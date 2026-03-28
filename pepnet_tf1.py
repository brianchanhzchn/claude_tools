"""
PEPNet: Parameter and Embedding Personalized Network
论文: "PEPNet: Parameter and Embedding Personalized Network for Infusing
       with Personalized Prior Information" (KDD 2023)

TensorFlow 兼容实现（避免 tf.layers，兼容 Keras 3）
所有全连接层均用 tf.matmul + tf.Variable 手动实现。

数据集: 模拟 MovieLens 多域多任务场景
  - 多域: 按用户年龄段分 3 个域 (年轻/中年/老年)
  - 多任务: Rating 回归 + Watched 分类
"""

import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import os
import time
import logging
from collections import defaultdict

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ============================================================
# 工具函数：手动实现全连接层（绕开 tf.layers / Keras）
# ============================================================

def dense(x, out_dim, name, activation=None):
    """
    手动全连接层：y = x @ W + b
    使用 Xavier 初始化，完全不依赖 tf.layers 或 Keras。
    """
    in_dim = x.shape[-1].value
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        limit = np.sqrt(6.0 / (in_dim + out_dim))
        W = tf.get_variable(
            'W', shape=[in_dim, out_dim],
            initializer=tf.random_uniform_initializer(-limit, limit))
        b = tf.get_variable(
            'b', shape=[out_dim],
            initializer=tf.zeros_initializer())
        y = tf.matmul(x, W) + b
        if activation == 'relu':
            y = tf.nn.relu(y)
        elif activation == 'sigmoid':
            y = tf.sigmoid(y)
    return y


def dropout(x, rate, is_training):
    """训练时随机 Dropout，推理时直接返回"""
    keep_prob = 1.0 - rate
    return tf.cond(
        is_training,
        lambda: tf.nn.dropout(x, keep_prob=keep_prob),
        lambda: x)


# ============================================================
# 1. 数据生成 —— 模拟 MovieLens 多域多任务场景
# ============================================================

def generate_movielens_like_data(n_users=3000, n_items=2000,
                                  n_samples=50000, seed=42):
    """
    生成模拟 MovieLens 风格的多域多任务数据集。

    域划分 (domain_id):
        0 = 年轻用户 (age < 25)
        1 = 中年用户 (25 <= age < 45)
        2 = 老年用户 (age >= 45)

    任务:
        task_rating  : 用户评分 [1,5] → 回归
        task_watched : 是否完整观看   → 分类 (0/1)
    """
    np.random.seed(seed)

    user_ages   = np.random.randint(15, 70, size=n_users)
    user_gender = np.random.randint(0, 2,  size=n_users)
    user_domain = np.where(user_ages < 25, 0,
                  np.where(user_ages < 45, 1, 2))

    n_genres  = 18
    n_authors = 500
    item_genre      = np.random.randint(0, n_genres,  size=n_items)
    item_author     = np.random.randint(0, n_authors, size=n_items)
    item_popularity = np.random.beta(2, 5, size=n_items)

    user_ids = np.random.randint(0, n_users, size=n_samples)
    item_ids = np.random.randint(0, n_items, size=n_samples)

    domain_user_count = np.random.poisson(20, size=n_samples).astype(np.float32)
    domain_user_count /= (domain_user_count.max() + 1e-8)

    user_interest = np.random.uniform(0, 1, size=n_users)
    base_rating   = (user_interest[user_ids] * 2.5 +
                     item_popularity[item_ids] * 1.5 +
                     np.random.normal(0, 0.5, size=n_samples))
    rating = np.clip(base_rating + 1.5, 1.0, 5.0).astype(np.float32)

    watched_prob = 1.0 / (1.0 + np.exp(-(rating - 3.0)))
    watched      = (np.random.uniform(size=n_samples) < watched_prob).astype(np.int32)

    age_bucket = (user_ages[user_ids] // 10).astype(np.int32)

    data = {
        'user_id'                  : user_ids.astype(np.int32),
        'item_id'                  : item_ids.astype(np.int32),
        'author_id'                : item_author[item_ids].astype(np.int32),
        'domain_id'                : user_domain[user_ids].astype(np.int32),
        'user_age_bucket'          : age_bucket,
        'user_gender'              : user_gender[user_ids].astype(np.int32),
        'item_genre'               : item_genre[item_ids].astype(np.int32),
        'item_popularity'          : item_popularity[item_ids].astype(np.float32),
        'domain_user_behavior_cnt' : domain_user_count,
        'label_rating'             : rating,
        'label_watched'            : watched,
    }
    stats = {
        'n_users'      : n_users,
        'n_items'      : n_items,
        'n_authors'    : n_authors,
        'n_genres'     : n_genres,
        'n_domains'    : 3,
        'n_tasks'      : 2,
        'n_age_buckets': 7,
    }
    return data, stats


def train_val_test_split(data, train_ratio=0.8, val_ratio=0.1, seed=42):
    n   = len(data['user_id'])
    idx = np.arange(n)
    np.random.seed(seed)
    np.random.shuffle(idx)
    t1  = int(n * train_ratio)
    t2  = int(n * (train_ratio + val_ratio))
    sub = lambda d, i: {k: v[i] for k, v in d.items()}
    return sub(data, idx[:t1]), sub(data, idx[t1:t2]), sub(data, idx[t2:])


def batch_iter(data, batch_size=1024, shuffle=True, seed=None):
    n   = len(data['user_id'])
    idx = np.arange(n)
    if shuffle:
        np.random.RandomState(seed).shuffle(idx)
    for s in range(0, n, batch_size):
        e = min(s + batch_size, n)
        yield {k: v[idx[s:e]] for k, v in data.items()}


# ============================================================
# 2. PEPNet 网络（手动 dense，无 tf.layers / Keras）
# ============================================================

class PEPNet:
    """
    PEPNet 完整实现：
      - Gate Neural Unit  (Gate NU)
      - Embedding Personalized Network (EPNet)
      - Parameter Personalized Network (PPNet)
      - 多任务 DNN Tower

    全部使用 tf.matmul + tf.Variable，不依赖 tf.layers 或 Keras。
    """

    def __init__(self, stats,
                 emb_dim=40,
                 gate_hidden=128,
                 dnn_hidden=(256, 128, 64),
                 gamma=2.0,
                 dropout_rate=0.1,
                 l2_reg=1e-5,
                 scope_prefix='pepnet'):
        self.stats        = stats
        self.emb_dim      = emb_dim
        self.gate_hidden  = gate_hidden
        self.dnn_hidden   = list(dnn_hidden)
        self.gamma        = gamma
        self.dropout_rate = dropout_rate
        self.l2_reg       = l2_reg
        self.n_tasks      = stats['n_tasks']
        self.n_domains    = stats['n_domains']
        self.scope_prefix = scope_prefix
        self._build_graph()

    # ----------------------------------------------------------
    # 占位符
    # ----------------------------------------------------------
    def _build_placeholders(self):
        with tf.name_scope(self.scope_prefix + '/inputs'):
            self.ph_user_id     = tf.placeholder(tf.int32,   [None], name='user_id')
            self.ph_item_id     = tf.placeholder(tf.int32,   [None], name='item_id')
            self.ph_author_id   = tf.placeholder(tf.int32,   [None], name='author_id')
            self.ph_domain_id   = tf.placeholder(tf.int32,   [None], name='domain_id')
            self.ph_age_bucket  = tf.placeholder(tf.int32,   [None], name='age_bucket')
            self.ph_gender      = tf.placeholder(tf.int32,   [None], name='gender')
            self.ph_genre       = tf.placeholder(tf.int32,   [None], name='genre')
            self.ph_item_pop    = tf.placeholder(tf.float32, [None], name='item_pop')
            self.ph_domain_cnt  = tf.placeholder(tf.float32, [None], name='domain_cnt')
            self.ph_label_rating  = tf.placeholder(tf.float32, [None], name='label_rating')
            self.ph_label_watched = tf.placeholder(tf.int32,   [None], name='label_watched')
            self.ph_is_training   = tf.placeholder(tf.bool,           name='is_training')

    # ----------------------------------------------------------
    # Embedding 表（tf.get_variable，不用 Keras）
    # ----------------------------------------------------------
    def _build_embeddings(self):
        s    = self.stats
        d    = self.emb_dim
        init = tf.glorot_uniform_initializer()
        sp   = self.scope_prefix
        with tf.variable_scope(sp + '/embeddings'):
            self.emb_user   = tf.get_variable('user',   [s['n_users'],       d], initializer=init)
            self.emb_item   = tf.get_variable('item',   [s['n_items'],       d], initializer=init)
            self.emb_author = tf.get_variable('author', [s['n_authors'],     d], initializer=init)
            self.emb_domain = tf.get_variable('domain', [s['n_domains'],     d], initializer=init)
            self.emb_age    = tf.get_variable('age',    [s['n_age_buckets'], d], initializer=init)
            self.emb_gender = tf.get_variable('gender', [2,                  d], initializer=init)
            self.emb_genre  = tf.get_variable('genre',  [s['n_genres'],      d], initializer=init)

    # ----------------------------------------------------------
    # 查询 Embedding，拼接主干特征 E
    # ----------------------------------------------------------
    def _lookup_embeddings(self):
        eu = tf.nn.embedding_lookup(self.emb_user,   self.ph_user_id)
        ei = tf.nn.embedding_lookup(self.emb_item,   self.ph_item_id)
        ea = tf.nn.embedding_lookup(self.emb_author, self.ph_author_id)
        eg = tf.nn.embedding_lookup(self.emb_age,    self.ph_age_bucket)
        en = tf.nn.embedding_lookup(self.emb_gender, self.ph_gender)
        ec = tf.nn.embedding_lookup(self.emb_genre,  self.ph_genre)

        pop = tf.expand_dims(self.ph_item_pop,   -1)
        cnt = tf.expand_dims(self.ph_domain_cnt, -1)

        # 主干 Embedding  E: [B, 6d+2]
        self.E = tf.concat([eu, ei, ea, eg, en, ec, pop, cnt], axis=1)

        # EPNet 输入：域侧特征 [B, d+1]
        ed = tf.nn.embedding_lookup(self.emb_domain, self.ph_domain_id)
        self.domain_feat = tf.concat([ed, cnt], axis=1)

        # PPNet 输入：user/item/author prior [B, 3d]
        self.prior = tf.concat([eu, ei, ea], axis=1)

    # ----------------------------------------------------------
    # Gate Neural Unit  (论文公式 2-3)
    # ----------------------------------------------------------
    def _gate_nu(self, x, out_dim, scope):
        """
        两层网络：
          Layer1: ReLU(xW1 + b1)   — 特征交叉
          Layer2: γ·Sigmoid(hW2 + b2) — 生成门控 δ ∈ [0, γ]
        """
        sp = self.scope_prefix + '/' + scope
        h     = dense(x,  self.gate_hidden, sp + '/gate_l1', activation='relu')
        logit = dense(h,  out_dim,          sp + '/gate_l2', activation=None)
        delta = self.gamma * tf.sigmoid(logit)
        return delta

    # ----------------------------------------------------------
    # EPNet  (公式 5-6)
    # ----------------------------------------------------------
    def _epnet(self):
        """
        δ_domain = Gate_NU( domain_feat ⊕ stop_grad(E) )
        O_ep     = δ_domain ⊗ E
        """
        e_dim       = self.E.shape[1].value
        epnet_input = tf.concat([self.domain_feat,
                                  tf.stop_gradient(self.E)], axis=1)
        delta_domain = self._gate_nu(epnet_input, e_dim, scope='epnet_gate')
        O_ep = delta_domain * self.E
        return O_ep

    # ----------------------------------------------------------
    # PPNet  (公式 7-9)
    # ----------------------------------------------------------
    def _ppnet(self, O_ep, layer_idx, hidden_dim):
        """
        δ_task = Gate_NU( prior ⊕ stop_grad(O_ep) )
        形状: [B, hidden_dim * n_tasks]
        """
        ppnet_input = tf.concat([self.prior,
                                  tf.stop_gradient(O_ep)], axis=1)
        out_dim    = hidden_dim * self.n_tasks
        delta_task = self._gate_nu(ppnet_input, out_dim,
                                    scope=f'ppnet_gate_l{layer_idx}')
        return delta_task

    # ----------------------------------------------------------
    # 多任务 DNN Tower（注入 PPNet 门控）
    # ----------------------------------------------------------
    def _multi_task_dnn(self, O_ep, scope_tag=''):
        """
        每层:
          1. 线性变换 H = x @ W + b
          2. PPNet 门控: H = δ_task ⊗ H
          3. ReLU + Dropout (非末层)
        """
        task_hiddens = [O_ep] * self.n_tasks

        for l, h_dim in enumerate(self.dnn_hidden):
            delta_task = self._ppnet(O_ep, l, h_dim)
            delta_list = tf.split(delta_task, self.n_tasks, axis=1)

            new_hiddens = []
            for t in range(self.n_tasks):
                sc = f'{self.scope_prefix}/{scope_tag}dnn_t{t}_l{l}'
                h  = dense(task_hiddens[t], h_dim, sc + '/linear')
                h  = delta_list[t] * h            # PPNet 门控
                if l < len(self.dnn_hidden) - 1:
                    h = tf.nn.relu(h)
                    h = dropout(h, self.dropout_rate, self.ph_is_training)
                new_hiddens.append(h)

            task_hiddens = new_hiddens

        # 输出层
        sp = self.scope_prefix + '/' + scope_tag
        pred_rating = 1.0 + 4.0 * tf.sigmoid(tf.squeeze(
            dense(task_hiddens[0], 1, sp + 'out_rating'), axis=1))

        pred_watched_logit = tf.squeeze(
            dense(task_hiddens[1], 1, sp + 'out_watched'), axis=1)

        return pred_rating, pred_watched_logit

    # ----------------------------------------------------------
    # 损失
    # ----------------------------------------------------------
    def _build_loss(self, pred_rating, pred_watched_logit):
        labels_f = tf.cast(self.ph_label_watched, tf.float32)
        loss_rating  = tf.reduce_mean(
            tf.square(self.ph_label_rating - pred_rating))
        loss_watched = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                labels=labels_f, logits=pred_watched_logit))
        l2 = self.l2_reg * tf.add_n(
            [tf.nn.l2_loss(v) for v in tf.trainable_variables()
             if 'b:0' not in v.name and self.scope_prefix in v.name])
        total = loss_rating + loss_watched + l2
        return total, loss_rating, loss_watched

    # ----------------------------------------------------------
    # 指标
    # ----------------------------------------------------------
    def _build_metrics(self, pred_rating, pred_watched_logit):
        mae      = tf.reduce_mean(tf.abs(self.ph_label_rating - pred_rating))
        pred_prob = tf.sigmoid(pred_watched_logit)
        labels_f  = tf.cast(self.ph_label_watched, tf.float32)
        auc, auc_update = tf.metrics.auc(
            labels=labels_f, predictions=pred_prob,
            name=self.scope_prefix + '_auc')
        return mae, auc, auc_update

    # ----------------------------------------------------------
    # 构建完整计算图
    # ----------------------------------------------------------
    def _build_graph(self):
        logger.info(f"Building PEPNet graph [{self.scope_prefix}] ...")
        tf.reset_default_graph()

        self._build_placeholders()
        self._build_embeddings()
        self._lookup_embeddings()

        O_ep = self._epnet()
        self.pred_rating, self.pred_watched_logit = self._multi_task_dnn(O_ep)

        (self.total_loss,
         self.loss_rating,
         self.loss_watched) = self._build_loss(
            self.pred_rating, self.pred_watched_logit)

        (self.mae,
         self.auc,
         self.auc_update) = self._build_metrics(
            self.pred_rating, self.pred_watched_logit)

        # 优化器：Embedding → AdaGrad；DNN → Adam（同论文）
        all_vars = tf.trainable_variables()
        emb_vars = [v for v in all_vars if '/embeddings/' in v.name]
        dnn_vars = [v for v in all_vars if '/embeddings/' not in v.name]

        self.global_step = tf.Variable(0, trainable=False, name='global_step')
        opt_emb = tf.train.AdagradOptimizer(learning_rate=0.05)
        opt_dnn = tf.train.AdamOptimizer(learning_rate=5e-4)

        grads    = tf.gradients(self.total_loss, emb_vars + dnn_vars)
        n_emb    = len(emb_vars)
        train_emb = opt_emb.apply_gradients(zip(grads[:n_emb], emb_vars))
        train_dnn = opt_dnn.apply_gradients(
            zip(grads[n_emb:], dnn_vars), global_step=self.global_step)
        self.train_op = tf.group(train_emb, train_dnn)

        self.init_global = tf.global_variables_initializer()
        self.init_local  = tf.local_variables_initializer()
        self.saver       = tf.train.Saver(max_to_keep=3)

        total_params = sum(np.prod(v.shape.as_list()) for v in all_vars)
        logger.info(f"  Trainable parameters: {total_params:,}")
        logger.info("  Graph built OK.")

    # ----------------------------------------------------------
    # Feed dict 辅助
    # ----------------------------------------------------------
    def _make_feed(self, batch, training=True):
        return {
            self.ph_user_id      : batch['user_id'],
            self.ph_item_id      : batch['item_id'],
            self.ph_author_id    : batch['author_id'],
            self.ph_domain_id    : batch['domain_id'],
            self.ph_age_bucket   : batch['user_age_bucket'],
            self.ph_gender       : batch['user_gender'],
            self.ph_genre        : batch['item_genre'],
            self.ph_item_pop     : batch['item_popularity'],
            self.ph_domain_cnt   : batch['domain_user_behavior_cnt'],
            self.ph_label_rating : batch['label_rating'],
            self.ph_label_watched: batch['label_watched'],
            self.ph_is_training  : training,
        }

    # ----------------------------------------------------------
    # 训练 / 评估 / 预测
    # ----------------------------------------------------------
    def train_epoch(self, sess, data, batch_size=1024):
        losses, lr_list, lw_list = [], [], []
        for batch in batch_iter(data, batch_size, shuffle=True):
            feed = self._make_feed(batch, training=True)
            _, tl, lr, lw = sess.run(
                [self.train_op, self.total_loss,
                 self.loss_rating, self.loss_watched],
                feed_dict=feed)
            losses.append(tl); lr_list.append(lr); lw_list.append(lw)
        return np.mean(losses), np.mean(lr_list), np.mean(lw_list)

    def evaluate(self, sess, data, batch_size=2048):
        sess.run(self.init_local)
        maes = []
        for batch in batch_iter(data, batch_size, shuffle=False):
            feed = self._make_feed(batch, training=False)
            mae_val, _ = sess.run([self.mae, self.auc_update], feed_dict=feed)
            maes.append(mae_val)
        auc_val = sess.run(self.auc)
        return np.mean(maes), auc_val

    def predict(self, sess, data, batch_size=2048):
        ratings, watcheds = [], []
        for batch in batch_iter(data, batch_size, shuffle=False):
            feed = self._make_feed(batch, training=False)
            pr, pw = sess.run(
                [self.pred_rating, self.pred_watched_logit], feed_dict=feed)
            ratings.append(pr)
            watcheds.append(1.0 / (1.0 + np.exp(-pw)))
        return np.concatenate(ratings), np.concatenate(watcheds)


# ============================================================
# 3. 消融实验用子类
# ============================================================

class PEPNetAblation(PEPNet):
    """可选择性禁用 EPNet 或 PPNet 的消融版本"""

    def __init__(self, stats, use_epnet=True, use_ppnet=True,
                 scope_prefix='pepnet_abl', **kwargs):
        self.use_epnet = use_epnet
        self.use_ppnet = use_ppnet
        super().__init__(stats, scope_prefix=scope_prefix, **kwargs)

    def _build_graph(self):
        logger.info(f"  [{self.scope_prefix}] "
                    f"EPNet={'ON' if self.use_epnet else 'OFF'}, "
                    f"PPNet={'ON' if self.use_ppnet else 'OFF'}")
        tf.reset_default_graph()
        self._build_placeholders()
        self._build_embeddings()
        self._lookup_embeddings()

        O_ep = self._epnet() if self.use_epnet else self.E
        self.pred_rating, self.pred_watched_logit = \
            self._multi_task_dnn_abl(O_ep)

        (self.total_loss,
         self.loss_rating,
         self.loss_watched) = self._build_loss(
            self.pred_rating, self.pred_watched_logit)
        (self.mae,
         self.auc,
         self.auc_update) = self._build_metrics(
            self.pred_rating, self.pred_watched_logit)

        all_vars  = tf.trainable_variables()
        emb_vars  = [v for v in all_vars if '/embeddings/' in v.name]
        dnn_vars  = [v for v in all_vars if '/embeddings/' not in v.name]
        self.global_step = tf.Variable(0, trainable=False)
        opt_emb   = tf.train.AdagradOptimizer(0.05)
        opt_dnn   = tf.train.AdamOptimizer(5e-4)
        grads     = tf.gradients(self.total_loss, emb_vars + dnn_vars)
        n_emb     = len(emb_vars)
        train_emb = opt_emb.apply_gradients(zip(grads[:n_emb], emb_vars))
        train_dnn = opt_dnn.apply_gradients(
            zip(grads[n_emb:], dnn_vars), global_step=self.global_step)
        self.train_op    = tf.group(train_emb, train_dnn)
        self.init_global = tf.global_variables_initializer()
        self.init_local  = tf.local_variables_initializer()
        self.saver       = tf.train.Saver(max_to_keep=1)

    def _multi_task_dnn_abl(self, O_ep):
        task_hiddens = [O_ep] * self.n_tasks
        for l, h_dim in enumerate(self.dnn_hidden):
            if self.use_ppnet:
                delta_task = self._ppnet(O_ep, l, h_dim)
                delta_list = tf.split(delta_task, self.n_tasks, axis=1)
            new_hiddens = []
            for t in range(self.n_tasks):
                sc = f'{self.scope_prefix}/abl_t{t}_l{l}'
                h  = dense(task_hiddens[t], h_dim, sc + '/linear')
                if self.use_ppnet:
                    h = delta_list[t] * h
                if l < len(self.dnn_hidden) - 1:
                    h = tf.nn.relu(h)
                    h = dropout(h, self.dropout_rate, self.ph_is_training)
                new_hiddens.append(h)
            task_hiddens = new_hiddens

        sp = self.scope_prefix
        pred_rating = 1.0 + 4.0 * tf.sigmoid(tf.squeeze(
            dense(task_hiddens[0], 1, sp + '/abl_out_rating'), axis=1))
        pred_watched_logit = tf.squeeze(
            dense(task_hiddens[1], 1, sp + '/abl_out_watched'), axis=1)
        return pred_rating, pred_watched_logit


# ============================================================
# 4. 训练主流程
# ============================================================

def train_pepnet(n_epochs=20, batch_size=1024,
                 save_dir='/tmp/pepnet_ckpt'):
    logger.info("=" * 60)
    logger.info("PEPNet Training Pipeline")
    logger.info("=" * 60)

    data, stats = generate_movielens_like_data(
        n_users=3000, n_items=2000, n_samples=50000)
    train_data, val_data, test_data = train_val_test_split(data)
    logger.info(f"  Train: {len(train_data['user_id']):,}  "
                f"Val: {len(val_data['user_id']):,}  "
                f"Test: {len(test_data['user_id']):,}")

    model = PEPNet(
        stats, emb_dim=40, gate_hidden=128,
        dnn_hidden=(256, 128, 64), gamma=2.0,
        dropout_rate=0.1, l2_reg=1e-5)

    os.makedirs(save_dir, exist_ok=True)
    cfg = tf.ConfigProto()
    cfg.gpu_options.allow_growth = True

    best_val_auc = 0.0
    history      = defaultdict(list)

    with tf.Session(config=cfg) as sess:
        sess.run(model.init_global)
        sess.run(model.init_local)

        logger.info(f"\n{'Epoch':>5} | {'Loss':>8} | {'Rating':>7} | "
                    f"{'Watched':>8} | {'ValMAE':>7} | {'ValAUC':>7} | Time")
        logger.info("-" * 62)

        for epoch in range(1, n_epochs + 1):
            t0 = time.time()
            tl, lr, lw   = model.train_epoch(sess, train_data, batch_size)
            val_mae, val_auc = model.evaluate(sess, val_data, batch_size)
            elapsed = time.time() - t0

            logger.info(f"{epoch:>5} | {tl:>8.4f} | {lr:>7.4f} | "
                        f"{lw:>8.4f} | {val_mae:>7.4f} | {val_auc:>7.4f} | "
                        f"{elapsed:.1f}s")
            history['train_loss'].append(tl)
            history['val_mae'].append(val_mae)
            history['val_auc'].append(val_auc)

            if val_auc > best_val_auc:
                best_val_auc = val_auc
                model.saver.save(sess, os.path.join(save_dir, 'best.ckpt'))
                logger.info(f"  ★ Best saved  (val_auc={best_val_auc:.4f})")

        logger.info("\n" + "=" * 40)
        test_mae, test_auc = model.evaluate(sess, test_data, batch_size)
        logger.info(f"  Test MAE  : {test_mae:.4f}")
        logger.info(f"  Test AUC  : {test_auc:.4f}")

        pred_r, pred_w = model.predict(sess, test_data, batch_size)
        logger.info("\n  Sample predictions (first 5):")
        logger.info(f"  {'TrueRating':>10} | {'PredRating':>10} | "
                    f"{'TrueWatch':>9} | {'PredProb':>8}")
        logger.info("  " + "-" * 47)
        for i in range(5):
            logger.info(f"  {test_data['label_rating'][i]:>10.2f} | "
                        f"{pred_r[i]:>10.2f} | "
                        f"{test_data['label_watched'][i]:>9d} | "
                        f"{pred_w[i]:>8.4f}")

    return history, {'test_mae': test_mae, 'test_auc': test_auc}


# ============================================================
# 5. 消融实验（对应论文 Figure 4 / RQ2）
# ============================================================

def ablation_study(n_epochs=10, batch_size=1024):
    logger.info("\n" + "=" * 60)
    logger.info("Ablation Study")
    logger.info("=" * 60)

    data, stats = generate_movielens_like_data(
        n_users=3000, n_items=2000, n_samples=30000, seed=0)
    train_data, _, test_data = train_val_test_split(data)

    configs = {
        'Baseline (no EPNet/PPNet)': dict(use_epnet=False, use_ppnet=False),
        'EPNet only (no PPNet)'    : dict(use_epnet=True,  use_ppnet=False),
        'PPNet only (no EPNet)'    : dict(use_epnet=False, use_ppnet=True),
        'Full PEPNet'              : dict(use_epnet=True,  use_ppnet=True),
    }

    results = {}
    for i, (name, cfg) in enumerate(configs.items()):
        model = PEPNetAblation(
            stats, scope_prefix=f'abl_{i}',
            emb_dim=40, gate_hidden=64,
            dnn_hidden=(128, 64), gamma=2.0, **cfg)
        tf_cfg = tf.ConfigProto()
        tf_cfg.gpu_options.allow_growth = True
        with tf.Session(config=tf_cfg) as sess:
            sess.run(model.init_global)
            for _ in range(n_epochs):
                model.train_epoch(sess, train_data, batch_size)
            mae, auc = model.evaluate(sess, test_data, batch_size)
        logger.info(f"  {name:<30}  MAE={mae:.4f}  AUC={auc:.4f}")
        results[name] = {'mae': mae, 'auc': auc}

    return results


# ============================================================
# 6. 入口
# ============================================================

if __name__ == '__main__':
    logger.info("PEPNet — TensorFlow (Keras-3 compatible)")
    logger.info("Paper: KDD 2023, Kuaishou\n")

    history, final = train_pepnet(n_epochs=20, batch_size=1024)

    print("\n" + "=" * 40)
    print(f"Test MAE  (Rating) : {final['test_mae']:.4f}")
    print(f"Test AUC  (Watched): {final['test_auc']:.4f}")
    print("=" * 40)
