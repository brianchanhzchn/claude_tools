"""
PEPNet v2 — 三项改进 + 消融实验
========================================
基于论文: KDD 2023, Kuaishou Technology

改进 1 — Attention Gate NU
    原版 Gate NU 用线性层生成门控，无法捕捉先验特征间的交叉依赖。
    改进：在 Gate NU 第一层后加入单头自注意力（Q/K/V），让不同
    先验特征互相感知后再生成门控向量。

改进 2 — GradNorm 动态任务权重
    原版多任务损失均等加和，稀疏任务（Follow 0.3%）梯度被稠密
    任务（EffView 48%）压制，造成 task seesaw。
    改进：引入可学习的任务不确定度权重（Uncertainty Weighting），
    每个任务的损失权重由 log(σ_t²) 自动学习，稀疏任务权重更高。

改进 3 — Graph EPNet（域图结构建模）
    原版 EPNet 对各域独立处理，忽略域间的用户重叠和行为迁移关系。
    改进：把域关系建成图（域节点 + 相似度边），用单层 GCN 对域
    Embedding 做消息传递，再把聚合后的域表示输入 Gate NU。

消融框架
    8 种配置逐步叠加，清晰量化每项改进的独立收益和叠加收益。

技术约束
    全部使用 tf.matmul + tf.Variable，兼容 Keras 3，无 tf.layers。
"""

import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import os, time, logging
from collections import defaultdict

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ================================================================
# 基础算子（全部手动实现，不依赖 tf.layers / Keras）
# ================================================================

def dense(x, out_dim, name, activation=None):
    """Xavier 初始化的手动全连接层"""
    in_dim = x.shape[-1].value
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        lim = np.sqrt(6.0 / (in_dim + out_dim))
        W = tf.get_variable('W', [in_dim, out_dim],
                            initializer=tf.random_uniform_initializer(-lim, lim))
        b = tf.get_variable('b', [out_dim],
                            initializer=tf.zeros_initializer())
        y = tf.matmul(x, W) + b
    if   activation == 'relu':    return tf.nn.relu(y)
    elif activation == 'sigmoid': return tf.sigmoid(y)
    elif activation == 'tanh':    return tf.tanh(y)
    return y


def dropout(x, rate, is_training):
    return tf.cond(is_training,
                   lambda: tf.nn.dropout(x, keep_prob=1.0 - rate),
                   lambda: x)


def layer_norm(x, name):
    """简化版 Layer Normalization"""
    d = x.shape[-1].value
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        scale = tf.get_variable('scale', [d], initializer=tf.ones_initializer())
        bias  = tf.get_variable('bias',  [d], initializer=tf.zeros_initializer())
    mean, var = tf.nn.moments(x, axes=[-1], keep_dims=True)
    return scale * (x - mean) / tf.sqrt(var + 1e-6) + bias


# ================================================================
# 数据生成
# ================================================================

def generate_data(n_users=3000, n_items=2000, n_samples=50000, seed=42):
    """
    模拟 Kuaishou 风格的多域多任务数据集。
    域 0/1/2 = 年轻/中年/老年用户（模拟三个 Tab）
    任务 0: Rating 回归 [1,5]
    任务 1: Watched 分类 (0/1)
    """
    np.random.seed(seed)
    n_genres, n_authors = 18, 500

    user_ages   = np.random.randint(15, 70, size=n_users)
    user_gender = np.random.randint(0, 2,   size=n_users)
    user_domain = np.where(user_ages < 25, 0, np.where(user_ages < 45, 1, 2))

    item_genre      = np.random.randint(0, n_genres,  size=n_items)
    item_author     = np.random.randint(0, n_authors, size=n_items)
    item_popularity = np.random.beta(2, 5, size=n_items)

    uid = np.random.randint(0, n_users, size=n_samples)
    iid = np.random.randint(0, n_items, size=n_samples)

    domain_cnt = np.random.poisson(20, size=n_samples).astype(np.float32)
    domain_cnt /= domain_cnt.max() + 1e-8

    user_interest = np.random.uniform(0, 1, size=n_users)
    base  = user_interest[uid] * 2.5 + item_popularity[iid] * 1.5
    base += np.random.normal(0, 0.5, size=n_samples)
    rating  = np.clip(base + 1.5, 1.0, 5.0).astype(np.float32)
    watched = (np.random.uniform(size=n_samples) <
               1 / (1 + np.exp(-(rating - 3)))).astype(np.int32)

    data = {
        'user_id'                  : uid.astype(np.int32),
        'item_id'                  : iid.astype(np.int32),
        'author_id'                : item_author[iid].astype(np.int32),
        'domain_id'                : user_domain[uid].astype(np.int32),
        'user_age_bucket'          : (user_ages[uid] // 10).astype(np.int32),
        'user_gender'              : user_gender[uid].astype(np.int32),
        'item_genre'               : item_genre[iid].astype(np.int32),
        'item_popularity'          : item_popularity[iid].astype(np.float32),
        'domain_user_behavior_cnt' : domain_cnt,
        'label_rating'             : rating,
        'label_watched'            : watched,
    }
    stats = dict(n_users=n_users, n_items=n_items, n_authors=n_authors,
                 n_genres=n_genres, n_domains=3, n_tasks=2, n_age_buckets=7)
    return data, stats


def split(data, tr=0.8, va=0.1, seed=42):
    n = len(data['user_id'])
    idx = np.arange(n); np.random.seed(seed); np.random.shuffle(idx)
    t1, t2 = int(n*tr), int(n*(tr+va))
    sub = lambda i: {k: v[i] for k, v in data.items()}
    return sub(idx[:t1]), sub(idx[t1:t2]), sub(idx[t2:])


def batcher(data, bs=1024, shuffle=True, seed=None):
    n = len(data['user_id'])
    idx = np.arange(n)
    if shuffle: np.random.RandomState(seed).shuffle(idx)
    for s in range(0, n, bs):
        e = min(s + bs, n)
        yield {k: v[idx[s:e]] for k, v in data.items()}


# ================================================================
# PEPNet v2 —— 带三项改进开关的统一模型
# ================================================================

class PEPNetV2:
    """
    统一模型，通过三个布尔开关控制改进项：
      use_attn_gate  : 改进1 — 注意力 Gate NU
      use_grad_norm  : 改进2 — 动态任务权重（Uncertainty Weighting）
      use_graph_epnet: 改进3 — Graph EPNet

    其余参数与原版保持一致，便于公平比较。
    """

    def __init__(self, stats,
                 emb_dim=40,
                 gate_hidden=128,
                 dnn_hidden=(256, 128, 64),
                 gamma=2.0,
                 dropout_rate=0.1,
                 l2_reg=1e-5,
                 # ── 改进开关 ──
                 use_attn_gate=False,
                 use_grad_norm=False,
                 use_graph_epnet=False,
                 # ── 注意力超参 ──
                 attn_heads=4,
                 # ── 域图超参（GCN 邻接矩阵） ──
                 domain_adj=None,
                 scope_prefix='pep'):

        self.stats           = stats
        self.emb_dim         = emb_dim
        self.gate_hidden     = gate_hidden
        self.dnn_hidden      = list(dnn_hidden)
        self.gamma           = gamma
        self.dropout_rate    = dropout_rate
        self.l2_reg          = l2_reg
        self.use_attn_gate   = use_attn_gate
        self.use_grad_norm   = use_grad_norm
        self.use_graph_epnet = use_graph_epnet
        self.attn_heads      = attn_heads
        self.n_tasks         = stats['n_tasks']
        self.n_domains       = stats['n_domains']
        self.sp              = scope_prefix

        # 域邻接矩阵（默认：全连接 + 自环，行归一化）
        if domain_adj is None:
            nd = stats['n_domains']
            A  = np.ones((nd, nd), dtype=np.float32)
            np.fill_diagonal(A, 2.0)              # 自环权重加倍
            D  = A.sum(axis=1, keepdims=True)
            self.domain_adj = (A / D).astype(np.float32)
        else:
            self.domain_adj = domain_adj

        self._build_graph()

    # ──────────────────────────────────────────
    # 占位符
    # ──────────────────────────────────────────
    def _placeholders(self):
        with tf.name_scope(self.sp + '/in'):
            self.ph_uid    = tf.placeholder(tf.int32,   [None], 'uid')
            self.ph_iid    = tf.placeholder(tf.int32,   [None], 'iid')
            self.ph_aid    = tf.placeholder(tf.int32,   [None], 'aid')
            self.ph_did    = tf.placeholder(tf.int32,   [None], 'did')
            self.ph_age    = tf.placeholder(tf.int32,   [None], 'age')
            self.ph_gen    = tf.placeholder(tf.int32,   [None], 'gen')
            self.ph_gre    = tf.placeholder(tf.int32,   [None], 'gre')
            self.ph_pop    = tf.placeholder(tf.float32, [None], 'pop')
            self.ph_cnt    = tf.placeholder(tf.float32, [None], 'cnt')
            self.ph_lbl_r  = tf.placeholder(tf.float32, [None], 'lbl_r')
            self.ph_lbl_w  = tf.placeholder(tf.int32,   [None], 'lbl_w')
            self.ph_train  = tf.placeholder(tf.bool,   [],     name="train")

    # ──────────────────────────────────────────
    # Embedding 表
    # ──────────────────────────────────────────
    def _embeddings(self):
        s, d, sp = self.stats, self.emb_dim, self.sp
        init = tf.glorot_uniform_initializer()
        with tf.variable_scope(sp + '/emb'):
            self.E_user   = tf.get_variable('user',   [s['n_users'],       d], initializer=init)
            self.E_item   = tf.get_variable('item',   [s['n_items'],       d], initializer=init)
            self.E_author = tf.get_variable('author', [s['n_authors'],     d], initializer=init)
            self.E_domain = tf.get_variable('domain', [s['n_domains'],     d], initializer=init)
            self.E_age    = tf.get_variable('age',    [s['n_age_buckets'], d], initializer=init)
            self.E_gender = tf.get_variable('gender', [2,                  d], initializer=init)
            self.E_genre  = tf.get_variable('genre',  [s['n_genres'],      d], initializer=init)

    # ──────────────────────────────────────────
    # 查询 Embedding，组装主干 E
    # ──────────────────────────────────────────
    def _lookup(self):
        eu = tf.nn.embedding_lookup(self.E_user,   self.ph_uid)
        ei = tf.nn.embedding_lookup(self.E_item,   self.ph_iid)
        ea = tf.nn.embedding_lookup(self.E_author, self.ph_aid)
        eg = tf.nn.embedding_lookup(self.E_age,    self.ph_age)
        en = tf.nn.embedding_lookup(self.E_gender, self.ph_gen)
        ec = tf.nn.embedding_lookup(self.E_genre,  self.ph_gre)
        pop = tf.expand_dims(self.ph_pop, -1)
        cnt = tf.expand_dims(self.ph_cnt, -1)

        self.E        = tf.concat([eu, ei, ea, eg, en, ec, pop, cnt], axis=1)  # [B, 6d+2]
        self.prior    = tf.concat([eu, ei, ea], axis=1)                        # [B, 3d]

        # 域侧特征（Graph EPNet 改进前/后通用）
        ed_raw = tf.nn.embedding_lookup(self.E_domain, self.ph_did)            # [B, d]
        self.eu, self.ei, self.ea = eu, ei, ea
        self.pop, self.cnt = pop, cnt
        self._ed_raw = ed_raw

    # ──────────────────────────────────────────
    # 改进3: Graph EPNet — 域图卷积
    # ──────────────────────────────────────────
    def _graph_domain_emb(self):
        """
        单层 GCN：H' = ReLU( Â · H · W )
        Â = 行归一化邻接矩阵（含自环），已在 __init__ 中预计算。
        H = 域 Embedding 矩阵 [n_domains, d]

        作用：域 i 的新表示融合了相邻域 j 的信息，
        让 EPNet 的门控感知到域间用户行为的迁移关系。
        """
        sp = self.sp
        nd, d = self.n_domains, self.emb_dim
        A_hat = tf.constant(self.domain_adj, dtype=tf.float32, name='A_hat')  # [D, D]

        # GCN 线性变换权重
        with tf.variable_scope(sp + '/gcn', reuse=tf.AUTO_REUSE):
            lim = np.sqrt(6.0 / (d + d))
            W_gcn = tf.get_variable('W', [d, d],
                                    initializer=tf.random_uniform_initializer(-lim, lim))

        H_agg = tf.nn.relu(
            tf.matmul(tf.matmul(A_hat, self.E_domain), W_gcn)
        )  # [n_domains, d]

        # 按 batch 中每个样本的 domain_id 取聚合后的域表示
        ed_gcn = tf.nn.embedding_lookup(H_agg, self.ph_did)   # [B, d]
        return ed_gcn

    # ──────────────────────────────────────────
    # 改进1: Attention Gate NU
    # ──────────────────────────────────────────
    def _attention_gate_nu(self, x, out_dim, scope):
        """
        在原版 Gate NU 的 ReLU 层后，插入单头自注意力模块：

          1. x → ReLU(xW₁) → h           (特征交叉，同原版)
          2. h → reshape → [B, n_feat, head_dim]
          3. 自注意力: softmax(QKᵀ/√d) V  → h_attn
          4. 残差 + LayerNorm             → h_out
          5. h_out → γ·Sigmoid(h_out W₂) → δ ∈ [0, γ]

        注意力让不同先验特征之间互相"感知"，
        例如 user_age 可以影响 item_genre 维度的门控强度。
        """
        sp = self.sp + '/' + scope
        heads    = self.attn_heads
        h_all    = dense(x, self.gate_hidden, sp + '/l1', activation='relu')
        # gate_hidden 必须能被 heads 整除
        head_dim = self.gate_hidden // heads                   # e.g. 128//4=32

        # 把隐层切成 [B, heads, head_dim]，每个 head 是一个"特征片段"
        B_size = tf.shape(h_all)[0]
        h_3d   = tf.reshape(h_all, [B_size, heads, head_dim])  # [B, H, d_h]

        # Q / K / V 投影（共享权重，省参数）
        Q = dense(h_all, self.gate_hidden, sp + '/Q', activation=None)
        K = dense(h_all, self.gate_hidden, sp + '/K', activation=None)
        V = dense(h_all, self.gate_hidden, sp + '/V', activation=None)

        Q3 = tf.reshape(Q, [B_size, heads, head_dim])          # [B, H, d_h]
        K3 = tf.reshape(K, [B_size, heads, head_dim])
        V3 = tf.reshape(V, [B_size, heads, head_dim])

        # 注意力得分: [B, H, H]
        scale  = tf.sqrt(tf.cast(head_dim, tf.float32))
        scores = tf.matmul(Q3, K3, transpose_b=True) / scale   # [B, H, H]
        attn   = tf.nn.softmax(scores, axis=-1)

        # 加权 V: [B, H, d_h] → [B, gate_hidden]
        h_attn = tf.reshape(
            tf.matmul(attn, V3),
            [B_size, self.gate_hidden])

        # 残差 + LayerNorm
        h_out  = layer_norm(h_all + h_attn, sp + '/ln')

        # 生成门控
        logit  = dense(h_out, out_dim, sp + '/l2', activation=None)
        delta  = self.gamma * tf.sigmoid(logit)
        return delta

    # ──────────────────────────────────────────
    # Gate NU 调度（根据开关选版本）
    # ──────────────────────────────────────────
    def _gate_nu(self, x, out_dim, scope):
        if self.use_attn_gate:
            return self._attention_gate_nu(x, out_dim, scope)
        sp = self.sp + '/' + scope
        h     = dense(x,  self.gate_hidden, sp + '/gate_l1', activation='relu')
        logit = dense(h,  out_dim,          sp + '/gate_l2', activation=None)
        return self.gamma * tf.sigmoid(logit)

    # ──────────────────────────────────────────
    # EPNet（公式 5-6，含改进3开关）
    # ──────────────────────────────────────────
    def _epnet(self):
        e_dim = self.E.shape[1].value

        if self.use_graph_epnet:
            ed = self._graph_domain_emb()              # GCN 聚合域表示 [B, d]
        else:
            ed = self._ed_raw                          # 原版静态域 emb [B, d]

        domain_feat = tf.concat([ed, self.cnt], axis=1)
        ep_in       = tf.concat([domain_feat, tf.stop_gradient(self.E)], axis=1)
        delta       = self._gate_nu(ep_in, e_dim, scope='epnet_gate')
        return delta * self.E

    # ──────────────────────────────────────────
    # PPNet（公式 7-9）
    # ──────────────────────────────────────────
    def _ppnet(self, O_ep, layer_idx, hidden_dim):
        pp_in  = tf.concat([self.prior, tf.stop_gradient(O_ep)], axis=1)
        out_d  = hidden_dim * self.n_tasks
        return self._gate_nu(pp_in, out_d, scope=f'ppnet_gate_l{layer_idx}')

    # ──────────────────────────────────────────
    # 多任务 DNN Tower
    # ──────────────────────────────────────────
    def _dnn(self, O_ep):
        task_h = [O_ep] * self.n_tasks
        for l, h_dim in enumerate(self.dnn_hidden):
            delta_task = self._ppnet(O_ep, l, h_dim)
            deltas     = tf.split(delta_task, self.n_tasks, axis=1)
            new_h = []
            for t in range(self.n_tasks):
                sc = f'{self.sp}/dnn_t{t}_l{l}'
                h  = dense(task_h[t], h_dim, sc + '/fc')
                h  = deltas[t] * h
                if l < len(self.dnn_hidden) - 1:
                    h = tf.nn.relu(h)
                    h = dropout(h, self.dropout_rate, self.ph_train)
                new_h.append(h)
            task_h = new_h

        pred_r = 1.0 + 4.0 * tf.sigmoid(tf.squeeze(
            dense(task_h[0], 1, self.sp + '/out_r'), axis=1))
        pred_w_logit = tf.squeeze(
            dense(task_h[1], 1, self.sp + '/out_w'), axis=1)
        return pred_r, pred_w_logit

    # ──────────────────────────────────────────
    # 改进2: Uncertainty Weighting 损失
    # ──────────────────────────────────────────
    def _build_loss(self, pred_r, pred_w_logit):
        """
        原版: L = L_rating + L_watched  (固定权重 1:1)

        改进: Uncertainty Weighting (Kendall et al. 2018)
            L = (1/2σ₀²)·L_rating + log(σ₀)
              + (1/2σ₁²)·L_watched + log(σ₁)
            σ₀, σ₁ 是可学习参数，代表每个任务的输出不确定度。
            稀疏任务（watched）的 σ 会自动学习到较小值（权重更高）。
            用 log(σ²) 参数化，避免 σ 变为负数。
        """
        labels_f = tf.cast(self.ph_lbl_w, tf.float32)
        L_r = tf.reduce_mean(tf.square(self.ph_lbl_r - pred_r))
        L_w = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                labels=labels_f, logits=pred_w_logit))

        if self.use_grad_norm:
            with tf.variable_scope(self.sp + '/task_weights', reuse=tf.AUTO_REUSE):
                # log_var_t = log(σ_t²)，初始化为 0（σ=1，即与原版等价）
                lv0 = tf.get_variable('log_var_0', [],
                                      initializer=tf.zeros_initializer())
                lv1 = tf.get_variable('log_var_1', [],
                                      initializer=tf.zeros_initializer())
            # 1/(2σ²) = exp(-log_var)
            w0 = tf.exp(-lv0)
            w1 = tf.exp(-lv1)
            # 加上 log(σ) = 0.5*log_var 作为正则，防止 σ→∞
            task_loss = w0 * L_r + 0.5 * lv0 + w1 * L_w + 0.5 * lv1
            self.task_weights = tf.stack([w0, w1])    # 供监控用
        else:
            task_loss = L_r + L_w
            self.task_weights = tf.constant([1.0, 1.0])

        # L2 正则（只作用于本模型变量）
        l2 = self.l2_reg * tf.add_n(
            [tf.nn.l2_loss(v) for v in tf.trainable_variables()
             if '/b:0' not in v.name and self.sp in v.name])

        return task_loss + l2, L_r, L_w

    # ──────────────────────────────────────────
    # 指标
    # ──────────────────────────────────────────
    def _build_metrics(self, pred_r, pred_w_logit):
        mae      = tf.reduce_mean(tf.abs(self.ph_lbl_r - pred_r))
        pred_p   = tf.sigmoid(pred_w_logit)
        labels_f = tf.cast(self.ph_lbl_w, tf.float32)
        auc, auc_upd = tf.metrics.auc(
            labels=labels_f, predictions=pred_p,
            name=self.sp + '_auc')
        return mae, auc, auc_upd

    # ──────────────────────────────────────────
    # 完整计算图
    # ──────────────────────────────────────────
    def _build_graph(self):
        flags = (f"AttnGate={'ON' if self.use_attn_gate   else 'off'} | "
                 f"GradNorm={'ON' if self.use_grad_norm   else 'off'} | "
                 f"GraphEP={'ON'  if self.use_graph_epnet else 'off'}")
        logger.info(f"  [{self.sp}] {flags}")
        tf.reset_default_graph()

        self._placeholders()
        self._embeddings()
        self._lookup()

        O_ep = self._epnet()
        self.pred_r, self.pred_w_logit = self._dnn(O_ep)

        self.total_loss, self.loss_r, self.loss_w = \
            self._build_loss(self.pred_r, self.pred_w_logit)
        self.mae, self.auc, self.auc_upd = \
            self._build_metrics(self.pred_r, self.pred_w_logit)

        all_v = tf.trainable_variables()
        emb_v = [v for v in all_v if '/emb/' in v.name]
        dnn_v = [v for v in all_v if '/emb/' not in v.name]

        self.global_step = tf.Variable(0, trainable=False)
        opt_e = tf.train.AdagradOptimizer(0.05)
        opt_d = tf.train.AdamOptimizer(5e-4)
        grads = tf.gradients(self.total_loss, emb_v + dnn_v)
        ne    = len(emb_v)
        t_e   = opt_e.apply_gradients(zip(grads[:ne], emb_v))
        t_d   = opt_d.apply_gradients(zip(grads[ne:], dnn_v),
                                       global_step=self.global_step)
        self.train_op    = tf.group(t_e, t_d)
        self.init_global = tf.global_variables_initializer()
        self.init_local  = tf.local_variables_initializer()
        self.saver       = tf.train.Saver(max_to_keep=1)

        n_params = sum(np.prod(v.shape.as_list()) for v in all_v)
        logger.info(f"  [{self.sp}] params: {n_params:,}")

    # ──────────────────────────────────────────
    # Feed dict
    # ──────────────────────────────────────────
    def _feed(self, b, train=True):
        return {
            self.ph_uid  : b['user_id'],
            self.ph_iid  : b['item_id'],
            self.ph_aid  : b['author_id'],
            self.ph_did  : b['domain_id'],
            self.ph_age  : b['user_age_bucket'],
            self.ph_gen  : b['user_gender'],
            self.ph_gre  : b['item_genre'],
            self.ph_pop  : b['item_popularity'],
            self.ph_cnt  : b['domain_user_behavior_cnt'],
            self.ph_lbl_r: b['label_rating'],
            self.ph_lbl_w: b['label_watched'],
            self.ph_train: train,
        }

    # ──────────────────────────────────────────
    # 训练 / 评估 / 预测
    # ──────────────────────────────────────────
    def train_epoch(self, sess, data, bs=1024):
        tl, lr, lw, tw = [], [], [], []
        for b in batcher(data, bs):
            fd = self._feed(b, True)
            _, l, r, w, weights = sess.run(
                [self.train_op, self.total_loss,
                 self.loss_r, self.loss_w, self.task_weights],
                feed_dict=fd)
            tl.append(l); lr.append(r); lw.append(w); tw.append(weights)
        return np.mean(tl), np.mean(lr), np.mean(lw), np.mean(tw, axis=0)

    def evaluate(self, sess, data, bs=2048):
        sess.run(self.init_local)
        maes = []
        for b in batcher(data, bs, shuffle=False):
            fd = self._feed(b, False)
            m, _ = sess.run([self.mae, self.auc_upd], feed_dict=fd)
            maes.append(m)
        return float(np.mean(maes)), float(sess.run(self.auc))

    def predict(self, sess, data, bs=2048):
        rs, ws = [], []
        for b in batcher(data, bs, shuffle=False):
            fd = self._feed(b, False)
            r, w = sess.run([self.pred_r, self.pred_w_logit], feed_dict=fd)
            rs.append(r); ws.append(1 / (1 + np.exp(-w)))
        return np.concatenate(rs), np.concatenate(ws)


# ================================================================
# 训练单个配置，返回结果
# ================================================================

def run_config(name, cfg_flags, stats, train_data, val_data, test_data,
               n_epochs=15, bs=1024, seed_idx=0):
    """
    训练一个消融配置，返回 test MAE / AUC 及每轮历史。
    seed_idx 保证不同配置用相同的参数空间初始化逻辑。
    """
    tf.reset_default_graph()
    np.random.seed(42 + seed_idx)
    tf.set_random_seed(42 + seed_idx)

    model = PEPNetV2(
        stats, emb_dim=40, gate_hidden=128,
        dnn_hidden=(256, 128, 64), gamma=2.0,
        dropout_rate=0.1, l2_reg=1e-5,
        scope_prefix=f'pep_{seed_idx}',
        **cfg_flags)

    cfg = tf.ConfigProto()
    cfg.gpu_options.allow_growth = True

    best_auc   = 0.0
    best_mae   = 999.0
    history    = defaultdict(list)

    with tf.Session(config=cfg) as sess:
        sess.run(model.init_global)
        sess.run(model.init_local)

        for ep in range(1, n_epochs + 1):
            t0 = time.time()
            tl, lr, lw, tw = model.train_epoch(sess, train_data, bs)
            v_mae, v_auc   = model.evaluate(sess, val_data, bs)
            elapsed = time.time() - t0

            history['train_loss'].append(tl)
            history['val_mae'].append(v_mae)
            history['val_auc'].append(v_auc)
            history['task_w0'].append(float(tw[0]))
            history['task_w1'].append(float(tw[1]))

            if v_auc > best_auc:
                best_auc, best_mae = v_auc, v_mae

            logger.info(
                f"    [{name[:28]:<28}] ep{ep:02d} "
                f"loss={tl:.4f}  val_mae={v_mae:.4f}  val_auc={v_auc:.4f}  "
                f"tw=[{tw[0]:.2f},{tw[1]:.2f}]  {elapsed:.1f}s")

        test_mae, test_auc = model.evaluate(sess, test_data, bs)

    return {
        'name'    : name,
        'test_mae': test_mae,
        'test_auc': test_auc,
        'best_val_auc': best_auc,
        'history' : dict(history),
    }


# ================================================================
# 消融实验主函数
# ================================================================

def ablation_study(n_epochs=15, bs=1024):
    """
    8 种配置的完整消融实验，依次叠加三项改进：

    配置编号  AttnGate  GradNorm  GraphEP   说明
    ────────  ────────  ────────  ───────   ────────────────────────────
    C0        off       off       off       原版 PEPNet（基线）
    C1        ON        off       off       仅加注意力 Gate NU
    C2        off       ON        off       仅加动态任务权重
    C3        off       off       ON        仅加 Graph EPNet
    C4        ON        ON        off       改进1 + 改进2
    C5        ON        off       ON        改进1 + 改进3
    C6        off       ON        ON        改进2 + 改进3
    C7        ON        ON        ON        全部三项改进（完整 v2）
    """
    logger.info("=" * 70)
    logger.info("PEPNet v2 消融实验 — 8 种配置")
    logger.info("=" * 70)

    # 所有配置共用同一份数据，保证可比性
    data, stats = generate_data(n_users=3000, n_items=2000, n_samples=50000)
    train_data, val_data, test_data = split(data)
    logger.info(f"数据集: train={len(train_data['user_id']):,}  "
                f"val={len(val_data['user_id']):,}  "
                f"test={len(test_data['user_id']):,}\n")

    configs = [
        # name,                         attn,  grad,  graph
        ("C0 | Baseline (原版 PEPNet)", False, False, False),
        ("C1 | + Attn Gate NU",         True,  False, False),
        ("C2 | + GradNorm",             False, True,  False),
        ("C3 | + Graph EPNet",          False, False, True ),
        ("C4 | + Attn + GradNorm",      True,  True,  False),
        ("C5 | + Attn + Graph",         True,  False, True ),
        ("C6 | + GradNorm + Graph",     False, True,  True ),
        ("C7 | Full v2 (全部改进)",      True,  True,  True ),
    ]

    all_results = []
    for idx, (name, attn, grad, graph) in enumerate(configs):
        logger.info(f"\n{'─'*60}")
        logger.info(f"配置 {idx}/7 : {name}")
        logger.info(f"{'─'*60}")
        res = run_config(
            name,
            dict(use_attn_gate=attn,
                 use_grad_norm=grad,
                 use_graph_epnet=graph),
            stats, train_data, val_data, test_data,
            n_epochs=n_epochs, bs=bs, seed_idx=idx)
        all_results.append(res)

    # ── 打印汇总表 ──
    print_ablation_table(all_results)
    return all_results


def print_ablation_table(results):
    """打印格式化的消融对比表，计算相对基线的增益"""
    baseline = results[0]
    b_mae = baseline['test_mae']
    b_auc = baseline['test_auc']

    banner = "=" * 78
    print(f"\n{banner}")
    print("消融实验结果汇总")
    print(banner)
    print(f"{'配置':<32} {'MAE':>7} {'ΔMAE':>8} {'AUC':>7} {'ΔAUC':>8}  改进项")
    print("-" * 78)

    improvement_tags = {
        "C0": "—",
        "C1": "AttnGate",
        "C2": "GradNorm",
        "C3": "GraphEP",
        "C4": "AttnGate+GradNorm",
        "C5": "AttnGate+GraphEP",
        "C6": "GradNorm+GraphEP",
        "C7": "全部改进",
    }

    for r in results:
        tag  = r['name'][:2]
        dmae = r['test_mae'] - b_mae
        dauc = r['test_auc'] - b_auc
        sign_mae = "+" if dmae > 0 else ""
        sign_auc = "+" if dauc > 0 else ""
        imp  = improvement_tags.get(tag, "")
        star = " ★" if tag == "C7" else ""
        print(f"{r['name']:<32} "
              f"{r['test_mae']:>7.4f} "
              f"{sign_mae}{dmae:>7.4f} "
              f"{r['test_auc']:>7.4f} "
              f"{sign_auc}{dauc:>7.4f}  "
              f"{imp}{star}")

    print("=" * 78)

    # 单项改进增益分析
    print("\n各改进项独立增益（相对基线 C0）：")
    print(f"  改进1 Attention Gate NU : "
          f"MAE {results[1]['test_mae']-b_mae:+.4f} | "
          f"AUC {results[1]['test_auc']-b_auc:+.4f}")
    print(f"  改进2 GradNorm (UW)     : "
          f"MAE {results[2]['test_mae']-b_mae:+.4f} | "
          f"AUC {results[2]['test_auc']-b_auc:+.4f}")
    print(f"  改进3 Graph EPNet       : "
          f"MAE {results[3]['test_mae']-b_mae:+.4f} | "
          f"AUC {results[3]['test_auc']-b_auc:+.4f}")

    c7_mae = results[7]['test_mae']
    c7_auc = results[7]['test_auc']
    sum_ind_mae = sum(r['test_mae']-b_mae for r in results[1:4])
    sum_ind_auc = sum(r['test_auc']-b_auc for r in results[1:4])
    print(f"\n  三项叠加实际增益: MAE {c7_mae-b_mae:+.4f} | AUC {c7_auc-b_auc:+.4f}")
    print(f"  三项独立增益之和: MAE {sum_ind_mae:+.4f} | AUC {sum_ind_auc:+.4f}")
    synergy = (c7_auc - b_auc) - sum_ind_auc
    print(f"  协同效应 (AUC)  : {synergy:+.4f}  "
          f"{'(正协同，改进间相互增强)' if synergy > 0 else '(负协同，存在冗余)'}")
    print()


# ================================================================
# 入口
# ================================================================

if __name__ == '__main__':
    logger.info("PEPNet v2 — 三项改进 + 消融实验")
    logger.info("Paper: KDD 2023 | Improvements: 2024\n")

    results = ablation_study(n_epochs=15, bs=1024)

    # 输出最终最优配置
    best = max(results, key=lambda r: r['test_auc'])
    print(f"\n最优配置: {best['name']}")
    print(f"  Test MAE  : {best['test_mae']:.4f}")
    print(f"  Test AUC  : {best['test_auc']:.4f}")

