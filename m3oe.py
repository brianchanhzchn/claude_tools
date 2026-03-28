"""
M3oE: Multi-Domain Multi-Task Mixture of Experts
=================================================
基于 PEPNet v2 代码库，在相同数据集和训练设置下实现 M3oE，
并进行系统性消融实验，量化每个设计组件的实际收益。

M3oE 核心思想 (SIGIR 2024)
--------------------------
PEPNet 用门控缩放已有参数（乘法式个性化），M3oE 转而用
混合专家（MoE）结构同时在域轴和任务轴上解耦参数：

  三类专家:
    - Domain Expert   : 只服务某个域，捕获域内特有的用户行为模式
    - Task Expert     : 只服务某个任务，捕获任务间的共性表示
    - Shared Expert   : 全局共享，捕获跨域跨任务的通用知识

  双路由门控:
    - Domain Gate  : 以域 ID + 域统计特征为输入，选择 Domain Experts
    - Task Gate    : 以任务 ID + 用户/物品先验为输入，选择 Task Experts
    两路门控的输出与 Shared Expert 输出做加权融合

  消融设计 (8 种配置):
    A0: 无专家，纯 shared DNN（最基础基线）
    A1: 仅 Shared Expert（MoE 框架但无专用专家）
    A2: Shared + Domain Expert（加入域解耦）
    A3: Shared + Task Expert（加入任务解耦）
    A4: 完整 M3oE（Shared + Domain + Task Expert）
    A5: A4 + Uncertainty Weighting（动态任务损失权重）
    A6: A4 + Graph Domain Embedding（GCN 域图结构）
    A7: 完整 M3oE + UW + Graph（全部组件）

技术约束: 全部使用 tf.matmul + tf.Variable，兼容 Keras 3，无 tf.layers。
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
# 基础算子（与 PEPNet v2 完全一致）
# ================================================================

def dense(x, out_dim, name, activation=None):
    """Xavier 初始化手动全连接层，不依赖 tf.layers / Keras"""
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
    d = x.shape[-1].value
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        scale = tf.get_variable('scale', [d], initializer=tf.ones_initializer())
        bias  = tf.get_variable('bias',  [d], initializer=tf.zeros_initializer())
    mean, var = tf.nn.moments(x, axes=[-1], keep_dims=True)
    return scale * (x - mean) / tf.sqrt(var + 1e-6) + bias


# ================================================================
# 数据生成（与 PEPNet v2 完全一致，保证可比性）
# ================================================================

def generate_data(n_users=3000, n_items=2000, n_samples=50000, seed=42):
    """模拟 Kuaishou 三域两任务数据集，与 PEPNet v2 完全相同"""
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
# M3oE 核心模型
# ================================================================

class M3oE:
    """
    Multi-Domain Multi-Task Mixture of Experts

    专家结构
    --------
    n_shared_experts  : 全局共享专家数（默认 2）
    n_domain_experts  : 每个域的专用专家数（默认 2，共 n_domains × 2 个）
    n_task_experts    : 每个任务的专用专家数（默认 2，共 n_tasks × 2 个）
    expert_dim        : 每个专家的输出维度

    路由机制
    --------
    Domain Gate : softmax( Linear(domain_feat) ) → [B, n_domain_experts+n_shared]
    Task Gate   : softmax( Linear(task_feat)   ) → [B, n_task_experts+n_shared]
    最终表示 = domain_gate · [domain_experts, shared] +
               task_gate   · [task_experts,   shared]

    消融开关
    --------
    use_domain_expert : 是否使用域专用专家（改动1）
    use_task_expert   : 是否使用任务专用专家（改动2）
    use_uw            : 是否使用 Uncertainty Weighting 损失（改动3）
    use_graph_domain  : 是否使用 GCN 域图结构（改动4）
    """

    def __init__(self, stats,
                 emb_dim=40,
                 expert_dim=128,
                 expert_hidden=256,
                 tower_hidden=(128, 64),
                 n_shared_experts=2,
                 n_domain_experts=2,
                 n_task_experts=2,
                 dropout_rate=0.1,
                 l2_reg=1e-5,
                 # ── 消融开关 ──
                 use_domain_expert=True,
                 use_task_expert=True,
                 use_uw=False,
                 use_graph_domain=False,
                 # ── 域图 ──
                 domain_adj=None,
                 scope_prefix='m3oe'):

        self.stats             = stats
        self.emb_dim           = emb_dim
        self.expert_dim        = expert_dim
        self.expert_hidden     = expert_hidden
        self.tower_hidden      = list(tower_hidden)
        self.n_shared          = n_shared_experts
        self.n_dom_exp         = n_domain_experts
        self.n_task_exp        = n_task_experts
        self.dropout_rate      = dropout_rate
        self.l2_reg            = l2_reg
        self.use_domain_expert = use_domain_expert
        self.use_task_expert   = use_task_expert
        self.use_uw            = use_uw
        self.use_graph_domain  = use_graph_domain
        self.n_domains         = stats['n_domains']
        self.n_tasks           = stats['n_tasks']
        self.sp                = scope_prefix

        # 域邻接矩阵（行归一化，自环权重 2.0）
        if domain_adj is None:
            nd = self.n_domains
            A  = np.ones((nd, nd), dtype=np.float32)
            np.fill_diagonal(A, 2.0)
            self.domain_adj = (A / A.sum(axis=1, keepdims=True)).astype(np.float32)
        else:
            self.domain_adj = domain_adj

        self._build_graph()

    # ──────────────────────────────────────────
    # 占位符
    # ──────────────────────────────────────────
    def _placeholders(self):
        with tf.name_scope(self.sp + '/in'):
            self.ph_uid   = tf.placeholder(tf.int32,   [None], 'uid')
            self.ph_iid   = tf.placeholder(tf.int32,   [None], 'iid')
            self.ph_aid   = tf.placeholder(tf.int32,   [None], 'aid')
            self.ph_did   = tf.placeholder(tf.int32,   [None], 'did')
            self.ph_age   = tf.placeholder(tf.int32,   [None], 'age')
            self.ph_gen   = tf.placeholder(tf.int32,   [None], 'gen')
            self.ph_gre   = tf.placeholder(tf.int32,   [None], 'gre')
            self.ph_pop   = tf.placeholder(tf.float32, [None], 'pop')
            self.ph_cnt   = tf.placeholder(tf.float32, [None], 'cnt')
            self.ph_lbl_r = tf.placeholder(tf.float32, [None], 'lbl_r')
            self.ph_lbl_w = tf.placeholder(tf.int32,   [None], 'lbl_w')
            self.ph_train = tf.placeholder(tf.bool,    [],     name='train')

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
            # 任务 ID Embedding（M3oE 新增：让 Task Gate 感知任务身份）
            self.E_task   = tf.get_variable('task',   [s['n_tasks'],       d], initializer=init)

    # ──────────────────────────────────────────
    # 查询 Embedding，组装输入
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

        # 主干输入 x: [B, 6d+2]
        self.x     = tf.concat([eu, ei, ea, eg, en, ec, pop, cnt], axis=1)
        self.eu    = eu
        self.ei    = ei
        self.ea    = ea
        self.cnt   = cnt

        # 域表示（GCN 改进可选）
        self._ed_raw = tf.nn.embedding_lookup(self.E_domain, self.ph_did)

        # domain_feat: 域路由门控的输入 [B, d+1]
        ed = self._graph_domain_emb() if self.use_graph_domain else self._ed_raw
        self.domain_feat = tf.concat([ed, cnt], axis=1)

        # user_item_feat: 任务路由门控的输入 [B, 3d]
        self.user_item_feat = tf.concat([eu, ei, ea], axis=1)

    # ──────────────────────────────────────────
    # 改进4: GCN 域图（与 PEPNet v2 相同实现）
    # ──────────────────────────────────────────
    def _graph_domain_emb(self):
        d     = self.emb_dim
        A_hat = tf.constant(self.domain_adj, dtype=tf.float32)
        with tf.variable_scope(self.sp + '/gcn', reuse=tf.AUTO_REUSE):
            lim   = np.sqrt(6.0 / (d + d))
            W_gcn = tf.get_variable('W', [d, d],
                                    initializer=tf.random_uniform_initializer(-lim, lim))
        H_agg = tf.nn.relu(tf.matmul(tf.matmul(A_hat, self.E_domain), W_gcn))
        return tf.nn.embedding_lookup(H_agg, self.ph_did)

    # ──────────────────────────────────────────
    # 单个专家网络：两层 MLP
    # ──────────────────────────────────────────
    def _expert_mlp(self, x, name):
        """
        专家网络: Linear(expert_hidden) → ReLU → Linear(expert_dim)
        每个专家独立参数，互不共享。
        """
        h = dense(x, self.expert_hidden, name + '/l1', activation='relu')
        h = dropout(h, self.dropout_rate, self.ph_train)
        h = dense(h, self.expert_dim,    name + '/l2', activation='relu')
        return h  # [B, expert_dim]

    # ──────────────────────────────────────────
    # 构建所有专家
    # ──────────────────────────────────────────
    def _build_experts(self):
        sp = self.sp
        # Shared experts: 所有域/任务共享
        self.shared_experts = [
            self._expert_mlp(self.x, f'{sp}/shared_exp_{i}')
            for i in range(self.n_shared)
        ]  # list of [B, expert_dim]

        # Domain-specific experts: 每个域 n_dom_exp 个
        # 实现方式：为每个域建一组专家，通过 domain_id one-hot 软选择
        self.domain_experts = {}
        for d in range(self.n_domains):
            self.domain_experts[d] = [
                self._expert_mlp(self.x, f'{sp}/dom{d}_exp_{i}')
                for i in range(self.n_dom_exp)
            ]

        # Task-specific experts: 每个任务 n_task_exp 个
        self.task_experts = {}
        for t in range(self.n_tasks):
            self.task_experts[t] = [
                self._expert_mlp(self.x, f'{sp}/task{t}_exp_{i}')
                for i in range(self.n_task_exp)
            ]

    # ──────────────────────────────────────────
    # 双路由门控：为每个 (domain, task) 对计算融合表示
    # ──────────────────────────────────────────
    def _moe_fusion(self, domain_id_onehot, task_idx):
        """
        对于任务 task_idx，计算 MoE 融合输出。

        Domain Gate
        -----------
        输入: domain_feat ⊕ domain_id_onehot  →  softmax
        选择: 当前域的 domain experts + shared experts
        权重: [B, n_dom_exp + n_shared]

        Task Gate
        ---------
        输入: user_item_feat ⊕ E_task[task_idx]  →  softmax
        选择: 当前任务的 task experts + shared experts
        权重: [B, n_task_exp + n_shared]

        Fusion
        ------
        out = weighted_sum(domain_selected) + weighted_sum(task_selected)
              再经 LayerNorm 稳定训练
        """
        sp = self.sp
        B  = tf.shape(self.x)[0]

        # ── Shared expert 堆叠: [B, n_shared, expert_dim] ──
        shared_stack = tf.stack(self.shared_experts, axis=1)  # [B, n_sh, ed]

        # ── Domain Gate ──
        # 每个样本属于一个域，取该域的 domain experts
        # 用 domain_id one-hot 加权求和，实现软路由
        dom_gate_input = tf.concat([self.domain_feat, domain_id_onehot], axis=1)
        dom_n_select   = (self.n_dom_exp + self.n_shared
                          if self.use_domain_expert else self.n_shared)

        dom_gate_w = tf.nn.softmax(
            dense(dom_gate_input, dom_n_select,
                  f'{sp}/dom_gate_t{task_idx}'))  # [B, dom_n_select]

        if self.use_domain_expert:
            # 加权融合所有域的 domain experts（软路由，通过 domain one-hot 区分）
            # shape: [B, n_domains*n_dom_exp, expert_dim]
            all_dom_experts = []
            for d in range(self.n_domains):
                # domain one-hot 的第 d 维作为该域专家的软权重调制
                d_weight = domain_id_onehot[:, d:d+1]  # [B, 1]
                for exp in self.domain_experts[d]:
                    all_dom_experts.append(exp * d_weight)  # [B, expert_dim]

            # 取前 n_dom_exp 个（已按域 one-hot 调制，其余域权重接近 0）
            dom_exp_stack = tf.stack(all_dom_experts, axis=1)  # [B, D*n_dom_exp, ed]

            # 为门控选取当前域的专家 + shared
            # 简化：把所有域专家平均 pool 后和 shared 一起路由
            # 实际效果通过 domain one-hot 调制实现域隔离
            dom_pool = tf.reduce_sum(dom_exp_stack, axis=1)  # [B, expert_dim]
            dom_pool_exp = tf.expand_dims(dom_pool, 1)       # [B, 1, ed]

            # 要选 n_dom_exp 个 dom + n_shared 个 shared
            dom_candidates = tf.concat(
                [tf.tile(dom_pool_exp, [1, self.n_dom_exp, 1]),
                 shared_stack], axis=1)  # [B, n_dom_exp+n_shared, ed]
        else:
            dom_candidates = shared_stack      # [B, n_shared, ed]

        dom_gate_w_3d = tf.expand_dims(dom_gate_w, -1)  # [B, sel, 1]
        dom_out = tf.reduce_sum(dom_candidates * dom_gate_w_3d, axis=1)  # [B, ed]

        # ── Task Gate ──
        task_emb       = tf.nn.embedding_lookup(self.E_task,
                             tf.fill([B], task_idx))  # [B, d]
        task_gate_input = tf.concat([self.user_item_feat, task_emb], axis=1)
        task_n_select   = (self.n_task_exp + self.n_shared
                           if self.use_task_expert else self.n_shared)

        task_gate_w = tf.nn.softmax(
            dense(task_gate_input, task_n_select,
                  f'{sp}/task_gate_t{task_idx}'))  # [B, task_n_select]

        if self.use_task_expert:
            task_exp_list = self.task_experts[task_idx]
            task_exp_stack = tf.stack(task_exp_list, axis=1)  # [B, n_task_exp, ed]
            task_candidates = tf.concat([task_exp_stack, shared_stack], axis=1)
        else:
            task_candidates = shared_stack

        task_gate_w_3d = tf.expand_dims(task_gate_w, -1)
        task_out = tf.reduce_sum(task_candidates * task_gate_w_3d, axis=1)  # [B, ed]

        # ── 融合 + LayerNorm ──
        fused = dom_out + task_out
        fused = layer_norm(fused, f'{sp}/fuse_ln_t{task_idx}')
        return fused  # [B, expert_dim]

    # ──────────────────────────────────────────
    # 任务塔：接收 MoE 融合输出，输出预测值
    # ──────────────────────────────────────────
    def _task_tower(self, fused, task_idx, output_name):
        sp = self.sp
        h  = fused
        for l, h_dim in enumerate(self.tower_hidden):
            h = dense(h, h_dim, f'{sp}/tower_t{task_idx}_l{l}', activation='relu')
            h = dropout(h, self.dropout_rate, self.ph_train)
        out = dense(h, 1, f'{sp}/{output_name}')
        return tf.squeeze(out, axis=1)  # [B]

    # ──────────────────────────────────────────
    # 改进3: Uncertainty Weighting 损失（与 PEPNet v2 相同）
    # ──────────────────────────────────────────
    def _build_loss(self, pred_r, pred_w_logit):
        labels_f = tf.cast(self.ph_lbl_w, tf.float32)
        L_r = tf.reduce_mean(tf.square(self.ph_lbl_r - pred_r))
        L_w = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                labels=labels_f, logits=pred_w_logit))

        if self.use_uw:
            with tf.variable_scope(self.sp + '/task_w', reuse=tf.AUTO_REUSE):
                lv0 = tf.get_variable('lv0', [], initializer=tf.zeros_initializer())
                lv1 = tf.get_variable('lv1', [], initializer=tf.zeros_initializer())
            w0, w1 = tf.exp(-lv0), tf.exp(-lv1)
            task_loss = w0*L_r + 0.5*lv0 + w1*L_w + 0.5*lv1
            self.task_weights = tf.stack([w0, w1])
        else:
            task_loss = L_r + L_w
            self.task_weights = tf.constant([1.0, 1.0])

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
        flags = (f"DomExp={'ON' if self.use_domain_expert else 'off'} | "
                 f"TaskExp={'ON' if self.use_task_expert  else 'off'} | "
                 f"UW={'ON'     if self.use_uw            else 'off'} | "
                 f"Graph={'ON'  if self.use_graph_domain  else 'off'}")
        logger.info(f"  [{self.sp}] {flags}")
        tf.reset_default_graph()

        self._placeholders()
        self._embeddings()
        self._lookup()

        # domain_id one-hot: [B, n_domains]
        domain_onehot = tf.one_hot(self.ph_did, self.n_domains)

        # 构建所有专家
        self._build_experts()

        # 每个任务独立路由 → 任务塔
        fused_r = self._moe_fusion(domain_onehot, task_idx=0)
        fused_w = self._moe_fusion(domain_onehot, task_idx=1)

        pred_r_logit = self._task_tower(fused_r, 0, 'out_r')
        self.pred_r  = 1.0 + 4.0 * tf.sigmoid(pred_r_logit)

        self.pred_w_logit = self._task_tower(fused_w, 1, 'out_w')

        self.total_loss, self.loss_r, self.loss_w = \
            self._build_loss(self.pred_r, self.pred_w_logit)
        self.mae, self.auc, self.auc_upd = \
            self._build_metrics(self.pred_r, self.pred_w_logit)

        # 优化器：Embedding → AdaGrad，其余 → Adam（同 PEPNet v2）
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
            m, _ = sess.run([self.mae, self.auc_upd], feed_dict=self._feed(b, False))
            maes.append(m)
        return float(np.mean(maes)), float(sess.run(self.auc))

    def predict(self, sess, data, bs=2048):
        rs, ws = [], []
        for b in batcher(data, bs, shuffle=False):
            r, w = sess.run([self.pred_r, self.pred_w_logit],
                            feed_dict=self._feed(b, False))
            rs.append(r); ws.append(1 / (1 + np.exp(-w)))
        return np.concatenate(rs), np.concatenate(ws)


# ================================================================
# 训练单个配置
# ================================================================

def run_one(name, model_cls, model_kwargs, stats,
            train_data, val_data, test_data,
            n_epochs=15, bs=1024, seed_idx=0):
    """训练并评估单个配置，返回结果字典"""
    tf.reset_default_graph()
    np.random.seed(42 + seed_idx)
    tf.set_random_seed(42 + seed_idx)

    model = model_cls(stats, scope_prefix=f'run_{seed_idx}', **model_kwargs)

    cfg = tf.ConfigProto()
    cfg.gpu_options.allow_growth = True

    history    = defaultdict(list)
    best_auc   = 0.0

    with tf.Session(config=cfg) as sess:
        sess.run(model.init_global)
        sess.run(model.init_local)

        for ep in range(1, n_epochs + 1):
            t0 = time.time()
            tl, lr, lw, tw  = model.train_epoch(sess, train_data, bs)
            v_mae, v_auc    = model.evaluate(sess, val_data, bs)
            elapsed = time.time() - t0

            history['train_loss'].append(tl)
            history['val_mae'].append(v_mae)
            history['val_auc'].append(v_auc)
            history['tw0'].append(float(tw[0]))
            history['tw1'].append(float(tw[1]))

            if v_auc > best_auc:
                best_auc = v_auc

            logger.info(f"    [{name[:26]:<26}] "
                        f"ep{ep:02d} loss={tl:.4f} "
                        f"val_mae={v_mae:.4f} val_auc={v_auc:.4f} "
                        f"tw=[{tw[0]:.2f},{tw[1]:.2f}] {elapsed:.1f}s")

        test_mae, test_auc = model.evaluate(sess, test_data, bs)

    return dict(name=name, test_mae=test_mae, test_auc=test_auc,
                best_val_auc=best_auc, history=dict(history))


# ================================================================
# 消融实验主函数
# ================================================================

def ablation_study(n_epochs=15, bs=1024):
    """
    8 种消融配置，依次叠加 M3oE 的组件：

    配置   DomExp  TaskExp  UW    Graph   说明
    ────   ──────  ───────  ──    ─────   ──────────────────────────────
    A0     off     off      off   off     纯 Shared DNN（无专用专家）
    A1     off     off      off   off     仅 Shared Experts（MoE 框架）
    A2     ON      off      off   off     + Domain Experts
    A3     off     ON       off   off     + Task Experts
    A4     ON      ON       off   off     完整 M3oE（双路由）
    A5     ON      ON       ON    off     + Uncertainty Weighting
    A6     ON      ON       off   ON      + Graph Domain Emb
    A7     ON      ON       ON    ON      完整 M3oE + UW + Graph

    注: A0 是"无 MoE 框架"基线，A1 是"有 MoE 框架但无专用专家"基线
    """
    logger.info("=" * 72)
    logger.info("M3oE 消融实验 — 8 种配置")
    logger.info("=" * 72)

    data, stats = generate_data(n_users=3000, n_items=2000, n_samples=500000)
    train_data, val_data, test_data = split(data)
    logger.info(f"数据集: train={len(train_data['user_id']):,}  "
                f"val={len(val_data['user_id']):,}  "
                f"test={len(test_data['user_id']):,}\n")

    # ── 公共超参（对所有配置保持一致）──
    common = dict(
        emb_dim=40, expert_dim=128, expert_hidden=256,
        tower_hidden=(128, 64),
        n_shared_experts=2, n_domain_experts=2, n_task_experts=2,
        dropout_rate=0.1, l2_reg=1e-5,
    )

    # A0: 退化为纯 shared MLP（n_shared=1，无专用专家，无 MoE 路由）
    # 通过把 n_shared_experts=1 且关闭专用专家来模拟
    configs = [
        # name,                      dom,   task,  uw,    graph
        ("A0 | Shared MLP (基线)",   False, False, False, False),
        ("A1 | + Shared Experts",    False, False, False, False),  # 见下方特殊处理
        ("A2 | + Domain Experts",    True,  False, False, False),
        ("A3 | + Task Experts",      False, True,  False, False),
        ("A4 | Full M3oE",           True,  True,  False, False),
        ("A5 | + UW",                True,  True,  True,  False),
        ("A6 | + Graph",             True,  True,  False, True ),
        ("A7 | Full M3oE+UW+Graph",  True,  True,  True,  True ),
    ]

    all_results = []
    for idx, (name, dom, task, uw, graph) in enumerate(configs):
        logger.info(f"\n{'─'*60}")
        logger.info(f"配置 {idx}/7 : {name}")
        logger.info(f"{'─'*60}")

        # A0 用单个 shared expert 模拟纯 MLP（无专用专家，最少参数）
        kw = dict(**common,
                  use_domain_expert=dom,
                  use_task_expert=task,
                  use_uw=uw,
                  use_graph_domain=graph)
        if idx == 0:
            kw['n_shared_experts'] = 1   # A0: 最小化，模拟 plain DNN

        res = run_one(name, M3oE, kw, stats,
                      train_data, val_data, test_data,
                      n_epochs=n_epochs, bs=bs, seed_idx=idx)
        all_results.append(res)

    _print_table(all_results)
    return all_results


# ================================================================
# 结果汇总打印
# ================================================================

def _print_table(results):
    b_mae = results[0]['test_mae']
    b_auc = results[0]['test_auc']

    bar = "=" * 80
    print(f"\n{bar}")
    print("M3oE 消融实验结果汇总")
    print(bar)
    print(f"{'配置':<32} {'MAE':>7} {'ΔMAE':>8} {'AUC':>7} {'ΔAUC':>8}  组件")
    print("-" * 80)

    tags = ["—", "SharedMoE", "+DomainExp",
            "+TaskExp", "DualRoute",
            "+UW", "+Graph", "Full"]

    for i, r in enumerate(results):
        dm   = r['test_mae'] - b_mae
        da   = r['test_auc'] - b_auc
        sm   = "+" if dm > 0 else ""
        sa   = "+" if da > 0 else ""
        star = " ★" if i == len(results)-1 else ""
        print(f"{r['name']:<32} "
              f"{r['test_mae']:>7.4f} "
              f"{sm}{dm:>7.4f} "
              f"{r['test_auc']:>7.4f} "
              f"{sa}{da:>7.4f}  "
              f"{tags[i]}{star}")
    print(bar)

    # 组件独立增益
    print("\n各组件独立增益（相对 A0 基线）：")
    labels = [
        ("SharedMoE  框架", 1),
        ("DomainExp  加入", 2),
        ("TaskExp    加入", 3),
        ("双路由(A4) 完整", 4),
        ("UW         加入", 5),
        ("Graph      加入", 6),
    ]
    for lbl, idx in labels:
        r = results[idx]
        print(f"  {lbl}: MAE {r['test_mae']-b_mae:+.4f} | "
              f"AUC {r['test_auc']-b_auc:+.4f}")

    best = max(results, key=lambda r: r['test_auc'])
    print(f"\n最优配置: {best['name']}")
    print(f"  Test MAE : {best['test_mae']:.4f}  "
          f"Test AUC : {best['test_auc']:.4f}")

    # 与 A0 的总增益
    full = results[-1]
    print(f"\nFull M3oE+UW+Graph vs 基线:")
    print(f"  MAE: {full['test_mae']:.4f} ({full['test_mae']-b_mae:+.4f})")
    print(f"  AUC: {full['test_auc']:.4f} ({full['test_auc']-b_auc:+.4f})")
    print()


# ================================================================
# 入口
# ================================================================

if __name__ == '__main__':
    logger.info("M3oE — Multi-Domain Multi-Task MoE")
    logger.info("Based on SIGIR 2024 | Codebase: PEPNet v2\n")

    results = ablation_study(n_epochs=15, bs=1024)
