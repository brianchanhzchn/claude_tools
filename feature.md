# PEPNet 特征放置指南

> 推荐系统五类特征在 PEPNet 各模块的最优放置策略
>
> 基于 `pepnet_tf1.py` 代码实现，结合 PEPNet 论文（KDD 2023）的模块设计逻辑

---

## 目录

1. [核心放置原则](#1-核心放置原则)
2. [PEPNet 模块速查](#2-pepnet-模块速查)
3. [用户特征](#3-用户特征)
4. [商品特征](#4-商品特征)
5. [交叉特征](#5-交叉特征)
6. [上下文特征](#6-上下文特征)
7. [序列特征](#7-序列特征)
8. [完整放置汇总表](#8-完整放置汇总表)
9. [代码集成示例](#9-代码集成示例)

---

## 1. 核心放置原则

五类特征在 PEPNet 中有四个可注入的位置，每个位置对应不同的语义职责：

| 注入位置 | 语义职责 | 适合的特征类型 |
|---------|---------|--------------|
| Embedding 层（主干 E） | 基础表示学习，所有特征的起点 | ID 类、属性类、离散化的稠密特征 |
| EPNet 门控输入 | 域个性化——"在哪个场景下" | domain_id、域统计特征、上下文 |
| PPNet 门控 prior | 任务个性化——"对谁、推什么" | user_id、item_id、author_id |
| DNN 第一层输入 | 显式高阶交叉，预计算的匹配信号 | 交叉特征、FM 输出、DIN 分数 |

一句话总结：**身份类**（谁）进 PPNet，**场景类**（在哪）进 EPNet，**匹配类**（多合适）进 DNN 输入，**时序类**（最近干什么）用序列编码器压缩后进 PPNet。

---

## 2. PEPNet 模块速查

```
原始特征
    │
    ▼
┌─────────────────────────────────┐
│        Embedding 层              │  ← 所有 ID 类特征在此变为稠密向量 E
└──────────────┬──────────────────┘
               │
    ┌──────────┴──────────┐
    │                     │
    ▼                     ▼
┌─────────┐         ┌──────────────┐
│  EPNet  │         │  (主干 E)    │
│ 域门控  │         │  O_ep 输出   │
└────┬────┘         └──────┬───────┘
     │  δ_domain ⊗ E       │
     └──────────┬──────────┘
                ▼
┌───────────────────────────────────┐
│       DNN 第一层                   │  ← 交叉特征在此 concat 输入
└───────────────┬───────────────────┘
                │  每层注入 PPNet 门控
                ▼
┌───────────────────────────────────┐
│  PPNet（每个 DNN 层独立生成门控）   │  ← prior = user+item+author
│  δ_task ⊗ H                       │
└───────────────┬───────────────────┘
                ▼
┌───────────────────────────────────┐
│         多任务输出塔               │
└───────────────────────────────────┘
```

---

## 3. 用户特征

### 放置位置

- **Embedding 层（主干 E）**：用户 ID 及所有用户属性
- **PPNet 的 prior 输入**：user_id + 关键属性（age、gender 等）

### 设计原因

用户特征承担两个职责：

1. 进主干 E，让模型学习该用户的长期稳定偏好表示
2. 进 PPNet prior，让任务门控感知"当前用户对不同任务的偏好权重"——例如年轻用户更倾向点赞，老年用户更倾向完整观看，PPNet 需要这个信息来为不同任务生成不同强度的门控

### 典型特征清单

| 特征 | 类型 | 主干 E | PPNet prior |
|------|------|:------:|:-----------:|
| user_id | ID | ✓ | ✓ |
| user_age_bucket | 离散 | ✓ | ✓ |
| user_gender | 离散 | ✓ | ✓ |
| user_level | 离散 | ✓ | |
| user_register_days | 稠密 | ✓（分桶） | |

### 代码实现

```python
# Embedding 层查询
eu_id  = tf.nn.embedding_lookup(emb_user_id,  ph_uid)     # [B, d]
eu_age = tf.nn.embedding_lookup(emb_age,      ph_age)     # [B, d]
eu_gen = tf.nn.embedding_lookup(emb_gender,   ph_gender)  # [B, d]

# 主干 E 包含全部用户特征
E = tf.concat([eu_id, eu_age, eu_gen, ...], axis=1)

# PPNet prior 包含 user_id + 关键属性
# 论文公式(7): O_prior = E(F_u) ⊕ E(F_i) ⊕ E(F_a)
prior = tf.concat([eu_id, eu_age, eu_gen, ei_id, ea_id], axis=1)
delta_task = gate_nu(prior ⊕ stop_grad(O_ep))
```

---

## 4. 商品特征

### 放置位置

- **Embedding 层（主干 E）**：item_id、属性特征、稠密特征
- **PPNet 的 prior 输入**：item_id、author_id（ID 类）

### 设计原因

`item_id` 和 `author_id` 必须进 PPNet prior，因为不同物品天然对应不同的任务完成率——高质量视频的完整观看率远高于普通内容，PPNet 需要感知这个差异来动态调整任务门控权重。属性类特征（category、price 等）进主干 E 即可，不需要进门控。

### 典型特征清单

| 特征 | 类型 | 主干 E | PPNet prior |
|------|------|:------:|:-----------:|
| item_id | ID | ✓ | ✓ |
| author_id | ID | ✓ | ✓ |
| item_category | 离散 | ✓ | |
| item_popularity | 稠密 | ✓（直接 expand_dims） | |
| item_duration | 稠密 | ✓（分桶） | |
| item_quality_score | 稠密 | ✓ | |

### 代码实现

```python
ei_id  = tf.nn.embedding_lookup(emb_item_id,   ph_iid)   # [B, d]
ea_id  = tf.nn.embedding_lookup(emb_author_id, ph_aid)   # [B, d]
ei_cat = tf.nn.embedding_lookup(emb_category,  ph_cat)   # [B, d]

# 稠密特征直接 expand_dims 后拼入 E
ei_pop = tf.expand_dims(ph_item_popularity, -1)           # [B, 1]
ei_dur = tf.expand_dims(ph_item_duration,   -1)           # [B, 1]

# 主干 E
E = tf.concat([eu_id, ei_id, ea_id, ei_cat, ei_pop, ei_dur, ...], axis=1)

# PPNet prior 只取 ID 类
prior = tf.concat([eu_id, ei_id, ea_id], axis=1)          # 论文原始设置
```

---

## 5. 交叉特征

### 放置位置

- **DNN 第一层的输入**（与 EPNet 输出 O_ep concat 后一起进入 DNN）

### 设计原因

交叉特征（FM 二阶交叉、DIN 注意力分数、统计类 CTR 等）是**预计算好的显式特征交叉**，描述的是"这个用户和这个商品的匹配程度"，语义层次高于单独的用户或商品特征。

- 不放 Embedding 层：那里处理的是原始 ID，交叉特征已经是高层表示
- 不放门控 prior：门控只需要个性化身份信息，不需要匹配分数
- 放 DNN 输入：让网络在已有高阶信号的基础上继续学习，效率最高

注意：使用 EPNet 处理后的 `O_ep` 而不是原始 `E` 作为 DNN 输入起点，让域门控的个性化效果在交叉特征层就已经生效。

### 典型特征清单

| 特征 | 来源 | 说明 |
|------|------|------|
| FM 二阶交叉 | 离线/在线计算 | user embedding × item embedding 点积 |
| DIN 注意力分数 | 序列模型输出 | 用户历史序列对当前物品的注意力权重 |
| 历史 CTR | 统计特征 | 该用户在该类目下的历史点击率 |
| 历史 CVR | 统计特征 | 该用户的历史转化率 |
| 共现统计 | 统计特征 | user-item 共现次数、协同过滤分数 |

### 代码实现

```python
# 方式1: FM 二阶交叉（内积）
cross_fm = tf.reduce_sum(eu_id * ei_id, axis=-1, keepdims=True)  # [B, 1]

# 方式2: DIN 注意力分数（序列对当前物品的相关性）
cross_din = din_attention(eu_id, seq_emb, ei_id)   # [B, 1]

# 方式3: 统计类交叉特征（稠密标量）
cross_ctr = tf.expand_dims(ph_user_item_ctr, -1)   # [B, 1]

# 全部 concat 后接入 DNN 第一层
# O_ep 是 EPNet 处理后的个性化 Embedding，不是原始 E
dnn_input = tf.concat([O_ep, cross_fm, cross_din, cross_ctr], axis=1)

# DNN 第一层
h1 = dense(dnn_input, hidden_dim, 'dnn_l0', activation='relu')
# 随后 PPNet 在每层注入任务门控...
```

---

## 6. 上下文特征

### 放置位置

- **EPNet 门控输入（核心）**：domain_id、域统计特征（domain_user_cnt 等）
- **Embedding 层（辅助）**：time_of_day、position、request_type 等

### 设计原因

上下文特征描述的是"当前推荐发生在什么场景下"，与 EPNet 的使命完全吻合：**根据域信息对 Embedding 做个性化调整**。这是 PEPNet 设计中最精妙的映射关系。

- `domain_id` 是 EPNet 的核心输入（论文公式 5）
- `domain_user_cnt`（该用户在本域的历史行为次数）作为域门控的补充信号，帮助区分"新用户进入新域"和"老用户在熟悉域"两种场景
- 时段、位置等特征同时进主干 E，让 DNN 也能感知到它们的统计规律

### 典型特征清单

| 特征 | EPNet 门控 | 主干 E | 说明 |
|------|:----------:|:------:|------|
| domain_id | ✓（必须） | ✓ | 场景/Tab 标识 |
| domain_user_cnt | ✓ | | 用户在该域的历史行为次数 |
| domain_item_exp | ✓ | | 物品在该域的曝光次数 |
| time_of_day | | ✓ | 请求时段（早/中/晚/深夜） |
| day_of_week | | ✓ | 工作日 vs 周末 |
| request_position | | ✓ | 曝光位（首屏/次屏/...） |
| network_type | | ✓ | WiFi/4G/5G |

### 代码实现

```python
# domain 相关特征：EPNet 的核心输入
ed_domain = tf.nn.embedding_lookup(emb_domain, ph_did)      # [B, d]
cnt_dense  = tf.expand_dims(ph_domain_cnt, -1)              # [B, 1]
exp_dense  = tf.expand_dims(ph_domain_item_exp, -1)         # [B, 1]

# 时段、位置等：进主干 E
et_time = tf.nn.embedding_lookup(emb_time_bucket, ph_time)  # [B, d]
ep_pos  = tf.nn.embedding_lookup(emb_position,    ph_pos)   # [B, d]

# 主干 E 包含时段、位置等上下文
E = tf.concat([eu_id, ei_id, ..., et_time, ep_pos], axis=1)

# EPNet 门控输入 = domain embedding ⊕ 域统计特征 ⊕ stop_grad(E)
# 论文公式(5): δ_domain = Gate_NU(E(F_d) ⊕ stop_grad(E))
domain_feat = tf.concat([ed_domain, cnt_dense, exp_dense], axis=1)
epnet_input = tf.concat([domain_feat, tf.stop_gradient(E)], axis=1)
delta_domain = gate_nu(epnet_input, e_dim, scope='epnet_gate')
O_ep = delta_domain * E
```

---

## 7. 序列特征

### 放置位置

- **独立的序列编码器**（新增模块）→ 输出作为"动态用户向量"进入 **PPNet prior**

### 设计原因

这是现有 `pepnet_tf1.py` 代码中**唯一缺失的特征类型**，也是最有提升空间的改进点。

静态 `user_id` Embedding 表达的是用户的长期稳定偏好，但用户兴趣会随时间漂移——昨天刚看完科技视频的用户，今天更可能继续看科技内容。序列特征捕捉的正是这种**动态兴趣状态**。

序列特征不能简单拼入主干 E，原因是：
1. 序列长度可变，不能直接拼接
2. 顺序信息重要，需要专门处理
3. 需要与当前候选物品做相关性计算（target attention）

编码后的动态用户向量送入 PPNet prior，与静态 `user_id` 并列，让任务门控同时感知"长期偏好"和"即时兴趣"。

### 编码方案选择

| 方案 | 适用场景 | 复杂度 |
|------|---------|--------|
| Mean Pooling | 序列短、兴趣稳定 | 最低 |
| GRU/LSTM | 需要捕捉时序依赖 | 中 |
| Target Attention（DIN） | 需要与候选物品相关性 | 中 |
| Transformer（BST） | 长序列、复杂兴趣 | 高 |

### 代码实现（Target Attention / DIN 风格）

```python
def encode_sequence_din(seq_iids, seq_len, target_ei, emb_item,
                        max_len=50, scope='seq_enc'):
    """
    Target-Attention 序列编码（DIN 风格）

    参数
    ----
    seq_iids  : [B, L]  历史交互物品 ID 序列
    seq_len   : [B]     每个样本的真实序列长度（用于 mask padding）
    target_ei : [B, d]  当前候选物品的 Embedding
    emb_item  : [V, d]  物品 Embedding 表
    max_len   : int     序列最大长度 L

    返回
    ----
    user_dyn  : [B, d]  动态用户兴趣向量
    """
    # 序列中每个物品的 Embedding
    seq_emb = tf.nn.embedding_lookup(emb_item, seq_iids)   # [B, L, d]

    # Target Attention：计算每个历史物品与候选物品的相关性
    # [B, L, d] × [B, d, 1] → [B, L, 1] → [B, L]
    scores = tf.squeeze(
        tf.matmul(seq_emb, tf.expand_dims(target_ei, -1)),
        axis=-1)                                             # [B, L]

    # Mask padding 位置（填充 -inf，softmax 后权重接近 0）
    mask   = tf.sequence_mask(seq_len, maxlen=max_len, dtype=tf.float32)
    scores = scores * mask + (1.0 - mask) * (-1e9)

    # Softmax → 注意力权重
    attn = tf.nn.softmax(scores, axis=-1)                   # [B, L]

    # 加权求和 → 动态兴趣向量
    user_dyn = tf.reduce_sum(
        seq_emb * tf.expand_dims(attn, -1), axis=1)         # [B, d]

    return user_dyn


# ── 在 _lookup() 中集成 ──

# 序列占位符（新增）
ph_seq_iids = tf.placeholder(tf.int32,   [None, MAX_SEQ_LEN], 'seq_iids')
ph_seq_len  = tf.placeholder(tf.int32,   [None],              'seq_len')

# 编码序列 → 动态用户向量
user_dyn = encode_sequence_din(
    ph_seq_iids, ph_seq_len, ei_id, emb_item_id)            # [B, d]

# PPNet prior：静态 user_id + 动态序列向量 + item/author ID
# 动态向量替换或补充静态 user_id，让门控感知即时兴趣
prior = tf.concat([eu_id, user_dyn, ei_id, ea_id], axis=1)

# （可选）Layer Norm 稳定动态向量的量级
user_dyn = layer_norm(user_dyn, 'seq_ln')
```

### GRU 方案（适合更长的序列）

```python
def encode_sequence_gru(seq_emb, seq_len, scope='gru_enc'):
    """
    GRU 序列编码，取最后一个有效时间步的隐层状态
    seq_emb : [B, L, d]
    seq_len : [B]
    返回   : [B, hidden_dim]
    """
    cell = tf.nn.rnn_cell.GRUCell(num_units=hidden_dim)
    outputs, _ = tf.nn.dynamic_rnn(
        cell, seq_emb,
        sequence_length=seq_len,
        dtype=tf.float32,
        scope=scope)                   # outputs: [B, L, hidden_dim]

    # 取最后一个有效时间步（不是 padding 的最后一步）
    idx     = tf.stack([tf.range(tf.shape(seq_len)[0]),
                        seq_len - 1], axis=1)
    last_h  = tf.gather_nd(outputs, idx)  # [B, hidden_dim]
    return last_h
```

---

## 8. 完整放置汇总表

| 特征类型 | 具体示例 | Embedding 层 | EPNet 门控 | PPNet prior | DNN 输入 | 序列编码器 |
|---------|---------|:---:|:---:|:---:|:---:|:---:|
| **用户特征** | user_id | ✓ | | ✓ | | |
| | user_age, gender | ✓ | | ✓ | | |
| | user_level | ✓ | | | | |
| **商品特征** | item_id | ✓ | | ✓ | | |
| | author_id | ✓ | | ✓ | | |
| | item_category | ✓ | | | | |
| | item_popularity | ✓（稠密直接拼） | | | | |
| **交叉特征** | FM 二阶交叉 | | | | ✓ | |
| | DIN 注意力分数 | | | | ✓ | |
| | 历史 CTR/CVR | | | | ✓ | |
| **上下文特征** | domain_id | ✓ | ✓（必须）| | | |
| | domain_user_cnt | | ✓ | | | |
| | time_of_day, position | ✓ | | | | |
| **序列特征** | 历史点击序列 | | | ✓（编码后） | | ✓ |
| | 历史观看序列 | | | ✓（编码后） | | ✓ |

---

## 9. 代码集成示例

将五类特征整合到 `pepnet_tf1.py` 的完整 `_lookup_embeddings` 方法：

```python
def _lookup_embeddings(self):
    """
    五类特征的完整查询与组装
    """
    # ── 1. 用户特征 ──
    eu_id  = tf.nn.embedding_lookup(self.emb_user,   self.ph_uid)
    eu_age = tf.nn.embedding_lookup(self.emb_age,    self.ph_age)
    eu_gen = tf.nn.embedding_lookup(self.emb_gender, self.ph_gen)

    # ── 2. 商品特征 ──
    ei_id  = tf.nn.embedding_lookup(self.emb_item,   self.ph_iid)
    ea_id  = tf.nn.embedding_lookup(self.emb_author, self.ph_aid)
    ei_cat = tf.nn.embedding_lookup(self.emb_genre,  self.ph_gre)
    ei_pop = tf.expand_dims(self.ph_pop, -1)           # 稠密特征直接拼

    # ── 3. 上下文特征 ──
    ed_dom = tf.nn.embedding_lookup(self.emb_domain, self.ph_did)
    cnt    = tf.expand_dims(self.ph_cnt, -1)            # 域统计，进 EPNet
    et_time= tf.nn.embedding_lookup(self.emb_time,   self.ph_time)  # 进主干E

    # ── 4. 序列特征（新增）──
    user_dyn = encode_sequence_din(
        self.ph_seq_iids, self.ph_seq_len, ei_id,
        self.emb_item, scope='seq_enc')                # [B, d]
    user_dyn = layer_norm(user_dyn, 'seq_ln')

    # ── 5. 交叉特征（新增）──
    cross_fm  = tf.reduce_sum(eu_id * ei_id, axis=-1, keepdims=True)  # [B,1]
    cross_ctr = tf.expand_dims(self.ph_user_item_ctr, -1)             # [B,1]

    # ── 主干 Embedding E（用户 + 商品 + 上下文基础特征）──
    self.E = tf.concat([
        eu_id, ei_id, ea_id,      # ID 类
        eu_age, eu_gen, ei_cat,   # 属性类
        ei_pop, cnt,              # 稠密类
        et_time,                  # 上下文基础
    ], axis=1)                    # [B, n*d + m]

    # ── EPNet 输入（域门控：domain + 域统计）──
    self.domain_feat = tf.concat([ed_dom, cnt], axis=1)

    # ── PPNet prior（任务门控：user + item + author + 动态序列）──
    self.prior = tf.concat([
        eu_id, user_dyn,          # 静态长期 + 动态即时兴趣
        ei_id, ea_id,             # 物品、作者
    ], axis=1)

    # ── DNN 输入附加特征（交叉特征，在 EPNet 后 concat）──
    self.cross_feats = tf.concat([cross_fm, cross_ctr], axis=1)


# 在 _multi_task_dnn 中使用 cross_feats
def _multi_task_dnn(self, O_ep):
    # 交叉特征与 O_ep 合并后作为 DNN 第一层输入
    dnn_input = tf.concat([O_ep, self.cross_feats], axis=1)
    task_hiddens = [dnn_input] * self.n_tasks
    # ... 后续 PPNet 门控注入逻辑不变
```

### 新增占位符

```python
def _build_placeholders(self):
    # ... 原有占位符 ...

    # 序列特征（新增）
    self.ph_seq_iids = tf.placeholder(tf.int32,   [None, MAX_SEQ_LEN], 'seq_iids')
    self.ph_seq_len  = tf.placeholder(tf.int32,   [None],              'seq_len')

    # 上下文扩展（新增）
    self.ph_time     = tf.placeholder(tf.int32,   [None], 'time_bucket')
    self.ph_position = tf.placeholder(tf.int32,   [None], 'position')

    # 交叉特征（新增）
    self.ph_user_item_ctr = tf.placeholder(tf.float32, [None], 'user_item_ctr')
```

---

## 附：设计决策速查

**Q：交叉特征为什么不进 Embedding 层？**

Embedding 层处理的是原始离散 ID，目的是将稀疏的 one-hot 表示映射为稠密向量。交叉特征已经是高层匹配信号，直接进 Embedding 层意味着将这些特征再做一次低层嵌入，反而会损失信息。

**Q：序列特征为什么不直接 mean pooling 后进主干 E？**

Mean pooling 会丢失序列的顺序信息，也无法与当前候选物品做相关性计算。对于电商、短视频这类强序列依赖的场景，DIN 风格的 target attention 效果明显好于 mean pooling。

**Q：domain_id 为什么既进主干 E 又进 EPNet？**

两者的作用不同。进主干 E 是让 DNN 感知"当前域的统计规律"（例如该域的点击率基准值）；进 EPNet 是让门控网络知道"应该以哪个域的视角来调整 Embedding 的特征重要性"。前者是 DNN 的输入特征，后者是个性化调整的条件。

**Q：PPNet prior 里加入序列编码向量后，静态 user_id 还需要保留吗？**

需要。静态 user_id 编码的是长期稳定偏好（兴趣标签、人口属性），动态序列向量编码的是即时兴趣（最近看了什么），两者互补。实验中同时保留两者的效果通常优于单独使用任一种。

---

*文档更新时间：2026-03-28*
*对应代码文件：`pepnet_tf1.py`、`pepnet_v2.py`*
