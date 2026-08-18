# 07 · Phoenix ML 模型架构

> 核心知识点:**两塔检索模型**、**Candidate Isolation 注意力掩码**、**多哈希 Embedding**、L2 归一化与点积检索、Grok-1 Transformer 的推荐系统改造。

源码位置:`phoenix/recsys_model.py`、`phoenix/recsys_retrieval_model.py`(JAX + Haiku 实现)。

---

## 1. 零手工特征的哲学

传统推荐系统依赖大量人工特征(作者历史 CTR、内容分类、时间衰减...)。Phoenix 的宣言是:**所有相关性信号由 Transformer 从用户行为序列中自动学习**。输入只有:

- 用户哈希、行为历史(帖哈希 + 作者哈希 + 动作类型 + 产品界面);
- 候选(帖哈希 + 作者哈希 + 产品界面)。

收益:砍掉特征工程管线和特征存储,模型升级即能力升级;代价:表达能力完全押注在模型与数据上。

## 2. 检索阶段:两塔模型(Two-Tower)

目标:从百万级语料中毫秒级找出与用户最相关的几百条。关键是把"相似度计算"转化为**可预先建索引的向量检索**:

```
用户侧:  user_hashes + history ──Transformer(User Tower)──▶ user_vec [D]  ──L2归一化──┐
                                                                                     ├─ dot product ─ top-K
候选侧:  post+author 哈希 emb ──MLP(Candidate Tower)──▶ item_vec [N, D] ──L2归一化──┘
```

`CandidateTower` 的实现:

```python
hidden = jnp.dot(post_author_embedding, proj_1)   # 升维
hidden = jax.nn.silu(hidden)                       # 非线性
candidate_embeddings = jnp.dot(hidden, proj_2)     # 投影到共享空间
candidate_representation = candidate_embeddings / candidate_norm   # L2 归一化
```

**为什么归一化**:两向量 L2 归一化后,点积 = 余弦相似度。候选塔输出可**离线批量算好并灌入 ANN 索引**(如 FAISS/HNSW),在线只需算用户向量 + 一次近邻查询,把"百万次打分"变成"一次检索"。

**架构不对称的合理性**:User Tower 用 Transformer(要理解行为序列上下文,表达力优先),Candidate Tower 用轻量 MLP(要对全库帖离线计算,成本优先)。

## 3. 精排阶段:Candidate Isolation 注意力掩码

排序模型把输入拼成一条序列:`[User(1) | History(S) | Candidates(C)]`,然后用**定制注意力掩码**控制信息流动:

| Query \ Key | User | History | Candidates |
|---|---|---|---|
| User | ✓ | ✓ | ✗ |
| History | ✓ | ✓ | ✗ |
| Candidate_i | ✓ | ✓ | **仅自己(对角线)** |

实现上由三段 padding mask 拼接 + candidate 起始偏移构造:

```python
padding_mask = jnp.concatenate(
    [user_padding_mask, history_padding_mask, candidate_padding_mask], axis=1)
candidate_start_offset = user_padding_mask.shape[1] + history_padding_mask.shape[1]
```

**为什么要隔离候选**:如果候选之间可以互相 attend,那么某个候选的分数会依赖"同批还有哪些候选"——同一篇帖在两个不同 batch 里得分不同。隔离后:

1. **分数确定性**:候选分只取决于 (用户, 该候选),与 batch 组成无关;
2. **可缓存**:分数可以按 (user, post) 缓存复用,不必每次重算整个 batch;
3. **可增量**:增删候选不需要重跑其他候选的推理。

这是用一点表达力(放弃"候选间相对比较")换取巨大的工程收益——**模型设计服从于服务架构**的典范。

## 4. 多哈希 Embedding:免词表的 ID 表示

推荐系统的 ID 空间(用户、帖、作者)是亿级且持续增长的,传统 embedding table 需要维护"ID → 行号"的词表映射服务,且新实体无法即时表示。Phoenix 的解法:

```python
class HashConfig:
    num_user_hashes: int = 2      # 每个实体用 2 个哈希函数
    num_item_hashes: int = 2
    num_author_hashes: int = 2
```

- 每个 ID 经多个哈希函数映射到固定大小的表,查出多个 embedding;
- `block_user_reduce` / `block_history_reduce` / `block_candidate_reduce` 把多哈希 embedding 拼接后**线性投影**融合成单向量;
- 多哈希缓解冲突(两个 ID 在一个哈希下撞车的概率,远低于同时撞所有哈希);
- **hash 0 保留为 padding**:`user_padding_mask = (user_hashes[:, 0] != 0)`,无效位不进入注意力。

本质上这是 **hashing trick + 多哈希去冲突 + 可学习融合** 的组合,用极小的表支撑无限 ID 空间,新帖/新用户零成本接入。

## 5. 多动作输出头

模型输出 `[B, num_candidates, num_actions]` 的 logits,覆盖 15+ 动作(离散概率 + 连续值如 dwell_time)。多头共享同一个 Transformer trunk——**多任务学习**让各动作的相关信号互相增益(喜欢与分享在表征层面强相关),同时一次推理产出排序所需的全部信号(见文档 06 的加权合成)。

## 6. Grok-1 移植的工程含义

Transformer 核心(注意力、层归一化等)直接移植自 xAI 开源的 Grok-1,说明:**推荐排序与语言建模在架构上已经统一**——都是"给定上下文序列,预测下一个/每个单元的输出分布"。差异只在输入构造(哈希 embedding 替代 token embedding)与掩码设计(candidate isolation 替代因果掩码)。这提示一个趋势:LLM 架构技术(RoPE、MoE、KV cache 等)可以持续迁移到推荐排序模型。

## 7. 可迁移场景

- **检索系统**:两塔 + L2 归一化 + ANN,是从语义搜索到广告召回的通用方案;
- **需要分数可缓存/可复现的排序**:candidate isolation 掩码思路;
- **高基数 ID 特征的表示**:多哈希 embedding 免词表方案;
- **多目标预估**:单 trunk 多输出头的多任务结构。

---

**上一篇**:[06 · 推荐打分与排序策略](06-推荐打分与排序策略.md) | **下一篇**:[08 · 错误处理与可观测性](08-错误处理与可观测性.md)
