# x-algorithm 学习笔记索引

本系列文档从 **x-algorithm**(X "For You" 推荐系统)源码中提炼出值得学习的知识点，按主题分类整理。每篇文档包含：原理说明、源码实例、设计动机分析与可迁移场景。

## 文档目录

| # | 文档 | 主题 | 难度 |
|---|------|------|------|
| 01 | [总体架构与推荐系统范式](01-总体架构与推荐系统范式.md) | 两阶段推荐、双路召回、组件职责划分 | ⭐ |
| 02 | [Rust Trait 插件式流水线框架](02-rust-trait-流水线框架.md) | trait 设计、模板方法、泛型与 dyn 对象、并行策略 | ⭐⭐⭐ |
| 03 | [Rust 并发编程模式](03-rust-并发编程模式.md) | DashMap、Arc、tokio、Semaphore 背压、spawn_blocking | ⭐⭐⭐ |
| 04 | [Kafka 实时流处理工程实践](04-kafka-实时流处理.md) | 分区并行消费、攒批、追赶检测、失败哲学 | ⭐⭐⭐ |
| 05 | [内存数据结构设计](05-内存数据结构设计.md) | 索引/时间线分离、引用最小化、墓碑机制、保留策略 | ⭐⭐ |
| 06 | [推荐打分与排序策略](06-推荐打分与排序策略.md) | 多动作加权、作者多样性衰减、OON 折扣、双阶段过滤 | ⭐⭐ |
| 07 | [Phoenix ML 模型架构](07-phoenix-ml-模型架构.md) | 两塔检索、Candidate Isolation、Hash Embedding | ⭐⭐⭐⭐ |
| 08 | [错误处理与可观测性](08-错误处理与可观测性.md) | 组件级容错、结构化日志、防御性校验、Metrics | ⭐⭐ |

## 阅读建议

- **推荐系统入门**：按 01 → 06 → 07 顺序阅读，建立"召回 → 过滤 → 精排 → 重排"的完整认知;
- **Rust 工程进阶**：按 02 → 03 → 04 → 05 顺序阅读，重点学习 trait 抽象与并发原语的组合使用;
- **系统设计者**：01 与 08 必读，理解组件职责边界与生产级容错思路。

## 源码导航速查

```
candidate-pipeline/     通用流水线框架(trait 定义 + execute 模板方法)
home-mixer/             编排层:gRPC 服务 + PhoenixCandidatePipeline 组装
  ├── query_hydrators/  查询级数据水合(UAS 行为序列、用户特征)
  ├── sources/          候选召回(Thunder / Phoenix Retrieval)
  ├── candidate_hydrators/ 候选级水合(核心数据、作者、视频时长、订阅)
  ├── filters/          打分前过滤链
  ├── scorers/          打分链(Phoenix → Weighted → Diversity → OON)
  ├── selectors/        TopK 选择器
  └── side_effects/     异步副作用(缓存已服务帖)
thunder/                实时内存帖库(Kafka 摄取 + PostStore + gRPC 查询)
phoenix/                ML 模型(JAX/Haiku:两塔检索 + 排序 Transformer)
```
