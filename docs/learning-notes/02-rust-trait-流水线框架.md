# 02 · Rust Trait 插件式流水线框架

> 核心知识点:用 Rust trait 系统实现**模板方法模式 + 策略模式**的组合框架;`async_trait`、`dyn` 对象与泛型的取舍;差异化的并行/顺序执行策略。

源码位置:`candidate-pipeline/`(框架)+ `home-mixer/candidate_pipeline/phoenix_candidate_pipeline.rs`(实例化)。

---

## 1. 框架核心:一个 trait 固化流程,八类组件自由插拔

`CandidatePipeline<Q, C>` 泛型化于查询类型 `Q` 与候选类型 `C`,通过 8 个 getter 暴露挂载点:

```rust
#[async_trait]
pub trait CandidatePipeline<Q, C>: Send + Sync {
    fn query_hydrators(&self) -> &[Box<dyn QueryHydrator<Q>>];
    fn sources(&self) -> &[Box<dyn Source<Q, C>>];
    fn hydrators(&self) -> &[Box<dyn Hydrator<Q, C>>];
    fn filters(&self) -> &[Box<dyn Filter<Q, C>>];
    fn scorers(&self) -> &[Box<dyn Scorer<Q, C>>];
    fn selector(&self) -> &dyn Selector<Q, C>;
    fn post_selection_hydrators(&self) -> &[Box<dyn Hydrator<Q, C>>];
    fn post_selection_filters(&self) -> &[Box<dyn Filter<Q, C>>];
    fn side_effects(&self) -> Arc<Vec<Box<dyn SideEffect<Q, C>>>>;
    fn result_size(&self) -> usize;

    async fn execute(&self, query: Q) -> PipelineResult<Q, C> { /* 模板方法 */ }
}
```

**设计精髓**:

1. **模板方法模式**:`execute()` 是 trait 的默认方法,固化执行顺序(水合→召回→水合→过滤→打分→选择→后处理),实现方**只能提供组件,无法改变流程语义**。框架逻辑(并行调度、错误处理、日志)与业务逻辑(组件实现)彻底解耦;
2. **静态分发 vs 动态分发**:Pipeline 本身用泛型(`Q, C`),编译期单态化零开销;组件集合用 `Box<dyn Trait>`,运行期多态换取"异构组件装入同一 Vec"的灵活性。这是 Rust 框架设计的经典搭配——**性能关键路径用泛型,扩展点用 trait 对象**。

## 2. 组件 trait 的统一协议

每个组件 trait 遵循同一套协议,学习这套协议就掌握了整个框架:

```rust
#[async_trait]
pub trait Filter<Q, C>: Any + Send + Sync {
    fn enable(&self, _query: &Q) -> bool { true }          // ① 动态开关
    async fn filter(&self, query: &Q, candidates: Vec<C>)
        -> Result<FilterResult<C>, String>;                 // ② 核心逻辑
    fn name(&self) -> &'static str {                        // ③ 自描述名称
        util::short_type_name(type_name_of_val(self))
    }
}
```

- **`enable()` 默认 true**:支持按请求特征动态跳过组件(灰度、实验、降级)的零成本挂载点;
- **`name()` 默认实现**:用 `type_name_of_val` 反射出类型短名,日志/metrics 自动获得可读组件标识,无需手写;
- **`Result<_, String>`**:错误上抛给框架统一处理(记录而不中断),组件不用关心容错策略。

### Scorer 的"只更新自己字段"契约

```rust
pub trait Scorer<Q, C>: Send + Sync {
    async fn score(&self, query: &Q, candidates: &[C]) -> Result<Vec<C>, String>;
    fn update(&self, candidate: &mut C, scored: C);   // 只拷贝本 scorer 负责的字段
    fn update_all(&self, candidates: &mut [C], scored: Vec<C>) { /* 默认 zip 遍历 */ }
}
```

注意 `score` 的文档约定:**"返回的 Vec 必须与输入等长同序,不允许在 scorer 中丢候选——丢候选请用 Filter"**。这是框架用**文档契约 + 长度校验**(`run_hydrators` 中的 `length_mismatch` 检查)共同维护的不变量,防止打分阶段静默丢数据。

`update` 模式让多个 scorer 可以各自往候选上写不同字段(PhoenixScorer 写 `phoenix_scores`,WeightedScorer 写 `weighted_score`...),互不覆盖,实现**特征字段的增量累积**。

## 3. 差异化并行策略(框架的性能内核)

框架并非"能并行就并行",而是按**组件间的依赖语义**选择执行方式:

| 阶段 | 执行方式 | 原因 |
|------|----------|------|
| QueryHydrator | `join_all` 并行 | 各水合器读写查询的不同字段,互不依赖 |
| Source | `join_all` 并行 | 召回源相互独立,延迟取 max 而非 sum |
| Hydrator | `join_all` 并行 | 每个 hydrator 独立处理全量候选再按位合并 |
| Filter | 顺序循环 | 前一个的输出是后一个的输入;`FilterResult` 分区语义需要依次传递 |
| Scorer | 顺序循环 | 链式依赖:Weighted 依赖 Phoenix 写入的分数字段 |

```rust
// 并行:fetch_candidates
let source_futures = sources.iter().map(|s| s.get_candidates(query));
let results = join_all(source_futures).await;

// 顺序:run_filters
for filter in filters.iter().filter(|f| f.enable(query)) {
    let backup = candidates.clone();
    match filter.filter(query, candidates).await {
        Ok(result) => { candidates = result.kept; ... }
        Err(err) => { candidates = backup; /* 回滚 */ }
    }
}
```

Filter 顺序执行时还做了**失败回滚**:先 `clone()` 备份,过滤器失败则恢复备份并继续后续过滤——单个过滤器故障不影响整条链。

## 4. Selector 的默认行为下沉

```rust
pub trait Selector<Q, C>: Send + Sync {
    fn select(&self, _query: &Q, candidates: Vec<C>) -> Vec<C> {
        let mut sorted = self.sort(candidates);
        if let Some(limit) = self.size() { sorted.truncate(limit); }
        sorted
    }
    fn score(&self, candidate: &C) -> f64;            // 唯一必须实现
    fn size(&self) -> Option<usize> { None }          // 可选
}
```

实现方(如 `TopKScoreSelector`)只需提供"分数从哪取"和"取几个",排序、截断、`enable` 全部由默认方法给出。**把尽可能多的行为做成默认方法**,是降低实现方负担的关键技巧。

## 5. 实例化:依赖注入 + 环境分离

`PhoenixCandidatePipeline::build_with_clients(...)` 接收所有外部客户端的 `Arc<dyn Trait>` 参数,`prod()` 负责创建生产实现后调用它。收益:

- **可测试性**:测试时注入 mock client,无需改一行组装代码;
- **可替换性**:客户端接口 trait 化(`PhoenixPredictionClient`、`StratoClient`...),换实现不换流水线。

## 6. 可迁移场景

- 任何**多阶段数据处理管线**(ETL、审核流、风控决策流)都可套用这套"trait 组件 + 模板方法 + 差异化并行"结构;
- `enable()` + `name()` + `Result<_, String>` 的统一协议是**插件系统**的最小完备设计;
- 学习重点:Rust 中"泛型做性能路径、dyn 做扩展点、默认方法做行为下沉"的三板斧。

---

**上一篇**:[01 · 总体架构](01-总体架构与推荐系统范式.md) | **下一篇**:[03 · Rust 并发编程模式](03-rust-并发编程模式.md)
