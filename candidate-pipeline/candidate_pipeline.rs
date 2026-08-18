// =============================================================================
// candidate_pipeline.rs — 候选流水线核心执行引擎
// 职责：定义 CandidatePipeline trait 及其完整的默认 execute 实现，
//       串联推荐系统的全部阶段：
//         1. QueryHydrator   — 查询水合（并行）：填充用户特征、行为序列等
//         2. Source          — 候选召回（并行）：从 Thunder/Phoenix 等源拉取候选
//         3. Hydrator        — 候选水合（并行）：填充作者信息、视频时长等
//         4. Filter          — 过滤（顺序）：去重、年龄过滤、拉黑过滤等
//         5. Scorer          — 打分（顺序）：ML 模型打分、启发式加权
//         6. Selector        — 选择（单个）：按分数排序并截取 Top-K
//         7. PostSelectionHydrator — 后置水合（并行）：为已选候选补充展示字段
//         8. PostSelectionFilter   — 后置过滤（顺序）：最终裁剪
//         9. SideEffect      — 副作用（后台并行）：缓存、日志等不影响返回的动作
// 设计模式：
//   - trait 提供默认 execute 实现（模板方法模式），实现方只需提供组件列表
//   - 各阶段的并行/顺序执行模型由框架统一控制
// 边界情况说明：
//   - 所有组件均 fail-open：单个组件失败只记录错误，不清空整个结果
//   - Hydrator/Scorer 返回长度不符时丢弃其结果（防索引错位）
//   - Filter 失败时回滚到过滤前的候选集
//   - SideEffect 在后台 spawn 执行，不阻塞结果返回
// =============================================================================

// 各阶段 trait 定义
use crate::filter::Filter;
use crate::hydrator::Hydrator;
use crate::query_hydrator::QueryHydrator;
use crate::scorer::Scorer;
use crate::selector::Selector;
use crate::side_effect::{SideEffect, SideEffectInput};
use crate::source::Source;
// join_all 并发执行多个 future 并收集结果（用于并行阶段）
use futures::future::join_all;
use log::{error, info, warn};
// Arc 用于共享查询与副作用输入（避免多次 clone）
use std::sync::Arc;
use tonic::async_trait;

/// 流水线阶段枚举：用于日志中标识当前执行到的阶段
/// Copy + Clone + Debug：轻量值语义，可低成本传递与打印
#[derive(Copy, Clone, Debug)]
pub enum PipelineStage {
    /// 查询水合阶段（最先执行）
    QueryHydrator,
    /// 候选召回阶段
    Source,
    /// 候选水合阶段（召回后、过滤前）
    Hydrator,
    /// 后置水合阶段（选择后）
    PostSelectionHydrator,
    /// 过滤阶段（水合后、打分前）
    Filter,
    /// 后置过滤阶段（后置水合后）
    PostSelectionFilter,
    /// 打分阶段（过滤后、选择前）
    Scorer,
}

/// 流水线执行结果：携带各中间阶段的候选快照与最终结果
/// 泛型参数：Q — 查询类型；C — 候选类型
/// 边界：中间快照（retrieved/filtered）用于调试与指标分析，
///       会额外占用内存（候选被 clone 多次）
pub struct PipelineResult<Q, C> {
    /// 召回并水合后的候选（过滤前的快照）
    pub retrieved_candidates: Vec<C>,
    /// 所有过滤阶段（含后置过滤）移除的候选汇总
    pub filtered_candidates: Vec<C>,
    /// 最终选中的候选（已截断至 result_size）
    pub selected_candidates: Vec<C>,
    /// 已水合的查询（Arc 共享，供调用方与副作用使用）
    pub query: Arc<Q>,
}

/// 为查询提供稳定的请求标识（用于日志追踪与分布式 tracing）
/// 实现方（如 PipelineQuery）需提供 request_id 方法
pub trait HasRequestId {
    fn request_id(&self) -> &str;
}

/// 候选流水线核心 trait：定义流水线的组件组成与执行逻辑
/// 泛型参数：
///   - Q: 查询类型，必须实现 HasRequestId（日志追踪）且可 Clone
///   - C: 候选类型，必须可 Clone
/// 约束：Send + Sync + 'static — 可跨线程安全共享（tokio 多线程运行时必需）
/// 设计：execute 提供完整默认实现（模板方法），
///       实现方只需提供各阶段的组件列表（query_hydrators/sources/...）
#[async_trait]
pub trait CandidatePipeline<Q, C>: Send + Sync
where
    Q: HasRequestId + Clone + Send + Sync + 'static,
    C: Clone + Send + Sync + 'static,
{
    /// 查询水合器列表（并行执行）
    fn query_hydrators(&self) -> &[Box<dyn QueryHydrator<Q>>];
    /// 候选源列表（并行执行）
    fn sources(&self) -> &[Box<dyn Source<Q, C>>];
    /// 候选水合器列表（并行执行，召回后过滤前）
    fn hydrators(&self) -> &[Box<dyn Hydrator<Q, C>>];
    /// 过滤器列表（顺序执行，水合后打分前）
    fn filters(&self) -> &[Box<dyn Filter<Q, C>>];
    /// 打分器列表（顺序执行，过滤后选择前）
    fn scorers(&self) -> &[Box<dyn Scorer<Q, C>>];
    /// 选择器（单个，排序并截取 Top-K）
    fn selector(&self) -> &dyn Selector<Q, C>;
    /// 后置水合器列表（并行执行，选择后）
    fn post_selection_hydrators(&self) -> &[Box<dyn Hydrator<Q, C>>];
    /// 后置过滤器列表（顺序执行，后置水合后）
    fn post_selection_filters(&self) -> &[Box<dyn Filter<Q, C>>];
    /// 副作用列表（后台并行执行）
    /// 边界：返回 Arc<Vec<...>> 而非 &[...]——因为副作用要 move 进 spawn 的
    ///       后台任务，需要共享所有权而非借用
    fn side_effects(&self) -> Arc<Vec<Box<dyn SideEffect<Q, C>>>>;
    /// 最终结果数量上限（execute 末尾统一截断）
    fn result_size(&self) -> usize;

    /// 执行完整流水线（模板方法：固定阶段顺序，组件由实现方提供）
    /// 参数：query — 原始请求查询（所有权传入）
    /// 返回：PipelineResult，包含各阶段快照与最终选中候选
    /// 边界：任一组件失败均 fail-open（记录错误继续），不会中断整个流水线
    async fn execute(&self, query: Q) -> PipelineResult<Q, C> {
        // 阶段 1：查询水合（并行）——填充用户特征、行为序列等
        let hydrated_query = self.hydrate_query(query).await;

        // 阶段 2：候选召回（并行）——从各数据源拉取候选并合并
        let candidates = self.fetch_candidates(&hydrated_query).await;

        // 阶段 3：候选水合（并行）——填充作者信息、视频时长等
        let hydrated_candidates = self.hydrate(&hydrated_query, candidates).await;

        // 阶段 4：过滤（顺序）——划分保留/移除
        // 边界：传入 hydrated_candidates.clone()——原副本留给
        //       PipelineResult.retrieved_candidates 作为过滤前快照
        let (kept_candidates, mut filtered_candidates) = self
            .filter(&hydrated_query, hydrated_candidates.clone())
            .await;

        // 阶段 5：打分（顺序）——为候选计算分数
        let scored_candidates = self.score(&hydrated_query, kept_candidates).await;

        // 阶段 6：选择（单个）——按分数排序并截取 Top-K
        let selected_candidates = self.select(&hydrated_query, scored_candidates);

        // 阶段 7：后置水合（并行）——为已选候选补充展示字段
        let post_selection_hydrated_candidates = self
            .hydrate_post_selection(&hydrated_query, selected_candidates)
            .await;

        // 阶段 8：后置过滤（顺序）——最终裁剪
        let (mut final_candidates, post_selection_filtered_candidates) = self
            .filter_post_selection(&hydrated_query, post_selection_hydrated_candidates)
            .await;
        // 将后置过滤移除的候选并入总移除列表（供调试/指标）
        filtered_candidates.extend(post_selection_filtered_candidates);

        // 最终截断：确保结果不超过 result_size
        // 边界：truncate 超出部分被静默丢弃（不计入 filtered_candidates）
        final_candidates.truncate(self.result_size());

        // 将查询包装为 Arc 共享（供副作用与返回结果复用，避免多次 clone）
        let arc_hydrated_query = Arc::new(hydrated_query);
        // 构造副作用输入（查询 + 最终候选的克隆）
        let input = Arc::new(SideEffectInput {
            query: arc_hydrated_query.clone(),
            selected_candidates: final_candidates.clone(),
        });
        // 阶段 9：副作用（后台并行，不阻塞返回）
        self.run_side_effects(input);

        // 组装并返回流水线结果
        PipelineResult {
            retrieved_candidates: hydrated_candidates,
            filtered_candidates,
            selected_candidates: final_candidates,
            query: arc_hydrated_query,
        }
    }

    /// Run all query hydrators in parallel and merge results into the query.
    async fn hydrate_query(&self, query: Q) -> Q {
        let request_id = query.request_id().to_string();
        let hydrators: Vec<_> = self
            .query_hydrators()
            .iter()
            .filter(|h| h.enable(&query))
            .collect();
        let hydrate_futures = hydrators.iter().map(|h| h.hydrate(&query));
        let results = join_all(hydrate_futures).await;

        let mut hydrated_query = query;
        for (hydrator, result) in hydrators.iter().zip(results) {
            match result {
                Ok(hydrated) => {
                    hydrator.update(&mut hydrated_query, hydrated);
                }
                Err(err) => {
                    error!(
                        "request_id={} stage={:?} component={} failed: {}",
                        request_id,
                        PipelineStage::QueryHydrator,
                        hydrator.name(),
                        err
                    );
                }
            }
        }
        hydrated_query
    }

    /// Run all candidate sources in parallel and collect results.
    async fn fetch_candidates(&self, query: &Q) -> Vec<C> {
        let request_id = query.request_id().to_string();
        let sources: Vec<_> = self.sources().iter().filter(|s| s.enable(query)).collect();
        let source_futures = sources.iter().map(|s| s.get_candidates(query));
        let results = join_all(source_futures).await;

        let mut collected = Vec::new();
        for (source, result) in sources.iter().zip(results) {
            match result {
                Ok(mut candidates) => {
                    info!(
                        "request_id={} stage={:?} component={} fetched {} candidates",
                        request_id,
                        PipelineStage::Source,
                        source.name(),
                        candidates.len()
                    );
                    collected.append(&mut candidates);
                }
                Err(err) => {
                    error!(
                        "request_id={} stage={:?} component={} failed: {}",
                        request_id,
                        PipelineStage::Source,
                        source.name(),
                        err
                    );
                }
            }
        }
        collected
    }

    /// Run all candidate hydrators in parallel and merge results into candidates.
    async fn hydrate(&self, query: &Q, candidates: Vec<C>) -> Vec<C> {
        self.run_hydrators(query, candidates, self.hydrators(), PipelineStage::Hydrator)
            .await
    }

    /// Run post-selection candidate hydrators in parallel and merge results into candidates.
    async fn hydrate_post_selection(&self, query: &Q, candidates: Vec<C>) -> Vec<C> {
        self.run_hydrators(
            query,
            candidates,
            self.post_selection_hydrators(),
            PipelineStage::PostSelectionHydrator,
        )
        .await
    }

    /// Shared helper to hydrate with a provided hydrator list.
    async fn run_hydrators(
        &self,
        query: &Q,
        mut candidates: Vec<C>,
        hydrators: &[Box<dyn Hydrator<Q, C>>],
        stage: PipelineStage,
    ) -> Vec<C> {
        let request_id = query.request_id().to_string();
        let hydrators: Vec<_> = hydrators.iter().filter(|h| h.enable(query)).collect();
        let expected_len = candidates.len();
        let hydrate_futures = hydrators.iter().map(|h| h.hydrate(query, &candidates));
        let results = join_all(hydrate_futures).await;
        for (hydrator, result) in hydrators.iter().zip(results) {
            match result {
                Ok(hydrated) => {
                    if hydrated.len() == expected_len {
                        hydrator.update_all(&mut candidates, hydrated);
                    } else {
                        warn!(
                            "request_id={} stage={:?} component={} skipped: length_mismatch expected={} got={}",
                            request_id,
                            stage,
                            hydrator.name(),
                            expected_len,
                            hydrated.len()
                        );
                    }
                }
                Err(err) => {
                    error!(
                        "request_id={} stage={:?} component={} failed: {}",
                        request_id,
                        stage,
                        hydrator.name(),
                        err
                    );
                }
            }
        }
        candidates
    }

    /// Run all filters sequentially. Each filter partitions candidates into kept and removed.
    async fn filter(&self, query: &Q, candidates: Vec<C>) -> (Vec<C>, Vec<C>) {
        self.run_filters(query, candidates, self.filters(), PipelineStage::Filter)
            .await
    }

    /// Run post-scoring filters sequentially on already-scored candidates.
    async fn filter_post_selection(&self, query: &Q, candidates: Vec<C>) -> (Vec<C>, Vec<C>) {
        self.run_filters(
            query,
            candidates,
            self.post_selection_filters(),
            PipelineStage::PostSelectionFilter,
        )
        .await
    }

    // Shared helper to run filters sequentially from a provided filter list.
    async fn run_filters(
        &self,
        query: &Q,
        mut candidates: Vec<C>,
        filters: &[Box<dyn Filter<Q, C>>],
        stage: PipelineStage,
    ) -> (Vec<C>, Vec<C>) {
        let request_id = query.request_id().to_string();
        let mut all_removed = Vec::new();
        for filter in filters.iter().filter(|f| f.enable(query)) {
            let backup = candidates.clone();
            match filter.filter(query, candidates).await {
                Ok(result) => {
                    candidates = result.kept;
                    all_removed.extend(result.removed);
                }
                Err(err) => {
                    error!(
                        "request_id={} stage={:?} component={} failed: {}",
                        request_id,
                        stage,
                        filter.name(),
                        err
                    );
                    candidates = backup;
                }
            }
        }
        info!(
            "request_id={} stage={:?} kept {}, removed {}",
            request_id,
            stage,
            candidates.len(),
            all_removed.len()
        );
        (candidates, all_removed)
    }

    /// Run all scorers sequentially and apply their results to candidates.
    async fn score(&self, query: &Q, mut candidates: Vec<C>) -> Vec<C> {
        let request_id = query.request_id().to_string();
        let expected_len = candidates.len();
        for scorer in self.scorers().iter().filter(|s| s.enable(query)) {
            match scorer.score(query, &candidates).await {
                Ok(scored) => {
                    if scored.len() == expected_len {
                        scorer.update_all(&mut candidates, scored);
                    } else {
                        warn!(
                            "request_id={} stage={:?} component={} skipped: length_mismatch expected={} got={}",
                            request_id,
                            PipelineStage::Scorer,
                            scorer.name(),
                            expected_len,
                            scored.len()
                        );
                    }
                }
                Err(err) => {
                    error!(
                        "request_id={} stage={:?} component={} failed: {}",
                        request_id,
                        PipelineStage::Scorer,
                        scorer.name(),
                        err
                    );
                }
            }
        }
        candidates
    }

    /// Select (sort/truncate) candidates using the configured selector
    fn select(&self, query: &Q, candidates: Vec<C>) -> Vec<C> {
        if self.selector().enable(query) {
            self.selector().select(query, candidates)
        } else {
            candidates
        }
    }

    // Run all side effects in parallel
    fn run_side_effects(&self, input: Arc<SideEffectInput<Q, C>>) {
        let side_effects = self.side_effects();
        tokio::spawn(async move {
            let futures = side_effects
                .iter()
                .filter(|se| se.enable(input.query.clone()))
                .map(|se| se.run(input.clone()));
            let _ = join_all(futures).await;
        });
    }
}
