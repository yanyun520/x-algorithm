// 目的：引入候选结构 PostCandidate。
// 影响：本过滤器校验并划分候选集合。
use crate::candidate_pipeline::candidate::PostCandidate;
// 目的：引入查询对象 ScoredPostsQuery。
// 影响：接口对齐需要（当前未读取查询内容）。
use crate::candidate_pipeline::query::ScoredPostsQuery;
// 目的：引入 tonic 异步 trait 支持。
// 影响：使 filter 方法成为异步方法。
use tonic::async_trait;
// 目的：引入过滤器 trait 与结果类型。
// 影响：让本类型以标准过滤器接入流水线。
use xai_candidate_pipeline::filter::{Filter, FilterResult};

// 目的：定义核心数据完整性过滤器（无内部状态）。
// 影响：剔除核心数据缺失的候选，保证下游内容可展示性。
pub struct CoreDataHydrationFilter;

// 目的：声明实现异步 Filter。
// 影响：流水线可在过滤阶段调用本过滤器。
#[async_trait]
impl Filter<ScoredPostsQuery, PostCandidate> for CoreDataHydrationFilter {
    // 目的：实现完整性过滤主逻辑。
    // 影响：剔除没有作者或没有正文的候选。
    async fn filter(
        // 目的：引用本过滤器状态（无内部字段）。
        // 影响：仅调用签名需要。
        &self,
        // 目的：接收查询对象（当前未使用，故命名为 _query）。
        // 影响：保持接口统一。
        _query: &ScoredPostsQuery,
        // 目的：接收待过滤候选列表（按值传入）。
        // 影响：本过滤器拥有所有权后直接分区。
        candidates: Vec<PostCandidate>,
    // 目的：声明返回值：过滤结果（保留+剔除）。
    // 影响：流水线据此更新候选集合。
    ) -> Result<FilterResult<PostCandidate>, String> {
        // 目的：按完整性条件分区候选。
        // 影响：具备作者且非空正文的候选被保留。
        let (kept, removed) = candidates
            // 目的：转迭代器消费候选。
            // 影响：逐个处理并移入对应分区。
            .into_iter()
            // 目的：分区谓词：作者非默认0 且正文去除空白后非空。
            // 影响：缺作者或缺正文的候选被剔除。
            .partition(|c| c.author_id != 0 && !c.tweet_text.trim().is_empty());
        // 目的：返回过滤结果。
        // 影响：脏数据候选不再参与后续打分与输出。
        Ok(FilterResult { kept, removed })
    }
}
