// 目的：引入候选结构 PostCandidate。
// 影响：本过滤器操作并划分候选集合。
use crate::candidate_pipeline::candidate::PostCandidate;
// 目的：引入查询对象 ScoredPostsQuery。
// 影响：本过滤器从查询中读取当前用户 ID。
use crate::candidate_pipeline::query::ScoredPostsQuery;
// 目的：引入 tonic 异步 trait 支持。
// 影响：使 filter 方法成为异步方法。
use tonic::async_trait;
// 目的：引入过滤器 trait 与结果类型。
// 影响：让本类型以标准过滤器接入流水线。
use xai_candidate_pipeline::filter::{Filter, FilterResult};

/// Filter that removes tweets where the author is the viewer.
// 目的：结构注释：剔除作者为当前用户本人的帖子。
// 影响：首页聚焦他人内容，不推荐用户自己的帖子。
pub struct SelfTweetFilter;

// 目的：声明实现异步 Filter。
// 影响：流水线可在过滤阶段调用本过滤器。
#[async_trait]
impl Filter<ScoredPostsQuery, PostCandidate> for SelfTweetFilter {
    // 目的：实现本人帖子过滤主逻辑。
    // 影响：剔除作者为当前查看者的候选。
    async fn filter(
        // 目的：引用本过滤器状态（无内部字段）。
        // 影响：仅调用签名需要。
        &self,
        // 目的：接收查询对象。
        // 影响：从中读取当前用户 ID。
        query: &ScoredPostsQuery,
        // 目的：接收待过滤候选列表（按值传入）。
        // 影响：本过滤器拥有所有权后直接分区。
        candidates: Vec<PostCandidate>,
    // 目的：声明返回值：过滤结果（保留+剔除）。
    // 影响：流水线据此更新候选集合。
    ) -> Result<FilterResult<PostCandidate>, String> {
        // 目的：将查询用户 ID 转为 u64。
        // 影响：与候选的 author_id 类型对齐以比较。
        let viewer_id = query.user_id as u64;
        // 目的：按作者是否为本人分区。
        // 影响：非本人作品保留，本人作品剔除。
        let (kept, removed): (Vec<_>, Vec<_>) = candidates
            // 目的：转迭代器消费候选。
            // 影响：逐个处理并移入对应分区。
            .into_iter()
            // 目的：分区谓词：作者 ID 不等于当前用户。
            // 影响：本人请假条进 kept，本人帖子进 removed。
            .partition(|c| c.author_id != viewer_id);

        // 目的：返回过滤结果。
        // 影响：用户自己的帖子不再进入打分与输出。
        Ok(FilterResult { kept, removed })
    }
}
