// 目的：引入候选结构 PostCandidate。
// 影响：本过滤器操作并划分候选集合。
use crate::candidate_pipeline::candidate::PostCandidate;
// 目的：引入查询对象 ScoredPostsQuery。
// 影响：本过滤器从查询中读取用户订阅列表。
use crate::candidate_pipeline::query::ScoredPostsQuery;
// 目的：引入 HashSet 集合类型。
// 影响：以 O(1) 复杂度判断订阅作者关系。
use std::collections::HashSet;
// 目的：引入 tonic 异步 trait 支持。
// 影响：使 filter 方法成为异步方法。
use tonic::async_trait;
// 目的：引入过滤器 trait 与结果类型。
// 影响：让本类型以标准过滤器接入流水线。
use xai_candidate_pipeline::filter::{Filter, FilterResult};

/// Filters out subscription-only posts from authors the viewer is not subscribed to.
// 目的：结构注释：剔除用户未订阅作者的专属订阅内容。
// 影响：保证仅展示用户有权限查看的付费内容。
pub struct IneligibleSubscriptionFilter;

// 目的：声明实现异步 Filter。
// 影响：流水线可在过滤阶段调用本过滤器。
#[async_trait]
impl Filter<ScoredPostsQuery, PostCandidate> for IneligibleSubscriptionFilter {
    // 目的：实现订阅资格过滤主逻辑。
    // 影响：剔除无订阅权限的付费内容候选。
    async fn filter(
        // 目的：引用本过滤器状态（无内部字段）。
        // 影响：仅调用签名需要。
        &self,
        // 目的：接收查询对象。
        // 影响：从中读取用户已订阅的作者 ID。
        query: &ScoredPostsQuery,
        // 目的：接收待过滤候选列表（按值传入）。
        // 影响：本过滤器拥有所有权后直接分区。
        candidates: Vec<PostCandidate>,
    // 目的：声明返回值：过滤结果（保留+剔除）。
    // 影响：流水线据此更新候选集合。
    ) -> Result<FilterResult<PostCandidate>, String> {
        // 目的：将用户的订阅作者 ID 转为 u64 集合。
        // 影响：快速判断候选订阅作者是否已被订阅。
        let subscribed_user_ids: HashSet<u64> = query
            // 目的：访问查询中的订阅特征。
            // 影响：订阅关系的数据来源。
            .user_features
            // 目的：迭代订阅作者 ID。
            // 影响：逐个处理每条订阅关系。
            .subscribed_user_ids
            // 目的：迭代元素。
            // 影响：为收集做准备。
            .iter()
            // 目的：将每个 ID 转为 u64。
            // 影响：与候选的作者 ID 类型对齐。
            .map(|id| *id as u64)
            // 目的：收集进 HashSet。
            // 影响：后续可 O(1) 判断订阅关系。
            .collect();

        // 目的：按订阅资格分区候选。
        // 影响：有权限的内容被保留，无权限的被剔除。
        let (kept, removed): (Vec<_>, Vec<_>) =
            candidates
                // 目的：转迭代器消费候选。
                // 影响：逐个处理并移入对应分区。
                .into_iter()
                // 目的：分区谓词：按订阅作者判断可见性。
                // 影响：未订阅作者的专属内容被剔除。
                .partition(|candidate| match candidate.subscription_author_id {
                    // 目的：候选存在订阅作者时，需用户已订阅该作者。
                    // 影响：未订阅则进入 removed。
                    Some(author_id) => subscribed_user_ids.contains(&author_id),
                    // 目的：候选无订阅作者时视为公开内容。
                    // 影响：无资格限制，直接保留。
                    None => true,
                });

        // 目的：返回过滤结果。
        // 影响：无订阅权限的付费内容从结果中消失。
        Ok(FilterResult { kept, removed })
    }
}
