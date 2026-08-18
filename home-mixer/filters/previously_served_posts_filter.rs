// 目的：引入候选结构 PostCandidate。
// 影响：本过滤器操作并划分候选集合。
use crate::candidate_pipeline::candidate::PostCandidate;
// 目的：引入查询对象 ScoredPostsQuery。
// 影响：本过滤器从查询中读取已服务 ID 与请求类型标记。
use crate::candidate_pipeline::query::ScoredPostsQuery;
// 目的：引入候选关联 ID 提取工具。
// 影响：获取候选涉及的帖子 ID 以匹配已服务列表。
use crate::util::candidates_util::get_related_post_ids;
// 目的：引入 tonic 异步 trait 支持。
// 影响：使 filter 方法成为异步方法。
use tonic::async_trait;
// 目的：引入过滤器 trait 与结果类型。
// 影响：让本类型以标准过滤器接入流水线。
use xai_candidate_pipeline::filter::{Filter, FilterResult};

// 目的：定义已服务帖子过滤器（无内部状态）。
// 影响：剔除近期已下发给用户的帖子，减少重复展示。
pub struct PreviouslyServedPostsFilter;

// 目的：声明实现异步 Filter。
// 影响：流水线可在过滤阶段调用本过滤器。
#[async_trait]
impl Filter<ScoredPostsQuery, PostCandidate> for PreviouslyServedPostsFilter {
    // 目的：定义启用条件：仅在下拉加载更多请求时启用。
    // 影响：首次请求不做已服务过滤，避免过度裁剪。
    fn enable(&self, query: &ScoredPostsQuery) -> bool {
        // 目的：返回请求是否为 bottom（加载更多）类型。
        // 影响：仅加载更多场景执行本过滤。
        query.is_bottom_request
    }

    // 目的：实现已服务过滤主逻辑。
    // 影响：剔除已服务列表中的候选。
    async fn filter(
        // 目的：引用本过滤器状态（无内部字段）。
        // 影响：仅调用签名需要。
        &self,
        // 目的：接收查询对象。
        // 影响：从中读取已服务 ID 列表。
        query: &ScoredPostsQuery,
        // 目的：接收待过滤候选列表（按值传入）。
        // 影响：本过滤器拥有所有权后直接分区。
        candidates: Vec<PostCandidate>,
    // 目的：声明返回值：过滤结果（保留+剔除）。
    // 影响：流水线据此更新候选集合。
    ) -> Result<FilterResult<PostCandidate>, String> {
        // 目的：按是否属于已服务集合分区候选。
        // 影响：已服务候选进 removed（剔除），未服务进 kept（保留）。
        let (removed, kept): (Vec<_>, Vec<_>) = candidates.into_iter().partition(|c| {
            // 目的：取候选关联帖子 ID 并判断任一是否已服务。
            // 影响：关联帖已服务则整条候选视为重复。
            get_related_post_ids(c)
                // 目的：迭代关联 ID。
                // 影响：逐一比对。
                .iter()
                // 目的：判断关联 ID 是否在已服务列表中。
                // 影响：命中即视为已服务。
                .any(|id| query.served_ids.contains(id))
        });

        // 目的：返回过滤结果。
        // 影响：近期已下发的帖子不再重复进入结果。
        Ok(FilterResult { kept, removed })
    }
}
