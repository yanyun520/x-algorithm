// 目的：引入候选结构 PostCandidate。
// 影响：本过滤器操作并划分候选集合。
use crate::candidate_pipeline::candidate::PostCandidate;
// 目的：引入查询对象 ScoredPostsQuery。
// 影响：接口对齐需要（当前未读取查询内容）。
use crate::candidate_pipeline::query::ScoredPostsQuery;
// 目的：引入 HashSet 集合类型。
// 影响：以 O(1) 复杂度判断帖子 ID 是否重复。
use std::collections::HashSet;
// 目的：引入 tonic 异步 trait 支持。
// 影响：使 filter 方法成为异步方法。
use tonic::async_trait;
// 目的：引入过滤器 trait 与结果类型。
// 影响：让本类型以标准过滤器接入流水线。
use xai_candidate_pipeline::filter::{Filter, FilterResult};

// 目的：定义重复帖过滤器（无内部状态）。
// 影响：去除同一 tweet_id 的重复候选。
pub struct DropDuplicatesFilter;

// 目的：声明实现异步 Filter。
// 影响：流水线可在过滤阶段调用本过滤器。
#[async_trait]
impl Filter<ScoredPostsQuery, PostCandidate> for DropDuplicatesFilter {
    // 目的：实现重复剔除主逻辑。
    // 影响：同一帖子仅保留首次出现的候选。
    async fn filter(
        // 目的：引用本过滤器状态（无内部字段）。
        // 影响：仅调用签名需要。
        &self,
        // 目的：接收查询对象（当前未使用，故命名为 _query）。
        // 影响：保持接口统一。
        _query: &ScoredPostsQuery,
        // 目的：接收待过滤候选列表（按值传入）。
        // 影响：本过滤器拥有所有权后逐条判定。
        candidates: Vec<PostCandidate>,
    // 目的：声明返回值：过滤结果（保留+剔除）。
    // 影响：流水线据此更新候选集合。
    ) -> Result<FilterResult<PostCandidate>, String> {
        // 目的：创建已见帖子 ID 集合。
        // 影响：用于追踪哪些帖子已经出现过。
        let mut seen_ids = HashSet::new();
        // 目的：初始化保留集合。
        // 影响：存放首次出现的候选。
        let mut kept = Vec::new();
        // 目的：初始化剔除集合。
        // 影响：存放重复出现的候选。
        let mut removed = Vec::new();

        // 目的：遍历每个候选。
        // 影响：逐条判断帖子 ID 是否重复。
        for candidate in candidates {
            // 目的：尝试将 ID 插入已见集合（insert 返回是否为新元素）。
            // 影响：首次出现返回 true，重复出现返回 false。
            if seen_ids.insert(candidate.tweet_id) {
                // 目的：首次出现则保留。
                // 影响：该候选进入后续阶段。
                kept.push(candidate);
            } else {
                // 目的：重复出现则剔除。
                // 影响：该候选不再参与评分与输出。
                removed.push(candidate);
            }
        }

        // 目的：返回过滤结果。
        // 影响：结果中不再包含重复帖子。
        Ok(FilterResult { kept, removed })
    }
}
