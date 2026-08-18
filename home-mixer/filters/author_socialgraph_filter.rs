// 目的：引入候选结构 PostCandidate。
// 影响：本过滤器操作并划分候选集合。
use crate::candidate_pipeline::candidate::PostCandidate;
// 目的：引入查询对象 ScoredPostsQuery。
// 影响：本过滤器从查询中读取拉黑/静音列表。
use crate::candidate_pipeline::query::ScoredPostsQuery;
// 目的：引入 tonic 异步 trait 支持。
// 影响：使 filter 方法成为异步方法。
use tonic::async_trait;
// 目的：引入过滤器 trait 与结果类型。
// 影响：让本类型以标准过滤器接入流水线。
use xai_candidate_pipeline::filter::{Filter, FilterResult};

// Remove candidates that are blocked or muted by the viewer
// 目的：结构注释：剔除被查看用户拉黑或静音的作者内容。
// 影响：尊重用户的社交设置，不推荐其明确拒绝的内容。
pub struct AuthorSocialgraphFilter;

// 目的：声明实现异步 Filter。
// 影响：流水线可在过滤阶段调用本过滤器。
#[async_trait]
impl Filter<ScoredPostsQuery, PostCandidate> for AuthorSocialgraphFilter {
    // 目的：实现社交关系过滤主逻辑。
    // 影响：将候选划分为保留与剔除两种结果。
    async fn filter(
        // 目的：引用本过滤器状态（无内部字段）。
        // 影响：仅调用签名需要。
        &self,
        // 目的：接收查询对象。
        // 影响：从中读取用户拉黑/静音列表。
        query: &ScoredPostsQuery,
        // 目的：接收待过滤候选列表（按值传入）。
        // 影响：本过滤器拥有所有权后逐条判定。
        candidates: Vec<PostCandidate>,
    // 目的：声明返回值：过滤结果（保留+剔除）。
    // 影响：流水线据此更新候选集合。
    ) -> Result<FilterResult<PostCandidate>, String> {
        // 目的：克隆用户的拉黑列表。
        // 影响：避免多次借用查询对象。
        let viewer_blocked_user_ids = query.user_features.blocked_user_ids.clone();
        // 目的：克隆用户的静音列表。
        // 影响：避免多次借用查询对象。
        let viewer_muted_user_ids = query.user_features.muted_user_ids.clone();

        // 目的：检查用户是否没有任何拉黑/静音设置。
        // 影响：无设置时跳过逐条判定，直接整体保留。
        if viewer_blocked_user_ids.is_empty() && viewer_muted_user_ids.is_empty() {
            // 目的：返回全部保留的空结果。
            // 影响：无剔除项，候选原样进入下一阶段。
            return Ok(FilterResult {
                // 目的：候选整体保留。
                // 影响：过滤对结果无影响。
                kept: candidates,
                // 目的：无剔除项。
                // 影响：removed 保持为空。
                removed: Vec::new(),
            });
        }

        // 目的：初始化保留集合。
        // 影响：存放未被拉黑/静音的作者内容。
        let mut kept: Vec<PostCandidate> = Vec::new();
        // 目的：初始化剔除集合。
        // 影响：存放被拉黑/静音的作者内容。
        let mut removed: Vec<PostCandidate> = Vec::new();

        // 目的：遍历每个候选。
        // 影响：逐条判断作者是否在设置列表中。
        for candidate in candidates {
            // 目的：将作者 ID 转为 i64。
            // 影响：与特征列表中的元素类型对齐。
            let author_id = candidate.author_id as i64;
            // 目的：判断作者是否被静音。
            // 影响：得到静音判定结果。
            let muted = viewer_muted_user_ids.contains(&author_id);
            // 目的：判断作者是否被拉黑。
            // 影响：得到拉黑判定结果。
            let blocked = viewer_blocked_user_ids.contains(&author_id);
            // 目的：任一命中即剔除。
            // 影响：被拉黑或静音的内容不进结果。
            if muted || blocked {
                // 目的：移入剔除集合。
                // 影响：该候选不再参与评分与输出。
                removed.push(candidate);
            } else {
                // 目的：移入保留集合。
                // 影响：该候选继续进入后续阶段。
                kept.push(candidate);
            }
        }

        // 目的：返回过滤结果。
        // 影响：剔除的用户内容从最终结果中消失。
        Ok(FilterResult { kept, removed })
    }
}
