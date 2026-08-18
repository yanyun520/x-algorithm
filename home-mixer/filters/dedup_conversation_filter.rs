// 目的：引入候选结构 PostCandidate。
// 影响：本过滤器操作并划分候选集合。
use crate::candidate_pipeline::candidate::PostCandidate;
// 目的：引入查询对象 ScoredPostsQuery。
// 影响：接口对齐需要（当前未读取查询内容）。
use crate::candidate_pipeline::query::ScoredPostsQuery;
// 目的：引入 HashMap 集合类型。
// 影响：记录每个会话当前保留的候选索引与其最佳分数。
use std::collections::HashMap;
// 目的：引入 tonic 异步 trait 支持。
// 影响：使 filter 方法成为异步方法。
use tonic::async_trait;
// 目的：引入过滤器 trait 与结果类型。
// 影响：让本类型以标准过滤器接入流水线。
use xai_candidate_pipeline::filter::{Filter, FilterResult};

/// Keeps only the highest-scored candidate per branch of a conversation tree
// 目的：结构注释：每个会话树（分支）仅保留分数最高的候选。
// 影响：避免同一对话的多条回复同时占据结果位置，提升多样性。
pub struct DedupConversationFilter;

// 目的：声明实现异步 Filter。
// 影响：流水线可在过滤阶段调用本过滤器。
#[async_trait]
impl Filter<ScoredPostsQuery, PostCandidate> for DedupConversationFilter {
    // 目的：实现会话去重主逻辑。
    // 影响：同会话中低分候选被剔除。
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
        // 目的：初始化保留集合。
        // 影响：存放各会话最高分候选。
        let mut kept: Vec<PostCandidate> = Vec::new();
        // 目的：初始化剔除集合。
        // 影响：存放同会话低分候选。
        let mut removed: Vec<PostCandidate> = Vec::new();
        // 目的：创建会话映射（会话 ID -> 保留索引与最佳分数）。
        // 影响：跟踪每个会话当前的最佳候选位置。
        let mut best_per_convo: HashMap<u64, (usize, f64)> = HashMap::new();

        // 目的：遍历每个候选。
        // 影响：逐条判断是否为本会话最高分。
        for candidate in candidates {
            // 目的：计算候选所属会话 ID。
            // 影响：作为去重的分组键。
            let conversation_id = get_conversation_id(&candidate);
            // 目的：读取候选得分，缺失时按 0 处理。
            // 影响：用于同类会话候选之间的分数比较。
            let score = candidate.score.unwrap_or(0.0);

            // 目的：若本会话已有保留候选，则进行分数比较。
            // 影响：根据比较结果决定新旧候选去留。
            if let Some((kept_idx, best_score)) = best_per_convo.get_mut(&conversation_id) {
                // 目的：新候选分数更高时替换保留位。
                // 影响：高分候选上位，低分候选被剔除。
                if score > *best_score {
                    // 目的：用新候选替换原保留位并取出旧候选。
                    // 影响：旧候选被移入剔除集合。
                    let previous = std::mem::replace(&mut kept[*kept_idx], candidate);
                    // 目的：旧候选进入剔除集合。
                    // 影响：剔除集合包含同会话低分候选。
                    removed.push(previous);
                    // 目的：更新本会话最佳分数。
                    // 影响：后续比较以此成绩为准。
                    *best_score = score;
                } else {
                    // 目的：新候选分数不更高则直接剔除。
                    // 影响：该候选不再进入结果。
                    removed.push(candidate);
                }
            } else {
                // 目的：本会话首次出现，直接作为当前最佳。
                // 影响：该候选进入保留集合。
                let idx = kept.len();
                // 目的：记录会话 ID 对应的保留索引与分数。
                // 影响：供后续同会话候选比较。
                best_per_convo.insert(conversation_id, (idx, score));
                // 目的：候选加入保留集合。
                // 影响：该会话当前的代表候选被保留。
                kept.push(candidate);
            }
        }

        // 目的：返回过滤结果。
        // 影响：结果中每个会话仅保留最高分候选。
        Ok(FilterResult { kept, removed })
    }
}

// 目的：定义会话 ID 的计算函数。
// 影响：为候选去重提供稳定分组依据。
fn get_conversation_id(candidate: &PostCandidate) -> u64 {
    // 目的：取候选祖先 ID 中最小的作为会话根。
    // 影响：同一会话树的所有分支共享最小祖先作为会话 ID。
    candidate
        // 目的：访问候选的祖先列表。
        // 影响：提供会话树节点信息。
        .ancestors
        // 目的：迭代祖先 ID。
        // 影响：逐个比较寻找最小值。
        .iter()
        // 目的：拷贝迭代值。
        // 影响：得到可直接比较的 u64 元素。
        .copied()
        // 目的：取最小祖先 ID。
        // 影响：最小值作为会话根标识。
        .min()
        // 目的：无祖先时以自身 tweet_id 兜底。
        // 影响：非回复帖以自身为独立会话。
        .unwrap_or(candidate.tweet_id as u64)
}
