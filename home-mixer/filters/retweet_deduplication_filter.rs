// 目的：引入候选结构 PostCandidate。
// 影响：本过滤器操作并划分候选集合。
use crate::candidate_pipeline::candidate::PostCandidate;
// 目的：引入查询对象 ScoredPostsQuery。
// 影响：接口对齐需要（当前未读取查询内容）。
use crate::candidate_pipeline::query::ScoredPostsQuery;
// 目的：引入 HashSet 集合类型。
// 影响：以 O(1) 复杂度追踪已出现的原始帖 ID。
use std::collections::HashSet;
// 目的：引入 tonic 异步 trait 支持。
// 影响：使 filter 方法成为异步方法。
use tonic::async_trait;
// 目的：引入过滤器 trait 与结果类型。
// 影响：让本类型以标准过滤器接入流水线。
use xai_candidate_pipeline::filter::{Filter, FilterResult};

/// Deduplicates retweets, keeping only the first occurrence of a tweet
/// (whether as an original or as a retweet).
// 目的：结构注释：对转发去重，同一原始帖（无论原帖还是转发形态）仅保留首次出现。
// 影响：避免原帖与其多条转发同时出现在结果中。
pub struct RetweetDeduplicationFilter;

// 目的：声明实现异步 Filter。
// 影响：流水线可在过滤阶段调用本过滤器。
#[async_trait]
impl Filter<ScoredPostsQuery, PostCandidate> for RetweetDeduplicationFilter {
    // 目的：实现转发去重主逻辑。
    // 影响：同一原始帖只允许一个候选进入结果。
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
        // 目的：创建已见原始帖 ID 集合。
        // 影响：追踪已出现的原始帖，配合去重判定。
        let mut seen_tweet_ids: HashSet<u64> = HashSet::new();
        // 目的：初始化保留集合。
        // 影响：存放首次出现的候选。
        let mut kept = Vec::new();
        // 目的：初始化剔除集合。
        // 影响：存放重复出现的候选。
        let mut removed = Vec::new();

        // 目的：遍历每个候选。
        // 影响：逐条判断是否为重复内容。
        for candidate in candidates {
            // 目的：按候选的转发目标分情形处理。
            // 影响：区分转发帖与原始帖两种情况。
            match candidate.retweeted_tweet_id {
                // 目的：候选为转发帖，其原始帖 ID 为 retweeted_id。
                // 影响：以原始帖 ID 作为去重键。
                Some(retweeted_id) => {
                    // Remove if we've already seen this tweet (as original or retweet)
                    // 目的：若原始帖已经出现过，则判定重复。
                    // 影响：避免同一条内容以多个形态出现。
                    if seen_tweet_ids.insert(retweeted_id) {
                        // 目的：首次出现则保留转发候选。
                        // 影响：该候选进入后续阶段。
                        kept.push(candidate);
                    } else {
                        // 目的：重复出现则剔除。
                        // 影响：该转发不再参与输出。
                        removed.push(candidate);
                    }
                }
                // 目的：候选为原始帖（无转发目标）。
                // 影响：自身即去重键。
                None => {
                    // Mark this original tweet ID as seen so retweets of it get filtered
                    // 目的：登记原始帖 ID，使其转发被识别为重复。
                    // 影响：后续该帖的转发候选会被剔除。
                    seen_tweet_ids.insert(candidate.tweet_id as u64);
                    // 目的：保留原始帖候选。
                    // 影响：原始内容进入后续阶段。
                    kept.push(candidate);
                }
            }
        }

        // 目的：返回过滤结果。
        // 影响：结果中不再包含被转发重复的内容。
        Ok(FilterResult { kept, removed })
    }
}
