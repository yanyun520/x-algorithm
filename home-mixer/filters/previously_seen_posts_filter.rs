// 目的：引入候选结构 PostCandidate。
// 影响：本过滤器操作并划分候选集合。
use crate::candidate_pipeline::candidate::PostCandidate;
// 目的：引入查询对象 ScoredPostsQuery。
// 影响：本过滤器从查询中读取已见 ID 与布隆过滤器条目。
use crate::candidate_pipeline::query::ScoredPostsQuery;
// 目的：引入布隆过滤器工具。
// 影响：以概率方式快速判断帖子是否已见。
use crate::util::bloom_filter::BloomFilter;
// 目的：引入候选关联 ID 提取工具。
// 影响：获取候选涉及的帖子 ID（原帖/转发/回复等）。
use crate::util::candidates_util::get_related_post_ids;
// 目的：引入 tonic 异步 trait 支持。
// 影响：使 filter 方法成为异步方法。
use tonic::async_trait;
// 目的：引入过滤器 trait 与结果类型。
// 影响：让本类型以标准过滤器接入流水线。
use xai_candidate_pipeline::filter::{Filter, FilterResult};

/// Filter out previously seen posts using a Bloom Filter and
/// the seen IDs sent in the request directly from the client
// 目的：结构注释：结合布隆过滤器与客户端上报的已见 ID 剔除已见帖子。
// 影响：避免重复推荐用户已经看过的内容。
pub struct PreviouslySeenPostsFilter;

// 目的：声明实现异步 Filter。
// 影响：流水线可在过滤阶段调用本过滤器。
#[async_trait]
impl Filter<ScoredPostsQuery, PostCandidate> for PreviouslySeenPostsFilter {
    // 目的：实现已见过滤主逻辑。
    // 影响：剔除已被用户看过的候选。
    async fn filter(
        // 目的：引用本过滤器状态（无内部字段）。
        // 影响：仅调用签名需要。
        &self,
        // 目的：接收查询对象。
        // 影响：从中读取已见 ID 与布隆条目。
        query: &ScoredPostsQuery,
        // 目的：接收待过滤候选列表（按值传入）。
        // 影响：本过滤器拥有所有权后直接分区。
        candidates: Vec<PostCandidate>,
    // 目的：声明返回值：过滤结果（保留+剔除）。
    // 影响：流水线据此更新候选集合。
    ) -> Result<FilterResult<PostCandidate>, String> {
        // 目的：将查询中的布隆过滤器条目解码为过滤器实例列表。
        // 影响：用于后续高效判断帖子是否已被客户端记录。
        let bloom_filters = query
            // 目的：访问查询的布隆条目。
            // 影响：数据来源为客户端压缩的上报集合。
            .bloom_filter_entries
            // 目的：迭代条目。
            // 影响：逐个转换。
            .iter()
            // 目的：将每条目转为 BloomFilter 实例。
            // 影响：得到可直接查询的过滤器集合。
            .map(BloomFilter::from_entry)
            // 目的：收集为向量。
            // 影响：形成过滤器列表。
            .collect::<Vec<_>>();

        // 目的：按是否已见分区候选。
        // 影响：已见候选进 removed（剔除），未见进 kept（保留）。
        let (removed, kept): (Vec<_>, Vec<_>) = candidates.into_iter().partition(|c| {
            // 目的：取候选涉及的帖子 ID 并判断任一是否已见。
            // 影响：关联帖已见则整条候选视为已见。
            get_related_post_ids(c).iter().any(|&post_id| {
                // 目的：判断显式已见列表是否包含该 ID。
                // 影响：命中即视为已见。
                query.seen_ids.contains(&post_id)
                // 目的：或判断任一布隆过滤器是否可能包含该 ID。
                // 影响：概率命中也视为已见（可容忍少量误判）。
                || bloom_filters
                        // 目的：迭代布隆过滤器列表。
                        // 影响：逐一检查。
                        .iter()
                        // 目的：检查过滤器是否可能包含该帖子 ID。
                        // 影响：两个来源任一命中即为已见。
                        .any(|filter| filter.may_contain(post_id))
            })
        });

        // 目的：返回过滤结果。
        // 影响：已看过的内容不再进入后续阶段。
        Ok(FilterResult { kept, removed })
    }
}
