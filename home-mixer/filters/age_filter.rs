// 目的：引入候选结构 PostCandidate。
// 影响：本过滤器操作并划分候选集合。
use crate::candidate_pipeline::candidate::PostCandidate;
// 目的：引入查询对象 ScoredPostsQuery。
// 影响：接口对齐需要（当前未读取查询内容）。
use crate::candidate_pipeline::query::ScoredPostsQuery;
// 目的：引入雪花 ID 工具模块。
// 影响：从 tweet_id（雪花 ID）解码出帖子的创建时间。
use crate::util::snowflake;
// 目的：引入 Duration 类型。
// 影响：表达帖子允许的最大年龄窗口。
use std::time::Duration;
// 目的：引入 tonic 异步 trait 支持。
// 影响：使 filter 方法成为异步方法。
use tonic::async_trait;
// 目的：引入过滤器 trait 与结果类型。
// 影响：让本类型以标准过滤器接入流水线。
use xai_candidate_pipeline::filter::{Filter, FilterResult};

/// Filter that removes tweets older than a specified duration.
// 目的：结构注释：本过滤器剔除超过指定时长的旧帖。
// 影响：保证推荐内容新鲜度。
pub struct AgeFilter {
    // 目的：保存允许的最大帖龄。
    // 影响：决定哪些帖子被保留、哪些被剔除。
    pub max_age: Duration,
}

// 目的：为过滤器实现构造与判断逻辑。
// 影响：提供实例化与年龄判断的复用能力。
impl AgeFilter {
    // 目的：定义构造函数。
    // 影响：按传入时长创建过滤器实例。
    pub fn new(max_age: Duration) -> Self {
        // 目的：构造结构体实例。
        // 影响：过滤器携带年龄窗口配置。
        Self { max_age }
    }

    // 目的：判断单条帖子是否在允许年龄范围内。
    // 影响：为分区过滤提供判定依据。
    fn is_within_age(&self, tweet_id: i64) -> bool {
        // 目的：解码帖子创建时间并计算已存在时长。
        // 影响：得到帖子的实际年龄。
        snowflake::duration_since_creation_opt(tweet_id)
            // 目的：判断年龄是否不超过最大窗口。
            // 影响：不超龄返回 true（保留）。
            .map(|age| age <= self.max_age)
            // 目的：解码失败时保守地视为不保留。
            // 影响：异常数据帖被剔除，保证质量。
            .unwrap_or(false)
    }
}

// 目的：声明实现异步 Filter。
// 影响：流水线可在过滤阶段调用本过滤器。
#[async_trait]
impl Filter<ScoredPostsQuery, PostCandidate> for AgeFilter {
    // 目的：实现年龄过滤主逻辑。
    // 影响：将候选划分为保留（新鲜）与剔除（超龄）两组。
    async fn filter(
        // 目的：引用本过滤器状态。
        // 影响：访问其中的年龄窗口配置。
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
        // 目的：按年龄是否达标分区候选。
        // 影响：分别产出 kept 与 removed 两个集合。
        let (kept, removed): (Vec<_>, Vec<_>) = candidates
            // 目的：转迭代器消费候选。
            // 影响：逐个处理并移入对应分区。
            .into_iter()
            // 目的：以年龄判断为分区谓词。
            // 影响：true 进 kept，false 进 removed。
            .partition(|c| self.is_within_age(c.tweet_id));

        // 目的：返回过滤结果。
        // 影响：超龄帖被移除，新鲜帖继续参与后续打分。
        Ok(FilterResult { kept, removed })
    }
}
