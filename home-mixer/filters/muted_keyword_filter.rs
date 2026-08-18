// 目的：引入候选结构 PostCandidate。
// 影响：本过滤器操作并划分候选集合。
use crate::candidate_pipeline::candidate::PostCandidate;
// 目的：引入查询对象 ScoredPostsQuery。
// 影响：本过滤器从查询中读取用户静音关键词。
use crate::candidate_pipeline::query::ScoredPostsQuery;
// 目的：引入 Arc 智能指针。
// 影响：共享分词器实例，避免重复初始化。
use std::sync::Arc;
// 目的：引入 tonic 异步 trait 支持。
// 影响：使 filter 方法成为异步方法。
use tonic::async_trait;
// 目的：引入过滤器 trait 与结果类型。
// 影响：让本类型以标准过滤器接入流水线。
use xai_candidate_pipeline::filter::{Filter, FilterResult};
// 目的：引入推文文本匹配相关工具（分词器、匹配组等）。
// 影响：实现关键词分词与命中匹配能力。
use xai_post_text::{MatchTweetGroup, TokenSequence, TweetTokenizer, UserMutes};

// 目的：定义静音关键词过滤器结构。
// 影响：剔除正文命中用户静音关键词的候选。
pub struct MutedKeywordFilter {
    // 目的：持有共享的分词器。
    // 影响：统一分词规则，供关键词与正文共同使用。
    pub tokenizer: Arc<TweetTokenizer>,
}

// 目的：为过滤器实现构造逻辑。
// 影响：提供实例化入口。
impl MutedKeywordFilter {
    // 目的：定义构造函数。
    // 影响：创建并共享分词器。
    pub fn new() -> Self {
        // 目的：创建推文分词器。
        // 影响：提供标准的推文文本分词能力。
        let tokenizer = TweetTokenizer::new();
        // 目的：构造过滤器实例。
        // 影响：分词器以 Arc 形式共享。
        Self {
            // 目的：包装分词器为共享引用。
            // 影响：多请求并发时复用同一分词器。
            tokenizer: Arc::new(tokenizer),
        }
    }
}

// 目的：声明实现异步 Filter。
// 影响：流水线可在过滤阶段调用本过滤器。
#[async_trait]
impl Filter<ScoredPostsQuery, PostCandidate> for MutedKeywordFilter {
    // 目的：为该过滤器挂接调用统计埋点。
    // 影响：监控静音过滤的调用频率与时延。
    #[xai_stats_macro::receive_stats]
    // 目的：实现静音关键词过滤主逻辑。
    // 影响：剔除命中用户静音关键词的候选。
    async fn filter(
        // 目的：引用本过滤器状态。
        // 影响：访问共享分词器。
        &self,
        // 目的：接收查询对象。
        // 影响：从中读取用户静音关键词列表。
        query: &ScoredPostsQuery,
        // 目的：接收待过滤候选列表（按值传入）。
        // 影响：本过滤器拥有所有权后逐条判定。
        candidates: Vec<PostCandidate>,
    // 目的：声明返回值：过滤结果（保留+剔除）。
    // 影响：流水线据此更新候选集合。
    ) -> Result<FilterResult<PostCandidate>, String> {
        // 目的：克隆用户的静音关键词列表。
        // 影响：避免长期借用查询对象。
        let muted_keywords = query.user_features.muted_keywords.clone();

        // 目的：判断用户是否没有设置任何静音关键词。
        // 影响：无关键词时直接整体保留候选。
        if muted_keywords.is_empty() {
            // 目的：返回全部保留的空结果。
            // 影响：过滤对结果无影响。
            return Ok(FilterResult {
                // 目的：候选整体保留。
                // 影响：全部进入后续阶段。
                kept: candidates,
                // 目的：无剔除项。
                // 影响：removed 保持为空。
                removed: vec![],
            });
        }

        // 目的：将各静音关键词分词为序列。
        // 影响：准备规则的词形表示。
        let tokenized = muted_keywords.iter().map(|k| self.tokenizer.tokenize(k));
        // 目的：收集所有关键词的词序列。
        // 影响：构成静音规则的集合。
        let token_sequences: Vec<TokenSequence> = tokenized.collect::<Vec<_>>();
        // 目的：组装用户静音规则对象。
        // 影响：封装关键词序列为可匹配策略。
        let user_mutes = UserMutes::new(token_sequences);
        // 目的：创建推文匹配器。
        // 影响：提供对候选正文的命中判断。
        let matcher = MatchTweetGroup::new(user_mutes);

        // 目的：初始化保留集合。
        // 影响：存放未命中静音词的候选。
        let mut kept = Vec::new();
        // 目的：初始化剔除集合。
        // 影响：存放命中静音词的候选。
        let mut removed = Vec::new();

        // 目的：遍历每个候选。
        // 影响：逐条判断正文是否命中静音词。
        for candidate in candidates {
            // 目的：将候选正文分词为序列。
            // 影响：为匹配器提供可比对的词形。
            let tweet_text_token_sequence = self.tokenizer.tokenize(&candidate.tweet_text);
            // 目的：判断正文是否命中静音关键词。
            // 影响：得到匹配判定结果。
            if matcher.matches(&tweet_text_token_sequence) {
                // Matches muted keywords - should be removed/filtered out
                // 目的：命中时移入剔除集合。
                // 影响：该候选不再参与评分与输出。
                removed.push(candidate);
            } else {
                // Does not match muted keywords - keep it
                // 目的：未命中时进入保留集合。
                // 影响：该候选继续参与后续阶段。
                kept.push(candidate);
            }
        }

        // 目的：返回过滤结果。
        // 影响：命中静音关键词的内容从结果中消失。
        Ok(FilterResult { kept, removed })
    }
}
