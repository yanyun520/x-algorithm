// 目的：引入候选结构 PostCandidate。
// 影响：本增强器产出并更新候选的基础内容字段。
use crate::candidate_pipeline::candidate::PostCandidate;
// 目的：引入查询对象 ScoredPostsQuery。
// 影响：本增强器签名需与流水线接口对齐（当前未读取查询内容）。
use crate::candidate_pipeline::query::ScoredPostsQuery;
// 目的：引入 TES 客户端 trait。
// 影响：为批量获取推文核心数据提供异步调用能力。
use crate::clients::tweet_entity_service_client::TESClient;
// 目的：引入 Arc 智能指针。
// 影响：共享 TES 客户端，避免每请求重复建连。
use std::sync::Arc;
// 目的：引入 tonic 异步 trait 支持。
// 影响：使 hydrate 方法可在异步运行时执行网络请求。
use tonic::async_trait;
// 目的：引入 Hydrator trait。
// 影响：本类型以标准增强器身份接入流水线框架。
use xai_candidate_pipeline::hydrator::Hydrator;

// 目的：定义核心数据增强器结构。
// 影响：为候选补充作者/正文/回复转发等基础字段。
pub struct CoreDataCandidateHydrator {
    // 目的：持有 TES 客户端引用。
    // 影响：发起核心数据批量查询的唯一通道。
    pub tes_client: Arc<dyn TESClient + Send + Sync>,
}

// 目的：为增强器实现构造函数。
// 影响：提供注入 TES 客户端的实例化入口。
impl CoreDataCandidateHydrator {
    // 目的：定义异步构造方法。
    // 影响：返回携带客户端的增强器实例。
    pub async fn new(tes_client: Arc<dyn TESClient + Send + Sync>) -> Self {
        // 目的：构造结构体实例。
        // 影响：客户端引用被保存供 hydrate 使用。
        Self { tes_client }
    }
}

// 目的：声明实现异步 Hydrator。
// 影响：流水线可在增强阶段调用。
#[async_trait]
impl Hydrator<ScoredPostsQuery, PostCandidate> for CoreDataCandidateHydrator {
    // 目的：为该增强器挂接调用统计埋点。
    // 影响：监控核心数据增强的调用频率与时延。
    #[xai_stats_macro::receive_stats]
    // 目的：实现核心数据增强主逻辑。
    // 影响：补齐候选的可点击/可展示所必需的内容字段。
    async fn hydrate(
        // 目的：引用本增强器状态。
        // 影响：访问其中的 TES 客户端。
        &self,
        // 目的：接收查询对象（当前未使用，故命名为 _query）。
        // 影响：保持接口统一，为未来按查询定制留空间。
        _query: &ScoredPostsQuery,
        // 目的：接收待增强候选列表。
        // 影响：作为批量查询的输入。
        candidates: &[PostCandidate],
    // 目的：声明返回值与错误类型。
    // 影响：成功返回增强候选，失败返回字符串错误。
    ) -> Result<Vec<PostCandidate>, String> {
        // 目的：取出 TES 客户端引用。
        // 影响：便于后续批量调用。
        let client = &self.tes_client;

        // 目的：提取全部候选的 tweet_id。
        // 影响：构成一次批量查询的键集合。
        let tweet_ids = candidates.iter().map(|c| c.tweet_id).collect::<Vec<_>>();

        // 目的：批量请求推文核心数据。
        // 影响：得到 tweet_id -> 核心数据的映射。
        let post_features = client.get_tweet_core_datas(tweet_ids.clone()).await;
        // 目的：转换远程调用错误为字符串。
        // 影响：调用失败时经 ? 提前返回错误。
        let post_features = post_features.map_err(|e| e.to_string())?;

        // 目的：预分配增强结果容器。
        // 影响：避免扩容开销，长度与输入一致。
        let mut hydrated_candidates = Vec::with_capacity(candidates.len());
        // 目的：按原始顺序遍历每个 tweet_id。
        // 影响：保证增强结果与输入候选一一对应。
        for tweet_id in tweet_ids {
            // 目的：按 tweet_id 查取核心数据。
            // 影响：缺失时返回 None，安全回退。
            let post_features = post_features.get(&tweet_id);
            // 目的：解包可选值并取内部引用。
            // 影响：得到核心数据结构或空值。
            let core_data = post_features.and_then(|x| x.as_ref());
            // 目的：取出正文文本。
            // 影响：用于后续填充候选正文。
            let text = core_data.map(|x| x.text.clone());
            // 目的：构造增强后的候选副本。
            // 影响：携带核心数据，供后续过滤/展示使用。
            let hydrated = PostCandidate {
                // 目的：写入作者 ID；缺失时取默认值 0。
                // 影响：核心数据完整性过滤器据 0 判定剔除。
                author_id: core_data.map(|x| x.author_id).unwrap_or_default(),
                // 目的：写入被转发用户 ID。
                // 影响：用于屏幕名补充与模型特征。
                retweeted_user_id: core_data.and_then(|x| x.source_user_id),
                // 目的：写入被转发帖 ID。
                // 影响：用于转发去重与预测映射。
                retweeted_tweet_id: core_data.and_then(|x| x.source_tweet_id),
                // 目的：写入回复目标帖 ID。
                // 影响：用于识别回复关系并构建会话链。
                in_reply_to_tweet_id: core_data.and_then(|x| x.in_reply_to_tweet_id),
                // 目的：写入正文文本；缺失时取空串。
                // 影响：完整性过滤器据空正文判定剔除。
                tweet_text: text.unwrap_or_default(),
                // 目的：其余字段保持默认。
                // 影响：由其它增强器逐步填充。
                ..Default::default()
            };
            // 目的：将增强结果追加到容器。
            // 影响：保持与输入候选的顺序一致。
            hydrated_candidates.push(hydrated);
        }

        // 目的：返回增强结果。
        // 影响：流水线据此刷新原候选基础字段。
        Ok(hydrated_candidates)
    }

    // 目的：定义合并增强结果回候选的方法。
    // 影响：把核心数据各字段同步到原候选。
    fn update(&self, candidate: &mut PostCandidate, hydrated: PostCandidate) {
        // 目的：同步被转发用户 ID。
        // 影响：候选获得转发上下文。
        candidate.retweeted_user_id = hydrated.retweeted_user_id;
        // 目的：同步被转发帖 ID。
        // 影响：候选获得转发目标。
        candidate.retweeted_tweet_id = hydrated.retweeted_tweet_id;
        // 目的：同步回复目标帖 ID。
        // 影响：候选获得回复上下文。
        candidate.in_reply_to_tweet_id = hydrated.in_reply_to_tweet_id;
        // 目的：同步正文文本。
        // 影响：候选携带可展示的正文内容。
        candidate.tweet_text = hydrated.tweet_text;
    }
}
