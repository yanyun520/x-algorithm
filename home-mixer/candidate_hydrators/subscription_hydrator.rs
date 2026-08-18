// 目的：引入候选结构 PostCandidate。
// 影响：本增强器产出并更新候选的订阅作者字段。
use crate::candidate_pipeline::candidate::PostCandidate;
// 目的：引入查询对象 ScoredPostsQuery。
// 影响：接口对齐需要（当前未读取查询内容）。
use crate::candidate_pipeline::query::ScoredPostsQuery;
// 目的：引入 TES 客户端 trait。
// 影响：为批量获取帖子订阅作者提供异步调用能力。
use crate::clients::tweet_entity_service_client::TESClient;
// 目的：引入 Arc 智能指针。
// 影响：共享 TES 客户端，降低连接开销。
use std::sync::Arc;
// 目的：引入 tonic 异步 trait 支持。
// 影响：使 hydrate 方法可在异步运行时执行网络请求。
use tonic::async_trait;
// 目的：引入 Hydrator trait。
// 影响：本类型以标准增强器身份接入流水线框架。
use xai_candidate_pipeline::hydrator::Hydrator;

// 目的：定义订阅作者增强器结构。
// 影响：为候选标记其订阅归属作者。
pub struct SubscriptionHydrator {
    // 目的：持有 TES 客户端引用。
    // 影响：发起订阅作者批量查询的唯一通道。
    pub tes_client: Arc<dyn TESClient + Send + Sync>,
}

// 目的：为增强器实现构造函数。
// 影响：提供注入 TES 客户端的实例化入口。
impl SubscriptionHydrator {
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
impl Hydrator<ScoredPostsQuery, PostCandidate> for SubscriptionHydrator {
    // 目的：为该增强器挂接调用统计埋点。
    // 影响：监控订阅增强的调用频率与时延。
    #[xai_stats_macro::receive_stats]
    // 目的：实现订阅作者增强主逻辑。
    // 影响：为候选补充订阅作者标识供资格过滤。
    async fn hydrate(
        // 目的：引用本增强器状态。
        // 影响：访问其中的 TES 客户端。
        &self,
        // 目的：接收查询对象（当前未使用，故命名为 _query）。
        // 影响：保持接口统一。
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

        // 目的：批量请求帖子的订阅作者 ID。
        // 影响：得到 tweet_id -> 订阅作者的映射。
        let post_features = client.get_subscription_author_ids(tweet_ids.clone()).await;
        // 目的：转换远程调用错误为字符串。
        // 影响：调用失败时经 ? 提前返回错误。
        let post_features = post_features.map_err(|e| e.to_string())?;

        // 目的：预分配增强结果容器。
        // 影响：避免扩容开销，长度与输入一致。
        let mut hydrated_candidates = Vec::with_capacity(candidates.len());
        // 目的：按原始顺序遍历每个 tweet_id。
        // 影响：保证增强结果与输入候选一一对应。
        for tweet_id in tweet_ids {
            // 目的：按 tweet_id 查取订阅作者。
            // 影响：缺失时返回 None，安全回退。
            let post_features = post_features.get(&tweet_id);
            // 目的：解包嵌套 Option 取订阅作者 ID。
            // 影响：得到订阅作者或空值。
            let subscription_author_id = post_features.and_then(|x| *x);
            // 目的：构造增强后的候选副本。
            // 影响：携带订阅作者标识。
            let hydrated = PostCandidate {
                // 目的：写入订阅作者 ID。
                // 影响：资格过滤器据此判断付费内容可见性。
                subscription_author_id,
                // 目的：其余字段保持默认。
                // 影响：由其它增强器负责填充。
                ..Default::default()
            };
            // 目的：追加到结果容器。
            // 影响：保持与输入候选顺序一致。
            hydrated_candidates.push(hydrated);
        }

        // 目的：返回增强结果。
        // 影响：流水线据此刷新原候选订阅字段。
        Ok(hydrated_candidates)
    }

    // 目的：定义合并增强结果回候选的方法。
    // 影响：把订阅作者同步到原候选。
    fn update(&self, candidate: &mut PostCandidate, hydrated: PostCandidate) {
        // 目的：同步订阅作者 ID。
        // 影响：候选携带订阅归属信息。
        candidate.subscription_author_id = hydrated.subscription_author_id;
    }
}
