// 目的：引入候选结构 PostCandidate。
// 影响：本增强器产出并更新候选的可见性过滤原因字段。
use crate::candidate_pipeline::candidate::PostCandidate;
// 目的：引入查询对象 ScoredPostsQuery。
// 影响：本增强器从查询中读取查看者上下文。
use crate::candidate_pipeline::query::ScoredPostsQuery;
// 目的：引入 futures 的 join 并发组合子。
// 影响：同时发起站内/站外两路 VF 查询，降低总时延。
use futures::future::join;
// 目的：引入 HashMap 集合类型。
// 影响：汇总两路 VF 查询结果到统一映射表。
use std::collections::HashMap;
// 目的：引入 Arc 智能指针。
// 影响：共享 VF 客户端，降低连接开销。
use std::sync::Arc;
// 目的：引入 tonic 异步 trait 支持。
// 影响：使 hydrate 方法可在异步运行时执行网络请求。
use tonic::async_trait;
// 目的：引入 Hydrator trait。
// 影响：本类型以标准增强器身份接入流水线框架。
use xai_candidate_pipeline::hydrator::Hydrator;
// 目的：引入 GetTwitterContextViewer trait。
// 影响：将查询对象转为 VF 所需查看者上下文。
use xai_twittercontext_proto::GetTwitterContextViewer;
// 目的：引入 TwitterContextViewer 结构。
// 影响：作为 VF 请求中的用户上下文载体。
use xai_twittercontext_proto::TwitterContextViewer;
// 目的：引入过滤原因类型。
// 影响：携带安全过滤的判定结果。
use xai_visibility_filtering::models::FilteredReason;
// 目的：引入安全级别枚举。
// 影响：区分站内/站外使用的安全策略等级。
use xai_visibility_filtering::vf_client::SafetyLevel;
// 目的：引入站内（主时间线）与站外（推荐）两种安全级别常量。
// 影响：分别用于站内候选与站外候选的打分方案。
use xai_visibility_filtering::vf_client::SafetyLevel::{TimelineHome, TimelineHomeRecommendations};
// 目的：引入 VF 客户端 trait。
// 影响：为批量安全判定提供异步调用能力。
use xai_visibility_filtering::vf_client::VisibilityFilteringClient;

// 目的：定义可见性过滤增强器结构。
// 影响：为已选候选补充安全过滤原因，是选后阶段的关键增强器。
pub struct VFCandidateHydrator {
    // 目的：持有 VF 客户端引用。
    // 影响：发起安全判定的唯一通道。
    pub vf_client: Arc<dyn VisibilityFilteringClient + Send + Sync>,
}

// 目的：为增强器实现构造函数与内部工具方法。
// 影响：提供实例化与批量查询复用的封装。
impl VFCandidateHydrator {
    // 目的：定义异步构造方法。
    // 影响：返回携带 VF 客户端的增强器实例。
    pub async fn new(vf_client: Arc<dyn VisibilityFilteringClient + Send + Sync>) -> Self {
        // 目的：构造结构体实例。
        // 影响：客户端引用被保存供 hydrate 使用。
        Self { vf_client }
    }

    // 目的：封装单批 VF 查询的公共逻辑。
    // 影响：提供给站内与站外两路查询复用，减少重复代码。
    async fn fetch_vf_results(
        // 目的：接收 VF 客户端引用。
        // 影响：执行真实的安全判定请求。
        client: &Arc<dyn VisibilityFilteringClient + Send + Sync>,
        // 目的：接收待判定的帖子 ID 列表。
        // 影响：作为本次查询的输入集合。
        tweet_ids: Vec<i64>,
        // 目的：接收采用的安全级别。
        // 影响：决定安全策略类型。
        safety_level: SafetyLevel,
        // 目的：接收目标用户 ID。
        // 影响：安全策略以该用户视角判定。
        for_user_id: i64,
        // 目的：接收查看者上下文。
        // 影响：提供应用/国家/语言等策略上下文。
        context: Option<TwitterContextViewer>,
    // 目的：声明返回值：帖子ID到过滤原因的映射。
    // 影响：调用方汇总后为候选填充过滤原因。
    ) -> Result<HashMap<i64, Option<FilteredReason>>, String> {
        // 目的：判断帖子列表是否为空。
        // 影响：空列表可跳过远程调用直接返回空映射。
        if tweet_ids.is_empty() {
            // 目的：返回空映射。
            // 影响：避免无意义的网络请求。
            return Ok(HashMap::new());
        }

        // 目的：发起 VF 批量判定请求。
        // 影响：得到帖子级别的过滤结果。
        client
            .get_result(tweet_ids, safety_level, for_user_id, context)
            // 目的：等待异步判定完成。
            // 影响：拿到最终过滤结果。
            .await
            // 目的：转换远程错误为字符串。
            // 影响：调用失败时把错误透传给上层。
            .map_err(|e| e.to_string())
    }
}

// 目的：声明实现异步 Hydrator。
// 影响：流水线可在选后增强阶段调用。
#[async_trait]
impl Hydrator<ScoredPostsQuery, PostCandidate> for VFCandidateHydrator {
    // 目的：为该增强器挂接调用统计埋点。
    // 影响：监控安全判定的调用频率与时延。
    #[xai_stats_macro::receive_stats]
    // 目的：实现可见性过滤增强主逻辑。
    // 影响：为候选补充安全过滤原因，供 VFFilter 决策。
    async fn hydrate(
        // 目的：引用本增强器状态。
        // 影响：访问其中的 VF 客户端。
        &self,
        // 目的：接收查询对象。
        // 影响：提取查看者上下文与用户 ID。
        query: &ScoredPostsQuery,
        // 目的：接收待增强候选列表。
        // 影响：作为过滤判定的输入集合。
        candidates: &[PostCandidate],
    // 目的：声明返回值与错误类型。
    // 影响：成功返回增强候选，失败返回字符串错误。
    ) -> Result<Vec<PostCandidate>, String> {
        // 目的：将查询转为查看者上下文。
        // 影响：为 VF 策略提供用户/应用/地区上下文。
        let context = query.get_viewer();
        // 目的：读取查询中的用户 ID。
        // 影响：作为 VF 判定的目标用户。
        let user_id = query.user_id;
        // 目的：取出 VF 客户端引用。
        // 影响：便于后续批量调用。
        let client = &self.vf_client;

        // 目的：初始化站内候选 ID 容器。
        // 影响：收集站内待判定帖子。
        let mut in_network_ids = Vec::new();
        // 目的：初始化站外候选 ID 容器。
        // 影响：收集站外待判定帖子。
        let mut oon_ids = Vec::new();
        // 目的：遍历候选并按站内外分流。
        // 影响：形成两条独立的安全判定队列。
        for candidate in candidates.iter() {
            // 目的：依据候选的 in_network 标记分流。
            // 影响：站内进入 TimelineHome 判定，站外进入推荐判定。
            if candidate.in_network.unwrap_or(false) {
                // 目的：将站内候选 ID 入列。
                // 影响：站内批次集合增大。
                in_network_ids.push(candidate.tweet_id);
            } else {
                // 目的：将站外候选 ID 入列。
                // 影响：站外批次集合增大。
                oon_ids.push(candidate.tweet_id);
            }
        }

        // 目的：构造站内 VF 查询 Future（未 await，先并发准备）。
        // 影响：与站外查询并行执行。
        let in_network_future = Self::fetch_vf_results(
            // 目的：传客户端引用。
            // 影响：共享同一客户端执行查询。
            client,
            // 目的：传站内候选 ID。
            // 影响：判定对象为站内内容。
            in_network_ids,
            // 目的：指定主时间线安全级别。
            // 影响：站内内容按主时间线策略判定。
            TimelineHome,
            // 目的：传目标用户 ID。
            // 影响：策略按用户视角生效。
            user_id,
            // 目的：克隆上下文副本（本路独立拥有）。
            // 影响：两路查询各自持有上下文。
            context.clone(),
        );

        // 目的：构造站外 VF 查询 Future。
        // 影响：与站内查询并行执行。
        let oon_future = Self::fetch_vf_results(
            // 目的：传客户端引用。
            // 影响：共享同一客户端执行查询。
            client,
            // 目的：传站外候选 ID。
            // 影响：判定对象为站外内容。
            oon_ids,
            // 目的：指定推荐时间线安全级别。
            // 影响：站外内容按推荐策略判定。
            TimelineHomeRecommendations,
            // 目的：传目标用户 ID。
            // 影响：策略按用户视角生效。
            user_id,
            // 目的：移入上下文。
            // 影响：本路独占使用上下文。
            context,
        );

        // 目的：并发等待两路结果返回。
        // 影响：相比串行查询降低约一半时延。
        let (in_network_result, oon_result) = join(in_network_future, oon_future).await;
        // 目的：创建统一结果映射表。
        // 影响：合并站内/站外两路结果。
        let mut result: HashMap<i64, Option<FilteredReason>> = HashMap::new();
        // 目的：并入站内结果并透传错误。
        // 影响：站内判定失败则整体失败。
        result.extend(in_network_result?);
        // 目的：并入站外结果并透传错误。
        // 影响：站外判定失败则整体失败。
        result.extend(oon_result?);

        // 目的：预分配结果容器。
        // 影响：避免扩容开销，长度与输入一致。
        let mut hydrated_candidates = Vec::with_capacity(candidates.len());
        // 目的：遍历全部候选。
        // 影响：逐个维映射过滤原因。
        for candidate in candidates {
            // 目的：按 tweet_id 查取过滤原因。
            // 影响：得到该候选的判定结果（可能缺失）。
            let visibility_reason = result.get(&candidate.tweet_id);
            // 目的：缺失时以 None 兜底。
            // 影响：保证候选始终有合法的字段值。
            let visibility_reason = visibility_reason.unwrap_or(&None);
            // 目的：构造增强后的候选副本。
            // 影响：携带可见性过滤原因。
            let hydrated = PostCandidate {
                // 目的：写入过滤原因（克隆引用值）。
                // 影响：VFFilter 据此决定候选去留。
                visibility_reason: visibility_reason.clone(),
                // 目的：其余字段保持默认。
                // 影响：由其它增强器负责填充。
                ..Default::default()
            };
            // 目的：追加到结果容器。
            // 影响：保持与输入候选顺序一致。
            hydrated_candidates.push(hydrated);
        }
        // 目的：返回增强结果。
        // 影响：流水线据此刷新原候选的过滤原因。
        Ok(hydrated_candidates)
    }

    // 目的：定义合并增强结果回候选的方法。
    // 影响：把过滤原因同步到原候选。
    fn update(&self, candidate: &mut PostCandidate, hydrated: PostCandidate) {
        // 目的：同步可见性过滤原因。
        // 影响：候选携带安全判定信息供过滤与输出。
        candidate.visibility_reason = hydrated.visibility_reason;
    }
}
