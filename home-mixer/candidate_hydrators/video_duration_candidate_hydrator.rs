// 目的：引入候选结构 PostCandidate。
// 影响：本增强器产出并更新候选的视频时长字段。
use crate::candidate_pipeline::candidate::PostCandidate;
// 目的：引入 MediaInfo 枚举（尾部片断不截断注解对应的枚举值）。
// 影响：用于识别候选媒体是否为视频。
use crate::candidate_pipeline::candidate_features::MediaInfo;
// 目的：引入查询对象 ScoredPostsQuery。
// 影响：接口对齐需要（当前未读取查询内容）。
use crate::candidate_pipeline::query::ScoredPostsQuery;
// 目的：引入 TES 客户端 trait。
// 影响：为批量获取帖子媒体实体提供异步调用能力。
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

// 目的：定义视频时长增强器结构。
// 影响：为候选补充视频时长特征。
pub struct VideoDurationCandidateHydrator {
    // 目的：持有 TES 客户端引用。
    // 影响：发起媒体实体批量查询的唯一通道。
    pub tes_client: Arc<dyn TESClient + Send + Sync>,
}

// 目的：为增强器实现构造函数。
// 影响：提供注入 TES 客户端的实例化入口。
impl VideoDurationCandidateHydrator {
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
impl Hydrator<ScoredPostsQuery, PostCandidate> for VideoDurationCandidateHydrator {
    // 目的：为该增强器挂接调用统计埋点。
    // 影响：监控视频时长增强的调用频率与时延。
    #[xai_stats_macro::receive_stats]
    // 目的：实现视频时长增强主逻辑。
    // 影响：为候选补充视频时长（毫秒）。
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

        // 目的：批量请求帖子的媒体实体。
        // 影响：得到 tweet_id -> 媒体实体列表的映射。
        let post_features = client.get_tweet_media_entities(tweet_ids.clone()).await;
        // 目的：转换远程调用错误为字符串。
        // 影响：调用失败时经 ? 提前返回错误。
        let post_features = post_features.map_err(|e| e.to_string())?;

        // 目的：预分配增强结果容器。
        // 影响：避免扩容开销，长度与输入一致。
        let mut hydrated_candidates = Vec::with_capacity(candidates.len());
        // 目的：按原始顺序遍历每个 tweet_id。
        // 影响：保证增强结果与输入候选一一对应。
        for tweet_id in tweet_ids {
            // 目的：按 tweet_id 查取媒体实体。
            // 影响：缺失时返回 None，安全回退。
            let post_features = post_features.get(&tweet_id);
            // 目的：解包可选值取实体列表引用。
            // 影响：获得媒体实体列表或空值。
            let media_entities = post_features.and_then(|x| x.as_ref());

            // 目的：在媒体实体列表中查找首个视频并取时长。
            // 影响：确定候选是否为视频及视频时长。
            let video_duration_ms = media_entities.and_then(|entities| {
                // 目的：遍历所有媒体实体。
                // 影响：逐个检查是否为视频。
                entities.iter().find_map(|entity| {
                    // 目的：匹配 VideoInfo 媒体类型。
                    // 影响：命中时取出时长，未命中则继续查找。
                    if let Some(MediaInfo::VideoInfo(video_info)) = &entity.media_info {
                        // 目的：返回视频时长（毫秒）。
                        // 影响：作为候选的视频时长值。
                        Some(video_info.duration_millis)
                    } else {
                        // 目的：非视频媒体返回 None。
                        // 影响：继续查找下一条实体。
                        None
                    }
                })
            });

            // 目的：构造增强后的候选副本。
            // 影响：携带视频时长特征。
            let hydrated = PostCandidate {
                // 目的：写入视频时长。
                // 影响：加权评分器据此判定 VQV 权重资格。
                video_duration_ms,
                // 目的：其余字段保持默认。
                // 影响：由其它增强器负责填充。
                ..Default::default()
            };
            // 目的：追加到结果容器。
            // 影响：保持与输入候选顺序一致。
            hydrated_candidates.push(hydrated);
        }

        // 目的：返回增强结果。
        // 影响：流水线据此刷新原候选的视频时长字段。
        Ok(hydrated_candidates)
    }

    // 目的：定义合并增强结果回候选的方法。
    // 影响：把视频时长同步到原候选。
    fn update(&self, candidate: &mut PostCandidate, hydrated: PostCandidate) {
        // 目的：同步视频时长。
        // 影响：候选携带视频特征供评分使用。
        candidate.video_duration_ms = hydrated.video_duration_ms;
    }
}
