// 目的：引用候选辅助 trait，用于获取候选关联的屏幕名映射。
// 影响：后续可调用 candidate.get_screen_names() 生成返回给客户端的名字集合。
use crate::candidate_pipeline::candidate::CandidateHelpers;
// 目的：引用 Phoenix 候选流水线实现。
// 影响：服务器持有该流水线以执行完整的候选生成-筛选-打分-选择流程。
use crate::candidate_pipeline::phoenix_candidate_pipeline::PhoenixCandidatePipeline;
// 目的：引用查询对象类型 ScoredPostsQuery。
// 影响：将 proto 请求转换为流水线内部查询结构。
use crate::candidate_pipeline::query::ScoredPostsQuery;
// 目的：引入 info 日志宏，记录请求与响应信息。
// 影响：请求日志可关联 request_id 排查问题与统计耗时。
use log::info;
// 目的：引入 Arc 智能指针，实现流水线的共享持有。
// 影响：保证流水线在多请求并发访问下安全共享且只初始化一次。
use std::sync::Arc;
// 目的：引入 Instant 时间测量类型，统计单次请求耗时。
// 影响：记录请求处理耗时并写入日志用于性能监控。
use std::time::Instant;
// 目的：引入 tonic 的 Request/Response/Status 类型。
// 影响：实现 gRPC 服务接口的收发与错误状态返回。
use tonic::{Request, Response, Status};
// 目的：引入候选流水线的抽象 trait，用于调用统一的 execute 方法。
// 影响：服务器只需依赖抽象接口执行流水线，与具体实现解耦。
use xai_candidate_pipeline::candidate_pipeline::CandidatePipeline;
// 目的：引入本服务生成的 proto crate 并命名为 pb。
// 影响：统一访问 ScoredPostsQuery/ScoredPost 等 proto 类型。
use xai_home_mixer_proto as pb;
// 目的：直接引入响应用到的 proto 类型 ScoredPost 与 ScoredPostsResponse。
// 影响：减少调用处的类型前缀冗余，构造响应结构更直接。
use xai_home_mixer_proto::{ScoredPost, ScoredPostsResponse};

// 目的：定义 HomeMixerServer 服务端实现结构体。
// 影响：作为 gRPC get_scored_posts 接口的具体后端实现。
pub struct HomeMixerServer {
    // 目的：以 Arc 持有 Phoenix 候选流水线实例。
    // 影响：保证流水线在服务生命周期内只创建一次并被并发请求共享。
    phx_candidate_pipeline: Arc<PhoenixCandidatePipeline>,
}

// 目的：为 HomeMixerServer 实现构造方法。
// 影响：提供统一的实例创建入口。
impl HomeMixerServer {
    // 目的：异步构造 HomeMixerServer。
    // 影响：内部完成全部远程客户端初始化，失败则调用方报错。
    pub async fn new() -> Self {
        // 目的：构造服务器实例。
        // 影响：将创建好的流水线包装进 Arc 存入结构体。
        HomeMixerServer {
            // 目的：调用流水线的 prod() 工厂方法创建生产环境流水线。
            // 影响：一次性初始化 UAS/Phoenix/Thunder/Strato/TES/Gizmoduck/VF 等客户端。
            phx_candidate_pipeline: Arc::new(PhoenixCandidatePipeline::prod().await),
        }
    }
}

// 目的：标记该 impl 块使用 tonic 的异步 trait 支持。
// 影响：使 async trait 方法可被编译为兼容 gRPC 的 Future。
#[tonic::async_trait]
// 目的：为 HomeMixerServer 实现 proto 生成的 ScoredPostsService 服务接口。
// 影响：使服务器满足 gRPC 路由注册要求，接口对外可调用。
impl pb::scored_posts_service_server::ScoredPostsService for HomeMixerServer {
    // 目的：为该方法挂接统计埋点，自动记录调用次数/耗时等指标。
    // 影响：请求指标会上报到监控系统，辅助容量与延迟分析。
    #[xai_stats_macro::receive_stats]
    // 目的：实现 get_scored_posts 异步接口定义。
    // 影响：客户端调用该接口后在此完成全部推荐流水线处理。
    async fn get_scored_posts(
        // 目的：接收 gRPC 上下文自引用。
        // 影响：提供对服务状态（流水线）的访问权限。
        &self,
        // 目的：接收 gRPC 请求，内部包裹 proto 的 ScoredPostsQuery。
        // 影响：请求的元数据（headers 等）与消息体一同进入处理流程。
        request: Request<pb::ScoredPostsQuery>,
    // 目的：声明返回 gRPC 响应包装的 ScoredPostsResponse，失败时返回 Status。
    // 影响：正常返回帖子列表；异常时返回 gRPC 错误码给客户端。
    ) -> Result<Response<ScoredPostsResponse>, Status> {
        // 目的：取出请求内层的 proto 查询对象。
        // 影响：后续全部基于该对象读取参数。
        let proto_query = request.into_inner();

        // 目的：校验 viewer_id 是否合法（必须非零）。
        // 影响：非法请求直接返回 INVALID_ARGUMENT 错误，避免下游对无效用户做无效计算。
        if proto_query.viewer_id == 0 {
            // 目的：返回 invalid_argument 状态并携带错误说明。
            // 影响：客户端收到明确的参数错误，可尽早纠正请求。
            return Err(Status::invalid_argument("viewer_id must be specified"));
        }

        // 目的：记录请求处理开始时间。
        // 影响：用于计算本次请求的端到端耗时。
        let start = Instant::now();
        // 目的：将 proto 请求字段组装为内部 ScoredPostsQuery，并生成唯一 request_id。
        // 影响：请求进入统一的数据结构，便于流水线各阶段传递与日志关联。
        let query = ScoredPostsQuery::new(
            // 目的：透传用户 ID。
            // 影响：决定推荐结果面向的用户。
            proto_query.viewer_id,
            // 目的：透传客户端应用 ID。
            // 影响：用于区分不同客户端应用上下文。
            proto_query.client_app_id,
            // 目的：透传国家码。
            // 影响：作为上下文特征传入模型与过滤。
            proto_query.country_code,
            // 目的：透传语言码。
            // 影响：作为上下文特征传入模型与过滤。
            proto_query.language_code,
            // 目的：透传已见帖子 ID 列表。
            // 影响：用于过滤用户已看过的帖子。
            proto_query.seen_ids,
            // 目的：透传已服务帖子 ID 列表。
            // 影响：用于过滤近期已下发给用户的帖子。
            proto_query.served_ids,
            // 目的：透传是否仅站内内容标记。
            // 影响：决定是否启用站外 Phoenix 召回与缓存副作用。
            proto_query.in_network_only,
            // 目的：透传是否为下拉加载更多的标记。
            // 影响：决定是否启用已服务过滤等加载更多逻辑。
            proto_query.is_bottom_request,
            // 目的：透传布隆过滤器条目。
            // 影响：用于压缩存储客户端已见集合，提升重复过滤效率。
            proto_query.bloom_filter_entries,
        );
        // 目的：记录请求进入流水线前的日志。
        // 影响：通过 request_id 可追踪单个请求全链路日志。
        info!("Scored Posts request - request_id {}", query.request_id);
        // 目的：调用候选流水线执行完整推荐流程。
        // 影响：返回包含已选候选、查询结果等在内的流水线结果对象。
        let pipeline_result = self.phx_candidate_pipeline.execute(query).await;

        // 目的：定义响应帖子列表变量，从流水线选中候选中转换。
        // 影响：决定了最终返回给客户端的帖子集合。
        let scored_posts: Vec<ScoredPost> = pipeline_result
            // 目的：取得流水线最终选中的候选列表。
            // 影响：列表内容就是本次推荐结果。
            .selected_candidates
            // 目的：转为迭代器逐个处理候选。
            // 影响：逐个将候选映射为 proto 结构。
            .into_iter()
            // 目的：将每个 PostCandidate 映射为 ScoredPost。
            // 影响：完成内部结构到对外传输结构的转换。
            .map(|candidate| {
                // 目的：计算该候选关联的用户（作者/转发者）屏幕名映射。
                // 影响：客户端可直接用屏幕名做展示，无需额外查询。
                let screen_names = candidate.get_screen_names();
                // 目的：构造 proto 响应消息条目。
                // 影响：逐字段把候选信息复制到可序列化结构。
                ScoredPost {
                    // 目的：转换帖子 ID 为 u64。
                    // 影响：响应的 tweet_id 字段与该候选一一对应。
                    tweet_id: candidate.tweet_id as u64,
                    // 目的：输出作者 ID。
                    // 影响：客户端据此区分内容作者。
                    author_id: candidate.author_id,
                    // 目的：输出被转发帖 ID，无则置 0。
                    // 影响：客户端可判断该展示为转发形态。
                    retweeted_tweet_id: candidate.retweeted_tweet_id.unwrap_or(0),
                    // 目的：输出被转发用户 ID，无则置 0。
                    // 影响：客户端可展示原始作者信息。
                    retweeted_user_id: candidate.retweeted_user_id.unwrap_or(0),
                    // 目的：输出回复目标帖子 ID，无则置 0。
                    // 影响：客户端可识别回复上下文。
                    in_reply_to_tweet_id: candidate.in_reply_to_tweet_id.unwrap_or(0),
                    // 目的：输出最终分数并转为 f32。
                    // 影响：客户端/前端可依分数做二次排序或展示权重。
                    score: candidate.score.unwrap_or(0.0) as f32,
                    // 目的：输出是否站内标记。
                    // 影响：客户端可区分站内/站外内容的展现样式。
                    in_network: candidate.in_network.unwrap_or(false),
                    // 目的：输出服务类型（如 ForYouPhoenixRetrieval/ForYouInNetwork）。
                    // 影响：客户端可了解内容来源，支持召回侧归因分析。
                    served_type: candidate.served_type.map(|t| t as i32).unwrap_or_default(),
                    // 目的：输出最近一次打分的毫秒时间戳。
                    // 影响：提供新鲜度/时效信息给客户端。
                    last_scored_timestamp_ms: candidate.last_scored_at_ms.unwrap_or(0),
                    // 目的：输出模型打分请求 ID。
                    // 影响：便于按预测请求归因（与 Phoenix 模型日志关联）。
                    prediction_request_id: candidate.prediction_request_id.unwrap_or(0),
                    // 目的：输出会话祖先帖子 ID 列表。
                    // 影响：客户端可构建会话（回复树）展示关系。
                    ancestors: candidate.ancestors,
                    // 目的：输出屏幕名映射。
                    // 影响：直接供客户端渲染作者名。
                    screen_names,
                    // 目的：输出可见性过滤原因（如有）。
                    // 影响：客户端了解该帖子被限制/标记的原因。
                    visibility_reason: candidate.visibility_reason.map(|r| r.into()),
                }
            })
            // 目的：收集所有映射结果为向量。
            // 影响：得到最终响应的帖子列表。
            .collect();

        // 目的：记录响应日志，包含请求 ID、帖子数量与耗时。
        // 影响：提供请求维度的性能与结果规模监控数据。
        info!(
            "Scored Posts response - request_id {} - {} posts ({} ms)",
            // 目的：输出流水线查询对象的请求 ID。
            // 影响：与请求日志配对，完成全链路追踪。
            pipeline_result.query.request_id,
            // 目的：输出结果帖子数量。
            // 影响：可反推选择器规模与过滤强度。
            scored_posts.len(),
            // 目的：输出端到端耗时毫秒数。
            // 影响：作为性能指标的日志形式。
            start.elapsed().as_millis()
        );
        // 目的：包装最终响应并返回。
        // 影响：客户端获得包含 scored_posts 的 gRPC 成功响应。
        Ok(Response::new(ScoredPostsResponse { scored_posts }))
    }
}
