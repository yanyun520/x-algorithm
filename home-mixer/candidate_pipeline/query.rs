// 目的：引入查询特征结构 UserFeatures。
// 影响：查询对象内置用户维度特征字段。
use crate::candidate_pipeline::query_features::UserFeatures;
// 目的：引入请求 ID 生成工具函数。
// 影响：为每个请求生成唯一 ID 便于日志与归因。
use crate::util::request_util::generate_request_id;
// 目的：引入 HasRequestId trait。
// 影响：使查询对象具备统一的请求 ID 访问接口。
use xai_candidate_pipeline::candidate_pipeline::HasRequestId;
// 目的：引入 proto 的布隆过滤器条目类型。
// 影响：接收客户端压缩后的已见集合信息。
use xai_home_mixer_proto::ImpressionBloomFilterEntry;
// 目的：引入 Twitter 上下文相关 trait 与结构。
// 影响：可将查询转换为 TwitterContextViewer 供可见性过滤(VF)使用。
use xai_twittercontext_proto::{GetTwitterContextViewer, TwitterContextViewer};

// 目的：为 ScoredPostsQuery 派生 Clone/Default/Debug。
// 影响：支持查询对象整体复制、默认构造与调试打印，便于流水线各阶段产生派生副本。
#[derive(Clone, Default, Debug)]
// 目的：定义贯穿候选流水线的查询上下文对象。
// 影响：承载请求参数，并作为各阶段传递和更新的核心数据载体。
pub struct ScoredPostsQuery {
    // 目的：记录查看者用户 ID。
    // 影响：决定推荐对象与过滤依据。
    pub user_id: i64,
    // 目的：记录客户端应用 ID。
    // 影响：作为上下文特征透传给模型与 VF。
    pub client_app_id: i32,
    // 目的：记录请求国家码。
    // 影响：作为上下文特征影响召回与过滤。
    pub country_code: String,
    // 目的：记录请求语言码。
    // 影响：作为上下文特征影响召回与过滤。
    pub language_code: String,
    // 目的：记录用户已见帖子 ID 列表。
    // 影响：用于过滤用户已经看过的帖子。
    pub seen_ids: Vec<i64>,
    // 目的：记录近期已服务给用户的帖子 ID 列表。
    // 影响：用于减少重复下发。
    pub served_ids: Vec<i64>,
    // 目的：标记是否仅要站内内容。
    // 影响：控制 Phoenix 站外召回与缓存副作用是否启用。
    pub in_network_only: bool,
    // 目的：标记是否为下拉加载更多（bottom）请求。
    // 影响：控制已服务过滤等加载更多逻辑。
    pub is_bottom_request: bool,
    // 目的：记录客户端上报的布隆过滤器条目。
    // 影响：参与已见帖子过滤，压缩重复信息传输。
    pub bloom_filter_entries: Vec<ImpressionBloomFilterEntry>,
    // 目的：记录用户行为序列（由查询增强阶段填充）。
    // 影响：作为 Phoenix 召回与打分的核心特征。
    pub user_action_sequence: Option<xai_recsys_proto::UserActionSequence>,
    // 目的：记录用户静态特征（静音、拉黑、关注等）。
    // 影响：驱动多个过滤器的行为。
    pub user_features: UserFeatures,
    // 目的：记录本次请求的唯一 ID。
    // 影响：贯穿日志与远程调用归因，端到端追踪。
    pub request_id: String,
}

// 目的：为 ScoredPostsQuery 实现构造函数。
// 影响：提供从 proto 请求参数到内部查询对象的转换入口。
impl ScoredPostsQuery {
    // 目的：定义构造函数签名，接收全部客户端参数。
    // 影响：外部只需传入请求字段即可得到查询对象。
    pub fn new(
        // 目的：接收查看者用户 ID。
        // 影响：绑定本次推荐针对的用户。
        user_id: i64,
        // 目的：接收客户端应用 ID。
        // 影响：绑定客户端上下文。
        client_app_id: i32,
        // 目的：接收国家码。
        // 影响：作为上下文特征。
        country_code: String,
        // 目的：接收语言码。
        // 影响：作为上下文特征。
        language_code: String,
        // 目的：接收已见帖子 ID。
        // 影响：用于重复过滤。
        seen_ids: Vec<i64>,
        // 目的：接收已服务帖子 ID。
        // 影响：用于减少重复下发。
        served_ids: Vec<i64>,
        // 目的：接收仅站内标记。
        // 影响：控制召回通道选择。
        in_network_only: bool,
        // 目的：接收 bottom 请求标记。
        // 影响：控制加载更多逻辑。
        is_bottom_request: bool,
        // 目的：接收布隆过滤器条目。
        // 影响：参与已见过滤。
        bloom_filter_entries: Vec<ImpressionBloomFilterEntry>,
    // 目的：声明返回 Self 类型。
    // 影响：链式构建后返回完整查询对象。
    ) -> Self {
        // 目的：拼接生成请求 ID（随机ID + 用户ID）。
        // 影响：使同一用户在连续请求中也能区分不同 request_id，便于日志追踪。
        let request_id = format!("{}-{}", generate_request_id(), user_id);
        // 目的：构建查询对象本体。
        // 影响：所有参数就位，等待流水线各阶段填充特征。
        Self {
            // 目的：保存查看者用户 ID。
            // 影响：后续阶段读取决策对象。
            user_id,
            // 目的：保存客户端应用 ID。
            // 影响：作为上下文特征。
            client_app_id,
            // 目的：保存国家码。
            // 影响：作为上下文特征。
            country_code,
            // 目的：保存语言码。
            // 影响：作为上下文特征。
            language_code,
            // 目的：保存已见帖子 ID。
            // 影响：供重复过滤使用。
            seen_ids,
            // 目的：保存已服务帖子 ID。
            // 影响：供已服务过滤使用。
            served_ids,
            // 目的：保存仅站内标记。
            // 影响：控制召回与副作用。
            in_network_only,
            // 目的：保存 bottom 标记。
            // 影响：控制加载更多过滤。
            is_bottom_request,
            // 目的：保存布隆过滤器条目。
            // 影响：供已见过滤加速。
            bloom_filter_entries,
            // 目的：动作序列初始为空。
            // 影响：由查询增强阶段后续填充。
            user_action_sequence: None,
            // 目的：用户特征初始化为默认空值。
            // 影响：由用户特征增强阶段后续填充。
            user_features: UserFeatures::default(),
            // 目的：保存生成的请求 ID。
            // 影响：贯穿全链路日志。
            request_id,
        }
    }
}

// 目的：让 ScoredPostsQuery 实现 GetTwitterContextViewer，将查询转换为 TV 上下文。
// 影响：使 VF 可见性过滤客户端无需感知查询内部结构。
impl GetTwitterContextViewer for ScoredPostsQuery {
    // 目的：实现 get_viewer 方法。
    // 影响：返回包装好的 TwitterContextViewer 供 VF 使用。
    fn get_viewer(&self) -> Option<TwitterContextViewer> {
        // 目的：构造并返回查看者上下文。
        // 影响：向 VF 提供用户/应用/国家/语言等安全判定上下文。
        Some(TwitterContextViewer {
            // 目的：透传查看者用户 ID。
            // 影响：安全策略按用户维度判定。
            user_id: self.user_id,
            // 目的：透传客户端应用 ID（转为 i64）。
            // 影响：安全策略可区分不同客户端。
            client_application_id: self.client_app_id as i64,
            // 目的：透传国家码副本。
            // 影响：处理按地区的合规策略。
            request_country_code: self.country_code.clone(),
            // 目的：透传语言码副本。
            // 影响：处理语言相关策略。
            request_language_code: self.language_code.clone(),
            // 目的：使用默认值填充其余字段。
            // 影响：未提供的上下文项取默认，保证结构可构建。
            ..Default::default()
        })
    }
}

// 目的：让 ScoredPostsQuery 实现 HasRequestId。
// 影响：流水线框架可统一读取查询的请求 ID。
impl HasRequestId for ScoredPostsQuery {
    // 目的：实现请求 ID 访问方法。
    // 影响：框架层可按 ID 记录日志与关联数据。
    fn request_id(&self) -> &str {
        // 目的：返回内部保存的请求 ID。
        // 影响：调用方获得用于日志/追踪的标识。
        &self.request_id
    }
}
