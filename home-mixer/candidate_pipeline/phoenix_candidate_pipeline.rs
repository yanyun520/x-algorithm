// 目的：引入核心数据增强器，用于补全候选的作者/正文/回复转发字段。
// 影响：流水线候选在打分前获得可展示所需的基础内容数据。
use crate::candidate_hydrators::core_data_candidate_hydrator::CoreDataCandidateHydrator;
// 目的：引入用户资料增强器，用于补全候选作者/转发者的屏幕名与粉丝数。
// 影响：为展示层与可能需要作者规模的逻辑提供数据。
use crate::candidate_hydrators::gizmoduck_hydrator::GizmoduckCandidateHydrator;
// 目的：引入站内判定增强器，标记候选是否属于站内内容。
// 影响：为 OON 加权与可见性分流提供依据。
use crate::candidate_hydrators::in_network_candidate_hydrator::InNetworkCandidateHydrator;
// 目的：引入订阅作者增强器，标记帖子的订阅作者。
// 影响：供订阅资格过滤判断付费内容可见性。
use crate::candidate_hydrators::subscription_hydrator::SubscriptionHydrator;
// 目的：引入可见性过滤增强器，获取候选的安全过滤原因。
// 影响：为后置 VFFilter 提供过滤决策数据。
use crate::candidate_hydrators::vf_candidate_hydrator::VFCandidateHydrator;
// 目的：引入视频时长增强器，提取候选视频时长。
// 影响：为 VQV 权重资格判定提供视频信息。
use crate::candidate_hydrators::video_duration_candidate_hydrator::VideoDurationCandidateHydrator;
// 目的：引入候选结构 PostCandidate。
// 影响：流水线各阶段统一操作该候选类型。
use crate::candidate_pipeline::candidate::PostCandidate;
// 目的：引入查询对象 ScoredPostsQuery。
// 影响：流水线各阶段统一接收该查询上下文。
use crate::candidate_pipeline::query::ScoredPostsQuery;
// 目的：引入 Gizmoduck 客户端抽象与生产实现。
// 影响：为池创建用户资料客户端提供类型。
use crate::clients::gizmoduck_client::{GizmoduckClient, ProdGizmoduckClient};
// 目的：引入 Phoenix 预测客户端抽象与生产实现。
// 影响：为候选打分提供模型调用能力。
use crate::clients::phoenix_prediction_client::{
    PhoenixPredictionClient, ProdPhoenixPredictionClient,
};
// 目的：引入 Phoenix 检索客户端抽象与生产实现。
// 影响：为站点外召回提供模型检索能力。
use crate::clients::phoenix_retrieval_client::{
    PhoenixRetrievalClient, ProdPhoenixRetrievalClient,
};
// 目的：引入 S2S（服务到服务）TLS 证书路径常量。
// 影响：用于创建带双向认证的可见性过滤客户端。
use crate::clients::s2s::{S2S_CHAIN_PATH, S2S_CRT_PATH, S2S_KEY_PATH};
// 目的：引入社交关系图客户端（当前保留引用）。
// 影响：预留未来基于社交关系图的特征/过滤能力。
use crate::clients::socialgraph_client::SocialGraphClient;
// 目的：引入 Strato 存储客户端抽象与生产实现。
// 影响：为用户特征读写与请求信息缓存提供存储通道。
use crate::clients::strato_client::{ProdStratoClient, StratoClient};
// 目的：引入 Thunder 客户端。
// 影响：为站内好友动态召回提供通道选择能力。
use crate::clients::thunder_client::ThunderClient;
// 目的：引入 TES（推文实体服务）客户端抽象与生产实现。
// 影响：为核心数据、视频时长、订阅作者等获取推文实体信息。
use crate::clients::tweet_entity_service_client::{ProdTESClient, TESClient};
// 目的：引入用户行为序列获取器。
// 影响：为查询增强阶段拉取用户近期行为序列。
use crate::clients::uas_fetcher::UserActionSequenceFetcher;
// 目的：引入帖龄过滤器，剔除超过最大时长的帖子。
// 影响：保证推荐内容新鲜度。
use crate::filters::age_filter::AgeFilter;
// 目的：引入社交关系过滤器，剔除拉黑/静音作者内容。
// 影响：尊重用户的社交设置。
use crate::filters::author_socialgraph_filter::AuthorSocialgraphFilter;
// 目的：引入核心数据完整性过滤器。
// 影响：剔除缺少作者或正文的脏候选。
use crate::filters::core_data_hydration_filter::CoreDataHydrationFilter;
// 目的：引入会话去重过滤器。
// 影响：同一会话树仅保留最高分候选。
use crate::filters::dedup_conversation_filter::DedupConversationFilter;
// 目的：引入重复帖过滤器。
// 影响：去除重复 tweet_id。
use crate::filters::drop_duplicates_filter::DropDuplicatesFilter;
// 目的：引入订阅资格过滤器。
// 影响：剔除用户未订阅的付费内容。
use crate::filters::ineligible_subscription_filter::IneligibleSubscriptionFilter;
// 目的：引入静音关键词过滤器。
// 影响：剔除命中用户静音关键词的帖子。
use crate::filters::muted_keyword_filter::MutedKeywordFilter;
// 目的：引入已见帖子过滤器。
// 影响：剔除用户已经看过的帖子。
use crate::filters::previously_seen_posts_filter::PreviouslySeenPostsFilter;
// 目的：引入已服务帖子过滤器。
// 影响：在加载更多场景减少重复下发。
use crate::filters::previously_served_posts_filter::PreviouslyServedPostsFilter;
// 目的：引入转发去重过滤器。
// 影响：同一原始帖只保留一条。
use crate::filters::retweet_deduplication_filter::RetweetDeduplicationFilter;
// 目的：引入本人帖子过滤器。
// 影响：剔除用户自己发布的帖子。
use crate::filters::self_tweet_filter::SelfTweetFilter;
// 目的：引入可见性（安全）过滤器。
// 影响：依据 VF 结果剔除不符合安全策略的内容。
use crate::filters::vf_filter::VFFilter;
// 目的：引入全局参数模块。
// 影响：读取 MAX_POST_AGE、RESULT_SIZE 等配置。
use crate::params;
// 目的：引入用户行为序列查询增强器。
// 影响：在召回前补充用户行为序列到查询。
use crate::query_hydrators::user_action_seq_query_hydrator::UserActionSeqQueryHydrator;
// 目的：引入用户特征查询增强器。
// 影响：在召回前补充用户静态特征到查询。
use crate::query_hydrators::user_features_query_hydrator::UserFeaturesQueryHydrator;
// 目的：引入作者多样性打分器。
// 影响：提升作者多样性，抑制同作者刷屏。
use crate::scorers::author_diversity_scorer::AuthorDiversityScorer;
// 目的：引入 OON 打分器。
// 影响：对站外内容降权。
use crate::scorers::oon_scorer::OONScorer;
// 目的：引入 Phoenix 打分器。
// 影响：调用模型产出候选各行为概率。
use crate::scorers::phoenix_scorer::PhoenixScorer;
// 目的：引入加权评分器。
// 影响：汇总各行为概率为候选总分。
use crate::scorers::weighted_scorer::WeightedScorer;
// 目的：引入 TopK 选择器。
// 影响：按分数取前 K 个候选作为最终结果。
use crate::selectors::TopKScoreSelector;
// 目的：引入请求信息缓存副作用。
// 影响：请求结束后把结果写缓存。
use crate::side_effects::cache_request_info_side_effect::CacheRequestInfoSideEffect;
// 目的：引入 Phoenix 召回源。
// 影响：提供站外候选入口。
use crate::sources::phoenix_source::PhoenixSource;
// 目的：引入 Thunder 召回源。
// 影响：提供站内好友动态候选入口。
use crate::sources::thunder_source::ThunderSource;
// 目的：引入 Arc 智能指针。
// 影响：共享远程客户端，避免重复连接开销。
use std::sync::Arc;
// 目的：引入 Duration 类型。
// 影响：构造帖龄过滤的时间窗口。
use std::time::Duration;
// 目的：引入 tonic 的 async_trait 支持。
// 影响：使流水线 trait 实现支持异步方法。
use tonic::async_trait;
// 目的：引入候选流水线核心 trait。
// 影响：统一流水线的执行契约。
use xai_candidate_pipeline::candidate_pipeline::CandidatePipeline;
// 目的：引入过滤器 trait。
// 影响：定义流水线各过滤器的接口。
use xai_candidate_pipeline::filter::Filter;
// 目的：引入增强器 trait。
// 影响：定义候选增强阶段接口。
use xai_candidate_pipeline::hydrator::Hydrator;
// 目的：引入查询增强器 trait。
// 影响：定义查询增强阶段接口。
use xai_candidate_pipeline::query_hydrator::QueryHydrator;
// 目的：引入打分器 trait。
// 影响：定义候选打分阶段接口。
use xai_candidate_pipeline::scorer::Scorer;
// 目的：引入选择器 trait。
// 影响：定义候选选择阶段接口。
use xai_candidate_pipeline::selector::Selector;
// 目的：引入副作用 trait。
// 影响：定义流水线结束后的旁路操作接口。
use xai_candidate_pipeline::side_effect::SideEffect;
// 目的：引入召回源 trait。
// 影响：定义候选召回阶段的接口。
use xai_candidate_pipeline::source::Source;
// 目的：引入可见性过滤客户端抽象与生产实现。
// 影响：为池创建 VF 客户端。
use xai_visibility_filtering::vf_client::{
    ProdVisibilityFilteringClient, VisibilityFilteringClient,
};

// 目的：定义 Phoenix 候选流水线结构，持有各阶段组件。
// 影响：作为本服务推荐流程的编排容器，集中管理所有阶段。
pub struct PhoenixCandidatePipeline {
    // 目的：查询增强组件列表（用户序列、用户特征）。
    // 影响：执行顺序上最先运行，为召回准备上下文。
    query_hydrators: Vec<Box<dyn QueryHydrator<ScoredPostsQuery>>>,
    // 目的：召回源组件列表（Phoenix 站外、Thunder 站内）。
    // 影响：决定候选的最初来源。
    sources: Vec<Box<dyn Source<ScoredPostsQuery, PostCandidate>>>,
    // 目的：候选增强组件列表（站内标记、核心数据、视频时长、订阅、用户资料）。
    // 影响：在打分前补齐候选特征。
    hydrators: Vec<Box<dyn Hydrator<ScoredPostsQuery, PostCandidate>>>,
    // 目的：候选过滤组件列表。
    // 影响：按规则剔除不合格候选，控制候选质量。
    filters: Vec<Box<dyn Filter<ScoredPostsQuery, PostCandidate>>>,
    // 目的：打分组件列表（Phoenix、加权、作者多样性、OON）。
    // 影响：依次产出候选的最终排序分数。
    scorers: Vec<Box<dyn Scorer<ScoredPostsQuery, PostCandidate>>>,
    // 目的：选择器（TopK）。
    // 影响：决定最终返回的候选集合。
    selector: TopKScoreSelector,
    // 目的：选后增强组件列表（可见性过滤）。
    // 影响：对已选候选补充安全过滤所需数据。
    post_selection_hydrators: Vec<Box<dyn Hydrator<ScoredPostsQuery, PostCandidate>>>,
    // 目的：选后过滤组件列表（安全过滤、会话去重）。
    // 影响：对最终候选做安全与展示层约束。
    post_selection_filters: Vec<Box<dyn Filter<ScoredPostsQuery, PostCandidate>>>,
    // 目的：副作用组件列表（缓存请求信息）。
    // 影响：请求结束前落缓存，优化后续体验。
    side_effects: Arc<Vec<Box<dyn SideEffect<ScoredPostsQuery, PostCandidate>>>>,
}

// 目的：为 PhoenixCandidatePipeline 实现构建方法。
// 影响：集中完成所有阶段组件的组装。
impl PhoenixCandidatePipeline {
    // 目的：以显式传入的客户端参数构建流水线（便于测试注入桩客户端）。
    // 影响：生产与测试环境共用同一套组装逻辑。
    async fn build_with_clients(
        // 目的：接收用户行为序列获取器。
        // 影响：用于构造查询增强器；内部可注入 mock 便于测试。
        uas_fetcher: Arc<UserActionSequenceFetcher>,
        // 目的：接收 Phoenix 预测客户端。
        // 影响：用于候选打分阶段调用模型。
        phoenix_client: Arc<dyn PhoenixPredictionClient + Send + Sync>,
        // 目的：接收 Phoenix 检索客户端。
        // 影响：用于站外候选召回。
        phoenix_retrieval_client: Arc<dyn PhoenixRetrievalClient + Send + Sync>,
        // 目的：接收 Thunder 客户端。
        // 影响：用于站内候选召回。
        thunder_client: Arc<ThunderClient>,
        // 目的：接收 Strato 客户端。
        // 影响：用于用户特征读写与请求信息缓存。
        strato_client: Arc<dyn StratoClient + Send + Sync>,
        // 目的：接收 TES 客户端。
        // 影响：用于核心数据/视频时长/订阅作者等实体信息。
        tes_client: Arc<dyn TESClient + Send + Sync>,
        // 目的：接收 Gizmoduck 客户端。
        // 影响：用于用户资料（屏幕名、粉丝数）获取。
        gizmoduck_client: Arc<dyn GizmoduckClient + Send + Sync>,
        // 目的：接收可见性过滤客户端。
        // 影响：用于选后安全过滤判定。
        vf_client: Arc<dyn VisibilityFilteringClient + Send + Sync>,
    // 目的：声明返回值类型为完整流水线。
    // 影响：组装完成后返回可执行的流水线实例。
    ) -> PhoenixCandidatePipeline {
        // Query Hydrators
        // 目的：组装查询增强组件列表。
        // 影响：在候选召回前依次执行，为查询补充上下文。
        let query_hydrators: Vec<Box<dyn QueryHydrator<ScoredPostsQuery>>> = vec![
            // 目的：实例化用户行为序列增强器并装箱。
            // 影响：拉取并聚合用户行为序列注入查询。
            Box::new(UserActionSeqQueryHydrator::new(uas_fetcher)),
            // 目的：实例化用户特征增强器并装箱。
            // 影响：从 Strato 读取用户静态特征注入查询。
            Box::new(UserFeaturesQueryHydrator {
                // 目的：注入克隆的 Strato 客户端。
                // 影响：该增强器独享一个客户端引用（共享连接）。
                strato_client: strato_client.clone(),
            }),
        ];

        // Sources
        // 目的：构建 Phoenix 站外召回源并装箱。
        // 影响：站外候选入口。
        let phoenix_source = Box::new(PhoenixSource {
            // 目的：将检索客户端注入源。
            // 影响：源可调用模型检索站外候选。
            phoenix_retrieval_client,
        });
        // 目的：构建 Thunder 站内召回源并装箱。
        // 影响：站内好友动态入口。
        let thunder_source = Box::new(ThunderSource { thunder_client });
        // 目的：整理召回源列表。
        // 影响：流水线将并行/顺序调用各源获取候选。
        let sources: Vec<Box<dyn Source<ScoredPostsQuery, PostCandidate>>> =
            vec![phoenix_source, thunder_source];

        // Hydrators
        // 目的：组装候选增强组件列表。
        // 影响：在候选召回后、过滤前逐个补齐候选特征。
        let hydrators: Vec<Box<dyn Hydrator<ScoredPostsQuery, PostCandidate>>> = vec![
            // 目的：实例化站内标记增强器。
            // 影响：标记候选是否为站内内容。
            Box::new(InNetworkCandidateHydrator),
            // 目的：实例化核心数据增强器（异步初始化 TES 引用）。
            // 影响：补全候选的作者/正文/转发回复字段。
            Box::new(CoreDataCandidateHydrator::new(tes_client.clone()).await),
            // 目的：实例化视频时长增强器。
            // 影响：补全候选的视频时长特征。
            Box::new(VideoDurationCandidateHydrator::new(tes_client.clone()).await),
            // 目的：实例化订阅作者增强器。
            // 影响：补全候选的订阅作者标识。
            Box::new(SubscriptionHydrator::new(tes_client.clone()).await),
            // 目的：实例化用户资料增强器。
            // 影响：补全候选作者/转发者的屏幕名与粉丝数。
            Box::new(GizmoduckCandidateHydrator::new(gizmoduck_client).await),
        ];

        // Filters
        // 目的：组装候选过滤组件列表。
        // 影响：在打分前按要求剔除不合格候选。
        let filters: Vec<Box<dyn Filter<ScoredPostsQuery, PostCandidate>>> = vec![
            // 目的：注册重复帖过滤。
            // 影响：先去除重复 tweet_id。
            Box::new(DropDuplicatesFilter),
            // 目的：注册核心数据完整性过滤。
            // 影响：剔除缺少作者或正文的候选。
            Box::new(CoreDataHydrationFilter),
            // 目的：注册帖龄过滤，窗口取 MAX_POST_AGE 秒。
            // 影响：剔除超龄旧帖。
            Box::new(AgeFilter::new(Duration::from_secs(params::MAX_POST_AGE))),
            // 目的：注册本人帖子过滤。
            // 影响：剔除用户自己的帖子。
            Box::new(SelfTweetFilter),
            // 目的：注册转发去重过滤。
            // 影响：同一原始帖仅保留一条。
            Box::new(RetweetDeduplicationFilter),
            // 目的：注册订阅资格过滤。
            // 影响：剔除用户未订阅的付费内容。
            Box::new(IneligibleSubscriptionFilter),
            // 目的：注册已见帖子过滤。
            // 影响：剔除客户端已上报看过的帖子。
            Box::new(PreviouslySeenPostsFilter),
            // 目的：注册已服务帖子过滤。
            // 影响：加载更多场景下减少重复下发。
            Box::new(PreviouslyServedPostsFilter),
            // 目的：注册静音关键词过滤。
            // 影响：剔除命中用户静音关键词的帖子。
            Box::new(MutedKeywordFilter::new()),
            // 目的：注册社交关系过滤。
            // 影响：剔除拉黑/静音作者内容。
            Box::new(AuthorSocialgraphFilter),
        ];

        // Scorers
        // 目的：构造 Phoenix 打分器并装箱。
        // 影响：调用模型产出候选各行为概率。
        let phoenix_scorer = Box::new(PhoenixScorer { phoenix_client });
        // 目的：构造加权评分器并装箱。
        // 影响：汇总行为概率为候选总分（weighted_score）。
        let weighted_scorer = Box::new(WeightedScorer);
        // 目的：构造作者多样性评分器并装箱。
        // 影响：按作者出现次数衰减分数。
        let author_diversity_scorer = Box::new(AuthorDiversityScorer::default());
        // 目的：构造 OON 评分器并装箱。
        // 影响：对站外内容降权。
        let oon_scorer = Box::new(OONScorer);
        // 目的：整理打分组件列表。
        // 影响：流水线按顺序执行打分并叠加更新候选。
        let scorers: Vec<Box<dyn Scorer<ScoredPostsQuery, PostCandidate>>> = vec![
            phoenix_scorer,
            weighted_scorer,
            author_diversity_scorer,
            oon_scorer,
        ];

        // Selector
        // 目的：实例化 TopK 选择器。
        // 影响：按最终分数取前 K 个候选。
        let selector = TopKScoreSelector;

        // Post-selection hydrators
        // 目的：组装选后增强组件列表。
        // 影响：对已选候选补充安全过滤所需数据。
        let post_selection_hydrators: Vec<Box<dyn Hydrator<ScoredPostsQuery, PostCandidate>>> =
            // 目的：实例化可见性过滤增强器。
            // 影响：异步获取候选安全过滤原因。
            vec![Box::new(VFCandidateHydrator::new(vf_client.clone()).await)];

        // Post-selection filters
        // 目的：组装选后过滤组件列表。
        // 影响：对最终候选做展示层约束。
        let post_selection_filters: Vec<Box<dyn Filter<ScoredPostsQuery, PostCandidate>>> =
            // 目的：注册安全过滤与会话去重过滤。
            // 影响：剔除安全不合规与同会话低分候选。
            vec![Box::new(VFFilter), Box::new(DedupConversationFilter)];

        // Side Effects
        // 目的：组装副作用组件列表。
        // 影响：请求结束前执行旁路操作。
        let side_effects: Arc<Vec<Box<dyn SideEffect<ScoredPostsQuery, PostCandidate>>>> =
            // 目的：实例化请求信息缓存副作用。
            // 影响：把本次服务结果写入 Strato 缓存。
            Arc::new(vec![Box::new(CacheRequestInfoSideEffect { strato_client })]);

        // 目的：汇总所有组件为最终流水线对象。
        // 影响：返回后即可被 CandidatePipeline::execute 驱动执行。
        PhoenixCandidatePipeline {
            // 目的：保存查询增强组件。
            // 影响：查询阶段调用。
            query_hydrators,
            // 目的：保存候选增强组件。
            // 影响：增强阶段调用。
            hydrators,
            // 目的：保存过滤组件。
            // 影响：过滤阶段调用。
            filters,
            // 目的：保存召回源组件。
            // 影响：召回阶段调用。
            sources,
            // 目的：保存打分组件。
            // 影响：打分阶段调用。
            scorers,
            // 目的：保存选择器。
            // 影响：选择阶段调用。
            selector,
            // 目的：保存选后增强组件。
            // 影响：选后增强阶段调用。
            post_selection_hydrators,
            // 目的：保存选后过滤组件。
            // 影响：选后过滤阶段调用。
            post_selection_filters,
            // 目的：保存副作用组件。
            // 影响：流水线收尾阶段调用。
            side_effects,
        }
    }

    // 目的：生产环境工厂方法，创建并连接全部真实远程客户端。
    // 影响：供服务器启动入口调用；任一客户端初始化失败则 panic 退出。
    pub async fn prod() -> PhoenixCandidatePipeline {
        // 目的：创建用户行为序列获取器并包装为 Arc。
        // 影响：失败时直接 panic，确保启动即暴露配置错误。
        let uas_fetcher =
            Arc::new(UserActionSequenceFetcher::new().expect("Failed to create UAS fetcher"));
        // 目的：创建社交关系图客户端（当前保留引用未使用）。
        // 影响：预留能力；变量名以 _ 开头避免未使用告警。
        let _sgs_client = Arc::new(SocialGraphClient::new());
        // 目的：创建 Phoenix 预测客户端并包装为 Arc。
        // 影响：失败时 panic，保证打分能力可用。
        let phoenix_client = Arc::new(
            // 目的：调用生产客户端构造函数。
            // 影响：完成与 Phoenix 服务的连接建立。
            ProdPhoenixPredictionClient::new()
                // 目的：等待连接完成。
                // 影响：得到可用的客户端实例。
                .await
                // 目的：初始化失败则终止启动并给出明确错误。
                // 影响：避免运行期才暴露出连接问题。
                .expect("Failed to create Phoenix prediction client"),
        );
        // 目的：创建 Phoenix 检索客户端并包装为 Arc。
        // 影响：失败时 panic。
        let phoenix_retrieval_client = Arc::new(
            // 目的：调用生产检索客户端构造函数。
            // 影响：完成与检索服务连接。
            ProdPhoenixRetrievalClient::new()
                // 目的：等待连接完成。
                // 影响：得到可用的检索客户端。
                .await
                // 目的：失败即终止。
                // 影响：保证召回入口可用。
                .expect("Failed to create Phoenix retrieval client"),
        );
        // 目的：创建 Thunder 客户端并包装为 Arc。
        // 影响：为站内召回提供通道。
        let thunder_client = Arc::new(ThunderClient::new().await);
        // 目的：创建 Strato 客户端并包装为 Arc。
        // 影响：失败时 panic，保证存储能力可用。
        let strato_client = Arc::new(
            // 目的：调用生产 Strato 构造函数。
            // 影响：完成与存储服务连接。
            ProdStratoClient::new()
                // 目的：等待连接完成。
                // 影响：得到可用的存储客户端。
                .await
                // 目的：失败即终止。
                // 影响：保证用户特征读写可用。
                .expect("Failed to create Strato client"),
        );
        // 目的：创建 TES 客户端并包装为 Arc。
        // 影响：失败时 panic，保证推文实体查询可用。
        let tes_client = Arc::new(
            // 目的：调用生产 TES 构造函数。
            // 影响：完成与推文实体服务连接。
            ProdTESClient::new()
                // 目的：等待连接完成。
                // 影响：得到可用的 TES 客户端。
                .await
                // 目的：失败即终止。
                // 影响：保证增强阶段依赖可用。
                .expect("Failed to create TES client"),
        );
        // 目的：创建 Gizmoduck 客户端并包装为 Arc。
        // 影响：失败时 panic，保证用户资料查询可用。
        let gizmoduck_client = Arc::new(
            // 目的：调用生产 Gizmoduck 构造函数。
            // 影响：完成与用户资料服务连接。
            ProdGizmoduckClient::new()
                // 目的：等待连接完成。
                // 影响：得到可用的用户资料客户端。
                .await
                // 目的：失败即终止。
                // 影响：保证屏幕名/粉丝数可用。
                .expect("Failed to create Gizmoduck client"),
        );
        // 目的：创建可见性过滤客户端并包装为 Arc。
        // 影响：失败时 panic，保证选后安全过滤能力。
        let vf_client = Arc::new(
            // 目的：调用生产 VF 构造函数。
            // 影响：完成与安全过滤服务连接。
            ProdVisibilityFilteringClient::new(
                // 目的：传入 S2S 证书链路径。
                // 影响：用于双向 TLS 认证。
                S2S_CHAIN_PATH.clone(),
                // 目的：传入 S2S 客户端证书路径。
                // 影响：用于服务身份认证。
                S2S_CRT_PATH.clone(),
                // 目的：传入 S2S 私钥路径。
                // 影响：用于 TLS 握手签名。
                S2S_KEY_PATH.clone()
            )
            // 目的：等待连接完成。
            // 影响：得到可用的 VF 客户端。
            .await
            // 目的：失败即终止。
            // 影响：保证安全过滤链路完整。
            .expect("Failed to create VF client"),
        );
        // 目的：调用统一组装函数完成流水线构建。
        // 影响：返回完整可执行流水线。
        PhoenixCandidatePipeline::build_with_clients(
            // 目的：传入 UAS 获取器。
            // 影响：供查询增强阶段使用。
            uas_fetcher,
            // 目的：传入 Phoenix 预测客户端。
            // 影响：供打分阶段使用。
            phoenix_client,
            // 目的：传入 Phoenix 检索客户端。
            // 影响：供站外召回使用。
            phoenix_retrieval_client,
            // 目的：传入 Thunder 客户端。
            // 影响：供站内召回使用。
            thunder_client,
            // 目的：传入 Strato 客户端。
            // 影响：供特征与缓存使用。
            strato_client,
            // 目的：传入 TES 客户端。
            // 影响：供实体增强使用。
            tes_client,
            // 目的：传入 Gizmoduck 客户端。
            // 影响：供用户资料增强使用。
            gizmoduck_client,
            // 目的：传入 VF 客户端。
            // 影响：供选后安全过滤使用。
            vf_client,
        )
        // 目的：等待异步组装完成。
        // 影响：返回最终流水线对象。
        .await
    }
}

// 目的：标记使用异步 trait 支持。
// 影响：使该 impl 块中的方法可被框架以 async 方式调度。
#[async_trait]
// 目的：为 PhoenixCandidatePipeline 实现候选流水线抽象 trait。
// 影响：流水线框架可统一驱动本实现完成各阶段执行。
impl CandidatePipeline<ScoredPostsQuery, PostCandidate> for PhoenixCandidatePipeline {
    // 目的：向框架暴露查询增强组件切片。
    // 影响：框架在查询阶段依次调用这些组件。
    fn query_hydrators(&self) -> &[Box<dyn QueryHydrator<ScoredPostsQuery>>] {
        // 目的：返回内部持有的查询增强组件。
        // 影响：框架据此执行查询上下文增强。
        &self.query_hydrators
    }

    // 目的：向框架暴露召回源组件切片。
    // 影响：框架在召回阶段调用各源获取候选。
    fn sources(&self) -> &[Box<dyn Source<ScoredPostsQuery, PostCandidate>>] {
        // 目的：返回内部持有的召回源组件。
        // 影响：框架据此执行候选召回。
        &self.sources
    }
    // 目的：向框架暴露候选增强组件切片。
    // 影响：框架在增强阶段调用各组件补齐候选特征。
    fn hydrators(&self) -> &[Box<dyn Hydrator<ScoredPostsQuery, PostCandidate>>] {
        // 目的：返回内部持有的增强组件。
        // 影响：框架据此执行候选增强。
        &self.hydrators
    }

    // 目的：向框架暴露过滤组件切片。
    // 影响：框架在过滤阶段依次执行过滤。
    fn filters(&self) -> &[Box<dyn Filter<ScoredPostsQuery, PostCandidate>>] {
        // 目的：返回内部持有的过滤组件。
        // 影响：框架据此剔除不合格候选。
        &self.filters
    }

    // 目的：向框架暴露打分组件切片。
    // 影响：框架在打分阶段依次执行打分。
    fn scorers(&self) -> &[Box<dyn Scorer<ScoredPostsQuery, PostCandidate>>] {
        // 目的：返回内部持有的打分组件。
        // 影响：框架据此计算候选分数。
        &self.scorers
    }

    // 目的：向框架暴露选择器。
    // 影响：框架在选择阶段按分数选取最终候选。
    fn selector(&self) -> &dyn Selector<ScoredPostsQuery, PostCandidate> {
        // 目的：返回内部持有的 TopK 选择器。
        // 影响：框架据此获取选择行为与结果规模。
        &self.selector
    }

    // 目的：向框架暴露选后增强组件切片。
    // 影响：框架在选后阶段补齐最终候选特征。
    fn post_selection_hydrators(&self) -> &[Box<dyn Hydrator<ScoredPostsQuery, PostCandidate>>] {
        // 目的：返回内部持有的选后增强组件。
        // 影响：框架据此为已选候选补充安全数据。
        &self.post_selection_hydrators
    }

    // 目的：向框架暴露选后过滤组件切片。
    // 影响：框架在选后阶段对最终候选施加约束。
    fn post_selection_filters(&self) -> &[Box<dyn Filter<ScoredPostsQuery, PostCandidate>>] {
        // 目的：返回内部持有的选后过滤组件。
        // 影响：框架据此对已选候选做安全/会话过滤。
        &self.post_selection_filters
    }

    // 目的：向框架暴露副作用组件（以 Arc 共享）。
    // 影响：框架在收尾阶段执行旁路操作。
    fn side_effects(&self) -> Arc<Vec<Box<dyn SideEffect<ScoredPostsQuery, PostCandidate>>>> {
        // 目的：克隆共享的副作用容器引用。
        // 影响：避免拷贝组件本身，降低开销且保持共享。
        Arc::clone(&self.side_effects)
    }

    // 目的：向框架声明最终结果规模。
    // 影响：框架据此约束最终输出候选数量。
    fn result_size(&self) -> usize {
        // 目的：返回配置的 RESULT_SIZE 常量。
        // 影响：决定响应中最终返回的帖子条数。
        params::RESULT_SIZE
    }
}
