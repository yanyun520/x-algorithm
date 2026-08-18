// 目的：引入候选结构 PostCandidate。
// 影响：本增强器产出并更新候选的作者/转发者展示信息。
use crate::candidate_pipeline::candidate::PostCandidate;
// 目的：引入查询对象 ScoredPostsQuery。
// 影响：接口对齐需要（当前未读取查询内容）。
use crate::candidate_pipeline::query::ScoredPostsQuery;
// 目的：引入 Gizmoduck 客户端 trait。
// 影响：为批量获取用户资料提供异步调用能力。
use crate::clients::gizmoduck_client::GizmoduckClient;
// 目的：引入 Arc 智能指针。
// 影响：共享 Gizmoduck 客户端，降低连接开销。
use std::sync::Arc;
// 目的：引入 tonic 异步 trait 支持。
// 影响：使 hydrate 方法可在异步运行时执行网络请求。
use tonic::async_trait;
// 目的：引入 Hydrator trait。
// 影响：本类型以标准增强器身份接入流水线框架。
use xai_candidate_pipeline::hydrator::Hydrator;

// 目的：定义用户资料增强器结构。
// 影响：为候选作者与转发者补充屏幕名与粉丝数。
pub struct GizmoduckCandidateHydrator {
    // 目的：持有 Gizmoduck 客户端引用。
    // 影响：发起用户资料批量查询的唯一通道。
    pub gizmoduck_client: Arc<dyn GizmoduckClient + Send + Sync>,
}

// 目的：为增强器实现构造函数。
// 影响：提供注入 Gizmoduck 客户端的实例化入口。
impl GizmoduckCandidateHydrator {
    // 目的：定义异步构造方法。
    // 影响：返回携带客户端的增强器实例。
    pub async fn new(gizmoduck_client: Arc<dyn GizmoduckClient + Send + Sync>) -> Self {
        // 目的：构造结构体实例。
        // 影响：客户端引用被保存供 hydrate 使用。
        Self { gizmoduck_client }
    }
}

// 目的：声明实现异步 Hydrator。
// 影响：流水线可在增强阶段调用。
#[async_trait]
impl Hydrator<ScoredPostsQuery, PostCandidate> for GizmoduckCandidateHydrator {
    // 目的：为该增强器挂接调用统计埋点。
    // 影响：监控用户资料增强的调用频率与时延。
    #[xai_stats_macro::receive_stats]
    // 目的：实现用户资料增强主逻辑。
    // 影响：为候选补充作者/转发者的屏幕名与粉丝数。
    async fn hydrate(
        // 目的：引用本增强器状态。
        // 影响：访问其中的 Gizmoduck 客户端。
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
        // 目的：取出 Gizmoduck 客户端引用。
        // 影响：便于后续批量调用。
        let client = &self.gizmoduck_client;

        // 目的：提取所有候选的作者 ID。
        // 影响：一次性收集需查询的作者集合。
        let author_ids: Vec<_> = candidates.iter().map(|c| c.author_id).collect();
        // 目的：将作者 ID 转为 i64。
        // 影响：匹配 Gizmoduck 客户端入参类型。
        let author_ids: Vec<_> = author_ids.iter().map(|&x| x as i64).collect();
        // 目的：提取所有候选的被转发用户 ID（可能为 None）。
        // 影响：收集转发表情涉及的作者集合。
        let retweet_user_ids: Vec<_> = candidates.iter().map(|c| c.retweeted_user_id).collect();
        // 目的：展平 Option，仅保留存在的转发者 ID。
        // 影响：剔除空值，缩小查询集合。
        let retweet_user_ids: Vec<_> = retweet_user_ids.iter().flatten().collect();
        // 目的：将转发者 ID 转为 i64。
        // 影响：匹配客户端入参类型。
        let retweet_user_ids: Vec<_> = retweet_user_ids.iter().map(|&&x| x as i64).collect();

        // 目的：预分配合并后的查询 ID 容器（容量=两者之和）。
        // 影响：避免重复扩容，容纳全部需查询用户。
        let mut user_ids_to_fetch = Vec::with_capacity(author_ids.len() + retweet_user_ids.len());
        // 目的：追加作者 ID。
        // 影响：保证作者被查询。
        user_ids_to_fetch.extend(author_ids);
        // 目的：追加转发者 ID。
        // 影响：保证转发者被查询。
        user_ids_to_fetch.extend(retweet_user_ids);
        // 目的：对集合去重。
        // 影响：同一用户仅查询一次，减少远程请求量。
        user_ids_to_fetch.dedup();

        // 目的：批量请求用户资料。
        // 影响：得到 user_id -> 用户结果映射。
        let users = client.get_users(user_ids_to_fetch).await;
        // 目的：转换远程调用错误为字符串。
        // 影响：调用失败时经 ? 提前返回错误。
        let users = users.map_err(|e| e.to_string())?;

        // 目的：预分配增强结果容器。
        // 影响：避免扩容开销，长度与输入一致。
        let mut hydrated_candidates = Vec::with_capacity(candidates.len());

        // 目的：遍历每个候选。
        // 影响：逐个候选提取对应的用户资料。
        for candidate in candidates {
            // 目的：按作者 ID 查取用户数据并解包引用。
            // 影响：得到作者的用户结果（可能缺失）。
            let user = users
                .get(&(candidate.author_id as i64))
                .and_then(|user| user.as_ref());
            // 目的：从用户结果中取计数信息。
            // 影响：获取作者粉丝数。
            let user_counts = user.and_then(|user| user.user.as_ref().map(|u| &u.counts));
            // 目的：从用户结果中取资料信息。
            // 影响：获取作者屏幕名。
            let user_profile = user.and_then(|user| user.user.as_ref().map(|u| &u.profile));

            // 目的：提取作者粉丝数并转为 i32。
            // 影响：写入候选供后续使用（展示/权重）。
            let author_followers_count: Option<i32> =
                user_counts.map(|x| x.followers_count).map(|x| x as i32);
            // 目的：克隆作者屏幕名。
            // 影响：供响应输出与组装。
            let author_screen_name: Option<String> = user_profile.map(|x| x.screen_name.clone());

            // 目的：若候选存在被转发者，则按转发者 ID 查取用户。
            // 影响：获取转发者的用户数据（可能缺失）。
            let retweet_user = candidate
                .retweeted_user_id
                .and_then(|retweeted_user_id| users.get(&(retweeted_user_id as i64)))
                .and_then(|user| user.as_ref());
            // 目的：提取转发者资料。
            // 影响：获取转发者屏幕名。
            let retweet_profile =
                retweet_user.and_then(|user| user.user.as_ref().map(|u| &u.profile));
            // 目的：克隆转发者屏幕名。
            // 影响：供响应输出与组装。
            let retweeted_screen_name: Option<String> =
                retweet_profile.map(|x| x.screen_name.clone());

            // 目的：构造增强后的候选副本。
            // 影响：携带作者与转发者展示信息。
            let hydrated = PostCandidate {
                // 目的：写入作者粉丝数。
                // 影响：候选具备作者影响力特征。
                author_followers_count,
                // 目的：写入作者屏幕名。
                // 影响：响应可直接展示作者名。
                author_screen_name,
                // 目的：写入转发者屏幕名。
                // 影响：响应可直接展示转发作者名。
                retweeted_screen_name,
                // 目的：其余字段保持默认。
                // 影响：由其它增强器负责填充。
                ..Default::default()
            };
            // 目的：追加到结果容器。
            // 影响：保持与输入候选顺序一致。
            hydrated_candidates.push(hydrated);
        }

        // 目的：返回增强结果。
        // 影响：流水线据此刷新原候选展示字段。
        Ok(hydrated_candidates)
    }

    // 目的：定义合并增强结果回候选的方法。
    // 影响：把展示信息字段同步到原候选。
    fn update(&self, candidate: &mut PostCandidate, hydrated: PostCandidate) {
        // 目的：同步作者粉丝数。
        // 影响：候选携带粉丝数特征。
        candidate.author_followers_count = hydrated.author_followers_count;
        // 目的：同步作者屏幕名。
        // 影响：候选携带作者名字。
        candidate.author_screen_name = hydrated.author_screen_name;
        // 目的：同步转发者屏幕名。
        // 影响：候选携带转发者名字。
        candidate.retweeted_screen_name = hydrated.retweeted_screen_name;
    }
}
