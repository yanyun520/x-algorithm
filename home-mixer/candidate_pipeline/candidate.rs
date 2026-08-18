// 目的：引入 HashMap 标准集合，用于构建用户ID到屏幕名的映射。
// 影响：get_screen_names 返回的映射即依赖该类型。
use std::collections::HashMap;
// 目的：引入本服务的 proto crate，用于引用 ServedType 枚举。
// 影响：候选结构可记录其内容服务来源类型。
use xai_home_mixer_proto as pb;
// 目的：引入可见性过滤模型（FilteredReason 等）。
// 影响：候选可携带安全过滤原因，供后置过滤与响应输出使用。
use xai_visibility_filtering::models as vf;

// 目的：为 PostCandidate 派生 Clone/Debug/Default，便于复制、调试与快捷初始化。
// 影响：流水线中各阶段可安全复制候选，用 Default 快速构造部分字段更新的新对象。
#[derive(Clone, Debug, Default)]
// 目的：定义贯穿流水线的候选帖子数据结构。
// 影响：所有阶段（召回/增强/过滤/打分/选择）都以该结构传递帖子信息。
pub struct PostCandidate {
    // 目的：记录帖子唯一 ID（雪花 ID）。
    // 影响：用于去重、查询特征、过滤与响应输出。
    pub tweet_id: i64,
    // 目的：记录帖子作者用户 ID。
    // 影响：用于站内判定、社交关系过滤与展示。
    pub author_id: u64,
    // 目的：记录帖子正文文本。
    // 影响：用于核心数据完整性校验与静音关键词匹配。
    pub tweet_text: String,
    // 目的：记录该帖回复的目标帖子 ID（可选）。
    // 影响：用于识别回复关系并构造会话祖先链。
    pub in_reply_to_tweet_id: Option<u64>,
    // 目的：记录被转发的原始帖子 ID（可选）。
    // 影响：用于转发去重、Phoenix 预测映射与响应输出。
    pub retweeted_tweet_id: Option<u64>,
    // 目的：记录被转发的用户 ID（可选）。
    // 影响：用于补充转发者屏幕名与模型特征。
    pub retweeted_user_id: Option<u64>,
    // 目的：记录 Phoenix 模型输出的各类行为分数。
    // 影响：作为加权评分器的输入，决定最终候选得分。
    pub phoenix_scores: PhoenixScores,
    // 目的：记录本次预测对应的预测请求 ID（可选）。
    // 影响：供日志与线上归因追踪单次模型打分。
    pub prediction_request_id: Option<u64>,
    // 目的：记录最近一次打分的毫秒时间戳（可选）。
    // 影响：输出到响应，标识内容打分时效。
    pub last_scored_at_ms: Option<u64>,
    // 目的：记录加权归一化后的候选分数（可选）。
    // 影响：作者多样性打分与最终排序的输入分数。
    pub weighted_score: Option<f64>,
    // 目的：记录最终排序分数（可选）。
    // 影响：选择器按该分数选取 TopK 候选。
    pub score: Option<f64>,
    // 目的：记录内容服务类型（如在站外/站内何种召回通道产出）。
    // 影响：输出到响应供客户端区分内容来源。
    pub served_type: Option<pb::ServedType>,
    // 目的：标记该帖是否属于站内（作者被关注或本人）。
    // 影响：决定 OON 加权与敏感过滤的安全级别分流。
    pub in_network: Option<bool>,
    // 目的：记录会话树的祖先帖子 ID 列表。
    // 影响：用于会话去重与响应中的会话关系展示。
    pub ancestors: Vec<u64>,
    // 目的：记录视频时长（毫秒，可选）。
    // 影响：决定视频帖是否可获得 VQV 权重加成。
    pub video_duration_ms: Option<i32>,
    // 目的：记录作者粉丝数（可选）。
    // 影响：可被用于权重/展示逻辑（当前未直接参与分数计算）。
    pub author_followers_count: Option<i32>,
    // 目的：记录作者屏幕名（可选）。
    // 影响：输出到响应供客户端直接展示作者名。
    pub author_screen_name: Option<String>,
    // 目的：记录被转发作者屏幕名（可选）。
    // 影响：输出到响应供客户端直接展示转发作者名。
    pub retweeted_screen_name: Option<String>,
    // 目的：记录可见性（安全）过滤原因（可选）。
    // 影响：后置 VFFilter 依据该字段决定候选去留并输出原因。
    pub visibility_reason: Option<vf::FilteredReason>,
    // 目的：记录帖子的订阅作者 ID（可选）。
    // 影响：订阅资格过滤器据此判断是否为付费订阅内容。
    pub subscription_author_id: Option<u64>,
}

// 目的：为 PhoenixScores 派生 Clone/Debug/Default。
// 影响：可整体复制、调试与初始化各行为分数默认值。
#[derive(Clone, Debug, Default)]
// 目的：定义 Phoenix 模型预测出的候选行为分数集合。
// 影响：加权评分器按权重累加这些分数生成候选总分。
pub struct PhoenixScores {
    // 目的：收藏（赞）行为概率。
    // 影响：参与加权总分计算，权重为 FAVORITE_WEIGHT。
    pub favorite_score: Option<f64>,
    // 目的：回复行为概率。
    // 影响：参与加权总分计算，权重为 REPLY_WEIGHT。
    pub reply_score: Option<f64>,
    // 目的：转发行为概率。
    // 影响：参与加权总分计算，权重为 RETWEET_WEIGHT。
    pub retweet_score: Option<f64>,
    // 目的：图片展开行为概率。
    // 影响：体现内容吸引用户展开图片的期望，参与加权。
    pub photo_expand_score: Option<f64>,
    // 目的：点击行为概率。
    // 影响：参与加权总分计算，权重为 CLICK_WEIGHT。
    pub click_score: Option<f64>,
    // 目的：点击作者主页行为概率。
    // 影响：体现关注转化潜力，参与加权。
    pub profile_click_score: Option<f64>,
    // 目的：视频有效观看（VQV）行为概率。
    // 影响：仅在此帖为符合时长条件的视频时参与加权。
    pub vqv_score: Option<f64>,
    // 目的：分享行为概率。
    // 影响：参与加权总分计算，权重为 SHARE_WEIGHT。
    pub share_score: Option<f64>,
    // 目的：通过私信分享行为概率。
    // 影响：参与加权总分计算，权重为 SHARE_VIA_DM_WEIGHT。
    pub share_via_dm_score: Option<f64>,
    // 目的：通过复制链接分享行为概率。
    // 影响：参与加权总分计算，权重为 SHARE_VIA_COPY_LINK_WEIGHT。
    pub share_via_copy_link_score: Option<f64>,
    // 目的：停留（长时间阅读）行为概率。
    // 影响：参与加权总分计算，权重为 DWELL_WEIGHT。
    pub dwell_score: Option<f64>,
    // 目的：引用（Quote）行为概率。
    // 影响：参与加权总分计算，权重为 QUOTE_WEIGHT。
    pub quote_score: Option<f64>,
    // 目的：点击被引用帖行为概率。
    // 影响：参与加权总分计算，权重为 QUOTED_CLICK_WEIGHT。
    pub quoted_click_score: Option<f64>,
    // 目的：关注作者行为概率。
    // 影响：体现内容/作者吸引力，参与加权。
    pub follow_author_score: Option<f64>,
    // 目的：不感兴趣行为概率。
    // 影响：作为负向信号参与加权，降低内容得分。
    pub not_interested_score: Option<f64>,
    // 目的：拉黑作者行为概率。
    // 影响：负向信号，参与加权降低内容得分。
    pub block_author_score: Option<f64>,
    // 目的：静音作者行为概率。
    // 影响：负向信号，参与加权降低内容得分。
    pub mute_author_score: Option<f64>,
    // 目的：举报行为概率。
    // 影响：强烈的负向信号，参与加权降低内容得分。
    pub report_score: Option<f64>,
    // Continuous actions
    // 目的：记录连续型行为——预计停留时长（秒）。
    // 影响：与 DwellTime 权重相乘计入总分，反映内容留人能力。
    pub dwell_time: Option<f64>,
}

// 目的：定义候选辅助能力 trait。
// 影响：为候选类型提供与展示相关的便捷方法接口。
pub trait CandidateHelpers {
    // 目的：声明获取候选关联用户（作者与转发者）屏幕名映射的方法。
    // 影响：返回 user_id -> 屏幕名，用于响应输出。
    fn get_screen_names(&self) -> HashMap<u64, String>;
}

// 目的：为 PostCandidate 实现 CandidateHelpers trait。
// 影响：使候选对象直接具备获取屏幕名映射的能力。
impl CandidateHelpers for PostCandidate {
    // 目的：实现屏幕名映射的获取逻辑。
    // 影响：收集作者与转发者两个维度的屏幕名。
    fn get_screen_names(&self) -> HashMap<u64, String> {
        // 目的：创建空的用户名映射表。
        // 影响：作为返回值累积容器。
        let mut screen_names = HashMap::<u64, String>::new();
        // 目的：若存在作者屏幕名，则按作者 ID 写入映射。
        // 影响：客户端可通过作者 ID 查到屏幕名。
        if let Some(author_screen_name) = self.author_screen_name.clone() {
            // 目的：将作者 ID -> 屏幕名插入映射。
            // 影响：完成作者名字的登记。
            screen_names.insert(self.author_id, author_screen_name);
        }
        // 目的：若同时存在转发者屏幕名与转发者 ID，则登记转发者名字。
        // 影响：让转发作者的屏幕名也可被响应使用。
        if let (Some(retweeted_screen_name), Some(retweeted_user_id)) =
            (self.retweeted_screen_name.clone(), self.retweeted_user_id)
        {
            // 目的：将转发者 ID -> 屏幕名插入映射。
            // 影响：完成转发者名字的登记，并覆盖同 ID 冲突情况。
            screen_names.insert(retweeted_user_id, retweeted_screen_name);
        }
        // 目的：返回累积的屏幕名映射。
        // 影响：调用方得到最终名字集合用于响应。
        screen_names
    }
}
