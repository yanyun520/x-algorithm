// 目的：引入 serde 的序列化/反序列化 trait。
// 影响：使 UserFeatures 可与 Thrift/JSON 等格式互转，供 Strato 存储读取。
use serde::{Deserialize, Serialize};

// 目的：为 UserFeatures 派生 Debug/Clone/序列化/反序列化/比较/默认实现。
// 影响：支持复制、调试、编解码与默认初始化，便于跨服务传输。
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
// 目的：设置 serde 字段名映射为 camelCase（如 mutedKeywords）。
// 影响：与下游存储（Strato）中的字段命名保持一致，保证正确解码。
#[serde(rename_all = "camelCase")]
// 目的：定义用户特征数据结构。
// 影响：为过滤与召回提供用户维度上下文。
pub struct UserFeatures {
    // 目的：记录用户设置的静音关键词列表。
    // 影响：驱动静音关键词过滤器，剔除命中关键词的帖子。
    pub muted_keywords: Vec<String>,
    // 目的：记录用户拉黑的用户 ID 列表。
    // 影响：社交关系过滤器据此剔除拉黑作者的内容。
    pub blocked_user_ids: Vec<i64>,
    // 目的：记录用户静音的用户 ID 列表。
    // 影响：社交关系过滤器据此剔除静音作者的内容。
    pub muted_user_ids: Vec<i64>,
    // 目的：记录用户关注的用户 ID 列表。
    // 影响：用于站内判定与 Thunder 站内召回。
    pub followed_user_ids: Vec<i64>,
    // 目的：记录用户订阅的创作者用户 ID 列表。
    // 影响：订阅资格过滤器据此判断付费内容可看性。
    pub subscribed_user_ids: Vec<i64>,
}