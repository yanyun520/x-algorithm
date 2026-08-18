// 目的：引入 serde 序列化/反序列化 trait。
// 影响：使下列各数据结构可在不同服务间以字节形式传输与解析。
use serde::{Deserialize, Serialize};

// 目的：为 PureCoreData 派生通用实现（调试、克隆、编解码、比较、默认值）。
// 影响：结构可直接被 Thrift 解码器生成并用于候选字段填充。
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
// 目的：将序列化字段名映射为 camelCase（如 sourceTweetId）。
// 影响：与外部服务返回的 JSON/Thrift 字段命名对齐，保证正确反序列化。
#[serde(rename_all = "camelCase")]
// 目的：定义帖子核心数据的轻量结构。
// 影响：作为 TES 返回的核心数据载体，用于填充候选基础字段。
pub struct PureCoreData {
    // 目的：帖子作者 ID。
    // 影响：映射为候选的 author_id。
    pub author_id: u64,
    // 目的：帖子正文文本。
    // 影响：映射为候选的 tweet_text，供校验与关键词过滤。
    pub text: String,
    // 目的：源帖 ID（转发时指向被转发帖）。
    // 影响：映射为候选的 retweeted_tweet_id。
    pub source_tweet_id: Option<u64>,
    // 目的：源用户 ID（转发时指向被转发作者）。
    // 影响：映射为候选的 retweeted_user_id。
    pub source_user_id: Option<u64>,
    // 目的：回复目标帖 ID。
    // 影响：映射为候选的 in_reply_to_tweet_id。
    pub in_reply_to_tweet_id: Option<u64>,
    // 目的：回复目标用户 ID。
    // 影响：可为会话构建提供线索（当前未直接使用）。
    pub in_reply_to_user_id: Option<u64>,
}

// 目的：为 ExclusiveTweetControl 派生通用实现。
// 影响：支持解析订阅/私密会话等受控推文控制字段。
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
// 目的：字段名映射为 camelCase。
// 影响：与外部数据格式对齐。
#[serde(rename_all = "camelCase")]
// 目的：定义私密/受控帖文的会话作者控制信息。
// 影响：为订阅内容判定提供会话维度的作者信息（当前保留备用）。
pub struct ExclusiveTweetControl {
    // 目的：会话所属作者 ID。
    // 影响：可据此判断私密会话的归属者（当前未参与过滤）。
    pub conversation_author_id: i64,
}

// 目的：定义媒体实体集合的类型别名。
// 影响：简化后续类型签名，表示帖子包含的多媒体实体列表。
pub type MediaEntities = Vec<MediaEntity>;

// 目的：为 MediaEntity 派生通用实现。
// 影响：支持媒体实体列表的克隆/编解码。
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
// 目的：字段名映射为 camelCase。
// 影响：与外部媒体数据格式对齐。
#[serde(rename_all = "camelCase")]
// 目的：定义单条媒体实体。
// 影响：通过 media_info 承载视频等具体媒体信息。
pub struct MediaEntity {
    // 目的：媒体详细信息（可选）。
    // 影响：为视频信息提取提供入口。
    pub media_info: Option<MediaInfo>,
}

// 目的：为 MediaInfo 枚举派生通用实现（不含 Default，因枚举无默认值）。
// 影响：支持按具体媒体类型匹配。
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
// 目的：枚举序列化字段名映射为 camelCase。
// 影响：与外部媒体数据结构对齐。
#[serde(rename_all = "camelCase")]
// 目的：定义媒体信息类型的判别枚举。
// 影响：区分视频等不同媒体类型，便于按类型提取特征。
pub enum MediaInfo {
    // 目的：标记该媒体为视频信息。
    // 影响：匹配到时即可读取视频时长。
    VideoInfo(VideoInfo),
}

// 目的：为 VideoInfo 派生通用实现。
// 影响：支持视频信息的克隆与编解码。
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
// 目的：字段名映射为 camelCase。
// 影响：与外部视频数据格式对齐。
#[serde(rename_all = "camelCase")]
// 目的：定义视频信息结构。
// 影响：承载视频时长，用于 VQV 权重资格判定。
pub struct VideoInfo {
    // 目的：视频时长（毫秒）。
    // 影响：大于阈值时让候选获得 VQV 权重加成。
    pub duration_millis: i32,
}

// 目的：为 Share 派生通用实现。
// 影响：支持分享来源信息的解析。
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
// 目的：字段名映射为 camelCase。
// 影响：与外部数据格式对齐。
#[serde(rename_all = "camelCase")]
// 目的：定义分享来源信息。
// 影响：记录某帖经由分享形态出现的源帖与源用户（当前保留备用）。
pub struct Share {
    // 目的：源帖 ID。
    // 影响：标识分享内容原始出处。
    pub source_tweet_id: u64,
    // 目的：源用户 ID。
    // 影响：标识分享内容原始作者。
    pub source_user_id: u64,
}

// 目的：为 Reply 派生通用实现。
// 影响：支持回复关系信息的解析。
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
// 目的：字段名映射为 camelCase。
// 影响：与外部数据格式对齐。
#[serde(rename_all = "camelCase")]
// 目的：定义回复关系信息。
// 影响：记录回复目标，用于会话祖先链的构建。
pub struct Reply {
    // 目的：被回复的帖子 ID。
    // 影响：作为会话祖先之一写入候选。
    pub in_reply_to_tweet_id: Option<u64>,
    // 目的：被回复的用户 ID。
    // 影响：标识回复对象（当前未直接使用）。
    pub in_reply_to_user_id: u64,
}

// 目的：为 GizmoduckUserCounts 派生通用实现。
// 影响：支持用户计数数据的解析。
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
// 目的：字段名映射为 camelCase。
// 影响：与 Gizmoduck 返回格式对齐。
#[serde(rename_all = "camelCase")]
// 目的：定义用户计数信息。
// 影响：承载粉丝数等公开计数。
pub struct GizmoduckUserCounts {
    // 目的：粉丝数。
    // 影响：映射为候选的 author_followers_count。
    pub followers_count: u32,
}

// 目的：为 GizmoduckUserProfile 派生通用实现。
// 影响：支持用户资料信息的解析。
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
// 目的：字段名映射为 camelCase。
// 影响：与 Gizmoduck 返回格式对齐。
#[serde(rename_all = "camelCase")]
// 目的：定义用户资料概要。
// 影响：承载屏幕名等展示信息。
pub struct GizmoduckUserProfile {
    // 目的：用户屏幕名。
    // 影响：映射为候选的 author_screen_name / retweeted_screen_name。
    pub screen_name: String,
}

// 目的：为 GizmoduckUser 派生通用实现。
// 影响：支持用户整体信息的解析。
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
// 目的：字段名映射为 camelCase。
// 影响：与 Gizmoduck 返回格式对齐。
#[serde(rename_all = "camelCase")]
// 目的：定义用户综合信息（ID + 资料 + 计数）。
// 影响：作为用户查询结果的主要内容单元。
pub struct GizmoduckUser {
    // 目的：用户 ID。
    // 影响：标识该用户数据对应的用户。
    pub user_id: u64,
    // 目的：用户资料。
    // 影响：提供屏幕名等展示信息。
    pub profile: GizmoduckUserProfile,
    // 目的：用户计数。
    // 影响：提供粉丝数等数据。
    pub counts: GizmoduckUserCounts,
}

// 目的：为 GizmoduckUserResult 派生通用实现。
// 影响：支持单用户查询结果的可选包装解析。
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
// 目的：字段名映射为 camelCase。
// 影响：与 Gizmoduck 返回格式对齐。
#[serde(rename_all = "camelCase")]
// 目的：定义单用户查询结果的包装结构。
// 影响：用户不存在时可安全地以 None 表达，避免下游误用。
pub struct GizmoduckUserResult {
    // 目的：用户数据（可选）。
    // 影响：存在时取用资料与计数，缺失时按无数据处理。
    pub user: Option<GizmoduckUser>,
}
