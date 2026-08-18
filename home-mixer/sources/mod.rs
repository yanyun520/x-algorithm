// 目的：声明并对外开放 Phoenix 召回源，通过 Phoenix 检索服务获取站外候选帖子。
// 影响：提供站外(OON)召回候选，是主要的内容入口之一。
pub mod phoenix_source;
// 目的：声明并对外开放 Thunder 召回源，通过 Thunder 服务获取用户关注对象的站内候选帖子。
// 影响：提供站内(In-Network)召回候选，保证好友动态即时性。
pub mod thunder_source;
