// 目的：声明并对外开放用户动作序列增强模块，负责拉取并聚合用户近期行为序列。
// 影响：为 Phoenix 召回与打分模型提供用户行为特征。
pub mod user_action_seq_query_hydrator;
// 目的：声明并对外开放用户特征增强模块，负责从 Strato 读取静音关键词、拉黑/关注/订阅等特征。
// 影响：为过滤器与源提供用户维度上下文，决定后续过滤与召回行为。
pub mod user_features_query_hydrator;
