// 目的：声明并对外开放候选结构模块（PostCandidate、PhoenixScores 及其工具方法）。
// 影响：使流水线各组件与服务器层能统一操作候选对象。
pub mod candidate;
// 目的：声明并对外开放候选属性模块（CoreData、媒体实体、用户信息等反序列化数据结构）。
// 影响：供各 Hydrator 解析远程服务返回数据，把外部数据映射为候选字段。
pub mod candidate_features;
// 目的：声明并对外开放 Phoenix 候选流水线模块，负责按阶段组装 Hydrator/Filter/Scorer/Selector 等。
// 影响：作为本服务候选处理流程的核心编排器，被服务器直接调用执行。
pub mod phoenix_candidate_pipeline;
// 目的：声明并对外开放查询对象模块（ScoredPostsQuery 及其 gRPC 上下文适配）。
// 影响：贯穿整个流水线的请求上下文载体，同时可转换为 TwitterContextViewer 供 VF 使用。
pub mod query;
// 目的：声明并对外开放查询特征模块（UserFeatures：静音关键词、拉黑/静音/关注/订阅用户）。
// 影响：为过滤器与源提供用户维度特征，决定过滤与召回行为。
pub mod query_features;
