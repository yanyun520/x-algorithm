// 目的：声明并编译候选数据增强(hydration)子模块，负责在候选进入打分前补充帖子/作者等属性。
// 影响：使本 crate 内可以通过 `crate::candidate_hydrators::*` 引用该模块中的各类 Hydrator。
mod candidate_hydrators;
// 目的：声明候选流水线子模块，定义候选(PostCandidate)、查询(ScoredPostsQuery)以及 Phoenix 候选流水线的组装逻辑。
// 影响：将候选生成、筛选、打分、选取等步骤的部件集中在一个模块中，便于统一组装和复用。
mod candidate_pipeline;
// 目的：声明 clients 子模块，并提供为公开可见的模块，内部封装 Thrift/gRPC 客户端（Gizmoduck、Strato、TES、Phoenix、Thunder 等）。
// 影响：供应上层组件获得远程服务访问能力；因含敏感凭证/内部接口，故从开源发布中排除。
pub mod clients; // Excluded from open source release for security reasons
// 目的：声明候选筛选子模块，集中定义从候选集中剔除不合规帖子的各类 Filter。
// 影响：让流水线能按顺序执行去重、年龄、拉黑/静音、订阅、沉没/已服务、敏感内容等过滤。
mod filters;
// 目的：声明参数模块，集中存放端口、权重、阈值、结果条数等全局配置常量，并对外开放。
// 影响：使配置文件与业务代码解耦，便于调整线上参数而无需改动逻辑代码；因包含内部配置而从开源发布中排除。
pub mod params; // Excluded from open source release for security reasons
// 目的：声明查询增强子模块，负责在候选检索前补充用户动作序列与用户特征到查询对象中。
// 影响：为召回和打分阶段提供必要的用户侧上下文特征。
mod query_hydrators;
// 目的：声明打分器子模块并对外开放，集中封装最终给候选排序用的各类 Scorer。
// 影响：公开导出后可由外部（如测试工具）直接引用这些打分器实现。
pub mod scorers;
// 目的：声明候选选择子模块，负责从打分后的候选中挑选最终结果（如 TopK）。
// 影响：决定响应中最终返回给用户的帖子数量与集合。
mod selectors;
// 目的：声明 HTTP/gRPC 服务器子模块，封装 HomeMixerServer 的 gRPC 服务实现。
// 影响：对外暴露 get_scored_posts 接口，处理请求并返回打分后的帖子列表。
mod server;
// 目的：声明副作用子模块，用于在流水线完成后执行与返回结果无关的旁路操作（如缓存请求信息）。
// 影响：在响应返回到用户前/后异步落盘缓存，提升下一次请求的体验与命中率。
mod side_effects;
// 目的：声明候选召回源子模块，定义从 Phoenix(OTF 重排/召回)与 Thunder(好友动态) 拉取候选帖子的 Source。
// 影响：决定候选帖子最初来自哪些召回通道，是整条流水线的数据入口。
mod sources;
// 目的：声明工具模块，集中放置请求 ID 生成、布隆过滤器、雪花 ID、分数归一化等通用函数。
// 影响：为流水线各环节提供公共工具能力；因内部实现细节而从开源发布中排除。
pub mod util; // Excluded from open source release for security reasons

// 目的：重新导出 HomeMixerServer，使外部依赖方通过 `xai_home_mixer::HomeMixerServer` 即可直接使用服务实现。
// 影响：简化代码库对外 API 入口，避免外部引用 `server` 模块内部路径。
pub use server::HomeMixerServer;
