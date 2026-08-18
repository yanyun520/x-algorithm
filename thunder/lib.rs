// =============================================================================
// lib.rs — thunder 服务的库入口文件
// 作用：将 thunder 服务内部的所有子模块统一暴露给二进制入口（main.rs）以及
//       外部依赖（如 home-mixer）。Rust 中 lib.rs 相当于该 crate 的"根模块"，
//       此处仅做模块声明（pub mod），不包含具体业务逻辑。
// 边界情况说明：
//   - 若某个被声明的模块对应的 .rs 文件缺失，编译期会报 E0432（unresolved module）。
//   - 模块声明的顺序不影响编译结果，但按功能分组便于维护。
//   - pub mod 表示对外可见；若改为 mod（非 pub），则外部 crate 无法引用该模块。
// =============================================================================

// 命令行参数解析模块：定义 clap 的 Args 结构体，承载所有启动参数
pub mod args;

// 全局常量配置模块：集中管理如 MAX_INPUT_LIST_SIZE、MAX_POSTS_TO_RETURN 等阈值
pub mod config;

// 反序列化模块：负责将 Kafka 消息的 Thrift / protobuf 字节流还原为结构体
pub mod deserializer;

// Kafka 消费与生产子模块集合（含 v1 与 v2 两个版本的 tweet 事件监听器）
pub mod kafka;

// Kafka 辅助函数模块：封装 consumer 创建、消息批量反序列化等通用逻辑
pub mod kafka_utils;

// Prometheus 指标模块：定义计数器、直方图、计时器等可观测性埋点
pub mod metrics;

// Twitter/X 内部可观测性平台 o2 的集成模块
pub mod o2;

// 帖子存储子模块集合：核心内存数据结构 PostStore 所在位置
pub mod posts;

// Thrift schema 定义模块：由 thrift IDL 自动生成，描述 tweet 事件的数据结构
pub mod schema;

// Strato 客户端模块：Strato 是 X 内部的社会关系图存储服务，用于查询用户关注列表
pub mod strato_client;

// gRPC 服务实现模块：ThunderServiceImpl 提供 GetInNetworkPosts RPC 接口
pub mod thunder_service;
