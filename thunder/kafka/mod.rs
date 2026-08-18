// =============================================================================
// kafka/mod.rs — Kafka 子模块的入口声明
// 职责：声明 kafka 目录下的三个子模块，使其对外可见
// 模块说明：
//   - tweet_events_listener: v1 版本，消费 Thrift 格式的 tweet 事件，
//     转换为 protobuf 后写入 in-network topic（非 serving 模式使用）
//   - tweet_events_listener_v2: v2 版本，直接消费 protobuf 格式的
//     in-network 事件并写入 PostStore（serving 模式使用）
//   - utils: 通用工具函数，包括 Kafka consumer 创建和消息批量反序列化
// 边界情况说明：
//   - 若任一子模块文件缺失，编译期报错
//   - pub mod 表示这些模块对 crate 外部可见（如 main.rs 可直接引用）
// =============================================================================

// v1 tweet 事件监听器：消费 Thrift 事件 + 生产 protobuf 事件
pub mod tweet_events_listener;
// v2 tweet 事件监听器：消费 protobuf 事件并直接写入 PostStore
pub mod tweet_events_listener_v2;
// Kafka 通用工具：consumer 创建、消息批量反序列化
pub mod utils;
