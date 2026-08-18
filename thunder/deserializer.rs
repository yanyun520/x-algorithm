// =============================================================================
// deserializer.rs — Kafka 消息反序列化模块
// 职责：将 Kafka 中收到的字节流（payload）还原为 Rust 结构体
// 支持三种格式：
//   1. Thrift binary → TweetEvent（v1 旧格式）
//   2. Thrift binary → Event（通用 Thrift 事件）
//   3. Protobuf → InNetworkEvent（v2 新格式）
// 边界情况说明：
//   - payload 为空或格式损坏时，反序列化返回 Err，由调用方决定跳过或告警
//   - Thrift 的 TBinaryInputProtocol 要求严格的字段顺序与类型匹配
//   - protobuf 的 decode 对未知字段有前向兼容性（跳过未知 tag）
// =============================================================================

// 从 schema 模块导入 Thrift 定义的事件类型
use crate::schema::{events::Event, tweet_events::TweetEvent};
// anyhow::Context 为 Result 添加错误上下文
use anyhow::{Context, Result};
// prost::Message 提供 protobuf 的 decode / encode 方法
use prost::Message;
// thrift::protocol 提供 Thrift 二进制协议的读写能力
use thrift::protocol::{TBinaryInputProtocol, TSerializable};
// InNetworkEvent 是 protobuf 定义的新版网络内事件类型
use xai_thunder_proto::InNetworkEvent;

/// 将 Thrift 二进制消息反序列化为 TweetEvent
/// 参数：payload — Kafka 消息的原始字节切片
/// 返回：Result<TweetEvent>，成功返回事件结构体，失败返回带上下文的错误
/// 边界：
///   - 空字节流 → Thrift 读取失败，返回 Err("Failed to deserialize TweetEvent")
///   - 字节流截断 → 同上
///   - 字段类型不匹配 → Thrift 抛出异常，被 .context() 包装
pub fn deserialize_tweet_event(payload: &[u8]) -> Result<TweetEvent> {
    // 将字节切片包装为 Cursor，实现 Read trait 以供 Thrift 协议读取
    let mut cursor = std::io::Cursor::new(payload);
    // 创建 Thrift 二进制输入协议，第二个参数 true 表示严格读取（含协议版本校验）
    // 边界：strict=true 时若消息头不含正确的 Thrift 版本号会报错
    let mut protocol = TBinaryInputProtocol::new(&mut cursor, true);

    // 从协议读取 TweetEvent 结构体，失败时附加上下文信息
    TweetEvent::read_from_in_protocol(&mut protocol).context("Failed to deserialize TweetEvent")
}

/// 将 Thrift 二进制消息反序列化为 Event（通用事件类型）
/// 与 deserialize_tweet_event 类似，但目标类型为 Event
pub fn deserialize_event(payload: &[u8]) -> Result<Event> {
    let mut cursor = std::io::Cursor::new(payload);
    let mut protocol = TBinaryInputProtocol::new(&mut cursor, true);

    Event::read_from_in_protocol(&mut protocol).context("Failed to deserialize Event")
}

/// 将 protobuf 二进制消息反序列化为 InNetworkEvent（v2 新格式）
/// 参数：payload — Kafka 消息的原始字节切片
/// 返回：Result<InNetworkEvent>
/// 边界：
///   - 空字节流 → decode 返回 Err（protobuf 要求至少有消息头）
///   - 未知字段 → 自动跳过，不影响反序列化（前向兼容）
///   - 必填字段缺失 → decode 成功但字段为 None，由调用方处理
pub fn deserialize_tweet_event_v2(payload: &[u8]) -> Result<InNetworkEvent> {
    // prost 的 decode 直接从字节切片解析 protobuf 消息
    InNetworkEvent::decode(payload).context("Failed to deserialize InNetworkEvent")
}
