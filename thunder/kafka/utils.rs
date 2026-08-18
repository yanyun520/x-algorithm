// =============================================================================
// kafka/utils.rs — Kafka 通用工具函数
// 职责：
//   1. create_kafka_consumer：创建并启动一个 Kafka 消费者，返回线程安全的句柄
//   2. deserialize_kafka_messages：批量反序列化 Kafka 消息，跳过解析失败的消息
// 边界情况说明：
//   - consumer.start() 失败时返回带上下文的 Err
//   - 消息 payload 为 None（如 tombstone 消息）时跳过
//   - 单条消息反序列化失败时记录错误并继续处理其余消息（不中断批次）
// =============================================================================

use anyhow::{Context, Result};
use std::sync::Arc;
use tokio::sync::RwLock;
use xai_kafka::{KafkaMessage, config::KafkaConsumerConfig, consumer::KafkaConsumer};

use crate::metrics;

/// 创建并启动一个 Kafka 消费者
/// 参数：config — Kafka 消费者配置（topic、group_id、SSL 等）
/// 返回：Result<Arc<RwLock<KafkaConsumer>>>，线程安全的消费者句柄
/// 边界：
///   - 配置无效（如 topic 不存在、认证失败）时 start() 返回 Err
///   - RwLock 允许并发读（如查询 lag）和独占写（如 poll、commit）
pub async fn create_kafka_consumer(
    config: KafkaConsumerConfig,
) -> Result<Arc<RwLock<KafkaConsumer>>> {
    // 创建消费者实例（此时尚未连接 Kafka）
    let mut consumer = KafkaConsumer::new(config);
    // 启动消费者：建立连接、订阅 topic、加入 consumer group
    // 失败时附加上下文信息
    consumer
        .start()
        .await
        .context("Failed to start Kafka consumer")?;

    // 包装为 Arc<RwLock> 以便在多个异步任务间共享
    Ok(Arc::new(RwLock::new(consumer)))
}

/// 批量反序列化 Kafka 消息
/// 泛型参数：
///   - T: 反序列化后的目标类型
///   - F: 反序列化函数（接收字节切片，返回 Result<T>）
/// 参数：
///   - messages: Kafka 消息向量
///   - deserializer: 反序列化函数（如 deserialize_tweet_event / deserialize_tweet_event_v2）
/// 返回：Result<Vec<T>>，仅包含成功反序列化的消息
/// 边界情况：
///   - 空消息列表：返回空 Vec
///   - payload 为 None（tombstone 删除标记）：跳过该消息
///   - 单条消息解析失败：记录 error 日志并递增 KAFKA_MESSAGES_FAILED_PARSE 指标，
///     继续处理后续消息——保证单条坏消息不会拖垮整个批次
pub fn deserialize_kafka_messages<T, F>(
    messages: Vec<KafkaMessage>,
    deserializer: F,
) -> Result<Vec<T>>
where
    F: Fn(&[u8]) -> Result<T>,
{
    // 记录批量处理耗时指标（RAII：函数结束时自动上报）
    let _timer = metrics::Timer::new(metrics::BATCH_PROCESSING_TIME.clone());

    // 预分配容量，避免多次扩容
    let mut kafka_data = Vec::with_capacity(messages.len());

    // 遍历每条消息
    for msg in messages.iter() {
        // 仅处理有 payload 的消息
        // 边界：payload 为 None 的消息（如 tombstone）被跳过
        if let Some(payload) = &msg.payload {
            // 调用反序列化函数
            match deserializer(payload) {
                Ok(deserialized_msg) => {
                    // 成功：加入结果向量
                    kafka_data.push(deserialized_msg);
                }
                Err(e) => {
                    // 失败：记录错误并递增指标，继续处理下一条
                    // 边界：此处不返回 Err，避免单条坏消息中断整个批次
                    log::error!("Failed to parse Kafka message: {}", e);
                    metrics::KAFKA_MESSAGES_FAILED_PARSE.inc();
                }
            }
        }
    }

    Ok(kafka_data)
}
