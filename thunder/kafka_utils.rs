// =============================================================================
// kafka_utils.rs — Kafka 启动与配置的顶层入口
// 职责：
//   1. 读取 SASL 密码（环境变量优先，命令行参数兜底）
//   2. 根据 is_serving 模式构建不同的 KafkaConsumerConfig
//   3. serving 模式：启动 v2 消费器（直接将事件灌入 PostStore）
//   4. 非 serving 模式：启动 v1 消费器 + 生产器（从 Thrift topic 消费，
//      转换为 protobuf 后写入 in-network topic）
// 边界情况说明：
//   - SASL 密码缺失时 start_kafka 返回 Err（? 传播 None 的 Option）
//   - serving 模式下 group_id 附加 UUID 后缀，确保每次重启从最新 offset 开始
//   - 非 serving 模式下 enable_auto_commit=false，由代码手动提交 offset
//   - max_partition_fetch_bytes：serving 模式 100MB，非 serving 模式 10MB
// =============================================================================

use anyhow::{Context, Result};
use std::sync::Arc;
// KafkaProducerConfig / KafkaConsumerConfig / KafkaConfig / SslConfig 来自 xai_kafka 库
use xai_kafka::KafkaProducerConfig;
use xai_kafka::config::{KafkaConfig, KafkaConsumerConfig, SslConfig};
// WilyConfig 是 X 内部的 Kafka 客户端监控配置
use xai_wily::WilyConfig;

use crate::{
    args,
    kafka::{
        tweet_events_listener::start_tweet_event_processing,
        tweet_events_listener_v2::start_tweet_event_processing_v2,
    },
};

// Tweet 事件 Thrift topic 名称（空字符串表示使用默认/占位值）
// 边界：实际部署时需替换为真实 topic 名称
const TWEET_EVENT_TOPIC: &str = "";
// Tweet 事件 Thrift Kafka 集群地址（空字符串同上）
const TWEET_EVENT_DEST: &str = "";

// InNetwork 事件 protobuf topic 的目标地址和名称
const IN_NETWORK_EVENTS_DEST: &str = "";
const IN_NETWORK_EVENTS_TOPIC: &str = "";

/// 启动 Kafka 消费/生产任务
/// 参数：
///   - args: 命令行参数
///   - post_store: 共享的帖子存储（serving 模式下 v2 消费器直接写入）
///   - user: consumer group id 的后缀标识（非 serving 模式使用）
///   - tx: catchup 信号发送器（serving 模式下每个线程追赶完毕后发送）
/// 返回：Result<()>，SASL 密码缺失时返回 Err
pub async fn start_kafka(
    args: &args::Args,
    post_store: Arc<crate::posts::post_store::PostStore>,
    user: &str,
    tx: tokio::sync::mpsc::Sender<i64>,
) -> Result<()> {
    // ---- 读取 SASL 密码 ----
    // 优先从环境变量读取，若不存在则使用命令行参数
    // 边界：若两者都为 None，? 操作符将 None 转为 Err（anyhow error）
    let sasl_password = std::env::var("")
        .ok()
        .or(args.sasl_password.clone())?;

    // 生产者 SASL 密码：同样环境变量优先，命令行兜底
    // 边界：允许为 None（某些集群不需要密码）
    let producer_sasl_password = std::env::var("")
        .ok()
        .or(args.producer_sasl_password.clone());

    // ---- serving 模式：启动 v2 消费器 ----
    if args.is_serving {
        // 生成唯一 UUID 作为 consumer group id 后缀
        // 边界：每次重启使用不同 UUID，配合 auto_offset_reset 实现从最新位置开始消费
        let unique_id = uuid::Uuid::new_v4().to_string();

        // 构建 v2 Kafka 消费器配置
        let v2_tweet_events_consumer_config = KafkaConsumerConfig {
            base_config: KafkaConfig {
                // 使用 in_network_events 的消费目标地址
                dest: args.in_network_events_consumer_dest.clone(),
                // 消费 in-network protobuf topic
                topic: IN_NETWORK_EVENTS_TOPIC.to_string(),
                // 启用 Wily 监控
                wily_config: Some(WilyConfig::default()),
                // SSL/SASL 配置：使用生产者凭据（因为 in-network topic 由生产者写入）
                ssl: Some(SslConfig {
                    security_protocol: args.security_protocol.clone(),
                    sasl_mechanism: Some(args.producer_sasl_mechanism.clone()),
                    sasl_username: Some(args.producer_sasl_username.clone()),
                    sasl_password: producer_sasl_password.clone(),
                }),
                // 其余字段使用默认值
                ..Default::default()
            },
            // group_id = 基础 group_id + UUID 后缀
            group_id: format!("{}-{}", args.kafka_group_id, unique_id),
            // auto_offset_reset：新 group 从最新还是最早开始消费
            auto_offset_reset: args.auto_offset_reset.clone(),
            fetch_timeout_ms: args.fetch_timeout_ms,
            // 单分区最大拉取字节数：100MB
            // 边界：若消息超过此大小会被丢弃
            max_partition_fetch_bytes: Some(1024 * 1024 * 100),
            // skip_to_latest：是否跳到最新 offset
            skip_to_latest: args.skip_to_latest,
            ..Default::default()
        };

        // 启动 v2 tweet 事件处理（直接写入 PostStore）
        // 边界：此函数内部 spawn 多个异步任务，不阻塞当前调用
        start_tweet_event_processing_v2(
            v2_tweet_events_consumer_config,
            Arc::clone(&post_store),
            args,
            tx,
        )
        .await;
    }

    // ---- 非 serving 模式：启动 v1 消费器 + 生产器 ----
    // 此模式从 Thrift topic 消费原始 tweet 事件，转换为 protobuf 后写入 in-network topic
    if !args.is_serving {
        // 构建 v1 Kafka 消费器配置
        let tweet_events_consumer_config = KafkaConsumerConfig {
            base_config: KafkaConfig {
                dest: TWEET_EVENT_DEST.to_string(),
                topic: TWEET_EVENT_TOPIC.to_string(),
                wily_config: Some(WilyConfig::default()),
                // 使用消费者凭据
                ssl: Some(SslConfig {
                    security_protocol: args.security_protocol.clone(),
                    sasl_mechanism: Some(args.sasl_mechanism.clone()),
                    sasl_username: Some(args.sasl_username.clone()),
                    sasl_password: Some(sasl_password.clone()),
                }),
                ..Default::default()
            },
            // group_id = 基础 group_id + user 后缀
            group_id: format!("{}-{}", args.kafka_group_id, user),
            auto_offset_reset: args.auto_offset_reset.clone(),
            // 关闭自动提交，由代码手动 commit offset
            // 边界：手动提交确保消息处理成功后才推进 offset，避免丢消息
            enable_auto_commit: false,
            fetch_timeout_ms: args.fetch_timeout_ms,
            // 单分区最大拉取字节数：10MB（比 serving 模式小）
            max_partition_fetch_bytes: Some(1024 * 1024 * 10),
            // partitions=None 表示消费所有分区
            partitions: None,
            skip_to_latest: args.skip_to_latest,
            ..Default::default()
        };

        // 构建 Kafka 生产者配置（用于写入 in-network topic）
        let producer_config = KafkaProducerConfig {
            base_config: KafkaConfig {
                dest: IN_NETWORK_EVENTS_DEST.to_string(),
                topic: IN_NETWORK_EVENTS_TOPIC.to_string(),
                wily_config: Some(WilyConfig::default()),
                ssl: Some(SslConfig {
                    security_protocol: args.security_protocol.clone(),
                    sasl_mechanism: Some(args.producer_sasl_mechanism.clone()),
                    sasl_username: Some(args.producer_sasl_username.clone()),
                    sasl_password: producer_sasl_password.clone(),
                }),
                ..Default::default()
            },
            ..Default::default()
        };

        // 启动 v1 tweet 事件处理（消费 Thrift + 生产 protobuf）
        start_tweet_event_processing(tweet_events_consumer_config, producer_config, args).await;
    }

    Ok(())
}
