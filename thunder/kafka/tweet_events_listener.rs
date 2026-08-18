// =============================================================================
// kafka/tweet_events_listener.rs — v1 版 tweet 事件监听器（非 serving 模式使用）
// 职责：
//   1. 从 Thrift 格式的 tweet 事件 topic 消费消息
//   2. 反序列化为 TweetEvent，提取创建/删除事件
//   3. 转换为 protobuf 格式的 InNetworkEvent 并写入 in-network topic
//   4. 监控各分区 lag 并上报 Prometheus 指标
// 边界情况说明：
//   - 生产者启动失败时直接 panic（v1 模式是 feeder，必须能生产）
//   - 消费线程异常退出时 panic，保证故障快速暴露
//   - 单条消息解析失败不影响批次处理（见 utils::deserialize_kafka_messages）
//   - 删除事件若帖子已超过保留期则跳过（避免无意义删除）
// =============================================================================

use anyhow::{Context, Result};
use log::{error, info, warn};
use prost::Message;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;
use tokio::sync::RwLock;
use xai_kafka::{KafkaMessage, config::KafkaConsumerConfig, consumer::KafkaConsumer};
use xai_kafka::{KafkaProducer, KafkaProducerConfig};
use xai_thunder_proto::{
    InNetworkEvent, LightPost, TweetCreateEvent, TweetDeleteEvent, in_network_event,
};

use crate::{
    args::Args,
    crate::config::MIN_VIDEO_DURATION_MS,
    deserializer::deserialize_tweet_event,
    kafka::utils::{create_kafka_consumer, deserialize_kafka_messages},
    metrics,
    schema::{tweet::Tweet, tweet_events::TweetEventData},
};

/// 批次处理日志计数器：每处理 1000 个批次输出一次里程碑日志
/// 使用 AtomicUsize 保证多线程安全递增
static BATCH_LOG_COUNTER: AtomicUsize = AtomicUsize::new(0);

/// 监控 Kafka 分区 lag 并更新 Prometheus 指标
/// 参数：
///   - consumer: 共享的消费者句柄
///   - topic: 主题名称（用于指标标签）
///   - interval_secs: 监控间隔（秒）
/// 边界：
///   - 该函数是无限循环，仅在任务被取消时退出
///   - get_partition_lags 失败时仅告警，不中断监控循环
async fn monitor_partition_lag(
    consumer: Arc<RwLock<KafkaConsumer>>,
    topic: String,
    interval_secs: u64,
) {
    // 创建定时器，每 interval_secs 触发一次
    let mut interval = tokio::time::interval(Duration::from_secs(interval_secs));

    // 无限循环：持续监控直到任务被取消
    loop {
        // 等待下一个 tick（首次 tick 立即触发）
        interval.tick().await;

        // 获取消费者读锁（多个监控任务可并发读）
        let consumer = consumer.read().await;
        // 查询各分区 lag（消费落后量）
        match consumer.get_partition_lags().await {
            Ok(lag_info) => {
                // 遍历每个分区的 lag 信息
                for partition_lag in lag_info {
                    // 将分区 ID 转为字符串作为指标标签
                    let partition_str = partition_lag.partition_id.to_string();

                    // 上报该分区的 lag 值到 Prometheus gauge
                    metrics::KAFKA_PARTITION_LAG
                        .with_label_values(&[&topic, &partition_str])
                        .set(partition_lag.lag as f64);
                }
            }
            Err(e) => {
                // 查询失败：仅告警，继续下一轮监控
                warn!("Failed to get partition lag info: {}", e);
            }
        }
    }
}

/// 判断一条 tweet 是否为合格视频帖（时长 >= MIN_VIDEO_DURATION_MS）
/// 边界情况：
///   - 无媒体信息 → false
///   - 媒体数量不为 1（0 个或多个）→ false（只接受单媒体帖）
///   - 媒体不是视频 → false
///   - 视频时长缺失 → false（unwrap_or(false)）
///   - 视频时长 >= 阈值 → true
fn is_eligible_video(tweet: &Tweet) -> bool {
    // 获取媒体列表；无媒体则不是视频帖
    let Some(media) = tweet.media.as_ref() else {
        return false;
    };

    // 模式匹配：仅当媒体列表恰好有 1 个元素时绑定到 first_media
    // 边界：0 个或多个媒体都返回 false
    let [first_media] = media.as_slice() else {
        return false;
    };

    // 检查第一个媒体的类型是否为视频
    // 边界：非视频媒体（如图片、GIF）返回 false
    let Some(crate::schema::tweet_media::MediaInfo::VideoInfo(video_info)) =
        first_media.media_info.as_ref()
    else {
        return false;
    };

    // 视频时长 >= 最小阈值才算合格视频
    // 边界：duration_millis 为 None 时返回 false
    video_info
        .duration_millis
        .map(|d| d >= MIN_VIDEO_DURATION_MS)
        .unwrap_or(false)
}

/// 在后台启动分区 lag 监控任务
/// 参数：
///   - consumer: 共享消费者句柄
///   - topic: 主题名称
///   - interval_secs: 监控间隔（秒）
pub fn start_partition_lag_monitor(
    consumer: Arc<RwLock<KafkaConsumer>>,
    topic: String,
    interval_secs: u64,
) {
    // 在 tokio 运行时上派生一个异步任务
    tokio::spawn(async move {
        info!(
            "Starting partition lag monitoring task for topic '{}' (interval: {}s)",
            topic, interval_secs
        );
        // 进入无限监控循环
        monitor_partition_lag(consumer, topic, interval_secs).await;
    });
}

/// 在后台启动 tweet 事件处理循环，支持配置多个处理线程
/// 参数：
///   - base_config: 基础消费者配置（不含分区分配）
///   - producer_config: 生产者配置（用于写入 in-network topic）
///   - args: 命令行参数
/// 边界：
///   - 非 serving 模式下生产者启动失败会 panic（feeder 必须能生产）
///   - serving 模式下不创建生产者（v1 模式仅用于非 serving 场景）
pub async fn start_tweet_event_processing(
    base_config: KafkaConsumerConfig,
    producer_config: KafkaProducerConfig,
    args: &Args,
) {
    // 读取分区总数和线程数配置
    let num_partitions = args.tweet_events_num_partitions as usize;
    let kafka_num_threads = args.kafka_num_threads;

    // 生成所有分区 ID 列表 [0, 1, ..., num_partitions-1]
    let partitions_to_use: Vec<i32> = (0..num_partitions as i32).collect();
    // 计算每个线程负责的分区数（向上取整）
    // 边界：若 kafka_num_threads > num_partitions，部分线程可能分不到分区
    let partitions_per_thread = num_partitions.div_ceil(kafka_num_threads);

    info!(
        "Starting {} message processing threads for {} partitions ({} partitions per thread)",
        kafka_num_threads, num_partitions, partitions_per_thread
    );

    // 创建 Kafka 生产者（仅非 serving 模式）
    let producer = if !args.is_serving {
        info!("Kafka producer enabled, starting producer...");
        // 创建生产者实例并包装为 Arc<RwLock>
        let producer = Arc::new(RwLock::new(KafkaProducer::new(producer_config)));
        // 启动生产者；失败则 panic
        // 边界：生产者启动失败是致命错误，直接 panic 快速失败
        if let Err(e) = producer.write().await.start().await {
            panic!("Failed to start Kafka producer: {:#}", e);
        }
        Some(producer)
    } else {
        info!("Kafka producer disabled, skipping producer initialization");
        None
    };

    // 派生多个处理线程，每个负责一部分分区
    spawn_processing_threads(base_config, partitions_to_use, producer, args);
}

/// 派生多个处理线程，每个线程负责一部分分区
/// 参数：
///   - base_config: 基础消费者配置
///   - partitions_to_use: 所有分区 ID 列表
///   - producer: 可选的生产器句柄
///   - args: 命令行参数
/// 边界：
///   - 线程数多于分区数时，多余线程不派生（start_idx >= total_partitions 时 break）
///   - 每个线程的消费者配置会覆盖 partitions 字段，只消费分配给自己的分区
fn spawn_processing_threads(
    base_config: KafkaConsumerConfig,
    partitions_to_use: Vec<i32>,
    producer: Option<Arc<RwLock<KafkaProducer>>>,
    args: &Args,
) {
    let total_partitions = partitions_to_use.len();
    // 重新计算每线程分区数（与调用方一致）
    let partitions_per_thread = total_partitions.div_ceil(args.kafka_num_threads);

    // 为每个线程 ID 派生一个异步任务
    for thread_id in 0..args.kafka_num_threads {
        // 计算本线程负责的分区区间 [start_idx, end_idx)
        let start_idx = thread_id * partitions_per_thread;
        let end_idx = ((thread_id + 1) * partitions_per_thread).min(total_partitions);

        // 边界：若起始索引超出分区总数，说明线程数多于分区数，停止派生
        if start_idx >= total_partitions {
            break;
        }

        // 提取本线程负责的分区列表
        let thread_partitions = partitions_to_use[start_idx..end_idx].to_vec();
        // 克隆基础配置并设置本线程的分区
        let mut thread_config = base_config.clone();
        thread_config.partitions = Some(thread_partitions.clone());

        // 克隆需要在异步任务中使用的数据
        let producer_clone = producer.as_ref().map(Arc::clone);
        let topic = thread_config.base_config.topic.clone();
        let lag_monitor_interval_secs = args.lag_monitor_interval_secs;
        let batch_size = args.kafka_batch_size;
        let post_retention_sec = args.post_retention_seconds;

        // 派生处理线程任务
        tokio::spawn(async move {
            info!(
                "Starting message processing thread {} for partitions {:?}",
                thread_id, thread_partitions
            );

            // 创建本线程的 Kafka 消费者
            match create_kafka_consumer(thread_config).await {
                Ok(consumer) => {
                    // 为本线程的分区启动 lag 监控
                    start_partition_lag_monitor(
                        Arc::clone(&consumer),
                        topic,
                        lag_monitor_interval_secs,
                    );

                    // 进入主处理循环
                    // 边界：处理循环异常退出是致命错误，直接 panic
                    if let Err(e) = process_tweet_events(
                        consumer,
                        batch_size,
                        producer_clone,
                        post_retention_sec as i64,
                    )
                    .await
                    {
                        panic!(
                            "Tweet events processing thread {} exited unexpectedly: {:#}. This is a critical failure - the feeder cannot function without tweet event processing.",
                            thread_id, e
                        );
                    }
                }
                Err(e) => {
                    // 消费者创建失败：panic 快速失败
                    panic!(
                        "Failed to create consumer for thread {}: {:#}",
                        thread_id, e
                    );
                }
            }
        });
    }
}

/// 处理一批消息：反序列化、提取帖子、发送到生产者
/// 参数：
///   - messages: 一批 Kafka 消息
///   - batch_num: 批次编号（用于日志）
///   - producer: 可选的生产者句柄
///   - post_retention_sec: 帖子保留时长（秒），用于过滤过期删除事件
/// 边界情况：
///   - nullcast 帖子（不进入时间线的帖子）被跳过
///   - 删除事件中帖子已超过保留期则跳过（无需再删除）
///   - 生产者发送失败仅告警，不中断批次
///   - 事件字段缺失时 unwrap 会 panic（Thrift 数据完整性由上游保证）
async fn process_message_batch(
    messages: Vec<KafkaMessage>,
    batch_num: usize,
    producer: Option<Arc<RwLock<KafkaProducer>>>,
    post_retention_sec: i64,
) -> Result<()> {
    // 批量反序列化 Thrift 消息为 TweetEvent
    // 边界：单条失败会被跳过（见 utils），此处 ? 仅在极端情况下返回 Err
    let results = deserialize_kafka_messages(messages, deserialize_tweet_event)?;

    // 创建事件和删除事件列表
    let mut create_tweets = Vec::new();
    let mut delete_tweets = Vec::new();
    // 记录批次中第一条帖子的 ID 和用户 ID（用于里程碑日志）
    let mut first_post_id = 0;
    let mut first_user_id = 0;

    // 记录本批次消息总数（用于日志）
    let len_posts = results.len();

    // 获取当前 Unix 时间戳（秒），用于删除事件的新鲜度判断
    let now_secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs() as i64;

    // 遍历反序列化后的每个事件
    for tweet_event in results {
        // 解包事件数据（Thrift 的 union 字段）
        // 边界：data 为 None 时 panic，依赖上游数据完整性
        let data = tweet_event.data.unwrap();

        match data {
            // ---- 创建事件 ----
            TweetEventData::TweetCreateEvent(create_event) => {
                // 记录批次第一条帖子的 ID 和用户 ID
                first_post_id = create_event.tweet.as_ref().unwrap().id.unwrap();
                first_user_id = create_event.user.as_ref().unwrap().id.unwrap();

                // 解包 tweet 和 core_data
                let tweet = create_event.tweet.as_ref().unwrap();
                let core_data = tweet.core_data.as_ref().unwrap();

                // 跳过 nullcast 帖子（不进入任何用户时间线的帖子）
                // 边界：nullcast 为 None 时视为非 nullcast，正常处理
                if let Some(nullcast) = core_data.nullcast
                    && nullcast
                {
                    continue;
                }

                // 将 Thrift 事件转换为轻量级 LightPost 结构
                create_tweets.push(LightPost {
                    post_id: tweet.id.unwrap(),
                    author_id: create_event.user.as_ref().unwrap().id.unwrap(),
                    created_at: core_data.created_at_secs.unwrap(),
                    // 回复的帖子 ID（无回复则为 None）
                    in_reply_to_post_id: core_data
                        .reply
                        .as_ref()
                        .and_then(|r| r.in_reply_to_status_id),
                    // 回复的用户 ID（无回复则为 None）
                    in_reply_to_user_id: core_data
                        .reply
                        .as_ref()
                        .and_then(|r| r.in_reply_to_user_id),
                    // 是否为转推（share 字段存在即为转推）
                    is_retweet: core_data.share.is_some(),
                    // 是否为回复（reply 字段存在即为回复）
                    is_reply: core_data.reply.is_some(),
                    // 转推来源帖子 ID
                    source_post_id: core_data.share.as_ref().and_then(|s| s.source_status_id),
                    // 转推来源用户 ID
                    source_user_id: core_data.share.as_ref().and_then(|s| s.source_user_id),
                    // 是否为合格视频帖
                    has_video: is_eligible_video(tweet),
                    // 会话 ID（用于回复链追踪）
                    conversation_id: core_data.conversation_id,
                });
            }
            // ---- 删除事件 ----
            TweetEventData::TweetDeleteEvent(delete_event) => {
                // 获取被删除帖子的创建时间
                let created_at_secs = delete_event
                    .tweet
                    .as_ref()
                    .unwrap()
                    .core_data
                    .as_ref()
                    .unwrap()
                    .created_at_secs
                    .unwrap();
                // 边界：若帖子已超过保留期，跳过删除（下游已无此帖）
                if now_secs - created_at_secs > post_retention_sec {
                    continue;
                }
                // 记录要删除的帖子 ID
                delete_tweets.push(delete_event.tweet.as_ref().unwrap().id.unwrap());
            }
            // ---- 引用帖删除事件 ----
            TweetEventData::QuotedTweetDeleteEvent(delete_event) => {
                // 记录被删除的引用帖 ID
                delete_tweets.push(delete_event.quoting_tweet_id.unwrap());
            }
            // ---- 其他事件类型 ----
            _ => {
                log::info!("Other non post creation/deletion event")
            }
        }
    }

    // ---- 将事件发送到生产者（仅当生产者启用时）----
    if let Some(ref producer) = producer {
        // 预分配发送任务列表
        let mut send_tasks = Vec::with_capacity(create_tweets.len());
        // 为每个创建事件构造 InNetworkEvent 并异步发送
        for light_post in &create_tweets {
            // 将 LightPost 包装为 protobuf 的 TweetCreateEvent
            let event = InNetworkEvent {
                event_variant: Some(in_network_event::EventVariant::TweetCreateEvent(
                    TweetCreateEvent {
                        post_id: light_post.post_id,
                        author_id: light_post.author_id,
                        created_at: light_post.created_at,
                        in_reply_to_post_id: light_post.in_reply_to_post_id,
                        in_reply_to_user_id: light_post.in_reply_to_user_id,
                        is_retweet: light_post.is_retweet,
                        is_reply: light_post.is_reply,
                        source_post_id: light_post.source_post_id,
                        source_user_id: light_post.source_user_id,
                        has_video: light_post.has_video,
                        conversation_id: light_post.conversation_id,
                    },
                )),
            };
            // 编码为 protobuf 字节
            let payload = event.encode_to_vec();
            // 克隆生产者句柄供异步任务使用
            let producer_clone = Arc::clone(producer);
            // 派生发送任务（并发发送提高吞吐）
            send_tasks.push(tokio::spawn(async move {
                // 获取生产者读锁
                let producer_lock = producer_clone.read().await;
                // 发送消息；失败仅告警
                if let Err(e) = producer_lock.send(&payload).await {
                    warn!("Failed to send InNetworkEvent to producer: {:#}", e);
                }
            }));
        }

        // 为每个删除事件构造 InNetworkEvent 并异步发送
        for post_id in &delete_tweets {
            let event = InNetworkEvent {
                event_variant: Some(in_network_event::EventVariant::TweetDeleteEvent(
                    TweetDeleteEvent {
                        post_id: *post_id,
                        // 删除时间使用当前时间
                        deleted_at: now_secs,
                    },
                )),
            };
            let payload = event.encode_to_vec();
            let producer_clone = Arc::clone(producer);
            send_tasks.push(tokio::spawn(async move {
                let producer_lock = producer_clone.read().await;
                if let Err(e) = producer_lock.send(&payload).await {
                    warn!("Failed to send InNetworkEvent to producer: {:#}", e);
                }
            }));
        }

        // 等待所有发送任务完成
        // 边界：任务 panic 时记录错误，不中断批次处理
        for task in send_tasks {
            if let Err(e) = task.await {
                error!("Error writing to kafka {}", e);
            }
        }
    }

    // ---- 里程碑日志 - 每 1000 个批次输出一次 ----
    // 原子递增批次计数器，返回递增前的值
    let batch_count = BATCH_LOG_COUNTER.fetch_add(1, Ordering::Relaxed);
    // 边界：is_multiple_of(1000) 在 batch_count=0 时也为 true（0 是任何数的倍数）
    if batch_count.is_multiple_of(1000) {
        info!(
            "Batch processing milestone: processed {} batches total, latest batch {} had {} posts (first: post_id={}, user_id={})",
            batch_count + 1,
            batch_num,
            len_posts,
            first_post_id,
            first_user_id
        );
    }

    Ok(())
}

/// 主消息处理循环：轮询 Kafka、批量累积消息、处理批次并提交 offset
/// 参数：
///   - consumer: 共享消费者句柄
///   - batch_size: 批次大小阈值（达到后触发处理）
///   - producer: 可选生产者句柄
///   - post_retention_sec: 帖子保留时长（秒）
/// 边界情况：
///   - 轮询失败时告警并休眠 100ms 后重试（避免忙循环）
///   - 批次处理失败时返回 Err，由调用方 panic（致命错误）
///   - 消息不足 batch_size 时持续累积，不处理（可能延迟处理）
///   - 处理成功后手动提交 offset，确保不丢消息
async fn process_tweet_events(
    consumer: Arc<RwLock<KafkaConsumer>>,
    batch_size: usize,
    producer: Option<Arc<RwLock<KafkaProducer>>>,
    post_retention_sec: i64,
) -> Result<()> {
    // 消息累积缓冲区
    let mut message_buffer = Vec::new();
    // 批次编号
    let mut batch_num = 0;

    // 无限循环：持续消费直到任务被取消
    loop {
        // 轮询 Kafka 获取消息（最多 100 条）
        // 需要写锁因为 poll 会更新内部状态
        let poll_result = {
            let mut consumer_lock = consumer.write().await;
            consumer_lock.poll(100).await
        };

        match poll_result {
            Ok(messages) => {
                // 将新消息追加到缓冲区
                message_buffer.extend(messages);

                // 当缓冲区达到批次大小时触发处理
                if message_buffer.len() >= batch_size {
                    batch_num += 1;

                    // 取出缓冲区所有消息（std::mem::take 清空缓冲区）
                    let messages = std::mem::take(&mut message_buffer);
                    let producer_clone = producer.clone();

                    // 处理批次
                    // 边界：处理失败返回 Err，由调用方 panic 快速失败
                    process_message_batch(messages, batch_num, producer_clone, post_retention_sec)
                        .await
                        .context("Error processing tweet event batch")?;

                    // 批次处理成功后手动提交 offset
                    // 边界：commit 失败返回 Err，同样触发 panic
                    consumer.write().await.commit_offsets()?;
                }
            }
            Err(e) => {
                // 轮询失败：告警、递增指标、休眠后重试
                warn!("Error polling messages: {:#}", e);
                metrics::KAFKA_POLL_ERRORS.inc();
                // 休眠 100ms 避免失败时忙循环
                tokio::time::sleep(Duration::from_millis(100)).await;
            }
        }
    }
}
