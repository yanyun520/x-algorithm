// =============================================================================
// kafka/tweet_events_listener_v2.rs — v2 版 tweet 事件监听器（serving 模式使用）
// 职责：
//   1. 从 protobuf 格式的 in-network 事件 topic 消费消息
//   2. 反序列化为 InNetworkEvent，提取创建/删除事件
//   3. 直接写入 PostStore（内存帖子存储）
//   4. 通过信号量限制并发写入，避免与 serving 请求争抢 CPU
//   5. 初始追赶完成后通过 mpsc channel 通知主线程
// 边界情况说明：
//   - 与 v1 不同，v2 不生产消息，只消费并写入 PostStore
//   - 初始追赶阶段不获取信号量（允许全速灌入数据）
//   - 追赶完成后获取信号量（限制并发，为 serving 请求留出 CPU）
//   - 消费线程异常退出时 panic，保证故障快速暴露
// =============================================================================

use anyhow::Result;
use log::{info, warn};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};
use tokio::sync::{RwLock, Semaphore};
use xai_kafka::{KafkaMessage, config::KafkaConsumerConfig, consumer::KafkaConsumer};

use xai_thunder_proto::{LightPost, TweetDeleteEvent, in_network_event};

use crate::{
    args::Args,
    deserializer::deserialize_tweet_event_v2,
    kafka::utils::{create_kafka_consumer, deserialize_kafka_messages},
    metrics,
    posts::post_store::PostStore,
};

/// 反序列化日志计数器：每处理 1000 个批次输出一次反序列化性能日志
static DESER_LOG_COUNTER: AtomicUsize = AtomicUsize::new(0);

/// 在后台启动 v2 tweet 事件处理循环，支持配置多个处理线程
/// 参数：
///   - base_config: 基础消费者配置
///   - post_store: 共享的帖子存储（写入目标）
///   - args: 命令行参数
///   - tx: catchup 信号发送器（每个线程追赶完毕后发送）
pub async fn start_tweet_event_processing_v2(
    base_config: KafkaConsumerConfig,
    post_store: Arc<PostStore>,
    args: &Args,
    tx: tokio::sync::mpsc::Sender<i64>,
) {
    // 读取分区总数和线程数配置
    let num_partitions = args.kafka_tweet_events_v2_num_partitions;
    let kafka_num_threads = args.kafka_num_threads;

    // 生成所有分区 ID 列表 [0, 1, ..., num_partitions-1]
    let partitions_to_use: Vec<i32> = (0..num_partitions as i32).collect();
    // 计算每个线程负责的分区数（向上取整）
    let partitions_per_thread = num_partitions.div_ceil(kafka_num_threads);

    info!(
        "Starting {} message processing threads for {} partitions ({} partitions per thread)",
        kafka_num_threads, num_partitions, partitions_per_thread
    );

    // 派生多个处理线程
    spawn_processing_threads_v2(base_config, partitions_to_use, post_store, args, tx);
}

/// 派生多个处理线程，每个线程负责一部分分区
/// 参数：
///   - base_config: 基础消费者配置
///   - partitions_to_use: 所有分区 ID 列表
///   - post_store: 共享的帖子存储
///   - args: 命令行参数
///   - tx: catchup 信号发送器
/// 边界：
///   - 线程数多于分区数时，多余线程不派生
///   - 共享信号量许可数为 3，限制同时进行的批次写入
fn spawn_processing_threads_v2(
    base_config: KafkaConsumerConfig,
    partitions_to_use: Vec<i32>,
    post_store: Arc<PostStore>,
    args: &Args,
    tx: tokio::sync::mpsc::Sender<i64>,
) {
    let total_partitions = partitions_to_use.len();
    // 每线程分区数（向上取整）
    let partitions_per_thread = total_partitions.div_ceil(args.kafka_num_threads);

    // 创建共享信号量，限制同时进行的批次写入数量
    // 边界：许可数 3 是经验值——允许一定并发写入，同时为 serving 请求留出 CPU
    let semaphore = Arc::new(Semaphore::new(3));

    // 为每个线程 ID 派生一个异步任务
    for thread_id in 0..args.kafka_num_threads {
        // 计算本线程负责的分区区间 [start_idx, end_idx)
        let start_idx = thread_id * partitions_per_thread;
        let end_idx = ((thread_id + 1) * partitions_per_thread).min(total_partitions);

        // 边界：线程数多于分区数时停止派生
        if start_idx >= total_partitions {
            break;
        }

        // 提取本线程负责的分区列表
        let thread_partitions = partitions_to_use[start_idx..end_idx].to_vec();
        // 克隆基础配置并设置本线程的分区
        let mut thread_config = base_config.clone();
        thread_config.partitions = Some(thread_partitions.clone());

        // 克隆需要在异步任务中使用的数据
        let post_store_clone = Arc::clone(&post_store);
        let topic = thread_config.base_config.topic.clone();
        let lag_monitor_interval_secs = args.lag_monitor_interval_secs;
        let batch_size = args.kafka_batch_size;
        let tx_clone = tx.clone();
        let semaphore_clone = Arc::clone(&semaphore);

        // 派生处理线程任务
        tokio::spawn(async move {
            info!(
                "Starting message processing thread {} for partitions {:?}",
                thread_id, thread_partitions
            );

            // 创建本线程的 Kafka 消费者
            match create_kafka_consumer(thread_config).await {
                Ok(consumer) => {
                    // 复用 v1 模块的 lag 监控函数
                    crate::kafka::tweet_events_listener::start_partition_lag_monitor(
                        Arc::clone(&consumer),
                        topic,
                        lag_monitor_interval_secs,
                    );

                    // 进入主处理循环
                    // 边界：处理循环异常退出是致命错误，直接 panic
                    if let Err(e) = process_tweet_events_v2(
                        consumer,
                        post_store_clone,
                        batch_size,
                        tx_clone,
                        semaphore_clone,
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

/// 处理单个批次：反序列化、提取创建/删除事件
/// 参数：messages — 一批 Kafka 消息
/// 返回：(创建帖子列表, 删除事件列表)
/// 边界情况：
///   - event_variant 为 None 时 unwrap 会 panic（依赖上游数据完整性）
///   - is_reply 的判断是宽松的：显式标记 OR 存在回复目标帖 OR 存在回复目标用户
///   - 每 1000 个批次输出一次反序列化性能日志
fn deserialize_batch(
    messages: Vec<KafkaMessage>,
) -> Result<(Vec<LightPost>, Vec<TweetDeleteEvent>)> {
    // 记录反序列化开始时间
    let start_time = Instant::now();
    // 记录消息总数（用于性能日志）
    let num_messages = messages.len();
    // 批量反序列化 protobuf 消息为 InNetworkEvent
    let results = deserialize_kafka_messages(messages, deserialize_tweet_event_v2)?;
    // 计算反序列化耗时
    let deser_elapsed = start_time.elapsed();
    // 每 1000 个批次输出一次性能日志
    if DESER_LOG_COUNTER
        .fetch_add(1, Ordering::Relaxed)
        .is_multiple_of(1000)
    {
        info!(
            "Deserialized {} messages in {:?} ({:.2} msgs/sec)",
            num_messages,
            deser_elapsed,
            // 计算每秒处理消息数；边界：耗时极短时数值可能很大
            num_messages as f64 / deser_elapsed.as_secs_f64()
        );
    }

    // 预分配创建/删除列表
    let mut create_tweets = Vec::with_capacity(results.len());
    let mut delete_tweets = Vec::with_capacity(10);

    // 遍历反序列化后的每个事件
    for tweet_event in results {
        // 解包事件变体（protobuf oneof 字段）
        // 边界：event_variant 为 None 时 panic
        match tweet_event.event_variant.unwrap() {
            // ---- 创建事件 ----
            in_network_event::EventVariant::TweetCreateEvent(create_event) => {
                // 将 protobuf 事件转换为 LightPost
                create_tweets.push(LightPost {
                    post_id: create_event.post_id,
                    author_id: create_event.author_id,
                    created_at: create_event.created_at,
                    in_reply_to_post_id: create_event.in_reply_to_post_id,
                    in_reply_to_user_id: create_event.in_reply_to_user_id,
                    is_retweet: create_event.is_retweet,
                    // 宽松的回复判断：显式标记 OR 存在回复目标帖 OR 存在回复目标用户
                    // 边界：上游可能只填了部分字段，这里做兜底判断
                    is_reply: create_event.is_reply
                        || create_event.in_reply_to_post_id.is_some()
                        || create_event.in_reply_to_user_id.is_some(),
                    source_post_id: create_event.source_post_id,
                    source_user_id: create_event.source_user_id,
                    has_video: create_event.has_video,
                    conversation_id: create_event.conversation_id,
                });
            }
            // ---- 删除事件 ----
            in_network_event::EventVariant::TweetDeleteEvent(delete_event) => {
                delete_tweets.push(delete_event);
            }
        }
    }

    Ok((create_tweets, delete_tweets))
}

/// 主消息处理循环：轮询 Kafka、批量累积消息、写入 PostStore
/// 参数：
///   - consumer: 共享消费者句柄
///   - post_store: 共享的帖子存储
///   - batch_size: 批次大小阈值
///   - tx: catchup 信号发送器
///   - semaphore: 并发写入信号量
/// 边界情况：
///   - 初始追赶阶段（init_data_downloaded=false）不获取信号量，全速灌入数据
///   - 追赶完成后获取信号量（最多 3 个并发批次），为 serving 请求留出 CPU
///   - 追赶完成的判断：总 lag < 分区数 × batch_size（接近追上）
///   - 轮询失败时告警并休眠 100ms 后重试
async fn process_tweet_events_v2(
    consumer: Arc<RwLock<KafkaConsumer>>,
    post_store: Arc<PostStore>,
    batch_size: usize,
    tx: tokio::sync::mpsc::Sender<i64>,
    semaphore: Arc<Semaphore>,
) -> Result<()> {
    // 消息累积缓冲区
    let mut message_buffer = Vec::new();
    // 批次计数
    let mut batch_count = 0_usize;
    // 初始数据是否已追赶完成
    let mut init_data_downloaded = false;

    // 无限循环：持续消费直到任务被取消
    loop {
        // 轮询 Kafka 获取消息（最多 batch_size 条）
        let poll_result = {
            let mut consumer_lock = consumer.write().await;
            consumer_lock.poll(batch_size).await
        };

        match poll_result {
            Ok(messages) => {
                // ---- 判断初始追赶是否完成 ----
                // 仅在尚未完成时检查
                let catchup_sender = if !init_data_downloaded {
                    // 获取消费者读锁查询 lag
                    let consumer_lock = consumer.read().await;
                    if let Ok(lags) = consumer_lock.get_partition_lags().await {
                        // 计算所有分区总 lag
                        let total_lag: i64 = lags.iter().map(|l| l.lag).sum();
                        // 边界：总 lag 小于 分区数×batch_size 视为已追上
                        //   （剩余未消费消息不足一个批次，可认为追赶完成）
                        if total_lag < (lags.len() * batch_size) as i64 {
                            init_data_downloaded = true;
                            // 返回发送器，稍后通知主线程
                            Some((tx.clone(), total_lag))
                        } else {
                            None
                        }
                    } else {
                        // 查询 lag 失败：暂不判定完成
                        None
                    }
                } else {
                    None
                };

                // 将新消息追加到缓冲区
                message_buffer.extend(messages);

                // 当缓冲区达到批次大小时触发处理
                if message_buffer.len() >= batch_size {
                    batch_count += 1;
                    // 取出缓冲区所有消息
                    let messages = std::mem::take(&mut message_buffer);
                    let post_store_clone = Arc::clone(&post_store);

                    // 追赶完成后获取信号量许可，限制并发写入
                    // 边界：追赶阶段不获取许可（全速灌入）；
                    //       追赶后获取许可（最多 3 个并发批次，为 serving 留 CPU）
                    //       unwrap 在信号量关闭时 panic（此处不会发生）
                    let permit = if init_data_downloaded {
                        Some(semaphore.clone().acquire_owned().await.unwrap())
                    } else {
                        None
                    };

                    // 将批次处理发送到阻塞线程池执行
                    // 边界：spawn_blocking 失败时忽略错误（_ = ...）
                    let _ = tokio::task::spawn_blocking(move || {
                        // 持有许可直到任务完成（RAII）
                        let _permit = permit;
                        // 反序列化并写入 PostStore
                        match deserialize_batch(messages) {
                            // 反序列化失败：仅告警，不中断循环
                            Err(e) => warn!("Error processing batch {}: {:#}", batch_count, e),
                            Ok((light_posts, delete_posts)) => {
                                // 写入创建帖子和删除标记
                                post_store_clone.insert_posts(light_posts);
                                post_store_clone.mark_as_deleted(delete_posts);
                            }
                        };
                    })
                    .await;

                    // 若本批次触发了追赶完成，通知主线程
                    if let Some((sender, lag)) = catchup_sender {
                        info!("Completed kafka init for a single thread");
                        // 发送 lag 值作为信号
                        // 边界：接收方已关闭时发送失败，仅记录错误
                        if let Err(e) = sender.send(lag).await {
                            log::error!("error sending {}", e);
                        }
                    }
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
