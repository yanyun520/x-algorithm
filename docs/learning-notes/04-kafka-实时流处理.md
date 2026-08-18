# 04 · Kafka 实时流处理工程实践

> 核心知识点:分区级并行消费模型、攒批摊薄开销、lag 驱动的"追赶/稳态"双模式、失败即 panic 的 critical-path 哲学。

源码位置:`thunder/kafka/tweet_events_listener_v2.rs`、`thunder/kafka/utils.rs`。

---

## 1. 分区并行模型:Kafka 并行度的正确打开方式

Kafka 的并行单位是**分区**(同分区内保序,跨分区无序)。v2 监听器的做法:

```rust
let partitions_to_use: Vec<i32> = (0..num_partitions as i32).collect();
let partitions_per_thread = num_partitions.div_ceil(kafka_num_threads);

for thread_id in 0..args.kafka_num_threads {
    let thread_partitions = partitions_to_use[start_idx..end_idx].to_vec();
    thread_config.partitions = Some(thread_partitions.clone());
    tokio::spawn(async move {
        let consumer = create_kafka_consumer(thread_config).await; // 每线程独立 consumer
        process_tweet_events_v2(consumer, ...).await
    });
}
```

要点:

- **手动分区分配**(`thread_config.partitions = Some(...)`)而非 consumer group 的自动 rebalance——每个 tokio 任务独占固定分区子集,无再均衡开销与不确定性;
- **每线程独立 consumer 实例**,避免多任务竞争同一把 consumer 锁;
- `div_ceil` 处理分区数不整除线程数的情况,越界直接 `break`。

这种"静态分片 + 独立 worker"的模式,本质上和 DashMap 的分片思想一致:**用数据划分换取无竞争并行**。

## 2. 攒批处理:摊薄固定开销

```rust
let mut message_buffer = Vec::new();
loop {
    let messages = consumer_lock.poll(batch_size).await?;
    message_buffer.extend(messages);
    if message_buffer.len() >= batch_size {
        let messages = std::mem::take(&mut message_buffer);  // 取出所有权,buffer 复用
        tokio::task::spawn_blocking(move || { /* 反序列化 + 入库 */ }).await;
    }
}
```

- **为什么攒批**:逐条处理会让每条消息都承担一次反序列化上下文、锁获取、spawn 开销;攒成 batch 后这些固定成本被摊薄,吞吐显著提升;
- **`std::mem::take`**:用空 Vec 换出满 buffer 的所有权,避免拷贝,也避免重复分配——是"双缓冲"思想的轻量实现。

## 3. 追赶检测与双模式切换

服务重启后,Kafka 里积压了大量历史消息。此时应该**全速追赶**;追上之后应该**限速保在线延迟**。如何判定"追上了"?

```rust
if !init_data_downloaded {
    if let Ok(lags) = consumer_lock.get_partition_lags().await {
        let total_lag: i64 = lags.iter().map(|l| l.lag).sum();
        if total_lag < (lags.len() * batch_size) as i64 {
            init_data_downloaded = true;                 // 切换模式
            Some((tx.clone(), total_lag))                // 通知主流程
        } else { None }
    } else { None }
}
```

- **判定阈值**:`总 lag < 分区数 × 批大小`,即"剩余消息一个批次就能消化完"时认为已追上;
- **模式切换的体现**:追赶期 `permit = None`(不获取信号量,全速),稳态期必须拿到 `Semaphore` 许可(限流,见文档 03);
- **完成通知**:通过 `mpsc` channel 把"初始化完成 + 追赶的消息量"发给主流程,主流程可据此开始对外服务( readiness gate 的简易实现)。

## 4. 错误处理的分层哲学

同一份代码里体现了三种错误策略,对应错误的三种严重程度:

```rust
// ① 可恢复瞬时错误:记指标 + 退避重试
Err(e) => {
    warn!("Error polling messages: {:#}", e);
    metrics::KAFKA_POLL_ERRORS.inc();
    tokio::time::sleep(Duration::from_millis(100)).await;
}

// ② 单批数据损坏:记日志,丢这批,继续
match deserialize_batch(messages) {
    Err(e) => warn!("Error processing batch {}: {:#}", batch_count, e),
    Ok(...) => { /* 入库 */ }
}

// ③ 基础设施级致命错误:直接 panic
if let Err(e) = process_tweet_events_v2(...).await {
    panic!("Tweet events processing thread {} exited unexpectedly: {:#}. \
            This is a critical failure - the feeder cannot function without \
            tweet event processing.", thread_id, e);
}
```

**为什么 panic 是合理的**:帖文事件流是 thunder 的生命线,消费者线程永久退出意味着 in-network 数据停止更新——服务还活着但数据已腐坏,这是"静默降级"的最坏形态。**快速失败(fail-fast)让运维立即感知、触发重启**,远好于带着腐坏数据继续服务。

判断原则:错误发生时问一句——"继续运行还有意义吗?"答案是否定时,panic 比吞掉错误更负责任。

## 5. 乱序与幂等的配套设计

Kafka 只保证分区内有序,跨分区、跨 topic 的事件可能乱序(create/delete 颠倒)。thunder 的应对(见文档 05):

- 删除事件写入**墓碑表** `deleted_posts`,后续迟到的重复 create 会被拦截;
- `finalize_init` 阶段按墓碑表再次统一清理主索引。

**经验**:流处理系统必须假设事件乱序与重复,存储层设计幂等与对账机制,而不是依赖上游保序。

## 6. 可迁移场景

- 任何 Kafka/消息队列消费服务:静态分区分配 + 攒批 + 背压三件套;
- 需要"启动追数据、稳态限流"的同步器:lag 阈值判定 + 模式切换;
- 长连接消费 worker 的错误策略分级:重试 / 跳过 / panic 的决策框架。

---

**上一篇**:[03 · Rust 并发编程模式](03-rust-并发编程模式.md) | **下一篇**:[05 · 内存数据结构设计](05-内存数据结构设计.md)
