# 03 · Rust 并发编程模式

> 核心知识点:`DashMap` 分片并发哈希表、`Arc` 共享所有权、tokio 异步任务模型、`spawn_blocking` 隔离 CPU 密集任务、`Semaphore` 背压、原子计数器节流日志。

源码位置:`thunder/posts/post_store.rs`、`thunder/kafka/tweet_events_listener_v2.rs`、`thunder/thunder_service.rs`。

---

## 1. DashMap:读多写少场景的分片锁

`PostStore` 用 5 个 `DashMap` 构建并发索引:

```rust
pub struct PostStore {
    posts: Arc<DashMap<i64, LightPost>>,                       // 主索引
    original_posts_by_user: Arc<DashMap<i64, VecDeque<TinyPost>>>,
    secondary_posts_by_user: Arc<DashMap<i64, VecDeque<TinyPost>>>,
    video_posts_by_user: Arc<DashMap<i64, VecDeque<TinyPost>>>,
    deleted_posts: Arc<DashMap<i64, bool>>,
}
```

**为什么用 DashMap 而不是 `RwLock<HashMap>`**:DashMap 内部按 key 哈希分成多个分片(shard),每个分片独立加锁。不同用户的时间线读写落在不同分片上时**完全无竞争**,而全局 `RwLock` 会让任意写操作阻塞所有读。这正是 thunder "写路径高频摄入 + 读路径低延迟查询"场景的最佳选择。

**entry API 的原子性**:

```rust
let mut user_posts_entry = self.original_posts_by_user.entry(author_id).or_default();
user_posts_entry.push_back(tiny_post);
```

`entry().or_default()` 把"查不到则创建 + 获取可变引用"合成一次原子操作,避免了"先 get 判断存在、再 insert"的 check-then-act 竞态。entry guard 持有期间锁定对应分片,操作完自动释放。

## 2. Arc:跨任务共享所有权

项目中 `Arc` 的三个典型用法:

```rust
// ① 共享大对象,跨 tokio 任务
let post_store_clone = Arc::clone(&post_store);
tokio::spawn(async move { /* 使用 post_store_clone */ });

// ② 共享不可变配置/客户端(trait 对象)
phoenix_client: Arc<dyn PhoenixPredictionClient + Send + Sync>

// ③ 跨异步边界传递只读结果
let arc_hydrated_query = Arc::new(hydrated_query);
let input = Arc::new(SideEffectInput { query: arc_hydrated_query.clone(), ... });
```

要点:`Arc` + 内部不可变性(字段本身是 `Arc<DashMap>` 这种"共享的可并发结构")组合,使得克隆 `PostStore` 极廉价(只是引用计数 +1),同时所有持有者看到同一份数据。

## 3. spawn_blocking:别把 CPU 密集任务塞进 async runtime

Kafka 消息反序列化是纯 CPU 计算。如果在 async 任务里直接执行,会**阻塞 executor 线程**,拖累同线程上所有其他异步任务(包括在线 gRPC 查询)。项目的做法:

```rust
let _ = tokio::task::spawn_blocking(move || {
    let _permit = permit; // permit 移入闭包,任务结束才释放
    match deserialize_batch(messages) {
        Ok((light_posts, delete_posts)) => {
            post_store_clone.insert_posts(light_posts);
            post_store_clone.mark_as_deleted(delete_posts);
        }
        Err(e) => warn!("Error processing batch: {:#}", e),
    };
}).await;
```

**经验法则**:async 任务里只做 I/O 等待和轻量计算;任何可能超过几十微秒的纯计算(编解码、压缩、大 Vec 排序)都应丢进 `spawn_blocking`,由 tokio 专门的阻塞线程池执行。

## 4. Semaphore 背压:保护在线延迟

两处信号量的用途不同,都值得学习:

**(a) Kafka 摄入限流**(`tweet_events_listener_v2.rs`):

```rust
let semaphore = Arc::new(Semaphore::new(3));  // 最多 3 个批处理并发
let permit = if init_data_downloaded {
    Some(semaphore.clone().acquire_owned().await.unwrap())
} else {
    None  // 追赶阶段不限速,全速追 lag
};
```

- 初始追赶:不限速,尽快追平历史消息;
- 稳态运行:最多 3 个批次同时在阻塞池处理,**给在线查询请求保留 CPU**。

**(b) gRPC 请求并发上限**(`thunder_service.rs`):

```rust
request_semaphore: Arc<Semaphore>,  // max_concurrent_requests
```

防止突发流量打爆服务(过载保护),配合 `IN_FLIGHT_REQUESTS`/`REJECTED_REQUESTS` 指标观测。

**模式总结**:信号量是"有界并发"的标准工具——不是队列排队,而是许可制,拿不到就等(或拒绝),把过载控制在系统边界。

## 5. 原子计数器:日志节流

高频路径上每次打日志会成为性能瓶颈,用原子计数器做"每 N 次采样一次":

```rust
static DESER_LOG_COUNTER: AtomicUsize = AtomicUsize::new(0);

if DESER_LOG_COUNTER.fetch_add(1, Ordering::Relaxed).is_multiple_of(1000) {
    info!("Deserialized {} messages in {:?} ({:.2} msgs/sec)", ...);
}
```

`Ordering::Relaxed` 足够——计数只用于采样,不需要同步语义,零内存序开销。

## 6. 并发原语选型速查

| 场景 | 选型 | 项目实例 |
|------|------|----------|
| 读多写少的并发 map | `DashMap` | PostStore 全部索引 |
| 跨任务共享大对象 | `Arc<T>` | post_store / client |
| CPU 密集计算 | `tokio::task::spawn_blocking` | deserialize_batch |
| 有界并发/背压 | `tokio::sync::Semaphore` | 摄入限流、请求上限 |
| 跨任务一次性通知 | `tokio::sync::mpsc` | 追赶完成通知 `tx.send(lag)` |
| 无锁计数/采样 | `AtomicUsize(Relaxed)` | 日志节流 |
| consumer 互斥访问 | `tokio::sync::RwLock` | `Arc<RwLock<KafkaConsumer>>` |

---

**上一篇**:[02 · Rust Trait 流水线框架](02-rust-trait-流水线框架.md) | **下一篇**:[04 · Kafka 实时流处理](04-kafka-实时流处理.md)
