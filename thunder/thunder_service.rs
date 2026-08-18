// =============================================================================
// thunder_service.rs — gRPC 服务 ThunderServiceImpl 的实现
// 职责：
//   1. 实现 InNetworkPostsService trait，提供 GetInNetworkPosts RPC 方法
//   2. 通过 Semaphore 限制并发请求数，防止过载
//   3. 当请求未携带 following 列表时，通过 StratoClient 远程拉取
//   4. 从 PostStore 检索帖子，按时间排序后返回
//   5. 收集并上报丰富的 Prometheus 指标（延迟、帖子数、新鲜度等）
// 边界情况说明：
//   - 信号量满时请求被立即拒绝（RESOURCE_EXHAUSTED），而非排队
//   - StratoClient 拉取失败时返回 INTERNAL 错误
//   - spawn_blocking 任务 panic 时返回 INTERNAL 错误
//   - following/exclude 列表超过 MAX_INPUT_LIST_SIZE 时截断并告警
// =============================================================================

// lazy_static! 宏用于定义全局静态变量（此处未直接使用但保留以备扩展）
use lazy_static::lazy_static;
// debug! / info! / warn! 分别对应不同日志级别
use log::{debug, info, warn};
// Reverse 用于将排序顺序反转（实现从新到旧排列）
use std::cmp::Reverse;
// HashSet 用于 exclude_tweet_ids 的高效去重与查找
use std::collections::HashSet;
// Arc 用于在异步任务间共享 PostStore / StratoClient / Semaphore
use std::sync::Arc;
// SystemTime / UNIX_EPOCH 用于获取当前 Unix 时间戳；Instant 用于计时
use std::time::{Instant, SystemTime, UNIX_EPOCH};
// Semaphore 用于限制并发请求数
use tokio::sync::Semaphore;
// tonic 的 Request / Response / Status 是 gRPC 的标准类型
use tonic::{Request, Response, Status};

// 从 protobuf 生成的代码中导入请求/响应类型和服务 trait
use xai_thunder_proto::{
    GetInNetworkPostsRequest, GetInNetworkPostsResponse, LightPost,
    in_network_posts_service_server::{InNetworkPostsService, InNetworkPostsServiceServer},
};

// 从 config 模块导入全局阈值常量
use crate::config::{
    MAX_INPUT_LIST_SIZE, MAX_POSTS_TO_RETURN, MAX_VIDEOS_TO_RETURN,
};
// 从 metrics 模块导入所有需要的 Prometheus 指标
use crate::metrics::{
    GET_IN_NETWORK_POSTS_COUNT, GET_IN_NETWORK_POSTS_DURATION,
    GET_IN_NETWORK_POSTS_DURATION_WITHOUT_STRATO, GET_IN_NETWORK_POSTS_EXCLUDED_SIZE,
    GET_IN_NETWORK_POSTS_FOLLOWING_SIZE, GET_IN_NETWORK_POSTS_FOUND_FRESHNESS_SECONDS,
    GET_IN_NETWORK_POSTS_FOUND_POSTS_PER_AUTHOR, GET_IN_NETWORK_POSTS_FOUND_REPLY_RATIO,
    GET_IN_NETWORK_POSTS_FOUND_TIME_RANGE_SECONDS, GET_IN_NETWORK_POSTS_FOUND_UNIQUE_AUTHORS,
    GET_IN_NETWORK_POSTS_MAX_RESULTS, IN_FLIGHT_REQUESTS, REJECTED_REQUESTS, Timer,
};
// PostStore 是核心内存帖子存储
use crate::posts::post_store::PostStore;
// StratoClient 用于远程查询用户关注列表
use crate::strato_client::StratoClient;

// ThunderServiceImpl 是 gRPC 服务的具体实现结构体
pub struct ThunderServiceImpl {
    /// PostStore：按 user_id 索引的内存帖子存储，用于检索 in-network 帖子
    post_store: Arc<PostStore>,
    /// StratoClient：当请求未携带 following 列表时，远程拉取关注用户列表
    strato_client: Arc<StratoClient>,
    /// Semaphore：限制同时处理的 gRPC 请求数，防止服务器过载
    /// 边界：超过此数量的请求会被立即拒绝（而非排队），返回 RESOURCE_EXHAUSTED
    request_semaphore: Arc<Semaphore>,
}

impl ThunderServiceImpl {
    /// 构造函数：创建新的 ThunderServiceImpl 实例
    /// 参数：
    ///   - post_store: 共享的帖子存储
    ///   - strato_client: Strato 远程客户端
    ///   - max_concurrent_requests: 最大并发请求数（Semaphore 许可数）
    pub fn new(
        post_store: Arc<PostStore>,
        strato_client: Arc<StratoClient>,
        max_concurrent_requests: usize,
    ) -> Self {
        info!(
            "Initializing ThunderService with max_concurrent_requests={}",
            max_concurrent_requests
        );
        Self {
            post_store,
            strato_client,
            // 创建 Semaphore，许可数 = max_concurrent_requests
            // 边界：若 max_concurrent_requests=0，所有请求都会被拒绝
            request_semaphore: Arc::new(Semaphore::new(max_concurrent_requests)),
        }
    }

    /// 创建 gRPC 服务端实例
    /// 配置 zstd 压缩以减少网络传输量
    /// 边界：accept_compressed / send_compressed 分别控制入站和出站压缩
    pub fn server(self) -> InNetworkPostsServiceServer<Self> {
        InNetworkPostsServiceServer::new(self)
            // 接受客户端发送的 zstd 压缩请求
            .accept_compressed(tonic::codec::CompressionEncoding::Zstd)
            // 向客户端发送 zstd 压缩响应
            .send_compressed(tonic::codec::CompressionEncoding::Zstd)
    }

    /// 分析已找到的帖子集合，计算统计指标并上报 Prometheus
    /// 参数：
    ///   - posts: 帖子切片
    ///   - stage: 阶段标签（如 "retrieved" / "scored"），用于区分不同处理阶段
    /// 边界情况：
    ///   - posts 为空时直接返回，不上报任何指标
    ///   - SystemTime 早于 UNIX_EPOCH 时 unwrap 会 panic（实际不会发生）
    ///   - unique_author_count=0 时跳过 posts_per_author 计算（避免除零）
    fn analyze_and_report_post_statistics(posts: &[LightPost], stage: &str) {
        // 空集合无需分析
        if posts.is_empty() {
            debug!("[{}] No posts found for analysis", stage);
            return;
        }

        // 获取当前 Unix 时间戳（秒）
        // 边界：unwrap 在系统时钟早于 1970-01-01 时 panic，实际不可能发生
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs() as i64;

        // 计算最新帖子的新鲜度（当前时间 - 最新帖子创建时间）
        // .max() 返回 Option（空集合时为 None，但前面已排除空集合）
        let time_since_most_recent = posts
            .iter()
            .map(|post| post.created_at)
            .max()
            .map(|most_recent| now - most_recent);

        // 计算最旧帖子的年龄（当前时间 - 最旧帖子创建时间）
        let time_since_oldest = posts
            .iter()
            .map(|post| post.created_at)
            .min()
            .map(|oldest| now - oldest);

        // 统计回复帖数量（is_reply=true 的帖子）
        let reply_count = posts.iter().filter(|post| post.is_reply).count();
        // 原创帖数量 = 总数 - 回复帖数量
        let original_count = posts.len() - reply_count;

        // 统计唯一作者数（使用 HashSet 去重）
        let unique_authors: HashSet<_> = posts.iter().map(|post| post.author_id).collect();
        let unique_author_count = unique_authors.len();

        // ---- 上报 Prometheus 指标 ----

        // 新鲜度指标：最新帖子距今多少秒
        if let Some(freshness) = time_since_most_recent {
            GET_IN_NETWORK_POSTS_FOUND_FRESHNESS_SECONDS
                .with_label_values(&[stage])
                .observe(freshness as f64);
        }

        // 时间跨度指标：最旧帖子与最新帖子之间的时间差（秒）
        if let (Some(oldest), Some(newest)) = (time_since_oldest, time_since_most_recent) {
            let time_range = oldest - newest;
            GET_IN_NETWORK_POSTS_FOUND_TIME_RANGE_SECONDS
                .with_label_values(&[stage])
                .observe(time_range as f64);
        }

        // 回复率指标：回复帖 / 总帖数
        // 边界：posts.len() > 0（前面已排除空集合），不会除零
        let reply_ratio = reply_count as f64 / posts.len() as f64;
        GET_IN_NETWORK_POSTS_FOUND_REPLY_RATIO
            .with_label_values(&[stage])
            .observe(reply_ratio);

        // 唯一作者数指标
        GET_IN_NETWORK_POSTS_FOUND_UNIQUE_AUTHORS
            .with_label_values(&[stage])
            .observe(unique_author_count as f64);

        // 每作者帖子数指标（仅当有唯一作者时计算，避免除零）
        if unique_author_count > 0 {
            let posts_per_author = posts.len() as f64 / unique_author_count as f64;
            GET_IN_NETWORK_POSTS_FOUND_POSTS_PER_AUTHOR
                .with_label_values(&[stage])
                .observe(posts_per_author);
        }

        // 输出详细统计日志（debug 级别）
        debug!(
            "[{}] Post statistics: total={}, original={}, replies={}, unique_authors={}, posts_per_author={:.2}, reply_ratio={:.2}, time_since_most_recent={:?}s, time_range={:?}s",
            stage,
            posts.len(),
            original_count,
            reply_count,
            unique_author_count,
            if unique_author_count > 0 {
                posts.len() as f64 / unique_author_count as f64
            } else {
                0.0
            },
            reply_ratio,
            time_since_most_recent,
            if let (Some(o), Some(n)) = (time_since_oldest, time_since_most_recent) {
                Some(o - n)
            } else {
                None
            }
        );
    }
}

// #[tonic::async_trait] 自动为 trait 生成 async fn 的实现样板
#[tonic::async_trait]
impl InNetworkPostsService for ThunderServiceImpl {
    /// GetInNetworkPosts RPC 方法：获取用户关注网络内的帖子
    /// 完整流程：
    ///   1. 获取信号量许可（满则拒绝）
    ///   2. 若未携带 following 列表，从 Strato 拉取
    ///   3. 截断 following / exclude 列表至 MAX_INPUT_LIST_SIZE
    ///   4. 在 spawn_blocking 中从 PostStore 检索帖子并按时间排序
    ///   5. 返回结果并上报指标
    async fn get_in_network_posts(
        &self,
        request: Request<GetInNetworkPostsRequest>,
    ) -> Result<Response<GetInNetworkPostsResponse>, Status> {
        // ---- 并发控制 ----
        // 尝试非阻塞地获取信号量许可
        // 边界：若所有许可已被占用，try_acquire 返回 Err，请求被立即拒绝
        //   这避免了排队等待导致的尾延迟放大
        let _permit = match self.request_semaphore.try_acquire() {
            Ok(permit) => {
                // 成功获取许可，增加在途请求计数
                IN_FLIGHT_REQUESTS.inc();
                permit
            }
            Err(_) => {
                // 许可耗尽，增加拒绝计数并返回 RESOURCE_EXHAUSTED
                REJECTED_REQUESTS.inc();
                return Err(Status::resource_exhausted(
                    "Server at capacity, please retry",
                ));
            }
        };

        // ---- 在途请求计数守卫 ----
        // 使用 RAII 模式：当 _in_flight_guard 离开作用域时自动减少计数
        // 边界：无论请求成功还是出错，Drop 都会执行，确保计数准确
        struct InFlightGuard;
        impl Drop for InFlightGuard {
            fn drop(&mut self) {
                IN_FLIGHT_REQUESTS.dec();
            }
        }
        let _in_flight_guard = InFlightGuard;

        // 启动总延迟计时器（RAII：离开作用域时自动记录耗时到直方图）
        let _total_timer = Timer::new(GET_IN_NETWORK_POSTS_DURATION.clone());

        // 解包 gRPC 请求获取内部消息体
        let req = request.into_inner();

        // 若开启 debug 模式，记录请求参数概要
        if req.debug {
            info!(
                "Received GetInNetworkPosts request: user_id={}, following_count={}, exclude_tweet_ids={}",
                req.user_id,
                req.following_user_ids.len(),
                req.exclude_tweet_ids.len(),
            );
        }

        // ---- 获取关注用户列表 ----
        // 若请求未携带 following_user_ids 且开启了 debug 模式，
        // 则从 Strato 远程拉取关注列表
        // 边界：仅在 debug=true 且列表为空时才拉取——生产环境通常由调用方传入列表
        //   若 Strato 拉取失败，返回 INTERNAL 错误
        let following_user_ids = if req.following_user_ids.is_empty() && req.debug {
            info!(
                "Following list is empty, fetching from Strato for user {}",
                req.user_id
            );

            match self
                .strato_client
                .fetch_following_list(req.user_id as i64, MAX_INPUT_LIST_SIZE as i32)
                .await
            {
                Ok(following_list) => {
                    info!(
                        "Fetched {} following users from Strato for user {}",
                        following_list.len(),
                        req.user_id
                    );
                    // 将 i64 列表转换为 u64 列表以匹配 protobuf 字段类型
                    // 边界：i64 负值转换为 u64 会变成巨大的正数，但 Strato 不应返回负 ID
                    following_list.into_iter().map(|id| id as u64).collect()
                }
                Err(e) => {
                    warn!(
                        "Failed to fetch following list from Strato for user {}: {}",
                        req.user_id, e
                    );
                    return Err(Status::internal(format!(
                        "Failed to fetch following list: {}",
                        e
                    )));
                }
            }
        } else {
            // 直接使用请求中携带的 following 列表
            req.following_user_ids
        };

        // ---- 上报请求参数指标 ----
        GET_IN_NETWORK_POSTS_FOLLOWING_SIZE.observe(following_user_ids.len() as f64);
        GET_IN_NETWORK_POSTS_EXCLUDED_SIZE.observe(req.exclude_tweet_ids.len() as f64);

        // 启动不含 Strato 调用的处理延迟计时器
        let _processing_timer = Timer::new(GET_IN_NETWORK_POSTS_DURATION_WITHOUT_STRATO.clone());

        // ---- 确定 max_results ----
        // 优先使用请求指定的值；未指定时根据请求类型使用默认值
        // 边界：max_results=0 视为"未指定"，使用默认值
        let max_results = if req.max_results > 0 {
            req.max_results as usize
        } else if req.is_video_request {
            // 视频请求使用 MAX_VIDEOS_TO_RETURN
            MAX_VIDEOS_TO_RETURN
        } else {
            // 普通帖子请求使用 MAX_POSTS_TO_RETURN
            MAX_POSTS_TO_RETURN
        };
        GET_IN_NETWORK_POSTS_MAX_RESULTS.observe(max_results as f64);

        // ---- 截断 following_user_ids ----
        let following_count = following_user_ids.len();
        if following_count > MAX_INPUT_LIST_SIZE {
            warn!(
                "Limiting following_user_ids from {} to {} entries for user {}",
                following_count, MAX_INPUT_LIST_SIZE, req.user_id
            );
        }
        // take(MAX_INPUT_LIST_SIZE) 只保留前 K 个元素
        // 边界：截断可能导致部分关注用户的帖子被遗漏
        let following_user_ids: Vec<u64> = following_user_ids
            .into_iter()
            .take(MAX_INPUT_LIST_SIZE)
            .collect();

        // ---- 截断 exclude_tweet_ids ----
        let exclude_count = req.exclude_tweet_ids.len();
        if exclude_count > MAX_INPUT_LIST_SIZE {
            warn!(
                "Limiting exclude_tweet_ids from {} to {} entries for user {}",
                exclude_count, MAX_INPUT_LIST_SIZE, req.user_id
            );
        }
        let exclude_tweet_ids: Vec<u64> = req
            .exclude_tweet_ids
            .into_iter()
            .take(MAX_INPUT_LIST_SIZE)
            .collect();

        // ---- 在 spawn_blocking 中执行帖子检索 ----
        // 克隆 Arc 引用以便 move 到阻塞线程
        let post_store = Arc::clone(&self.post_store);
        let request_user_id = req.user_id as i64;

        // spawn_blocking 将 CPU 密集型任务放到独立的阻塞线程池执行
        // 避免阻塞 tokio 的异步运行时线程
        // 边界：若阻塞线程池满，任务排队等待；若任务 panic，JoinError 被转为 INTERNAL
        let proto_posts = tokio::task::spawn_blocking(move || {
            // 将 exclude_tweet_ids 转为 HashSet 以实现 O(1) 查找
            // 边界：重复 ID 会被自动去重
            let exclude_tweet_ids: HashSet<i64> =
                exclude_tweet_ids.iter().map(|&id| id as i64).collect();

            // 记录检索开始时间，用于超时控制
            let start_time = Instant::now();

            // 根据请求类型从 PostStore 检索帖子
            // 视频请求走 get_videos_by_users，普通请求走 get_all_posts_by_users
            let all_posts: Vec<LightPost> = if req.is_video_request {
                post_store.get_videos_by_users(
                    &following_user_ids,
                    &exclude_tweet_ids,
                    start_time,
                    request_user_id,
                )
            } else {
                post_store.get_all_posts_by_users(
                    &following_user_ids,
                    &exclude_tweet_ids,
                    start_time,
                    request_user_id,
                )
            };

            // 分析检索结果并上报指标（阶段标签："retrieved"）
            ThunderServiceImpl::analyze_and_report_post_statistics(&all_posts, "retrieved");

            // 按时间新鲜度排序并截取 max_results 条
            let scored_posts = score_recent(all_posts, max_results);

            // 分析排序后结果并上报指标（阶段标签："scored"）
            ThunderServiceImpl::analyze_and_report_post_statistics(&scored_posts, "scored");

            scored_posts
        })
        .await
        // 将 JoinError 转为 gRPC INTERNAL 状态
        .map_err(|e| Status::internal(format!("Failed to process posts: {}", e)))?;

        if req.debug {
            info!(
                "Returning {} posts for user {}",
                proto_posts.len(),
                req.user_id
            );
        }

        // 记录返回帖子数量指标
        GET_IN_NETWORK_POSTS_COUNT.observe(proto_posts.len() as f64);

        // 构造 gRPC 响应
        let response = GetInNetworkPostsResponse { posts: proto_posts };

        Ok(Response::new(response))
    }
}

/// 按帖子创建时间排序（最新优先），并截取前 max_results 条
/// 参数：
///   - light_posts: 待排序的帖子向量
///   - max_results: 最大返回数量
/// 边界情况：
///   - 空向量：排序后仍为空，take 返回空
///   - max_results=0：take(0) 返回空向量
///   - max_results > len：take 返回全部帖子
///   - created_at 相同的帖子：sort_unstable 不保证相对顺序（不稳定排序）
fn score_recent(mut light_posts: Vec<LightPost>, max_results: usize) -> Vec<LightPost> {
    // 按 created_at 降序排列（Reverse 使最大值排最前）
    // sort_unstable 比 sort 快但不保持相等元素的原始顺序
    light_posts.sort_unstable_by_key(|post| Reverse(post.created_at));

    // 截取前 max_results 条帖子
    light_posts.into_iter().take(max_results).collect()
}
