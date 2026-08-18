// =============================================================================
// posts/post_store.rs — 核心内存帖子存储 PostStore
// 职责：
//   1. 按 user_id 索引帖子，支持原创帖 / 回复帖 / 视频帖三类时间线
//   2. 支持帖子插入、删除标记、批量查询、过期清理
//   3. 提供统计日志与自动清理后台任务
// 数据结构：
//   - posts: post_id → LightPost 全量数据
//   - original_posts_by_user: user_id → 原创帖 TinyPost 队列
//   - secondary_posts_by_user: user_id → 回复/转推帖 TinyPost 队列
//   - video_posts_by_user: user_id → 视频帖 TinyPost 队列
//   - deleted_posts: post_id → 删除标记
// 边界情况说明：
//   - 使用 DashMap 实现无锁并发读写（分段锁）
//   - 删除事件与创建事件可能乱序到达，用 deleted_posts 标记兜底
//   - 查询有超时保护，避免遍历过多用户导致请求超时
//   - 自动清理会压缩队列容量，避免内存膨胀
// =============================================================================

use anyhow::Result;
use dashmap::DashMap;
use log::info;
use std::collections::{HashSet, VecDeque};
use std::sync::Arc;
use std::time::{Duration, Instant};
use xai_thunder_proto::{LightPost, TweetDeleteEvent};

use crate::config::{
    DELETE_EVENT_KEY, MAX_ORIGINAL_POSTS_PER_AUTHOR, MAX_REPLY_POSTS_PER_AUTHOR,
    MAX_TINY_POSTS_PER_USER_SCAN, MAX_VIDEO_POSTS_PER_AUTHOR,
};
use crate::metrics::{
    POST_STORE_DELETED_POSTS, POST_STORE_DELETED_POSTS_FILTERED, POST_STORE_ENTITY_COUNT,
    POST_STORE_POSTS_RETURNED, POST_STORE_POSTS_RETURNED_RATIO, POST_STORE_REQUEST_TIMEOUTS,
    POST_STORE_REQUESTS, POST_STORE_TOTAL_POSTS, POST_STORE_USER_COUNT,
};

/// 存储在用户时间线中的最小帖子引用（仅 ID 和时间戳）
/// 设计目的：时间线只需按时间排序和过滤，无需存储完整帖子数据，
///           大幅减少内存占用（完整数据只存一份在 posts map 中）
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct TinyPost {
    /// 帖子 ID
    pub post_id: i64,
    /// 创建时间戳（Unix 秒）
    pub created_at: i64,
}

impl TinyPost {
    /// 创建新的 TinyPost
    /// 参数：post_id — 帖子 ID；created_at — 创建时间戳
    pub fn new(post_id: i64, created_at: i64) -> Self {
        TinyPost {
            post_id,
            created_at,
        }
    }
}

/// 线程安全的帖子存储，按用户 ID 分组
/// 注意：LightPost 定义在 protobuf schema（in-network.proto）中
/// 边界：所有字段均为 Arc<DashMap>，支持跨线程共享与并发读写
#[derive(Clone)]
pub struct PostStore {
    /// 完整帖子数据，按 post_id 索引
    posts: Arc<DashMap<i64, LightPost>>,
    /// 原创帖时间线：user_id → TinyPost 队列（非回复、非转推）
    original_posts_by_user: Arc<DashMap<i64, VecDeque<TinyPost>>>,
    /// 次级帖时间线：user_id → TinyPost 队列（回复和转推）
    secondary_posts_by_user: Arc<DashMap<i64, VecDeque<TinyPost>>>,
    /// 视频帖时间线：user_id → TinyPost 队列
    video_posts_by_user: Arc<DashMap<i64, VecDeque<TinyPost>>>,
    /// 删除标记：post_id → true（用于处理创建/删除乱序）
    deleted_posts: Arc<DashMap<i64, bool>>,
    /// 帖子保留时长（秒），超过则被清理
    retention_seconds: u64,
    /// 查询超时（0 表示不超时）
    request_timeout: Duration,
}

impl PostStore {
    /// 创建新的空 PostStore
    /// 参数：
    ///   - retention_seconds: 帖子保留时长（秒）
    ///   - request_timeout_ms: 查询超时（毫秒，0 表示不超时）
    pub fn new(retention_seconds: u64, request_timeout_ms: u64) -> Self {
        PostStore {
            // 初始化五个 DashMap（均为空）
            posts: Arc::new(DashMap::new()),
            original_posts_by_user: Arc::new(DashMap::new()),
            secondary_posts_by_user: Arc::new(DashMap::new()),
            video_posts_by_user: Arc::new(DashMap::new()),
            deleted_posts: Arc::new(DashMap::new()),
            retention_seconds,
            request_timeout: Duration::from_millis(request_timeout_ms),
        }
    }

    /// 标记帖子为已删除
    /// 参数：posts — 删除事件列表
    /// 边界情况：
    ///   - 从 posts map 移除完整数据
    ///   - 在 deleted_posts 中记录标记（防止后续创建事件重新插入）
    ///   - 删除事件统一记录在 DELETE_EVENT_KEY 用户的时间线下，
    ///     便于 trim 时按时间清理删除标记
    pub fn mark_as_deleted(&self, posts: Vec<TweetDeleteEvent>) {
        for post in posts.into_iter() {
            // 从完整数据 map 中移除该帖子
            self.posts.remove(&post.post_id);
            // 记录删除标记
            self.deleted_posts.insert(post.post_id, true);

            // 将删除事件追加到 DELETE_EVENT_KEY 用户的时间线
            // 边界：DELETE_EVENT_KEY 是特殊用户 ID，用于统一管理删除标记的过期
            let mut user_posts_entry = self
                .original_posts_by_user
                .entry(DELETE_EVENT_KEY)
                .or_default();
            user_posts_entry.push_back(TinyPost {
                post_id: post.post_id,
                // 使用删除时间作为 created_at，用于过期判断
                created_at: post.deleted_at,
            });
        }
    }

    /// 批量插入帖子到存储
    /// 参数：posts — 待插入的帖子列表
    /// 边界情况：
    ///   - 过滤掉未来时间戳的帖子（时钟偏差保护）
    ///   - 过滤掉超过保留期的帖子
    ///   - 插入前按 created_at 排序，保证时间线有序
    pub fn insert_posts(&self, mut posts: Vec<LightPost>) {
        // 获取当前 Unix 时间戳
        // 边界：unwrap_or_default 在时钟异常时回退为 0
        let current_time = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs() as i64;
        // 过滤：仅保留创建时间在过去且未超过保留期的帖子
        // 边界：created_at >= current_time 的帖子（未来时间）被丢弃
        posts.retain(|p| {
            p.created_at < current_time
                && current_time - p.created_at <= (self.retention_seconds as i64)
        });

        // 按创建时间排序（升序），保证时间线按时间顺序追加
        posts.sort_unstable_by_key(|p| p.created_at);

        // 调用内部插入逻辑
        Self::insert_posts_internal(self, posts);
    }

    /// 完成初始化：排序所有时间线、清理过期帖子、处理乱序删除
    /// 边界：
    ///   - 排序保证时间线按时间升序（后续查询从尾部反向遍历取最新）
    ///   - 清理过期帖子释放内存
    ///   - 删除标记兜底：feeder 中创建/删除事件顺序可能丢失，
    ///     此处确保所有被标记删除的帖子从 posts map 中移除
    pub async fn finalize_init(&self) -> Result<()> {
        // 对所有用户时间线按时间排序
        self.sort_all_user_posts().await;
        // 清理过期帖子
        self.trim_old_posts().await;

        // 遍历所有删除标记，确保对应帖子已从 posts map 移除
        // 边界：这是处理 create/delete 乱序的关键兜底逻辑
        for entry in self.deleted_posts.iter() {
            self.posts.remove(entry.key());
        }

        Ok(())
    }
}

    /// 内部插入逻辑：将帖子写入各时间线
    /// 参数：posts — 已过滤、已排序的帖子列表
    /// 边界情况：
    ///   - 已删除的帖子（deleted_posts 中有标记）被跳过
    ///   - 已存在的帖子 ID 不重复插入（幂等）
    ///   - 转推帖若来源帖有视频，则继承视频属性
    ///   - 回复帖永远不算视频帖
    fn insert_posts_internal(&self, posts: Vec<LightPost>) {
        for post in posts {
            // 提取帖子关键字段
            let post_id = post.post_id;
            let author_id = post.author_id;
            let created_at = post.created_at;
            // 原创帖 = 非回复 且 非转推
            let is_original = !post.is_reply && !post.is_retweet;

            // 边界：若帖子已被标记删除（乱序到达的删除事件先于创建事件），跳过
            if self.deleted_posts.contains_key(&post_id) {
                continue;
            }

            // 存储完整帖子数据
            let old = self.posts.insert(post_id, post);
            if old.is_some() {
                // 边界：帖子已存在则不重复插入时间线（幂等保护）
                continue;
            }

            // 创建时间线引用
            let tiny_post = TinyPost::new(post_id, created_at);

            // 使用 entry API 获取对应用户时间线的可变引用
            if is_original {
                // 原创帖 → 原创时间线
                let mut user_posts_entry =
                    self.original_posts_by_user.entry(author_id).or_default();
                user_posts_entry.push_back(tiny_post.clone());
            } else {
                // 回复/转推 → 次级时间线
                let mut user_posts_entry =
                    self.secondary_posts_by_user.entry(author_id).or_default();
                user_posts_entry.push_back(tiny_post.clone());
            }

            // 判断是否为视频帖
            let mut video_eligible = post.has_video;

            // 边界：若帖子本身无视频标记，但它是转推且来源帖有视频，
            //       则继承来源帖的视频属性（转推视频帖应出现在视频时间线）
            if !video_eligible
                && post.is_retweet
                && let Some(source_post_id) = post.source_post_id
                && let Some(source_post) = self.posts.get(&source_post_id)
            {
                // 仅当来源帖是原创帖且有视频时才继承
                video_eligible = !source_post.is_reply && source_post.has_video;
            }

            // 边界：回复帖永远不算视频帖（即使回复的是视频帖）
            if post.is_reply {
                video_eligible = false;
            }

            // 若为视频帖，同时加入视频时间线
            if video_eligible {
                let mut user_posts_entry = self.video_posts_by_user.entry(author_id).or_default();
                user_posts_entry.push_back(tiny_post);
            }
        }
    }

    /// 从多个用户获取视频帖
    /// 参数：
    ///   - user_ids: 关注用户 ID 列表
    ///   - exclude_tweet_ids: 需要排除的帖子 ID 集合
    ///   - start_time: 查询开始时间（用于超时判断）
    ///   - request_user_id: 发起请求的用户 ID
    /// 返回：视频帖列表
    pub fn get_videos_by_users(
        &self,
        user_ids: &[i64],
        exclude_tweet_ids: &HashSet<i64>,
        start_time: Instant,
        request_user_id: i64,
    ) -> Vec<LightPost> {
        // 从视频时间线查询，每作者最多 MAX_VIDEO_POSTS_PER_AUTHOR 条
        // 边界：following_users 传空集合（视频查询不做回复链过滤）
        let video_posts = self.get_posts_from_map(
            &self.video_posts_by_user,
            user_ids,
            MAX_VIDEO_POSTS_PER_AUTHOR,
            exclude_tweet_ids,
            &HashSet::new(),
            start_time,
            request_user_id,
        );

        // 上报返回帖子数指标
        POST_STORE_POSTS_RETURNED.observe(video_posts.len() as f64);
        video_posts
    }

    /// 从多个用户获取所有帖子（原创 + 次级）
    /// 参数：
    ///   - user_ids: 关注用户 ID 列表
    ///   - exclude_tweet_ids: 需要排除的帖子 ID 集合
    ///   - start_time: 查询开始时间（用于超时判断）
    ///   - request_user_id: 发起请求的用户 ID
    /// 返回：原创帖 + 次级帖合并列表
    pub fn get_all_posts_by_users(
        &self,
        user_ids: &[i64],
        exclude_tweet_ids: &HashSet<i64>,
        start_time: Instant,
        request_user_id: i64,
    ) -> Vec<LightPost> {
        // 将关注用户列表转为集合，用于次级帖的回复链过滤
        let following_users_set: HashSet<i64> = user_ids.iter().copied().collect();

        // 查询原创帖（每作者最多 MAX_ORIGINAL_POSTS_PER_AUTHOR 条）
        // 边界：原创帖不做回复链过滤（following_users 传空集合）
        let mut all_posts = self.get_posts_from_map(
            &self.original_posts_by_user,
            user_ids,
            MAX_ORIGINAL_POSTS_PER_AUTHOR,
            exclude_tweet_ids,
            &HashSet::new(),
            start_time,
            request_user_id,
        );

        // 查询次级帖（回复/转推，每作者最多 MAX_REPLY_POSTS_PER_AUTHOR 条）
        // 边界：次级帖需要回复链过滤（传入 following_users_set）
        let secondary_posts = self.get_posts_from_map(
            &self.secondary_posts_by_user,
            user_ids,
            MAX_REPLY_POSTS_PER_AUTHOR,
            exclude_tweet_ids,
            &following_users_set,
            start_time,
            request_user_id,
        );

        // 合并原创帖和次级帖
        all_posts.extend(secondary_posts);
        // 上报返回帖子数指标
        POST_STORE_POSTS_RETURNED.observe(all_posts.len() as f64);
        all_posts
    }

    /// 从指定时间线 map 查询帖子（核心查询逻辑）
    /// 参数：
    ///   - posts_map: 要查询的时间线 map（原创/次级/视频）
    ///   - user_ids: 关注用户 ID 列表
    ///   - max_per_user: 每用户最多返回条数
    ///   - exclude_tweet_ids: 排除的帖子 ID 集合
    ///   - following_users: 关注用户集合（用于回复链过滤；空集合表示不过滤）
    ///   - start_time: 查询开始时间（用于超时判断）
    ///   - request_user_id: 发起请求的用户 ID
    /// 返回：过滤后的帖子列表
    /// 边界情况：
    ///   - 查询超时则中断遍历并上报超时指标
    ///   - 已删除帖子被过滤（deleted_posts 标记）
    ///   - 转推自己的帖子被过滤（避免自转推）
    ///   - 回复链过滤：仅保留回复原创帖或回复关注用户的帖子
    #[allow(clippy::too_many_arguments)]
    pub fn get_posts_from_map(
        &self,
        posts_map: &Arc<DashMap<i64, VecDeque<TinyPost>>>,
        user_ids: &[i64],
        max_per_user: usize,
        exclude_tweet_ids: &HashSet<i64>,
        following_users: &HashSet<i64>,
        start_time: Instant,
        request_user_id: i64,
    ) -> Vec<LightPost> {
        // 递增查询请求计数指标
        POST_STORE_REQUESTS.inc();
        // 结果列表
        let mut light_posts = Vec::new();

        // 可候选帖子总数（用于计算返回率指标）
        let mut total_eligible: usize = 0;

        // 遍历每个关注用户
        for (i, user_id) in user_ids.iter().enumerate() {
            // 边界：超时保护——若已超过请求超时阈值，中断遍历
            //   request_timeout 为 0 时永不超时（is_zero 判断）
            if !self.request_timeout.is_zero() && start_time.elapsed() >= self.request_timeout {
                log::error!(
                    "Timed out fetching posts for user={}; Processed: {}/{}. Stage: {}",
                    request_user_id,
                    i,
                    user_ids.len(),
                    // 通过 following_users 是否为空判断当前阶段
                    if following_users.is_empty() {
                        "original"
                    } else {
                        "secondary"
                    }
                );
                // 上报超时指标
                POST_STORE_REQUEST_TIMEOUTS.inc();
                break;
            }

            // 获取该用户的时间线（若存在）
            if let Some(user_posts_ref) = posts_map.get(user_id) {
                let user_posts = user_posts_ref.value();
                // 累加可候选帖子数
                total_eligible += user_posts.len();

                // 从最新帖子开始遍历（反向迭代器）
                // 边界：take(MAX_TINY_POSTS_PER_USER_SCAN) 限制扫描深度，
                //       避免用户长期不活跃时遍历到远古帖子
                let tiny_posts_iter = user_posts
                    .iter()
                    .rev()
                    .filter(|post| !exclude_tweet_ids.contains(&post.post_id))
                    .take(MAX_TINY_POSTS_PER_USER_SCAN);

                // 轻量查找：通过 TinyPost 获取完整 LightPost 数据
                // 边界：立即复制值以释放读锁，避免在写者等待时嵌套获取读锁导致死锁
                let light_post_iter_1 = tiny_posts_iter
                    .filter_map(|tiny_post| self.posts.get(&tiny_post.post_id).map(|r| *r.value()));

                // 过滤已删除的帖子
                let light_post_iter = light_post_iter_1.filter(|post| {
                    if self.deleted_posts.get(&post.post_id).is_some() {
                        // 已删除：递增过滤计数并排除
                        POST_STORE_DELETED_POSTS_FILTERED.inc();
                        false
                    } else {
                        true
                    }
                });

                // 过滤转推自己的帖子（避免用户看到自己转推自己的内容）
                let light_post_iter = light_post_iter.filter(|post| {
                    !(post.is_retweet && post.source_user_id == Some(request_user_id))
                });

                // 回复链过滤（仅次级帖查询时启用）
                let filtered_post_iter = light_post_iter.filter(|post| {
                    // following_users 为空表示不做回复链过滤（原创/视频查询）
                    if following_users.is_empty() {
                        return true;
                    }
                    // 非回复帖直接通过
                    post.in_reply_to_post_id.is_none_or(|reply_to_post_id| {
                        // 查找被回复的帖子
                        if let Some(replied_to_post) = self.posts.get(&reply_to_post_id) {
                            // 若被回复帖是原创帖，直接通过
                            if !replied_to_post.is_retweet && !replied_to_post.is_reply {
                                return true;
                            }

                            // 被回复帖是回复/转推：需要更复杂的判断
                            // 仅当满足以下两个条件才通过：
                            //   1. 被回复帖回复的是本会话的原始帖（回复链闭合）
                            //   2. 本回复的回复对象是关注用户
                            return post.conversation_id.is_some_and(|convo_id| {
                                // 被回复帖的回复目标 == 会话原始帖
                                let reply_to_reply_to_original =
                                    replied_to_post.in_reply_to_post_id == Some(convo_id);
                                // 本回复的回复对象在关注列表中
                                let reply_to_followed_user = post
                                    .in_reply_to_user_id
                                    .map(|uid| following_users.contains(&uid))
                                    .unwrap_or(false);

                                // 两个条件同时满足才通过
                                reply_to_reply_to_original && reply_to_followed_user
                            });
                        }

                        // 被回复帖不存在（可能已被删除）：不通过
                        false
                    })
                });

                // 取每用户最多 max_per_user 条并加入结果
                light_posts.extend(filtered_post_iter.take(max_per_user));
            }
        }

        // 上报返回率指标（返回数 / 可候选数）
        // 边界：total_eligible 为 0 时跳过（避免除零）
        if total_eligible > 0 {
            let ratio = light_posts.len() as f64 / total_eligible as f64;
            POST_STORE_POSTS_RETURNED_RATIO.observe(ratio);
        }

        light_posts
    }

    /// 启动后台任务，每 5 秒输出一次 PostStore 统计信息并更新 Prometheus 指标
    /// 边界：
    ///   - 该任务无限循环，仅在 tokio 运行时关闭时终止
    ///   - 遍历 DashMap 计算各时间线总帖子数（O(用户数) 开销）
    pub fn start_stats_logger(self: Arc<Self>) {
        tokio::spawn(async move {
            // 创建 5 秒定时器
            let mut interval = tokio::time::interval(Duration::from_secs(5));

            // 无限循环
            loop {
                // 等待下一个 tick
                interval.tick().await;

                // 获取基础统计：用户数、总帖子数、删除标记数
                let user_count = self.original_posts_by_user.len();
                let total_posts = self.posts.len();
                let deleted_posts = self.deleted_posts.len();

                // 累加各时间线的帖子总数
                let original_posts_count: usize = self
                    .original_posts_by_user
                    .iter()
                    .map(|entry| entry.value().len())
                    .sum();
                let secondary_posts_count: usize = self
                    .secondary_posts_by_user
                    .iter()
                    .map(|entry| entry.value().len())
                    .sum();
                let video_posts_count: usize = self
                    .video_posts_by_user
                    .iter()
                    .map(|entry| entry.value().len())
                    .sum();

                // 更新 Prometheus gauge 指标
                POST_STORE_USER_COUNT.set(user_count as f64);
                POST_STORE_TOTAL_POSTS.set(total_posts as f64);
                POST_STORE_DELETED_POSTS.set(deleted_posts as f64);

                // 更新带标签的实体计数指标
                POST_STORE_ENTITY_COUNT
                    .with_label_values(&["users"])
                    .set(user_count as f64);
                POST_STORE_ENTITY_COUNT
                    .with_label_values(&["posts"])
                    .set(total_posts as f64);
                POST_STORE_ENTITY_COUNT
                    .with_label_values(&["original_posts"])
                    .set(original_posts_count as f64);
                POST_STORE_ENTITY_COUNT
                    .with_label_values(&["secondary_posts"])
                    .set(secondary_posts_count as f64);
                POST_STORE_ENTITY_COUNT
                    .with_label_values(&["video_posts"])
                    .set(video_posts_count as f64);
                POST_STORE_ENTITY_COUNT
                    .with_label_values(&["deleted_posts"])
                    .set(deleted_posts as f64);

                // 输出统计日志
                info!(
                    "PostStore Stats: {} users, {} total posts, {} deleted posts",
                    user_count, total_posts, deleted_posts
                );
            }
        });
    }

    /// 启动后台任务，定期清理超过保留期的帖子
    /// 参数：interval_minutes — 清理间隔（分钟）
    /// 边界：
    ///   - 清理间隔过短会增加 CPU 开销；过长会导致内存膨胀
    ///   - 每次清理后仅在有删除时输出日志
    pub fn start_auto_trim(self: Arc<Self>, interval_minutes: u64) {
        tokio::spawn(async move {
            // 创建定时器（分钟转秒）
            let mut interval = tokio::time::interval(Duration::from_secs(interval_minutes * 60));

            // 无限循环
            loop {
                // 等待下一个 tick
                interval.tick().await;
                // 执行清理，返回清理数量
                let trimmed = self.trim_old_posts().await;
                // 仅在有清理时输出日志（避免日志噪音）
                if trimmed > 0 {
                    info!("Auto-trim: removed {} old posts", trimmed);
                }
            }
        });
    }

    /// 手动清理所有用户时间线中超过保留期的帖子
    /// 返回：清理的帖子总数
    /// 边界情况：
    ///   - 使用 spawn_blocking 避免阻塞异步运行时
    ///   - 队列容量超过实际长度 2 倍时压缩容量（内存优化）
    ///   - 空时间线的用户条目被移除
    ///   - DELETE_EVENT_KEY 时间线中的删除标记也会被清理
    pub async fn trim_old_posts(&self) -> usize {
        // 克隆所有需要访问的 Arc 引用
        let posts_map = Arc::clone(&self.posts);
        let original_posts_by_user = Arc::clone(&self.original_posts_by_user);
        let secondary_posts_by_user = Arc::clone(&self.secondary_posts_by_user);
        let video_posts_by_user = Arc::clone(&self.video_posts_by_user);
        let deleted_posts = Arc::clone(&self.deleted_posts);
        let retention_seconds = self.retention_seconds;

        // 在阻塞线程池中执行清理
        tokio::task::spawn_blocking(move || {
            // 获取当前 Unix 时间戳
            let current_time = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs();

            // 累计清理总数
            let mut total_trimmed = 0;

            // 内部闭包：清理单个时间线 map
            // 边界：
            //   - 时间线按时间升序排列，队首是最旧的帖子，从队首开始清理
            //   - 遇到未过期的帖子即停止（后续帖子更新，无需继续）
            //   - 删除标记（DELETE_EVENT_KEY 用户）同时从 deleted_posts 移除
            let trim_map = |posts_by_user: &DashMap<i64, VecDeque<TinyPost>>,
                            posts_map: &DashMap<i64, LightPost>,
                            deleted_posts: &DashMap<i64, bool>|
             -> usize {
                let mut trimmed = 0;
                // 记录需要移除的空用户条目
                let mut users_to_remove = Vec::new();

                // 遍历每个用户的时间线（可变迭代）
                for mut entry in posts_by_user.iter_mut() {
                    let user_id = *entry.key();
                    let user_posts = entry.value_mut();

                    // 从队首（最旧）开始清理
                    while let Some(oldest_post) = user_posts.front() {
                        // 边界：帖子年龄超过保留期则清理
                        if current_time - (oldest_post.created_at as u64) > retention_seconds {
                            // 弹出最旧帖子
                            let trimmed_post = user_posts.pop_front().unwrap();
                            // 从完整数据 map 中移除
                            posts_map.remove(&trimmed_post.post_id);

                            // 边界：若该条目是删除标记（DELETE_EVENT_KEY 用户），
                            //       同时从 deleted_posts 移除标记
                            if user_id == DELETE_EVENT_KEY {
                                deleted_posts.remove(&trimmed_post.post_id);
                            }
                            trimmed += 1;
                        } else {
                            // 遇到未过期帖子即停止（时间线有序，后续都更新）
                            break;
                        }
                    }

                    // 内存优化：若队列容量超过实际长度 2 倍，压缩容量
                    // 边界：shrink_to 是尽力而为，不保证精确压缩
                    if user_posts.capacity() > user_posts.len() * 2 {
                        let new_cap = user_posts.len() as f32 * 1.5_f32;
                        user_posts.shrink_to(new_cap as usize);
                    }

                    // 记录空时间线的用户（稍后移除）
                    if user_posts.is_empty() {
                        users_to_remove.push(user_id);
                    }
                }

                // 移除空时间线的用户条目
                // 边界：remove_if 带条件，避免误删并发插入的新条目
                for user_id in users_to_remove {
                    posts_by_user.remove_if(&user_id, |_, posts| posts.is_empty());
                }

                trimmed
            };

            // 清理原创帖时间线
            total_trimmed += trim_map(&original_posts_by_user, &posts_map, &deleted_posts);
            // 清理次级帖时间线
            total_trimmed += trim_map(&secondary_posts_by_user, &posts_map, &deleted_posts);
            // 清理视频帖时间线（返回值不累加，避免重复计数）
            // 边界：视频帖与原创/次级帖共享 posts map，重复清理会重复计数
            trim_map(&video_posts_by_user, &posts_map, &deleted_posts);

            total_trimmed
        })
        .await
        // 边界：spawn_blocking 失败（运行时关闭）时 panic
        .expect("spawn_blocking failed")
    }

    /// 对所有用户时间线按创建时间排序（升序）
    /// 边界：
    ///   - 使用 make_contiguous 将 VecDeque 转为连续切片后排序（性能优化）
    ///   - 排序后队首为最旧、队尾为最新，查询时从尾部反向遍历取最新
    pub async fn sort_all_user_posts(&self) {
        // 克隆所有需要访问的 Arc 引用
        let original_posts_by_user = Arc::clone(&self.original_posts_by_user);
        let secondary_posts_by_user = Arc::clone(&self.secondary_posts_by_user);
        let video_posts_by_user = Arc::clone(&self.video_posts_by_user);

        // 在阻塞线程池中执行排序
        tokio::task::spawn_blocking(move || {
            // 排序原创帖时间线
            for mut entry in original_posts_by_user.iter_mut() {
                let user_posts = entry.value_mut();
                user_posts
                    .make_contiguous()
                    .sort_unstable_by_key(|a| a.created_at);
            }
            // 排序次级帖时间线
            for mut entry in secondary_posts_by_user.iter_mut() {
                let user_posts = entry.value_mut();
                user_posts
                    .make_contiguous()
                    .sort_unstable_by_key(|a| a.created_at);
            }
            // 排序视频帖时间线
            for mut entry in video_posts_by_user.iter_mut() {
                let user_posts = entry.value_mut();
                user_posts
                    .make_contiguous()
                    .sort_unstable_by_key(|a| a.created_at);
            }
        })
        .await
        // 边界：spawn_blocking 失败（运行时关闭）时 panic
        .expect("spawn_blocking failed");
    }

    /// 清空存储中的所有数据
    /// 边界：
    ///   - 清空操作是原子的（逐个 clear）
    ///   - 清空后所有查询返回空结果
    pub fn clear(&self) {
        self.posts.clear();
        self.original_posts_by_user.clear();
        self.secondary_posts_by_user.clear();
        self.video_posts_by_user.clear();
        info!("PostStore cleared");
    }
}

impl Default for PostStore {
    /// 默认配置：保留 2 天，无查询超时
    /// 边界：request_timeout=0 表示查询永不超时
    fn default() -> Self {
        // 2 天 = 2 * 24 * 60 * 60 秒
        Self::new(2 * 24 * 60 * 60, 0)
    }
}
