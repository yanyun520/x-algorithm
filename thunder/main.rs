// =============================================================================
// main.rs — thunder 服务的二进制入口点
// 职责：
//   1. 解析命令行参数
//   2. 初始化 PostStore（内存帖子存储）与 StratoClient（关注列表查询客户端）
//   3. 启动 gRPC + HTTP 服务器
//   4. 启动 Kafka 消费线程，将 tweet 事件实时灌入 PostStore
//   5. 在 serving 模式下等待 Kafka 追赶完毕后标记就绪，并启动定期清理任务
// 边界情况说明：
//   - 若 Kafka 消费线程启动失败，整个 main 返回 Err，进程退出。
//   - 若 is_serving=false，则不会等待 Kafka catchup 信号，也不会启动 stats logger
//     和 auto-trim 任务——该模式主要用于本地开发 / 回放测试。
//   - http_server 创建失败时通过 anyhow 的 ? 传播错误并退出。
// =============================================================================

// anyhow::Context 为 Result 添加上下文信息，便于错误追踪
use anyhow::{Context, Result};
// axum::Router 用于构建 HTTP 路由（此处 HTTP 路由为空，仅用于 gRPC 侧）
use axum::Router;
// clap::Parser 从命令行参数自动解析为 Args 结构体
use clap::Parser;
// log::info 输出 INFO 级别日志
use log::info;
// Arc 提供线程安全的引用计数共享，用于在多个异步任务间共享 PostStore / StratoClient
use std::sync::Arc;
// Duration 表示时间跨度；Instant 用于测量 Kafka 初始化耗时
use std::time::{Duration, Instant};
// tonic::service::Routes 是 tonic 的 gRPC 路由集合
use tonic::service::Routes;
// xai_http_server 提供统一的 HTTP + gRPC 服务器封装
use xai_http_server::{CancellationToken, GrpcConfig, HttpServer};

// 从 thunder 库中导入所需模块
use thunder::{
    args, kafka_utils, posts::post_store::PostStore, strato_client::StratoClient,
    thunder_service::ThunderServiceImpl,
};

// #[tokio::main] 将 async fn main 转换为 tokio 运行时入口
// 边界：若运行时内部 panic，整个进程终止
#[tokio::main]
async fn main() -> Result<()> {
    // 初始化 env_logger，从 RUST_LOG 环境变量读取日志级别
    // 边界：若未设置 RUST_LOG，默认级别为 off（无日志输出）
    env_logger::init();
    // 解析命令行参数为 Args 结构体；解析失败时 clap 自动打印帮助并退出
    let args = args::Args::parse();

    // ---- 初始化 PostStore ----
    // PostStore 是 thunder 的核心内存数据结构，按 user_id 索引帖子
    // post_retention_seconds：帖子保留时长（超过则被 auto-trim 清理）
    // request_timeout_ms：查询超时阈值（0 表示不超时）
    let post_store = Arc::new(PostStore::new(
        args.post_retention_seconds,
        args.request_timeout_ms,
    ));
    // 日志输出保留天数与超时配置；将秒换算为天便于人工阅读
    info!(
        "Initialized PostStore for in-memory post storage (retention: {} seconds / {:.1} days, request_timeout: {}ms)",
        args.post_retention_seconds,
        args.post_retention_seconds as f64 / 86400.0,
        args.request_timeout_ms
    );

    // ---- 初始化 StratoClient ----
    // StratoClient 用于在请求未携带 following 列表时，按 user_id 远程拉取关注列表
    let strato_client = Arc::new(StratoClient::new());
    info!("Initialized StratoClient");

    // ---- 创建 ThunderService ----
    // 将 PostStore、StratoClient 和并发限制传入服务实现
    // max_concurrent_requests：通过 Semaphore 限制同时处理的 gRPC 请求数
    //   边界：超过此值的请求会被立即拒绝（RESOURCE_EXHAUSTED），而非排队等待
    let thunder_service = ThunderServiceImpl::new(
        Arc::clone(&post_store),
        Arc::clone(&strato_client),
        args.max_concurrent_requests,
    );
    info!(
        "Initialized with max_concurrent_requests={}",
        args.max_concurrent_requests
    );
    // 从 ThunderServiceImpl 获取 gRPC 服务端并包装为 tonic Routes
    let routes = Routes::new(thunder_service.server());

    // ---- 配置 gRPC ----
    // GrpcConfig 绑定监听端口与路由
    let grpc_config = GrpcConfig::new(args.grpc_port, routes);

    // ---- 创建 HTTP/gRPC 服务器 ----
    // 参数依次：HTTP 端口、HTTP 路由（空）、可选 gRPC 配置、取消令牌、就绪检查间隔
    // .context() 在失败时附加 "Failed to create HTTP server" 上下文
    // 边界：若端口被占用，创建失败，main 返回 Err 并退出
    let mut http_server = HttpServer::new(
        args.http_port,
        Router::new(),
        Some(grpc_config),
        CancellationToken::new(),
        Duration::from_secs(10),
    )
    .await
    .context("Failed to create HTTP server")?;

    // 若开启了性能分析（profiling），在 3000 端口启动 pprof/分析服务器
    // 边界：若 3000 端口被占用，spawn_server 内部处理错误，不影响主服务
    if args.enable_profiling {
        xai_profiling::spawn_server(3000, CancellationToken::new()).await;
    }

    // ---- 创建 Kafka 事件通道 ----
    // mpsc channel 用于 Kafka 消费线程在追赶完毕后向主线程发送信号
    // 通道容量设为 kafka_num_threads，每个消费线程最多发送一条 catchup 信号
    // 边界：若所有线程都未发送信号（如非 serving 模式），rx.recv() 会永远阻塞
    let (tx, mut rx) = tokio::sync::mpsc::channel::<i64>(args.kafka_num_threads);
    // 启动 Kafka 消费任务；传入 post_store 的 Arc 副本和 channel sender
    // 空字符串 "" 作为 user 参数（用于 consumer group id 后缀）
    kafka_utils::start_kafka(&args, post_store.clone(), "", tx).await?;

    // ---- serving 模式下的初始化流程 ----
    if args.is_serving {
        // 等待所有 Kafka 消费线程完成初始数据追赶
        // 每收到一条信号代表一个线程已追上最新 offset
        // 边界：若 kafka_num_threads 个信号未全部到达，此处会一直阻塞，
        //       服务器不会标记为 ready，从而避免向未就绪的实例路由流量
        let start = Instant::now();
        for _ in 0..args.kafka_num_threads {
            rx.recv().await;
        }
        info!("Kafka init took {:?}", start.elapsed());

        // 完成初始化：对所有用户帖子排序并清理过期帖子
        // 边界：finalize_init 内部还会处理 create/delete 事件乱序问题
        post_store.finalize_init().await?;

        // 启动 PostStore 统计日志任务（每 5 秒输出一次用户数、帖子数等指标）
        Arc::clone(&post_store).start_stats_logger();
        info!("Started PostStore stats logger",);

        // 启动自动清理任务，每 2 分钟移除超过保留期的帖子
        // 边界：清理间隔过短会增加 CPU 开销；过长会导致内存膨胀
        Arc::clone(&post_store).start_auto_trim(2);
        info!(
            "Started PostStore auto-trim task (interval: 2 minutes, retention: {:.1} days)",
            args.post_retention_seconds as f64 / 86400.0
        );
    }

    // ---- 标记服务器就绪 ----
    // set_readiness(true) 通知负载均衡器此实例可接收流量
    // 边界：在非 serving 模式下也会标记就绪，但此时 PostStore 可能为空
    http_server.set_readiness(true);
    info!("HTTP/gRPC server is ready");

    // ---- 等待终止信号 ----
    // 阻塞直到收到 SIGTERM / SIGINT 或 CancellationToken 被触发
    http_server.wait_for_termination().await;
    info!("Server terminated");

    Ok(())
}
