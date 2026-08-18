// 目的：引入 clap 的 Parser 派生宏，用声明式方式定义命令行参数。
// 影响：使 Args 结构体能从命令行解析启动参数，未传参数时给出友好报错。
use clap::Parser;
// 目的：引入 info 日志宏，用于输出运行状态信息。
// 影响：运行日志中可观察到服务启动/就绪/关停等关键节点。
use log::info;
// 目的：引入 std::time::Duration，用于表达 20 秒的优雅停机等待时长。
// 影响：控制 HttpServer 收到终止信号后允许处理中的请求完成的时长。
use std::time::Duration;

// 目的：引入 tonic 压缩编码类型，声明 gRPC 支持 Gzip/Zstd 压缩传输。
// 影响：允许与支持压缩的客户端双向压缩传输，降低带宽开销。
use tonic::codec::CompressionEncoding;
// 目的：引入 RoutesBuilder，用于向 gRPC 服务器注册多个服务。
// 影响：可把业务服务与反射服务同时挂到同一 gRPC 端口。
use tonic::service::RoutesBuilder;
// 目的：引入 tonic-reflection 的 Builder，用于构建 gRPC 反射服务。
// 影响：可让客户端运行时查询 proto 定义，便于工具调试与自动生成客户端。
use tonic_reflection::server::Builder;

// 目的：引入本服务生成的 proto crate 并命名为 pb，统一访问所有 proto 类型。
// 影响：后续代码可通过 pb:: 前缀访问 ScoredPostsQuery、ScoredPostsResponse 等类型。
use xai_home_mixer_proto as pb;
// 目的：引入通用 HTTP 服务器工具（取消令牌、gRPC 配置、HTTP 服务器）。
// 影响：复用基础设施以统一启动 HTTP + gRPC 服务并管理生命周期。
use xai_http_server::{CancellationToken, GrpcConfig, HttpServer};

// 目的：引入 HomeMixerServer 服务实现。
// 影响：此处持有实际的业务逻辑实现，供 gRPC 服务注册。
use xai_home_mixer::HomeMixerServer;
// 目的：引入 params 模块，读取 MAX_GRPC_MESSAGE_SIZE 等配置常量。
// 影响：控制 gRPC 消息的最大编解码大小，避免超大请求/响应被截断。
use xai_home_mixer::params;

// 目的：为 Args 结构体派生 Parser 与 Debug，支持从命令行解析并便于调试打印。
// 影响：实现命令行参数到结构体字段的自动绑定。
#[derive(Parser, Debug)]
// 目的：为 clap 命令设置描述信息，出现在 --help 输出中。
// 影响：提升命令行工具的可发现性与可维护性。
#[command(about = "HomeMixer gRPC Server")]
struct Args {
    // 目的：声明 --grpc_port 长选项，类型为 u16。
    // 影响：使 gRPC 服务监听该端口，缺省或非法值将导致启动失败。
    #[arg(long)]
    grpc_port: u16,
    // 目的：声明 --metrics_port 长选项，类型为 u16。
    // 影响：使指标/健康检查 HTTP 服务监听该端口。
    #[arg(long)]
    metrics_port: u16,
    // 目的：声明 --reload_interval_minutes 长选项（当前未被使用，保留配置）。
    // 影响：预留模型/配置定期重载的时间周期接口。
    #[arg(long)]
    reload_interval_minutes: u64,
    // 目的：声明 --chunk_size 长选项（当前未被使用，保留配置）。
    // 影响：预留分批处理大小配置，方便未来控制批量请求拆分。
    #[arg(long)]
    chunk_size: usize,
}

// 目的：xai_stats_macro 的 main 宏，初始化统计/指标上报框架，并以 home-mixer 作为服务名。
// 影响：服务启动后自动上报进程级指标（如启动时间、请求量等）到监控系统。
#[xai_stats_macro::main(name = "home-mixer")]
// 目的：将异步 main 包装为 tokio 运行时，提供 async/await 执行环境。
// 影响：使 gRPC 服务器能够以异步方式并发处理大量请求。
#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // 目的：解析命令行参数并写入 args，失败时进程直接退出并打印用法。
    // 影响：所有后续启动配置来源于此解析结果。
    let args = Args::parse();
    // 目的：初始化日志系统（xai_init_utils::init().log() 启用日志）。
    // 影响：使 info! 等日志输出生效，便于观测启动过程。
    xai_init_utils::init().log();
    // 目的：初始化 rustls/TLS 基础设施，为后续加密通信提供证书环境。
    // 影响：保证后续需要 TLS 的远程调用（如 VF、Phoenix）可以正常握手。
    xai_init_utils::init().rustls();
    // 目的：记录本次启动所用的端口、重载间隔与块大小配置。
    // 影响：运维可通过日志核对所选配置是否符合预期。
    info!(
        "Starting server with gRPC port: {}, metrics port: {}, reload interval: {} minutes, chunk size: {}",
        args.grpc_port, args.metrics_port, args.reload_interval_minutes, args.chunk_size,
    );

    // Create the service implementation
    // 目的：异步构造 HomeMixerServer，内部会初始化并连接 Phoenix 流水线的全部远程客户端。
    // 影响：完成所有候选处理组件的准备；此处失败则服务无法启动。
    let service = HomeMixerServer::new().await;
    // Keep a reference to stats_receiver before service is moved
    // 目的：构建 gRPC 反射服务，注册本服务的 proto 文件描述符集合。
    // 影响：使客户端工具（如 grpcurl）能力便地查询/调用本服务接口。
    let reflection_service = Builder::configure()
        .register_encoded_file_descriptor_set(pb::FILE_DESCRIPTOR_SET)
        .build_v1()?;

    // 目的：创建 gRPC 路由构建器，用于收集待注册的全部 gRPC 服务。
    // 影响：之后 add_service 的服务都会挂载到同一 gRPC 端口。
    let mut grpc_routes = RoutesBuilder::default();

    // 目的：注册 ScoredPostsService 业务服务到路由表中。
    // 影响：使 get_scored_posts 接口在 gRPC 端口上对外可用。
    grpc_routes.add_service(
        // 目的：用 service 实现创建 gRPC 服务端适配器。
        // 影响：将 trait 实现绑定到 proto 生成的服务 HTTP/2 路由。
        pb::scored_posts_service_server::ScoredPostsServiceServer::new(service)
            // 目的：设置最大解码消息大小，拒绝超过阈值的上行请求。
            // 影响：防止超大体量请求导致内存压力或解析异常。
            .max_decoding_message_size(params::MAX_GRPC_MESSAGE_SIZE)
            // 目的：设置最大编码消息大小，限制下行响应体积。
            // 影响：避免构造超大的响应导致传输失败。
            .max_encoding_message_size(params::MAX_GRPC_MESSAGE_SIZE)
            // 目的：声明接受 Gzip 压缩的请求。
            // 影响：允许客户端以 Gzip 压缩请求体，降低上行带宽。
            .accept_compressed(CompressionEncoding::Gzip)
            // 目的：声明接受 Zstd 压缩的请求。
            // 影响：允许客户端以 Zstd 压缩请求体，提供另一种压缩选择。
            .accept_compressed(CompressionEncoding::Zstd)
            // 目的：声明响应默认以 Gzip 压缩发送。
            // 影响：下行响应压缩后传输，减少带宽占用。
            .send_compressed(CompressionEncoding::Gzip)
            // 目的：声明响应默认以 Zstd 压缩发送。
            // 影响：对支持 Zstd 的客户端提供更高压缩比的传输。
            .send_compressed(CompressionEncoding::Zstd),
    );

    // 目的：把反射服务也注册进路由表。
    // 影响：同一端口同时提供服务调用与 proto 反射查询能力。
    grpc_routes.add_service(reflection_service);

    // 目的：封装 gRPC 服务配置（监听端口与路由表）。
    // 影响：作为 HttpServer 的组成部分，决定 gRPC 的对外监听地址与可路由服务集合。
    let grpc_config = GrpcConfig::new(args.grpc_port, grpc_routes.routes());

    // 目的：创建默认空路由的 axum HTTP 路由器。
    // 影响：为后续通过 HTTP 扩展接口（如自定义 HTTP 端点）预留位置。
    let http_router = axum::Router::default();

    // 目的：创建统一的 HTTP 服务器实例，整合 metrics 端口、HTTP 路由与 gRPC 配置。
    // 影响：服务启动后同时提供指标端口与 gRPC 业务端口。
    let mut server = HttpServer::new(
        // 目的：传入指标监控端口。
        // 影响：健康检查与指标采集在该端口生效。
        args.metrics_port,
        // 目的：传入 HTTP 路由。
        // 影响：扩展的 HTTP 接口会挂载在此路由器上。
        http_router,
        // 目的：传入 gRPC 配置（端口+路由）。
        // 影响：gRPC 业务接口（get_scored_posts）随服务器启动而生效。
        Some(grpc_config),
        // 目的：创建取消令牌，用于主动触发服务器停止。
        // 影响：提供外部中止服务的能力。
        CancellationToken::new(),
        // 目的：设置 20 秒的优雅停机超时。
        // 影响：收到停机信号后最多等待 20 秒让在途请求完成，超时强制退出。
        Duration::from_secs(20),
    )
    // 目的：异步完成服务器启动（绑定端口、初始化资源）。
    // 影响：若端口被占用或初始化失败则返回错误并终止进程。
    .await?;

    // 目的：将服务标记为就绪状态（readiness 探针通过）。
    // 影响：负载均衡器/部署系统认为该实例已可接入流量。
    server.set_readiness(true);
    // 目的：记录服务就绪日志。
    // 影响：运维确认实例已完成启动并对外提供服务。
    info!("Server ready");
    // 目的：阻塞等待终止信号，保持进程存活。
    // 影响：进程在收到 SIGTERM/SIGINT 前持续运行并处理请求。
    server.wait_for_termination().await;
    // 目的：记录服务器已完成优雅关停。
    // 影响：运维日志可确认进程正常退出。
    info!("Server shutdown complete");
    // 目的：以 Ok(()) 返回，代表程序正常结束。
    // 影响：进程以退出码 0 结束，表示成功退出。
    Ok(())
}
