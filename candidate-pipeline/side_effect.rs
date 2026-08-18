// =============================================================================
// side_effect.rs — 副作用（SideEffect）trait 定义
// 职责：定义流水线末尾"副作用阶段"的抽象接口。副作用是不影响返回结果的
//       异步动作（如缓存请求信息供下次使用、上报曝光日志、写审计数据）
// 执行模型：多个副作用【并行】执行，且整个阶段被 tokio::spawn 丢到后台——
//           流水线不等待副作用完成即返回结果（fire-and-forget）
// 边界情况说明：
//   - 副作用失败（Err）只影响自身，不影响主流程（结果已返回）
//   - SideEffectInput 使用 Arc 共享查询与候选，避免多次 clone 的内存开销
//   - enable 接收 Arc<Q>（而非 &Q）因为副作用在 spawn 的独立任务中运行，
//     需要所有权式的共享引用
//   - 进程退出时未完成的副作用可能丢失（fire-and-forget 的固有代价）
// =============================================================================

use crate::util;
use std::any::type_name_of_val;
// Arc 用于在多个后台副作用任务间共享输入，避免深拷贝
use std::sync::Arc;
use tonic::async_trait;

/// 副作用输入：包装查询与已选候选的共享引用
/// 边界：Clone 仅克隆 Arc 与 Vec（浅拷贝），成本低；
///       多个副作用任务共享同一份 query Arc
#[derive(Clone)]
pub struct SideEffectInput<Q, C> {
    /// 已水合的查询（Arc 共享）
    pub query: Arc<Q>,
    /// 最终选中的候选列表
    pub selected_candidates: Vec<C>,
}

/// 副作用 trait：不影响流水线返回结果的动作
/// 泛型参数：
///   - Q: 查询类型
///   - C: 候选类型
/// 约束：Send + Sync + 'static 保证可跨线程安全共享并可 move 进 spawn 的任务
#[async_trait]
pub trait SideEffect<Q, C>: Send + Sync
where
    Q: Clone + Send + Sync + 'static,
    C: Clone + Send + Sync + 'static,
{
    /// 判断此副作用是否应执行
    /// 默认返回 true
    /// 边界：参数是 Arc<Q> 而非 &Q——因为副作用运行在 spawn 的独立任务中，
    ///       需要所有权式的共享引用（无法借用主流程的栈上数据）
    fn enable(&self, _query: Arc<Q>) -> bool {
        true
    }

    /// 执行副作用动作（如写缓存、上报日志）
    /// 参数：input — 共享的查询与选中候选
    /// 返回：Result<(), String>；Err 仅被记录，不影响已返回的主流程结果
    async fn run(&self, input: Arc<SideEffectInput<Q, C>>) -> Result<(), String>;

    /// 返回用于日志/指标的稳定组件名（默认取类型短名）
    /// 边界：依赖缺失的 util 模块（见 lib.rs 头注释）
    fn name(&self) -> &'static str {
        util::short_type_name(type_name_of_val(self))
    }
}
