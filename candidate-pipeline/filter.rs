// =============================================================================
// filter.rs — 过滤器（Filter）trait 定义
// 职责：定义流水线中"过滤阶段"的抽象接口。过滤器将候选集划分为两部分：
//   - kept（保留）：继续流向下一个阶段（打分/选择）
//   - removed（移除）：被排除，但仍收集在 PipelineResult.filtered_candidates
//     中供调试与指标分析
// 执行模型：多个过滤器【顺序】执行（前一个的 kept 作为后一个的输入），
//           因为过滤往往有依赖关系（如先去重再过滤低质内容）
// 边界情况说明：
//   - 过滤器失败（返回 Err）时，流水线会回滚到过滤前的候选集（fail-open），
//     保证单个过滤器故障不会清空整个结果集
//   - 与 Hydrator/Scorer 不同，Filter 允许改变候选数量（这正是它的职责）
//   - Any 约束允许运行时向下转型（downcast）到具体类型
// =============================================================================

// Any 支持运行时类型检查与向下转型；type_name_of_val 在编译期获取值的完整类型路径
use std::any::{Any, type_name_of_val};
// async_trait 宏将 trait 中的 async fn 转换为返回 Box<dyn Future> 的普通方法，
// 使 trait 对象（Box<dyn Filter>）可以承载异步方法
use tonic::async_trait;

use crate::util;

/// 过滤结果：kept 与 removed 的分区
/// 边界：kept.len() + removed.len() 应等于输入 candidates.len()，
///       但这由实现方保证，框架层不强制校验（实现方若丢失候选属于 bug）
pub struct FilterResult<C> {
    /// 通过过滤、继续后续阶段的候选
    pub kept: Vec<C>,
    /// 被过滤掉的候选（收集用于调试/指标，不进入最终结果）
    pub removed: Vec<C>,
}

/// 过滤器 trait：顺序执行，将候选划分为保留与移除两个集合
/// 泛型参数：
///   - Q: 查询类型（如 home-mixer 的 PipelineQuery）
///   - C: 候选类型（如 home-mixer 的 Candidate）
/// 约束说明：
///   - Any: 允许运行时向下转型到具体过滤器类型
///   - Send + Sync: 可安全地跨线程共享与传递（tokio 多线程运行时必需）
///   - 'static: 无非静态生命周期引用，可存入 Box<dyn Filter> 长期持有
#[async_trait]
pub trait Filter<Q, C>: Any + Send + Sync
where
    Q: Clone + Send + Sync + 'static,
    C: Clone + Send + Sync + 'static,
{
    /// 判断此过滤器是否对该查询启用
    /// 默认返回 true（始终启用）；实现方可按查询特征覆盖
    /// 边界：enable=false 的过滤器在流水线中被完全跳过，不产生任何日志
    fn enable(&self, _query: &Q) -> bool {
        true
    }

    /// 执行过滤：依据某种准则评估每个候选
    /// 参数：
    ///   - query: 当前请求的查询上下文
    ///   - candidates: 待过滤的候选列表（所有权传入）
    /// 返回：Result<FilterResult<C>, String>
    ///   - Ok: 包含 kept（继续）与 removed（排除）的分区结果
    ///   - Err: 过滤器自身故障（如依赖服务不可用），流水线将回滚候选集
    /// 边界：实现方应保证 kept + removed 覆盖全部输入候选，不丢不重
    async fn filter(&self, query: &Q, candidates: Vec<C>) -> Result<FilterResult<C>, String>;

    /// 返回用于日志/指标的稳定组件名
    /// 默认实现：取 Rust 类型完整路径的短名（如 "AgeFilter"）
    /// 边界：依赖缺失的 util 模块（见 lib.rs 头注释）；实现方可覆盖为自定义名称
    fn name(&self) -> &'static str {
        util::short_type_name(type_name_of_val(self))
    }
}
