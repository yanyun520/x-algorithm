// =============================================================================
// source.rs — 候选源（Source）trait 定义
// 职责：定义流水线中"召回阶段"的抽象接口。候选源从各数据源召回候选
//       （如 Thunder 的 in-network 帖子、Phoenix 的 ML 检索结果）
// 执行模型：多个候选源【并行】执行（各自独立召回，互不依赖），
//           框架将所有源的结果【追加合并】成一个候选列表（不去重、不排序）
// 边界情况说明：
//   - 单个源失败（Err）仅记录错误，其他源的结果正常合并（fail-open），
//     保证单个数据源故障不会清空整个召回结果
//   - 源失败时贡献 0 条候选，下游阶段照常运行
//   - 不同源可能返回重复候选（如同一帖子被两个源同时召回），
//     去重责任在下游的 Filter（如 DropDuplicatesFilter）
//   - Any 约束允许运行时向下转型到具体源类型
// =============================================================================

use std::any::{Any, type_name_of_val};
use tonic::async_trait;

use crate::util;

/// 候选源 trait：并行执行，召回候选
/// 泛型参数：
///   - Q: 查询类型（水合后的查询，包含召回所需的用户特征等）
///   - C: 候选类型
/// 约束：Any 允许向下转型；Send + Sync + 'static 保证可跨线程安全共享
#[async_trait]
pub trait Source<Q, C>: Any + Send + Sync
where
    Q: Clone + Send + Sync + 'static,
    C: Clone + Send + Sync + 'static,
{
    /// 判断此源是否对该查询启用
    /// 默认返回 true；实现方可按查询特征覆盖（如仅视频请求启用视频源）
    fn enable(&self, _query: &Q) -> bool {
        true
    }

    /// 从数据源召回候选
    /// 参数：query — 水合后的查询（含用户特征、行为序列等召回依据）
    /// 返回：Result<Vec<C>, String>
    ///   - Ok: 召回的候选列表（可为空——源无结果不算错误）
    ///   - Err: 源自身故障（如下游服务不可用），框架记录错误并跳过该源
    async fn get_candidates(&self, query: &Q) -> Result<Vec<C>, String>;

    /// 返回用于日志/指标的稳定组件名（默认取类型短名）
    /// 边界：依赖缺失的 util 模块（见 lib.rs 头注释）
    fn name(&self) -> &'static str {
        util::short_type_name(type_name_of_val(self))
    }
}
