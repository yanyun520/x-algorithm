// =============================================================================
// query_hydrator.rs — 查询水合器（QueryHydrator）trait 定义
// 职责：定义流水线中"查询水合阶段"（最先执行）的抽象接口。
//       在召回候选之前，为请求查询填充附加字段（如用户特征、用户行为序列），
//       这些字段供后续的 Source（召回）、Filter（过滤）、Scorer（打分）使用
// 执行模型：多个查询水合器【并行】执行，各自返回一份"部分填充"的查询副本，
//           框架按注册顺序依次合并（update）回原查询
// 设计模式：与 Hydrator 相同的 hydrate → update 两步走：
//   - hydrate: 异步获取数据，返回填充了本水合器字段的完整查询副本
//   - update: 只把本水合器负责的字段拷贝回原查询（字段级合并）
// 边界情况说明：
//   - 查询水合失败（Err）时仅记录错误，查询以未水合状态继续流转（fail-open），
//     后续阶段需容忍相关字段为空（如打分器对缺失特征做降级处理）
//   - 与候选水合器不同，查询水合器操作单个查询而非候选列表，
//     因此不存在"长度不匹配"的校验问题
//   - 合并顺序 = 注册顺序：若两个水合器写同一字段，后注册者胜出（应避免）
// =============================================================================

use std::any::{Any, type_name_of_val};
use tonic::async_trait;

use crate::util;

/// 查询水合器 trait：并行执行，更新查询字段
/// 泛型参数：Q — 查询类型（如 home-mixer 的 PipelineQuery）
/// 约束：Any 允许向下转型；Send + Sync + 'static 保证可跨线程安全共享
#[async_trait]
pub trait QueryHydrator<Q>: Any + Send + Sync
where
    Q: Clone + Send + Sync + 'static,
{
    /// 判断此查询水合器是否对该查询启用
    /// 默认返回 true；实现方可按查询特征覆盖（如仅登录用户启用用户特征水合）
    fn enable(&self, _query: &Q) -> bool {
        true
    }

    /// 水合查询：执行异步操作（如远程调用），返回填充了本水合器字段的新查询
    /// 参数：query — 当前请求的查询（不可变引用，并行水合器共享同一份输入）
    /// 返回：Result<Q, String>；Err 表示水合器自身故障
    async fn hydrate(&self, query: &Q) -> Result<Q, String>;

    /// 用水合结果更新查询
    /// 只应拷贝本水合器负责的字段（字段级合并），
    /// 避免覆盖其他并行水合器已填充的字段
    fn update(&self, query: &mut Q, hydrated: Q);

    /// 返回用于日志/指标的稳定组件名（默认取类型短名）
    /// 边界：依赖缺失的 util 模块（见 lib.rs 头注释）
    fn name(&self) -> &'static str {
        util::short_type_name(type_name_of_val(self))
    }
}
