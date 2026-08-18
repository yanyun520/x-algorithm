// =============================================================================
// selector.rs — 选择器（Selector）trait 定义
// 职责：定义流水线中"选择阶段"的抽象接口。选择器对打分后的候选
//       按分数降序排序，并可选地截取前 K 条（Top-K 选择）
// 执行模型：流水线中只有一个选择器（与多 Source/多 Scorer 不同），
//           同步执行（无异步操作，纯内存排序）
// 边界情况说明：
//   - 分数为 NaN 时 partial_cmp 返回 None，回退为 Ordering::Equal
//     （NaN 候选保持相对位置，不会被排序误杀，但也无法参与有效排序）
//   - size() 默认返回 None（不截断）；流水线最终仍会按 result_size() 截断
//   - enable=false 时跳过排序截断，候选原样进入后置水合阶段
//   - 此 trait 未使用 async_trait（无异步方法），也未约束 Any
// =============================================================================

use crate::util;
use std::any::type_name_of_val;

/// 选择器 trait：默认行为为"按配置排序并截取"
/// 泛型参数：
///   - Q: 查询类型
///   - C: 候选类型
/// 约束：Send + Sync + 'static 保证可跨线程安全共享
pub trait Selector<Q, C>: Send + Sync
where
    Q: Clone + Send + Sync + 'static,
    C: Clone + Send + Sync + 'static,
{
    /// 默认选择流程：先排序，再按 size() 截取
    /// 参数：query — 当前查询；candidates — 打分后的候选（所有权传入）
    /// 返回：排序（可选截取）后的候选
    /// 边界：size() 为 None 时不截断，返回全部排序结果
    fn select(&self, _query: &Q, candidates: Vec<C>) -> Vec<C> {
        // 第一步：按分数降序排序
        let mut sorted = self.sort(candidates);
        // 第二步：若提供了数量上限则截取前 K 条
        if let Some(limit) = self.size() {
            sorted.truncate(limit);
        }
        sorted
    }

    /// 判断此选择器是否对该查询启用
    /// 默认返回 true；enable=false 时流水线跳过排序截断，候选原样通过
    fn enable(&self, _query: &Q) -> bool {
        true
    }

    /// 从单个候选中提取用于排序的分数
    /// 实现方必须提供（无默认实现）——不同候选结构的分数字段位置不同
    fn score(&self, candidate: &C) -> f64;

    /// 按分数降序排序候选
    /// 默认实现：sort_by + partial_cmp
    /// 边界：分数为 NaN 时 partial_cmp 返回 None，unwrap_or 回退为 Equal，
    ///       NaN 候选保持相对位置（不会 panic）
    fn sort(&self, candidates: Vec<C>) -> Vec<C> {
        let mut sorted = candidates;
        // 注意比较方向：self.score(b) 在前——降序（高分在前）
        sorted.sort_by(|a, b| {
            self.score(b)
                .partial_cmp(&self.score(a))
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        sorted
    }

    /// 可选地提供选择数量上限；默认 None 表示不截断
    /// 边界：即使此处不截断，流水线最终也会按 result_size() 统一截断
    fn size(&self) -> Option<usize> {
        None
    }

    /// 返回用于日志/指标的稳定组件名（默认取类型短名）
    /// 边界：依赖缺失的 util 模块（见 lib.rs 头注释）
    fn name(&self) -> &'static str {
        util::short_type_name(type_name_of_val(self))
    }
}
