// =============================================================================
// scorer.rs — 打分器（Scorer）trait 定义
// 职责：定义流水线中"打分阶段"的抽象接口。打分器为候选计算分数
//       （如 ML 模型打分、启发式加权打分），分数供后续 Selector 排序截取
// 执行模型：多个打分器【顺序】执行（与 Hydrator 的并行不同），
//           因为后一个打分器可能依赖前一个打分器写入的分数字段
//           （如加权打分器聚合多个基础打分器的结果）
// 设计模式：与 Hydrator 相同的 score → update 两步走：
//   - score: 异步批量计算，返回与输入同序同长的候选副本（带分数字段）
//   - update/update_all: 只把本打分器负责的字段（通常是 score）拷贝回原候选
// 边界情况说明：
//   - 返回向量长度与输入不一致时，整个打分器的结果被【丢弃】（warn 日志），
//     候选保持未打分状态继续流转——防止索引错位导致分数张冠李戴
//   - 打分器【不允许】丢弃或重排候选；若需裁剪应使用过滤阶段
//   - 打分器失败（Err）时仅记录错误，候选继续流转（fail-open，分数留默认值），
//     Selector 需容忍缺失分数（如 partial_cmp 回退为 Equal）
// =============================================================================

use crate::util;
use std::any::type_name_of_val;
use tonic::async_trait;

/// 打分器 trait：更新候选字段（如 score 字段），顺序执行
/// 泛型参数：
///   - Q: 查询类型
///   - C: 候选类型
/// 约束：Send + Sync + 'static 保证可跨线程安全共享并存入 trait 对象
/// 注意：与 Filter/Hydrator/Source 不同，Scorer 未约束 Any（无需向下转型）
#[async_trait]
pub trait Scorer<Q, C>: Send + Sync
where
    Q: Clone + Send + Sync + 'static,
    C: Clone + Send + Sync + 'static,
{
    /// 判断此打分器是否对该查询启用
    /// 默认返回 true；实现方可按查询特征覆盖（如仅特定实验组启用 ML 打分）
    fn enable(&self, _query: &Q) -> bool {
        true
    }

    /// 打分候选：执行异步操作（如调用 ML 模型服务），返回填充了分数字段的新候选
    ///
    /// 重要约束：返回向量必须与输入具有【相同的候选、相同的顺序】。
    /// 打分器中丢弃候选是不允许的——请改用过滤（Filter）阶段。
    /// 边界：若违反此约束（长度不符），框架会丢弃整个结果并告警，
    ///       防止按索引合并时分数错位
    async fn score(&self, query: &Q, candidates: &[C]) -> Result<Vec<C>, String>;

    /// 用打分结果更新单个候选
    /// 只应拷贝本打分器负责的字段（通常是 score），避免覆盖其他字段
    fn update(&self, candidate: &mut C, scored: C);

    /// 用 scored 中的字段批量更新所有候选
    /// 默认实现：按索引逐对调用 update
    /// 边界：zip 按较短一侧截断——框架层已校验长度一致后才调用
    fn update_all(&self, candidates: &mut [C], scored: Vec<C>) {
        for (c, s) in candidates.iter_mut().zip(scored) {
            self.update(c, s);
        }
    }

    /// 返回用于日志/指标的稳定组件名（默认取类型短名）
    /// 边界：依赖缺失的 util 模块（见 lib.rs 头注释）
    fn name(&self) -> &'static str {
        util::short_type_name(type_name_of_val(self))
    }
}
