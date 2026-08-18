// =============================================================================
// hydrator.rs — 候选水合器（Hydrator）trait 定义
// 职责：定义流水线中"候选水合阶段"的抽象接口。水合器为候选填充附加字段
//       （如作者社交数据、视频时长、订阅状态），这些字段供后续的过滤与打分使用
// 执行模型：多个水合器【并行】执行（各自独立获取数据，互不依赖），
//           每个水合器返回一份"部分填充"的候选副本，框架按索引合并回原候选
// 设计模式：hydrate → update 两步走
//   - hydrate: 异步批量获取数据，返回与输入同序同长的新候选向量
//   - update/update_all: 只把本水合器负责的字段拷贝回原候选（字段级合并），
//     避免并行水合器之间互相覆盖对方已填充的字段
// 边界情况说明：
//   - 返回向量长度与输入不一致时，整个水合器的结果被【丢弃】（warn 日志），
//     原候选保持未水合状态继续流转——防止索引错位导致字段张冠李戴
//   - 水合器【不允许】丢弃或重排候选（与 Filter 的核心区别）；
//     若需裁剪候选应使用过滤阶段
//   - 水合器失败（Err）时仅记录错误，候选继续流转（fail-open，字段留空）
// =============================================================================

use crate::util;
use std::any::{Any, type_name_of_val};
use tonic::async_trait;

/// 候选水合器 trait：并行执行，更新候选字段
/// 泛型参数：
///   - Q: 查询类型
///   - C: 候选类型
/// 约束：Any 允许向下转型；Send + Sync + 'static 保证可跨线程安全共享
#[async_trait]
pub trait Hydrator<Q, C>: Any + Send + Sync
where
    Q: Clone + Send + Sync + 'static,
    C: Clone + Send + Sync + 'static,
{
    /// 判断此水合器是否对该查询启用
    /// 默认返回 true；实现方可按查询特征覆盖（如仅视频请求启用视频时长水合）
    fn enable(&self, _query: &Q) -> bool {
        true
    }

    /// 水合候选：执行异步操作（如远程调用），返回填充了本水合器字段的新候选
    ///
    /// 重要约束：返回向量必须与输入具有【相同的候选、相同的顺序】。
    /// 水合器中丢弃候选是不允许的——请改用过滤（Filter）阶段。
    /// 边界：若违反此约束（长度不符），框架会丢弃整个结果并告警，
    ///       防止按索引合并时字段错位
    async fn hydrate(&self, query: &Q, candidates: &[C]) -> Result<Vec<C>, String>;

    /// 用水合结果更新单个候选
    /// 只应拷贝本水合器负责的字段（字段级合并），
    /// 避免覆盖其他并行水合器已填充的字段
    fn update(&self, candidate: &mut C, hydrated: C);

    /// 用 hydrated 中的字段批量更新所有候选
    /// 默认实现：按索引逐对调用 update
    /// 边界：zip 按较短一侧截断——框架层已校验长度一致后才调用，
    ///       此处不会发生静默截断
    fn update_all(&self, candidates: &mut [C], hydrated: Vec<C>) {
        for (c, h) in candidates.iter_mut().zip(hydrated) {
            self.update(c, h);
        }
    }

    /// 返回用于日志/指标的稳定组件名（默认取类型短名）
    /// 边界：依赖缺失的 util 模块（见 lib.rs 头注释）
    fn name(&self) -> &'static str {
        util::short_type_name(type_name_of_val(self))
    }
}
