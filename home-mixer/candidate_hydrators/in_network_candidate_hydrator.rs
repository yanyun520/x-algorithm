// 目的：引入候选结构 PostCandidate。
// 影响：本增强器产出并更新候选项的 in_network 字段。
use crate::candidate_pipeline::candidate::PostCandidate;
// 目的：引入查询对象 ScoredPostsQuery。
// 影响：本增强器从查询中读取用户关注关系等上下文。
use crate::candidate_pipeline::query::ScoredPostsQuery;
// 目的：引入 HashSet 集合类型。
// 影响：将用户关注 ID 列表转为集合，实现 O(1) 的成员判断。
use std::collections::HashSet;
// 目的：引入 tonic 异步 trait 支持。
// 影响：使本增强器的 hydrate 方法成为异步方法。
use tonic::async_trait;
// 目的：引入 Hydrator trait。
// 影响：本类型以标准增强器身份接入流水线框架。
use xai_candidate_pipeline::hydrator::Hydrator;

// 目的：定义站内判定增强器（无内部状态）。
// 影响：为每个候选标记是否为站内内容（作者被关注或为本人）。
pub struct InNetworkCandidateHydrator;

// 目的：声明实现异步 Hydrator。
// 影响：流水线可在增强阶段调用其 hydrate。
#[async_trait]
impl Hydrator<ScoredPostsQuery, PostCandidate> for InNetworkCandidateHydrator {
    // 目的：为该增强器挂接调用统计埋点。
    // 影响：监控每次增强的调用次数与耗时。
    #[xai_stats_macro::receive_stats]
    // 目的：实现候选增强主逻辑。
    // 影响：产出与输入候选一一对应的、带 in_network 标记的新候选列表。
    async fn hydrate(
        // 目的：引用当前查询上下文。
        // 影响：从中读取用户 ID 与关注列表。
        &self,
        // 目的：接收查询对象引用。
        // 影响：提供用户维度数据源。
        query: &ScoredPostsQuery,
        // 目的：接收待增强候选列表。
        // 影响：作为输入候选集合。
        candidates: &[PostCandidate],
    // 目的：声明返回值与错误类型。
    // 影响：成功返回增强后的候选，失败返回字符串错误。
    ) -> Result<Vec<PostCandidate>, String> {
        // 目的：将查询中的用户 ID 转为 u64 作为 viewer 标识。
        // 影响：用于判断候选是否为用户本人发布。
        let viewer_id = query.user_id as u64;
        // 目的：将用户关注的作者 ID 列表转为 u64 集合。
        // 影响：快速判断候选作者是否处于好友圈。
        let followed_ids: HashSet<u64> = query
            // 目的：访问查询中的关注用户特征。
            // 影响：关注关系的数据来源。
            .user_features
            // 目的：迭代关注用户 ID。
            // 影响：逐个处理每条关注关系。
            .followed_user_ids
            // 目的：拷贝迭代中的 ID 到局部变量。
            // 影响：避免借用冲突，便于类型转换。
            .iter()
            // 目的：按值取出每个关注 ID。
            // 影响：得到可直接转换的数组元素。
            .copied()
            // 目的：将每个 ID 转为 u64。
            // 影响：与候选 author_id 的类型（u64）对齐。
            .map(|id| id as u64)
            // 目的：收集进 HashSet。
            // 影响：后续可 O(1) 判断作者是否被关注。
            .collect();

        // 目的：对全部候选执行站内判定映射。
        // 影响：得到与输入等长的增强结果列表。
        let hydrated_candidates = candidates
            // 目的：开始迭代候选。
            // 影响：逐个计算站内属性。
            .iter()
            // 目的：为每个候选计算站内标记。
            // 影响：生成新的候选副本并写入 in_network。
            .map(|candidate| {
                // 目的：判断候选作者是否为当前用户本人。
                // 影响：本人内容视为站内，会被推荐逻辑特殊处理。
                let is_self = candidate.author_id == viewer_id;
                // 目的：计算是否站内（本人或作者被关注）。
                // 影响：站内外判定结果用于后续 OON 加权与安全分流。
                let is_in_network = is_self || followed_ids.contains(&candidate.author_id);
                // 目的：构造增强后的候选副本。
                // 影响：仅更新 in_network 字段，其余字段保持默认。
                PostCandidate {
                    // 目的：写入站内标记。
                    // 影响：后续阶段可读取该标记进行分流。
                    in_network: Some(is_in_network),
                    // 目的：其余字段取默认空值。
                    // 影响：由其它增强器负责填充具体内容。
                    ..Default::default()
                }
            })
            // 目的：收集所有增强结果。
            // 影响：形成最终的增强候选列表。
            .collect();

        // 目的：返回增强结果。
        // 影响：流水线据此更新原候选的 in_network。
        Ok(hydrated_candidates)
    }

    // 目的：定义将增强结果合并回原候选的方法。
    // 影响：把新计算的 in_network 写入候选对象。
    fn update(&self, candidate: &mut PostCandidate, hydrated: PostCandidate) {
        // 目的：将增强值同步到原候选。
        // 影响：候选对象最终携带站内标记。
        candidate.in_network = hydrated.in_network;
    }
}
