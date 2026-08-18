// =============================================================================
// lib.rs — candidate-pipeline crate 的库入口文件
// 作用：声明推荐系统候选流水线框架的所有子模块。该 crate 是一个通用的、
//       基于 Rust trait 的流水线抽象层，被 home-mixer 等上层服务复用来组装
//       "查询水合 → 候选召回 → 候选水合 → 过滤 → 打分 → 选择 → 后置水合/过滤 → 副作用"
//       的完整推荐流程。
// 边界情况说明：
//   - 注意：本仓库中 util.rs 文件缺失（被 .gitignore 或未提交），
//     但下方 `pub mod util;` 与各 trait 文件中的 `util::short_type_name(...)`
//     都引用了它。若直接编译本 crate 会报 E0432（unresolved module）。
//     util 模块预期提供 short_type_name 函数：将 Rust 完整类型路径
//     （如 crate::filters::age_filter::AgeFilter）截取为短名（AgeFilter），
//     用于日志与指标标签。
//   - pub mod 表示模块对外部 crate 可见；home-mixer 通过
//     use candidate_pipeline::filter::Filter 等路径引用这些抽象。
// =============================================================================

// 核心流水线执行引擎：定义 CandidatePipeline trait 及其默认 execute 实现，
// 串联所有阶段并返回 PipelineResult
pub mod candidate_pipeline;

// 过滤器抽象：将候选集划分为"保留"与"移除"两部分，顺序执行
pub mod filter;

// 候选水合器抽象：为候选填充附加字段（如作者信息、视频时长），并行执行
pub mod hydrator;

// 查询水合器抽象：为请求查询填充附加字段（如用户特征、行为序列），并行执行
pub mod query_hydrator;

// 打分器抽象：为候选计算分数（如 ML 模型打分、启发式打分），顺序执行
pub mod scorer;

// 选择器抽象：对打分后的候选排序并截取 Top-K
pub mod selector;

// 副作用抽象：不影响返回结果的异步动作（如缓存请求信息），在后台并行执行
pub mod side_effect;

// 候选源抽象：从各数据源（如 Thunder、Phoenix）召回候选，并行执行
pub mod source;

// 工具函数模块：提供类型短名提取等辅助功能
// 边界：本仓库中该文件缺失，编译会失败（见文件头说明）
pub mod util;
