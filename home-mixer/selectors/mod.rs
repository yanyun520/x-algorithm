// 目的：声明（私有）top_k_score_selector 模块，定义按分数取 TopK 的候选选择器。
// 影响：该模块仅在 crate 内部可见，避免暴露内部实现细节。
mod top_k_score_selector;

// 目的：重新导出 TopKScoreSelector，使流水线与本 crate 其他模块可以直接使用。
// 影响：统一对外命名空间，简化选择器的引用路径。
pub use top_k_score_selector::TopKScoreSelector;
