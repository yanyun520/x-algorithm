// 目的：声明并对外开放 core_data 增强模块（补充作者、正文、回复/转发关联 ID）。
// 影响：供流水线通过 `crate::candidate_hydrators::core_data_candidate_hydrator` 初始化该 Hydrator。
pub mod core_data_candidate_hydrator;
// 目的：声明并对外开放 Gizmoduck 用户资料增强模块（补充粉丝数、屏幕名等）。
// 影响：供流水线获取候选作者/转发用户的展示信息，服务于展示与评分。
pub mod gizmoduck_hydrator;
// 目的：声明并对外开放站内(好友关系)增强模块（标记候选是否属于站内内容）。
// 影响：供后续 OON 评分与敏感过滤按站内/站外分流处理。
pub mod in_network_candidate_hydrator;
// 目的：声明并对外开放订阅作者增强模块（补充帖子的订阅作者 ID）。
// 影响：供订阅资格过滤器判断候选是否为仅订阅用户可见内容。
pub mod subscription_hydrator;
// 目的：声明并对外开放可见性过滤(VF)增强模块（获取每个候选的过滤原因）。
// 影响：供后置 VFFilter 依据安全策略决定候选是否可展示。
pub mod vf_candidate_hydrator;
// 目的：声明并对外开放视频时长增强模块（提取帖子的视频时长毫秒数）。
// 影响：供加权评分器判断候选是否满足 VQV 权重生效条件。
pub mod video_duration_candidate_hydrator;
