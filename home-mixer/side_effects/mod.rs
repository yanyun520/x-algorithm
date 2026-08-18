// 目的：声明并对外开放缓存请求信息副作用模块，负责在请求结束后把本次服务结果写入 Strato 缓存。
// 影响：供流水线在返回前调用，为后续请求提供缓存命中依据。
pub mod cache_request_info_side_effect;
