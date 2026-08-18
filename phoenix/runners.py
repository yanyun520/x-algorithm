# Copyright 2026 X.AI Corp.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import functools  # 导入标准库 functools，提供 lru_cache 缓存与 partial 等函数式工具
import logging  # 导入标准库 logging，用于记录运行日志
from abc import ABC, abstractmethod  # 从 abc 导入 ABC 与 abstractmethod，用于定义抽象基类和抽象方法约束
from dataclasses import dataclass  # 导入 dataclass 装饰器，自动为数据类生成 __init__ 等方法
from typing import Any, List, NamedTuple, Optional, Tuple  # 导入类型注解工具，用于声明函数参数与返回值类型

import haiku as hk  # 导入 Haiku 神经网络库，以面向对象方式在 JAX 上定义可组合模块
import jax  # 导入 JAX 框架，提供自动微分、JIT 编译与并行计算能力
import jax.numpy as jnp  # 导入 JAX 的 numpy 兼容接口，用于操作 JAX 数组
import numpy as np  # 导入 numpy，用于在 CPU 侧构造全零 dummy 数据与随机示例数据

from grok import TrainingState  # 从 grok 模块导入训练状态容器，用于封装模型参数
from recsys_retrieval_model import PhoenixRetrievalModelConfig  # 导入检索模型的配置类，定义检索模型超参数
from recsys_retrieval_model import RetrievalOutput as ModelRetrievalOutput  # 导入检索模型输出结构并重命名，避免与本文件同名类冲突

from recsys_model import (  # 从排序模型模块导入以下类型
    PhoenixModelConfig,  # 排序模型配置类，定义模型超参数与结构
    RecsysBatch,  # 排序批次数据结构，包含用户/历史/候选的哈希、动作与产品表面特征
    RecsysEmbeddings,  # 预查好的嵌入数据结构，包含用户/历史/候选的嵌入向量
    RecsysModelOutput,  # 排序模型输出结构，包含 logits 等字段
)

rank_logger = logging.getLogger("rank")  # 获取名为 "rank" 的日志记录器，用于排序/检索流程的日志输出


def create_dummy_batch_from_config(  # 定义函数：根据配置构造全零的 dummy 批次，用于参数初始化时的 JAX 追踪
    hash_config: Any,  # 哈希配置对象，包含 num_user_hashes / num_item_hashes / num_author_hashes
    history_len: int,  # 历史序列长度
    num_candidates: int,  # 候选数量
    num_actions: int,  # 动作类型数量
    batch_size: int = 1,  # 批次大小，默认 1（初始化仅需 1 个样本即可确定形状）
) -> RecsysBatch:  # 返回 RecsysBatch 类型
    """Create a dummy batch for initialization.

    Args:
        hash_config: HashConfig with num_user_hashes, num_item_hashes, num_author_hashes
        history_len: History sequence length
        num_candidates: Number of candidates
        num_actions: Number of action types
        batch_size: Batch size

    Returns:
        RecsysBatch with zeros
    """
    return RecsysBatch(  # 构造并返回一个 RecsysBatch 对象
        user_hashes=np.zeros((batch_size, hash_config.num_user_hashes), dtype=np.int32),  # 用户哈希：全零数组，形状 (B, 用户哈希数)，全零值用于触发 JAX 追踪以确定参数形状
        history_post_hashes=np.zeros(  # 历史帖子哈希：全零数组，形状 (B, 历史长度, 帖子哈希数)
            (batch_size, history_len, hash_config.num_item_hashes), dtype=np.int32
        ),
        history_author_hashes=np.zeros(  # 历史作者哈希：全零数组，形状 (B, 历史长度, 作者哈希数)
            (batch_size, history_len, hash_config.num_author_hashes), dtype=np.int32
        ),
        history_actions=np.zeros((batch_size, history_len, num_actions), dtype=np.float32),  # 历史动作：全零数组，形状 (B, 历史长度, 动作数)
        history_product_surface=np.zeros((batch_size, history_len), dtype=np.int32),  # 历史产品表面：全零数组，形状 (B, 历史长度)
        candidate_post_hashes=np.zeros(  # 候选帖子哈希：全零数组，形状 (B, 候选数, 帖子哈希数)
            (batch_size, num_candidates, hash_config.num_item_hashes), dtype=np.int32
        ),
        candidate_author_hashes=np.zeros(  # 候选作者哈希：全零数组，形状 (B, 候选数, 作者哈希数)
            (batch_size, num_candidates, hash_config.num_author_hashes), dtype=np.int32
        ),
        candidate_product_surface=np.zeros((batch_size, num_candidates), dtype=np.int32),  # 候选产品表面：全零数组，形状 (B, 候选数)
    )


def create_dummy_embeddings_from_config(  # 定义函数：根据配置构造全零的 dummy 嵌入，用于参数初始化时的 JAX 追踪
    hash_config: Any,  # 哈希配置对象，决定各哈希维度
    emb_size: int,  # 嵌入向量维度 D
    history_len: int,  # 历史序列长度
    num_candidates: int,  # 候选数量
    batch_size: int = 1,  # 批次大小，默认 1
) -> RecsysEmbeddings:  # 返回 RecsysEmbeddings 类型
    """Create dummy embeddings for initialization.

    Args:
        hash_config: HashConfig with num_user_hashes, num_item_hashes, num_author_hashes
        emb_size: Embedding dimension
        history_len: History sequence length
        num_candidates: Number of candidates
        batch_size: Batch size

    Returns:
        RecsysEmbeddings with zeros
    """
    return RecsysEmbeddings(  # 构造并返回 RecsysEmbeddings 对象
        user_embeddings=np.zeros(  # 用户嵌入：全零数组，形状 (B, 用户哈希数, D)
            (batch_size, hash_config.num_user_hashes, emb_size), dtype=np.float32
        ),
        history_post_embeddings=np.zeros(  # 历史帖子嵌入：全零数组，形状 (B, 历史长度, 帖子哈希数, D)
            (batch_size, history_len, hash_config.num_item_hashes, emb_size), dtype=np.float32
        ),
        candidate_post_embeddings=np.zeros(  # 候选帖子嵌入：全零数组，形状 (B, 候选数, 帖子哈希数, D)
            (batch_size, num_candidates, hash_config.num_item_hashes, emb_size),
            dtype=np.float32,
        ),
        history_author_embeddings=np.zeros(  # 历史作者嵌入：全零数组，形状 (B, 历史长度, 作者哈希数, D)
            (batch_size, history_len, hash_config.num_author_hashes, emb_size), dtype=np.float32
        ),
        candidate_author_embeddings=np.zeros(  # 候选作者嵌入：全零数组，形状 (B, 候选数, 作者哈希数, D)
            (batch_size, num_candidates, hash_config.num_author_hashes, emb_size),
            dtype=np.float32,
        ),
    )


@dataclass  # 将此类标记为数据类，基于字段自动生成构造逻辑
class BaseModelRunner(ABC):  # 定义模型运行器抽象基类，继承 ABC 以支持抽象方法约束
    """Base class for model runners with shared initialization logic."""

    bs_per_device: float = 2.0  # 每块设备上的 batch size 因子，默认 2.0，用于按设备数计算总 batch size
    rng_seed: int = 42  # 随机数种子，默认 42，保证初始化过程可复现

    @property  # 将 model 方法声明为只读属性，调用时无需加括号
    @abstractmethod  # 声明为抽象方法，强制子类必须实现
    def model(self) -> Any:  # 抽象属性：返回模型配置对象
        """Return the model config."""
        pass  # 占位，具体逻辑由子类实现

    @property  # 将 _model_name 声明为只读属性
    def _model_name(self) -> str:  # 返回模型名称字符串，用于日志输出
        """Return model name for logging."""
        return "model"  # 默认返回 "model"，子类可覆盖以区分不同模型

    @abstractmethod  # 抽象方法，强制子类实现
    def make_forward_fn(self):  # 创建前向函数，返回被 hk.transform 转换后的纯函数
        """Create the forward function. Must be implemented by subclasses."""
        pass  # 占位

    def initialize(self):  # 初始化模型运行器：设置精度、计算 batch size、构建前向函数
        """Initialize the model runner."""
        self.model.initialize()  # 调用模型配置的 initialize 方法，完成模型内部状态的初始化
        self.model.fprop_dtype = jnp.bfloat16  # 将前向传播精度设为 bfloat16，加速计算并节省显存
        num_local_gpus = len(jax.local_devices())  # 获取当前进程可见的本地设备（GPU/TPU）数量

        self.batch_size = max(1, int(self.bs_per_device * num_local_gpus))  # 计算总 batch size = 每设备 batch × 设备数，并至少取 1

        rank_logger.info(f"Initializing {self._model_name}...")  # 记录日志：正在初始化哪个模型
        self.forward = self.make_forward_fn()  # 构建前向函数（hk.transform 后的 init/apply 对）并保存到实例


@dataclass  # 数据类装饰器，自动生成构造方法
class BaseInferenceRunner(ABC):  # 定义推理运行器抽象基类，继承 ABC 支持抽象方法
    """Base class for inference runners with shared dummy data creation."""

    name: str  # 推理运行器名称字段，用于标识该推理器

    @property  # 属性装饰器
    @abstractmethod  # 抽象方法
    def runner(self) -> BaseModelRunner:  # 返回底层模型运行器对象
        """Return the underlying model runner."""
        pass  # 占位

    def _get_num_actions(self) -> int:  # 获取动作类型数量
        """Get number of actions. Override in subclasses if needed."""
        model_config = self.runner.model  # 取得底层模型配置对象
        if hasattr(model_config, "num_actions"):  # 判断配置是否显式定义了 num_actions 属性
            return model_config.num_actions  # 若已定义则直接返回
        return 19  # 否则返回默认值 19（与 ACTIONS 列表长度一致）

    def create_dummy_batch(self, batch_size: int = 1) -> RecsysBatch:  # 构造全零 dummy 批次
        """Create a dummy batch for initialization."""
        model_config = self.runner.model  # 取得模型配置
        return create_dummy_batch_from_config(  # 调用模块级函数，按配置生成全零批次
            hash_config=model_config.hash_config,  # 传入哈希配置
            history_len=model_config.history_seq_len,  # 传入历史序列长度
            num_candidates=model_config.candidate_seq_len,  # 传入候选序列长度
            num_actions=self._get_num_actions(),  # 传入动作数量
            batch_size=batch_size,  # 传入批次大小
        )

    def create_dummy_embeddings(self, batch_size: int = 1) -> RecsysEmbeddings:  # 构造全零 dummy 嵌入
        """Create dummy embeddings for initialization."""
        model_config = self.runner.model  # 取得模型配置
        return create_dummy_embeddings_from_config(  # 调用模块级函数，按配置生成全零嵌入
            hash_config=model_config.hash_config,  # 传入哈希配置
            emb_size=model_config.emb_size,  # 传入嵌入维度
            history_len=model_config.history_seq_len,  # 传入历史长度
            num_candidates=model_config.candidate_seq_len,  # 传入候选长度
            batch_size=batch_size,  # 传入批次大小
        )

    @abstractmethod  # 抽象方法，子类必须实现
    def initialize(self):  # 初始化推理运行器
        """Initialize the inference runner. Must be implemented by subclasses."""
        pass  # 占位


ACTIONS: List[str] = [  # 定义动作列表，描述模型输出的 19 种用户行为/参与类型的业务含义
    "favorite_score",  # 收藏/点赞动作的预测分数
    "reply_score",  # 回复动作的预测分数
    "repost_score",  # 转发动作的预测分数
    "photo_expand_score",  # 图片展开动作的预测分数
    "click_score",  # 点击动作的预测分数
    "profile_click_score",  # 点击进入作者主页动作的预测分数
    "vqv_score",  # 视频质量投票（vqv）动作的预测分数
    "share_score",  # 分享动作的预测分数
    "share_via_dm_score",  # 通过私信分享动作的预测分数
    "share_via_copy_link_score",  # 通过复制链接分享动作的预测分数
    "dwell_score",  # 停留（浏览时长达标）动作的预测分数
    "quote_score",  # 引用转发动作的预测分数
    "quoted_click_score",  # 点击被引用内容动作的预测分数
    "follow_author_score",  # 关注作者动作的预测分数
    "not_interested_score",  # 标记不感兴趣动作的预测分数
    "block_author_score",  # 屏蔽作者动作的预测分数
    "mute_author_score",  # 静音作者动作的预测分数
    "report_score",  # 举报动作的预测分数
    "dwell_time",  # 停留时长（回归目标，非概率）
]


class RankingOutput(NamedTuple):  # 定义排序输出的命名元组，字段不可变、可按位置或名字访问
    """Output from ranking candidates.

    Contains both the raw scores array and individual probability fields
    for each engagement type.
    """

    scores: jax.Array  # 全部动作的概率矩阵，形状 (B, 候选数, 19)

    ranked_indices: jax.Array  # 按主分数降序排列后的候选索引，形状 (B, 候选数)

    p_favorite_score: jax.Array  # 收藏/点赞概率，形状 (B, 候选数)
    p_reply_score: jax.Array  # 回复概率
    p_repost_score: jax.Array  # 转发概率
    p_photo_expand_score: jax.Array  # 图片展开概率
    p_click_score: jax.Array  # 点击概率
    p_profile_click_score: jax.Array  # 主页点击概率
    p_vqv_score: jax.Array  # 视频质量投票概率
    p_share_score: jax.Array  # 分享概率
    p_share_via_dm_score: jax.Array  # 私信分享概率
    p_share_via_copy_link_score: jax.Array  # 复制链接分享概率
    p_dwell_score: jax.Array  # 停留概率
    p_quote_score: jax.Array  # 引用概率
    p_quoted_click_score: jax.Array  # 被引用内容点击概率
    p_follow_author_score: jax.Array  # 关注作者概率
    p_not_interested_score: jax.Array  # 不感兴趣概率
    p_block_author_score: jax.Array  # 屏蔽作者概率
    p_mute_author_score: jax.Array  # 静音作者概率
    p_report_score: jax.Array  # 举报概率
    p_dwell_time: jax.Array  # 停留时长回归值


@dataclass  # 数据类装饰器
class ModelRunner(BaseModelRunner):  # 排序模型运行器，继承 BaseModelRunner
    """Runner for the recommendation ranking model."""

    _model: PhoenixModelConfig = None  # type: ignore  # 私有模型配置字段，默认 None，实例化时通过构造注入

    def __init__(self, model: PhoenixModelConfig, bs_per_device: float = 2.0, rng_seed: int = 42):  # 构造函数：接收模型配置、每设备 batch 因子、随机种子
        self._model = model  # 保存模型配置到私有字段
        self.bs_per_device = bs_per_device  # 保存每设备 batch 因子
        self.rng_seed = rng_seed  # 保存随机种子

    @property  # 属性装饰器
    def model(self) -> PhoenixModelConfig:  # 返回模型配置（实现基类的抽象属性）
        return self._model  # 返回私有配置字段

    @property  # 属性装饰器
    def _model_name(self) -> str:  # 覆盖基类方法，返回模型名称
        return "ranking model"  # 返回 "ranking model" 用于日志

    def make_forward_fn(self):  # type: ignore  # 创建前向函数并用 hk.transform 转换成纯函数式
        def forward(batch: RecsysBatch, recsys_embeddings: RecsysEmbeddings):  # 定义内部前向函数，接收批次与嵌入
            out = self.model.make()(batch, recsys_embeddings)  # 实例化 Haiku 排序模型并调用，得到模型输出
            return out  # 返回模型输出

        return hk.transform(forward)  # 把含状态的前向函数转换为无状态的 (init, apply) 纯函数对

    def init(  # 定义参数初始化方法
        self, rng: jax.Array, data: RecsysBatch, embeddings: RecsysEmbeddings  # 接收随机 key、dummy 批次、dummy 嵌入
    ) -> TrainingState:  # 返回训练状态
        assert self.forward is not None  # 断言前向函数已构建（initialize 已调用）
        rng, init_rng = jax.random.split(rng)  # 分裂随机 key：原 key 继续用，新 key 用于参数初始化
        params = self.forward.init(init_rng, data, embeddings)  # 用 dummy 数据调用 init 触发 JAX 追踪，确定并初始化参数形状
        return TrainingState(params=params)  # 将参数封装为 TrainingState 返回

    def load_or_init(  # 定义“加载或初始化”方法（当前实现总是重新初始化）
        self,
        init_data: RecsysBatch,  # 用于初始化的 dummy 批次
        init_embeddings: RecsysEmbeddings,  # 用于初始化的 dummy 嵌入
    ):
        rng = jax.random.PRNGKey(self.rng_seed)  # 根据固定种子创建 PRNGKey，保证初始化可复现
        state = self.init(rng, init_data, init_embeddings)  # 调用 init 完成参数初始化
        return state  # 返回训练状态


@dataclass  # 数据类装饰器
class RecsysInferenceRunner(BaseInferenceRunner):  # 排序推理运行器，继承 BaseInferenceRunner
    """Inference runner for the recommendation ranking model."""

    _runner: ModelRunner  # 底层排序模型运行器字段

    def __init__(self, runner: ModelRunner, name: str):  # 构造函数：接收底层运行器与名称
        self.name = name  # 保存推理器名称
        self._runner = runner  # 保存底层运行器

    @property  # 属性装饰器
    def runner(self) -> ModelRunner:  # 返回底层运行器（实现基类抽象属性）
        return self._runner  # 返回私有字段

    def initialize(self):  # 初始化推理运行器：构造 dummy 数据、初始化参数、编译推理函数
        """Initialize the inference runner."""
        runner = self.runner  # 取得底层模型运行器

        dummy_batch = self.create_dummy_batch(batch_size=1)  # 构造全零 dummy 批次（1 个样本即可确定形状）
        dummy_embeddings = self.create_dummy_embeddings(batch_size=1)  # 构造全零 dummy 嵌入

        runner.initialize()  # 调用底层运行器初始化：设置 bfloat16、计算 batch size、构建前向函数

        state = runner.load_or_init(dummy_batch, dummy_embeddings)  # 用 dummy 数据初始化模型参数
        self.params = state.params  # 保存初始化得到的参数，供后续推理复用

        @functools.lru_cache  # 用 lru_cache 缓存模型实例，避免重复实例化（Haiku 模块需保持单例以复用参数）
        def model():  # 定义返回模型实例的缓存函数
            return runner.model.make()  # 实例化 Haiku 排序模型并返回

        def hk_forward(  # 定义前向推理辅助函数
            batch: RecsysBatch, recsys_embeddings: RecsysEmbeddings  # 接收批次与嵌入
        ) -> RecsysModelOutput:  # 返回模型输出
            return model()(batch, recsys_embeddings)  # 调用缓存的模型实例计算前向输出

        def hk_rank_candidates(  # 定义候选排序辅助函数
            batch: RecsysBatch, recsys_embeddings: RecsysEmbeddings  # 接收批次与嵌入
        ) -> RankingOutput:  # 返回排序输出
            """Rank candidates by their predicted engagement scores."""
            output = hk_forward(batch, recsys_embeddings)  # 先执行前向得到模型输出
            logits = output.logits  # 取出 logits（未经激活的原始分数），形状 (B, 候选数, 19)

            probs = jax.nn.sigmoid(logits)  # 用 sigmoid 把 logits 压到 [0,1]，得到每个动作的独立概率

            primary_scores = probs[:, :, 0]  # 取第 0 个动作（favorite_score）作为主排序分数，形状 (B, 候选数)

            ranked_indices = jnp.argsort(-primary_scores, axis=-1)  # 对主分数取负后升序排序，等价于按分数降序，返回候选索引

            return RankingOutput(  # 构造排序输出命名元组
                scores=probs,  # 完整概率矩阵 (B, 候选数, 19)
                ranked_indices=ranked_indices,  # 按主分数降序的索引 (B, 候选数)
                p_favorite_score=probs[:, :, 0],  # 拆出收藏概率
                p_reply_score=probs[:, :, 1],  # 拆出回复概率
                p_repost_score=probs[:, :, 2],  # 拆出转发概率
                p_photo_expand_score=probs[:, :, 3],  # 拆出图片展开概率
                p_click_score=probs[:, :, 4],  # 拆出点击概率
                p_profile_click_score=probs[:, :, 5],  # 拆出主页点击概率
                p_vqv_score=probs[:, :, 6],  # 拆出视频质量投票概率
                p_share_score=probs[:, :, 7],  # 拆出分享概率
                p_share_via_dm_score=probs[:, :, 8],  # 拆出私信分享概率
                p_share_via_copy_link_score=probs[:, :, 9],  # 拆出复制链接分享概率
                p_dwell_score=probs[:, :, 10],  # 拆出停留概率
                p_quote_score=probs[:, :, 11],  # 拆出引用概率
                p_quoted_click_score=probs[:, :, 12],  # 拆出被引用内容点击概率
                p_follow_author_score=probs[:, :, 13],  # 拆出关注作者概率
                p_not_interested_score=probs[:, :, 14],  # 拆出不感兴趣概率
                p_block_author_score=probs[:, :, 15],  # 拆出屏蔽作者概率
                p_mute_author_score=probs[:, :, 16],  # 拆出静音作者概率
                p_report_score=probs[:, :, 17],  # 拆出举报概率
                p_dwell_time=probs[:, :, 18],  # 拆出停留时长回归值
            )

        rank_ = hk.without_apply_rng(hk.transform(hk_rank_candidates))  # 将排序函数转换为纯函数并移除 apply 时的 rng 参数
        self.rank_candidates = rank_.apply  # 保存 apply 函数为实例方法，供 rank 调用

    def rank(self, batch: RecsysBatch, recsys_embeddings: RecsysEmbeddings) -> RankingOutput:  # 对外排序接口
        """Rank candidates for the given batch.

        Args:
            batch: RecsysBatch containing hashes, actions, product surfaces
            recsys_embeddings: RecsysEmbeddings containing pre-looked-up embeddings

        Returns:
            RankingOutput with scores and ranked indices
        """
        return self.rank_candidates(self.params, batch, recsys_embeddings)  # 调用编译后的 apply 执行推理：参数 + 批次 + 嵌入


def create_example_batch(  # 定义函数：构造一个随机示例批次，用于测试排序/检索模型的前向与推理流程
    batch_size: int,  # 批次大小 B，决定本批包含多少个用户/请求
    emb_size: int,  # 嵌入向量维度 D
    history_len: int,  # 历史序列长度，模拟用户过去浏览过的帖子数量
    num_candidates: int,  # 候选数量 C，本次需要打分的候选帖子数
    num_actions: int,  # 动作类型数量（与 ACTIONS 列表长度一致，通常为 19）
    num_user_hashes: int = 2,  # 每个用户对应的哈希数量，默认 2（用户可由多个哈希共同标识）
    num_item_hashes: int = 2,  # 每个帖子对应的哈希数量，默认 2
    num_author_hashes: int = 2,  # 每个作者对应的哈希数量，默认 2
    product_surface_vocab_size: int = 16,  # 产品表面（product surface）词表大小，默认 16
    num_user_embeddings: int = 100000,  # 用户嵌入表规模，默认 10 万，作为用户哈希取值上限
    num_post_embeddings: int = 100000,  # 帖子嵌入表规模，默认 10 万，作为帖子哈希取值上限
    num_author_embeddings: int = 100000,  # 作者嵌入表规模，默认 10 万，作为作者哈希取值上限
) -> Tuple[RecsysBatch, RecsysEmbeddings]:  # 返回 (RecsysBatch, RecsysEmbeddings) 二元组
    """Create an example batch with random data for testing.

    This simulates a recommendation scenario where:
    - We have a user with some embedding
    - The user has interacted with some posts in their history
    - We want to rank a set of candidate posts

    Note on embedding table sizes:
        The num_*_embeddings parameters define the size of the embedding tables for each
        entity type. Hash values are generated in the range [1, num_*_embeddings) to ensure
        they can be used as valid indices into the corresponding embedding tables.
        Hash value 0 is reserved for padding/invalid entries.

    Returns:
        Tuple of (RecsysBatch, RecsysEmbeddings)
    """
    rng = np.random.default_rng(42)  # 用固定种子 42 创建 numpy 随机数生成器，保证每次生成的测试数据一致、可复现

    # 生成用户哈希：在 [1, num_user_embeddings) 区间随机取整数（避开 0 以留作 padding），形状 (B, 用户哈希数)
    user_hashes = rng.integers(1, num_user_embeddings, size=(batch_size, num_user_hashes)).astype(
        np.int32  # 转为 int32，与模型 embedding 查找所要求的索引 dtype 保持一致
    )

    # 生成历史帖子哈希：形状 (B, 历史长度, 帖子哈希数)，取值范围 [1, num_post_embeddings)
    history_post_hashes = rng.integers(
        1, num_post_embeddings, size=(batch_size, history_len, num_item_hashes)
    ).astype(np.int32)  # 转为 int32，作为帖子嵌入表的索引

    for b in range(batch_size):  # 遍历批次中的每个用户，为其历史序列设置变长有效长度
        valid_len = rng.integers(history_len // 2, history_len + 1)  # 随机抽取有效历史长度（介于历史长度一半与全长之间）
        history_post_hashes[b, valid_len:, :] = 0  # 将有效长度之后的帖子哈希置 0，模拟 padding/无效位置

    # 生成历史作者哈希：形状 (B, 历史长度, 作者哈希数)，取值范围 [1, num_author_embeddings)
    history_author_hashes = rng.integers(
        1, num_author_embeddings, size=(batch_size, history_len, num_author_hashes)
    ).astype(np.int32)  # 转为 int32，作为作者嵌入表的索引
    for b in range(batch_size):  # 遍历每个用户，为历史作者哈希设置变长有效长度
        valid_len = rng.integers(history_len // 2, history_len + 1)  # 随机抽取有效历史长度
        history_author_hashes[b, valid_len:, :] = 0  # 将有效长度之后的作者哈希置 0，作为 padding

    # 生成历史动作标签：形状 (B, 历史长度, 动作数)，用随机数与阈值 0.7 比较得到 0/1 二值标签
    history_actions = (rng.random(size=(batch_size, history_len, num_actions)) > 0.7).astype(
        np.float32  # 转为 float32，与模型输入的动作特征 dtype 一致
    )

    # 生成历史产品表面特征：形状 (B, 历史长度)，取值范围 [0, product_surface_vocab_size)
    history_product_surface = rng.integers(
        0, product_surface_vocab_size, size=(batch_size, history_len)
    ).astype(np.int32)  # 转为 int32，作为产品表面类别索引

    # 生成候选帖子哈希：形状 (B, 候选数, 帖子哈希数)，取值范围 [1, num_post_embeddings)
    candidate_post_hashes = rng.integers(
        1, num_post_embeddings, size=(batch_size, num_candidates, num_item_hashes)
    ).astype(np.int32)  # 转为 int32，作为候选帖子的嵌入表索引

    # 生成候选作者哈希：形状 (B, 候选数, 作者哈希数)，取值范围 [1, num_author_embeddings)
    candidate_author_hashes = rng.integers(
        1, num_author_embeddings, size=(batch_size, num_candidates, num_author_hashes)
    ).astype(np.int32)  # 转为 int32，作为候选作者的嵌入表索引

    # 生成候选产品表面特征：形状 (B, 候选数)，取值范围 [0, product_surface_vocab_size)
    candidate_product_surface = rng.integers(
        0, product_surface_vocab_size, size=(batch_size, num_candidates)
    ).astype(np.int32)  # 转为 int32，作为候选产品表面类别索引

    batch = RecsysBatch(  # 用上述随机生成的特征组装 RecsysBatch 数据对象
        user_hashes=user_hashes,  # 传入用户哈希
        history_post_hashes=history_post_hashes,  # 传入历史帖子哈希
        history_author_hashes=history_author_hashes,  # 传入历史作者哈希
        history_actions=history_actions,  # 传入历史动作标签
        history_product_surface=history_product_surface,  # 传入历史产品表面特征
        candidate_post_hashes=candidate_post_hashes,  # 传入候选帖子哈希
        candidate_author_hashes=candidate_author_hashes,  # 传入候选作者哈希
        candidate_product_surface=candidate_product_surface,  # 传入候选产品表面特征
    )

    embeddings = RecsysEmbeddings(  # 用随机正态分布生成嵌入，组装 RecsysEmbeddings 数据对象
        user_embeddings=rng.normal(size=(batch_size, num_user_hashes, emb_size)).astype(np.float32),  # 用户嵌入 (B, 用户哈希数, D)，转为 float32
        history_post_embeddings=rng.normal(  # 历史帖子嵌入 (B, 历史长度, 帖子哈希数, D)
            size=(batch_size, history_len, num_item_hashes, emb_size)
        ).astype(np.float32),  # 转为 float32
        candidate_post_embeddings=rng.normal(  # 候选帖子嵌入 (B, 候选数, 帖子哈希数, D)
            size=(batch_size, num_candidates, num_item_hashes, emb_size)
        ).astype(np.float32),  # 转为 float32
        history_author_embeddings=rng.normal(  # 历史作者嵌入 (B, 历史长度, 作者哈希数, D)
            size=(batch_size, history_len, num_author_hashes, emb_size)
        ).astype(np.float32),  # 转为 float32
        candidate_author_embeddings=rng.normal(  # 候选作者嵌入 (B, 候选数, 作者哈希数, D)
            size=(batch_size, num_candidates, num_author_hashes, emb_size)
        ).astype(np.float32),  # 转为 float32
    )

    return batch, embeddings  # 返回构造好的批次与嵌入二元组


class RetrievalOutput(NamedTuple):  # 定义检索输出的命名元组，字段不可变、可按键名访问
    """Output from retrieval inference.

    Contains user representations and retrieved candidates.
    """

    user_representation: jax.Array  # 用户表征向量，形状 (B, D)

    top_k_indices: jax.Array  # 检索到的 top-k 候选在语料中的索引，形状 (B, top_k)

    top_k_scores: jax.Array  # 检索到的 top-k 候选的相似度分数，形状 (B, top_k)


@dataclass  # 数据类装饰器，自动生成 __init__ 等基础方法
class RetrievalModelRunner(BaseModelRunner):  # 检索模型运行器，继承 BaseModelRunner 共享初始化逻辑
    """Runner for the Phoenix retrieval model."""

    _model: PhoenixRetrievalModelConfig = None  # type: ignore  # 私有检索模型配置字段，默认 None，构造时注入

    def __init__(  # 构造函数：接收检索模型配置、每设备 batch 因子、随机种子
        self,
        model: PhoenixRetrievalModelConfig,  # 检索模型配置对象
        bs_per_device: float = 2.0,  # 每设备 batch 因子，默认 2.0
        rng_seed: int = 42,  # 随机种子，默认 42
    ):
        self._model = model  # 保存模型配置到私有字段
        self.bs_per_device = bs_per_device  # 保存每设备 batch 因子
        self.rng_seed = rng_seed  # 保存随机种子

    @property  # 属性装饰器
    def model(self) -> PhoenixRetrievalModelConfig:  # 返回模型配置（实现基类抽象属性）
        return self._model  # 返回私有配置字段

    @property  # 属性装饰器
    def _model_name(self) -> str:  # 覆盖基类方法，返回模型名称
        return "retrieval model"  # 返回 "retrieval model" 用于日志输出

    def make_forward_fn(self):  # type: ignore  # 创建前向函数并用 hk.transform 转成纯函数式
        def forward(  # 定义内部前向函数，接收批次、嵌入、语料嵌入与 top_k
            batch: RecsysBatch,  # 批次数据
            recsys_embeddings: RecsysEmbeddings,  # 预查好的嵌入
            corpus_embeddings: jax.Array,  # 语料候选嵌入矩阵 (N, D)
            top_k: int,  # 每个用户检索的候选数量
        ) -> ModelRetrievalOutput:  # 返回检索模型输出
            model = self.model.make()  # 实例化 Haiku 检索模型
            out = model(batch, recsys_embeddings, corpus_embeddings, top_k)  # 调用模型做检索前向，得到输出

            _ = model.build_candidate_representation(batch, recsys_embeddings)  # 显式构建候选表征，确保其参数在 init 时被追踪注册
            return out  # 返回检索输出

        return hk.transform(forward)  # 将含状态前向函数转换为 (init, apply) 纯函数对

    def init(  # 定义检索模型参数初始化方法
        self,
        rng: jax.Array,  # 随机 key
        data: RecsysBatch,  # dummy 批次
        embeddings: RecsysEmbeddings,  # dummy 嵌入
        corpus_embeddings: jax.Array,  # dummy 语料嵌入
        top_k: int,  # dummy top_k
    ) -> TrainingState:  # 返回训练状态
        assert self.forward is not None  # 断言前向函数已构建（initialize 已调用）
        rng, init_rng = jax.random.split(rng)  # 分裂随机 key，新 key 用于参数初始化
        params = self.forward.init(init_rng, data, embeddings, corpus_embeddings, top_k)  # 用 dummy 数据触发 JAX 追踪并初始化参数
        return TrainingState(params=params)  # 将参数封装为 TrainingState 返回

    def load_or_init(  # 定义“加载或初始化”方法（当前实现总是重新初始化）
        self,
        init_data: RecsysBatch,  # dummy 批次
        init_embeddings: RecsysEmbeddings,  # dummy 嵌入
        corpus_embeddings: jax.Array,  # dummy 语料嵌入
        top_k: int,  # dummy top_k
    ):
        rng = jax.random.PRNGKey(self.rng_seed)  # 用固定种子创建 PRNGKey，保证初始化可复现
        state = self.init(rng, init_data, init_embeddings, corpus_embeddings, top_k)  # 调用 init 完成参数初始化
        return state  # 返回训练状态


@dataclass  # 数据类装饰器
class RecsysRetrievalInferenceRunner(BaseInferenceRunner):  # 检索推理运行器，继承 BaseInferenceRunner
    """Inference runner for the Phoenix retrieval model.

    This runner provides methods for:
    1. Encoding users to get user representations
    2. Encoding candidates to get candidate embeddings
    3. Retrieving top-k candidates from a corpus
    """

    _runner: RetrievalModelRunner = None  # type: ignore  # 底层检索模型运行器字段，默认 None，构造时注入

    corpus_embeddings: jax.Array | None = None  # 语料嵌入缓存字段，可空，检索时作为候选集合使用
    corpus_post_ids: jax.Array | None = None  # 语料对应的帖子 ID 缓存字段，可空，便于映射回业务实体

    def __init__(self, runner: RetrievalModelRunner, name: str):  # 构造函数：接收底层运行器与名称
        self.name = name  # 保存推理器名称
        self._runner = runner  # 保存底层运行器
        self.corpus_embeddings = None  # 初始化语料嵌入为 None
        self.corpus_post_ids = None  # 初始化语料帖子 ID 为 None

    @property  # 属性装饰器
    def runner(self) -> RetrievalModelRunner:  # 返回底层运行器（实现基类抽象属性）
        return self._runner  # 返回私有字段

    def initialize(self):  # 初始化检索推理器：构造 dummy 数据、初始化参数、编译 encode/retrieve 函数
        """Initialize the retrieval inference runner."""
        runner = self.runner  # 取得底层模型运行器

        dummy_batch = self.create_dummy_batch(batch_size=1)  # 构造全零 dummy 批次（1 个样本即可确定形状）
        dummy_embeddings = self.create_dummy_embeddings(batch_size=1)  # 构造全零 dummy 嵌入
        dummy_corpus = jnp.zeros((10, runner.model.emb_size), dtype=jnp.float32)  # 构造 10 条全零 dummy 语料嵌入 (10, D)，用于追踪确定语料形状
        dummy_top_k = 5  # dummy top_k 设为 5，用于追踪确定检索输出形状

        runner.initialize()  # 调用底层运行器初始化：设置 bfloat16、计算 batch size、构建前向函数

        state = runner.load_or_init(dummy_batch, dummy_embeddings, dummy_corpus, dummy_top_k)  # 用 dummy 数据初始化检索模型参数
        self.params = state.params  # 保存初始化得到的参数，供后续推理复用

        @functools.lru_cache  # 用 lru_cache 缓存模型实例，避免重复实例化（Haiku 模块需保持单例以复用参数）
        def model():  # 定义返回模型实例的缓存函数
            return runner.model.make()  # 实例化 Haiku 检索模型并返回

        def hk_encode_user(batch: RecsysBatch, recsys_embeddings: RecsysEmbeddings) -> jax.Array:  # 定义用户编码辅助函数
            """Encode user to get user representation."""
            m = model()  # 取得缓存的模型实例
            user_rep, _ = m.build_user_representation(batch, recsys_embeddings)  # 构建用户表征，忽略第二个返回值
            return user_rep  # 返回用户表征 (B, D)

        def hk_encode_candidates(  # 定义候选编码辅助函数
            batch: RecsysBatch, recsys_embeddings: RecsysEmbeddings  # 接收批次与嵌入
        ) -> jax.Array:  # 返回候选表征数组
            """Encode candidates to get candidate representations."""
            m = model()  # 取得缓存的模型实例
            cand_rep, _ = m.build_candidate_representation(batch, recsys_embeddings)  # 构建候选表征，忽略第二个返回值
            return cand_rep  # 返回候选表征 (B, C, D)

        def hk_retrieve(  # 定义检索辅助函数
            batch: RecsysBatch,  # 批次数据
            recsys_embeddings: RecsysEmbeddings,  # 预查好的嵌入
            corpus_embeddings: jax.Array,  # 语料嵌入矩阵 (N, D)
            top_k: int,  # 每个用户检索数量
        ) -> "RetrievalOutput":  # 返回检索输出
            """Retrieve top-k candidates from corpus."""
            m = model()  # 取得缓存的模型实例
            return m(batch, recsys_embeddings, corpus_embeddings, top_k)  # 调用模型完成检索，返回 RetrievalOutput

        encode_user_ = hk.without_apply_rng(hk.transform(hk_encode_user))  # 将用户编码函数转为纯函数并移除 apply 时的 rng
        encode_candidates_ = hk.without_apply_rng(hk.transform(hk_encode_candidates))  # 将候选编码函数转为纯函数并移除 rng
        retrieve_ = hk.without_apply_rng(hk.transform(hk_retrieve))  # 将检索函数转为纯函数并移除 rng

        self.encode_user_fn = encode_user_.apply  # 保存用户编码 apply 函数为实例方法
        self.encode_candidates_fn = encode_candidates_.apply  # 保存候选编码 apply 函数为实例方法
        self.retrieve_fn = retrieve_.apply  # 保存检索 apply 函数为实例方法

    def encode_user(self, batch: RecsysBatch, recsys_embeddings: RecsysEmbeddings) -> jax.Array:  # 对外用户编码接口
        """Encode users to get user representations.

        Args:
            batch: RecsysBatch containing user and history information
            recsys_embeddings: RecsysEmbeddings containing pre-looked-up embeddings

        Returns:
            User representations [B, D]
        """
        return self.encode_user_fn(self.params, batch, recsys_embeddings)  # 调用编译后的 apply 计算用户表征

    def encode_candidates(  # 对外候选编码接口
        self, batch: RecsysBatch, recsys_embeddings: RecsysEmbeddings  # 接收批次与嵌入
    ) -> jax.Array:  # 返回候选表征
        """Encode candidates to get candidate representations.

        Args:
            batch: RecsysBatch containing candidate information
            recsys_embeddings: RecsysEmbeddings containing pre-looked-up embeddings

        Returns:
            Candidate representations [B, C, D]
        """
        return self.encode_candidates_fn(self.params, batch, recsys_embeddings)  # 调用编译后的 apply 计算候选表征

    def set_corpus(  # 设置检索语料的方法
        self,
        corpus_embeddings: jax.Array,  # 预计算的候选嵌入矩阵 (N, D)
        corpus_post_ids: jax.Array,  # 与嵌入对应的帖子 ID 数组 (N)
    ):
        """Set the corpus embeddings for retrieval.

        Args:
            corpus_embeddings: Pre-computed candidate embeddings [N, D]
            corpus_post_ids: Optional post IDs corresponding to embeddings [N]
        """
        self.corpus_embeddings = corpus_embeddings  # 缓存语料嵌入，供 retrieve 在未显式传入时使用
        self.corpus_post_ids = corpus_post_ids  # 缓存语料帖子 ID，便于后续映射回业务实体

    def retrieve(  # 对外检索接口
        self,
        batch: RecsysBatch,  # 批次数据
        recsys_embeddings: RecsysEmbeddings,  # 预查好的嵌入
        top_k: int = 100,  # 每个用户检索候选数，默认 100
        corpus_embeddings: Optional[jax.Array] = None,  # 可选语料嵌入，未传则回退到缓存的 corpus
    ) -> RetrievalOutput:  # 返回检索输出
        """Retrieve top-k candidates for users.

        Args:
            batch: RecsysBatch containing user and history information
            recsys_embeddings: RecsysEmbeddings containing pre-looked-up embeddings
            top_k: Number of candidates to retrieve per user
            corpus_embeddings: Optional corpus embeddings (uses set_corpus if not provided)

        Returns:
            RetrievalOutput with user representations and top-k candidates
        """
        if corpus_embeddings is None:  # 判断是否显式传入语料嵌入
            corpus_embeddings = self.corpus_embeddings  # 未传入则回退到缓存的语料嵌入

        return self.retrieve_fn(self.params, batch, recsys_embeddings, corpus_embeddings, top_k)  # 调用编译后的 apply 执行检索


def create_example_corpus(  # 定义函数：构造示例语料嵌入，用于测试检索流程
    corpus_size: int,  # 语料中的候选数量 N
    emb_size: int,  # 嵌入向量维度 D
    seed: int = 123,  # 随机种子，默认 123，保证数据可复现
) -> Tuple[jax.Array, jax.Array]:  # 返回 (语料嵌入, 帖子 ID) 二元组
    """Create example corpus embeddings for testing retrieval.

    Args:
        corpus_size: Number of candidates in corpus
        emb_size: Embedding dimension
        seed: Random seed

    Returns:
        Tuple of (corpus_embeddings [N, D], corpus_post_ids [N])
    """
    rng = np.random.default_rng(seed)  # 用指定种子创建 numpy 随机数生成器

    corpus_embeddings = rng.normal(size=(corpus_size, emb_size)).astype(np.float32)  # 生成标准正态分布语料嵌入 (N, D) 并转为 float32
    norms = np.linalg.norm(corpus_embeddings, axis=-1, keepdims=True)  # 计算每个嵌入向量的 L2 范数 (N, 1)，用于后续归一化
    corpus_embeddings = corpus_embeddings / np.maximum(norms, 1e-12)  # 逐向量除以范数做 L2 归一化，1e-12 防止除零

    corpus_post_ids = np.arange(corpus_size, dtype=np.int64)  # 生成 0..N-1 的帖子 ID 数组 (N)

    return jnp.array(corpus_embeddings), jnp.array(corpus_post_ids)  # 转成 JAX 数组并返回
