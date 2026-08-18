# Copyright 2026 X.AI Corp.
# 以下为 Apache 2.0 开源许可证的版权声明，用于声明本文件的开源协议
#
# Licensed under the Apache License, Version 2.0 (the "License");
# 声明本文件遵循 Apache License 2.0 开源协议
# you may not use this file except in compliance with the License.
# 除非符合许可证要求，否则不得使用本文件
# You may obtain a copy of the License at
# 完整许可证可在以下网址获取
#
#     http://www.apache.org/licenses/LICENSE-2.0
# 指向 Apache 2.0 许可证全文的 URL
#
# Unless required by applicable law or agreed to in writing, software
# 除非适用法律要求或书面同意，否则依"现状"分发本软件
# distributed under the License is distributed on an "AS IS" BASIS,
# 不提供任何明示或暗示的担保
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# 既不担保可商用，也不担保不侵权
# See the License for the specific language governing permissions and
# 具体权限与限制请参见许可证原文
# limitations under the License.

import logging  # 导入标准日志库，用于记录模型运行时的告警与调试信息
from dataclasses import dataclass  # 导入 dataclass 装饰器，用于以简洁语法定义配置/数据容器类
from typing import Any, NamedTuple, Optional, Tuple  # 导入类型注解工具，用于声明可空、元组、命名元组等类型

import haiku as hk  # 导入 Haiku 库并取别名 hk，Haiku 是 JAX 上用于构建可训练模块的函数式神经网络库
import jax  # 导入 JAX 主库，提供函数式自动微分、jit 编译与设备数组支持
import jax.numpy as jnp  # 导入 JAX 的 NumPy 兼容 API 并取别名 jnp，用于张量运算

from grok import (  # 从本地 grok 模块导入 Grok-1 Transformer 的核心组件，用于复用其主干网络
    TransformerConfig,  # Transformer 主干的配置类（层数、维度、头数等）
    Transformer,  # Grok-1 Transformer 主干模块，处理序列建模
    layer_norm,  # 层归一化函数，用于对输出嵌入做归一化
)

logger = logging.getLogger(__name__)  # 创建以当前模块名命名的日志器，便于在日志中定位本文件输出


@dataclass  # 使用 dataclass 装饰器，自动生成 __init__、__repr__ 等方法，简化配置类定义
class HashConfig:  # 定义哈希 embedding 配置类，控制用户/物品/作者各自使用多少个哈希函数查表
    """Configuration for hash-based embeddings."""  # 原有英文 docstring，说明本类用于配置基于哈希的 embedding

    num_user_hashes: int = 2  # 用户侧使用的哈希函数个数，默认 2；多哈希可降低哈希冲突并丰富用户表示
    num_item_hashes: int = 2  # 物品（推文）侧使用的哈希函数个数，默认 2；用于生成推文的多哈希 embedding
    num_author_hashes: int = 2  # 作者侧使用的哈希函数个数，默认 2；用于生成作者的多哈希 embedding


@dataclass  # 使用 dataclass 装饰器，定义一个纯数据容器类，存放预先查表得到的各类 embedding
class RecsysEmbeddings:  # 定义推荐 embedding 容器类，封装所有"提前查表"得到的 embedding 张量
    """Container for pre-looked-up embeddings from the embedding tables.  # 原有英文 docstring：存放从 embedding 表中预先查得的张量

    These embeddings are looked up from hash tables before being passed to the model.  # 这些 embedding 在送入模型前已从哈希表查得
    The block_*_reduce functions will combine multiple hash embeddings into single representations.  # block_*_reduce 系列函数会把多哈希 embedding 融合成单一表示
    """

    user_embeddings: jax.typing.ArrayLike  # 用户 embedding 张量，形状 [B, num_user_hashes, D]，来自用户哈希查表
    history_post_embeddings: jax.typing.ArrayLike  # 历史推文 embedding，形状 [B, S, num_item_hashes, D]，S 为历史序列长度
    candidate_post_embeddings: jax.typing.ArrayLike  # 候选推文 embedding，形状 [B, C, num_item_hashes, D]，C 为候选数量
    history_author_embeddings: jax.typing.ArrayLike  # 历史推文作者 embedding，形状 [B, S, num_author_hashes, D]
    candidate_author_embeddings: jax.typing.ArrayLike  # 候选推文作者 embedding，形状 [B, C, num_author_hashes, D]


class RecsysModelOutput(NamedTuple):  # 使用 NamedTuple 定义模型输出结构，便于以字段名访问且保持不可变性
    """Output of the recommendation model."""  # 原有英文 docstring：推荐模型的输出

    logits: jax.Array  # 模型输出的预测 logits，形状 [B, C, num_actions]，每个候选对应多个动作的得分


class RecsysBatch(NamedTuple):  # 使用 NamedTuple 定义输入批次结构，封装送入模型的特征数据（不含 embedding）
    """Input batch for the recommendation model.  # 原有英文 docstring：推荐模型的输入批次

    Contains the feature data (hashes, actions, product surfaces) but NOT the embeddings.  # 包含特征数据（哈希、动作、产品面）但不包含 embedding
    Embeddings are passed separately via RecsysEmbeddings.  # embedding 通过 RecsysEmbeddings 单独传入
    """

    user_hashes: jax.typing.ArrayLike  # 用户哈希值，形状 [B, num_user_hashes]；约定 0 表示 padding/无效
    history_post_hashes: jax.typing.ArrayLike  # 历史推文哈希，形状 [B, S, num_item_hashes]；用于生成 padding mask
    history_author_hashes: jax.typing.ArrayLike  # 历史推文作者哈希，形状 [B, S, num_author_hashes]
    history_actions: jax.typing.ArrayLike  # 历史动作多热向量，形状 [B, S, num_actions]；表示用户对历史推文的多种交互
    history_product_surface: jax.typing.ArrayLike  # 历史推文的产品面索引，形状 [B, S]；单热离散特征（如 For You/Following）
    candidate_post_hashes: jax.typing.ArrayLike  # 候选推文哈希，形状 [B, C, num_item_hashes]
    candidate_author_hashes: jax.typing.ArrayLike  # 候选推文作者哈希，形状 [B, C, num_author_hashes]
    candidate_product_surface: jax.typing.ArrayLike  # 候选推文的产品面索引，形状 [B, C]


def block_user_reduce(  # 定义用户多哈希 embedding 融合函数：把多哈希用户 embedding 压缩为单一表示
    user_hashes: jnp.ndarray,  # 用户哈希值，形状 [B, num_user_hashes]，用于判断有效性（0=padding）
    user_embeddings: jnp.ndarray,  # 用户多哈希 embedding，形状 [B, num_user_hashes, D]
    num_user_hashes: int,  # 用户哈希函数个数，决定拼接后的特征维度
    emb_size: int,  # embedding 维度 D，决定投影后输出的维度
    embed_init_scale: float = 1.0,  # 投影矩阵初始化的缩放系数，控制初始化方差
) -> Tuple[jax.Array, jax.Array]:  # 返回 (融合后用户 embedding [B,1,D], 用户 padding mask [B,1])
    """Combine multiple user hash embeddings into a single user representation.  # 原有英文 docstring：把多个用户哈希 embedding 组合成单一用户表示

    Args:
        user_hashes: [B, num_user_hashes] - hash values (0 = invalid/padding)  # 用户哈希，0 表示无效/padding
        user_embeddings: [B, num_user_hashes, D] - looked-up embeddings  # 已查表的用户多哈希 embedding
        num_user_hashes: number of hash functions used  # 使用的哈希函数个数
        emb_size: embedding dimension D  # embedding 维度
        embed_init_scale: initialization scale for projection  # 投影矩阵初始化缩放系数

    Returns:
        user_embedding: [B, 1, D] - combined user embedding  # 融合后的用户 embedding，序列长度为 1
        user_padding_mask: [B, 1] - True where user is valid  # 用户有效性掩码，True 表示有效
    """
    B = user_embeddings.shape[0]  # 取批次大小 B，表示本批次样本数
    D = emb_size  # 取 embedding 维度 D，便于后续拼接与投影计算

    # 将多哈希 embedding 在最后一维拼接：[B, num_user_hashes, D] -> [B, 1, num_user_hashes*D]
    user_embedding = user_embeddings.reshape((B, 1, num_user_hashes * D))

    # 创建 VarianceScaling 初始化器，mode="fan_out" 适配前向传播的方差保持
    embed_init = hk.initializers.VarianceScaling(embed_init_scale, mode="fan_out")
    # 获取投影矩阵参数 proj_mat_1，形状 [num_user_hashes*D, D]，把拼接特征压回 D 维
    proj_mat_1 = hk.get_parameter(
        "proj_mat_1",  # 参数名，用于在 Haiku 参数树中唯一定位
        [num_user_hashes * D, D],  # 参数形状：输入维度为拼接后维度，输出维度为 D
        dtype=jnp.float32,  # 参数精度为 float32，保证训练稳定性
        # 使用转置技巧：先对反转后的 shape 初始化再转置，使 fan_out 模式按"输出维度"计算增益
        init=lambda shape, dtype: embed_init(list(reversed(shape)), dtype).T,
    )

    # 将拼接 embedding 与投影矩阵相乘：[B, 1, num_user_hashes*D] @ [num_user_hashes*D, D] -> [B, 1, D]
    user_embedding = jnp.dot(user_embedding.astype(proj_mat_1.dtype), proj_mat_1).astype(
        user_embeddings.dtype
    )  # 投影后 cast 回原 embedding 的精度，避免精度不一致

    # hash 0 is reserved for padding)  # 原有英文注释：哈希 0 保留给 padding
    # 用第 0 个哈希是否非 0 判断用户有效性：[B] -> [B, 1]，并转为布尔类型
    user_padding_mask = (user_hashes[:, 0] != 0).reshape(B, 1).astype(jnp.bool_)

    return user_embedding, user_padding_mask  # 返回融合后用户 embedding 与对应 padding mask


def block_history_reduce(  # 定义历史序列多哈希 embedding 融合函数：把历史推文/作者/动作/产品面融合为序列表示
    history_post_hashes: jnp.ndarray,  # 历史推文哈希，形状 [B, S, num_item_hashes]；用于生成 padding mask
    history_post_embeddings: jnp.ndarray,  # 历史推文多哈希 embedding，形状 [B, S, num_item_hashes, D]
    history_author_embeddings: jnp.ndarray,  # 历史作者多哈希 embedding，形状 [B, S, num_author_hashes, D]
    history_product_surface_embeddings: jnp.ndarray,  # 历史产品面 embedding，形状 [B, S, D]
    history_actions_embeddings: jnp.ndarray,  # 历史动作 embedding，形状 [B, S, D]
    num_item_hashes: int,  # 物品哈希函数个数
    num_author_hashes: int,  # 作者哈希函数个数
    embed_init_scale: float = 1.0,  # 投影矩阵初始化缩放系数
) -> Tuple[jax.Array, jax.Array]:  # 返回 (历史序列 embedding [B,S,D], 历史 padding mask [B,S])
    """Combine history embeddings (post, author, actions, product_surface) into sequence.  # 原有英文 docstring：把历史 embedding（推文、作者、动作、产品面）融合成序列

    Args:
        history_post_hashes: [B, S, num_item_hashes]  # 历史推文哈希
        history_post_embeddings: [B, S, num_item_hashes, D]  # 历史推文多哈希 embedding
        history_author_embeddings: [B, S, num_author_hashes, D]  # 历史作者多哈希 embedding
        history_product_surface_embeddings: [B, S, D]  # 历史产品面 embedding
        history_actions_embeddings: [B, S, D]  # 历史动作 embedding
        num_item_hashes: number of hash functions for items  # 物品哈希数
        num_author_hashes: number of hash functions for authors  # 作者哈希数
        emb_size: embedding dimension D  # embedding 维度
        embed_init_scale: initialization scale  # 初始化缩放系数

    Returns:
        history_embeddings: [B, S, D]  # 融合后的历史序列 embedding
        history_padding_mask: [B, S]  # 历史有效性掩码
    """
    B, S, _, D = history_post_embeddings.shape  # 解析批次 B、历史序列长度 S、embedding 维度 D

    # 把历史推文多哈希 embedding 在最后一维拼接：[B, S, num_item_hashes, D] -> [B, S, num_item_hashes*D]
    history_post_embeddings_reshaped = history_post_embeddings.reshape((B, S, num_item_hashes * D))
    # 把历史作者多哈希 embedding 在最后一维拼接：[B, S, num_author_hashes, D] -> [B, S, num_author_hashes*D]
    history_author_embeddings_reshaped = history_author_embeddings.reshape(
        (B, S, num_author_hashes * D)
    )

    # 将推文、作者、动作、产品面 embedding 在特征维拼接：得到 [B, S, (num_item_hashes+num_author_hashes)*D + 2D]
    post_author_embedding = jnp.concatenate(
        [
            history_post_embeddings_reshaped,  # 推文多哈希拼接特征
            history_author_embeddings_reshaped,  # 作者多哈希拼接特征
            history_actions_embeddings,  # 动作 embedding（多热投影得到）
            history_product_surface_embeddings,  # 产品面 embedding（查表得到）
        ],
        axis=-1,  # 沿最后一维（特征维）拼接
    )

    # 创建 VarianceScaling 初始化器用于投影矩阵，fan_out 模式保持前向方差
    embed_init = hk.initializers.VarianceScaling(embed_init_scale, mode="fan_out")
    # 获取投影矩阵 proj_mat_3，形状 [拼接特征维, D]，把拼接特征压回 D 维
    proj_mat_3 = hk.get_parameter(
        "proj_mat_3",  # 参数名，区别于用户/候选投影矩阵
        [post_author_embedding.shape[-1], D],  # 输入维度取拼接后的特征维，输出维度为 D
        dtype=jnp.float32,  # 参数精度 float32
        # 转置技巧：对反转 shape 初始化后转置，使 fan_out 按输出维度计算
        init=lambda shape, dtype: embed_init(list(reversed(shape)), dtype).T,
    )

    # 投影：[B, S, 拼接维] @ [拼接维, D] -> [B, S, D]，并 cast 回原精度
    history_embedding = jnp.dot(post_author_embedding.astype(proj_mat_3.dtype), proj_mat_3).astype(
        post_author_embedding.dtype
    )

    # 显式 reshape 以确保形状为 [B, S, D]（投影后本应如此，这里做形状断言）
    history_embedding = history_embedding.reshape(B, S, D)

    # 用历史推文第 0 个哈希是否非 0 生成 padding mask：[B, S, num_item_hashes] -> [B, S]
    history_padding_mask = (history_post_hashes[:, :, 0] != 0).reshape(B, S)

    return history_embedding, history_padding_mask  # 返回历史序列 embedding 与 padding mask


def block_candidate_reduce(  # 定义候选多哈希 embedding 融合函数：把候选推文/作者/产品面融合为候选序列表示
    candidate_post_hashes: jnp.ndarray,  # 候选推文哈希，形状 [B, C, num_item_hashes]；用于生成 padding mask
    candidate_post_embeddings: jnp.ndarray,  # 候选推文多哈希 embedding，形状 [B, C, num_item_hashes, D]
    candidate_author_embeddings: jnp.ndarray,  # 候选作者多哈希 embedding，形状 [B, C, num_author_hashes, D]
    candidate_product_surface_embeddings: jnp.ndarray,  # 候选产品面 embedding，形状 [B, C, D]
    num_item_hashes: int,  # 物品哈希函数个数
    num_author_hashes: int,  # 作者哈希函数个数
    embed_init_scale: float = 1.0,  # 投影矩阵初始化缩放系数
) -> Tuple[jax.Array, jax.Array]:  # 返回 (候选序列 embedding [B,C,D], 候选 padding mask [B,C])
    """Combine candidate embeddings (post, author, product_surface) into sequence.  # 原有英文 docstring：把候选 embedding（推文、作者、产品面）融合成序列

    Args:
        candidate_post_hashes: [B, C, num_item_hashes]  # 候选推文哈希
        candidate_post_embeddings: [B, C, num_item_hashes, D]  # 候选推文多哈希 embedding
        candidate_author_embeddings: [B, C, num_author_hashes, D]  # 候选作者多哈希 embedding
        candidate_product_surface_embeddings: [B, C, D]  # 候选产品面 embedding
        num_item_hashes: number of hash functions for items  # 物品哈希数
        num_author_hashes: number of hash functions for authors  # 作者哈希数
        emb_size: embedding dimension D  # embedding 维度
        embed_init_scale: initialization scale  # 初始化缩放系数

    Returns:
        candidate_embeddings: [B, C, D]  # 融合后的候选序列 embedding
        candidate_padding_mask: [B, C]  # 候选有效性掩码
    """
    B, C, _, D = candidate_post_embeddings.shape  # 解析批次 B、候选数 C、embedding 维度 D

    # 候选推文多哈希拼接：[B, C, num_item_hashes, D] -> [B, C, num_item_hashes*D]
    candidate_post_embeddings_reshaped = candidate_post_embeddings.reshape(
        (B, C, num_item_hashes * D)
    )
    # 候选作者多哈希拼接：[B, C, num_author_hashes, D] -> [B, C, num_author_hashes*D]
    candidate_author_embeddings_reshaped = candidate_author_embeddings.reshape(
        (B, C, num_author_hashes * D)
    )

    # 候选推文、作者、产品面 embedding 在特征维拼接：[B, C, (num_item_hashes+num_author_hashes)*D + D]
    post_author_embedding = jnp.concatenate(
        [
            candidate_post_embeddings_reshaped,  # 候选推文多哈希拼接特征
            candidate_author_embeddings_reshaped,  # 候选作者多哈希拼接特征
            candidate_product_surface_embeddings,  # 候选产品面 embedding
        ],
        axis=-1,  # 沿最后一维（特征维）拼接
    )

    # 创建 VarianceScaling 初始化器用于候选投影矩阵
    embed_init = hk.initializers.VarianceScaling(embed_init_scale, mode="fan_out")
    # 获取投影矩阵 proj_mat_2，形状 [拼接特征维, D]，把候选拼接特征压回 D 维
    proj_mat_2 = hk.get_parameter(
        "proj_mat_2",  # 参数名，区别于用户/历史投影矩阵
        [post_author_embedding.shape[-1], D],  # 输入维度取拼接后的特征维，输出维度为 D
        dtype=jnp.float32,  # 参数精度 float32
        # 转置技巧：对反转 shape 初始化后转置，使 fan_out 按输出维度计算
        init=lambda shape, dtype: embed_init(list(reversed(shape)), dtype).T,
    )

    # 投影：[B, C, 拼接维] @ [拼接维, D] -> [B, C, D]，并 cast 回原精度
    candidate_embedding = jnp.dot(
        post_author_embedding.astype(proj_mat_2.dtype), proj_mat_2
    ).astype(post_author_embedding.dtype)

    # 用候选推文第 0 个哈希是否非 0 生成 padding mask，并转为布尔类型：[B, C, num_item_hashes] -> [B, C]
    candidate_padding_mask = (candidate_post_hashes[:, :, 0] != 0).reshape(B, C).astype(jnp.bool_)

    return candidate_embedding, candidate_padding_mask  # 返回候选序列 embedding 与 padding mask


@dataclass  # 使用 dataclass 装饰器，定义推荐模型的整体配置类，含主干配置、维度、动作数等
class PhoenixModelConfig:  # 定义 Phoenix 推荐模型配置类，聚合 Transformer 配置与推荐专用超参
    """Configuration for the recommendation system model."""  # 原有英文 docstring：推荐系统模型的配置

    model: TransformerConfig  # Grok-1 Transformer 主干的配置（层数、维度、注意力头数等）
    emb_size: int  # 推荐模型统一使用的 embedding 维度 D
    num_actions: int  # 多动作输出头的动作数量（如点赞、转发、评论等）
    history_seq_len: int = 128  # 历史序列长度 S，默认 128，表示回看用户最近 128 个交互
    candidate_seq_len: int = 32  # 候选序列长度 C，默认 32，表示每批最多对 32 个候选打分

    name: Optional[str] = None  # 模型实例名称，可选，用于日志与参数命名空间区分
    fprop_dtype: Any = jnp.bfloat16  # 前向传播使用的数据精度，默认 bfloat16 以节省显存与加速

    hash_config: HashConfig = None  # type: ignore  # 多哈希 embedding 配置，默认为 None，在 __post_init__ 中初始化

    product_surface_vocab_size: int = 16  # 产品面词表大小，默认 16，用于产品面 embedding 查表

    _initialized = False  # 内部标记，指示模型是否已调用 initialize()，防止未初始化使用

    def __post_init__(self):  # dataclass 的后置初始化钩子，在 __init__ 之后自动调用
        if self.hash_config is None:  # 若未显式提供 hash_config
            self.hash_config = HashConfig()  # 则使用默认的 HashConfig（各类哈希数均为 2）

    def initialize(self):  # 显式初始化方法，标记模型为已初始化状态
        self._initialized = True  # 将内部初始化标记置为 True
        return self  # 返回 self 以支持链式调用

    def make(self):  # 工厂方法，根据配置构建可执行的 PhoenixModel 模块实例
        if not self._initialized:  # 若尚未调用 initialize()
            logger.warning(f"PhoenixModel {self.name} is not initialized. Initializing.")  # 记录告警并自动初始化
            self.initialize()  # 自动执行初始化

        return PhoenixModel(  # 构造并返回 PhoenixModel 实例
            model=self.model.make(),  # 调用 TransformerConfig.make() 生成 Transformer 主干
            config=self,  # 将本配置对象传入模型，供其读取超参
            fprop_dtype=self.fprop_dtype,  # 传入前向传播精度
        )


@dataclass  # 使用 dataclass 装饰器，将 PhoenixModel 定义为 Haiku 模块（虽然继承 hk.Module，dataclass 用于自动生成字段处理）
class PhoenixModel(hk.Module):  # 定义 Phoenix 推荐模型主模块，继承 Haiku 模块以接入参数管理
    """A transformer-based recommendation model for ranking candidates."""  # 原有英文 docstring：基于 Transformer 的候选排序推荐模型

    model: Transformer  # Grok-1 Transformer 主干实例，负责对拼接序列做自注意力建模
    config: PhoenixModelConfig  # 模型配置，提供维度、动作数、哈希配置等
    fprop_dtype: Any = jnp.bfloat16  # 前向传播精度，默认 bfloat16
    name: Optional[str] = None  # 模块名称，可选

    def _get_action_embeddings(  # 定义私有方法：把多热动作向量转换为 embedding
        self,
        actions: jax.Array,  # 动作多热向量，形状 [B, S, num_actions]，1 表示发生，0 表示未发生
    ) -> jax.Array:  # 返回动作 embedding，形状 [B, S, D]
        """Convert multi-hot action vectors to embeddings.  # 原有英文 docstring：把多热动作向量转换为 embedding

        Uses a learned projection matrix to map the signed action vector  # 使用可学习投影矩阵将带符号的动作向量
        to the embedding dimension. This works for any number of actions.  # 映射到 embedding 维度，适用于任意动作数
        """
        config = self.config  # 取模型配置，便于读取 embedding 维度
        _, _, num_actions = actions.shape  # 解析动作维度 num_actions（前两维 B、S 不在此使用）
        D = config.emb_size  # 取 embedding 维度 D

        # 创建 VarianceScaling 初始化器用于动作投影矩阵
        embed_init = hk.initializers.VarianceScaling(1.0, mode="fan_out")
        # 获取动作投影矩阵 action_projection，形状 [num_actions, D]，把动作向量映射到 D 维
        action_projection = hk.get_parameter(
            "action_projection",  # 参数名
            [num_actions, D],  # 形状：动作数 × embedding 维度
            dtype=jnp.float32,  # 参数精度 float32
            init=embed_init,  # 使用 VarianceScaling 初始化
        )

        # 将 0/1 多热动作转为 ±1 带符号向量：发生=+1，未发生=-1，强调动作正负信号
        actions_signed = (2 * actions - 1).astype(jnp.float32)

        # 投影：[B, S, num_actions] @ [num_actions, D] -> [B, S, D]
        action_emb = jnp.dot(actions_signed.astype(action_projection.dtype), action_projection)

        # 生成有效性掩码：只要任一动作发生即视为有效，形状 [B, S, 1]，用于屏蔽无动作的 padding 位
        valid_mask = jnp.any(actions, axis=-1, keepdims=True)
        action_emb = action_emb * valid_mask  # 将无动作位置 embedding 置零，避免引入噪声

        return action_emb.astype(self.fprop_dtype)  # 转换为前向传播精度后返回

    def _single_hot_to_embeddings(  # 定义私有方法：把单热离散索引通过查表转为 embedding
        self,
        input: jax.Array,  # 离散索引张量，形状 [B, S] 或 [B, C]，值为词表中的索引
        vocab_size: int,  # 词表大小，决定 embedding 表行数
        emb_size: int,  # embedding 维度，决定 embedding 表列数
        name: str,  # embedding 表参数名，用于在 Haiku 参数树中定位
    ) -> jax.Array:  # 返回 embedding，形状 [B, S, emb_size] 或 [B, C, emb_size]
        """Convert single-hot indices to embeddings via lookup table.  # 原有英文 docstring：通过查表把单热索引转为 embedding

        Args:
            input: [B, S] tensor of categorical indices  # 离散索引张量
            vocab_size: size of the vocabulary  # 词表大小
            emb_size: embedding dimension  # embedding 维度
            name: name for the embedding table parameter  # embedding 表参数名

        Returns:
            embeddings: [B, S, emb_size]  # 查表得到的 embedding
        """
        # 创建 VarianceScaling 初始化器用于 embedding 表
        embed_init = hk.initializers.VarianceScaling(1.0, mode="fan_out")
        # 获取 embedding 表参数，形状 [vocab_size, emb_size]，每行对应一个词的 embedding
        embedding_table = hk.get_parameter(
            name,  # 参数名（如 product_surface_embedding_table）
            [vocab_size, emb_size],  # 形状：词表大小 × embedding 维度
            dtype=jnp.float32,  # 参数精度 float32
            init=embed_init,  # 使用 VarianceScaling 初始化
        )

        # 将索引转为 one-hot 张量：[B, S] -> [B, S, vocab_size]
        input_one_hot = jax.nn.one_hot(input, vocab_size)
        # 通过矩阵乘法实现查表：[B, S, vocab_size] @ [vocab_size, emb_size] -> [B, S, emb_size]
        output = jnp.dot(input_one_hot, embedding_table)
        return output.astype(self.fprop_dtype)  # 转为前向传播精度后返回

    def _get_unembedding(self) -> jax.Array:  # 定义私有方法：获取用于解码到 logits 的"反嵌入"矩阵
        """Get the unembedding matrix for decoding to logits."""  # 原有英文 docstring：获取用于解码到 logits 的反嵌入矩阵
        config = self.config  # 取模型配置
        # 创建 VarianceScaling 初始化器用于反嵌入矩阵
        embed_init = hk.initializers.VarianceScaling(1.0, mode="fan_out")
        # 获取反嵌入矩阵 unembeddings，形状 [emb_size, num_actions]，将 D 维隐状态映射为 num_actions 个动作得分
        unembed_mat = hk.get_parameter(
            "unembeddings",  # 参数名
            [config.emb_size, config.num_actions],  # 形状：embedding 维度 × 动作数
            dtype=jnp.float32,  # 参数精度 float32
            init=embed_init,  # 使用 VarianceScaling 初始化
        )
        return unembed_mat  # 返回反嵌入矩阵

    def build_inputs(  # 定义输入构建方法：把批次特征与预查 embedding 组装成 Transformer 的输入序列
        self,
        batch: RecsysBatch,  # 输入批次，含哈希、动作、产品面等特征
        recsys_embeddings: RecsysEmbeddings,  # 预查得到的各类 embedding
    ) -> Tuple[jax.Array, jax.Array, int]:  # 返回 (输入 embedding, padding mask, 候选起始偏移)
        """Build input embeddings from batch and pre-looked-up embeddings.  # 原有英文 docstring：从批次与预查 embedding 构建输入 embedding

        Args:
            batch: RecsysBatch containing hashes, actions, product surfaces  # 批次含哈希、动作、产品面
            recsys_embeddings: RecsysEmbeddings containing pre-looked-up embeddings  # 预查 embedding

        Returns:
            embeddings: [B, 1 + history_len + num_candidates, D]  # 输入序列 embedding（用户1 + 历史 + 候选）
            padding_mask: [B, 1 + history_len + num_candidates]  # 序列有效性掩码
            candidate_start_offset: int - position where candidates start  # 候选在序列中的起始位置
        """
        config = self.config  # 取模型配置
        hash_config = config.hash_config  # 取多哈希 embedding 配置

        # 将历史产品面索引查表转为 embedding：[B, S] -> [B, S, D]
        history_product_surface_embeddings = self._single_hot_to_embeddings(
            batch.history_product_surface,  # type: ignore  # 历史产品面索引
            config.product_surface_vocab_size,  # 词表大小
            config.emb_size,  # embedding 维度
            "product_surface_embedding_table",  # 参数名（历史与候选共享同一张表）
        )
        # 将候选产品面索引查表转为 embedding：[B, C] -> [B, C, D]（与历史共用同一张表）
        candidate_product_surface_embeddings = self._single_hot_to_embeddings(
            batch.candidate_product_surface,  # type: ignore  # 候选产品面索引
            config.product_surface_vocab_size,  # 词表大小（与历史相同）
            config.emb_size,  # embedding 维度
            "product_surface_embedding_table",  # 参数名相同，实现历史与候选共享 embedding 表
        )

        # 将历史动作多热向量转为 embedding：[B, S, num_actions] -> [B, S, D]
        history_actions_embeddings = self._get_action_embeddings(batch.history_actions)  # type: ignore

        # 融合用户多哈希 embedding 为单一用户表示：返回 [B, 1, D] 与用户 padding mask [B, 1]
        user_embeddings, user_padding_mask = block_user_reduce(
            batch.user_hashes,  # type: ignore  # 用户哈希
            recsys_embeddings.user_embeddings,  # type: ignore  # 用户多哈希 embedding
            hash_config.num_user_hashes,  # 用户哈希数
            config.emb_size,  # embedding 维度
            1.0,  # 初始化缩放系数
        )

        # 融合历史推文/作者/动作/产品面为历史序列 embedding：返回 [B, S, D] 与历史 padding mask [B, S]
        history_embeddings, history_padding_mask = block_history_reduce(
            batch.history_post_hashes,  # type: ignore  # 历史推文哈希
            recsys_embeddings.history_post_embeddings,  # type: ignore  # 历史推文多哈希 embedding
            recsys_embeddings.history_author_embeddings,  # type: ignore  # 历史作者多哈希 embedding
            history_product_surface_embeddings,  # 历史产品面 embedding
            history_actions_embeddings,  # 历史动作 embedding
            hash_config.num_item_hashes,  # 物品哈希数
            hash_config.num_author_hashes,  # 作者哈希数
            1.0,  # 初始化缩放系数
        )

        # 融合候选推文/作者/产品面为候选序列 embedding：返回 [B, C, D] 与候选 padding mask [B, C]
        candidate_embeddings, candidate_padding_mask = block_candidate_reduce(
            batch.candidate_post_hashes,  # type: ignore  # 候选推文哈希
            recsys_embeddings.candidate_post_embeddings,  # type: ignore  # 候选推文多哈希 embedding
            recsys_embeddings.candidate_author_embeddings,  # type: ignore  # 候选作者多哈希 embedding
            candidate_product_surface_embeddings,  # 候选产品面 embedding
            hash_config.num_item_hashes,  # 物品哈希数
            hash_config.num_author_hashes,  # 作者哈希数
            1.0,  # 初始化缩放系数
        )

        # 在序列维拼接：用户(1) + 历史(S) + 候选(C) -> [B, 1+S+C, D]，构成 Transformer 输入序列
        embeddings = jnp.concatenate(
            [user_embeddings, history_embeddings, candidate_embeddings], axis=1
        )
        # 拼接对应的 padding mask：[B, 1] + [B, S] + [B, C] -> [B, 1+S+C]，标记每个位置是否有效
        padding_mask = jnp.concatenate(
            [user_padding_mask, history_padding_mask, candidate_padding_mask], axis=1
        )

        # 计算候选在序列中的起始偏移 = 用户段长度(1) + 历史段长度(S)，供 Transformer 构造 Candidate Isolation 掩码
        candidate_start_offset = user_padding_mask.shape[1] + history_padding_mask.shape[1]

        return embeddings.astype(self.fprop_dtype), padding_mask, candidate_start_offset  # 返回输入 embedding、padding mask、候选起始偏移

    def __call__(  # 定义模型前向传播：将批次转换为 logits，是 hk.Module 的核心调用入口
        self,
        batch: RecsysBatch,  # 输入批次，含哈希、动作、产品面等特征
        recsys_embeddings: RecsysEmbeddings,  # 预查得到的各类 embedding
    ) -> RecsysModelOutput:  # 返回模型输出，含每个候选每个动作的 logits
        """Forward pass for ranking candidates.  # 原有英文 docstring：用于候选排序的前向传播

        Args:
            batch: RecsysBatch containing hashes, actions, product surfaces  # 批次含哈希、动作、产品面
            recsys_embeddings: RecsysEmbeddings containing pre-looked-up embeddings  # 预查 embedding

        Returns:
            RecsysModelOutput containing logits for each candidate. Shape = [B, num_candidates, num_actions]  # 输出 logits 形状
        """
        # 调用 build_inputs 构建输入序列 embedding、padding mask 与候选起始偏移
        embeddings, padding_mask, candidate_start_offset = self.build_inputs(
            batch, recsys_embeddings
        )

        # transformer  # 原有英文注释：transformer
        # 调用 Grok-1 Transformer 主干对输入序列建模；candidate_start_offset 用于构造 Candidate Isolation 掩码,
        # 使候选之间互不可见，但都能 attend 到用户与历史，从而实现 per-candidate 的独立打分
        model_output = self.model(
            embeddings,  # 输入序列 embedding [B, 1+S+C, D]
            padding_mask,  # padding mask [B, 1+S+C]
            candidate_start_offset=candidate_start_offset,  # 候选起始偏移，供掩码构造使用
        )

        out_embeddings = model_output.embeddings  # 取 Transformer 输出的隐状态序列，形状 [B, 1+S+C, D]

        # 对输出隐状态做层归一化，稳定数值并提升打分质量
        out_embeddings = layer_norm(out_embeddings)

        # 切片取出候选位置的隐状态：[B, 1+S+C, D] -> [B, C, D]，只对候选打分
        candidate_embeddings = out_embeddings[:, candidate_start_offset:, :]

        unembeddings = self._get_unembedding()  # 获取反嵌入矩阵 [D, num_actions]
        # 解码：[B, C, D] @ [D, num_actions] -> [B, C, num_actions]，得到每个候选各动作的得分
        logits = jnp.dot(candidate_embeddings.astype(unembeddings.dtype), unembeddings)
        logits = logits.astype(self.fprop_dtype)  # 转为前向传播精度

        return RecsysModelOutput(logits=logits)  # 封装为命名元组返回，便于下游取用 logits 字段
