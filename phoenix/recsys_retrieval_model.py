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

import logging  # 导入 Python 标准库的 logging 模块，用于在模型初始化、告警等场景记录日志
from dataclasses import dataclass  # 从 dataclasses 导入 dataclass 装饰器，用于自动为配置类和模块类生成 __init__ 等样板方法
from typing import Any, NamedTuple, Optional, Tuple  # 从 typing 导入类型注解工具：Any 表示任意类型、NamedTuple 用于定义命名元组、Optional 表示可空类型、Tuple 表示元组

import haiku as hk  # 导入 DeepMind 的 Haiku 库，作为 JAX 之上的神经网络模块化框架，用于定义模块和参数
import jax  # 导入 JAX 库，提供自动微分和 XLA 编译的高性能数值计算能力
import jax.numpy as jnp  # 导入 JAX 的 numpy 兼容接口，提供支持 JIT/自动微分的数组运算

from grok import TransformerConfig, Transformer  # 从 grok 模块导入 TransformerConfig 配置类和 Transformer 模型类，User Tower 将复用该 Transformer
from recsys_model import (  # 从 recsys_model 模块导入检索模型所需的共享组件
    HashConfig,  # HashConfig：多哈希（multi-hash）配置，定义 user/item/author 各自使用的哈希数量
    RecsysBatch,  # RecsysBatch：推荐系统一个批次输入的结构化数据（哈希、行为、产品位面等）
    RecsysEmbeddings,  # RecsysEmbeddings：预先查表得到的所有 embedding 集合（user/history/candidate 的 post 与 author embedding）
    block_history_reduce,  # block_history_reduce：将历史序列的多哈希 embedding 归约合并为单个向量的工具函数
    block_user_reduce,  # block_user_reduce：将用户特征的多哈希 embedding 归约合并为单个向量的工具函数
)

logger = logging.getLogger(__name__)  # 创建当前模块专属的 logger 实例，后续用它输出初始化告警等日志信息

EPS = 1e-12  # 定义一个极小正数 EPS（1e-12），用作 L2 归一化时防止除以零的兜底值
INF = 1e12  # 定义一个很大的数 INF（1e12），用作屏蔽无效 corpus 候选时置为的极小相似度（-INF）


class RetrievalOutput(NamedTuple):  # 定义命名元组 RetrievalOutput，作为检索模型的统一输出容器
    """Output of the retrieval model."""

    user_representation: jax.Array  # user_representation 字段：每个用户的 L2 归一化表示向量，shape [B, D]
    top_k_indices: jax.Array  # top_k_indices 字段：为每个用户检索到的 top-k 候选在 corpus 中的索引，shape [B, K]
    top_k_scores: jax.Array  # top_k_scores 字段：对应 top-k 候选的相似度分数，shape [B, K]


@dataclass  # 用 dataclass 装饰 CandidateTower，使 hk.Module 的字段定义更简洁（自动生成构造逻辑）
class CandidateTower(hk.Module):  # 定义 CandidateTower 类：两塔检索模型中的"候选塔"，把帖子+作者 embedding 投影到共享空间
    """Candidate tower that projects post+author embeddings to a shared embedding space.

    This tower takes the concatenated embeddings of a post and its author,
    and projects them to a normalized representation suitable for similarity search.
    """

    emb_size: int  # emb_size 字段：最终候选表示的目标维度（与 User Tower 输出维度一致，保证能做点积）
    name: Optional[str] = None  # name 字段：可选的模块命名（Haiku 模块的命名空间），默认为 None

    def __call__(self, post_author_embedding: jax.Array) -> jax.Array:  # 定义前向传播方法，接收拼接后的 post+author embedding 并返回归一化表示
        """Project post+author embeddings to normalized representation.

        Args:
            post_author_embedding: Concatenated post and author embeddings
                Shape: [B, C, num_hashes, D] or [B, num_hashes, D]

        Returns:
            Normalized candidate representation
                Shape: [B, C, D] or [B, D]
        """
        if len(post_author_embedding.shape) == 4:  # 判断输入是否为 4 维（带候选数量 C 维），即 [B, C, num_hashes, D] 形态
            B, C, _, _ = post_author_embedding.shape  # 若是 4 维，则解包得到 batch 大小 B 和候选数量 C，忽略哈希数与 D 维
            post_author_embedding = jnp.reshape(post_author_embedding, (B, C, -1))  # 把 [B, C, num_hashes, D] 展平为 [B, C, num_hashes*D]，拼接所有哈希的 embedding
        else:  # 否则输入是 3 维（无候选数量 C 维），即 [B, num_hashes, D] 形态
            B, _, _ = post_author_embedding.shape  # 解包得到 batch 大小 B，忽略哈希数与 D 维
            post_author_embedding = jnp.reshape(post_author_embedding, (B, -1))  # 把 [B, num_hashes, D] 展平为 [B, num_hashes*D]，拼接所有哈希的 embedding

        embed_init = hk.initializers.VarianceScaling(1.0, mode="fan_out")  # 定义 VarianceScaling 权重初始化器（scale=1.0，fan_out 模式），用于两层 MLP 的权重初始化

        proj_1 = hk.get_parameter(  # 获取/创建 MLP 第一层权重矩阵 proj_1
            "candidate_tower_projection_1",  # 参数名为 candidate_tower_projection_1，用于唯一标识该权重
            [post_author_embedding.shape[-1], self.emb_size * 2],  # 形状为 [输入拼接维度, 2*emb_size]，先把输入升维到 2 倍以增强表达能力
            dtype=jnp.float32,  # 参数数据类型为 float32，保证训练与推理的数值精度
            init=embed_init,  # 使用上面定义的 VarianceScaling 初始化器初始化该权重
        )

        proj_2 = hk.get_parameter(  # 获取/创建 MLP 第二层权重矩阵 proj_2
            "candidate_tower_projection_2",  # 参数名为 candidate_tower_projection_2，用于唯一标识该权重
            [self.emb_size * 2, self.emb_size],  # 形状为 [2*emb_size, emb_size]，把中间隐藏层降维到最终的 emb_size 维表示
            dtype=jnp.float32,  # 参数数据类型为 float32
            init=embed_init,  # 同样使用 VarianceScaling 初始化器
        )

        hidden = jnp.dot(post_author_embedding.astype(proj_1.dtype), proj_1)  # 第一层线性变换：输入投影到隐藏层 [..., 2*emb_size]，先转换为与权重一致的类型
        hidden = jax.nn.silu(hidden)  # 对隐藏层施加 SiLU（Swish）激活函数，引入非线性，提升 MLP 的表达能力
        candidate_embeddings = jnp.dot(hidden.astype(proj_2.dtype), proj_2)  # 第二层线性变换：隐藏层投影到最终 emb_size 维候选向量

        candidate_norm_sq = jnp.sum(candidate_embeddings**2, axis=-1, keepdims=True)  # 计算候选向量每一维的平方和，得到 L2 范数的平方 [..., 1]，keepdims 保持维度以便广播
        candidate_norm = jnp.sqrt(jnp.maximum(candidate_norm_sq, EPS))  # 开方得到 L2 范数，用 maximum(EPS) 防止范数为 0 时出现除零错误
        candidate_representation = candidate_embeddings / candidate_norm  # L2 归一化：把候选向量除以自身范数，使其模长为 1，从而点积等价于余弦相似度

        return candidate_representation.astype(post_author_embedding.dtype)  # 把结果转换回输入 dtype（如 bfloat16）后返回，保证下游计算类型一致


@dataclass  # 用 dataclass 装饰 PhoenixRetrievalModelConfig，便于声明式地组织模型配置
class PhoenixRetrievalModelConfig:  # 定义 Phoenix 检索模型的配置类，承载模型结构、维度等超参数
    """Configuration for the Phoenix Retrieval Model.

    This model uses the same transformer architecture as the Phoenix ranker
    for encoding user representations.
    """

    model: TransformerConfig  # model 字段：底层 Transformer 的配置，用于构建 User Tower 的编码器
    emb_size: int  # emb_size 字段：统一的 embedding 维度 D，User Tower 与 Candidate Tower 共享
    history_seq_len: int = 128  # history_seq_len 字段：用户历史行为序列的最大长度，默认 128
    candidate_seq_len: int = 32  # candidate_seq_len 字段：候选序列的最大长度，默认 32

    name: Optional[str] = None  # name 字段：可选的模型命名，用于日志输出与区分实例
    fprop_dtype: Any = jnp.bfloat16  # fprop_dtype 字段：前向传播使用的数据类型，默认 bfloat16 以加速并省显存

    hash_config: HashConfig = None  # type: ignore  # hash_config 字段：多哈希配置，默认 None 会在 __post_init__ 中被替换为默认值

    product_surface_vocab_size: int = 16  # product_surface_vocab_size 字段：产品位面（product surface）的词典大小，默认 16

    _initialized: bool = False  # _initialized 字段：内部标记，记录配置是否已执行过 initialize()

    def __post_init__(self):  # dataclass 初始化后自动调用的钩子，用于填充默认依赖项
        if self.hash_config is None:  # 判断 hash_config 是否未被显式设置
            self.hash_config = HashConfig()  # 若未设置，则用默认 HashConfig 实例填充，避免后续访问 None

    def initialize(self):  # 定义 initialize 方法，标记配置已初始化
        self._initialized = True  # 把内部标记 _initialized 置为 True，表示已完成初始化
        return self  # 返回 self，支持链式调用（如 config.initialize().make()）

    def make(self):  # 定义 make 方法，根据配置构建实际的 PhoenixRetrievalModel 实例
        if not self._initialized:  # 判断配置是否尚未初始化
            logger.warning(f"PhoenixRetrievalModel {self.name} is not initialized. Initializing.")  # 若未初始化，输出一条告警日志提示用户
            self.initialize()  # 自动调用 initialize() 完成初始化

        return PhoenixRetrievalModel(  # 返回构建好的 PhoenixRetrievalModel 实例
            model=self.model.make(),  # 调用底层 TransformerConfig.make() 实例化 Transformer 编码器
            config=self,  # 把当前配置对象自身传入模型
            fprop_dtype=self.fprop_dtype,  # 把前向传播 dtype 传入模型
        )


@dataclass  # 用 dataclass 装饰 PhoenixRetrievalModel，组织其字段
class PhoenixRetrievalModel(hk.Module):  # 定义 Phoenix 检索模型主体：两塔检索模型（User Tower + Candidate Tower）
    """A two-tower retrieval model using the Phoenix transformer for user encoding.

    This model implements the two-tower architecture for efficient retrieval:
    - User Tower: Encodes user features + history using the Phoenix transformer
    - Candidate Tower: Projects candidate embeddings to a shared space

    The user and candidate representations are L2-normalized, enabling efficient
    approximate nearest neighbor (ANN) search using dot product similarity.
    """

    model: Transformer  # model 字段：底层 Transformer 编码器实例，用于编码用户表示
    config: PhoenixRetrievalModelConfig  # config 字段：模型配置对象，提供 emb_size、hash_config 等超参数
    fprop_dtype: Any = jnp.bfloat16  # fprop_dtype 字段：前向传播数据类型，默认 bfloat16
    name: Optional[str] = None  # name 字段：可选的模块命名

    def _get_action_embeddings(  # 定义私有方法 _get_action_embeddings：把多热（multi-hot）行为向量转换为 embedding
        self,
        actions: jax.Array,  # actions 参数：多热行为向量，shape [B, T, num_actions]，每个位置为 0/1 表示是否发生某类行为
    ) -> jax.Array:
        """Convert multi-hot action vectors to embeddings."""
        config = self.config  # 取出配置对象，便于后续读取 emb_size 等字段
        _, _, num_actions = actions.shape  # 解包 actions 的形状，取最后一维 num_actions 作为行为类别数
        D = config.emb_size  # 读取统一的 embedding 维度 D

        embed_init = hk.initializers.VarianceScaling(1.0, mode="fan_out")  # 定义 VarianceScaling 权重初始化器，用于行为投影矩阵
        action_projection = hk.get_parameter(  # 获取/创建行为投影矩阵 action_projection
            "action_projection",  # 参数名为 action_projection
            [num_actions, D],  # 形状为 [num_actions, D]，把每个行为类别映射为一个 D 维向量
            dtype=jnp.float32,  # 参数数据类型为 float32
            init=embed_init,  # 使用 VarianceScaling 初始化器
        )

        actions_signed = (2 * actions - 1).astype(jnp.float32)  # 符号化：把 0/1 多热值映射为 -1/+1（0->-1，1->+1），使"未发生"与"发生"两类行为获得相反符号的向量
        action_emb = jnp.dot(actions_signed.astype(action_projection.dtype), action_projection)  # 线性组合：用符号化权重对投影矩阵加权求和，得到行为 embedding [B, T, D]

        valid_mask = jnp.any(actions, axis=-1, keepdims=True)  # 计算有效掩码：判断每个位置是否存在任意一个行为（沿最后一维做 any），得到 [B, T, 1] 的布尔掩码
        action_emb = action_emb * valid_mask  # 掩码清零：把完全没有行为的位置 embedding 全部置零，避免"全 0 但被符号化成全 -1"引入噪声

        return action_emb.astype(self.fprop_dtype)  # 把行为 embedding 转换到前向传播 dtype（如 bfloat16）后返回

    def _single_hot_to_embeddings(  # 定义私有方法 _single_hot_to_embeddings：把单热（single-hot）整数索引通过查表转换为 embedding
        self,
        input: jax.Array,  # input 参数：单热整数索引张量，每个元素是一个类别 ID
        vocab_size: int,  # vocab_size 参数：词典大小（embedding 表的行数）
        emb_size: int,  # emb_size 参数：每个类别的 embedding 维度
        name: str,  # name 参数：embedding 表的参数名，用于唯一标识
    ) -> jax.Array:
        """Convert single-hot indices to embeddings via lookup table."""
        embed_init = hk.initializers.VarianceScaling(1.0, mode="fan_out")  # 定义 VarianceScaling 权重初始化器，用于 embedding 表
        embedding_table = hk.get_parameter(  # 获取/创建 embedding 查找表
            name,  # 用传入的 name 作为参数名
            [vocab_size, emb_size],  # 形状为 [vocab_size, emb_size]，每行是词典中一个类别的向量
            dtype=jnp.float32,  # 参数数据类型为 float32
            init=embed_init,  # 使用 VarianceScaling 初始化器
        )

        input_one_hot = jax.nn.one_hot(input, vocab_size)  # 把整数索引编码为 one-hot 向量 [..., vocab_size]，便于与 embedding 表做矩阵乘法完成查表
        output = jnp.dot(input_one_hot, embedding_table)  # 矩阵乘法查表：one-hot 与 embedding 表相乘，等价于按索引取出对应行的向量
        return output.astype(self.fprop_dtype)  # 把结果转换到前向传播 dtype 后返回

    def build_user_representation(  # 定义 User Tower 的构建方法：从用户特征与历史行为构建 L2 归一化的用户表示
        self,
        batch: RecsysBatch,  # batch 参数：一个批次的输入数据，包含哈希、行为、产品位面等
        recsys_embeddings: RecsysEmbeddings,  # recsys_embeddings 参数：预先查表好的各类 embedding
    ) -> Tuple[jax.Array, jax.Array]:
        """Build user representation from user features and history.

        Uses the Phoenix transformer to encode user + history embeddings
        into a single user representation vector.

        Args:
            batch: RecsysBatch containing hashes, actions, product surfaces
            recsys_embeddings: RecsysEmbeddings containing pre-looked-up embeddings

        Returns:
            user_representation: L2-normalized user embedding [B, D]
            user_norm: Pre-normalization L2 norm [B, 1]
        """
        config = self.config  # 取出配置对象，便于后续读取 emb_size、product_surface_vocab_size 等
        hash_config = config.hash_config  # 取出多哈希配置，后续读取 num_user_hashes、num_item_hashes、num_author_hashes

        history_product_surface_embeddings = self._single_hot_to_embeddings(  # 把历史序列的产品位面单热索引转换为 embedding
            batch.history_product_surface,  # type: ignore  # 输入：历史序列中每个位置的产品位面类别 ID
            config.product_surface_vocab_size,  # 词典大小为配置中的 product_surface_vocab_size
            config.emb_size,  # embedding 维度为统一维度 emb_size
            "product_surface_embedding_table",  # embedding 表参数名为 product_surface_embedding_table
        )

        history_actions_embeddings = self._get_action_embeddings(batch.history_actions)  # type: ignore  # 把历史序列的多热行为向量转换为行为 embedding

        user_embeddings, user_padding_mask = block_user_reduce(  # 调用 block_user_reduce 归约用户特征的多哈希 embedding，得到单一用户向量及其 padding 掩码
            batch.user_hashes,  # type: ignore  # 输入：用户特征的多哈希整数 ID
            recsys_embeddings.user_embeddings,  # type: ignore  # 输入：预先查表好的用户 embedding
            hash_config.num_user_hashes,  # 用户特征使用的哈希数量，用于分组归约
            config.emb_size,  # embedding 维度
            1.0,  # 归约时的缩放因子（1.0 表示不缩放）
        )

        history_embeddings, history_padding_mask = block_history_reduce(  # 调用 block_history_reduce 归约历史序列的多哈希 embedding（post/author/产品位面/行为）为单一向量
            batch.history_post_hashes,  # type: ignore  # 输入：历史帖子（post）的多哈希整数 ID
            recsys_embeddings.history_post_embeddings,  # type: ignore  # 输入：历史帖子预先查表好的 embedding
            recsys_embeddings.history_author_embeddings,  # type: ignore  # 输入：历史作者预先查表好的 embedding
            history_product_surface_embeddings,  # 输入：上一步得到的产品位面 embedding
            history_actions_embeddings,  # 输入：上一步得到的行为 embedding
            hash_config.num_item_hashes,  # 帖子（item）使用的哈希数量
            hash_config.num_author_hashes,  # 作者使用的哈希数量
            1.0,  # 归约缩放因子
        )

        embeddings = jnp.concatenate([user_embeddings, history_embeddings], axis=1)  # 沿序列维拼接用户向量与历史向量，构成 Transformer 的输入序列 [B, 1+T, D]
        padding_mask = jnp.concatenate([user_padding_mask, history_padding_mask], axis=1)  # 沿序列维拼接用户与历史的 padding 掩码，标识哪些位置是有效的

        model_output = self.model(  # 调用底层 Transformer 编码器，对拼接后的序列做自注意力编码
            embeddings.astype(self.fprop_dtype),  # 输入 embedding，转换到前向传播 dtype
            padding_mask,  # 传入 padding 掩码，让 Transformer 忽略无效位置
            candidate_start_offset=None,  # 候选起始偏移置为 None，表示当前只编码用户侧、不区分候选段
        )

        user_outputs = model_output.embeddings  # 取出 Transformer 输出的序列表示 [B, T, D]

        mask_float = padding_mask.astype(jnp.float32)[:, :, None]  # [B, T, 1]  # 把 padding 掩码转为 float 并扩维到 [B, T, 1]，便于与 [B, T, D] 的表示做广播
        user_embeddings_masked = user_outputs * mask_float  # 用掩码把无效位置的输出置零，只保留有效位置的表示
        user_embedding_sum = jnp.sum(user_embeddings_masked, axis=1)  # [B, D]  # 沿序列维求和，把所有有效位置的表示累加为单一用户向量 [B, D]
        mask_sum = jnp.sum(mask_float, axis=1)  # [B, 1]  # 沿序列维求和掩码，得到每个用户有效位置的个数 [B, 1]
        user_representation = user_embedding_sum / jnp.maximum(mask_sum, 1.0)  # 求平均（mean pooling）：累加和除以有效位置数，用 maximum(1.0) 防止除零

        user_norm_sq = jnp.sum(user_representation**2, axis=-1, keepdims=True)  # 计算用户向量的 L2 范数平方 [B, 1]
        user_norm = jnp.sqrt(jnp.maximum(user_norm_sq, EPS))  # 开方得到 L2 范数，用 maximum(EPS) 防止除零
        user_representation = user_representation / user_norm  # L2 归一化：把用户向量缩放到模长为 1，使点积等价于余弦相似度

        return user_representation, user_norm  # 返回归一化用户表示 [B, D] 及其归一化前的范数 [B, 1]

    def build_candidate_representation(  # 定义 Candidate Tower 的构建方法：把候选帖子+作者 embedding 投影为归一化表示
        self,
        batch: RecsysBatch,  # batch 参数：一个批次的输入数据，含候选哈希
        recsys_embeddings: RecsysEmbeddings,  # recsys_embeddings 参数：预先查表好的候选 embedding
    ) -> Tuple[jax.Array, jax.Array]:
        """Build candidate (item) representations.

        Projects post + author embeddings to a shared embedding space
        using the candidate tower MLP.

        Args:
            batch: RecsysBatch containing candidate hashes
            recsys_embeddings: RecsysEmbeddings containing pre-looked-up embeddings

        Returns:
            candidate_representation: L2-normalized candidate embeddings [B, C, D]
            candidate_padding_mask: Valid candidate mask [B, C]
        """
        config = self.config  # 取出配置对象，后续读取 emb_size 用于构建 CandidateTower

        candidate_post_embeddings = recsys_embeddings.candidate_post_embeddings  # 取出候选帖子的 embedding，shape [B, C, num_hashes, D]
        candidate_author_embeddings = recsys_embeddings.candidate_author_embeddings  # 取出候选作者的 embedding，shape [B, C, num_hashes, D]

        post_author_embedding = jnp.concatenate(  # 沿第 2 维（哈希维）拼接帖子与作者 embedding，供 CandidateTower 一起投影
            [candidate_post_embeddings, candidate_author_embeddings], axis=2  # 指定 axis=2，把帖子与作者的哈希 embedding 在哈希维度拼接
        )

        candidate_tower = CandidateTower(  # 实例化候选塔 MLP
            emb_size=config.emb_size,  # 传入统一的 emb_size 作为候选塔输出维度
        )
        candidate_representation = candidate_tower(post_author_embedding)  # 调用候选塔，把拼接后的输入投影并 L2 归一化，得到 [B, C, D]

        candidate_padding_mask = (batch.candidate_post_hashes[:, :, 0] != 0).astype(jnp.bool_)  # type: ignore  # 用第一个哈希 ID 是否非 0 判断候选是否有效，得到布尔掩码 [B, C]

        return candidate_representation, candidate_padding_mask  # 返回归一化候选表示与候选有效掩码

    def __call__(  # 定义 __call__ 方法：检索模型的推理入口，构建用户表示并对 corpus 做 top-k 检索
        self,
        batch: RecsysBatch,  # batch 参数：输入批次数据
        recsys_embeddings: RecsysEmbeddings,  # recsys_embeddings 参数：预先查表好的 embedding
        corpus_embeddings: jax.Array,  # corpus_embeddings 参数：整个候选语料的归一化向量 [N, D]
        top_k: int,  # top_k 参数：每个用户需要检索的候选数量 K
        corpus_mask: Optional[jax.Array] = None,  # corpus_mask 参数：可选的语料有效掩码 [N]，用于屏蔽无效候选
    ) -> RetrievalOutput:
        """Retrieve top-k candidates from corpus for each user.

        Args:
            batch: RecsysBatch containing hashes, actions, product surfaces
            recsys_embeddings: RecsysEmbeddings containing pre-looked-up embeddings
            corpus_embeddings: [N, D] normalized corpus candidate embeddings
            top_k: Number of candidates to retrieve
            corpus_mask: [N] optional mask for valid corpus entries

        Returns:
            RetrievalOutput containing user representation and top-k results
        """
        user_representation, _ = self.build_user_representation(batch, recsys_embeddings)  # 构建用户的 L2 归一化表示 [B, D]，忽略返回的范数

        top_k_indices, top_k_scores = self._retrieve_top_k(  # 调用 _retrieve_top_k 计算每个用户与 corpus 的相似度并取 top-k
            user_representation, corpus_embeddings, top_k, corpus_mask  # 传入用户表示、语料向量、top-k 值与语料掩码
        )

        return RetrievalOutput(  # 将结果封装为 RetrievalOutput 命名元组返回
            user_representation=user_representation,  # 用户表示向量
            top_k_indices=top_k_indices,  # top-k 候选索引 [B, K]
            top_k_scores=top_k_scores,  # top-k 相似度分数 [B, K]
        )

    def _retrieve_top_k(  # 定义私有方法 _retrieve_top_k：计算用户与语料的相似度并检索 top-k
        self,
        user_representation: jax.Array,  # user_representation 参数：归一化用户表示 [B, D]
        corpus_embeddings: jax.Array,  # corpus_embeddings 参数：归一化语料向量 [N, D]
        top_k: int,  # top_k 参数：每个用户检索的候选数量 K
        corpus_mask: Optional[jax.Array] = None,  # corpus_mask 参数：可选的语料有效掩码 [N]
    ) -> Tuple[jax.Array, jax.Array]:
        """Retrieve top-k candidates from a corpus for each user.

        Args:
            user_representation: [B, D] normalized user embeddings
            corpus_embeddings: [N, D] normalized corpus candidate embeddings
            top_k: Number of candidates to retrieve
            corpus_mask: [N] optional mask for valid corpus entries

        Returns:
            top_k_indices: [B, K] indices of top-k candidates
            top_k_scores: [B, K] similarity scores of top-k candidates
        """
        scores = jnp.matmul(user_representation, corpus_embeddings.T)  # 计算相似度矩阵 [B, N]：因两端均已 L2 归一化，点积即等于余弦相似度

        if corpus_mask is not None:  # 判断是否提供了语料有效掩码
            scores = jnp.where(corpus_mask[None, :], scores, -INF)  # 用掩码屏蔽无效候选：无效位置分数置为 -INF，保证 top-k 不会选中它们

        top_k_scores, top_k_indices = jax.lax.top_k(scores, top_k)  # 对每个用户取分数最高的 top_k 个候选，返回分数 [B, K] 与索引 [B, K]

        return top_k_indices, top_k_scores  # 返回 top-k 的索引与分数
