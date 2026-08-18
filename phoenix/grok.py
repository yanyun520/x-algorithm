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

# 本文件是 Grok-1 Transformer 模型的 JAX/Haiku 实现，被移植用于推荐系统排序（recsys ranking）。
# 主要组件：RMSNorm（均方根归一化）、RoPE 旋转位置编码、GQA 多头注意力（分组查询头）、
# SwiGLU 风格的 DenseBlock FFN、DecoderLayer 与 Transformer 主干，以及推荐系统专用的注意力掩码。

import logging  # 引入 Python 标准日志模块，用于在模型构建/调试时输出日志
from dataclasses import dataclass  # 引入 dataclass 装饰器，用于以简洁语法定义配置类（如 TransformerConfig）
from typing import NamedTuple, Optional, Sequence, Union  # 引入类型注解工具，NamedTuple 定义结构化输出，Optional/Sequence/Union 用于类型提示

import haiku as hk  # 引入 Haiku：Sonnet 的继任者，JAX 上的神经网络库，提供模块化参数管理（hk.Module、hk.get_parameter、hk.transform）
import jax  # 引入 JAX：自动微分 + XLA 编译的数值计算框架，支持 jit/vmap/grad 等函数式变换
import jax.numpy as jnp  # 引入 JAX 版 NumPy API，提供与 numpy 一致但可在 GPU/TPU 上运行、可微分的数组操作

logger = logging.getLogger(__name__)  # 创建以当前模块名为名的 logger，便于在 ffn_size 等函数中输出调试信息


class TrainingState(NamedTuple):  # 用 NamedTuple 定义训练状态容器，便于作为 JAX pytree 被 jit/scan 处理
    """Container for the training state."""

    params: hk.Params  # 存放 Haiku transform 后的所有模型参数（权重字典树），是训练过程中唯一需要维护的状态


def ffn_size(emb_size, widening_factor):  # 计算前馈网络（FFN）中间层维度：Grok 采用 SwiGLU/GeGLU 风格，需对齐到 8 的倍数
    _ffn_size = int(widening_factor * emb_size) * 2 // 3  # 先按 widening_factor 放大，再乘 2/3 是 SwiGLU 类 FFN 的惯例（两个门控分支等效容量约等于标准 FFN）
    _ffn_size = _ffn_size + (8 - _ffn_size) % 8  # ensure it's a multiple of 8  # 将维度向上调整到 8 的倍数，便于张量核心（Tensor Core）高效计算
    logger.debug(f"emd_size: {emb_size} adjusted ffn_size: {_ffn_size}")  # 输出调试日志，记录嵌入维度与调整后的 FFN 维度
    return _ffn_size  # 返回最终 FFN 中间层维度


def make_recsys_attn_mask(  # 构造推荐系统推理专用的注意力掩码：用户+历史用因果掩码，候选之间相互独立
    seq_len: int,  # 序列总长度（用户 token + 历史 + 候选）
    candidate_start_offset: int,  # 候选项在序列中的起始位置，<该位置为用户+历史，>=该位置为候选
    dtype: jnp.dtype = jnp.float32,  # 掩码的数据类型，默认 float32
) -> jax.Array:  # 返回 jax.Array 类型的注意力掩码
    """Create attention mask for recommendation system inference.

    Creates a mask where:
    - Positions 0 to candidate_start_offset-1 (user+history): causal attention
    - Positions candidate_start_offset onwards (candidates): can attend to user+history
      and themselves (self-attention), but NOT to other candidates

    This ensures each candidate is scored independently based on user+history context.

    Args:
        seq_len: Total sequence length (user + history + candidates)
        candidate_start_offset: Position where candidates start in the sequence
        dtype: Data type for the mask

    Returns:
        Attention mask of shape [1, 1, seq_len, seq_len] where 1 means "can attend"
    """
    # Start with causal mask for the full sequence
    # 先构造一个全序列的下三角因果掩码：位置 i 只能注意到位置 <=i 的元素
    # 形状 [1, 1, seq_len, seq_len]，1 表示允许注意，0 表示禁止
    causal_mask = jnp.tril(jnp.ones((1, 1, seq_len, seq_len), dtype=dtype))  # tril 取下三角，保证自回归方向的因果性

    # Zero out candidate-to-candidate attention (bottom-right block)
    # 将候选区到候选区的整块注意力置 0：候选之间不应互相参考，确保独立打分
    attn_mask = causal_mask.at[:, :, candidate_start_offset:, candidate_start_offset:].set(0)  # 使用 .at[].set() 在右下角候选块全置 0

    # Add back self-attention for candidates (diagonal of the candidate block)
    # 恢复候选块的对角线（自注意力）：每个候选可以看到自己，但看不到其他候选
    candidate_indices = jnp.arange(candidate_start_offset, seq_len)  # 生成候选位置的索引数组 [offset, offset+1, ..., seq_len-1]
    attn_mask = attn_mask.at[:, :, candidate_indices, candidate_indices].set(1)  # 在候选块对角线置 1，恢复候选自身的自注意力

    return attn_mask  # 返回最终的推荐系统注意力掩码，形状 [1, 1, seq_len, seq_len]


class MHAOutput(NamedTuple):  # 多头注意力输出的结构化容器，便于作为 pytree 在 JAX 中传递
    """Outputs of the multi-head attention operation."""

    embeddings: jax.Array  # 注意力计算后的嵌入输出，形状通常为 [B, T, model_size]


class DecoderOutput(NamedTuple):  # 单个解码层的输出容器
    embeddings: jax.Array  # 解码层输出嵌入，形状 [B, T, D]


class TransformerOutput(NamedTuple):  # 整个 Transformer 主干的输出容器
    embeddings: jax.Array  # Transformer 最终输出嵌入，形状 [B, T, D]


@dataclass  # 使用 dataclass 自动生成 __init__ 等方法，简洁地定义配置参数
class TransformerConfig:  # Transformer 模型的配置类，封装所有超参数
    emb_size: int  # 嵌入维度 D（模型隐藏维度）
    key_size: int  # 每个注意力头的维度（key/query 的维度）
    num_q_heads: int  # 查询头（Query head）数量
    num_kv_heads: int  # 键/值头（Key/Value head）数量，当 < num_q_heads 时启用 GQA（分组查询注意力）
    num_layers: int  # Transformer 解码层的层数
    widening_factor: float = 4.0  # FFN 的扩展系数，默认 4.0（配合 2/3 调整后约为标准 4x FFN 容量)

    attn_output_multiplier: float = 1.0  # 注意力 logits 的缩放因子，用于控制注意力分布的尖锐程度（Grok 特有)

    name: Optional[str] = None  # 可选的模块名称，用于参数命名空间隔离

    def make(self) -> "Transformer":  # 工厂方法：根据当前配置实例化并返回一个 Transformer 模块
        return Transformer(  # 用配置中的超参数构造 Transformer 实例
            num_q_heads=self.num_q_heads,  # 传入查询头数量
            num_kv_heads=self.num_kv_heads,  # 传入键/值头数量
            widening_factor=self.widening_factor,  # 传入 FFN 扩展系数
            key_size=self.key_size,  # 传入每个头的维度
            attn_output_multiplier=self.attn_output_multiplier,  # 传入注意力输出缩放因子
            num_layers=self.num_layers,  # 传入层数
        )


def hk_rms_norm(  # 便捷封装函数：对输入张量应用 RMSNorm（均方根归一化）
    x: jax.Array,  # 输入张量，待归一化
    fixed_scale=False,  # 是否使用固定缩放（不创建可学习 scale 参数）
) -> jax.Array:  # 返回归一化后的张量，形状与输入相同
    """Applies a unique LayerNorm to x with default settings."""
    ln = RMSNorm(axis=-1, create_scale=not fixed_scale)  # 实例化 RMSNorm 模块，沿最后一维归一化，按需创建可学习 scale
    return ln(x)  # 对 x 执行 RMSNorm 并返回结果


class Linear(hk.Linear):  # 自定义 Linear 层，继承 hk.Linear，但强制权重以 float32 存储、前向时再转回前向 dtype（数值稳定性）
    def __init__(  # 构造函数：定义输出维度、是否带偏置、模块名
        self,
        output_size: int,  # 输出特征维度
        with_bias: bool = True,  # 是否使用偏置项，默认 True
        name: Optional[str] = None,  # 模块名称，用于参数命名
    ):
        super().__init__(  # 调用父类 hk.Linear 的构造函数完成基础初始化
            output_size=output_size,  # 传入输出维度
            with_bias=with_bias,  # 传入是否带偏置
            name=name,  # 传入模块名
        )

    def __call__(  # type: ignore  # 前向计算：覆盖父类 __call__，自定义参数初始化与 dtype 处理
        self,
        inputs: jax.Array,  # 输入张量，形状 [..., input_size]
    ) -> jax.Array:  # 返回输出张量，形状 [..., output_size]
        """Computes a linear transform of the input."""

        fprop_dtype = inputs.dtype  # 记录输入的前向数据类型（如 bfloat16），计算时用该类型，但参数存储用 float32
        if not inputs.shape:  # 若输入是标量（无 shape）
            raise ValueError("Input must not be scalar.")  # 抛出错误：Linear 不支持标量输入

        input_size = inputs.shape[-1]  # 取输入最后一维作为输入维度，用于构造权重矩阵形状
        output_size = self.output_size  # 取目标输出维度

        w = hk.get_parameter(  # 通过 Haiku 获取/创建权重参数 w（自动注册到模块参数树）
            "w", [input_size, output_size], jnp.float32, init=hk.initializers.Constant(0)  # 形状 [input_size, output_size]，存储为 float32，初始化为 0
        )

        out = jnp.dot(inputs, w.astype(fprop_dtype))  # 矩阵乘法：inputs @ w，并将 w 转为前向 dtype 以匹配计算精度
        if self.with_bias:  # 若启用偏置
            b = hk.get_parameter(  # 获取/创建偏置参数 b，形状 [output_size]，float32 存储，初始化为 0
                "b", [self.output_size], jnp.float32, init=hk.initializers.Constant(0)
            )
            b = jnp.broadcast_to(b, out.shape)  # 将偏置广播到与 out 相同的形状，以便逐元素相加
            out = out + b.astype(fprop_dtype)  # 将偏置转为前向 dtype 后加到输出上

        return out  # 返回线性变换结果，形状 [..., output_size]


class RMSNorm(hk.RMSNorm):  # 自定义 RMSNorm，继承 hk.RMSNorm，覆盖前向以使用 float32 计算（提升数值稳定性）
    def __init__(  # 构造函数：指定归一化轴、epsilon、模块名、是否创建可学习 scale
        self,
        axis: Union[int, Sequence[int], slice],  # 归一化的轴，通常为 -1（最后一维）
        eps: float = 1e-5,  # 防 除 0 的小常数
        name: Optional[str] = None,  # 模块名
        create_scale: bool = True,  # 是否创建可学习的缩放参数 scale
    ):
        super().__init__(axis, eps, create_scale=create_scale, name=name)  # 调用父类构造函数完成初始化

    def __call__(self, inputs: jax.Array):  # 前向：对 inputs 沿最后一维做 RMSNorm
        fprop_dtype = inputs.dtype  # 记录前向 dtype，最终输出转回该类型
        param_shape = (inputs.shape[-1],)  # scale 参数形状为 [特征维度]
        if self.create_scale:  # 若需要可学习 scale
            scale = hk.get_parameter(  # 获取/创建 scale 参数，float32 存储，初始化为 0
                "scale",
                param_shape,
                dtype=jnp.float32,
                init=hk.initializers.Constant(0),
            )
            scale = jnp.broadcast_to(scale.astype(jnp.float32), inputs.shape)  # 将 scale 广播到与输入相同的形状
        else:  # 若不创建可学习 scale
            scale = 1.0  # 使用固定缩放 1.0
        inputs = inputs.astype(jnp.float32)  # 将输入转为 float32 进行归一化计算，避免低精度下的数值不稳定
        scale = jnp.float32(scale)  # 确保 scale 为 float32 类型
        mean_squared = jnp.mean(jnp.square(inputs), axis=[-1], keepdims=True)  # 计算最后一维的均方值（mean of squares），保留维度以便广播
        mean_squared = jnp.broadcast_to(mean_squared, inputs.shape)  # 将均方值广播到与输入相同的形状

        normed_inputs = inputs * jax.lax.rsqrt(mean_squared + self.eps)  # 用 rsqrt（倒数平方根，XLA 优化算子）做归一化：x / sqrt(mean_sq + eps)

        outputs = scale * normed_inputs  # 应用可学习缩放 scale，得到归一化输出

        return outputs.astype(fprop_dtype)  # 将输出转回前向 dtype 返回，保持计算图 dtype 一致


def rotate_half(  # RoPE 用的旋转辅助函数：将特征向量后半部分取负并与前半拼接，实现二维旋转
    x: jax.Array,  # 输入张量，最后一维为待旋转的特征维度（须为偶数）
) -> jax.Array:  # 返回旋转后的张量，形状与输入相同
    """Obtain the rotated counterpart of each feature"""
    x1, x2 = jnp.split(x, 2, axis=-1)  # 沿最后一维将 x 分成两半：x1=前半，x2=后半
    return jnp.concatenate((-x2, x1), axis=-1)  # 拼接 (-x2, x1)，相当于对每对 (x_{2i}, x_{2i+1}) 做 90° 旋转


class RotaryEmbedding(hk.Module):  # RoPE 旋转位置编码模块：将位置信息以旋转矩阵形式注入 query/key
    """Applies rotary embeddings (RoPE) to the input sequence tensor,
    as described in https://arxiv.org/abs/2104.09864.

    Attributes:
        dim (int): Dimensionality of the feature vectors
        base_exponent (int): Base exponent to compute embeddings from
    """

    def __init__(  # 构造函数：指定旋转作用维度、模块名、基数指数
        self,
        dim: int,  # 旋转作用的总特征维度（须为偶数）
        name: Optional[str] = None,  # 模块名
        base_exponent: int = 10000,  # 频率基数，常见值为 10000，控制不同维度的频率分布
    ):
        super().__init__(name)  # 调用父类 hk.Module 构造函数
        self.dim = dim  # 保存旋转维度
        self.base_exponent = base_exponent  # 保存基数指数
        assert self.dim % 2 == 0  # 断言维度为偶数：RoPE 需要将特征两两配对做 2D 旋转

    def __call__(  # 前向：对输入 x 应用 RoPE 旋转
        self,
        x: jax.Array,  # 输入张量，形状 [..., seq_dim_size, num_heads, dim]
        seq_dim: int,  # 指定哪个轴是序列轴（用于获取位置索引）
        offset: jax.Array,  # 位置偏移量，支持标量或每批次一个偏移
        const_position: Optional[int] = None,  # 若给定，则所有位置使用该常量位置（用于特定推理场景）
        t: Optional[jax.Array] = None,  # 可选的自定义位置张量，覆盖默认的位置序列
    ) -> jax.Array:  # 返回应用 RoPE 后的张量，形状与输入相同
        fprop_dtype = x.dtype  # 记录前向 dtype，最终输出转回该类型
        # Compute the per-dimension frequencies
        # 计算每个频率对的指数：0, 2, 4, ..., dim-2（共 dim/2 个）
        exponents = jnp.arange(0, self.dim, 2, dtype=jnp.float32)  # 生成偶数序列 [0,2,...,dim-2]，用于构造不同频率
        inv_freq = jnp.asarray(  # 计算逆频率：1 / base^(2i/dim)，频率从高到低分布
            1.0 / (self.base_exponent ** (exponents / self.dim)), dtype=jnp.float32  # base_exponent^(exponents/dim) 给出从 1 到 base 的指数衰减
        )

        if jnp.shape(offset) == ():  # 若 offset 是标量
            # Offset can be a scalar or one offset per batch element.
            offset = jnp.expand_dims(offset, 0)  # 将标量 offset 扩展为形状 [1]，统一后续处理

        # Compute the per element phase (to pass into sin and cos)
        # 计算每个元素的相位（送入 sin/cos）
        if const_position:  # 若使用常量位置
            t = const_position * jnp.ones(  # 构造一个全部为 const_position 的位置序列，形状 [1, seq_len]
                (
                    1,
                    x.shape[seq_dim],
                ),
                dtype=jnp.float32,
            )
        elif t is None:  # 若未提供自定义 t，则用默认位置序列
            t = jnp.arange(x.shape[seq_dim], dtype=jnp.float32) + jnp.expand_dims(offset, -1)  # 位置 = [0,1,...,seq_len-1] + offset，支持批次偏移
        phase = jnp.einsum("bi,j->bij", t, inv_freq)  # 通过 einsum 计算相位矩阵：position * inv_freq，形状 [batch, seq_len, dim/2]
        phase = jnp.tile(phase, reps=(1, 2))[:, :, None, :]  # 将相位在频率维复制一次（cos/sin 各需一份），并在头维度插入一维以便广播，形状 [batch, seq_len, 1, dim]

        x = x * jnp.cos(phase) + rotate_half(x) * jnp.sin(phase)  # RoPE 核心公式：x*cos(θ) + rotate_half(x)*sin(θ)，等价于对每对特征做 2D 旋转
        x = x.astype(fprop_dtype)  # 将结果转回前向 dtype

        return x  # 返回注入位置信息后的张量


class MultiHeadAttention(hk.Module):  # 多头注意力模块（含 GQA 分组查询头），Grok-1 的核心注意力实现
    def __init__(  # 构造函数：配置头数、头维度、是否带偏置、输出缩放等
        self,
        num_q_heads: int,  # 查询头数量
        num_kv_heads: int,  # 键/值头数量（GQA 时 < num_q_heads）
        key_size: int,  # 每个头的维度
        *,
        with_bias: bool = True,  # 是否在线性投影中使用偏置
        value_size: Optional[int] = None,  # 每个值头的维度，默认等于 key_size
        model_size: Optional[int] = None,  # 输出投影的维度，默认 key_size * num_q_heads
        attn_output_multiplier: float = 1.0,  # 注意力 logits 的缩放因子
        name: Optional[str] = None,  # 模块名
    ):
        super().__init__(name=name)  # 调用父类构造函数，传入模块名
        self.num_q_heads = num_q_heads  # 保存查询头数量
        self.num_kv_heads = num_kv_heads  # 保存键/值头数量
        self.key_size = key_size  # 保存每个头的维度
        self.value_size = value_size or key_size  # 设置值头维度，默认等于 key_size
        self.model_size = model_size or key_size * num_q_heads  # 设置输出维度，默认为 key_size * 查询头数
        self.attn_output_multiplier = attn_output_multiplier  # 保存注意力输出缩放因子
        self.with_bias = with_bias  # 保存是否使用偏置

    def __call__(  # 前向计算：执行多头注意力（含 RoPE、GQA、注意力裁剪、softmax）
        self,
        query: jax.Array,  # 查询张量，形状 [B, T_q, D]
        key: jax.Array,  # 键张量，形状 [B, T_k, D]
        value: jax.Array,  # 值张量，形状 [B, T_v, D]
        mask: jax.Array,  # 注意力掩码，形状 [B, 1, T_q, T_k] 或可广播形式
    ) -> MHAOutput:  # 返回 MHAOutput，包含注意力输出嵌入
        # In shape hints below, we suppress the leading dims [...] for brevity.
        # Hence e.g. [A, B] should be read in every case as [..., A, B].
        projection = self._linear_projection  # 取线性投影函数的引用，便于后续对 Q/K/V 投影

        # Check that the keys and values have consistent batch size and sequence length.
        # 校验 key 与 value 的批次和序列长度一致
        assert key.shape[:2] == value.shape[:2], f"key/value shape: {key.shape}/{value.shape}"  # 断言 key/value 的前两维（批次、序列）相同

        if mask is not None:  # 若提供了掩码
            assert mask.ndim == 4  # 断言掩码为 4 维（[B, H, T_q, T_k]）
            assert mask.shape[0] in {  # 断言掩码批次维度为 1 或与 query 批次一致
                1,
                query.shape[0],
            }, f"mask/query shape: {mask.shape}/{query.shape}"
            assert key.shape[0] in {  # 断言 key 批次维度为 1 或与 query 批次一致
                1,
                query.shape[0],
            }, f"key/query shape: {key.shape}/{query.shape}"
            assert mask.shape[1] == 1  # 断言掩码的头维度为 1（广播到所有头）
            assert mask.shape[2] in {  # 断言掩码的查询序列维度为 1 或与 query 序列一致
                1,
                query.shape[1],
            }, f"mask/query shape: {mask.shape}/{query.shape}"
            assert mask.shape[3] in {  # 断言掩码的键序列维度为 1 或与 key 序列一致
                1,
                key.shape[1],
            }, f"mask/query shape: {mask.shape}/{key.shape}"

        # Compute key/query/values (overload K/Q/V to denote the respective sizes).
        # 计算 key/query/value 投影（K/Q/V 分别表示对应尺寸）
        assert self.num_q_heads % self.num_kv_heads == 0  # 断言查询头数是键/值头数的整数倍（GQA 要求）
        query_heads = projection(query, self.key_size, self.num_q_heads, name="query")  # 投影 query 到 [B, T_q, num_q_heads, key_size]
        key_heads = projection(key, self.key_size, self.num_kv_heads, name="key")  # 投影 key 到 [B, T_k, num_kv_heads, key_size]
        value_heads = projection(value, self.value_size, self.num_kv_heads, name="value")  # 投影 value 到 [B, T_v, num_kv_heads, value_size]

        rotate = RotaryEmbedding(dim=self.key_size, base_exponent=int(1e4))  # 创建 RoPE 模块，作用维度为 key_size，基数 10000
        key_heads = rotate(key_heads, seq_dim=1, offset=0)  # 对 key 应用 RoPE，注入位置信息（offset=0 表示从序列起点开始）
        query_heads = rotate(query_heads, seq_dim=1, offset=0)  # 对 query 应用 RoPE，注入位置信息

        b, t, h, d = query_heads.shape  # 解析 query 形状：b=批次, t=查询序列长度, h=查询头数, d=头维度
        _, _, kv_h, _ = key_heads.shape  # 解析 key 的 kv_h=键/值头数
        assert h % kv_h == 0, f"query_heads {h} must be a multiple of kv_heads {kv_h}"  # 再次断言查询头数是键/值头数的整数倍

        query_heads = jnp.reshape(query_heads, (b, t, kv_h, h // kv_h, d))  # 重塑 query：将查询头按 GQA 分组，形状 [B, T_q, kv_h, group_size, d]

        # Compute attention weights.
        # 计算注意力权重
        # Attention softmax is always carried out in fp32.
        # 注意力 softmax 始终在 fp32 下进行，以保证数值稳定
        attn_logits = jnp.einsum("...thHd,...Thd->...hHtT", query_heads, key_heads).astype(  # 通过 einsum 计算 Q·K^T：每个 query 头与其对应 kv 组的点积
            jnp.float32  # 结果转为 fp32，形状 [..., kv_h, group_size, t, T]
        )
        attn_logits *= self.attn_output_multiplier  # 应用注意力输出缩放因子（Grok 特有，控制 logits 量级）
        max_attn_val = jnp.array(30.0, dtype=attn_logits.dtype)  # 定义注意力裁剪上限 30.0
        attn_logits = max_attn_val * jnp.tanh(attn_logits / max_attn_val)  # 用 tanh 将 logits 软裁剪到 [-30, 30]，防止注意力过度尖锐（Grok 特有技巧)

        mask = mask[:, :, None, :, :]  # 在掩码中插入头组维度，使其可广播到 attn_logits 的 [..., h, H, t, T] 形状

        if mask is not None:  # 若存在掩码
            if mask.ndim != attn_logits.ndim:  # 校验掩码与 logits 维度一致
                raise ValueError(
                    f"Mask dimensionality {mask.ndim} must match logits dimensionality "
                    f"{attn_logits.ndim} for {mask.shape}/{attn_logits.shape}."
                )
            attn_logits = jnp.where(mask, attn_logits, -1e30)  # 用掩码屏蔽：允许注意位置保留 logits，禁止位置设为 -1e30（softmax 后趋近 0）
        attn_weights = jax.nn.softmax(attn_logits).astype(query.dtype)  # [H, T', T]  # 对最后一维做 softmax 得到注意力权重，并转回 query 的 dtype

        # Weight the values by the attention and flatten the head vectors.
        # 用注意力权重加权 value，并展平头维度
        attn = jnp.einsum("...hHtT,...Thd->...thHd", attn_weights, value_heads)  # 注意力权重 @ value：按头组与组内头加权求和，形状 [B, T_q, kv_h, group_size, d]
        leading_dims = attn.shape[:2]  # 取前两维（批次、查询序列长度）
        attn = jnp.reshape(attn, (*leading_dims, -1))  # [T', H*V]  # 展平头维度：将 (kv_h, group_size, d) 合并为一维，形状 [B, T_q, num_q_heads*value_size]

        # Apply another projection to get the final embeddings.
        # 应用最终输出投影，得到最终嵌入
        final_projection = Linear(self.model_size, with_bias=False)  # 创建输出投影层，输出维度为 model_size，无偏置
        return MHAOutput(final_projection(attn))  # 对注意力输出做线性投影，封装为 MHAOutput 返回

    @hk.transparent  # 标记为透明模块：不创建额外的参数命名空间，参数直接归属于调用者
    def _linear_projection(  # 线性投影辅助函数：将输入投影到多头形式
        self,
        x: jax.Array,  # 输入张量，形状 [..., D]
        head_size: int,  # 每个头的维度
        num_heads: int,  # 头数量
        name: Optional[str] = None,  # 投影层名称
    ) -> jax.Array:  # 返回投影后张量，形状 [..., num_heads, head_size]
        y = Linear(num_heads * head_size, with_bias=False, name=name)(x)  # 线性投影到 [num_heads * head_size]，无偏置
        *leading_dims, _ = x.shape  # 取出除最后一维外的所有前导维度
        return y.reshape((*leading_dims, num_heads, head_size))  # 重塑为多头形式：[..., num_heads, head_size]


@dataclass  # 使用 dataclass 定义 MHABlock，封装一个注意力子层
class MHABlock(hk.Module):  # 多头注意力块：对输入做归一化后执行注意力
    """A MHA Block"""

    num_q_heads: int  # 查询头数
    num_kv_heads: int  # 键/值头数
    key_size: int  # 每个头的维度
    attn_output_multiplier: float = 1.0  # 注意力输出缩放因子

    @hk.transparent  # 透明模块：参数归属外层调用者
    def __call__(  # 前向：执行注意力块
        self,
        inputs: jax.Array,  # [B, T, D]  # 输入嵌入，形状 [批次, 序列, 嵌入维度]
        mask: jax.Array,  # [B, 1, T, T] or [B, 1, 1, T] or B[1, 1, 1, 1]  # 注意力掩码
    ) -> MHAOutput:  # 返回 MHAOutput
        _, _, model_size = inputs.shape  # 取输入的嵌入维度作为 model_size
        assert mask.ndim == 4, f"shape: {mask.shape}"  # 断言掩码为 4 维
        assert mask.shape[2] in {1, inputs.shape[1]}, str(mask.shape)  # 断言掩码查询序列维度合法
        assert mask.shape[3] in {1, inputs.shape[1]}, str(mask.shape)  # 断言掩码键序列维度合法
        side_input = inputs  # 将输入作为 side_input（此处 key/value 与 query 相同，即自注意力)

        def attn_block(query, key, value, mask) -> MHAOutput:  # 定义注意力子函数，构造 MultiHeadAttention 并前向
            return MultiHeadAttention(  # 实例化多头注意力模块
                num_q_heads=self.num_q_heads,  # 传入查询头数
                num_kv_heads=self.num_kv_heads,  # 传入键/值头数
                key_size=self.key_size,  # 传入头维度
                model_size=model_size,  # 传入输出维度（等于输入嵌入维度）
                attn_output_multiplier=self.attn_output_multiplier,  # 传入注意力缩放因子
            )(query, key, value, mask)  # 执行注意力前向计算

        attn_output = attn_block(inputs, side_input, side_input, mask)  # 以 inputs 作为 query/key/value 执行自注意力
        h_attn = attn_output.embeddings  # 取出注意力输出嵌入

        return MHAOutput(embeddings=h_attn)  # 封装为 MHAOutput 返回


@dataclass  # 使用 dataclass 定义 DenseBlock，封装前馈子层
class DenseBlock(hk.Module):  # 稠密前馈块（SwiGLU/GeGLU 风格 FFN）
    num_q_heads: int  # 查询头数（此处用于与层配置对齐，FFN 内部不直接使用）
    num_kv_heads: int  # 键/值头数（同上）
    key_size: int  # 头维度（同上）
    widening_factor: float = 4.0  # FFN 扩展系数

    @hk.transparent  # 透明模块：参数归属外层调用者
    def __call__(  # 前向：执行 SwiGLU 风格前馈
        self,
        inputs: jax.Array,  # [B, T, D]  # 输入嵌入，形状 [批次, 序列, 嵌入维度]
    ) -> jax.Array:  # [B, T, D]  # 返回与输入同形状的输出嵌入
        _, _, model_size = inputs.shape  # 取输入嵌入维度
        h_v = Linear(  # 门控分支 v：投影到 FFN 中间维度，作为门控的"值"路径
            ffn_size(model_size, self.widening_factor),  # 中间维度由 ffn_size 计算
            with_bias=False,  # 无偏置
            name="linear_v",  # 命名为 linear_v
        )(inputs)  # 输出形状 [B, T, ffn_size]
        h_w1 = jax.nn.gelu(  # 门控分支 w1：投影后经 GELU 激活，作为门控信号
            Linear(
                ffn_size(model_size, self.widening_factor),  # 中间维度同上
                with_bias=False,  # 无偏置
            )(inputs)  # 投影后形状 [B, T, ffn_size]
        )  # GELU 激活后形状不变
        h_dense = Linear(model_size, with_bias=False)(h_w1 * h_v)  # SwiGLU/GeGLU 核心：门控相乘 (w1 * v) 后投影回 model_size，形状 [B, T, D]

        return h_dense  # 返回前馈块输出


@dataclass  # 使用 dataclass 定义 DecoderLayer，封装一个 Transformer 解码层
class DecoderLayer(hk.Module):  # 单个 Transformer 解码层：注意力 + 前馈 + 归一化 + 残差
    """A transformer stack."""

    num_q_heads: int  # 查询头数
    num_kv_heads: int  # 键/值头数
    key_size: int  # 头维度
    num_layers: int  # 总层数（用于命名/配置传递）
    layer_index: Optional[int] = None  # 当前层索引（可选，用于调试或差异化命名）
    widening_factor: float = 4.0  # FFN 扩展系数
    name: Optional[str] = None  # 层名
    attn_output_multiplier: float = 1.0  # 注意力输出缩放因子

    def __call__(  # 前向：执行一个解码层（注意力 + 前馈 + 残差）
        self,
        inputs: jax.Array,  # [B, T, D]  # 输入嵌入，形状 [批次, 序列, 嵌入维度]
        mask: jax.Array,  # [B, 1, T, T] or [B, 1, 1, T]  # 注意力掩码
        padding_mask: Optional[jax.Array],  # 填充掩码（本实现未使用）
    ) -> DecoderOutput:  # 返回 DecoderOutput
        """Transforms input embedding sequences to output embedding sequences."""
        del padding_mask  # Unused.  # 显式删除未使用的 padding_mask，避免参数未使用告警

        def layer_norm(x):  # 定义本层使用的归一化函数：RMSNorm
            return hk_rms_norm(x)  # 调用 hk_rms_norm 对 x 做均方根归一化

        h = inputs  # 初始化残差流 h 为输入

        attn_output = MHABlock(  # 构造注意力块并对归一化后的 h 执行自注意力
            num_q_heads=self.num_q_heads,  # 传入查询头数
            num_kv_heads=self.num_kv_heads,  # 传入键/值头数
            key_size=self.key_size,  # 传入头维度
            attn_output_multiplier=self.attn_output_multiplier,  # 传入注意力缩放因子
        )(layer_norm(h), mask)  # 先对 h 做 RMSNorm，再执行注意力（Pre-Norm 架构）
        h_attn = attn_output.embeddings  # 取注意力输出嵌入

        h_attn = layer_norm(h_attn)  # 对注意力输出再做一次 RMSNorm（Grok 风格：子层输出归一化后再加到残差）
        h += h_attn  # 残差连接：将归一化后的注意力输出加回残差流 h

        def base_dense_block(h):  # 定义前馈子函数
            h = DenseBlock(  # 构造前馈块
                num_q_heads=self.num_q_heads,  # 传入头数（对齐配置）
                num_kv_heads=self.num_kv_heads,  # 传入键/值头数
                key_size=self.key_size,  # 传入头维度
                widening_factor=self.widening_factor,  # 传入扩展系数
            )(h)  # 对 h 执行前馈计算
            return h  # 返回前馈输出

        h_dense = base_dense_block(layer_norm(h))  # 先对残差流 h 做 RMSNorm，再执行前馈块（Pre-Norm）

        h_dense = layer_norm(h_dense)  # 对前馈输出再做 RMSNorm（与注意力分支一致的归一化模式）
        h += h_dense  # 残差连接：将归一化后的前馈输出加回残差流 h

        return DecoderOutput(  # 封装并返回解码层输出
            embeddings=h,  # 输出嵌入为更新后的残差流
        )


def layer_norm(x):  # 模块级归一化函数：对 x 应用 RMSNorm
    return hk_rms_norm(x)  # 调用 hk_rms_norm 完成归一化


@dataclass  # 使用 dataclass 定义 Transformer 主干模块
class Transformer(hk.Module):  # Transformer 主干：堆叠多个 DecoderLayer
    """A transformer stack."""

    num_q_heads: int  # 查询头数
    num_kv_heads: int  # 键/值头数
    key_size: int  # 头维度
    widening_factor: float  # FFN 扩展系数
    attn_output_multiplier: float  # 注意力输出缩放因子
    num_layers: int  # 解码层数量
    name: Optional[str] = None  # 主干名

    def __call__(  # 前向：执行整个 Transformer 主干
        self,
        embeddings: jax.Array,  # [B, T, D]  # 输入嵌入，形状 [批次, 序列, 嵌入维度]
        mask: jax.Array,  # [B, T]  # 填充掩码，形状 [批次, 序列]，True 表示有效位置
        candidate_start_offset: Optional[int] = None,  # 候选起始偏移：用于推荐系统推理的候选独立打分
    ) -> TransformerOutput:  # 返回 TransformerOutput
        """Transforms input embedding sequences to output embedding sequences.

        Args:
            embeddings: Input embeddings of shape [B, T, D]
            mask: Padding mask of shape [B, T], True for valid positions
            candidate_start_offset: If provided, positions >= this offset are treated as
                candidates that can only attend to positions before the offset (user+history)
                and themselves (self-attention), but not to other candidates.
                Used for recommendation system inference.

        Returns:
            TransformerOutput containing the output embeddings.
        """

        fprop_dtype = embeddings.dtype  # 记录前向 dtype，用于掩码类型转换
        _, seq_len, _ = embeddings.shape  # 取序列长度 seq_len
        padding_mask = mask.copy()  # 备份原始填充掩码，供各层使用（本实现各层未直接使用，但保留以备扩展）
        mask = mask[:, None, None, :]  # [B, H=1, T'=1, T]  # 将填充掩码扩展为 4 维，便于与注意力掩码相乘

        if candidate_start_offset is not None:  # 若指定了候选起始偏移（推荐系统推理模式）
            # Use recommendation system attention mask where candidates attend to
            # user+history and themselves, but not to other candidates
            # 使用推荐系统注意力掩码：候选可关注用户+历史及自身，但不能关注其他候选
            attn_mask = make_recsys_attn_mask(seq_len, candidate_start_offset, fprop_dtype)  # 构造推荐系统掩码，形状 [1, 1, seq_len, seq_len]
            mask = mask * attn_mask  # 将填充掩码与推荐系统掩码相乘，得到最终注意力掩码
        else:  # 否则使用标准因果掩码（自回归序列建模）
            # Standard causal mask for autoregressive sequence modelling
            # 标准因果掩码：下三角，保证位置 i 只能注意 <=i 的位置
            causal_mask = jnp.tril(jnp.ones((1, 1, seq_len, seq_len))).astype(
                fprop_dtype
            )  # [B=1, H=1, T, T]  # 构造下三角因果掩码并转为前向 dtype
            mask = mask * causal_mask  # [B, H=1, T, T]  # 与填充掩码相乘，得到最终因果+填充掩码

        h = embeddings  # 初始化隐藏状态 h 为输入

        def block(  # 定义单层前向函数：构造一个 DecoderLayer 并执行
            h,  # 当前隐藏状态
            mask,  # 注意力掩码
            padding_mask,  # 填充掩码
            layer_index: Optional[int] = None,  # 层索引
            widening_factor: Optional[int] = None,  # 可选的覆盖扩展系数
            name: Optional[str] = None,  # 层名
        ) -> DecoderOutput:  # 返回 DecoderOutput
            return DecoderLayer(  # 构造解码层
                num_q_heads=self.num_q_heads,  # 传入查询头数
                num_kv_heads=self.num_kv_heads,  # 传入键/值头数
                key_size=self.key_size,  # 传入头维度
                widening_factor=widening_factor or self.widening_factor,  # 优先用传入的扩展系数，否则用配置默认值
                num_layers=self.num_layers,  # 传入总层数
                attn_output_multiplier=self.attn_output_multiplier,  # 传入注意力缩放因子
                name=name,  # 传入层名（用于参数命名空间）
                layer_index=layer_index,  # 传入层索引
            )(h, mask, padding_mask)  # 执行该层前向

        for i in range(self.num_layers):  # 逐层堆叠：循环 num_layers 次
            decoder_output = block(  # 执行第 i 层
                h,  # 传入当前隐藏状态
                mask,  # 传入注意力掩码
                padding_mask,  # 传入填充掩码
                layer_index=i,  # 标记层索引
                name=f"decoder_layer_{i}",  # 每层独立命名，参数互不共享
            )
            h = decoder_output.embeddings  # 更新隐藏状态为该层输出

        return TransformerOutput(  # 封装并返回 Transformer 主干输出
            embeddings=h,  # 最终输出嵌入为最后一层的输出
        )
