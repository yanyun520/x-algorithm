# Copyright 2026 X.AI Corp.
# 版权声明：X.AI Corp. 2026 年版权所有
#
# Licensed under the Apache License, Version 2.0 (the "License");
# Apache 2.0 许可证头部声明，允许在遵守许可证的前提下使用本代码
# you may not use this file except in compliance with the License.
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import jax.numpy as jnp  # 导入 JAX 的 NumPy 兼容数组库，用于张量运算与 dtype 定义
import numpy as np  # 导入标准 NumPy，用于在测试中构造期望数组并做断言比较
import pytest  # 导入 pytest 测试框架，提供以 test_ 开头方法自动发现的测试风格

from grok import make_recsys_attn_mask  # 从被测模块 grok 导入候选隔离注意力掩码构造函数


class TestMakeRecsysAttnMask:
    """Tests for the make_recsys_attn_mask function."""

    # 测试类：集中验证 make_recsys_attn_mask 函数的各类掩码构造行为
    # 影响范围：确保推荐场景下 user/history 与 candidate 之间的注意力隔离规则正确

    def test_output_shape(self):
        """Test that the output has the correct shape [1, 1, seq_len, seq_len]."""
        # 验证输出形状：掩码应为 [1, 1, seq_len, seq_len]，兼容多头注意力的广播维度
        seq_len = 10  # 序列总长度：包含 user + history + candidate 的位置数
        candidate_start_offset = 5  # 候选起始偏移：位置 0..4 为 user/history，5..9 为 candidate

        mask = make_recsys_attn_mask(seq_len, candidate_start_offset)  # 调用被测函数生成注意力掩码

        assert mask.shape == (1, 1, seq_len, seq_len)  # 断言掩码形状符合 [batch, head, q, k] 的广播格式

    def test_user_history_has_causal_attention(self):
        """Test that user+history positions (before candidate_start_offset) have causal attention."""
        # 验证 user+history 区域采用因果注意力：每个位置只能看到自身及之前的位置
        seq_len = 8  # 序列长度 8，便于枚举所有位置组合
        candidate_start_offset = 5  # 前 5 个位置属于 user/history 区域

        mask = make_recsys_attn_mask(seq_len, candidate_start_offset)  # 生成掩码
        mask_2d = mask[0, 0]  # 去掉前两个广播维度，得到 [seq_len, seq_len] 的二维矩阵便于逐元素检查

        for i in range(candidate_start_offset):  # 遍历 user/history 区域中的每个 query 位置 i
            for j in range(candidate_start_offset):  # 遍历 user/history 区域中的每个 key 位置 j
                if j <= i:  # 因果规则：key 位置不超过 query 位置时允许关注
                    assert mask_2d[i, j] == 1, f"Position {i} should attend to position {j}"  # 断言允许关注
                else:  # 否则属于"未来"位置
                    assert mask_2d[i, j] == 0, (  # 断言禁止关注未来，保证自回归性
                        f"Position {i} should NOT attend to future position {j}"
                    )

    def test_candidates_attend_to_user_history(self):
        """Test that candidates can attend to all user+history positions."""
        # 验证候选位置可以关注到全部 user/history 位置（这是推荐场景的核心信息流向）
        seq_len = 8  # 序列长度 8
        candidate_start_offset = 5  # candidate 位于位置 5..7

        mask = make_recsys_attn_mask(seq_len, candidate_start_offset)  # 生成掩码
        mask_2d = mask[0, 0]  # 提取二维矩阵

        for candidate_pos in range(candidate_start_offset, seq_len):  # 遍历每个 candidate 位置
            for history_pos in range(candidate_start_offset):  # 遍历每个 user/history 位置
                assert mask_2d[candidate_pos, history_pos] == 1, (  # 断言 candidate 能看到全部 history
                    f"Candidate at {candidate_pos} should attend to user+history at {history_pos}"
                )

    def test_candidates_attend_to_themselves(self):
        """Test that candidates can attend to themselves (self-attention)."""
        # 验证候选位置的自注意力：candidate 必须能看到自身（对角线为 1）
        seq_len = 8  # 序列长度 8
        candidate_start_offset = 5  # candidate 区域 5..7

        mask = make_recsys_attn_mask(seq_len, candidate_start_offset)  # 生成掩码
        mask_2d = mask[0, 0]  # 提取二维矩阵

        for candidate_pos in range(candidate_start_offset, seq_len):  # 遍历每个 candidate 位置
            assert mask_2d[candidate_pos, candidate_pos] == 1, (  # 断言对角线元素为 1，即允许自注意力
                f"Candidate at {candidate_pos} should attend to itself"
            )

    def test_candidates_do_not_attend_to_other_candidates(self):
        """Test that candidates cannot attend to other candidates."""
        # 验证候选隔离：candidate 之间不能互相关注（避免信息泄漏，保证各候选独立打分）
        seq_len = 8  # 序列长度 8
        candidate_start_offset = 5  # candidate 区域 5..7

        mask = make_recsys_attn_mask(seq_len, candidate_start_offset)  # 生成掩码
        mask_2d = mask[0, 0]  # 提取二维矩阵

        for query_pos in range(candidate_start_offset, seq_len):  # 遍历每个 candidate 作为 query
            for key_pos in range(candidate_start_offset, seq_len):  # 遍历每个 candidate 作为 key
                if query_pos != key_pos:  # 排除自身（自注意力已在其它测试中验证）
                    assert mask_2d[query_pos, key_pos] == 0, (  # 断言不同 candidate 之间掩码为 0
                        f"Candidate at {query_pos} should NOT attend to candidate at {key_pos}"
                    )

    def test_full_mask_structure(self):
        """Test the complete mask structure with a small example."""
        # 验证完整掩码结构：用一个小例子整体检查 user/history 因果 + candidate 隔离的复合模式
        # Sequence: [user, h1, h2, c1, c2, c3]
        # 序列含义：user 位置 + 2 个 history + 3 个 candidate
        # Positions:  0     1   2   3   4   5
        # 位置编号对应上述元素

        seq_len = 6  # 序列长度 6
        candidate_start_offset = 3  # 前 3 个为 user/history，后 3 个为 candidate

        mask = make_recsys_attn_mask(seq_len, candidate_start_offset)  # 生成掩码
        mask_2d = mask[0, 0]  # 提取二维矩阵

        # Expected mask structure:
        # 期望的掩码结构（行=query，列=key，1=可关注，0=不可关注）
        # Query positions are rows, key positions are columns
        # 1 = can attend, 0 = cannot attend
        #
        #        Keys:  u   h1  h2  c1  c2  c3
        # Query u   :   1   0   0   0   0   0
        # Query h1  :   1   1   0   0   0   0
        # Query h2  :   1   1   1   0   0   0
        # Query c1  :   1   1   1   1   0   0   <- c1 attends to user+history + self
        # Query c2  :   1   1   1   0   1   0   <- c2 attends to user+history + self
        # Query c3  :   1   1   1   0   0   1   <- c3 attends to user+history + self

        expected = np.array(  # 构造期望的掩码矩阵
            [
                [1, 0, 0, 0, 0, 0],  # user
                [1, 1, 0, 0, 0, 0],  # h1
                [1, 1, 1, 0, 0, 0],  # h2
                [1, 1, 1, 1, 0, 0],  # c1: user+history + self
                [1, 1, 1, 0, 1, 0],  # c2: user+history + self
                [1, 1, 1, 0, 0, 1],  # c3: user+history + self
            ],
            dtype=np.float32,  # 指定 float32 类型，与默认输出 dtype 保持一致
        )

        np.testing.assert_array_equal(  # 用 NumPy 断言逐元素完全相等
            np.array(mask_2d),  # 将 JAX 数组转换为 NumPy 数组以便比较
            expected,  # 期望矩阵
            err_msg="Full mask structure does not match expected pattern",  # 失败时的错误信息
        )

    def test_dtype_preserved(self):
        """Test that the specified dtype is used."""
        # 验证 dtype 参数生效：分别用 float32 与 float16 构造掩码并检查类型
        seq_len = 5  # 序列长度 5
        candidate_start_offset = 3  # candidate 起始偏移 3

        mask_f32 = make_recsys_attn_mask(seq_len, candidate_start_offset, dtype=jnp.float32)  # 用 float32 生成
        mask_f16 = make_recsys_attn_mask(seq_len, candidate_start_offset, dtype=jnp.float16)  # 用 float16 生成

        assert mask_f32.dtype == jnp.float32  # 断言 float32 掩码的 dtype 正确
        assert mask_f16.dtype == jnp.float16  # 断言 float16 掩码的 dtype 正确

    def test_single_candidate(self):
        """Test edge case with a single candidate."""
        # 边界场景：仅有一个 candidate 时的掩码结构
        seq_len = 4  # 序列长度 4
        candidate_start_offset = 3  # 前 3 个为 user/history，仅位置 3 为 candidate

        mask = make_recsys_attn_mask(seq_len, candidate_start_offset)  # 生成掩码
        mask_2d = mask[0, 0]  # 提取二维矩阵

        expected = np.array(  # 构造期望矩阵：history 因果 + 单个 candidate 自注意力
            [
                [1, 0, 0, 0],
                [1, 1, 0, 0],
                [1, 1, 1, 0],
                [1, 1, 1, 1],
            ],
            dtype=np.float32,  # 指定 float32
        )

        np.testing.assert_array_equal(np.array(mask_2d), expected)  # 断言与期望矩阵完全一致

    def test_all_candidates(self):
        """Test edge case where all positions except first are candidates."""
        # 边界场景：除第一个位置外全部为 candidate（只有 1 个 user，其余都是 candidate）
        seq_len = 4  # 序列长度 4
        candidate_start_offset = 1  # 仅位置 0 为 user，位置 1..3 均为 candidate

        mask = make_recsys_attn_mask(seq_len, candidate_start_offset)  # 生成掩码
        mask_2d = mask[0, 0]  # 提取二维矩阵

        expected = np.array(  # 构造期望矩阵：user 自身 + 各 candidate 仅看到 user 和自身
            [
                [1, 0, 0, 0],  # user
                [1, 1, 0, 0],  # c1: user + self
                [1, 0, 1, 0],  # c2: user + self
                [1, 0, 0, 1],  # c3: user + self
            ],
            dtype=np.float32,  # 指定 float32
        )

        np.testing.assert_array_equal(np.array(mask_2d), expected)  # 断言与期望矩阵完全一致


if __name__ == "__main__":  # 当作为脚本直接运行时
    pytest.main([__file__, "-v"])  # 以详细模式（-v）运行本文件中的所有测试
