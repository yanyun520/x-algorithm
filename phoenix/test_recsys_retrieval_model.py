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

"""Tests for the Phoenix Retrieval Model."""
# 模块 docstring：本文件是 Phoenix 检索模型（候选塔 + 完整检索模型 + 推理运行器）的单元测试

import unittest
# 导入 Python 标准库 unittest 测试框架，提供 TestCase 基类、setUp 夹具与 assert* 断言方法
# 作用：与 pytest 不同，unittest 采用"类继承 + test_ 方法命名"来组织和自动发现测试用例

import haiku as hk
# 导入 DeepMind 的 Haiku 库：将普通 Python 函数转换为可参数化的 JAX 模块
# 关键 API：hk.transform 做函数式转换，hk.without_apply_rng 去掉 apply 阶段的随机数依赖

import jax
# 导入 JAX 核心库，提供 jax.random.PRNGKey / jax.random.normal 等随机数与底层数值基础设施

import jax.numpy as jnp
# 导入 JAX 的 NumPy 兼容数组库，用于张量运算（如 jnp.sqrt、jnp.sum、jnp.all、jnp.ones）

import numpy as np
# 导入标准 NumPy，用于 np.testing 数值近似断言以及把 JAX 数组转成 np.array 做比较

from grok import TransformerConfig
# 从 grok 模块导入 Transformer 配置类：定义内部 Transformer 的层数、头数、宽化因子等超参数

from recsys_model import HashConfig
# 从 recsys_model 导入哈希配置类：定义 user/item/author 三个维度各自使用的哈希桶数量

from recsys_retrieval_model import (
    CandidateTower,
    PhoenixRetrievalModelConfig,
)
# 从被测模块 recsys_retrieval_model 导入候选塔 CandidateTower 与检索模型配置类
# 作用：这两个是本文件直接被测的核心对象

from runners import (
    RecsysRetrievalInferenceRunner,
    RetrievalModelRunner,
    create_example_batch,
    create_example_corpus,
)
# 从 runners 模块导入推理运行器、底层模型运行器，以及构造测试批次/语料库的工具函数
# create_example_batch 生成 batch + embeddings，create_example_corpus 生成语料库嵌入 + 帖子 id


class TestCandidateTower(unittest.TestCase):
    """Tests for the CandidateTower module."""
    # 测试类：验证候选塔 CandidateTower 的输出形状与 L2 归一化性质
    # 继承 unittest.TestCase，使框架能自动发现并执行类中所有 test_ 开头的方法

    def test_candidate_tower_output_shape(self):
        """Test that candidate tower produces correct output shape."""
        # 测试方法：验证候选塔把输入聚合后输出的形状为 (batch_size, num_candidates, emb_size)
        emb_size = 64  # 每个候选 token 的嵌入维度：64，决定输出向量的最后一维大小
        batch_size = 4  # 批次大小：一次处理 4 个样本
        num_candidates = 8  # 每个样本包含的候选数量：8 个候选
        num_hashes = 4  # 每个候选被哈希拆分成的子向量数量（multi-hash 编码）

        def forward(x):
            # 定义前向函数：这是 Haiku 的标准模块定义方式
            tower = CandidateTower(emb_size=emb_size)  # 实例化候选塔，指定嵌入维度 64
            return tower(x)  # 对输入 x 执行候选塔的前向计算并返回输出

        forward_fn = hk.without_apply_rng(hk.transform(forward))
        # 将 forward 纯函数转换为 Haiku 模块：hk.transform 生成 init/apply 两个函数
        # hk.without_apply_rng 表示 apply（推理）阶段不依赖随机数，从而去掉 rng 参数
        # 结果：得到纯函数式的 forward_fn.init 与 forward_fn.apply 两个入口

        rng = jax.random.PRNGKey(0)  # 用固定种子 0 创建伪随机数生成器密钥，保证测试结果可复现
        x = jax.random.normal(rng, (batch_size, num_candidates, num_hashes, emb_size))
        # 用上述 rng 生成标准正态分布随机输入张量，形状 (4, 8, 4, 64)
        # 作用：模拟候选塔输入——每个候选由 4 个哈希子向量拼接表示

        params = forward_fn.init(rng, x)  # 用随机种子初始化模型参数（对输入做一次前向以推导参数结构）
        output = forward_fn.apply(params, x)  # 用初始化得到的参数执行真实推理，得到模型输出

        self.assertEqual(output.shape, (batch_size, num_candidates, emb_size))
        # 断言输出形状为 (4, 8, 64)：候选塔把 (..., num_hashes, emb_size) 聚合成了 emb_size 维向量

    def test_candidate_tower_normalized(self):
        """Test that candidate tower output is L2 normalized."""
        # 测试方法：验证候选塔输出沿最后一维做了 L2 归一化（每个向量范数为 1）
        emb_size = 64  # 嵌入维度 64
        batch_size = 4  # 批次大小 4
        num_candidates = 8  # 候选数量 8
        num_hashes = 4  # 哈希子向量数量 4

        def forward(x):
            # 定义前向函数：构造候选塔
            tower = CandidateTower(emb_size=emb_size)  # 实例化候选塔
            return tower(x)  # 前向计算

        forward_fn = hk.without_apply_rng(hk.transform(forward))
        # 函数式转换：去掉 apply 阶段的 rng 依赖，得到 init/apply 两个纯函数

        rng = jax.random.PRNGKey(0)  # 固定种子随机密钥，保证可复现
        x = jax.random.normal(rng, (batch_size, num_candidates, num_hashes, emb_size))
        # 生成随机输入 (4, 8, 4, 64)

        params = forward_fn.init(rng, x)  # 初始化参数
        output = forward_fn.apply(params, x)  # 推理得到输出

        norms = jnp.sqrt(jnp.sum(output**2, axis=-1))
        # 沿最后一维（emb_size）计算每个输出向量的 L2 范数：先逐元素平方再求和再开方
        # 结果：得到形状为 (4, 8) 的范数矩阵，每个元素对应一个候选向量的范数
        np.testing.assert_array_almost_equal(norms, jnp.ones_like(norms), decimal=5)
        # 断言范数矩阵与全 1 矩阵近似相等（容差 5 位小数），即每个向量范数≈1，证明做了 L2 归一化

    def test_candidate_tower_mean_pooling(self):
        """Test candidate tower with mean pooling (no linear projection)."""
        # 测试方法：验证候选塔使用均值池化（无线性投影）路径时仍满足形状与归一化要求
        emb_size = 64  # 嵌入维度 64
        batch_size = 4  # 批次大小 4
        num_candidates = 8  # 候选数量 8
        num_hashes = 4  # 哈希子向量数量 4

        def forward(x):
            # 定义前向函数：构造候选塔
            tower = CandidateTower(emb_size=emb_size)  # 实例化候选塔（默认走均值池化路径）
            return tower(x)  # 前向计算

        forward_fn = hk.without_apply_rng(hk.transform(forward))
        # 函数式转换，去掉 rng 依赖

        rng = jax.random.PRNGKey(0)  # 固定种子随机密钥
        x = jax.random.normal(rng, (batch_size, num_candidates, num_hashes, emb_size))
        # 生成随机输入 (4, 8, 4, 64)

        params = forward_fn.init(rng, x)  # 初始化参数
        output = forward_fn.apply(params, x)  # 推理得到输出

        self.assertEqual(output.shape, (batch_size, num_candidates, emb_size))
        # 断言均值池化后的输出形状仍为 (4, 8, 64)

        norms = jnp.sqrt(jnp.sum(output**2, axis=-1))
        # 计算每个候选向量的 L2 范数
        np.testing.assert_array_almost_equal(norms, jnp.ones_like(norms), decimal=5)
        # 断言均值池化后的输出同样做了 L2 归一化，范数≈1


class TestPhoenixRetrievalModel(unittest.TestCase):
    """Tests for the full Phoenix Retrieval Model."""

    def setUp(self):
        """Set up test fixtures."""
        # 测试夹具：每个测试方法执行前由 unittest 自动调用，初始化共享的维度与配置对象
        # 作用：避免在每个测试方法中重复书写相同配置，并保证各测试之间相互独立
        self.emb_size = 64  # 嵌入维度 64
        self.history_seq_len = 16  # 用户历史序列长度 16
        self.candidate_seq_len = 8  # 候选序列长度 8
        self.batch_size = 2  # 批次大小 2
        self.num_actions = 19  # 用户行为动作种类数量 19（用于动作序列的 one-hot 维度）
        self.corpus_size = 100  # 语料库规模 100（检索的候选池大小）
        self.top_k = 10  # top-k 检索返回的前 k 个结果数量 10

        self.hash_config = HashConfig(
            num_user_hashes=2,
            num_item_hashes=2,
            num_author_hashes=2,
        )
        # 构造哈希配置：user/item/author 三个维度各使用 2 个哈希桶
        # 作用：多哈希编码用于把稀疏高维 id 映射为低维稠密向量

        self.config = PhoenixRetrievalModelConfig(
            emb_size=self.emb_size,
            history_seq_len=self.history_seq_len,
            candidate_seq_len=self.candidate_seq_len,
            hash_config=self.hash_config,
            product_surface_vocab_size=16,
            model=TransformerConfig(
                emb_size=self.emb_size,
                widening_factor=2,
                key_size=32,
                num_q_heads=2,
                num_kv_heads=2,
                num_layers=1,
                attn_output_multiplier=0.125,
            ),
        )
        # 构造完整检索模型配置：嵌入维度、序列长度、哈希配置、产品面词表大小 16，
        # 以及内部 Transformer（宽化因子 2、key 尺寸 32、2 个 Q 头、2 个 KV 头、1 层、注意力输出缩放 0.125）
        # 作用：该配置通过 make() 方法实例化出被测模型

    def _create_test_batch(self) -> tuple:
        """Create test batch and embeddings."""
        # 辅助方法：构造测试用的批次数据和嵌入，返回 (batch, embeddings) 元组
        return create_example_batch(
            batch_size=self.batch_size,
            emb_size=self.emb_size,
            history_len=self.history_seq_len,
            num_candidates=self.candidate_seq_len,
            num_actions=self.num_actions,
            num_user_hashes=self.hash_config.num_user_hashes,
            num_item_hashes=self.hash_config.num_item_hashes,
            num_author_hashes=self.hash_config.num_author_hashes,
            product_surface_vocab_size=16,
        )
        # 调用工具函数生成模拟输入批次：包含历史行为、候选、哈希 id 等字段及其对应嵌入
        # 作用：为各测试提供结构一致、数值随机的输入数据

    def _create_test_corpus(self):
        """Create test corpus embeddings."""
        # 辅助方法：构造测试语料库嵌入
        return create_example_corpus(self.corpus_size, self.emb_size)
        # 生成规模为 100、维度为 64 的语料库嵌入（同时返回语料帖子 id）
        # 作用：作为检索的候选池，模型在其上做相似度检索

    def test_model_forward(self):
        """Test model forward pass produces correct output shapes."""
        # 测试方法：验证模型完整前向过程输出的三个字段（用户表示、top-k 索引、top-k 分数）形状都正确

        def forward(batch, embeddings, corpus_embeddings, top_k):
            # 定义前向函数：接收批次、嵌入、语料库嵌入与 top_k 四个参数
            model = self.config.make()  # 用配置实例化完整检索模型
            return model(batch, embeddings, corpus_embeddings, top_k)  # 执行前向并返回输出

        forward_fn = hk.without_apply_rng(hk.transform(forward))
        # 函数式转换：去掉 apply 阶段 rng，得到 init/apply 两个纯函数

        batch, embeddings = self._create_test_batch()  # 构造测试批次与嵌入
        corpus_embeddings, _ = self._create_test_corpus()  # 构造语料库嵌入（忽略返回的 id）

        rng = jax.random.PRNGKey(0)  # 固定种子随机密钥
        params = forward_fn.init(rng, batch, embeddings, corpus_embeddings, self.top_k)
        # 用随机种子初始化模型参数，输入包括批次、嵌入、语料库嵌入与 top_k=10
        output = forward_fn.apply(params, batch, embeddings, corpus_embeddings, self.top_k)
        # 用参数执行推理，得到最终输出（含用户表示、top-k 索引与分数）

        self.assertEqual(output.user_representation.shape, (self.batch_size, self.emb_size))
        # 断言用户表示形状为 (2, 64)：每个样本得到一个 64 维用户向量
        self.assertEqual(output.top_k_indices.shape, (self.batch_size, self.top_k))
        # 断言 top-k 索引形状为 (2, 10)：每个样本返回 10 个候选索引
        self.assertEqual(output.top_k_scores.shape, (self.batch_size, self.top_k))
        # 断言 top-k 分数形状为 (2, 10)：每个样本返回 10 个相似度分数

    def test_user_representation_normalized(self):
        """Test that user representations are L2 normalized."""
        # 测试方法：验证模型输出的用户表示做了 L2 归一化（范数为 1）

        def forward(batch, embeddings, corpus_embeddings, top_k):
            # 定义前向函数：完整模型调用
            model = self.config.make()  # 实例化模型
            return model(batch, embeddings, corpus_embeddings, top_k)  # 前向计算

        forward_fn = hk.without_apply_rng(hk.transform(forward))
        # 函数式转换，去掉 rng 依赖

        batch, embeddings = self._create_test_batch()  # 构造测试批次
        corpus_embeddings, _ = self._create_test_corpus()  # 构造语料库嵌入

        rng = jax.random.PRNGKey(0)  # 固定种子
        params = forward_fn.init(rng, batch, embeddings, corpus_embeddings, self.top_k)
        # 初始化参数
        output = forward_fn.apply(params, batch, embeddings, corpus_embeddings, self.top_k)
        # 推理得到输出

        norms = jnp.sqrt(jnp.sum(output.user_representation**2, axis=-1))
        # 沿最后一维计算每个用户表示的 L2 范数，得到形状 (2,) 的范数向量
        np.testing.assert_array_almost_equal(norms, jnp.ones(self.batch_size), decimal=5)
        # 断言范数向量≈全 1（容差 5 位小数），证明用户表示已做 L2 归一化

    def test_candidate_representation_normalized(self):
        """Test that candidate representations from build_candidate_representation are L2 normalized."""
        # 测试方法：验证由 build_candidate_representation 构造的候选表示做了 L2 归一化

        def forward(batch, embeddings):
            # 定义前向函数：只调用候选表示构建方法（不涉及语料库检索）
            model = self.config.make()  # 实例化模型
            cand_rep, _ = model.build_candidate_representation(batch, embeddings)
            # 构建候选表示，返回候选表示及其它信息（此处用 _ 忽略第二个返回值）
            return cand_rep  # 返回候选表示

        forward_fn = hk.without_apply_rng(hk.transform(forward))
        # 函数式转换，去掉 rng 依赖

        batch, embeddings = self._create_test_batch()  # 构造测试批次
        # 注意：本测试不需要语料库，只验证候选表示构建阶段的归一化

        rng = jax.random.PRNGKey(0)  # 固定种子
        params = forward_fn.init(rng, batch, embeddings)  # 初始化参数（仅需 batch 与 embeddings）
        cand_rep = forward_fn.apply(params, batch, embeddings)  # 推理得到候选表示

        norms = jnp.sqrt(jnp.sum(cand_rep**2, axis=-1))
        # 沿最后一维计算每个候选表示的 L2 范数，得到形状 (2, 8) 的范数矩阵
        np.testing.assert_array_almost_equal(
            norms, jnp.ones((self.batch_size, self.candidate_seq_len)), decimal=5
        )
        # 断言范数矩阵≈全 1 矩阵（形状 (2, 8)，容差 5 位小数），证明候选表示已做 L2 归一化

    def test_retrieve_top_k(self):
        """Test top-k retrieval through __call__."""
        # 测试方法：通过 __call__ 验证 top-k 检索的索引范围合法性与分数降序单调性

        def forward(batch, embeddings, corpus_embeddings, top_k):
            # 定义前向函数：完整模型调用
            model = self.config.make()  # 实例化模型
            return model(batch, embeddings, corpus_embeddings, top_k)  # 前向计算

        forward_fn = hk.without_apply_rng(hk.transform(forward))
        # 函数式转换，去掉 rng 依赖

        batch, embeddings = self._create_test_batch()  # 构造测试批次
        corpus_embeddings, _ = self._create_test_corpus()  # 构造语料库嵌入

        rng = jax.random.PRNGKey(0)  # 固定种子
        params = forward_fn.init(rng, batch, embeddings, corpus_embeddings, self.top_k)
        # 初始化参数
        output = forward_fn.apply(params, batch, embeddings, corpus_embeddings, self.top_k)
        # 推理得到 top-k 输出

        self.assertEqual(output.top_k_indices.shape, (self.batch_size, self.top_k))
        # 断言 top-k 索引形状为 (2, 10)
        self.assertEqual(output.top_k_scores.shape, (self.batch_size, self.top_k))
        # 断言 top-k 分数形状为 (2, 10)

        self.assertTrue(jnp.all(output.top_k_indices >= 0))
        # 断言所有 top-k 索引都 >= 0：索引不能为负数
        self.assertTrue(jnp.all(output.top_k_indices < self.corpus_size))
        # 断言所有 top-k 索引都 < 100：索引必须落在语料库有效范围内

        for b in range(self.batch_size):
            # 遍历批次中的每个样本
            scores = np.array(output.top_k_scores[b])  # 取出第 b 个样本的 top-k 分数并转成 NumPy 数组
            self.assertTrue(np.all(scores[:-1] >= scores[1:]))  # 断言分数单调非增（降序排列）
        # 作用：验证 top-k 结果是按相似度从高到低排序返回的


class TestRetrievalInferenceRunner(unittest.TestCase):
    """Tests for the retrieval inference runner."""

    def setUp(self):
        """Set up test fixtures."""
        # 测试夹具：初始化维度、哈希配置与模型配置（结构与上一测试类一致，但无需语料库相关字段）
        self.emb_size = 64  # 嵌入维度 64
        self.history_seq_len = 16  # 历史序列长度 16
        self.candidate_seq_len = 8  # 候选序列长度 8
        self.batch_size = 2  # 批次大小 2
        self.num_actions = 19  # 动作种类数量 19

        self.hash_config = HashConfig(
            num_user_hashes=2,
            num_item_hashes=2,
            num_author_hashes=2,
        )
        # 构造哈希配置：user/item/author 各 2 个哈希桶

        self.config = PhoenixRetrievalModelConfig(
            emb_size=self.emb_size,
            history_seq_len=self.history_seq_len,
            candidate_seq_len=self.candidate_seq_len,
            hash_config=self.hash_config,
            product_surface_vocab_size=16,
            model=TransformerConfig(
                emb_size=self.emb_size,
                widening_factor=2,
                key_size=32,
                num_q_heads=2,
                num_kv_heads=2,
                num_layers=1,
                attn_output_multiplier=0.125,
            ),
        )
        # 构造完整检索模型配置（含内部 Transformer 超参数），供运行器内部实例化模型使用

    def test_runner_initialization(self):
        """Test that runner initializes correctly."""
        # 测试方法：验证推理运行器能正确初始化并生成模型参数
        runner = RecsysRetrievalInferenceRunner(
            runner=RetrievalModelRunner(
                model=self.config,
                bs_per_device=0.125,
            ),
            name="test_retrieval",
        )
        # 构造推理运行器：内部用 RetrievalModelRunner 包装模型配置，
        # bs_per_device=0.125 表示每设备批次大小（用于参数分片/批处理策略），name 指定运行器名称
        # 作用：这是被测对象，封装了初始化、用户编码与检索能力

        runner.initialize()  # 调用初始化方法：触发参数初始化等准备工作

        self.assertIsNotNone(runner.params)  # 断言初始化后 runner.params 不为 None，即参数已成功生成

    def test_runner_encode_user(self):
        """Test user encoding through runner."""
        # 测试方法：通过运行器编码用户，验证得到的用户表示形状正确
        runner = RecsysRetrievalInferenceRunner(
            runner=RetrievalModelRunner(
                model=self.config,
                bs_per_device=0.125,
            ),
            name="test_retrieval",
        )
        # 构造推理运行器（同前）
        runner.initialize()  # 初始化运行器，生成参数

        batch, embeddings = create_example_batch(
            batch_size=self.batch_size,
            emb_size=self.emb_size,
            history_len=self.history_seq_len,
            num_candidates=self.candidate_seq_len,
            num_actions=self.num_actions,
            num_user_hashes=self.hash_config.num_user_hashes,
            num_item_hashes=self.hash_config.num_item_hashes,
            num_author_hashes=self.hash_config.num_author_hashes,
        )
        # 直接调用工具函数构造测试批次与嵌入
        # 注意：此处未传 product_surface_vocab_size，与夹具中的完整调用略有差异

        user_rep = runner.encode_user(batch, embeddings)  # 调用运行器编码用户，得到用户表示

        self.assertEqual(user_rep.shape, (self.batch_size, self.emb_size))
        # 断言用户表示形状为 (2, 64)

    def test_runner_retrieve(self):
        """Test retrieval through runner."""
        # 测试方法：通过运行器执行检索，验证返回的用户表示与 top-k 结果形状正确
        runner = RecsysRetrievalInferenceRunner(
            runner=RetrievalModelRunner(
                model=self.config,
                bs_per_device=0.125,
            ),
            name="test_retrieval",
        )
        # 构造推理运行器
        runner.initialize()  # 初始化运行器

        batch, embeddings = create_example_batch(
            batch_size=self.batch_size,
            emb_size=self.emb_size,
            history_len=self.history_seq_len,
            num_candidates=self.candidate_seq_len,
            num_actions=self.num_actions,
            num_user_hashes=self.hash_config.num_user_hashes,
            num_item_hashes=self.hash_config.num_item_hashes,
            num_author_hashes=self.hash_config.num_author_hashes,
        )
        # 构造测试批次与嵌入

        corpus_size = 100  # 语料库规模 100
        corpus_embeddings, corpus_post_ids = create_example_corpus(corpus_size, self.emb_size)
        # 构造语料库嵌入及其帖子 id，作为检索的候选池
        runner.set_corpus(corpus_embeddings, corpus_post_ids)  # 把语料库设置到运行器中，供检索使用

        top_k = 10  # 设置 top-k 为 10
        output = runner.retrieve(batch, embeddings, top_k=top_k)  # 执行检索，得到输出

        self.assertEqual(output.user_representation.shape, (self.batch_size, self.emb_size))
        # 断言用户表示形状为 (2, 64)
        self.assertEqual(output.top_k_indices.shape, (self.batch_size, top_k))
        # 断言 top-k 索引形状为 (2, 10)
        self.assertEqual(output.top_k_scores.shape, (self.batch_size, top_k))
        # 断言 top-k 分数形状为 (2, 10)


if __name__ == "__main__":
    # 当本文件作为脚本直接运行时进入该分支
    unittest.main()
    # 调用 unittest 的测试运行入口，自动发现并执行所有继承 TestCase 的测试类
