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

import logging  # 导入 Python 标准库 logging 模块，用于配置日志级别并输出运行日志

import numpy as np  # 导入 NumPy 数值计算库并别名 np，用于数组创建与向量化/矩阵运算

from grok import TransformerConfig  # 从 grok 模块导入 TransformerConfig 配置类，用于定义 Transformer 主干网络的结构超参数
from recsys_model import HashConfig  # 从 recsys_model 导入多哈希特征配置类 HashConfig
from recsys_retrieval_model import PhoenixRetrievalModelConfig  # 从 recsys_retrieval_model 导入检索模型总配置类 PhoenixRetrievalModelConfig
from runners import (  # 从 runners 模块导入检索演示所需的多个组件（分多行导入）
    RecsysRetrievalInferenceRunner,  # 检索模型的推理运行器，封装 initialize / set_corpus / retrieve 等流程
    RetrievalModelRunner,  # 检索模型的底层运行器，负责实际的前向计算
    create_example_batch,  # 构造示例用户批次的函数
    create_example_corpus,  # 构造示例语料库（corpus）的函数
    ACTIONS,  # 交互动作列表
)


def main():  # 定义主函数 main，作为检索模型端到端推理演示的执行入口
    # Model configuration - same architecture as Phoenix ranker
    emb_size = 128  # Embedding dimension；设置 embedding 向量维度为 128，与排序模型保持一致
    num_actions = len(ACTIONS)  # Number of explicit engagement actions；统计显式交互动作总数，作为多标签输出的维度
    history_seq_len = 32  # Max history length；设置用户历史行为序列的最大长度 32
    candidate_seq_len = 8  # Max candidates per batch (for training)；训练时每批候选数量上限 8（此处仅用于示例批次构造）

    # Hash configuration
    hash_config = HashConfig(  # 构造多哈希配置对象，用于把高维稀疏的 ID 类特征映射到低维稠密空间
        num_user_hashes=2,  # 用户 ID 使用 2 个不同的哈希函数
        num_item_hashes=2,  # 帖子（物品）ID 使用 2 个不同的哈希函数
        num_author_hashes=2,  # 作者 ID 使用 2 个不同的哈希函数
    )

    # Configure the retrieval model - uses same transformer as Phoenix
    retrieval_model_config = PhoenixRetrievalModelConfig(  # 构造检索模型的总配置对象
        emb_size=emb_size,  # 传入 embedding 维度，保证全局一致
        history_seq_len=history_seq_len,  # 传入用户历史序列最大长度
        candidate_seq_len=candidate_seq_len,  # 传入候选序列最大长度
        hash_config=hash_config,  # 挂载多哈希配置，供特征编码层使用
        product_surface_vocab_size=16,  # 设置产品界面词表大小为 16，作为该分类特征 embedding 表的行数
        model=TransformerConfig(  # 传入与 Phoenix 排序模型相同的 Transformer 主干配置
            emb_size=emb_size,  # Transformer 内部 embedding 维度
            widening_factor=2,  # SwiGLU 前馈层宽度放大倍数：中间层维度 = key_size * widening_factor，取 2 表示放大 2 倍
            key_size=64,  # 每个注意力头的 query/key 维度设为 64
            num_q_heads=2,  # query 注意力头数设为 2
            num_kv_heads=2,  # key/value 注意力头数设为 2；与 query 头数相等时即为标准多头注意力(MHA)
            num_layers=2,  # Transformer 堆叠层数设为 2
            attn_output_multiplier=0.125,  # 注意力输出缩放系数：对注意力输出乘以 0.125，用于稳定训练
        ),
    )

    # Create inference runner
    inference_runner = RecsysRetrievalInferenceRunner(  # 创建检索模型的推理运行器，封装初始化、语料装载与检索流程
        runner=RetrievalModelRunner(  # 构造底层检索模型运行器，负责实际的前向计算
            model=retrieval_model_config,  # 将检索模型配置传入运行器
            bs_per_device=0.125,  # 每设备批次大小为 0.125：表示 8 个设备共同承担 1 个样本的批次，跨设备分摊 batch
        ),
        name="retrieval_local",  # 为该运行器命名，便于日志打印与标识
    )

    print("Initializing retrieval model...")  # 打印提示信息：开始初始化检索模型
    inference_runner.initialize()  # 执行检索模型初始化（构建参数、随机初始化权重等）
    print("Model initialized!")  # 打印提示信息：模型初始化完成

    # Create example batch with simulated user and history
    print("\n" + "=" * 70)  # 打印由 70 个等号组成的分隔线，醒目标记演示标题
    print("RETRIEVAL SYSTEM DEMO")  # 打印演示标题：检索系统演示
    print("=" * 70)  # 再次打印分隔线，形成标题框

    batch_size = 2  # Two users for demo；设置批次大小为 2，即本次演示模拟 2 个用户
    example_batch, example_embeddings = create_example_batch(  # 构造模拟的示例用户批次与示例 embedding，返回 (批次特征, embedding)
        batch_size=batch_size,  # 传入批次大小（2 个用户）
        emb_size=emb_size,  # 传入 embedding 维度
        history_len=history_seq_len,  # 传入用户历史序列长度
        num_candidates=candidate_seq_len,  # 传入候选帖子数量
        num_actions=num_actions,  # 传入交互动作数量
        num_user_hashes=hash_config.num_user_hashes,  # 传入用户哈希函数个数（2）
        num_item_hashes=hash_config.num_item_hashes,  # 传入物品哈希函数个数（2）
        num_author_hashes=hash_config.num_author_hashes,  # 传入作者哈希函数个数（2）
        product_surface_vocab_size=16,  # 传入产品界面词表大小（16）
    )

    # Count valid history items
    valid_history_count = int((example_batch.history_post_hashes[:, :, 0] != 0).sum())  # type: ignore；统计历史中第一个帖子哈希不为 0 的有效条目数（0 视为占位/空）
    print(f"\nUsers have viewed {valid_history_count} posts total in their history")  # 打印所有用户历史中实际浏览过的帖子总数

    # Step 1: Create a corpus of candidate posts
    print("\n" + "-" * 70)  # 打印由 70 个短横线组成的分隔线
    print("STEP 1: Creating Candidate Corpus")  # 打印步骤 1 标题：构造候选语料库
    print("-" * 70)  # 打印分隔线

    corpus_size = 1000  # Simulated corpus of 1000 posts；设置模拟语料库规模为 1000 篇帖子
    corpus_embeddings, corpus_post_ids = create_example_corpus(  # 构造示例语料库，返回语料 embedding 矩阵与帖子 ID 列表
        corpus_size=corpus_size,  # 传入语料库规模
        emb_size=emb_size,  # 传入 embedding 维度
        seed=456,  # 传入随机种子 456，保证生成的假数据可复现
    )
    print(f"Corpus size: {corpus_size} posts")  # 打印语料库中的帖子数量
    print(f"Corpus embeddings shape: {corpus_embeddings.shape}")  # 打印语料 embedding 矩阵的形状

    # Set corpus for retrieval
    inference_runner.set_corpus(corpus_embeddings, corpus_post_ids)  # 将语料 embedding 与帖子 ID 载入检索运行器，作为后续检索的候选池

    # Step 2: Retrieve top-k candidates for each user
    print("\n" + "-" * 70)  # 打印分隔线
    print("STEP 2: Retrieving Top-K Candidates")  # 打印步骤 2 标题：检索 top-k 候选
    print("-" * 70)  # 打印分隔线

    top_k = 10  # 设置每个用户需要检索出的候选数量为 10
    retrieval_output = inference_runner.retrieve(  # 调用检索方法，基于用户表示在语料库中检索最相似的 top-k 帖子
        example_batch,  # 传入示例批次（用户特征）
        example_embeddings,  # 传入示例 embedding
        top_k=top_k,  # 传入 top-k 值
    )

    print(f"\nRetrieved top {top_k} candidates for each of {batch_size} users:")  # 打印检索结果提示：为每个用户检索出 top-k 个候选

    top_k_indices = np.array(retrieval_output.top_k_indices)  # 把输出中的 top-k 候选索引转为 numpy 数组
    top_k_scores = np.array(retrieval_output.top_k_scores)  # 把输出中的 top-k 相似度分数转为 numpy 数组

    for user_idx in range(batch_size):  # 遍历每一个用户
        print(f"\n  User {user_idx + 1}:")  # 打印当前用户的编号（从 1 开始）
        print(f"    {'Rank':<6} {'Post ID':<12} {'Score':<12}")  # 打印表头：名次、帖子 ID、分数
        print(f"    {'-' * 30}")  # 打印表头下方的分隔线
        for rank in range(top_k):  # 遍历该用户的 top-k 个检索结果
            post_id = top_k_indices[user_idx, rank]  # 取出第 rank 个候选对应的帖子 ID
            score = top_k_scores[user_idx, rank]  # 取出第 rank 个候选对应的相似度分数
            bar = "█" * int((score + 1) * 10) + "░" * (20 - int((score + 1) * 10))  # 绘制分数条：cosine 相似度范围 [-1,1] 经 (score+1)*10 线性映射到 [0,20] 个实心方块
            print(f"    {rank + 1:<6} {post_id:<12} {bar} {score:.4f}")  # 打印名次、帖子 ID、分数条以及保留 4 位小数的分数

    print("\n" + "=" * 70)  # 打印分隔线
    print("Demo complete!")  # 打印提示：演示完成
    print("=" * 70)  # 打印分隔线


if __name__ == "__main__":  # 判断本文件是否作为主程序直接运行（而非被 import 导入）
    logging.basicConfig(level=logging.INFO)  # 配置日志级别为 INFO，使运行期间的日志能够输出到控制台
    main()  # 调用主函数，启动检索模型的推理演示流程
