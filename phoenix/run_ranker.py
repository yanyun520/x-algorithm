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

import logging  # 导入 Python 标准库 logging 模块，用于配置日志级别并输出运行过程中的日志信息

import numpy as np  # 导入 NumPy 数值计算库并别名 np，用于创建数组以及进行向量化/矩阵运算

from grok import TransformerConfig  # 从 grok 模块导入 TransformerConfig 配置类，用于定义 Transformer 主干网络的结构超参数
from recsys_model import PhoenixModelConfig, HashConfig  # 从 recsys_model 导入排序模型总配置类 PhoenixModelConfig 与多哈希特征配置类 HashConfig
from runners import RecsysInferenceRunner, ModelRunner, create_example_batch, ACTIONS  # 从 runners 导入推理运行器 RecsysInferenceRunner、底层模型运行器 ModelRunner、示例批次构造函数 create_example_batch 以及交互动作列表 ACTIONS


def main():  # 定义主函数 main，作为整个排序模型推理演示脚本的执行入口
    # Model configuration
    emb_size = 128  # Embedding dimension；设置 embedding 向量维度为 128，决定所有特征和隐藏表示的宽度
    num_actions = len(ACTIONS)  # Number of explicit engagement actions；统计显式交互动作（如点赞/收藏等）的总数，作为多标签分类的输出维度
    history_seq_len = 32  # Max history length；设置用户历史行为序列的最大长度（最多记录 32 条历史交互）
    candidate_seq_len = 8  # Max candidates to rank；设置单次参与排序的候选帖子最大数量为 8

    # Hash configuration
    hash_config = HashConfig(  # 构造多哈希配置对象，用于把高维稀疏的 ID 类特征映射到低维稠密空间
        num_user_hashes=2,  # 用户 ID 使用 2 个不同的哈希函数，降低哈希冲突、增强特征表达
        num_item_hashes=2,  # 帖子（物品）ID 使用 2 个不同的哈希函数
        num_author_hashes=2,  # 作者 ID 使用 2 个不同的哈希函数
    )

    recsys_model = PhoenixModelConfig(  # 构造 Phoenix 排序模型的总配置对象，聚合所有子配置
        emb_size=emb_size,  # 传入 embedding 维度，保证全局维度一致
        num_actions=num_actions,  # 传入动作数量，确定模型输出头需要预测的标签数
        history_seq_len=history_seq_len,  # 传入用户历史序列的最大长度
        candidate_seq_len=candidate_seq_len,  # 传入候选序列的最大长度
        hash_config=hash_config,  # 挂载多哈希配置，供特征编码层使用
        product_surface_vocab_size=16,  # 设置产品界面（product surface）词表大小为 16，作为该分类特征 embedding 表的行数
        model=TransformerConfig(  # 传入 Transformer 主干网络的配置
            emb_size=emb_size,  # Transformer 内部 embedding 维度，与整体保持一致
            widening_factor=2,  # SwiGLU 前馈层宽度放大倍数：中间层维度 = key_size * widening_factor，取 2 表示放大 2 倍
            key_size=64,  # 每个注意力头的 query/key 维度设为 64
            num_q_heads=2,  # query 注意力头数设为 2
            num_kv_heads=2,  # key/value 注意力头数设为 2；与 query 头数相等时等价于标准多头注意力(MHA)，若小于则为分组查询注意力(GQA)
            num_layers=2,  # Transformer 堆叠层数设为 2，属于演示用的轻量配置
            attn_output_multiplier=0.125,  # 注意力输出缩放系数：对注意力输出乘以 0.125，用于稳定训练和收敛数值范围
        ),
    )

    # Create inference runner
    inference_runner = RecsysInferenceRunner(  # 创建排序模型的推理运行器，封装模型初始化与 rank 推理流程
        runner=ModelRunner(  # 构造底层模型运行器，负责实际的前向计算以及设备/批次管理
            model=recsys_model,  # 将排序模型配置传入运行器
            bs_per_device=0.125,  # 每设备批次大小为 0.125：表示 8 个设备共同承担 1 个样本的批次，演示时用跨设备分摊的方式分摊 batch
        ),
        name="recsys_local",  # 为该运行器命名，便于日志打印与运行器标识
    )

    print("Initializing model...")  # 打印提示信息：开始初始化模型
    inference_runner.initialize()  # 执行模型初始化（构建参数、随机初始化权重等），为后续推理做准备
    print("Model initialized!")  # 打印提示信息：模型初始化完成

    # Create example batch with simulated posts
    print("\n" + "=" * 70)  # 打印由 70 个等号组成的分隔线，在界面上醒目地区分演示标题
    print("RECOMMENDATION SYSTEM DEMO")  # 打印演示标题：推荐系统演示
    print("=" * 70)  # 再次打印分隔线，形成标题框

    batch_size = 1  # 设置批次大小为 1，即本次演示仅对一个用户进行排序
    example_batch, example_embeddings = create_example_batch(  # 调用构造函数生成模拟的示例批次与示例 embedding，返回 (批次特征, embedding)
        batch_size=batch_size,  # 传入批次大小（1 个用户）
        emb_size=emb_size,  # 传入 embedding 维度
        history_len=history_seq_len,  # 传入用户历史序列长度
        num_candidates=candidate_seq_len,  # 传入候选帖子数量
        num_actions=num_actions,  # 传入交互动作数量
        num_user_hashes=hash_config.num_user_hashes,  # 传入用户哈希函数个数（2）
        num_item_hashes=hash_config.num_item_hashes,  # 传入物品哈希函数个数（2）
        num_author_hashes=hash_config.num_author_hashes,  # 传入作者哈希函数个数（2）
        product_surface_vocab_size=recsys_model.product_surface_vocab_size,  # 传入产品界面词表大小（16）
    )

    action_names = [action.replace("_", " ").title() for action in ACTIONS]  # 把动作名中的下划线替换为空格并首字母大写，得到更易读的展示名列表

    # Count valid history items (where first post hash is non-zero)
    valid_history_count = int((example_batch.history_post_hashes[:, :, 0] != 0).sum())  # type: ignore；统计历史中第一个帖子哈希不为 0 的有效条目数（0 视为占位/空）
    print(f"\nUser has viewed {valid_history_count} posts in their history")  # 打印该用户历史中实际浏览过的帖子数量
    print(f"Ranking {candidate_seq_len} candidate posts...")  # 打印提示：开始对 8 个候选帖子进行排序

    # Rank candidates
    ranking_output = inference_runner.rank(example_batch, example_embeddings)  # 调用推理运行器的 rank 方法，对候选帖子打分并排序，返回排序输出对象

    # Display results
    scores = np.array(ranking_output.scores[0])  # [num_candidates, num_actions]；取出第 0 个用户的分数矩阵并转为 numpy 数组，形状为 [候选数, 动作数]
    ranked_indices = np.array(ranking_output.ranked_indices[0])  # [num_candidates]；取出第 0 个用户的排序后候选索引并转为 numpy 数组

    print("\n" + "-" * 70)  # 打印由 70 个短横线组成的分隔线
    print("RANKING RESULTS (ordered by predicted 'Favorite Score' probability)")  # 打印排序结果标题，说明结果按预测的“收藏概率”从高到低排序
    print("-" * 70)  # 打印分隔线

    for rank, idx in enumerate(ranked_indices):  # 遍历排序后的索引列表，rank 为名次，idx 为该候选在 scores 中的原始索引
        idx = int(idx)  # 将候选索引强制转换为整数，确保可用作数组下标
        print(f"\nRank {rank + 1}: ")  # 打印当前候选的名次（从 1 开始）
        print("  Predicted engagement probabilities:")  # 打印提示：以下为该候选对各类交互动作的预测概率
        for action_idx, action_name in enumerate(action_names):  # 遍历每个动作名及其下标，逐项展示该候选在每个动作维度上的预测概率
            prob = float(scores[idx, action_idx])  # 取出该候选在对应动作维度上的预测概率并转为浮点数
            bar = "█" * int(prob * 20) + "░" * (20 - int(prob * 20))  # 用实心方块█与空心方块░绘制长度 20 的概率条：prob 线性映射为 0~20 个实心块
            print(f"    {action_name:24s}: {bar} {prob:.3f}")  # 打印动作名（左对齐宽 24）、概率条以及保留 3 位小数的概率值

    print("\n" + "=" * 70)  # 打印分隔线
    print("Demo complete!")  # 打印提示：演示完成
    print("=" * 70)  # 打印分隔线


if __name__ == "__main__":  # 判断本文件是否作为主程序直接运行（而非被 import 导入），满足条件才执行下面的入口逻辑
    logging.basicConfig(level=logging.INFO)  # 配置日志级别为 INFO，使运行期间的 INFO 级别日志能够输出到控制台
    main()  # 调用主函数，启动排序模型的推理演示流程
