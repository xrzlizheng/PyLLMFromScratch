#!/usr/bin/env python
# coding: utf-8
"""
@Author: lizheng
@Date: 2025-04-15
@Description: https://blog.csdn.net/qq_36603091/article/details/147288188
具体可以查看博文https://blog.csdn.net/qq_36603091/article/details/147288188
"""
# 导入必要的库
import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
import math
import os
import collections # 用于BPE类似的处理（如果扩展）
import re          # 用于初始分词

#流程详解看博客https://blog.csdn.net/qq_36603091/article/details/147288188
torch.manual_seed(1337)

print(f"PyTorch version: {torch.__version__}")

# --- 设备配置 ---
# 理论：设置设备（如果可用则使用GPU 'cuda'，否则使用CPU）用于张量运算。
# 这确保模型和数据能在可用硬件上高效处理。
device = 'cuda' if torch.cuda.is_available() else 'cpu'

print(f"Using device: {device}")
print("Libraries imported and device configured.")







# 定义用于训练的原始文本语料库
corpus_raw = """
Alice was beginning to get very tired of sitting by her sister on the
bank, and of having nothing to do: once or twice she had peeped into the
book her sister was reading, but it had no pictures or conversations in
it, 'and what is the use of a book,' thought Alice 'without pictures or
conversation?'
So she was considering in her own mind (as well as she could, for the
hot day made her feel very sleepy and stupid), whether the pleasure
of making a daisy-chain would be worth the trouble of getting up and
picking the daisies, when suddenly a White Rabbit with pink eyes ran
close by her.
"""

print(f"Training corpus defined (length: {len(corpus_raw)} characters).")





# 在原始语料库中找到所有唯一字符
chars = sorted(list(set(corpus_raw)))
vocab_size = len(chars)

# 创建字符到整数的映射（编码）
char_to_int = { ch:i for i,ch in enumerate(chars) }

# 创建整数到字符的映射（解码）
int_to_char = { i:ch for i,ch in enumerate(chars) }

print(f"Created character vocabulary of size: {vocab_size}")
print(f"Vocabulary: {''.join(chars)}")
# 可选：打印映射
# print(f"字符到整数映射示例: {{k: char_to_int[k] for k in list(char_to_int)[:5]}}")
# print(f"整数到字符映射示例: {{k: int_to_char[k] for k in list(int_to_char)[:5]}}")





# 将整个语料库编码为整数ID列表
encoded_corpus = [char_to_int[ch] for ch in corpus_raw]

# 将列表转换为PyTorch张量
full_data_sequence = torch.tensor(encoded_corpus, dtype=torch.long, device=device)

print(f"Encoded corpus into a tensor of shape: {full_data_sequence.shape}")
# 可选：显示前几个编码的ID
# print(f"前50个编码的token ID: {full_data_sequence[:50].tolist()}")





# --- 模型架构超参数 ---
# vocab_size已从数据中确定
d_model = 128         # 嵌入维度（显著减少）
n_layers = 4          # Transformer块的数量（减少）
n_heads = 4           # 注意力头的数量
block_size = 64       # 最大上下文长度（序列长度）
rms_norm_eps = 1e-5   # RMSNorm稳定性的epsilon值
rope_theta = 10000.0  # RoPE的theta参数（从Llama 4的500k减少）

# --- MoE特定超参数 ---
num_local_experts = 4      # 每个MoE层的专家数量（从16减少）
num_experts_per_tok = 2   # 每个token路由到的专家数量（Top-K，从4减少？）
intermediate_size_expert = d_model * 2  # 每个专家MLP内的隐藏维度（缩小）
intermediate_size_shared = d_model * 2  # 共享MLP内的隐藏维度（缩小）

# --- 注意力超参数 ---
# d_k（每个头的维度）将从d_model和n_heads派生

# --- 训练超参数 ---
learning_rate = 5e-4  # 学习率
batch_size = 16       # 并行处理的序列数量
epochs = 3000         # 训练迭代次数（根据需要调整）
eval_interval = 300  # 打印损失的频率

# --- 派生超参数 ---
assert d_model % n_heads == 0, "d_model必须能被n_heads整除"
d_k = d_model // n_heads # 每个头的键/查询/值的维度
expert_dim = intermediate_size_expert # 为清晰起见的别名
shared_expert_dim = intermediate_size_shared # 为清晰起见的别名

print("--- 超参数已定义 ---")
print(f"词汇表大小 (vocab_size): {vocab_size}")
print(f"嵌入维度 (d_model): {d_model}")
print(f"层数 (n_layers): {n_layers}")
print(f"注意力头数量 (n_heads): {n_heads}")
print(f"每个头的维度 (d_k): {d_k}")
print(f"最大序列长度 (block_size): {block_size}")
print(f"RMSNorm Epsilon (rms_norm_eps): {rms_norm_eps}")
print(f"RoPE Theta (rope_theta): {rope_theta}")
print("--- MoE特定参数 ---")
print(f"每层的本地专家数量 (num_local_experts): {num_local_experts}")
print(f"每个Token的专家数量 (num_experts_per_tok): {num_experts_per_tok}")
print(f"专家中间维度 (expert_dim): {expert_dim}")
print(f"共享MLP中间维度 (shared_expert_dim): {shared_expert_dim}")
print("--- 训练特定参数 ---")
print(f"学习率: {learning_rate}")
print(f"批量大小: {batch_size}")
print(f"训练轮数: {epochs}")



# ### 步骤3：准备训练数据
# 
# **目标：** 将编码数据（`full_data_sequence`）结构化为适合训练下一个token预测任务的输入（`x`）和目标（`y`）对。

# #### 步骤3.1：创建输入(x)和目标(y)对
# 
# **理论：** 模型通过前面的token学习预测下一个token。我们从编码语料库中创建长度为`block_size`的重叠序列。对于从索引`i`开始的输入序列`x`，对应的目标序列`y`从索引`i+1`开始。`y`中的每个token是对应`x`中token的目标预测。

# 创建列表以保存所有可能的输入(x)和目标(y)序列
all_x = []
all_y = []

# 遍历编码语料库张量以提取重叠序列
num_total_tokens = len(full_data_sequence)
for i in range(num_total_tokens - block_size):
    # 提取输入序列块
    x_chunk = full_data_sequence[i : i + block_size]
    # 提取目标序列块（右移一个位置）
    y_chunk = full_data_sequence[i + 1 : i + block_size + 1]
    all_x.append(x_chunk)
    all_y.append(y_chunk)

# 将张量列表堆叠成单个大张量
train_x = torch.stack(all_x)
train_y = torch.stack(all_y)

num_sequences_available = train_x.shape[0]
print(f"Created {num_sequences_available} overlapping input/target sequence pairs.")
print(f"Shape of train_x: {train_x.shape}") # 应该是(num_sequences, block_size)
print(f"Shape of train_y: {train_y.shape}") # 应该是(num_sequences, block_size)

# 可选：验证设备
# print(f"train_x所在设备: {train_x.device}") # 可能仍在CPU上，在批处理时移动

# **输出说明：** 代码从编码文本中提取所有可能的长度为`block_size`的序列及其对应的目标序列。然后将这些堆叠成两个张量，`train_x`和`train_y`。输出显示了创建的序列对数量及其形状。

# #### 步骤3.2：批处理策略（随机采样）
# 
# **理论：** 对于训练，我们以批次处理数据。我们不使用正式的DataLoader，而是在每个训练步骤中从可用序列（`0`到`num_sequences_available - 1`）中随机采样`batch_size`个索引。然后我们从`train_x`和`train_y`中检索相应的序列并将其移动到正确的`device`。

# 检查是否有足够的序列满足所需的批量大小
if num_sequences_available < batch_size:
    print(f"警告：序列数量({num_sequences_available})小于批量大小({batch_size})。调整批量大小。")
    batch_size = num_sequences_available

print(f"数据准备就绪。将随机采样大小为{batch_size}的批次。")
print("批次将在训练循环中移动到设备。")
# 在循环中如何选择批次的示例：
# indices = torch.randint(0, num_sequences_available, (batch_size,))
# xb = train_x[indices].to(device)
# yb = train_y[indices].to(device)

# **输出说明：** 确认批处理策略。检查给定序列数量的批量大小是否可行，并解释将在训练循环中使用随机采样创建批次。

# ### 步骤4：模型组件初始化（内联）
# 
# **目标：** 为我们的类Llama 4 MoE模型初始化可学习参数和固定组件。我们将在训练循环中使用的每层组件存储在列表中。

# #### 步骤4.1：Token嵌入层
# 
# **理论：** 该层将离散的token ID（我们的字符ID）映射到维度为`d_model`的密集向量。它本质上是一个查找表，其中每行对应词汇表中的一个token。输入形状`(B, T)` -> 输出形状`(B, T, C)`，其中`B`是批量大小，`T`是序列长度（`block_size`），`C`是嵌入维度（`d_model`）。

# 初始化token嵌入表
token_embedding_table = nn.Embedding(vocab_size, d_model).to(device)

print(f"已初始化Token嵌入层:")
print(f"  输入词汇表大小: {vocab_size}")
print(f"  输出嵌入维度 (d_model): {d_model}")
print(f"  权重形状: {token_embedding_table.weight.shape}")
print(f"  设备: {token_embedding_table.weight.device}")

# **输出说明：** 创建`nn.Embedding`层，指定词汇表大小和所需的嵌入维度（`d_model`）。它被移动到目标`device`。输出确认层的配置和其可学习权重矩阵的形状（`vocab_size` x `d_model`）。

# #### 步骤4.2：旋转位置嵌入（RoPE）预计算
# 
# **理论：** RoPE通过基于查询（Q）和键（K）向量的绝对位置旋转特征对来注入位置信息。它在概念上使用复数。我们基于`rope_theta`和每个注意力头的维度（`d_k`）预计算逆频率（`inv_freq`）。实际的旋转角度（`freqs_cis`）取决于token位置，在前向传播中动态计算。
# 公式：
# $$ \theta_i = \frac{1}{\text{rope_theta}^{2i / d_k}} $$
# 其中 $i \in [0, 1, ..., d_k/2 - 1]$。我们预计算`inv_freq`，它对应于这个$\theta_i$。

# 预计算RoPE的逆频率
# 公式：1.0 / (rope_theta ** (torch.arange(0, d_k, 2) / d_k))
rope_freq_indices = torch.arange(0, d_k, 2, dtype=torch.float, device=device)
inv_freq = 1.0 / (rope_theta ** (rope_freq_indices / d_k))

print("预计算的RoPE逆频率 (inv_freq):")
print(f"  形状: {inv_freq.shape}") # 应该是(d_k / 2,)
print(f"  值（前5个）: {inv_freq[:5].tolist()}")
print(f"  设备: {inv_freq.device}")
# 'freqs_cis'（复数）将在前向传播中使用这些inv_freq和position_ids计算

# **输出说明：** 这段代码基于`rope_theta`和`d_k`超参数计算`inv_freq`张量。这个张量包含用于RoPE的基本频率。其形状是`(d_k / 2,)`。实际的旋转值（`freqs_cis`）将在前向传播中基于token位置动态计算。

# #### 步骤4.3：RMSNorm层初始化
# 
# **理论：** RMSNorm是Layer Normalization的简化版。它通过其均方根对输入进行归一化，然后通过可学习的权重参数`gamma`进行缩放（但通常缺少LayerNorm的可学习偏置`beta`）。
# 公式：$$ \text{RMSNorm}(x) = \frac{x}{\sqrt{\text{mean}(x^2) + \epsilon}} * \gamma $$
# 我们需要在注意力块之前、MoE/FFN块之前和输出层之前各有一个RMSNorm层。

# 存储每个Transformer块的RMSNorm层权重的列表
rmsnorm_weights_input = []      # MHA之前的RMSNorm
rmsnorm_weights_post_attn = []  # MoE/FFN之前的RMSNorm

print(f"为{n_layers}层初始化RMSNorm权重...")
for i in range(n_layers):
    # 用于注意力输入的RMSNorm权重
    # 初始化权重为torch.ones，类似于nn.LayerNorm的默认gamma
    weight_in = nn.Parameter(torch.ones(d_model, device=device))
    rmsnorm_weights_input.append(weight_in)

    # 用于MoE/FFN输入的RMSNorm权重（注意力后）
    weight_post = nn.Parameter(torch.ones(d_model, device=device))
    rmsnorm_weights_post_attn.append(weight_post)
    print(f"  为第{i+1}层初始化RMSNorm权重 (输入: {weight_in.shape}, 注意力后: {weight_post.shape})")

# 输出层之前的最终RMSNorm
final_rmsnorm_weight = nn.Parameter(torch.ones(d_model, device=device))

print(f"已初始化最终RMSNorm权重，形状: {final_rmsnorm_weight.shape}")
print("RMSNorm权重已初始化（作为nn.Parameter）。归一化逻辑将内联实现。")

# **输出说明：** 为模型中需要的每个RMSNorm实例创建可学习的权重参数（`gamma`）（每层两个，加上一个最终的）。这些存储为包含初始化为1的张量的`nn.Parameter`对象，形状为`(d_model,)`。实际的归一化计算将在前向传播中使用这些权重内联执行。

# #### 步骤4.4：注意力层初始化（MHA）
# 
# **理论：** 初始化每个Transformer块中需要的多头注意力的线性投影层。我们需要：
# 1.  一个组合投影层，同时为所有头生成查询（Q）、键（K）和值（V）向量。输入`(B, T, C)` -> 输出`(B, T, 3*C)`。
# 2.  一个输出投影层，将所有头的结果组合回模型维度。输入`(B, T, C)` -> 输出`(B, T, C)`。

# 存储每个Transformer块的注意力层的列表
mha_qkv_linears = []    # Q、K、V投影的组合线性层
mha_output_linears = [] # MHA的输出线性层

print(f"为{n_layers}层初始化注意力（MHA）线性层...")
for i in range(n_layers):
    # 组合QKV投影层
    # 在大型transformer QKV投影中偏置通常为False
    qkv_linear = nn.Linear(d_model, 3 * d_model, bias=False).to(device)
    mha_qkv_linears.append(qkv_linear)

    # 输出投影层
    # 这里的偏置也通常为False，但可以为True
    output_linear = nn.Linear(d_model, d_model, bias=False).to(device)
    mha_output_linears.append(output_linear)
    print(f"  为第{i+1}层初始化MHA线性层 (QKV: {qkv_linear.weight.shape}, 输出: {output_linear.weight.shape})")

print("注意力（MHA）线性层已初始化。")

# **输出说明：** 创建负责将输入投影到Q、K、V空间并将注意力输出投影回`d_model`的`nn.Linear`层，用于每个`n_layers`。输出确认创建并显示这些线性层的权重矩阵的形状。

# #### 步骤4.5：专家混合（MoE）层初始化
# 
# **理论：** 初始化MoE块的组件，这些块替代了每个Transformer层中的标准FFN。对于每一层，我们需要：
# 1.  **路由器：** 一个将输入隐藏状态（`d_model`）映射到每个专家（`num_local_experts`）的logits的线性层。
# 2.  **专家：** 一组`num_local_experts`个独立的MLP网络。每个专家MLP通常使用门控激活（SiLU）。我们初始化这些专家MLP的权重。
#     *   门控/上投影：线性`d_model` -> `2 * expert_dim`。
#     *   下投影：线性`expert_dim` -> `d_model`。
# 3.  **共享专家：** 一个标准MLP（结构与一个专家相同，但使用`shared_expert_dim`）处理所有token。
# 
# *注意：* 我们直接将权重初始化为`nn.Parameter`以紧密遵循参考脚本的结构，而不是使用`nn.Linear`。共享专家使用标准`nn.Linear`。

# 存储每层MoE组件的列表
moe_routers = []             # 路由器线性层
moe_expert_gate_up_proj = [] # 专家门控/上投影权重
moe_expert_down_proj = []    # 专家下投影权重
shared_expert_gate_proj = [] # 共享专家门控投影
shared_expert_up_proj = []   # 共享专家上投影
shared_expert_down_proj = [] # 共享专家下投影

print(f"为{n_layers}层初始化MoE和共享MLP组件...")
print(f"  每层专家数量: {num_local_experts}")
print(f"  专家维度: {expert_dim}")
print(f"  共享MLP维度: {shared_expert_dim}")

for i in range(n_layers):
    # 1. 路由器
    router_linear = nn.Linear(d_model, num_local_experts, bias=False).to(device)
    moe_routers.append(router_linear)

    # 2. 专家（权重作为参数）
    # 门控/上投影权重：(num_experts, d_model, 2 * expert_dim)
    gate_up_w = nn.Parameter(torch.empty(num_local_experts, d_model, 2 * expert_dim, device=device))
    nn.init.normal_(gate_up_w, mean=0.0, std=0.02) # 示例初始化
    moe_expert_gate_up_proj.append(gate_up_w)

    # 下投影权重：(num_experts, expert_dim, d_model)
    down_w = nn.Parameter(torch.empty(num_local_experts, expert_dim, d_model, device=device))
    nn.init.normal_(down_w, mean=0.0, std=0.02) # 示例初始化
    moe_expert_down_proj.append(down_w)

    # 3. 共享专家（标准MLP层）
    shared_gate = nn.Linear(d_model, shared_expert_dim, bias=False).to(device)
    shared_up = nn.Linear(d_model, shared_expert_dim, bias=False).to(device)
    shared_down = nn.Linear(shared_expert_dim, d_model, bias=False).to(device)
    shared_expert_gate_proj.append(shared_gate)
    shared_expert_up_proj.append(shared_up)
    shared_expert_down_proj.append(shared_down)

    print(f"  已初始化第{i+1}层的MoE组件:")
    print(f"    路由器权重: {router_linear.weight.shape}")
    print(f"    专家门控/上投影权重: {gate_up_w.shape}")
    print(f"    专家下投影权重: {down_w.shape}")
    print(f"    共享门控权重: {shared_gate.weight.shape}")
    print(f"    共享上投影权重: {shared_up.weight.shape}")
    print(f"    共享下投影权重: {shared_down.weight.shape}")

print("MoE和共享MLP组件已初始化。")
# 激活函数（内联使用）
activation_fn = nn.SiLU()

# **输出说明：** 这个块为每一层初始化MoE组件。对于每一层，它创建：
# *   一个线性`router`层。
# *   所有`num_local_experts`的`gate_up_proj`和`down_proj`的权重`Parameter`张量。
# *   `shared_expert` MLP的`nn.Linear`层。
# 输出确认初始化并显示每层中每个组件的权重张量的形状。

# #### 步骤4.6：最终输出层初始化
# 
# **理论：** 这个最终线性层将最后一个Transformer块的输出（经过最终RMSNorm后）从模型维度`d_model`映射回词汇表大小`vocab_size`。输出表示每个可能的下一个token的原始分数（logits）。

# 最终线性层（语言建模头）
output_linear_layer = nn.Linear(d_model, vocab_size, bias=False).to(device)

print(f"已初始化最终输出线性层:")
print(f"  输入维度 (d_model): {d_model}")
print(f"  输出维度 (vocab_size): {vocab_size}")
print(f"  权重形状: {output_linear_layer.weight.shape}")
print(f"  设备: {output_linear_layer.weight.device}")

# **输出说明：** 初始化负责在词汇表上产生logits的最终`nn.Linear`层。输出确认其输入/输出维度和权重形状。

# #### 步骤4.7：因果掩码预计算
# 
# **理论：** 对于仅解码器的语言模型，我们需要一个因果注意力掩码来防止注意力头在训练和生成过程中查看未来的token。这通常是一个下三角矩阵，允许注意的位置有一个值（例如，1或0），不允许的位置有另一个值（例如，0或-无穷大）。我们为最大`block_size`预计算这个掩码。

# 创建用于因果自注意力的下三角掩码
# 允许注意的位置值为1，掩码的位置值为0。
# 形状：(1, 1, block_size, block_size) 用于与(B, n_heads, T, T)广播
causal_mask = torch.tril(torch.ones(block_size, block_size, device=device))
causal_mask = causal_mask.view(1, 1, block_size, block_size)

print("预计算的因果注意力掩码:")
print(f"  形状: {causal_mask.shape}")
print(f"  需要梯度: {causal_mask.requires_grad}")
# 可选：可视化掩码的前几行和列
# print(causal_mask[0, 0, :5, :5])



