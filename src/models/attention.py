import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Head(nn.Module):
    """
    单个注意力头。

    该模块计算注意力分数并将其应用于值。
    它包括key、query和value投影，并使用因果掩码
    以防止关注未来的token。

    参数:
        head_size (int): key、query和value投影的维度。
        n_embed (int): 输入嵌入的维度。
        context_length (int): 输入序列的最大长度，用于因果掩码。
    """
    def __init__(self, head_size: int, n_embed: int, context_length: int) -> None:
        """
        初始化注意力头。

        参数:
            head_size (int): key、query和value投影的维度。
            n_embed (int): 输入嵌入的维度。
            context_length (int): 输入序列的最大长度。
        """
        super().__init__()
        self.key = nn.Linear(n_embed, head_size, bias=False)   # Key投影
        self.query = nn.Linear(n_embed, head_size, bias=False) # Query投影
        self.value = nn.Linear(n_embed, head_size, bias=False) # Value投影
        # 用于因果掩码的下三角矩阵
        self.register_buffer('tril', torch.tril(torch.ones(context_length, context_length)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        通过注意力头的前向传播。

        参数:
            x (torch.Tensor): 形状为(B, T, C)的输入张量。

        返回:
            torch.Tensor): 应用注意力后的输出张量。
        """
        B, T, C = x.shape
        head_size = self.key.out_features
        k = self.key(x)     # (B, T, head_size)
        q = self.query(x)   # (B, T, head_size)
        scale_factor = 1 / math.sqrt(head_size)
        # Calculate attention weights: (B, T, head_size) @ (B, head_size, T) -> (B, T, T)
        attn_weights = q @ k.transpose(-2, -1) * scale_factor
        # Apply causal masking
        attn_weights = attn_weights.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        attn_weights = F.softmax(attn_weights, dim=-1)
        v = self.value(x)   # (B, T, head_size)
        # Apply attention weights to values
        out = attn_weights @ v # (B, T, T) @ (B, T, head_size) -> (B, T, head_size)
        return out

class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention module.

    This module combines multiple attention heads in parallel. The outputs of each head
    are concatenated to form the final output.

    Args:
        n_head (int): The number of parallel attention heads.
        n_embed (int): The dimensionality of the input embedding.
        context_length (int): The maximum length of the input sequence.
    """
    def __init__(self, n_head: int, n_embed: int, context_length: int) -> None:
        """
        Initializes the multi-head attention module.

        Args:
            n_head (int): The number of parallel attention heads.
            n_embed (int): The dimensionality of the input embedding.
            context_length (int): The maximum length of the input sequence.
        """
        super().__init__()
        self.heads = nn.ModuleList([Head(n_embed // n_head, n_embed, context_length) for _ in range(n_head)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the multi-head attention.

        Args:
            x (torch.Tensor): Input tensor of shape (B, T, C).

        Returns:
            torch.Tensor: Output tensor after concatenating the outputs of all heads.
        """
        # Concatenate the output of each head along the last dimension (C)
        x = torch.cat([h(x) for h in self.heads], dim=-1)
        return x

if __name__ == '__main__':
    # Example Usage (optional, for testing the module independently)
    batch_size = 2
    sequence_length = 5
    embedding_dim = 32
    num_heads = 4
    context_len = 5
    input_tensor = torch.randn(batch_size, sequence_length, embedding_dim)

    multihead_attn = MultiHeadAttention(n_head=num_heads, n_embed=embedding_dim, context_length=context_len)
    output_tensor = multihead_attn(input_tensor)

    print("MultiHeadAttention Input Shape:", input_tensor.shape)
    print("MultiHeadAttention Output Shape:", output_tensor.shape)
