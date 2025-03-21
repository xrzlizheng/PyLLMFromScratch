import torch
import torch.nn as nn
from src.models.attention import MultiHeadAttention
from src.models.mlp import MLP

class Block(nn.Module):
    """
    一个Transformer模块。

    该模块由一个多头注意力层和一个MLP组成，
    带有层归一化和残差连接。

    参数:
        n_head (int): 多头注意力层中的注意力头数量。
        n_embed (int): 输入嵌入的维度。
        context_length (int): 输入序列的最大长度。
    """
    def __init__(self, n_head: int, n_embed: int, context_length: int) -> None:
        """
        初始化Transformer模块。

        参数:
            n_head (int): 注意力头数量。
            n_embed (int): 嵌入空间的维度。
            context_length (int): 最大序列长度。
        """
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embed)
        self.attn = MultiHeadAttention(n_head, n_embed, context_length)
        self.ln2 = nn.LayerNorm(n_embed)
        self.mlp = MLP(n_embed)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Transformer模块的前向传播。

        参数:
            x (torch.Tensor): 输入张量。

        返回:
            torch.Tensor: 经过模块处理后的输出张量。
        """
        # Apply multi-head attention with residual connection
        x = x + self.attn(self.ln1(x))
        # Apply MLP with residual connection
        x = x + self.mlp(self.ln2(x))
        return x

    def forward_embedding(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        专注于embedding和attention部分的前向传播。

        参数:
            x (torch.Tensor): 输入张量。

        返回:
            tuple: 包含经过MLP embedding后的输出和残差的元组。
        """
        res = x + self.attn(self.ln1(x))
        x = self.mlp.forward_embedding(self.ln2(res))
        return x, res

if __name__ == '__main__':
    # Example Usage (optional, for testing the module independently)
    batch_size = 2
    sequence_length = 5
    embedding_dim = 32
    num_heads = 4
    context_len = 5
    input_tensor = torch.randn(batch_size, sequence_length, embedding_dim)

    transformer_block = Block(n_head=num_heads, n_embed=embedding_dim, context_length=context_len)
    output_tensor = transformer_block(input_tensor)

    print("Transformer Block Input Shape:", input_tensor.shape)
    print("Transformer Block Output Shape:", output_tensor.shape)