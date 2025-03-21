import torch
import torch.nn as nn
from torch import Tensor

class MLP(nn.Module):
    """
    一个简单的单隐藏层多层感知机。

    该模块用于Transformer块中的前馈处理。
    它扩展输入嵌入的维度，应用ReLU激活函数，然后将其投影回原始嵌入维度。

    参数:
        n_embed (int): 输入嵌入的维度。
    """
    def __init__(self, n_embed: int) -> None:
        """
        初始化MLP模块。

        参数:
            n_embed (int): 输入嵌入的维度。
        """
        super().__init__()
        self.hidden = nn.Linear(n_embed, 4 * n_embed)  # Linear layer to expand embedding size
        self.relu = nn.ReLU()                        # ReLU activation function
        self.proj = nn.Linear(4 * n_embed, n_embed)  # Linear layer to project back to original size

    def forward(self, x: Tensor) -> Tensor:
        """
        MLP的前向传播。

        参数:
            x (torch.Tensor): 输入张量，形状为(B, T, C)，其中B是批量大小，
                              T是序列长度，C是嵌入维度。

        返回:
            torch.Tensor: 与输入形状相同的输出张量。
        """
        x = self.forward_embedding(x)
        x = self.project_embedding(x)
        return x

    def forward_embedding(self, x: Tensor) -> Tensor:
        """
        应用隐藏线性层，然后进行ReLU激活。

        参数:
            x (torch.Tensor): 输入张量。

        返回:
            torch.Tensor: 经过隐藏层和ReLU后的输出。
        """
        x = self.relu(self.hidden(x))
        return x

    def project_embedding(self, x: Tensor) -> Tensor:
        """
        应用投影线性层。

        参数:
            x (torch.Tensor): 输入张量。

        返回:
            torch.Tensor: 经过投影层后的输出。
        """
        x = self.proj(x)
        return x

if __name__ == '__main__':
    # Example Usage (optional, for testing the module independently)
    batch_size = 2
    sequence_length = 3
    embedding_dim = 16
    input_tensor = torch.randn(batch_size, sequence_length, embedding_dim)

    mlp_module = MLP(n_embed=embedding_dim)
    output_tensor = mlp_module(input_tensor)

    print("MLP Input Shape:", input_tensor.shape)
    print("MLP Output Shape:", output_tensor.shape)