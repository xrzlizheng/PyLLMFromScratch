import torch
from torch import nn

class MLP(nn.Module):
    emb_dim: int
    relu: nn.Module
    in_linear: nn.Linear
    out_linear: nn.Linear
    
    def __init__(self, emb_dim: int):
        super().__init__()
        self.emb_dim = emb_dim
        self.in_linear = nn.Linear(emb_dim, emb_dim * 4)
        self.relu = nn.ReLU()
        self.out_linear = nn.Linear(emb_dim * 4, emb_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.in_linear(x)
        x = self.relu(x)
        x = self.out_linear(x)
        return x


if __name__ == "__main__":
    mlp = MLP(10)
    x = torch.randn((5, 64, 10))
    y = mlp(x)
    print(x.shape)
    print(y.shape)