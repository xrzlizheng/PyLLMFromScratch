import torch
from torch import nn
from src.models.mlp import MLP
from src.models.attention import MultiHeadAttn

class TransformerBlock(nn.Module):
    emb_dim: int
    attn: MultiHeadAttn
    ln_attn: nn.LayerNorm
    mlp: MLP
    ln_mlp: nn.LayerNorm
    
    def __init__(self, emb_dim: int, num_heads: int):
        super().__init__()
        self.attn = MultiHeadAttn(emb_dim, num_heads)
        self.ln_attn = nn.LayerNorm(emb_dim)
        self.mlp = MLP(emb_dim)
        self.ln_mlp = nn.LayerNorm(emb_dim)
    
    def forward(self, x: torch.Tensor):
        x += self.attn(self.ln_attn(x))
        x += self.mlp(self.ln_mlp(x))
        return x