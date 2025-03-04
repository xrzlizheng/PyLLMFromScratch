import torch
from torch import nn

class Attention(nn.Module):
    emb_dim: int
    attn_dim: int
    q_head: nn.Linear
    k_head: nn.Linear
    v_head: nn.Linear
    
    def __init__(self, emb_dim: int, attn_dim: int):
        super().__init__()
        self.emb_dim = emb_dim
        self.attn_dim = attn_dim
        self.q_head = nn.Linear(emb_dim, attn_dim, bias=False)
        self.k_head = nn.Linear(emb_dim, attn_dim, bias=False)
        self.v_head = nn.Linear(emb_dim, attn_dim, bias=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, ctx_len, emb_dim)
        assert len(x.shape) == 3 and x.shape[-1] == self.emb_dim
        
        q: torch.Tensor = self.q_head(x) # (B, ctx_len, attn_dim)
        k: torch.Tensor = self.k_head(x)
        k = torch.transpose(k, -1, -2) # (B, attn_dim, ctx_len)
        attn = q @ k * (self.attn_dim ** -0.5)
        attn = torch.tril(attn, diagonal=0) + torch.triu(torch.ones_like(attn) * (-torch.inf), diagonal=1)
        attn = nn.functional.softmax(attn, dim=-1) # (B, ctx_len, ctx_len)
        
        v: torch.Tensor = self.v_head(x) # (B, ctx_len, attn_dim)
        return attn @ v    


class MultiHeadAttn(nn.Module):
    emb_dim: int
    num_heads: int
    heads: nn.ModuleList
    
    def __init__(self, emb_dim: int, num_heads: int):
        super().__init__()
        self.emb_dim = emb_dim
        self.num_heads = num_heads
        
        assert emb_dim % num_heads == 0
        
        self.heads = nn.ModuleList([
            Attention(emb_dim, emb_dim // num_heads)
            for _ in range(num_heads)
        ])
    
    def forward(self, x: torch.Tensor):
        return torch.cat(
            [attn(x) for attn in self.heads],
            dim = -1
        )

if __name__ == "__main__":
    mha = MultiHeadAttn(16, 4)
    x = torch.randn((5, 64, 16))
    y = mha(x)
    print(x.shape)
    print(y.shape)