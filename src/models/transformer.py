import torch
from torch import nn
from src.models.transformer_block import TransformerBlock

class Transformer(nn.Module):
    text_dim: int
    ctx_len: int
    emb_dim: int
    num_heads: int
    num_blocks: int
    
    text_emb: nn.Embedding
    pos_emb: nn.Embedding
    blocks: nn.ModuleList
    lm_head: nn.Linear
    
    model_device: torch.device
    
    def __init__(self, text_dim: int, ctx_len: int, emb_dim: int, num_heads: int, num_blocks: int, device: torch.device = None):
        super().__init__()
        self.text_dim = text_dim
        self.ctx_len = ctx_len
        self.emb_dim = emb_dim
        self.num_heads = num_heads
        self.num_blocks = num_blocks
        self.model_device = device
        
        self.text_emb = nn.Embedding(text_dim, emb_dim)
        self.pos_emb = nn.Embedding(ctx_len, emb_dim)
        self.blocks = nn.ModuleList([
            TransformerBlock(emb_dim, num_heads)
            for _ in range(num_blocks)
        ])
        self.ln_lmhead = nn.LayerNorm(emb_dim)
        self.lm_head = nn.Linear(emb_dim, text_dim)
    
    def forward(self, x: torch.Tensor, get_probs: bool = False, keep_prompt_seg: bool = False):
        """Transformer forward pass. Input 2-D tensor contains raw tokens, NOT one-hot encoded."""
        assert len(x.shape) == 2 and x.shape[-1] <= self.ctx_len
        
        x = self.text_emb(x) + self.pos_emb(torch.arange(x.shape[-1], device=self.model_device))
        assert len(x.shape) == 3 and x.shape[2] == self.emb_dim
        
        for block in self.blocks:
            x = block(x)
        
        x = self.lm_head(self.ln_lmhead(x))
        assert len(x.shape) == 3 and x.shape[2] == self.text_dim
        
        if get_probs:
            # Compute probs instead of logits
            x = nn.functional.softmax(x, dim=-1)
        
        if not keep_prompt_seg:
            # Only return the next-token results
            x = x[:, -1, :]
        
        return x

if __name__ == "__main__":
    transformer = Transformer(8, 8, 32, 4, 5)
    x = torch.randint(8, (3, 7))
    y = transformer(x, get_probs = True, keep_prompt_seg = True)
    print(x.shape)
    print(y.shape)
    print(y)