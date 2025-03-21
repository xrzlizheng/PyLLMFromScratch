import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.transformer_block import Block

class Transformer(nn.Module):
    """
    主要的Transformer模型。

    该类将token和position embeddings与一系列Transformer块结合，
    并使用最后的线性层进行语言建模。

    参数:
        n_head (int): 每个transformer块中的注意力头数量。
        n_embed (int): 嵌入空间的维度。
        context_length (int): 输入序列的最大长度。
        vocab_size (int): 词汇表大小。
        N_BLOCKS (int): 模型中transformer块的数量。
    """
    def __init__(self, n_head: int, n_embed: int, context_length: int, vocab_size: int, N_BLOCKS: int) -> None:
        """
        初始化Transformer模型。

        参数:
            n_head (int): 注意力头数量。
            n_embed (int): 嵌入维度。
            context_length (int): 最大序列长度。
            vocab_size (int): 词汇表大小。
            N_BLOCKS (int): transformer块数量。
        """
        super().__init__()
        self.context_length = context_length
        self.N_BLOCKS = N_BLOCKS
        self.token_embed = nn.Embedding(vocab_size, n_embed)
        self.position_embed = nn.Embedding(context_length, n_embed)
        self.attn_blocks = nn.ModuleList([Block(n_head, n_embed, context_length) for _ in range(N_BLOCKS)])
        self.layer_norm = nn.LayerNorm(n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size)
        self.register_buffer('pos_idxs', torch.arange(context_length))

    def _pre_attn_pass(self, idx: torch.Tensor) -> torch.Tensor:
        """
        结合token和position embeddings。

        参数:
            idx (torch.Tensor): 输入的token索引。

        返回:
            torch.Tensor: token和position embeddings的和。
        """
        B, T = idx.shape
        tok_embedding = self.token_embed(idx)
        pos_embedding = self.position_embed(self.pos_idxs[:T])
        return tok_embedding + pos_embedding

    def forward(self, idx: torch.Tensor, targets: torch.Tensor = None) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Transformer的前向传播。

        参数:
            idx (torch.Tensor): 输入的token索引。
            targets (torch.Tensor, 可选): 用于计算loss的目标token索引。默认为None。

        返回:
            tuple: Logits和loss（如果提供了targets）。
        """
        x = self._pre_attn_pass(idx)
        for block in self.attn_blocks:
            x = block(x)
        x = self.layer_norm(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            B, T, C = logits.shape
            flat_logits = logits.view(B * T, C)
            targets = targets.view(B * T).long()
            loss = F.cross_entropy(flat_logits, targets)
        return logits, loss

    def forward_embedding(self, idx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        专注于embedding和attention块的前向传播。

        参数:
            idx (torch.Tensor): 输入的token索引。

        返回:
            tuple: 经过attention块后的输出和残差。
        """
        x = self._pre_attn_pass(idx)
        residual = x
        for block in self.attn_blocks:
            x, residual = block.forward_embedding(x)
        return x, residual

    def generate(self, idx: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        """
        给定初始序列生成新的tokens。

        参数:
            idx (torch.Tensor): 初始的token索引序列。
            max_new_tokens (int): 要生成的token数量。

        返回:
            torch.Tensor: 扩展后的token序列。
        """
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.context_length:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

if __name__ == '__main__':
    # Example Usage (optional, for testing the module independently)
    batch_size = 2
    sequence_length = 5
    vocab_size = 100
    embedding_dim = 32
    num_heads = 4
    num_blocks = 2
    context_len = 5
    input_indices = torch.randint(0, vocab_size, (batch_size, sequence_length))

    transformer_model = Transformer(n_head=num_heads, n_embed=embedding_dim, context_length=context_len, vocab_size=vocab_size, N_BLOCKS=num_blocks)
    logits, loss = transformer_model(input_indices, targets=input_indices) # Using input as target for simplicity

    print("Transformer Logits Shape:", logits.shape)
    print("Transformer Loss:", loss)

    # Example of generating tokens
    start_indices = input_indices[:, :1]  # Take the first token of each sequence as start
    generated_tokens = transformer_model.generate(start_indices, max_new_tokens=5)
    print("Generated Tokens Shape:", generated_tokens.shape)