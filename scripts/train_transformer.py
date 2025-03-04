from data_loader.data_loader import get_batch_iterator
from src.models import Transformer
from config import config
from typing import Iterator, Literal, Tuple
from tqdm import tqdm
import torch
import numpy as np

try:
    device = torch.device(config.DEVICE)
except:
    device = torch.device('cpu')

model = Transformer(
    text_dim=config.VOCAB_SIZE,
    ctx_len=config.CONTEXT_LENGTH,
    emb_dim=config.N_EMBED,
    num_heads=config.N_HEAD,
    num_blocks=config.N_BLOCKS,
).to(device)

criterion = torch.nn.CrossEntropyLoss()

def create_dataloader(split: Literal["train", "val"]) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
    dataloader = get_batch_iterator(
        (config.TRAIN_PATH if split == "train" else config.DEV_PATH),
        config.T_BATCH_SIZE,
        config.CONTEXT_LENGTH,
    )
    return dataloader

train_dataloader = create_dataloader("train")
val_dataloader = create_dataloader("val")

def compute_loss(logits: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    gt = gt.type(torch.LongTensor).to(device)
    
    B, CL, VOCAB = logits.shape
    logits = torch.moveaxis(logits, 2, 1)
    assert logits.shape == (B, VOCAB, CL)
    assert gt.shape == (B, CL)
    
    loss = criterion(logits, gt)
    return loss

@torch.no_grad()
def evaluate_model(model: Transformer, num_batches: int = config.T_EVAL_ITERS, split: Literal["train", "val"] = "val") -> float:
    if split == "val":
        dataloader = val_dataloader
    else:
        dataloader = create_dataloader(split)
    
    losses = []
    for _ in range(num_batches):
        x, y = next(dataloader)
        assert len(x.shape) == 2 and x.shape[0] == y.shape[0]
        
        losses.append(
            compute_loss(
                model(x),
                y
            ).item()
        )
    
    return np.mean(losses)

optimizer = torch.optim.Adam(
    model.parameters(), 
    config.T_LR,
)

print(f"All components count: {sum(1 for param in model.parameters())}")
print(f"All parameter count: {sum(param.numel() for param in model.parameters())}")
print(f"Trainable parameter count: {sum(param.numel() for param in model.parameters() if param.requires_grad)}")

train_losses = []
eval_losses = []

for step in tqdm(list(range(config.T_TRAIN_STEPS))):
    x, y_gt = next(train_dataloader)
    x = x.to(device)
    y_gt = y_gt.to(device)
    logits = model(x, get_probs=False, keep_prompt_seg=True)
    
    optimizer.zero_grad(set_to_none=True)
    loss = compute_loss(logits, y_gt)
    train_losses.append(loss.item())
    loss.backward()
    optimizer.step()
    
    if step % config.T_EVAL_STEPS == 0:
        eval_loss = evaluate_model(model)
        print(f"Eval loss at step {step:7d}: {eval_loss:1.6f}")
        eval_losses.append((step, eval_loss))
    
    if step % config.T_LR_DECAY_STEP == 0:
        for gr in optimizer.param_groups:
            gr["lr"] *= config.T_LR_DECAY_RATE

torch.save(
    {
        "model_state": model.state_dict(),
        "optim_state": optimizer.state_dict(),
        "train_losses": train_losses,
        "eval_losses": eval_losses,
    },
    config.T_OUT_PATH
)