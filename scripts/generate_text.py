import torch
import tiktoken
import argparse
from config.config import default_config as config
from src.models.transformer import Transformer  # Assuming your Transformer class is in this module

def generate_text(model_path: str, input_text: str, max_new_tokens: int = 100, device: str = 'cuda') -> str:
    """
    使用预训练的Transformer模型生成文本。

    参数:
        model_path (str): 保存的模型检查点路径。
        input_text (str): 用于生成文本的初始文本。
        max_new_tokens (int): 要生成的最大新token数量。
        device (str): 'cuda' 或 'cpu'，运行模型的设备。

    返回:
        str: 生成的文本。
    """
    # 加载模型检查点
    checkpoint = torch.load(model_path, map_location=torch.device(device))

    # 使用config.py中的配置初始化模型
    model = Transformer(
        n_head=config['n_head'],
        n_embed=config['n_embed'],
        context_length=config['context_length'],
        vocab_size=config['vocab_size'],
        N_BLOCKS=config['n_blocks']
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval().to(device)

    # 加载tokenizer
    enc = tiktoken.get_encoding("r50k_base")

    start_ids = enc.encode_ordinary(input_text)
    context = torch.tensor(start_ids, dtype=torch.long, device=device).unsqueeze(0)

    # 生成过程
    with torch.no_grad():
        generated_tokens = model.generate(context, max_new_tokens=max_new_tokens)[0].tolist()

    # 解码生成的tokens
    output_text = enc.decode(generated_tokens)

    return output_text

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate text using a pre-trained Transformer model.")
    parser.add_argument('--model_path', type=str, help='Path to the saved model checkpoint.')
    parser.add_argument('--input_text', type=str, help='The initial text to start generation from.')
    parser.add_argument('--max_new_tokens', type=int, default=100, help='Maximum number of new tokens to generate.')

    args = parser.parse_args()

    generated = generate_text(args.model_path, args.input_text, args.max_new_tokens)
    print(f"Generated text:\n{generated}")

if __name__ == "__main__":
    main()