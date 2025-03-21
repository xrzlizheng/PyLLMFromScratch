# 用python一步步从零开始训练LLM
具体见

## 训练数据信息

训练数据来自Pile数据集，这是一个多样化的、开源的、大规模的语言模型训练数据集。Pile数据集包含22个不同的数据集，包括书籍、文章、网站等文本。Pile数据集的总大小为825GB

## 硬件要求

您需要GPU来训练模型。Colab或Kaggle T4可以用于训练1300万+参数的模型，但对于数十亿参数的训练会失败。请看以下比较：

| GPU名称              | 显存  | 数据大小  | 20亿LLM训练 | 1300万LLM训练 |
|-----------------------|---------|------------|------------------|------------------|
| NVIDIA A100           | 40 GB   | 大        | ✔                | ✔                |
| NVIDIA V100           | 16 GB   | 中        | ✘                | ✔                |
| AMD Radeon VII        | 16 GB   | 中        | ✘                | ✔                |
| NVIDIA RTX 3090       | 24 GB   | 大        | ✔                | ✔                |
| Tesla P100            | 16 GB   | 中        | ✘                | ✔                |
| NVIDIA RTX 3080       | 10 GB   | 中        | ✘                | ✔                |
| AMD Radeon RX 6900 XT | 16 GB   | 大        | ✘                | ✔                |
| NVIDIA GTX 1080 Ti    | 11 GB   | 中        | ✘                | ✔                |
| Tesla T4              | 16 GB   | 小        | ✘                | ✔                |
| NVIDIA Quadro RTX 8000| 48 GB   | 大        | ✔                | ✔                |

1300万LLM训练是指训练1300万+参数的模型，20亿LLM训练是指训练20亿+参数的模型。数据大小分为小、中、大。小数据大小约为1GB，中数据大小约为5GB，大数据大小约为10GB。

## 代码结构

代码库的组织结构如下：
```bash
train-llm-from-scratch/
├── src/          
│   ├── models/   
│   │   ├── mlp.py       # 多层感知器（MLP）模块的定义
│   │   ├── attention.py # 注意力机制的定义（单头、多头）
│   │   ├── transformer_block.py # 单个Transformer块的定义
│   │   ├── transformer.py     # 主Transformer模型的定义
├── config/       
│   └── config.py    # 包含默认配置（模型参数、文件路径等）
├── data_loader/  
│   └── data_loader.py # 包含创建数据加载器/迭代器的函数
├── scripts/      
│   ├── train_transformer.py # 训练Transformer模型的脚本
│   ├── data_download.py   # 下载数据集的脚本
│   ├── data_preprocess.py # 预处理下载数据的脚本
│   ├── generate_text.py   # 使用训练模型生成文本的脚本
├── data/         # 存储数据集的目录
│   ├── train/     # 包含训练数据
│   └── val/       # 包含验证数据
├── models/       # 保存训练模型的目录
```

`scripts/`目录包含下载数据集、预处理数据、训练模型和使用训练模型生成文本的脚本。`src/models/`目录包含Transformer模型、多层感知器（MLP）、注意力机制和Transformer块的实现。`config/`目录包含默认参数的配置文件。`data_loader/`目录包含创建数据加载器/迭代器的函数。

## 使用说明

克隆仓库并导航到目录：
```bash
git clone https://github.com/xrzlizheng/PyLLMFromScratch.git
cd train-llm-from-scratch
```

如果遇到导入问题，请确保将pythonpath更改为项目的根目录：
```bash
export PYTHONPATH="${PYTHONPATH}:/path/to/train-llm-from-scratch"

# 或者如果您已经在train-llm-from-scratch目录中
export PYTHONPATH="$PYTHONPATH:."
```

安装所需的依赖项：
```bash
pip install -r requirements.txt
```

您可以修改`src/models/transformer.py`中的Transformer架构和`config/config.py`中的训练配置。

要下载训练数据，请运行：
```bash
python scripts/data_download.py
```

该脚本支持以下参数：
* `--train_max`: 要下载的最大训练文件数。默认为1（最大等于30）每个文件大约11GB。
* `--train_dir`: 存储训练数据的目录。默认为`data/train`。
* `--val_dir`: 存储验证数据的目录。默认为`data/val`。

要预处理下载的数据，请运行：
```bash
python scripts/data_preprocess.py
```

该脚本支持以下参数：
- `--train_dir`: 存储训练数据文件的目录（默认为`data/train`）。
- `--val_dir`: 存储验证数据文件的目录（默认为`data/val`）。
- `--out_train_file`: 存储处理后的训练数据的HDF5格式路径（默认为`data/train/pile_train.h5`）。
- `--out_val_file`: 存储处理后的验证数据的HDF5格式路径（默认为`data/val/pile_dev.h5`）。
- `--tokenizer_name`: 用于处理数据的tokenizer名称（默认为`r50k_base`）。
- `--max_data`: 从每个数据集（训练和验证）中处理的最大JSON对象数（[行](#training-data-info)）。默认为1000。

现在数据已经预处理完毕，您可以通过将`config/config.py`中的配置更改为以下内容来训练1300万参数的LLM：

```python
# 定义词汇表大小和Transformer配置（30亿）
VOCAB_SIZE = 50304          # 词汇表中唯一token的数量
CONTEXT_LENGTH = 128        # 模型的最大序列长度
N_EMBED = 128               # 嵌入空间的维度
N_HEAD = 8                  # 每个Transformer块中的注意力头数
N_BLOCKS = 1               # 模型中的Transformer块数
```

要训练模型，请运行：
```bash
python scripts/train_transformer.py
```

它将开始训练模型，并将训练好的模型保存在`models/`默认目录或配置文件中指定的目录中。

要使用训练好的模型生成文本，请运行：
```bash
python scripts/generate_text.py --model_path models/your_model.pth --input_text 你好
```

该脚本支持以下参数：
- `--model_path`: 训练好的模型的路径。
- `--input_text`: 生成新文本的初始文本提示。
- `--max_new_tokens`: 生成的最大token数（默认为100）。

它将使用训练好的模型根据输入提示生成文本。


---

## 参考

- [train-llm-from-scratch](https://github.com/FareedKhan-dev/train-llm-from-scratch)
- [The Pile Dataset](https://pile.eleuther.ai/)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [OpenAI's tiktoken](https://github.com/openai/tiktoken)
