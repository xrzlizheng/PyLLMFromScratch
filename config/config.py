# --- 配置 ---

# 定义词汇表大小和transformer配置(30亿参数)
VOCAB_SIZE = 50304          # 词汇表中唯一token的数量
CONTEXT_LENGTH = 512        # 模型的最大序列长度
N_EMBED = 2048              # 嵌入空间的维度
N_HEAD = 16                 # 每个transformer块中的注意力头数量
N_BLOCKS = 64               # 模型中的transformer块数量

# Paths to training and development datasets
TRAIN_PATH = "data/train/pile_train.h5"  # 训练数据集的文件路径
DEV_PATH = "data/val/pile_dev.h5"      # 验证数据集的文件路径

# Transformer training parameters
T_BATCH_SIZE = 32          # 每个训练批次的样本数量
T_CONTEXT_LENGTH = 16      # 训练批次的上下文长度
T_TRAIN_STEPS = 200000     # 总训练步数
T_EVAL_STEPS = 1000        # 执行评估的频率(以步数为单位)
T_EVAL_ITERS = 250         # 评估模型的迭代次数
T_LR_DECAY_STEP = 50000    # 学习率衰减的步数
T_LR = 5e-4                # 训练的初始学习率
T_LR_DECAYED = 5e-5        # 衰减后的学习率
T_OUT_PATH = "models/transformer_B.pt"  # 保存训练模型的路径

# Device configuration
DEVICE = 'cuda'              # 设备配置，使用CUDA

# 将所有配置存储在字典中以便于访问和修改
default_config = {
    'vocab_size': VOCAB_SIZE,
    'context_length': CONTEXT_LENGTH,
    'n_embed': N_EMBED,
    'n_head': N_HEAD,
    'n_blocks': N_BLOCKS,
    'train_path': TRAIN_PATH,
    'dev_path': DEV_PATH,
    't_batch_size': T_BATCH_SIZE,
    't_context_length': T_CONTEXT_LENGTH,
    't_train_steps': T_TRAIN_STEPS,
    't_eval_steps': T_EVAL_STEPS,
    't_eval_iters': T_EVAL_ITERS,
    't_lr_decay_step': T_LR_DECAY_STEP,
    't_lr': T_LR,
    't_lr_decayed': T_LR_DECAYED,
    't_out_path': T_OUT_PATH,
    'device': DEVICE,
}