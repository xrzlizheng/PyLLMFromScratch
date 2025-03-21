import torch
import numpy as np
import h5py
from typing import Iterator, Tuple

def get_batch_iterator(data_path: str, batch_size: int, context_length: int, device: str = "cpu") -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
    """
    创建一个迭代器，用于从HDF5文件生成数据批次。

    参数:
        data_path (str): 包含token化数据的HDF5文件路径。
        batch_size (int): 每个批次中的序列数量。
        context_length (int): 每个序列的长度。
        device (str, 可选): 加载数据的设备('cpu'或'cuda')，默认为"cpu"。

    生成:
        tuple: 包含输入序列(xb)和目标序列(yb)的元组。
    """
    # 以只读模式打开HDF5文件
    with h5py.File(data_path, 'r') as hdf5_file:

        # 提取token化序列的数据集
        dataset = hdf5_file['tokens']

        # 获取数据集的总大小
        dataset_size = dataset.shape[0]

        # 计算可以从数据中生成的示例(序列)数量
        n_examples = (dataset_size - 1) // context_length

        # 创建示例索引数组并随机打乱顺序
        example_idxs = np.arange(n_examples)
        np.random.shuffle(example_idxs)

        # 初始化epoch计数器和示例计数器
        epochs = 0
        counter = 0

        while True:
            # 检查当前批次是否超过可用示例数量
            if counter + batch_size > n_examples:
                # 再次打乱索引并将计数器重置为0
                np.random.shuffle(example_idxs)
                counter = 0
                print(f"Finished epoch {epochs}")  # 当一个epoch结束时打印epoch编号
                epochs += 1  # 增加epoch计数器

            # 选择一批随机索引来生成序列
            random_indices = example_idxs[counter:counter+batch_size] * context_length

            # Retrieve sequences from the dataset based on the random indices
            random_samples = torch.tensor(np.array([dataset[idx:idx+context_length+1] for idx in random_indices]))

            # Separate the input sequences (xb) and target sequences (yb)
            xb = random_samples[:, :context_length].to(device)  # Input sequence (first half of the random sample)
            yb = random_samples[:, 1:context_length+1].to(device)  # Target sequence (second half of the random sample)

            # Increment the counter to move to the next batch
            counter += batch_size

            # Yield the input and target sequences as a tuple for the current batch
            yield xb, yb

if __name__ == '__main__':
    # Example Usage (requires a dummy HDF5 file for testing)
    # Create a dummy HDF5 file
    import os
    dummy_data_path = "dummy_data.h5"
    if not os.path.exists(dummy_data_path):
        with h5py.File(dummy_data_path, 'w') as f:
            f.create_dataset('tokens', data=np.arange(1000))

    batch_size = 4
    context_length = 10
    for xb, yb in get_batch_iterator(dummy_data_path, batch_size, context_length):
        print("Input Batch Shape:", xb.shape)
        print("Target Batch Shape:", yb.shape)
        break