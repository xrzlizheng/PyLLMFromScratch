import os
import argparse
import requests
from tqdm import tqdm
from typing import List

# 数据集文件的基础URL
BASE_URL = "https://huggingface.co/datasets/monology/pile-uncopyrighted/resolve/main"
VAL_URL = f"{BASE_URL}/val.jsonl.zst"  # 验证数据集的URL
TRAIN_URLS = [f"{BASE_URL}/train/{i:02d}.jsonl.zst" for i in range(65)]  # 65个训练文件的URL（如果需要可以调整范围）

def download_file(url: str, file_name: str) -> None:
    """
    从给定URL下载文件并保存到指定文件名。
    使用tqdm显示进度条。
    
    参数:
        url (str): 要下载的文件的URL。
        file_name (str): 文件保存的本地路径。
    """
    print(f"正在下载: {file_name}...")
    response = requests.get(url, stream=True)  # 流式传输文件内容
    total_size = int(response.headers.get('content-length', 0))  # 获取文件总大小（如果可用）
    block_size = 1024  # 进度条的每个块大小
    with open(file_name, 'wb') as f:  # 以二进制模式打开文件进行写入
        for chunk in tqdm(response.iter_content(block_size), total=total_size // block_size, desc="正在下载", leave=True):
            f.write(chunk)  # 将每个块写入文件

def download_dataset(val_url: str, train_urls: List[str], val_dir: str, train_dir: str, max_train_files: int) -> None:
    """
    管理数据集的下载，包括验证和训练文件。
    
    参数:
        val_url (str): 验证数据集的URL。
        train_urls (list): 训练数据集文件的URL列表。
        val_dir (str): 验证文件存储的目录。
        train_dir (str): 训练文件存储的目录。
        max_train_files (int): 要下载的最大训练文件数。
    """
    # Define the path for the validation file
    val_file_path = os.path.join(val_dir, "val.jsonl.zst")
    if not os.path.exists(val_file_path):  # Check if the validation file already exists
        print(f"Validation file not found. Downloading from {val_url}...")
        download_file(val_url, val_file_path)  # Download the validation file
    else:
        print("Validation data already present. Skipping download.")

    # Loop through the training file URLs and download if not already present
    for idx, url in enumerate(train_urls[:max_train_files]):  # Limit to max_train_files
        file_name = f"{idx:02d}.jsonl.zst"  # Format file name (e.g., 00.jsonl.zst)
        file_path = os.path.join(train_dir, file_name)  # Construct the full file path
        if not os.path.exists(file_path):  # Check if the file already exists
            print(f"Training file {file_name} not found. Downloading...")
            download_file(url, file_path)  # Download the training file
        else:
            print(f"Training file {file_name} already present. Skipping download.")

def main() -> None:
    """
    Main function to parse arguments and orchestrate the dataset download process.
    """
    # Parse command-line arguments using argparse
    parser = argparse.ArgumentParser(description="Download PILE dataset.")  # Description of the script
    parser.add_argument('--train_max', type=int, default=1, help="Max number of training files to download.")  # Max training files
    parser.add_argument('--train_dir', default="data/train", help="Directory for storing training data.")  # Training directory
    parser.add_argument('--val_dir', default="data/val", help="Directory for storing validation data.")  # Validation directory

    args = parser.parse_args()  # Parse the arguments provided by the user

    # Ensure directories for training and validation data exist
    os.makedirs(args.train_dir, exist_ok=True)  # Create training directory if it doesn't exist
    os.makedirs(args.val_dir, exist_ok=True)  # Create validation directory if it doesn't exist

    # Start downloading the dataset
    download_dataset(VAL_URL, TRAIN_URLS, args.val_dir, args.train_dir, args.train_max)

    print("Dataset downloaded successfully.")  # Indicate successful download

if __name__ == "__main__":
    # Entry point of the script
    main()