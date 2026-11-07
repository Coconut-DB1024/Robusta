import os
import numpy as np

# 句向量 .npy 文件路径
EMB_FILE = "output/predicate_embeddings_codet5.npy"

# 对应的原始谓词文本文件（可选，用来对照行号）
PREDICATE_FILE = "output/predicate.txt"


def main():
    # 1. 读取 .npy 向量文件
    if not os.path.exists(EMB_FILE):
        raise FileNotFoundError(f'Embedding file not found: {EMB_FILE}')

    embeddings = np.load(EMB_FILE)  # 形状大致是 (num_sentences, hidden_dim)

    print('[INFO] Embedding matrix loaded successfully.')
    print(f'[INFO] Embedding matrix shape: {embeddings.shape}')  # 比如 (14, 512)
    print(f'[INFO] Embedding data type: {embeddings.dtype}')

    # 2. 可选：把对应的谓词读出来，对照看
    predicates = []
    if os.path.exists(PREDICATE_FILE):
        with open(PREDICATE_FILE, "r", encoding="utf-8") as f:
            predicates = [line.strip() for line in f if line.strip()]
        print(f'[INFO] Predicate text lines: {len(predicates)}')
    else:
        print(f'[Warning] Predicate text file not found: {PREDICATE_FILE}; displaying embeddings only.')

    # 3. 打印前几条看看
    num_to_show = min(10, embeddings.shape[0])
    for i in range(num_to_show):
        print("\n==============================")
        print(f'[DEBUG] Embedding for predicate {i}:')
        if i < len(predicates):
            print(f'  Original predicate: {predicates[i]}')
        print(f'  Embedding shape: {embeddings[i].shape}')  # (hidden_dim,)
        print(f'  First 10 dimensions: {embeddings[i][:20]}')  # 只打印前 10 个数，避免太长
    print('\n[INFO] read_predicate_embeddings.py completed.')


if __name__ == "__main__":
    main()