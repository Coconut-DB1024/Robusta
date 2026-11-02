import os
import numpy as np

# 句向量 .npy 文件路径
EMB_FILE = "output/predicate_embeddings_codet5.npy"

# 对应的原始谓词文本文件（可选，用来对照行号）
PREDICATE_FILE = "output/predicate.txt"


def main():
    # 1. 读取 .npy 向量文件
    if not os.path.exists(EMB_FILE):
        raise FileNotFoundError(f"未找到向量文件：{EMB_FILE}")

    embeddings = np.load(EMB_FILE)  # 形状大致是 (num_sentences, hidden_dim)

    print("[信息] 成功加载向量矩阵。")
    print(f"[信息] 向量矩阵形状: {embeddings.shape}")  # 比如 (14, 512)
    print(f"[信息] 向量数据类型: {embeddings.dtype}")

    # 2. 可选：把对应的谓词读出来，对照看
    predicates = []
    if os.path.exists(PREDICATE_FILE):
        with open(PREDICATE_FILE, "r", encoding="utf-8") as f:
            predicates = [line.strip() for line in f if line.strip()]
        print(f"[信息] 谓词文本行数: {len(predicates)}")
    else:
        print(f"[警告] 未找到谓词文本文件：{PREDICATE_FILE}，将只显示向量。")

    # 3. 打印前几条看看
    num_to_show = min(10, embeddings.shape[0])
    for i in range(num_to_show):
        print("\n==============================")
        print(f"[调试] 第 {i} 条谓词向量：")
        if i < len(predicates):
            print(f"  原始谓词: {predicates[i]}")
        print(f"  向量形状: {embeddings[i].shape}")  # (hidden_dim,)
        print(f"  向量前 10 维: {embeddings[i][:20]}")  # 只打印前 10 个数，避免太长
    print("\n[信息] read_predicate_embeddings.py 执行完成。")


if __name__ == "__main__":
    main()