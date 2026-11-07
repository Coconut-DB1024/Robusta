import os
import numpy as np
os.environ["OMP_NUM_THREADS"] = "1"
from sklearn.cluster import KMeans
import json
import csv
from datetime import datetime
from collections import defaultdict

def clustering(
        embedding_path="output/predicate_embeddings_codet5.npy",
        text_path="output/predicate.txt",
        output_dir="output_cluster",
        n_clusters=8,
        max_iter=100,  # KMeans最大迭代次数
        n_init=10,  # KMeans运行次数
        random_state=42,  # 随机种子
        iteration=1,  # 当前聚类编号
        weight_strategy="uniform"  # 当前权重记为1（暂时未用到，保持原样）
):
    final_output_dir = os.path.join("output_cluster", f"iter_{iteration}")
    os.makedirs(final_output_dir, exist_ok=True)  # 创建目录（如果不存在则创建，存在则覆盖内容）

    # 加载向量
    embeddings = np.load(embedding_path)  # shape: (N_op, D)
    N_op, D = embeddings.shape  # 获取样本数和维度

    # 加载对应的文本（谓词整句）
    with open(text_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
    assert len(lines) == N_op, f"Embedding count {N_op} != text line count {len(lines)}"

    operator_texts = lines  # 每一个 embedding 对应一整句谓词

    # 执行KMeans聚类（欧式距离）
    if iteration == 0:
        # 第一次迭代，使用随机初始化
        kmeans = KMeans(
            n_clusters=n_clusters,
            max_iter=max_iter,
            n_init=n_init,
            random_state=random_state
        )
    else:
        # 加载上一次聚类的中心
        prev_centers_path = os.path.join("output_cluster", f"iter_{iteration - 1}", "cluster_centers.npy")

        if os.path.exists(prev_centers_path):
            # 加载上一次的中心
            prev_centers = np.load(prev_centers_path)
            # 新 Encoder 输出为 512 维；旧 VAE 中心为 256 维时不能混用。
            if prev_centers.ndim != 2 or prev_centers.shape[1] != D:
                print(
                    f"Warning: Previous cluster center dimension {getattr(prev_centers, 'shape', None)} != current embedding dimension {D}; using the default KMeans initialization."
                )
                prev_centers = None

            # 检查上一次的中心数量是否与本次 n_clusters 匹配
            if prev_centers is not None and prev_centers.shape[0] != n_clusters:
                print(
                    f"Warning: Previous cluster centers count ({prev_centers.shape[0]}) != n_clusters ({n_clusters}). Adjusting..."
                )
                # 如果数量不匹配，进行调整
                if prev_centers.shape[0] > n_clusters:
                    # 如果上一次的中心数量多于本次，取前n_clusters个
                    prev_centers = prev_centers[:n_clusters]
                else:
                    # 如果上一次的中心数量少于本次，用随机点补充
                    additional_centers = np.random.rand(n_clusters - prev_centers.shape[0], prev_centers.shape[1])
                    prev_centers = np.vstack([prev_centers, additional_centers])

            if prev_centers is not None:
                # 使用上一次的中心作为初始化
                kmeans = KMeans(
                    n_clusters=n_clusters,
                    max_iter=max_iter,
                    n_init=1,  # 只运行一次初始化，因为已经提供了初始中心
                    init=prev_centers,
                    random_state=random_state
                )
            else:
                kmeans = KMeans(
                    n_clusters=n_clusters,
                    max_iter=max_iter,
                    n_init=n_init,
                    random_state=random_state
                )
        else:
            # 如果没有上一次的结果，使用随机初始化
            kmeans = KMeans(
                n_clusters=n_clusters,
                max_iter=max_iter,
                n_init=n_init,
                random_state=random_state
            )

    # 对嵌入向量进行聚类，返回每个样本的簇标签（整数 0 ~ K-1）
    labels = kmeans.fit_predict(embeddings)
    # 获取每个簇的中心向量
    centers = kmeans.cluster_centers_
    # 获取类内误差平方和
    inertia = kmeans.inertia_

    # 保存聚类标签和簇中心
    np.save(os.path.join(final_output_dir, "operator_cluster_labels.npy"), labels)
    np.save(os.path.join(final_output_dir, "cluster_centers.npy"), centers.astype(np.float32))

    # 按簇汇总信息，写 cluster_summary.csv
    cluster_to_ops = defaultdict(list)
    for op_id, label in enumerate(labels):
        cluster_to_ops[label].append(op_id)

    summary_path = os.path.join(final_output_dir, "cluster_summary.csv")
    with open(summary_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["cluster_id", "cluster_size", "operator_ids", "operator_texts"])
        for cid in sorted(cluster_to_ops.keys()):
            op_ids = cluster_to_ops[cid]
            op_texts = [operator_texts[i] for i in op_ids]  # 这里现在是整句谓词
            writer.writerow([
                cid,
                len(op_ids),
                "|".join(map(str, op_ids)),
                "[]".join(op_texts)
            ])

    # WeightedSupConLoss 直接由 batch 内 cluster labels 构造正/负样本，
    # 因此不再枚举或保存未被任何 active 代码读取的 contrastive_pairs.npy。

    # 保存聚类元信息
    meta = {
        "iteration": iteration,
        "n_clusters": n_clusters,
        "n_operators": N_op,
        "inertia": float(inertia),
        "max_iter": max_iter,
        "n_init": n_init,
        "random_state": random_state,
        "timestamp": datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    }
    with open(os.path.join(final_output_dir, "cluster_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f'[INFO] Clustering iteration {iteration + 1} completed')
