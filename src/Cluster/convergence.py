import os
import shutil
import json
import numpy as np
from datetime import datetime

from sklearn.metrics import adjusted_rand_score  # 用于计算 ARI

# 原有阈值（现在主要用于打印参考，不再作为硬收敛条件）
LABEL_CHANGE_THRESHOLD = 0.10        # 标签变化比例阈值（仅做参考）
INERTIA_DECREASE_THRESHOLD = 0.01    # inertia下降比例阈值（仅做参考）

# ★ 新增：基于 ARI 的收敛阈值，越接近 1 表示两次聚类越相似
ARI_THRESHOLD = 0.95


def is_converged(i):
    """
    判断第 i 轮聚类是否相对于第 i-1 轮已经“收敛”。

    之前版本：
        - 直接比较标签数组 current_labels != prev_labels 得到 label_change_ratio
        - 同时看 inertia 是否下降且下降比例 < 0.01

    问题：
        - KMeans 的簇标签顺序是无意义的，可能同一个簇换了编号，导致 label_change_ratio 接近 1.0，
          但实际上簇结构没怎么变。
        - 外循环中 encoder 在更新，inertia 在不同轮次之间也不一定单调下降。

    现在版本：
        - 保留原来的 label_change_ratio 和 inertia 的计算，只用于“打印参考”
        - 真正的收敛判定改为：使用 ARI(adjusted_rand_score) 来衡量两次聚类结果的相似度。
          当 ARI >= ARI_THRESHOLD（例如 0.95）时，认为聚类结构基本稳定，可以视为收敛。
    """
    # 只有 i>=1 时才能和上一轮对比
    if i < 1:
        return False

    # 获取当前迭代 i 的路径
    current_labels_path = os.path.join("output_cluster", f"iter_{i}", "operator_cluster_labels.npy")
    current_meta_path = os.path.join("output_cluster", f"iter_{i}", "cluster_meta.json")

    # 获取前一次迭代 i-1 的路径
    prev_labels_path = os.path.join("output_cluster", f"iter_{i - 1}", "operator_cluster_labels.npy")
    prev_meta_path = os.path.join("output_cluster", f"iter_{i - 1}", "cluster_meta.json")

    # 检查文件是否存在
    if (
        not os.path.exists(current_labels_path)
        or not os.path.exists(current_meta_path)
        or not os.path.exists(prev_labels_path)
        or not os.path.exists(prev_meta_path)
    ):
        print(f'Warning: Missing files; cannot determine convergence (i={i})')
        return False

    # 读取标签数据
    current_labels = np.load(current_labels_path)
    prev_labels = np.load(prev_labels_path)

    # 计算标签变化比例（直接比对整数标签，仅作为参考）
    diff_count = np.sum(current_labels != prev_labels)  # 计算标签不同的样本数
    total_samples = len(current_labels)
    label_change_ratio = diff_count / total_samples

    # 计算 ARI（Permutation 不变，真正反映簇划分是否稳定）
    ari = adjusted_rand_score(prev_labels, current_labels)

    # 读取 inertia 数据
    with open(current_meta_path, "r", encoding="utf-8") as f:
        current_meta = json.load(f)
        current_inertia = current_meta["inertia"]

    with open(prev_meta_path, "r", encoding="utf-8") as f:
        prev_meta = json.load(f)
        prev_inertia = prev_meta["inertia"]

    # 计算 inertia 变化比例（仍然作为参考信息）
    if prev_inertia <= 0:
        inertia_decrease_ratio = 0.0
    else:
        inertia_decrease_ratio = (prev_inertia - current_inertia) / prev_inertia

    # 新的收敛条件：主要看 ARI
    # 当 ARI 足够接近 1（如 >= 0.95）时，认为两次聚类结果高度一致，视为“收敛”
    converged = ari >= ARI_THRESHOLD

    # 打印判断结果
    if converged:
        print(f'[INFO] Clustering converged (iteration {i + 1})')
    else:
        print(
            f'[INFO] Not converged (iteration {i + 1}): label_change_ratio={label_change_ratio:.4f}, inertia_change_ratio={inertia_decrease_ratio:.4f}, ARI={ari:.4f} (< {ARI_THRESHOLD})'
        )

    return converged


def save_final_result(i):
    source_dir = os.path.join("output_cluster", f"iter_{i}")
    target_dir = "data/cluster_final_output"

    # 确保目标目录存在（如果存在则删除）
    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)

    # 直接复制，允许覆盖已存在的目录
    shutil.copytree(source_dir, target_dir, dirs_exist_ok=True)

    # print(f"聚类收敛，结果已保存到: {target_dir}")
    return True
