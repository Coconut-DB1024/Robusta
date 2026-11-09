import os
import shutil
import json
import numpy as np
from datetime import datetime

from sklearn.metrics import adjusted_rand_score  


LABEL_CHANGE_THRESHOLD = 0.10        
INERTIA_DECREASE_THRESHOLD = 0.01    


ARI_THRESHOLD = 0.95


def is_converged(i):
    if i < 1:
        return False

    current_labels_path = os.path.join("output_cluster", f"iter_{i}", "operator_cluster_labels.npy")
    current_meta_path = os.path.join("output_cluster", f"iter_{i}", "cluster_meta.json")

    prev_labels_path = os.path.join("output_cluster", f"iter_{i - 1}", "operator_cluster_labels.npy")
    prev_meta_path = os.path.join("output_cluster", f"iter_{i - 1}", "cluster_meta.json")

    if (
        not os.path.exists(current_labels_path)
        or not os.path.exists(current_meta_path)
        or not os.path.exists(prev_labels_path)
        or not os.path.exists(prev_meta_path)
    ):
        print(f'Warning: Missing files; cannot determine convergence (i={i})')
        return False

    current_labels = np.load(current_labels_path)
    prev_labels = np.load(prev_labels_path)

    diff_count = np.sum(current_labels != prev_labels)  
    total_samples = len(current_labels)
    label_change_ratio = diff_count / total_samples

    ari = adjusted_rand_score(prev_labels, current_labels)

    with open(current_meta_path, "r", encoding="utf-8") as f:
        current_meta = json.load(f)
        current_inertia = current_meta["inertia"]

    with open(prev_meta_path, "r", encoding="utf-8") as f:
        prev_meta = json.load(f)
        prev_inertia = prev_meta["inertia"]

    if prev_inertia <= 0:
        inertia_decrease_ratio = 0.0
    else:
        inertia_decrease_ratio = (prev_inertia - current_inertia) / prev_inertia

    converged = ari >= ARI_THRESHOLD

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

    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)

    shutil.copytree(source_dir, target_dir, dirs_exist_ok=True)

    return True
