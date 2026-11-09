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
        max_iter=100,  
        n_init=10,  
        random_state=42,  
        iteration=1,  
        weight_strategy="uniform"  
):
    final_output_dir = os.path.join("output_cluster", f"iter_{iteration}")
    os.makedirs(final_output_dir, exist_ok=True)  

    embeddings = np.load(embedding_path)  
    N_op, D = embeddings.shape  

    with open(text_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
    assert len(lines) == N_op, f"Embedding count {N_op} != text line count {len(lines)}"

    operator_texts = lines  

    if iteration == 0:
        kmeans = KMeans(
            n_clusters=n_clusters,
            max_iter=max_iter,
            n_init=n_init,
            random_state=random_state
        )
    else:
        prev_centers_path = os.path.join("output_cluster", f"iter_{iteration - 1}", "cluster_centers.npy")

        if os.path.exists(prev_centers_path):
            prev_centers = np.load(prev_centers_path)

            if prev_centers.ndim != 2 or prev_centers.shape[1] != D:
                print(
                    f"Warning: Previous cluster center dimension {getattr(prev_centers, 'shape', None)} != current embedding dimension {D}; using the default KMeans initialization."
                )
                prev_centers = None

            if prev_centers is not None and prev_centers.shape[0] != n_clusters:
                print(
                    f"Warning: Previous cluster centers count ({prev_centers.shape[0]}) != n_clusters ({n_clusters}). Adjusting..."
                )

                if prev_centers.shape[0] > n_clusters:
                    prev_centers = prev_centers[:n_clusters]
                else:
                    additional_centers = np.random.rand(n_clusters - prev_centers.shape[0], prev_centers.shape[1])
                    prev_centers = np.vstack([prev_centers, additional_centers])

            if prev_centers is not None:
                kmeans = KMeans(
                    n_clusters=n_clusters,
                    max_iter=max_iter,
                    n_init=1,  
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
            kmeans = KMeans(
                n_clusters=n_clusters,
                max_iter=max_iter,
                n_init=n_init,
                random_state=random_state
            )

    labels = kmeans.fit_predict(embeddings)

    centers = kmeans.cluster_centers_

    inertia = kmeans.inertia_

    np.save(os.path.join(final_output_dir, "operator_cluster_labels.npy"), labels)
    np.save(os.path.join(final_output_dir, "cluster_centers.npy"), centers.astype(np.float32))

    cluster_to_ops = defaultdict(list)
    for op_id, label in enumerate(labels):
        cluster_to_ops[label].append(op_id)

    summary_path = os.path.join(final_output_dir, "cluster_summary.csv")
    with open(summary_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["cluster_id", "cluster_size", "operator_ids", "operator_texts"])
        for cid in sorted(cluster_to_ops.keys()):
            op_ids = cluster_to_ops[cid]
            op_texts = [operator_texts[i] for i in op_ids]  
            writer.writerow([
                cid,
                len(op_ids),
                "|".join(map(str, op_ids)),
                "[]".join(op_texts)
            ])

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
