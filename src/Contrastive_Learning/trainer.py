import os
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .dataset import ClusterAwareBatchSampler, ContrastiveDataset
from .loss import WeightedSupConLoss


def get_project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def run_contrastive_learning(
    iter_n: int,
    model_dir_name: str = "codet5-small",
    batch_size: int = 12,
    num_epochs: int = 8,
    learning_rate: float = 1e-5,
    temperature: float = 0.07,
):
    """一轮 CodeT5 Encoder 监督对比学习。"""

    project_root = get_project_root()
    text_file = project_root / "output" / "predicate.txt"
    # 保留旧路径参数以兼容外部调用；训练本身不读取该文件。
    embeddings_file = project_root / "output" / "predicate_embeddings_codet5.npy"

    cluster_dir = project_root / "output_cluster" / f"iter_{iter_n}"
    labels_file = cluster_dir / "operator_cluster_labels.npy"
    cluster_summary_file = cluster_dir / "cluster_summary.csv"
    model_dir = project_root / model_dir_name
    encoder_state_path = model_dir / "encoder_state.pt"

    print("========== Encoder-only WeightedSupConLoss 阶段开始 ==========")
    print(f"使用的聚类结果目录: {cluster_dir}")
    print(f"使用的 CodeT5 Encoder 目录: {model_dir}")
    print(f"Encoder checkpoint: {encoder_state_path}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")

    if str(project_root) not in sys.path:
        sys.path.append(str(project_root))

    from transformer_encoder.sql_vae_codet5 import CodeT5Encoder

    model = CodeT5Encoder(
        model_name_or_path=str(model_dir),
        device=torch.device(device),
    )
    tokenizer = model.tokenizer

    if encoder_state_path.exists():
        print(f"[信息] 加载上一轮 Encoder checkpoint: {encoder_state_path}")
        state_dict = torch.load(encoder_state_path, map_location=device)
        model.load_encoder_state_dict(state_dict)
    else:
        print("[信息] 未发现 encoder_state.pt，本轮从原始 CodeT5-small 权重开始。")

    model.train()

    dataset = ContrastiveDataset(
        text_file=text_file,
        embeddings_file=embeddings_file,
        labels_file=labels_file,
        cluster_summary_file=cluster_summary_file,
        tokenizer=tokenizer,
    )
    batch_sampler = ClusterAwareBatchSampler(
        labels=dataset.labels,
        batch_size=batch_size,
        shuffle=True,
    )
    dataloader = DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        collate_fn=dataset.collate_fn,
    )

    print(f"数据集大小: {len(dataset)} 个样本")
    print(f"每个 epoch 的 batch 数量: {len(dataloader)}")

    # 只更新 Encoder（其中包含 T5 的共享输入 embedding）。
    encoder_parameters = model.base_model.get_encoder().parameters()
    optimizer = torch.optim.AdamW(encoder_parameters, lr=learning_rate)
    contrastive_criterion = WeightedSupConLoss(temperature=temperature)

    for epoch in range(num_epochs):
        print(f"\n=========== Epoch {epoch + 1}/{num_epochs} ===========")
        loss_sum = 0.0
        step_count = 0

        for batch_idx, batch in enumerate(dataloader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            weights = batch["weights"].to(device)

            # Encoder mean pooling 与聚类使用相同的句向量表示。
            _token_hidden, pooled_representation = model.encode_encoder_only(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            contrastive_loss = contrastive_criterion(
                embeddings=pooled_representation,
                labels=labels,
                sample_weights=weights,
            )

            optimizer.zero_grad()
            contrastive_loss.backward()
            optimizer.step()

            loss_value = float(contrastive_loss.detach().item())
            loss_sum += loss_value
            step_count += 1

            if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == len(dataloader):
                stats = getattr(contrastive_criterion, "last_stats", {})
                print(
                    f"[Encoder] Epoch {epoch + 1}/{num_epochs} | "
                    f"Batch {batch_idx + 1}/{len(dataloader)} | "
                    f"WeightedSupCon: {loss_value:.4f} | "
                    f"valid_anchor={stats.get('valid_anchors', 0)} | "
                    f"skipped_anchor={stats.get('skipped_anchors', 0)}"
                )

        print(f"[Encoder] Epoch {epoch + 1} 平均损失: {loss_sum / max(step_count, 1):.4f}")

    os.makedirs(model_dir, exist_ok=True)
    torch.save(model.encoder_state_dict(), encoder_state_path)
    print(f"已保存 Encoder checkpoint（未覆盖基础 CodeT5）: {encoder_state_path}")
    print("========== Encoder-only WeightedSupConLoss 阶段完成 ==========")
