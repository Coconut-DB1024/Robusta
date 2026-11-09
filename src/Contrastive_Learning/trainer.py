import os
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .dataset import ClusterAwareBatchSampler, ContrastiveDataset
from .loss import WeightedSupConLoss


def get_project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _print_training_progress(completed, total):
    fraction = min(completed / total, 1.0) if total > 0 else 1.0
    filled = int(fraction * 40)
    bar = "#" * filled + "-" * (40 - filled)
    print(f"[{bar}] {fraction * 100:.2f}%", flush=True)


def run_contrastive_learning(
    iter_n: int,
    model_dir_name: str = "codet5-small",
    batch_size: int = 12,
    num_epochs: int = 8,
    learning_rate: float = 1e-5,
    temperature: float = 0.07,
):
    project_root = get_project_root()
    text_file = project_root / "output" / "predicate.txt"

    embeddings_file = project_root / "output" / "predicate_embeddings_codet5.npy"

    cluster_dir = project_root / "output_cluster" / f"iter_{iter_n}"
    labels_file = cluster_dir / "operator_cluster_labels.npy"
    cluster_summary_file = cluster_dir / "cluster_summary.csv"
    model_dir = project_root / model_dir_name
    encoder_state_path = model_dir / "encoder_state.pt"

    print('========== Encoder-only WeightedSupConLoss stage started ==========')

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if str(project_root) not in sys.path:
        sys.path.append(str(project_root))

    from transformer_encoder.sql_vae_codet5 import CodeT5Encoder

    model = CodeT5Encoder(
        model_name_or_path=str(model_dir),
        device=torch.device(device),
    )
    tokenizer = model.tokenizer

    if encoder_state_path.exists():
        state_dict = torch.load(encoder_state_path, map_location=device)
        model.load_encoder_state_dict(state_dict)
    else:
        print('[INFO] encoder_state.pt not found; starting this round with original CodeT5-small weights.')

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

    encoder_parameters = model.base_model.get_encoder().parameters()
    optimizer = torch.optim.AdamW(encoder_parameters, lr=learning_rate)
    contrastive_criterion = WeightedSupConLoss(temperature=temperature)

    total_samples = len(dataset) * num_epochs
    completed_samples = 0
    _print_training_progress(completed_samples, total_samples)
    last_progress_time = time.monotonic()

    for epoch in range(num_epochs):
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            weights = batch["weights"].to(device)

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

            completed_samples += input_ids.size(0)
            now = time.monotonic()
            if now - last_progress_time >= 10.0 and completed_samples < total_samples:
                _print_training_progress(completed_samples, total_samples)
                last_progress_time = now

    if total_samples > 0:
        _print_training_progress(completed_samples, total_samples)

    os.makedirs(model_dir, exist_ok=True)
    torch.save(model.encoder_state_dict(), encoder_state_path)
    print('========== Encoder-only WeightedSupConLoss stage completed ==========')
