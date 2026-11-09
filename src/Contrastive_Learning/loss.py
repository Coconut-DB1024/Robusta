from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class WeightedSupConLoss(nn.Module): 
    def __init__(self, temperature: float = 0.07): 
        super().__init__()
        self.temperature = temperature
        self.last_stats = {
            "valid_anchors": 0,
            "skipped_anchors": 0,
            "batch_size": 0,
            "cluster_counts": {},
        }

    def forward(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
        sample_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor: 
        device = embeddings.device

        labels = labels.to(device)

        if sample_weights is None:
            sample_weights = torch.ones_like(labels, dtype=torch.float32)
        else:
            sample_weights = sample_weights.to(device=device, dtype=torch.float32)

        batch_size = embeddings.shape[0]
        unique_labels, unique_counts = torch.unique(labels, return_counts=True)
        self.last_stats = {
            "valid_anchors": 0,
            "skipped_anchors": int(batch_size),
            "batch_size": int(batch_size),
            "cluster_counts": {
                int(label.item()): int(count.item())
                for label, count in zip(unique_labels, unique_counts)
            },
        }

        if batch_size <= 1:
            return embeddings.sum() * 0.0

        embeddings = F.normalize(embeddings, p=2, dim=1)

        similarity_matrix = torch.matmul(embeddings, embeddings.T) / self.temperature

        labels_eq = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()

        self_mask = torch.eye(batch_size, dtype=torch.float32, device=device)

        positive_mask = labels_eq * (1.0 - self_mask)

        logits = similarity_matrix
        logits_mask = 1.0 - self_mask
        masked_logits = logits.masked_fill(self_mask.bool(), float("-inf"))
        log_prob_denominator = torch.logsumexp(masked_logits, dim=1, keepdim=True)

        log_prob = logits - log_prob_denominator

        sum_pos_log_prob = (positive_mask * log_prob).sum(dim=1)

        num_pos = positive_mask.sum(dim=1)

        valid_mask = num_pos > 0

        if valid_mask.sum() == 0:
            return embeddings.sum() * 0.0

        mean_log_prob_pos = sum_pos_log_prob[valid_mask] / num_pos[valid_mask]

        valid_weights = sample_weights[valid_mask]
        self.last_stats["valid_anchors"] = int(valid_mask.sum().item())
        self.last_stats["skipped_anchors"] = int((~valid_mask).sum().item())

        loss = - (valid_weights * mean_log_prob_pos).sum() / (valid_weights.sum() + 1e-12)

        return loss
