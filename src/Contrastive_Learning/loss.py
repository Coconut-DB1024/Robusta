from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class WeightedSupConLoss(nn.Module): #带权重的监督对比损失
    def __init__(self, temperature: float = 0.07): #temperature: 温度系数,控制 softmax
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
    ) -> torch.Tensor: #计算一个 batch 的带权对比损失。
        device = embeddings.device

        labels = labels.to(device)

        # 如果没有传权重，则默认每个样本权重为 1
        if sample_weights is None:
            sample_weights = torch.ones_like(labels, dtype=torch.float32)
        else:
            sample_weights = sample_weights.to(device=device, dtype=torch.float32)

        # batch 大小
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

        # 如果 batch 中样本太少，无法做对比，直接返回 0
        if batch_size <= 1:
            return embeddings.sum() * 0.0

        # 向量 L2 归一化
        # 使相似度更接近“余弦相似度”
        embeddings = F.normalize(embeddings, p=2, dim=1)

        # 计算相似度矩阵,结果是一个 (B, B) 的矩阵
        # sim(i,j) = dot(z_i, z_j) / temperature
        similarity_matrix = torch.matmul(embeddings, embeddings.T) / self.temperature

        # 构造标签相等矩阵和自身掩码
        # labels_eq[i,j] = 1 表示 i 和 j 属于同一簇；0 表示不同簇
        labels_eq = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()

        # self_mask[i,j] = 1 当且仅当 i == j，表示自身位置
        self_mask = torch.eye(batch_size, dtype=torch.float32, device=device)

        # 正样本掩码：同簇且不是自身
        positive_mask = labels_eq * (1.0 - self_mask)

        # 对除自身以外的位置做 log-sum-exp，数学公式与原实现一致但更稳定。
        logits = similarity_matrix
        logits_mask = 1.0 - self_mask
        masked_logits = logits.masked_fill(self_mask.bool(), float("-inf"))
        log_prob_denominator = torch.logsumexp(masked_logits, dim=1, keepdim=True)

        # 计算 log_prob(i, j)
        # log_prob[i,j] = sim(i,j)/tau - log(sum_{a != i} exp(sim(i,a)/tau))
        log_prob = logits - log_prob_denominator

        # 针对正样本求平均 log_prob
        # sum_pos_log_prob[i] = sum_{p in P(i)} log_prob[i,p]
        sum_pos_log_prob = (positive_mask * log_prob).sum(dim=1)

        # num_pos[i] = 正样本数量 |P(i)|
        num_pos = positive_mask.sum(dim=1)

        # 有些样本可能在当前 batch 内没有正样本（num_pos = 0），我们需要跳过它们
        valid_mask = num_pos > 0

        if valid_mask.sum() == 0:
            # 如果整个 batch 都没有有效的锚点，返回 0
            return embeddings.sum() * 0.0

        # 对每个有效锚点 i，计算平均 log_prob：sum / num_pos
        mean_log_prob_pos = sum_pos_log_prob[valid_mask] / num_pos[valid_mask]

        # 对应的样本权重也只取有效锚点的部分
        valid_weights = sample_weights[valid_mask]
        self.last_stats["valid_anchors"] = int(valid_mask.sum().item())
        self.last_stats["skipped_anchors"] = int((~valid_mask).sum().item())

        # L_i = - mean_log_prob_pos[i]
        # 带权平均：sum( w_i * L_i ) / sum(w_i)
        loss = - (valid_weights * mean_log_prob_pos).sum() / (valid_weights.sum() + 1e-12)

        return loss
