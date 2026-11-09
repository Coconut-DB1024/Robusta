from __future__ import annotations

from functools import partial
from typing import Any, Callable, Dict, Tuple, Type

import numpy as np
import torch
from torch import nn

from stable_baselines3.common.distributions import CategoricalDistribution
from stable_baselines3.common.policies import ActorCriticPolicy


class CandidateEdgeScorer(nn.Module):
    def __init__(
        self,
        features_dim: int,
        candidate_pool_size: int,
        candidate_feature_dim: int,
        activation_fn: Type[nn.Module],
    ) -> None:
        super().__init__()
        self.candidate_pool_size = int(candidate_pool_size)
        self.candidate_feature_dim = int(candidate_feature_dim)
        expected_dim = self.candidate_pool_size * self.candidate_feature_dim
        if int(features_dim) != expected_dim:
            raise ValueError(
                "candidate policy observation size mismatch: "
                f"features_dim={features_dim}, expected={expected_dim}"
            )

        edge_hidden_dim = 128
        edge_context_dim = 64
        value_hidden_dim = 128
        self.edge_encoder = nn.Sequential(
            nn.Linear(self.candidate_feature_dim, edge_hidden_dim),
            activation_fn(),
            nn.Linear(edge_hidden_dim, edge_context_dim),
            activation_fn(),
        )
        self.edge_score = nn.Linear(edge_context_dim, 1)
        self.value_context = nn.Sequential(
            nn.Linear(edge_context_dim * 2, value_hidden_dim),
            activation_fn(),
        )

        self.latent_dim_pi = self.candidate_pool_size
        self.latent_dim_vf = value_hidden_dim

    def _reshape(self, features: torch.Tensor) -> torch.Tensor:
        return features.reshape(
            features.shape[0],
            self.candidate_pool_size,
            self.candidate_feature_dim,
        )

    def forward_actor(self, features: torch.Tensor) -> torch.Tensor:
        rows = self._reshape(features)
        encoded = self.edge_encoder(rows)
        logits = self.edge_score(encoded).squeeze(-1)
        valid_rows = rows.abs().sum(dim=-1) > 0
        return logits.masked_fill(~valid_rows, -1e9)

    def forward_critic(self, features: torch.Tensor) -> torch.Tensor:
        rows = self._reshape(features)
        encoded = self.edge_encoder(rows)
        valid_rows = (rows.abs().sum(dim=-1) > 0).unsqueeze(-1)
        valid_float = valid_rows.to(dtype=encoded.dtype)
        denominator = valid_float.sum(dim=1).clamp_min(1.0)
        mean_context = (encoded * valid_float).sum(dim=1) / denominator
        max_context = encoded.masked_fill(~valid_rows, -1e9).max(dim=1).values
        any_valid = valid_rows.any(dim=1)
        max_context = torch.where(any_valid, max_context, torch.zeros_like(max_context))
        return self.value_context(torch.cat([mean_context, max_context], dim=-1))

    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.forward_actor(features), self.forward_critic(features)


class CandidateScoringPolicy(ActorCriticPolicy):
    def __init__(
        self,
        *args: Any,
        candidate_pool_size: int = 64,
        candidate_feature_dim: int = 15,
        **kwargs: Any,
    ) -> None:
        self.candidate_pool_size = int(candidate_pool_size)
        self.candidate_feature_dim = int(candidate_feature_dim)
        super().__init__(*args, **kwargs)

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = CandidateEdgeScorer(
            features_dim=self.features_dim,
            candidate_pool_size=self.candidate_pool_size,
            candidate_feature_dim=self.candidate_feature_dim,
            activation_fn=self.activation_fn,
        )

    def _build(self, lr_schedule: Callable[[float], float]) -> None:
        self._build_mlp_extractor()
        if not isinstance(self.action_dist, CategoricalDistribution):
            raise TypeError("CandidateScoringPolicy requires a Discrete action space")

        self.action_net = nn.Identity()
        self.value_net = nn.Linear(self.mlp_extractor.latent_dim_vf, 1)

        if self.ortho_init:
            module_gains = {
                self.features_extractor: np.sqrt(2),
                self.mlp_extractor: np.sqrt(2),
                self.value_net: 1.0,
            }
            if not self.share_features_extractor:
                del module_gains[self.features_extractor]
                module_gains[self.pi_features_extractor] = np.sqrt(2)
                module_gains[self.vf_features_extractor] = np.sqrt(2)
            for module, gain in module_gains.items():
                module.apply(partial(self.init_weights, gain=gain))
            self.mlp_extractor.edge_score.apply(partial(self.init_weights, gain=0.01))

        self.optimizer = self.optimizer_class(
            self.parameters(),
            lr=lr_schedule(1),
            **self.optimizer_kwargs,
        )

    def _get_constructor_parameters(self) -> Dict[str, Any]:
        data = super()._get_constructor_parameters()
        data.update(
            candidate_pool_size=self.candidate_pool_size,
            candidate_feature_dim=self.candidate_feature_dim,
        )
        return data
