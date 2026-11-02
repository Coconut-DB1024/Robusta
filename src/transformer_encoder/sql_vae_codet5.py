"""CodeT5 encoder used by predicate encoding and contrastive learning."""

import json
import os
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from RL.config import DB_CONFIG


def _resolve_masking_rules_path(db_type: Optional[str] = None) -> str:
    """Return the existing SQL vocabulary configuration for the active DBMS."""
    normalized_db_type = str(
        db_type if db_type is not None else getattr(DB_CONFIG, "db_type", "mysql")
    ).strip().lower()
    if not normalized_db_type:
        normalized_db_type = "mysql"
    return os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "masking-config",
        f"{normalized_db_type}_masking_rules.json",
    )


class _EncoderBackbone(nn.Module):
    """Keep the pretrained shared embedding and encoder, with no decoder."""

    def __init__(self, pretrained_model):
        super().__init__()
        self.config = pretrained_model.config
        self.shared = pretrained_model.shared
        self.encoder = pretrained_model.get_encoder()

    def get_encoder(self):
        return self.encoder


def _preserve_initialization_rng(hidden_size: int) -> None:
    """Preserve seeded training after removing unused parameter initialization.

    The previous stage-one constructor consumed these six random draws after
    resizing token embeddings. Advancing the same generator keeps subsequent
    dropout and DataLoader draws unchanged, without creating discarded modules,
    model parameters, or a selectable latent-model path.
    """
    for count in (
        hidden_size * 256, 256,
        hidden_size * 256, 256,
        256 * hidden_size, hidden_size,
    ):
        torch.empty(count).uniform_()


class CodeT5Encoder(nn.Module):
    """Encode SQL with CodeT5 and attention-mask-aware mean pooling."""

    def __init__(
        self,
        model_name_or_path: str = "./codet5-small",
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        vocabulary_path = _resolve_masking_rules_path()
        if os.path.exists(vocabulary_path):
            with open(vocabulary_path, "r", encoding="utf-8") as handle:
                rules = json.load(handle)
            custom_tokens = rules.get("sql_vocabulary", []) + rules.get("functions", [])
            num_added = self.tokenizer.add_tokens(custom_tokens)
            if num_added > 0:
                print(f"[CodeT5Encoder] Added {num_added} SQL vocabulary tokens.")
        else:
            print(f"[CodeT5Encoder] SQL vocabulary configuration not found: {vocabulary_path}")

        # Keep the same pretrained loader and embedding resize operation so
        # existing weights, added-token initialization and RNG behavior match.
        pretrained_model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path)
        pretrained_model.resize_token_embeddings(len(self.tokenizer))
        self.hidden_size = pretrained_model.config.d_model
        self.base_model = _EncoderBackbone(pretrained_model)
        del pretrained_model
        _preserve_initialization_rng(self.hidden_size)

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.to(self.device)

    def encode_encoder_only(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        encoder_outputs = self.base_model.get_encoder()(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        token_embeddings = encoder_outputs.last_hidden_state
        return token_embeddings, self.mean_pooling(token_embeddings, attention_mask)

    def encoder_state_dict(self) -> Dict[str, Dict[str, torch.Tensor]]:
        """Keep the existing encoder/shared checkpoint format."""
        return {
            "encoder": self.base_model.get_encoder().state_dict(),
            "shared": self.base_model.shared.state_dict(),
        }

    def load_encoder_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load existing structured or flat encoder checkpoints."""
        if "encoder" in state_dict:
            self.base_model.get_encoder().load_state_dict(state_dict["encoder"], strict=False)
            if "shared" in state_dict:
                self.base_model.shared.load_state_dict(state_dict["shared"], strict=False)
            return
        self.base_model.get_encoder().load_state_dict(state_dict, strict=False)

    def mean_pooling(
        self,
        token_embeddings: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        mask_expanded = attention_mask.unsqueeze(-1)
        masked_embeddings = token_embeddings * mask_expanded
        sum_embeddings = masked_embeddings.sum(dim=1)
        lengths = mask_expanded.sum(dim=1).clamp(min=1e-8)
        return sum_embeddings / lengths


# Preserve import resolution for untouched modules outside stage one.
# This name exposes only the encoder; no VAE implementation or mode remains.
SQLVAEModel = CodeT5Encoder
