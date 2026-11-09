import json
import logging
import os
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from RL.config import DB_CONFIG


class _EmbeddingResizeLogFilter(logging.Filter):
    def filter(self, record):
        return not record.getMessage().startswith(
            "The new embeddings will be initialized from a multivariate normal distribution"
        )


def _resolve_masking_rules_path(db_type: Optional[str] = None) -> str:
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
    def __init__(self, pretrained_model):
        super().__init__()
        self.config = pretrained_model.config
        self.shared = pretrained_model.shared
        self.encoder = pretrained_model.get_encoder()

    def get_encoder(self):
        return self.encoder


def _preserve_initialization_rng(hidden_size: int) -> None:
    for count in (
        hidden_size * 256, 256,
        hidden_size * 256, 256,
        256 * hidden_size, hidden_size,
    ):
        torch.empty(count).uniform_()


class CodeT5Encoder(nn.Module):
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
            self.tokenizer.add_tokens(custom_tokens)
        else:
            print(f"[CodeT5Encoder] SQL vocabulary configuration not found: {vocabulary_path}")

        pretrained_model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path)
        resize_logger = logging.getLogger("transformers.modeling_utils")
        resize_log_filter = _EmbeddingResizeLogFilter()
        resize_logger.addFilter(resize_log_filter)
        try:
            pretrained_model.resize_token_embeddings(len(self.tokenizer))
        finally:
            resize_logger.removeFilter(resize_log_filter)
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
        return {
            "encoder": self.base_model.get_encoder().state_dict(),
            "shared": self.base_model.shared.state_dict(),
        }

    def load_encoder_state_dict(self, state_dict: Dict[str, Any]) -> None:
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


SQLVAEModel = CodeT5Encoder
