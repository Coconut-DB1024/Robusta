from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, Sampler


class ContrastiveDataset(Dataset):
    def __init__(
        self,
        text_file: Path,
        embeddings_file: Optional[Path],
        labels_file: Path,
        cluster_summary_file: Path,
        tokenizer=None,
        max_length: Optional[int] = None,
    ):
        super().__init__()

        with open(text_file, "r", encoding="utf-8") as f:
            self.texts: List[str] = [
                line.strip() for line in f.readlines() if line.strip()
            ]

        self.embeddings_file = embeddings_file

        self.labels: np.ndarray = np.load(labels_file)
        n_texts = len(self.texts)
        n_labels = len(self.labels)
        if n_texts != n_labels:
            raise ValueError(
                f'Data count mismatch: {n_texts} texts and {n_labels} labels. Please check.'
            )

        df = pd.read_csv(cluster_summary_file)
        cluster_size_map: Dict[int, int] = {
            int(row["cluster_id"]): int(row["cluster_size"])
            for _, row in df.iterrows()
        }

        weights = []
        for cid in self.labels:
            size = cluster_size_map.get(int(cid), 1)
            weights.append(1.0 / np.sqrt(size))
        self.sample_weights: np.ndarray = np.asarray(weights, dtype=np.float32)

        self.tokenizer = tokenizer
        self.max_length = max_length
        self.tokenized_inputs: Optional[List[torch.Tensor]] = None
        if tokenizer is not None:
            encoded = tokenizer(
                self.texts,
                truncation=True,
                padding=False,
                max_length=max_length,
                add_special_tokens=True,
            )
            self.tokenized_inputs = [
                torch.tensor(ids, dtype=torch.long)
                for ids in encoded["input_ids"]
            ]

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int):
        item = {
            "text": self.texts[idx],
            "label": torch.tensor(int(self.labels[idx]), dtype=torch.long),
            "weight": torch.tensor(float(self.sample_weights[idx]), dtype=torch.float32),
        }
        if self.tokenized_inputs is not None:
            item["input_ids"] = self.tokenized_inputs[idx]
        return item

    def collate_fn(self, batch):
        if self.tokenized_inputs is None:
            raise RuntimeError('ContrastiveDataset.collate_fn requires a tokenizer.')

        pad_id = self.tokenizer.pad_token_id
        input_ids = pad_sequence(
            [item["input_ids"] for item in batch],
            batch_first=True,
            padding_value=pad_id,
        )
        attention_mask = (input_ids != pad_id).long()
        return {
            "texts": tuple(item["text"] for item in batch),
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": torch.stack([item["label"] for item in batch]),
            "weights": torch.stack([item["weight"] for item in batch]),
        }


class ClusterAwareBatchSampler(Sampler[List[int]]):
    def __init__(self, labels: np.ndarray, batch_size: int, shuffle: bool = True):
        self.labels = np.asarray(labels)
        self.batch_size = max(2, int(batch_size))
        self.shuffle = bool(shuffle)

    def __iter__(self):
        rng = np.random.default_rng()
        groups: Dict[int, List[int]] = {}
        for index, label in enumerate(self.labels):
            groups.setdefault(int(label), []).append(int(index))

        pair_units: List[List[int]] = []
        for indices in groups.values():
            if self.shuffle:
                rng.shuffle(indices)
            pair_units.extend(
                [indices[pos : pos + 2] for pos in range(0, len(indices), 2)]
            )

        if self.shuffle:
            rng.shuffle(pair_units)

        batch: List[int] = []
        for unit in pair_units:
            if batch and len(batch) + len(unit) > self.batch_size:
                yield batch
                batch = []
            batch.extend(unit)
        if batch:
            yield batch

    def __len__(self) -> int:
        counts: Dict[int, int] = {}
        for label in self.labels:
            counts[int(label)] = counts.get(int(label), 0) + 1
        unit_lengths: List[int] = []
        for count in counts.values():
            unit_lengths.extend([2] * (count // 2))
            if count % 2:
                unit_lengths.append(1)

        batches = 0
        current_size = 0
        for unit_length in unit_lengths:
            if current_size and current_size + unit_length > self.batch_size:
                batches += 1
                current_size = 0
            current_size += unit_length
        return batches + int(current_size > 0)
