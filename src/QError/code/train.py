import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from mscn.data import get_train_datasets
from mscn.model import SetConv


def unnormalize_torch(vals, min_val, max_val):
    vals = (vals * (max_val - min_val)) + min_val
    return torch.exp(vals)


def qerror_loss(preds, targets, min_val, max_val):
    qerror = []
    preds = unnormalize_torch(preds, min_val, max_val)
    targets = unnormalize_torch(targets, min_val, max_val)
    for i in range(len(targets)):
        if (preds[i] > targets[i]).item():
            qerror.append(preds[i] / targets[i])
        else:
            qerror.append(targets[i] / preds[i])
    return torch.mean(torch.cat(qerror))


def train_mscn(config, data_dir, checkpoint, progress):
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    sample_count = config["materialized_samples"]
    dicts, bounds, lo, hi, _, _, _, _, train_data, _ = get_train_datasets(
        config["queries"], sample_count, data_dir
    )
    table, column, operator, join = dicts
    initial_rng_state = torch.get_rng_state().clone()
    model = SetConv(len(table) + sample_count, len(column) + len(operator) + 1,
                    len(join), config["hidden_units"])
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    loader = DataLoader(train_data, batch_size=config["batch_size"], shuffle=config["shuffle"])
    total_steps = config["epochs"] * len(loader)
    progress("Training", 0, total_steps)
    for epoch in range(config["epochs"]):
        model.train()
        for step, batch in enumerate(loader, 1):
            samples, predicates, joins, labels, sm, pm, jm = batch
            optimizer.zero_grad()
            loss = qerror_loss(model(samples, predicates, joins, sm, pm, jm), labels.float(), lo, hi)
            if not torch.isfinite(loss):
                raise ValueError(f"Non-finite MSCN training loss at epoch {epoch + 1}.")
            loss.backward()
            optimizer.step()
            progress("Training", epoch * len(loader) + step, total_steps)
    torch.save({
        "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
        "config": config, "dicts": dicts, "column_min_max_vals": bounds,
        "min_val": lo, "max_val": hi, "initial_rng_state": initial_rng_state,
        "final_rng_state": torch.get_rng_state(),
    }, checkpoint)
