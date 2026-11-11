from contextlib import contextmanager
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import struct
from tempfile import TemporaryDirectory, gettempdir
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from experiment_core import encode, infer_features, make_model, normalized_labels, vectorized_qerror
from mscn.data import load_data
from mscn.model import SetConv
from mscn.util import unnormalize_labels
from train import train_mscn

METHODS = {"mscn": "MSCN", "robusta": "Robusta"}
WORKLOADS = {"job-light": ("JOB-light", 70), "job-exclude": ("JOB-exclude", 42)}
MODES = {"predict", "train_and_predict"}


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


class Progress:
    def __init__(self, method, workload):
        self.method = method
        self.workload = workload
        self.last_print = -math.inf
        self.phase = None

    def __call__(self, phase, completed, total):
        name = self.method if phase == "Training" and self.method == "MSCN" else f"{self.method} {self.workload}"
        label = f"{name} {phase.lower()}"
        if phase != self.phase:
            print(f"Starting {label}", flush=True)
        now = time.monotonic()
        if phase != self.phase or completed == total or now - self.last_print >= 10:
            width = 20
            filled = completed * width // total
            bar = "#" * filled + "-" * (width - filled)
            print(f"{label} [{bar}] {completed / total:6.1%}", flush=True)
            self.last_print = now
            self.phase = phase

    def finish(self, phase):
        name = self.method if phase == "Training" else f"{self.method} {self.workload}"
        print(f"{name} {phase.lower()} completed", flush=True)


def local_directory(path, parent):
    if path.is_symlink() or getattr(path, "is_junction", lambda: False)() or path.resolve().parent != parent.resolve():
        raise ValueError(f"Unsafe output directory: {path}")
    return path


@contextmanager
def fresh_output(root, name):
    output = local_directory(root / "output", root)
    output.mkdir(exist_ok=True)
    target_key = os.path.normcase(str((output / name).resolve()))
    lock_key = hashlib.sha256(target_key.encode("utf-8")).hexdigest()
    lock_path = Path(gettempdir()) / f"qerror-reproduce-{lock_key}.lock"
    with lock_path.open("a+b") as lock:
        lock.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise RuntimeError(f"Experiment is already running: {name}") from exc
        folder = local_directory(output / name, output)
        if folder.exists():
            shutil.rmtree(folder)
        folder.mkdir()
        yield folder


def archive_output(folder, archive_root):
    archive_root = Path(archive_root).absolute()
    if archive_root.resolve() != archive_root:
        raise ValueError(f"Unsafe output directory: {archive_root}")
    archive_root.mkdir(parents=True, exist_ok=True)
    target = local_directory(archive_root / folder.name, archive_root)
    with TemporaryDirectory(prefix=".qerror-results-", dir=archive_root) as temporary:
        staged = Path(temporary) / folder.name
        shutil.copytree(folder, staged)
        if target.exists():
            shutil.rmtree(target)
        staged.replace(target)


def load_checkpoint(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def result_row(query, sql, prediction, truth):
    prediction, truth = int(prediction), int(truth)
    if prediction < 1 or truth < 1:
        raise ValueError(f"Non-positive cardinality for query {query}.")
    return {
        "query": query, "sql": sql, "prediction": prediction, "true_cardinality": truth,
        "q_error": max(prediction / truth, truth / prediction),
    }


def write_results(folder, method, workload, mode, rows, count):
    if len(rows) != count or len({row["query"] for row in rows}) != count:
        raise ValueError(f"Expected {count} distinct query results for {workload}.")
    with (folder / "per_query.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    errors = np.asarray([row["q_error"] for row in rows], dtype=np.float64)
    p90, p95, p99 = np.percentile(errors, [90, 95, 99])
    summary = {
        "method": method, "workload": workload, "mode": mode, "query_count": len(rows),
        "max_q_error": float(errors.max()), "mean_q_error": float(errors.mean()),
        "median_q_error": float(np.median(errors)),
        "p90_q_error": float(p90), "p95_q_error": float(p95), "p99_q_error": float(p99),
    }
    (folder / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


def mscn_predictions(checkpoint_path, workloads, workload):
    checkpoint = load_checkpoint(checkpoint_path)
    table, column, operator, join = checkpoint["dicts"]
    config = checkpoint["config"]
    sample_count = config["materialized_samples"]
    model = SetConv(len(table) + sample_count, len(column) + len(operator) + 1, len(join),
                    config["hidden_units"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    if workload == "JOB-light":
        prefix = workloads / "job-light"
        joins, predicates, tables, bitmaps, truths = load_data(str(prefix), sample_count)
        sql = prefix.with_suffix(".sql").read_text(encoding="utf-8").splitlines()
        estimates, _ = infer_features(model, checkpoint, tables, joins, predicates, bitmaps)
        return [result_row(f"JL{i:02}", statement, prediction, truth)
                for i, (statement, prediction, truth) in enumerate(zip(sql, estimates, truths, strict=True), 1)]

    folder = workloads / "job-exclude"
    unique = read_json(folder / "unique_queries.json")
    bitmaps = []
    with (folder / "JOB-exclude-unique.bitmaps").open("rb") as stream:
        for record in unique:
            count, = struct.unpack("<I", stream.read(4))
            if count != len(record["tables"]):
                raise ValueError("Bitmap table order differs from the original evaluation.")
            bitmaps.append(np.vstack([np.unpackbits(np.frombuffer(stream.read(125), dtype=np.uint8))
                                      for _ in range(count)]))
        if stream.read(1):
            raise ValueError("Unexpected trailing bitmap bytes.")
    predictions, _ = infer_features(model, checkpoint, [r["tables"] for r in unique],
                                    [r["joins"] for r in unique], [r["predicates"] for r in unique], bitmaps)
    estimates = {row["sql_sha256"]: p for row, p in zip(unique, predictions, strict=True)}
    truths = {r["sql_sha256"]: int(r["true_cardinality"])
              for r in read_json(folder / "cardinalities_unique.json")}
    rows = []
    for row in read_json(folder / "queries.json"):
        key = row["sql_sha256"]
        query = row["original_job_id"]
        if truths[key] > 0:
            rows.append(result_row(query, row["sql"], estimates[key], truths[key]))
        elif query != "8b":
            raise ValueError(f"Unexpected zero-cardinality query: {query}.")
    return rows


def robusta_input(root, item):
    source = root / item["dataset"]
    if sha256(source) != item["dataset_sha256"]:
        raise ValueError(f"Original dataset differs for query {item['query']}.")
    data = read_json(source)
    training, validation = data["training"], data["validation"]
    labels, lo, hi = normalized_labels(training + validation)
    if (len(training), len(validation), float(lo), float(hi)) != (
            item["train_rows"], item["validation_rows"], item["normalization_min"], item["normalization_max"]):
        raise ValueError(f"Original split or normalization differs for query {item['query']}.")
    return data, labels, lo, hi


def train_robusta(root, items, config, model_dir, progress):
    encoder = load_checkpoint(root / "data/models/robusta_encoding_reference.pt")
    total = sum(item["epochs"] for item in items)
    progress("Training", 0, total)
    completed = 0
    for item in items:
        data, labels, lo, hi = robusta_input(root, item)
        seed = item["model_seed"]
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        model = make_model(encoder)
        weights = b"".join(v.numpy().tobytes() for v in model.state_dict().values())
        if hashlib.sha256(weights).hexdigest() != item["initial_weight_sha256"]:
            raise ValueError(f"Initial weights differ for query {item['query']}.")
        optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
        training = data["training"]
        dataset = encode(training, encoder, data["target"], labels[:len(training)], item["max_predicates"])
        loader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=config["shuffle"])
        for epoch in range(1, item["epochs"] + 1):
            model.train()
            for sample, predicate, join, y, sm, pm, jm in loader:
                optimizer.zero_grad()
                loss = vectorized_qerror(model(sample, predicate, join, sm, pm, jm), y.float(), lo, hi)
                if not torch.isfinite(loss):
                    raise ValueError(f"Non-finite loss for query {item['query']} at epoch {epoch}.")
                loss.backward()
                optimizer.step()
            progress("Training", completed + epoch, total)
        completed += item["epochs"]
        torch.save({
            "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
            "epoch": item["epochs"], "seed": seed, "dicts": encoder["dicts"],
            "column_min_max_vals": encoder["column_min_max_vals"], "min_val": float(lo), "max_val": float(hi),
        }, model_dir / f"{item['query']}.pt")


def robusta_prediction(root, item, checkpoint):
    data, _, lo, hi = robusta_input(root, item)
    encoder = load_checkpoint(checkpoint)
    model = make_model(encoder)
    model.load_state_dict(encoder["model_state_dict"])
    model.eval()
    target = data["target"]
    dataset = encode([target], encoder, target, max_predicates=item["max_predicates"])
    sample, predicate, join, _, sm, pm, jm = dataset.tensors
    with torch.no_grad():
        estimate = model(sample, predicate, join, sm, pm, jm).numpy().flatten()
    cast = np.float64 if item["inference_bounds_scalar_policy"] == "numpy_float64" else float
    prediction = int(np.maximum(unnormalize_labels(estimate, cast(lo), cast(hi)), 1)[0])
    return result_row(item["query"], target["sql"], prediction, target["true_cardinality"])


def run(root, config_path=None, archive_root=None):
    selected = read_json(root / "experiment.json" if config_path is None else Path(config_path))
    method, workload_key, mode = (selected.get(k) for k in ("method", "workload", "mode"))
    if method not in METHODS or workload_key not in WORKLOADS or mode not in MODES:
        raise ValueError("Invalid experiment.json. See README.md for valid options.")
    workload, count = WORKLOADS[workload_key]
    data = root / "data"
    settings = read_json(data / "experiment_settings.json")
    config = settings[method]
    torch.set_num_threads(config["cpu_threads"])
    progress = Progress(METHODS[method], workload_key)
    retrain = mode == "train_and_predict"
    with fresh_output(root, f"{method}_{workload_key}_{mode}") as folder:
        models = folder / "models"
        if retrain:
            models.mkdir()
        if method == "mscn":
            if retrain:
                for name, expected in config["input_hashes"].items():
                    if sha256(root / name) != expected:
                        raise ValueError(f"Original MSCN input differs: {name}.")
                checkpoint = models / "model.pt"
                train_mscn(config, data, checkpoint, progress)
                progress.finish("Training")
            else:
                checkpoint = data / "models/mscn_seed43_epoch100.pt"
                if sha256(checkpoint) != config["checkpoint_sha256"]:
                    raise ValueError("Original MSCN checkpoint differs.")
            progress("Prediction", 0, count)
            rows = mscn_predictions(checkpoint, data / "workloads", workload)
        else:
            items = [item for item in settings["queries"] if item["workload"] == workload]
            if len(items) != count:
                raise ValueError(f"Expected {count} Robusta query settings for {workload}.")
            if retrain:
                train_robusta(root, items, config, models, progress)
                progress.finish("Training")
            progress("Prediction", 0, count)
            rows = []
            for index, item in enumerate(items, 1):
                if retrain:
                    checkpoint = models / f"{item['query']}.pt"
                else:
                    checkpoint = root / item["checkpoint"]
                    if sha256(checkpoint) != item["checkpoint_sha256"]:
                        raise ValueError(f"Original checkpoint differs for query {item['query']}.")
                rows.append(robusta_prediction(root, item, checkpoint))
                if index < count:
                    progress("Prediction", index, count)
        write_results(folder, METHODS[method], workload, mode, rows, count)
        progress("Prediction", count, count)
        progress.finish("Prediction")
        if archive_root is not None:
            archive_output(folder, archive_root)
            print(f"Experiment completed. Results are saved in data/Qerror/{folder.name}/.", flush=True)
