from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import hnswlib
except ImportError:  
    hnswlib = None

from RL.config import (
    CANDIDATE_BITMAP_CONFIG,
    HNSW_EF_CONSTRUCTION,
    HNSW_EF_SEARCH,
    HNSW_M,
    HNSW_SPACE,
    OPERATOR_PROTOTYPES_PATH,
)
from RL.operator_similarity import normalize_operator_key


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _file_fingerprint(path: Path) -> Dict[str, Any]:
    path = Path(path)
    try:
        stat = path.stat()
        return {
            "path": str(path),
            "size": int(stat.st_size),
            "mtime_ns": int(stat.st_mtime_ns),
        }
    except OSError:
        return {"path": str(path), "size": -1, "mtime_ns": -1}


def _load_operator_prototypes(path: Path) -> Tuple[Dict[str, np.ndarray], int]:
    path = Path(path)
    if not path.exists():
        return {}, 0

    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        return {}, 0

    raw_vectors: List[Tuple[str, np.ndarray]] = []
    dimensions: Dict[int, int] = {}
    for raw_key, raw_item in payload.items():
        if not isinstance(raw_item, dict):
            continue
        raw_vector = raw_item.get("vector")
        try:
            vector = np.asarray(raw_vector, dtype=np.float32).reshape(-1)
        except Exception:
            continue
        if vector.size == 0 or not np.all(np.isfinite(vector)):
            continue

        key = normalize_operator_key(raw_key)
        if not key:
            continue
        raw_vectors.append((key, vector))
        dimensions[int(vector.size)] = dimensions.get(int(vector.size), 0) + 1

    if not dimensions:
        return {}, 0

    prototype_dim = max(dimensions.items(), key=lambda item: (item[1], item[0]))[0]
    prototypes: Dict[str, np.ndarray] = {}
    for key, vector in raw_vectors:
        if int(vector.size) != prototype_dim:
            continue
        norm = float(np.linalg.norm(vector))
        prototypes[key] = vector / norm if norm > 0.0 else vector

    return prototypes, int(prototype_dim)


def _required_slot_counts(edge_records: Sequence[Dict[str, Any]]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for record in edge_records:
        outer_node_id = str(record.get("outer_node_id", ""))
        if not outer_node_id:
            continue
        slot_count = max(0, _safe_int(record.get("slot_index", 0))) + 1
        counts[outer_node_id] = max(int(counts.get(outer_node_id, 0)), slot_count)
    return counts


def compute_structural_entropies(
    edge_records: Sequence[Dict[str, Any]],
) -> np.ndarray:
    if not edge_records:
        return np.zeros((0,), dtype=np.float32)

    slot_counts = _required_slot_counts(edge_records)
    raw_entropy = np.zeros((len(edge_records),), dtype=np.float32)

    for idx, record in enumerate(edge_records):
        outer_node_id = str(record.get("outer_node_id", ""))
        inner_node_id = str(record.get("inner_node_id", ""))
        outer_slots = max(1, int(slot_counts.get(outer_node_id, 1)))
        inner_slots = max(0, int(slot_counts.get(inner_node_id, 0)))
        nested_cost = 1 if inner_slots > 0 else 0
        recursive_cost = 1 if outer_node_id and outer_node_id == inner_node_id else 0
        complexity = outer_slots + inner_slots + nested_cost + recursive_cost
        raw_entropy[idx] = float(math.log2(1.0 + float(complexity)))

    min_value = float(np.min(raw_entropy))
    max_value = float(np.max(raw_entropy))
    if max_value <= min_value:
        return np.zeros_like(raw_entropy, dtype=np.float32)

    normalized = (raw_entropy - min_value) / (max_value - min_value)
    return np.clip(normalized, 0.0, 1.0).astype(np.float32)


def _type_feature_keys(edge_records: Sequence[Dict[str, Any]]) -> List[str]:
    keys = set()
    for record in edge_records:
        keys.add(f"outer:{record.get('outer_return_type', '')}")
        keys.add(f"slot:{record.get('slot_required_type', '')}")
        keys.add(f"inner:{record.get('inner_return_type', '')}")
    return sorted(key for key in keys if not key.endswith(":"))


def build_candidate_edge_vectors(
    edge_records: Sequence[Dict[str, Any]],
    *,
    prototypes_path: Path = OPERATOR_PROTOTYPES_PATH,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    prototypes, prototype_dim = _load_operator_prototypes(Path(prototypes_path))
    n = len(edge_records)
    if n == 0 or prototype_dim <= 0:
        missing_outer_names = sorted({
            normalize_operator_key(record.get("outer_token", ""))
            for record in edge_records
            if normalize_operator_key(record.get("outer_token", ""))
        })
        missing_inner_names = sorted({
            normalize_operator_key(record.get("inner_token", ""))
            for record in edge_records
            if normalize_operator_key(record.get("inner_token", ""))
        })
        return np.zeros((n, 0), dtype=np.float32), {
            "prototype_count": int(len(prototypes)),
            "prototype_dim": int(prototype_dim),
            "vector_dim": 0,
            "missing_outer_prototype_count": int(n),
            "missing_inner_prototype_count": int(n),
            "missing_outer_operator_names": missing_outer_names,
            "missing_inner_operator_names": missing_inner_names,
            "type_feature_count": 0,
        }

    type_keys = _type_feature_keys(edge_records)
    type_index = {key: idx for idx, key in enumerate(type_keys)}

    vector_dim = (2 * prototype_dim) + 3 + len(type_keys)
    vectors = np.zeros((n, vector_dim), dtype=np.float32)
    max_slot = max(
        1,
        max((_safe_int(record.get("slot_index", 0)) for record in edge_records), default=0),
    )
    missing_outer = 0
    missing_inner = 0
    missing_outer_names = set()
    missing_inner_names = set()

    for idx, record in enumerate(edge_records):
        outer_key = normalize_operator_key(record.get("outer_token", ""))
        inner_key = normalize_operator_key(record.get("inner_token", ""))
        outer_vector = prototypes.get(outer_key)
        inner_vector = prototypes.get(inner_key)

        if outer_vector is None:
            missing_outer += 1
            if outer_key:
                missing_outer_names.add(outer_key)
        else:
            vectors[idx, :prototype_dim] = outer_vector

        if inner_vector is None:
            missing_inner += 1
            if inner_key:
                missing_inner_names.add(inner_key)
        else:
            vectors[idx, prototype_dim:(2 * prototype_dim)] = inner_vector

        cursor = 2 * prototype_dim
        vectors[idx, cursor] = float(max(0, _safe_int(record.get("slot_index", 0))) / max_slot)
        vectors[idx, cursor + 1] = 1.0 if outer_vector is not None else 0.0
        vectors[idx, cursor + 2] = 1.0 if inner_vector is not None else 0.0

        for type_key in (
            f"outer:{record.get('outer_return_type', '')}",
            f"slot:{record.get('slot_required_type', '')}",
            f"inner:{record.get('inner_return_type', '')}",
        ):
            type_idx = type_index.get(type_key)
            if type_idx is not None:
                vectors[idx, cursor + 3 + type_idx] = 1.0

    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    vectors = np.divide(
        vectors,
        norms,
        out=np.zeros_like(vectors),
        where=norms > 0.0,
    ).astype(np.float32)

    return vectors, {
        "prototype_count": int(len(prototypes)),
        "prototype_dim": int(prototype_dim),
        "vector_dim": int(vector_dim),
        "missing_outer_prototype_count": int(missing_outer),
        "missing_inner_prototype_count": int(missing_inner),
        "missing_outer_operator_names": sorted(missing_outer_names),
        "missing_inner_operator_names": sorted(missing_inner_names),
        "type_feature_count": int(len(type_keys)),
    }


def _build_hnsw_neighbors(
    vectors: np.ndarray,
    *,
    index_path: Path,
    neighbor_count: int,
) -> Tuple[np.ndarray, np.ndarray]:
    n, vector_dim = vectors.shape
    k = max(0, min(int(neighbor_count), max(0, n - 1)))
    neighbor_indices = np.full((n, k), -1, dtype=np.int32)
    neighbor_distances = np.full((n, k), np.inf, dtype=np.float32)
    if n == 0 or vector_dim == 0 or k == 0:
        return neighbor_indices, neighbor_distances
    if hnswlib is None:
        raise RuntimeError("candidate_hnsw_enabled=True, but hnswlib is not installed")

    index_path = Path(index_path)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index = hnswlib.Index(space=HNSW_SPACE, dim=int(vector_dim))
    index.init_index(
        max_elements=int(n),
        ef_construction=int(HNSW_EF_CONSTRUCTION),
        M=int(HNSW_M),
    )
    labels = np.arange(n, dtype=np.int32)
    index.add_items(vectors, labels)
    index.set_ef(max(int(HNSW_EF_SEARCH), k + 1))
    index.save_index(str(index_path))

    query_k = min(n, k + 1)
    raw_labels, raw_distances = index.knn_query(vectors, k=query_k)
    for row_idx in range(n):
        cursor = 0
        for label, distance in zip(raw_labels[row_idx], raw_distances[row_idx]):
            label_int = int(label)
            if label_int == row_idx:
                continue
            neighbor_indices[row_idx, cursor] = label_int
            neighbor_distances[row_idx, cursor] = float(distance)
            cursor += 1
            if cursor >= k:
                break

    return neighbor_indices, neighbor_distances


def build_candidate_edge_organization(
    edge_records: List[Dict[str, Any]],
    *,
    prototypes_path: Path = OPERATOR_PROTOTYPES_PATH,
) -> Dict[str, Any]:
    enabled = bool(getattr(CANDIDATE_BITMAP_CONFIG, "candidate_organization_enabled", True))
    hnsw_enabled = enabled and bool(
        getattr(CANDIDATE_BITMAP_CONFIG, "candidate_hnsw_enabled", True)
    )
    entropy_enabled = enabled and bool(
        getattr(CANDIDATE_BITMAP_CONFIG, "candidate_entropy_enabled", True)
    )

    entropies = (
        compute_structural_entropies(edge_records)
        if entropy_enabled
        else np.zeros((len(edge_records),), dtype=np.float32)
    )
    for idx, record in enumerate(edge_records):
        record["structural_entropy"] = float(entropies[idx])

    vectors_path = Path(getattr(CANDIDATE_BITMAP_CONFIG, "candidate_edge_vectors_path"))
    index_path = Path(getattr(CANDIDATE_BITMAP_CONFIG, "candidate_edge_hnsw_index_path"))
    neighbors_path = Path(getattr(CANDIDATE_BITMAP_CONFIG, "candidate_edge_neighbors_path"))
    report_path = Path(
        getattr(CANDIDATE_BITMAP_CONFIG, "candidate_edge_organization_report_path")
    )
    for path in (vectors_path, index_path, neighbors_path, report_path):
        path.parent.mkdir(parents=True, exist_ok=True)

    neighbor_count = max(1, int(getattr(CANDIDATE_BITMAP_CONFIG, "candidate_hnsw_k", 32)))
    prototype_fingerprint = _file_fingerprint(Path(prototypes_path))
    if report_path.exists() and neighbors_path.exists() and vectors_path.exists():
        try:
            with report_path.open("r", encoding="utf-8") as f:
                previous_report = json.load(f)
            previous_hnsw = previous_report.get("hnsw", {})
            previous_entropy = previous_report.get("entropy", {})
            previous_fingerprint = previous_report.get("prototype_fingerprint", {})
            previous_neighbor_count = int(previous_hnsw.get("neighbor_count", 0))
            cache_is_current = (
                int(previous_report.get("edge_count", -1)) == len(edge_records)
                and bool(previous_report.get("enabled", False)) == bool(enabled)
                and bool(previous_hnsw.get("enabled", False)) == bool(hnsw_enabled)
                and bool(previous_entropy.get("enabled", False)) == bool(entropy_enabled)
                and previous_fingerprint == prototype_fingerprint
                and (not hnsw_enabled or previous_neighbor_count == min(
                    neighbor_count,
                    max(0, len(edge_records) - 1),
                ))
                and all("structural_entropy" in record for record in edge_records)
            )
            if cache_is_current:
                unresolved_outer = list(
                    previous_report.get("vectors", {}).get(
                        "missing_outer_operator_names", []
                    )
                    or []
                )
                if hnsw_enabled and unresolved_outer:
                    raise RuntimeError(
                        "candidate HNSW preflight failed: unresolved nonterminal outer "
                        f"operator prototypes: {sorted(str(x) for x in unresolved_outer)}"
                    )
                previous_report["reused"] = True
                return previous_report
        except Exception:
            pass

    if enabled:
        vectors, vector_report = build_candidate_edge_vectors(
            edge_records,
            prototypes_path=Path(prototypes_path),
        )
    else:
        vectors = np.zeros((len(edge_records), 0), dtype=np.float32)
        vector_report = {
            "prototype_count": 0,
            "prototype_dim": 0,
            "vector_dim": 0,
            "missing_outer_prototype_count": 0,
            "missing_inner_prototype_count": 0,
            "missing_outer_operator_names": [],
            "missing_inner_operator_names": [],
            "type_feature_count": 0,
        }

    unresolved_outer = list(vector_report.get("missing_outer_operator_names", []) or [])
    if hnsw_enabled and unresolved_outer:
        raise RuntimeError(
            "candidate HNSW preflight failed: unresolved nonterminal outer operator "
            f"prototypes: {sorted(str(x) for x in unresolved_outer)}"
        )

    np.save(vectors_path, vectors)

    hnsw_built = False
    hnsw_error: Optional[str] = None
    try:
        if hnsw_enabled and vectors.shape[1] > 0:
            neighbor_indices, neighbor_distances = _build_hnsw_neighbors(
                vectors,
                index_path=index_path,
                neighbor_count=neighbor_count,
            )
            hnsw_built = True
        else:
            neighbor_indices = np.full((len(edge_records), 0), -1, dtype=np.int32)
            neighbor_distances = np.full((len(edge_records), 0), np.inf, dtype=np.float32)
    except Exception as exc:
        hnsw_error = str(exc)
        neighbor_indices = np.full((len(edge_records), 0), -1, dtype=np.int32)
        neighbor_distances = np.full((len(edge_records), 0), np.inf, dtype=np.float32)

    np.savez_compressed(
        neighbors_path,
        neighbor_indices=neighbor_indices,
        neighbor_distances=neighbor_distances,
    )

    entropy_report = {
        "enabled": bool(entropy_enabled),
        "min": float(np.min(entropies)) if entropies.size else 0.0,
        "mean": float(np.mean(entropies)) if entropies.size else 0.0,
        "max": float(np.max(entropies)) if entropies.size else 0.0,
    }
    report = {
        "version": "candidate_edge_organization_v1",
        "enabled": bool(enabled),
        "edge_count": int(len(edge_records)),
        "prototype_fingerprint": prototype_fingerprint,
        "reused": False,
        "hnsw": {
            "enabled": bool(hnsw_enabled),
            "built": bool(hnsw_built),
            "space": str(HNSW_SPACE),
            "neighbor_count": int(neighbor_indices.shape[1]),
            "error": hnsw_error,
        },
        "entropy": entropy_report,
        "vectors": vector_report,
        "outputs": {
            "vectors_path": str(vectors_path),
            "hnsw_index_path": str(index_path),
            "neighbors_path": str(neighbors_path),
            "report_path": str(report_path),
        },
    }
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    return report


def load_candidate_edge_neighbors(
    path: Path,
    *,
    expected_edge_count: int,
) -> Tuple[np.ndarray, np.ndarray]:
    path = Path(path)
    if not path.exists():
        return (
            np.full((expected_edge_count, 0), -1, dtype=np.int32),
            np.full((expected_edge_count, 0), np.inf, dtype=np.float32),
        )

    with np.load(path) as payload:
        indices = np.asarray(payload.get("neighbor_indices", []), dtype=np.int32)
        distances = np.asarray(payload.get("neighbor_distances", []), dtype=np.float32)
    if (
        indices.ndim != 2
        or distances.ndim != 2
        or indices.shape != distances.shape
        or indices.shape[0] != int(expected_edge_count)
    ):
        return (
            np.full((expected_edge_count, 0), -1, dtype=np.int32),
            np.full((expected_edge_count, 0), np.inf, dtype=np.float32),
        )
    return indices, distances


def compute_neighbor_success_tendencies(
    neighbor_indices: np.ndarray,
    neighbor_distances: np.ndarray,
    observed_bitmap: np.ndarray,
    failed_bitmap: np.ndarray,
    *,
    min_known_neighbors: int,
) -> np.ndarray:
    n = int(len(observed_bitmap))
    tendencies = np.full((n,), 0.5, dtype=np.float32)
    if neighbor_indices.ndim != 2 or neighbor_indices.shape[0] != n:
        return tendencies
    if neighbor_distances.shape != neighbor_indices.shape:
        return tendencies

    observed = np.asarray(observed_bitmap, dtype=np.uint8).reshape(-1)
    failed = np.asarray(failed_bitmap, dtype=np.uint8).reshape(-1)
    minimum = max(1, int(min_known_neighbors))

    for row_idx in range(n):
        weighted_success = 0.0
        total_weight = 0.0
        known_count = 0
        valid_count = 0
        for neighbor_idx, distance in zip(
            neighbor_indices[row_idx],
            neighbor_distances[row_idx],
        ):
            idx = int(neighbor_idx)
            if idx < 0 or idx >= n or not np.isfinite(distance):
                continue
            valid_count += 1
            if observed[idx] > 0:
                label = 1.0
            elif failed[idx] > 0:
                label = 0.0
            else:
                continue

            weight = 1.0 / (1.0 + max(0.0, float(distance)))
            weighted_success += weight * label
            total_weight += weight
            known_count += 1

        if known_count < minimum or total_weight <= 0.0 or valid_count <= 0:
            continue

        success_rate = weighted_success / total_weight
        confidence = min(1.0, float(known_count) / float(valid_count))
        tendencies[row_idx] = float(
            np.clip(0.5 + (success_rate - 0.5) * confidence, 0.0, 1.0)
        )

    return tendencies
