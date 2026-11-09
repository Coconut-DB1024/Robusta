from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

try:
    from RL.config import (
        OPERATOR_PROTOTYPES_PATH,
        OPERATOR_SIMILARITY_PATH,
        OPERATOR_EMBEDDING_METADATA_PATH,
    )
except Exception:
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    OUTPUT_DIR = PROJECT_ROOT / "output"
    OPERATOR_PROTOTYPES_PATH = OUTPUT_DIR / "operator_prototypes.json"
    OPERATOR_SIMILARITY_PATH = OUTPUT_DIR / "operator_similarity.json"
    OPERATOR_EMBEDDING_METADATA_PATH = OUTPUT_DIR / "operator_embedding_metadata.json"


def normalize_operator_key(op: Any) -> str:
    text = str(op).strip()
    if not text:
        return ""

    if "::" in text:
        text = text.split("::", 1)[0].strip()

    text = " ".join(text.split())

    if not text:
        return ""

    if text.endswith("()"):
        return text[:-2].upper() + "()"

    if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", text):
        return text.upper()

    if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*(\s+[A-Za-z_][A-Za-z0-9_]*)+", text):
        return text.upper()

    return text


def _load_json_file(path: Path, *, allow_missing: bool = False, default: Any = None) -> Any:
    if not path.exists():
        if allow_missing:
            return default
        raise FileNotFoundError(
            f'Operator-similarity file not found: {path}\\nRun Stage 1 pre_rl / recycle_pre_rl first, or run python -m transformer_encoder.operator_embedding_extractor manually.'
        )

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        x = float(value)
        if math.isnan(x) or math.isinf(x):
            return default
        return x
    except Exception:
        return default


class OperatorSimilarityIndex:
    def __init__(
        self,
        similarity_path: Optional[Path] = None,
        prototypes_path: Optional[Path] = None,
        metadata_path: Optional[Path] = None,
        *,
        auto_load: bool = True,
        allow_missing: bool = False,
    ):
        self.similarity_path = Path(similarity_path or OPERATOR_SIMILARITY_PATH)
        self.prototypes_path = Path(prototypes_path or OPERATOR_PROTOTYPES_PATH)
        self.metadata_path = Path(metadata_path or OPERATOR_EMBEDDING_METADATA_PATH)

        self.allow_missing = allow_missing

        self._similarity: Dict[str, List[Dict[str, Any]]] = {}
        self._prototypes: Dict[str, Dict[str, Any]] = {}
        self._metadata: Dict[str, Any] = {}

        if auto_load:
            self.reload()

    def reload(self) -> None:
        raw_similarity = _load_json_file(
            self.similarity_path,
            allow_missing=self.allow_missing,
            default={},
        )
        raw_prototypes = _load_json_file(
            self.prototypes_path,
            allow_missing=True,
            default={},
        )
        raw_metadata = _load_json_file(
            self.metadata_path,
            allow_missing=True,
            default={},
        )

        self._similarity = self._normalize_similarity_payload(raw_similarity)
        self._prototypes = self._normalize_prototypes_payload(raw_prototypes)
        self._metadata = raw_metadata if isinstance(raw_metadata, dict) else {}

    @staticmethod
    def _normalize_similarity_payload(payload: Any) -> Dict[str, List[Dict[str, Any]]]:
        normalized: Dict[str, List[Dict[str, Any]]] = {}

        if not isinstance(payload, dict):
            return normalized

        for raw_op, raw_candidates in payload.items():
            op = normalize_operator_key(raw_op)
            if not op:
                continue

            candidates: List[Dict[str, Any]] = []

            if isinstance(raw_candidates, list):
                for idx, item in enumerate(raw_candidates):
                    if isinstance(item, dict):
                        cand_op = normalize_operator_key(item.get("op", ""))
                        if not cand_op:
                            continue

                        sim = _as_float(item.get("sim", item.get("similarity", 0.0)))
                        rank = int(item.get("rank", idx + 1))
                        hop = int(item.get("hop", 1))

                        candidates.append(
                            {
                                "op": cand_op,
                                "sim": sim,
                                "rank": rank,
                                "hop": hop,
                            }
                        )

                    elif isinstance(item, str):
                        cand_op = normalize_operator_key(item)
                        if cand_op:
                            candidates.append(
                                {
                                    "op": cand_op,
                                    "sim": 0.0,
                                    "rank": idx + 1,
                                    "hop": 1,
                                }
                            )

            candidates.sort(key=lambda x: (-float(x.get("sim", 0.0)), int(x.get("rank", 10**9))))
            normalized[op] = candidates

        return normalized

    @staticmethod
    def _normalize_prototypes_payload(payload: Any) -> Dict[str, Dict[str, Any]]:
        normalized: Dict[str, Dict[str, Any]] = {}

        if not isinstance(payload, dict):
            return normalized

        for raw_op, raw_data in payload.items():
            op = normalize_operator_key(raw_op)
            if not op:
                continue

            if isinstance(raw_data, dict):
                normalized[op] = raw_data
            else:
                normalized[op] = {"vector": raw_data}

        return normalized

    @property
    def is_loaded(self) -> bool:
        return bool(self._similarity)

    @property
    def operators(self) -> List[str]:
        return sorted(self._similarity.keys())

    @property
    def metadata(self) -> Dict[str, Any]:
        return dict(self._metadata)

    def has_operator(self, op: Any) -> bool:
        key = normalize_operator_key(op)
        return key in self._similarity

    def get_raw_table(self) -> Dict[str, List[Dict[str, Any]]]:
        return {
            op: [dict(item) for item in candidates]
            for op, candidates in self._similarity.items()
        }

    def get_prototype(self, op: Any) -> Optional[Dict[str, Any]]:
        key = normalize_operator_key(op)
        value = self._prototypes.get(key)
        if value is None:
            return None
        return dict(value)

    def get_operator_count(self, op: Any) -> int:
        key = normalize_operator_key(op)

        proto = self._prototypes.get(key)
        if isinstance(proto, dict) and "count" in proto:
            try:
                return int(proto["count"])
            except Exception:
                pass

        counts = self._metadata.get("operator_counts", {})
        if isinstance(counts, dict):
            try:
                return int(counts.get(key, 0))
            except Exception:
                return 0

        return 0

    def get_similar_ops(
        self,
        op: Any,
        *,
        top_k: Optional[int] = 8,
        min_sim: Optional[float] = None,
        include_self: bool = False,
        exclude_ops: Optional[Iterable[Any]] = None,
        max_hop: int = 1,
        allow_missing: bool = True,
    ) -> List[Dict[str, Any]]:
        key = normalize_operator_key(op)
        if not key:
            return []

        if key not in self._similarity:
            if allow_missing:
                return []
            raise KeyError(f'Operator not present in operator_similarity: {op!r} -> {key!r}')

        excluded: Set[str] = {normalize_operator_key(x) for x in (exclude_ops or [])}
        if not include_self:
            excluded.add(key)

        candidates: Dict[str, Dict[str, Any]] = {}

        def add_candidate(
            cand_op: str,
            sim: float,
            *,
            rank: int,
            hop: int,
            via: Optional[str] = None,
        ) -> None:
            cand_key = normalize_operator_key(cand_op)
            if not cand_key or cand_key in excluded:
                return

            if min_sim is not None and sim < float(min_sim):
                return

            old = candidates.get(cand_key)
            item = {
                "op": cand_key,
                "sim": float(sim),
                "rank": int(rank),
                "hop": int(hop),
            }
            if via is not None:
                item["via"] = via

            if old is None or float(item["sim"]) > float(old.get("sim", 0.0)):
                candidates[cand_key] = item

        direct = self._similarity.get(key, [])
        for idx, item in enumerate(direct):
            cand_op = normalize_operator_key(item.get("op", ""))
            sim = _as_float(item.get("sim", 0.0))
            rank = int(item.get("rank", idx + 1))
            add_candidate(cand_op, sim, rank=rank, hop=1)

        if max_hop >= 2:
            for mid_item in direct:
                mid_op = normalize_operator_key(mid_item.get("op", ""))
                if not mid_op or mid_op not in self._similarity:
                    continue

                sim_1 = _as_float(mid_item.get("sim", 0.0))
                if sim_1 <= 0:
                    continue

                for idx, second_item in enumerate(self._similarity.get(mid_op, [])):
                    cand_op = normalize_operator_key(second_item.get("op", ""))
                    sim_2 = _as_float(second_item.get("sim", 0.0))
                    if sim_2 <= 0:
                        continue

                    combined_sim = sim_1 * sim_2
                    add_candidate(
                        cand_op,
                        combined_sim,
                        rank=idx + 1,
                        hop=2,
                        via=mid_op,
                    )

        results = list(candidates.values())
        results.sort(
            key=lambda x: (
                int(x.get("hop", 1)),
                -float(x.get("sim", 0.0)),
                int(x.get("rank", 10**9)),
                str(x.get("op", "")),
            )
        )

        if top_k is not None:
            results = results[: int(top_k)]

        return results

    def get_similarity_score(
        self,
        op_a: Any,
        op_b: Any,
        *,
        default: float = 0.0,
        include_self_score: float = 1.0,
        max_hop: int = 2,
        symmetric: bool = True,
    ) -> float:
        key_a = normalize_operator_key(op_a)
        key_b = normalize_operator_key(op_b)

        if not key_a or not key_b:
            return float(default)

        if key_a == key_b:
            return float(include_self_score)

        def query_forward(src: str, dst: str) -> Optional[float]:
            if src not in self._similarity:
                return None

            candidates = self.get_similar_ops(
                src,
                top_k=None,
                min_sim=None,
                include_self=False,
                exclude_ops=None,
                max_hop=max_hop,
                allow_missing=True,
            )
            for item in candidates:
                if normalize_operator_key(item.get("op", "")) == dst:
                    return _as_float(item.get("sim", default), default)
            return None

        score = query_forward(key_a, key_b)
        if score is not None:
            return float(score)

        if symmetric:
            score = query_forward(key_b, key_a)
            if score is not None:
                return float(score)

        return float(default)

    def summary(self) -> Dict[str, Any]:
        return {
            "similarity_path": str(self.similarity_path),
            "prototypes_path": str(self.prototypes_path),
            "metadata_path": str(self.metadata_path),
            "num_operators": len(self._similarity),
            "operators": self.operators,
            "metadata": self._metadata,
        }


def load_operator_similarity_index(
    *,
    similarity_path: Optional[Path] = None,
    prototypes_path: Optional[Path] = None,
    metadata_path: Optional[Path] = None,
    allow_missing: bool = False,
) -> OperatorSimilarityIndex:
    return OperatorSimilarityIndex(
        similarity_path=similarity_path,
        prototypes_path=prototypes_path,
        metadata_path=metadata_path,
        auto_load=True,
        allow_missing=allow_missing,
    )


if __name__ == "__main__":
    index = load_operator_similarity_index(allow_missing=True)
    summary = index.summary()

    print("[operator_similarity] summary:")
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    for op in [">", "<", "=", "!=", "LIKE", "NOT LIKE", "AND", "OR", "ROUND()"]:
        candidates = index.get_similar_ops(op, top_k=8, min_sim=None, max_hop=1)
        if candidates:
            print(f"{op} -> {candidates}")
