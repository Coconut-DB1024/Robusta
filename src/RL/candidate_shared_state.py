from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np


def completion_retry_needed(result: Mapping[str, Any]) -> bool:
    """A failed sibling completion advances rotation, never target-failure evidence."""
    return (
        bool(result.get("safe_closure_used", False))
        and not bool(result.get("db_success", False))
        and bool(str(result.get("predicate", result.get("predicate_template", "")) or "").strip())
        and str(result.get("failure_kind", "") or "").lower() not in {
            "downstream_construct_failed", "db_connection_failed", "environment_error",
        }
        and not any(bool(result.get(key, False)) for key in (
            "environment_failure", "timed_out", "db_timed_out", "cancelled", "db_cancelled",
        ))
    )


def advance_completion_variant(record: Dict[str, Any]) -> int:
    # One bounded integer per edge; no accumulating predicates or Cartesian products.
    index = (int(record.get("completion_variant_index", 0) or 0) + 1) % (1 << 63)
    record["completion_variant_index"] = index
    return index


def normalize_failure_signature(predicate: Any) -> str:
    """Normalize one probe predicate for cross-environment failure de-duplication."""
    text = str(predicate or "").strip()
    if not text:
        return ""
    return re.sub(r"\s+", " ", text).lower()


@dataclass
class CandidateSharedState:
    """Small Manager-backed state shared by candidate-bitmap workers and callback."""

    observed: Any
    confirmed_failed: Any
    reservations: Any
    failure_signatures: Any
    incremental_graph_edges: Any
    state_events: Any
    metadata: Any
    lock: Any
    completion_variants: Any = None

    def _append_state_event_locked(self, event: Mapping[str, Any]) -> int:
        version = int(self.metadata.get("state_version", 0)) + 1
        payload = dict(event)
        payload["state_version"] = version
        self.state_events.append(payload)
        self.metadata["state_version"] = version
        return version

    def state_version(self) -> int:
        return int(self.metadata.get("state_version", 0))

    def snapshot_if_changed(self, last_version: int) -> Optional[Dict[str, Any]]:
        """Return completed-edge and incremental-graph state only when it changed."""
        current = self.state_version()
        if current == int(last_version):
            return None
        with self.lock:
            current = int(self.metadata.get("state_version", 0))
            if current == int(last_version):
                return None
            if int(last_version) < 0:
                return {
                    "state_version": current,
                    "full_snapshot": True,
                    "observed": dict(self.observed),
                    "confirmed_failed": dict(self.confirmed_failed),
                    "incremental_graph_edges": dict(self.incremental_graph_edges),
                    "events": [],
                }
            start = max(0, int(last_version))
            return {
                "state_version": current,
                "full_snapshot": False,
                "events": list(self.state_events[start:current]),
            }

    def is_observed(self, edge_id: str) -> bool:
        return str(edge_id or "") in self.observed

    def is_confirmed_failed(self, edge_id: str) -> bool:
        return str(edge_id or "") in self.confirmed_failed

    def mark_success(self, edge_id: str, edge_index: int) -> bool:
        """Atomically mark success and return True only for the first global success."""
        edge_key = str(edge_id or "")
        if not edge_key:
            return False
        with self.lock:
            first_success = edge_key not in self.observed
            changed = first_success or edge_key in self.confirmed_failed
            self.observed[edge_key] = int(edge_index)
            self.confirmed_failed.pop(edge_key, None)
            self.failure_signatures.pop(edge_key, None)
            if first_success:
                self.incremental_graph_edges[edge_key] = int(edge_index)
            if changed:
                self._append_state_event_locked({
                    "kind": "observed",
                    "edge_id": edge_key,
                    "edge_index": int(edge_index),
                    "remove_confirmed_failed": True,
                    "incremental_graph_edge": bool(first_success),
                })
            return bool(first_success)

    def get_failure_signatures(self, edge_id: str) -> List[str]:
        raw = self.failure_signatures.get(str(edge_id or ""), [])
        return [str(item) for item in list(raw or []) if str(item)]

    def get_completion_variant(self, edge_id: str) -> int:
        if self.completion_variants is None:
            return 0
        return int(self.completion_variants.get(str(edge_id or ""), 0))

    def record_failure_signature(
        self,
        edge_id: str,
        edge_index: int,
        predicate: Any,
        threshold: int,
    ) -> Dict[str, Any]:
        """Record one distinct target-DB failure and confirm failure at threshold N."""
        edge_key = str(edge_id or "")
        signature = normalize_failure_signature(predicate)
        limit = max(1, int(threshold))
        if not edge_key or not signature:
            return {"added": False, "confirmed": False, "count": 0}

        with self.lock:
            if edge_key in self.observed:
                return {"added": False, "confirmed": False, "count": 0}

            signatures = [
                str(item)
                for item in list(self.failure_signatures.get(edge_key, []) or [])
                if str(item)
            ]
            added = signature not in signatures
            if added:
                signatures.append(signature)
                self.failure_signatures[edge_key] = signatures

            confirmed_before = edge_key in self.confirmed_failed
            confirmed = len(signatures) >= limit
            if confirmed:
                self.confirmed_failed[edge_key] = int(edge_index)
            if confirmed and not confirmed_before:
                self._append_state_event_locked({
                    "kind": "confirmed_failed",
                    "edge_id": edge_key,
                    "edge_index": int(edge_index),
                })

            return {
                "added": bool(added),
                "confirmed": bool(confirmed),
                "count": int(len(signatures)),
            }

    def record_result(self, result: Mapping[str, Any], threshold: int) -> Dict[str, Any]:
        """Apply one callback result to memory without replacing the disk callback."""
        edge_id = str(result.get("edge_id", "") or "")
        try:
            edge_index = int(result.get("edge_index", -1))
        except Exception:
            edge_index = -1
        if not edge_id or edge_index < 0:
            return {"applied": False, "reason": "invalid_edge"}

        db_success = bool(result.get("db_success", False))
        parse_success = bool(result.get("parse_success", False))
        target_hit = bool(result.get("target_edge_hit", False))
        if db_success and parse_success and target_hit:
            first_success = self.mark_success(edge_id, edge_index)
            return {"applied": True, "status": "observed", "first_success": first_success}

        failure_kind = str(result.get("failure_kind", "") or "").lower()
        safe_closure_used = bool(result.get("safe_closure_used", False))
        if completion_retry_needed(result) and self.completion_variants is not None:
            with self.lock:
                record = {"completion_variant_index": self.get_completion_variant(edge_id)}
                index = advance_completion_variant(record)
                self.completion_variants[edge_id] = index
            # Do not append state_events: rotation is fetched only for the selected edge.
            return {"applied": True, "status": "completion_retry", "completion_variant_index": index}
        if (
            (not db_success)
            and parse_success
            and target_hit
            and not safe_closure_used
            and failure_kind not in {
                "context_inconclusive",
                "downstream_construct_failed",
                "db_connection_failed",
                "environment_error",
            }
        ):
            recorded = self.record_failure_signature(
                edge_id,
                edge_index,
                result.get("predicate", ""),
                threshold,
            )
            return {"applied": True, "status": "target_db_failure", **recorded}

        return {"applied": False, "reason": failure_kind or "not_target_evidence"}

    def try_reserve_edge(
        self,
        edge_id: str,
        edge_index: int,
        *,
        owner: str,
        timeout_seconds: float,
    ) -> Dict[str, Any]:
        """Atomically reserve one PPO-selected edge without scanning the whole map."""
        edge_key = str(edge_id or "")
        if not edge_key:
            return {"token": "", "conflict_count": 0, "expired_count": 0}

        timeout = max(1.0, float(timeout_seconds))
        now = time.time()
        owner_text = str(owner or "worker")
        expired_count = 0

        with self.lock:
            metadata = dict(self.reservations.get(edge_key, {}) or {})
            if metadata:
                try:
                    claimed_at = float(metadata.get("claim_time", 0.0))
                except Exception:
                    claimed_at = 0.0
                if claimed_at > 0.0 and now - claimed_at < timeout:
                    return {"token": "", "conflict_count": 1, "expired_count": 0}
                self.reservations.pop(edge_key, None)
                expired_count = 1

            token = f"{owner_text}:{time.time_ns()}"
            self.reservations[edge_key] = {
                "token": token,
                "owner": owner_text,
                "edge_index": int(edge_index),
                "claim_time": now,
            }

        return {
            "token": token,
            "conflict_count": 0,
            "expired_count": int(expired_count),
        }

    def release_reservations(self, claims: Mapping[str, str]) -> int:
        """Release only reservations whose ownership token still matches."""
        released = 0
        with self.lock:
            for raw_edge_id, raw_token in dict(claims or {}).items():
                edge_id = str(raw_edge_id or "")
                token = str(raw_token or "")
                metadata = dict(self.reservations.get(edge_id, {}) or {})
                if edge_id and token and str(metadata.get("token", "")) == token:
                    self.reservations.pop(edge_id, None)
                    released += 1
        return int(released)


def create_candidate_shared_state(
    manager: Any,
    edge_records: Sequence[Mapping[str, Any]],
    bitmaps: Mapping[str, np.ndarray],
) -> CandidateSharedState:
    """Create and seed the shared state from the candidate files loaded by the main process."""
    records = [dict(record) for record in edge_records if isinstance(record, Mapping)]
    n = len(records)
    observed_bitmap = np.asarray(bitmaps.get("observed_bitmap", np.zeros(n)), dtype=np.uint8)
    failed_bitmap = np.asarray(bitmaps.get("failed_bitmap", np.zeros(n)), dtype=np.uint8)

    observed: Dict[str, int] = {}
    confirmed_failed: Dict[str, int] = {}
    signatures: Dict[str, List[str]] = {}
    completion_variants: Dict[str, int] = {}
    for record in records:
        try:
            edge_index = int(record.get("edge_index", -1))
        except Exception:
            continue
        edge_id = str(record.get("edge_id", "") or "")
        if not edge_id or edge_index < 0 or edge_index >= n:
            continue
        completion_index = int(record.get("completion_variant_index", 0) or 0)
        if completion_index:
            completion_variants[edge_id] = completion_index
        if int(observed_bitmap[edge_index]) > 0:
            observed[edge_id] = edge_index
        elif int(failed_bitmap[edge_index]) > 0:
            confirmed_failed[edge_id] = edge_index
        raw_signatures = record.get("target_db_failure_signatures", [])
        normalized = [
            normalize_failure_signature(item)
            for item in list(raw_signatures or [])
            if normalize_failure_signature(item)
        ]
        if normalized:
            signatures[edge_id] = list(dict.fromkeys(normalized))

    return CandidateSharedState(
        observed=manager.dict(observed),
        confirmed_failed=manager.dict(confirmed_failed),
        reservations=manager.dict(),
        failure_signatures=manager.dict(signatures),
        incremental_graph_edges=manager.dict(),
        state_events=manager.list(),
        metadata=manager.dict({"state_version": 0}),
        lock=manager.RLock(),
        completion_variants=manager.dict(completion_variants),
    )


def create_candidate_shared_state_from_files(
    manager: Any,
    edge_index_path: Path,
    bitmap_npz_path: Path,
) -> CandidateSharedState:
    """Load candidate artifacts once in the main process and seed Manager proxies."""
    edge_path = Path(edge_index_path)
    bitmap_path = Path(bitmap_npz_path)
    with edge_path.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    edge_records = payload.get("edges", []) if isinstance(payload, dict) else []
    if not isinstance(edge_records, list):
        raise ValueError(f"candidate edge index 'edges' must be a list: {edge_path}")
    with np.load(bitmap_path) as loaded:
        bitmaps = {name: np.asarray(loaded[name], dtype=np.uint8) for name in loaded.files}
    return create_candidate_shared_state(manager, edge_records, bitmaps)
