import os  
import json  
import random  
import re
import time  
import gzip  
from bisect import bisect_right
from contextlib import contextmanager

from pkg.stage3_runtime import (
    Stage3DeadlineReached,
    check_deadline,
    decode_counts_in_place,
    encode_count,
    estimate_downstream_reserve,
    format_large_int,
    remaining_seconds,
    write_status_json,
)
from RL.joint_signature_evidence import (
    classify_oracle_error, classify_sqlserver_error, SQLSERVER_FAILURE_WEIGHTS,
)

try:
    from RL.config import DB_CONFIG, STAGE3_CONFIG
except Exception:
    DB_CONFIG = None
    STAGE3_CONFIG = None

try:
    from RL.operator_similarity import (
        load_operator_similarity_index,
        normalize_operator_key,
    )
except Exception:
    load_operator_similarity_index = None
    normalize_operator_key = lambda value: str(value or "").strip().upper()


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PREDICATE_CONTRIBUTION_PATH = os.path.join(
    PROJECT_ROOT,
    "output",
    "graph_runtime_stats",
    "predicate_contribution.json",
)


_STAGE3_SLOT_ORDER_CACHE = {}


_ORACLE_DB_TYPES = frozenset({"oracle", "oracledb", "oracle_ai26"})

_ORACLE_SAFE_ARGUMENT_LITERALS = {
    "SQRT()": {"0": "4"},
    "EXP()": {"0": "1"},
    "LN()": {"0": "1"},
    "LOG()": {"0": "2"},
    "POWER()": {"0": "2", "1": "3"},
    "COSH()": {"0": "1"},
    "SINH()": {"0": "1"},
    "TANH()": {"0": "1"},
    "TO_NUMBER()": {"0": "'1'"},
    "NLSSORT()": {"1": "'NLS_SORT=BINARY'"},
    "FROM_TZ()": {"1": "'+00:00'"},
}


_SAMPLING_TIMING_SECONDS = (
    "sampler_setup_seconds",
    "counted_dag_load_seconds",
    "counted_dag_decode_seconds",
    "successor_ordering_seconds",
    "decode_cache_preparation_seconds",
    "graph_preparation_seconds",
    "predicate_traversal_render_seconds",
    "predicate_output_write_seconds",
    "sampling_overhead_seconds",
)
_SAMPLING_TIMING_COUNTERS = (
    "predicate_traversal_attempts",
    "predicate_output_write_calls",
)


def _new_sampling_timings():
    return {
        **{name: 0.0 for name in _SAMPLING_TIMING_SECONDS},
        **{name: 0 for name in _SAMPLING_TIMING_COUNTERS},
    }


def _merge_sampling_timings(target, source):
    if not isinstance(source, dict):
        return
    for name in _SAMPLING_TIMING_SECONDS:
        target[name] = target.get(name, 0.0) + max(
            0.0,
            float(source.get(name, 0.0) or 0.0),
        )
    for name in _SAMPLING_TIMING_COUNTERS:
        target[name] = target.get(name, 0) + max(
            0,
            int(source.get(name, 0) or 0),
        )


def _print_sampling_progress(completed_roots, total_roots):
    fraction = completed_roots / total_roots if total_roots > 0 else 1.0
    filled = min(40, int(fraction * 40))
    bar = "#" * filled + "-" * (40 - filled)

    percent = min(fraction * 100, 99.99) if completed_roots < total_roots else 100.0
    print(f"  [Sampling] [{bar}] {percent:.2f}%", flush=True)


@contextmanager
def _open_predicate_writer(
    output_path,
    *,
    compressed,
    compression_level,
    buffer_bytes,
):
    with open(output_path, "wb", buffering=buffer_bytes) as raw_file:
        if not compressed:
            yield raw_file
            return

        with gzip.GzipFile(
            filename="",
            mode="wb",
            fileobj=raw_file,
            compresslevel=compression_level,
            mtime=0,
        ) as gzip_file:
            yield gzip_file


class PredicateSampler:
    def __init__(
        self,
        step_size=1,
        max_sqls_per_file=None,
        runtime_signature_feedback=None,
        enumerate_all=False,
        localized_random=False,
    ):
        self.step_size = step_size

        self.enumerate_all = bool(enumerate_all)
        self.localized_random = bool(localized_random)
        if self.enumerate_all and self.localized_random:
            raise ValueError(
                "Enumeration and localized-random sampling cannot be enabled together."
            )
        self.local_random_seed = int(
            getattr(STAGE3_CONFIG, "stage3_random_seed", 42)
        )
        self.local_random_root_pool_size = max(
            1,
            int(getattr(STAGE3_CONFIG, "stage3_random_root_pool_size", 16)),
        )
        self.local_random_root_candidate_pool_size = max(
            self.local_random_root_pool_size,
            int(
                getattr(
                    STAGE3_CONFIG,
                    "stage3_random_root_candidate_pool_size",
                    64,
                )
            ),
        )
        self.local_random_window_ratio = min(
            1.0,
            max(
                0.0,
                float(
                    getattr(
                        STAGE3_CONFIG,
                        "stage3_random_local_window_ratio",
                        0.001,
                    )
                ),
            ),
        )
        self.local_random_window_min_size = max(
            1,
            int(
                getattr(
                    STAGE3_CONFIG,
                    "stage3_random_local_window_min_size",
                    10000000,
                )
            ),
        )
        self.local_random_root_block_size = max(
            1,
            int(
                getattr(
                    STAGE3_CONFIG,
                    "stage3_random_root_block_size",
                    4096,
                )
            ),
        )

        self.local_random_rng = random.Random(self.local_random_seed)
        self.db_type = str(
            getattr(DB_CONFIG, "db_type", "mysql") or "mysql"
        ).lower().strip()
        self.rule_config = self._load_rule_config()
        self.render_config = (
            self.rule_config.get("sql_render", {})
            if isinstance(self.rule_config, dict)
            else {}
        )
        self.operator_styles = self._load_operator_styles(self.render_config)
        literal_profiles = self.render_config.get("literal_profiles", {}) if isinstance(self.render_config, dict) else {}
        self.allow_hex_int_literals = bool(literal_profiles.get("allow_hex_int_literals", True))
        candidate_bitmap = (
            self.rule_config.get("candidate_bitmap", {})
            if isinstance(self.rule_config, dict)
            else {}
        )
        self.safe_argument_literals = candidate_bitmap.get(
            "non_target_slot_fixed_sql",
            {},
        )
        self.contextual_null_cast_enabled = bool(
            getattr(STAGE3_CONFIG, "stage3_contextual_null_cast_enabled", True)
        )

        self.exact_column_type_guidance_enabled = bool(
            getattr(STAGE3_CONFIG, "stage3_exact_column_type_guidance_enabled", True)
        )
        self.safe_value_guidance_enabled = bool(
            getattr(STAGE3_CONFIG, "stage3_safe_value_guidance_enabled", True)
        )
        safe_value_ratio_key = (
            "stage3_oracle_safe_value_exploration_ratio"
            if self.db_type in _ORACLE_DB_TYPES
            else "stage3_safe_value_exploration_ratio"
        )
        self.safe_value_exploration_ratio = min(
            1.0,
            max(
                0.0,
                float(
                    getattr(
                        STAGE3_CONFIG,
                        safe_value_ratio_key,
                        getattr(
                            STAGE3_CONFIG,
                            "stage3_safe_value_exploration_ratio",
                            0.20,
                        ),
                    )
                ),
            ),
        )
        self.similarity_order_enabled = bool(
            getattr(STAGE3_CONFIG, "stage3_similarity_order_enabled", True)
        )
        self.success_guided_sampling_enabled = bool(
            getattr(STAGE3_CONFIG, "stage3_success_guided_sampling_enabled", True)
        )

        self.candidate_pool_multiplier = max(
            1,
            int(getattr(STAGE3_CONFIG, "stage3_candidate_pool_multiplier", 2)),
        )

        self.guided_candidate_ratio = min(
            1.0,
            max(0.0, float(getattr(STAGE3_CONFIG, "stage3_guided_candidate_ratio", 0.70))),
        )
        sampling_step = float(
            getattr(STAGE3_CONFIG, "stage3_sampling_step", 64.0)
        )
        if not 0.0 < sampling_step < float("inf"):
            raise ValueError("stage3_sampling_step must be a finite number greater than 0")
        try:
            self.sampling_density = 4.0 * (64.0 / sampling_step) ** 1.5
        except OverflowError:
            raise ValueError("stage3_sampling_step produces an unsupported sampling density") from None
        if not 0.0 < self.sampling_density < float("inf"):
            raise ValueError("stage3_sampling_step produces an unsupported sampling density")
        self.evidence_weight_strength = min(
            0.50,
            max(
                0.0,
                float(
                    getattr(
                        STAGE3_CONFIG,
                        "stage3_evidence_weight_strength",
                        0.25,
                    )
                ),
            ),
        )
        self.evidence_min_observations = max(
            1,
            int(
                getattr(
                    STAGE3_CONFIG,
                    "stage3_evidence_min_observations",
                    4,
                )
            ),
        )

        unknown_exploration_ratio_key = (
            "stage3_oracle_implicit_conversion_exploration_ratio"
            if self.db_type in _ORACLE_DB_TYPES
            else "stage3_type_projection_unknown_exploration_ratio"
        )
        if self.db_type == "sqlserver":
            unknown_exploration_ratio_key = "stage3_sqlserver_implicit_conversion_exploration_ratio"
        self.unknown_exploration_ratio = min(
            1.0,
            max(
                0.0,
                float(
                    getattr(
                        STAGE3_CONFIG,
                        unknown_exploration_ratio_key,
                        0.05 if self.db_type == "sqlserver" else getattr(
                            STAGE3_CONFIG,
                            "stage3_type_projection_unknown_exploration_ratio",
                            0.08,
                        ),
                    )
                ),
            ),
        )
        self.runtime_signature_feedback_enabled = bool(
            getattr(
                STAGE3_CONFIG,
                "stage3_runtime_signature_feedback_enabled",
                True,
            )
        )
        self.runtime_signature_feedback = dict(runtime_signature_feedback or {})
        self.runtime_feedback_min_observations = max(
            1,
            int(
                getattr(
                    STAGE3_CONFIG,
                    "stage3_runtime_signature_feedback_min_observations",
                    4,
                )
            ),
        )
        self.runtime_feedback_strength = min(
            0.99,
            max(
                0.0,
                float(
                    getattr(
                        STAGE3_CONFIG,
                        "stage3_runtime_signature_feedback_strength",
                        0.90,
                    )
                ),
            ),
        )
        self.runtime_feedback_min_factor = min(
            1.0,
            max(
                0.01,
                float(
                    getattr(
                        STAGE3_CONFIG,
                        "stage3_runtime_signature_feedback_min_factor",
                        0.10,
                    )
                ),
            ),
        )
        self.runtime_feedback_tiered_enabled = bool(
            getattr(
                STAGE3_CONFIG,
                "stage3_runtime_signature_feedback_tiered_enabled",
                True,
            )
        )
        self.runtime_feedback_high_rate = min(
            1.0,
            max(
                0.0,
                float(
                    getattr(
                        STAGE3_CONFIG,
                        "stage3_runtime_signature_feedback_high_rate",
                        0.95,
                    )
                ),
            ),
        )
        self.runtime_feedback_medium_rate = min(
            self.runtime_feedback_high_rate,
            max(
                0.0,
                float(
                    getattr(
                        STAGE3_CONFIG,
                        "stage3_runtime_signature_feedback_medium_rate",
                        0.90,
                    )
                ),
            ),
        )
        self.runtime_feedback_low_rate = min(
            self.runtime_feedback_medium_rate,
            max(
                0.0,
                float(
                    getattr(
                        STAGE3_CONFIG,
                        "stage3_runtime_signature_feedback_low_rate",
                        0.80,
                    )
                ),
            ),
        )
        self.runtime_feedback_high_factor = max(
            0.0,
            float(
                getattr(
                    STAGE3_CONFIG,
                    "stage3_runtime_signature_feedback_high_factor",
                    1.80,
                )
            ),
        )
        self.runtime_feedback_medium_factor = max(
            0.0,
            float(
                getattr(
                    STAGE3_CONFIG,
                    "stage3_runtime_signature_feedback_medium_factor",
                    0.50,
                )
            ),
        )
        self.runtime_feedback_low_factor = max(
            0.0,
            float(
                getattr(
                    STAGE3_CONFIG,
                    "stage3_runtime_signature_feedback_low_factor",
                    0.10,
                )
            ),
        )
        self.runtime_feedback_very_low_factor = max(
            0.0,
            float(
                getattr(
                    STAGE3_CONFIG,
                    "stage3_runtime_signature_feedback_very_low_factor",
                    0.01,
                )
            ),
        )

        normalized_db_type = str(
            getattr(DB_CONFIG, "db_type", "") or ""
        ).strip().lower()
        self.component_feedback_enabled = bool(
            normalized_db_type == "mysql"
            and
            not self.enumerate_all
            and not self.localized_random
            and getattr(
                STAGE3_CONFIG,
                "stage3_component_signature_feedback_enabled",
                True,
            )
        )
        self.component_feedback_trace_stride = max(
            1,
            int(
                getattr(
                    STAGE3_CONFIG,
                    "stage3_component_feedback_trace_stride",
                    16,
                )
            ),
        )
        self.component_feedback_block_size = max(
            1,
            int(
                getattr(
                    STAGE3_CONFIG,
                    "stage3_component_feedback_block_size",
                    256,
                )
            ),
        )
        self.component_feedback_max_keys_per_block = max(
            1,
            int(
                getattr(
                    STAGE3_CONFIG,
                    "stage3_component_feedback_max_keys_per_block",
                    32,
                )
            ),
        )
        self.operator_similarity_index = self._load_operator_similarity_index()
        self._similarity_score_cache = {}
        self._similarity_neighbor_cache = {}

        self._similarity_token_order_cache = {}

        self.success_edge_pair_counts = {}
        self.last_sample_metadata = {
            "round_complete": True,
            "stop_root_id": None,
            "stop_index": None,
            "stop_step": None,
            "signature_spans": [],
            "signature_catalog": {},
        }
        self._decode_cache_nodes = None
        self._decode_cache = {}

        self.last_sampling_timings = _new_sampling_timings()

    def _load_rule_config(self):
        rules_db_type = "oracle" if self.db_type in _ORACLE_DB_TYPES else self.db_type
        config_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "configuration",
            f"{rules_db_type}_rules.json",
        )
        if not os.path.exists(config_path):
            return {}
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
            return config if isinstance(config, dict) else {}
        except Exception as exc:
            print(f"[Sampler] Warning: failed to load DBMS rules: {config_path} | {exc}")
            return {}

    def _load_operator_styles(self, render_config):
        templates = render_config.get("operator_templates", {}) if isinstance(render_config, dict) else {}
        return {
            str(token).upper(): str(template.get("style", "")).strip().lower()
            for token, template in templates.items()
            if isinstance(template, dict)
        }

    def _load_operator_similarity_index(self):
        if not self.similarity_order_enabled or load_operator_similarity_index is None:
            return None
        try:
            index = load_operator_similarity_index(allow_missing=True)
            if not bool(getattr(index, "is_loaded", False)):
                return None
            return index
        except Exception as exc:
            print(f"[Sampler] Warning: failed to load operator similarity: {exc}")
            return None

    @staticmethod
    def _parse_edge_key(edge_key):
        text = str(edge_key or "")
        if "::slot" not in text or "->" not in text:
            return None
        source_id, tail = text.rsplit("::slot", 1)
        slot_text, target_id = tail.split("->", 1)
        try:
            return source_id, int(slot_text), target_id
        except ValueError:
            return None

    @staticmethod
    def _pair_key(edge_a, edge_b):
        return tuple(sorted((str(edge_a), str(edge_b))))

    def _load_success_edge_pair_counts(self):
        if not self.success_guided_sampling_enabled:
            return {}
        if not os.path.exists(PREDICATE_CONTRIBUTION_PATH):
            return {}

        try:
            with open(PREDICATE_CONTRIBUTION_PATH, "r", encoding="utf-8") as f:
                payload = json.load(f)
        except Exception as exc:
            print(f"[Sampler] Warning: failed to load predicate contribution: {exc}")
            return {}

        pair_counts = {}
        records = payload.values() if isinstance(payload, dict) else []
        for record in records:
            if not isinstance(record, dict):
                continue
            if int(record.get("success_count", 0) or 0) <= 0:
                continue
            if not bool(record.get("parse_success", False)):
                continue

            grouped_edges = {}
            for edge_key in sorted(set(record.get("edges", []) or [])):
                parsed = self._parse_edge_key(edge_key)
                if parsed is None:
                    continue
                source_id, slot_id, _ = parsed
                grouped_edges.setdefault(source_id, []).append((slot_id, edge_key))

            weight = max(1, int(record.get("success_count", 1) or 1))
            for sibling_edges in grouped_edges.values():
                for i in range(len(sibling_edges)):
                    slot_i, edge_i = sibling_edges[i]
                    for j in range(i + 1, len(sibling_edges)):
                        slot_j, edge_j = sibling_edges[j]
                        if slot_i == slot_j:
                            continue
                        key = self._pair_key(edge_i, edge_j)
                        pair_counts[key] = pair_counts.get(key, 0) + weight

        return pair_counts

    @staticmethod
    def _original_node_id(node_id, nodes):
        node = nodes.get(node_id, {})
        base_node_id = str(node.get("base_node_id", "") or "").strip()
        if base_node_id:
            return base_node_id
        original_id = str(node.get("original_id", "") or "").strip()
        if original_id:
            return original_id

        parts = str(node_id).split("::")
        if parts and parts[-1].startswith("L") and parts[-1][1:].isdigit():
            return "::".join(parts[:-1])
        return str(node_id)

    @classmethod
    def _operator_token(cls, node_id, nodes):
        return cls._original_node_id(node_id, nodes).split("::", 1)[0]

    def _similarity_score_for_tokens(self, left_token, right_token):
        if self.operator_similarity_index is None:
            return 0.0
        left_key = normalize_operator_key(left_token)
        right_key = normalize_operator_key(right_token)
        cache_key = tuple(sorted((left_key, right_key)))
        score_cache = getattr(self, "_similarity_score_cache", None)
        if score_cache is None:
            score_cache = {}
            self._similarity_score_cache = score_cache
        neighbor_cache = getattr(self, "_similarity_neighbor_cache", None)
        if neighbor_cache is None:
            neighbor_cache = {}
            self._similarity_neighbor_cache = neighbor_cache
        if cache_key in score_cache:
            return score_cache[cache_key]

        if left_key == right_key:
            score = 1.0
        else:
            direct_score = getattr(self.operator_similarity_index, "get_similarity_score", None)
            if callable(direct_score):
                try:
                    score = float(direct_score(
                        left_token,
                        right_token,
                        default=0.0,
                        include_self_score=1.0,
                        max_hop=2,
                        symmetric=True,
                    ))
                except Exception:
                    score = None
            else:
                score = None

            if score is not None:
                score_cache[cache_key] = score
                return score

            def neighbor_scores(token, token_key):
                cached = neighbor_cache.get(token_key)
                if cached is not None:
                    return cached
                try:
                    candidates = self.operator_similarity_index.get_similar_ops(
                        token,
                        top_k=None,
                        min_sim=None,
                        include_self=False,
                        exclude_ops=None,
                        max_hop=2,
                        allow_missing=True,
                    )
                    cached = {
                        normalize_operator_key(item.get("op", "")): float(
                            item.get("sim", 0.0)
                        )
                        for item in candidates
                    }
                except Exception:
                    cached = {}
                neighbor_cache[token_key] = cached
                return cached

            left_scores = neighbor_scores(left_token, left_key)
            if right_key in left_scores:
                score = left_scores[right_key]
            else:
                score = neighbor_scores(right_token, right_key).get(left_key, 0.0)
        score_cache[cache_key] = score
        return score

    def _similarity_score(self, left_id, right_id, nodes):
        if self.operator_similarity_index is None:
            return 0.0
        return self._similarity_score_for_tokens(
            self._operator_token(left_id, nodes),
            self._operator_token(right_id, nodes),
        )

    def _ordered_similarity_tokens(self, token_keys):
        raw_tokens = tuple(sorted({str(token) for token in token_keys if str(token)}))
        order_cache = getattr(self, "_similarity_token_order_cache", None)
        if order_cache is None:
            order_cache = {}
            self._similarity_token_order_cache = order_cache
        cached = order_cache.get(raw_tokens)
        if cached is not None:
            return cached

        rankings = {}
        for source_token in raw_tokens:
            rankings[source_token] = tuple(sorted(
                raw_tokens,
                key=lambda candidate_token: (
                    -self._similarity_score_for_tokens(source_token, candidate_token),
                    candidate_token,
                ),
            ))
        order_cache[raw_tokens] = rankings
        return rankings

    def _order_slot_by_similarity(self, slot, nodes, original_ids=None):
        if original_ids is None:
            original_ids = {
                node_id: self._original_node_id(node_id, nodes)
                for node_id in slot
            }

        def token_for(node_id):
            original_id = original_ids.get(node_id)
            if original_id is None:
                original_id = self._original_node_id(node_id, nodes)
            return str(original_id).split("::", 1)[0]

        remaining = sorted(
            list(slot),
            key=lambda node_id: (token_for(node_id), str(node_id)),
        )
        if len(remaining) < 2 or self.operator_similarity_index is None:
            return remaining

        groups = {}
        for node_id in remaining:
            token = token_for(node_id)
            groups.setdefault(token, []).append(node_id)

        token_rankings = self._ordered_similarity_tokens(groups.keys())

        group_positions = {token: 0 for token in groups}

        def take_from_group(token):
            position = group_positions[token]
            group_positions[token] = position + 1
            return groups[token][position]

        ordered = []
        first_token = token_for(remaining[0])
        ordered.append(take_from_group(first_token))

        while len(ordered) < len(remaining):
            previous_token = token_for(ordered[-1])
            next_token = next(
                token
                for token in token_rankings.get(previous_token, tuple(sorted(groups)))
                if group_positions.get(token, 0) < len(groups.get(token, ()))
            )
            ordered.append(take_from_group(next_token))
        return ordered

    def _order_successors_in_place(self, nodes, deadline=None):
        if not self.similarity_order_enabled:
            return
        slot_order_cache = _STAGE3_SLOT_ORDER_CACHE
        for node in nodes.values():
            check_deadline(deadline, "successor similarity ordering")
            slots = node.get("slots", []) or []
            ordered_slots = []
            for slot in slots:
                original_ids = {
                    child_id: self._original_node_id(child_id, nodes)
                    for child_id in slot
                }
                cache_key = tuple(sorted(original_ids.values()))
                cached_order = slot_order_cache.get(cache_key)
                if cached_order is None:
                    ordered_slot = self._order_slot_by_similarity(
                        slot,
                        nodes,
                        original_ids=original_ids,
                    )
                    slot_order_cache[cache_key] = [original_ids[item] for item in ordered_slot]
                else:
                    rank = {original_id: index for index, original_id in enumerate(cached_order)}
                    ordered_slot = sorted(
                        slot,
                        key=lambda child_id: (
                            rank.get(original_ids[child_id], len(rank)),
                            str(child_id),
                        ),
                    )
                ordered_slots.append(ordered_slot)
            node["slots"] = ordered_slots

    def _score_candidate_edges(self, edge_keys):
        if not self.success_edge_pair_counts:
            return 0

        score = 0
        grouped_edges = {}
        for edge_key in sorted(set(edge_keys or [])):
            parsed = self._parse_edge_key(edge_key)
            if parsed is None:
                continue
            source_id, slot_id, _ = parsed
            grouped_edges.setdefault(source_id, []).append((slot_id, edge_key))

        for sibling_edges in grouped_edges.values():
            for i in range(len(sibling_edges)):
                slot_i, edge_i = sibling_edges[i]
                for j in range(i + 1, len(sibling_edges)):
                    slot_j, edge_j = sibling_edges[j]
                    if slot_i == slot_j:
                        continue
                    score += self.success_edge_pair_counts.get(
                        self._pair_key(edge_i, edge_j),
                        0,
                    )
        return score

    def _select_guided_candidates(self, candidates, target_count):
        target_count = min(max(0, int(target_count)), len(candidates))
        if target_count >= len(candidates):
            return sorted(candidates, key=lambda item: item["index"])

        guided_limit = int(round(target_count * self.guided_candidate_ratio))
        scored = sorted(
            (item for item in candidates if item.get("score", 0) > 0),
            key=lambda item: (-item["score"], item["index"]),
        )
        guided = scored[:guided_limit]
        guided_ids = {id(item) for item in guided}
        remaining = [item for item in candidates if id(item) not in guided_ids]
        random_count = target_count - len(guided)
        random_part = random.sample(remaining, random_count) if random_count > 0 else []
        return sorted(guided + random_part, key=lambda item: item["index"])

    @staticmethod
    def _runtime_signature_identity(node_id, node):
        node_data = node if isinstance(node, dict) else {}
        signature = node_data.get("stage3_type_signature", {}) or {}
        original_id = str(node_data.get("original_id", "") or "").strip()
        if not original_id:
            original_id = str(node_id or "")
            parts = original_id.split("::")
            if parts and parts[-1].startswith("L") and parts[-1][1:].isdigit():
                original_id = "::".join(parts[:-1])
        detail = {
            "signature_key": original_id,
            "outer_node_id": str(
                node_data.get("base_node_id", "") or original_id
            ),
            "token": str(node_data.get("token", "") or ""),
            "source": str(signature.get("source", "") or ""),
            "compatibility_class": str(
                signature.get("compatibility_class", "") or ""
            ),
            "arg_types": list(signature.get("arg_types", []) or []),
            "accepted_arg_types": list(
                signature.get("accepted_arg_types", []) or []
            ),
        }
        return original_id, detail

    def _new_component_feedback_tracker(
        self,
        nodes,
        signature_spans,
        signature_catalog,
    ):
        if not getattr(self, "component_feedback_enabled", False):
            return None

        details = {}
        for node_id, node in nodes.items():
            if not isinstance(node, dict) or bool(node.get("is_terminal")):
                continue
            if not (node.get("stage3_type_signature") or {}):
                continue
            signature_key, detail = self._runtime_signature_identity(node_id, node)
            details.setdefault(signature_key, detail)

        return {
            "signature_spans": signature_spans,
            "signature_catalog": signature_catalog,
            "details": details,
            "block_start": 1,
            "key_counts": {},
        }

    def _should_trace_component_feedback(self, tracker, next_case_idx):
        return bool(
            tracker
            and (int(next_case_idx) - 1)
            % self.component_feedback_trace_stride
            == 0
        )

    def _flush_component_feedback_block(self, tracker, end_case_idx):
        if not tracker or int(end_case_idx) < int(tracker["block_start"]):
            return
        key_counts = tracker["key_counts"]
        if key_counts:
            component_keys = [
                key
                for key, _count in sorted(
                    key_counts.items(),
                    key=lambda item: (-item[1], item[0]),
                )[:self.component_feedback_max_keys_per_block]
            ]
            tracker["signature_spans"].append({
                "start_case_idx": int(tracker["block_start"]),
                "end_case_idx": int(end_case_idx),
                "signature_keys": component_keys,
                "feedback_scope": "component_block",
            })
        tracker["block_start"] = int(end_case_idx) + 1
        tracker["key_counts"] = {}

    def _record_component_feedback_trace(
        self,
        tracker,
        case_idx,
        edge_keys,
    ):
        if not tracker:
            return
        traced_keys = set()
        for edge_key in edge_keys or []:
            parsed = self._parse_edge_key(edge_key)
            if parsed is None:
                continue
            source_id, _slot_id, target_id = parsed
            for signature_key in (source_id, target_id):
                if signature_key in tracker["details"]:
                    traced_keys.add(signature_key)
        for signature_key in traced_keys:
            tracker["key_counts"][signature_key] = (
                tracker["key_counts"].get(signature_key, 0) + 1
            )
            tracker["signature_catalog"].setdefault(
                signature_key,
                tracker["details"][signature_key],
            )
        if int(case_idx) % self.component_feedback_block_size == 0:
            self._flush_component_feedback_block(tracker, case_idx)

    @staticmethod
    def _oracle_error_kind(error_info):
        info = error_info if isinstance(error_info, dict) else {}
        category = classify_oracle_error(
            info.get("error_code"),
            info.get("error"),
            timed_out=bool(info.get("is_timeout", False)),
        )

        if category in {"execution_failure_count", "environment_failure_count"}:
            return "other_failure_count"
        return category

    def _runtime_feedback_factor(self, node_id, node):
        if not getattr(self, "runtime_signature_feedback_enabled", False):
            return 1.0
        signature_key, _detail = self._runtime_signature_identity(node_id, node)
        item = getattr(self, "runtime_signature_feedback", {}).get(
            signature_key,
            {},
        )
        if not isinstance(item, dict):
            return 1.0
        successes = max(0, int(item.get("success_count", 0) or 0))
        failures = max(0, int(item.get("failure_count", 0) or 0))
        normalized_db_type = str(getattr(DB_CONFIG, "db_type", "") or "").strip().lower()
        if normalized_db_type == "sqlserver":
            counts = {key: max(0, int(item.get(key, 0) or 0)) for key in SQLSERVER_FAILURE_WEIGHTS}
            counts["execution_failure_count"] += max(0, int(item.get("other_failure_count", 0) or 0))
            classified = sum(counts.values())
            raw = item.get("error_info")
            if raw is None and any(key in item for key in ("error_code", "error", "sqlstate", "is_timeout", "cancelled")):
                raw = item
            if not classified and failures and isinstance(raw, dict):
                kind = classify_sqlserver_error(
                    raw.get("error_code"), raw.get("error"), sqlstate=raw.get("sqlstate"),
                    timed_out=bool(raw.get("is_timeout") or raw.get("timed_out")),
                    cancelled=bool(raw.get("cancelled", False)),
                )
                counts[kind] = failures
                classified = failures

            counts["execution_failure_count"] += max(0, failures - classified)
            failures = sum(counts[key] * weight for key, weight in SQLSERVER_FAILURE_WEIGHTS.items())
        elif normalized_db_type == "mysql":
            type_failures = max(
                0, int(item.get("type_failure_count", 0) or 0)
            )
            value_failures = max(
                0, int(item.get("value_failure_count", 0) or 0)
            )
            other_failures = max(
                0, int(item.get("other_failure_count", 0) or 0)
            )
            timeout_failures = max(
                0, int(item.get("timeout_count", 0) or 0)
            )
            classified_failures = (
                type_failures
                + value_failures
                + other_failures
                + timeout_failures
            )
            if classified_failures > 0:
                failures = (
                    float(type_failures)
                    * max(0.0, float(getattr(
                        STAGE3_CONFIG,
                        "stage3_mysql_runtime_type_failure_weight",
                        1.0,
                    )))
                    + float(value_failures)
                    * max(0.0, float(getattr(
                        STAGE3_CONFIG,
                        "stage3_mysql_runtime_value_failure_weight",
                        0.15,
                    )))
                    + float(other_failures)
                    * max(0.0, float(getattr(
                        STAGE3_CONFIG,
                        "stage3_mysql_runtime_other_failure_weight",
                        0.05,
                    )))
                    + float(timeout_failures)
                    * max(0.0, float(getattr(
                        STAGE3_CONFIG,
                        "stage3_mysql_runtime_timeout_weight",
                        0.10,
                    )))
                )
        elif normalized_db_type in _ORACLE_DB_TYPES:
            type_failures = max(
                0, int(item.get("type_failure_count", 0) or 0)
            )
            value_failures = max(
                0, int(item.get("value_failure_count", 0) or 0)
            )
            execution_failures = max(
                0, int(item.get("execution_failure_count", 0) or 0)
            )
            other_failures = max(
                0,
                int(item.get("other_failure_count", 0) or 0),
                execution_failures,
            )
            timeout_failures = max(
                0, int(item.get("timeout_count", 0) or 0)
            )
            classified_failures = (
                type_failures
                + value_failures
                + other_failures
                + timeout_failures
            )

            raw_error_info = item.get("error_info")
            if raw_error_info is None and any(
                key in item for key in ("error_code", "error", "is_timeout")
            ):
                raw_error_info = item
            if classified_failures == 0 and failures > 0 and raw_error_info:
                raw_kind = self._oracle_error_kind(raw_error_info)
                if raw_kind == "type_failure_count":
                    type_failures = failures
                elif raw_kind == "value_failure_count":
                    value_failures = failures
                elif raw_kind == "timeout_count":
                    timeout_failures = failures
                else:
                    other_failures = failures
                classified_failures = failures

            if classified_failures > 0:
                failures = (
                    float(type_failures)
                    * max(0.0, float(getattr(
                        STAGE3_CONFIG,
                        "stage3_oracle_runtime_type_failure_weight",
                        1.0,
                    )))
                    + float(value_failures)
                    * max(0.0, float(getattr(
                        STAGE3_CONFIG,
                        "stage3_oracle_runtime_value_failure_weight",
                        0.40,
                    )))
                    + float(other_failures)
                    * max(0.0, float(getattr(
                        STAGE3_CONFIG,
                        "stage3_oracle_runtime_other_failure_weight",
                        0.20,
                    )))
                    + float(timeout_failures)
                    * max(0.0, float(getattr(
                        STAGE3_CONFIG,
                        "stage3_oracle_runtime_timeout_weight",
                        0.05,
                    )))
                )
        observations = successes + failures
        minimum = max(
            1,
            int(getattr(self, "runtime_feedback_min_observations", 4)),
        )
        if observations < minimum:
            return 1.0

        if getattr(self, "runtime_feedback_tiered_enabled", True):
            success_rate = successes / observations if observations > 0 else 0.0
            if success_rate >= float(self.runtime_feedback_high_rate):
                return float(self.runtime_feedback_high_factor)
            if success_rate >= float(self.runtime_feedback_medium_rate):
                return float(self.runtime_feedback_medium_factor)
            if success_rate >= float(self.runtime_feedback_low_rate):
                return float(self.runtime_feedback_low_factor)
            return float(self.runtime_feedback_very_low_factor)

        confidence = (successes + 1.0) / (observations + 2.0)
        reliability = min(1.0, observations / minimum)
        strength = float(getattr(self, "runtime_feedback_strength", 0.0) or 0.0)
        factor = 1.0 + strength * reliability * (2.0 * confidence - 1.0)
        return max(
            float(getattr(self, "runtime_feedback_min_factor", 0.10) or 0.10),
            factor,
        )

    def _evidence_density_factor(self, node, node_id=None):
        signature = node.get("stage3_type_signature", {}) if isinstance(node, dict) else {}
        if (signature or {}).get("compatibility_class") == "unknown":
            unknown_factor = float(
                getattr(self, "unknown_exploration_ratio", 0.08) or 0.0
            )
            if str((signature or {}).get("db_type", "")).lower() == "mysql":
                unknown_factor = float(
                    getattr(
                        STAGE3_CONFIG,
                        "stage3_mysql_unknown_evidence_factor",
                        0.05,
                    )
                    or 0.05
                )
            elif str((signature or {}).get("db_type", "")).lower() in _ORACLE_DB_TYPES:
                unknown_factor = float(
                    getattr(self, "unknown_exploration_ratio", 0.05) or 0.0
                )
            evidence_factor = max(
                1e-9,
                min(1.0, max(0.0, unknown_factor)),
            )
        elif not getattr(self, "success_guided_sampling_enabled", False):
            evidence_factor = 1.0
        else:
            strength = float(getattr(self, "evidence_weight_strength", 0.0) or 0.0)
            if strength <= 0.0 or not isinstance(node, dict):
                evidence_factor = 1.0
            else:
                signature = node.get("stage3_type_signature", {}) or {}
                successes = max(0, int(signature.get("observed_success_count", 0) or 0))
                type_failures = max(
                    0,
                    int(signature.get("observed_type_failure_count", 0) or 0),
                )
                value_failures = max(
                    0,
                    int(signature.get("observed_value_failure_count", 0) or 0),
                )
                execution_failures = max(
                    0,
                    int(signature.get("observed_execution_failure_count", 0) or 0),
                )
                db_type = str(signature.get("db_type", "") or "").lower()
                type_weight = 1.0
                value_weight = 1.0
                execution_weight = 1.0
                if db_type == "sqlserver":
                    type_weight = SQLSERVER_FAILURE_WEIGHTS["type_failure_count"]
                    value_weight = SQLSERVER_FAILURE_WEIGHTS["value_failure_count"]
                    execution_weight = SQLSERVER_FAILURE_WEIGHTS["execution_failure_count"]
                elif db_type == "mysql":
                    value_weight = max(
                        0.0,
                        float(getattr(STAGE3_CONFIG, "stage3_mysql_value_failure_weight", 0.05) or 0.0),
                    )
                    execution_weight = max(
                        0.0,
                        float(getattr(STAGE3_CONFIG, "stage3_mysql_execution_failure_weight", 0.02) or 0.0),
                    )
                elif db_type in _ORACLE_DB_TYPES:
                    type_weight = max(
                        0.0,
                        float(
                            getattr(
                                STAGE3_CONFIG,
                                "stage3_oracle_type_failure_weight",
                                1.0,
                            )
                            or 0.0
                        ),
                    )
                    value_weight = max(
                        0.0,
                        float(
                            getattr(
                                STAGE3_CONFIG,
                                "stage3_oracle_value_failure_weight",
                                0.40,
                            )
                            or 0.0
                        ),
                    )
                    execution_weight = max(
                        0.0,
                        float(
                            getattr(
                                STAGE3_CONFIG,
                                "stage3_oracle_execution_failure_weight",
                                0.20,
                            )
                            or 0.0
                        ),
                    )
                effective_failures = (
                    type_weight * type_failures
                    + value_weight * value_failures
                    + execution_weight * execution_failures
                )
                observations = successes + effective_failures
                if observations <= 0:
                    evidence_factor = 1.0
                else:
                    if db_type in {"mysql", "sqlserver"}:
                        confidence = (successes + 1.0) / (observations + 2.0)
                    else:
                        confidence = signature.get("evidence_confidence")
                    if confidence is None:
                        confidence = (successes + 1.0) / (observations + 2.0)
                    confidence = min(1.0, max(0.0, float(confidence)))
                    reliability = min(
                        1.0,
                        observations / max(1, int(getattr(self, "evidence_min_observations", 1))),
                    )
                    if db_type == "mysql":
                        strength = min(
                            strength,
                            max(
                                0.0,
                                float(
                                    getattr(
                                        STAGE3_CONFIG,
                                        "stage3_mysql_evidence_weight_strength",
                                        0.12,
                                    )
                                ),
                            ),
                        )
                    elif db_type in _ORACLE_DB_TYPES:
                        strength = min(
                            strength,
                            max(
                                0.0,
                                float(
                                    getattr(
                                        STAGE3_CONFIG,
                                        "stage3_oracle_evidence_weight_strength",
                                        0.25,
                                    )
                                ),
                            ),
                        )
                    evidence_factor = 1.0 + strength * reliability * (2.0 * confidence - 1.0)

        runtime_factor = self._runtime_feedback_factor(node_id, node)
        signature = (node.get("stage3_type_signature", {}) or {}) if isinstance(node, dict) else {}
        if str(signature.get("db_type", "")).lower() in _ORACLE_DB_TYPES:
            direct = max(0, int(signature.get("observed_direct_success_count", 0)))
            context = max(0, int(signature.get("observed_context_success_count", 0)))
            successes = max(direct + context, int(signature.get("observed_success_count", 0)))
            if successes and context:
                evidence_factor *= 1.0 - 0.25 * context / successes

        return max(1e-9, evidence_factor * runtime_factor)

    def _calculate_fixed_step(self, total_count, node=None, node_id=None):
        total_count = max(0, int(total_count))
        if total_count <= 1:
            return 1
        if self.enumerate_all:
            return 1
        density_factor = self._evidence_density_factor(node, node_id=node_id)
        expected_samples = max(
            1,
            int(round(self.sampling_density * total_count.bit_length() * density_factor)),
        )
        return max(1, (total_count + expected_samples - 1) // expected_samples)

    def _prepare_decode_cache(self, nodes, deadline=None):
        cache = {}

        count_by_id = {}
        for node_id, node in nodes.items():
            check_deadline(deadline, "sampling count-cache preparation")
            count_by_id[node_id] = int(node.get('count', 0) or 0)
        for node_id, node in nodes.items():
            check_deadline(deadline, "sampling decode-cache preparation")
            slot_infos = []
            for slot in node.get('slots', []) or []:
                check_deadline(deadline, "sampling slot-cache preparation")
                cumulative_ends = []
                child_ids = []
                slot_capacity = 0
                for child_id in slot:
                    if child_id not in count_by_id:
                        continue
                    child_count = count_by_id[child_id]
                    if child_count <= 0:
                        continue
                    slot_capacity += child_count
                    cumulative_ends.append(slot_capacity)
                    child_ids.append(child_id)
                slot_infos.append((slot_capacity, cumulative_ends, child_ids))
            cache[node_id] = slot_infos

        self._decode_cache = cache
        self._decode_cache_nodes = nodes

    def _ensure_decode_cache(self, nodes, deadline=None):
        if getattr(self, "_decode_cache_nodes", None) is not nodes:
            self._prepare_decode_cache(nodes, deadline=deadline)

    @staticmethod
    def _format_count_for_log(value):
        return format_large_int(value)

    def _local_random_window_width(self, total_count):
        total_count = max(1, int(total_count))
        numerator, denominator = self.local_random_window_ratio.as_integer_ratio()
        ratio_width = (
            (total_count * numerator + denominator - 1) // denominator
            if numerator > 0
            else 0
        )
        return min(
            total_count,
            max(1, self.local_random_window_min_size, ratio_width),
        )

    def _prepare_local_random_roots(self, boolean_roots, nodes):
        ranked_roots = sorted(
            (
                root_id
                for root_id in boolean_roots
                if int((nodes.get(root_id) or {}).get("count", 0) or 0) > 0
            ),
            key=lambda root_id: (
                int((nodes.get(root_id) or {}).get("count", 0) or 0),
                root_id,
            ),
        )

        band_names = ("small", "medium", "large")
        band_size, band_remainder = divmod(len(ranked_roots), len(band_names))
        selected_count = min(self.local_random_root_pool_size, len(ranked_roots))
        quota, quota_remainder = divmod(selected_count, len(band_names))
        selected_roots = []
        root_bands = []
        start = 0
        for band_index, band_name in enumerate(band_names):
            end = start + band_size + (band_index < band_remainder)
            candidates = ranked_roots[start:end]

            take = quota + (band_index < quota_remainder)
            chosen = self.local_random_rng.sample(candidates, take)
            selected_roots.extend(chosen)
            root_bands.append({
                "band": band_name,
                "candidate_root_count": len(candidates),
                "selected_root_count": len(chosen),
                "minimum_combination_count": (
                    encode_count(int(nodes[candidates[0]]["count"])) if candidates else None
                ),
                "maximum_combination_count": (
                    encode_count(int(nodes[candidates[-1]]["count"])) if candidates else None
                ),
                "selected_root_ids": chosen,
            })
            start = end

        root_specs = []
        root_metadata = []
        for root_id in selected_roots:
            root_node = nodes[root_id]
            total_count = int(root_node.get("count", 0) or 0)
            window_width = self._local_random_window_width(total_count)
            maximum_start = total_count - window_width
            window_start = (
                self.local_random_rng.randrange(maximum_start + 1)
                if maximum_start > 0
                else 0
            )
            window_end = window_start + window_width
            signature_key, signature_detail = self._runtime_signature_identity(
                root_id,
                root_node,
            )
            root_specs.append({
                "root_id": root_id,
                "node": root_node,
                "total_count": total_count,
                "window_start": window_start,
                "window_end": window_end,
                "window_width": window_width,
                "signature_key": signature_key,
                "signature_detail": signature_detail,
            })
            root_metadata.append({
                "root_id": root_id,
                "total_count": encode_count(total_count),
                "window_start": encode_count(window_start),
                "window_end_exclusive": encode_count(window_end),
                "window_width": encode_count(window_width),
            })

        metadata = {
            "enabled": True,
            "seed": self.local_random_seed,
            "allow_duplicates": True,
            "sampling_with_replacement": True,
            "available_root_count": len(ranked_roots),
            "candidate_root_count": len(ranked_roots),
            "selected_root_count": len(root_specs),
            "configured_root_pool_size": self.local_random_root_pool_size,

            "configured_candidate_root_pool_size": (
                self.local_random_root_candidate_pool_size
            ),
            "candidate_pool_limit_applied": False,
            "root_selection_strategy": "combination_count_tertiles",
            "root_bands": root_bands,
            "local_window_ratio": self.local_random_window_ratio,
            "local_window_min_size": self.local_random_window_min_size,
            "root_block_size": self.local_random_root_block_size,
            "selected_roots": root_metadata,
            "blocks_started": 0,
            "draws_attempted": 0,
            "generated_count": 0,
        }
        return root_specs, metadata

    def _iter_local_random(
        self,
        *,
        boolean_roots,
        nodes,
        deadline,
        stop_check,
        stop_check_interval,
        progress_callback,
        progress_interval,
        progress_min_seconds,
    ):
        if deadline is None and stop_check is None:
            raise ValueError(
                "Localized-random sampling requires a deadline or stop callback."
            )

        root_specs, random_metadata = self._prepare_local_random_roots(
            boolean_roots,
            nodes,
        )
        signature_spans = []
        signature_catalog = {}
        component_tracker = self._new_component_feedback_tracker(
            nodes,
            signature_spans,
            signature_catalog,
        )
        generated_count = 0
        blocks_started = 0
        draws_attempted = 0
        next_stop_check = 0
        last_progress_log = 0.0
        stopped = False

        self.last_sample_metadata = {
            "round_complete": True,
            "stop_root_id": None,
            "stop_index": None,
            "stop_step": None,
            "stop_reason": None,
            "generated_count": 0,
            "signature_spans": signature_spans,
            "signature_catalog": signature_catalog,
            "random_sampling": random_metadata,
        }

        print(
            "  [Localized Random] "
            f"seed={self.local_random_seed}, "
            f"roots={len(root_specs)}/{len(boolean_roots)}, "
            f"candidate_roots={random_metadata['candidate_root_count']}, "
            f"allow_duplicates=True, block_size={self.local_random_root_block_size}"
        )
        for spec in root_specs:
            print(
                f"  [Localized Random Root] {spec['root_id']}, "
                f"Count: {self._format_count_for_log(spec['total_count'])}, "
                f"Window Start: {self._format_count_for_log(spec['window_start'])}, "
                f"Window Width: {self._format_count_for_log(spec['window_width'])}"
            )

        def stop_sampling(root_id, target_index, reason):
            nonlocal stopped
            stopped = True
            random_metadata["blocks_started"] = blocks_started
            random_metadata["draws_attempted"] = draws_attempted
            random_metadata["generated_count"] = generated_count
            self.last_sample_metadata.update({
                "round_complete": False,
                "stop_root_id": root_id,
                "stop_index": target_index,
                "stop_step": None,
                "stop_reason": reason,
                "generated_count": generated_count,
            })

        while root_specs and not stopped:
            if deadline is not None and time.monotonic() >= deadline:
                stop_sampling(None, None, "deadline")
                break

            spec = root_specs[self.local_random_rng.randrange(len(root_specs))]
            root_id = spec["root_id"]
            signature_key = spec["signature_key"]
            signature_catalog.setdefault(
                signature_key,
                spec["signature_detail"],
            )
            block_start_case = generated_count + 1
            blocks_started += 1

            for _ in range(self.local_random_root_block_size):
                if deadline is not None and time.monotonic() >= deadline:
                    stop_sampling(root_id, None, "deadline")
                    break

                if stop_check is not None and generated_count >= next_stop_check:
                    stop_decision = stop_check(generated_count)
                    next_stop_check = generated_count + max(
                        1,
                        int(stop_check_interval),
                    )
                    if stop_decision:
                        decision_detail = (
                            stop_decision
                            if isinstance(stop_decision, dict)
                            else {}
                        )
                        stop_sampling(
                            root_id,
                            None,
                            decision_detail.get("reason", "downstream_reserve"),
                        )
                        break

                target_index = self.local_random_rng.randrange(
                    spec["window_start"],
                    spec["window_end"],
                )
                draws_attempted += 1
                next_case_idx = generated_count + 1
                collect_component_trace = self._should_trace_component_feedback(
                    component_tracker,
                    next_case_idx,
                )
                render_started = time.monotonic()
                try:
                    sql, edge_keys = self._generate_predicate_with_trace(
                        root_id,
                        target_index,
                        nodes,
                        collect_trace=collect_component_trace,
                        deadline=deadline,
                    )
                except Stage3DeadlineReached:
                    stop_sampling(root_id, target_index, "deadline")
                    break
                finally:
                    self.last_sampling_timings[
                        "predicate_traversal_render_seconds"
                    ] += max(0.0, time.monotonic() - render_started)
                    self.last_sampling_timings["predicate_traversal_attempts"] += 1

                if not sql or "ERROR" in sql:
                    continue

                generated_count += 1
                self.last_sample_metadata["generated_count"] = generated_count
                if collect_component_trace:
                    self._record_component_feedback_trace(
                        component_tracker,
                        generated_count,
                        edge_keys,
                    )
                yield sql

                now = time.monotonic()
                if (
                    progress_callback is not None
                    and progress_interval > 0
                    and generated_count % progress_interval == 0
                ):
                    progress_callback(generated_count)
                if (
                    progress_interval > 0
                    and generated_count % progress_interval == 0
                    and (
                        last_progress_log == 0.0
                        or now - last_progress_log >= progress_min_seconds
                    )
                ):
                    current_time = time.strftime(
                        "%Y-%m-%d %H:%M:%S",
                        time.localtime(),
                    )
                    print(
                        f"  [Progress] Time: {current_time}, "
                        f"Generated predicates so far: {generated_count}"
                    )
                    last_progress_log = now

            if generated_count >= block_start_case:
                if (
                    signature_spans
                    and signature_spans[-1]["signature_key"] == signature_key
                    and signature_spans[-1]["end_case_idx"] == block_start_case - 1
                ):
                    signature_spans[-1]["end_case_idx"] = generated_count
                else:
                    signature_spans.append({
                        "start_case_idx": block_start_case,
                        "end_case_idx": generated_count,
                        "signature_key": signature_key,
                    })

        random_metadata["blocks_started"] = blocks_started
        random_metadata["draws_attempted"] = draws_attempted
        random_metadata["generated_count"] = generated_count
        self._flush_component_feedback_block(component_tracker, generated_count)
        self.last_sample_metadata["generated_count"] = generated_count

    def iter_file(
        self,
        dag_data,
        deadline=None,
        stop_check=None,
        stop_check_interval=1000,
        progress_callback=None,
    ):
        self.last_sampling_timings = _new_sampling_timings()
        nodes = dag_data['nodes']
        generated_count = 0
        progress_interval = int(
            getattr(STAGE3_CONFIG, "stage3_predicate_progress_interval", 1000)
        )
        progress_min_seconds = max(
            0.0,
            float(
                getattr(
                    STAGE3_CONFIG,
                    "stage3_predicate_progress_min_seconds",
                    5.0,
                )
            ),
        )
        signature_spans = []
        signature_catalog = {}
        component_tracker = self._new_component_feedback_tracker(
            nodes,
            signature_spans,
            signature_catalog,
        )
        self.last_sample_metadata = {
            "round_complete": True,
            "stop_root_id": None,
            "stop_index": None,
            "stop_step": None,
            "stop_reason": None,
            "generated_count": 0,
            "signature_spans": signature_spans,
            "signature_catalog": signature_catalog,
        }
        stop_check_interval = max(1, int(stop_check_interval))
        next_stop_check = 0
        check_deadline(deadline, "sampling initialization")
        ordering_started = time.monotonic()
        try:
            self._order_successors_in_place(nodes, deadline=deadline)
        finally:
            self.last_sampling_timings["successor_ordering_seconds"] += max(
                0.0,
                time.monotonic() - ordering_started,
            )
        cache_started = time.monotonic()
        try:
            self._prepare_decode_cache(nodes, deadline=deadline)
        finally:
            self.last_sampling_timings["decode_cache_preparation_seconds"] += max(
                0.0,
                time.monotonic() - cache_started,
            )

        boolean_roots = []
        for nid, node in nodes.items():
            if node.get('return_type') == 'T_BOOL' and not node.get('is_terminal'):
                if node.get('slots'):  
                    boolean_roots.append(nid)

        def get_node_level(node_id):
            try:
                level_str = node_id.split('::')[-1]
                return int(level_str.replace('L', ''))
            except ValueError:
                return -1  

        boolean_roots.sort(key=get_node_level, reverse=True)

        if not boolean_roots:
            print('  [Warning] No valid Boolean root nodes found in the current graph; skipping sampling.')
            return

        if self.localized_random:
            yield from self._iter_local_random(
                boolean_roots=boolean_roots,
                nodes=nodes,
                deadline=deadline,
                stop_check=stop_check,
                stop_check_interval=stop_check_interval,
                progress_callback=progress_callback,
                progress_interval=progress_interval,
                progress_min_seconds=progress_min_seconds,
            )
            return

        total_progress_roots = sum(
            1 for root_id in boolean_roots
            if nodes.get(root_id) and nodes[root_id].get("count", 0) != 0
        )
        completed_roots = 0
        last_sampling_progress_log = time.monotonic()

        def log_sampling_progress(now, *, force=False):
            nonlocal last_sampling_progress_log
            if force or now - last_sampling_progress_log >= 15.0:
                _print_sampling_progress(completed_roots, total_progress_roots)
                last_sampling_progress_log = now

        log_sampling_progress(last_sampling_progress_log, force=True)
        for root_id in boolean_roots:
            root_node = nodes.get(root_id)
            if not root_node:
                continue

            total_count = root_node.get('count', 0)

            if total_count == 0:
                continue

            signature_key, signature_detail = self._runtime_signature_identity(
                root_id,
                root_node,
            )
            signature_catalog.setdefault(signature_key, signature_detail)
            root_start_case = generated_count + 1
            fixed_step = self._calculate_fixed_step(
                total_count,
                node=root_node,
                node_id=root_id,
            )

            current_index = 0
            while current_index < total_count:
                if deadline is not None and time.monotonic() >= deadline:
                    self.last_sample_metadata = {
                        "round_complete": False,
                        "stop_root_id": root_id,
                        "stop_index": current_index,
                        "stop_step": fixed_step,
                        "stop_reason": "deadline",
                        "generated_count": generated_count,
                        "signature_spans": signature_spans,
                        "signature_catalog": signature_catalog,
                    }
                    break

                if stop_check is not None and generated_count >= next_stop_check:
                    stop_decision = stop_check(generated_count)
                    next_stop_check = generated_count + stop_check_interval
                    if stop_decision:
                        decision_detail = (
                            stop_decision if isinstance(stop_decision, dict) else {}
                        )
                        self.last_sample_metadata = {
                            "round_complete": False,
                            "stop_root_id": root_id,
                            "stop_index": current_index,
                            "stop_step": fixed_step,
                            "stop_reason": decision_detail.get(
                                "reason",
                                "downstream_reserve",
                            ),
                            "generated_count": generated_count,
                            "signature_spans": signature_spans,
                            "signature_catalog": signature_catalog,
                        }
                        break

                next_case_idx = generated_count + 1
                collect_component_trace = self._should_trace_component_feedback(
                    component_tracker,
                    next_case_idx,
                )
                render_started = time.monotonic()
                try:
                    sql, edge_keys = self._generate_predicate_with_trace(
                        root_id,
                        current_index,
                        nodes,
                        collect_trace=collect_component_trace,
                        deadline=deadline,
                    )
                except Stage3DeadlineReached:
                    self.last_sample_metadata = {
                        "round_complete": False,
                        "stop_root_id": root_id,
                        "stop_index": current_index,
                        "stop_step": fixed_step,
                        "stop_reason": "deadline",
                        "generated_count": generated_count,
                        "signature_spans": signature_spans,
                        "signature_catalog": signature_catalog,
                    }
                    break
                finally:
                    self.last_sampling_timings[
                        "predicate_traversal_render_seconds"
                    ] += max(0.0, time.monotonic() - render_started)
                    self.last_sampling_timings["predicate_traversal_attempts"] += 1

                if sql and "ERROR" not in sql:
                    generated_count += 1
                    self.last_sample_metadata["generated_count"] = generated_count
                    if collect_component_trace:
                        self._record_component_feedback_trace(
                            component_tracker,
                            generated_count,
                            edge_keys,
                        )

                    yield sql
                    now = time.monotonic()
                    if (
                        progress_callback is not None
                        and progress_interval > 0
                        and generated_count % progress_interval == 0
                    ):
                        progress_callback(generated_count)
                    log_sampling_progress(now)

                current_index += fixed_step

            if generated_count >= root_start_case:
                signature_spans.append({
                    "start_case_idx": root_start_case,
                    "end_case_idx": generated_count,
                    "signature_key": signature_key,
                })

            if not self.last_sample_metadata["round_complete"]:
                break

            completed_roots += 1
            if completed_roots < total_progress_roots:
                log_sampling_progress(time.monotonic())

        self._flush_component_feedback_block(component_tracker, generated_count)
        self.last_sample_metadata["generated_count"] = generated_count
        if total_progress_roots > 0:
            log_sampling_progress(time.monotonic(), force=True)

    def sample_file(self, dag_data, deadline=None):
        return list(self.iter_file(dag_data, deadline=deadline))

    def generate_predicate(self, node_id, target_index, nodes, deadline=None):
        self._ensure_decode_cache(nodes, deadline=deadline)
        sql, _ = self._generate_predicate_with_trace(
            node_id,
            target_index,
            nodes,
            collect_trace=False,
            deadline=deadline,
        )
        return sql

    @staticmethod
    def _slot_expected_types(node, slot_index):
        signature = node.get("stage3_type_signature", {}) or {}
        accepted = signature.get("accepted_arg_types", []) or []
        if not 0 <= int(slot_index) < len(accepted):
            return []
        return [str(item).strip() for item in (accepted[slot_index] or []) if str(item).strip()]

    @staticmethod
    def _is_castable_type_name(type_name):
        normalized = " ".join(str(type_name or "").strip().split())
        if not normalized or normalized.startswith(("__", "any", "internal")):
            return False
        if normalized in {"unknown", "record", "void", "cstring"}:
            return False
        return bool(
            re.fullmatch(
                r'(?:"[A-Za-z_][A-Za-z0-9_]*"|[A-Za-z_][A-Za-z0-9_]*)'
                r'(?:\s+[A-Za-z_][A-Za-z0-9_]*)*(?:\[\])?',
                normalized,
            )
        )

    def _render_contextual_null(self, expected_types):
        if (
            not getattr(self, "contextual_null_cast_enabled", False)
            or str(getattr(self, "db_type", "")).lower() not in {"postgresql", "postgres", "pg"}
        ):
            return "NULL"
        candidates = sorted(
            {
                " ".join(str(item).strip().split())
                for item in (expected_types or [])
                if self._is_castable_type_name(item)
            }
        )
        if not candidates:
            return "NULL"

        if len(candidates) != 1:
            return "NULL"
        return f"CAST(NULL AS {candidates[0]})"

    @staticmethod
    def _safe_literal_matches_terminal(token, literal):
        upper_token = str(token or "").upper()
        text = str(literal or "").strip()
        if not text:
            return False
        if text.upper() == "NULL":
            return True
        if "STRING" in upper_token or upper_token.startswith("HOLE_T_STR"):
            return len(text) >= 2 and text[0] in {"'", '"'} and text[-1] == text[0]
        if "BOOL" in upper_token:
            return text.upper() in {"TRUE", "FALSE", "0", "1"}
        if any(marker in upper_token for marker in ("INT", "FLOAT", "CMP")):
            return bool(re.fullmatch(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)", text))
        return False

    @staticmethod
    def _oracle_literal_matches_expected_types(literal, expected_types):
        normalized = {
            " ".join(str(item or "").strip().lower().split())
            for item in (expected_types or [])
            if str(item or "").strip()
        }
        if len(normalized) != 1:
            return True
        expected = next(iter(normalized))
        text = str(literal or "").strip()
        if expected in {
            "date", "timestamp", "timestamp without time zone",
            "timestamp with time zone", "timestamp with local time zone",
            "interval year to month", "interval day to second",
        }:
            return bool(re.match(r"^(DATE|TIMESTAMP|INTERVAL)\b", text, re.IGNORECASE))
        if expected in {
            "number", "integer", "smallint", "bigint", "float", "binary_float",
            "binary_double", "numeric", "decimal",
        }:
            return bool(re.fullmatch(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)", text))
        if expected in {"raw", "long raw", "blob"}:
            return bool(re.match(r"^(HEXTORAW|TO_BLOB)\s*\(", text, re.IGNORECASE))
        if expected == "json":
            return bool(re.match(r"^JSON\s*\(", text, re.IGNORECASE))
        if expected == "boolean":
            return text.upper() in {"TRUE", "FALSE"}
        return True

    def _safe_argument_literal(
        self,
        token,
        parent_token,
        parent_slot_index,
        *,
        apply_exploration=True,
    ):
        if (
            not getattr(self, "safe_value_guidance_enabled", False)
            or parent_token is None
            or parent_slot_index is None
            or not str(token or "").upper().startswith(("VALUE_", "HOLE_"))
        ):
            return None
        exploration_ratio = float(
            getattr(self, "safe_value_exploration_ratio", 0.20) or 0.0
        )
        if apply_exploration and random.random() < exploration_ratio:
            return None

        parent_key = str(parent_token).strip().upper()
        slot_key = str(int(parent_slot_index))
        configured_profiles = getattr(self, "safe_argument_literals", {}) or {}
        slot_values = configured_profiles.get(parent_key, {})
        if not isinstance(slot_values, dict):
            return None
        literal = slot_values.get(slot_key)
        if literal is None and getattr(self, "db_type", "") in _ORACLE_DB_TYPES:
            literal = _ORACLE_SAFE_ARGUMENT_LITERALS.get(parent_key, {}).get(slot_key)
        if literal is None or not self._safe_literal_matches_terminal(token, literal):
            return None
        return str(literal)

    def _oracle_expected_literal(self, token, expected_types):
        if getattr(self, "db_type", "") not in _ORACLE_DB_TYPES or not expected_types:
            return None
        normalized = {
            " ".join(str(item or "").strip().lower().split())
            for item in expected_types
            if str(item or "").strip()
        }
        if len(normalized) != 1:
            return None
        expected = next(iter(normalized))
        if expected in {
            "date",
            "timestamp",
            "timestamp without time zone",
            "timestamp with time zone",
            "timestamp with local time zone",
        }:
            if expected == "date":
                return "DATE '2020-01-01'"
            if expected == "timestamp with time zone":
                return "TIMESTAMP '2020-01-01 00:00:00 +00:00'"
            return "TIMESTAMP '2020-01-01 00:00:00'"
        if expected == "interval year to month":
            return "INTERVAL '1-0' YEAR TO MONTH"
        if expected == "interval day to second":
            return "INTERVAL '1' DAY"
        if expected in {"varchar2", "char", "long"}:
            return "'probe_str'"
        if expected in {"nvarchar2", "nchar"}:
            return "N'probe_str'"
        if expected == "clob":
            return "TO_CLOB('probe_str')"
        if expected == "nclob":
            return "TO_NCLOB(N'probe_str')"
        if expected in {"raw", "long raw"}:
            return "HEXTORAW('00')"
        if expected == "blob":
            return "TO_BLOB(HEXTORAW('00'))"
        if expected == "json":
            return "JSON('{\"probe\":1}')"
        if expected == "xmltype":
            return "XMLTYPE('<probe/>')"
        if expected == "boolean":
            return "TRUE"
        if expected in {
            "number", "integer", "smallint", "bigint", "float",
            "binary_float", "binary_double", "numeric", "decimal",
        }:
            return "1"
        return None

    def _generate_predicate_with_trace(
        self,
        node_id,
        target_index,
        nodes,
        collect_trace=True,
        deadline=None,
        expected_types=None,
        parent_token=None,
        parent_slot_index=None,
    ):
        check_deadline(deadline, "single-predicate rendering")
        node = nodes[node_id]

        if not node['slots']:
            return self.instantiate_terminal(
                node,
                expected_types=expected_types,
                parent_token=parent_token,
                parent_slot_index=parent_slot_index,
            ), []

        child_sqls = []  
        temp_index = target_index  

        selected_edges = []
        decode_cache = getattr(self, "_decode_cache", None)
        if decode_cache is None:
            decode_cache = {}
            self._decode_cache = decode_cache
        decode_infos = decode_cache.get(node_id)
        if decode_infos is None:
            self._ensure_decode_cache(nodes, deadline=deadline)
            decode_infos = self._decode_cache.get(node_id, [])

        for slot_index in range(len(node['slots']) - 1, -1, -1):
            slot_capacity, cumulative_ends, child_ids = decode_infos[slot_index]

            if slot_capacity == 0:
                return "ERROR_DEAD_END", []

            local_index = temp_index % slot_capacity

            temp_index = temp_index // slot_capacity

            child_position = bisect_right(cumulative_ends, local_index)
            selected_child_id = (
                child_ids[child_position]
                if child_position < len(child_ids)
                else None
            )
            previous_end = (
                cumulative_ends[child_position - 1]
                if child_position > 0
                else 0
            )
            child_relative_index = local_index - previous_end

            if selected_child_id:
                child_expected_types = self._slot_expected_types(node, slot_index)
                child_sql, child_edges = self._generate_predicate_with_trace(
                    selected_child_id,
                    child_relative_index,
                    nodes,
                    collect_trace=collect_trace,
                    deadline=deadline,
                    expected_types=child_expected_types,
                    parent_token=node.get("token"),
                    parent_slot_index=slot_index,
                )

                child_sqls.insert(0, child_sql)
                if collect_trace:
                    source_id = self._original_node_id(node_id, nodes)
                    target_id = self._original_node_id(selected_child_id, nodes)
                    selected_edges.append(f"{source_id}::slot{slot_index}->{target_id}")
                    selected_edges.extend(child_edges)
            else:
                child_sqls.insert(0, "ERROR_MISSING_CHILD")

        return self.format_sql(node['token'], child_sqls), selected_edges

    def instantiate_terminal(
        self,
        node,
        *,
        expected_types=None,
        parent_token=None,
        parent_slot_index=None,
    ):
        token = node['token']

        if getattr(self, "db_type", "") == "sqlserver":
            literal = self._instantiate_sqlserver_literal(
                token, expected_types, parent_token, parent_slot_index,
            )
            if literal is not None:
                return literal

        if token.endswith("()"):
            return token

        if token == "NULL" or (token.startswith("VALUE_") and "NULL" in token):
            return self._render_contextual_null(expected_types)

        literal_token = token.startswith("HOLE_") or token.startswith("VALUE_")
        oracle_guidance_allowed = True
        db_type = str(getattr(self, "db_type", "") or "").lower().strip()
        if db_type in _ORACLE_DB_TYPES and literal_token:
            exploration_ratio = float(
                getattr(self, "safe_value_exploration_ratio", 0.05) or 0.0
            )
            oracle_guidance_allowed = random.random() >= exploration_ratio

        safe_literal = None
        if oracle_guidance_allowed:
            safe_literal = self._safe_argument_literal(
                token,
                parent_token,
                parent_slot_index,
                apply_exploration=db_type not in _ORACLE_DB_TYPES,
            )
            if (
                safe_literal is not None
                and db_type in _ORACLE_DB_TYPES
                and not self._oracle_literal_matches_expected_types(
                    safe_literal,
                    expected_types,
                )
            ):
                safe_literal = None
        if safe_literal is not None:
            return safe_literal

        oracle_literal = (
            self._oracle_expected_literal(token, expected_types)
            if oracle_guidance_allowed
            else None
        )
        if oracle_literal is not None and literal_token:
            return oracle_literal

        if token.startswith("HOLE_T_CMP"):
            if random.random() < 0.8:
                return str(random.randint(-100, 1000))
            else:
                return f"'{self.random_string()}'"

        elif token.startswith("HOLE_T_STR"):
            base_str = self.random_string()
            if random.random() < 0.5:
                return f"'{base_str}%'"  
            else:
                return f"'{base_str}'"

        elif token.startswith("HOLE_T_BOOL"):
            if db_type in _ORACLE_DB_TYPES:
                return random.choice(["TRUE", "FALSE"])
            return str(random.choice([0, 1]))

        elif token.startswith("VALUE_"):
            if "SINGULAR" in token:
                if "INT" in token:
                    candidates = ["0", "-1", "1", "2147483647", "-2147483648"]
                    if self.allow_hex_int_literals:
                        candidates.extend(["0xFFFFFFFF", "0x00"])
                    return random.choice(candidates)
                elif "STRING" in token:
                    return random.choice(["''", "' '", "'\\0'"])
                elif "FLOAT" in token:
                    return random.choice(["0.0", "-0.0", "999999.99"])
                else:
                    return "0"
            else:
                if "INT" in token:
                    if self.allow_hex_int_literals and random.random() < 0.2:
                        return random.choice(["0xFF", "0x7F", "0x0A", "0x55", "0xAA"])
                    else:
                        return str(random.randint(2, 1000))
                elif "STRING" in token:
                    return f"'{random.choice(['typical_str', 'test_data', 'admin'])}'"
                elif "FLOAT" in token:
                    return str(round(random.uniform(1.0, 100.0), 2))
                else:
                    return "100"

        elif token.startswith("COLUMN_") or token.startswith("COL_"):
            if getattr(self, "exact_column_type_guidance_enabled", True):
                if getattr(self, "db_type", "") == "sqlserver":
                    from RL.sqlserver_evidence_types import normalize_sqlserver_type

                    types = {normalize_sqlserver_type(item)
                             for item in (node.get("stage3_result_types", []) or [])}
                    types.discard("")
                    exact = (next(iter(types)) if len(types) == 1 else
                             normalize_sqlserver_type(node.get("stage3_column_type")))
                    if exact and exact not in {"condition", "null"}:
                        return f"col(__TYPE__{exact})"
                exact_type = " ".join(
                    str(node.get("stage3_column_type", "") or "").strip().split()
                )
                if not exact_type:
                    exact_types = sorted({
                        " ".join(str(item).strip().split())
                        for item in (node.get("stage3_result_types", []) or [])
                        if self._is_castable_type_name(item)
                    })
                    if len(exact_types) == 1:
                        exact_type = exact_types[0]
                if exact_type and self._is_castable_type_name(exact_type):
                    return f"col(__TYPE__{exact_type})"

            parts = token.split("_")
            col_type = parts[1] if len(parts) > 1 else "CMP"

            return f"col({col_type}_mock)"

        else:
            return token

    @staticmethod
    def _sqlserver_literal_family(literal):
        text = str(literal).strip()
        if text.upper() in {"TRUE", "FALSE"} or re.fullmatch(r"\(\s*1\s*=\s*[01]\s*\)", text):
            return "bool"
        if re.fullmatch(r"0[xX][0-9A-Fa-f]*", text):
            return "binary"
        if re.fullmatch(r"[Nn]?'(?:[^']|'')*'", text):
            return "string"
        if re.fullmatch(r"[-+]?\d+", text):
            return "int"
        if re.fullmatch(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?", text):
            return "float"
        return None

    @staticmethod
    def _sqlserver_terminal_accepts_literal(token, family, literal=None):
        upper = str(token).upper()
        if upper.startswith("HOLE_T_CMP"):
            return family in {"int", "float", "string", "binary"}
        if "BOOL" in upper:
            return family == "bool"
        if "STRING" in upper or upper.startswith("HOLE_T_STR"):
            return family == "string"
        if upper.endswith("_DATE"):
            return bool(re.fullmatch(
                r"'(?:\d{8}|\d{4}-\d{2}-\d{2}(?:T[0-9:+.\-]+)?|\d{2}:\d{2}:\d{2})'",
                str(literal),
            ))
        if upper.endswith("_UNIQUEIDENTIFIER"):
            return bool(re.fullmatch(r"'[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}'", str(literal)))
        if upper.endswith("_BINARY"):
            return family == "binary"
        if upper.endswith("_INT"):
            return family == "int"
        if upper.endswith("_FLOAT"):
            return family in {"int", "float"}
        return False

    @staticmethod
    def _sqlserver_expected_literal(expected_types):
        normalized = {str(item).strip().lower() for item in (expected_types or []) if str(item).strip()}
        if len(normalized) != 1:
            return None
        expected = next(iter(normalized))
        match = re.fullmatch(r"([a-z][a-z0-9]*)(?:\s*\(\s*(?:max|\d+)(?:\s*,\s*\d+)?\s*\))?", expected)
        if not match:
            return None
        base_type = match.group(1)

        temporal = {
            "date": "'20240115'",
            "time": "'12:34:56'",
            "smalldatetime": "'2024-01-15T12:34:00'",
            "datetime": "'2024-01-15T12:34:56'",
            "datetime2": "'2024-01-15T12:34:56'",
            "datetimeoffset": "'2024-01-15T12:34:56+00:00'",
            "uniqueidentifier": "'01234567-89ab-cdef-0123-456789abcdef'",
        }
        if base_type in temporal:
            return temporal[base_type]
        if base_type in {"binary", "varbinary", "image", "timestamp", "rowversion"}:
            return "0x0000000000000001" if base_type in {"timestamp", "rowversion"} else "0x01"
        if base_type in {"bit", "tinyint", "smallint", "int", "integer", "bigint"}:
            return "1"
        if base_type in {"float", "real", "decimal", "numeric", "money", "smallmoney"}:
            return "1.0"
        if base_type in {"nvarchar", "nchar", "ntext"}:
            return "N'中'"
        if base_type in {"char", "varchar", "text", "string"}:
            return "'a'"
        return None

    def _instantiate_sqlserver_literal(self, token, expected_types, parent_token, parent_slot_index):
        upper = str(token).upper()
        if upper in {"TRUE", "FALSE"}:
            return "(1 = 1)" if upper == "TRUE" else "(1 = 0)"
        if not upper.startswith(("VALUE_", "HOLE_")) or "NULL" in upper:
            return None

        guided = (
            getattr(self, "safe_value_guidance_enabled", False)
            and random.random() >= float(getattr(self, "safe_value_exploration_ratio", 0.20) or 0.0)
        )
        expected_literal = self._sqlserver_expected_literal(expected_types)
        if guided:
            profiles = getattr(self, "safe_argument_literals", {}) or {}
            slots = profiles.get(str(parent_token or "").strip().upper(), {})
            fixed = slots.get(str(parent_slot_index)) if isinstance(slots, dict) else None
            family = self._sqlserver_literal_family(fixed)

            expected_family = self._sqlserver_literal_family(expected_literal)
            if (fixed is not None and self._sqlserver_terminal_accepts_literal(upper, family, fixed)
                    and (expected_literal is None or fixed == expected_literal
                         or (family == "string" and expected_literal in {"'a'", "N'中'"})
                         or (family in {"int", "float"} and expected_family in {"int", "float"}))):
                if upper.endswith("_FLOAT") and family == "int":
                    return str(fixed) + ".0"
                return {"TRUE": "(1 = 1)", "FALSE": "(1 = 0)"}.get(str(fixed).upper(), str(fixed))
            if expected_literal is not None and self._sqlserver_terminal_accepts_literal(upper, expected_family, expected_literal):
                if upper.endswith("_FLOAT") and expected_family == "int":
                    return expected_literal + ".0"
                return expected_literal
            if upper.endswith("_INT") and expected_literal == "1.0":
                return "1"

        if upper.startswith("HOLE_T_BOOL"):
            return random.choice(["(1 = 1)", "(1 = 0)"])
        if upper.startswith("HOLE_T_CMP"):
            if random.random() < 0.8:
                return str(random.randint(-100, 1000))
            return "N'" + self.random_string().replace("'", "''") + "'"
        if upper.startswith("HOLE_T_STR"):
            value = self.random_string() + ("%" if random.random() < 0.5 else "")
            return "N'" + value.replace("'", "''") + "'"
        if upper.endswith("_INT"):
            if "SINGULAR" in upper:
                return random.choice(["0", "-1", "1", "2147483647", "-2147483648"])
            return str(random.randint(2, 1000))
        if upper.endswith("_FLOAT"):
            if "SINGULAR" in upper:
                return random.choice(["0.0", "-0.0", "999999.99"])
            return str(round(random.uniform(1.0, 100.0), 2))
        if upper.endswith("_STRING"):
            values = ["", " ", "'"] if "SINGULAR" in upper else ["typical_str", "test_data", "中文", "O'Brien"]
            return "N'" + random.choice(values).replace("'", "''") + "'"

        if upper.endswith("_BINARY"):
            return random.choice(["0x", "0x00", "0xFF", "0x0123456789ABCDEF"])
        if upper.endswith("_DATE"):
            return "'20240115'"
        if upper.endswith("_UNIQUEIDENTIFIER"):
            return "'01234567-89ab-cdef-0123-456789abcdef'"
        return None

    def _legacy_format_sql(self, token, child_sqls):
        if token.endswith("()") or token == "col()":
            func_name = token.replace("()", "")

            args_str = ", ".join(child_sqls)
            return f"{func_name}({args_str})"

        elif len(child_sqls) == 1:
            return f"({token} {child_sqls[0]})"

        elif len(child_sqls) == 2:
            return f"({child_sqls[0]} {token} {child_sqls[1]})"

        elif len(child_sqls) == 3 and token.upper() == "BETWEEN":
            return f"({child_sqls[0]} BETWEEN {child_sqls[1]} AND {child_sqls[2]})"

        else:
            return f"{token}({', '.join(child_sqls)})"

    def _format_in_rhs(self, rhs_sql):
        rhs = str(rhs_sql).strip()

        if not rhs:
            return "(NULL)"

        if rhs.startswith("(") and rhs.endswith(")"):
            return rhs

        return f"({rhs})"

    def format_sql(self, token, child_sqls):
        token_text = str(token).strip()
        upper_token = token_text.upper()
        op_style = self.operator_styles.get(upper_token, "")

        if getattr(self, "db_type", "") == "sqlserver" and token_text.lower() != "col()":
            signature = (getattr(self, "rule_config", {}) or {}).get("definitions", {}).get(upper_token)
            if not signature or not op_style:
                raise ValueError(f"SQL Server operator is outside the configured scope: {token_text}")
            if (len(child_sqls) < 2 if op_style == "in_list" else len(child_sqls) != len(signature[1])):
                raise ValueError(f"SQL Server operator has incorrect arity: {token_text}")

            child_sqls = [
                {"TRUE": "(1 = 1)", "FALSE": "(1 = 0)"}.get(str(child).strip().upper(), child)
                for child in child_sqls
            ]

        if token_text.endswith("()") or token_text == "col()":
            func_name = token_text.replace("()", "")
            args_str = ", ".join(child_sqls)
            return f"{func_name}({args_str})"

        if op_style == "between" and len(child_sqls) == 3:
            return f"({child_sqls[0]} {token_text} {child_sqls[1]} AND {child_sqls[2]})"

        if op_style == "in_list" and len(child_sqls) >= 2:
            rhs_items = ", ".join(child_sqls[1:])
            return f"({child_sqls[0]} {token_text} ({rhs_items}))"

        if op_style == "postfix" and len(child_sqls) == 1:
            return f"({child_sqls[0]} {token_text})"

        if op_style == "prefix" and len(child_sqls) == 1:
            return f"({token_text} {child_sqls[0]})"

        if op_style == "binary_infix" and len(child_sqls) == 2:
            return f"({child_sqls[0]} {token_text} {child_sqls[1]})"

        if len(child_sqls) == 3 and upper_token == "BETWEEN":
            return f"({child_sqls[0]} BETWEEN {child_sqls[1]} AND {child_sqls[2]})"

        if len(child_sqls) == 3 and upper_token == "NOT BETWEEN":
            return f"({child_sqls[0]} NOT BETWEEN {child_sqls[1]} AND {child_sqls[2]})"

        if len(child_sqls) == 2 and upper_token == "IN":
            rhs = self._format_in_rhs(child_sqls[1])
            return f"({child_sqls[0]} IN {rhs})"

        if len(child_sqls) == 2 and upper_token == "NOT IN":
            rhs = self._format_in_rhs(child_sqls[1])
            return f"({child_sqls[0]} NOT IN {rhs})"

        if len(child_sqls) == 1:
            return f"({token_text} {child_sqls[0]})"

        if len(child_sqls) == 2:
            return f"({child_sqls[0]} {token_text} {child_sqls[1]})"

        return f"{token_text}({', '.join(child_sqls)})"

    def random_string(self, length=5):
        import string
        return ''.join(random.choices(string.ascii_letters, k=length))


def _depth_scoped_predicate_name(file_name, max_depth, output_label=None):
    stem, extension = os.path.splitext(file_name)
    if output_label is not None:
        label = str(output_label).strip()
        if not re.fullmatch(r"[A-Za-z0-9_-]+", label):
            raise ValueError("Stage-3 output label must contain only letters, digits, '_' or '-'")
        return f"{stem}-{label}{extension}"
    if max_depth is None:
        return file_name
    return f"{stem}-{int(max_depth)}{extension}"


def sampler(
    deadline=None,
    input_files=None,
    max_depth=None,
    downstream_cost_model=None,
    runtime_signature_feedback=None,
    progress_callback=None,
    enumerate_all=False,
    localized_random=False,
    output_label=None,
):
    sampling_start = time.monotonic()
    sampling_timings = _new_sampling_timings()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    input_dir = os.path.join(base_dir, 'counted_graph')  
    output_dir = os.path.join(base_dir, 'generated_sqls')  

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    if not os.path.exists(input_dir):
        print(f"Error: Input directory not found: {input_dir}")
        return {
            "success": False,
            "round_complete": False,
            "phase": "sampling",
            "error": f"Input directory not found: {input_dir}",
            "total_generated": 0,
            "output_files": [],
            "sampling_elapsed_seconds": time.monotonic() - sampling_start,
            "sampling_timings": sampling_timings,
        }

    sampler_setup_started = time.monotonic()
    try:
        sampler = PredicateSampler(
            step_size=1,
            runtime_signature_feedback=runtime_signature_feedback,
            enumerate_all=enumerate_all,
            localized_random=localized_random,
        )
    finally:
        sampling_timings["sampler_setup_seconds"] += max(
            0.0,
            time.monotonic() - sampler_setup_started,
        )
    if input_files is None:
        json_files = [
            os.path.join(input_dir, file_name)
            for file_name in os.listdir(input_dir)
            if file_name.endswith('.json')
        ]
    else:
        json_files = [str(path) for path in input_files if str(path).endswith('.json')]

    if not json_files:
        print("No counted JSON files found.")
        return {
            "success": False,
            "round_complete": False,
            "phase": "sampling",
            "error": "No counted JSON files found.",
            "total_generated": 0,
            "output_files": [],
            "sampling_elapsed_seconds": time.monotonic() - sampling_start,
            "sampling_timings": sampling_timings,
        }

    print(f"Found {len(json_files)} files. Starting sampling...")

    total_generated = 0
    output_buffer_bytes = max(
        4096,
        int(getattr(STAGE3_CONFIG, "stage3_output_buffer_bytes", 1048576)),
    )
    compress_predicates = bool(
        getattr(STAGE3_CONFIG, "stage3_compress_predicate_output", True)
    )
    predicate_compression_level = min(
        9,
        max(
            1,
            int(
                getattr(
                    STAGE3_CONFIG,
                    "stage3_predicate_compression_level",
                    1,
                )
            ),
        ),
    )
    reserve_enabled = bool(
        downstream_cost_model is not None
        and deadline is not None
        and getattr(STAGE3_CONFIG, "stage3_downstream_reserve_enabled", True)
    )
    reserve_min_seconds = max(
        0.0,
        float(
            getattr(
                STAGE3_CONFIG,
                "stage3_downstream_reserve_min_seconds",
                120.0,
            )
        ),
    )
    reserve_safety_factor = max(
        1.0,
        float(
            getattr(
                STAGE3_CONFIG,
                "stage3_downstream_reserve_safety_factor",
                1.25,
            )
        ),
    )
    reserve_fixed_seconds = max(
        0.0,
        float(
            getattr(
                STAGE3_CONFIG,
                "stage3_downstream_reserve_fixed_seconds",
                30.0,
            )
        ),
    )
    reserve_check_interval = max(
        1,
        int(
            getattr(
                STAGE3_CONFIG,
                "stage3_downstream_reserve_check_interval",
                1000,
            )
        ),
    )
    round_complete = True
    stop_metadata = {
        "stop_root_id": None,
        "stop_index": None,
        "stop_step": None,
        "stop_reason": None,
    }
    output_files = []
    output_text_bytes = {}
    output_text_characters = {}
    total_text_bytes = 0
    total_text_characters = 0
    signature_spans_by_file = {}
    signature_catalog = {}
    random_sampling_by_file = {}
    stratified_layers_by_file = {}
    reserve_status = {
        "enabled": reserve_enabled,
        "triggered": False,
        "check_interval_predicates": reserve_check_interval,
        "minimum_seconds": reserve_min_seconds,
        "safety_factor": reserve_safety_factor,
        "fixed_seconds": reserve_fixed_seconds,
        "remaining_seconds": remaining_seconds(deadline),
        "predicate_count": 0,
        "predicate_text_bytes": 0,
    }
    error_message = None

    for input_path in sorted(json_files):
        file_name = os.path.basename(input_path)

        iter_file_started = False
        try:
            check_deadline(deadline, "counted DAG loading")

            load_started = time.monotonic()
            try:
                with open(input_path, 'r', encoding='utf-8') as f:
                    dag_data = json.load(f)
            finally:
                sampling_timings["counted_dag_load_seconds"] += max(
                    0.0,
                    time.monotonic() - load_started,
                )
            decode_started = time.monotonic()
            try:
                decode_counts_in_place(dag_data, deadline=deadline)
            finally:
                sampling_timings["counted_dag_decode_seconds"] += max(
                    0.0,
                    time.monotonic() - decode_started,
                )

            plain_output_name = file_name.replace(
                "counted_dag_",
                "output_sql_",
            ).replace(".json", ".txt")

            plain_output_name = _depth_scoped_predicate_name(
                plain_output_name,
                max_depth,
                output_label=output_label,
            )
            output_name = (
                f"{plain_output_name}.gz"
                if compress_predicates
                else plain_output_name
            )
            output_path = os.path.join(output_dir, output_name)
            partial_path = output_path + ".partial"

            if deadline is not None and (enumerate_all or localized_random):
                from .stratified_sampling import sample_stratified_file
                iter_file_started = True
                files_left = len(json_files) - sorted(json_files).index(input_path)
                file_deadline = time.monotonic() + max(0.0, deadline - time.monotonic()) / files_left
                stratified = sample_stratified_file(
                    sampler, dag_data, output_dir=output_dir, output_name=output_name,
                    deadline=file_deadline, compressed=compress_predicates,
                    compression_level=predicate_compression_level,
                    buffer_bytes=output_buffer_bytes,
                    progress_callback=(
                        (lambda count: progress_callback(total_generated + count))
                        if progress_callback is not None else None
                    ),
                )
                for item in stratified["outputs"]:
                    path = item["path"]
                    name = os.path.basename(path)
                    output_files.append(path)
                    output_text_bytes[name] = item["text_bytes"]
                    output_text_characters[name] = item["text_characters"]
                    signature_spans_by_file[name] = item["signature_spans"]
                    stratified_layers_by_file[name] = item["layer"]
                    total_generated += item["generated"]
                    total_text_bytes += item["text_bytes"]
                    total_text_characters += item["text_characters"]
                    if localized_random:
                        random_sampling_by_file[name] = item["layer"]["random_sampling"]
                signature_catalog.update(stratified["signature_catalog"])
                round_complete = round_complete and stratified["round_complete"]
                if not stratified["round_complete"]:
                    stop_metadata["stop_reason"] = "deadline"
                continue

            file_generated = 0
            file_text_bytes = 0
            file_text_characters = 0

            def should_stop_for_downstream(generated_in_file):
                nonlocal reserve_status
                generated_so_far = total_generated + generated_in_file
                text_bytes_so_far = total_text_bytes + file_text_bytes
                estimate = estimate_downstream_reserve(
                    downstream_cost_model,
                    predicate_count=generated_so_far,
                    predicate_text_bytes=text_bytes_so_far,
                    minimum_seconds=reserve_min_seconds,
                    safety_factor=reserve_safety_factor,
                    fixed_seconds=reserve_fixed_seconds,
                )
                seconds_left = remaining_seconds(deadline)
                reserve_status = {
                    **reserve_status,
                    **estimate,
                    "remaining_seconds": seconds_left,
                    "predicate_count": generated_so_far,
                    "predicate_text_bytes": text_bytes_so_far,
                }
                if (
                    seconds_left is not None
                    and seconds_left <= estimate["reserve_seconds"]
                ):
                    reserve_status["triggered"] = True
                    print(
                        "[Stage3] Sampling stopped to preserve downstream time: "
                        f"remaining={seconds_left:.2f}s, "
                        f"reserved={estimate['reserve_seconds']:.2f}s, "
                        f"predicates={generated_so_far}, "
                        f"text_bytes={text_bytes_so_far}"
                    )
                    return {"reason": "downstream_reserve"}
                return None

            with _open_predicate_writer(
                partial_path,
                compressed=compress_predicates,
                compression_level=predicate_compression_level,
                buffer_bytes=output_buffer_bytes,
            ) as output_file:
                if round_complete:
                    iter_file_started = True
                    for sql in sampler.iter_file(
                        dag_data,
                        deadline=deadline,
                        stop_check=(
                            should_stop_for_downstream if reserve_enabled else None
                        ),
                        stop_check_interval=reserve_check_interval,
                        progress_callback=(
                            (
                                lambda generated_in_file: progress_callback(
                                    total_generated + generated_in_file
                                )
                            )
                            if progress_callback is not None
                            else None
                        ),
                    ):
                        line = sql + "\n"
                        encoded_line = line.encode("utf-8")
                        write_started = time.monotonic()
                        try:
                            output_file.write(encoded_line)
                        finally:
                            sampling_timings[
                                "predicate_output_write_seconds"
                            ] += max(
                                0.0,
                                time.monotonic() - write_started,
                            )
                            sampling_timings[
                                "predicate_output_write_calls"
                            ] += 1
                        file_generated += 1
                        file_text_bytes += len(encoded_line)
                        file_text_characters += len(line)

                    file_metadata = sampler.last_sample_metadata
                    signature_spans_by_file[output_name] = list(
                        file_metadata.get("signature_spans", []) or []
                    )
                    signature_catalog.update(
                        file_metadata.get("signature_catalog", {}) or {}
                    )
                    if localized_random:
                        random_sampling_by_file[output_name] = dict(
                            file_metadata.get("random_sampling", {}) or {}
                        )
                    if not file_metadata["round_complete"]:
                        round_complete = False
                        stop_metadata = {
                            "stop_root_id": file_metadata.get("stop_root_id"),
                            "stop_index": file_metadata.get("stop_index"),
                            "stop_step": file_metadata.get("stop_step"),
                            "stop_reason": file_metadata.get("stop_reason"),
                        }

            os.replace(partial_path, output_path)
            output_files.append(output_path)
            output_text_bytes[output_name] = file_text_bytes
            output_text_characters[output_name] = file_text_characters
            file_size = os.path.getsize(output_path)
            elapsed = max(time.monotonic() - sampling_start, 1e-9)
            storage_ratio = file_size / max(file_text_bytes, 1)
            print(f"Generated {file_generated} SQLs from {file_name}.")
            total_generated += file_generated
            total_text_bytes += file_text_bytes
            total_text_characters += file_text_characters

        except Stage3DeadlineReached as e:
            round_complete = False
            error_message = None
            print(f"[Stage3] Deadline reached during sampling: {e.phase}")
            break
        except Exception as e:
            print(f"Error sampling {file_name}: {e}")
            error_message = str(e)
            round_complete = False
            break
        finally:
            if iter_file_started:
                _merge_sampling_timings(
                    sampling_timings,
                    sampler.last_sampling_timings,
                )

    if reserve_enabled:
        reserve_status.update(
            estimate_downstream_reserve(
                downstream_cost_model,
                predicate_count=total_generated,
                predicate_text_bytes=total_text_bytes,
                minimum_seconds=reserve_min_seconds,
                safety_factor=reserve_safety_factor,
                fixed_seconds=reserve_fixed_seconds,
            )
        )
        reserve_status["remaining_seconds"] = remaining_seconds(deadline)
        reserve_status["predicate_count"] = total_generated
        reserve_status["predicate_text_bytes"] = total_text_bytes
    sampling_elapsed = max(0.0, time.monotonic() - sampling_start)
    sampling_timings["graph_preparation_seconds"] = sum(
        float(sampling_timings.get(name, 0.0) or 0.0)
        for name in (
            "counted_dag_load_seconds",
            "counted_dag_decode_seconds",
            "successor_ordering_seconds",
            "decode_cache_preparation_seconds",
        )
    )
    accounted_seconds = sum(
        float(sampling_timings.get(name, 0.0) or 0.0)
        for name in (
            "sampler_setup_seconds",
            "graph_preparation_seconds",
            "predicate_traversal_render_seconds",
            "predicate_output_write_seconds",
        )
    )

    sampling_timings["sampling_overhead_seconds"] = max(
        0.0,
        sampling_elapsed - accounted_seconds,
    )
    report = {
        "success": error_message is None,
        "round_complete": round_complete,
        "phase": "sampling",
        "error": error_message,
        "total_generated": total_generated,
        "output_text_bytes": total_text_bytes,
        "output_text_characters": total_text_characters,
        "output_text_bytes_by_file": output_text_bytes,
        "output_text_characters_by_file": output_text_characters,
        "sampling_elapsed_seconds": sampling_elapsed,
        "output_files": output_files,
        "max_depth": max_depth,
        "sampling_timings": sampling_timings,
        "sampling_mode": (
            "localized_random"
            if localized_random
            else ("enumeration" if enumerate_all else "stepwise")
        ),
        "predicate_storage": {
            "format": "gzip" if compress_predicates else "plain_text",
            "compression_level": (
                predicate_compression_level if compress_predicates else None
            ),
            "lossless_complete_output": True,
        },
        "downstream_reserve": reserve_status,
        "signature_spans_by_file": signature_spans_by_file,
        "signature_catalog": signature_catalog,
        **({"stratified_layers_by_file": stratified_layers_by_file}
           if stratified_layers_by_file else {}),
        **(
            {"random_sampling_by_file": random_sampling_by_file}
            if localized_random
            else {}
        ),
        **stop_metadata,
    }
    status_report = dict(report)
    status_report.pop("signature_spans_by_file", None)
    status_report.pop("signature_catalog", None)
    status_report["signature_span_count"] = sum(
        len(spans) for spans in signature_spans_by_file.values()
    )
    status_report["signature_catalog_size"] = len(signature_catalog)
    status_report["stop_index"] = encode_count(status_report.get("stop_index"))
    status_report["stop_step"] = encode_count(status_report.get("stop_step"))
    status_report["output_files"] = [os.path.abspath(path) for path in output_files]
    status_report["output_sizes_bytes"] = {
        os.path.basename(path): os.path.getsize(path)
        for path in output_files
        if os.path.exists(path)
    }
    try:
        write_status_json(
            status_report,
            os.path.join(output_dir, "stage3_sampling_status.json"),
        )
    except Exception as exc:
        print(f"[Stage3] Warning: failed to write sampling status: {exc}")
    return report


if __name__ == "__main__":
    sampler()
