from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

try:
    from RL.candidate_shared_state import advance_completion_variant, completion_retry_needed
except ModuleNotFoundError:
    from candidate_shared_state import advance_completion_variant, completion_retry_needed

try:
    from RL.config import (
        CANDIDATE_BITMAP_CONFIG,
        CANDIDATE_BITMAP_DIR,
        CANDIDATE_BITMAP_NPZ_PATH,
        CANDIDATE_BITMAP_REPORT_PATH,
        CANDIDATE_EDGE_INDEX_PATH,
        CANDIDATE_EDGE_SCORES_PATH,
        CANDIDATE_OBSERVED_REPORT_PATH,
        CANDIDATE_MISSING_EDGES_PATH,
        CANDIDATE_PROBE_GRAPH_PATH,
        CANDIDATE_PROBE_REPORT_PATH,
        CANDIDATE_PROBE_PREDICATE_PATH,
        CANDIDATE_PROBE_PREDICATE_REPORT_PATH,
        CANDIDATE_PROBE_COMPLETED_SQL_PATH,
        CANDIDATE_PROBE_VALIDATION_REPORT_PATH,
        CANDIDATE_BITMAP_UPDATE_LOG_PATH,
        CANDIDATE_BITMAP_UPDATE_REPORT_PATH,
        DB_CONFIG,
        GLOBAL_KNOWLEDGE_GRAPH_PATH,
        STAGE2_MODE,
    )
except Exception:
    STAGE2_MODE = None
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    OUTPUT_DIR = PROJECT_ROOT / "output"
    CANDIDATE_BITMAP_DIR = OUTPUT_DIR / "candidate_bitmap"
    CANDIDATE_EDGE_INDEX_PATH = CANDIDATE_BITMAP_DIR / "candidate_edge_index.json"
    CANDIDATE_BITMAP_NPZ_PATH = CANDIDATE_BITMAP_DIR / "candidate_bitmaps.npz"
    CANDIDATE_EDGE_SCORES_PATH = CANDIDATE_BITMAP_DIR / "candidate_edge_scores.jsonl"
    CANDIDATE_BITMAP_REPORT_PATH = CANDIDATE_BITMAP_DIR / "candidate_bitmap_report.json"
    CANDIDATE_OBSERVED_REPORT_PATH = CANDIDATE_BITMAP_DIR / "candidate_observed_report.json"
    CANDIDATE_MISSING_EDGES_PATH = CANDIDATE_BITMAP_DIR / "candidate_missing_edges.jsonl"
    CANDIDATE_PROBE_GRAPH_PATH = CANDIDATE_BITMAP_DIR / "probe_graph.json"
    CANDIDATE_PROBE_REPORT_PATH = CANDIDATE_BITMAP_DIR / "probe_graph_report.json"
    CANDIDATE_PROBE_PREDICATE_PATH = CANDIDATE_BITMAP_DIR / "probe_predicate.txt"
    CANDIDATE_PROBE_PREDICATE_REPORT_PATH = CANDIDATE_BITMAP_DIR / "probe_predicate_report.json"
    CANDIDATE_PROBE_COMPLETED_SQL_PATH = CANDIDATE_BITMAP_DIR / "probe_completed_sql.txt"
    CANDIDATE_PROBE_VALIDATION_REPORT_PATH = CANDIDATE_BITMAP_DIR / "probe_validation_report.json"
    CANDIDATE_BITMAP_UPDATE_LOG_PATH = CANDIDATE_BITMAP_DIR / "candidate_bitmap_update_log.jsonl"
    CANDIDATE_BITMAP_UPDATE_REPORT_PATH = CANDIDATE_BITMAP_DIR / "candidate_bitmap_update_report.json"
    GLOBAL_KNOWLEDGE_GRAPH_PATH = PROJECT_ROOT / "pkg" / "knowledge_graph" / "global_knowledge_graph.json"

    class _FallbackCandidateBitmapConfig:
        enabled = True
        rules_path = PROJECT_ROOT / "pkg" / "configuration" / "postgresql_rules.json"
        bitmap_dir = CANDIDATE_BITMAP_DIR
        edge_index_path = CANDIDATE_EDGE_INDEX_PATH
        bitmap_npz_path = CANDIDATE_BITMAP_NPZ_PATH
        edge_scores_path = CANDIDATE_EDGE_SCORES_PATH
        report_path = CANDIDATE_BITMAP_REPORT_PATH
        observed_report_path = CANDIDATE_OBSERVED_REPORT_PATH
        missing_edges_path = CANDIDATE_MISSING_EDGES_PATH
        probe_graph_path = CANDIDATE_PROBE_GRAPH_PATH
        probe_report_path = CANDIDATE_PROBE_REPORT_PATH
        probe_predicate_path = CANDIDATE_PROBE_PREDICATE_PATH
        probe_predicate_report_path = CANDIDATE_PROBE_PREDICATE_REPORT_PATH
        probe_completed_sql_path = CANDIDATE_PROBE_COMPLETED_SQL_PATH
        probe_validation_report_path = CANDIDATE_PROBE_VALIDATION_REPORT_PATH
        bitmap_update_log_path = CANDIDATE_BITMAP_UPDATE_LOG_PATH
        bitmap_update_report_path = CANDIDATE_BITMAP_UPDATE_REPORT_PATH
        probe_predicate_max_depth = 3
        global_knowledge_graph_path = GLOBAL_KNOWLEDGE_GRAPH_PATH
        open_strict_initially = True
        open_weak_initially = False
        open_try_initially = False
        include_unknown_as_try = False
        expand_valid_type_terminals = False
        weak_fail_count_to_disable = 3
        try_fail_count_to_disable = 2
        strict_fail_count_to_disable = 5
        target_db_failure_signature_threshold = 10
        candidate_bitmap_probe_variant_enabled = True
        candidate_bitmap_probe_variant_max_attempts = 20
        strict_base_score = 1.0
        weak_base_score = 0.45
        try_base_score = 0.15
        candidate_representation_guidance_enabled = True
        pattern_transfer_score_weight = 0.30
        refresh_pattern_transfer_before_rl = True
        pattern_transfer_refresh_steps = 20000
        pattern_transfer_refresh_min_observed_added = 128
        pattern_transfer_top_k_similar_outer = 8
        pattern_transfer_min_outer_sim = 0.45
        pattern_transfer_max_outer_hop = 1
        pattern_transfer_include_self_outer = True
        pattern_transfer_exact_inner_match_score = 1.0
        pattern_transfer_same_return_type_match_score = 0.65
        pattern_transfer_same_terminal_kind_match_score = 0.35
        pattern_transfer_inner_operator_similarity_weight = 0.50
        write_human_readable_report = True

    CANDIDATE_BITMAP_CONFIG = _FallbackCandidateBitmapConfig()

try:
    from RL.operator_similarity import load_operator_similarity_index, normalize_operator_key
except Exception:
    load_operator_similarity_index = None

    def normalize_operator_key(op: Any) -> str:
        text = str(op).strip()
        if "::" in text:
            text = text.split("::", 1)[0].strip()
        if text.endswith("()"):
            return text[:-2].upper() + "()"
        if text.replace("_", "").replace(" ", "").isalnum():
            return " ".join(text.split()).upper()
        return text


@dataclass(frozen=True)
class TokenSignature:
    token: str

    return_type: str

    coarse_return_type: str

    arg_types: Tuple[str, ...]

    is_terminal: bool


@dataclass(frozen=True)
class CandidateEdge:
    edge_index: int
    edge_id: str
    outer_token: str
    outer_node_id: str
    outer_return_type: str
    slot_index: int
    slot_required_type: str
    inner_token: str
    inner_node_id: str
    inner_return_type: str
    candidate_level: str
    pattern_transfer_score: float
    edge_score: float
    outer_group: str = "other"
    inner_group: str = "other"
    edge_group: str = "other"

    def to_dict(self) -> Dict[str, Any]:
        is_core_to_extension = self.edge_group == "core_to_extension"
        is_core_internal = self.edge_group == "core_internal"
        is_extension_outer = self.edge_group == "extension_outer"

        return {
            "edge_index": self.edge_index,
            "edge_id": self.edge_id,
            "outer_token": self.outer_token,
            "outer_node_id": self.outer_node_id,
            "outer_return_type": self.outer_return_type,
            "slot_index": self.slot_index,
            "slot_required_type": self.slot_required_type,
            "inner_token": self.inner_token,
            "inner_node_id": self.inner_node_id,
            "inner_return_type": self.inner_return_type,
            "candidate_level": self.candidate_level,
            "pattern_transfer_score": self.pattern_transfer_score,
            "edge_score": self.edge_score,
            "outer_group": self.outer_group,
            "inner_group": self.inner_group,
            "edge_group": self.edge_group,
            "is_core_to_extension": bool(is_core_to_extension),
            "is_core_internal": bool(is_core_internal),
            "is_extension_outer": bool(is_extension_outer),
            "success_count": 0,
            "fail_count": 0,
            "new_edge_count": 0,
        }


def _load_json(path: Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f'JSON root must be an object: {path}')
    return data


def _candidate_rules_preflight_report_path() -> Path:
    return CANDIDATE_BITMAP_DIR / "candidate_rules_preflight_report.json"


def _candidate_safe_closure_smoke_report_path() -> Path:
    return CANDIDATE_BITMAP_DIR / "candidate_safe_closure_smoke_report.json"


def _as_token_list(value: Any) -> List[str]:
    if not isinstance(value, list):
        return []
    return [str(x).strip() for x in value if str(x).strip()]


def _normalized_definition_tokens(
    rules: Dict[str, Any],
    exclude_prefixes: Iterable[str],
) -> set:
    definitions = rules.get("definitions", {}) or {}
    if not isinstance(definitions, dict):
        return set()

    tokens = set()
    for raw_token in definitions.keys():
        token = str(raw_token).strip()
        if not token:
            continue
        if _is_comment_definition_key(token, exclude_prefixes):
            continue
        tokens.add(normalize_operator_key(token))
    return tokens


def _normalized_terminal_tokens(rules: Dict[str, Any]) -> set:
    bitmap_cfg = rules.get("candidate_bitmap", {}) or {}
    if not isinstance(bitmap_cfg, dict):
        return set()

    terminal_tokens = bitmap_cfg.get("terminal_tokens", {}) or {}
    if not isinstance(terminal_tokens, dict):
        return set()

    return {
        normalize_operator_key(str(token).strip())
        for token in terminal_tokens.keys()
        if str(token).strip()
    }


def _token_exists_for_rules_group(
    token: str,
    definition_tokens: set,
    terminal_tokens: set,
) -> bool:
    norm = normalize_operator_key(token)
    return norm in definition_tokens or norm in terminal_tokens


def _looks_like_forbidden_placeholder(token: str) -> bool:
    text = str(token).strip().upper()
    return (
        text == "??"
        or "HOLE" in text
        or text.startswith("VALUE_SINGULAR")
    )


def _scan_initial_predicate_files_for_rule_issues(
    rules_path: Path,
    excluded_tokens: List[str],
) -> Dict[str, Any]:
    report = {
        "checked_files": [],
        "missing_files": [],
        "fatal_hits": [],
        "warning_hits": [],
        "syntax_sugar_hits": [],
    }

    try:
        project_root = Path(rules_path).resolve().parents[2]
    except Exception:
        project_root = Path.cwd()

    try:
        from RL.config import get_database_data_path
    except ModuleNotFoundError:
        from config import get_database_data_path

    collection_name = get_database_data_path("Correct_Collection.txt").name
    candidate_paths = [
        project_root / "data" / collection_name,
        project_root / collection_name,
    ]

    excluded_scan_tokens = [
        token for token in excluded_tokens
        if token and not token.startswith("//")
    ]

    fatal_patterns = ["??", "HOLE", "HOLE_T_CMP", "HOLE_T_STR", "HOLE_T_BOOL"]
    syntax_sugar_tokens = {
        "IS NULL",
        "IS NOT NULL",
        "IS TRUE",
        "IS FALSE",
    }

    for path in candidate_paths:
        if not path.exists():
            report["missing_files"].append(str(path))
            continue

        report["checked_files"].append(str(path))
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except Exception as exc:
            report["warning_hits"].append({
                "file": str(path),
                "token": "<READ_ERROR>",
                "message": str(exc),
            })
            continue

        upper_text = text.upper()

        for pattern in fatal_patterns:
            if pattern.upper() in upper_text:
                report["fatal_hits"].append({
                    "file": str(path),
                    "token": pattern,
                    "message": 'The initial predicate collection contains deprecated placeholders or HOLE nodes.',
                })

        for token in excluded_scan_tokens:
            token_text = str(token).strip()
            if not token_text:
                continue

            if _looks_like_forbidden_placeholder(token_text):
                continue

            escaped = re.escape(token_text)
            if re.search(r"[A-Za-z0-9_]", token_text):
                pattern = re.compile(r"(?<![A-Za-z0-9_])" + escaped + r"(?![A-Za-z0-9_])", re.IGNORECASE)
                hit = bool(pattern.search(text))
            else:
                hit = token_text in text

            if hit:
                if token_text.upper() in syntax_sugar_tokens:
                    report["syntax_sugar_hits"].append({
                        "file": str(path),
                        "token": token_text,
                        "message": 'This token is syntactic sugar expressible with IS + NULL/TRUE/FALSE; recorded only, not treated as a warning.',
                    })
                else:
                    report["warning_hits"].append({
                        "file": str(path),
                        "token": token_text,
                        "message": 'The initial predicate collection contains tokens in excluded_tokens; parser/type_inference may not support them under the reduced rules.',
                    })

    return report


def validate_candidate_rules_preflight(
    rules: Dict[str, Any],
    rules_path: Optional[Path] = None,
    *,
    write_report: bool = True,
    raise_on_error: bool = True,
) -> Dict[str, Any]:
    bitmap_cfg = rules.get("candidate_bitmap", {}) or {}
    if not isinstance(bitmap_cfg, dict):
        bitmap_cfg = {}

    definitions = rules.get("definitions", {}) or {}
    terminal_tokens = bitmap_cfg.get("terminal_tokens", {}) or {}
    aliases = bitmap_cfg.get("coarse_type_aliases", {}) or {}
    exclude_prefixes = bitmap_cfg.get("exclude_definition_prefixes", ["//"]) or ["//"]

    errors: List[Dict[str, Any]] = []
    warnings: List[Dict[str, Any]] = []
    notes: List[Dict[str, Any]] = []

    definition_tokens = _normalized_definition_tokens(rules, exclude_prefixes)
    terminal_token_set = _normalized_terminal_tokens(rules)

    if not isinstance(definitions, dict):
        errors.append({
            "kind": "definitions_not_object",
            "message": 'rules.definitions must be an object.',
        })
        definitions = {}

    for raw_token, raw_rule in definitions.items():
        token = str(raw_token).strip()
        if not token:
            errors.append({
                "kind": "empty_definition_token",
                "token": raw_token,
                "message": 'definitions contains an empty token.',
            })
            continue

        if _is_comment_definition_key(token, exclude_prefixes):
            continue

        if not isinstance(raw_rule, list) or len(raw_rule) < 2:
            errors.append({
                "kind": "invalid_definition_signature",
                "token": token,
                "rule": raw_rule,
                "message": 'Each definition must have the form [return_type, [arg_types...]].',
            })
            continue

        if not isinstance(raw_rule[1], list):
            errors.append({
                "kind": "invalid_definition_args",
                "token": token,
                "rule": raw_rule,
                "message": 'The second item in a definition must be an argument-type list.',
            })

    if not isinstance(terminal_tokens, dict):
        errors.append({
            "kind": "terminal_tokens_not_object",
            "message": 'candidate_bitmap.terminal_tokens must be an object.',
        })
        terminal_tokens = {}

    bad_terminal_tokens = []
    for raw_token in terminal_tokens.keys():
        token = str(raw_token).strip()
        if _looks_like_forbidden_placeholder(token):
            bad_terminal_tokens.append(token)

    if bad_terminal_tokens:
        errors.append({
            "kind": "bad_terminal_tokens",
            "tokens": sorted(bad_terminal_tokens),
            "message": 'HOLE_* and VALUE_SINGULAR_* tokens are not allowed in terminal_tokens.',
        })

    group_names = [
        "core_scored_tokens",
        "extension_scored_tokens",
        "exploratory_tokens",
    ]

    group_missing = {}
    for group_name in group_names:
        group_tokens = _as_token_list(bitmap_cfg.get(group_name, []))
        if not group_tokens:
            if group_name == "exploratory_tokens":
                continue
            warnings.append({
                "kind": "empty_group",
                "group": group_name,
                "message": f'candidate_bitmap.{group_name} is empty or missing.',
            })
            continue

        missing = [
            token for token in group_tokens
            if not _token_exists_for_rules_group(token, definition_tokens, terminal_token_set)
        ]
        if missing:
            group_missing[group_name] = missing
            errors.append({
                "kind": "group_token_missing",
                "group": group_name,
                "tokens": missing,
                "message": f'{group_name} contains tokens not defined in definitions / terminal_tokens.',
            })

    excluded_tokens = _as_token_list(bitmap_cfg.get("excluded_tokens", []))
    excluded_present = []
    for token in excluded_tokens:
        norm = normalize_operator_key(token)
        if norm in definition_tokens or norm in terminal_token_set:
            excluded_present.append(token)

    if excluded_present:
        errors.append({
            "kind": "excluded_token_is_active",
            "tokens": sorted(excluded_present),
            "message": 'Tokens in excluded_tokens should not also appear in definitions or terminal_tokens.',
        })

    precedence = rules.get("precedence", {}) or {}
    if isinstance(precedence, dict):
        precedence_unknown = []
        for token in precedence.keys():
            norm = normalize_operator_key(str(token).strip())
            if norm not in definition_tokens and norm not in terminal_token_set:
                precedence_unknown.append(str(token).strip())
        if precedence_unknown:
            warnings.append({
                "kind": "precedence_token_not_in_definitions",
                "tokens": sorted(set(precedence_unknown)),
                "message": 'precedence contains tokens not defined in definitions / terminal_tokens; warning only.',
            })

    unary_operators = rules.get("unary_operators", []) or []
    unary_unknown = []
    for token in unary_operators:
        norm = normalize_operator_key(str(token).strip())
        if norm not in definition_tokens and norm not in terminal_token_set:
            unary_unknown.append(str(token).strip())
    if unary_unknown:
        warnings.append({
            "kind": "unary_token_not_in_definitions",
            "tokens": sorted(set(unary_unknown)),
            "message": 'unary_operators contains tokens not defined in definitions / terminal_tokens; warning only.',
        })

    terminal_null_type = str(terminal_tokens.get("NULL", "")).strip().upper()
    alias_null_type = str(aliases.get("NULL", "")).strip().upper()
    if terminal_null_type and alias_null_type and terminal_null_type != alias_null_type:
        notes.append({
            "kind": "null_type_alias_mismatch",
            "terminal_tokens.NULL": terminal_null_type,
            "coarse_type_aliases.NULL": alias_null_type,
            "message": 'NULL has different types in terminal_tokens and coarse_type_aliases; note only.',
        })

    predicate_scan = {}
    if rules_path is not None:
        predicate_scan = _scan_initial_predicate_files_for_rule_issues(Path(rules_path), excluded_tokens)
        for hit in predicate_scan.get("fatal_hits", []):
            errors.append({
                "kind": "initial_predicate_forbidden_placeholder",
                **hit,
            })
        if predicate_scan.get("warning_hits"):
            warnings.append({
                "kind": "initial_predicate_excluded_token_hits",
                "hits": predicate_scan.get("warning_hits", [])[:30],
                "message": 'The initial predicate collection contains tokens in excluded_tokens; check whether these predicates still belong to the current stage.',
            })

    report = {
        "version": "stage20_3_candidate_rules_preflight_v1",
        "rules_path": str(rules_path) if rules_path is not None else "",
        "ok": len(errors) == 0,
        "summary": {
            "definitions_count": len(definition_tokens),
            "terminal_tokens_count": len(terminal_token_set),
            "core_scored_tokens_count": len(_as_token_list(bitmap_cfg.get("core_scored_tokens", []))),
            "extension_scored_tokens_count": len(_as_token_list(bitmap_cfg.get("extension_scored_tokens", []))),
            "excluded_tokens_count": len(excluded_tokens),
            "error_count": len(errors),
            "warning_count": len(warnings),
            "note_count": len(notes),
        },
        "errors": errors,
        "warnings": warnings,
        "notes": notes,
        "checks": {
            "definition_tokens": sorted(definition_tokens),
            "terminal_tokens": sorted(terminal_token_set),
            "group_missing": group_missing,
            "excluded_present": sorted(excluded_present),
            "bad_terminal_tokens": sorted(bad_terminal_tokens),
            "predicate_scan": predicate_scan,
        },
    }

    if write_report:
        report_path = _candidate_rules_preflight_report_path()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with report_path.open("w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

    if errors and raise_on_error:
        detail = json.dumps(report["summary"], ensure_ascii=False)
        raise ValueError(f'[candidate_rules_preflight] Database rules JSON validation failed: {detail}')

    return report
def _normalize_type(raw_type: Any, aliases: Dict[str, str]) -> str:
    if raw_type is None:
        return "T_ANY"

    text = str(raw_type).strip().upper()
    if not text:
        return "T_ANY"

    return str(aliases.get(text, text)).upper()


def _node_id(token: str, return_type: str) -> str:
    return f"{token}::{return_type}"


def _is_comment_definition_key(key: str, exclude_prefixes: Iterable[str]) -> bool:
    text = str(key)
    return any(text.startswith(prefix) for prefix in exclude_prefixes)


def _infer_terminal_type(token: str, terminal_tokens: Dict[str, str]) -> str:
    text = str(token).strip().upper()

    if text in terminal_tokens:
        return str(terminal_tokens[text]).strip().upper()

    if text.startswith("COL_"):
        return text.split("_", 1)[1].strip().upper()

    if text.startswith("VALUE_"):
        parts = text.split("_")
        if parts:
            return parts[-1].strip().upper()

    if text in {"TRUE", "FALSE"}:
        return "T_BOOL"

    if text == "NULL":
        return "T_CMP"

    return "T_ANY"


def _is_sqlserver() -> bool:
    return str(getattr(globals().get("DB_CONFIG"), "db_type", "") or "").strip().lower() == "sqlserver"


def _build_terminal_tokens(rules: Dict[str, Any]) -> Dict[str, str]:
    bitmap_cfg = rules.get("candidate_bitmap", {}) or {}
    terminal_tokens = {
        str(k).strip().upper(): str(v).strip().upper()
        for k, v in (bitmap_cfg.get("terminal_tokens", {}) or {}).items()
    }

    expand_valid_type_terminals = bool(
        getattr(
            CANDIDATE_BITMAP_CONFIG,
            "expand_valid_type_terminals",
            bitmap_cfg.get("expand_valid_type_terminals", False),
        )
    )

    if _is_sqlserver():
        expand_valid_type_terminals = bool(bitmap_cfg.get("expand_valid_type_terminals", False))

    if expand_valid_type_terminals:
        valid_types = rules.get("valid_types", []) or []
        for raw_type in valid_types:
            type_name = str(raw_type).strip().upper()
            if not type_name:
                continue

            terminal_tokens.setdefault(f"COL_{type_name}", type_name)
            terminal_tokens.setdefault(f"VALUE_TYPICAL_{type_name}", type_name)

    if not _is_sqlserver():
        terminal_tokens.setdefault("NULL", "T_CMP")
        terminal_tokens.setdefault("TRUE", "T_BOOL")
        terminal_tokens.setdefault("FALSE", "T_BOOL")

    return terminal_tokens


def collect_token_signatures(rules: Dict[str, Any]) -> Dict[str, TokenSignature]:
    bitmap_cfg = rules.get("candidate_bitmap", {}) or {}
    aliases = bitmap_cfg.get("coarse_type_aliases", {}) or {}
    exclude_prefixes = bitmap_cfg.get("exclude_definition_prefixes", ["//"]) or ["//"]

    definitions = rules.get("definitions", {}) or {}
    signatures: Dict[str, TokenSignature] = {}

    for raw_token, raw_rule in definitions.items():
        token = str(raw_token).strip()
        if not token:
            continue
        if _is_comment_definition_key(token, exclude_prefixes):
            continue
        if not isinstance(raw_rule, list) or len(raw_rule) < 2:
            continue

        return_type = _normalize_type(raw_rule[0], aliases)
        arg_types = tuple(_normalize_type(x, aliases) for x in (raw_rule[1] or []))

        norm_token = normalize_operator_key(token)
        signatures[norm_token] = TokenSignature(
            token=norm_token,
            return_type=return_type,
            coarse_return_type=return_type,
            arg_types=arg_types,
            is_terminal=False,
        )

    terminal_tokens = _build_terminal_tokens(rules)
    for token, return_type in terminal_tokens.items():
        graph_return_type = _infer_terminal_type(token, terminal_tokens)
        signatures[token] = TokenSignature(
            token=token,
            return_type=graph_return_type,
            coarse_return_type=_normalize_type(graph_return_type, aliases),
            arg_types=tuple(),
            is_terminal=True,
        )

    return dict(sorted(signatures.items(), key=lambda item: item[0]))


def _representation_guidance_enabled() -> bool:
    return bool(
        getattr(
            CANDIDATE_BITMAP_CONFIG,
            "candidate_representation_guidance_enabled",
            True,
        )
    )


def _load_similarity_index():
    if not _representation_guidance_enabled():
        return None
    if load_operator_similarity_index is None:
        return None

    try:
        return load_operator_similarity_index(allow_missing=True)
    except Exception:
        return None


def _compatibility_level(
    required_type: str,
    actual_type: str,
    rules: Dict[str, Any],
    *,
    include_unknown_as_try: bool,
) -> Optional[str]:
    bitmap_cfg = rules.get("candidate_bitmap", {}) or {}
    compatibility = bitmap_cfg.get("compatibility", {}) or {}

    aliases = bitmap_cfg.get("coarse_type_aliases", {}) or {}

    required = _normalize_type(required_type, aliases)
    actual = _normalize_type(actual_type, aliases)

    for level in ("strict", "weak", "try"):
        table = compatibility.get(level, {}) or {}
        allowed = {str(x).strip().upper() for x in table.get(required, [])}
        if actual in allowed:
            return level

    if required == actual:
        return "strict"

    if required == "T_ANY" or actual == "T_ANY":
        return "strict"

    if include_unknown_as_try:
        return "try"

    return None


def _base_score_for_level(level: str) -> float:
    if level == "strict":
        return float(getattr(CANDIDATE_BITMAP_CONFIG, "strict_base_score", 1.0))
    if level == "weak":
        return float(getattr(CANDIDATE_BITMAP_CONFIG, "weak_base_score", 0.45))
    return float(getattr(CANDIDATE_BITMAP_CONFIG, "try_base_score", 0.15))


def _edge_score_from_pattern_transfer(level: str, pattern_transfer_score: float) -> float:
    transfer_weight = float(
        getattr(CANDIDATE_BITMAP_CONFIG, "pattern_transfer_score_weight", 0.30)
    )
    transfer_score = (
        max(0.0, min(1.0, float(pattern_transfer_score or 0.0)))
        if _representation_guidance_enabled()
        else 0.0
    )
    return float(_base_score_for_level(level) + transfer_weight * transfer_score)


def _pattern_transfer_clamp01(value: Any) -> float:
    try:
        score = float(value)
    except Exception:
        return 0.0

    if score < 0.0:
        return 0.0
    if score > 1.0:
        return 1.0
    return score


def _split_graph_node_id(node_id: Any) -> Tuple[str, str]:
    text = str(node_id or "").strip()
    if "::" in text:
        token, return_type = text.rsplit("::", 1)
        return token.strip(), return_type.strip()
    return text, ""


def _pattern_node_kind(token: Any, *, is_terminal: bool) -> str:
    token_text = str(token or "").strip()
    token_upper = token_text.upper()

    if token_upper.startswith("COL_"):
        return "column"

    if token_upper.startswith("VALUE_"):
        return "value"

    if token_upper in {"NULL", "TRUE", "FALSE"}:
        return "literal"

    if token_upper.startswith("HOLE_"):
        return "hole"

    if token_text.endswith("()"):
        return "operator"

    if is_terminal:
        return "terminal"

    return "operator"


def _build_pattern_node_info(
    graph_data: Dict[str, Any],
    edge_records: List[Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    nodes = graph_data.get("nodes", {})
    if not isinstance(nodes, dict):
        nodes = {}

    outer_node_ids = {
        str(record.get("outer_node_id", ""))
        for record in edge_records
        if str(record.get("outer_node_id", ""))
    }

    node_info: Dict[str, Dict[str, Any]] = {}

    def put_info(
        node_id: Any,
        *,
        token: Any = None,
        return_type: Any = None,
        is_terminal: Optional[bool] = None,
    ) -> None:
        node_id_text = str(node_id or "").strip()
        if not node_id_text:
            return

        split_token, split_return_type = _split_graph_node_id(node_id_text)
        token_text = str(token if token is not None else split_token).strip()
        return_type_text = str(return_type if return_type is not None else split_return_type).strip()

        if is_terminal is None:
            is_terminal_value = node_id_text not in outer_node_ids
        else:
            is_terminal_value = bool(is_terminal)

        kind = _pattern_node_kind(token_text, is_terminal=is_terminal_value)

        node_info[node_id_text] = {
            "node_id": node_id_text,
            "token": token_text,
            "norm_token": normalize_operator_key(token_text),
            "return_type": return_type_text,
            "is_terminal": bool(is_terminal_value),
            "kind": kind,
        }

    for node_id, payload in nodes.items():
        if not isinstance(payload, dict):
            split_token, split_return_type = _split_graph_node_id(node_id)
            put_info(
                node_id,
                token=split_token,
                return_type=split_return_type,
                is_terminal=str(node_id) not in outer_node_ids,
            )
            continue

        raw_slots = payload.get("slots", [])
        slot_count = len(raw_slots) if isinstance(raw_slots, list) else 0
        is_terminal = bool(payload.get("is_terminal", False)) or slot_count == 0

        split_token, split_return_type = _split_graph_node_id(node_id)
        put_info(
            node_id,
            token=payload.get("token", split_token),
            return_type=payload.get("return_type", split_return_type),
            is_terminal=is_terminal,
        )

    for record in edge_records:
        outer_node_id = str(record.get("outer_node_id", ""))
        if outer_node_id:
            put_info(
                outer_node_id,
                token=record.get("outer_token", _split_graph_node_id(outer_node_id)[0]),
                return_type=record.get("outer_return_type", _split_graph_node_id(outer_node_id)[1]),
                is_terminal=False,
            )

        inner_node_id = str(record.get("inner_node_id", ""))
        if inner_node_id and inner_node_id not in node_info:
            put_info(
                inner_node_id,
                token=record.get("inner_token", _split_graph_node_id(inner_node_id)[0]),
                return_type=record.get("inner_return_type", _split_graph_node_id(inner_node_id)[1]),
                is_terminal=inner_node_id not in outer_node_ids,
            )

    return node_info


def _build_observed_slot_pattern_index(
    graph_data: Dict[str, Any],
    node_info: Dict[str, Dict[str, Any]],
) -> Tuple[Dict[str, Dict[int, set]], Dict[str, set]]:
    nodes = graph_data.get("nodes", {})
    if not isinstance(nodes, dict):
        return {}, {}

    observed_slot_index: Dict[str, Dict[int, set]] = {}
    token_to_outer_node_ids: Dict[str, set] = {}

    for outer_node_id, payload in nodes.items():
        outer_node_id_text = str(outer_node_id)
        if not isinstance(payload, dict):
            continue

        slots = payload.get("slots", [])
        if not isinstance(slots, list) or not slots:
            continue

        outer_info = node_info.get(outer_node_id_text, {})
        outer_token = normalize_operator_key(
            outer_info.get("token") or _split_graph_node_id(outer_node_id_text)[0]
        )
        if outer_token:
            token_to_outer_node_ids.setdefault(outer_token, set()).add(outer_node_id_text)

        slot_map = observed_slot_index.setdefault(outer_node_id_text, {})

        for slot_index, slot_items in enumerate(slots):
            if not isinstance(slot_items, list):
                continue

            for inner_node_id in slot_items:
                inner_node_id_text = str(inner_node_id)
                if not inner_node_id_text:
                    continue
                slot_map.setdefault(int(slot_index), set()).add(inner_node_id_text)

    return observed_slot_index, token_to_outer_node_ids


def _get_similar_outer_tokens(
    outer_token: Any,
    similarity_index: Any,
) -> List[Dict[str, Any]]:
    norm_outer = normalize_operator_key(outer_token)
    if not norm_outer:
        return []

    result: List[Dict[str, Any]] = []

    if bool(getattr(CANDIDATE_BITMAP_CONFIG, "pattern_transfer_include_self_outer", True)):
        result.append({
            "op": norm_outer,
            "sim": 1.0,
            "hop": 0,
            "rank": 0,
            "source": "self",
        })

    if similarity_index is not None:
        try:
            similar_ops = similarity_index.get_similar_ops(
                norm_outer,
                top_k=int(getattr(CANDIDATE_BITMAP_CONFIG, "pattern_transfer_top_k_similar_outer", 8)),
                min_sim=float(getattr(CANDIDATE_BITMAP_CONFIG, "pattern_transfer_min_outer_sim", 0.45)),
                include_self=False,
                exclude_ops=None,
                max_hop=int(getattr(CANDIDATE_BITMAP_CONFIG, "pattern_transfer_max_outer_hop", 1)),
                allow_missing=True,
            )
        except Exception:
            similar_ops = []

        for item in similar_ops:
            cand_op = normalize_operator_key(item.get("op", ""))
            if not cand_op:
                continue

            result.append({
                "op": cand_op,
                "sim": _pattern_transfer_clamp01(item.get("sim", 0.0)),
                "hop": int(item.get("hop", 1)),
                "rank": int(item.get("rank", len(result) + 1)),
                "source": "operator_similarity",
            })

    dedup: Dict[str, Dict[str, Any]] = {}
    for item in result:
        op = normalize_operator_key(item.get("op", ""))
        if not op:
            continue

        old = dedup.get(op)
        if old is None or float(item.get("sim", 0.0)) > float(old.get("sim", 0.0)):
            dedup[op] = item

    final = list(dedup.values())
    final.sort(key=lambda x: (-float(x.get("sim", 0.0)), int(x.get("hop", 0)), int(x.get("rank", 10**9)), str(x.get("op", ""))))
    return final


def _pattern_inner_match_score(
    candidate_inner_id: str,
    observed_inner_id: str,
    *,
    node_info: Dict[str, Dict[str, Any]],
    similarity_index: Any,
) -> float:
    candidate_inner_id = str(candidate_inner_id or "")
    observed_inner_id = str(observed_inner_id or "")

    if not candidate_inner_id or not observed_inner_id:
        return 0.0

    exact_score = float(
        getattr(CANDIDATE_BITMAP_CONFIG, "pattern_transfer_exact_inner_match_score", 1.0)
    )
    if candidate_inner_id == observed_inner_id:
        return _pattern_transfer_clamp01(exact_score)

    candidate_info = node_info.get(candidate_inner_id, {})
    observed_info = node_info.get(observed_inner_id, {})

    candidate_token = candidate_info.get("token") or _split_graph_node_id(candidate_inner_id)[0]
    observed_token = observed_info.get("token") or _split_graph_node_id(observed_inner_id)[0]

    candidate_return_type = str(candidate_info.get("return_type") or _split_graph_node_id(candidate_inner_id)[1]).upper()
    observed_return_type = str(observed_info.get("return_type") or _split_graph_node_id(observed_inner_id)[1]).upper()

    candidate_kind = str(candidate_info.get("kind") or _pattern_node_kind(candidate_token, is_terminal=True))
    observed_kind = str(observed_info.get("kind") or _pattern_node_kind(observed_token, is_terminal=True))

    if candidate_kind == "hole" or observed_kind == "hole":
        return 0.0

    best = 0.0

    if candidate_return_type and observed_return_type and candidate_return_type == observed_return_type:
        best = max(
            best,
            float(getattr(CANDIDATE_BITMAP_CONFIG, "pattern_transfer_same_return_type_match_score", 0.65)),
        )

    non_operator_kinds = {"column", "value", "literal", "terminal", "hole"}
    if candidate_kind == observed_kind and candidate_kind in non_operator_kinds:
        best = max(
            best,
            float(getattr(CANDIDATE_BITMAP_CONFIG, "pattern_transfer_same_terminal_kind_match_score", 0.35)),
        )

    if candidate_kind == "operator" and observed_kind == "operator" and similarity_index is not None:
        try:
            get_inner_similarity = getattr(similarity_index, "get_" + "similarity" + "_score")
            inner_sim = get_inner_similarity(
                candidate_token,
                observed_token,
                default=0.0,
                include_self_score=1.0,
                max_hop=int(getattr(CANDIDATE_BITMAP_CONFIG, "pattern_transfer_max_outer_hop", 1)),
                symmetric=True,
            )
        except Exception:
            inner_sim = 0.0

        best = max(
            best,
            float(getattr(CANDIDATE_BITMAP_CONFIG, "pattern_transfer_inner_operator_similarity_weight", 0.50))
            * _pattern_transfer_clamp01(inner_sim),
        )

    return _pattern_transfer_clamp01(best)


def _compute_pattern_transfer_score_for_record(
    record: Dict[str, Any],
    *,
    observed_slot_index: Dict[str, Dict[int, set]],
    token_to_outer_node_ids: Dict[str, set],
    node_info: Dict[str, Dict[str, Any]],
    similarity_index: Any,

    similar_outer_cache: Optional[Dict[str, List[Dict[str, Any]]]] = None,
    inner_match_cache: Optional[Dict[Tuple[str, str], float]] = None,
) -> float:
    if not _representation_guidance_enabled():
        return 0.0

    try:
        slot_index = int(record.get("slot_index", -1))
    except Exception:
        return 0.0

    if slot_index < 0:
        return 0.0

    outer_token = record.get("outer_token") or _split_graph_node_id(record.get("outer_node_id", ""))[0]
    inner_node_id = str(record.get("inner_node_id", ""))

    if not inner_node_id:
        return 0.0

    best_score = 0.0

    normalized_outer_token = normalize_operator_key(outer_token)
    if similar_outer_cache is not None:
        similar_outer_tokens = similar_outer_cache.get(normalized_outer_token)
        if similar_outer_tokens is None:
            similar_outer_tokens = _get_similar_outer_tokens(outer_token, similarity_index)
            similar_outer_cache[normalized_outer_token] = similar_outer_tokens
    else:
        similar_outer_tokens = _get_similar_outer_tokens(outer_token, similarity_index)

    for outer_item in similar_outer_tokens:
        similar_outer_token = normalize_operator_key(outer_item.get("op", ""))
        outer_sim = _pattern_transfer_clamp01(outer_item.get("sim", 0.0))
        if not similar_outer_token or outer_sim <= 0.0:
            continue

        for observed_outer_node_id in token_to_outer_node_ids.get(similar_outer_token, set()):
            slot_map = observed_slot_index.get(str(observed_outer_node_id), {})
            observed_inners = slot_map.get(slot_index, set())
            if not observed_inners:
                continue

            for observed_inner_node_id in observed_inners:
                observed_inner_node_id = str(observed_inner_node_id)
                cache_key = (inner_node_id, observed_inner_node_id)
                if inner_match_cache is not None and cache_key in inner_match_cache:
                    inner_match = inner_match_cache[cache_key]
                else:
                    inner_match = _pattern_inner_match_score(
                        inner_node_id,
                        observed_inner_node_id,
                        node_info=node_info,
                        similarity_index=similarity_index,
                    )
                    if inner_match_cache is not None:
                        inner_match_cache[cache_key] = inner_match
                if inner_match <= 0.0:
                    continue

                best_score = max(best_score, outer_sim * inner_match)

    return _pattern_transfer_clamp01(best_score)


def _pattern_transfer_settings_fingerprint() -> Tuple[Any, ...]:
    names = (
        "candidate_representation_guidance_enabled",
        "pattern_transfer_score_weight",
        "strict_base_score", "weak_base_score", "try_base_score",
        "pattern_transfer_top_k_similar_outer",
        "pattern_transfer_min_outer_sim", "pattern_transfer_max_outer_hop",
        "pattern_transfer_include_self_outer",
        "pattern_transfer_exact_inner_match_score",
        "pattern_transfer_same_return_type_match_score",
        "pattern_transfer_same_terminal_kind_match_score",
        "pattern_transfer_inner_operator_similarity_weight",
    )
    return tuple(getattr(CANDIDATE_BITMAP_CONFIG, name, None) for name in names)


def _pattern_transfer_file_fingerprint(similarity_index: Any) -> Tuple[Any, ...]:
    paths = (
        getattr(similarity_index, "similarity_path", None),
        getattr(similarity_index, "prototypes_path", None),
        getattr(similarity_index, "metadata_path", None),
    )
    result = []
    for raw_path in paths:
        try:
            stat = Path(raw_path).stat()
            result.append((str(raw_path), int(stat.st_size), int(stat.st_mtime_ns)))
        except Exception:
            result.append((str(raw_path or ""), None, None))
    return tuple(result)


def _pattern_transfer_edge_fingerprint(edge_records: List[Dict[str, Any]]) -> str:
    digest = hashlib.sha1()
    for record in edge_records:
        if not isinstance(record, dict):
            continue
        digest.update(
            "|".join(
                str(record.get(name, ""))
                for name in ("edge_id", "outer_node_id", "slot_index", "inner_node_id")
            ).encode("utf-8", errors="ignore")
        )
        digest.update(b"\n")
    return digest.hexdigest()


def _normalized_rule_token_set(raw_tokens: Any) -> set:
    if not isinstance(raw_tokens, list):
        return set()
    return {
        normalize_operator_key(str(token).strip())
        for token in raw_tokens
        if str(token).strip()
    }


def _candidate_rule_groups(rules: Dict[str, Any]) -> Dict[str, set]:
    bitmap_cfg = rules.get("candidate_bitmap", {}) or {}
    if not isinstance(bitmap_cfg, dict):
        bitmap_cfg = {}

    terminal_tokens = bitmap_cfg.get("terminal_tokens", {}) or {}
    if not isinstance(terminal_tokens, dict):
        terminal_tokens = {}

    return {
        "core": _normalized_rule_token_set(bitmap_cfg.get("core_scored_tokens", [])),
        "extension": _normalized_rule_token_set(bitmap_cfg.get("extension_scored_tokens", [])),
        "terminal": {
            normalize_operator_key(str(token).strip())
            for token in terminal_tokens.keys()
            if str(token).strip()
        },
    }


def _candidate_token_group(token: str, *, is_terminal: bool, groups: Dict[str, set]) -> str:
    norm = normalize_operator_key(token)

    if norm in groups.get("core", set()):
        return "core"

    if norm in groups.get("extension", set()):
        return "extension"

    if is_terminal or norm in groups.get("terminal", set()):
        return "core_terminal"

    return "other"


def _candidate_edge_group(outer_group: str, inner_group: str) -> str:
    if outer_group == "core" and inner_group == "extension":
        return "core_to_extension"

    if outer_group == "core" and inner_group in {"core", "core_terminal"}:
        return "core_internal"

    if outer_group == "extension":
        return "extension_outer"

    return "other"


def build_candidate_edges(rules: Dict[str, Any]) -> List[CandidateEdge]:
    signatures = collect_token_signatures(rules)
    include_unknown_as_try = bool(getattr(CANDIDATE_BITMAP_CONFIG, "include_unknown_as_try", False))

    bitmap_cfg = rules.get("candidate_bitmap", {}) or {}
    if not isinstance(bitmap_cfg, dict):
        bitmap_cfg = {}

    slot_type_constraint_mode = str(
        bitmap_cfg.get("slot_type_constraint_mode", "compatibility") or "compatibility"
    ).strip().lower()

    groups = _candidate_rule_groups(rules)

    edges: List[CandidateEdge] = []
    seen_edge_ids = set()

    outer_signatures = [sig for sig in signatures.values() if not sig.is_terminal and sig.arg_types]
    inner_signatures = list(signatures.values())

    for outer_sig in outer_signatures:
        outer_node_id = _node_id(outer_sig.token, outer_sig.return_type)

        for slot_index, required_type in enumerate(outer_sig.arg_types):
            for inner_sig in inner_signatures:
                level = _compatibility_level(
                    required_type,
                    inner_sig.coarse_return_type,
                    rules,
                    include_unknown_as_try=include_unknown_as_try,
                )
                if level is None:
                    if slot_type_constraint_mode == "level_hint_only":
                        level = "try"
                    else:
                        continue

                inner_node_id = _node_id(inner_sig.token, inner_sig.return_type)
                edge_id = f"{outer_node_id}::slot_{slot_index}->{inner_node_id}"

                if edge_id in seen_edge_ids:
                    continue
                seen_edge_ids.add(edge_id)

                pattern_transfer_score = 0.0
                edge_score = _edge_score_from_pattern_transfer(level, pattern_transfer_score)
                outer_group = _candidate_token_group(
                    outer_sig.token,
                    is_terminal=outer_sig.is_terminal,
                    groups=groups,
                )
                inner_group = _candidate_token_group(
                    inner_sig.token,
                    is_terminal=inner_sig.is_terminal,
                    groups=groups,
                )
                edge_group = _candidate_edge_group(outer_group, inner_group)

                edges.append(
                    CandidateEdge(
                        edge_index=len(edges),
                        edge_id=edge_id,
                        outer_token=outer_sig.token,
                        outer_node_id=outer_node_id,
                        outer_return_type=outer_sig.return_type,
                        slot_index=slot_index,
                        slot_required_type=str(required_type),
                        inner_token=inner_sig.token,
                        inner_node_id=inner_node_id,
                        inner_return_type=inner_sig.return_type,
                        candidate_level=level,
                        pattern_transfer_score=pattern_transfer_score,
                        edge_score=edge_score,
                        outer_group=outer_group,
                        inner_group=inner_group,
                        edge_group=edge_group,
                    )
                )

    return edges


def build_bitmaps(edges: List[CandidateEdge]) -> Dict[str, np.ndarray]:
    n = len(edges)

    strict_candidate_bitmap = np.zeros(n, dtype=np.uint8)
    weak_candidate_bitmap = np.zeros(n, dtype=np.uint8)
    try_candidate_bitmap = np.zeros(n, dtype=np.uint8)

    observed_bitmap = np.zeros(n, dtype=np.uint8)
    failed_bitmap = np.zeros(n, dtype=np.uint8)
    probing_bitmap = np.zeros(n, dtype=np.uint8)

    for edge in edges:
        if edge.candidate_level == "strict":
            strict_candidate_bitmap[edge.edge_index] = 1
        elif edge.candidate_level == "weak":
            weak_candidate_bitmap[edge.edge_index] = 1
        elif edge.candidate_level == "try":
            try_candidate_bitmap[edge.edge_index] = 1

    return {
        "strict_candidate_bitmap": strict_candidate_bitmap,
        "weak_candidate_bitmap": weak_candidate_bitmap,
        "try_candidate_bitmap": try_candidate_bitmap,
        "observed_bitmap": observed_bitmap,
        "failed_bitmap": failed_bitmap,
        "probing_bitmap": probing_bitmap,
    }


def _apply_candidate_edge_organization(
    edge_records: List[Dict[str, Any]],
) -> Dict[str, Any]:
    try:
        from RL.candidate_edge_organization import build_candidate_edge_organization

        return build_candidate_edge_organization(edge_records)
    except Exception as exc:
        if (
            bool(getattr(CANDIDATE_BITMAP_CONFIG, "candidate_organization_enabled", True))
            and bool(getattr(CANDIDATE_BITMAP_CONFIG, "candidate_hnsw_enabled", True))
            and "preflight failed" in str(exc).lower()
        ):
            raise

        print(f"[Candidate Organization Warning] build failed: {exc}", flush=True)
        for record in edge_records:
            record["structural_entropy"] = 0.0
        return {
            "version": "candidate_edge_organization_v1",
            "enabled": bool(
                getattr(CANDIDATE_BITMAP_CONFIG, "candidate_organization_enabled", True)
            ),
            "edge_count": int(len(edge_records)),
            "error": str(exc),
        }


def save_outputs(edges: List[CandidateEdge], bitmaps: Dict[str, np.ndarray]) -> Dict[str, Any]:
    bitmap_dir = Path(getattr(CANDIDATE_BITMAP_CONFIG, "bitmap_dir", CANDIDATE_BITMAP_DIR))
    edge_index_path = Path(getattr(CANDIDATE_BITMAP_CONFIG, "edge_index_path", CANDIDATE_EDGE_INDEX_PATH))
    bitmap_npz_path = Path(getattr(CANDIDATE_BITMAP_CONFIG, "bitmap_npz_path", CANDIDATE_BITMAP_NPZ_PATH))
    edge_scores_path = Path(getattr(CANDIDATE_BITMAP_CONFIG, "edge_scores_path", CANDIDATE_EDGE_SCORES_PATH))
    report_path = Path(getattr(CANDIDATE_BITMAP_CONFIG, "report_path", CANDIDATE_BITMAP_REPORT_PATH))

    bitmap_dir.mkdir(parents=True, exist_ok=True)

    edge_records = [edge.to_dict() for edge in edges]
    organization_report = _apply_candidate_edge_organization(edge_records)

    with edge_index_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "version": "stage17_1_candidate_bitmap_v1",
                "representation_guidance_enabled": bool(_representation_guidance_enabled()),
                "num_edges": len(edge_records),
                "edges": edge_records,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    np.savez_compressed(bitmap_npz_path, **bitmaps)

    with edge_scores_path.open("w", encoding="utf-8") as f:
        for record in edge_records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    report = build_report(edges, bitmaps)
    report["candidate_edge_organization"] = organization_report

    if bool(getattr(CANDIDATE_BITMAP_CONFIG, "write_human_readable_report", True)):
        with report_path.open("w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

    return report


def build_report(edges: List[CandidateEdge], bitmaps: Dict[str, np.ndarray]) -> Dict[str, Any]:
    by_level: Dict[str, int] = {"strict": 0, "weak": 0, "try": 0}
    by_outer: Dict[str, int] = {}
    by_inner: Dict[str, int] = {}

    for edge in edges:
        by_level[edge.candidate_level] = by_level.get(edge.candidate_level, 0) + 1
        by_outer[edge.outer_node_id] = by_outer.get(edge.outer_node_id, 0) + 1
        by_inner[edge.inner_node_id] = by_inner.get(edge.inner_node_id, 0) + 1

    top_outer = sorted(by_outer.items(), key=lambda item: (-item[1], item[0]))[:30]
    top_inner = sorted(by_inner.items(), key=lambda item: (-item[1], item[0]))[:30]

    return {
        "version": "stage17_1_candidate_bitmap_v1",
        "representation_guidance_enabled": bool(_representation_guidance_enabled()),
        "num_edges": len(edges),
        "by_level": by_level,
        "bitmap_counts": {
            name: int(np.asarray(bitmap).sum())
            for name, bitmap in bitmaps.items()
        },
        "top_outer_nodes": [{"node_id": k, "edge_count": v} for k, v in top_outer],
        "top_inner_nodes": [{"node_id": k, "edge_count": v} for k, v in top_inner],
        "outputs": {
            "edge_index_path": str(getattr(CANDIDATE_BITMAP_CONFIG, "edge_index_path", CANDIDATE_EDGE_INDEX_PATH)),
            "bitmap_npz_path": str(getattr(CANDIDATE_BITMAP_CONFIG, "bitmap_npz_path", CANDIDATE_BITMAP_NPZ_PATH)),
            "edge_scores_path": str(getattr(CANDIDATE_BITMAP_CONFIG, "edge_scores_path", CANDIDATE_EDGE_SCORES_PATH)),
            "report_path": str(getattr(CANDIDATE_BITMAP_CONFIG, "report_path", CANDIDATE_BITMAP_REPORT_PATH)),
        },
    }


def _edge_id_from_parts(outer_node_id: str, slot_index: int, inner_node_id: str) -> str:
    return f"{outer_node_id}::slot_{slot_index}->{inner_node_id}"


def _load_edge_index_records(edge_index_path: Path) -> List[Dict[str, Any]]:
    path = Path(edge_index_path)
    if not path.exists():
        raise FileNotFoundError(
            f'Candidate edge index is missing; build the candidate bitmap first: {path}'
        )

    data = _load_json(path)
    records = data.get("edges", [])
    if not isinstance(records, list):
        raise ValueError(f"The 'edges' field in candidate_edge_index.json must be a list: {path}")

    return [x for x in records if isinstance(x, dict)]


def _load_npz_bitmaps(bitmap_npz_path: Path, expected_size: int) -> Dict[str, np.ndarray]:
    path = Path(bitmap_npz_path)
    if not path.exists():
        raise FileNotFoundError(
            f'Candidate bitmap NPZ is missing; build the candidate bitmap first: {path}'
        )

    result: Dict[str, np.ndarray] = {}

    required_names = [
        "strict_candidate_bitmap",
        "weak_candidate_bitmap",
        "try_candidate_bitmap",
        "observed_bitmap",
        "failed_bitmap",
        "probing_bitmap",
        "missing_candidate_bitmap",
        "available_initial_bitmap",
    ]

    with np.load(path) as loaded:
        for name in required_names:
            if name in loaded:
                arr = np.asarray(loaded[name], dtype=np.uint8)
                if arr.shape[0] != expected_size:
                    raise ValueError(
                        f'{name} length does not match the number of candidate edges: {arr.shape[0]} != {expected_size}'
                    )
                result[name] = arr.copy()
            else:
                result[name] = np.zeros(expected_size, dtype=np.uint8)

    return result


def _iter_graph_edges_from_json(graph_path: Path) -> Tuple[List[str], List[Tuple[str, int, str]]]:
    path = Path(graph_path)
    if not path.exists():
        raise FileNotFoundError(
            f'global_knowledge_graph.json is missing. Expected location: pkg/knowledge_graph/global_knowledge_graph.json; current path: {path}'
        )

    data = _load_json(path)
    nodes = data.get("nodes", {})
    if not isinstance(nodes, dict):
        raise ValueError(f"The 'nodes' field in global_knowledge_graph.json must be a dict: {path}")

    node_ids = sorted(str(node_id) for node_id in nodes.keys())
    edges: List[Tuple[str, int, str]] = []

    for outer_node_id, payload in nodes.items():
        if not isinstance(payload, dict):
            continue

        slots = payload.get("slots", [])
        if not isinstance(slots, list):
            continue

        for slot_index, slot_items in enumerate(slots):
            if not isinstance(slot_items, list):
                continue

            for inner_node_id in slot_items:
                edges.append((str(outer_node_id), int(slot_index), str(inner_node_id)))

    edges.sort(key=lambda item: (item[0], item[1], item[2]))
    return node_ids, edges


def _candidate_union_bitmap(bitmaps: Dict[str, np.ndarray]) -> np.ndarray:
    strict = np.asarray(bitmaps.get("strict_candidate_bitmap", []), dtype=np.uint8)
    weak = np.asarray(bitmaps.get("weak_candidate_bitmap", []), dtype=np.uint8)
    try_bits = np.asarray(bitmaps.get("try_candidate_bitmap", []), dtype=np.uint8)

    return ((strict > 0) | (weak > 0) | (try_bits > 0)).astype(np.uint8)


def _initial_open_candidate_bitmap(bitmaps: Dict[str, np.ndarray]) -> np.ndarray:
    n = len(np.asarray(bitmaps.get("strict_candidate_bitmap", []), dtype=np.uint8))
    opened = np.zeros(n, dtype=bool)

    if bool(getattr(CANDIDATE_BITMAP_CONFIG, "open_strict_initially", True)):
        opened |= np.asarray(bitmaps.get("strict_candidate_bitmap", np.zeros(n)), dtype=np.uint8) > 0

    if bool(getattr(CANDIDATE_BITMAP_CONFIG, "open_weak_initially", False)):
        opened |= np.asarray(bitmaps.get("weak_candidate_bitmap", np.zeros(n)), dtype=np.uint8) > 0

    if bool(getattr(CANDIDATE_BITMAP_CONFIG, "open_try_initially", False)):
        opened |= np.asarray(bitmaps.get("try_candidate_bitmap", np.zeros(n)), dtype=np.uint8) > 0

    return opened.astype(np.uint8)


def _level_counts_from_bitmap(
    edge_records: List[Dict[str, Any]],
    bitmap: np.ndarray,
) -> Dict[str, int]:
    result: Dict[str, int] = {"strict": 0, "weak": 0, "try": 0}

    for record in edge_records:
        try:
            idx = int(record.get("edge_index", -1))
        except Exception:
            continue

        if idx < 0 or idx >= len(bitmap):
            continue

        if int(bitmap[idx]) <= 0:
            continue

        level = str(record.get("candidate_level", "unknown"))
        result[level] = result.get(level, 0) + 1

    return result


def _write_missing_edges(
    edge_records: List[Dict[str, Any]],
    missing_bitmap: np.ndarray,
    missing_edges_path: Path,
) -> int:
    path = Path(missing_edges_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with path.open("w", encoding="utf-8") as f:
        for record in edge_records:
            try:
                idx = int(record.get("edge_index", -1))
            except Exception:
                continue

            if idx < 0 or idx >= len(missing_bitmap):
                continue

            if int(missing_bitmap[idx]) <= 0:
                continue

            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1

    return count


def build_and_save_observed_bitmap(graph_path: Optional[Path] = None) -> Dict[str, Any]:
    edge_index_path = Path(getattr(CANDIDATE_BITMAP_CONFIG, "edge_index_path", CANDIDATE_EDGE_INDEX_PATH))
    bitmap_npz_path = Path(getattr(CANDIDATE_BITMAP_CONFIG, "bitmap_npz_path", CANDIDATE_BITMAP_NPZ_PATH))
    observed_report_path = Path(
        getattr(CANDIDATE_BITMAP_CONFIG, "observed_report_path", CANDIDATE_OBSERVED_REPORT_PATH)
    )
    missing_edges_path = Path(
        getattr(CANDIDATE_BITMAP_CONFIG, "missing_edges_path", CANDIDATE_MISSING_EDGES_PATH)
    )
    graph_file = Path(
        graph_path
        or getattr(CANDIDATE_BITMAP_CONFIG, "global_knowledge_graph_path", GLOBAL_KNOWLEDGE_GRAPH_PATH)
    )

    edge_records = _load_edge_index_records(edge_index_path)
    edge_id_to_index: Dict[str, int] = {}

    for record in edge_records:
        edge_id = str(record.get("edge_id", ""))
        if not edge_id:
            continue

        try:
            edge_index = int(record.get("edge_index"))
        except Exception:
            continue

        edge_id_to_index[edge_id] = edge_index

    n = len(edge_records)
    bitmaps = _load_npz_bitmaps(bitmap_npz_path, n)

    graph_node_ids, graph_edges = _iter_graph_edges_from_json(graph_file)

    observed_bitmap = np.zeros(n, dtype=np.uint8)
    unmatched_graph_edges: List[Dict[str, Any]] = []
    matched_graph_edge_count = 0

    for outer_node_id, slot_index, inner_node_id in graph_edges:
        edge_id = _edge_id_from_parts(outer_node_id, slot_index, inner_node_id)
        edge_index = edge_id_to_index.get(edge_id)

        if edge_index is None:
            if len(unmatched_graph_edges) < 200:
                unmatched_graph_edges.append(
                    {
                        "edge_id": edge_id,
                        "outer_node_id": outer_node_id,
                        "slot_index": slot_index,
                        "inner_node_id": inner_node_id,
                    }
                )
            continue

        observed_bitmap[edge_index] = 1
        matched_graph_edge_count += 1

    bitmaps["observed_bitmap"] = observed_bitmap

    candidate_union_bitmap = _candidate_union_bitmap(bitmaps)
    failed_bitmap = np.asarray(bitmaps.get("failed_bitmap", np.zeros(n)), dtype=np.uint8)
    probing_bitmap = np.asarray(bitmaps.get("probing_bitmap", np.zeros(n)), dtype=np.uint8)

    missing_candidate_bitmap = (
        (candidate_union_bitmap > 0)
        & (observed_bitmap == 0)
        & (failed_bitmap == 0)
        & (probing_bitmap == 0)
    ).astype(np.uint8)

    open_candidate_bitmap = _initial_open_candidate_bitmap(bitmaps)
    available_initial_bitmap = (
        (open_candidate_bitmap > 0)
        & (observed_bitmap == 0)
        & (failed_bitmap == 0)
        & (probing_bitmap == 0)
    ).astype(np.uint8)

    bitmaps["missing_candidate_bitmap"] = missing_candidate_bitmap
    bitmaps["available_initial_bitmap"] = available_initial_bitmap

    np.savez_compressed(bitmap_npz_path, **bitmaps)

    missing_edge_count_written = _write_missing_edges(
        edge_records,
        missing_candidate_bitmap,
        missing_edges_path,
    )

    candidate_total = int(candidate_union_bitmap.sum())
    observed_total = int(observed_bitmap.sum())
    missing_total = int(missing_candidate_bitmap.sum())
    available_initial_total = int(available_initial_bitmap.sum())
    graph_edge_total = len(graph_edges)

    report = {
        "version": "stage17_2_observed_bitmap_v1",
        "global_knowledge_graph_path": str(graph_file),
        "candidate_edge_index_path": str(edge_index_path),
        "bitmap_npz_path": str(bitmap_npz_path),
        "observed_report_path": str(observed_report_path),
        "missing_edges_path": str(missing_edges_path),
        "graph": {
            "node_count": len(graph_node_ids),
            "edge_count": graph_edge_total,
            "matched_candidate_edge_count": matched_graph_edge_count,
            "unmatched_graph_edge_count": graph_edge_total - matched_graph_edge_count,
            "graph_edge_match_rate": (
                matched_graph_edge_count / graph_edge_total if graph_edge_total > 0 else 0.0
            ),
            "unmatched_graph_edges_sample": unmatched_graph_edges,
        },
        "candidate": {
            "candidate_edge_count": candidate_total,
            "observed_edge_count": observed_total,
            "missing_edge_count": missing_total,
            "available_initial_edge_count": available_initial_total,
            "coverage_rate": observed_total / candidate_total if candidate_total > 0 else 0.0,
        },
        "by_level": {
            "candidate": _level_counts_from_bitmap(edge_records, candidate_union_bitmap),
            "observed": _level_counts_from_bitmap(edge_records, observed_bitmap),
            "missing": _level_counts_from_bitmap(edge_records, missing_candidate_bitmap),
            "available_initial": _level_counts_from_bitmap(edge_records, available_initial_bitmap),
        },
        "bitmap_counts": {
            name: int(np.asarray(bitmap).sum())
            for name, bitmap in bitmaps.items()
        },
        "notes": [
            'observed_bitmap marks candidate edges already verified in the formal knowledge graph.',
            "missing_candidate_bitmap = candidate_union & ~observed & ~failed & ~probing。",
            'available_initial_bitmap is calculated from the open_strict/weak/try settings; strict is enabled by default.',
            'global_knowledge_graph.json is located under pkg/knowledge_graph by default, not output.',
        ],
    }

    observed_report_path.parent.mkdir(parents=True, exist_ok=True)
    with observed_report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    return report


def check_observed_bitmap_graph_consistency(
    graph_path: Optional[Path] = None,
    *,
    edge_index_path: Optional[Path] = None,
    bitmap_npz_path: Optional[Path] = None,
    report_path: Optional[Path] = None,
) -> Dict[str, Any]:
    resolved_edge_index_path = Path(
        edge_index_path
        or getattr(CANDIDATE_BITMAP_CONFIG, "edge_index_path", CANDIDATE_EDGE_INDEX_PATH)
    )
    resolved_bitmap_npz_path = Path(
        bitmap_npz_path
        or getattr(CANDIDATE_BITMAP_CONFIG, "bitmap_npz_path", CANDIDATE_BITMAP_NPZ_PATH)
    )
    graph_file = Path(
        graph_path
        or getattr(CANDIDATE_BITMAP_CONFIG, "global_knowledge_graph_path", GLOBAL_KNOWLEDGE_GRAPH_PATH)
    )
    resolved_report_path = Path(
        report_path
        or resolved_bitmap_npz_path.parent / "candidate_bitmap_graph_consistency_report.json"
    )

    edge_records = _load_edge_index_records(resolved_edge_index_path)
    bitmaps = _load_npz_bitmaps(resolved_bitmap_npz_path, len(edge_records))
    observed_bitmap = np.asarray(
        bitmaps.get("observed_bitmap", np.zeros(len(edge_records))),
        dtype=np.uint8,
    )

    records_by_edge_id: Dict[str, Dict[str, Any]] = {}
    observed_edge_ids = set()
    for record in edge_records:
        edge_id = str(record.get("edge_id", "") or "")
        try:
            edge_index = int(record.get("edge_index", -1))
        except Exception:
            continue
        if not edge_id or edge_index < 0 or edge_index >= len(observed_bitmap):
            continue
        records_by_edge_id[edge_id] = record
        if int(observed_bitmap[edge_index]) > 0:
            observed_edge_ids.add(edge_id)

    _graph_node_ids, graph_edges = _iter_graph_edges_from_json(graph_file)
    graph_candidate_edge_ids = {
        edge_id
        for outer_node_id, slot_index, inner_node_id in graph_edges
        for edge_id in [_edge_id_from_parts(outer_node_id, slot_index, inner_node_id)]
        if edge_id in records_by_edge_id
    }

    def _difference_record(edge_id: str) -> Dict[str, Any]:
        record = records_by_edge_id[edge_id]
        try:
            slot_index = int(record.get("slot_index", -1))
        except Exception:
            slot_index = -1
        outer_token = str(
            record.get("outer_token", _split_graph_node_id(str(record.get("outer_node_id", "")))[0])
            or ""
        )
        inner_token = str(
            record.get("inner_token", _split_graph_node_id(str(record.get("inner_node_id", "")))[0])
            or ""
        )
        return {
            "edge_id": edge_id,
            "display_edge": f"{outer_token}.slot_{slot_index}->{inner_token}",
        }

    bitmap_only_ids = sorted(observed_edge_ids - graph_candidate_edge_ids)
    graph_only_ids = sorted(graph_candidate_edge_ids - observed_edge_ids)
    bitmap_only_edges = [_difference_record(edge_id) for edge_id in bitmap_only_ids]
    graph_only_edges = [_difference_record(edge_id) for edge_id in graph_only_ids]

    report = {
        "version": "candidate_bitmap_graph_consistency_v1",
        "global_knowledge_graph_path": str(graph_file),
        "candidate_edge_index_path": str(resolved_edge_index_path),
        "bitmap_npz_path": str(resolved_bitmap_npz_path),
        "report_path": str(resolved_report_path),
        "is_consistent": not bitmap_only_edges and not graph_only_edges,
        "observed_bitmap_count": len(observed_edge_ids),
        "graph_candidate_edge_count": len(graph_candidate_edge_ids),
        "bitmap_only_count": len(bitmap_only_edges),
        "graph_only_count": len(graph_only_edges),
        "bitmap_only_edges": bitmap_only_edges,
        "graph_only_edges": graph_only_edges,
    }

    resolved_report_path.parent.mkdir(parents=True, exist_ok=True)
    with resolved_report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    return report


def sync_observed_bitmap_to_global_graph(graph_path: Optional[Path] = None) -> Dict[str, Any]:
    edge_index_path = Path(getattr(CANDIDATE_BITMAP_CONFIG, "edge_index_path", CANDIDATE_EDGE_INDEX_PATH))
    bitmap_npz_path = Path(getattr(CANDIDATE_BITMAP_CONFIG, "bitmap_npz_path", CANDIDATE_BITMAP_NPZ_PATH))
    graph_file = Path(
        graph_path
        or getattr(CANDIDATE_BITMAP_CONFIG, "global_knowledge_graph_path", GLOBAL_KNOWLEDGE_GRAPH_PATH)
    )

    edge_records = _load_edge_index_records(edge_index_path)
    bitmaps = _load_npz_bitmaps(bitmap_npz_path, len(edge_records))
    observed_bitmap = np.asarray(
        bitmaps.get("observed_bitmap", np.zeros(len(edge_records))),
        dtype=np.uint8,
    )

    graph_data = _load_json(graph_file)
    nodes = graph_data.get("nodes", {})
    if not isinstance(nodes, dict):
        raise ValueError(f"The 'nodes' field in global_knowledge_graph.json must be a dict: {graph_file}")
    graph_data["nodes"] = nodes

    roots = graph_data.get("roots", [])
    if not isinstance(roots, list):
        roots = []
    graph_data["roots"] = roots

    added_edges: List[Dict[str, Any]] = []
    skipped_edges: List[Dict[str, Any]] = []

    for record in edge_records:
        try:
            edge_index = int(record.get("edge_index", -1))
            slot_index = int(record.get("slot_index", -1))
        except Exception:
            continue

        if edge_index < 0 or edge_index >= len(observed_bitmap):
            continue
        if int(observed_bitmap[edge_index]) <= 0:
            continue

        outer_node_id = str(record.get("outer_node_id", ""))
        inner_node_id = str(record.get("inner_node_id", ""))
        if not outer_node_id or not inner_node_id or slot_index < 0:
            continue
        if _graph_has_edge(graph_data, outer_node_id, slot_index, inner_node_id):
            continue

        if outer_node_id not in nodes or not isinstance(nodes.get(outer_node_id), dict):
            nodes[outer_node_id] = _make_probe_node_payload(
                node_id=outer_node_id,
                token=str(record.get("outer_token", _split_graph_node_id(outer_node_id)[0])),
                return_type=str(record.get("outer_return_type", _split_graph_node_id(outer_node_id)[1])),
                edge_records=edge_records,
                force_non_terminal=True,
                min_slot_count=slot_index + 1,
            )

        if inner_node_id not in nodes or not isinstance(nodes.get(inner_node_id), dict):
            nodes[inner_node_id] = _make_probe_node_payload(
                node_id=inner_node_id,
                token=str(record.get("inner_token", _split_graph_node_id(inner_node_id)[0])),
                return_type=str(record.get("inner_return_type", _split_graph_node_id(inner_node_id)[1])),
                edge_records=edge_records,
                force_non_terminal=False,
                min_slot_count=0,
            )

        outer_payload = nodes.get(outer_node_id)
        if not isinstance(outer_payload, dict):
            skipped_edges.append(
                {
                    "edge_id": record.get("edge_id"),
                    "reason": "outer_payload_not_dict",
                }
            )
            continue

        _ensure_node_slots(outer_payload, slot_index + 1)
        slot_items = outer_payload["slots"][slot_index]
        if not isinstance(slot_items, list):
            slot_items = []
            outer_payload["slots"][slot_index] = slot_items

        if inner_node_id not in slot_items:
            slot_items.append(inner_node_id)
            slot_items.sort()
            added_edges.append(
                {
                    "edge_id": record.get("edge_id"),
                    "outer_node_id": outer_node_id,
                    "slot_index": slot_index,
                    "inner_node_id": inner_node_id,
                }
            )

        if (
            str(record.get("outer_return_type", "")).upper() == "T_BOOL"
            and outer_node_id not in roots
        ):
            roots.append(outer_node_id)
            roots.sort()

    before_refresh_edge_count = _graph_edge_count_from_data(graph_data)
    graph_file.parent.mkdir(parents=True, exist_ok=True)
    with graph_file.open("w", encoding="utf-8") as f:
        json.dump(graph_data, f, indent=2, ensure_ascii=False)

    observed_report = build_and_save_observed_bitmap(graph_file)

    return {
        "version": "stage17_final_observed_bitmap_to_graph_sync_v1",
        "global_knowledge_graph_path": str(graph_file),
        "candidate_edge_index_path": str(edge_index_path),
        "bitmap_npz_path": str(bitmap_npz_path),
        "observed_bitmap_count": int(observed_bitmap.sum()),
        "added_edge_count": len(added_edges),
        "skipped_edge_count": len(skipped_edges),
        "graph_edge_count_after_sync": before_refresh_edge_count,
        "observed_report": observed_report,
        "added_edges_sample": added_edges[:50],
        "skipped_edges_sample": skipped_edges[:50],
    }


def _graph_edge_count_from_data(graph_data: Dict[str, Any]) -> int:
    nodes = graph_data.get("nodes", {})
    if not isinstance(nodes, dict):
        return 0

    count = 0
    for payload in nodes.values():
        if not isinstance(payload, dict):
            continue
        slots = payload.get("slots", [])
        if not isinstance(slots, list):
            continue
        for slot_items in slots:
            if isinstance(slot_items, list):
                count += len(slot_items)
    return count


def _required_slot_count_for_node(
    node_id: str,
    edge_records: List[Dict[str, Any]],
) -> int:
    max_slot_index = -1

    for record in edge_records:
        if str(record.get("outer_node_id", "")) != str(node_id):
            continue

        try:
            slot_index = int(record.get("slot_index", -1))
        except Exception:
            continue

        if slot_index > max_slot_index:
            max_slot_index = slot_index

    return max_slot_index + 1 if max_slot_index >= 0 else 0


def _make_probe_node_payload(
    *,
    node_id: str,
    token: str,
    return_type: str,
    edge_records: List[Dict[str, Any]],
    force_non_terminal: bool = False,
    min_slot_count: int = 0,
) -> Dict[str, Any]:
    slot_count = max(_required_slot_count_for_node(node_id, edge_records), int(min_slot_count))
    is_terminal = False if force_non_terminal else slot_count == 0

    return {
        "token": str(token),
        "return_type": str(return_type),
        "is_terminal": bool(is_terminal),
        "slots": [[] for _ in range(slot_count)],
    }


def _ensure_node_slots(payload: Dict[str, Any], min_slot_count: int) -> None:
    slots = payload.get("slots", [])
    if not isinstance(slots, list):
        slots = []

    while len(slots) < min_slot_count:
        slots.append([])

    payload["slots"] = slots


def _graph_has_edge(
    graph_data: Dict[str, Any],
    outer_node_id: str,
    slot_index: int,
    inner_node_id: str,
) -> bool:
    nodes = graph_data.get("nodes", {})
    if not isinstance(nodes, dict):
        return False

    outer_payload = nodes.get(outer_node_id)
    if not isinstance(outer_payload, dict):
        return False

    slots = outer_payload.get("slots", [])
    if not isinstance(slots, list):
        return False

    if slot_index < 0 or slot_index >= len(slots):
        return False

    slot_items = slots[slot_index]
    if not isinstance(slot_items, list):
        return False

    return inner_node_id in slot_items


def _find_edge_record(
    edge_records: List[Dict[str, Any]],
    *,
    probe_edge_id: Optional[str] = None,
    probe_edge_index: Optional[int] = None,
) -> Optional[Dict[str, Any]]:
    if probe_edge_id:
        for record in edge_records:
            if str(record.get("edge_id", "")) == str(probe_edge_id):
                return record
        return None

    if probe_edge_index is not None:
        for record in edge_records:
            try:
                idx = int(record.get("edge_index", -1))
            except Exception:
                continue
            if idx == int(probe_edge_index):
                return record
        return None

    return None


def _probe_edge_support_rank(
    record: Dict[str, Any],
    *,
    formal_node_ids: Iterable[str],
    edge_records: List[Dict[str, Any]],
) -> int:
    node_set = {str(x) for x in formal_node_ids}

    outer_node_id = str(record.get("outer_node_id", ""))
    inner_node_id = str(record.get("inner_node_id", ""))

    outer_preexisting = outer_node_id in node_set
    inner_preexisting = inner_node_id in node_set
    is_self_loop = outer_node_id == inner_node_id

    inner_required_slot_count = _required_slot_count_for_node(inner_node_id, edge_records)
    inner_terminal_like = inner_required_slot_count == 0

    if outer_preexisting and inner_preexisting and not is_self_loop:
        return 60

    if outer_preexisting and inner_preexisting:
        return 50

    if outer_preexisting and inner_terminal_like:
        return 45

    if outer_preexisting:
        return 35

    if inner_preexisting and not is_self_loop:
        return 25

    if not is_self_loop:
        return 10

    return 0


def _select_probe_edge_record(
    edge_records: List[Dict[str, Any]],
    bitmaps: Dict[str, np.ndarray],
    *,
    formal_node_ids: Iterable[str],
    probe_edge_id: Optional[str] = None,
    probe_edge_index: Optional[int] = None,
    allow_observed: bool = False,
) -> Dict[str, Any]:
    explicit_record = _find_edge_record(
        edge_records,
        probe_edge_id=probe_edge_id,
        probe_edge_index=probe_edge_index,
    )

    observed_bitmap = np.asarray(bitmaps.get("observed_bitmap", []), dtype=np.uint8)
    failed_bitmap = np.asarray(bitmaps.get("failed_bitmap", []), dtype=np.uint8)
    probing_bitmap = np.asarray(bitmaps.get("probing_bitmap", []), dtype=np.uint8)

    if explicit_record is not None:
        idx = int(explicit_record.get("edge_index", -1))
        if idx < 0 or idx >= len(edge_records):
            raise ValueError(f'The explicitly specified probe edge index is invalid: {idx}')

        if not allow_observed:
            if idx < len(observed_bitmap) and int(observed_bitmap[idx]) > 0:
                raise ValueError(
                    f"The explicitly specified probe edge is already in observed_bitmap and should not be tested as a missing edge: {explicit_record.get('edge_id')}"
                )
            if idx < len(failed_bitmap) and int(failed_bitmap[idx]) > 0:
                raise ValueError(
                    f"The explicitly specified probe edge is already in failed_bitmap and should not be tested as a missing edge: {explicit_record.get('edge_id')}"
                )

        if idx < len(probing_bitmap) and int(probing_bitmap[idx]) > 0:
            raise ValueError(
                f"The explicitly specified probe edge is already in probing_bitmap and should not be tested again: {explicit_record.get('edge_id')}"
            )

        return explicit_record

    masks_to_try = [
        np.asarray(bitmaps.get("available_initial_bitmap", []), dtype=np.uint8),
        np.asarray(bitmaps.get("missing_candidate_bitmap", []), dtype=np.uint8),
    ]

    best_record: Optional[Dict[str, Any]] = None
    best_key: Optional[Tuple[float, int]] = None

    for mask in masks_to_try:
        if mask.size == 0:
            continue

        for record in edge_records:
            try:
                idx = int(record.get("edge_index", -1))
            except Exception:
                continue

            if idx < 0 or idx >= len(mask):
                continue

            if int(mask[idx]) <= 0:
                continue

            if not allow_observed:
                if idx < len(observed_bitmap) and int(observed_bitmap[idx]) > 0:
                    continue
                if idx < len(failed_bitmap) and int(failed_bitmap[idx]) > 0:
                    continue

            if idx < len(probing_bitmap) and int(probing_bitmap[idx]) > 0:
                continue

            try:
                edge_score = float(record.get("edge_score", 0.0))
            except Exception:
                edge_score = 0.0

            support_rank = _probe_edge_support_rank(
                record,
                formal_node_ids=formal_node_ids,
                edge_records=edge_records,
            )

            level = str(record.get("candidate_level", "try")).lower()
            level_rank = {
                "strict": 2,
                "weak": 1,
                "try": 0,
            }.get(level, 0)

            key = (support_rank, level_rank, edge_score, -idx)
            if best_key is None or key > best_key:
                best_key = key
                best_record = record

        if best_record is not None:
            return best_record

    raise ValueError('No missing / available edge was found for probe_graph; run --build-observed and check the bitmaps first.')


def _refresh_missing_and_available_bitmaps(bitmaps: Dict[str, np.ndarray]) -> None:
    n = len(np.asarray(bitmaps.get("strict_candidate_bitmap", []), dtype=np.uint8))
    candidate_union_bitmap = _candidate_union_bitmap(bitmaps)

    observed_bitmap = np.asarray(bitmaps.get("observed_bitmap", np.zeros(n)), dtype=np.uint8)
    failed_bitmap = np.asarray(bitmaps.get("failed_bitmap", np.zeros(n)), dtype=np.uint8)
    probing_bitmap = np.asarray(bitmaps.get("probing_bitmap", np.zeros(n)), dtype=np.uint8)

    missing_candidate_bitmap = (
        (candidate_union_bitmap > 0)
        & (observed_bitmap == 0)
        & (failed_bitmap == 0)
        & (probing_bitmap == 0)
    ).astype(np.uint8)

    open_candidate_bitmap = _initial_open_candidate_bitmap(bitmaps)
    available_initial_bitmap = (
        (open_candidate_bitmap > 0)
        & (observed_bitmap == 0)
        & (failed_bitmap == 0)
        & (probing_bitmap == 0)
    ).astype(np.uint8)

    bitmaps["missing_candidate_bitmap"] = missing_candidate_bitmap
    bitmaps["available_initial_bitmap"] = available_initial_bitmap


@contextmanager
def _candidate_bitmap_file_lock(target_path: Path, timeout_seconds: float = 30.0, poll_interval: float = 0.05):
    target_path = Path(target_path)
    lock_path = target_path.with_name(target_path.name + ".lock")

    start = time.time()
    fd = None

    while True:
        try:
            fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_RDWR)
            os.write(fd, str(os.getpid()).encode("utf-8"))
            break
        except FileExistsError:
            if time.time() - start > timeout_seconds:
                raise TimeoutError(f'Time limit exceeded while waiting for the candidate bitmap file lock: {lock_path}')
            time.sleep(poll_interval)

    try:
        yield
    finally:
        try:
            if fd is not None:
                os.close(fd)
        finally:
            try:
                lock_path.unlink()
            except FileNotFoundError:
                pass


def _candidate_bitmap_atomic_write_json(path: Path, data: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        delete=False,
        dir=str(path.parent),
        suffix=".tmp",
    ) as tmp:
        json.dump(data, tmp, indent=2, ensure_ascii=False)
        tmp.write("\n")
        tmp_path = Path(tmp.name)

    os.replace(str(tmp_path), str(path))


def _candidate_bitmap_atomic_save_npz(path: Path, bitmaps: Dict[str, np.ndarray]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile(
        mode="wb",
        delete=False,
        dir=str(path.parent),
        suffix=".npz",
    ) as tmp:
        tmp_path = Path(tmp.name)

    try:
        np.savez_compressed(tmp_path, **bitmaps)
        os.replace(str(tmp_path), str(path))
    finally:
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except FileNotFoundError:
            pass


def _candidate_bitmap_append_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _candidate_bitmap_atomic_write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        delete=False,
        dir=str(path.parent),
        suffix=".jsonl.tmp",
    ) as tmp:
        for row in rows:
            tmp.write(json.dumps(row, ensure_ascii=False) + "\n")
        tmp_path = Path(tmp.name)

    os.replace(str(tmp_path), str(path))


def _candidate_bitmap_as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value

    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "success", "ok"}

    return bool(value)


def _candidate_bitmap_fail_threshold(level: str) -> int:
    _ = level

    return max(1, int(getattr(CANDIDATE_BITMAP_CONFIG, "target_db_failure_signature_threshold", 10)))


def _candidate_bitmap_normalize_failure_signature(predicate: Any) -> str:
    text = str(predicate or "").strip()
    if not text:
        return ""
    text = re.sub(r"\s+", " ", text)
    return text.lower()


def _candidate_bitmap_probe_variant_enabled() -> bool:
    return bool(getattr(CANDIDATE_BITMAP_CONFIG, "candidate_bitmap_probe_variant_enabled", True))


def _candidate_bitmap_probe_variant_max_attempts() -> int:
    return max(1, int(getattr(CANDIDATE_BITMAP_CONFIG, "candidate_bitmap_probe_variant_max_attempts", 20)))


def _candidate_bitmap_tried_probe_signatures(record: Dict[str, Any]) -> List[str]:
    raw = record.get("tried_probe_signatures", [])
    if not isinstance(raw, list):
        return []

    signatures: List[str] = []
    seen = set()
    for item in raw:
        signature = _candidate_bitmap_normalize_failure_signature(item)
        if signature and signature not in seen:
            signatures.append(signature)
            seen.add(signature)
    return signatures


def _candidate_bitmap_record_tried_probe_signature(record: Dict[str, Any], predicate: Any) -> bool:
    signature = _candidate_bitmap_normalize_failure_signature(predicate)
    if not signature:
        return False

    signatures = _candidate_bitmap_tried_probe_signatures(record)
    if signature in signatures:
        record["tried_probe_signatures"] = signatures
        return False

    signatures.append(signature)
    record["tried_probe_signatures"] = signatures
    return True


def _candidate_bitmap_result_target_hit(result: Dict[str, Any]) -> bool:
    if "target_edge_hit" in result:
        return _candidate_bitmap_as_bool(result.get("target_edge_hit"))
    return _candidate_bitmap_as_bool(result.get("candidate_bitmap_target_edge_hit"))


def _candidate_bitmap_result_success(result: Dict[str, Any]) -> bool:
    return (
        _candidate_bitmap_as_bool(result.get("db_success"))
        and _candidate_bitmap_as_bool(result.get("parse_success"))
        and _candidate_bitmap_result_target_hit(result)
    )


def _candidate_bitmap_result_should_skip_failure(result: Dict[str, Any]) -> bool:
    status = str(result.get("status", "") or result.get("result_status", "")).lower()
    try:
        db_type = str(getattr(DB_CONFIG, "db_type", "") or "").lower().strip()
    except Exception:
        db_type = ""
    if db_type == "sqlserver":
        from RL.joint_signature_evidence import classify_sqlserver_error
        if _candidate_bitmap_as_bool(result.get("environment_failure")):
            return True
        kind = classify_sqlserver_error(
            result.get("db_error_code", result.get("error_code")),
            result.get("db_error", result.get("error")),
            sqlstate=result.get("db_sqlstate", result.get("sqlstate")),
            timed_out=any(_candidate_bitmap_as_bool(result.get(key))
                          for key in ("timed_out", "db_timed_out", "is_timeout")),
            cancelled=_candidate_bitmap_as_bool(result.get("db_cancelled", result.get("cancelled"))),
        )
        if kind in {"timeout_count", "environment_failure_count"}:
            return True
    if db_type in {"oracle", "oracledb", "oracle_ai26"} and (
        _candidate_bitmap_as_bool(result.get("timed_out"))
        or _candidate_bitmap_as_bool(result.get("db_timed_out"))
        or _candidate_bitmap_as_bool(result.get("environment_failure"))
    ):
        return True
    if status in {"db_connection_failed", "environment_error"}:
        return True

    failure_kind = str(result.get("failure_kind", "") or "").lower()
    if failure_kind in {
        "context_inconclusive",
        "downstream_construct_failed",
        "db_connection_failed",
        "environment_error",
    }:
        return True

    if _candidate_bitmap_as_bool(result.get("safe_closure_used")):
        return True

    db_error = str(result.get("db_error", "") or "").lower()
    db_error_code = str(
        result.get("db_error_code", result.get("error_code", "")) or ""
    ).upper().strip()
    if (
        "access denied for user" in db_error
        or "failed to connect to mysql" in db_error
        or "can't connect to mysql" in db_error
        or "connection refused" in db_error
    ):
        return True

    if db_type in {"oracle", "oracledb", "oracle_ai26"}:
        try:
            numeric_code = int(db_error_code.replace("ORA-", "").replace("DPY-", ""))
        except (TypeError, ValueError):
            numeric_code = None
        if numeric_code in {
            18, 20, 54, 60, 1012, 1013, 1017, 1033, 1034, 1089, 1219, 12170,
            12514, 12537, 12541, 12543, 12545, 3113, 3114, 3135,
            4024, 6000, 6001, 6005, 4011, 2049, 4031, 4036, 1652,
        }:
            return True
        if any(
            marker in db_error
            for marker in (
                "failed to connect to oracle",
                "oracle schema is unavailable",
                "oracle schema has no column",
                "oracle not available",
                "no listener",
                "not logged on",
                "end-of-file on communication channel",
                "connection lost contact",
                "connection closed",
                "dpy-600",
                "dpy-4011",
                "maximum number of sessions",
                "maximum number of processes",
                "shared memory",
                "out of memory",
            )
        ):
            return True

    return False


def candidate_bitmap_environment_failure(result: Dict[str, Any]) -> bool:
    if not isinstance(result, dict):
        return False
    if str(result.get("status", "") or result.get("result_status", "")).lower() in {
        "db_connection_failed", "environment_error",
    }:
        return True
    if str(result.get("failure_kind", "")).lower() in {
        "db_connection_failed", "environment_error",
    }:
        return True
    return _candidate_bitmap_result_should_skip_failure(result)


def _candidate_bitmap_result_edge_key(result: Dict[str, Any]) -> str:
    edge_id = str(result.get("edge_id", "") or result.get("candidate_bitmap_edge_id", ""))
    if edge_id:
        return f"id::{edge_id}"

    raw_idx = result.get("edge_index", result.get("candidate_bitmap_edge_index", ""))
    return f"idx::{raw_idx}"


def _candidate_bitmap_sort_results_success_first(
    results: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    indexed = list(enumerate(results or []))
    indexed.sort(
        key=lambda item: (
            _candidate_bitmap_result_edge_key(item[1]),
            0 if _candidate_bitmap_result_success(item[1]) else 1,
            int(item[0]),
        )
    )
    return [item[1] for item in indexed]


def _candidate_bitmap_stage20_5_update_summary(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    group_counts: Dict[str, int] = {}
    status_by_group: Dict[str, Dict[str, int]] = {}

    soft_priority_active_count = 0
    failed_retry_count = 0
    core_to_extension_count = 0
    soft_priority_bonus_sum = 0.0

    for row in rows or []:
        if not isinstance(row, dict):
            continue

        group = str(row.get("edge_group", "") or "unknown")
        status = str(row.get("status", "") or "unknown")

        group_counts[group] = group_counts.get(group, 0) + 1
        status_map = status_by_group.setdefault(group, {})
        status_map[status] = status_map.get(status, 0) + 1

        if bool(row.get("is_core_to_extension", False)):
            core_to_extension_count += 1

        if bool(row.get("soft_priority_active", False)):
            soft_priority_active_count += 1

        if bool(row.get("failed_retry", False)):
            failed_retry_count += 1

        try:
            soft_priority_bonus_sum += float(row.get("soft_priority_bonus", 0.0) or 0.0)
        except Exception:
            pass

    return {
        "edge_group_counts": group_counts,
        "status_by_edge_group": status_by_group,
        "core_to_extension_count": int(core_to_extension_count),
        "soft_priority_active_count": int(soft_priority_active_count),
        "failed_retry_count": int(failed_retry_count),
        "soft_priority_bonus_sum": float(soft_priority_bonus_sum),
    }


def refresh_candidate_edge_scores_by_pattern_transfer(
    *,
    edge_index_path: Optional[Path] = None,
    edge_scores_path: Optional[Path] = None,
    bitmap_npz_path: Optional[Path] = None,
    graph_path: Optional[Path] = None,
) -> Dict[str, Any]:
    edge_index_file = Path(
        edge_index_path
        or getattr(CANDIDATE_BITMAP_CONFIG, "edge_index_path", CANDIDATE_EDGE_INDEX_PATH)
    )
    edge_scores_file = Path(
        edge_scores_path
        or getattr(CANDIDATE_BITMAP_CONFIG, "edge_scores_path", CANDIDATE_EDGE_SCORES_PATH)
    )
    bitmap_file = Path(
        bitmap_npz_path
        or getattr(CANDIDATE_BITMAP_CONFIG, "bitmap_npz_path", CANDIDATE_BITMAP_NPZ_PATH)
    )
    graph_file = Path(
        graph_path
        or getattr(CANDIDATE_BITMAP_CONFIG, "global_knowledge_graph_path", GLOBAL_KNOWLEDGE_GRAPH_PATH)
    )

    now = time.time()

    similarity_index = _load_similarity_index()
    graph_data = _load_json(graph_file)

    with _candidate_bitmap_file_lock(bitmap_file):
        edge_index_data = _load_json(edge_index_file)
        edge_records = edge_index_data.get("edges", [])
        if not isinstance(edge_records, list):
            raise ValueError(f"The 'edges' field in candidate_edge_index.json must be a list: {edge_index_file}")

        node_info = _build_pattern_node_info(graph_data, edge_records)
        observed_slot_index, token_to_outer_node_ids = _build_observed_slot_pattern_index(
            graph_data,
            node_info,
        )

        removed_legacy_similarity_count = 0
        positive_score_count = 0
        max_score = 0.0
        score_sum = 0.0
        top_samples: List[Dict[str, Any]] = []

        similar_outer_cache: Dict[str, List[Dict[str, Any]]] = {}
        inner_match_cache: Dict[Tuple[str, str], float] = {}

        current_patterns = sorted(
            (str(outer_id), int(slot), str(inner_id))
            for outer_id, slot_map in observed_slot_index.items()
            for slot, inner_ids in slot_map.items()
            for inner_id in inner_ids
        )
        previous_patterns_raw = edge_index_data.get("pattern_transfer_observed_patterns")
        previous_patterns = None
        if isinstance(previous_patterns_raw, list):
            try:
                previous_patterns = {
                    (str(item[0]), int(item[1]), str(item[2]))
                    for item in previous_patterns_raw
                    if isinstance(item, (list, tuple)) and len(item) == 3
                }
            except Exception:
                previous_patterns = None

        current_settings = _pattern_transfer_settings_fingerprint()
        current_files = _pattern_transfer_file_fingerprint(similarity_index)
        current_edge_fingerprint = _pattern_transfer_edge_fingerprint(edge_records)
        incremental_valid = (
            previous_patterns is not None
            and edge_index_data.get("pattern_transfer_cache_version") == "incremental_v1"
            and edge_index_data.get("pattern_transfer_settings") == list(current_settings)
            and edge_index_data.get("pattern_transfer_files") == [list(x) for x in current_files]
            and int(edge_index_data.get("pattern_transfer_cached_edge_count", -1)) == len(edge_records)
            and edge_index_data.get("pattern_transfer_edge_fingerprint") == current_edge_fingerprint
        )
        changed_patterns = set(current_patterns) - (previous_patterns or set())
        removed_patterns = (previous_patterns or set()) - set(current_patterns)

        affected_edge_indices: Optional[set] = None
        if incremental_valid and not removed_patterns:
            affected_edge_indices = set()

            candidate_groups: Dict[Tuple[str, int], List[Tuple[int, Dict[str, Any]]]] = {}
            candidate_outer_tokens = set()
            for idx, candidate in enumerate(edge_records):
                if not isinstance(candidate, dict):
                    continue
                token = normalize_operator_key(
                    candidate.get("outer_token")
                    or _split_graph_node_id(candidate.get("outer_node_id", ""))[0]
                )
                try:
                    slot = int(candidate.get("slot_index", -1))
                except Exception:
                    continue
                if token and slot >= 0:
                    candidate_outer_tokens.add(token)
                    candidate_groups.setdefault((token, slot), []).append((idx, candidate))

            reverse_outer: Dict[str, List[Tuple[str, float]]] = {}
            for candidate_token in candidate_outer_tokens:
                similar_items = similar_outer_cache.get(candidate_token)
                if similar_items is None:
                    similar_items = _get_similar_outer_tokens(candidate_token, similarity_index)
                    similar_outer_cache[candidate_token] = similar_items
                for item in similar_items:
                    observed_token = normalize_operator_key(item.get("op", ""))
                    similarity = _pattern_transfer_clamp01(item.get("sim", 0.0))
                    if observed_token and similarity > 0.0:
                        reverse_outer.setdefault(observed_token, []).append((candidate_token, similarity))
            for outer_id, slot, inner_id in changed_patterns:
                outer_info = node_info.get(outer_id, {})
                changed_outer_token = normalize_operator_key(
                    outer_info.get("token") or _split_graph_node_id(outer_id)[0]
                )
                for candidate_token, _outer_sim in reverse_outer.get(changed_outer_token, []):
                    for idx, candidate in candidate_groups.get((candidate_token, slot), []):
                        candidate_inner = str(candidate.get("inner_node_id", ""))
                        pair = (candidate_inner, inner_id)
                        if pair not in inner_match_cache:
                            inner_match_cache[pair] = _pattern_inner_match_score(
                                candidate_inner,
                                inner_id,
                                node_info=node_info,
                                similarity_index=similarity_index,
                            )
                        if inner_match_cache[pair] > 0.0:
                            affected_edge_indices.add(idx)

        for record_index, record in enumerate(edge_records):
            if not isinstance(record, dict):
                continue

            legacy_similarity_key = "similarity" + "_score"
            if legacy_similarity_key in record:
                record.pop(legacy_similarity_key, None)
                removed_legacy_similarity_count += 1

            cached_score = record.get("pattern_transfer_score")
            use_cached_score = (
                affected_edge_indices is not None
                and record_index not in affected_edge_indices
                and isinstance(cached_score, (int, float))
            )
            if use_cached_score:
                pattern_transfer_score = _pattern_transfer_clamp01(cached_score)
            else:
                pattern_transfer_score = _compute_pattern_transfer_score_for_record(
                    record,
                    observed_slot_index=observed_slot_index,
                    token_to_outer_node_ids=token_to_outer_node_ids,
                    node_info=node_info,
                    similarity_index=similarity_index,
                    similar_outer_cache=similar_outer_cache,
                    inner_match_cache=inner_match_cache,
                )

            level = str(record.get("candidate_level", "try")).lower()
            edge_score = _edge_score_from_pattern_transfer(level, pattern_transfer_score)

            record["pattern_transfer_score"] = float(pattern_transfer_score)
            record["edge_score"] = float(edge_score)

            score_sum += float(pattern_transfer_score)
            max_score = max(max_score, float(pattern_transfer_score))

            if pattern_transfer_score > 0.0:
                positive_score_count += 1
                if len(top_samples) < 50:
                    top_samples.append({
                        "edge_index": record.get("edge_index"),
                        "edge_id": record.get("edge_id"),
                        "candidate_level": record.get("candidate_level"),
                        "pattern_transfer_score": float(pattern_transfer_score),
                        "edge_score": float(edge_score),
                    })

        top_samples.sort(
            key=lambda x: (
                -float(x.get("pattern_transfer_score", 0.0)),
                str(x.get("edge_id", "")),
            )
        )
        top_samples = top_samples[:20]

        organization_report = _apply_candidate_edge_organization(edge_records)

        edge_index_data["edges"] = edge_records
        edge_index_data["last_score_refresh_version"] = "stage18_2_slot_pattern_transfer_v1"
        edge_index_data["last_score_refresh_time"] = now
        edge_index_data["pattern_transfer_cache_version"] = "incremental_v1"
        edge_index_data["pattern_transfer_observed_patterns"] = [list(item) for item in current_patterns]
        edge_index_data["pattern_transfer_settings"] = list(current_settings)
        edge_index_data["pattern_transfer_files"] = [list(item) for item in current_files]
        edge_index_data["pattern_transfer_cached_edge_count"] = int(len(edge_records))
        edge_index_data["pattern_transfer_edge_fingerprint"] = current_edge_fingerprint
        representation_guidance_enabled = _representation_guidance_enabled()
        edge_index_data["representation_guidance_enabled"] = bool(
            representation_guidance_enabled
        )
        edge_index_data["score_strategy"] = (
            "slot_pattern_transfer"
            if representation_guidance_enabled
            else "candidate_level_base_only"
        )
        edge_index_data["score_strategy_note"] = (
            'pattern_transfer_score uses sim(current_outer, observed_outer) and observed inner patterns in the same slot; it does not use direct sim(outer, inner).'
            if representation_guidance_enabled
            else 'Representation transfer scoring is disabled; pattern_transfer_score=0, and edge_score retains only the candidate-level base score.'
        )

        _candidate_bitmap_atomic_write_json(edge_index_file, edge_index_data)
        _candidate_bitmap_atomic_write_jsonl(edge_scores_file, edge_records)

    edge_count = len(edge_records)
    report = {
        "version": "stage18_2_slot_pattern_transfer_v1",
        "time": now,
        "edge_index_path": str(edge_index_file),
        "edge_scores_path": str(edge_scores_file),
        "global_knowledge_graph_path": str(graph_file),
        "representation_guidance_enabled": bool(representation_guidance_enabled),
        "edge_count": int(edge_count),
        "positive_pattern_transfer_score_count": int(positive_score_count),
        "zero_pattern_transfer_score_count": int(edge_count - positive_score_count),
        "max_pattern_transfer_score": float(max_score),
        "mean_pattern_transfer_score": float(score_sum / edge_count) if edge_count > 0 else 0.0,
        "removed_legacy_direct_similarity_field_count": int(removed_legacy_similarity_count),
        "observed_outer_node_count": int(len(observed_slot_index)),
        "observed_outer_token_count": int(len(token_to_outer_node_ids)),
        "operator_similarity_loaded": bool(similarity_index is not None and getattr(similarity_index, "is_loaded", False)),
        "candidate_edge_organization": organization_report,
        "incremental_refresh": bool(affected_edge_indices is not None),
        "recomputed_edge_count": int(
            len(edge_records) if affected_edge_indices is None else len(affected_edge_indices)
        ),
        "top_positive_samples": top_samples,
        "notes": [
            'This refresh does not rebuild the candidate edge universe.',
            'This refresh does not modify observed_bitmap / failed_bitmap / probing_bitmap.',
            'pattern_transfer_score is based on similar observed outer operators and accepted inner nodes in the same slot.',
            'Direct sim(outer, inner) is intentionally not used here.',
            'Legacy direct outer-inner similarity fields in candidate_edge_index records are removed during refresh.',
        ],
    }

    return report


def update_candidate_bitmaps_from_results(
    results: List[Dict[str, Any]],
    *,
    edge_index_path: Optional[Path] = None,
    bitmap_npz_path: Optional[Path] = None,
    update_log_path: Optional[Path] = None,
    update_report_path: Optional[Path] = None,
    missing_edges_path: Optional[Path] = None,
) -> Dict[str, Any]:
    clean_results = _candidate_bitmap_sort_results_success_first(
        [x for x in (results or []) if isinstance(x, dict)]
    )

    edge_index_file = Path(
        edge_index_path
        or getattr(CANDIDATE_BITMAP_CONFIG, "edge_index_path", CANDIDATE_EDGE_INDEX_PATH)
    )
    bitmap_file = Path(
        bitmap_npz_path
        or getattr(CANDIDATE_BITMAP_CONFIG, "bitmap_npz_path", CANDIDATE_BITMAP_NPZ_PATH)
    )
    log_file = Path(
        update_log_path
        or getattr(CANDIDATE_BITMAP_CONFIG, "bitmap_update_log_path", CANDIDATE_BITMAP_UPDATE_LOG_PATH)
    )
    report_file = Path(
        update_report_path
        or getattr(CANDIDATE_BITMAP_CONFIG, "bitmap_update_report_path", CANDIDATE_BITMAP_UPDATE_REPORT_PATH)
    )
    missing_file = Path(
        missing_edges_path
        or getattr(CANDIDATE_BITMAP_CONFIG, "missing_edges_path", CANDIDATE_MISSING_EDGES_PATH)
    )

    now = time.time()

    if not clean_results:
        report = {
            "version": "stage17_6b_candidate_bitmap_update_v1",
            "time": now,
            "input_result_count": 0,
            "applied_count": 0,
            "skipped_count": 0,
            "note": 'No candidate bitmap results to update',
        }
        report_file.parent.mkdir(parents=True, exist_ok=True)
        _candidate_bitmap_atomic_write_json(report_file, report)
        return report

    with _candidate_bitmap_file_lock(bitmap_file):
        edge_index_data = _load_json(edge_index_file)
        edge_records = edge_index_data.get("edges", [])
        if not isinstance(edge_records, list):
            raise ValueError(f"The 'edges' field in candidate_edge_index.json must be a list: {edge_index_file}")

        n = len(edge_records)
        bitmaps = _load_npz_bitmaps(bitmap_file, n)

        observed_bitmap = np.asarray(bitmaps.get("observed_bitmap", np.zeros(n)), dtype=np.uint8)
        failed_bitmap = np.asarray(bitmaps.get("failed_bitmap", np.zeros(n)), dtype=np.uint8)
        probing_bitmap = np.asarray(bitmaps.get("probing_bitmap", np.zeros(n)), dtype=np.uint8)

        edge_id_to_index: Dict[str, int] = {}
        for record in edge_records:
            try:
                idx = int(record.get("edge_index", -1))
            except Exception:
                continue
            if idx < 0 or idx >= n:
                continue
            edge_id = str(record.get("edge_id", ""))
            if edge_id:
                edge_id_to_index[edge_id] = idx

        applied_rows: List[Dict[str, Any]] = []
        skipped_rows: List[Dict[str, Any]] = []

        observed_added_count = 0
        observed_existing_count = 0
        observed_failure_ignored_count = 0
        context_inconclusive_count = 0
        failure_count_added = 0
        failed_disabled_count = 0
        probing_cleared_count = 0

        for result in clean_results:
            edge_id = str(result.get("edge_id", "") or result.get("candidate_bitmap_edge_id", ""))
            raw_idx = result.get("edge_index", result.get("candidate_bitmap_edge_index", None))

            edge_index: Optional[int] = None
            try:
                if raw_idx is not None:
                    edge_index = int(raw_idx)
            except Exception:
                edge_index = None

            if edge_index is None or edge_index < 0 or edge_index >= n:
                edge_index = edge_id_to_index.get(edge_id)

            if edge_index is None or edge_index < 0 or edge_index >= n:
                skipped_rows.append({
                    "time": now,
                    "reason": 'Edge index not found',
                    "edge_id": edge_id,
                    "raw_edge_index": raw_idx,
                    "result": result,
                })
                continue

            record = edge_records[edge_index]
            record_edge_id = str(record.get("edge_id", ""))
            if edge_id and record_edge_id and edge_id != record_edge_id:
                skipped_rows.append({
                    "time": now,
                    "reason": 'edge_id mismatch',
                    "edge_id": edge_id,
                    "record_edge_id": record_edge_id,
                    "edge_index": edge_index,
                    "result": result,
                })
                continue

            if not edge_id:
                edge_id = record_edge_id

            was_probing = int(probing_bitmap[edge_index]) > 0
            if was_probing:
                probing_cleared_count += 1
            probing_bitmap[edge_index] = 0

            record["attempt_count"] = int(record.get("attempt_count", 0)) + 1
            record["last_update_time"] = now
            record["last_predicate"] = str(result.get("predicate", result.get("predicate_template", "")) or "")
            record["last_completed_sql"] = str(result.get("completed_sql", "") or "")
            record["last_db_success"] = bool(_candidate_bitmap_as_bool(result.get("db_success")))
            record["last_parse_success"] = bool(_candidate_bitmap_as_bool(result.get("parse_success")))
            record["last_target_edge_hit"] = bool(_candidate_bitmap_result_target_hit(result))
            record["last_error"] = str(result.get("db_error", "") or result.get("parse_error", "") or "")

            db_success = bool(_candidate_bitmap_as_bool(result.get("db_success")))
            parse_success = bool(_candidate_bitmap_as_bool(result.get("parse_success")))
            target_edge_hit = bool(_candidate_bitmap_result_target_hit(result))
            predicate_text = str(result.get("predicate", result.get("predicate_template", "")) or "")

            is_success = _candidate_bitmap_result_success(result)
            skip_failure = _candidate_bitmap_result_should_skip_failure(result)
            if completion_retry_needed(result):
                advance_completion_variant(record)
            if str(result.get("failure_kind", "") or "").lower() == "context_inconclusive":
                context_inconclusive_count += 1

            if (not is_success) and (not skip_failure) and predicate_text:
                _candidate_bitmap_record_tried_probe_signature(record, predicate_text)

            if is_success:
                was_observed = int(observed_bitmap[edge_index]) > 0
                observed_bitmap[edge_index] = 1
                failed_bitmap[edge_index] = 0

                record["success_count"] = int(record.get("success_count", 0)) + 1
                record["fail_count"] = 0
                record["target_db_failure_signatures"] = []

                if not was_observed:
                    record["new_edge_count"] = int(record.get("new_edge_count", 0)) + 1
                    observed_added_count += 1
                else:
                    observed_existing_count += 1

                record["last_update_status"] = "observed"

                applied_status = "observed"

            elif skip_failure:
                record["last_update_status"] = "skipped_environment_error"
                applied_status = "skipped_environment_error"

            elif int(observed_bitmap[edge_index]) > 0:
                observed_failure_ignored_count += 1
                failed_bitmap[edge_index] = 0
                record["last_update_status"] = "observed_failure_ignored"
                applied_status = "observed_failure_ignored"

            elif (not db_success) and parse_success and target_edge_hit:
                signature = _candidate_bitmap_normalize_failure_signature(predicate_text)
                if not signature:
                    record["probe_invalid_count"] = int(record.get("probe_invalid_count", 0)) + 1
                    record["last_update_status"] = "probe_construct_failed"
                    applied_status = "probe_construct_failed"
                else:
                    raw_signatures = record.get("target_db_failure_signatures", [])
                    if isinstance(raw_signatures, list):
                        signatures = [str(x) for x in raw_signatures if str(x)]
                    else:
                        signatures = []

                    if signature not in signatures:
                        signatures.append(signature)
                        record["target_db_failure_signatures"] = signatures
                        fail_count = len(signatures)
                        record["fail_count"] = fail_count
                        failure_count_added += 1

                        threshold = _candidate_bitmap_fail_threshold(str(record.get("candidate_level", "strict")))
                        if fail_count >= threshold:
                            if int(failed_bitmap[edge_index]) <= 0:
                                failed_disabled_count += 1
                            failed_bitmap[edge_index] = 1
                            record["last_update_status"] = "target_db_failed_disabled"
                            applied_status = "target_db_failed_disabled"
                        else:
                            record["last_update_status"] = "target_db_failed_counted"
                            applied_status = "target_db_failed_counted"
                    else:
                        record["fail_count"] = len(signatures)
                        record["last_update_status"] = "target_db_failed_duplicate"
                        applied_status = "target_db_failed_duplicate"

            else:
                record["probe_invalid_count"] = int(record.get("probe_invalid_count", 0)) + 1

                if not predicate_text:
                    record["construct_failed_count"] = int(record.get("construct_failed_count", 0)) + 1
                    record["last_update_status"] = "probe_construct_failed"
                    applied_status = "probe_construct_failed"
                elif db_success:
                    record["target_miss_count"] = int(record.get("target_miss_count", 0)) + int(not target_edge_hit)
                    record["parse_invalid_count"] = int(record.get("parse_invalid_count", 0)) + int(not parse_success)
                    record["last_update_status"] = "probe_invalid_db_success_no_target"
                    applied_status = "probe_invalid_db_success_no_target"
                else:
                    record["target_miss_count"] = int(record.get("target_miss_count", 0)) + int(parse_success and not target_edge_hit)
                    record["parse_invalid_count"] = int(record.get("parse_invalid_count", 0)) + int(not parse_success)
                    record["last_update_status"] = "probe_invalid_db_failed_no_target"
                    applied_status = "probe_invalid_db_failed_no_target"

            applied_rows.append({
                "time": now,
                "edge_index": edge_index,
                "edge_id": edge_id,
                "status": applied_status,
                "candidate_level": str(record.get("candidate_level", "")),

                "edge_group": str(result.get("edge_group", record.get("edge_group", "")) or ""),
                "is_core_to_extension": bool(
                    result.get("is_core_to_extension", record.get("is_core_to_extension", False))
                ),
                "soft_priority_active": bool(result.get("soft_priority_active", False)),
                "soft_priority_bonus": float(result.get("soft_priority_bonus", 0.0) or 0.0),
                "failed_retry": bool(result.get("failed_retry", False)),
                "success_count": int(record.get("success_count", 0)),
                "fail_count": int(record.get("fail_count", 0)),
                "new_edge_count": int(record.get("new_edge_count", 0)),
                "db_success": db_success,
                "parse_success": parse_success,
                "target_edge_hit": target_edge_hit,
                "predicate": predicate_text,
                "db_error": result.get("db_error"),
                "parse_error": result.get("parse_error"),

                "failure_kind": str(result.get("failure_kind", "") or ""),
                "downstream_construct_failed": bool(
                    _candidate_bitmap_as_bool(result.get("downstream_construct_failed"))
                ),
                "downstream_failure_reason": str(
                    result.get("downstream_failure_reason", "") or ""
                ),
                "inner_ready": bool(_candidate_bitmap_as_bool(result.get("inner_ready"))),
                "safe_closure_used": bool(
                    _candidate_bitmap_as_bool(result.get("safe_closure_used"))
                ),
                "global_new_edge": bool(
                    _candidate_bitmap_as_bool(result.get("global_new_edge"))
                ),
                "step": result.get("step"),
                "source": result.get("source", "candidate_bitmap_callback"),
            })

        bitmaps["observed_bitmap"] = observed_bitmap
        bitmaps["failed_bitmap"] = failed_bitmap
        bitmaps["probing_bitmap"] = probing_bitmap

        _refresh_missing_and_available_bitmaps(bitmaps)

        _candidate_bitmap_atomic_save_npz(bitmap_file, bitmaps)

        edge_index_data["edges"] = edge_records
        edge_index_data["last_update_version"] = "stage17_6b_candidate_bitmap_update_v1"
        edge_index_data["last_update_time"] = now
        _candidate_bitmap_atomic_write_json(edge_index_file, edge_index_data)

        missing_edge_count_written = _write_missing_edges(
            edge_records,
            np.asarray(bitmaps.get("missing_candidate_bitmap", np.zeros(n)), dtype=np.uint8),
            missing_file,
        )

        _candidate_bitmap_append_jsonl(log_file, applied_rows + skipped_rows)

        stage20_5_update_summary = _candidate_bitmap_stage20_5_update_summary(applied_rows)

        report = {
            "version": "stage17_6b_candidate_bitmap_update_v1",
            "time": now,
            "edge_index_path": str(edge_index_file),
            "bitmap_npz_path": str(bitmap_file),
            "update_log_path": str(log_file),
            "update_report_path": str(report_file),
            "missing_edges_path": str(missing_file),
            "input_result_count": len(clean_results),
            "applied_count": len(applied_rows),
            "skipped_count": len(skipped_rows),
            "observed_added_count": int(observed_added_count),
            "observed_existing_count": int(observed_existing_count),
            "observed_failure_ignored_count": int(observed_failure_ignored_count),
            "context_inconclusive_count": int(context_inconclusive_count),
            "failure_count_added": int(failure_count_added),
            "failed_disabled_count": int(failed_disabled_count),
            "probing_cleared_count": int(probing_cleared_count),
            "missing_edge_count_written": int(missing_edge_count_written),
            "bitmap_counts": {
                name: int(np.asarray(bitmap).sum())
                for name, bitmap in bitmaps.items()
            },
            "by_level": {
                "observed": _level_counts_from_bitmap(edge_records, observed_bitmap),
                "failed": _level_counts_from_bitmap(edge_records, failed_bitmap),
                "missing": _level_counts_from_bitmap(
                    edge_records,
                    np.asarray(bitmaps.get("missing_candidate_bitmap", np.zeros(n)), dtype=np.uint8),
                ),
                "available_initial": _level_counts_from_bitmap(
                    edge_records,
                    np.asarray(bitmaps.get("available_initial_bitmap", np.zeros(n)), dtype=np.uint8),
                ),
            },
            "stage20_5_update_summary": stage20_5_update_summary,
            "applied_sample": applied_rows[:20],
            "skipped_sample": skipped_rows[:20],
            "notes": [
                'observed means DB success, parser success, and a target-edge hit.',
                'Only distinct predicates with DB failure, parser success, and a target-edge hit increase fail_count; failed_bitmap is set at the shared threshold N.',
                'Each writeback clears probing_bitmap and recalculates missing / available bitmaps.',
                'db_connection_failed / environment_error are not counted in failed_bitmap.',
                'DB failures after temporary completion of non-target slots are recorded as context_inconclusive and do not count as target-edge failure evidence.',
                'If an edge is already observed, subsequent failures are recorded as observed_failure_ignored.',
                'Results are grouped by edge, with successful results applied first to avoid contamination within a batch.',
                'stage20_5_update_summary is logging metadata only and does not affect bitmap updates.',
            ],
        }

        _candidate_bitmap_atomic_write_json(report_file, report)

    return report


def build_probe_graph_data(
    formal_graph_data: Dict[str, Any],
    edge_records: List[Dict[str, Any]],
    bitmaps: Dict[str, np.ndarray],
    *,
    probe_edge_id: Optional[str] = None,
    probe_edge_index: Optional[int] = None,
    allow_observed: bool = False,
    graph_path: Optional[Path] = None,
    probe_graph_path: Optional[Path] = None,
    probe_report_path: Optional[Path] = None,
    bitmap_npz_path: Optional[Path] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    if not isinstance(formal_graph_data, dict):
        raise ValueError('formal_graph_data must be a dict')

    formal_nodes = formal_graph_data.get("nodes", {})
    if not isinstance(formal_nodes, dict):
        raise ValueError('The nodes field in global_knowledge_graph.json must be a dict')

    formal_node_ids = set(str(node_id) for node_id in formal_nodes.keys())
    formal_node_count = len(formal_nodes)
    formal_edge_count = _graph_edge_count_from_data(formal_graph_data)

    target_record = _select_probe_edge_record(
        edge_records,
        bitmaps,
        formal_node_ids=formal_node_ids,
        probe_edge_id=probe_edge_id,
        probe_edge_index=probe_edge_index,
        allow_observed=allow_observed,
    )

    target_edge_index = int(target_record["edge_index"])
    outer_node_id = str(target_record["outer_node_id"])
    inner_node_id = str(target_record["inner_node_id"])
    slot_index = int(target_record["slot_index"])

    probe_graph_data = {
        k: v for k, v in formal_graph_data.items()
        if k not in {"nodes", "roots", "probe_metadata"}
    }

    nodes = dict(formal_nodes)
    probe_graph_data["nodes"] = nodes

    formal_roots = formal_graph_data.get("roots", [])
    roots = list(formal_roots) if isinstance(formal_roots, list) else []
    probe_graph_data["roots"] = roots

    target_edge_existed_before = _graph_has_edge(
        formal_graph_data,
        outer_node_id,
        slot_index,
        inner_node_id,
    )

    if target_edge_existed_before and not allow_observed:
        raise ValueError(f"The target edge already exists in the formal graph: {target_record.get('edge_id')}")

    outer_node_preexisting = outer_node_id in nodes
    inner_node_preexisting = inner_node_id in nodes

    created_nodes: List[str] = []

    if outer_node_id in nodes and isinstance(nodes.get(outer_node_id), dict):
        outer_payload_copy = dict(nodes[outer_node_id])
        raw_slots = outer_payload_copy.get("slots", [])
        if isinstance(raw_slots, list):
            outer_payload_copy["slots"] = [
                list(slot_items) if isinstance(slot_items, list) else []
                for slot_items in raw_slots
            ]
        else:
            outer_payload_copy["slots"] = []
        nodes[outer_node_id] = outer_payload_copy

    if outer_node_id not in nodes:
        nodes[outer_node_id] = _make_probe_node_payload(
            node_id=outer_node_id,
            token=str(target_record.get("outer_token", "")),
            return_type=str(target_record.get("outer_return_type", "")),
            edge_records=edge_records,
            force_non_terminal=True,
            min_slot_count=slot_index + 1,
        )
        created_nodes.append(outer_node_id)

    if inner_node_id not in nodes:
        nodes[inner_node_id] = _make_probe_node_payload(
            node_id=inner_node_id,
            token=str(target_record.get("inner_token", "")),
            return_type=str(target_record.get("inner_return_type", "")),
            edge_records=edge_records,
            force_non_terminal=False,
            min_slot_count=0,
        )
        created_nodes.append(inner_node_id)

    outer_payload = nodes[outer_node_id]
    if not isinstance(outer_payload, dict):
        raise ValueError(f"probe outer node payload invalid: {outer_node_id}")

    _ensure_node_slots(outer_payload, slot_index + 1)

    slot_items = outer_payload["slots"][slot_index]
    if not isinstance(slot_items, list):
        slot_items = []
        outer_payload["slots"][slot_index] = slot_items

    added_target_edge = False
    if inner_node_id not in slot_items:
        slot_items.append(inner_node_id)
        added_target_edge = True

    if outer_node_id not in roots:
        roots.append(outer_node_id)

    probe_graph_data["probe_metadata"] = {
        "version": "stage17_3_probe_graph_v1",
        "source_global_knowledge_graph_path": str(graph_path or "<memory>"),
        "target_edge": target_record,
        "target_edge_existed_in_formal_graph_before": target_edge_existed_before,
        "added_target_edge_to_probe_graph": added_target_edge,
        "created_nodes": created_nodes,
        "note": 'In-memory probe_graph, not the formal global_knowledge_graph.json',
    }

    probe_node_count = len(probe_graph_data.get("nodes", {}))
    probe_edge_count = _graph_edge_count_from_data(probe_graph_data)

    report = {
        "version": "stage17_3_probe_graph_v1",
        "formal_graph_path": str(graph_path or "<memory>"),
        "probe_graph_path": str(probe_graph_path or "<memory>"),
        "probe_report_path": str(probe_report_path or "<memory>"),
        "bitmap_npz_path": str(bitmap_npz_path or "<memory>"),
        "target_edge": {
            "edge_index": target_edge_index,
            "edge_id": str(target_record.get("edge_id", "")),
            "candidate_level": str(target_record.get("candidate_level", "")),
            "edge_score": float(target_record.get("edge_score", 0.0)),
            "pattern_transfer_score": float(target_record.get("pattern_transfer_score", 0.0)),
            "support_rank": _probe_edge_support_rank(
                target_record,
                formal_node_ids=formal_node_ids,
                edge_records=edge_records,
            ),
            "outer_node_id": outer_node_id,
            "slot_index": slot_index,
            "inner_node_id": inner_node_id,
        },
        "formal_graph": {
            "node_count": formal_node_count,
            "edge_count": formal_edge_count,
            "target_edge_existed_before": target_edge_existed_before,
            "outer_node_preexisting": outer_node_preexisting,
            "inner_node_preexisting": inner_node_preexisting,
        },
        "probe_graph": {
            "node_count": probe_node_count,
            "edge_count": probe_edge_count,
            "created_node_count": len(created_nodes),
            "created_nodes": created_nodes,
            "added_target_edge": added_target_edge,
            "root_added_or_present": outer_node_id in probe_graph_data.get("roots", []),
        },
        "bitmap": {
            "mark_probing": False,
            "probing_bitmap_updated": False,
        },
        "runtime": {
            "persisted_to_disk": False,
        },
        "notes": [
            'probe_graph is a temporary copy of the formal graph with one target edge inserted.',
            'This function does not modify global_knowledge_graph.json.',
            'Stage 17-11 RL uses in-memory probe_graph to avoid growth of runtime_probe files.',
        ],
    }

    return probe_graph_data, report


def build_and_save_probe_graph(
    graph_path: Optional[Path] = None,
    *,
    probe_edge_id: Optional[str] = None,
    probe_edge_index: Optional[int] = None,
    probe_graph_path: Optional[Path] = None,
    probe_report_path: Optional[Path] = None,
    allow_observed: bool = False,
    mark_probing: bool = False,
) -> Dict[str, Any]:
    edge_index_path = Path(getattr(CANDIDATE_BITMAP_CONFIG, "edge_index_path", CANDIDATE_EDGE_INDEX_PATH))
    bitmap_npz_path = Path(getattr(CANDIDATE_BITMAP_CONFIG, "bitmap_npz_path", CANDIDATE_BITMAP_NPZ_PATH))
    graph_file = Path(
        graph_path
        or getattr(CANDIDATE_BITMAP_CONFIG, "global_knowledge_graph_path", GLOBAL_KNOWLEDGE_GRAPH_PATH)
    )

    output_graph_path = Path(
        probe_graph_path
        or getattr(CANDIDATE_BITMAP_CONFIG, "probe_graph_path", CANDIDATE_PROBE_GRAPH_PATH)
    )
    output_report_path = Path(
        probe_report_path
        or getattr(CANDIDATE_BITMAP_CONFIG, "probe_report_path", CANDIDATE_PROBE_REPORT_PATH)
    )

    edge_records = _load_edge_index_records(edge_index_path)
    bitmaps = _load_npz_bitmaps(bitmap_npz_path, len(edge_records))

    formal_graph_data = _load_json(graph_file)
    formal_nodes = formal_graph_data.get("nodes", {})
    if not isinstance(formal_nodes, dict):
        raise ValueError(f"The 'nodes' field in global_knowledge_graph.json must be a dict: {graph_file}")

    formal_node_ids = set(str(node_id) for node_id in formal_nodes.keys())
    formal_node_count = len(formal_nodes)
    formal_edge_count = _graph_edge_count_from_data(formal_graph_data)

    target_record = _select_probe_edge_record(
        edge_records,
        bitmaps,
        formal_node_ids=formal_node_ids,
        probe_edge_id=probe_edge_id,
        probe_edge_index=probe_edge_index,
        allow_observed=allow_observed,
    )

    target_edge_index = int(target_record["edge_index"])
    outer_node_id = str(target_record["outer_node_id"])
    inner_node_id = str(target_record["inner_node_id"])
    slot_index = int(target_record["slot_index"])

    probe_graph_data = json.loads(json.dumps(formal_graph_data, ensure_ascii=False))

    nodes = probe_graph_data.setdefault("nodes", {})
    if not isinstance(nodes, dict):
        raise ValueError(f"The 'nodes' field in global_knowledge_graph.json must be a dict: {graph_file}")

    roots = probe_graph_data.setdefault("roots", [])
    if not isinstance(roots, list):
        roots = []
        probe_graph_data["roots"] = roots

    target_edge_existed_before = _graph_has_edge(
        formal_graph_data,
        outer_node_id,
        slot_index,
        inner_node_id,
    )

    if target_edge_existed_before and not allow_observed:
        raise ValueError(
            f"The target edge already exists in the formal graph and should not be used as a probe edge: {target_record.get('edge_id')}"
        )

    outer_node_preexisting = outer_node_id in nodes
    inner_node_preexisting = inner_node_id in nodes

    created_nodes: List[str] = []

    if outer_node_id not in nodes:
        nodes[outer_node_id] = _make_probe_node_payload(
            node_id=outer_node_id,
            token=str(target_record.get("outer_token", "")),
            return_type=str(target_record.get("outer_return_type", "")),
            edge_records=edge_records,
            force_non_terminal=True,
            min_slot_count=slot_index + 1,
        )
        created_nodes.append(outer_node_id)

    if inner_node_id not in nodes:
        nodes[inner_node_id] = _make_probe_node_payload(
            node_id=inner_node_id,
            token=str(target_record.get("inner_token", "")),
            return_type=str(target_record.get("inner_return_type", "")),
            edge_records=edge_records,
            force_non_terminal=False,
            min_slot_count=0,
        )
        created_nodes.append(inner_node_id)

    outer_payload = nodes[outer_node_id]
    if not isinstance(outer_payload, dict):
        raise ValueError(f"probe outer node payload is invalid: {outer_node_id}")

    _ensure_node_slots(outer_payload, slot_index + 1)

    slot_items = outer_payload["slots"][slot_index]
    if not isinstance(slot_items, list):
        slot_items = []
        outer_payload["slots"][slot_index] = slot_items

    added_target_edge = False
    if inner_node_id not in slot_items:
        slot_items.append(inner_node_id)
        added_target_edge = True

    if outer_node_id not in roots:
        roots.append(outer_node_id)

    probe_graph_data["probe_metadata"] = {
        "version": "stage17_3_probe_graph_v1",
        "source_global_knowledge_graph_path": str(graph_file),
        "target_edge": target_record,
        "target_edge_existed_in_formal_graph_before": target_edge_existed_before,
        "added_target_edge_to_probe_graph": added_target_edge,
        "created_nodes": created_nodes,
        "note": 'This file is a temporary probe_graph, not the formal global_knowledge_graph.json.',
    }

    output_graph_path.parent.mkdir(parents=True, exist_ok=True)
    with output_graph_path.open("w", encoding="utf-8") as f:
        json.dump(probe_graph_data, f, indent=2, ensure_ascii=False)

    if mark_probing:
        probing_bitmap = np.asarray(bitmaps.get("probing_bitmap", np.zeros(len(edge_records))), dtype=np.uint8)
        if len(probing_bitmap) != len(edge_records):
            probing_bitmap = np.zeros(len(edge_records), dtype=np.uint8)

        probing_bitmap[target_edge_index] = 1
        bitmaps["probing_bitmap"] = probing_bitmap
        _refresh_missing_and_available_bitmaps(bitmaps)
        np.savez_compressed(bitmap_npz_path, **bitmaps)

    probe_node_count = len(probe_graph_data.get("nodes", {}))
    probe_edge_count = _graph_edge_count_from_data(probe_graph_data)

    report = {
        "version": "stage17_3_probe_graph_v1",
        "formal_graph_path": str(graph_file),
        "probe_graph_path": str(output_graph_path),
        "probe_report_path": str(output_report_path),
        "bitmap_npz_path": str(bitmap_npz_path),
        "target_edge": {
            "edge_index": target_edge_index,
            "edge_id": str(target_record.get("edge_id", "")),
            "candidate_level": str(target_record.get("candidate_level", "")),
            "edge_score": float(target_record.get("edge_score", 0.0)),
            "pattern_transfer_score": float(target_record.get("pattern_transfer_score", 0.0)),
            "support_rank": _probe_edge_support_rank(
                target_record,
                formal_node_ids=formal_node_ids,
                edge_records=edge_records,
            ),
            "outer_node_id": outer_node_id,
            "slot_index": slot_index,
            "inner_node_id": inner_node_id,
        },
        "formal_graph": {
            "node_count": formal_node_count,
            "edge_count": formal_edge_count,
            "target_edge_existed_before": target_edge_existed_before,
            "outer_node_preexisting": outer_node_preexisting,
            "inner_node_preexisting": inner_node_preexisting,
        },
        "probe_graph": {
            "node_count": probe_node_count,
            "edge_count": probe_edge_count,
            "created_node_count": len(created_nodes),
            "created_nodes": created_nodes,
            "added_target_edge": added_target_edge,
            "root_added_or_present": outer_node_id in probe_graph_data.get("roots", []),
        },
        "bitmap": {
            "mark_probing": bool(mark_probing),
            "probing_bitmap_updated": bool(mark_probing),
        },
        "notes": [
            'probe_graph is a temporary copy of the formal graph with only the current target edge added.',
            'This function does not modify pkg/knowledge_graph/global_knowledge_graph.json.',
            'If the target edge references missing nodes, skeleton nodes are created only in probe_graph.',
            'When no edge is explicitly specified, missing edges supported by existing formal nodes are preferred.',
            'Predicates built in later stages must contain the target edge in probe_graph.',
        ],
    }

    output_report_path.parent.mkdir(parents=True, exist_ok=True)
    with output_report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    return report


def _load_probe_graph_data(probe_graph_path: Path) -> Dict[str, Any]:
    path = Path(probe_graph_path)
    if not path.exists():
        raise FileNotFoundError(f'probe_graph.json is missing; run --build-probe first: {path}')

    data = _load_json(path)
    nodes = data.get("nodes", {})
    if not isinstance(nodes, dict):
        raise ValueError(f'The nodes field in probe_graph.json must be a dict: {path}')

    metadata = data.get("probe_metadata", {})
    if not isinstance(metadata, dict):
        raise ValueError(f'probe_graph.json has no probe_metadata; cannot determine the target edge: {path}')

    target_edge = metadata.get("target_edge", {})
    if not isinstance(target_edge, dict):
        raise ValueError(f'probe_metadata.target_edge in probe_graph.json must be a dict: {path}')

    return data


def _edge_record_lookup(edge_records: List[Dict[str, Any]]) -> Dict[Tuple[str, int, str], Dict[str, Any]]:
    lookup: Dict[Tuple[str, int, str], Dict[str, Any]] = {}

    for record in edge_records:
        try:
            slot_index = int(record.get("slot_index", -1))
        except Exception:
            continue

        if slot_index < 0:
            continue

        outer_node_id = str(record.get("outer_node_id", ""))
        inner_node_id = str(record.get("inner_node_id", ""))

        if not outer_node_id or not inner_node_id:
            continue

        lookup[(outer_node_id, slot_index, inner_node_id)] = record

    return lookup


def _get_node_payload(graph_data: Dict[str, Any], node_id: str) -> Optional[Dict[str, Any]]:
    nodes = graph_data.get("nodes", {})
    if not isinstance(nodes, dict):
        return None

    payload = nodes.get(node_id)
    if isinstance(payload, dict):
        return payload

    return None


def _node_is_terminal(payload: Dict[str, Any]) -> bool:
    slots = payload.get("slots", [])
    if not isinstance(slots, list):
        slots = []

    return bool(payload.get("is_terminal", False)) or len(slots) == 0


def _terminal_to_probe_sql(token: str, return_type: str) -> str:
    token_text = str(token).strip()
    token_upper = token_text.upper()
    return_upper = str(return_type).strip().upper()

    sqlserver = _is_sqlserver()
    if sqlserver:
        terminals = _configured_probe_terminal_tokens()
        if token_upper not in terminals or terminals[token_upper] != return_upper:
            raise ValueError(f"SQL Server probe terminal is not declared with this type: {token_text}::{return_upper}")

    if token_upper == "NULL":
        return "NULL"

    if token_upper == "TRUE":
        return "TRUE"

    if token_upper == "FALSE":
        return "FALSE"

    if token_upper in _callable_terminal_token_set():
        return token_text

    if token_upper in _bare_terminal_token_set():
        return token_upper

    if token_upper.startswith("COL_"):
        return token_upper

    if token_upper.startswith("HOLE"):
        raise ValueError(f'The deprecated HOLE token should not appear in probe SQL: {token_text}')

    if token_upper.startswith("VALUE_"):
        if sqlserver:
            profile_key = {"INT": "cmp", "FLOAT": "float", "STRING": "string"}.get(return_upper)
            value = (_load_sql_render_config().get("literal_profiles", {}) or {}).get(profile_key)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"SQL Server probe literal has no configured rendering: {token_text}")
            return value
        if "STRING" in token_upper or return_upper in {"STRING", "T_STR"}:
            return "'probe_str'"
        if "FLOAT" in token_upper or return_upper in {"FLOAT", "DOUBLE", "DECIMAL", "REAL"}:
            return "1.5"
        if "BOOL" in token_upper or return_upper == "T_BOOL":
            return "TRUE"
        return "1"

    if sqlserver:
        raise ValueError(f"SQL Server probe terminal has no configured rendering: {token_text}")

    if return_upper == "T_BOOL":
        return "TRUE"

    if return_upper in {"STRING", "T_STR"}:
        return "'probe_str'"

    return "1"


def _default_safe_terminal_for_required_type(required_type: Any) -> Tuple[str, str]:
    type_text = str(required_type or "").strip().upper()

    if _is_sqlserver():
        terminals = _configured_probe_terminal_tokens()
        aliases = _load_candidate_bitmap_rule_behavior().get("coarse_type_aliases", {}) or {}
        required = _normalize_type(type_text, aliases)
        if required in {"BOOL", "BOOLEAN"}:
            required = "T_BOOL"
        elif required == "T_ANY":
            required = "T_CMP"

        defaults = ["TRUE", "FALSE", "VALUE_TYPICAL_INT", "VALUE_TYPICAL_FLOAT", "VALUE_TYPICAL_STRING", "NULL"]
        bool_default = str((_load_sql_render_config().get("literal_profiles", {}) or {}).get("bool", "")).upper()
        if bool_default in {"TRUE", "FALSE"}:
            defaults.remove(bool_default)
            defaults.insert(0, bool_default)
        if not type_text.startswith("T_"):
            defaults.sort(key=lambda token: terminals.get(token) != type_text)
        for token in defaults:
            declared_type = terminals.get(token)
            if declared_type and _normalize_type(declared_type, aliases) == required:
                return _terminal_to_probe_sql(token, declared_type), _node_id(token, declared_type)
        raise ValueError(f"SQL Server rules have no safe terminal for required type {type_text!r}.")

    if type_text in {"T_BOOL", "BOOL", "BOOLEAN"}:
        return _terminal_to_probe_sql("TRUE", "T_BOOL"), "TRUE::T_BOOL"

    if type_text in {"T_STR", "STRING"}:
        return _terminal_to_probe_sql("VALUE_TYPICAL_STRING", "STRING"), "VALUE_TYPICAL_STRING::STRING"

    if type_text in {"FLOAT", "DOUBLE", "DECIMAL", "REAL"}:
        return _terminal_to_probe_sql("VALUE_TYPICAL_FLOAT", "FLOAT"), "VALUE_TYPICAL_FLOAT::FLOAT"

    if type_text in {"T_CMP", "T_ANY", "INT", "INTEGER"}:
        return _terminal_to_probe_sql("VALUE_TYPICAL_INT", "INT"), "VALUE_TYPICAL_INT::INT"

    return _terminal_to_probe_sql("VALUE_TYPICAL_INT", "INT"), "VALUE_TYPICAL_INT::INT"


def _slot_required_types_from_edge_records(
    node_id: str,
    edge_records: List[Dict[str, Any]],
) -> List[str]:
    slot_to_type: Dict[int, str] = {}

    for record in edge_records:
        if str(record.get("outer_node_id", "")) != str(node_id):
            continue

        try:
            slot_index = int(record.get("slot_index", -1))
        except Exception:
            continue

        if slot_index < 0:
            continue

        slot_to_type.setdefault(
            slot_index,
            str(record.get("slot_required_type", "T_ANY") or "T_ANY"),
        )

    if not slot_to_type:
        return []

    return [
        slot_to_type.get(slot_index, "T_ANY")
        for slot_index in range(max(slot_to_type.keys()) + 1)
    ]


def _target_inner_safe_closure_disabled(token: Any) -> bool:
    token_norm = normalize_operator_key(str(token or ""))
    return token_norm in {
        "IN",
        "NOT IN",
        "IS",
        "IS NULL",
        "IS NOT NULL",
        "IS TRUE",
        "IS FALSE",
    }


_CANDIDATE_BITMAP_RULE_BEHAVIOR_CACHE: Optional[Tuple[str, int, Dict[str, Any]]] = None


def _load_candidate_bitmap_rule_behavior() -> Dict[str, Any]:
    global _CANDIDATE_BITMAP_RULE_BEHAVIOR_CACHE

    rules_path = Path(getattr(CANDIDATE_BITMAP_CONFIG, "rules_path"))
    try:
        mtime_ns = int(rules_path.stat().st_mtime_ns)
    except Exception:
        mtime_ns = -1

    cache_key = str(rules_path)
    if (
        _CANDIDATE_BITMAP_RULE_BEHAVIOR_CACHE is not None
        and _CANDIDATE_BITMAP_RULE_BEHAVIOR_CACHE[0] == cache_key
        and _CANDIDATE_BITMAP_RULE_BEHAVIOR_CACHE[1] == mtime_ns
    ):
        return _CANDIDATE_BITMAP_RULE_BEHAVIOR_CACHE[2]

    try:
        rules = _load_json(rules_path)
        bitmap_cfg = rules.get("candidate_bitmap", {}) or {}
        if not isinstance(bitmap_cfg, dict):
            bitmap_cfg = {}
    except Exception:
        bitmap_cfg = {}

    _CANDIDATE_BITMAP_RULE_BEHAVIOR_CACHE = (cache_key, mtime_ns, bitmap_cfg)
    return bitmap_cfg


def _non_target_slot_safe_fallback_enabled() -> bool:
    return STAGE2_MODE == "full"


def _configured_probe_terminal_tokens() -> Dict[str, str]:
    bitmap_cfg = _load_candidate_bitmap_rule_behavior()
    return {
        str(token).strip().upper(): str(ret).strip().upper()
        for token, ret in (bitmap_cfg.get("terminal_tokens", {}) or {}).items()
    }


def _callable_terminal_token_set() -> set:
    bitmap_cfg = _load_candidate_bitmap_rule_behavior()
    return {
        normalize_operator_key(str(token).strip())
        for token in bitmap_cfg.get("callable_terminal_tokens", []) or []
        if str(token).strip()
    }


def _bare_terminal_token_set() -> set:
    bitmap_cfg = _load_candidate_bitmap_rule_behavior()
    terminal_tokens = bitmap_cfg.get("terminal_tokens", {}) or {}
    return {
        normalize_operator_key(str(token).strip())
        for token in terminal_tokens.keys()
        if str(token).strip()
        and not str(token).strip().endswith("()")
        and not str(token).strip().upper().startswith(("COL_", "VALUE_"))
    }


def _non_target_slot_fixed_sql_values(outer_node_id: str, slot_index: int) -> List[str]:
    if not _non_target_slot_safe_fallback_enabled():
        return []

    bitmap_cfg = _load_candidate_bitmap_rule_behavior()

    fixed_sql_cfg = bitmap_cfg.get(
        "probe_non_target_slot_fixed_sql", bitmap_cfg.get("non_target_slot_fixed_sql", {})
    ) or {}
    if not isinstance(fixed_sql_cfg, dict):
        return []

    outer_token = normalize_operator_key(str(outer_node_id).split("::", 1)[0])
    token_cfg = None
    for raw_token, raw_slot_cfg in fixed_sql_cfg.items():
        if normalize_operator_key(str(raw_token).strip()) == outer_token:
            token_cfg = raw_slot_cfg
            break

    if not isinstance(token_cfg, dict):
        return []

    fixed_sql = token_cfg.get(str(int(slot_index)))
    if fixed_sql is None:
        fixed_sql = token_cfg.get(int(slot_index))

    values = fixed_sql if isinstance(fixed_sql, list) else [fixed_sql]
    result: List[str] = []
    for value in values:
        fixed_sql_text = str(value).strip() if value is not None else ""
        if not fixed_sql_text:
            continue
        if _is_sqlserver():
            condition_constant = re.fullmatch(r"\(\s*1\s*=\s*([01])\s*\)", fixed_sql_text)
            if condition_constant:
                token = "TRUE" if condition_constant.group(1) == "1" else "FALSE"
                fixed_sql_text = _terminal_to_probe_sql(token, "T_BOOL")
        if fixed_sql_text not in result:
            result.append(fixed_sql_text)
    return result


def _non_target_slot_fixed_sql(
    outer_node_id: str, slot_index: int, variant_index: int = 0,
) -> Optional[str]:
    values = _non_target_slot_fixed_sql_values(outer_node_id, slot_index)
    return values[int(variant_index) % len(values)] if values else None


def _fixed_sql_result_for_non_target_slot(
    outer_node_id: str,
    slot_index: int,
    variant_index: int = 0,
) -> Optional[Dict[str, Any]]:
    values = _non_target_slot_fixed_sql_values(outer_node_id, slot_index)
    if not values:
        return None

    return {
        "sql": values[int(variant_index) % len(values)],
        "variant_count": len(values),
        "used_edges": [],
        "used_nodes": [],
        "safe_closure_used": True,
        "non_target_slot_safe_fallback_used": True,
        "non_target_slot_fixed_sql_used": True,
    }


_SQL_RENDER_CONFIG_CACHE: Optional[Tuple[str, int, Dict[str, Any]]] = None


def _load_sql_render_config() -> Dict[str, Any]:
    global _SQL_RENDER_CONFIG_CACHE

    rules_path = Path(getattr(CANDIDATE_BITMAP_CONFIG, "rules_path"))
    try:
        mtime_ns = int(rules_path.stat().st_mtime_ns)
    except Exception:
        mtime_ns = -1

    cache_key = str(rules_path)
    if (
        _SQL_RENDER_CONFIG_CACHE is not None
        and _SQL_RENDER_CONFIG_CACHE[0] == cache_key
        and _SQL_RENDER_CONFIG_CACHE[1] == mtime_ns
    ):
        return _SQL_RENDER_CONFIG_CACHE[2]

    try:
        rules = _load_json(rules_path)
        render_config = rules.get("sql_render", {}) or {}
        if not isinstance(render_config, dict):
            render_config = {}
    except Exception:
        render_config = {}

    _SQL_RENDER_CONFIG_CACHE = (cache_key, mtime_ns, render_config)
    return render_config


def _sql_render_operator_templates() -> Dict[str, Dict[str, Any]]:
    render_config = _load_sql_render_config()
    templates = render_config.get("operator_templates", {}) or {}
    if not isinstance(templates, dict):
        return {}

    result: Dict[str, Dict[str, Any]] = {}
    for raw_token, raw_template in templates.items():
        if not isinstance(raw_template, dict):
            continue

        norm_token = normalize_operator_key(str(raw_token).strip())
        if norm_token:
            result[norm_token] = raw_template

    return result


def _render_probe_sql_by_template(
    token: str,
    child_sqls: List[str],
    template: Dict[str, Any],
) -> Optional[str]:
    token_text = str(token).strip()
    style = str(template.get("style", "") or "").strip().lower()

    if style == "binary_infix" and len(child_sqls) == 2:
        return f"({child_sqls[0]} {token_text} {child_sqls[1]})"

    if style == "between" and len(child_sqls) == 3:
        return f"({child_sqls[0]} {token_text} {child_sqls[1]} AND {child_sqls[2]})"

    if style == "in_list" and len(child_sqls) >= 2:
        left_sql = child_sqls[0]
        list_items = list(child_sqls[1:])

        try:
            min_items = int(template.get("min_items", 1))
        except Exception:
            min_items = 1

        min_items = max(1, min_items)
        while len(list_items) < min_items:
            list_items.append("1")

        return f"({left_sql} {token_text} ({', '.join(list_items)}))"

    if style == "function_call":
        func_name = token_text[:-2] if token_text.endswith("()") else token_text
        return f"{func_name}({', '.join(child_sqls)})"

    if style == "prefix" and len(child_sqls) == 1:
        return f"({token_text} {child_sqls[0]})"

    if style == "postfix" and len(child_sqls) == 1:
        return f"({child_sqls[0]} {token_text})"

    return None


def _format_probe_sql(token: str, child_sqls: List[str]) -> str:
    token_text = str(token).strip()

    templates = _sql_render_operator_templates()
    template = templates.get(normalize_operator_key(token_text))
    if template is not None:
        rendered = _render_probe_sql_by_template(token_text, child_sqls, template)
        if rendered is not None:
            return rendered

    if _is_sqlserver():
        raise ValueError(f"SQL Server probe operator has no matching render template: {token_text}")

    if token_text.endswith("()"):
        func_name = token_text[:-2]
        return f"{func_name}({', '.join(child_sqls)})"

    if len(child_sqls) == 1:
        return f"({token_text} {child_sqls[0]})"

    if len(child_sqls) == 2:
        return f"({child_sqls[0]} {token_text} {child_sqls[1]})"

    if len(child_sqls) > 2:
        return f"{token_text}({', '.join(child_sqls)})"

    return token_text


def _candidate_level_rank(level: str) -> int:
    level_text = str(level).lower()
    if level_text == "strict":
        return 0
    if level_text == "weak":
        return 1
    if level_text == "try":
        return 2
    return 3


def _probe_child_sort_key(
    *,
    graph_data: Dict[str, Any],
    edge_lookup: Dict[Tuple[str, int, str], Dict[str, Any]],
    outer_node_id: str,
    slot_index: int,
    child_node_id: str,
) -> Tuple[int, int, int, int, str]:
    payload = _get_node_payload(graph_data, child_node_id) or {}
    is_terminal = _node_is_terminal(payload) if payload else False

    edge_record = edge_lookup.get((outer_node_id, slot_index, child_node_id), {})
    level_rank = _candidate_level_rank(str(edge_record.get("candidate_level", "unknown")))

    slots = payload.get("slots", []) if isinstance(payload, dict) else []
    slot_count = len(slots) if isinstance(slots, list) else 0

    self_loop_penalty = 1 if child_node_id == outer_node_id else 0

    return (
        0 if is_terminal else 1,
        level_rank,
        slot_count,
        self_loop_penalty,
        str(child_node_id),
    )


def _safe_fallback_for_non_target_slot(
    *,
    outer_node_id: str,
    slot_index: int,
    edge_lookup: Dict[Tuple[str, int, str], Dict[str, Any]],
    variant_index: int = 0,
) -> Optional[Dict[str, Any]]:
    if not _non_target_slot_safe_fallback_enabled():
        return None

    fixed_result = _fixed_sql_result_for_non_target_slot(outer_node_id, slot_index, variant_index)
    if fixed_result is not None:
        return fixed_result

    required_type = "T_ANY"
    slot_records: List[Dict[str, Any]] = []
    for record in edge_lookup.values():
        if str(record.get("outer_node_id", "")) != str(outer_node_id):
            continue

        try:
            record_slot_index = int(record.get("slot_index", -1))
        except Exception:
            continue

        if record_slot_index == int(slot_index):
            required_type = str(record.get("slot_required_type", "T_ANY") or "T_ANY")
            slot_records.append(record)

    nonterminal_node_ids = {
        str(record.get("outer_node_id", ""))
        for record in edge_lookup.values()
        if str(record.get("outer_node_id", ""))
    }
    terminal_candidates = []
    for record in slot_records:
        inner_node_id = str(record.get("inner_node_id", "") or "")
        if not inner_node_id or inner_node_id in nonterminal_node_ids:
            continue
        if str(record.get("candidate_level", "")).lower() != "strict":
            continue
        inner_token = str(record.get("inner_token", "") or "")
        inner_return_type = str(record.get("inner_return_type", "") or "")
        try:
            child_sql = _terminal_to_probe_sql(inner_token, inner_return_type)
        except Exception:
            continue
        terminal_candidates.append((inner_node_id, child_sql))

    if terminal_candidates:
        terminal_candidates.sort(key=lambda item: item[0])
        selected_position = int(variant_index) % len(terminal_candidates)
        child_node_id, child_sql = terminal_candidates[selected_position]
        return {
            "sql": child_sql,
            "variant_count": len(terminal_candidates),
            "used_edges": [
                _edge_id_from_parts(outer_node_id, int(slot_index), child_node_id)
            ],
            "used_nodes": [child_node_id],
            "safe_closure_used": True,
            "non_target_slot_safe_fallback_used": True,
            "non_target_slot_fallback_source": "strict_terminal_candidate",
        }

    child_sql, child_node_id = _default_safe_terminal_for_required_type(required_type)

    used_edges: List[str] = []
    if (outer_node_id, int(slot_index), child_node_id) in edge_lookup:
        used_edges.append(_edge_id_from_parts(outer_node_id, int(slot_index), child_node_id))

    return {
        "sql": child_sql,
        "variant_count": 1,
        "used_edges": used_edges,
        "used_nodes": [child_node_id],
        "safe_closure_used": True,
        "non_target_slot_safe_fallback_used": True,
        "non_target_slot_fallback_source": "required_type_default",
    }


def _try_build_probe_expr(
    *,
    graph_data: Dict[str, Any],
    node_id: str,
    edge_lookup: Dict[Tuple[str, int, str], Dict[str, Any]],
    target_outer_node_id: str,
    target_slot_index: int,
    target_inner_node_id: str,
    depth_remaining: int,
    path: List[str],
    target_edge_consumed: bool = False,
    variant_index: int = 0,
) -> Optional[Dict[str, Any]]:
    payload = _get_node_payload(graph_data, node_id)
    if payload is None:
        return None

    token = str(payload.get("token", node_id.split("::", 1)[0]))
    return_type = str(payload.get("return_type", ""))

    if _node_is_terminal(payload):
        return {
            "sql": _terminal_to_probe_sql(token, return_type),
            "used_edges": [],
            "used_nodes": [node_id],
            "safe_closure_used": False,
        }

    if depth_remaining <= 0:
        return None

    slots = payload.get("slots", [])
    if not isinstance(slots, list) or len(slots) == 0:
        return None

    next_path = path + [node_id]
    child_sqls: List[str] = []
    used_edges: List[str] = []
    used_nodes: List[str] = [node_id]
    safe_closure_used = False

    completion_variant_index = int(variant_index)

    for slot_index, slot_items in enumerate(slots):
        if not isinstance(slot_items, list):
            slot_items = []

        force_target_child = (
            not bool(target_edge_consumed)
            and node_id == target_outer_node_id
            and int(slot_index) == int(target_slot_index)
        )

        allow_outer_sibling_fallback = (
            not path
            and node_id == target_outer_node_id
            and int(slot_index) != int(target_slot_index)
        )

        if allow_outer_sibling_fallback:
            fixed_result = _fixed_sql_result_for_non_target_slot(
                node_id, slot_index, completion_variant_index,
            )
            if fixed_result is not None:
                child_sqls.append(str(fixed_result["sql"]))
                completion_variant_index //= int(fixed_result["variant_count"])
                safe_closure_used = True
                continue

        if force_target_child:
            candidate_children = [target_inner_node_id]
        else:
            candidate_children = list(slot_items)

        candidate_children = [
            str(child_id)
            for child_id in candidate_children
            if _get_node_payload(graph_data, str(child_id)) is not None
        ]

        if not candidate_children:
            if allow_outer_sibling_fallback:
                fallback_result = _safe_fallback_for_non_target_slot(
                    outer_node_id=node_id,
                    slot_index=slot_index,
                    edge_lookup=edge_lookup,
                    variant_index=completion_variant_index,
                )
                if fallback_result is not None:
                    child_sqls.append(str(fallback_result["sql"]))
                    completion_variant_index //= int(fallback_result["variant_count"])
                    used_edges.extend(str(x) for x in fallback_result.get("used_edges", []))
                    used_nodes.extend(str(x) for x in fallback_result.get("used_nodes", []))
                    safe_closure_used = True
                    continue

            return None

        candidate_children.sort(
            key=lambda child_id: _probe_child_sort_key(
                graph_data=graph_data,
                edge_lookup=edge_lookup,
                outer_node_id=node_id,
                slot_index=slot_index,
                child_node_id=child_id,
            )
        )

        rotate_completion = allow_outer_sibling_fallback and _non_target_slot_safe_fallback_enabled()
        if not force_target_child and len(candidate_children) > 1:
            child_offset = int(completion_variant_index if rotate_completion else variant_index) % len(candidate_children)
            if not rotate_completion:
                variant_index = int(variant_index) // len(candidate_children)
            candidate_children = candidate_children[child_offset:] + candidate_children[:child_offset]

        selected_child_result: Optional[Dict[str, Any]] = None
        selected_child_id: Optional[str] = None

        for child_node_id in candidate_children:
            child_payload = _get_node_payload(graph_data, child_node_id)
            if child_payload is None:
                continue

            if (
                child_node_id in next_path
                and not _node_is_terminal(child_payload)
                and depth_remaining <= 1
            ):
                continue

            child_result = _try_build_probe_expr(
                graph_data=graph_data,
                node_id=child_node_id,
                edge_lookup=edge_lookup,
                target_outer_node_id=target_outer_node_id,
                target_slot_index=target_slot_index,
                target_inner_node_id=target_inner_node_id,
                depth_remaining=depth_remaining - 1,
                path=next_path,
                target_edge_consumed=bool(target_edge_consumed or force_target_child),
                variant_index=variant_index,
            )

            if child_result is not None:
                selected_child_result = child_result
                selected_child_id = child_node_id
                break

        if selected_child_result is None or selected_child_id is None:
            if allow_outer_sibling_fallback:
                fallback_result = _safe_fallback_for_non_target_slot(
                    outer_node_id=node_id,
                    slot_index=slot_index,
                    edge_lookup=edge_lookup,
                    variant_index=completion_variant_index,
                )
                if fallback_result is not None:
                    child_sqls.append(str(fallback_result["sql"]))
                    completion_variant_index //= int(fallback_result["variant_count"])
                    used_edges.extend(str(x) for x in fallback_result.get("used_edges", []))
                    used_nodes.extend(str(x) for x in fallback_result.get("used_nodes", []))
                    safe_closure_used = True
                    continue

            return None

        if rotate_completion:
            completion_variant_index //= len(candidate_children)
        edge_id = _edge_id_from_parts(node_id, slot_index, selected_child_id)
        child_sqls.append(str(selected_child_result["sql"]))
        used_edges.append(edge_id)
        used_edges.extend(str(x) for x in selected_child_result.get("used_edges", []))
        used_nodes.extend(str(x) for x in selected_child_result.get("used_nodes", []))
        if bool(selected_child_result.get("safe_closure_used", False)):
            safe_closure_used = True

    return {
        "sql": _format_probe_sql(token, child_sqls),
        "used_edges": used_edges,
        "used_nodes": used_nodes,
        "safe_closure_used": safe_closure_used,
    }


def build_probe_predicate_from_graph_data(
    probe_graph_data: Dict[str, Any],
    *,
    edge_records: Optional[List[Dict[str, Any]]] = None,
    max_depth: Optional[int] = None,
    probe_graph_path: Optional[Path] = None,
    predicate_output_path: Optional[Path] = None,
    predicate_report_path: Optional[Path] = None,
) -> Dict[str, Any]:
    if not isinstance(probe_graph_data, dict):
        raise ValueError('probe_graph_data must be a dict')

    nodes = probe_graph_data.get("nodes", {})
    if not isinstance(nodes, dict):
        raise ValueError('The nodes field in probe_graph_data must be a dict')

    metadata = probe_graph_data.get("probe_metadata", {})
    if not isinstance(metadata, dict):
        raise ValueError('probe_graph_data has no probe_metadata')

    target_edge = metadata.get("target_edge", {})
    if not isinstance(target_edge, dict):
        raise ValueError('probe_metadata.target_edge must be a dict')

    depth_limit = int(
        max_depth
        if max_depth is not None
        else getattr(CANDIDATE_BITMAP_CONFIG, "probe_predicate_max_depth", 3)
    )

    target_outer_node_id = str(target_edge.get("outer_node_id", ""))
    target_inner_node_id = str(target_edge.get("inner_node_id", ""))
    target_slot_index = int(target_edge.get("slot_index", -1))

    if not target_outer_node_id or not target_inner_node_id or target_slot_index < 0:
        raise ValueError('probe_metadata.target_edge is missing outer_node_id / inner_node_id / slot_index')

    if edge_records is None:
        edge_records = _load_edge_index_records(
            Path(getattr(CANDIDATE_BITMAP_CONFIG, "edge_index_path", CANDIDATE_EDGE_INDEX_PATH))
        )

    edge_lookup = _edge_record_lookup(edge_records)

    target_edge_id = _edge_id_from_parts(
        target_outer_node_id,
        target_slot_index,
        target_inner_node_id,
    )

    target_record: Dict[str, Any] = {}
    for record in edge_records:
        if str(record.get("edge_id", "")) == target_edge_id:
            target_record = record
            break

    tried_probe_signatures = _candidate_bitmap_tried_probe_signatures(target_record)
    variant_enabled = _candidate_bitmap_probe_variant_enabled()
    variant_max_attempts = _candidate_bitmap_probe_variant_max_attempts() if variant_enabled else 1
    variant_base_index = len(tried_probe_signatures) if variant_enabled else 0
    completion_variant_index = int(target_record.get("completion_variant_index", 0) or 0)
    if variant_enabled:
        variant_base_index += completion_variant_index
    variant_attempts = 0
    variant_duplicate_count = 0
    variant_construct_failed_count = 0
    selected_variant_index: Optional[int] = None
    build_result: Optional[Dict[str, Any]] = None

    for attempt_offset in range(variant_max_attempts):
        variant_attempts = attempt_offset + 1
        current_variant_index = variant_base_index + attempt_offset
        candidate_build_result = _try_build_probe_expr(
            graph_data=probe_graph_data,
            node_id=target_outer_node_id,
            edge_lookup=edge_lookup,
            target_outer_node_id=target_outer_node_id,
            target_slot_index=target_slot_index,
            target_inner_node_id=target_inner_node_id,
            depth_remaining=depth_limit,
            path=[],
            target_edge_consumed=False,
            variant_index=current_variant_index,
        )

        if candidate_build_result is None:
            variant_construct_failed_count += 1
            continue

        candidate_signature = _candidate_bitmap_normalize_failure_signature(
            candidate_build_result.get("sql", "")
        )
        if variant_enabled and candidate_signature and candidate_signature in tried_probe_signatures:
            variant_duplicate_count += 1
            continue

        build_result = candidate_build_result
        selected_variant_index = current_variant_index
        break

    status = "success"
    blocked_reason = ""
    predicate_sql = ""
    used_edges: List[str] = []
    used_nodes: List[str] = []

    if build_result is None:
        status = "blocked"
        blocked_reason = 'Not all argument slots could be filled within max_depth'
        if variant_enabled and variant_duplicate_count > 0:
            blocked_reason = "probe_variant_exhausted"
    else:
        predicate_sql = str(build_result.get("sql", ""))
        used_edges = [str(x) for x in build_result.get("used_edges", [])]
        used_nodes = [str(x) for x in build_result.get("used_nodes", [])]

        if target_edge_id not in used_edges:
            status = "blocked"
            blocked_reason = 'The constructed predicate does not contain the target edge'

    report = {
        "version": "stage17_4_probe_predicate_v1",
        "status": status,
        "blocked_reason": blocked_reason,
        "probe_graph_path": str(probe_graph_path or "<memory>"),
        "predicate_output_path": str(predicate_output_path or "<memory>"),
        "predicate_report_path": str(predicate_report_path or "<memory>"),
        "max_depth": depth_limit,
        "target_edge": {
            "edge_id": target_edge_id,
            "outer_node_id": target_outer_node_id,
            "slot_index": target_slot_index,
            "inner_node_id": target_inner_node_id,
            "candidate_level": str(target_edge.get("candidate_level", "")),
            "edge_score": float(target_edge.get("edge_score", 0.0)),
            "pattern_transfer_score": float(target_edge.get("pattern_transfer_score", 0.0)),
        },
        "predicate": {
            "sql": predicate_sql,
            "contains_target_edge": target_edge_id in used_edges,
            "safe_closure_used": bool(build_result.get("safe_closure_used", False)) if build_result else False,
            "used_edge_count": len(used_edges),
            "used_node_count": len(set(used_nodes)),
            "used_edges": used_edges,
            "used_nodes": sorted(set(used_nodes)),
            "contains_abstract_column_token": "COL_" in predicate_sql,
        },
        "runtime": {
            "persisted_to_disk": False,
        },
        "probe_variant": {
            "enabled": bool(variant_enabled),
            "max_attempts": int(variant_max_attempts),
            "attempts": int(variant_attempts),
            "base_variant_index": int(variant_base_index),
            "completion_variant_index": completion_variant_index,
            "selected_variant_index": selected_variant_index,
            "tried_signature_count": len(tried_probe_signatures),
            "duplicate_count": int(variant_duplicate_count),
            "construct_failed_count": int(variant_construct_failed_count),
        },
        "notes": [
            'The predicate is built from the local probe_graph to validate the target edge.',
            'The target outer slot is forced to use the target inner node once; after consuming the target edge, the remaining slots are filled normally to avoid recursive self-edge expansion.',
            'Stage 17-11 RL uses in-memory probe predicates to avoid growth of runtime_probe files.',
        ],
    }

    return report


def _find_smoke_edge_record(
    edge_records: List[Dict[str, Any]],
    *,
    outer_node_id: str,
    slot_index: int,
    inner_node_id: str,
) -> Dict[str, Any]:
    target_edge_id = _edge_id_from_parts(outer_node_id, slot_index, inner_node_id)
    for record in edge_records:
        if str(record.get("edge_id", "")) == target_edge_id:
            return record
    raise ValueError(f'The smoke-test target edge was not found among in-memory candidate edges: {target_edge_id}')


def _make_smoke_terminal_node(node_id: str, token: str, return_type: str) -> Dict[str, Any]:
    return {
        "token": token,
        "return_type": return_type,
        "is_terminal": True,
        "slots": [],
    }


def _make_smoke_outer_node(
    *,
    token: str,
    return_type: str,
    slot_count: int,
    slots: Optional[List[List[str]]] = None,
) -> Dict[str, Any]:
    if slots is None:
        slots = [[] for _ in range(slot_count)]
    return {
        "token": token,
        "return_type": return_type,
        "is_terminal": False,
        "slots": slots,
    }


def _build_safe_closure_smoke_graph(
    *,
    target_record: Dict[str, Any],
    outer_slots: List[List[str]],
    extra_nodes: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    outer_node_id = str(target_record["outer_node_id"])
    outer_token = str(target_record.get("outer_token", _split_graph_node_id(outer_node_id)[0]))
    outer_return_type = str(target_record.get("outer_return_type", _split_graph_node_id(outer_node_id)[1]))

    nodes: Dict[str, Dict[str, Any]] = {
        outer_node_id: _make_smoke_outer_node(
            token=outer_token,
            return_type=outer_return_type,
            slot_count=len(outer_slots),
            slots=outer_slots,
        )
    }
    nodes.update(extra_nodes)

    return {
        "nodes": nodes,
        "roots": [outer_node_id],
        "probe_metadata": {
            "version": "stage20_4_safe_closure_smoke_v1",
            "target_edge": target_record,
            "note": 'In-memory smoke graph; the formal global_knowledge_graph.json is not modified',
        },
    }


def _run_one_safe_closure_smoke_case(
    *,
    case: Dict[str, Any],
    edge_records: List[Dict[str, Any]],
    rules_path: Path,
) -> Dict[str, Any]:
    target_record = _find_smoke_edge_record(
        edge_records,
        outer_node_id=str(case["outer_node_id"]),
        slot_index=int(case["slot_index"]),
        inner_node_id=str(case["inner_node_id"]),
    )

    probe_graph_data = _build_safe_closure_smoke_graph(
        target_record=target_record,
        outer_slots=case["outer_slots"],
        extra_nodes=case["extra_nodes"],
    )

    predicate_report = build_probe_predicate_from_graph_data(
        probe_graph_data,
        edge_records=edge_records,
        max_depth=int(case.get("max_depth", 1)),
        probe_graph_path=None,
        predicate_output_path=None,
        predicate_report_path=None,
    )

    predicate_sql = str(predicate_report.get("predicate", {}).get("sql", "") or "")
    parse_result = {
        "success": False,
        "error": "not_run",
        "target_edge_hit": False,
        "extracted_edge_count": 0,
    }

    if predicate_sql:
        parsed = _parse_probe_predicate_edges(predicate_sql, rules_path)
        extracted_edges = [str(x) for x in parsed.get("edges", [])]
        target_edge_id = str(predicate_report.get("target_edge", {}).get("edge_id", ""))
        parse_result = {
            "success": bool(parsed.get("success", False)),
            "error": str(parsed.get("error", "")),
            "target_edge_hit": target_edge_id in extracted_edges,
            "extracted_edge_count": len(extracted_edges),
            "extracted_edges_sample": extracted_edges[:20],
        }

    expected_status = str(case.get("expected_status", "success"))
    expected_safe = bool(case.get("expected_safe_closure_used", False))
    expected_parse_success = bool(case.get("expected_parse_success", expected_status == "success"))

    actual_status = str(predicate_report.get("status", ""))
    actual_safe = bool(predicate_report.get("predicate", {}).get("safe_closure_used", False))
    actual_parse_success = bool(parse_result.get("success", False))

    passed = (
        actual_status == expected_status
        and actual_safe == expected_safe
        and (
            not expected_parse_success
            or (
                actual_parse_success
                and bool(parse_result.get("target_edge_hit", False))
            )
        )
    )

    return {
        "name": str(case.get("name", "")),
        "passed": bool(passed),
        "expected": {
            "status": expected_status,
            "safe_closure_used": expected_safe,
            "parse_success": expected_parse_success,
        },
        "actual": {
            "status": actual_status,
            "blocked_reason": str(predicate_report.get("blocked_reason", "")),
            "safe_closure_used": actual_safe,
            "sql": predicate_sql,
            "parse": parse_result,
        },
        "target_edge": predicate_report.get("target_edge", {}),
        "predicate": predicate_report.get("predicate", {}),
    }


def run_candidate_safe_closure_smoke(
    rules_path: Optional[Path] = None,
    *,
    write_report: bool = True,
    raise_on_error: bool = True,
) -> Dict[str, Any]:
    path = Path(rules_path or getattr(CANDIDATE_BITMAP_CONFIG, "rules_path"))
    rules = _load_json(path)

    preflight_report = validate_candidate_rules_preflight(
        rules,
        path,
        write_report=True,
        raise_on_error=True,
    )

    edge_records = [edge.to_dict() for edge in build_candidate_edges(rules)]

    value_int = _make_smoke_terminal_node("VALUE_TYPICAL_INT::INT", "VALUE_TYPICAL_INT", "INT")
    true_node = _make_smoke_terminal_node("TRUE::T_BOOL", "TRUE", "T_BOOL")

    cases = [
        {
            "name": "eq_slot0_to_abs_target_inner_uses_safe_closure",
            "outer_node_id": "=::T_BOOL",
            "slot_index": 0,
            "inner_node_id": "ABS()::T_CMP",
            "outer_slots": [["ABS()::T_CMP"], ["VALUE_TYPICAL_INT::INT"]],
            "extra_nodes": {
                "ABS()::T_CMP": _make_smoke_outer_node(token="ABS()", return_type="T_CMP", slot_count=1),
                "VALUE_TYPICAL_INT::INT": value_int,
            },
            "expected_status": "success",
            "expected_safe_closure_used": True,
            "expected_parse_success": True,
        },
        {
            "name": "eq_slot0_to_length_target_inner_uses_safe_closure",
            "outer_node_id": "=::T_BOOL",
            "slot_index": 0,
            "inner_node_id": "LENGTH()::T_CMP",
            "outer_slots": [["LENGTH()::T_CMP"], ["VALUE_TYPICAL_INT::INT"]],
            "extra_nodes": {
                "LENGTH()::T_CMP": _make_smoke_outer_node(token="LENGTH()", return_type="T_CMP", slot_count=1),
                "VALUE_TYPICAL_INT::INT": value_int,
            },
            "expected_status": "success",
            "expected_safe_closure_used": True,
            "expected_parse_success": True,
        },
        {
            "name": "and_slot0_to_regexp_target_inner_uses_safe_closure",
            "outer_node_id": "AND::T_BOOL",
            "slot_index": 0,
            "inner_node_id": "REGEXP::T_BOOL",
            "outer_slots": [["REGEXP::T_BOOL"], ["TRUE::T_BOOL"]],
            "extra_nodes": {
                "REGEXP::T_BOOL": _make_smoke_outer_node(token="REGEXP", return_type="T_BOOL", slot_count=2),
                "TRUE::T_BOOL": true_node,
            },
            "expected_status": "success",
            "expected_safe_closure_used": True,
            "expected_parse_success": True,
        },
        {
            "name": "and_slot0_to_not_between_target_inner_uses_safe_closure",
            "outer_node_id": "AND::T_BOOL",
            "slot_index": 0,
            "inner_node_id": "NOT BETWEEN::T_BOOL",
            "outer_slots": [["NOT BETWEEN::T_BOOL"], ["TRUE::T_BOOL"]],
            "extra_nodes": {
                "NOT BETWEEN::T_BOOL": _make_smoke_outer_node(token="NOT BETWEEN", return_type="T_BOOL", slot_count=3),
                "TRUE::T_BOOL": true_node,
            },
            "expected_status": "success",
            "expected_safe_closure_used": True,
            "expected_parse_success": True,
        },
        {
            "name": "abs_as_target_outer_does_not_use_safe_closure",
            "outer_node_id": "ABS()::T_CMP",
            "slot_index": 0,
            "inner_node_id": "VALUE_TYPICAL_INT::INT",
            "outer_slots": [["VALUE_TYPICAL_INT::INT"]],
            "extra_nodes": {
                "VALUE_TYPICAL_INT::INT": value_int,
            },
            "expected_status": "success",
            "expected_safe_closure_used": False,
            "expected_parse_success": True,
        },
        {
            "name": "and_slot0_to_is_target_inner_is_not_generic_safe_closed",
            "outer_node_id": "AND::T_BOOL",
            "slot_index": 0,
            "inner_node_id": "IS::T_BOOL",
            "outer_slots": [["IS::T_BOOL"], ["TRUE::T_BOOL"]],
            "extra_nodes": {
                "IS::T_BOOL": _make_smoke_outer_node(token="IS", return_type="T_BOOL", slot_count=2),
                "TRUE::T_BOOL": true_node,
            },
            "expected_status": "blocked",
            "expected_safe_closure_used": False,
            "expected_parse_success": False,
        },
    ]

    results = [
        _run_one_safe_closure_smoke_case(
            case=case,
            edge_records=edge_records,
            rules_path=path,
        )
        for case in cases
    ]

    passed_count = sum(1 for item in results if item.get("passed"))
    failed = [item for item in results if not item.get("passed")]

    report = {
        "version": "stage20_4_safe_closure_smoke_v1",
        "rules_path": str(path),
        "ok": len(failed) == 0,
        "summary": {
            "case_count": len(results),
            "passed_count": passed_count,
            "failed_count": len(failed),
            "candidate_edge_count_in_memory": len(edge_records),
            "preflight_ok": bool(preflight_report.get("ok", False)),
        },
        "results": results,
        "notes": [
            'This smoke test builds candidate edges only in memory.',
            'It does not call save_outputs or rebuild candidate_edge_index.json or candidate_bitmaps.npz.',
            'It does not connect to the database or run RL.',
            'It validates target-inner-only safe closure and confirms parser target-edge hits for successful cases.',
        ],
    }

    if write_report:
        report_path = _candidate_safe_closure_smoke_report_path()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with report_path.open("w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

    if failed and raise_on_error:
        detail = json.dumps(report["summary"], ensure_ascii=False)
        raise ValueError(f'[safe_closure_smoke] Validation failed: {detail}')

    return report


def build_probe_predicate_for_edge(
    *,
    formal_graph_data: Dict[str, Any],
    edge_records: List[Dict[str, Any]],
    bitmaps: Dict[str, np.ndarray],
    probe_edge_id: Optional[str] = None,
    probe_edge_index: Optional[int] = None,
    allow_observed: bool = False,
    max_depth: Optional[int] = None,
    graph_path: Optional[Path] = None,
) -> Dict[str, Any]:
    probe_graph_data, probe_graph_report = build_probe_graph_data(
        formal_graph_data=formal_graph_data,
        edge_records=edge_records,
        bitmaps=bitmaps,
        probe_edge_id=probe_edge_id,
        probe_edge_index=probe_edge_index,
        allow_observed=allow_observed,
        graph_path=graph_path,
    )

    probe_predicate_report = build_probe_predicate_from_graph_data(
        probe_graph_data,
        edge_records=edge_records,
        max_depth=max_depth,
        probe_graph_path=None,
        predicate_output_path=None,
        predicate_report_path=None,
    )

    return {
        "probe_graph_data": probe_graph_data,
        "probe_graph_report": probe_graph_report,
        "probe_predicate_report": probe_predicate_report,
    }


def build_and_save_probe_predicate(
    probe_graph_path: Optional[Path] = None,
    *,
    predicate_output_path: Optional[Path] = None,
    predicate_report_path: Optional[Path] = None,
    max_depth: Optional[int] = None,
    edge_records: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    input_probe_graph_path = Path(
        probe_graph_path
        or getattr(CANDIDATE_BITMAP_CONFIG, "probe_graph_path", CANDIDATE_PROBE_GRAPH_PATH)
    )
    output_predicate_path = Path(
        predicate_output_path
        or getattr(CANDIDATE_BITMAP_CONFIG, "probe_predicate_path", CANDIDATE_PROBE_PREDICATE_PATH)
    )
    output_report_path = Path(
        predicate_report_path
        or getattr(CANDIDATE_BITMAP_CONFIG, "probe_predicate_report_path", CANDIDATE_PROBE_PREDICATE_REPORT_PATH)
    )

    probe_graph_data = _load_probe_graph_data(input_probe_graph_path)
    report = build_probe_predicate_from_graph_data(
        probe_graph_data,
        edge_records=edge_records,
        max_depth=max_depth,
        probe_graph_path=input_probe_graph_path,
        predicate_output_path=output_predicate_path,
        predicate_report_path=output_report_path,
    )

    predicate_sql = str(report.get("predicate", {}).get("sql", "") or "")
    status = str(report.get("status", ""))

    output_predicate_path.parent.mkdir(parents=True, exist_ok=True)
    with output_predicate_path.open("w", encoding="utf-8") as f:
        if status == "success":
            f.write(predicate_sql + "\n")
        else:
            f.write("")

    report.setdefault("runtime", {})
    report["runtime"]["persisted_to_disk"] = True

    output_report_path.parent.mkdir(parents=True, exist_ok=True)
    with output_report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    return report


def _read_text_file(path: Path) -> str:
    with Path(path).open("r", encoding="utf-8") as f:
        return f.read().strip()


def _ast_node_id(node: Any) -> str:
    token = str(getattr(node, "token", ""))
    return_type = str(getattr(node, "return_type", "") or "")
    return f"{token}::{return_type}"


def _extract_edges_from_typed_ast(node: Any) -> List[str]:
    if node is None:
        return []

    edges: List[str] = []
    parent_id = _ast_node_id(node)

    children = getattr(node, "children", []) or []
    for slot_index, child in enumerate(children):
        child_id = _ast_node_id(child)
        edges.append(_edge_id_from_parts(parent_id, slot_index, child_id))
        edges.extend(_extract_edges_from_typed_ast(child))

    return edges


def _parse_probe_predicate_edges(predicate_sql: str, rules_path: Path) -> Dict[str, Any]:
    try:
        import sys

        project_root = Path(__file__).resolve().parents[1]
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))

        from pkg.parser import SimpleParser
        from pkg.type_inference import TypeInferencer

        rules = _load_json(rules_path)
        parser = SimpleParser(config=rules)
        ast = parser.parse(predicate_sql)

        if ast is None:
            return {
                "success": False,
                "error": 'The parser returned None',
                "edges": [],
                "ast": None,
            }

        inferencer = TypeInferencer(rules)
        inferencer.infer(ast)

        edges = _extract_edges_from_typed_ast(ast)

        return {
            "success": True,
            "error": "",
            "edges": edges,
            "ast": ast.to_dict() if hasattr(ast, "to_dict") else None,
        }
    except Exception as exc:
        return {
            "success": False,
            "error": str(exc),
            "edges": [],
            "ast": None,
        }


def _complete_and_execute_probe_predicate(
    predicate_sql: str,
    *,
    expected_return_type: Optional[str] = None,
) -> Dict[str, Any]:
    executor = None
    completed_sql = ""

    try:
        from RL.db_executor import create_db_executor
        from RL.heuristic_completion import HeuristicCompletionClient

        executor = create_db_executor(autoconnect=True)
        completion_client = HeuristicCompletionClient(executor)
        completed_sql = completion_client.complete_sql(
            predicate_sql,
            expected_return_type=expected_return_type,
        )

        db_result = executor.execute_query(
            sql=completed_sql,
            params=None,
            fetch="all",
            autocommit=True,
        )

        return {
            "completion_success": bool(completed_sql),
            "completed_sql": completed_sql,
            "db_success": bool(db_result.success),
            "db_error": db_result.error,
            "db_error_code": db_result.error_code,
            "db_rowcount": db_result.rowcount,
            "db_rows_returned": len(db_result.rows or []),
            "db_execution_time": db_result.execution_time,
            "db_warning_count": db_result.warning_count,
            **({"db_sqlstate": getattr(db_result, "sqlstate", None),
                "db_cancelled": bool(getattr(db_result, "cancelled", False)),
                "db_timed_out": bool(getattr(db_result, "timed_out", False))}
               if str(getattr(DB_CONFIG, "db_type", "")).lower().strip() == "sqlserver" else {}),
        }
    except Exception as exc:
        return {
            "completion_success": bool(completed_sql),
            "completed_sql": completed_sql,
            "db_success": False,
            "db_error": str(exc),
            "db_error_code": None,
            "db_rowcount": 0,
            "db_rows_returned": 0,
            "db_execution_time": 0.0,
            "db_warning_count": 0,
        }
    finally:
        if executor is not None:
            try:
                executor.close()
            except Exception:
                pass


def build_and_save_probe_validation(
    predicate_path: Optional[Path] = None,
    *,
    predicate_report_path: Optional[Path] = None,
    completed_sql_output_path: Optional[Path] = None,
    validation_report_path: Optional[Path] = None,
    rules_path: Optional[Path] = None,
) -> Dict[str, Any]:
    input_predicate_path = Path(
        predicate_path
        or getattr(CANDIDATE_BITMAP_CONFIG, "probe_predicate_path", CANDIDATE_PROBE_PREDICATE_PATH)
    )
    input_predicate_report_path = Path(
        predicate_report_path
        or getattr(
            CANDIDATE_BITMAP_CONFIG,
            "probe_predicate_report_path",
            CANDIDATE_PROBE_PREDICATE_REPORT_PATH,
        )
    )
    output_completed_sql_path = Path(
        completed_sql_output_path
        or getattr(
            CANDIDATE_BITMAP_CONFIG,
            "probe_completed_sql_path",
            CANDIDATE_PROBE_COMPLETED_SQL_PATH,
        )
    )
    output_validation_report_path = Path(
        validation_report_path
        or getattr(
            CANDIDATE_BITMAP_CONFIG,
            "probe_validation_report_path",
            CANDIDATE_PROBE_VALIDATION_REPORT_PATH,
        )
    )
    db_rules_path = Path(
        rules_path
        or getattr(CANDIDATE_BITMAP_CONFIG, "rules_path")
    )

    predicate_sql = _read_text_file(input_predicate_path)
    predicate_report = _load_json(input_predicate_report_path)

    target_edge = predicate_report.get("target_edge", {}) or {}
    target_edge_id = str(target_edge.get("edge_id", ""))

    parse_result = _parse_probe_predicate_edges(predicate_sql, db_rules_path)
    extracted_edges = [str(x) for x in parse_result.get("edges", [])]
    target_edge_hit = target_edge_id in extracted_edges

    expected_return_type = str(target_edge.get("outer_return_type", "") or "")
    if not expected_return_type:
        outer_node_id = str(target_edge.get("outer_node_id", "") or "")
        if "::" in outer_node_id:
            expected_return_type = outer_node_id.rsplit("::", 1)[-1]

    execution_result = _complete_and_execute_probe_predicate(
        predicate_sql,
        expected_return_type=expected_return_type,
    )

    completed_sql = str(execution_result.get("completed_sql", ""))
    output_completed_sql_path.parent.mkdir(parents=True, exist_ok=True)
    with output_completed_sql_path.open("w", encoding="utf-8") as f:
        if completed_sql:
            f.write(completed_sql + "\n")
        else:
            f.write("")

    if not predicate_sql:
        status = "blocked"
        blocked_reason = 'probe_predicate.txt is empty.'
    elif not parse_result.get("success", False):
        status = "parser_failed"
        blocked_reason = str(parse_result.get("error", 'Parser failed'))
    elif not target_edge_hit:
        status = "target_edge_miss"
        blocked_reason = 'Parser edge extraction did not hit the target edge.'
    elif not execution_result.get("completion_success", False):
        status = "completion_failed"
        blocked_reason = 'heuristic_completion produced no executable SQL.'
    elif not execution_result.get("db_success", False):
        status = "execution_failed"
        blocked_reason = str(execution_result.get("db_error", 'Database execution failed'))
    else:
        status = "success"
        blocked_reason = ""

    report = {
        "version": "stage17_5_probe_validation_v1",
        "status": status,
        "blocked_reason": blocked_reason,
        "predicate_path": str(input_predicate_path),
        "predicate_report_path": str(input_predicate_report_path),
        "completed_sql_path": str(output_completed_sql_path),
        "validation_report_path": str(output_validation_report_path),
        "rules_path": str(db_rules_path),
        "target_edge": target_edge,
        "predicate": {
            "sql": predicate_sql,
            "contains_abstract_column_token": "COL_" in predicate_sql,
        },
        "parser": {
            "success": bool(parse_result.get("success", False)),
            "error": str(parse_result.get("error", "")),
            "target_edge_hit": bool(target_edge_hit),
            "extracted_edge_count": len(extracted_edges),
            "extracted_edges": extracted_edges,
            "typed_ast": parse_result.get("ast"),
            "note": 'The parser extracts edges from the original probe predicate, not from heuristic-completed SQL.',
        },
        "completion": {
            "success": bool(execution_result.get("completion_success", False)),
            "completed_sql": completed_sql,
        },
        "database": {
            "success": bool(execution_result.get("db_success", False)),
            "error": execution_result.get("db_error"),
            "error_code": execution_result.get("db_error_code"),
            "rowcount": execution_result.get("db_rowcount"),
            "rows_returned": execution_result.get("db_rows_returned"),
            "execution_time": execution_result.get("db_execution_time"),
            "warning_count": execution_result.get("db_warning_count"),
            **({"sqlstate": execution_result.get("db_sqlstate"),
                "cancelled": execution_result.get("db_cancelled", False),
                "timed_out": execution_result.get("db_timed_out", False)}
               if str(getattr(DB_CONFIG, "db_type", "")).lower().strip() == "sqlserver" else {}),
        },
        "notes": [
            'This validation is a Stage 2 probe-edge dry run, not Stage 3 BugFinding.',
            'This function does not update Correct_Collection / Incorrect_Collection or the formal global_knowledge_graph.json.',
            'success means predicate construction, parser target-edge hit, heuristic SQL completion, and DB execution all succeeded.',
            'Subsequent RL integration logic determines how to update observed / failed / probing bitmaps.',
        ],
    }

    output_validation_report_path.parent.mkdir(parents=True, exist_ok=True)
    with output_validation_report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    return report


def build_and_save_candidate_bitmap(rules_path: Optional[Path] = None) -> Dict[str, Any]:
    path = Path(rules_path or getattr(CANDIDATE_BITMAP_CONFIG, "rules_path"))
    rules = _load_json(path)

    preflight_report = validate_candidate_rules_preflight(
        rules,
        path,
        write_report=True,
        raise_on_error=True,
    )

    edges = build_candidate_edges(rules)
    bitmaps = build_bitmaps(edges)
    report = save_outputs(edges, bitmaps)
    report["rules_preflight"] = {
        "ok": bool(preflight_report.get("ok", False)),
        "summary": preflight_report.get("summary", {}),
        "report_path": str(_candidate_rules_preflight_report_path()),
    }

    return report


def main() -> None:
    parser = argparse.ArgumentParser(description='Stage 17: build the candidate edge universe, observed bitmap, and probe graph.')
    parser.add_argument(
        "--rules",
        type=str,
        default=str(getattr(CANDIDATE_BITMAP_CONFIG, "rules_path")),
        help='Database rules JSON path; defaults to CANDIDATE_BITMAP_CONFIG.rules_path.',
    )
    parser.add_argument(
        "--build-observed",
        action="store_true",
        help='After rebuilding candidate bitmaps, map edges from global_knowledge_graph.json to observed_bitmap.',
    )
    parser.add_argument(
        "--build-probe",
        action="store_true",
        help='Insert one missing candidate edge into a graph copy to build a temporary probe_graph.',
    )
    parser.add_argument(
        "--graph",
        type=str,
        default=str(getattr(CANDIDATE_BITMAP_CONFIG, "global_knowledge_graph_path", GLOBAL_KNOWLEDGE_GRAPH_PATH)),
        help='Path to pkg/knowledge_graph/global_knowledge_graph.json.',
    )
    parser.add_argument(
        "--probe-edge-id",
        type=str,
        default=None,
        help='Optional: specify the exact candidate edge_id to insert into probe_graph.',
    )
    parser.add_argument(
        "--probe-edge-index",
        type=int,
        default=None,
        help='Optional: specify the candidate edge_index to insert into probe_graph.',
    )
    parser.add_argument(
        "--allow-observed-probe",
        action="store_true",
        help='Allow building a probe graph for an already observed edge. Defaults to False.',
    )
    parser.add_argument(
        "--mark-probing",
        action="store_true",
        help='If set, mark the selected edge in probing_bitmap and refresh the derived missing/available bitmaps.',
    )
    parser.add_argument(
        "--probe-graph-output",
        type=str,
        default=str(getattr(CANDIDATE_BITMAP_CONFIG, "probe_graph_path", CANDIDATE_PROBE_GRAPH_PATH)),
        help='Output path for temporary probe_graph.json.',
    )
    parser.add_argument(
        "--probe-report-output",
        type=str,
        default=str(getattr(CANDIDATE_BITMAP_CONFIG, "probe_report_path", CANDIDATE_PROBE_REPORT_PATH)),
        help='Output path for probe_graph_report.json.',
    )
    parser.add_argument(
        "--build-probe-predicate",
        action="store_true",
        help='Build a predicate expression that must contain the target edge from probe_graph.json.',
    )
    parser.add_argument(
        "--probe-input",
        type=str,
        default=str(getattr(CANDIDATE_BITMAP_CONFIG, "probe_graph_path", CANDIDATE_PROBE_GRAPH_PATH)),
        help='Input probe_graph.json path for --build-probe-predicate.',
    )
    parser.add_argument(
        "--probe-predicate-output",
        type=str,
        default=str(getattr(CANDIDATE_BITMAP_CONFIG, "probe_predicate_path", CANDIDATE_PROBE_PREDICATE_PATH)),
        help='Output text path for the generated probe predicate.',
    )
    parser.add_argument(
        "--probe-predicate-report-output",
        type=str,
        default=str(getattr(CANDIDATE_BITMAP_CONFIG, "probe_predicate_report_path", CANDIDATE_PROBE_PREDICATE_REPORT_PATH)),
        help='Output path for probe_predicate_report.json.',
    )
    parser.add_argument(
        "--probe-predicate-max-depth",
        type=int,
        default=int(getattr(CANDIDATE_BITMAP_CONFIG, "probe_predicate_max_depth", 3)),
        help='Maximum recursion depth when constructing a probe predicate.',
    )
    parser.add_argument(
        "--validate-probe-predicate",
        action="store_true",
        help='Validate the probe predicate through parser edge extraction, heuristic SQL completion, and DB execution.',
    )
    parser.add_argument(
        "--probe-predicate-input",
        type=str,
        default=str(getattr(CANDIDATE_BITMAP_CONFIG, "probe_predicate_path", CANDIDATE_PROBE_PREDICATE_PATH)),
        help='Input probe_predicate.txt path for --validate-probe-predicate.',
    )
    parser.add_argument(
        "--probe-predicate-report-input",
        type=str,
        default=str(getattr(CANDIDATE_BITMAP_CONFIG, "probe_predicate_report_path", CANDIDATE_PROBE_PREDICATE_REPORT_PATH)),
        help='Input probe_predicate_report.json path for --validate-probe-predicate.',
    )
    parser.add_argument(
        "--probe-completed-sql-output",
        type=str,
        default=str(getattr(CANDIDATE_BITMAP_CONFIG, "probe_completed_sql_path", CANDIDATE_PROBE_COMPLETED_SQL_PATH)),
        help='Output path for heuristic-completed SQL.',
    )
    parser.add_argument(
        "--probe-validation-report-output",
        type=str,
        default=str(getattr(CANDIDATE_BITMAP_CONFIG, "probe_validation_report_path", CANDIDATE_PROBE_VALIDATION_REPORT_PATH)),
        help='Output path for probe_validation_report.json.',
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help='Only validate database rules JSON and write candidate_rules_preflight_report.json; do not build candidate bitmaps.',
    )
    parser.add_argument(
        "--smoke-safe-closure",
        action="store_true",
        help='Run the Stage 20-4 in-memory smoke test for target-inner-only safe closure; do not rebuild candidate bitmaps.',
    )
    args = parser.parse_args()

    if args.preflight_only:
        rules_path = Path(args.rules)
        rules = _load_json(rules_path)
        preflight_report = validate_candidate_rules_preflight(
            rules,
            rules_path,
            write_report=True,
            raise_on_error=True,
        )
        print('[candidate_bitmap] Rules preflight check completed')
        print(json.dumps(preflight_report, indent=2, ensure_ascii=False))
        return

    if args.smoke_safe_closure:
        smoke_report = run_candidate_safe_closure_smoke(
            Path(args.rules),
            write_report=True,
            raise_on_error=True,
        )
        print('[candidate_bitmap] Safe closure smoke check completed')
        print(json.dumps(smoke_report, indent=2, ensure_ascii=False))
        return

    report = build_and_save_candidate_bitmap(Path(args.rules))

    print('[candidate_bitmap] Candidate bitmap construction completed')
    print(json.dumps(report, indent=2, ensure_ascii=False))

    if args.build_observed or args.build_probe:
        observed_report = build_and_save_observed_bitmap(Path(args.graph))
        print('[candidate_bitmap] Observed bitmap construction completed')
        print(json.dumps(observed_report, indent=2, ensure_ascii=False))

    if args.build_probe:
        probe_report = build_and_save_probe_graph(
            Path(args.graph),
            probe_edge_id=args.probe_edge_id,
            probe_edge_index=args.probe_edge_index,
            probe_graph_path=Path(args.probe_graph_output),
            probe_report_path=Path(args.probe_report_output),
            allow_observed=bool(args.allow_observed_probe),
            mark_probing=bool(args.mark_probing),
        )
        print('[candidate_bitmap] Probe graph construction completed')
        print(json.dumps(probe_report, indent=2, ensure_ascii=False))

    if args.build_probe_predicate:
        probe_input_path = Path(args.probe_graph_output) if args.build_probe else Path(args.probe_input)
        predicate_report = build_and_save_probe_predicate(
            probe_input_path,
            predicate_output_path=Path(args.probe_predicate_output),
            predicate_report_path=Path(args.probe_predicate_report_output),
            max_depth=int(args.probe_predicate_max_depth),
        )
        print('[candidate_bitmap] Probe predicate construction completed')
        print(json.dumps(predicate_report, indent=2, ensure_ascii=False))

    if args.validate_probe_predicate:
        predicate_input_path = Path(args.probe_predicate_output) if args.build_probe_predicate else Path(args.probe_predicate_input)
        predicate_report_input_path = (
            Path(args.probe_predicate_report_output)
            if args.build_probe_predicate
            else Path(args.probe_predicate_report_input)
        )

        validation_report = build_and_save_probe_validation(
            predicate_input_path,
            predicate_report_path=predicate_report_input_path,
            completed_sql_output_path=Path(args.probe_completed_sql_output),
            validation_report_path=Path(args.probe_validation_report_output),
            rules_path=Path(args.rules),
        )
        print('[candidate_bitmap] Probe validation completed')
        print(json.dumps(validation_report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
