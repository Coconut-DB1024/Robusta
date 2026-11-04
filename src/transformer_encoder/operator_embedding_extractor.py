import argparse
import json
import os
import re
import sys
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import torch

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from transformer_encoder.sql_vae_codet5 import CodeT5Encoder
from RL.config import DB_CONFIG

try:
    from pkg.parser import SimpleParser
except Exception:
    SimpleParser = None


MODEL_DIR = os.path.join(PROJECT_ROOT, "codet5-small")
STATE_PATH = os.path.join(MODEL_DIR, "encoder_state.pt")
INPUT_PREDICATE_FILE = os.path.join(PROJECT_ROOT, "output", "predicate.txt")

OUTPUT_PROTOTYPES_FILE = os.path.join(PROJECT_ROOT, "output", "operator_prototypes.json")
OUTPUT_SIMILARITY_FILE = os.path.join(PROJECT_ROOT, "output", "operator_similarity.json")
OUTPUT_METADATA_FILE = os.path.join(PROJECT_ROOT, "output", "operator_embedding_metadata.json")

DEFAULT_TOP_K = 16

TYPE_WRAPPER_NAMES = {
    "col",
    "int",
    "float",
    "decimal",
    "double",
    "string",
    "char",
    "varchar",
    "date",
    "timestamp",
    "time",
    "hex",
    "bit",
    "bool",
    "boolean",
}

LITERAL_WORDS = {
    "TRUE",
    "FALSE",
    "NULL",
}

# These keywords should not be treated as function names even when followed by "(".
# For example, IN (...) is predicate operator syntax, not an IN() function.
SQL_KEYWORDS_NOT_FUNCTIONS = {
    "IN",
    "LIKE",
    "BETWEEN",
    "REGEXP",
    "IS",
    "AND",
    "OR",
    "XOR",
    "NOT",
}


def _safe_norm(vec: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm < eps:
        return vec
    return vec / norm


def _normalize_operator_name(op: str) -> str:
    text = " ".join(str(op).strip().split())
    if not text:
        return text

    if text.endswith("()"):
        return text[:-2].upper() + "()"

    if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", text):
        return text.upper()

    if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*(\s+[A-Za-z_][A-Za-z0-9_]*)+", text):
        return text.upper()

    return text


def _load_db_parser_config() -> Tuple[str, str, Dict[str, Any]]:
    db_type = str(getattr(DB_CONFIG, "db_type", "") or "").lower().strip()
    if not db_type:
        raise ValueError("DB_CONFIG.db_type is empty; cannot select parser rules.")

    config_path = os.path.join(
        PROJECT_ROOT,
        "pkg",
        "configuration",
        f"{db_type}_rules.json",
    )
    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"Parser rules for db_type={db_type!r} do not exist: {config_path}"
        )

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    if not isinstance(config, dict):
        raise ValueError(f"Parser rules must be a JSON object: {config_path}")

    return db_type, os.path.abspath(config_path), config


def _collect_parser_operators(parser_config: Dict[str, Any]) -> List[str]:
    if SimpleParser is None:
        raise RuntimeError("SimpleParser is unavailable; cannot extract configured operators.")

    parser = SimpleParser(parser_config)
    ops = (
        set(parser.precedence.keys())
        | set(parser.unary_operators)
        | set(parser.operator_templates.keys())
    )
    definitions = parser_config.get("definitions", {}) or {}
    if isinstance(definitions, dict):
        # Candidate-bitmap outer operators come from definitions; several
        # function operators are not present in precedence/operator_templates.
        ops |= {
            str(operator)
            for operator in definitions.keys()
            if str(operator).strip() and not str(operator).strip().startswith("//")
        }
    ops.discard("UNARY_HIGH")
    return sorted(
        {_normalize_operator_name(op) for op in ops if str(op).strip()},
        key=len,
        reverse=True,
    )


def _operator_prototype_aliases(parser_config: Dict[str, Any]) -> Dict[str, List[str]]:
    """Load optional prototype-only aliases without changing parser semantics."""
    bitmap_config = parser_config.get("candidate_bitmap", {}) or {}
    raw_aliases = bitmap_config.get("operator_prototype_aliases", {}) or {}
    if not isinstance(raw_aliases, dict):
        return {}
    aliases: Dict[str, List[str]] = {}
    for raw_operator, raw_targets in raw_aliases.items():
        operator = _normalize_operator_name(str(raw_operator))
        targets = raw_targets if isinstance(raw_targets, list) else [raw_targets]
        normalized_targets = [
            _normalize_operator_name(str(target))
            for target in targets
            if str(target).strip()
        ]
        if operator and normalized_targets:
            aliases[operator] = normalized_targets
    return aliases


def _find_ignored_spans(text: str) -> List[Tuple[int, int]]:
    ignored: List[Tuple[int, int]] = []

    for m in re.finditer(r"'(?:\\.|[^'\\])*'|\"(?:\\.|[^\"\\])*\"", text):
        ignored.append((m.start(), m.end()))

    wrapper_pattern = re.compile(
        r"\b(" + "|".join(re.escape(x) for x in sorted(TYPE_WRAPPER_NAMES, key=len, reverse=True)) + r")\s*\([^)]*\)",
        flags=re.IGNORECASE,
    )
    for m in wrapper_pattern.finditer(text):
        ignored.append((m.start(), m.end()))

    return ignored


def _span_inside_ignored(start: int, end: int, ignored_spans: List[Tuple[int, int]]) -> bool:
    for s, e in ignored_spans:
        if start >= s and end <= e:
            return True
    return False


def _find_function_operator_spans(text: str, ignored_spans: List[Tuple[int, int]]) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []

    for m in re.finditer(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\(", text):
        name = m.group(1)
        name_lower = name.lower()
        name_upper = name.upper()
        if name_lower in TYPE_WRAPPER_NAMES:
            continue

        if name_upper in SQL_KEYWORDS_NOT_FUNCTIONS:
            continue

        start, end = m.start(1), m.end(1)
        if _span_inside_ignored(start, end, ignored_spans):
            continue

        op = _normalize_operator_name(name + "()")
        if op in LITERAL_WORDS:
            continue

        results.append(
            {
                "op": op,
                "start": start,
                "end": end,
                "surface": text[start:end],
                "kind": "function",
            }
        )

    return results


def _build_operator_regex(operators: List[str]) -> re.Pattern:
    word_patterns: List[str] = []
    symbol_patterns: List[str] = []

    for op in operators:
        if not op:
            continue

        if op.endswith("()"):
            continue

        if re.fullmatch(r"[A-Z]+(\s+[A-Z]+)*", op):
            parts = op.split()
            pattern = r"\b" + r"\s+".join(re.escape(p) for p in parts) + r"\b"
            word_patterns.append(pattern)
        else:
            symbol_patterns.append(re.escape(op))

    all_patterns = []
    all_patterns.extend(sorted(word_patterns, key=len, reverse=True))
    all_patterns.extend(sorted(symbol_patterns, key=len, reverse=True))

    if not all_patterns:
        return re.compile(r"a^")

    return re.compile("|".join(f"(?:{p})" for p in all_patterns), flags=re.IGNORECASE)


def _drop_between_separator_and(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    去掉 BETWEEN / NOT BETWEEN 语法中的分隔符 AND。

    例如：
        col(age) BETWEEN int(18) AND int(30)

    算子抽取时应该保留：
        BETWEEN

    不应该额外把中间的 AND 当成逻辑 AND 算子。
    """
    cleaned: List[Dict[str, Any]] = []
    waiting_between_and = False

    for item in items:
        op = item.get("op")

        if op in {"BETWEEN", "NOT BETWEEN", "BETWEEN SYMMETRIC", "NOT BETWEEN SYMMETRIC"}:
            cleaned.append(item)
            waiting_between_and = True
            continue

        if waiting_between_and and op == "AND":
            waiting_between_and = False
            continue

        cleaned.append(item)

    return cleaned


def find_operator_spans(
    text: str,
    parser_config: Dict[str, Any],
) -> List[Dict[str, Any]]:
    operators = _collect_parser_operators(parser_config)
    ignored_spans = _find_ignored_spans(text)

    results: List[Dict[str, Any]] = []
    results.extend(_find_function_operator_spans(text, ignored_spans))

    op_regex = _build_operator_regex(operators)
    for m in op_regex.finditer(text):
        start, end = m.start(), m.end()
        if _span_inside_ignored(start, end, ignored_spans):
            continue

        surface = text[start:end]
        op = _normalize_operator_name(surface)
        if not op or op in LITERAL_WORDS:
            continue

        results.append(
            {
                "op": op,
                "start": start,
                "end": end,
                "surface": surface,
                "kind": "operator",
            }
        )

    if str(parser_config.get("sql_render", {}).get("version", "")).startswith("sqlserver_"):
        # FROM followed by '(' belongs to IS [NOT] DISTINCT FROM, not a
        # function call. Keep real functions after the complete operator.
        operator_spans = [item for item in results if item["kind"] == "operator"]
        results = [
            item for item in results
            if item["kind"] != "function" or not any(
                op["start"] <= item["start"] and item["end"] <= op["end"]
                for op in operator_spans
            )
        ]

    results.sort(key=lambda x: (x["start"], -(x["end"] - x["start"])))

    deduped: List[Dict[str, Any]] = []
    seen_spans = set()
    for item in results:
        key = (item["start"], item["end"], item["op"])
        if key in seen_spans:
            continue
        seen_spans.add(key)
        deduped.append(item)

    deduped = _drop_between_separator_and(deduped)

    return deduped


def _token_indices_for_char_span(
    offsets: torch.Tensor,
    start: int,
    end: int,
) -> List[int]:
    indices: List[int] = []

    for idx, pair in enumerate(offsets.tolist()):
        token_start, token_end = int(pair[0]), int(pair[1])

        if token_start == 0 and token_end == 0:
            continue

        if token_start < end and token_end > start:
            indices.append(idx)

    return indices


def _encode_operator_occurrences(
    text: str,
    tokenizer,
    model: CodeT5Encoder,
    device: torch.device,
    parser_config: Dict[str, Any],
) -> List[Dict[str, Any]]:
    spans = find_operator_spans(text, parser_config)
    if not spans:
        return []

    encoded = tokenizer(
        text,
        return_tensors="pt",
        return_offsets_mapping=True,
        truncation=True,
        padding="longest",
    )

    offsets = encoded.pop("offset_mapping")[0]
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)

    with torch.no_grad():
        encoder_outputs = model.base_model.get_encoder()(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )

    token_embeddings = encoder_outputs.last_hidden_state[0]

    occurrences: List[Dict[str, Any]] = []
    for span in spans:
        token_indices = _token_indices_for_char_span(
            offsets=offsets,
            start=int(span["start"]),
            end=int(span["end"]),
        )

        if not token_indices:
            occurrences.append(
                {
                    "op": span["op"],
                    "surface": span["surface"],
                    "start": span["start"],
                    "end": span["end"],
                    "kind": span["kind"],
                    "vector": None,
                    "skip_reason": "no_token_overlap",
                }
            )
            continue

        idx_tensor = torch.tensor(token_indices, dtype=torch.long, device=device)
        vec = token_embeddings.index_select(0, idx_tensor).mean(dim=0).detach().cpu().numpy()

        occurrences.append(
            {
                "op": span["op"],
                "surface": span["surface"],
                "start": span["start"],
                "end": span["end"],
                "kind": span["kind"],
                "vector": vec,
                "token_indices": token_indices,
            }
        )

    return occurrences


def load_predicates(path: str) -> List[str]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"未找到谓词文件: {path}")

    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def load_model(model_dir: str, state_path: str, device: torch.device) -> CodeT5Encoder:
    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"未找到模型目录: {model_dir}")

    model = CodeT5Encoder(
        model_name_or_path=model_dir,
        device=device,
    )

    if os.path.exists(state_path):
        state_dict = torch.load(state_path, map_location=device)
        model.load_encoder_state_dict(state_dict)
    else:
        print(f"[operator_embedding_extractor 警告] 未找到 Encoder 权重，将使用原始 CodeT5 权重: {state_path}")

    model.eval()
    return model


def build_operator_prototypes(
    predicates: List[str],
    tokenizer,
    model: CodeT5Encoder,
    device: torch.device,
    parser_config: Dict[str, Any],
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Any]]:
    vectors_by_op: Dict[str, List[np.ndarray]] = defaultdict(list)
    examples_by_op: Dict[str, List[str]] = defaultdict(list)
    failed_lines: List[Dict[str, Any]] = []
    skipped_occurrences: List[Dict[str, Any]] = []

    processed_count = 0

    for idx, text in enumerate(predicates):
        try:
            occurrences = _encode_operator_occurrences(
                text,
                tokenizer,
                model,
                device,
                parser_config,
            )
            processed_count += 1

            if not occurrences:
                failed_lines.append(
                    {
                        "line_no": idx + 1,
                        "predicate": text,
                        "reason": "no_operator_found",
                    }
                )
                continue

            for occ in occurrences:
                op = occ["op"]
                vec = occ.get("vector")
                if vec is None:
                    skipped_occurrences.append(
                        {
                            "line_no": idx + 1,
                            "predicate": text,
                            "op": op,
                            "surface": occ.get("surface"),
                            "reason": occ.get("skip_reason", "unknown"),
                        }
                    )
                    continue

                vectors_by_op[op].append(vec)
                if len(examples_by_op[op]) < 5:
                    examples_by_op[op].append(text)

        except Exception as exc:
            failed_lines.append(
                {
                    "line_no": idx + 1,
                    "predicate": text,
                    "reason": repr(exc),
                }
            )

    prototypes: Dict[str, Dict[str, Any]] = {}
    for op, vectors in sorted(vectors_by_op.items(), key=lambda x: x[0]):
        matrix = np.stack(vectors, axis=0)
        prototype = matrix.mean(axis=0)
        prototype = _safe_norm(prototype)

        prototypes[op] = {
            "vector": prototype.astype(float).tolist(),
            "count": int(len(vectors)),
            "examples": examples_by_op.get(op, []),
            "prototype_source": "observed",
            "prototype_source_operator": op,
        }

    configured_operators = _collect_parser_operators(parser_config)
    prototype_aliases = _operator_prototype_aliases(parser_config)
    unresolved_operators: List[str] = []

    # Fill operators absent from predicate.txt without changing the observed
    # prototypes: explicit semantic alias first, then a model embedding of the
    # operator's canonical parser spelling.
    for operator in sorted(configured_operators):
        if operator in prototypes:
            continue

        alias_source = next(
            (
                alias
                for alias in prototype_aliases.get(operator, [])
                if alias in prototypes
            ),
            None,
        )
        if alias_source is not None:
            prototypes[operator] = {
                "vector": list(prototypes[alias_source]["vector"]),
                "count": 0,
                "examples": [],
                "prototype_source": "alias",
                "prototype_source_operator": alias_source,
            }
            continue

        canonical_occurrences = _encode_operator_occurrences(
            operator,
            tokenizer,
            model,
            device,
            parser_config,
        )
        canonical_vector = next(
            (
                occurrence.get("vector")
                for occurrence in canonical_occurrences
                if occurrence.get("vector") is not None
                and _normalize_operator_name(str(occurrence.get("op", ""))) == operator
            ),
            None,
        )
        if canonical_vector is None:
            unresolved_operators.append(operator)
            continue
        prototypes[operator] = {
            "vector": _safe_norm(np.asarray(canonical_vector, dtype=np.float32)).astype(float).tolist(),
            "count": 0,
            "examples": [],
            "prototype_source": "canonical_fallback",
            "prototype_source_operator": operator,
        }

    source_counts: Dict[str, int] = defaultdict(int)
    for data in prototypes.values():
        source_counts[str(data.get("prototype_source", "observed"))] += 1

    metadata = {
        "num_predicates": len(predicates),
        "num_processed_predicates": processed_count,
        "num_failed_predicates": len(failed_lines),
        "num_skipped_occurrences": len(skipped_occurrences),
        "num_operators": len(prototypes),
        "operator_counts": {op: data["count"] for op, data in prototypes.items()},
        "prototype_source_counts": dict(source_counts),
        "prototype_sources": {
            op: {
                "source": data.get("prototype_source", "observed"),
                "source_operator": data.get("prototype_source_operator", op),
            }
            for op, data in prototypes.items()
        },
        "configured_operator_count": len(configured_operators),
        "unresolved_configured_operators": unresolved_operators,
        "failed_lines": failed_lines[:200],
        "skipped_occurrences": skipped_occurrences[:200],
    }

    return prototypes, metadata


def build_operator_similarity(
    prototypes: Dict[str, Dict[str, Any]],
    top_k: int = DEFAULT_TOP_K,
    min_similarity: Optional[float] = None,
) -> Dict[str, List[Dict[str, Any]]]:
    ops = sorted(prototypes.keys())
    if not ops:
        return {}

    matrix = np.stack(
        [_safe_norm(np.asarray(prototypes[op]["vector"], dtype=np.float32)) for op in ops],
        axis=0,
    )

    sim_matrix = matrix @ matrix.T

    similarity: Dict[str, List[Dict[str, Any]]] = {}
    for i, op in enumerate(ops):
        candidates: List[Dict[str, Any]] = []

        order = np.argsort(-sim_matrix[i])
        rank = 1
        for j in order:
            if i == int(j):
                continue

            sim = float(sim_matrix[i, int(j)])
            if min_similarity is not None and sim < min_similarity:
                continue

            candidates.append(
                {
                    "op": ops[int(j)],
                    "sim": sim,
                    "rank": rank,
                }
            )
            rank += 1

            if len(candidates) >= top_k:
                break

        similarity[op] = candidates

    return similarity


def save_json(path: str, data: Any) -> None:
    output_dir = os.path.dirname(path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def run_extraction(
    predicate_file: str = INPUT_PREDICATE_FILE,
    model_dir: str = MODEL_DIR,
    state_path: str = STATE_PATH,
    output_prototypes_file: str = OUTPUT_PROTOTYPES_FILE,
    output_similarity_file: str = OUTPUT_SIMILARITY_FILE,
    output_metadata_file: str = OUTPUT_METADATA_FILE,
    top_k: int = DEFAULT_TOP_K,
    min_similarity: Optional[float] = None,
    device_name: str = "auto",
) -> Dict[str, Any]:
    db_type, parser_config_path, parser_config = _load_db_parser_config()

    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)

    predicates = load_predicates(predicate_file)

    model = load_model(
        model_dir=model_dir,
        state_path=state_path,
        device=device,
    )
    tokenizer = model.tokenizer

    prototypes, metadata = build_operator_prototypes(
        predicates=predicates,
        tokenizer=tokenizer,
        model=model,
        device=device,
        parser_config=parser_config,
    )

    # Record the exact DBMS rules used to make generated artifacts auditable.
    metadata["db_type"] = db_type
    metadata["parser_config_path"] = parser_config_path

    similarity = build_operator_similarity(
        prototypes=prototypes,
        top_k=top_k,
        min_similarity=min_similarity,
    )

    save_json(output_prototypes_file, prototypes)
    save_json(output_similarity_file, similarity)
    save_json(output_metadata_file, metadata)

    return {
        "prototypes": prototypes,
        "similarity": similarity,
        "metadata": metadata,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract contextual operator embeddings from predicate.txt and build operator similarity table."
    )

    parser.add_argument("--predicate-file", default=INPUT_PREDICATE_FILE)
    parser.add_argument("--model-dir", default=MODEL_DIR)
    parser.add_argument("--state-path", default=STATE_PATH)
    parser.add_argument("--output-prototypes", default=OUTPUT_PROTOTYPES_FILE)
    parser.add_argument("--output-similarity", default=OUTPUT_SIMILARITY_FILE)
    parser.add_argument("--output-metadata", default=OUTPUT_METADATA_FILE)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--min-similarity", type=float, default=None)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_extraction(
        predicate_file=args.predicate_file,
        model_dir=args.model_dir,
        state_path=args.state_path,
        output_prototypes_file=args.output_prototypes,
        output_similarity_file=args.output_similarity,
        output_metadata_file=args.output_metadata,
        top_k=args.top_k,
        min_similarity=args.min_similarity,
        device_name=args.device,
    )
