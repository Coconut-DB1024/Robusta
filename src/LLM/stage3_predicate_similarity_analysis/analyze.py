from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import mmap
import os
import shutil
import sqlite3
import sys
import time
import traceback
import tempfile
from collections import OrderedDict, deque
from array import array
from contextlib import ExitStack
from dataclasses import asdict, replace
from itertools import islice
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .config import CONFIG, SimilarityAnalysisConfig
from .report import atomic_json_dump, write_similarity_reports
from .ranked_selection import RankedCandidates, skip_ratio
from .similarity import (
    lsh_band_keys,
    one_permutation_minhash,
)
from .ast_similarity import (
    AstTreeIndexBuilder,
    AstTreeStore,
    database_dialect,
    dependency_versions,
)
from .ast_distance import distance_backend_info, distance_lower_bound, exact_distance
from .tokenizer import build_shingle_hashes, predicate_digest, tokenize_predicate


# Bounded LRU cache: repeated labeled-tree pairs reuse TED without allowing
# millions of pairs to accumulate in memory during large Stage-3 runs.
_AST_DISTANCE_CACHE_LIMIT = 100_000


class SimilarityAnalysisError(RuntimeError):
    pass


def _open_predicate_file(path: Path):
    if path.name.lower().endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", newline=None)
    return open(path, "r", encoding="utf-8", newline=None)


def _sha256_file(path: Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(block_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _validate_config(config: SimilarityAnalysisConfig) -> None:
    database_dialect()
    if config.ast_index_workers is not None and int(config.ast_index_workers) < 1:
        raise ValueError("ast_index_workers must be None or a positive integer")
    if int(config.ast_index_batch_size) < 1:
        raise ValueError("ast_index_batch_size must be a positive integer")
    if int(config.shingle_size) <= 0:
        raise ValueError("shingle_size must be greater than zero")
    if int(config.minhash_size) <= 0:
        raise ValueError("minhash_size must be greater than zero")
    if (
        int(config.lsh_rows_per_band) <= 0
        or int(config.minhash_size) % int(config.lsh_rows_per_band) != 0
    ):
        raise ValueError("lsh_rows_per_band must divide minhash_size")
    if config.window_radius is not None and int(config.window_radius) < 0:
        raise ValueError("window_radius must be None or >= 0")
    if (
        config.max_predicates_to_read is not None
        and int(config.max_predicates_to_read) <= 0
    ):
        raise ValueError("max_predicates_to_read must be None or greater than zero")
    for name in (
        "window_target_pair_budget",
        "window_radius_min",
        "window_radius_max",
        "lsh_bucket_fetch_multiplier",
        "sqlite_insert_batch_predicates",
    ):
        if int(getattr(config, name)) < 0:
            raise ValueError(f"{name} must be >= 0")
    if int(config.window_radius_min) > int(config.window_radius_max):
        raise ValueError("window_radius_min must be <= window_radius_max")
    if int(config.sqlite_insert_batch_predicates) == 0:
        raise ValueError("sqlite_insert_batch_predicates must be greater than zero")


def _relative_to_project(path: Path, project_root: Path) -> str:
    try:
        return path.resolve().relative_to(project_root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def load_input_manifest(*, project_root: Path, input_path: Path):
    """Read one complete predicate TXT, preserving order and duplicate lines."""
    resolved = Path(input_path).resolve()
    if not resolved.is_file():
        raise SimilarityAnalysisError(f"Predicate file does not exist: {resolved}")
    with _open_predicate_file(resolved) as handle:
        count = sum(1 for line in handle if line.strip())
    input_files = [{
        "file_id": 0, "round_index": 0, "depth": 0,
        "path": resolved, "relative_path": _relative_to_project(resolved, project_root),
    }]
    rounds = [{
        "round_index": 0, "depth": 0, "raw_expected_predicates": count,
        "expected_predicates": count, "round_complete": True, "file_ids": [0],
    }]
    return input_files, rounds


def _auto_bounded_value(
    *,
    configured: Optional[int],
    target_budget: int,
    minimum: int,
    maximum: int,
    predicate_count: int,
) -> int:
    if predicate_count <= 1:
        return 0
    if configured is not None:
        return min(max(0, int(configured)), predicate_count - 1)
    estimate = int(target_budget) // max(1, predicate_count)
    return min(
        predicate_count - 1,
        max(int(minimum), min(int(maximum), estimate)),
    )


def _create_lsh_database(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA journal_mode=OFF")
    connection.execute("PRAGMA synchronous=OFF")
    connection.execute("PRAGMA temp_store=FILE")
    connection.execute("PRAGMA locking_mode=EXCLUSIVE")
    connection.execute(
        "CREATE TABLE lsh_buckets (bucket BLOB NOT NULL, predicate_id INTEGER NOT NULL)"
    )
    return connection


def _print_progress(phase: str, current: int, total: int) -> None:
    fraction = min(max(0, current) / total, 1.0) if total > 0 else 0.0
    filled = int(fraction * 40)
    bar = "#" * filled + "-" * (40 - filled)
    # An unfinished phase must not round up to 100.00%.
    percent = min(fraction * 100, 99.99) if current < total else fraction * 100
    print(f"[PredicateSimilarity] {phase}: [{bar}] {percent:.2f}%", flush=True)


def _build_index(
    *,
    input_files: List[Dict[str, Any]],
    rounds: List[Dict[str, Any]],
    work_dir: Path,
    config: SimilarityAnalysisConfig,
    build_lsh: bool,
) -> Dict[str, Any]:
    sqlite_path = work_dir / "lsh.sqlite"
    connection = _create_lsh_database(sqlite_path) if build_lsh else None
    try:
        return _build_index_contents(
            input_files=input_files,
            rounds=rounds,
            work_dir=work_dir,
            config=config,
            connection=connection,
        )
    except Exception:
        if connection is not None:
            connection.close()
        raise


def _build_index_contents(
    *,
    input_files: List[Dict[str, Any]],
    rounds: List[Dict[str, Any]],
    work_dir: Path,
    config: SimilarityAnalysisConfig,
    connection: Optional[sqlite3.Connection],
) -> Dict[str, Any]:
    shingle_path = work_dir / "shingle_hashes.bin"
    predicate_hash_path = work_dir / "predicate_hashes.bin"
    sqlite_path = work_dir / "lsh.sqlite"

    offsets = array("Q")
    shingle_counts = array("I")
    token_counts = array("I")
    source_file_ids = array("I")
    source_lines = array("Q")
    round_actual_counts = [0 for _ in rounds]
    skipped_blank_lines = 0
    shingle_elements = 0
    ast_builder = (
        AstTreeIndexBuilder(work_dir / "ast_trees.bin", database_dialect())
    )
    pending_rows: List[Tuple[sqlite3.Binary, int]] = []
    predicates_since_flush = 0
    raw_manifest_predicate_count = sum(
        int(round_info.get("raw_expected_predicates", round_info["expected_predicates"]))
        for round_info in rounds
    )
    manifest_predicate_count = sum(
        int(round_info["expected_predicates"]) for round_info in rounds
    )
    configured_read_limit = (
        None
        if config.max_predicates_to_read is None
        else int(config.max_predicates_to_read)
    )
    analysis_target_count = (
        manifest_predicate_count
        if configured_read_limit is None
        else min(configured_read_limit, manifest_predicate_count)
    )
    input_truncated_by_limit = analysis_target_count < manifest_predicate_count

    ast_workers = 1
    if ast_builder is not None:
        if config.ast_index_workers is None:
            ast_workers = min(4, max(1, (os.cpu_count() or 1) - 1))
            if analysis_target_count < 20_000:
                ast_workers = 1
        else:
            ast_workers = int(config.ast_index_workers)
        print(f"[PredicateSimilarity] AST indexing workers: {ast_workers}; "
              f"batch size: {config.ast_index_batch_size}", flush=True)

    progress_enabled = int(config.progress_interval_predicates) > 0
    last_progress_log = time.monotonic()
    if progress_enabled:
        _print_progress("indexing", 0, analysis_target_count)

    # The parent alone owns source filtering, numbering, hashes and disk output.
    # SQL parsing is the expensive parallel step. A bounded queue collects its
    # results in original order, preserving window membership even across files.
    with ExitStack() as resources:
        shingle_handle = resources.enter_context(open(shingle_path, "wb"))
        predicate_hash_handle = resources.enter_context(open(predicate_hash_path, "wb"))
        ast_parser = None
        if ast_builder is not None:
            from .ast_index_workers import OrderedAstParser

            ast_parser = resources.enter_context(OrderedAstParser(
                ast_builder.dialect,
                workers=ast_workers, batch_size=int(config.ast_index_batch_size),
            ))
        for file_info in input_files:
            source_path = Path(file_info["path"])
            file_info["compressed_size_bytes"] = source_path.stat().st_size
            file_info["sha256"] = (
                None if input_truncated_by_limit else _sha256_file(source_path)
            )
            file_start = len(offsets)
            if input_truncated_by_limit and file_start >= analysis_target_count:
                file_info["predicate_count"] = 0
                file_info["start_predicate_id"] = None
                file_info["end_predicate_id"] = None
                file_info["analysis_skipped_due_to_read_limit"] = True
                continue

            # Count admitted rows, not completed worker results: prefetch must
            # never overshoot the configured cross-file predicate read cap.
            remaining_for_file = analysis_target_count - file_start

            def source_records(input_handle):
                nonlocal skipped_blank_lines
                case_index = admitted = 0
                for line_number, line in enumerate(input_handle, start=1):
                    if input_truncated_by_limit and admitted >= remaining_for_file:
                        break
                    raw_predicate = line.strip()
                    if not raw_predicate:
                        skipped_blank_lines += 1
                        continue
                    case_index += 1
                    tokens = tokenize_predicate(
                        raw_predicate,
                        normalize_unquoted_tokens=config.normalize_unquoted_tokens,
                    )
                    hashes = build_shingle_hashes(tokens, config.shingle_size)
                    if not hashes:
                        skipped_blank_lines += 1
                        continue
                    admitted += 1
                    yield raw_predicate, line_number, tokens, hashes

            with _open_predicate_file(source_path) as input_handle:
                records = source_records(input_handle)
                metadata_batches = deque()

                def raw_batches():
                    while True:
                        batch = list(islice(records, int(config.ast_index_batch_size)))
                        if not batch:
                            return
                        metadata_batches.append(batch)
                        yield [row[0] for row in batch]

                def ordered_rows():
                    if ast_parser is None:
                        for record in records:
                            yield record, None
                    else:
                        for outcomes in ast_parser.parse_batches(raw_batches()):
                            batch = metadata_batches.popleft()
                            if len(batch) != len(outcomes):
                                raise SimilarityAnalysisError("AST worker batch length mismatch")
                            yield from zip(batch, outcomes)

                for (raw_predicate, line_number, tokens, hashes), parsed in ordered_rows():
                    if ast_builder is not None:
                        ast_builder.add_encoded(*parsed)
                    predicate_id = len(offsets)
                    offsets.append(shingle_elements)
                    shingle_counts.append(len(hashes))
                    token_counts.append(len(tokens))
                    source_file_ids.append(int(file_info["file_id"]))
                    source_lines.append(line_number)
                    array("Q", hashes).tofile(shingle_handle)
                    shingle_elements += len(hashes)
                    predicate_hash_handle.write(predicate_digest(raw_predicate))
                    round_actual_counts[int(file_info["round_index"])] += 1

                    if connection is not None:
                        signature = one_permutation_minhash(hashes, config.minhash_size)
                        for bucket in lsh_band_keys(signature, config.lsh_rows_per_band):
                            pending_rows.append((sqlite3.Binary(bucket), predicate_id))
                        predicates_since_flush += 1
                        if predicates_since_flush >= int(config.sqlite_insert_batch_predicates):
                            connection.executemany(
                                "INSERT INTO lsh_buckets(bucket, predicate_id) VALUES (?, ?)",
                                pending_rows,
                            )
                            connection.commit()
                            pending_rows.clear()
                            predicates_since_flush = 0

                    if progress_enabled:
                        now = time.monotonic()
                        if now - last_progress_log >= 10.0 and len(offsets) < analysis_target_count:
                            _print_progress("indexing", len(offsets), analysis_target_count)
                            last_progress_log = now

            file_info["predicate_count"] = len(offsets) - file_start
            file_info["start_predicate_id"] = file_start if len(offsets) > file_start else None
            file_info["end_predicate_id"] = len(offsets) - 1 if len(offsets) > file_start else None
            file_info["analysis_stopped_at_read_limit"] = bool(
                input_truncated_by_limit and len(offsets) >= analysis_target_count
            )
    if input_truncated_by_limit:
        print(
            "[PredicateSimilarity] predicate read limit reached: "
            f"analyzed={len(offsets)}, manifest_total={manifest_predicate_count}",
            flush=True,
        )

    if connection is not None:
        if pending_rows:
            connection.executemany(
                "INSERT INTO lsh_buckets(bucket, predicate_id) VALUES (?, ?)",
                pending_rows,
            )
        connection.commit()
        connection.execute(
            "CREATE INDEX idx_lsh_bucket_predicate ON lsh_buckets(bucket, predicate_id)"
        )
        connection.commit()

    remaining_target = analysis_target_count
    for round_info, actual in zip(rounds, round_actual_counts):
        expected = int(round_info["expected_predicates"])
        round_target = min(expected, remaining_target)
        remaining_target -= round_target
        round_info["analysis_target_predicates"] = int(round_target)
        round_info["actual_predicates"] = int(actual)
        round_info["analysis_truncated_by_read_limit"] = bool(
            round_target < expected
        )
        if int(round_target) != int(actual):
            raise SimilarityAnalysisError(
                "predicate count does not match the analysis target: "
                f"depth={round_info['depth']}, "
                f"target={round_target}, actual={actual}"
            )

    ast_store = ast_builder.finish() if ast_builder is not None else None
    if progress_enabled:
        _print_progress("indexing", len(offsets), analysis_target_count)

    return {
        "shingle_path": shingle_path,
        "predicate_hash_path": predicate_hash_path,
        "sqlite_path": sqlite_path,
        "connection": connection,
        "offsets": offsets,
        "shingle_counts": shingle_counts,
        "token_counts": token_counts,
        "source_file_ids": source_file_ids,
        "source_lines": source_lines,
        "skipped_blank_lines": skipped_blank_lines,
        "total_shingles": shingle_elements,
        "raw_manifest_predicate_count": raw_manifest_predicate_count,
        "manifest_predicate_count": manifest_predicate_count,
        "excluded_by_validation_count": (
            raw_manifest_predicate_count - manifest_predicate_count
            if config.executable_only
            else 0
        ),
        "configured_read_limit": configured_read_limit,
        "analysis_target_count": analysis_target_count,
        "input_truncated_by_limit": input_truncated_by_limit,
        "ast_store": ast_store,
        "ast_index_workers": ast_workers if ast_builder is not None else None,
    }


def _build_lsh_candidate_query(band_count: int) -> str:
    """Build one bounded SQLite statement covering every band for a predicate."""
    parts = []
    for band_index in range(int(band_count)):
        parts.append(
            f"SELECT {band_index} AS band_index, predicate_id FROM ("
            "SELECT predicate_id FROM lsh_buckets "
            "WHERE bucket = ? AND predicate_id > ? "
            "ORDER BY predicate_id LIMIT ?)"
        )
    return " UNION ALL ".join(parts)


def _calculate_similarities(
    *,
    index: Mapping[str, Any],
    config: SimilarityAnalysisConfig,
    window_radius: int,
    lsh_candidate_limit: int,
    top_skip_fraction: float,
) -> Dict[str, Any]:
    offsets = index["offsets"]
    shingle_counts = index["shingle_counts"]
    predicate_count = len(offsets)
    connection: Optional[sqlite3.Connection] = index["connection"]
    ast_store: Optional[AstTreeStore] = index.get("ast_store")
    # AST analysis uses the original fixed 1/1/2 edit costs only. The optional
    # ranking changes which candidate score is selected, never the TED metric.
    ast_ranked = skip_ratio(top_skip_fraction) > 0

    ast_count = predicate_count
    best_scores = array("d", [-1.0]) * ast_count
    best_score_numerators = array("q", [0]) * ast_count
    best_score_denominators = array("I", [1]) * ast_count
    best_distances = array("I", [0]) * ast_count
    best_node_totals = array("I", [0]) * ast_count
    # Failed AST parses are not zero similarities and cannot occupy a rank.
    # Keep attempted candidate_count unchanged; rank only the scored candidates.
    rank_candidate_counts = array("I", [0]) * (ast_count if ast_ranked else 0)
    nearest_ids = array("q", [-1]) * predicate_count
    candidate_counts = array("I", [0]) * predicate_count
    match_sources = array("B", [0]) * predicate_count

    lsh_band_count = int(config.minhash_size) // int(config.lsh_rows_per_band)
    lsh_query = (
        _build_lsh_candidate_query(lsh_band_count)
        if connection is not None and lsh_candidate_limit > 0
        else ""
    )

    exact_pairs = 0
    window_pairs = 0
    lsh_pairs = 0
    combined_pairs = 0
    bucket_fetch_truncations = 0
    lsh_candidate_truncated_predicates = 0
    ast_distance_cache = OrderedDict()
    ast_cache_hits = 0
    ast_bound_skipped_pairs = 0
    ast_distance_evaluations = 0

    shingle_file = open(index["shingle_path"], "rb")
    shingle_map = (mmap.mmap(shingle_file.fileno(), 0, access=mmap.ACCESS_READ)
                   if predicate_count else None)
    shingle_view = memoryview(shingle_map if shingle_map is not None else b"").cast("Q")
    rank_selector = None

    def hashes_for(predicate_id: int):
        start = int(offsets[predicate_id])
        return shingle_view[start : start + int(shingle_counts[predicate_id])]


    def update_best_ast(
        owner_id: int,
        candidate_id: int,
        distance: int,
        node_total: int,
        source_code: int,
    ) -> None:
        if node_total <= 0:
            return
        value = int(distance)
        score_numerator = max(0, node_total - value)
        score_denominator = node_total + value
        if rank_selector is not None:
            # Reuse exact rational sorting and the bounded window/disk spool.
            # No floating-point rounding is used for ranks or equal-score ties.
            rank_selector.add(owner_id, candidate_id, score_numerator,
                              score_denominator, source_code)
            rank_candidate_counts[owner_id] += 1
            return
        old_numerator = best_score_numerators[owner_id]
        old_denominator = best_score_denominators[owner_id]
        score = score_numerator / score_denominator
        better = (
            int(nearest_ids[owner_id]) < 0
            or score_numerator * old_denominator > old_numerator * score_denominator
            or (
                score_numerator * old_denominator == old_numerator * score_denominator
                and int(candidate_id) < int(nearest_ids[owner_id])
            )
        )
        if better:
            best_score_numerators[owner_id] = score_numerator
            best_score_denominators[owner_id] = score_denominator
            best_scores[owner_id] = score
            best_distances[owner_id] = value
            best_node_totals[owner_id] = int(node_total)
            nearest_ids[owner_id] = candidate_id
            match_sources[owner_id] = source_code

    def ast_bound_can_improve(owner_id, candidate_id, numerator, denominator):
        """Test an upper score bound, including the original smaller-ID tie rule.

        A pair can be pruned ONLY if neither endpoint could improve. Pruned pairs
        still count as candidates, and are never cached as exact distances.
        """
        nearest = int(nearest_ids[owner_id])
        if nearest < 0:
            return True
        left = numerator * int(best_score_denominators[owner_id])
        right = int(best_score_numerators[owner_id]) * denominator
        return left > right or (left == right and candidate_id < nearest)

    progress_enabled = int(config.progress_interval_predicates) > 0
    last_progress_log = time.monotonic()
    try:
        if progress_enabled:
            _print_progress("exact comparison", 0, predicate_count)
        if skip_ratio(top_skip_fraction) > 0:
            rank_selector = RankedCandidates(
                top_skip_fraction,
                Path(index["shingle_path"]).with_name("ranked_candidates.sqlite")
                if lsh_query else None,
            )
        for predicate_id in range(predicate_count):
            candidate_sources: Dict[int, int] = {}
            window_stop = min(predicate_count, predicate_id + window_radius + 1)
            for candidate_id in range(predicate_id + 1, window_stop):
                candidate_sources[candidate_id] = 1

            if connection is not None and lsh_candidate_limit > 0:
                current_hashes = hashes_for(predicate_id)
                try:
                    signature = one_permutation_minhash(
                        current_hashes,
                        config.minhash_size,
                    )
                finally:
                    del current_hashes
                match_counts: Dict[int, int] = {}
                fetch_limit = max(
                    1,
                    int(lsh_candidate_limit)
                    * max(1, int(config.lsh_bucket_fetch_multiplier)),
                )
                query_parameters: List[Any] = []
                for bucket in lsh_band_keys(
                    signature,
                    config.lsh_rows_per_band,
                ):
                    query_parameters.extend((
                        sqlite3.Binary(bucket),
                        predicate_id,
                        fetch_limit + 1,
                    ))
                rows_per_band = [0] * lsh_band_count
                rows = connection.execute(lsh_query, query_parameters).fetchall()
                for band_index, raw_candidate_id in rows:
                    band_index = int(band_index)
                    rows_per_band[band_index] += 1
                    if rows_per_band[band_index] > fetch_limit:
                        bucket_fetch_truncations += 1
                        continue
                    candidate_id = int(raw_candidate_id)
                    match_counts[candidate_id] = match_counts.get(candidate_id, 0) + 1

                ranked = sorted(
                    match_counts.items(),
                    key=lambda item: (-item[1], item[0]),
                )
                if len(ranked) > lsh_candidate_limit:
                    lsh_candidate_truncated_predicates += 1
                for candidate_id, _matches in ranked[:lsh_candidate_limit]:
                    candidate_sources[candidate_id] = (
                        candidate_sources.get(candidate_id, 0) | 2
                    )

            for candidate_id in sorted(candidate_sources):
                source_code = candidate_sources[candidate_id]
                intersection, union = 0, 1

                candidate_counts[predicate_id] += 1
                candidate_counts[candidate_id] += 1
                if ast_store is None:
                    raise SimilarityAnalysisError("AST index is unavailable")
                if (ast_store.parse_status[predicate_id]
                        and ast_store.parse_status[candidate_id]):
                    left_nodes = int(ast_store.node_counts[predicate_id])
                    right_nodes = int(ast_store.node_counts[candidate_id])
                    node_total = left_nodes + right_nodes
                    left_fingerprint = int(
                        ast_store.tree_fingerprints[predicate_id]
                    )
                    right_fingerprint = int(
                        ast_store.tree_fingerprints[candidate_id]
                    )
                    if left_fingerprint == right_fingerprint:
                        distance = 0
                    else:
                        cache_key = (
                            min(left_fingerprint, right_fingerprint),
                            max(left_fingerprint, right_fingerprint),
                        )
                        cached_distance = ast_distance_cache.get(cache_key)
                        if cached_distance is not None:
                            ast_cache_hits += 1
                            ast_distance_cache.move_to_end(cache_key)
                            distance = cached_distance
                        else:
                            left_tree = ast_store.prepared_for(predicate_id)
                            right_tree = ast_store.prepared_for(candidate_id)
                            bound = distance_lower_bound(left_tree, right_tree)
                            upper_numerator = node_total - bound
                            upper_denominator = node_total + bound
                            # The old maximum-only bound cannot discard
                            # scores needed for a lower rank. Ranked mode
                            # keeps every valid candidate's exact score;
                            # repeated-pair caching/native TED still apply.
                            if not ast_ranked and not (
                                ast_bound_can_improve(predicate_id, candidate_id,
                                                     upper_numerator, upper_denominator)
                                or ast_bound_can_improve(candidate_id, predicate_id,
                                                        upper_numerator, upper_denominator)
                            ):
                                distance = None
                                ast_bound_skipped_pairs += 1
                            else:
                                distance = exact_distance(left_tree, right_tree, bound)
                                ast_distance_evaluations += 1
                                ast_distance_cache[cache_key] = distance
                                if len(ast_distance_cache) > _AST_DISTANCE_CACHE_LIMIT:
                                    ast_distance_cache.popitem(last=False)
                    if distance is not None:
                        update_best_ast(
                            predicate_id, candidate_id, distance, node_total, source_code
                        )
                        update_best_ast(
                            candidate_id, predicate_id, distance, node_total, source_code
                        )
                exact_pairs += 1
                if source_code == 1:
                    window_pairs += 1
                elif source_code == 2:
                    lsh_pairs += 1
                else:
                    combined_pairs += 1

            if rank_selector is not None:
                # Every earlier predicate has supplied its incoming matches;
                # this predicate's outgoing window/LSH matches are now complete.
                selected = rank_selector.finish(
                    predicate_id,
                    int(rank_candidate_counts[predicate_id]),
                )
                if selected is not None:
                    intersection, union, candidate_id, source_code = selected
                    best_score_numerators[predicate_id] = intersection
                    best_score_denominators[predicate_id] = union
                    best_scores[predicate_id] = intersection / union
                    best_distances[predicate_id] = (union - intersection) // 2
                    best_node_totals[predicate_id] = (union + intersection) // 2
                    nearest_ids[predicate_id] = candidate_id
                    match_sources[predicate_id] = source_code

            if progress_enabled:
                now = time.monotonic()
                if now - last_progress_log >= 10.0 and predicate_id + 1 < predicate_count:
                    _print_progress("exact comparison", predicate_id + 1, predicate_count)
                    last_progress_log = now
    finally:
        if rank_selector is not None:
            rank_selector.close()
        shingle_view.release()
        if shingle_map is not None:
            shingle_map.close()
        shingle_file.close()

    if progress_enabled:
        _print_progress("exact comparison", predicate_count, predicate_count)

    stats = {
        "predicate_count": predicate_count,
        "window_radius": int(window_radius),
        "lsh_candidate_limit": int(lsh_candidate_limit),
        "exact_pairs": int(exact_pairs),
        "window_only_pairs": int(window_pairs),
        "lsh_only_pairs": int(lsh_pairs),
        "window_and_lsh_pairs": int(combined_pairs),
        "bucket_fetch_truncations": int(bucket_fetch_truncations),
        "lsh_candidate_truncated_predicates": int(
            lsh_candidate_truncated_predicates
        ),
    }
    stats.update({
        "similarity_mode": "ast_ted",
        "top_skip_fraction": float(top_skip_fraction),
        "score_selection": "descending_rank" if ast_ranked else "maximum",
        "rank_rule": "floor(scored_candidate_count * top_skip_fraction) + 1",
        "ast_parsed_predicates": sum(
            1 for value in ast_store.parse_status if value
        ) if ast_store is not None else 0,
        "ast_parse_failed_predicates": sum(
            1 for value in ast_store.parse_status if not value
        ) if ast_store is not None else 0,
        "ast_parse_error_types": (
            dict(ast_store.parse_error_types) if ast_store is not None else {}
        ),
        "ast_distance_cache_hits": int(ast_cache_hits),
        "ast_bound_skipped_pairs": int(ast_bound_skipped_pairs),
        "ast_distance_evaluations": int(ast_distance_evaluations),
        "ast_distance_engine": distance_backend_info()["engine"],
    })

    return {
        "best_scores": best_scores,
        "best_distances": best_distances,
        "best_node_totals": best_node_totals,
        "nearest_ids": nearest_ids,
        "candidate_counts": candidate_counts,
        "match_sources": match_sources,
        "rank_candidate_counts": rank_candidate_counts if ast_ranked else None,
        "stats": stats,
    }


def run_analysis(
    *,
    project_root: Optional[Path] = None,
    input_path: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    config: SimilarityAnalysisConfig = CONFIG,
) -> Dict[str, Any]:
    """Analyze every nonblank line in the LLM experiment's satisfiable TXT."""
    # LLM comparison always uses window-only maximum AST similarity.
    # These two controls are fixed here, not exposed in config.py.
    lsh_candidate_limit = 0
    top_skip_fraction = 0
    config = replace(config, executable_only=False, max_predicates_to_read=None, window_radius=10)
    _validate_config(config)
    ast_dependency_info = (
        dependency_versions()
    )
    package_dir = Path(__file__).resolve().parent
    project_root = (
        Path(project_root).resolve()
        if project_root is not None
        else package_dir.parent.resolve()
    )
    input_path = (Path(input_path).resolve() if input_path is not None
                  else package_dir.parent / "results" / "satisfiable.txt")
    run_dir = (Path(output_dir).resolve() if output_dir is not None
               else package_dir.parent / "results" / "similarity_result")
    run_dir.mkdir(parents=True, exist_ok=True)
    work_dir = Path(tempfile.mkdtemp(prefix=".work-", dir=run_dir))

    manifest_path = run_dir / "run_manifest.json"
    manifest: Dict[str, Any] = {
        "status": "running",
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "project_root": project_root.as_posix(),
        "algorithm": {
            "tokenization": "SQL lexical tokens",
            "shingles": f"{config.shingle_size}-token set",
            "candidate_generation": "local window only (radius 10); LSH disabled",
            "final_score": (
                "normalized labeled AST tree edit similarity"
            ),
            "ast_dependencies": ast_dependency_info,
        },
        "configured": asdict(config),
    }
    manifest["algorithm"]["ast_similarity_formula"] = (
        "(nodes_a + nodes_b - TED) / (nodes_a + nodes_b + TED)"
    )
    manifest["algorithm"]["ast_tree_representation"] = "semantic-predicate-ast-v3"
    manifest["algorithm"]["ast_database_dialect"] = database_dialect()
    manifest["algorithm"]["ast_database_config_source"] = "LLM PostgreSQL experiment"
    manifest["algorithm"]["ast_edit_costs"] = {
        "insert": 1, "delete": 1, "different_label_replacement": 2,
        "equal_label_replacement": 0,
    }
    if skip_ratio(top_skip_fraction) > 0:
        manifest["algorithm"]["final_score"] = (
            "normalized labeled AST tree edit similarity at descending rank "
            "floor(scored_candidate_count * top_skip_fraction) + 1"
        )
    atomic_json_dump(manifest, manifest_path)

    timings: Dict[str, float] = {}
    index: Optional[Dict[str, Any]] = None
    connection: Optional[sqlite3.Connection] = None
    try:
        started = time.monotonic()
        backend = distance_backend_info()
        timings["ast_accelerator_setup"] = time.monotonic() - started
        manifest["algorithm"]["ast_distance_backend"] = backend
        print(f"[PredicateSimilarity] AST distance backend: {backend['engine']}", flush=True)
        if not backend["accelerated"]:
            print("[PredicateSimilarity] AST accelerator unavailable; using exact Python "
                  "fallback (slower). Optional acceleration: install numpy and numba. Reason: "
                  + str(backend["fallback_reason"]), flush=True)
        atomic_json_dump(manifest, manifest_path)
        input_files, rounds = load_input_manifest(
            project_root=project_root,
            input_path=input_path,
        )
        manifest["predicate_input_path"] = _relative_to_project(input_path, project_root)

        build_lsh = lsh_candidate_limit > 0
        phase_started = time.monotonic()
        index = _build_index(
            input_files=input_files,
            rounds=rounds,
            work_dir=work_dir,
            config=config,
            build_lsh=build_lsh,
        )
        timings["indexing"] = time.monotonic() - phase_started
        connection = index["connection"]
        predicate_count = len(index["offsets"])
        manifest["algorithm"]["ast_index_workers"] = index["ast_index_workers"]
        manifest["algorithm"]["ast_index_batch_size"] = int(config.ast_index_batch_size)

        window_radius = _auto_bounded_value(
            configured=config.window_radius,
            target_budget=config.window_target_pair_budget,
            minimum=config.window_radius_min,
            maximum=config.window_radius_max,
            predicate_count=predicate_count,
        )
        lsh_candidate_limit = min(lsh_candidate_limit, max(0, predicate_count - 1))

        manifest["effective"] = {
            "predicate_count": predicate_count,
            "executable_only": bool(config.executable_only),
            "raw_manifest_predicate_count": int(
                index["raw_manifest_predicate_count"]
            ),
            "excluded_by_validation_count": int(
                index["excluded_by_validation_count"]
            ),
            "manifest_predicate_count": int(index["manifest_predicate_count"]),
            "max_predicates_to_read": index["configured_read_limit"],
            "input_truncated_by_read_limit": bool(
                index["input_truncated_by_limit"]
            ),
            "window_radius": window_radius,
            "lsh_candidate_limit": lsh_candidate_limit,
            "top_skip_fraction": top_skip_fraction,
            "lsh_band_count": (
                int(config.minhash_size) // int(config.lsh_rows_per_band)
                if build_lsh
                else 0
            ),
        }
        manifest["predicate_rounds"] = rounds
        manifest["input_files"] = [
            {
                key: value
                for key, value in file_info.items()
                if key not in {"path", "_validation_status"}
            }
            for file_info in input_files
        ]
        manifest["index_stats"] = {
            "total_shingles": int(index["total_shingles"]),
            "skipped_blank_lines": int(index["skipped_blank_lines"]),
        }
        atomic_json_dump(manifest, manifest_path)

        phase_started = time.monotonic()
        similarities = _calculate_similarities(
            index=index,
            config=config,
            window_radius=window_radius,
            lsh_candidate_limit=lsh_candidate_limit,
            top_skip_fraction=top_skip_fraction,
        )
        similarities["stats"].update({
            "executable_only": bool(config.executable_only),
            "raw_manifest_predicate_count": int(
                index["raw_manifest_predicate_count"]
            ),
            "excluded_by_validation_count": int(
                index["excluded_by_validation_count"]
            ),
            "manifest_predicate_count": int(index["manifest_predicate_count"]),
            "max_predicates_to_read": index["configured_read_limit"],
            "input_truncated_by_read_limit": bool(
                index["input_truncated_by_limit"]
            ),
        })
        timings["candidate_search_and_exact_ast_ted"] = (
            time.monotonic() - phase_started
        )

        phase_started = time.monotonic()
        report_result = write_similarity_reports(
            run_dir=run_dir,
            input_files=manifest["input_files"],
            predicate_hash_path=index["predicate_hash_path"],
            token_counts=index["token_counts"],
            shingle_counts=index["shingle_counts"],
            source_file_ids=index["source_file_ids"],
            source_lines=index["source_lines"],
            nearest_ids=similarities["nearest_ids"],
            candidate_counts=similarities["candidate_counts"],
            match_sources=similarities["match_sources"],
            examples_per_bucket=config.examples_per_bucket,
            detail_gzip_compression_level=config.detail_gzip_compression_level,
            analysis_stats=similarities["stats"],
            timings=timings,
            best_scores=similarities.get("best_scores"),
            best_distances=similarities.get("best_distances"),
            best_node_totals=similarities.get("best_node_totals"),
            rank_candidate_counts=similarities.get("rank_candidate_counts"),
            ast_parse_status=(
                index["ast_store"].parse_status
                if index.get("ast_store") is not None else None
            ),
        )
        timings["reporting"] = time.monotonic() - phase_started
        # The first summary write happens inside report generation. Refresh only
        # its timing field so the final summary also records reporting overhead.
        report_result["summary"]["timings_seconds"] = {
            key: float(value) for key, value in timings.items()
        }
        atomic_json_dump(report_result["summary"], run_dir / "summary.json")

        manifest.update({
            "status": "completed",
            "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "analysis_stats": similarities["stats"],
            "timings_seconds": timings,
            "output_files": report_result["output_files"] + ["run_manifest.json"],
        })
        atomic_json_dump(manifest, manifest_path)
        print(
            f"[PredicateSimilarity] completed: predicates={predicate_count}",
            flush=True,
        )
        return {
            "success": True,
            "run_dir": str(run_dir),
            "manifest_path": str(manifest_path),
            "summary": report_result["summary"],
        }
    except Exception as exc:
        manifest.update({
            "status": "failed",
            "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "error": str(exc),
            "traceback": traceback.format_exc(limit=20),
            "timings_seconds": timings,
        })
        try:
            atomic_json_dump(manifest, manifest_path)
        except Exception:
            pass
        raise
    finally:
        if connection is not None:
            connection.close()
        if index is not None and index.get("ast_store") is not None:
            index["ast_store"].close()
        if not config.keep_work_directory:
            shutil.rmtree(work_dir, ignore_errors=True)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Analyze all predicates in the LLM satisfiable TXT file."
    )
    parser.add_argument("--project-root", type=Path, default=None)
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args(argv)

    try:
        run_analysis(
            project_root=args.project_root,
            input_path=args.input,
            output_dir=args.output_dir,
        )
        return 0
    except Exception as exc:
        print(f"[PredicateSimilarity] failed: {exc}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
