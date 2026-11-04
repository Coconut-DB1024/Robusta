from __future__ import annotations

import csv
import gzip
import json
import math
import mmap
import os
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

from .similarity import BUCKET_ORDER, score_bucket
from .ranked_selection import selected_rank, skip_ratio


SOURCE_NAMES = {
    0: "none",
    1: "window",
    2: "lsh",
    3: "window+lsh",
}


def atomic_json_dump(data: Any, output_path: Path) -> None:
    output_path = Path(output_path)
    partial = output_path.with_name(output_path.name + ".partial")
    with open(partial, "w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    os.replace(partial, output_path)


def _digest_hex(hash_view: memoryview, predicate_id: int) -> str:
    start = int(predicate_id) * 16
    return bytes(hash_view[start : start + 16]).hex()


def _locator(
    predicate_id: int,
    source_file_ids: Sequence[int],
    source_lines: Sequence[int],
    input_files: Sequence[Mapping[str, Any]],
    hash_view: memoryview,
) -> Dict[str, Any]:
    file_id = int(source_file_ids[predicate_id])
    return {
        "predicate_id": int(predicate_id),
        "source_file_id": file_id,
        "source_file": str(input_files[file_id]["relative_path"]),
        "source_line": int(source_lines[predicate_id]),
        "predicate_hash": _digest_hex(hash_view, predicate_id),
    }


def write_similarity_reports(
    *,
    run_dir: Path,
    input_files: Sequence[Mapping[str, Any]],
    predicate_hash_path: Path,
    token_counts: Sequence[int],
    shingle_counts: Sequence[int],
    source_file_ids: Sequence[int],
    source_lines: Sequence[int],
    nearest_ids: Sequence[int],
    candidate_counts: Sequence[int],
    match_sources: Sequence[int],
    examples_per_bucket: int,
    detail_gzip_compression_level: int,
    analysis_stats: Mapping[str, Any],
    timings: Mapping[str, float],
    best_scores: Sequence[float] | None = None,
    best_distances: Sequence[int | float] | None = None,
    best_node_totals: Sequence[int] | None = None,
    ast_parse_status: Sequence[int] | None = None,
    rank_candidate_counts: Sequence[int] | None = None,
) -> Dict[str, Any]:
    """Write bounded per-predicate detail plus compact distribution reports."""
    run_dir = Path(run_dir)
    predicate_count = len(nearest_ids)
    distribution = {label: 0 for label in BUCKET_ORDER}
    example_locators = {label: [] for label in BUCKET_ORDER}
    rank_ratio = skip_ratio(analysis_stats.get("top_skip_fraction", 0))
    scored_predicate_count = 0
    average_score_sum = 0.0
    running_mean = 0.0
    squared_deviation_sum = 0.0
    threshold_counts = {40: 0, 80: 0}

    details_path = run_dir / "predicate_similarity_details.jsonl.gz"
    hash_file = open(predicate_hash_path, "rb")
    hash_map = mmap.mmap(hash_file.fileno(), 0, access=mmap.ACCESS_READ)
    hash_view = memoryview(hash_map)
    try:
        with gzip.open(
            details_path,
            "wt",
            encoding="utf-8",
            compresslevel=max(1, min(9, int(detail_gzip_compression_level))),
            newline="\n",
        ) as detail_handle:
            for predicate_id in range(predicate_count):
                nearest_id = int(nearest_ids[predicate_id])
                similarity = (
                    float(best_scores[predicate_id])
                    if best_scores is not None and float(best_scores[predicate_id]) >= 0.0
                    else None
                )
                bucket = score_bucket(similarity) if similarity is not None else None
                distance = (
                    best_distances[predicate_id]
                    if best_distances is not None else None
                )
                node_total = (
                    int(best_node_totals[predicate_id])
                    if best_node_totals is not None else None
                )
                if rank_ratio and similarity is None:
                    # Ranked mode has no selected pair in this case. Keep
                    # the legacy maximum-mode output unchanged at fraction 0.
                    distance = node_total = None
                if bucket is not None:
                    distribution[bucket] += 1
                    scored_predicate_count += 1
                    average_score_sum += float(similarity)
                    # Welford accumulation keeps population variance stable
                    # without storing scores or rounding individual values.
                    delta = float(similarity) - running_mean
                    running_mean += delta / scored_predicate_count
                    squared_deviation_sum += delta * (float(similarity) - running_mean)
                    for percent in threshold_counts:
                        if similarity <= percent / 100.0:
                            threshold_counts[percent] += 1

                record = {
                    **_locator(
                        predicate_id,
                        source_file_ids,
                        source_lines,
                        input_files,
                        hash_view,
                    ),
                    "token_count": int(token_counts[predicate_id]),
                    "shingle_count": int(shingle_counts[predicate_id]),
                    "similarity_bucket": bucket,
                    "candidate_count": int(candidate_counts[predicate_id]),
                    "match_source": SOURCE_NAMES.get(
                        int(match_sources[predicate_id]),
                        "unknown",
                    ),
                    "nearest": None,
                }
                record.update({
                    "max_similarity": similarity,
                    "ast_parse_success": bool(
                        ast_parse_status is None or ast_parse_status[predicate_id]
                    ),
                    "max_tree_edit_distance": distance,
                    "ast_node_total": node_total,
                })
                if rank_ratio:
                    valid_candidate_count = (
                        int(rank_candidate_counts[predicate_id])
                        if rank_candidate_counts is not None
                        else int(candidate_counts[predicate_id]) if similarity is not None else 0
                    )
                    record["scored_candidate_count"] = valid_candidate_count
                    record["selected_rank"] = (
                        selected_rank(valid_candidate_count, rank_ratio)
                        if similarity is not None else 0
                    )
                if nearest_id >= 0:
                    record["nearest"] = _locator(
                        nearest_id,
                        source_file_ids,
                        source_lines,
                        input_files,
                        hash_view,
                    )

                detail_handle.write(
                    json.dumps(record, ensure_ascii=False, separators=(",", ":"))
                    + "\n"
                )

                if bucket is not None and len(example_locators[bucket]) < max(0, int(examples_per_bucket)):
                    example_locators[bucket].append(record)
    finally:
        hash_view.release()
        hash_map.close()
        hash_file.close()

    distribution_rows = []
    cumulative = 0
    for bucket in BUCKET_ORDER:
        count = int(distribution[bucket])
        cumulative += count
        row = {
            "bucket": bucket,
            "count": count,
            "percentage": (
                100.0 * count / predicate_count if predicate_count else 0.0
            ),
            "cumulative_count": cumulative,
        }
        row["scored_percentage"] = (
            100.0 * count / scored_predicate_count
            if scored_predicate_count else 0.0
        )
        distribution_rows.append(row)

    csv_path = run_dir / "similarity_distribution.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                ("bucket", "count", "percentage", "scored_percentage", "cumulative_count")
            ),
        )
        writer.writeheader()
        writer.writerows(distribution_rows)

    candidate_total = sum(int(value) for value in candidate_counts)
    summary = {
        "predicate_count": predicate_count,
        "distribution": distribution_rows,
        "average_candidates_per_predicate": (
            candidate_total / predicate_count if predicate_count else 0.0
        ),
        "analysis_stats": dict(analysis_stats),
        "timings_seconds": {key: float(value) for key, value in timings.items()},
    }
    summary.update({
        "scored_predicate_count": scored_predicate_count,
        "unscored_predicate_count": predicate_count - scored_predicate_count,
        "average_similarity": (
            average_score_sum / scored_predicate_count
            if scored_predicate_count else None
        ),
        "similarity_standard_deviation": (
            math.sqrt(max(0.0, squared_deviation_sum / scored_predicate_count))
            if scored_predicate_count else None
        ),
        "similarity_mode": "ast_ted",
        "similarity_thresholds": {
            f"at_most_{percent}_percent": {
                "count": count,
                "fraction": count / scored_predicate_count if scored_predicate_count else None,
            }
            for percent, count in threshold_counts.items()
        },
    })
    atomic_json_dump(summary, run_dir / "summary.json")
    atomic_json_dump(example_locators, run_dir / "examples_by_bucket.json")

    report_lines = [
        "# Stage-3 Predicate Similarity Report",
        "",
        f"- Predicates: {predicate_count}",
        f"- Executable-only filter: {bool(analysis_stats.get('executable_only', False))}",
        f"- Raw Stage-3 predicates in manifest: {int(analysis_stats.get('raw_manifest_predicate_count', predicate_count))}",
        f"- Excluded by Validation (not successful): {int(analysis_stats.get('excluded_by_validation_count', 0))}",
        f"- Exact candidate pairs: {int(analysis_stats.get('exact_pairs', 0))}",
        f"- Effective window radius: {analysis_stats.get('window_radius')}",
        f"- Effective LSH candidate limit: {analysis_stats.get('lsh_candidate_limit')}",
        f"- Average candidates per predicate: {summary['average_candidates_per_predicate']:.4f}",
        f"- Truncated LSH bucket fetches: {int(analysis_stats.get('bucket_fetch_truncations', 0))}",
        f"- Predicates whose LSH candidates were capped: {int(analysis_stats.get('lsh_candidate_truncated_predicates', 0))}",
        "",
        "| Similarity | Count | Percentage |",
        "|---|---:|---:|",
    ]
    report_lines[4:4] = [
        f"- Scored predicates: {scored_predicate_count}",
        f"- Unscored predicates: {predicate_count - scored_predicate_count}",
        f"- Similarity mode: ast_ted",
        f"- Average {'selected ' if rank_ratio else ''}similarity: {summary['average_similarity']}",
    ]
    if rank_ratio:
        report_lines[8:8] = [
            f"- Top skip fraction: {float(rank_ratio)}",
            "- Selection: descending rank floor(successfully scored candidate count * fraction) + 1; ties by smaller ID.",
            "- Parse failures and candidates without a scored AST pair do not enter the rank; unscored predicates have rank 0.",
            "- This selects one ranked candidate, not the mean of the highest-scoring fraction.",
            "- Legacy detail keys max_similarity/max_tree_edit_distance/nearest describe the selected rank's score, TED and predicate.",
        ]
    for row in distribution_rows:
        report_lines.append(
            f"| {row['bucket']} | {row['count']} | {row['percentage']:.6f}% |"
        )
    report_lines.extend([
        "",
        (
            "The local window and optional MinHash/LSH generate candidates only. "
            "AST scores use semantic predicate trees for the configured database dialect "
            "and normalized tree edit distance."
        ),
        ("The selected rank is over bounded candidates, not an all-pairs statistic."
         if rank_ratio else "The maximum is over bounded candidates, not an exhaustive all-pairs maximum."),
        "File id, line number, and predicate hash locate source text.",
        "",
    ])
    report_lines.insert(
        -1,
        "AST normalization: similarity = (nodes_a + nodes_b - TED) / "
        "(nodes_a + nodes_b + TED), "
        "with insert/delete cost 1 and different-label replacement cost 2. "
        "Columns, constants, types and project col(...) placeholders each form "
        "one node. Scalar properties stay in their originating node label; "
        "child order is retained without added ARG/ROLE nodes or labels "
        "(semantic-predicate-ast-v3).",
    )
    with open(run_dir / "report.md", "w", encoding="utf-8", newline="\n") as handle:
        handle.write("\n".join(report_lines))

    statistics_path = run_dir.parent / f"{run_dir.name}.txt"
    partial_path = statistics_path.with_name(statistics_path.name + ".partial")
    try:
        with open(partial_path, "w", encoding="utf-8", newline="\n") as handle:
            for label, key in (
                ("Average similarity", "average_similarity"),
                ("Standard deviation", "similarity_standard_deviation"),
            ):
                value = summary[key]
                formatted = f"{value:.6f}" if value is not None else "N/A"
                handle.write(f"{label}: {formatted}\n")
        os.replace(partial_path, statistics_path)
    finally:
        partial_path.unlink(missing_ok=True)

    return {
        "summary": summary,
        "output_files": [
            details_path.name,
            csv_path.name,
            "summary.json",
            "report.md",
            "examples_by_bucket.json",
            f"../{statistics_path.name}",
        ],
    }
