
from __future__ import annotations

import math
from pathlib import Path
import shutil
import tempfile
from typing import Any


def _format_stat(value, places=4, suffix=""):
    if value is None or not math.isfinite(float(value)):
        return "N/A"
    return f"{float(value):.{places}f}{suffix}"


def _summary_text(experiment: dict[str, Any], similarity: dict[str, Any]) -> str:
    generated = int(experiment["generated_predicates"])
    executable = int(experiment["db_executable_count"])
    satisfiable = int(experiment["db_satisfiable_count"])
    execution_rate = 100.0 * executable / generated if generated else None
    satisfaction_rate = 100.0 * satisfiable / executable if executable else None

    seconds = experiment.get("llm_request_total_seconds")
    total_speed = None
    if seconds is not None and math.isfinite(float(seconds)) and float(seconds) > 0:
        total_speed = generated / float(seconds)

    lines = [
        f"Knowledge graph nodes: {int(experiment['tokens_count'])}",
        f"Knowledge graph edges: {int(experiment['discovered_unique_edges'])}",
        f"Executability rate: {_format_stat(execution_rate, 2, '%')}",
        f"Satisfiability rate: {_format_stat(satisfaction_rate, 2, '%')}",
        f"Average similarity: {_format_stat(similarity.get('average_similarity'))}",
        f"Standard deviation: {_format_stat(similarity.get('similarity_standard_deviation'))}",
    ]
    for percent in (40, 80):
        threshold = similarity.get("similarity_thresholds", {}).get(
            f"at_most_{percent}_percent", {}
        )
        fraction = threshold.get("fraction")
        speed = total_speed * fraction if total_speed is not None and fraction is not None else None
        lines.append(
            f"Predicate generation speed (similarity <= {percent}%): "
            f"{_format_stat(speed, 4, ' predicates/s')}"
        )
    return "\n".join(lines) + "\n"


def archive_results(
    *,
    project_dir: Path,
    predicate_path: Path,
    experiment: dict[str, Any],
    similarity_result: dict[str, Any],
) -> Path | None:
    if experiment.get("run_status") != "complete" or not similarity_result.get("success"):
        return None

    summary = _summary_text(experiment, similarity_result["summary"])
    project_dir = Path(project_dir).resolve()
    data_dir = project_dir / "data"
    output_dir = data_dir / "LLM_results"
    if data_dir.resolve() != data_dir or output_dir.resolve() != output_dir:
        raise RuntimeError("Refusing to archive LLM results through a redirected data directory.")
    data_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix=".llm-results-", dir=data_dir) as temporary:
        staging_dir = Path(temporary)
        shutil.copyfile(predicate_path, staging_dir / "satisfiable_predicates.txt")
        (staging_dir / "experiment_summary.txt").write_text(summary, encoding="utf-8")
        if output_dir.exists():
            if data_dir.resolve() != data_dir or output_dir.resolve() != data_dir / "LLM_results":
                raise RuntimeError("Refusing to clear LLM results outside the expected data directory.")
            shutil.rmtree(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        for name in ("satisfiable_predicates.txt", "experiment_summary.txt"):
            (staging_dir / name).replace(output_dir / name)
    print("[Archive] Final LLM experiment results are saved in Robusta/data/LLM_results/.", flush=True)
    return output_dir
