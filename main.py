"""Launch the full experiment using the existing src configuration."""

from pathlib import Path
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile


def _file_state(path):
    if not path.is_file():
        return None
    stat = path.stat()
    return stat.st_mtime_ns, stat.st_ctime_ns, stat.st_size


def _clear_previous_results(project_dir):
    data_dir = project_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    data_root = data_dir.resolve()
    if data_root != project_dir.resolve() / "data":
        raise RuntimeError("Refusing to clear a data directory outside the project.")
    for old_output in data_dir.iterdir():
        if old_output.name.casefold() == "readme.md":
            continue
        if old_output.is_symlink() or not old_output.is_dir():
            old_output.unlink()
        else:
            if old_output.resolve().parent != data_root:
                raise RuntimeError(f"Refusing to clear an output outside data: {old_output}")
            shutil.rmtree(old_output)
    return data_dir


def _read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _format_stat(value, places=4, suffix=""):
    if value is None or not math.isfinite(float(value)):
        return "N/A"
    return f"{float(value):.{places}f}{suffix}"


def _summary_text(graph, stage3, similarity):
    """Summarize existing results without rerunning generation or validation."""
    nodes = graph["nodes"]
    edge_count = sum(
        len(slot) for node in nodes.values() for slot in node.get("slots", [])
    )
    generated = int(stage3["cumulative_generated"])
    tlp = stage3.get("tlp", {})
    executable = int(tlp.get("cases_validated", 0))
    satisfiable = int(tlp.get("cases_satisfiable", 0))
    execution_rate = 100.0 * executable / generated if generated else None
    satisfaction_rate = 100.0 * satisfiable / executable if executable else None

    timings = stage3.get("phase_timings", {})
    generation_times = [timings.get(key) for key in (
        "predicate_traversal_render_seconds", "tlp_generation_seconds",
    )]
    total_seconds = None
    if all(value is not None and math.isfinite(float(value)) and float(value) >= 0
           for value in generation_times):
        total_seconds = sum(float(value) for value in generation_times)
    total_speed = generated / total_seconds if total_seconds else None

    similarity = similarity or {}
    lines = [
        f"Knowledge graph nodes: {len(nodes)}",
        f"Knowledge graph edges: {edge_count}",
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


def _copy_predicates_without_rates(source_path, output_path):
    # Remove only the two rate lines; preserve the heading and all TLP variants.
    with source_path.open("r", encoding="utf-8", newline="") as source:
        for prefix in ("[Stage3] Executability rate:", "[Stage3] Satisfiability rate:"):
            if not source.readline().startswith(prefix):
                raise ValueError(f"Unexpected predicate report header: {source_path}")
        with output_path.open("w", encoding="utf-8", newline="") as output:
            shutil.copyfileobj(source, output, length=1024 * 1024)


def _archive_results(project_dir, previous_states, similarity_root, previous_runs):
    source_dir = project_dir / "src"
    predicate_path = source_dir / "BugFinding/generated_test_cases/satisfiable_predicates.txt"
    status_path = source_dir / "pkg/generated_sqls/stage3_run_status.json"
    for path in (predicate_path, status_path):
        state = _file_state(path)
        if state is None or state == previous_states[path]:
            print(f"[Archive] No new {path.name}; existing data retained.", flush=True)
            return

    similarity_runs = []
    for run_dir in set(similarity_root.glob("run_*")) - previous_runs:
        manifest_path = run_dir / "run_manifest.json"
        if not run_dir.is_dir() or not manifest_path.is_file():
            continue
        manifest = _read_json(manifest_path)
        if manifest.get("status") == "completed" and not manifest.get("skipped"):
            similarity_runs.append(run_dir)
    similarity = None
    if similarity_runs:
        latest_run = max(similarity_runs, key=lambda path: (path / "summary.json").stat().st_mtime_ns)
        similarity = _read_json(latest_run / "summary.json")
    else:
        print("[Archive] No completed similarity results from this run; statistics are N/A.", flush=True)

    summary = _summary_text(
        _read_json(source_dir / "pkg/knowledge_graph/global_knowledge_graph.json"),
        _read_json(status_path),
        similarity,
    )
    # Prepare both exports before removing old results, so report errors leave
    # the previous data intact. Temporary files never touch the source outputs.
    with tempfile.TemporaryDirectory(prefix=".robusta-results-", dir=project_dir) as temporary:
        staging_dir = Path(temporary)
        _copy_predicates_without_rates(predicate_path, staging_dir / "satisfiable_predicates.txt")
        (staging_dir / "experiment_summary.txt").write_text(summary, encoding="utf-8")
        data_dir = _clear_previous_results(project_dir)
        for name in ("satisfiable_predicates.txt", "experiment_summary.txt"):
            (staging_dir / name).replace(data_dir / name)
    print("[Archive] Final experiment results are saved in Robusta/data/.", flush=True)


def main() -> int:
    project_dir = Path(__file__).resolve().parent
    source_dir = project_dir / "src"
    tracked_files = [
        source_dir / "pkg/generated_sqls/stage3_run_status.json",
        source_dir / "BugFinding/generated_test_cases/satisfiable_predicates.txt",
    ]
    previous_states = {path: _file_state(path) for path in tracked_files}
    similarity_root = source_dir / "stage3_predicate_similarity_analysis/outputs"
    previous_runs = set(similarity_root.glob("run_*"))
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    result = subprocess.run(
        [sys.executable, str(source_dir / "main.py"), "auto_cycle"],
        cwd=source_dir,
        env=env,
    )
    if result.returncode == 0:
        _archive_results(project_dir, previous_states, similarity_root, previous_runs)
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
