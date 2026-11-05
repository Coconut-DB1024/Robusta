"""Launch the full experiment using the existing src configuration."""

from pathlib import Path
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile


_SOURCE_OUTPUT_DIRECTORIES = (
    "output",
    "output_cluster",
    "BugFinding/bug_reports",
    "BugFinding/generated_test_cases",
    "data/cluster_final_output",
    "pkg/counted_graph",
    "pkg/knowledge_graph",
    "pkg/unfolded_graph",
    "RL/checkpoints",
    "RL/rl_logs",
    "RL/time_collect",
    "stage3_predicate_similarity_analysis/outputs",
)
_GENERATED_SQLS_DIRECTORY = "pkg/generated_sqls"
_SOURCE_OUTPUT_FILES = (
    "codet5-small/encoder_state.pt",
    "RL/ppo_sql_model.zip",
)
_MODEL_FILES = (
    "config.json",
    "generation_config.json",
    "merges.txt",
    "model.safetensors",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
)
_MODEL_DOWNLOAD_HELP = (
    "Please run download_codet5_small.py to download the transformer model or "
    "download codet5-small.zip from GitHub Releases and extract it into the "
    "root directory of the Robusta project."
)


def _model_files_complete(directory):
    """Check the eight required filenames without inspecting file contents."""
    return directory.is_dir() and all((directory / name).is_file() for name in _MODEL_FILES)


def _prepare_transformer(project_dir):
    project_dir = Path(project_dir).resolve()
    source_dir = project_dir / "src"
    runtime_model = source_dir / "codet5-small"
    root_model = project_dir / "codet5-small"
    if _model_files_complete(runtime_model):
        return True
    if not _model_files_complete(root_model):
        if not runtime_model.exists() and not root_model.exists():
            problem = "Transformer model is missing. Execution paused."
        else:
            problem = "Transformer model files are incomplete. Exiting."
        print(f"{problem} {_MODEL_DOWNLOAD_HELP}", flush=True)
        return False

    # Keep replacement and temporary cleanup inside the actual src directory.
    if source_dir.resolve() != source_dir or runtime_model.resolve() != runtime_model:
        raise RuntimeError(f"Refusing to replace a redirected model directory: {runtime_model}")
    with tempfile.TemporaryDirectory(prefix=".codet5-copy-", dir=source_dir) as temporary:
        staging_dir = Path(temporary)
        copied_model = staging_dir / "model"
        previous_model = staging_dir / "previous"
        shutil.copytree(root_model, copied_model)
        if not _model_files_complete(copied_model):
            raise RuntimeError(f"Transformer model files are incomplete. Exiting. {_MODEL_DOWNLOAD_HELP}")
        # Preserve the previous copy until the complete new copy is installed.
        if runtime_model.exists():
            runtime_model.replace(previous_model)
        try:
            copied_model.replace(runtime_model)
        except BaseException:
            if previous_model.exists():
                previous_model.replace(runtime_model)
            raise
    return True


def _reset_source_outputs(project_dir, *, db_type):
    """Reset only the named outputs, once before starting the experiment."""
    source_dir = Path(project_dir).resolve() / "src"
    if source_dir.resolve() != source_dir:
        raise RuntimeError(f"Refusing to clear outputs through a redirected src directory: {source_dir}")
    if not (source_dir / "main.py").is_file():
        raise FileNotFoundError(f"Experiment entry point not found: {source_dir / 'main.py'}")

    directories = [source_dir / name for name in _SOURCE_OUTPUT_DIRECTORIES]
    files = [source_dir / name for name in _SOURCE_OUTPUT_FILES]
    generated_dir = source_dir / _GENERATED_SQLS_DIRECTORY
    # Validate every absolute target before deleting anything. Symlinks and
    # junctions must not redirect cleanup to model files, inputs or another tree.
    for path in directories + [generated_dir] + files:
        if path.resolve() != path or not path.is_relative_to(source_dir):
            raise RuntimeError(f"Refusing to clear a redirected output path: {path}")
        if path.exists():
            expected_type = path.is_file() if path in files else path.is_dir()
            if not expected_type:
                raise RuntimeError(f"Unexpected output path type: {path}")

    feedback_dir = generated_dir / "feedback"
    generated_children = list(generated_dir.iterdir()) if generated_dir.exists() else []
    for path in generated_children:
        if path.resolve() != path or path.parent != generated_dir:
            raise RuntimeError(f"Refusing to clear a redirected output path: {path}")
    if feedback_dir.exists() and not feedback_dir.is_dir():
        raise RuntimeError(f"Feedback path is not a directory: {feedback_dir}")

    # Match the existing Stage-3 feedback filenames and database aliases.
    feedback_db = {
        "mysql": "mysql",
        "postgresql": "postgresql", "postgres": "postgresql", "pg": "postgresql",
        "oracle": "oracle", "oracledb": "oracle", "oracle_ai26": "oracle",
        "sqlserver": "sqlserver",
    }.get(str(db_type or "").strip().lower())
    feedback_seed = None
    if feedback_db is not None:
        candidate = feedback_dir / f"stage3_signature_feedback-{feedback_db}.json"
        if candidate.resolve() != candidate:
            raise RuntimeError(f"Refusing to copy a redirected feedback file: {candidate}")
        if candidate.is_file():
            feedback_seed = candidate

    for directory in directories:
        if directory.exists():
            shutil.rmtree(directory)
        directory.mkdir(parents=True, exist_ok=True)
    # Preserve the baseline folder, but discard all per-run files and folders.
    generated_dir.mkdir(parents=True, exist_ok=True)
    for path in generated_children:
        if path == feedback_dir:
            continue
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
    if feedback_seed is not None:
        shutil.copy2(feedback_seed, generated_dir / feedback_seed.name)
    for path in files:
        path.unlink(missing_ok=True)


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
    project_dir = Path(__file__).resolve().parents[2]
    source_dir = project_dir / "src"
    if not _prepare_transformer(project_dir):
        return 1
    if str(source_dir) not in sys.path:
        sys.path.insert(0, str(source_dir))
    from RL.config import DB_CONFIG

    _reset_source_outputs(project_dir, db_type=DB_CONFIG.db_type)
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
