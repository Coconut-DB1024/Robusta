"""Launch the full experiment using the existing src configuration."""

from pathlib import Path
from datetime import datetime
import json
import shutil
import subprocess
import sys


def _file_state(path):
    if not path.is_file():
        return None
    stat = path.stat()
    return stat.st_mtime_ns, stat.st_ctime_ns, stat.st_size


def _archive_results(project_dir, result_files, previous_states, similarity_root, previous_runs):
    files = []
    for source, name in result_files:
        state = _file_state(source)
        if state is not None and state != previous_states[source]:
            files.append((source, name))
        else:
            print(f"[Archive] No new result for {name}; an older file will not be copied.", flush=True)

    similarity_runs = []
    for run_dir in sorted(set(similarity_root.glob("run_*")) - previous_runs):
        if not run_dir.is_dir():
            continue
        manifest_path = run_dir / "run_manifest.json"
        if not manifest_path.is_file():
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("status") == "completed" and not manifest.get("skipped"):
            similarity_runs.append(run_dir)
    if not similarity_runs:
        print("[Archive] No completed similarity results from this run.", flush=True)
    if not files and not similarity_runs:
        return

    data_dir = project_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    run_name = datetime.now().strftime("run_%Y%m%d_%H%M%S_%f")
    archive_dir = data_dir / run_name
    archive_dir.mkdir()
    for source, name in files:
        shutil.copy2(source, archive_dir / name)
    if similarity_runs:
        similarity_archive = archive_dir / "similarity"
        similarity_archive.mkdir()
        for run_dir in similarity_runs:
            shutil.copytree(run_dir, similarity_archive / run_dir.name,
                            ignore=shutil.ignore_patterns(".work"))
            statistics_path = run_dir.with_name(run_dir.name + ".txt")
            if statistics_path.is_file():
                shutil.copy2(statistics_path, similarity_archive / statistics_path.name)
    print(f"[Archive] Experiment results: {archive_dir}", flush=True)


def main() -> int:
    project_dir = Path(__file__).resolve().parent
    source_dir = project_dir / "src"
    result_files = [
        (source_dir / "pkg/knowledge_graph/global_knowledge_graph.json", "global_knowledge_graph.json"),
        (source_dir / "BugFinding/generated_test_cases/satisfiable_predicates.txt", "satisfiable_predicates.txt"),
    ]
    previous_states = {path: _file_state(path) for path, _ in result_files}
    similarity_root = source_dir / "stage3_predicate_similarity_analysis/outputs"
    previous_runs = set(similarity_root.glob("run_*"))
    result = subprocess.run(
        [sys.executable, str(source_dir / "main.py"), "auto_cycle"],
        cwd=source_dir,
    )
    if result.returncode == 0:
        _archive_results(project_dir, result_files, previous_states, similarity_root, previous_runs)
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
