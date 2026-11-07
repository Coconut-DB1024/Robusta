"""Start a fresh, independent LLM experiment using the shared configuration."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


def _clear_previous_results(project_dir: Path) -> None:
    llm_dir = project_dir / "src" / "LLM"
    results_dir = llm_dir / "results"
    # Validate the absolute cleanup target before any recursive deletion.
    if llm_dir.resolve() != llm_dir or results_dir.resolve() != results_dir:
        raise RuntimeError("Refusing to clear a redirected LLM results directory.")
    if results_dir.exists():
        shutil.rmtree(results_dir)


def main() -> int:
    project_dir = Path(__file__).resolve().parents[2]
    llm_dir = project_dir / "src" / "LLM"
    entry_point = llm_dir / "llm_pg_edge_discovery.py"
    config_path = project_dir / "configuration.json"

    try:
        config = json.loads(config_path.read_text(encoding="utf-8-sig"))
        if not isinstance(config, dict):
            raise ValueError(f"Expected a JSON object in {config_path}")
        # Check before cleanup so an unsupported DB leaves previous results intact.
        if config.get("db_type") != "postgresql":
            print("Only PostgreSQL is currently supported.", file=sys.stderr, flush=True)
            return 1
        if not entry_point.is_file():
            raise FileNotFoundError(f"LLM experiment entry point not found: {entry_point}")

        _clear_previous_results(project_dir)
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        result = subprocess.run(
            [sys.executable, str(entry_point)],
            cwd=llm_dir,
            env=env,
        )
        return result.returncode
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
