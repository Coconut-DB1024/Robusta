"""Start the independent PostgreSQL L-error experiment."""

import json
import os
from pathlib import Path
import subprocess
import sys


def main() -> int:
    project_dir = Path(__file__).resolve().parents[2]
    experiment_dir = project_dir / "src" / "L-error"
    entry_point = experiment_dir / "run.py"
    config_path = project_dir / "configuration-lerror.json"

    try:
        config = json.loads(config_path.read_text(encoding="utf-8-sig"))
        if not isinstance(config, dict):
            raise ValueError(f"Expected a JSON object in {config_path}")
        if config.get("db_type", "postgresql") != "postgresql":
            print("Only PostgreSQL is supported.", file=sys.stderr, flush=True)
            return 1
        if not entry_point.is_file():
            raise FileNotFoundError(f"L-error experiment entry point not found: {entry_point}")

        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        result = subprocess.run(
            [sys.executable, "-B", str(entry_point), *sys.argv[1:]],
            cwd=experiment_dir,
            env=env,
        )
        return result.returncode
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
