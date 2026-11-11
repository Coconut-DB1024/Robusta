import os
from pathlib import Path
import subprocess
import sys


def main() -> int:
    project_dir = Path(__file__).resolve().parents[2]
    experiment_dir = project_dir / "src" / "QError"
    environment = os.environ.copy()
    environment["PYTHONUNBUFFERED"] = "1"
    result = subprocess.run(
        [
            sys.executable,
            "-B",
            str(experiment_dir / "run_experiment.py"),
            "--config",
            str(project_dir / "configuration-qerror.json"),
            "--archive-dir",
            str(project_dir / "data" / "Qerror"),
        ],
        cwd=experiment_dir,
        env=environment,
    )
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
