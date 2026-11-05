import sys
import importlib.util
import os
from pathlib import Path
import subprocess

sys.dont_write_bytecode = True

DEFAULT_TIMEOUT_SECONDS = {
    "Robusta-TPC-H": 600,
    "Robusta-JOB": 600,
}

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "code"))


def select_runtime():
    if all(importlib.util.find_spec(name) is not None for name in ("psycopg", "pglast")):
        return None
    relative_python = Path("Scripts/python.exe") if os.name == "nt" else Path("bin/python")
    for directory in (ROOT / ".venv", ROOT.parent / ".venv"):
        candidate = directory / relative_python
        if candidate.is_file() and candidate.resolve() != Path(sys.executable).resolve():
            probe = subprocess.run([str(candidate), "-B", "-c", "import psycopg, pglast"],
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            if probe.returncode == 0:
                return subprocess.call([str(candidate), "-B", str(ROOT / "run.py"), *sys.argv[1:]])
    print("Missing dependencies: install Robusta/requirements.txt into your Python environment.", file=sys.stderr)
    return 2


if __name__ == "__main__":
    runtime_status = select_runtime()
    if runtime_status is not None:
        raise SystemExit(runtime_status)
    try:
        from config import DB_CONFIG, EXPERIMENT
    except ValueError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr, flush=True)
        raise SystemExit(1)
    from runner import main
    raise SystemExit(main(ROOT, EXPERIMENT, DEFAULT_TIMEOUT_SECONDS, DB_CONFIG))
