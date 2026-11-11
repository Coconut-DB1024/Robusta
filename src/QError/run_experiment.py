import argparse
import os
from pathlib import Path
import subprocess
import sys


if __name__ == "__main__":
    script = Path(__file__).resolve()
    root = script.parent
    environment = root / ".venv"
    python = environment / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    if python.is_file() and Path(sys.prefix).resolve() != environment.resolve():
        if Path(sys.executable).absolute() == python:
            raise SystemExit("The project Python environment could not be activated.")
        raise SystemExit(subprocess.call([str(python), str(script), *sys.argv[1:]]))

    sys.dont_write_bytecode = True
    sys.path.insert(0, str(root / "code"))
    from experiments import run

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path)
    parser.add_argument("--archive-dir", type=Path)
    arguments = parser.parse_args()
    run(root, config_path=arguments.config, archive_root=arguments.archive_dir)
