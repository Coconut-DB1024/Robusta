"""Launch the full experiment using the existing src configuration."""

from pathlib import Path
import subprocess
import sys


def main() -> int:
    source_dir = Path(__file__).resolve().parent / "src"
    result = subprocess.run(
        [sys.executable, str(source_dir / "main.py"), "auto_cycle"],
        cwd=source_dir,
    )
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
