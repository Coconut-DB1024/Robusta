import json
import runpy
import sys
from pathlib import Path


ROBUSTA_ROOT = Path(__file__).resolve().parents[2]
CONFIGURATION_PATH = ROBUSTA_ROOT / "configuration.json"
INITIALIZERS = {
    "mysql": "mysql_db_initializer.py",
    "postgresql": "postgresql_db_initializer.py",
    "oracle": "oracle_db_initializer.py",
    "sqlserver": "sqlserver_db_initializer.py",
}


def main():
    configuration = json.loads(CONFIGURATION_PATH.read_text(encoding="utf-8-sig"))
    if not isinstance(configuration, dict):
        raise ValueError("Robusta/configuration.json must contain a JSON object.")
    db_type = str(configuration.get("db_type", "")).strip().lower()
    if db_type not in INITIALIZERS:
        raise ValueError(
            "Unsupported db_type in Robusta/configuration.json. "
            "Supported values: mysql, postgresql, oracle, sqlserver."
        )
    source_path = str(ROBUSTA_ROOT / "src")
    if source_path not in sys.path:
        sys.path.insert(0, source_path)
    initializer_path = Path(__file__).resolve().parent / INITIALIZERS[db_type]
    runpy.run_path(str(initializer_path), run_name="__main__")


if __name__ == "__main__":
    main()
