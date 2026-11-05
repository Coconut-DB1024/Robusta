from dataclasses import dataclass, fields
import json
from pathlib import Path
from typing import Optional


CONFIGURATION_PATH = Path(__file__).resolve().parents[2] / "configuration-lerror.json"


@dataclass
class DBConfig:
    db_type: str = "postgresql"
    host: str = ""
    port: Optional[int] = None
    user: str = ""
    password: str = ""
    database: str = ""
    connect_timeout: int = 5


def _load_configuration():
    try:
        values = json.loads(CONFIGURATION_PATH.read_text(encoding="utf-8-sig"))
    except OSError as exc:
        raise ValueError(f"Cannot read L-error configuration: {CONFIGURATION_PATH}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Invalid JSON in {CONFIGURATION_PATH}: line {exc.lineno}, column {exc.colno}"
        ) from exc
    if not isinstance(values, dict):
        raise ValueError(f"{CONFIGURATION_PATH} must contain a JSON object")

    options = {key: value for key, value in values.items() if not key.startswith("_comment_")}
    experiment = options.pop("experiment", "all")
    if experiment not in ("all", "Robusta-TPC-H", "Robusta-JOB"):
        raise ValueError("configuration-lerror.json: experiment must be all, Robusta-TPC-H, or Robusta-JOB")
    unknown = options.keys() - {item.name for item in fields(DBConfig)}
    if unknown:
        raise ValueError(f"configuration-lerror.json: unknown fields: {sorted(unknown)}")

    database = DBConfig(**options)
    for name in ("db_type", "host", "user", "password", "database"):
        if not isinstance(getattr(database, name), str):
            raise ValueError(f"configuration-lerror.json: {name} must be a JSON string")
    if database.db_type != "postgresql":
        raise ValueError("仅支持postgresql")
    if database.port is not None and (
        type(database.port) is not int or not 1 <= database.port <= 65535
    ):
        raise ValueError("configuration-lerror.json: port must be null or an integer from 1 to 65535")
    if type(database.connect_timeout) is not int or database.connect_timeout < 1:
        raise ValueError("configuration-lerror.json: connect_timeout must be a positive integer")
    return database, experiment


DB_CONFIG, EXPERIMENT = _load_configuration()
