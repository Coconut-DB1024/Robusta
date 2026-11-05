from __future__ import annotations

import json
import math
from typing import Any

import psycopg

from .hints import PARAMETERS, HintConfig


SETTINGS_SQL = "SELECT " + ", ".join(
    f"current_setting('{parameter}')" for parameter in PARAMETERS
)


def error_details(error: Exception) -> dict[str, Any]:
    return {
        "type": type(error).__name__,
        "message": str(error),
        "sqlstate": getattr(error, "sqlstate", None),
    }


def connect_database(connection_options: dict[str, Any] | None = None) -> Any:
    options = dict(connection_options or {})
    options.setdefault("connect_timeout", 10)
    options.setdefault("application_name", "l_error")
    options["autocommit"] = True
    return psycopg.connect(**options)


def read_settings(cursor: Any) -> dict[str, str]:
    cursor.execute(SETTINGS_SQL)
    row = cursor.fetchone()
    if row is None or len(row) != len(PARAMETERS):
        raise ValueError("PostgreSQL did not return all six optimizer parameters")
    settings = dict(zip(PARAMETERS, row))
    if any(value not in ("on", "off") for value in settings.values()):
        raise ValueError("Unexpected PostgreSQL optimizer parameter value")
    return settings


def read_default_settings(connection: Any, statement_timeout_ms: int) -> dict[str, str]:
    try:
        with connection.cursor() as cursor:
            cursor.execute("BEGIN READ ONLY")
            cursor.execute(f"SET LOCAL statement_timeout = {statement_timeout_ms}")
            return read_settings(cursor)
    finally:
        connection.rollback()


def parse_explain(value: Any) -> tuple[Any, float]:
    explain = json.loads(value) if isinstance(value, str) else value
    if not isinstance(explain, list) or len(explain) != 1:
        raise ValueError("EXPLAIN JSON must contain exactly one query plan")
    top = explain[0]
    if not isinstance(top, dict) or not isinstance(top.get("Plan"), dict):
        raise ValueError("EXPLAIN JSON is missing its Plan object")
    latency = top.get("Execution Time")
    if (
        isinstance(latency, bool)
        or not isinstance(latency, (int, float))
        or not math.isfinite(latency)
        or latency < 0
    ):
        raise ValueError("EXPLAIN JSON has no valid top-level Execution Time")
    return explain, float(latency)


def empty_record(
    hint: HintConfig, repetition: int, execution_order: int
) -> dict[str, Any]:
    return {
        "hint_id": hint.hint_id,
        "repetition": repetition,
        "execution_order": execution_order,
        "requested_settings": hint.settings,
        "settings": {},
        "attempted": False,
        "explain_started": False,
        "success": False,
        "status": "not_executed",
        "latency_ms": None,
        "error": None,
        "explain": None,
    }


def execute_measurement(
    connection: Any,
    sql: str,
    hint: HintConfig,
    repetition: int,
    execution_order: int,
    statement_timeout_ms: int,
) -> dict[str, Any]:
    record = empty_record(hint, repetition, execution_order)
    if connection.closed or connection.broken:
        record["error"] = {
            "type": "ConnectionUnavailable",
            "message": "The original database session is unavailable; no reconnection was made",
            "sqlstate": None,
        }
        return record

    record["attempted"] = True
    try:
        with connection.cursor() as cursor:
            cursor.execute("BEGIN READ ONLY")
            cursor.execute(f"SET LOCAL statement_timeout = {statement_timeout_ms}")
            for parameter, value in hint.settings.items():
                cursor.execute(f"SET LOCAL {parameter} = {value}")
            record["settings"] = read_settings(cursor)
            record["explain_started"] = True
            cursor.execute("EXPLAIN (ANALYZE, FORMAT JSON)\n" + sql, prepare=False)
            row = cursor.fetchone()
            if row is None:
                raise ValueError("EXPLAIN returned no result")
            record["explain"] = row[0]
            explain, latency = parse_explain(row[0])
            record.update(
                explain=explain, latency_ms=latency, success=True, status="success"
            )
    except Exception as error:
        record["error"] = error_details(error)
        is_cancelled = getattr(error, "sqlstate", None) == "57014"
        record["status"] = (
            "timeout"
            if is_cancelled and "statement timeout" in str(error).lower()
            else "cancelled" if is_cancelled else "error"
        )
    finally:
        try:
            connection.rollback()
        except Exception as error:
            cleanup_error = error_details(error)
            record["cleanup_error"] = cleanup_error
            record["error"] = record["error"] or cleanup_error
            record["success"] = False
            record["status"] = "cleanup_error"
    return record
