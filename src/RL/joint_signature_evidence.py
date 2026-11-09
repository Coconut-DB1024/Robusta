from __future__ import annotations

from contextlib import contextmanager
from itertools import product
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional
import atexit
import hashlib
import json
import os
import re
import tempfile
import threading
import time
import uuid

from RL.runtime_logging import report_joint_signature_flush_failure


EVIDENCE_FILE_NAME = "joint_signatures.json"
_MAX_COMBINATIONS_PER_AST_NODE = 64
_ORACLE_MAX_COMBINATIONS_PER_AST_NODE = 96
_ORACLE_DB_TYPES = frozenset({"oracle", "oracledb", "oracle_ai26"})
SHARDED_EVIDENCE_DB_TYPES = _ORACLE_DB_TYPES | {"sqlserver"}


ORACLE_TYPE_FAILURE_CODES = frozenset({
    "902", "932", "939", "1760", "12704", "22848", "22849", "22950",
    "30081", "30175", "30495", "40796", "61801", "61803",
})
ORACLE_VALUE_FAILURE_CODES = frozenset({
    "1401", "1410", "1426", "1428", "1438", "1465", "1476", "1481",
    "1722", "6502", "12728", "12733", "22835", "22998", "30494",
    "31011", "40441", "40478", "40597", "61800", "64203",
})


SQLSERVER_FAILURE_WEIGHTS = {
    "type_failure_count": 1.0,
    "value_failure_count": 0.40,
    "execution_failure_count": 0.20,
    "timeout_count": 0.0,
    "environment_failure_count": 0.0,
}


def classify_sqlserver_error(error_code=None, error=None, *, sqlstate=None,
                             timed_out=False, cancelled=False) -> str:
    text = str(error or "")
    state = str(sqlstate or "").strip().upper()
    if not state:
        match = re.search(r"(?:^|\[|['\"])([A-Z0-9]{5})(?=\]|['\"]|\s*\|)", text)
        state = match.group(1) if match else ""
    codes = set(re.findall(r"\((-?\d+)\)(?=\s*(?:\([A-Za-z][A-Za-z0-9_]*\)|;|$|['\"]))", text))
    raw_code = str(error_code if error_code is not None else "").strip()
    if re.fullmatch(r"-?\d+", raw_code):
        codes.add(str(int(raw_code)))
    elif not state and re.fullmatch(r"[A-Z0-9]{5}", raw_code.upper()):
        state = raw_code.upper()
    lower = text.lower()

    if (state.startswith(("08", "IM")) or state in {"28000", "40001"}
            or codes & {"53", "64", "233", "258", "229", "297", "916", "1101", "1105", "1205", "1222",
                        "1807", "18456", "4060", "40501", "40197", "40613", "49918", "49919", "49920",
                        "701", "802", "8645", "8651", "9002", "10053", "10054", "10060"}
            or any(s in lower for s in ("failed to connect to sql server", "login failed", "login timeout",
                                        "communication link failure", "communications link failure",
                                        "connection is closed", "connection was closed", "connection reset",
                                        "connection refused", "transport-level error", "deadlock victim",
                                        "sql server schema is unavailable", "sql server schema has no column"))):
        return "environment_failure_count"
    if timed_out or state in {"HYT00", "HYT01", "S1T00"} or "-2" in codes or any(
            s in lower for s in ("query timeout", "timeout expired", "timed out")):
        return "timeout_count"
    if cancelled or state == "HY008" or any(s in lower for s in ("operation canceled", "operation cancelled", "query was canceled", "query was cancelled")):
        return "environment_failure_count"

    if codes & {"206", "257", "306", "402", "403", "421", "529", "8116", "8117", "4145"}:
        return "type_failure_count"

    if codes & {"241", "242", "245", "248", "289", "295", "517", "535", "536", "537", "8114", "8115", "8134", "8152", "2628", "3623", "9828", "9838", "9839", "13609"}:
        return "value_failure_count"
    if state.startswith("22"):
        return "value_failure_count"

    if not codes or codes <= {"0", "8180"}:
        if (any(s in lower for s in ("operand type clash:", "expression of non-boolean type"))
                or re.search(r"\bdata types .+ are incompatible in the\b", lower)
                or re.search(r"\bargument data type .+ is invalid for argument\b", lower)):
            return "type_failure_count"
        if any(s in lower for s in ("conversion failed when converting", "error converting data type",
                                    "arithmetic overflow", "divide by zero", "invalid length parameter",
                                    "string or binary data would be truncated")):
            return "value_failure_count"
    return "execution_failure_count"


def _config_value(name: str, default: Any) -> Any:
    try:
        from RL.config import RL_ENV_CONFIG

        return getattr(RL_ENV_CONFIG, name, default)
    except Exception:
        return default


def _current_db_type() -> str:
    try:
        from RL.config import DB_CONFIG

        return str(getattr(DB_CONFIG, "db_type", "mysql") or "mysql").lower().strip()
    except Exception:
        return "mysql"


def _node_return_type(node: Any) -> str:
    return str(getattr(node, "return_type", None) or "UNKNOWN")


def _node_physical_type(node: Any) -> str:
    for name in (
        "oracle_physical_type",
        "physical_type",
        "stage3_column_type",
        "column_type",
    ):
        value = getattr(node, name, None)
        if value is None and isinstance(node, dict):
            value = node.get(name)
        if value is not None and str(value).strip():
            return str(value).strip().lower()
    return ""


def _node_id(node: Any) -> str:
    return f"{getattr(node, 'token', 'UNKNOWN')}::{_node_return_type(node)}"


def _slot_id(parent_token: str, child_index: int) -> int:
    if str(parent_token).upper() in {"IN", "NOT IN"} and child_index >= 1:
        return 1
    return int(child_index)


def _subtree_fingerprint(node: Any, cache: Optional[Dict[int, str]] = None) -> str:
    if cache is None:
        cache = {}
    node_key = id(node)
    if node_key in cache:
        return cache[node_key]
    payload = {
        "token": str(getattr(node, "token", "UNKNOWN")),
        "node_type": (
            "terminal"
            if not (getattr(node, "children", []) or [])
            else str(getattr(node, "node_type", ""))
        ),
        "return_type": _node_return_type(node),
        "children": [_subtree_fingerprint(child, cache) for child in (getattr(node, "children", []) or [])],
    }
    if _current_db_type() in _ORACLE_DB_TYPES or _current_db_type() == "sqlserver":
        physical_type = _node_physical_type(node)
        if physical_type:
            payload["physical_type"] = physical_type
    fingerprint = hashlib.sha1(
        json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    cache[node_key] = fingerprint
    return fingerprint


def extract_joint_signatures(parse_result: Any) -> List[Dict[str, Any]]:
    if not bool(getattr(parse_result, "success", False)):
        return []
    ast_root = getattr(parse_result, "ast_root", None)
    if ast_root is None:
        return []

    records: List[Dict[str, Any]] = []
    fingerprint_cache: Dict[int, str] = {}

    max_combinations = (
        _ORACLE_MAX_COMBINATIONS_PER_AST_NODE
        if _current_db_type() in _ORACLE_DB_TYPES
        else _MAX_COMBINATIONS_PER_AST_NODE
    )

    def visit(node: Any) -> None:
        children = list(getattr(node, "children", []) or [])
        token = str(getattr(node, "token", "UNKNOWN"))
        if str(getattr(node, "node_type", "")) == "operator" and children:
            grouped: Dict[int, List[Any]] = {}
            for child_index, child in enumerate(children):
                grouped.setdefault(_slot_id(token, child_index), []).append(child)

            slot_ids = sorted(grouped)
            if slot_ids == list(range(len(slot_ids))):
                emitted = 0
                for chosen_children in product(*(grouped[idx] for idx in slot_ids)):
                    child_return_types = [
                        _node_return_type(child) for child in chosen_children
                    ]
                    slot_child_tokens = [
                        str(getattr(child, "token", "UNKNOWN"))
                        for child in chosen_children
                    ]
                    slot_child_node_types = [
                        str(getattr(child, "node_type", ""))
                        for child in chosen_children
                    ]
                    slot_child_fingerprints = [
                        _subtree_fingerprint(child, fingerprint_cache) for child in chosen_children
                    ]
                    slot_child_physical_types = [
                        _node_physical_type(child) for child in chosen_children
                    ]
                    tuple_payload = {
                        "outer_node_id": _node_id(node),
                        "slot_ids": slot_ids,
                        "slot_child_node_ids": [
                            _node_id(child) for child in chosen_children
                        ],
                    }
                    if _current_db_type() in _ORACLE_DB_TYPES or _current_db_type() == "sqlserver":
                        tuple_payload["slot_child_physical_types"] = [
                            str(value or "").strip().lower()
                            for value in slot_child_physical_types
                        ]
                    tuple_hash = hashlib.sha1(
                        json.dumps(
                            tuple_payload,
                            ensure_ascii=True,
                            sort_keys=True,
                            separators=(",", ":"),
                        ).encode("utf-8")
                    ).hexdigest()
                    record = {
                        "outer_node_id": _node_id(node),
                        "outer_token": token,
                        "outer_return_type": _node_return_type(node),
                        "slot_child_node_ids": [_node_id(child) for child in chosen_children],
                        "slot_child_return_types": child_return_types,

                        "arg_types": list(child_return_types),
                        "slot_ids": list(slot_ids),
                        "slot_child_tokens": slot_child_tokens,
                        "slot_child_node_types": slot_child_node_types,
                        "slot_child_fingerprints": slot_child_fingerprints,
                        "subtree_fingerprint": _subtree_fingerprint(node, fingerprint_cache),
                        "arity": len(slot_ids),
                        "tuple_hash": tuple_hash,
                    }
                    if _current_db_type() in _ORACLE_DB_TYPES:
                        record["slot_child_physical_types"] = slot_child_physical_types
                        record["physical_type"] = _node_physical_type(node)
                        if getattr(node, "oracle_local_binding", False):
                            record["oracle_binding_version"] = 1
                    if _current_db_type() == "sqlserver":
                        record["slot_child_physical_types"] = slot_child_physical_types
                        record["physical_type"] = _node_physical_type(node)
                        record["sqlserver_binding_version"] = int(bool(getattr(node, "sqlserver_local_binding", False)))
                    records.append(record)
                    emitted += 1
                    if emitted >= max_combinations:
                        break

        for child in children:
            visit(child)

    visit(ast_root)
    return records


def _evidence_key(db_type: str, record: Dict[str, Any]) -> str:
    payload = {
        "db_type": db_type,
        "outer_node_id": record.get("outer_node_id", ""),
        "slot_child_node_ids": list(record.get("slot_child_node_ids", []) or []),
    }
    if str(db_type or "").lower().strip() in _ORACLE_DB_TYPES:
        payload["slot_ids"] = list(record.get("slot_ids", []) or [])
        payload["arity"] = int(record.get("arity", 0) or 0)
        payload["subtree_fingerprint"] = str(record.get("subtree_fingerprint", "") or "")
        payload["slot_child_physical_types"] = [
            str(value or "").strip().lower()
            for value in (record.get("slot_child_physical_types", []) or [])
        ]
        if record.get("oracle_binding_version") == 1:
            physical = payload["slot_child_physical_types"]
            payload.pop("subtree_fingerprint", None)
            payload["oracle_binding_version"] = 1

            payload["unresolved_child_fingerprints"] = [
                "" if i < len(physical) and physical[i] else fp
                for i, fp in enumerate(record.get("slot_child_fingerprints", []) or [])
            ]
    if str(db_type).lower().strip() == "sqlserver":
        physical = list(record.get("slot_child_physical_types") or [])
        payload.update({
            "sqlserver_binding_version": int(record.get("sqlserver_binding_version", 0)),
            "slot_child_physical_types": physical,
            "physical_type": record.get("physical_type", ""),
            "unresolved_child_fingerprints": [
                "" if i < len(physical) and physical[i] else fp
                for i, fp in enumerate(record.get("slot_child_fingerprints") or [])
            ],
        })
    raw = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def _is_definite_type_failure(
    db_type: str,
    error_code: Optional[Any],
    error: Optional[str],
) -> bool:
    code = str(error_code or "").strip()
    text = str(error or "").lower()
    normalized_db = str(db_type or "").lower().strip()

    if normalized_db in {"postgresql", "postgres", "pg"}:
        return code in {"42804", "42809", "42846", "42883"} or any(
            marker in text
            for marker in (
                "operator does not exist",
                "function does not exist",
                "no function matches the given name and argument types",
                "cannot cast type",
                "datatype mismatch",
            )
        )

    if normalized_db == "mysql":
        return code in {"1267", "3854", "3995", "4078", "4079"} or any(
            marker in text
            for marker in (
                "illegal parameter data types",
                "illegal mix of collations",
                "illegal operand types",
                "cannot convert string",
                "cannot be used in conjunction with",
            )
        )

    if normalized_db in _ORACLE_DB_TYPES:
        oracle_code = code.replace("ORA-", "").replace("DPY-", "").lstrip("0") or "0"
        return oracle_code in ORACLE_TYPE_FAILURE_CODES or any(
            marker in text
            for marker in (
                "inconsistent datatypes",
                "cannot compare",
                "datatype mismatch",
                "invalid datatype",
                "illegal argument for function",
                "too many arguments for function",
                "argument should be of numeric or date/datetime type",
                "invalid type given for an argument",
                "cannot use nclob type as comparison key",
                "not supported for comparison",
                "cannot order objects without map or order method",
                "invalid data type for datetime/interval arithmetic",
                "only boolean column or attribute can be used as a predicate",
                "must be a column or an expression of boolean",
            )
        )

    return False


def _is_definite_value_failure(
    db_type: str,
    error_code: Optional[Any],
    error: Optional[str],
) -> bool:
    code = str(error_code or "").strip()
    normalized_db = str(db_type or "").lower().strip()
    if normalized_db in {"postgresql", "postgres", "pg"} and code.startswith("22"):
        return True
    if normalized_db == "mysql" and code in {"1264", "1265", "1292", "1366", "1411", "1525", "1690", "3141"}:
        return True
    if normalized_db in _ORACLE_DB_TYPES:
        oracle_code = (
            code.replace("ORA-", "").replace("DPY-", "").lstrip("0") or "0"
        )
        if oracle_code.startswith("18") or oracle_code in ORACLE_VALUE_FAILURE_CODES:
            return True

    text = str(error or "").lower()
    return any(
        marker in text
        for marker in (
            "division by zero",
            "numeric value out of range",
            "numeric field overflow",
            "negative substring length",
            "not a valid encoding",
            "truncated incorrect",
            "incorrect integer value",
            "incorrect datetime value",
            "incorrect timestamp value",
            "incorrect date value",
            "invalid json text",
        )
    )


def _is_environment_failure(
    db_type: str,
    error_code: Optional[Any],
    error: Optional[str],
    *,
    timed_out: bool = False,
) -> bool:
    if timed_out:
        return True
    code = str(error_code or "").strip().upper()
    text = str(error or "").lower()
    normalized_db = str(db_type or "").lower().strip()

    if normalized_db in {"postgresql", "postgres", "pg"}:
        if code.startswith(("08", "53")) or code in {"57014"}:
            return True
    elif normalized_db == "mysql":
        if code in {
            "1040", "1042", "1043", "1053", "1081", "1205", "1213",
            "1317", "2002", "2003", "2005", "2006", "2013", "2055",
            "3024",
        }:
            return True
    elif normalized_db in _ORACLE_DB_TYPES:
        try:
            numeric_code = int(code.replace("ORA-", "").replace("DPY-", ""))
        except (TypeError, ValueError):
            numeric_code = None
        if numeric_code in {
            18, 20, 54, 60, 1003, 1012, 1013, 1017, 1033, 1034, 1089, 1219, 12170,
            12514, 12537, 12541, 12543, 12545, 3113, 3114, 3135,
            4024, 600, 6000, 6001, 6005, 4011, 2049, 4031, 4036, 1652,
        }:
            return True

    return any(
        marker in text
        for marker in (
            "timed out",
            "timeout",
            "can't connect",
            "cannot connect",
            "could not connect",
            "connection refused",
            "connection reset",
            "connection lost",
            "lost connection",
            "server has gone away",
            "communications link failure",
            "too many connections",
            "out of memory",
            "resource temporarily unavailable",
            "disk full",
            "deadlock",
            "lock wait timeout",
            "query execution was interrupted",
            "statement timeout",
            "failed to connect to oracle",
            "oracle not available",
            "no listener",
            "not logged on",
            "end-of-file on communication channel",
            "connection lost contact",
            "connection closed",
            "dpy-600",
            "dpy-4011",
            "maximum number of sessions",
            "maximum number of processes",
            "shared memory",
        )
    )


def _normalize_oracle_error_code(
    error_code: Optional[Any],
    error: Optional[str],
) -> str:
    raw_code = str(error_code or "").strip().upper()
    raw_error = str(error or "").upper()
    match = re.search(r"\b(?:ORA|DPY)-?(\d{3,5})\b", raw_code)
    if match is None:
        match = re.search(r"\b(?:ORA|DPY)-?(\d{3,5})\b", raw_error)
    if match is not None:
        raw_code = match.group(1)
    raw_code = raw_code.replace("ORA-", "").replace("DPY-", "")
    return raw_code.lstrip("0") or ("0" if raw_code else "")


def classify_oracle_error(
    error_code: Optional[Any] = None,
    error: Optional[str] = None,
    *,
    timed_out: bool = False,
) -> str:
    code = _normalize_oracle_error_code(error_code, error)
    text = str(error or "").lower()
    if (
        timed_out
        or code in {"51", "1013", "4024"}
        or any(
            marker in text
            for marker in (
                "call timeout",
                "statement timeout",
                "user requested cancel",
                "timed out",
                "timeout",
                "cancelled",
                "canceled",
            )
        )
    ):
        return "timeout_count"
    if _is_environment_failure("oracle", code, error, timed_out=False):
        return "environment_failure_count"
    if _is_definite_type_failure("oracle", code, error):
        return "type_failure_count"
    if _is_definite_value_failure("oracle", code, error):
        return "value_failure_count"
    return "execution_failure_count"


@contextmanager
def _file_lock(target_path: Path, timeout_seconds: float = 10.0):
    lock_path = target_path.with_name(target_path.name + ".lock")
    start = time.time()
    fd = None
    while True:
        try:
            fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_RDWR)
            os.write(fd, str(os.getpid()).encode("ascii", errors="ignore"))
            break
        except FileExistsError:
            if time.time() - start > timeout_seconds:
                raise TimeoutError(f"joint signature evidence lock timeout: {lock_path}")
            time.sleep(0.05)
    try:
        yield
    finally:
        if fd is not None:
            os.close(fd)
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass


def _load_payload(path: Path) -> Dict[str, Any]:
    if not path.exists() or path.stat().st_size <= 0:
        return {"version": 1, "entries": {}}
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, dict) or not isinstance(payload.get("entries"), dict):
            return {"version": 1, "entries": {}}
        return payload
    except Exception:
        return {"version": 1, "entries": {}}


def _atomic_write_compact(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        delete=False,
        dir=str(path.parent),
        suffix=".tmp",
    ) as handle:
        json.dump(payload, handle, ensure_ascii=False, separators=(",", ":"))
        handle.write("\n")
        temp_path = Path(handle.name)
    os.replace(str(temp_path), str(path))


def _atomic_write_durable(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", delete=False,
            dir=str(path.parent), suffix=".tmp",
        ) as handle:
            temp_path = Path(handle.name)
            json.dump(payload, handle, ensure_ascii=False, separators=(",", ":"))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(str(temp_path), str(path))
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


def _load_payload_strict(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict) or not isinstance(payload.get("entries"), dict):
        raise ValueError(f"invalid joint signature evidence payload: {path}")
    return payload


class JointSignatureEvidenceStore:
    def __init__(self, stats_dir: Path) -> None:
        self.path = Path(stats_dir) / EVIDENCE_FILE_NAME
        self.enabled = bool(
            _config_value("stage2_joint_signature_evidence_enabled", True)
        )
        self.flush_events = max(
            1,
            int(_config_value("stage2_joint_signature_flush_events", 5000)),
        )
        self.max_entries = max(
            1,
            int(_config_value("stage2_joint_signature_max_entries", 50000)),
        )
        self.db_type = _current_db_type()
        self._pending: Dict[str, Dict[str, Any]] = {}
        self._pending_events = 0
        self._mutex = threading.Lock()
        self._oracle_typer = None
        self._sqlserver_typer = None
        atexit.register(self._flush_at_exit)

    def _flush_at_exit(self) -> None:
        for attempt in range(3):
            try:
                self.flush()
                if self._sqlserver_typer is not None:
                    self._sqlserver_typer.close()
                if self._oracle_typer is not None:
                    self._oracle_typer.close()
                    if not getattr(self, "_oracle_stats_reported", False):
                        print(f"[Oracle evidence metadata] {dict(self._oracle_typer.stats)}")
                        self._oracle_stats_reported = True
                return
            except TimeoutError as exc:
                if attempt >= 2:
                    report_joint_signature_flush_failure(exc, final=True)
                    return

                time.sleep(0.10 * (attempt + 1) + 0.005 * (os.getpid() % 20))
            except Exception as exc:
                report_joint_signature_flush_failure(exc, final=True)
                return

    def _observe(
        self,
        record: Dict[str, Any],
        counter: str,
        *,
        error_code: Optional[Any] = None,
    ) -> None:
        if not self.enabled:
            return
        key = _evidence_key(self.db_type, record)
        now = float(time.time())
        with self._mutex:
            item = self._pending.get(key)
            if item is None:
                item = {
                    "db_type": self.db_type,
                    "outer_node_id": str(record.get("outer_node_id", "")),
                    "outer_token": str(record.get("outer_token", "")),
                    "outer_return_type": str(record.get("outer_return_type", "")),
                    "slot_child_node_ids": list(
                        record.get("slot_child_node_ids", []) or []
                    ),
                    "slot_child_return_types": list(
                        record.get("slot_child_return_types", []) or []
                    ),
                    "arg_types": list(record.get("arg_types", []) or []),
                    "success_count": 0,
                    "type_failure_count": 0,
                    "value_failure_count": 0,
                    "first_seen_time": now,
                    "last_seen_time": now,
                }
                if self.db_type in {"mysql", "sqlserver"} or self.db_type in _ORACLE_DB_TYPES:
                    item["execution_failure_count"] = 0
                if self.db_type in _ORACLE_DB_TYPES or self.db_type == "sqlserver":
                    item.update({
                        "slot_ids": list(record.get("slot_ids", []) or []),
                        "slot_child_tokens": list(
                            record.get("slot_child_tokens", []) or []
                        ),
                        "slot_child_node_types": list(
                            record.get("slot_child_node_types", []) or []
                        ),
                        "slot_child_fingerprints": list(
                            record.get("slot_child_fingerprints", []) or []
                        ),
                        "slot_child_physical_types": list(
                            record.get("slot_child_physical_types", []) or []
                        ),
                        "physical_type": str(record.get("physical_type", "") or ""),
                        "subtree_fingerprint": str(
                            record.get("subtree_fingerprint", "") or ""
                        ),
                        "arity": int(record.get("arity", 0) or 0),
                        "tuple_hash": str(record.get("tuple_hash", "") or ""),
                        "last_error_code": "",
                        "error_code_counts": {},
                        "oracle_binding_version": int(record.get("oracle_binding_version", 0)),
                    })
                    if self.db_type == "sqlserver":
                        item.pop("oracle_binding_version", None)
                        item["sqlserver_binding_version"] = int(record.get("sqlserver_binding_version", 0))
                self._pending[key] = item
            if (self.db_type in _ORACLE_DB_TYPES
                    and record.get("oracle_binding_version") == 1
                    and not item.get("physical_type") and record.get("physical_type")):
                item["physical_type"] = str(record["physical_type"])
            item[counter] = int(item.get(counter, 0)) + 1
            if self.db_type in _ORACLE_DB_TYPES and counter == "success_count":
                tier = ("direct_success_count" if record.get("oracle_direct_target")
                        else "context_success_count")
                item[tier] = int(item.get(tier, 0)) + 1
            normalized_error_code = str(error_code or "").strip()
            if normalized_error_code and counter != "success_count":
                item["last_error_code"] = normalized_error_code
                counts = item.setdefault("error_code_counts", {})
                if isinstance(counts, dict):
                    counts[normalized_error_code] = int(
                        counts.get(normalized_error_code, 0)
                    ) + 1
                    if len(counts) > 8:
                        keep = sorted(
                            counts.items(), key=lambda pair: (-pair[1], pair[0])
                        )[:8]
                        item["error_code_counts"] = dict(keep)
            item["last_seen_time"] = now
            self._pending_events += 1

    def _typed_oracle_evidence(self, parse_result):
        if self.db_type not in _ORACLE_DB_TYPES or parse_result is None:
            return parse_result
        if not getattr(parse_result, "oracle_executed_sql", ""):
            return parse_result
        if self._oracle_typer is None:
            from RL.oracle_evidence_types import OracleEvidenceTyper
            self._oracle_typer = OracleEvidenceTyper()
        try:
            return self._oracle_typer.enrich(parse_result)
        except Exception as exc:
            if not getattr(self, "_oracle_enrichment_warned", False):
                print(f"[Oracle evidence] metadata enrichment failed: {exc}")
                self._oracle_enrichment_warned = True
            return parse_result

    def _typed_sqlserver_evidence(self, parse_result):
        if self.db_type != "sqlserver" or not getattr(parse_result, "sqlserver_executed_sql", ""):
            return parse_result
        try:
            if self._sqlserver_typer is None:
                from RL.sqlserver_evidence_types import SQLServerEvidenceTyper
                self._sqlserver_typer = SQLServerEvidenceTyper()
            return self._sqlserver_typer.enrich(parse_result)
        except Exception as exc:
            if not getattr(self, "_sqlserver_enrichment_warned", False):
                print(f"[SQL Server evidence] metadata enrichment failed: {exc}")
                self._sqlserver_enrichment_warned = True
            return parse_result

    def observe_success_batch(self, parse_results: Iterable[Any]) -> None:
        if not self.enabled:
            return
        for parse_result in parse_results:
            parse_result = self._typed_oracle_evidence(parse_result)
            parse_result = self._typed_sqlserver_evidence(parse_result)
            records = extract_joint_signatures(parse_result)
            if self.db_type in _ORACLE_DB_TYPES:
                target = getattr(parse_result, "oracle_target_edge", None) or {}
                matched = []
                for record in records:
                    children = record.get("slot_child_node_ids", [])
                    slot = target.get("slot_index", -1)
                    if (isinstance(slot, int) and 0 <= slot < len(children)
                            and record.get("outer_node_id") == target.get("outer_node_id")
                            and children[slot] == target.get("inner_node_id")):
                        matched.append(record)

                if len(matched) == 1:
                    matched[0]["oracle_direct_target"] = True
            for record in records:
                self._observe(record, "success_count")
        self._flush_if_needed()

    def observe_target_type_failure(
        self,
        parse_result: Any,
        *,
        target_edge: Optional[Dict[str, Any]],
        target_edge_hit: bool,
        error_code: Optional[Any],
        error: Optional[str],
        timed_out: bool,
        sqlstate: Optional[str] = None,
        cancelled: bool = False,
    ) -> None:
        if (
            not self.enabled
            or not target_edge_hit
            or timed_out
            or not isinstance(target_edge, dict)
        ):
            return

        if self.db_type == "sqlserver":
            failure_counter = classify_sqlserver_error(
                error_code, error, sqlstate=sqlstate, timed_out=timed_out, cancelled=cancelled,
            )
            if failure_counter in {"timeout_count", "environment_failure_count"}:
                return
        elif self.db_type in _ORACLE_DB_TYPES:
            failure_counter = classify_oracle_error(
                error_code,
                error,
                timed_out=timed_out,
            )
            if failure_counter in {
                "timeout_count",
                "environment_failure_count",
            }:
                return
        else:
            if _is_environment_failure(
                self.db_type,
                error_code,
                error,
                timed_out=timed_out,
            ):
                return
            if _is_definite_type_failure(self.db_type, error_code, error):
                failure_counter = "type_failure_count"
            elif _is_definite_value_failure(self.db_type, error_code, error):
                failure_counter = "value_failure_count"
            elif self.db_type == "mysql":
                failure_counter = "execution_failure_count"
            else:
                return

        outer_id = str(target_edge.get("outer_node_id", ""))
        inner_id = str(target_edge.get("inner_node_id", ""))
        target_tuple_hash = str(target_edge.get("tuple_hash", "") or "")
        try:
            slot_index = int(target_edge.get("slot_index", -1))
        except Exception:
            return

        matched_records: List[Dict[str, Any]] = []
        parse_result = self._typed_oracle_evidence(parse_result)
        parse_result = self._typed_sqlserver_evidence(parse_result)
        for record in extract_joint_signatures(parse_result):
            child_ids = list(record.get("slot_child_node_ids", []) or [])
            if target_tuple_hash and str(record.get("tuple_hash", "")) != target_tuple_hash:
                continue
            if (
                record.get("outer_node_id") == outer_id
                and 0 <= slot_index < len(child_ids)
                and child_ids[slot_index] == inner_id
            ):
                matched_records.append(record)

        if (self.db_type in _ORACLE_DB_TYPES or self.db_type == "sqlserver") and not target_tuple_hash:
            if len(matched_records) != 1:
                return
        for record in matched_records:
            self._observe(record, failure_counter, error_code=error_code)
        self._flush_if_needed()

    def _flush_if_needed(self) -> None:
        with self._mutex:
            should_flush = self._pending_events >= self.flush_events
        if should_flush:
            self.flush()

    @property
    def _shard_dir(self) -> Path:
        return self.path.with_name(self.path.stem + "_pending")

    def _flush_shard(self) -> None:
        with self._mutex:
            if not self._pending:
                return
            name = f"delta-{time.time_ns():020d}-{os.getpid()}-{uuid.uuid4().hex}.json"
            _atomic_write_durable(self._shard_dir / name, {
                "version": 1, "db_type": self.db_type, "entries": self._pending,
            })
            self._pending = {}
            self._pending_events = 0

    def _finish_shard_merge(self, checkpoint_path: Path, checkpoint: Dict[str, Any]) -> int:
        payload = checkpoint.get("payload")
        names = checkpoint.get("shards")
        if (not isinstance(payload, dict) or not isinstance(payload.get("entries"), dict)
                or not isinstance(names, list) or not names
                or any(not isinstance(name, str) or Path(name).name != name
                       or not name.startswith("delta-") or not name.endswith(".json")
                       for name in names)):
            raise ValueError(f"invalid joint signature merge checkpoint: {checkpoint_path}")

        _atomic_write_durable(self.path, payload)
        for name in names:
            (self._shard_dir / name).unlink(missing_ok=True)
        checkpoint_path.unlink()
        return len(names)

    def merge_pending_shards(self) -> int:
        if not self.enabled or self.db_type not in SHARDED_EVIDENCE_DB_TYPES:
            return 0
        self.flush()  
        checkpoint_path = self._shard_dir / "merge_checkpoint.json"
        merged = 0
        if checkpoint_path.exists():
            with checkpoint_path.open("r", encoding="utf-8") as handle:
                checkpoint = json.load(handle)
            merged += self._finish_shard_merge(checkpoint_path, checkpoint)
            del checkpoint  
        shards = sorted(self._shard_dir.glob("delta-*.json"))
        if not shards:
            return merged
        payload = (_load_payload_strict(self.path) if self.path.exists()
                   else {"version": 1, "entries": {}})
        compatible_dbs = _ORACLE_DB_TYPES if self.db_type in _ORACLE_DB_TYPES else {self.db_type}
        for shard in shards:
            delta = _load_payload_strict(shard)
            if delta.get("db_type") not in compatible_dbs:
                raise ValueError(f"joint signature delta database mismatch: {shard}")
            self._merge_pending(payload, delta["entries"])
        checkpoint = {"shards": [shard.name for shard in shards], "payload": payload}
        _atomic_write_durable(checkpoint_path, checkpoint)
        return merged + self._finish_shard_merge(checkpoint_path, checkpoint)

    def _merge_pending(self, payload: Dict[str, Any], pending: Dict[str, Any]) -> None:
        entries = payload.setdefault("entries", {})
        counters = (
            "success_count",
            "type_failure_count",
            "value_failure_count",
        )
        if self.db_type in {"mysql", "sqlserver"} or self.db_type in _ORACLE_DB_TYPES:
            counters += ("execution_failure_count",)
        if self.db_type in _ORACLE_DB_TYPES:
            counters += ("direct_success_count", "context_success_count")
        for key, incoming in pending.items():
            existing = entries.get(key)
            if not isinstance(existing, dict):
                entries[key] = dict(incoming)
                continue
            for counter in counters:
                existing[counter] = int(existing.get(counter, 0)) + int(
                    incoming.get(counter, 0)
                )
            if not existing.get("arg_types") and incoming.get("arg_types"):
                existing["arg_types"] = list(incoming["arg_types"])
            for metadata_key in (
                "slot_ids",
                "slot_child_tokens",
                "slot_child_node_types",
                "slot_child_fingerprints",
                "slot_child_physical_types",
            ):
                if not existing.get(metadata_key) and incoming.get(metadata_key):
                    existing[metadata_key] = list(incoming[metadata_key])
            if not existing.get("arity") and incoming.get("arity"):
                existing["arity"] = int(incoming["arity"])
            if not existing.get("tuple_hash") and incoming.get("tuple_hash"):
                existing["tuple_hash"] = str(incoming["tuple_hash"])
            if not existing.get("subtree_fingerprint") and incoming.get("subtree_fingerprint"):
                existing["subtree_fingerprint"] = str(incoming["subtree_fingerprint"])
            if not existing.get("physical_type") and incoming.get("physical_type"):
                existing["physical_type"] = str(incoming["physical_type"])
            if incoming.get("last_error_code"):
                existing["last_error_code"] = str(
                    incoming["last_error_code"]
                )
            incoming_codes = incoming.get("error_code_counts", {})
            if isinstance(incoming_codes, dict):
                merged_codes = existing.setdefault("error_code_counts", {})
                if isinstance(merged_codes, dict):
                    for code, count in incoming_codes.items():
                        merged_codes[str(code)] = int(
                            merged_codes.get(str(code), 0)
                        ) + int(count or 0)
                    if len(merged_codes) > 8:
                        keep = sorted(
                            merged_codes.items(),
                            key=lambda pair: (-pair[1], pair[0]),
                        )[:8]
                        existing["error_code_counts"] = dict(keep)
            existing["first_seen_time"] = min(
                float(existing.get("first_seen_time", incoming["first_seen_time"])),
                float(incoming["first_seen_time"]),
            )
            existing["last_seen_time"] = max(
                float(existing.get("last_seen_time", 0.0)),
                float(incoming["last_seen_time"]),
            )

        if len(entries) > self.max_entries:
            ranked = sorted(
                entries.items(),
                key=lambda pair: (
                    int(pair[1].get("success_count", 0))
                    + int(pair[1].get("type_failure_count", 0))
                    + int(pair[1].get("value_failure_count", 0))
                    + int(pair[1].get("execution_failure_count", 0)),
                    int(pair[1].get("success_count", 0)),
                    float(pair[1].get("last_seen_time", 0.0)),
                ),
                reverse=True,
            )
            payload["entries"] = dict(ranked[: self.max_entries])
            entries = payload["entries"]

        payload["version"] = 2
        payload["updated_at"] = float(time.time())
        payload["entry_count"] = len(entries)

    def flush(self) -> None:
        if not self.enabled:
            return
        if self.db_type in SHARDED_EVIDENCE_DB_TYPES:
            self._flush_shard()
            return
        with self._mutex:
            if not self._pending:
                return
            pending = self._pending
            self._pending = {}
            self._pending_events = 0

        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with _file_lock(self.path):
                payload = _load_payload(self.path)
                self._merge_pending(payload, pending)
                _atomic_write_compact(self.path, payload)
        except Exception:
            with self._mutex:
                counters = (
                    "success_count",
                    "type_failure_count",
                    "value_failure_count",
                )
                if self.db_type in {"mysql", "sqlserver"} or self.db_type in _ORACLE_DB_TYPES:
                    counters += ("execution_failure_count",)
                if self.db_type in _ORACLE_DB_TYPES:
                    counters += ("direct_success_count", "context_success_count")
                for key, incoming in pending.items():
                    current = self._pending.get(key)
                    if current is None:
                        self._pending[key] = incoming
                        continue
                    for counter in counters:
                        current[counter] = int(current.get(counter, 0)) + int(
                            incoming.get(counter, 0)
                        )
                self._pending_events += sum(
                    int(item.get("success_count", 0))
                    + int(item.get("type_failure_count", 0))
                    + int(item.get("value_failure_count", 0))
                    + int(item.get("execution_failure_count", 0))
                    for item in pending.values()
                )
            raise
