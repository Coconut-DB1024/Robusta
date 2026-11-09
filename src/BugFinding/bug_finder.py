import os
import sys
import glob
import time
import re  
import multiprocessing
import json
import itertools
import gzip
import base64
from bisect import bisect_left, bisect_right
from pathlib import Path
from typing import List, Dict, Optional, Union, Any, Iterator


project_root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

sys.path.append(project_root_path)


from RL.config import PROJECT_ROOT, DB_CONFIG, STAGE3_CONFIG


def _tlp_variant_sqls(group):
    if not group or len(group) < 4:
        return {}
    return {"true": group[1], "false": group[2], "null": group[3]}
from RL.db_executor import create_db_executor
from RL.joint_signature_evidence import classify_oracle_error, classify_sqlserver_error
from BugFinding.oracle_worker_pool import (
    OracleValidationPool, OracleValidationDeadline, OracleWorkerFailure,
)


INPUT_DIR = PROJECT_ROOT / "BugFinding" / "generated_test_cases"

REPORT_DIR = PROJECT_ROOT / "BugFinding" / "bug_reports"


VALIDATION_BATCH_SIZE = 1000
MAX_WORKERS = min(max(1, multiprocessing.cpu_count() - 2), 24)


os.makedirs(REPORT_DIR, exist_ok=True)


def _is_test_case_file(file_path: Union[str, Path]) -> bool:
    lower_path = str(file_path).lower()
    return lower_path.endswith(".sql") or lower_path.endswith(".sql.gz")


def _open_test_case_reader(file_path: Union[str, Path]):
    if str(file_path).lower().endswith(".sql.gz"):
        return gzip.open(file_path, "rt", encoding="utf-8")
    return open(file_path, "r", encoding="utf-8")


def _select_current_test_case_files(file_paths) -> List[str]:
    selected = {}
    for file_path in file_paths:
        path_text = str(file_path)
        if not _is_test_case_file(path_text):
            continue
        logical_path = (
            path_text[:-3]
            if path_text.lower().endswith(".sql.gz")
            else path_text
        )
        logical_key = os.path.normcase(os.path.abspath(logical_path))
        existing = selected.get(logical_key)
        if existing is None:
            selected[logical_key] = path_text
            continue

        try:
            current_mtime = os.path.getmtime(path_text)
            existing_mtime = os.path.getmtime(existing)
        except OSError:
            current_mtime = existing_mtime = 0.0
        if current_mtime > existing_mtime or (
            current_mtime == existing_mtime
            and path_text.lower().endswith(".sql.gz")
        ):
            selected[logical_key] = path_text

    return sorted(selected.values())


def _prune_stage3_report_files(retention_count: int):
    if retention_count <= 0:
        return

    patterns = (
        "scan_log_*.txt",
        "bugs_found_*.txt",
        "tlp_sql_errors_*.jsonl",
        "tlp_error_summary_*.json",
        "predicate_cardinalities_*.jsonl.gz",
    )
    report_dir = Path(REPORT_DIR)
    for pattern in patterns:
        paths = sorted(
            (path for path in report_dir.glob(pattern) if path.is_file()),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        for old_path in paths[retention_count:]:
            try:
                old_path.unlink()
            except OSError as exc:
                print(f"[TLP] Warning: failed to prune old report {old_path}: {exc}")


DSL_WRAPPERS = [
    "int", "float", "double", "decimal",  
    "string", "char", "varchar",          
    "date", "datetime", "timestamp", "time", 
    "col",                                
    "hex", "bin", "bit"                   
]


worker_executor = None


def worker_init(progress=None):
    global worker_executor
    worker_executor = create_db_executor(
        autoconnect=True,
        query_timeout_seconds=int(getattr(STAGE3_CONFIG, "tlp_query_timeout", 10)),
    )
    if progress is not None:
        worker_executor._stage3_worker_progress = progress
        initialization_error = getattr(worker_executor, "initialization_error", None)
        if initialization_error is not None:
            raise initialization_error


def worker_task(args):
    file_name, idx, group, tlp_mode = args
    global worker_executor

    if _is_oracle_db():
        from BugFinding.oracle_predicate_repair import execute_oracle_group
        def clean_oracle(sql):
            for _ in range(5):
                cleaned = WRAPPER_PATTERN.sub(lambda m: m.group(2).strip(), sql)
                if cleaned == sql:
                    break
                sql = cleaned
            return sql
        tested_group, values, failure, repair_summary = execute_oracle_group(
            worker_executor, group,
            cleaner=clean_oracle,
        )
        identity = {"file": file_name, "case_idx": idx, "tlp_mode": tlp_mode,
                    "oracle_value_repair": repair_summary}
        if failure is not None:
            return "skip", {**identity, **failure}
        tested_group = [clean_oracle(sql) for sql in tested_group]
        identity["counts"] = values
        identity["variant_sqls"] = _tlp_variant_sqls(tested_group)
        if values["total"] != values["true"] + values["false"] + values["null"]:
            return "bug", (file_name, idx, tested_group, values, tlp_mode, {}, repair_summary)
        return "ok", identity

    overflow_retries = []

    def _execute(sql, role):
        if _normalized_db_type() == "sqlserver":
            from BugFinding.sql_generator import cleanup_dsl
            sql = cleanup_dsl(sql)
        else:
            for _ in range(5):
                new_sql = WRAPPER_PATTERN.sub(lambda m: m.group(2).strip(), sql)
                if new_sql == sql:
                    break
                sql = new_sql

        res = worker_executor.execute_query(sql, fetch="one")

        if not res.success:
            retry_sql, rewrite_count = (
                bound_mysql_exp_overflow(
                    sql,
                    db_type=getattr(DB_CONFIG, "db_type", ""),
                    error_code=res.error_code,
                    error=res.error,
                    argument_limit=float(
                        getattr(STAGE3_CONFIG, "tlp_mysql_exp_argument_limit", 700.0)
                    ),
                )
                if (
                    _normalized_db_type() == "mysql"
                    and bool(
                        getattr(STAGE3_CONFIG, "tlp_mysql_overflow_retry_enabled", True)
                    )
                )
                else (sql, 0)
            )
            if rewrite_count > 0 and retry_sql != sql:
                retry_res = worker_executor.execute_query(retry_sql, fetch="one")
                sql = retry_sql
                retry_info = {
                    "role": role,
                    "rewrite_count": int(rewrite_count),
                    "success": bool(retry_res.success),
                }
                overflow_retries.append(retry_info)
                if retry_res.success:
                    if retry_res.rows:
                        return True, int(list(retry_res.rows[0].values())[0]), None, sql
                    return True, 0, None, sql
                res = retry_res
                sql = retry_sql
            return (
                False,
                None,
                {
                    "file": file_name,
                    "case_idx": idx,
                    "tlp_mode": tlp_mode,
                    "role": role,
                    "sql": sql,
                    "error": res.error,
                    "error_code": res.error_code,
                    "execution_time": res.execution_time,
                    "is_timeout": bool(res.timed_out),
                    **(_sqlserver_diagnostics(res) if _normalized_db_type() == "sqlserver" else {}),
                    "overflow_retry": {
                        "attempts": len(overflow_retries),
                        "successes": sum(
                            1 for item in overflow_retries if item["success"]
                        ),
                        "rewritten_arguments": sum(
                            item["rewrite_count"] for item in overflow_retries
                        ),
                    },
                },
                sql,
            )

        if res.rows:
            return True, int(list(res.rows[0].values())[0]), None, sql

        return True, 0, None, sql

    values = {}
    executed_group = list(group)
    for group_index, (role, sql) in enumerate([
        ("total", group[0]),
        ("true", group[1]),
        ("false", group[2]),
        ("null", group[3]),
    ]):
        ok, value, error_info, executed_sql = _execute(sql, role)
        executed_group[group_index] = executed_sql
        if not ok:
            return ("skip", error_info)
        values[role] = value

    check_sum = values["true"] + values["false"] + values["null"]

    retry_summary = {
        "attempts": len(overflow_retries),
        "successes": sum(1 for item in overflow_retries if item["success"]),
        "rewritten_arguments": sum(item["rewrite_count"] for item in overflow_retries),
    }

    if values["total"] != check_sum:
        return (
            "bug",
            (
                file_name,
                idx,
                executed_group,
                {
                    "total": values["total"],
                    "true": values["true"],
                    "false": values["false"],
                    "null": values["null"],
                },
                tlp_mode,
                retry_summary,
            )
        )

    return (
        "ok",
        {
            "tlp_mode": tlp_mode,
            "file": file_name,
            "case_idx": idx,
            "overflow_retry": retry_summary,
            "counts": values,
            "variant_sqls": _tlp_variant_sqls(executed_group),
        },
    )


WRAPPER_PATTERN = re.compile(
    r"\b(" + "|".join(DSL_WRAPPERS) + r")\s*\(\s*([^()]+)\s*\)",
    re.IGNORECASE
)

RESOURCE_FUNCTION_PATTERN = re.compile(
    r"\b(?:REPEAT|LPAD|RPAD|LEFT|RIGHT|SUBSTR|SUBSTRING)\s*\(",
    re.IGNORECASE,
)

ORACLE_RESOURCE_FUNCTION_PATTERN = re.compile(
    r"\b(?:LPAD|RPAD|SUBSTR|SUBSTRB)\s*\(",
    re.IGNORECASE,
)

SQLSERVER_RESOURCE_FUNCTION_PATTERN = re.compile(
    r"\b(?:REPLICATE|SPACE|LEFT|RIGHT|SUBSTRING)\s*\(",
    re.IGNORECASE,
)

ORACLE_DB_TYPES = frozenset({"oracle", "oracledb", "oracle_ai26"})
ORACLE_DPY_CODES = frozenset({"600", "4011"})


def _normalized_db_type() -> str:
    return str(getattr(DB_CONFIG, "db_type", "") or "").strip().lower()


def _is_oracle_db() -> bool:
    return _normalized_db_type() in ORACLE_DB_TYPES


def _has_oracle_tlp_special_nodes(sql_group):
    if not _is_oracle_db() or len(sql_group or []) < 2:
        return False
    sql = re.sub(
        r"\bN?Q'(?:\[[\s\S]*?\]|\{[\s\S]*?\}|\([\s\S]*?\)|<[\s\S]*?>|"
        r"(?P<delimiter>[^\s])[\s\S]*?(?P=delimiter))'"
        r"|'(?:''|[^'])*'|\"(?:\"\"|[^\"])*\"|--[^\r\n]*|/\*[\s\S]*?\*/",
        " ",
        str(sql_group[1]),
        flags=re.IGNORECASE,
    )
    return re.search(
        r"(?<![\w$#])(?:ROWNUM|SYSDATE|SYSTIMESTAMP|CURRENT_DATE|"
        r"CURRENT_TIMESTAMP|LOCALTIMESTAMP)(?![\w$#])"
        r"|(?<![\w$#])SYS_GUID\s*\(",
        sql,
        re.IGNORECASE,
    ) is not None


def _sqlserver_diagnostics(result):
    return {"sqlstate": getattr(result, "sqlstate", None),
            "cancelled": bool(getattr(result, "cancelled", False))}


def _sqlserver_error_category(error_info):
    info = error_info or {}
    return classify_sqlserver_error(
        info.get("error_code"), info.get("error"), sqlstate=info.get("sqlstate"),
        timed_out=bool(info.get("is_timeout") or info.get("timed_out")),
        cancelled=bool(info.get("cancelled", False)),
    )


def _oracle_error_code(error_info: Optional[Dict[str, Any]]) -> str:
    info = error_info if isinstance(error_info, dict) else {}
    raw_code = str(info.get("error_code", "") or "").strip().upper()
    raw_text = str(info.get("error", "") or "")
    match = re.search(r"\b(?:ORA|DPY)-?(\d{3,5})\b", raw_code)
    if match is None:
        match = re.search(r"\b(?:ORA|DPY)-?(\d{3,5})\b", raw_text.upper())
    if match is not None:
        raw_code = match.group(1)
    raw_code = raw_code.replace("ORA-", "").replace("DPY-", "")
    return raw_code.lstrip("0") or ("0" if raw_code else "")


def _oracle_error_label(error_info: Optional[Dict[str, Any]]) -> str:
    info = error_info if isinstance(error_info, dict) else {}
    raw_code = str(info.get("error_code", "") or "").strip().upper()
    raw_text = str(info.get("error", "") or "").upper()
    match = re.search(r"\b(ORA|DPY)-?(\d{3,5})\b", raw_code)
    if match is None:
        match = re.search(r"\b(ORA|DPY)-?(\d{3,5})\b", raw_text)
    if match is not None:
        prefix, digits = match.groups()
        width = 5 if prefix == "ORA" else 4
        return f"{prefix}-{digits.zfill(width)}"
    code = _oracle_error_code(info)
    if code in ORACLE_DPY_CODES:
        return f"DPY-{code.zfill(4)}"
    return f"ORA-{code.zfill(5)}" if code else "unknown"


def _oracle_error_category(error_info: Optional[Dict[str, Any]]) -> str:
    info = error_info if isinstance(error_info, dict) else {}
    category = classify_oracle_error(
        info.get("error_code"),
        info.get("error"),
        timed_out=bool(info.get("is_timeout", False)),
    )

    if category in {"execution_failure_count", "environment_failure_count"}:
        return "other_failure_count"
    return category


def _resource_function_limits(string_length_max: int) -> Dict[str, tuple]:
    length_max = max(0, int(string_length_max))
    if _normalized_db_type() == "sqlserver":
        return {
            "LEFT": (1, 0, length_max),
            "RIGHT": (1, 0, length_max),
            "SUBSTRING": (2, 0, length_max),
            "SPACE": (0, 0, length_max),
        }
    if _is_oracle_db():
        return {
            "LPAD": (1, 0, length_max),
            "RPAD": (1, 0, length_max),
            "SUBSTR": (2, 0, length_max),
            "SUBSTRB": (2, 0, length_max),
        }
    return {
        "LPAD": (1, 0, length_max),
        "RPAD": (1, 0, length_max),
        "LEFT": (1, -length_max, length_max),
        "RIGHT": (1, -length_max, length_max),
        "SUBSTR": (2, 0, length_max),
        "SUBSTRING": (2, 0, length_max),
    }


def _consume_sql_literal_or_comment(sql: str, start: int) -> Optional[int]:
    if sql.startswith("--", start):
        newline = sql.find("\n", start + 2)
        return len(sql) if newline < 0 else newline + 1

    if sql.startswith("/*", start):
        if _normalized_db_type() == "sqlserver":
            depth, index = 1, start + 2
            while index < len(sql) and depth:
                if sql.startswith("/*", index):
                    depth += 1
                    index += 2
                elif sql.startswith("*/", index):
                    depth -= 1
                    index += 2
                else:
                    index += 1
            return index
        end = sql.find("*/", start + 2)
        return len(sql) if end < 0 else end + 2

    quote = sql[start]
    if quote == "[" and _normalized_db_type() == "sqlserver":
        index = start + 1
        while index < len(sql):
            if sql[index] == "]":
                if index + 1 < len(sql) and sql[index + 1] == "]":
                    index += 2
                    continue
                return index + 1
            index += 1
        return len(sql)
    if quote in ("'", '"'):
        index = start + 1
        while index < len(sql):
            if sql[index] == quote:
                if index + 1 < len(sql) and sql[index + 1] == quote:
                    index += 2
                    continue
                return index + 1
            if (sql[index] == "\\" and index + 1 < len(sql)
                    and _normalized_db_type() != "sqlserver"):
                index += 2
                continue
            index += 1
        return len(sql)

    if quote == "$" and _normalized_db_type() != "sqlserver":
        match = re.match(r"\$(?:[A-Za-z_][A-Za-z0-9_]*)?\$", sql[start:])
        if match:
            delimiter = match.group(0)
            end = sql.find(delimiter, start + len(delimiter))
            return len(sql) if end < 0 else end + len(delimiter)

    return None


def _find_matching_parenthesis(sql: str, open_index: int) -> int:
    depth = 0
    index = open_index
    while index < len(sql):
        token_end = _consume_sql_literal_or_comment(sql, index)
        if token_end is not None:
            index = token_end
            continue
        if sql[index] == "(":
            depth += 1
        elif sql[index] == ")":
            depth -= 1
            if depth == 0:
                return index
        index += 1
    return -1


def _split_sql_arguments(arguments_sql: str) -> List[str]:
    if not arguments_sql.strip():
        return []

    arguments = []
    start = 0
    index = 0
    parenthesis_depth = 0
    bracket_depth = 0
    while index < len(arguments_sql):
        token_end = _consume_sql_literal_or_comment(arguments_sql, index)
        if token_end is not None:
            index = token_end
            continue

        char = arguments_sql[index]
        if char == "(":
            parenthesis_depth += 1
        elif char == ")":
            parenthesis_depth -= 1
        elif char == "[":
            bracket_depth += 1
        elif char == "]":
            bracket_depth -= 1
        elif char == "," and parenthesis_depth == 0 and bracket_depth == 0:
            arguments.append(arguments_sql[start:index])
            start = index + 1
        index += 1

    arguments.append(arguments_sql[start:])
    return arguments


def _wrap_bounded_argument(argument: str, minimum: int, maximum: int,
                           *, null_if_negative: bool = False) -> str:
    leading_length = len(argument) - len(argument.lstrip())
    trailing_length = len(argument) - len(argument.rstrip())
    leading = argument[:leading_length]
    trailing = argument[len(argument) - trailing_length:] if trailing_length else ""
    expression_end = len(argument) - trailing_length if trailing_length else len(argument)
    expression = argument[leading_length:expression_end]
    if _normalized_db_type() == "sqlserver" and "\n" in trailing:
        expression += trailing
        trailing = ""
    null_condition = f"({expression}) IS NULL"
    if null_if_negative:
        null_condition += f" OR ({expression}) < 0"
    return (
        f"{leading}CASE WHEN {null_condition} THEN NULL "
        f"ELSE LEAST(GREATEST(({expression}), {minimum}), {maximum}) END{trailing}"
    )


def _bound_sqlserver_replicate(arguments: List[str], count_max: int,
                               output_max_chars: int) -> str:
    count_max = max(0, int(count_max))
    budget = max(0, int(output_max_chars))

    input_length = (
        "LEN(CONVERT(nvarchar(max), stage3_replicate_bound.repeat_value) + N'#') - 1"
    )
    count_budget = (
        f"LEAST({count_max}, {budget} / "
        f"GREATEST(COALESCE({input_length}, 0), 1))"
    )

    bounded_count = (
        "CASE WHEN stage3_replicate_bound.raw_count IS NULL "
        "OR stage3_replicate_bound.raw_count < 0 THEN NULL "
        f"ELSE LEAST(stage3_replicate_bound.raw_count, {count_budget}) END"
    )

    return (
        f"(SELECT REPLICATE(stage3_replicate_bound.repeat_value, {bounded_count}) "
        f"FROM (VALUES (LEFT(({arguments[0]}), {budget}), ({arguments[1]}))) "
        "AS stage3_replicate_bound(repeat_value, raw_count))"
    )


def bound_resource_arguments(
    sql: str,
    *,
    enabled: bool,
    repeat_count_max: int,
    repeat_output_max_chars: int,
    string_length_max: int,
) -> tuple:
    is_sqlserver = _normalized_db_type() == "sqlserver"
    resource_pattern = SQLSERVER_RESOURCE_FUNCTION_PATTERN if is_sqlserver else (
        ORACLE_RESOURCE_FUNCTION_PATTERN
        if _is_oracle_db()
        else RESOURCE_FUNCTION_PATTERN
    )
    if not enabled or not sql or not resource_pattern.search(sql):
        return sql, {}

    limits = _resource_function_limits(string_length_max)
    counts: Dict[str, int] = {}

    def rewrite(segment: str) -> str:
        output = []
        index = 0
        while index < len(segment):
            token_end = _consume_sql_literal_or_comment(segment, index)
            if token_end is not None:
                output.append(segment[index:token_end])
                index = token_end
                continue

            char = segment[index]
            if char.isalpha() or char == "_":
                name_end = index + 1
                while name_end < len(segment) and (
                    segment[name_end].isalnum() or segment[name_end] in ("_", "$")
                ):
                    name_end += 1
                open_index = name_end
                while open_index < len(segment) and segment[open_index].isspace():
                    open_index += 1

                if open_index < len(segment) and segment[open_index] == "(":
                    close_index = _find_matching_parenthesis(segment, open_index)
                    if close_index >= 0:
                        function_name = segment[index:name_end]
                        arguments = _split_sql_arguments(
                            segment[open_index + 1:close_index]
                        )
                        rewritten_arguments = [rewrite(argument) for argument in arguments]
                        limit = limits.get(function_name.upper())

                        is_builtin = not (
                            is_sqlserver and segment[:index].rstrip().endswith((".", "@", "#", "$"))
                        )
                        if not is_builtin:
                            limit = None
                        if (is_sqlserver and is_builtin
                                and function_name.upper() == "REPLICATE"
                                and len(rewritten_arguments) == 2):
                            output.append(_bound_sqlserver_replicate(
                                rewritten_arguments, repeat_count_max, repeat_output_max_chars,
                            ))
                            counts["REPLICATE"] = counts.get("REPLICATE", 0) + 1
                            index = close_index + 1
                            continue
                        if (
                            not _is_oracle_db()
                            and not is_sqlserver
                            and function_name.upper() == "REPEAT"
                            and len(rewritten_arguments) >= 2
                        ):
                            input_expression = rewritten_arguments[0].strip()
                            count_expression = rewritten_arguments[1].strip()
                            output_budget = max(0, int(repeat_output_max_chars))
                            repeat_count_limit = max(0, int(repeat_count_max))
                            repeat_input_budget = (
                                f"LEAST({repeat_count_limit}, GREATEST(1, "
                                f"{output_budget} / GREATEST(COALESCE("
                                "CHAR_LENGTH(stage3_repeat_bound.repeat_value), 1), 1)))"
                            )
                            repeat_count_bound = (
                                "CASE WHEN stage3_repeat_bound.raw_count IS NULL THEN NULL "
                                "ELSE LEAST(GREATEST(stage3_repeat_bound.raw_count, 0), "
                                f"{repeat_input_budget}) END"
                            )
                            if str(getattr(DB_CONFIG, "db_type", "mysql")).lower() in {
                                "postgresql", "postgres", "pg"
                            }:
                                output.append(
                                    "(SELECT REPEAT("
                                    f"stage3_repeat_bound.repeat_value, {repeat_count_bound}) "
                                    "FROM LATERAL (VALUES (("
                                    f"{input_expression}), ({count_expression}))) "
                                    "AS stage3_repeat_bound(repeat_value, raw_count))"
                                )
                            else:
                                output.append(
                                    "LEFT(REPEAT("
                                    f"{input_expression}, LEAST(GREATEST(("
                                    f"{count_expression}), 0), {repeat_count_limit})), "
                                    f"{output_budget})"
                                )
                            counts["REPEAT"] = counts.get("REPEAT", 0) + 1
                            index = close_index + 1
                            continue
                        if limit is not None:
                            argument_index, minimum, maximum = limit
                            if argument_index < len(rewritten_arguments):
                                rewritten_arguments[argument_index] = _wrap_bounded_argument(
                                    rewritten_arguments[argument_index], minimum, maximum,
                                    null_if_negative=is_sqlserver and function_name.upper() == "SPACE",
                                )
                                counts[function_name.upper()] = (
                                    counts.get(function_name.upper(), 0) + 1
                                )

                        output.append(segment[index:open_index + 1])
                        output.append(",".join(rewritten_arguments))
                        output.append(")")
                        index = close_index + 1
                        continue

            output.append(char)
            index += 1
        return "".join(output)

    return rewrite(sql), counts


def bound_mysql_exp_overflow(
    sql: str,
    *,
    db_type: str,
    error_code: Any,
    error: Optional[str],
    argument_limit: float,
) -> tuple:
    if (
        str(db_type or "").lower() != "mysql"
        or str(error_code or "") != "1690"
        or "out of range" not in str(error or "").lower()
        or not re.search(r"\bEXP\s*\(", str(sql or ""), flags=re.IGNORECASE)
    ):
        return sql, 0

    limit = max(1.0, float(argument_limit))
    rewritten_count = 0

    def rewrite(segment: str) -> str:
        nonlocal rewritten_count
        output = []
        index = 0
        while index < len(segment):
            token_end = _consume_sql_literal_or_comment(segment, index)
            if token_end is not None:
                output.append(segment[index:token_end])
                index = token_end
                continue
            char = segment[index]
            if char.isalpha() or char == "_":
                name_end = index + 1
                while name_end < len(segment) and (
                    segment[name_end].isalnum() or segment[name_end] in ("_", "$")
                ):
                    name_end += 1
                open_index = name_end
                while open_index < len(segment) and segment[open_index].isspace():
                    open_index += 1
                if open_index < len(segment) and segment[open_index] == "(":
                    close_index = _find_matching_parenthesis(segment, open_index)
                    if close_index >= 0:
                        function_name = segment[index:name_end]
                        arguments = _split_sql_arguments(segment[open_index + 1:close_index])
                        rewritten_arguments = [rewrite(argument) for argument in arguments]
                        if function_name.upper() == "EXP" and rewritten_arguments:
                            rewritten_arguments[0] = _wrap_bounded_argument(
                                rewritten_arguments[0], -limit, limit
                            )
                            rewritten_count += 1
                        output.append(segment[index:open_index + 1])
                        output.append(",".join(rewritten_arguments))
                        output.append(")")
                        index = close_index + 1
                        continue
            output.append(char)
            index += 1
        return "".join(output)

    rewritten = rewrite(str(sql or ""))
    return rewritten, rewritten_count

class TLPBugFinder:
    def __init__(
        self,
        deadline=None,
        input_files=None,
        predicate_source_files=None,
        signature_spans_by_file=None,
        progress_callback=None,
        stratified_task_source=None,
    ):
        self.deadline = deadline

        self.stratified_task_source = stratified_task_source

        self.input_files = list(input_files) if input_files is not None else None

        self.predicate_source_files = {
            os.path.basename(str(tlp_path)): str(source_path)
            for source_path, tlp_path in (predicate_source_files or {}).items()
            if str(source_path).strip() and str(tlp_path).strip()
        }
        self.progress_callback = progress_callback
        self._progress_callback_warning_emitted = False
        self.signature_feedback: Dict[str, Dict[str, int]] = {}
        self.signature_feedback_covered_cases = 0
        self.signature_span_lookup = self._build_signature_span_lookup(
            signature_spans_by_file or {}
        )
        self.stopped_by_deadline = False
        self.worker_failure = None
        self.oracle_excluded_cases = 0
        self.oracle_excluded_wall_seconds = 0.0
        self.oracle_worker_restarts = 0
        self.oracle_fault_counts = {}
        self.oracle_excluded_bits = {}
        self.oracle_excluded_by_file = {}
        self._last_oracle_report_flush = 0.0

        self.tlp_mode = "where"
        self.error_sample_limit = int(getattr(STAGE3_CONFIG, "tlp_error_sample_limit", 100))
        self.error_sql_max_chars = int(getattr(STAGE3_CONFIG, "tlp_error_sql_max_chars", 600))
        self.error_message_max_chars = int(getattr(STAGE3_CONFIG, "tlp_error_message_max_chars", 300))
        self.bug_sample_limit = max(
            0,
            int(getattr(STAGE3_CONFIG, "tlp_bug_sample_limit", 100)),
        )
        self.bug_sql_max_chars = max(
            1,
            int(getattr(STAGE3_CONFIG, "tlp_bug_sql_max_chars", 2000)),
        )
        self.output_buffer_bytes = max(
            4096,
            int(getattr(STAGE3_CONFIG, "stage3_output_buffer_bytes", 1048576)),
        )
        self.report_retention_count = int(
            getattr(STAGE3_CONFIG, "stage3_report_retention_count", 20)
        )
        self.progress_log_interval = int(getattr(STAGE3_CONFIG, "tlp_progress_log_interval", 1000))
        self.query_timeout = int(getattr(STAGE3_CONFIG, "tlp_query_timeout", 10))
        self.pool_chunksize = max(1, int(getattr(STAGE3_CONFIG, "tlp_pool_chunksize", 10)))
        self.resource_argument_bounding_enabled = bool(
            getattr(STAGE3_CONFIG, "tlp_resource_argument_bounding_enabled", True)
        )
        self.repeat_count_max = max(
            0, int(getattr(STAGE3_CONFIG, "tlp_repeat_count_max", 1024))
        )
        self.repeat_output_max_chars = max(
            0, int(getattr(STAGE3_CONFIG, "tlp_repeat_output_max_chars", 4096))
        )
        self.string_length_max = max(
            0, int(getattr(STAGE3_CONFIG, "tlp_string_length_max", 4096))
        )
        self.resource_bound_sample_limit = max(
            0, int(getattr(STAGE3_CONFIG, "tlp_resource_bound_sample_limit", 5))
        )

        self.executor = create_db_executor(
            autoconnect=True,
            query_timeout_seconds=self.query_timeout,
        )

        self.total_cases_loaded = 0       
        self.total_cases_run = 0          
        self.total_cases_validated = 0    
        self.total_cases_satisfiable = 0  
        self.total_cases_skipped = 0      
        self.total_sql_errors = 0         
        self.total_validation_errors = 0  
        self.total_query_timeouts = 0     
        self.oracle_error_category_counts: Dict[str, int] = {
            "type_failure_count": 0,
            "value_failure_count": 0,
            "other_failure_count": 0,
            "timeout_count": 0,
        }
        self.total_bugs_found = 0         
        self.resource_bounded_sqls = 0    
        self.resource_bounded_arguments = 0
        self.resource_bound_function_counts: Dict[str, int] = {}

        self.error_summary: Dict[str, Dict[str, int]] = {
            "by_kind": {},
            "by_error_code": {},
            "by_role": {},
            "by_category": {},
        }
        self.overflow_retry_attempts = 0
        self.overflow_retry_successes = 0
        self.overflow_retry_rewritten_arguments = 0
        self.oracle_value_repair = {"attempts": 0, "successes": 0, "rewritten_arguments": 0}
        self.resource_bound_samples_written = 0
        self.error_samples_written = 0    
        self.bug_samples_written = 0      
        self.bug_samples_dropped = 0      
        self.start_time = 0               

        self.mode_stats = {}
        self.file_modes = {}

        _prune_stage3_report_files(self.report_retention_count)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.log_file = REPORT_DIR / f"scan_log_{timestamp}.txt"
        self.bug_file = REPORT_DIR / f"bugs_found_{timestamp}.txt"
        self.error_sample_file = REPORT_DIR / f"tlp_sql_errors_{timestamp}.jsonl"
        self.error_summary_path = REPORT_DIR / f"tlp_error_summary_{timestamp}.json"
        self.cardinality_results_path = REPORT_DIR / f"predicate_cardinalities_{timestamp}.jsonl.gz"
        self.oracle_fault_path = REPORT_DIR / f"oracle_worker_faults_{time.time_ns()}.jsonl"
        self._oracle_fault_handle = None
        self._log_handle = None
        self._bug_handle = None
        self._error_sample_handle = None
        self._cardinality_handle = None
        self.cardinality_predicates_written = 0

        self.validation_status_bits: Dict[str, bytearray] = {}
        self.validation_case_counts: Dict[str, int] = {}
        self.validation_counts_by_file: Dict[str, int] = {}
        self.validation_count = 0
        self.validation_status_path = (
            PROJECT_ROOT
            / "pkg"
            / "generated_sqls"
            / f"stage3_validation_status_{time.time_ns()}_{os.getpid()}.json.gz"
        )

    def _record_validated_case(self, counts):
        self.total_cases_validated += 1
        self.total_cases_satisfiable += int(
            any(int(counts.get(role, 0)) >= 1 for role in ("true", "false", "null"))
        )

    def _record_cardinality_result(self, file_name, case_idx, tlp_mode, counts, sqls):
        roles = ("true", "false", "null")
        if not all(role in counts and role in sqls for role in roles):
            return

        payload = {
            "db_type": str(getattr(DB_CONFIG, "db_type", "unknown")),
            "file": os.path.basename(str(file_name or "")),
            "case_idx": int(case_idx),
            "tlp_mode": str(tlp_mode or self.tlp_mode),
            "variants": [
                {
                    "role": role,
                    "cardinality": int(counts[role]),
                    "sql": str(sqls[role]),
                }
                for role in roles
            ],
        }
        if self._cardinality_handle is None:
            self.cardinality_results_path.parent.mkdir(parents=True, exist_ok=True)
            self._cardinality_handle = gzip.open(
                self.cardinality_results_path,
                "at",
                encoding="utf-8",
                compresslevel=1,
            )
        self._cardinality_handle.write(
            json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n"
        )
        self.cardinality_predicates_written += 1
        if self.cardinality_predicates_written % 128 == 0:
            self._cardinality_handle.flush()

    def _increment_error_summary(self, error_info: Dict[str, Any], *, error_kind: str) -> None:
        kind = str(error_kind or "unknown")
        code = (
            _oracle_error_label(error_info)
            if _is_oracle_db()
            else str(error_info.get("error_code") or "unknown")
        )
        role = str(error_info.get("role") or "unknown")
        if _normalized_db_type() == "sqlserver" and code == "unknown":
            code = str(error_info.get("sqlstate") or "unknown")
        if _is_oracle_db():
            category = _oracle_error_category(error_info)
        elif _normalized_db_type() == "sqlserver":
            category = _sqlserver_error_category(error_info)
        else:
            category = self._validation_failure_kind(error_info)
        for bucket, value in (
            ("by_kind", kind),
            ("by_error_code", code),
            ("by_role", role),
            ("by_category", category),
        ):
            target = self.error_summary[bucket]
            target[value] = int(target.get(value, 0)) + 1

    def _record_overflow_retry(self, payload: Optional[Dict[str, Any]]) -> None:
        item = payload or {}
        self.overflow_retry_attempts += int(item.get("attempts", 0) or 0)
        self.overflow_retry_successes += int(item.get("successes", 0) or 0)
        self.overflow_retry_rewritten_arguments += int(
            item.get("rewritten_arguments", 0) or 0
        )

    def _record_validation_status(self, status, payload) -> None:
        if status == "bug" and isinstance(payload, tuple) and len(payload) >= 2:
            file_name, case_idx = payload[0], payload[1]
            validated = True
        elif isinstance(payload, dict):
            file_name = payload.get("file")
            case_idx = payload.get("case_idx")
            validated = status == "ok"
        else:
            return

        file_name = os.path.basename(str(file_name or "")).strip()
        try:
            case_idx = int(case_idx)
        except (TypeError, ValueError):
            return
        if not file_name or case_idx <= 0:
            return

        source = getattr(self, "stratified_task_source", None)
        if source is not None:
            source.observe_result(file_name, validated)

        self.validation_case_counts[file_name] = max(
            case_idx,
            int(self.validation_case_counts.get(file_name, 0)),
        )
        bits = self.validation_status_bits.setdefault(file_name, bytearray())
        zero_based = case_idx - 1
        byte_index = zero_based >> 3
        if byte_index >= len(bits):
            bits.extend(b"\x00" * (byte_index + 1 - len(bits)))
        mask = 1 << (zero_based & 7)
        if validated and not (bits[byte_index] & mask):
            bits[byte_index] |= mask
            self.validation_count += 1
            self.validation_counts_by_file[file_name] = (
                int(self.validation_counts_by_file.get(file_name, 0)) + 1
            )

    def _record_oracle_fault(self, status, details):
        item = dict(details)
        if status == "excluded":
            name, idx = os.path.basename(item["file"]), int(item["case_idx"])
            bits = self.oracle_excluded_bits.setdefault(name, bytearray())
            byte, bit = (idx - 1) >> 3, 1 << ((idx - 1) & 7)
            if byte >= len(bits):
                bits.extend(b"\x00" * (byte + 1 - len(bits)))
            if bits[byte] & bit:
                return
            bits[byte] |= bit
            self.oracle_excluded_cases += 1
            self.oracle_excluded_by_file[name] = self.oracle_excluded_by_file.get(name, 0) + 1
            mode = str(item.get("tlp_mode", self.tlp_mode)).lower()
            stats = self.mode_stats.get(mode)
            if stats is not None:
                stats["loaded"] = max(0, stats.get("loaded", 0) - 1)
                stats["oracle_excluded_cases"] = stats.get("oracle_excluded_cases", 0) + 1
            source = self.stratified_task_source
            if source is not None:
                source.observe_excluded(name)
        kind = item["kind"]
        self.oracle_fault_counts[kind] = self.oracle_fault_counts.get(kind, 0) + 1

        if sum(self.oracle_fault_counts.values()) > self.error_sample_limit:
            item.pop("sql", None)
        item.pop("pending_case_indices", None)
        item["event"] = status
        if self._oracle_fault_handle is None:
            self._oracle_fault_handle = open(self.oracle_fault_path, "a", encoding="utf-8",
                                             buffering=self.output_buffer_bytes)
        self._oracle_fault_handle.write(json.dumps(item, ensure_ascii=False) + "\n")
        self._oracle_fault_handle.flush()
        self.log(f"[Oracle Recovery] {status}: file={item.get('file')}, "
                 f"case={item.get('case_idx')}, kind={kind}; continuing.")

    def _oracle_exclusion_summary(self):
        return {
            "oracle_excluded_cases": self.oracle_excluded_cases,
            "oracle_excluded_wall_seconds": self.oracle_excluded_wall_seconds,
            "oracle_worker_restarts": self.oracle_worker_restarts,
            "oracle_fault_counts": dict(self.oracle_fault_counts),
            "oracle_fault_path": str(self.oracle_fault_path) if self.oracle_fault_counts else None,
            "cases_loaded_raw": self.total_cases_loaded,
            "cases_loaded": max(0, self.total_cases_loaded - self.oracle_excluded_cases),
        }

    def _write_validation_status(self) -> Optional[str]:
        entries = []
        tlp_names = {
            os.path.basename(str(path))
            for path in (self.input_files or [])
            if str(path).strip()
        }
        tlp_names.update(self.validation_case_counts)
        for tlp_file in sorted(tlp_names):
            bits = self.validation_status_bits.get(tlp_file, bytearray())
            case_count = int(self.validation_case_counts.get(tlp_file, 0))

            required_bytes = (max(0, case_count) + 7) // 8
            if len(bits) < required_bytes:
                bits.extend(b"\x00" * (required_bytes - len(bits)))
            entries.append({
                "tlp_file": tlp_file,
                "predicate_file": self.predicate_source_files.get(tlp_file),
                "case_count": case_count,
                "validated_count": int(self.validation_counts_by_file.get(tlp_file, 0)),
                "encoding": "bitset_lsb_first",
                "validated_bitmap_base64": base64.b64encode(bytes(bits)).decode("ascii"),
            })
            if _is_oracle_db():
                excluded = self.oracle_excluded_bits.get(tlp_file, bytearray())
                entries[-1].update(
                    oracle_excluded_count=self.oracle_excluded_by_file.get(tlp_file, 0),
                    oracle_excluded_bitmap_base64=base64.b64encode(
                        bytes(excluded).ljust(required_bytes, b"\x00")).decode("ascii"),
                )
        payload = {
            "version": 1,
            "definition": "all_four_tlp_validation_selects_succeeded",
            "status": "completed",
            "entries": entries,
            "total_cases": int(sum(item["case_count"] for item in entries)),
            "validated_cases": int(self.validation_count),
        }
        if _is_oracle_db():
            payload["oracle_excluded_cases"] = self.oracle_excluded_cases
        partial_path = Path(str(self.validation_status_path) + ".partial")
        try:
            partial_path.parent.mkdir(parents=True, exist_ok=True)
            with gzip.open(
                partial_path,
                "wt",
                encoding="utf-8",
                compresslevel=1,
            ) as handle:
                json.dump(payload, handle, ensure_ascii=False, separators=(",", ":"))
            os.replace(partial_path, self.validation_status_path)
            return str(self.validation_status_path)
        except Exception as exc:
            try:
                partial_path.unlink(missing_ok=True)
            except OSError:
                pass
            print(f"[TLP] Warning: failed to write Validation status index: {exc}")
            return None

    @staticmethod
    def _build_signature_span_lookup(signature_spans_by_file):
        lookup = {}
        for file_name, raw_spans in (signature_spans_by_file or {}).items():
            primary_spans = []
            component_spans = []
            for raw in raw_spans or []:
                if not isinstance(raw, dict):
                    continue
                try:
                    start = int(raw.get("start_case_idx", 0) or 0)
                    end = int(raw.get("end_case_idx", 0) or 0)
                except (TypeError, ValueError):
                    continue
                key = str(raw.get("signature_key", "") or "").strip()
                if start <= 0 or end < start:
                    continue
                raw_keys = raw.get("signature_keys")
                if isinstance(raw_keys, list):
                    keys = [str(item or "").strip() for item in raw_keys if str(item or "").strip()]
                else:
                    keys = [key]
                if not keys:
                    continue
                item = (start, end, tuple(dict.fromkeys(keys)))
                if raw_keys is not None and not key:
                    component_spans.append(item)
                else:
                    primary_spans.append(item)
            primary_spans.sort(key=lambda item: (item[1], item[0], item[2]))
            component_spans.sort(key=lambda item: (item[0], item[1], item[2]))
            lookup[os.path.basename(str(file_name))] = {
                "primary_ends": [item[1] for item in primary_spans],
                "primary_spans": primary_spans,
                "component_starts": [item[0] for item in component_spans],
                "component_spans": component_spans,
            }
        return lookup

    def _signature_key_for_case(self, file_name, case_idx):
        keys = self._signature_keys_for_case(file_name, case_idx)
        return keys[0] if keys else None

    def _signature_keys_for_case(self, file_name, case_idx):
        item = self.signature_span_lookup.get(os.path.basename(str(file_name or "")))
        if not item:
            return []
        try:
            index_value = int(case_idx)
        except (TypeError, ValueError):
            return []
        keys = []
        primary_position = bisect_left(item["primary_ends"], index_value)
        if primary_position < len(item["primary_spans"]):
            start, end, primary_keys = item["primary_spans"][primary_position]
            if start <= index_value <= end:
                keys.extend(primary_keys)

        component_position = bisect_right(
            item["component_starts"],
            index_value,
        ) - 1
        if component_position >= 0:
            start, end, component_keys = item["component_spans"][component_position]
            if start <= index_value <= end:
                keys.extend(component_keys)
        return list(dict.fromkeys(keys))

    @staticmethod
    def _validation_failure_kind(error_info):
        if _normalized_db_type() == "sqlserver":
            return _sqlserver_error_category(error_info)
        if _is_oracle_db():
            return _oracle_error_category(error_info)
        if bool((error_info or {}).get("is_timeout", False)):
            return "timeout_count"
        code = str((error_info or {}).get("error_code", "") or "").strip()
        text = str((error_info or {}).get("error", "") or "").lower()
        db_type = str(getattr(DB_CONFIG, "db_type", "") or "").lower()
        if db_type in {"postgresql", "postgres", "pg"}:
            if code in {"42804", "42809", "42846", "42883"} or any(
                marker in text
                for marker in (
                    "operator does not exist",
                    "function does not exist",
                    "no function matches the given name and argument types",
                    "cannot cast type",
                    "datatype mismatch",
                )
            ):
                return "type_failure_count"
            if code.startswith("22"):
                return "value_failure_count"
        elif db_type == "mysql":
            if code in {"1267", "3854", "3995", "4078", "4079"} or any(
                marker in text
                for marker in (
                    "illegal parameter data types",
                    "illegal mix of collations",
                    "illegal operand types",
                    "cannot convert string",
                    "cannot be used in conjunction with",
                )
            ):
                return "type_failure_count"

            if code in {"1264", "1265", "1292", "1366", "1411", "1525", "1690", "3141", "3685", "3688", "3692"} or any(
                marker in text
                for marker in (
                    "illegal argument to a regular expression",
                    "syntax error in regular expression",
                    "incorrect description of a {min,max}",
                    "incorrect datetime value",
                    "incorrect timestamp value",
                    "incorrect date value",
                    "out of range",
                )
            ):
                return "value_failure_count"
        return "other_failure_count"

    def _observe_signature_result(
        self,
        file_name,
        case_idx,
        *,
        success,
        error_info=None,
    ):
        if (_normalized_db_type() == "sqlserver" and not success
                and str((error_info or {}).get("role", "")) == "total"):
            return
        keys = self._signature_keys_for_case(file_name, case_idx)
        if not keys:
            return
        self.signature_feedback_covered_cases += 1
        for key in keys:
            item = self.signature_feedback.setdefault(
                key,
                {
                    "success_count": 0,
                    "failure_count": 0,
                    "type_failure_count": 0,
                    "value_failure_count": 0,
                    "other_failure_count": 0,
                    "timeout_count": 0,
                },
            )
            if success:
                item["success_count"] += 1
                continue
            item["failure_count"] += 1
            kind = self._validation_failure_kind(error_info)
            item[kind] = int(item.get(kind, 0)) + 1

    def _emit_progress_callback(self, *, force=False):
        if self.progress_callback is None:
            return
        try:
            self.progress_callback({
                "cases_loaded": self.total_cases_loaded,
                "cases_run": self.total_cases_run,
                "cases_validated": self.total_cases_validated,
                "cases_skipped": self.total_cases_skipped,
                "sql_errors": self.total_sql_errors,
                "query_timeouts": self.total_query_timeouts,
                "force": bool(force),
                **(self._oracle_exclusion_summary() if _is_oracle_db() else {}),
            })
        except Exception as exc:
            if not self._progress_callback_warning_emitted:
                print(f"[Stage3 Metrics] progress callback disabled: {exc}")
                self._progress_callback_warning_emitted = True
            self.progress_callback = None

    def log(self, message: str, to_console: bool = True):
        if to_console:
            print(message)

        if self._log_handle is None:
            self._log_handle = open(
                self.log_file,
                'a',
                encoding='utf-8',
                buffering=self.output_buffer_bytes,
            )
        self._log_handle.write(message + "\n")
        if to_console:
            self._log_handle.flush()

    def _close_report_handles(self):
        for attr_name in (
            "_log_handle",
            "_bug_handle",
            "_error_sample_handle",
            "_oracle_fault_handle",
            "_cardinality_handle",
        ):
            handle = getattr(self, attr_name, None)
            if handle is not None:
                handle.close()
                setattr(self, attr_name, None)

    def _flush_oracle_reports(self, force=False):
        if not _is_oracle_db():
            return
        now = time.monotonic()
        if not force and now - self._last_oracle_report_flush < 5.0:
            return
        for attr in (
            "_log_handle",
            "_bug_handle",
            "_error_sample_handle",
            "_oracle_fault_handle",
            "_cardinality_handle",
        ):
            handle = getattr(self, attr, None)
            if handle is not None:
                handle.flush()
        self._last_oracle_report_flush = now

    def _truncate_text(self, text: Optional[str], limit: int) -> str:
        if text is None:
            return ""
        text = str(text)
        if len(text) <= limit:
            return text
        return text[:limit] + f"...<truncated {len(text) - limit} chars>"

    def _ensure_mode_stats(self, tlp_mode: str) -> Dict[str, int]:
        mode = str(tlp_mode or "unknown").lower()
        if mode not in self.mode_stats:
            self.mode_stats[mode] = {
                "loaded": 0,
                "processed": 0,
                "validated": 0,
                "skipped": 0,
                "sql_errors": 0,
                "query_timeouts": 0,
                "bugs": 0,
            }
        return self.mode_stats[mode]

    def _inc_mode_stat(self, tlp_mode: str, key: str, amount: int = 1):
        stats = self._ensure_mode_stats(tlp_mode)
        stats[key] = stats.get(key, 0) + amount

    def _record_sql_error(self, error_info: Dict[str, Any], *, error_kind: str):
        tlp_mode = str(error_info.get("tlp_mode") or self.tlp_mode or "unknown").lower()

        self.total_sql_errors += 1
        self._inc_mode_stat(tlp_mode, "sql_errors")
        self._increment_error_summary(error_info, error_kind=error_kind)

        category = (
            _oracle_error_category(error_info)
            if _is_oracle_db()
            else (
                self._validation_failure_kind(error_info)
            )
        )
        if _normalized_db_type() == "sqlserver":
            category = _sqlserver_error_category(error_info)
        is_timeout = category == "timeout_count"
        if is_timeout:
            self.total_query_timeouts += 1
            self._inc_mode_stat(tlp_mode, "query_timeouts")
        if _is_oracle_db():
            self.oracle_error_category_counts[category] = (
                int(self.oracle_error_category_counts.get(category, 0)) + 1
            )

        self.total_validation_errors += 1

        if self.error_samples_written >= self.error_sample_limit:
            return

        payload = {
            "kind": error_kind,
            "file": error_info.get("file"),
            "case_idx": error_info.get("case_idx"),
            "tlp_mode": tlp_mode,
            "role": error_info.get("role"),
            "error_code": (
                _oracle_error_label(error_info)
                if _is_oracle_db()
                else error_info.get("error_code")
            ),
            "error": self._truncate_text(error_info.get("error"), self.error_message_max_chars),
            "sql": self._truncate_text(error_info.get("sql"), self.error_sql_max_chars),
            "execution_time": error_info.get("execution_time"),
            "is_timeout": is_timeout,
        }
        if _normalized_db_type() == "sqlserver":
            payload.update(sqlstate=error_info.get("sqlstate"), cancelled=bool(error_info.get("cancelled", False)))
        if error_kind == "validation" or _is_oracle_db() or _normalized_db_type() == "sqlserver":
            payload["category"] = category

        if self._error_sample_handle is None:
            self._error_sample_handle = open(
                self.error_sample_file,
                "a",
                encoding="utf-8",
                buffering=self.output_buffer_bytes,
            )
        self._error_sample_handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

        self.error_samples_written += 1

    def _write_error_summary(self) -> Optional[str]:
        payload = {
            "version": 1,
            "complete_counts": True,
            "sql_text_is_sampled_separately": True,
            "total_sql_errors": int(self.total_sql_errors),
            "validation_errors": int(self.total_validation_errors),
            "query_timeouts": int(self.total_query_timeouts),
            "counts": {
                key: dict(sorted(values.items()))
                for key, values in self.error_summary.items()
            },
            "mysql_numeric_overflow_retry": {
                "attempts": int(self.overflow_retry_attempts),
                "successes": int(self.overflow_retry_successes),
                "failures": max(
                    0,
                    int(self.overflow_retry_attempts)
                    - int(self.overflow_retry_successes),
                ),
                "rewritten_arguments": int(
                    self.overflow_retry_rewritten_arguments
                ),
            },
        }
        if _is_oracle_db():
            payload["oracle_value_repair"] = dict(self.oracle_value_repair)
            payload["round_complete"] = not self.stopped_by_deadline and self.worker_failure is None
            payload["worker_failure"] = self.worker_failure
            payload["cases_loaded"] = self.total_cases_loaded
            payload["cases_run"] = self.total_cases_run
            payload.update(self._oracle_exclusion_summary())
            payload["oracle_error_categories"] = dict(
                self.oracle_error_category_counts
            )
        partial_path = Path(str(self.error_summary_path) + ".partial")
        try:
            with open(partial_path, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False, indent=2)
                handle.write("\n")
            os.replace(partial_path, self.error_summary_path)
            return str(self.error_summary_path)
        except Exception as exc:
            try:
                partial_path.unlink(missing_ok=True)
            except OSError:
                pass
            print(f"[TLP] Warning: failed to write SQL error summary: {exc}")
            return None

    def report_bug(self, file_name: str, case_idx: int, sql_group: List[str], counts: Dict[str, int]):
        self.total_bugs_found += 1

        special_node_notice = (
            "[TLP] Inconsistency detected; possibly caused by special nodes!"
            if _has_oracle_tlp_special_nodes(sql_group) else None
        )

        if self.bug_samples_written >= self.bug_sample_limit:
            self.bug_samples_dropped += 1
            return

        check_sum = counts['true'] + counts['false'] + counts['null']
        diff = counts['total'] - check_sum

        display_group = [
            self._truncate_text(sql, self.bug_sql_max_chars)
            for sql in (sql_group or [])
        ]
        while len(display_group) < 5:
            display_group.append("")

        report_content = [
            "==================================================",
            (
                f"{special_node_notice} File: {file_name}; case: {case_idx}"
                if special_node_notice else
                f"[BUG FOUND] 在文件: {file_name} (第 {case_idx} 组测试)"
            ),
            "--------------------------------------------------",
            f"逻辑公式校验失败: Total != True + False + Null",
            f"期望值 (Total) : {counts['total']}",
            f"实际值 (Sum)   : {check_sum} (T:{counts['true']} + F:{counts['false']} + N:{counts['null']})",
            f"差值 (Diff)    : {diff}",
            "--------------------------------------------------",
            "[检测 SQL / 相关 SQL]:",
            f"1. Check Total: {display_group[0]}",
            f"2. Check True : {display_group[1]}",
            f"3. Check False: {display_group[2]}",
            f"4. Check Null : {display_group[3]}",
            "==================================================\n"
        ]

        if self._bug_handle is None:
            self._bug_handle = open(
                self.bug_file,
                'a',
                encoding='utf-8',
                buffering=self.output_buffer_bytes,
            )
        self._bug_handle.write("\n".join(report_content))
        if _is_oracle_db():
            self._bug_handle.flush()
        self.bug_samples_written += 1

    def iter_sql_file(self, file_path: str) -> Iterator[List[str]]:
        group = []
        valid_sql_count = 0
        file_name = os.path.basename(file_path)
        detected_mode = self.tlp_mode

        with _open_test_case_reader(file_path) as f:
            for line in f:
                line = line.strip()

                if line.lower().startswith("-- tlp mode:"):
                    if line.split(":", 1)[1].strip().lower() != "where":
                        raise ValueError("Expected WHERE-TLP test cases")

                if not line or line.startswith("--"):
                    continue

                if not line.upper().startswith("SELECT "):
                    raise ValueError("Expected four SELECT statements per WHERE-TLP case")
                valid_sql_count += 1
                group.append(line)
                if len(group) == 4:
                    yield list(group)
                    group.clear()

        self.file_modes[file_name] = detected_mode

        if valid_sql_count % 4 != 0:
            self.log(
                f'[Warning] The number of SELECT statements in {os.path.basename(file_path)} is not a multiple of 4 ({valid_sql_count}); the incomplete trailing group was ignored.'
            )

    def parse_sql_file(self, file_path: str) -> List[List[str]]:
        return list(self.iter_sql_file(file_path))

    def _cleanup_sql(self, sql: str) -> str:
        if not sql:
            return sql

        if _normalized_db_type() == "sqlserver":
            from BugFinding.sql_generator import cleanup_dsl
            return cleanup_dsl(sql)

        curr_sql = sql
        for _ in range(5):
            new_sql = WRAPPER_PATTERN.sub(lambda m: m.group(2).strip(), curr_sql)
            if new_sql == curr_sql:
                break
            curr_sql = new_sql

        return curr_sql

    def _bound_sql_for_execution(self, sql: str) -> str:
        bounded_sql, function_counts = bound_resource_arguments(
            sql,
            enabled=self.resource_argument_bounding_enabled,
            repeat_count_max=self.repeat_count_max,
            repeat_output_max_chars=self.repeat_output_max_chars,
            string_length_max=self.string_length_max,
        )
        if not function_counts:
            return bounded_sql

        self.resource_bounded_sqls += 1
        bounded_argument_count = sum(function_counts.values())
        self.resource_bounded_arguments += bounded_argument_count
        for function_name, count in function_counts.items():
            self.resource_bound_function_counts[function_name] = (
                self.resource_bound_function_counts.get(function_name, 0) + count
            )

        if self.resource_bound_samples_written < self.resource_bound_sample_limit:
            functions = ",".join(
                f"{name}:{count}" for name, count in sorted(function_counts.items())
            )
            self.log(
                "[Resource Bound] tag=bounded_resource_argument "
                f"functions={functions} "
                f"original={self._truncate_text(sql, self.error_sql_max_chars)} "
                f"executed={self._truncate_text(bounded_sql, self.error_sql_max_chars)}",
                to_console=False,
            )

            if self._log_handle is not None:
                self._log_handle.flush()
            self.resource_bound_samples_written += 1

        return bounded_sql

    def _prepare_validation_group(self, group: List[str]) -> List[str]:
        prepared_group = list(group)
        for index in range(min(len(prepared_group), 4)):
            cleaned_sql = self._cleanup_sql(prepared_group[index])
            prepared_group[index] = self._bound_sql_for_execution(cleaned_sql)
        return prepared_group

    def execute_single_sql(
        self,
        sql: str,
        *,
        sql_role: str = "unknown",
        file_name: Optional[str] = None,
        case_idx: Optional[int] = None,
        tlp_mode: Optional[str] = None,
    ) -> Optional[int]:
        sql = self._bound_sql_for_execution(self._cleanup_sql(sql))

        if not sql.strip().upper().startswith("SELECT "):
            raise ValueError("WHERE-TLP validation accepts SELECT statements only")

        res = self.executor.execute_query(sql, fetch="one")

        if (
            not res.success
            and _normalized_db_type() == "mysql"
            and bool(
                getattr(STAGE3_CONFIG, "tlp_mysql_overflow_retry_enabled", True)
            )
        ):
            retry_sql, rewrite_count = bound_mysql_exp_overflow(
                sql,
                db_type=getattr(DB_CONFIG, "db_type", ""),
                error_code=res.error_code,
                error=res.error,
                argument_limit=float(
                    getattr(STAGE3_CONFIG, "tlp_mysql_exp_argument_limit", 700.0)
                ),
            )
            if rewrite_count > 0 and retry_sql != sql:
                retry_res = self.executor.execute_query(
                    retry_sql,
                    fetch="one",
                )
                self._record_overflow_retry({
                    "attempts": 1,
                    "successes": int(bool(retry_res.success)),
                    "rewritten_arguments": int(rewrite_count),
                })
                res = retry_res
                sql = retry_sql

        if not res.success:
            if (_normalized_db_type() == "sqlserver"
                    and getattr(self, "_sqlserver_group_error", None) == {}):
                self._sqlserver_group_error = {
                    "role": sql_role, "error_code": res.error_code, "error": res.error,
                    "is_timeout": bool(res.timed_out), **_sqlserver_diagnostics(res),
                }
            self._record_sql_error(
                {
                    "file": file_name,
                    "case_idx": case_idx,
                    "tlp_mode": tlp_mode or self.tlp_mode,
                    "role": sql_role,
                    "sql": sql,
                    "error": res.error,
                    "error_code": res.error_code,
                    "execution_time": res.execution_time,
                    "is_timeout": bool(res.timed_out),
                    **(_sqlserver_diagnostics(res) if _normalized_db_type() == "sqlserver" else {}),
                },
                error_kind="validation",
            )
            return None

        if res.rows and len(res.rows) > 0:
            row_dict = res.rows[0]

            count_val = list(row_dict.values())[0]
            return int(count_val)
        return 0

    def run_test_group(self, file_name: str, idx: int, group: List[str]):
        if _normalized_db_type() == "sqlserver":
            self._sqlserver_group_error = {}
        select_total = group[0]
        select_true = group[1]
        select_false = group[2]
        select_null = group[3]

        tlp_mode = self.tlp_mode
        self.total_cases_run += 1
        self._inc_mode_stat(tlp_mode, "processed")

        val_total = self.execute_single_sql(select_total, sql_role="total", file_name=file_name, case_idx=idx, tlp_mode=tlp_mode)
        val_true = self.execute_single_sql(select_true, sql_role="true", file_name=file_name, case_idx=idx, tlp_mode=tlp_mode)
        val_false = self.execute_single_sql(select_false, sql_role="false", file_name=file_name, case_idx=idx, tlp_mode=tlp_mode)
        val_null = self.execute_single_sql(select_null, sql_role="null", file_name=file_name, case_idx=idx, tlp_mode=tlp_mode)

        if any(x is None for x in [val_total, val_true, val_false, val_null]):
            self.total_cases_skipped += 1
            self._inc_mode_stat(tlp_mode, "skipped")
            if _normalized_db_type() == "sqlserver":
                self._observe_signature_result(file_name, idx, success=False, error_info=self._sqlserver_group_error)
            return

        check_sum = val_true + val_false + val_null

        if val_total != check_sum:
            counts = {
                'total': val_total,
                'true': val_true,
                'false': val_false,
                'null': val_null
            }
            self._inc_mode_stat(tlp_mode, "bugs")
            self.report_bug(file_name, idx, group, counts)

        self._record_validated_case({"true": val_true, "false": val_false, "null": val_null})
        self._inc_mode_stat(tlp_mode, "validated")
        if _normalized_db_type() == "sqlserver":
            self._observe_signature_result(file_name, idx, success=True)

    def run(self):
        self.start_time = time.time()
        self.log(f'=== Starting TLP Bug Finder ===')
        self.log(f'Batch size: {VALIDATION_BATCH_SIZE}')
        self.log(f'Concurrent workers: {MAX_WORKERS}')
        self.log(f'Per-SQL timeout: {self.query_timeout} seconds')
        self.log(f'Process pool chunksize: {self.pool_chunksize}')
        if _is_oracle_db():
            self.log(
                "Oracle worker guard: spawn; crash detection; "
                f"operation limit={max(30, self.query_timeout * 3)}s; "
                f"case limit={max(180, self.query_timeout * 15)}s; "
                "report flush interval=5s"
            )
            self.log(
                f'Oracle argument bounds: {self.resource_argument_bounding_enabled} (LPAD/RPAD/SUBSTR/SUBSTRB=0..{self.string_length_max})'
            )

        if self.input_files is None:
            candidate_files = glob.glob(str(INPUT_DIR / "*.sql"))
            candidate_files.extend(glob.glob(str(INPUT_DIR / "*.sql.gz")))
        else:
            candidate_files = [
                str(path) for path in self.input_files if _is_test_case_file(path)
            ]
        files = _select_current_test_case_files(candidate_files)

        if not files:
            self.log('[Error] No test-case files found. Run sql_generator.py first.')
            self.executor.close()
            self._close_report_handles()
            return {
                "success": False,
                "round_complete": False,
                "phase": "tlp_scan",
                "error": "No TLP test-case files found.",
            }

        compressed_count = sum(
            1 for path in files if str(path).lower().endswith(".sql.gz")
        )
        self.log(f'Found {len(files)} test files; starting scan...')

        def iter_tasks():
            if self.stratified_task_source is not None:
                for file_name, case_idx, group, file_tlp_mode in self.stratified_task_source:
                    if self.deadline is not None and time.monotonic() >= self.deadline:
                        self.stopped_by_deadline = True
                        return
                    self._inc_mode_stat(file_tlp_mode, "loaded")
                    self.total_cases_loaded += 1
                    self.validation_case_counts[file_name] = max(
                        case_idx, self.validation_case_counts.get(file_name, 0))
                    yield (file_name, case_idx, group, file_tlp_mode)
                if self.deadline is not None and time.monotonic() >= self.deadline:
                    self.stopped_by_deadline = True
                return
            for file_path in sorted(files):
                if self.deadline is not None and time.monotonic() >= self.deadline:
                    self.stopped_by_deadline = True
                    return
                file_name = os.path.basename(file_path)
                for case_idx, group in enumerate(self.iter_sql_file(file_path), start=1):
                    if self.deadline is not None and time.monotonic() >= self.deadline:
                        self.stopped_by_deadline = True
                        return
                    file_tlp_mode = self.file_modes.get(file_name, self.tlp_mode)
                    self._ensure_mode_stats(file_tlp_mode)
                    self._inc_mode_stat(file_tlp_mode, "loaded")
                    self.total_cases_loaded += 1
                    file_name = os.path.basename(file_path)
                    self.validation_case_counts[file_name] = max(
                        case_idx,
                        int(self.validation_case_counts.get(file_name, 0)),
                    )
                    yield (file_name, case_idx, group, file_tlp_mode)

        task_iter = iter_tasks()
        self.log('Streaming test groups and validating concurrently in batches...')

        pool = (
            OracleValidationPool(
                MAX_WORKERS, worker_init, worker_task,
                query_timeout=self.query_timeout, deadline=self.deadline,
                on_poll=self._flush_oracle_reports,
            )
            if _is_oracle_db()
            else multiprocessing.Pool(processes=MAX_WORKERS, initializer=worker_init)
        )
        pool_terminated = False
        try:
            batch_start = 0
            batch_tasks = list(itertools.islice(task_iter, VALIDATION_BATCH_SIZE))
            while batch_tasks:
                if self.deadline is not None and time.monotonic() >= self.deadline:
                    self.stopped_by_deadline = True
                    break

                next_task = next(task_iter, None)
                if self.stopped_by_deadline:
                    break
                has_next_batch = next_task is not None

                worker_batch_tasks = []
                prepared_resource_counts = {}
                for file_name, case_idx, group, tlp_mode in batch_tasks:
                    if _is_oracle_db():
                        before_counts = (self.resource_bounded_sqls, self.resource_bounded_arguments,
                                         dict(self.resource_bound_function_counts))
                    prepared_group = self._prepare_validation_group(group)
                    if _is_oracle_db():
                        prepared_resource_counts[(file_name, case_idx)] = (
                            self.resource_bounded_sqls - before_counts[0],
                            self.resource_bounded_arguments - before_counts[1],
                            {key: value - before_counts[2].get(key, 0)
                             for key, value in self.resource_bound_function_counts.items()},
                        )
                    worker_batch_tasks.append(
                        (
                            file_name,
                            case_idx,
                            prepared_group,
                            tlp_mode,
                        )
                    )

                excluded_in_batch = set()
                for status, payload in pool.imap_unordered(
                    worker_task,
                    worker_batch_tasks,
                    chunksize=self.pool_chunksize,
                ):
                    if _is_oracle_db() and status in ("excluded", "worker_restarted", "worker_unavailable"):
                        self._record_oracle_fault(status, payload)
                        if status == "excluded":
                            identity = (payload["file"], payload["case_idx"])
                            excluded_in_batch.add(identity)
                            sqls, arguments, functions = prepared_resource_counts.pop(identity)
                            self.resource_bounded_sqls -= sqls
                            self.resource_bounded_arguments -= arguments
                            for key, value in functions.items():
                                self.resource_bound_function_counts[key] -= value
                        continue
                    self._record_validation_status(status, payload)
                    if _is_oracle_db():
                        repair_summary = (payload.get("oracle_value_repair", {})
                                          if isinstance(payload, dict)
                                          else payload[6] if status == "bug" and len(payload) > 6 else {})
                        for key in self.oracle_value_repair:
                            self.oracle_value_repair[key] += int(repair_summary.get(key, 0))
                    self.total_cases_run += 1

                    payload_mode = self.tlp_mode
                    if isinstance(payload, dict):
                        payload_mode = payload.get("tlp_mode", self.tlp_mode)
                    elif status == "bug" and payload and len(payload) >= 5:
                        payload_mode = payload[4]

                    self._inc_mode_stat(payload_mode, "processed")

                    if status == "bug":
                        file_name, case_idx, group, counts, bug_mode = payload[:5]
                        self._record_overflow_retry(
                            payload[5] if len(payload) > 5 else None
                        )
                        self._record_validated_case(counts)
                        self._inc_mode_stat(bug_mode, "validated")
                        self._inc_mode_stat(bug_mode, "bugs")
                        self._observe_signature_result(
                            file_name,
                            case_idx,
                            success=True,
                        )
                        self._record_cardinality_result(
                            file_name,
                            case_idx,
                            bug_mode,
                            counts,
                            _tlp_variant_sqls(group),
                        )
                        self.report_bug(file_name, case_idx, group, counts)

                    elif status == "ok":
                        self._record_overflow_retry(
                            payload.get("overflow_retry")
                            if isinstance(payload, dict)
                            else None
                        )
                        self._record_validated_case(
                            payload.get("counts", {}) if isinstance(payload, dict) else {}
                        )
                        self._inc_mode_stat(payload_mode, "validated")
                        self._observe_signature_result(
                            payload.get("file") if isinstance(payload, dict) else None,
                            payload.get("case_idx") if isinstance(payload, dict) else None,
                            success=True,
                        )
                        if isinstance(payload, dict):
                            self._record_cardinality_result(
                                payload.get("file"),
                                payload.get("case_idx"),
                                payload_mode,
                                payload.get("counts", {}),
                                payload.get("variant_sqls", {}),
                            )

                    elif status == "skip":
                        self._record_overflow_retry(
                            payload.get("overflow_retry")
                            if isinstance(payload, dict)
                            else None
                        )
                        self.total_cases_skipped += 1
                        self._inc_mode_stat(payload_mode, "skipped")
                        self._observe_signature_result(
                            payload.get("file") if isinstance(payload, dict) else None,
                            payload.get("case_idx") if isinstance(payload, dict) else None,
                            success=False,
                            error_info=payload,
                        )
                        self._record_sql_error(payload, error_kind="validation")

                    if self.progress_log_interval > 0 and self.total_cases_run % self.progress_log_interval == 0:
                        print(".", end="", flush=True)
                        self._emit_progress_callback()

                    if self.deadline is not None and time.monotonic() >= self.deadline:
                        self.stopped_by_deadline = True
                        pool.terminate()
                        pool_terminated = True
                        break

                if _is_oracle_db():
                    self.oracle_excluded_wall_seconds = pool.excluded_wall_seconds
                    self.oracle_worker_restarts = pool.restarts
                    self._emit_progress_callback()
                if self.stopped_by_deadline:
                    break

                if not has_next_batch:
                    batch_start += len(batch_tasks)
                    batch_tasks = []
                    continue

                batch_start += len(batch_tasks)
                batch_tasks = [next_task]
                batch_tasks.extend(
                    itertools.islice(task_iter, VALIDATION_BATCH_SIZE - 1)
                )
        except OracleValidationDeadline:
            self.stopped_by_deadline = True
            pool.terminate()
            pool_terminated = True
        except OracleWorkerFailure as exc:
            self.worker_failure = exc.details
            self.log(f"[Oracle Worker Failure] {exc}")
            self.log(
                "[Oracle Worker Failure] Round incomplete; stopping scheduling and retaining received results. Full diagnostics are in this round's tlp_error_summary."
            )
            self._flush_oracle_reports(force=True)
            pool.terminate()
            pool_terminated = True
        finally:
            if not pool_terminated:
                pool.close()
            pool.join()
            if _is_oracle_db():
                self.oracle_excluded_wall_seconds = pool.excluded_wall_seconds
                self.oracle_worker_restarts = pool.restarts
            self._flush_oracle_reports(force=True)
        print("")
        self._emit_progress_callback(force=True)

        duration = time.time() - self.start_time
        streamed_tlp_seconds = 0.0
        if self.stratified_task_source is not None:
            streamed_tlp_seconds = self.stratified_task_source.elapsed_seconds
            duration = max(0.0, duration - streamed_tlp_seconds)
        raw_duration = duration
        if _is_oracle_db():
            duration = max(0.0, duration - self.oracle_excluded_wall_seconds)
            self.log(f"[Oracle Recovery] excluded={self.oracle_excluded_cases}, "
                     f"excluded_wall_seconds={self.oracle_excluded_wall_seconds:.2f}, "
                     f"raw_elapsed_seconds={raw_duration:.2f}")
        self.log("==================================================")
        self.log('Scan stopped early due to an Oracle worker failure (incomplete).' if self.worker_failure is not None else 'Scan completed.')
        self.log(f'Elapsed time          : {duration:.2f} seconds')
        if self.stratified_task_source is not None:
            self.log(f'Stratified streaming TLP construction: {streamed_tlp_seconds:.2f} seconds (excluded from the validation time above)')
        self.log(f"TLP mode   : {self.tlp_mode}")
        effective_loaded = self.total_cases_loaded - self.oracle_excluded_cases
        self.log(f'Loaded test groups    : {effective_loaded} groups')
        if _is_oracle_db():
            self.log(f'Original loaded groups: {self.total_cases_loaded} groups; excluded special faults: {self.oracle_excluded_cases} groups')
            if self.oracle_fault_counts:
                self.log(f'Special fault records: {self.oracle_fault_path}')
        self.log(f'Processed test groups : {self.total_cases_run} groups')
        self.log(f'Successfully validated: {self.total_cases_validated} groups')
        self.log(f'Skipped test groups   : {self.total_cases_skipped} groups')
        self.log(f'SQL Error             : {self.total_sql_errors} statements')
        self.log(f'  - validation error  : {self.total_validation_errors} statements')
        self.log(f'  - query timeout     : {self.total_query_timeouts} statements')
        self.log(f'Stopped by deadline   : {self.stopped_by_deadline}')
        error_summary_path = self._write_error_summary()

        self.log("==================================================")

        self.executor.close()
        self._close_report_handles()
        validation_status_path = self._write_validation_status()
        result = {
            "success": self.worker_failure is None,
            "round_complete": not self.stopped_by_deadline and self.worker_failure is None,
            "phase": "tlp_scan",
            "error": ("Oracle worker failure: " + self.worker_failure["kind"]
                      if self.worker_failure is not None else None),
            "stopped_by_deadline": self.stopped_by_deadline,
            "cases_loaded": self.total_cases_loaded,
            "cases_run": self.total_cases_run,
            "cases_validated": self.total_cases_validated,
            "cases_satisfiable": self.total_cases_satisfiable,
            "cases_skipped": self.total_cases_skipped,
            "sql_errors": self.total_sql_errors,
            "query_timeouts": self.total_query_timeouts,
            "error_summary_path": error_summary_path,
            "error_summary": {
                key: dict(values) for key, values in self.error_summary.items()
            },
            "mysql_numeric_overflow_retry": {
                "attempts": int(self.overflow_retry_attempts),
                "successes": int(self.overflow_retry_successes),
                "failures": max(
                    0,
                    int(self.overflow_retry_attempts)
                    - int(self.overflow_retry_successes),
                ),
            },
            "resource_bounded_sqls": self.resource_bounded_sqls,
            "resource_bounded_arguments": self.resource_bounded_arguments,
            "resource_bound_function_counts": dict(
                sorted(self.resource_bound_function_counts.items())
            ),
            "bugs": self.total_bugs_found,
            "elapsed_seconds": duration,
            "signature_feedback": self.signature_feedback,
            "signature_feedback_covered_cases": (
                self.signature_feedback_covered_cases
            ),
            "validation_status_path": validation_status_path,
            "validation_status_cases": int(self.validation_count),
            "cardinality_results_path": (
                str(self.cardinality_results_path)
                if self.cardinality_predicates_written
                else None
            ),
            "cardinality_predicates_written": int(
                self.cardinality_predicates_written
            ),
            "cardinality_values_written": int(
                self.cardinality_predicates_written * 3
            ),
        }
        if _is_oracle_db():
            result["worker_failure"] = self.worker_failure
            result.update(self._oracle_exclusion_summary())
            result["elapsed_seconds_raw"] = raw_duration
            result["success_rate"] = (self.total_cases_validated / self.total_cases_run
                                      if self.total_cases_run else None)
            result["cases_per_second"] = self.total_cases_run / duration if duration > 0 else None
            result["oracle_value_repair"] = dict(self.oracle_value_repair)
            result["oracle_error_categories"] = dict(
                self.oracle_error_category_counts
            )
        return result


if __name__ == "__main__":
    finder = TLPBugFinder()
    scan_result = finder.run()
    if scan_result.get("worker_failure"):
        sys.exit(1)
