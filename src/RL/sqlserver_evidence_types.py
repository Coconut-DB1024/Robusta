"""SQL Server physical types for private evidence copies, never graph identities.

Result-set description compiles expressions without executing/fetching them.
The bounded cache includes the SQL text: lengths, precision and literal-dependent
return metadata must not be guessed from a coarse operator name.
"""
from __future__ import annotations

import copy
import json
import re
import time
from collections import Counter
from pathlib import Path

from pkg.parser import SimpleParser


NUMERIC = frozenset({"bit", "tinyint", "smallint", "int", "bigint", "real", "float", "decimal", "money", "smallmoney"})
STRINGS = frozenset({"char", "varchar", "nchar", "nvarchar", "text", "ntext"})
BINARY = frozenset({"binary", "varbinary", "image", "rowversion"})
TEMPORAL = frozenset({"date", "time", "smalldatetime", "datetime", "datetime2", "datetimeoffset"})
SUPPORTED = NUMERIC | STRINGS | BINARY | TEMPORAL | {"uniqueidentifier", "condition", "null"}


def normalize_sqlserver_type(value):
    text = re.sub(r"\s+", "", str(value or "").lower())
    match = re.fullmatch(r"([a-z][a-z0-9]*)(\((?:max|\d+)(?:,\d+)?\))?", text)
    if not match:
        return ""
    base, modifiers = match.group(1), match.group(2) or ""
    base = {"integer": "int", "numeric": "decimal", "timestamp": "rowversion"}.get(base, base)
    if base not in SUPPORTED:
        return ""
    if base == "float" and modifiers in {"(53)", "(24)"}:
        return "float" if modifiers == "(53)" else "real"
    return base + modifiers


def base_type(value):
    return normalize_sqlserver_type(value).split("(", 1)[0]


def column_type(row):
    """Resolve aliases using sys.types' base type, retaining declared typmods."""
    row = {str(k).lower(): v for k, v in row.items()}
    base = base_type(row.get("base_type") or row.get("data_type") or row.get("type"))
    if not base:
        return ""
    if base in {"char", "varchar", "nchar", "nvarchar", "binary", "varbinary"}:
        length = row.get("max_length")
        if length is not None:
            length = int(length)
            width = "max" if length == -1 else str(length // 2 if base in {"nchar", "nvarchar"} else length)
            return f"{base}({width})"
    if base == "decimal" and row.get("precision") is not None and row.get("scale") is not None:
        return f"decimal({int(row['precision'])},{int(row['scale'])})"
    if base in {"time", "datetime2", "datetimeoffset"} and row.get("scale") is not None:
        return f"{base}({int(row['scale'])})"
    declared = normalize_sqlserver_type(row.get("type"))
    return declared if base_type(declared) == base else base


def stable_cast(source, target):
    """Type-only relations safe to share; value-sensitive coercions stay closed."""
    source, target = normalize_sqlserver_type(source), normalize_sqlserver_type(target)
    if not source or not target:
        return False
    if "condition" in {source, target}:
        return source == target
    if source == "null" or source == target:
        return True
    widening = {"bit": {"tinyint", "smallint", "int", "bigint"},
                "tinyint": {"smallint", "int", "bigint"},
                "smallint": {"int", "bigint"}, "int": {"bigint"}, "real": {"float"}}
    return target in widening.get(source, set())


def fixed_result_type(token, inputs=()):
    """Small exact return rules, not merge policies or invented overload tables."""
    token = str(token).upper()
    if token in {"AND", "OR", "NOT", "=", "<>", "!=", ">", ">=", "<", "<=", "!<", "!>",
                 "IS DISTINCT FROM", "IS NOT DISTINCT FROM", "IN", "NOT IN", "LIKE", "NOT LIKE",
                 "BETWEEN", "NOT BETWEEN", "IS NULL", "IS NOT NULL", "TRUE", "FALSE"}:
        return "condition"
    if token == "NULL":
        return "null"
    if token in {"DB_ID()", "@@TRANCOUNT", "XACT_STATE()", "ASCII()", "UNICODE()", "CHECKSUM()",
                 "BINARY_CHECKSUM()", "ISNUMERIC()", "ISDATE()", "ISJSON()", "DAY()", "MONTH()", "YEAR()", "DIFFERENCE()"}:
        return "int"
    if token == "@@SPID":
        return "smallint"
    if token == "PI()":
        return "float"
    if token in {"EOMONTH()", "DATEFROMPARTS()"}:
        return "date"
    if token == "BIT_COUNT()":
        return "bigint"
    if token == "GET_BIT()":
        return "bit"
    # Literal NULL is distinct from a typed null; ISNULL(NULL, x) uses x's type.
    if token == "ISNULL()" and len(inputs) == 2:
        return inputs[1] if inputs[0] == "null" else inputs[0]
    if token == "NULLIF()" and inputs:
        return inputs[0] if inputs[0] != "null" else ""
    values = list(inputs[1:] if token in {"IIF()", "CHOOSE()"} else inputs)
    if token in {"COALESCE()", "IIF()", "CHOOSE()", "GREATEST()", "LEAST()"}:
        # Mixed types require SQL Server's precedence/precision rules, obtained
        # by description. Do not infer them from independent sets of children.
        concrete = {v for v in values if v != "null"}
        if len(concrete) == 1 and "" not in concrete:
            return next(iter(concrete))
    return ""


class _SourceParser(SimpleParser):
    def parse_atom(self):
        start = self.pos
        node = super().parse_atom()
        node.sqlserver_source_sql = " ".join(self.tokens[start:self.pos])
        return node

    def parse_unary(self):
        start = self.pos
        node = super().parse_unary()
        node.sqlserver_source_sql = " ".join(self.tokens[start:self.pos])
        return node

    def parse_expression(self, min_prec):
        start = self.pos
        node = super().parse_expression(min_prec)
        node.sqlserver_source_sql = " ".join(self.tokens[start:self.pos])
        return node


class SQLServerEvidenceTyper:
    MAX_CALL_TYPES = 8192
    MAX_AST_NODES = 256
    MAX_METADATA_SECONDS = 30.0

    def __init__(self, executor=None):
        self.executor = executor
        self.owns_executor = executor is None
        self.cache = {}
        self.stats = Counter()
        self.metadata_seconds = 0.0
        self.unavailable = False
        path = Path(__file__).resolve().parents[1] / "pkg/configuration/sqlserver_rules.json"
        self.rules = json.loads(path.read_text(encoding="utf-8"))

    def close(self):
        if self.owns_executor and self.executor is not None:
            self.executor.close()

    def _describe(self, expression, table):
        key = (table, expression)
        if key in self.cache:
            self.stats["cache_hits"] += 1
            return self.cache[key]
        if self.unavailable or len(self.cache) >= self.MAX_CALL_TYPES or self.metadata_seconds >= self.MAX_METADATA_SECONDS:
            self.stats["budget_or_unavailable_skips"] += 1
            return ""
        self.cache[key] = ""
        started = time.monotonic()
        try:
            if self.executor is None:
                from RL.db_executor import create_db_executor
                self.executor = create_db_executor(db_type="sqlserver", autoconnect=True, query_timeout_seconds=5)
            sql = f"SELECT {expression} FROM {table} WHERE 1=0"
            result = self.executor.execute_query(
                "SELECT system_type_name, error_number, error_message "
                "FROM sys.dm_exec_describe_first_result_set(?, NULL, 0) "
                "WHERE column_ordinal = 1 OR error_number IS NOT NULL;",
                params=(sql,), fetch="all", autocommit=True,
            )
            self.stats["description_queries"] += 1
            if result.success:
                rows = result.rows or []
                if len(rows) == 1 and not rows[0].get("error_number"):
                    self.cache[key] = normalize_sqlserver_type(rows[0].get("system_type_name"))
            else:
                self.stats["description_failures"] += 1
                if getattr(result, "timed_out", False) or str(getattr(result, "error_code", "")) in {"229", "297", "916"}:
                    self.unavailable = True
        except Exception:
            self.stats["description_failures"] += 1
            self.unavailable = True
        finally:
            self.metadata_seconds += time.monotonic() - started
        return self.cache[key]

    def enrich(self, parse_result):
        if not getattr(parse_result, "success", False):
            return parse_result
        sql = str(getattr(parse_result, "sqlserver_executed_sql", "") or "")
        identifier = r'\[(?:[^\]]|\]\])+\]'
        match = re.fullmatch(rf'\s*SELECT TOP \(1\) 1 FROM ({identifier}\.{identifier}) WHERE (.+);\s*', sql, re.I | re.S)
        if not match:
            self.stats["missing_sql_context"] += 1
            return parse_result
        parser = _SourceParser(self.rules)
        source = parser.parse(match[2])
        if source is None or parser.pos != len(parser.tokens):
            self.stats["unaligned_sql"] += 1
            return parse_result
        original = parse_result.ast_root
        if original.return_type != "T_BOOL" and source.token == "IS NOT NULL" and len(source.children) == 1:
            source = source.children[0]
        result = copy.copy(parse_result)
        result.ast_root = copy.deepcopy(original)
        pairs = []

        def align(node, raw):
            if len(pairs) >= self.MAX_AST_NODES:
                return False
            children, raw_children = node.children or [], raw.children or []
            if node.token in {"TRUE", "FALSE"}:
                text = re.sub(r"[\s()]", "", raw.sqlserver_source_sql)
                if text != ("1=1" if node.token == "TRUE" else "1=0"):
                    return False
                pairs.append((node, raw))
                return True
            if len(children) != len(raw_children) or (children and node.token != raw.token):
                return False
            if not children:
                is_column = node.node_type == "column" or node.token.startswith("COL_")
                if is_column != (raw.node_type == "column" or raw.token.startswith("COL_")):
                    return False
                if not is_column and node.token != raw.token:
                    return False
            pairs.append((node, raw))
            return all(align(a, b) for a, b in zip(children, raw_children))

        if not align(result.ast_root, source):
            self.stats["unaligned_sql"] += 1
            return parse_result
        for node, raw in reversed(pairs):
            inputs = tuple(getattr(c, "physical_type", "") for c in node.children)
            dtype = fixed_result_type(node.token, inputs)
            if not dtype:
                dtype = self._describe(raw.sqlserver_source_sql, match[1])
            node.physical_type = dtype
            node.sqlserver_local_binding = True
            self.stats["typed_nodes" if dtype else "untyped_nodes"] += 1
        self.stats["enriched_predicates"] += 1
        return result
