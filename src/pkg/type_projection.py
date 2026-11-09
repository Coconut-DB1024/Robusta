from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple
import copy
import hashlib
import json
import os
import re
import tempfile
import time

from pkg.stage3_runtime import check_deadline


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EVIDENCE_PATH = PROJECT_ROOT / "output" / "graph_runtime_stats" / "joint_signatures.json"
DEFAULT_REPORT_PATH = PROJECT_ROOT / "output" / "stage3_type_projection_report.json"

_TYPE_ALIASES = {
    "int2": "smallint",
    "int4": "integer",
    "int8": "bigint",
    "float4": "real",
    "float8": "double precision",
    "bool": "boolean",
    "bpchar": "character",
    "varchar": "character varying",
    "timestamp": "timestamp without time zone",
    "timestamptz": "timestamp with time zone",
    "time": "time without time zone",
    "timetz": "time with time zone",
}
_POLYMORPHIC_PREFIXES = ("any", "internal")
_WILDCARD_TYPE = "__any__"

_INTEGER_TYPES = {"smallint", "integer", "bigint", "numeric"}
_FLOAT_TYPES = {"real", "double precision", "numeric"}
_STRING_TYPES = {"character", "character varying", "text", "name", '"char"'}
_DATE_TYPES = {
    "date",
    "time without time zone",
    "time with time zone",
    "timestamp without time zone",
    "timestamp with time zone",
    "interval",
}


_INTEGER_TYPES.update({"number", "int", "integer", "smallint"})
_FLOAT_TYPES.update({"number", "float", "binary_float", "binary_double"})
_STRING_TYPES.update({
    "char", "nchar", "varchar2", "nvarchar2", "clob", "nclob", "long",
})
_DATE_TYPES.update({
    "timestamp", "timestamp with local time zone",
    "interval year to month", "interval day to second",
})


def _normalize_type(value: Any) -> str:
    text = " ".join(str(value or "").strip().lower().split())
    if text.startswith("pg_catalog."):
        text = text[len("pg_catalog.") :]

    if "(" in text:
        text = " ".join(re.sub(r"\([^()]*\)", "", text).split())
    return _TYPE_ALIASES.get(text, text)


def _normalize_token(value: Any) -> str:
    return " ".join(str(value or "").strip().upper().split())


def _is_any_polymorphic(type_name: Any) -> bool:
    return _normalize_type(type_name).startswith("any")


def _polymorphic_target_accepts(source_type: str, target_type: str) -> bool:
    source = _normalize_type(source_type)
    target = _normalize_type(target_type)

    if target in {"anyelement", "anycompatible"}:
        return not _is_any_polymorphic(source)
    if target == "anynonarray":
        return not _is_any_polymorphic(source) and not source.endswith("[]")
    if target in {"anyarray", "anycompatiblearray"}:
        return source.endswith("[]")
    if target in {"anyrange", "anycompatiblerange"}:
        return source.endswith("range") and not source.endswith("multirange")
    if target in {"anymultirange", "anycompatiblemultirange"}:
        return source.endswith("multirange")

    return False


def _requires_correlated_polymorphic_binding(signature: "TypeSignature") -> bool:
    polymorphic_arg_count = sum(
        1 for arg_type in signature.arg_types if _is_any_polymorphic(arg_type)
    )
    return polymorphic_arg_count > 1 or _is_any_polymorphic(signature.result_type)


def _is_unbound_wildcard_signature(signature: "TypeSignature") -> bool:
    return any(
        _normalize_type(arg_type) == _WILDCARD_TYPE
        for arg_type in signature.arg_types
    ) or _normalize_type(signature.result_type) == _WILDCARD_TYPE


def _config_value(name: str, default: Any) -> Any:
    try:
        from RL.config import STAGE3_CONFIG

        return getattr(STAGE3_CONFIG, name, default)
    except Exception:
        return default


def _current_db_type() -> str:
    try:
        from RL.config import DB_CONFIG

        return str(getattr(DB_CONFIG, "db_type", "mysql") or "mysql").lower().strip()
    except Exception:
        return "mysql"


@dataclass(frozen=True)
class TypeSignature:
    arg_types: Tuple[str, ...]
    result_type: str
    source: str = "catalog"

    def normalized(self) -> "TypeSignature":
        return TypeSignature(
            tuple(_normalize_type(item) for item in self.arg_types),
            _normalize_type(self.result_type),
            self.source,
        )


@dataclass
class FunctionRecord:
    arg_types: Tuple[str, ...]
    result_type: str
    default_count: int = 0
    variadic_type: str = ""


@dataclass
class CatalogSnapshot:
    db_type: str
    operator_signatures: Dict[str, List[TypeSignature]] = field(default_factory=dict)
    function_records: Dict[str, List[FunctionRecord]] = field(default_factory=dict)
    implicit_casts: Set[Tuple[str, str]] = field(default_factory=set)
    terminal_types: Dict[str, Set[str]] = field(default_factory=dict)

    terminal_metadata: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)
    loaded: bool = False
    source: str = "none"
    error: Optional[str] = None
    _cast_cache: Dict[Tuple[str, str], bool] = field(default_factory=dict)
    _cast_adjacency: Optional[Dict[str, Set[str]]] = None

    def can_cast(self, source_type: str, target_type: str) -> bool:
        if self.db_type == "sqlserver":
            from RL.sqlserver_evidence_types import stable_cast
            return stable_cast(source_type, target_type)
        source = _normalize_type(source_type)
        target = _normalize_type(target_type)
        cache_key = (source, target)
        if cache_key in self._cast_cache:
            return self._cast_cache[cache_key]

        if self.db_type in {"oracle", "oracledb", "oracle_ai26"}:
            if source == _WILDCARD_TYPE:
                self._cast_cache[cache_key] = False
                return False
            if target == _WILDCARD_TYPE:
                self._cast_cache[cache_key] = False
                return False

            numeric_types = {
                "number", "numeric", "integer", "int", "smallint", "bigint",
                "float", "real", "binary_float", "binary_double",
            }
            if source in numeric_types and target in numeric_types:
                self._cast_cache[cache_key] = True
                return True
        if source == _WILDCARD_TYPE or target == _WILDCARD_TYPE or source == target:
            self._cast_cache[cache_key] = True
            return True

        if _is_any_polymorphic(target):
            compatible = _polymorphic_target_accepts(source, target)
            self._cast_cache[cache_key] = compatible
            return compatible

        compatible = (source, target) in self.implicit_casts
        self._cast_cache[cache_key] = compatible
        return compatible

    def function_signatures(self, token: str, arity: int) -> List[TypeSignature]:
        name = str(token or "").replace("()", "").strip().lower()
        signatures: List[TypeSignature] = []
        for record in self.function_records.get(name, []):
            declared = len(record.arg_types)
            minimum = max(0, declared - int(record.default_count))
            if record.variadic_type:
                if arity < minimum:
                    continue
                args = list(record.arg_types[: min(arity, declared)])
                while len(args) < arity:
                    args.append(record.variadic_type)
            else:
                if not minimum <= arity <= declared:
                    continue
                args = list(record.arg_types[:arity])
            signatures.append(TypeSignature(tuple(args), record.result_type).normalized())
        return _dedupe_signatures(signatures)


class CatalogProvider:
    db_types: Tuple[str, ...] = ()

    def load(self) -> CatalogSnapshot:
        raise NotImplementedError


class PostgreSQLCatalogProvider(CatalogProvider):
    db_types = ("postgresql", "postgres", "pg")

    @staticmethod
    def _rows(executor: Any, sql: str, params=None) -> List[Dict[str, Any]]:
        result = executor.execute_query(sql, params=params, fetch="all", autocommit=True)
        if not result.success:
            raise RuntimeError(result.error or "PostgreSQL catalog query failed")
        return list(result.rows or [])

    @staticmethod
    def _string_array(value: Any) -> Tuple[str, ...]:
        if isinstance(value, (list, tuple)):
            return tuple(_normalize_type(item) for item in value)
        text = str(value or "").strip().strip("{}")
        if not text:
            return ()
        return tuple(_normalize_type(item.strip('"')) for item in text.split(","))

    def load(self) -> CatalogSnapshot:
        from RL.config import get_target_table
        from RL.db_executor import create_db_executor

        snapshot = CatalogSnapshot(
            db_type="postgresql",
            loaded=False,
            source="live_postgresql_catalog",
        )
        executor = create_db_executor(
            db_type="postgresql",
            autoconnect=True,
            query_timeout_seconds=10,
        )
        try:
            operator_rows = self._rows(
                executor,
                """
                SELECT o.oprname AS token,
                       o.oprkind AS kind,
                       CASE WHEN o.oprleft = 0 THEN NULL
                            ELSE format_type(o.oprleft, NULL) END AS left_type,
                       CASE WHEN o.oprright = 0 THEN NULL
                            ELSE format_type(o.oprright, NULL) END AS right_type,
                       format_type(o.oprresult, NULL) AS result_type
                FROM pg_catalog.pg_operator AS o
                WHERE pg_catalog.pg_operator_is_visible(o.oid);
                """,
            )
            for row in operator_rows:
                args = []
                if row.get("left_type"):
                    args.append(_normalize_type(row["left_type"]))
                if row.get("right_type"):
                    args.append(_normalize_type(row["right_type"]))
                token = str(row.get("token", ""))
                snapshot.operator_signatures.setdefault(token, []).append(
                    TypeSignature(tuple(args), _normalize_type(row.get("result_type")))
                )

            function_rows = self._rows(
                executor,
                """
                SELECT p.proname AS token,
                       p.pronargdefaults AS default_count,
                       CASE WHEN p.provariadic = 0 THEN NULL
                            ELSE format_type(p.provariadic, NULL) END AS variadic_type,
                       ARRAY(
                           SELECT format_type(a.type_oid, NULL)
                           FROM unnest(p.proargtypes::oid[]) WITH ORDINALITY
                                AS a(type_oid, ord)
                           ORDER BY a.ord
                       ) AS arg_types,
                       format_type(p.prorettype, NULL) AS result_type
                FROM pg_catalog.pg_proc AS p
                WHERE p.prokind = 'f'
                  AND NOT p.proretset
                  AND pg_catalog.pg_function_is_visible(p.oid);
                """,
            )
            for row in function_rows:
                name = str(row.get("token", "")).lower()
                snapshot.function_records.setdefault(name, []).append(
                    FunctionRecord(
                        arg_types=self._string_array(row.get("arg_types")),
                        result_type=_normalize_type(row.get("result_type")),
                        default_count=int(row.get("default_count", 0) or 0),
                        variadic_type=_normalize_type(row.get("variadic_type")),
                    )
                )

            cast_rows = self._rows(
                executor,
                """
                SELECT format_type(c.castsource, NULL) AS source_type,
                       format_type(c.casttarget, NULL) AS target_type
                FROM pg_catalog.pg_cast AS c
                WHERE c.castcontext = 'i';
                """,
            )
            snapshot.implicit_casts = {
                (_normalize_type(row.get("source_type")), _normalize_type(row.get("target_type")))
                for row in cast_rows
            }

            table_name, table_schema, _ = get_target_table("postgresql")
            column_rows = self._rows(
                executor,
                """
                SELECT a.attname AS column_name,
                       format_type(a.atttypid, a.atttypmod) AS actual_type
                FROM pg_catalog.pg_attribute AS a
                JOIN pg_catalog.pg_class AS c ON c.oid = a.attrelid
                JOIN pg_catalog.pg_namespace AS n ON n.oid = c.relnamespace
                WHERE n.nspname = %s
                  AND c.relname = %s
                  AND a.attnum > 0
                  AND NOT a.attisdropped;
                """,
                params=(table_schema, table_name),
            )
            prefix_tokens = {
                "col_integer_": "COL_INT",
                "col_float_": "COL_FLOAT",
                "col_string_": "COL_STRING",
            }
            for row in column_rows:
                column_name = str(row.get("column_name", "")).lower()
                for prefix, terminal_token in prefix_tokens.items():
                    if column_name.startswith(prefix):
                        snapshot.terminal_types.setdefault(terminal_token, set()).add(
                            _normalize_type(row.get("actual_type"))
                        )
                        break

            snapshot.loaded = True
            return snapshot
        finally:
            executor.close()


class MySQLCatalogProvider(CatalogProvider):
    db_types = ("mysql",)

    @staticmethod
    def _rows(executor: Any, sql: str, params=None) -> List[Dict[str, Any]]:
        result = executor.execute_query(sql, params=params, fetch="all", autocommit=True)
        if not result.success:
            raise RuntimeError(result.error or "MySQL information_schema query failed")
        return list(result.rows or [])

    @staticmethod
    def _canonical_column_type(data_type: Any) -> str:
        value = str(data_type or "").strip().lower()
        if value in {"tinyint", "smallint", "mediumint", "int", "integer", "bigint", "bit"}:
            return "integer"
        if value in {"decimal", "numeric", "float", "double", "real"}:
            return "numeric"
        if value in {
            "char", "varchar", "tinytext", "text", "mediumtext", "longtext",
            "enum", "set",
        }:
            return "text"
        if value in {"date", "datetime", "timestamp", "time", "year"}:
            return "datetime"
        if value in {"binary", "varbinary", "tinyblob", "blob", "mediumblob", "longblob"}:
            return "binary"
        if value == "json":
            return "json"
        return value or _WILDCARD_TYPE

    def load(self) -> CatalogSnapshot:
        from RL.config import get_target_table
        from RL.db_executor import create_db_executor

        snapshot = CatalogSnapshot(
            db_type="mysql",
            loaded=False,
            source="live_mysql_information_schema",
        )
        executor = create_db_executor(
            db_type="mysql",
            autoconnect=True,
            query_timeout_seconds=10,
        )
        try:
            rows = self._rows(
                executor,
                """
                SELECT COLUMN_NAME, DATA_TYPE, COLUMN_TYPE,
                       CHARACTER_SET_NAME, COLLATION_NAME, IS_NULLABLE
                FROM information_schema.COLUMNS
                WHERE TABLE_SCHEMA = DATABASE()
                  AND TABLE_NAME = %s
                ORDER BY ORDINAL_POSITION;
                """,
                params=(get_target_table("mysql")[0],),
            )
            prefix_tokens = {
                "col_integer_": "COL_INT",
                "col_float_": "COL_FLOAT",
                "col_string_": "COL_STRING",
                "col_datetime_": "COL_DATETIME",
                "col_binary_": "COL_BINARY",
                "col_json_": "COL_JSON",
            }
            for row in rows:
                column_name = str(row.get("COLUMN_NAME", "")).lower()
                terminal_token = None
                for prefix, token in prefix_tokens.items():
                    if column_name.startswith(prefix):
                        terminal_token = token
                        break
                if terminal_token is None:
                    continue

                canonical_type = (
                    "json"
                    if terminal_token == "COL_JSON"
                    else self._canonical_column_type(row.get("DATA_TYPE"))
                )
                snapshot.terminal_types.setdefault(terminal_token, set()).add(
                    canonical_type
                )
                snapshot.terminal_metadata.setdefault(terminal_token, []).append(
                    {
                        "column_name": column_name,
                        "data_type": str(row.get("DATA_TYPE", "")),
                        "column_type": str(row.get("COLUMN_TYPE", "")),
                        "character_set": row.get("CHARACTER_SET_NAME"),
                        "collation": row.get("COLLATION_NAME"),
                        "nullable": row.get("IS_NULLABLE"),
                    }
                )

            snapshot.loaded = True
            return snapshot
        finally:
            executor.close()


def _oracle_catalog_type(
    data_type: Any,
    type_name: Any = None,
    pls_type: Any = None,
) -> str:
    value = data_type or type_name or pls_type
    text = " ".join(str(value or "").strip().lower().split())
    if text.startswith(("sys.", "public.")):
        text = text.split(".", 1)[1]
    base_text = " ".join(re.sub(r"\([^()]*\)", "", text).split())
    if base_text.startswith("timestamp with local") or base_text.startswith("timestamp local"):
        return "timestamp with local time zone"
    if base_text.startswith("timestamp with time"):
        return "timestamp with time zone"
    if base_text.startswith("timestamp"):
        return "timestamp without time zone"
    if base_text.startswith("interval year"):
        return "interval year to month"
    if base_text.startswith("interval day"):
        return "interval day to second"
    if base_text.startswith("number") or base_text in {"decimal", "numeric"}:
        return "number"
    if base_text in {"integer", "int", "smallint", "pls_integer", "binary_integer"}:
        return "integer"
    if base_text in {"float", "binary_float", "binary_double"}:
        return base_text
    if base_text in {"varchar", "varchar2"}:
        return "varchar2"
    if base_text in {"nvarchar", "nvarchar2"}:
        return "nvarchar2"
    if base_text in {"char", "nchar", "clob", "nclob", "long"}:
        return base_text
    if base_text in {"raw", "long raw", "blob", "bfile"}:
        return base_text
    if base_text in {"json", "boolean", "xmltype", "vector", "date"}:
        return base_text
    return base_text or _WILDCARD_TYPE


def _oracle_sqlfn_type(
    value: Any,
    resolved_args: Sequence[str] = (),
) -> str:
    text = " ".join(str(value or "").strip().upper().split())
    family_types = {
        "STRING": "varchar2",
        "CHARACTER": "varchar2",
        "NUMERIC": "number",
        "DATE": "date",
        "DATETIME": "date",
        "DATETYPE": "date",
        "BOOLEAN": "boolean",
        "BINARY": "raw",
        "RAW": "raw",
    }
    if text in family_types:
        return family_types[text]
    if text in {"", "ANY", "UNKNOWN", "UNDEFINED", "NULL"}:
        return _WILDCARD_TYPE

    arg_match = re.fullmatch(r"ARG(?:UMENT)?\s*(\d+)", text)
    if arg_match:
        position = int(arg_match.group(1))
        index = position - 1 if position > 0 else 0
        if 0 <= index < len(resolved_args):
            return _normalize_type(resolved_args[index])
        return _WILDCARD_TYPE

    return _oracle_catalog_type(value)


class OracleCatalogProvider(CatalogProvider):
    db_types = ("oracle", "oracledb", "oracle_ai26")

    @staticmethod
    def _rows(executor: Any, sql: str, params=None) -> List[Dict[str, Any]]:
        result = executor.execute_query(sql, params=params, fetch="all", autocommit=True)
        if not result.success:
            raise RuntimeError(result.error or "Oracle catalog query failed")
        return list(result.rows or [])

    @classmethod
    def _optional_rows(cls, executor: Any, sql: str) -> List[Dict[str, Any]]:
        try:
            return cls._rows(executor, sql)
        except Exception:
            return []

    @staticmethod
    def _value(row: Dict[str, Any], *names: str) -> Any:
        for name in names:
            if name in row and row[name] is not None:
                return row[name]
        upper = {str(key).upper(): value for key, value in row.items()}
        for name in names:
            value = upper.get(str(name).upper())
            if value is not None:
                return value
        return None

    @staticmethod
    def _add_operator(
        snapshot: CatalogSnapshot,
        token: str,
        arg_types: Sequence[str],
        result_type: str,
    ) -> None:
        signature = TypeSignature(tuple(arg_types), result_type).normalized()
        token_text = str(token or "").strip()
        if not token_text:
            return
        for key in {token_text, token_text.upper(), token_text.lower()}:
            snapshot.operator_signatures.setdefault(key, []).append(signature)

    @staticmethod
    def _add_function(
        snapshot: CatalogSnapshot,
        token: str,
        arg_types: Sequence[str],
        result_type: str,
        *,
        default_count: int = 0,
    ) -> None:
        name = str(token or "").replace("()", "").strip().lower()
        if not name:
            return
        snapshot.function_records.setdefault(name, []).append(
            FunctionRecord(
                arg_types=tuple(arg_types),
                result_type=_oracle_catalog_type(result_type),
                default_count=max(0, int(default_count or 0)),
            )
        )

    @classmethod
    def _seed_builtin_signatures(cls, snapshot: CatalogSnapshot) -> None:
        snapshot.terminal_types.update({
            "CURRENT_DATE": {"date"},
            "SYSDATE": {"date"},
            "CURRENT_TIMESTAMP": {"timestamp with time zone"},
            "SYSTIMESTAMP": {"timestamp with time zone"},
            "LOCALTIMESTAMP": {"timestamp without time zone"},
            "UID": {"number"},
            "ROWNUM": {"number"},
            "ORA_ROWSCN": {"number"},
            "ORA_INVOKING_USERID": {"number"},

            "CURRENT_USER": {"varchar2"},
            "SESSION_USER": {"varchar2"},
            "SYSTEM_USER": {"varchar2"},
            "CURRENT_SCHEMA": {"varchar2"},
            "DBTIMEZONE": {"varchar2"},
            "SESSIONTIMEZONE": {"varchar2"},
            "ORA_INVOKING_USER": {"varchar2"},
            "USER": {"varchar2"},
            "ROWID": {"rowid"},
            "SYS_GUID()": {"raw"},
        })

        numeric = ("number", "number")
        numeric_types = ("number", "integer", "float", "binary_float", "binary_double")
        string_types = ("varchar2", "nvarchar2", "char", "nchar", "clob", "nclob")
        comparable_string_types = ("varchar2", "nvarchar2", "char", "nchar")
        temporal_types = (
            "date", "timestamp without time zone", "timestamp with time zone",
            "timestamp with local time zone", "interval year to month",
            "interval day to second",
        )

        for token in ("+", "-", "*", "/"):
            cls._add_operator(snapshot, token, numeric, "number")
        for token in ("+", "-"):
            cls._add_operator(snapshot, token, ("number",), "number")

        equality_types = (
            numeric_types
            + comparable_string_types
            + temporal_types
            + ("boolean", "raw", "rowid")
        )
        for token in ("=", "!=", "<>"):
            for type_name in equality_types:
                cls._add_operator(snapshot, token, (type_name, type_name), "boolean")
        for token in ("<", "<=", ">", ">="):
            for type_name in numeric_types + comparable_string_types + temporal_types:
                cls._add_operator(snapshot, token, (type_name, type_name), "boolean")

        like_types = comparable_string_types
        for token in ("LIKE", "NOT LIKE", "LIKEC", "LIKE2", "LIKE4", "~~", "!~~"):
            for type_name in like_types:
                cls._add_operator(snapshot, token, (type_name, type_name), "boolean")
        cls._add_operator(snapshot, "||", ("varchar2", "varchar2"), "varchar2")
        cls._add_operator(snapshot, "IS JSON", ("json",), "boolean")
        cls._add_operator(snapshot, "IS NOT JSON", ("json",), "boolean")

        unary_numeric = (
            "ABS", "ACOS", "ASIN", "ATAN", "CEIL", "COS", "COSH", "EXP",
            "FLOOR", "LN", "LOG", "ROUND", "SIGN", "SIN", "SINH", "SQRT",
            "TAN", "TANH", "TRUNC",
        )
        for token in unary_numeric:
            cls._add_function(snapshot, token, ("number",), "number")
        for token in ("ATAN2", "BITAND", "MOD", "POWER", "REMAINDER"):
            cls._add_function(snapshot, token, numeric, "number")
        for token in ("ROUND", "TRUNC", "LOG"):
            cls._add_function(snapshot, token, numeric, "number")
        cls._add_function(snapshot, "WIDTH_BUCKET", ("number",) * 4, "number")
        for token in ("GREATEST", "LEAST"):
            cls._add_function(snapshot, token, numeric, "number")

        for token in ("LOWER", "UPPER", "INITCAP", "LTRIM", "RTRIM", "TRIM"):
            cls._add_function(snapshot, token, ("varchar2",), "varchar2")
        for token in ("ASCII", "LENGTH", "LENGTHB", "LENGTHC", "LENGTH2", "LENGTH4"):
            cls._add_function(snapshot, token, ("varchar2",), "number")
        for token in ("INSTR", "INSTRB"):
            cls._add_function(snapshot, token, ("varchar2", "varchar2"), "number")
        for token in ("SUBSTR", "SUBSTRB"):
            cls._add_function(snapshot, token, ("varchar2", "number", "number"), "varchar2")
        for token in ("REPLACE", "TRANSLATE"):
            cls._add_function(snapshot, token, ("varchar2",) * 3, "varchar2")
        for token in ("LPAD", "RPAD"):
            cls._add_function(snapshot, token, ("varchar2", "number", "varchar2"), "varchar2")
        cls._add_function(snapshot, "CONCAT", ("varchar2", "varchar2"), "varchar2")
        cls._add_function(snapshot, "CHR", ("number",), "varchar2")
        cls._add_function(snapshot, "NCHR", ("number",), "nvarchar2")
        for token in ("ASCIISTR", "UNISTR", "SOUNDEX"):
            cls._add_function(snapshot, token, ("varchar2",), "varchar2")
        cls._add_function(snapshot, "NLSSORT", ("varchar2", "varchar2"), "raw")
        cls._add_function(snapshot, "REGEXP_LIKE", ("varchar2", "varchar2"), "boolean")
        cls._add_function(snapshot, "REGEXP_LIKE", ("varchar2", "varchar2", "varchar2"), "boolean")
        for token in ("REGEXP_COUNT", "REGEXP_INSTR"):
            cls._add_function(snapshot, token, ("varchar2", "varchar2"), "number")
        cls._add_function(snapshot, "REGEXP_REPLACE", ("varchar2",) * 3, "varchar2")
        cls._add_function(snapshot, "REGEXP_SUBSTR", ("varchar2", "varchar2"), "varchar2")
        cls._add_function(snapshot, "LNNVL", ("boolean",), "boolean")

        for token in ("ADD_MONTHS", "LAST_DAY", "NEXT_DAY", "ROUND", "TRUNC"):
            cls._add_function(snapshot, token, ("date",), "date")
        cls._add_function(snapshot, "ROUND", ("date", "varchar2"), "date")
        cls._add_function(snapshot, "TRUNC", ("date", "varchar2"), "date")
        cls._add_function(snapshot, "ADD_MONTHS", ("date", "number"), "date")
        cls._add_function(snapshot, "MONTHS_BETWEEN", ("date", "date"), "number")
        for token in ("CURRENT_DATE", "SYSDATE"):
            cls._add_function(snapshot, token, (), "date")
        cls._add_function(snapshot, "CURRENT_TIMESTAMP", (), "timestamp with time zone")
        cls._add_function(snapshot, "SYSTIMESTAMP", (), "timestamp with time zone")
        cls._add_function(snapshot, "LOCALTIMESTAMP", (), "timestamp without time zone")

        cls._add_function(snapshot, "TO_NUMBER", ("varchar2",), "number")
        cls._add_function(snapshot, "TO_NUMBER", ("number",), "number")
        cls._add_function(snapshot, "TO_CHAR", ("number",), "varchar2")
        cls._add_function(snapshot, "TO_CHAR", ("date",), "varchar2")
        cls._add_function(snapshot, "TO_CHAR", ("number", "varchar2"), "varchar2")
        cls._add_function(snapshot, "TO_CHAR", ("date", "varchar2"), "varchar2")
        cls._add_function(snapshot, "TO_DATE", ("varchar2",), "date")
        cls._add_function(
            snapshot, "TO_TIMESTAMP", ("varchar2",), "timestamp without time zone"
        )
        cls._add_function(
            snapshot, "TO_TIMESTAMP_TZ", ("varchar2",), "timestamp with time zone"
        )
        cls._add_function(
            snapshot,
            "TO_TIMESTAMP_TZ",
            ("varchar2", "varchar2"),
            "timestamp with time zone",
        )
        cls._add_function(
            snapshot,
            "NUMTODSINTERVAL",
            ("number", "varchar2"),
            "interval day to second",
        )
        cls._add_function(
            snapshot,
            "NUMTOYMINTERVAL",
            ("number", "varchar2"),
            "interval year to month",
        )
        cls._add_function(
            snapshot,
            "FROM_TZ",
            ("timestamp without time zone", "varchar2"),
            "timestamp with time zone",
        )
        cls._add_function(
            snapshot,
            "SYS_EXTRACT_UTC",
            ("timestamp with time zone",),
            "timestamp without time zone",
        )
        cls._add_function(snapshot, "HEXTORAW", ("varchar2",), "raw")
        cls._add_function(snapshot, "RAWTOHEX", ("raw",), "varchar2")

        for token in ("JSON_EXISTS",):
            cls._add_function(snapshot, token, ("json", "varchar2"), "boolean")
        for token in ("JSON_VALUE", "JSON_QUERY"):
            cls._add_function(snapshot, token, ("json", "varchar2"), "varchar2")
        cls._add_function(snapshot, "JSON", ("varchar2",), "json")
        cls._add_function(snapshot, "JSON_SERIALIZE", ("json",), "varchar2")
        for token in ("NVL", "COALESCE", "NULLIF"):
            cls._add_function(snapshot, token, (_WILDCARD_TYPE, _WILDCARD_TYPE), _WILDCARD_TYPE)
        cls._add_function(
            snapshot,
            "NVL2",
            (_WILDCARD_TYPE, _WILDCARD_TYPE, _WILDCARD_TYPE),
            _WILDCARD_TYPE,
        )
        cls._add_function(
            snapshot,
            "DECODE",
            (_WILDCARD_TYPE, _WILDCARD_TYPE, _WILDCARD_TYPE),
            _WILDCARD_TYPE,
        )
        cls._add_function(snapshot, "TO_BOOLEAN", (_WILDCARD_TYPE,), "boolean")
        cls._add_function(snapshot, "STANDARD_HASH", ("varchar2", "varchar2"), "raw")

    @classmethod
    def _load_function_rows(
        cls,
        snapshot: CatalogSnapshot,
        rows: Iterable[Dict[str, Any]],
    ) -> None:
        grouped: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
        for row in rows:
            function_name = str(
                cls._value(row, "function_name", "object_name", "procedure_name", "token")
                or ""
            ).strip()
            if not function_name:
                continue
            package_name = str(cls._value(row, "package_name") or "").strip()
            overload = str(cls._value(row, "overload") or "").strip()
            key = (function_name.lower(), package_name.lower(), overload)
            item = grouped.setdefault(key, {"args": {}, "result": None, "defaults": set()})
            try:
                position = int(cls._value(row, "position") or 0)
            except (TypeError, ValueError):
                position = 0
            arg_type = _oracle_catalog_type(
                cls._value(row, "data_type", "actual_type"),
                cls._value(row, "type_name"),
                cls._value(row, "pls_type"),
            )
            if position == 0:
                item["result"] = arg_type
                continue
            in_out = str(cls._value(row, "in_out") or "IN").upper()
            if in_out not in {"IN", "IN/OUT"}:
                continue
            item["args"][position] = arg_type
            defaulted = str(cls._value(row, "defaulted") or "").upper()
            if defaulted in {"Y", "YES", "TRUE"}:
                item["defaults"].add(position)

        for (function_name, package_name, _overload), item in grouped.items():
            result_type = item.get("result")
            if not result_type:
                continue
            args = tuple(item["args"][position] for position in sorted(item["args"]))
            default_count = len(item["defaults"])
            aliases = {function_name}
            if package_name:
                aliases.add(f"{package_name}.{function_name}")
            for alias in aliases:
                cls._add_function(
                    snapshot,
                    alias,
                    args,
                    result_type,
                    default_count=default_count,
                )

    @classmethod
    def _load_operator_rows(
        cls,
        snapshot: CatalogSnapshot,
        rows: Iterable[Dict[str, Any]],
    ) -> None:
        grouped: Dict[Tuple[str, str], Dict[str, Any]] = {}
        for row in rows:
            token = str(cls._value(row, "operator_name", "token") or "").strip()
            if not token:
                continue
            binding_number = str(
                cls._value(row, "binding_number", "binding#", "binding_id") or ""
            )
            key = (token, binding_number)
            item = grouped.setdefault(
                key,
                {"result_type": None, "argument_count": 0, "args": {}},
            )
            result_type = cls._value(row, "return_type", "result_type")
            if result_type:
                item["result_type"] = _oracle_catalog_type(result_type)
            try:
                item["argument_count"] = max(
                    int(item["argument_count"] or 0),
                    int(cls._value(row, "number_of_arguments", "number_args", "arity") or 0),
                )
            except (TypeError, ValueError):
                pass

            position = cls._value(row, "position", "argument_position")
            argument_type = cls._value(row, "argument_type", "data_type")
            if position is not None and argument_type:
                try:
                    item["args"][int(position)] = _oracle_catalog_type(argument_type)
                except (TypeError, ValueError):
                    pass

            if not item["args"]:
                for index in range(1, min(int(item["argument_count"] or 0), 32) + 1):
                    value = cls._value(row, f"argument{index}_type", f"arg{index}_type")
                    if value:
                        item["args"][index] = _oracle_catalog_type(value)

        for (token, _binding_number), item in grouped.items():
            argument_count = max(0, min(int(item["argument_count"] or 0), 32))
            if argument_count <= 0 and item["args"]:
                argument_count = max(item["args"])
            position_offset = 0 if 0 in item["args"] else 1
            args = [
                item["args"].get(index + position_offset, _WILDCARD_TYPE)
                for index in range(argument_count)
            ]
            if args and item["result_type"]:
                cls._add_operator(snapshot, token, args, item["result_type"])

    @classmethod
    def _load_sql_function_rows(
        cls,
        snapshot: CatalogSnapshot,
        rows: Iterable[Dict[str, Any]],
    ) -> None:
        grouped: Dict[Tuple[str, str], Dict[str, Any]] = {}
        for row in rows:
            name = cls._value(row, "function_name", "name", "token")
            if not name:
                continue
            function_id = str(cls._value(row, "function_id", "func_id") or "")
            key = (function_id, str(name).upper())
            item = grouped.setdefault(
                key,
                {"name": str(name), "minimum": 0, "maximum": 0, "result": None, "args": {}},
            )
            try:
                minimum = max(0, int(cls._value(row, "minargs", "min_args") or 0))
                maximum = max(minimum, int(cls._value(row, "maxargs", "max_args") or minimum))
            except (TypeError, ValueError):
                continue
            item["minimum"] = minimum
            item["maximum"] = min(maximum, minimum + 8)
            item["result"] = cls._value(row, "return_type", "datatype")
            position = cls._value(row, "argument_position", "argnum", "position")
            argument_type = cls._value(row, "argument_type", "argument_datatype")
            if position is not None and argument_type:
                try:
                    item["args"][int(position)] = argument_type
                except (TypeError, ValueError):
                    pass

        for item in grouped.values():
            positions = item["args"]
            position_offset = 0 if 0 in positions else 1
            minimum = int(item["minimum"])
            maximum = int(item["maximum"])
            for arity in range(minimum, maximum + 1):
                resolved_args: List[str] = []
                for index in range(arity):
                    raw_type = positions.get(index + position_offset)
                    resolved_args.append(_oracle_sqlfn_type(raw_type, resolved_args))
                result_type = _oracle_sqlfn_type(item["result"], resolved_args)
                cls._add_function(snapshot, item["name"], resolved_args, result_type)

    def load(self) -> CatalogSnapshot:
        from RL.config import get_target_table
        from RL.db_executor import create_db_executor

        snapshot = CatalogSnapshot(
            db_type="oracle",
            loaded=False,
            source="live_oracle_catalog",
        )
        executor = create_db_executor(
            db_type="oracle",
            autoconnect=True,
            query_timeout_seconds=10,
        )
        try:
            self._seed_builtin_signatures(snapshot)

            column_rows = self._rows(
                executor,
                """
                SELECT table_name,
                       column_name,
                       data_type,
                       data_length,
                       data_precision,
                       data_scale,
                       char_length,
                       nullable
                FROM user_tab_columns
                WHERE table_name = :1
                ORDER BY table_name, column_id
                """,
                params=(get_target_table("oracle")[0],),
            )
            prefix_tokens = {
                "col_integer_": "COL_INT",
                "col_float_": "COL_FLOAT",
                "col_string_": "COL_STRING",
                "col_datetime_": "COL_DATETIME",
                "col_interval_": "COL_DATETIME",
                "col_binary_": "COL_BINARY",
                "col_json_": "COL_JSON",
                "col_boolean_": "COL_BOOL",
            }
            for row in column_rows:
                column_name = str(self._value(row, "column_name", "column") or "").lower()
                table_name = str(self._value(row, "table_name", "table") or "").lower()
                data_type = self._value(row, "data_type", "actual_type")
                canonical_type = _oracle_catalog_type(
                    data_type,
                    self._value(row, "type_name"),
                )
                column_metadata = {
                    "table_name": table_name,
                    "column_name": column_name,
                    "data_type": str(data_type or ""),
                    "canonical_type": canonical_type,
                    "data_length": self._value(row, "data_length"),
                    "data_precision": self._value(row, "data_precision"),
                    "data_scale": self._value(row, "data_scale"),
                    "char_length": self._value(row, "char_length"),
                    "nullable": self._value(row, "nullable"),
                }
                snapshot.terminal_metadata.setdefault(
                    "__ORACLE_SCHEMA_COLUMNS__", []
                ).append(column_metadata)
                terminal_token = None
                for prefix, token in prefix_tokens.items():
                    if column_name.startswith(prefix):
                        terminal_token = token
                        break
                if terminal_token is None:
                    continue
                snapshot.terminal_types.setdefault(terminal_token, set()).add(canonical_type)
                snapshot.terminal_metadata.setdefault(terminal_token, []).append(
                    dict(column_metadata)
                )

            function_rows = self._optional_rows(
                executor,
                """
                SELECT object_name AS function_name,
                       package_name,
                       overload,
                       position,
                       sequence,
                       argument_name,
                       data_type,
                       type_name,
                       pls_type,
                       in_out,
                       defaulted,
                       data_level
                FROM user_arguments
                WHERE data_level = 0
                  AND object_name IS NOT NULL
                ORDER BY object_name, package_name, overload, sequence
                """,
            )
            self._load_function_rows(snapshot, function_rows)

            operator_rows = self._optional_rows(
                executor,
                """
                SELECT b.operator_name,
                       b.binding# AS binding_number,
                       b.return_type,
                       b.number_args AS number_of_arguments,
                       a.position AS argument_position,
                       a.argument_type
                FROM user_opbindings b
                LEFT JOIN user_oparguments a
                  ON a.operator_name = b.operator_name
                 AND a.binding# = b.binding#
                ORDER BY b.operator_name, b.binding#, a.position
                """,
            )
            self._load_operator_rows(snapshot, operator_rows)

            all_operator_rows = self._optional_rows(
                executor,
                """
                SELECT b.operator_name,
                       b.binding# AS binding_number,
                       b.return_type,
                       b.number_args AS number_of_arguments,
                       a.position AS argument_position,
                       a.argument_type
                FROM all_opbindings b
                LEFT JOIN all_oparguments a
                  ON a.owner = b.owner
                 AND a.operator_name = b.operator_name
                 AND a.binding# = b.binding#
                WHERE b.owner = SYS_CONTEXT('USERENV', 'CURRENT_SCHEMA')
                ORDER BY b.operator_name, b.binding#, a.position
                """,
            )
            self._load_operator_rows(snapshot, all_operator_rows)

            sql_function_rows = self._optional_rows(
                executor,
                """
                SELECT f.func_id AS function_id,
                       f.name AS function_name,
                       f.minargs,
                       f.maxargs,
                       f.datatype AS return_type,
                       a.argnum AS argument_position,
                       a.datatype AS argument_type
                FROM v$sqlfn_metadata f
                LEFT JOIN v$sqlfn_arg_metadata a
                  ON a.func_id = f.func_id
                WHERE f.name IS NOT NULL
                ORDER BY f.func_id, a.argnum
                """,
            )
            self._load_sql_function_rows(snapshot, sql_function_rows)

            snapshot.implicit_casts = set()

            snapshot.loaded = True
            return snapshot
        finally:
            executor.close()


class SQLServerCatalogProvider(CatalogProvider):
    db_types = ("sqlserver",)

    def load(self) -> CatalogSnapshot:
        from RL.config import DB_CONFIG, get_target_table
        from RL.db_executor import create_db_executor
        from RL.sqlserver_evidence_types import column_type, base_type, NUMERIC, STRINGS

        snapshot = CatalogSnapshot(db_type="sqlserver", source="live_sqlserver_sys_columns")
        executor = None
        try:
            executor = create_db_executor(db_type="sqlserver", autoconnect=True, query_timeout_seconds=10)
            table_name, schema, _ = get_target_table("sqlserver", DB_CONFIG)

            result = executor.list_columns(table_name, schema=schema)
            if not result.success:
                raise RuntimeError(result.error or "SQL Server column metadata unavailable")
            for raw in result.rows or []:
                row = {str(k).lower(): v for k, v in raw.items()}
                dtype = column_type(row)
                if not dtype or row.get("is_hidden"):
                    continue
                base = base_type(dtype)
                if base in {"bit", "tinyint", "smallint", "int", "bigint"}:
                    token = "COL_INT"
                elif base in NUMERIC:
                    token = "COL_FLOAT"
                elif base in STRINGS:
                    token = "COL_STRING"
                else:
                    token = "COL_" + base.upper()
                snapshot.terminal_types.setdefault(token, set()).add(dtype)
                snapshot.terminal_metadata.setdefault(token, []).append({
                    **row, "physical_type": dtype, "table_schema": schema,
                    "table_name": table_name,
                })
            if not snapshot.terminal_metadata:
                raise RuntimeError("SQL Server target table has no supported visible columns")
            snapshot.loaded = True
        except Exception as exc:
            snapshot.error = str(exc)
        finally:
            if executor is not None:
                executor.close()
        return snapshot


_PROVIDER_REGISTRY: Dict[str, type] = {}
for _db_type in PostgreSQLCatalogProvider.db_types:
    _PROVIDER_REGISTRY[_db_type] = PostgreSQLCatalogProvider
for _db_type in MySQLCatalogProvider.db_types:
    _PROVIDER_REGISTRY[_db_type] = MySQLCatalogProvider
for _db_type in OracleCatalogProvider.db_types:
    _PROVIDER_REGISTRY[_db_type] = OracleCatalogProvider
for _db_type in SQLServerCatalogProvider.db_types:
    _PROVIDER_REGISTRY[_db_type] = SQLServerCatalogProvider

_CATALOG_CACHE: Dict[Tuple[Any, ...], CatalogSnapshot] = {}
_PROJECTION_CACHE: Dict[Tuple[Any, ...], Tuple[Dict[str, Any], Dict[str, Any]]] = {}


def register_catalog_provider(provider_class: type) -> None:
    for db_type in getattr(provider_class, "db_types", ()):
        _PROVIDER_REGISTRY[str(db_type).lower().strip()] = provider_class


def _catalog_cache_key(db_type: str) -> Tuple[Any, ...]:
    try:
        from RL.config import DB_CONFIG

        return (
            db_type,
            getattr(DB_CONFIG, "host", None),
            getattr(DB_CONFIG, "port", None),
            getattr(DB_CONFIG, "database", None),
            getattr(DB_CONFIG, "user", None),
            getattr(DB_CONFIG, "table_name", None),
        )
    except Exception:
        return (db_type,)


def load_catalog_snapshot(db_type: str) -> CatalogSnapshot:
    normalized_db = str(db_type or "").lower().strip()
    cache_key = _catalog_cache_key(normalized_db)
    if normalized_db == "sqlserver":
        from RL.config import DB_CONFIG
        cache_key += (str(getattr(DB_CONFIG, "schema", "dbo") or "dbo"),)
    if cache_key in _CATALOG_CACHE:
        return _CATALOG_CACHE[cache_key]

    provider_class = _PROVIDER_REGISTRY.get(normalized_db)
    if provider_class is None:
        snapshot = CatalogSnapshot(
            db_type=normalized_db,
            loaded=False,
            source="pass_through_no_provider",
        )
    else:
        try:
            snapshot = provider_class().load()
        except Exception as exc:
            snapshot = CatalogSnapshot(
                db_type=normalized_db,
                loaded=False,
                source="catalog_load_failed",
                error=str(exc),
            )
    _CATALOG_CACHE[cache_key] = snapshot
    return snapshot


def _dedupe_signatures(signatures: Iterable[TypeSignature]) -> List[TypeSignature]:
    unique: Dict[Tuple[Tuple[str, ...], str], TypeSignature] = {}
    for signature in signatures:
        normalized = signature.normalized()
        unique[(normalized.arg_types, normalized.result_type)] = normalized
    return list(unique.values())


def _operator_signatures(
    catalog: CatalogSnapshot,
    operator_token: str,
    arity: int,
) -> List[TypeSignature]:
    return _dedupe_signatures(
        signature
        for signature in catalog.operator_signatures.get(operator_token, [])
        if len(signature.arg_types) == arity
    )


def _catalog_signatures_for_node(
    catalog: CatalogSnapshot,
    token: str,
    arity: int,
) -> List[TypeSignature]:
    upper = _normalize_token(token)
    if str(token).strip().endswith("()"):
        return catalog.function_signatures(token, arity)

    if upper in {"AND", "OR"} and arity == 2:
        return [TypeSignature(("boolean", "boolean"), "boolean")]
    if upper == "NOT" and arity == 1:
        return [TypeSignature(("boolean",), "boolean")]
    if upper in {"IS NULL", "IS NOT NULL"} and arity == 1:
        return [TypeSignature((_WILDCARD_TYPE,), "boolean")]
    if upper in {
        "IS TRUE",
        "IS FALSE",
        "IS NOT TRUE",
        "IS NOT FALSE",
        "IS UNKNOWN",
        "IS NOT UNKNOWN",
    } and arity == 1:
        return [TypeSignature(("boolean",), "boolean")]
    if upper in {"IS NORMALIZED", "IS NOT NORMALIZED"} and arity == 1:
        return [TypeSignature(("text",), "boolean")]

    operator_aliases = {
        "!=": "<>",
        "LIKE": "~~",
        "NOT LIKE": "!~~",
        "ILIKE": "~~*",
        "NOT ILIKE": "!~~*",
        "SIMILAR TO": "~~",
    }
    if upper in {"IN", "NOT IN", "IS DISTINCT FROM", "IS NOT DISTINCT FROM"}:
        return _operator_signatures(catalog, "=", arity)
    if upper in {
        "BETWEEN",
        "BETWEEN SYMMETRIC",
        "NOT BETWEEN",
        "NOT BETWEEN SYMMETRIC",
    } and arity == 3:
        triples = []
        for signature in _operator_signatures(catalog, "<", 2):
            triples.append(
                TypeSignature(
                    (signature.arg_types[0], signature.arg_types[1], signature.arg_types[1]),
                    "boolean",
                )
            )
        return _dedupe_signatures(triples)

    catalog_token = operator_aliases.get(upper, str(token).strip())
    return _operator_signatures(catalog, catalog_token, arity)


def _is_boolean_type(type_name: str) -> bool:
    return _normalize_type(type_name) == "boolean"


def _is_string_type(type_name: str) -> bool:
    normalized = _normalize_type(type_name)
    return normalized in _STRING_TYPES or normalized in {"cstring", "unknown"}


def _coarse_accepts(return_type: str, exact_type: str) -> bool:
    coarse = str(return_type or "").upper()
    exact = _normalize_type(exact_type)
    if coarse == "T_BOOL":
        return _is_boolean_type(exact)
    if coarse in {"T_STR", "STRING"}:
        return _is_string_type(exact) or exact == "raw"
    if coarse == "INT":
        return exact in _INTEGER_TYPES
    if coarse == "FLOAT":
        return exact in _FLOAT_TYPES
    if coarse == "T_CMP":
        return not _is_boolean_type(exact)
    if coarse in {"T_ANY", "UNKNOWN"}:
        return True
    return True


def _type_bucket(type_name: str) -> str:
    normalized = _normalize_type(type_name)
    if normalized in {"smallint", "integer", "bigint", "int"}:
        return "integer"
    if normalized in {
        "real", "double precision", "numeric", "decimal", "number",
        "float", "binary_float", "binary_double",
    }:
        return "numeric"
    if _is_string_type(normalized):
        return "string"
    if normalized == "boolean":
        return "boolean"
    if normalized in _DATE_TYPES:
        return "temporal"
    if normalized.startswith(_POLYMORPHIC_PREFIXES):
        return normalized
    return normalized


def _group_catalog_signatures(
    signatures: Iterable[TypeSignature],
    reachable_arg_buckets: Optional[Set[str]] = None,
    *,
    allow_wildcard: bool = False,
) -> List[Dict[str, Any]]:
    groups: List[Dict[str, Any]] = []
    for signature in signatures:
        if _requires_correlated_polymorphic_binding(signature):
            continue
        arg_buckets = tuple(_type_bucket(item) for item in signature.arg_types)
        if reachable_arg_buckets is not None and any(
            bucket not in reachable_arg_buckets
            and not (allow_wildcard and bucket == _WILDCARD_TYPE)
            and not bucket.startswith(_POLYMORPHIC_PREFIXES)
            for bucket in arg_buckets
        ):
            continue
        exact_args = tuple(_normalize_type(item) for item in signature.arg_types)
        groups.append({
            "arg_types": exact_args,
            "arg_type_options": [{arg_type} for arg_type in exact_args],
            "result_types": {_normalize_type(signature.result_type)},
        })
    return groups


def _reachable_type_buckets(
    nodes: Dict[str, Dict[str, Any]],
    catalog: CatalogSnapshot,
) -> Set[str]:
    reachable_types: Set[str] = set()
    for node in nodes.values():
        if bool(node.get("is_terminal")):
            reachable_types.update(
                _normalize_type(item)
                for item in _terminal_result_types(node, catalog)
                if _normalize_type(item) != _WILDCARD_TYPE
            )

    changed = True
    while changed:
        changed = False
        for source_type, target_type in catalog.implicit_casts:
            source = _normalize_type(source_type)
            target = _normalize_type(target_type)
            if source in reachable_types and target not in reachable_types:
                reachable_types.add(target)
                changed = True
        for node in nodes.values():
            if bool(node.get("is_terminal")):
                continue
            signatures = _catalog_signatures_for_node(
                catalog,
                str(node.get("token", "")),
                len(node.get("slots", []) or []),
            )
            for signature in signatures:
                if _requires_correlated_polymorphic_binding(signature):
                    continue
                if any(
                    not any(
                        catalog.can_cast(source_type, expected_type)
                        for source_type in reachable_types
                    )
                    for expected_type in signature.arg_types
                ):
                    continue
                result_type = _normalize_type(signature.result_type)
                if result_type.startswith(_POLYMORPHIC_PREFIXES):
                    continue
                if result_type not in reachable_types:
                    reachable_types.add(result_type)
                    changed = True
    return {_type_bucket(item) for item in reachable_types}


def _terminal_result_types(
    node: Dict[str, Any],
    catalog: CatalogSnapshot,
) -> Set[str]:
    token = _normalize_token(node.get("token", ""))
    if token in catalog.terminal_types and catalog.terminal_types[token]:
        return set(catalog.terminal_types[token])
    if token in {"CURRENT_USER", "SESSION_USER"}:
        return {"name"}
    if token in {"TRUE", "FALSE"}:
        return {"boolean"}
    if token == "NULL":
        return {_WILDCARD_TYPE}
    if token.startswith("COL_INT"):
        return set(_INTEGER_TYPES)
    if token.startswith("COL_FLOAT"):
        return set(_FLOAT_TYPES)
    if token.startswith("COL_STRING"):
        return set(_STRING_TYPES)
    if token.startswith("VALUE_") and "INT" in token:
        return {"integer"}
    if token.startswith("VALUE_") and "FLOAT" in token:
        return {"numeric"}
    if token.startswith("VALUE_") and "STRING" in token:
        return set(_STRING_TYPES)
    if str(node.get("token", "")).endswith("()"):
        signatures = catalog.function_signatures(str(node.get("token", "")), 0)
        if signatures:
            return {signature.result_type for signature in signatures}
    return _fallback_result_types(node)


def _live_column_result_types(
    node: Dict[str, Any],
    catalog: CatalogSnapshot,
) -> Set[str]:
    token = _normalize_token(node.get("token", ""))
    if token.startswith("COLUMN_"):
        token = "COL_" + token[len("COLUMN_") :]
    if not token.startswith("COL_"):
        return set()
    return {
        _normalize_type(item)
        for item in catalog.terminal_types.get(token, set())
        if _normalize_type(item) and _normalize_type(item) != _WILDCARD_TYPE
    }


def _fallback_result_types(node: Dict[str, Any]) -> Set[str]:
    coarse = str(node.get("return_type", "") or "").upper()
    if coarse == "T_BOOL":
        return {"boolean"}
    if coarse in {"T_STR", "STRING"}:
        return set(_STRING_TYPES)
    if coarse == "INT":
        return set(_INTEGER_TYPES)
    if coarse == "FLOAT":
        return set(_FLOAT_TYPES)
    if coarse in {"T_CMP", "T_ANY", "UNKNOWN"}:
        return set().union(_INTEGER_TYPES, _FLOAT_TYPES, _STRING_TYPES, _DATE_TYPES)
    return {_normalize_type(coarse)} if coarse else {_WILDCARD_TYPE}


def _possible_node_result_types(
    node: Dict[str, Any],
    catalog: CatalogSnapshot,
) -> Set[str]:
    if bool(node.get("is_terminal")):
        return _terminal_result_types(node, catalog)
    signatures = _catalog_signatures_for_node(
        catalog,
        str(node.get("token", "")),
        len(node.get("slots", []) or []),
    )
    if catalog.db_type in {"oracle", "oracledb", "oracle_ai26"}:
        signatures = [
            signature
            for signature in signatures
            if not _is_unbound_wildcard_signature(signature)
        ]
    result_types = {
        _normalize_type(signature.result_type)
        for signature in signatures
        if not _requires_correlated_polymorphic_binding(signature)
        and _coarse_accepts(
            str(node.get("return_type", "")),
            signature.result_type,
        )
    }
    if result_types:
        return result_types

    if (
        catalog.db_type in {"oracle", "oracledb", "oracle_ai26"}
        and catalog.loaded
        and str(node.get("return_type", "") or "").upper()
        in {"T_CMP", "T_ANY", "UNKNOWN"}
    ):
        return {_WILDCARD_TYPE}
    return _fallback_result_types(node)


def _oracle_observed_result_types(
    token: Any,
    child_ids: Sequence[Any],
    result_types_by_original: Dict[str, Set[str]],
) -> Set[str]:
    normalized = _normalize_token(token)
    if normalized.endswith("()"):
        normalized = normalized[:-2].rstrip()
    ids = [str(item) for item in child_ids]
    if normalized in {"NVL", "COALESCE", "NULLIF"}:
        value_ids = ids
    elif normalized == "NVL2":
        value_ids = ids[1:]
    elif normalized == "DECODE":
        value_ids = ids[2::2]
        if len(ids) >= 3 and len(ids) % 2 == 0:
            value_ids.append(ids[-1])
    else:
        return set()
    concrete_sets = [
        {
            _normalize_type(item)
            for item in result_types_by_original.get(child_id, set())
            if _normalize_type(item) != _WILDCARD_TYPE
        }
        for child_id in value_ids
    ]
    concrete_sets = [items for items in concrete_sets if items]
    if not concrete_sets:
        return set()
    inferred = set.intersection(*concrete_sets)
    if inferred:
        return inferred

    numeric_types = {
        "number", "numeric", "integer", "int", "smallint", "bigint",
        "float", "real", "binary_float", "binary_double",
    }
    if all(items and items.issubset(numeric_types) for items in concrete_sets):
        return {"number"}
    return set()


def _oracle_single_result_family(result_types: Iterable[str]) -> Optional[str]:
    families: Set[str] = set()
    for value in result_types:
        normalized = _normalize_type(value)
        if not normalized or normalized == _WILDCARD_TYPE:
            continue
        if normalized in {
            "number", "numeric", "integer", "int", "smallint", "bigint",
            "float", "real", "binary_float", "binary_double",
        }:
            families.add("numeric")
        elif normalized in _STRING_TYPES:
            families.add("string")
        elif normalized in _DATE_TYPES:
            families.add("temporal")
        elif normalized == "boolean":
            families.add("boolean")
        elif normalized in {"raw", "blob"}:
            families.add("binary")
        elif normalized == "json":
            families.add("json")
        else:
            families.add(normalized)
    return next(iter(families)) if len(families) == 1 else None


def _compatible_result_set(
    catalog: CatalogSnapshot,
    result_types: Iterable[str],
    expected_type: str,
) -> bool:
    return any(catalog.can_cast(item, expected_type) for item in result_types)


def _load_evidence(
    path: Path,
    db_type: str,
    minimum_success: int,
    known_invalid_min_observations: int,
) -> Tuple[Dict[str, List[Dict[str, Any]]], Dict[str, int]]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    stats = {
        "entries": 0,
        "successful_entries": 0,
        "type_failure_entries": 0,
        "value_failure_entries": 0,
        "execution_failure_entries": 0,
        "known_invalid_entries": 0,
        "unknown_entries": 0,
    }
    if not path.exists() or path.stat().st_size <= 0:
        return grouped, stats
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except Exception:
        return grouped, stats

    aliases = {db_type}
    if db_type in {"postgresql", "postgres", "pg"}:
        aliases.update({"postgresql", "postgres", "pg"})
    if db_type in {"oracle", "oracledb", "oracle_ai26"}:
        aliases.update({"oracle", "oracledb", "oracle_ai26"})
    for record in (payload.get("entries", {}) or {}).values():
        if not isinstance(record, dict) or str(record.get("db_type", "")) not in aliases:
            continue
        stats["entries"] += 1
        if int(record.get("type_failure_count", 0) or 0) > 0:
            stats["type_failure_entries"] += 1
        if int(record.get("value_failure_count", 0) or 0) > 0:
            stats["value_failure_entries"] += 1
        if int(record.get("execution_failure_count", 0) or 0) > 0:
            stats["execution_failure_entries"] += 1
        evidence_class = _evidence_compatibility_class(
            record,
            minimum_success=minimum_success,
            known_invalid_min_observations=known_invalid_min_observations,
            db_type=db_type,
        )
        if evidence_class == "known_invalid":
            stats["known_invalid_entries"] += 1
        elif evidence_class == "unknown":
            stats["unknown_entries"] += 1
        outer_id = str(record.get("outer_node_id", ""))
        child_ids = list(record.get("slot_child_node_ids", []) or [])
        if outer_id and child_ids:
            grouped.setdefault(outer_id, []).append(record)
            if int(record.get("success_count", 0) or 0) >= minimum_success:
                stats["successful_entries"] += 1
    return grouped, stats


def _record_arg_types(record: Dict[str, Any]) -> Tuple[str, ...]:
    for field in ("arg_types", "slot_child_return_types"):
        values = record.get(field)
        if isinstance(values, (list, tuple)) and values:
            return tuple(str(value or "UNKNOWN") for value in values)
    return ()


def _evidence_compatibility_class(
    record: Dict[str, Any],
    *,
    minimum_success: int,
    known_invalid_min_observations: int,
    db_type: str = "",
) -> str:
    successes = max(0, int(record.get("success_count", 0) or 0))
    type_failures = max(0, int(record.get("type_failure_count", 0) or 0))
    if successes >= minimum_success:
        return "compatible"

    if successes == 0 and type_failures >= known_invalid_min_observations:
        return "known_invalid"
    return "unknown"


def _evidence_confidence(
    successes: int,
    type_failures: int,
    value_failures: int,
    execution_failures: int = 0,
    db_type: str = "",
) -> Optional[float]:
    successes = max(0, int(successes))
    type_weight = 1.0
    value_weight = 1.0
    execution_weight = 1.0
    normalized_db = str(db_type or "").lower().strip()
    if normalized_db == "sqlserver":
        from RL.joint_signature_evidence import SQLSERVER_FAILURE_WEIGHTS
        type_weight = SQLSERVER_FAILURE_WEIGHTS["type_failure_count"]
        value_weight = SQLSERVER_FAILURE_WEIGHTS["value_failure_count"]
        execution_weight = SQLSERVER_FAILURE_WEIGHTS["execution_failure_count"]
    elif normalized_db == "mysql":
        value_weight = max(
            0.0,
            float(_config_value("stage3_mysql_value_failure_weight", 0.05) or 0.0),
        )
        execution_weight = max(
            0.0,
            float(
                _config_value("stage3_mysql_execution_failure_weight", 0.02)
                or 0.0
            ),
        )
    elif normalized_db in {"oracle", "oracledb", "oracle_ai26"}:
        type_weight = max(
            0.0,
            float(_config_value("stage3_oracle_type_failure_weight", 1.0) or 0.0),
        )
        value_weight = max(
            0.0,
            float(_config_value("stage3_oracle_value_failure_weight", 0.40) or 0.0),
        )
        execution_weight = max(
            0.0,
            float(
                _config_value("stage3_oracle_execution_failure_weight", 0.20)
                or 0.0
            ),
        )
    failures = (
        type_weight * max(0, int(type_failures))
        + value_weight * max(0, int(value_failures))
        + execution_weight * max(0, int(execution_failures))
    )
    if successes + failures <= 0:
        return None
    return (successes + 1.0) / (successes + failures + 2.0)


def _oracle_empirical_override_allowed(
    record: Dict[str, Any],
    *,
    minimum_success: int,
) -> bool:
    successes = max(0, int(record.get("success_count", 0) or 0))
    type_failures = max(0, int(record.get("type_failure_count", 0) or 0))
    if successes < max(1, int(minimum_success)):
        return False
    if type_failures <= 0:
        return True

    return successes >= 4 * type_failures and successes >= 4 * max(
        1, int(minimum_success)
    )


_ORACLE_MAX_EMPIRICAL_OVERRIDES_PER_NODE = 96


_ORACLE_EXACT_VARIANT_SAFETY_CAP = 96


def _partition_slot_product(
    slots: Sequence[Sequence[str]],
    excluded: Iterable[Sequence[str]],
    *,
    max_rectangles: Optional[int] = None,
) -> List[Tuple[Tuple[str, ...], ...]]:
    normalized_slots = tuple(tuple(dict.fromkeys(slot)) for slot in slots)
    arity = len(normalized_slots)
    excluded_set = {
        tuple(str(item) for item in values)
        for values in excluded
        if len(values) == arity
    }

    bounded_limit = None
    if max_rectangles is not None:
        bounded_limit = max(0, int(max_rectangles))
    emitted = 0

    def decompose(
        slot_index: int,
        blocked_suffixes: Set[Tuple[str, ...]],
    ) -> List[Tuple[Tuple[str, ...], ...]]:
        nonlocal emitted
        if bounded_limit is not None and emitted >= bounded_limit:
            return []
        if slot_index >= arity:
            if () in blocked_suffixes:
                return []
            emitted += 1
            return [()]

        current_slot = normalized_slots[slot_index]
        suffix_length = arity - slot_index
        grouped: Dict[str, Set[Tuple[str, ...]]] = {}
        for suffix in blocked_suffixes:
            if len(suffix) != suffix_length or not suffix:
                continue
            grouped.setdefault(suffix[0], set()).add(suffix[1:])

        rectangles: List[Tuple[Tuple[str, ...], ...]] = []
        unrestricted = tuple(item for item in current_slot if item not in grouped)
        if unrestricted:
            rectangles.append((unrestricted,) + normalized_slots[slot_index + 1 :])
            emitted += 1
            if bounded_limit is not None and emitted >= bounded_limit:
                return rectangles

        for child_id in current_slot:
            if bounded_limit is not None and emitted >= bounded_limit:
                break
            child_blocked = grouped.get(child_id)
            if not child_blocked:
                continue
            for tail in decompose(slot_index + 1, child_blocked):
                rectangles.append(((child_id,),) + tail)
                if bounded_limit is not None and emitted >= bounded_limit:
                    break
        return rectangles

    return decompose(0, excluded_set)


_MYSQL_SAFE_MERGE_TOKENS = {
    "scalar": frozenset({"AND", "OR", "XOR", "&&", "||"}),

    "comparison": frozenset({
        "=", "!=", "<>", "<=>", "<", "<=", ">", ">=",
    }),

    "numeric": frozenset({
        "+", "-", "*", "/", "DIV", "MOD", "%", "&", "|", "^",
        "<<", ">>",
    }),

    "string": frozenset({"CONCAT", "CONCAT_WS", "REPLACE"}),
}

_MYSQL_TERMINAL_FAMILY_TOKENS = {
    "numeric_int": frozenset({"COL_INT", "VALUE_TYPICAL_INT"}),
    "numeric_float": frozenset({"COL_FLOAT", "VALUE_TYPICAL_FLOAT"}),
    "string": frozenset({"COL_STRING", "VALUE_TYPICAL_STRING"}),
    "boolean": frozenset({"TRUE", "FALSE"}),
}


def _mysql_merge_policy(token: Any, arity: int) -> Optional[str]:
    normalized = _normalize_token(token)
    if normalized in _MYSQL_SAFE_MERGE_TOKENS["scalar"] and arity == 2:
        return "scalar"
    if normalized in _MYSQL_SAFE_MERGE_TOKENS["comparison"] and arity == 2:
        return "comparison"
    if normalized in _MYSQL_SAFE_MERGE_TOKENS["numeric"] and arity == 2:
        return "numeric"
    if normalized in _MYSQL_SAFE_MERGE_TOKENS["string"] and arity >= 2:
        return "string"
    return None


def _mysql_merge_family(
    child_id: Any,
    return_type: Any,
    policy: str,
    original_nodes: Dict[str, Dict[str, Any]],
) -> Optional[str]:
    child_node = original_nodes.get(str(child_id)) or {}
    if not bool(child_node.get("is_terminal")):
        return None
    token = _normalize_token(child_node.get("token", ""))
    for family, tokens in _MYSQL_TERMINAL_FAMILY_TOKENS.items():
        if token not in tokens:
            continue
        if policy == "numeric" and not family.startswith("numeric_"):
            continue
        if policy == "string" and family != "string":
            continue
        if policy == "scalar" and family not in {
            "numeric_int", "numeric_float", "string", "boolean"
        }:
            continue
        if policy == "comparison" and family not in {
            "numeric_int", "numeric_float", "string", "boolean"
        }:
            continue
        return family
    return None


def _mysql_merge_key(
    token: Any,
    child_ids: Sequence[Any],
    child_return_types: Sequence[Any],
    original_nodes: Dict[str, Dict[str, Any]],
) -> Optional[Tuple[str, ...]]:
    policy = _mysql_merge_policy(token, len(child_ids))
    if policy is None:
        return None
    if len(child_return_types) != len(child_ids):
        child_return_types = [
            (original_nodes.get(str(child_id)) or {}).get("return_type", "")
            for child_id in child_ids
        ]
    families = tuple(
        _mysql_merge_family(child_id, return_type, policy, original_nodes)
        for child_id, return_type in zip(child_ids, child_return_types)
    )
    if any(family is None for family in families):
        return None
    return families


def _mysql_descriptor_key(
    descriptor: Dict[str, Any],
    token: Any,
    original_nodes: Dict[str, Dict[str, Any]],
) -> Optional[Tuple[str, ...]]:
    child_ids = tuple(str(item) for item in (descriptor.get("exact_child_ids") or []))
    if not child_ids:
        return None
    child_types = tuple(descriptor.get("observed_child_return_types") or ())
    return _mysql_merge_key(token, child_ids, child_types, original_nodes)


def _mysql_evidence_record_key(
    record: Dict[str, Any],
    token: Any,
    original_nodes: Dict[str, Dict[str, Any]],
) -> Optional[Tuple[str, ...]]:
    child_ids = tuple(str(item) for item in (record.get("slot_child_node_ids") or []))
    if not child_ids:
        return None
    child_types = tuple(record.get("slot_child_return_types") or ())
    return _mysql_merge_key(token, child_ids, child_types, original_nodes)


_ORACLE_SAFE_MERGE_TOKENS = {
    "boolean": frozenset({"AND", "OR"}),
    "comparison": frozenset({"=", "!=", "<>", "<", "<=", ">", ">="}),
    "numeric": frozenset({"+", "-", "*", "/"}),
    "string": frozenset({"||"}),
}

_ORACLE_TERMINAL_FAMILY_TOKENS = {
    "numeric_int": frozenset({"COL_INT", "VALUE_TYPICAL_INT"}),
    "numeric_float": frozenset({"COL_FLOAT", "VALUE_TYPICAL_FLOAT"}),
    "string": frozenset({"COL_STRING", "VALUE_TYPICAL_STRING"}),
    "boolean": frozenset({"TRUE", "FALSE"}),
}

_ORACLE_POLICY_FAMILIES = {
    "boolean": frozenset({"boolean"}),
    "comparison": frozenset({"numeric_int", "numeric_float", "string"}),
    "numeric": frozenset({"numeric_int", "numeric_float"}),
    "string": frozenset({"string"}),
}


def _oracle_merge_policy(token: Any, arity: int) -> Optional[str]:
    normalized = _normalize_token(token)
    if arity != 2:
        return None
    for policy, tokens in _ORACLE_SAFE_MERGE_TOKENS.items():
        if normalized in tokens:
            return policy
    return None


def _oracle_merge_family(
    child_id: Any,
    return_type: Any,
    policy: str,
    original_nodes: Dict[str, Dict[str, Any]],
) -> Optional[str]:
    child_node = original_nodes.get(str(child_id)) or {}
    if not bool(child_node.get("is_terminal")):
        return None
    token = _normalize_token(child_node.get("token", ""))
    for family, tokens in _ORACLE_TERMINAL_FAMILY_TOKENS.items():
        if token in tokens and family in _ORACLE_POLICY_FAMILIES.get(policy, ()):
            return family
    return None


def _oracle_merge_key(
    token: Any,
    child_ids: Sequence[Any],
    child_return_types: Sequence[Any],
    original_nodes: Dict[str, Dict[str, Any]],
) -> Optional[Tuple[str, ...]]:
    policy = _oracle_merge_policy(token, len(child_ids))
    if policy is None:
        return None
    if len(child_return_types) != len(child_ids):
        child_return_types = [
            (original_nodes.get(str(child_id)) or {}).get("return_type", "")
            for child_id in child_ids
        ]
    families = tuple(
        _oracle_merge_family(child_id, return_type, policy, original_nodes)
        for child_id, return_type in zip(child_ids, child_return_types)
    )
    if any(family is None for family in families):
        return None
    return families


def _oracle_descriptor_key(
    descriptor: Dict[str, Any],
    token: Any,
    original_nodes: Dict[str, Dict[str, Any]],
) -> Optional[Tuple[str, ...]]:
    child_ids = tuple(str(item) for item in (descriptor.get("exact_child_ids") or []))
    if not child_ids:
        return None
    child_types = tuple(descriptor.get("observed_child_return_types") or ())
    key = _oracle_merge_key(token, child_ids, child_types, original_nodes)
    if key is None:
        return None
    physical_types = tuple(
        str(value or "").strip().lower()
        for value in (descriptor.get("slot_child_physical_types") or [])
    )
    return key + (("physical",) + physical_types if any(physical_types) else ())


def _oracle_evidence_record_key(
    record: Dict[str, Any],
    token: Any,
    original_nodes: Dict[str, Dict[str, Any]],
) -> Optional[Tuple[str, ...]]:
    child_ids = tuple(str(item) for item in (record.get("slot_child_node_ids") or []))
    if not child_ids:
        return None
    child_types = _record_arg_types(record)
    key = _oracle_merge_key(token, child_ids, child_types, original_nodes)
    if key is None:
        return None
    physical_types = tuple(
        str(value or "").strip().lower()
        for value in (record.get("slot_child_physical_types") or [])
    )
    return key + (("physical",) + physical_types if any(physical_types) else ())


def _projection_descriptor_rank(
    descriptor: Dict[str, Any],
    *,
    db_type: str,
) -> Tuple[Any, ...]:
    successes = max(0, int(descriptor.get("observed_success_count", 0) or 0))
    type_failures = max(0, int(descriptor.get("observed_type_failure_count", 0) or 0))
    value_failures = max(0, int(descriptor.get("observed_value_failure_count", 0) or 0))
    execution_failures = max(
        0,
        int(descriptor.get("observed_execution_failure_count", 0) or 0),
    )
    confidence = _evidence_confidence(
        successes,
        type_failures,
        value_failures,
        execution_failures,
        db_type=db_type,
    )
    canonical = json.dumps(
        {
            "exact": descriptor.get("exact_child_ids"),
            "allowed": descriptor.get("allowed_child_ids_by_slot"),
            "source": descriptor.get("source"),
            "tuple_hash": descriptor.get("tuple_hash", ""),
        },
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )
    return (
        -(float(confidence) if confidence is not None else 0.0),
        -successes,
        type_failures + value_failures + execution_failures,
        1 if descriptor.get("source") == "unknown_exploration" else 0,
        canonical,
    )


def _mysql_descriptor_rank(descriptor: Dict[str, Any]) -> Tuple[Any, ...]:
    return _projection_descriptor_rank(descriptor, db_type="mysql")


def _oracle_descriptor_rank(descriptor: Dict[str, Any]) -> Tuple[Any, ...]:
    signature = json.dumps(
        {
            "arg_types": list(descriptor.get("arg_types") or []),
            "arg_type_options": [
                sorted(options)
                for options in (descriptor.get("arg_type_options") or [])
            ],
            "result_types": sorted(descriptor.get("result_types") or []),
        },
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )
    return (0 if descriptor.get("oracle_binding_version") == 1 else 1,) + _projection_descriptor_rank(
        descriptor,
        db_type="oracle",
    ) + (signature,)


def _finalize_evidence_descriptors(
    node: Dict[str, Any],
    descriptors: Sequence[Dict[str, Any]],
    observed_records: Sequence[Dict[str, Any]],
    original_nodes: Dict[str, Dict[str, Any]],
    *,
    max_variants: int,
    descriptor_key: Callable[
        [Dict[str, Any], Any, Dict[str, Dict[str, Any]]],
        Optional[Tuple[str, ...]],
    ],
    evidence_record_key: Callable[
        [Dict[str, Any], Any, Dict[str, Dict[str, Any]]],
        Optional[Tuple[str, ...]],
    ],
    rank_key: Callable[[Dict[str, Any]], Tuple[Any, ...]],
    metadata_prefix: str,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    token = node.get("token", "")
    blocked_families: Set[Tuple[str, ...]] = set()
    for record in observed_records:
        key = evidence_record_key(record, token, original_nodes)
        if key is None:
            continue
        successes = int(record.get("success_count", 0) or 0)
        failures = (
            int(record.get("type_failure_count", 0) or 0)
            + int(record.get("value_failure_count", 0) or 0)
            + int(record.get("execution_failure_count", 0) or 0)
        )
        if successes <= 0 or failures > 0:
            blocked_families.add(key)

    groups: Dict[Tuple[str, ...], List[Dict[str, Any]]] = {}
    passthrough: List[Dict[str, Any]] = []
    for descriptor in descriptors:
        eligible = (
            descriptor.get("source") == "observed"
            and descriptor.get("compatibility_class") == "compatible"
            and descriptor.get("exact_child_ids")
            and int(descriptor.get("observed_success_count", 0) or 0) > 0
            and not any(
                int(descriptor.get(field, 0) or 0) > 0
                for field in (
                    "observed_type_failure_count",
                    "observed_value_failure_count",
                    "observed_execution_failure_count",
                )
            )
        )
        key = (
            descriptor_key(descriptor, token, original_nodes)
            if eligible
            else None
        )
        if key is None or key in blocked_families:
            passthrough.append(descriptor)
        else:
            groups.setdefault(key, []).append(descriptor)

    merged: List[Dict[str, Any]] = []
    merged_count = 0
    for key, group in groups.items():
        if len(group) < 2:
            passthrough.extend(group)
            continue

        slot_members = [
            sorted({
                str((descriptor.get("exact_child_ids") or [])[slot])
                for descriptor in group
            })
            for slot in range(len(group[0].get("exact_child_ids", []) or []))
        ]
        expected = 1
        for members in slot_members:
            expected *= len(members)
        observed_tuples = {
            tuple(str(item) for item in (descriptor.get("exact_child_ids") or []))
            for descriptor in group
        }
        if expected <= 0 or len(observed_tuples) != expected:
            passthrough.extend(group)
            continue

        merged_descriptor = copy.deepcopy(group[0])
        merged_descriptor["exact_child_ids"] = None
        merged_descriptor["allowed_child_ids_by_slot"] = tuple(
            tuple(members) for members in slot_members
        )
        merged_descriptor["observed_child_return_types"] = None
        merged_descriptor["observed_success_count"] = sum(
            int(descriptor.get("observed_success_count", 0) or 0)
            for descriptor in group
        )
        merged_descriptor["observed_type_failure_count"] = 0
        merged_descriptor["observed_value_failure_count"] = 0
        merged_descriptor["observed_execution_failure_count"] = 0
        if metadata_prefix == "oracle":
            for field in ("observed_direct_success_count", "observed_context_success_count"):
                merged_descriptor[field] = sum(int(d.get(field, 0)) for d in group)
        merged_descriptor[f"{metadata_prefix}_safe_merge_family"] = key
        merged_descriptor[f"{metadata_prefix}_safe_merge_member_count"] = len(group)
        merged.append(merged_descriptor)
        merged_count += len(group) - 1

    merged_descriptors = passthrough + merged
    before_limit = len(merged_descriptors)
    bounded_limit = max(1, int(max_variants or 1))
    if before_limit > bounded_limit:
        unknown = [
            descriptor
            for descriptor in merged_descriptors
            if descriptor.get("source") == "unknown_exploration"
        ]
        known = [
            descriptor
            for descriptor in merged_descriptors
            if descriptor.get("source") != "unknown_exploration"
        ]
        known.sort(key=rank_key)
        reserved_unknown = min(1, len(unknown), bounded_limit)
        selected = known[: max(0, bounded_limit - reserved_unknown)]
        selected.extend(unknown[:reserved_unknown])
        selected.sort(key=rank_key)
        bounded = selected
    else:
        bounded = list(merged_descriptors)

    return bounded, {
        "variants_before_limit": before_limit,
        "variants_after_limit": len(bounded),
        "variants_pruned_by_limit": max(0, before_limit - len(bounded)),
        "safe_merge_reductions": merged_count,
        "observed_variants_after_limit": sum(
            1 for descriptor in bounded if descriptor.get("source") == "observed"
        ),
        "unknown_variants_after_limit": sum(
            1
            for descriptor in bounded
            if descriptor.get("source") == "unknown_exploration"
        ),
    }


def _finalize_mysql_descriptors(
    node: Dict[str, Any],
    descriptors: Sequence[Dict[str, Any]],
    observed_records: Sequence[Dict[str, Any]],
    original_nodes: Dict[str, Dict[str, Any]],
    *,
    max_variants: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    return _finalize_evidence_descriptors(
        node,
        descriptors,
        observed_records,
        original_nodes,
        max_variants=max_variants,
        descriptor_key=_mysql_descriptor_key,
        evidence_record_key=_mysql_evidence_record_key,
        rank_key=_mysql_descriptor_rank,
        metadata_prefix="mysql",
    )


def _finalize_oracle_descriptors(
    node: Dict[str, Any],
    descriptors: Sequence[Dict[str, Any]],
    observed_records: Sequence[Dict[str, Any]],
    original_nodes: Dict[str, Dict[str, Any]],
    *,
    max_variants: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    return _finalize_evidence_descriptors(
        node,
        descriptors,
        observed_records,
        original_nodes,
        max_variants=max_variants,
        descriptor_key=_oracle_descriptor_key,
        evidence_record_key=_oracle_evidence_record_key,
        rank_key=_oracle_descriptor_rank,
        metadata_prefix="oracle",
    )


def _mysql_evidence_descriptors(
    node: Dict[str, Any],
    observed_records: Sequence[Dict[str, Any]],
    possible_result_types: Set[str],
    *,
    minimum_success: int,
    known_invalid_min_observations: int,
    max_unknown_variants: int = 1,
    db_type: str = "mysql",
    include_unknown: bool = True,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    slots = [list(slot) for slot in (node.get("slots", []) or [])]
    arity = len(slots)
    slot_members = [set(slot) for slot in slots]
    descriptors: List[Dict[str, Any]] = []
    represented_tuples: Set[Tuple[str, ...]] = set()
    stats = {
        "blocked_combinations": 0,
        "evidence_variants": 0,
        "unobserved_variants": 0,
    }

    for record in observed_records:
        child_ids = tuple(str(item) for item in (record.get("slot_child_node_ids", []) or []))
        if len(child_ids) != arity or any(
            child_id not in slot_members[index]
            for index, child_id in enumerate(child_ids)
        ):
            continue
        type_failures = max(0, int(record.get("type_failure_count", 0) or 0))
        value_failures = max(0, int(record.get("value_failure_count", 0) or 0))
        execution_failures = max(
            0,
            int(record.get("execution_failure_count", 0) or 0),
        )
        success_count = max(0, int(record.get("success_count", 0) or 0))
        has_failures = type_failures + value_failures + execution_failures > 0

        if not has_failures and success_count <= 0:
            continue
        represented_tuples.add(child_ids)
        if not has_failures:
            descriptors.append({
                "source": "observed",
                "compatibility_class": "compatible",
                "arg_types": None,
                "arg_type_options": None,
                "result_types": set(possible_result_types),
                "exact_child_ids": child_ids,
                "allowed_child_ids_by_slot": None,
                "observed_child_return_types": tuple(
                    record.get("slot_child_return_types", []) or []
                ),
                "tuple_hash": str(record.get("tuple_hash", "") or ""),
                "slot_child_tokens": tuple(
                    record.get("slot_child_tokens", []) or []
                ),
                "slot_child_fingerprints": tuple(
                    record.get("slot_child_fingerprints", []) or []
                ),
                "slot_child_physical_types": tuple(
                    record.get("slot_child_physical_types", []) or []
                ),
                "physical_type": str(record.get("physical_type", "") or ""),
                "oracle_binding_version": int(record.get("oracle_binding_version", 0)),
                **({
                    "oracle_identity_before_enrichment": record.get("oracle_identity_before_enrichment"),
                    "observed_direct_success_count": int(record.get("direct_success_count", 0)),
                    "observed_context_success_count": int(record.get("context_success_count", 0)),
                } if db_type in {"oracle", "oracledb", "oracle_ai26"} else {}),
                "subtree_fingerprint": str(
                    record.get("subtree_fingerprint", "") or ""
                ),
                "observed_success_count": success_count,
                "observed_type_failure_count": 0,
                "observed_value_failure_count": 0,
                "observed_execution_failure_count": 0,
            })
            stats["evidence_variants"] += 1
            continue

        evidence_class = _evidence_compatibility_class(
            record,
            minimum_success=minimum_success,
            known_invalid_min_observations=known_invalid_min_observations,
            db_type=db_type,
        )
        if evidence_class == "known_invalid":
            stats["blocked_combinations"] += 1
            continue

        observed_class = (
            "weak_observed" if evidence_class == "unknown" else evidence_class
        )
        descriptors.append({
            "source": "observed",
            "compatibility_class": observed_class,
            "arg_types": None,
            "arg_type_options": None,
            "result_types": set(possible_result_types),
            "exact_child_ids": child_ids,
            "allowed_child_ids_by_slot": None,
            "observed_child_return_types": tuple(
                record.get("slot_child_return_types", []) or []
            ),
            "tuple_hash": str(record.get("tuple_hash", "") or ""),
            "slot_child_tokens": tuple(record.get("slot_child_tokens", []) or []),
            "slot_child_fingerprints": tuple(
                record.get("slot_child_fingerprints", []) or []
            ),
            "slot_child_physical_types": tuple(
                record.get("slot_child_physical_types", []) or []
            ),
            "physical_type": str(record.get("physical_type", "") or ""),
            "oracle_binding_version": int(record.get("oracle_binding_version", 0)),
            **({
                "oracle_identity_before_enrichment": record.get("oracle_identity_before_enrichment"),
                "observed_direct_success_count": int(record.get("direct_success_count", 0)),
                "observed_context_success_count": int(record.get("context_success_count", 0)),
            } if db_type in {"oracle", "oracledb", "oracle_ai26"} else {}),
            "subtree_fingerprint": str(
                record.get("subtree_fingerprint", "") or ""
            ),
            "observed_success_count": success_count,
            "observed_type_failure_count": type_failures,
            "observed_value_failure_count": value_failures,
            "observed_execution_failure_count": execution_failures,
        })
        stats["evidence_variants"] += 1

    bounded_unknown_variants = min(
        2,
        max(1, int(max_unknown_variants or 1)),
    )
    unknown_rectangles = (
        _partition_slot_product(
            slots,
            represented_tuples,
            max_rectangles=bounded_unknown_variants,
        )
        if include_unknown
        else ()
    )
    for rectangle in unknown_rectangles:
        descriptors.append({
            "source": "unknown_exploration",
            "compatibility_class": "unknown",
            "arg_types": None,
            "arg_type_options": None,
            "result_types": set(possible_result_types),
            "exact_child_ids": None,
            "allowed_child_ids_by_slot": rectangle,
            "observed_child_return_types": None,
            "observed_success_count": 0,
            "observed_type_failure_count": 0,
            "observed_value_failure_count": 0,
            "observed_execution_failure_count": 0,
        })
        stats["unobserved_variants"] += 1

    return descriptors, stats


def _edge_count(graph_data: Dict[str, Any]) -> int:
    return sum(
        len(slot)
        for node in (graph_data.get("nodes", {}) or {}).values()
        for slot in (node.get("slots", []) or [])
    )


def _oracle_variant_identity(descriptor: Dict[str, Any]) -> Dict[str, Any]:
    descriptor = {**descriptor, **(descriptor.get("oracle_identity_before_enrichment") or {})}
    physical = list(descriptor.get("slot_child_physical_types") or [])
    local = descriptor.get("oracle_binding_version") == 1
    return {
        "oracle_binding_version": int(descriptor.get("oracle_binding_version", 0)),
        "physical_type": descriptor.get("physical_type", ""),
        "slot_child_physical_types": physical,
        "binding_fingerprints": [
            "" if local and i < len(physical) and physical[i] else fp
            for i, fp in enumerate(descriptor.get("slot_child_fingerprints") or [])
        ],
        "legacy_subtree": "" if local else descriptor.get("subtree_fingerprint", ""),
    }


def _variant_id(base_node_id: str, descriptor: Dict[str, Any]) -> str:
    material = json.dumps(
        {
            "arg_types": list(descriptor.get("arg_types", []) or []),
            "arg_type_options": [
                sorted(options)
                for options in (descriptor.get("arg_type_options", []) or [])
            ],
            "result_types": sorted(descriptor.get("oracle_identity_result_types", descriptor.get("result_types", [])) or []),
            "exact_child_ids": list(descriptor.get("exact_child_ids", []) or []),
            **(_oracle_variant_identity(descriptor)
               if descriptor.get("oracle_evidence") else {}),
            "allowed_child_ids_by_slot": [
                sorted(options)
                for options in (
                    descriptor.get("allowed_child_ids_by_slot", []) or []
                )
            ],
        },
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )
    suffix = hashlib.sha1(material.encode("utf-8")).hexdigest()[:12]
    return f"{base_node_id}::TYPE_{suffix}"


def _projection_base_fingerprint(
    node_id: str,
    original_nodes: Dict[str, Dict[str, Any]],
    cache: Optional[Dict[str, str]] = None,
    _active: Optional[Set[str]] = None,
) -> str:
    if cache is None:
        cache = {}
    if node_id in cache:
        return cache[node_id]
    if _active is None:
        _active = set()
    node = original_nodes.get(node_id, {}) or {}

    if node_id in _active:
        cycle_payload = {
            "cycle_node": str(node_id),
            "token": str(node.get("token", "UNKNOWN")),
            "return_type": str(node.get("return_type", "UNKNOWN")),
        }
        return hashlib.sha1(
            json.dumps(
                cycle_payload,
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()

    _active.add(node_id)
    try:
        payload = {
            "token": str(node.get("token", "UNKNOWN")),
            "node_type": "operator" if node.get("slots") else "terminal",
            "return_type": str(node.get("return_type", "UNKNOWN")),
            "children": [
                _projection_base_fingerprint(child_id, original_nodes, cache, _active)
                for slot in (node.get("slots", []) or [])
                for child_id in (slot or [])[:1]
            ],
        }
        fingerprint = hashlib.sha1(
            json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        cache[node_id] = fingerprint
        return fingerprint
    finally:
        _active.discard(node_id)


def _projection_terminal_fingerprint(
    node_id: str,
    physical_type: str,
    original_nodes: Dict[str, Dict[str, Any]],
    cache: Optional[Dict[str, str]] = None,
) -> str:
    base = original_nodes.get(node_id, {}) or {}
    physical = str(physical_type or "").strip().lower()
    if not physical:
        return _projection_base_fingerprint(node_id, original_nodes, cache)
    payload = {
        "token": str(base.get("token", "UNKNOWN")),
        "node_type": "terminal",
        "return_type": str(base.get("return_type", "UNKNOWN")),
        "children": [],
        "physical_type": physical,
    }
    return hashlib.sha1(
        json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _terminal_type_variant_id(base_node_id: str, result_type: str) -> str:
    material = _normalize_type(result_type)
    suffix = hashlib.sha1(material.encode("utf-8")).hexdigest()[:12]
    return f"{base_node_id}::COLUMN_TYPE_{suffix}"


def _prune_projection(nodes: Dict[str, Dict[str, Any]]) -> int:
    removed_total = 0
    while True:
        dead = {
            node_id
            for node_id, node in nodes.items()
            if not bool(node.get("is_terminal"))
            and any(not slot for slot in (node.get("slots", []) or []))
        }
        if not dead:
            return removed_total
        removed_total += len(dead)
        for node_id in dead:
            nodes.pop(node_id, None)
        for node in nodes.values():
            if node.get("is_terminal"):
                continue
            node["slots"] = [
                [child_id for child_id in slot if child_id in nodes]
                for slot in (node.get("slots", []) or [])
            ]


def _atomic_report(path: Path, report: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        delete=False,
        dir=str(path.parent),
        suffix=".tmp",
    ) as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        temp_path = Path(handle.name)
    os.replace(str(temp_path), str(path))


def _path_fingerprint(path: Optional[Path]) -> Tuple[Any, ...]:
    if path is None or not path.exists():
        return (str(path or ""), None, None)
    stat = path.stat()
    return (str(path.resolve()), int(stat.st_mtime_ns), int(stat.st_size))


def _sqlserver_merge_key(item, token, original_nodes):
    from RL.sqlserver_evidence_types import base_type, NUMERIC
    ids = item.get("exact_child_ids") or item.get("slot_child_node_ids") or []
    types = tuple(item.get("slot_child_physical_types") or [])
    if len(ids) != 2 or len(types) != 2 or not all(types):
        return None
    if not all(original_nodes.get(cid, {}).get("is_terminal") for cid in ids):
        return None
    token = _normalize_token(token)
    if token in {"AND", "OR"} and types == ("condition", "condition"):
        return (token,) + types
    if token in {"=", "<>", "!=", ">", ">=", "<", "<=", "!<", "!>"}:
        if types[0] == types[1] and base_type(types[0]) in NUMERIC:
            return (token,) + types
    return None


def _project_sqlserver_graph(graph_data, *, evidence_path, report_path, deadline, catalog_snapshot):
    from RL.sqlserver_evidence_types import (
        normalize_sqlserver_type as normalize, base_type, fixed_result_type,
    )
    check_deadline(deadline, "SQL Server type projection")
    enabled = bool(_config_value("stage3_type_projection_enabled", True))
    nodes = graph_data.get("nodes", {}) or {}
    roots = graph_data.get("roots", []) or []
    limit = max(1, int(_config_value("stage3_sqlserver_max_projection_variants", 192) or 192))
    minimum = max(1, int(_config_value("stage3_type_projection_min_success_count", 1)))
    failure_minimum = max(1, int(_config_value("stage3_type_projection_known_invalid_min_observations", 4)))
    unknown_ratio = max(0.0, min(1.0, float(_config_value("stage3_sqlserver_implicit_conversion_exploration_ratio", 0.05))))
    report = {
        "version": 3, "updated_at": time.time(), "db_type": "sqlserver", "enabled": enabled,
        "original_nodes": len(nodes), "original_roots": len(roots), "original_edges": _edge_count(graph_data),
        "sqlserver_max_projection_variants": limit, "unknown_exploration_ratio": unknown_ratio,
        "exact_signature_compatibility": False, "signature_source": "stage2_described_calls",
        "known_invalid_min_observations": failure_minimum,
        "catalog_source": "disabled", "catalog_loaded": False, "catalog_error": None,
        "evidence": {}, "projection_effective": False,
    }

    def finish(graph, status):
        report.update(status=status, projected_nodes=len(graph.get("nodes", {})),
                      projected_roots=len(graph.get("roots", [])), projected_edges=_edge_count(graph))
        _atomic_report(Path(report_path), report)
        return graph, report

    if not enabled:
        return finish(graph_data, "disabled")
    catalog = catalog_snapshot
    if catalog is None:
        catalog = (load_catalog_snapshot("sqlserver") if _config_value("stage3_type_projection_catalog_enabled", True)
                   else CatalogSnapshot(db_type="sqlserver", source="disabled"))
    if catalog.db_type != "sqlserver":
        raise ValueError("SQL Server projection requires a SQL Server catalog snapshot")
    report.update(catalog_source=catalog.source, catalog_loaded=catalog.loaded, catalog_error=catalog.error)
    report["catalog_terminal_metadata_count"] = sum(map(len, catalog.terminal_metadata.values()))
    report["catalog_terminal_types"] = {k: sorted(v) for k, v in catalog.terminal_types.items()}
    evidence = {}
    if _config_value("stage3_type_projection_evidence_enabled", True):
        report["evidence_error"] = None
        try:
            if Path(evidence_path).exists():
                payload = json.loads(Path(evidence_path).read_text(encoding="utf-8"))
                if not isinstance(payload.get("entries", {}), dict):
                    raise ValueError("SQL Server evidence entries must be an object")
        except Exception as exc:
            report["evidence_error"] = str(exc)
            return finish(graph_data, "pass_through_evidence_load_failed")
        evidence, report["evidence"] = _load_evidence(Path(evidence_path), "sqlserver", minimum, failure_minimum)
    if not catalog.loaded and not evidence:
        return finish(graph_data, "pass_through_no_catalog_or_evidence")
    rules = json.loads((Path(__file__).parent / "configuration/sqlserver_rules.json").read_text(encoding="utf-8"))
    definitions = rules["definitions"]
    terminal_scope = rules["candidate_bitmap"]["terminal_tokens"]
    stats = {
        "variants_before_limit": 0, "variants_after_limit": 0, "variants_pruned_by_limit": 0,
        "safe_merge_reductions": 0, "observed_variants_after_limit": 0, "unknown_variants_after_limit": 0,
        "blocked_combinations": 0, "unresolved_variants": 0, "typed_observations": 0,

        "value_sensitive_variants": 0, "condition_scalar_mismatches": 0,
    }

    leaf_types = {nid: set(catalog.terminal_types.get(n.get("token"), set()))
                  for nid, n in nodes.items() if n.get("is_terminal")}
    for records in evidence.values():
        for record in records:
            for cid, dtype in zip(record.get("slot_child_node_ids") or [], record.get("slot_child_physical_types") or []):
                if cid in leaf_types and normalize(dtype):
                    token = nodes[cid].get("token", "")
                    if token.startswith("COL_") and catalog.loaded:
                        continue
                    leaf_types[cid].add(normalize(dtype))

    projected, variants, descriptors = {}, {}, {}
    for nid, node in nodes.items():
        check_deadline(deadline, "SQL Server terminal projection")
        token = node.get("token", "")
        if not node.get("is_terminal") or token not in terminal_scope:
            continue
        types = {normalize(t) for t in leaf_types.get(nid, set())} - {""}
        fixed = fixed_result_type(token)
        if fixed:
            types = {fixed}
        if not types:
            if token.startswith("COL_") and catalog.loaded:
                continue
            types = {""}
        for dtype in sorted(types):
            vid = nid + "::SS" + hashlib.sha1(dtype.encode()).hexdigest()[:12]
            copied = copy.deepcopy(node)
            copied.update(base_node_id=nid, stage3_result_types=[dtype] if dtype else [])
            if token.startswith("COL_") and dtype:
                copied["stage3_column_type"] = base_type(dtype)
                copied["stage3_column_metadata"] = [
                    row for row in catalog.terminal_metadata.get(token, [])
                    if row.get("physical_type") == dtype
                ]
            payload = {"token": token, "node_type": "terminal", "return_type": node.get("return_type"),
                       "children": []}
            if dtype:
                payload["physical_type"] = dtype
            fp = hashlib.sha1(json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
            copied["stage3_type_signature"] = {
                "db_type": "sqlserver", "source": "terminal_projection", "physical_type": dtype,
                "subtree_fingerprint": fp, "compatibility_class": "compatible" if dtype else "unknown",
            }
            projected[vid] = copied
            variants.setdefault(nid, []).append(vid)
            descriptors[vid] = copied["stage3_type_signature"]

    for nid, node in nodes.items():
        check_deadline(deadline, "SQL Server observed tuple projection")
        token = node.get("token", "")
        if node.get("is_terminal") or token not in definitions:
            continue
        records = evidence.get(nid, [])
        candidates, counts = _mysql_evidence_descriptors(
            node, records, set(), minimum_success=minimum,
            known_invalid_min_observations=failure_minimum, db_type="sqlserver",
            include_unknown=not records and unknown_ratio > 0,
            max_unknown_variants=1,
        )
        stats["blocked_combinations"] += counts["blocked_combinations"]
        kept = []
        required = definitions[token][1]
        for item in candidates:
            inputs = tuple(normalize(t) for t in item.get("slot_child_physical_types", []))
            inputs = inputs if len(inputs) == len(required) else ("",) * len(required)
            dtype = normalize(item.get("physical_type")) or fixed_result_type(token, inputs)
            unresolved = not dtype or not all(inputs)

            item.update(physical_type=dtype, slot_child_physical_types=inputs,
                        result_types={dtype} if dtype else set())
            stats["typed_observations"] += int(bool(dtype) and all(inputs))
            stats["unresolved_variants"] += int(unresolved)
            kept.append(item)
        finalized, counts = _finalize_evidence_descriptors(
            node, kept, records, nodes, max_variants=limit,
            descriptor_key=_sqlserver_merge_key, evidence_record_key=_sqlserver_merge_key,
            rank_key=lambda d: _projection_descriptor_rank(d, db_type="sqlserver"), metadata_prefix="sqlserver",
        )
        for key, value in counts.items():
            stats[key] += value
        for item in finalized:
            identity = {
                "base": nid, "children": item.get("exact_child_ids"),
                "allowed": item.get("allowed_child_ids_by_slot"),
                "inputs": item["slot_child_physical_types"], "output": item["physical_type"],

                "witness": [
                    "" if index < len(item["slot_child_physical_types"])
                    and item["slot_child_physical_types"][index] else fingerprint
                    for index, fingerprint in enumerate(item.get("slot_child_fingerprints") or [])
                ],
            }
            vid = nid + "::SS" + hashlib.sha1(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:16]
            if vid in projected:
                continue
            copied = copy.deepcopy(node)
            signature = {
                "db_type": "sqlserver", "source": item["source"],
                "compatibility_class": item["compatibility_class"], "physical_type": item["physical_type"],
                "accepted_arg_types": [[t] if t else [] for t in item["slot_child_physical_types"]],
                "slot_child_physical_types": list(item["slot_child_physical_types"]),
                "slot_child_fingerprints": list(item.get("slot_child_fingerprints") or []),
                "subtree_fingerprint": item.get("subtree_fingerprint", ""),
                "tuple_hash": item.get("tuple_hash", ""),
            }
            for field in ("success", "type_failure", "value_failure", "execution_failure"):
                signature["observed_" + field + "_count"] = item.get("observed_" + field + "_count", 0)
            signature["evidence_confidence"] = _evidence_confidence(
                signature["observed_success_count"], signature["observed_type_failure_count"],
                signature["observed_value_failure_count"], signature["observed_execution_failure_count"], db_type="sqlserver")
            copied.update(base_node_id=nid, slots=[], stage3_result_types=sorted(item["result_types"]),
                          stage3_type_signature=signature)
            projected[vid] = copied
            descriptors[vid] = item
            variants.setdefault(nid, []).append(vid)

    child_has_compatible_variant = {
        nid: any(projected[vid]["stage3_type_signature"]["compatibility_class"] != "unknown"
                 for vid in vids)
        for nid, vids in variants.items()
    }
    for nid, vids in variants.items():
        original = nodes[nid]
        if original.get("is_terminal"):
            continue
        for vid in vids:
            check_deadline(deadline, "SQL Server physical child binding")
            descriptor = descriptors[vid]
            slots = []
            for index, original_slot in enumerate(original.get("slots", [])):
                exact = descriptor.get("exact_child_ids")
                allowed = descriptor.get("allowed_child_ids_by_slot")
                ids = [exact[index]] if exact else (allowed[index] if allowed else original_slot)
                expected = descriptor["slot_child_physical_types"][index]
                witnesses = descriptor.get("slot_child_fingerprints") or []
                witness = witnesses[index] if index < len(witnesses) else ""
                children = []
                for cid in ids:
                    if cid not in original_slot:
                        continue
                    for cvid in variants.get(cid, []):
                        child = projected[cvid]["stage3_type_signature"]
                        actual = child.get("physical_type", "")
                        if expected:
                            if actual != expected:
                                continue
                        elif descriptor["source"] != "unknown_exploration":
                            if not witness or child.get("subtree_fingerprint") != witness:
                                continue
                        if (descriptor["compatibility_class"] != "unknown"
                                and child["compatibility_class"] == "unknown"
                                and not (descriptor["source"] == "observed"
                                         and not child_has_compatible_variant.get(cid, False))):
                            continue
                        children.append(cvid)
                slots.append(sorted(set(children)))
            projected[vid]["slots"] = slots

    report["sqlserver_risk_propagated_variants"] = 0
    removed = _prune_projection(projected)
    graph = {"nodes": projected, "roots": sorted({v for root in roots for v in variants.get(root, []) if v in projected})}
    report.update({"sqlserver_" + k: v for k, v in stats.items()})
    report.update(projection_removed_nodes=removed, projection_effective=True,
                  physical_type_binding=True, fallback_node_count=0,
                  evidence_only=not catalog.loaded,
                  empirical_only_variants=sum(not n.get("is_terminal") and n["stage3_type_signature"]["source"] == "observed"
                                              for n in projected.values()))
    return finish(graph, "projected" if graph["roots"] else "projected_empty")


def project_graph_for_stage3(
    graph_data: Dict[str, Any],
    *,
    source_path: Optional[Path] = None,
    evidence_path: Path = DEFAULT_EVIDENCE_PATH,
    report_path: Path = DEFAULT_REPORT_PATH,
    deadline: Optional[float] = None,
    catalog_snapshot: Optional[CatalogSnapshot] = None,
    db_type: Optional[str] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    check_deadline(deadline, "type projection initialization")
    normalized_db = str(db_type or _current_db_type()).lower().strip()
    if normalized_db == "sqlserver":
        return _project_sqlserver_graph(
            graph_data, evidence_path=evidence_path, report_path=report_path,
            deadline=deadline, catalog_snapshot=catalog_snapshot,
        )
    enabled = bool(_config_value("stage3_type_projection_enabled", True))
    catalog_enabled = bool(
        _config_value("stage3_type_projection_catalog_enabled", True)
    )
    evidence_enabled = bool(
        _config_value("stage3_type_projection_evidence_enabled", True)
    )
    minimum_success = max(
        1,
        int(_config_value("stage3_type_projection_min_success_count", 1)),
    )
    known_invalid_min_observations = max(
        1,
        int(
            _config_value(
                "stage3_type_projection_known_invalid_min_observations",
                4,
            )
        ),
    )
    unknown_exploration_ratio = min(
        1.0,
        max(
            0.0,
            float(
                _config_value(
                    "stage3_type_projection_unknown_exploration_ratio",
                    0.08,
                )
            ),
        ),
    )
    max_unknown_exploration_variants = min(
        2,
        max(
            1,
            int(
                _config_value(
                    "stage3_type_projection_max_unknown_variants",
                    1,
                )
                or 1
            ),
        ),
    )
    mysql_max_projection_variants = max(
        1,
        int(
            _config_value(
                "stage3_mysql_max_projection_variants",
                96,
            )
            or 96
        ),
    )
    oracle_max_projection_variants = max(
        1,
        int(
            _config_value(
                "stage3_oracle_max_projection_variants",
                192,
            )
            or 192
        ),
    )
    exact_column_type_guidance_enabled = bool(
        _config_value("stage3_exact_column_type_guidance_enabled", True)
    )

    cache_key = (
        _path_fingerprint(source_path),
        _path_fingerprint(evidence_path if evidence_enabled else None),
        normalized_db,
        _catalog_cache_key(normalized_db),
        enabled,
        catalog_enabled,
        evidence_enabled,
        minimum_success,
        known_invalid_min_observations,
        unknown_exploration_ratio,
        max_unknown_exploration_variants,
        mysql_max_projection_variants if normalized_db == "mysql" else None,
        (
            oracle_max_projection_variants
            if normalized_db in {"oracle", "oracledb", "oracle_ai26"}
            else None
        ),
        exact_column_type_guidance_enabled,
        id(catalog_snapshot) if catalog_snapshot is not None else None,
    )
    if cache_key in _PROJECTION_CACHE:
        projected, report = _PROJECTION_CACHE[cache_key]
        _atomic_report(report_path, report)
        return projected, report

    original_nodes = graph_data.get("nodes", {}) or {}
    original_roots = list(graph_data.get("roots", []) or [])
    report: Dict[str, Any] = {
        "version": 3,
        "updated_at": float(time.time()),
        "db_type": normalized_db,
        "enabled": enabled,
        "status": "pass_through",
        "catalog_source": "disabled",
        "catalog_loaded": False,
        "catalog_error": None,
        "evidence": {},
        "exact_signature_compatibility": False,
        "known_invalid_min_observations": known_invalid_min_observations,
        "unknown_exploration_ratio": unknown_exploration_ratio,
        "max_unknown_exploration_variants": max_unknown_exploration_variants,
        "exact_column_type_guidance_enabled": exact_column_type_guidance_enabled,
        "original_nodes": len(original_nodes),
        "original_roots": len(original_roots),
        "original_edges": _edge_count(graph_data),
    }
    if normalized_db == "mysql":
        report["mysql_max_projection_variants"] = mysql_max_projection_variants
    elif normalized_db in {"oracle", "oracledb", "oracle_ai26"}:
        report["oracle_max_projection_variants"] = oracle_max_projection_variants

    provider_available = normalized_db in _PROVIDER_REGISTRY
    if not enabled or not provider_available:
        report["status"] = (
            "disabled" if not enabled else "pass_through_no_dbms_provider"
        )
        report.update({
            "projected_nodes": len(original_nodes),
            "projected_roots": len(original_roots),
            "projected_edges": report["original_edges"],
        })
        _atomic_report(report_path, report)
        return graph_data, report

    if catalog_snapshot is not None:
        catalog = catalog_snapshot
    elif catalog_enabled:
        catalog = load_catalog_snapshot(normalized_db)
    else:
        catalog = CatalogSnapshot(
            db_type=normalized_db,
            loaded=False,
            source="disabled",
        )
    report["catalog_source"] = catalog.source
    report["catalog_loaded"] = bool(catalog.loaded)
    report["catalog_error"] = catalog.error
    report["catalog_terminal_types"] = {
        str(token): sorted(types)
        for token, types in (catalog.terminal_types or {}).items()
    }
    report["catalog_terminal_metadata_count"] = sum(
        len(items) for items in (catalog.terminal_metadata or {}).values()
    )

    report["exact_signature_compatibility"] = bool(
        catalog.loaded
        and (catalog.operator_signatures or catalog.function_records)
    )

    if evidence_enabled:
        evidence_by_outer, evidence_stats = _load_evidence(
            evidence_path,
            normalized_db,
            minimum_success,
            known_invalid_min_observations,
        )
    else:
        evidence_by_outer, evidence_stats = {}, {
            "entries": 0,
            "successful_entries": 0,
            "type_failure_entries": 0,
            "value_failure_entries": 0,
            "execution_failure_entries": 0,
            "known_invalid_entries": 0,
            "unknown_entries": 0,
        }
    report["evidence"] = evidence_stats
    if normalized_db in {"oracle", "oracledb", "oracle_ai26"}:
        from RL.oracle_evidence_types import complete_observed_types
        evidence_by_outer, report["oracle_type_completion"] = complete_observed_types(evidence_by_outer)

    evidence_driven_db = normalized_db == "mysql" or normalized_db in {
        "oracle", "oracledb", "oracle_ai26",
    }
    evidence_available = (
        int(evidence_stats.get("entries", 0)) > 0
        if evidence_driven_db
        else int(evidence_stats.get("successful_entries", 0)) > 0
    )
    if not catalog.loaded and not evidence_available:
        report["status"] = "pass_through_no_catalog_or_evidence"
        report.update({
            "projected_nodes": len(original_nodes),
            "projected_roots": len(original_roots),
            "projected_edges": report["original_edges"],
        })
        _atomic_report(report_path, report)
        return graph_data, report

    projected_nodes: Dict[str, Dict[str, Any]] = {}
    variants_by_original: Dict[str, List[str]] = {}
    result_types_by_variant: Dict[str, Set[str]] = {}
    descriptors_by_variant: Dict[str, Dict[str, Any]] = {}
    fallback_nodes: List[str] = []
    empirical_only_variants = 0
    catalog_variants = 0
    catalog_observed_variants = 0
    unknown_exploration_variants = 0
    mysql_partition_blocked_combinations = 0
    mysql_partition_observed_variants = 0
    mysql_partition_unobserved_variants = 0
    mysql_variants_before_limit = 0
    mysql_variants_after_limit = 0
    mysql_variants_pruned_by_limit = 0
    mysql_safe_merge_reductions = 0
    oracle_variants_before_limit = 0
    oracle_variants_after_limit = 0
    oracle_variants_pruned_by_limit = 0
    oracle_safe_merge_reductions = 0
    oracle_observed_variants_after_limit = 0
    oracle_unknown_variants_after_limit = 0
    oracle_catalog_incompatible_failures_pruned = 0
    oracle_mixed_type_evidence_pruned = 0
    oracle_empirical_overrides_capped = 0
    split_column_terminals = 0
    exact_column_terminal_variants = 0
    reachable_buckets = (
        _reachable_type_buckets(original_nodes, catalog)
        if catalog.loaded
        else set()
    )
    possible_result_types_by_original = {
        node_id: _possible_node_result_types(node, catalog)
        for node_id, node in original_nodes.items()
    }
    projection_fingerprint_cache: Dict[str, str] = {}

    oracle_column_evidence: Dict[str, Set[str]] = {}
    if normalized_db in {"oracle", "oracledb", "oracle_ai26"}:
        from RL.oracle_evidence_types import normalize_oracle_type
        typed_entries = 0
        for records in evidence_by_outer.values():
            for record in records:
                physical = record.get("slot_child_physical_types") or []
                typed_entries += int(any(physical))
                for child_id, dtype in zip(record.get("slot_child_node_ids") or [], physical):
                    child = original_nodes.get(child_id, {})
                    if dtype and child.get("is_terminal") and str(child.get("token", "")).startswith("COL_"):
                        oracle_column_evidence.setdefault(child_id, set()).add(normalize_oracle_type(dtype))
        report["oracle_evidence_with_physical_inputs"] = typed_entries

    for node_id, node in original_nodes.items():
        check_deadline(deadline, "terminal type projection")
        if not bool(node.get("is_terminal")):
            continue
        result_types = possible_result_types_by_original[node_id]
        live_column_types = (
            sorted(_live_column_result_types(node, catalog))
            if exact_column_type_guidance_enabled and catalog.loaded
            else []
        )
        if (normalized_db in {"oracle", "oracledb", "oracle_ai26"}
                and exact_column_type_guidance_enabled):
            live_column_types = sorted(set(live_column_types) | oracle_column_evidence.get(node_id, set()))
        if live_column_types:
            variant_ids = []
            for result_type in live_column_types:
                variant_id = _terminal_type_variant_id(node_id, result_type)
                copied = copy.deepcopy(node)
                copied["base_node_id"] = node_id
                copied["stage3_result_types"] = [result_type]

                copied["stage3_column_type"] = result_type

                terminal_fingerprint = _projection_terminal_fingerprint(
                    node_id, result_type, original_nodes, projection_fingerprint_cache
                )
                if normalized_db in {"oracle", "oracledb", "oracle_ai26"}:
                    copied["stage3_type_signature"] = {
                        "db_type": normalized_db,
                        "source": "terminal_projection",
                        "compatibility_class": "compatible",
                        "subtree_fingerprint": terminal_fingerprint,
                        "stage3_column_type": result_type,
                    }
                projected_nodes[variant_id] = copied
                variant_ids.append(variant_id)
                result_types_by_variant[variant_id] = {result_type}
                descriptors_by_variant[variant_id] = {
                    "subtree_fingerprint": terminal_fingerprint,
                    "physical_type": str(result_type or "").strip().lower(),
                    "compatibility_class": "compatible",
                }
            variants_by_original[node_id] = variant_ids
            split_column_terminals += 1
            exact_column_terminal_variants += len(variant_ids)
            continue

        copied = copy.deepcopy(node)
        copied["base_node_id"] = node_id
        copied["stage3_result_types"] = sorted(result_types)
        terminal_physical = ""
        if normalized_db in {"oracle", "oracledb", "oracle_ai26"}:
            token = str(node.get("token", "")).upper()

            if token.startswith("VALUE_") and token.endswith(("_INT", "_FLOAT")):
                terminal_physical = "number"
            elif token.startswith("VALUE_") and token.endswith("_STRING"):
                terminal_physical = "char"
            elif token in {"TRUE", "FALSE"}:
                terminal_physical = "boolean"
            elif token == "NULL":
                terminal_physical = "null"
            else:
                types = catalog.terminal_types.get(str(node.get("token", "")), set())
                if len(types) == 1:
                    terminal_physical = normalize_oracle_type(next(iter(types)))
        if normalized_db in {"oracle", "oracledb", "oracle_ai26"}:
            copied["stage3_type_signature"] = {
                "db_type": normalized_db,
                "source": "terminal_projection",
                "compatibility_class": "compatible",
                "subtree_fingerprint": _projection_base_fingerprint(
                    node_id, original_nodes, projection_fingerprint_cache
                ),
            }
        projected_nodes[node_id] = copied
        variants_by_original[node_id] = [node_id]
        result_types_by_variant[node_id] = result_types
        descriptors_by_variant[node_id] = {
            "subtree_fingerprint": _projection_base_fingerprint(
                node_id, original_nodes, projection_fingerprint_cache
            ),
            "physical_type": str(
                terminal_physical or copied.get("stage3_column_type", "") or ""
            ).strip().lower(),
            "compatibility_class": "compatible",
        }

    for node_id, node in original_nodes.items():
        check_deadline(deadline, "operator signature projection")
        if bool(node.get("is_terminal")):
            continue
        slots = list(node.get("slots", []) or [])
        token = str(node.get("token", ""))
        observed_records = evidence_by_outer.get(node_id, [])
        catalog_signatures = (
            _catalog_signatures_for_node(catalog, token, len(slots))
            if catalog.loaded
            else []
        )
        catalog_signatures = [
            signature
            for signature in catalog_signatures
            if _coarse_accepts(str(node.get("return_type", "")), signature.result_type)
        ]

        if normalized_db in {"oracle", "oracledb", "oracle_ai26"}:
            catalog_signatures = []
        oracle_catalog_has_signature = bool(catalog_signatures)
        if normalized_db in {"oracle", "oracledb", "oracle_ai26"}:
            catalog_signatures = [
                signature
                for signature in catalog_signatures
                if (
                    _normalize_token(token) in {"IS NULL", "IS NOT NULL"}
                    or not _is_unbound_wildcard_signature(signature)
                )
            ]
        oracle_has_concrete_signature = bool(catalog_signatures)
        oracle_has_wildcard_signature = bool(
            oracle_catalog_has_signature and not oracle_has_concrete_signature
        )

        signature_groups = _group_catalog_signatures(
            catalog_signatures,
            reachable_arg_buckets=reachable_buckets,

            allow_wildcard=(
                normalized_db not in {"oracle", "oracledb", "oracle_ai26"}
                or _normalize_token(token) in {"IS NULL", "IS NOT NULL"}
            ),
        )
        descriptors: List[Dict[str, Any]] = []
        oracle_empirical_candidates: List[Dict[str, Any]] = []
        represented_observations: Set[int] = set()
        oracle_catalog_matched_observations: Set[int] = set()
        oracle_evidence_mode = normalized_db in {
            "oracle", "oracledb", "oracle_ai26",
        }
        if normalized_db == "mysql":
            mysql_descriptors, mysql_partition_stats = _mysql_evidence_descriptors(
                node,
                observed_records,
                possible_result_types_by_original[node_id],
                minimum_success=minimum_success,
                known_invalid_min_observations=known_invalid_min_observations,
                max_unknown_variants=max_unknown_exploration_variants,
            )
            mysql_descriptors, mysql_variant_stats = _finalize_mysql_descriptors(
                node,
                mysql_descriptors,
                observed_records,
                original_nodes,
                max_variants=mysql_max_projection_variants,
            )
            descriptors.extend(mysql_descriptors)
            mysql_partition_blocked_combinations += int(
                mysql_partition_stats.get("blocked_combinations", 0)
            )
            mysql_partition_observed_variants += int(
                mysql_variant_stats.get("observed_variants_after_limit", 0)
            )
            mysql_partition_unobserved_variants += int(
                mysql_variant_stats.get("unknown_variants_after_limit", 0)
            )
            mysql_variants_before_limit += int(
                mysql_variant_stats.get("variants_before_limit", 0)
            )
            mysql_variants_after_limit += int(
                mysql_variant_stats.get("variants_after_limit", 0)
            )
            mysql_variants_pruned_by_limit += int(
                mysql_variant_stats.get("variants_pruned_by_limit", 0)
            )
            mysql_safe_merge_reductions += int(
                mysql_variant_stats.get("safe_merge_reductions", 0)
            )
        if normalized_db in {"oracle", "oracledb", "oracle_ai26"}:
            oracle_descriptors, oracle_evidence_stats = _mysql_evidence_descriptors(
                node,
                observed_records,
                possible_result_types_by_original[node_id],
                minimum_success=minimum_success,
                known_invalid_min_observations=known_invalid_min_observations,
                max_unknown_variants=max_unknown_exploration_variants,
                db_type=normalized_db,

                include_unknown=not bool(observed_records),
            )
            descriptors.extend(oracle_descriptors)
            oracle_observed_variants_after_limit += int(
                oracle_evidence_stats.get("evidence_variants", 0)
            )
            oracle_unknown_variants_after_limit += int(
                oracle_evidence_stats.get("unobserved_variants", 0)
            )
        for signature_group in (
            () if normalized_db in {"mysql", "oracle", "oracledb", "oracle_ai26"}
            else signature_groups
        ):
            observed_support = 0
            observed_type_failures = 0
            observed_value_failures = 0
            observed_execution_failures = 0
            oracle_known_invalid = False
            oracle_matched_records: List[Dict[str, Any]] = []
            oracle_matched_success_records: List[Dict[str, Any]] = []
            for evidence_index, record in enumerate(observed_records):
                child_ids = list(record.get("slot_child_node_ids", []) or [])
                child_return_types = list(_record_arg_types(record))
                if len(child_ids) != len(signature_group["arg_type_options"]):
                    continue
                record_matches = True
                for slot_index, (child_id, expected_options) in enumerate(zip(
                    child_ids,
                    signature_group["arg_type_options"],
                )):
                    if child_id not in original_nodes:
                        record_matches = False
                        break
                    child_result_types = set(
                        possible_result_types_by_original[child_id]
                    )
                    if slot_index < len(child_return_types):
                        coarse_child_type = child_return_types[slot_index]
                        child_result_types = {
                            result_type
                            for result_type in child_result_types
                            if result_type == _WILDCARD_TYPE
                            or _coarse_accepts(coarse_child_type, result_type)
                            }
                    child_is_oracle_null = (
                        oracle_evidence_mode
                        and _normalize_token(
                            (original_nodes.get(child_id) or {}).get("token", "")
                        ) == "NULL"
                    )
                    if not any(
                        (
                            _normalize_token(token) in {"IS NULL", "IS NOT NULL"}
                            and _normalize_type(expected_type) == _WILDCARD_TYPE
                        )
                        or (child_is_oracle_null and _normalize_type(expected_type) != _WILDCARD_TYPE)
                        or _compatible_result_set(
                            catalog,
                            child_result_types,
                            expected_type,
                        )
                        for expected_type in expected_options
                    ):
                        record_matches = False
                        break

                if record_matches:
                    if oracle_evidence_mode:
                        oracle_catalog_matched_observations.add(evidence_index)
                    success_count = int(record.get("success_count", 0) or 0)
                    evidence_class = _evidence_compatibility_class(
                        record,
                        minimum_success=minimum_success,
                        known_invalid_min_observations=known_invalid_min_observations,
                        db_type=normalized_db,
                    )
                    has_observations = any(
                        int(record.get(field, 0) or 0) > 0
                        for field in (
                            "success_count",
                            "type_failure_count",
                            "value_failure_count",
                            "execution_failure_count",
                        )
                    )
                    if success_count >= minimum_success or (
                        oracle_evidence_mode and has_observations
                    ):
                        represented_observations.add(evidence_index)
                    if oracle_evidence_mode and evidence_class == "known_invalid":
                        oracle_known_invalid = True
                    observed_support += success_count
                    observed_type_failures += int(
                        record.get("type_failure_count", 0) or 0
                    )
                    observed_value_failures += int(
                        record.get("value_failure_count", 0) or 0
                    )
                    observed_execution_failures += int(
                        record.get("execution_failure_count", 0) or 0
                    )
                    if (
                        oracle_evidence_mode
                        and success_count >= minimum_success
                        and evidence_class != "known_invalid"
                        and len(child_ids) == len(slots)
                    ):
                        oracle_matched_success_records.append(record)

                    if (
                        oracle_evidence_mode
                        and has_observations
                        and evidence_class != "known_invalid"
                        and len(child_ids) == len(slots)
                    ):
                        oracle_matched_records.append(record)
            if (
                oracle_evidence_mode
                and observed_support < minimum_success
                and oracle_known_invalid
            ):
                continue
            if (
                oracle_evidence_mode
                and observed_support <= 0
                and observed_type_failures > 0
            ):
                continue
            if oracle_evidence_mode and oracle_matched_records:
                seen_exact: Set[Tuple[str, ...]] = set()
                for record in oracle_matched_records:
                    exact_child_ids = tuple(
                        str(item) for item in (record.get("slot_child_node_ids", []) or [])
                    )
                    if exact_child_ids in seen_exact:
                        continue
                    seen_exact.add(exact_child_ids)
                    record_class = _evidence_compatibility_class(
                        record,
                        minimum_success=minimum_success,
                        known_invalid_min_observations=known_invalid_min_observations,
                        db_type=normalized_db,
                    )
                    descriptors.append({
                        "source": "both",
                        "compatibility_class": (
                            "compatible"
                            if record_class == "compatible"
                            else "weak_observed"
                        ),
                        "arg_types": tuple(signature_group["arg_types"]),
                        "arg_type_options": tuple(
                            set(options)
                            for options in signature_group["arg_type_options"]
                        ),
                        "result_types": set(signature_group["result_types"]),
                        "exact_child_ids": exact_child_ids,
                        "observed_child_return_types": _record_arg_types(record),
                        "observed_success_count": int(record.get("success_count", 0) or 0),
                        "observed_type_failure_count": int(record.get("type_failure_count", 0) or 0),
                        "observed_value_failure_count": int(record.get("value_failure_count", 0) or 0),
                        "observed_execution_failure_count": int(record.get("execution_failure_count", 0) or 0),
                    })
            else:
                descriptors.append({
                    "source": "both" if observed_support > 0 else "catalog",
                    "compatibility_class": "compatible",
                    "arg_types": tuple(signature_group["arg_types"]),
                    "arg_type_options": tuple(
                        set(options) for options in signature_group["arg_type_options"]
                    ),
                    "result_types": set(signature_group["result_types"]),
                    "exact_child_ids": None,
                    "observed_child_return_types": None,
                    "observed_success_count": observed_support,
                    "observed_type_failure_count": observed_type_failures,
                    "observed_value_failure_count": observed_value_failures,
                    "observed_execution_failure_count": observed_execution_failures,
                })

        for evidence_index, record in enumerate(observed_records):
            if normalized_db in {"mysql", "oracle", "oracledb", "oracle_ai26"}:
                break
            if evidence_index in represented_observations:
                continue
            if oracle_evidence_mode and catalog.loaded:
                if oracle_has_concrete_signature and evidence_index not in oracle_catalog_matched_observations:
                    oracle_catalog_incompatible_failures_pruned += 1
                    continue
                success_count = int(record.get("success_count", 0) or 0)
                type_failure_count = int(record.get("type_failure_count", 0) or 0)
                if (
                    evidence_index not in oracle_catalog_matched_observations
                    and success_count < minimum_success
                    and type_failure_count > 0
                ):
                    oracle_catalog_incompatible_failures_pruned += 1
                    continue
                if (
                    evidence_index not in oracle_catalog_matched_observations
                    and success_count > 0
                    and type_failure_count > 0
                    and not _oracle_empirical_override_allowed(
                        record,
                        minimum_success=minimum_success,
                    )
                ):
                    oracle_mixed_type_evidence_pruned += 1
                    continue
            evidence_class = _evidence_compatibility_class(
                record,
                minimum_success=minimum_success,
                known_invalid_min_observations=known_invalid_min_observations,
                db_type=normalized_db,
            )
            if evidence_class == "known_invalid":
                continue
            if evidence_class != "compatible" and not oracle_evidence_mode:
                continue
            child_ids = tuple(record.get("slot_child_node_ids", []) or [])
            if len(child_ids) != len(slots):
                continue
            empirical_result_types = set(possible_result_types_by_original[node_id])
            if oracle_evidence_mode:
                inferred_result_types = _oracle_observed_result_types(
                    token,
                    child_ids,
                    possible_result_types_by_original,
                )
                if inferred_result_types:
                    empirical_result_types = inferred_result_types
                elif _normalize_token(token).removesuffix("()") in {
                    "NVL", "COALESCE", "NULLIF", "NVL2", "DECODE",
                }:
                    continue
                elif _oracle_single_result_family(empirical_result_types) is None:
                    continue
            empirical_descriptor = {
                "source": "observed",
                "compatibility_class": (
                    "weak_observed"
                    if oracle_evidence_mode and evidence_class == "unknown"
                    else "compatible"
                ),
                "arg_types": _record_arg_types(record),
                "arg_type_options": None,
                "result_types": empirical_result_types,
                "exact_child_ids": child_ids,
                "observed_child_return_types": _record_arg_types(record),
                "observed_success_count": int(record.get("success_count", 0) or 0),
                "observed_type_failure_count": int(
                    record.get("type_failure_count", 0) or 0
                ),
                "observed_value_failure_count": int(
                    record.get("value_failure_count", 0) or 0
                ),
                "observed_execution_failure_count": int(
                    record.get("execution_failure_count", 0) or 0
                ),
            }
            if oracle_evidence_mode and catalog.loaded:
                oracle_empirical_candidates.append(empirical_descriptor)
            else:
                descriptors.append(empirical_descriptor)

        if oracle_evidence_mode and catalog.loaded and oracle_empirical_candidates:
            oracle_empirical_candidates.sort(key=_oracle_descriptor_rank)
            retained_empirical = oracle_empirical_candidates[
                :_ORACLE_MAX_EMPIRICAL_OVERRIDES_PER_NODE
            ]
            oracle_empirical_overrides_capped += max(
                0,
                len(oracle_empirical_candidates) - len(retained_empirical),
            )
            descriptors.extend(retained_empirical)

        if (
            normalized_db not in {"mysql", "oracle", "oracledb", "oracle_ai26"}
            and unknown_exploration_ratio > 0.0
            and not (
                oracle_evidence_mode
                and catalog.loaded
                and (oracle_has_concrete_signature or oracle_has_wildcard_signature)
            )
        ):
            observed_tuples: Set[Tuple[str, ...]] = set()
            for record in observed_records:
                child_ids = tuple(
                    str(item)
                    for item in (record.get("slot_child_node_ids", []) or [])
                )
                if len(child_ids) != len(slots):
                    continue
                if any(
                    child_id not in original_slot
                    for child_id, original_slot in zip(child_ids, slots)
                ):
                    continue
                observed_tuples.add(child_ids)

            for allowed_child_ids_by_slot in _partition_slot_product(
                slots,
                observed_tuples,
                max_rectangles=max_unknown_exploration_variants,
            ):
                if len(allowed_child_ids_by_slot) != len(slots):
                    continue
                descriptors.append({
                    "source": "unknown_exploration",
                    "compatibility_class": "unknown",
                    "arg_types": None,
                    "arg_type_options": None,
                    "result_types": set(
                        possible_result_types_by_original[node_id]
                    ),
                    "exact_child_ids": None,
                    "allowed_child_ids_by_slot": allowed_child_ids_by_slot,
                    "observed_child_return_types": None,
                    "observed_success_count": 0,
                    "observed_type_failure_count": 0,
                    "observed_execution_failure_count": 0,
                    "observed_value_failure_count": 0,
                })

        if not descriptors:
            if catalog.loaded:
                continue
            descriptors.append({
                "source": "fallback",
                "compatibility_class": "unknown",
                "arg_types": None,
                "arg_type_options": None,
                "result_types": _fallback_result_types(node),
                "exact_child_ids": None,
                "observed_child_return_types": None,
                "observed_success_count": 0,
                "observed_type_failure_count": 0,
                "observed_value_failure_count": 0,
                "observed_execution_failure_count": 0,
            })
            fallback_nodes.append(node_id)

        if oracle_evidence_mode:
            for descriptor in descriptors:
                descriptor["oracle_evidence"] = True
                original_identity = descriptor.get("oracle_identity_before_enrichment")
                if original_identity is not None:
                    old_type = original_identity.get("physical_type")
                    descriptor["oracle_identity_result_types"] = (
                        {normalize_oracle_type(old_type)} if old_type else set(descriptor["result_types"])
                    )
                physical = descriptor.get("physical_type")
                if physical:
                    descriptor["result_types"] = {normalize_oracle_type(physical)}

            descriptors, oracle_variant_stats = _finalize_oracle_descriptors(
                node,
                descriptors,
                observed_records,
                original_nodes,
                max_variants=oracle_max_projection_variants,
            )
            oracle_variants_before_limit += int(
                oracle_variant_stats.get("variants_before_limit", 0)
            )
            oracle_variants_after_limit += int(
                oracle_variant_stats.get("variants_after_limit", 0)
            )
            oracle_variants_pruned_by_limit += int(
                oracle_variant_stats.get("variants_pruned_by_limit", 0)
            )
            oracle_safe_merge_reductions += int(
                oracle_variant_stats.get("safe_merge_reductions", 0)
            )
            oracle_observed_variants_after_limit += int(
                oracle_variant_stats.get("observed_variants_after_limit", 0)
            )
            oracle_unknown_variants_after_limit += int(
                oracle_variant_stats.get("unknown_variants_after_limit", 0)
            )

        for descriptor in descriptors:
            variant_id = _variant_id(node_id, descriptor)
            variant_node = copy.deepcopy(node)
            variant_node["base_node_id"] = node_id
            variant_node["stage3_result_types"] = sorted(descriptor["result_types"])
            variant_node["stage3_type_signature"] = {
                "db_type": normalized_db,
                "source": descriptor["source"],
                "compatibility_class": descriptor["compatibility_class"],
                "arg_types": list(descriptor.get("arg_types") or []),
                "accepted_arg_types": [
                    sorted(options)
                    for options in (descriptor.get("arg_type_options") or [])
                ],
                "tuple_hash": str(descriptor.get("tuple_hash", "") or ""),
                "subtree_fingerprint": str(
                    descriptor.get("subtree_fingerprint", "") or ""
                ),
                "slot_child_fingerprints": list(
                    descriptor.get("slot_child_fingerprints", []) or []
                ),
                "slot_child_physical_types": list(
                    descriptor.get("slot_child_physical_types", []) or []
                ),
                "physical_type": str(descriptor.get("physical_type", "") or ""),
                **({"oracle_binding_version": int(descriptor.get("oracle_binding_version", 0)),
                    "observed_direct_success_count": int(descriptor.get("observed_direct_success_count", 0)),
                    "observed_context_success_count": int(descriptor.get("observed_context_success_count", 0))}
                   if oracle_evidence_mode else {}),
                "observed_success_count": int(
                    descriptor.get("observed_success_count", 0)
                ),
                "observed_type_failure_count": int(
                    descriptor.get("observed_type_failure_count", 0)
                ),
                "observed_value_failure_count": int(
                    descriptor.get("observed_value_failure_count", 0)
                ),
                "observed_execution_failure_count": int(
                    descriptor.get("observed_execution_failure_count", 0)
                ),
            }
            variant_node["stage3_type_signature"]["evidence_confidence"] = (
                _evidence_confidence(
                    variant_node["stage3_type_signature"]["observed_success_count"],
                    variant_node["stage3_type_signature"]["observed_type_failure_count"],
                    variant_node["stage3_type_signature"]["observed_value_failure_count"],
                    variant_node["stage3_type_signature"]["observed_execution_failure_count"],
                    db_type=normalized_db,
                )
            )
            variant_node["slots"] = []
            projected_nodes[variant_id] = variant_node
            variants_by_original.setdefault(node_id, []).append(variant_id)
            result_types_by_variant[variant_id] = set(descriptor["result_types"])
            descriptors_by_variant[variant_id] = descriptor
            if descriptor["source"] == "observed":
                empirical_only_variants += 1
            elif descriptor["source"] == "both":
                catalog_observed_variants += 1
            elif descriptor["source"] == "catalog":
                catalog_variants += 1
            elif descriptor["source"] == "unknown_exploration":
                unknown_exploration_variants += 1

    child_has_compatible_variant = {
        child_original_id: any(
            descriptors_by_variant.get(child_variant_id, {}).get(
                "compatibility_class",
                "compatible",
            )
            != "unknown"
            for child_variant_id in child_variant_ids
        )
        for child_original_id, child_variant_ids in variants_by_original.items()
    }
    compatible_result_cache: Dict[Tuple[str, str], bool] = {}

    for original_id, variant_ids in list(variants_by_original.items()):
        original_node = original_nodes.get(original_id, {})
        if bool(original_node.get("is_terminal")):
            continue
        original_slots = list(original_node.get("slots", []) or [])
        for variant_id in variant_ids:
            check_deadline(deadline, "projected edge filtering")
            descriptor = descriptors_by_variant[variant_id]
            compatibility_class = descriptor.get(
                "compatibility_class",
                "compatible",
            )
            expected_type_options = descriptor.get("arg_type_options")
            exact_child_ids = descriptor.get("exact_child_ids")
            observed_child_return_types = descriptor.get(
                "observed_child_return_types"
            )
            projected_slots: List[List[str]] = []
            for slot_index, original_slot in enumerate(original_slots):
                allowed_original_ids: Sequence[str]
                if exact_child_ids is not None:
                    allowed_original_ids = [exact_child_ids[slot_index]]
                elif descriptor.get("allowed_child_ids_by_slot") is not None:
                    allowed_original_ids = list(
                        descriptor["allowed_child_ids_by_slot"][slot_index]
                    )
                else:
                    allowed_original_ids = list(original_slot)

                projected_children: List[str] = []
                for child_original_id in allowed_original_ids:
                    if child_original_id not in original_slot:
                        continue
                    child_variant_ids = variants_by_original.get(child_original_id, [])
                    has_compatible_child = child_has_compatible_variant.get(
                        child_original_id,
                        False,
                    )
                    for child_variant_id in child_variant_ids:
                        child_descriptor = descriptors_by_variant.get(child_variant_id)
                        physical_binding_match = False
                        expected_child_fingerprints = descriptor.get(
                            "slot_child_fingerprints"
                        )
                        if (
                            normalized_db in {"oracle", "oracledb", "oracle_ai26"}
                            and expected_child_fingerprints
                            and slot_index < len(expected_child_fingerprints)
                        ):
                            expected_fingerprint = str(
                                expected_child_fingerprints[slot_index] or ""
                            )
                            actual_fingerprint = str(
                                (child_descriptor or {}).get(
                                    "subtree_fingerprint", ""
                                )
                                or ""
                            )

                            if (
                                normalized_db in {"oracle", "oracledb", "oracle_ai26"}
                                and not actual_fingerprint
                            ):
                                actual_fingerprint = _projection_base_fingerprint(
                                    child_original_id,
                                    original_nodes,
                                    projection_fingerprint_cache,
                                )
                            expected_physical_types = descriptor.get(
                                "slot_child_physical_types"
                            )
                            if (
                                expected_physical_types
                                and slot_index < len(expected_physical_types)
                            ):
                                expected_physical_type = str(
                                    expected_physical_types[slot_index] or ""
                                ).strip().lower()
                                actual_physical_type = str(
                                    (child_descriptor or {}).get("physical_type", "")
                                    or ""
                                ).strip().lower()
                                if expected_physical_type and (
                                    not actual_physical_type
                                    or actual_physical_type != expected_physical_type
                                ):
                                    continue
                                if expected_physical_type and actual_physical_type == expected_physical_type:
                                    physical_binding_match = True
                            if not expected_fingerprint:
                                continue
                            local_type_match = (
                                descriptor.get("oracle_binding_version") == 1
                                and physical_binding_match
                            )
                            if not local_type_match and actual_fingerprint != expected_fingerprint:
                                has_physical_binding = bool(
                                    expected_physical_types
                                    and any(
                                        str(value or "").strip()
                                        for value in expected_physical_types
                                    )
                                )
                                if has_physical_binding or (
                                    descriptor.get("oracle_binding_version") == 1
                                    and not original_nodes.get(child_original_id, {}).get("is_terminal")
                                ):
                                    continue
                                base_fingerprint = _projection_base_fingerprint(
                                    child_original_id,
                                    original_nodes,
                                    projection_fingerprint_cache,
                                )
                                if base_fingerprint != expected_fingerprint:
                                    continue
                        child_compatibility_class = (
                            child_descriptor.get("compatibility_class", "compatible")
                            if child_descriptor is not None
                            else "compatible"
                        )
                        if (
                            compatibility_class != "unknown"
                            and child_compatibility_class == "unknown"
                            and not (
                                descriptor.get("source") == "observed"
                                and not has_compatible_child
                            )
                        ):
                            continue

                        if physical_binding_match:
                            projected_children.append(child_variant_id)
                            continue
                        if expected_type_options is not None:
                            compatible = False
                            for expected_type in expected_type_options[slot_index]:
                                if (
                                    _normalize_token(token)
                                    in {"IS NULL", "IS NOT NULL"}
                                    and _normalize_type(expected_type) == _WILDCARD_TYPE
                                ):
                                    compatible = True
                                    break
                                if (
                                    normalized_db in {"oracle", "oracledb", "oracle_ai26"}
                                    and _normalize_token(
                                        (original_nodes.get(child_original_id) or {}).get(
                                            "token", ""
                                        )
                                    ) == "NULL"
                                ):
                                    compatible = True
                                    break
                                result_cache_key = (child_variant_id, str(expected_type))
                                cached_result = compatible_result_cache.get(
                                    result_cache_key
                                )
                                if cached_result is None:
                                    candidate_result_types = result_types_by_variant.get(
                                        child_variant_id,
                                        {_WILDCARD_TYPE},
                                    )
                                    if normalized_db in {
                                        "oracle", "oracledb", "oracle_ai26"
                                    } and _normalize_token(
                                        (original_nodes.get(child_original_id) or {}).get(
                                            "token", ""
                                        )
                                    ) != "NULL":
                                        candidate_result_types = {
                                            item
                                            for item in candidate_result_types
                                            if _normalize_type(item) != _WILDCARD_TYPE
                                        }
                                    cached_result = _compatible_result_set(
                                        catalog,
                                        candidate_result_types,
                                        expected_type,
                                    )
                                    compatible_result_cache[result_cache_key] = bool(
                                        cached_result
                                    )
                                if cached_result:
                                    compatible = True
                                    break
                            if not compatible:
                                continue
                        elif (
                            observed_child_return_types is not None
                            and slot_index < len(observed_child_return_types)
                        ):
                            coarse_child_type = observed_child_return_types[slot_index]
                            child_is_oracle_null = (
                                normalized_db in {"oracle", "oracledb", "oracle_ai26"}
                                and _normalize_token(
                                    (original_nodes.get(child_original_id) or {}).get(
                                        "token", ""
                                    )
                                ) == "NULL"
                            )
                            compatible = any(
                                (
                                    result_type == _WILDCARD_TYPE
                                    and (
                                        normalized_db not in {
                                            "oracle", "oracledb", "oracle_ai26"
                                        }
                                        or child_is_oracle_null
                                    )
                                )
                                or _coarse_accepts(coarse_child_type, result_type)
                                for result_type in result_types_by_variant.get(
                                    child_variant_id,
                                    {_WILDCARD_TYPE},
                                )
                            )
                            if not compatible:
                                continue
                        projected_children.append(child_variant_id)
                projected_slots.append(sorted(set(projected_children)))
            projected_nodes[variant_id]["slots"] = projected_slots

    merged_variants = 0
    removed_nodes = _prune_projection(projected_nodes)
    projected_roots = sorted({
        variant_id
        for root_id in original_roots
        for variant_id in variants_by_original.get(root_id, [])
        if variant_id in projected_nodes
    })
    projected_graph = {"roots": projected_roots, "nodes": projected_nodes}

    if not projected_roots:
        if normalized_db in {"oracle", "oracledb", "oracle_ai26"}:
            report["status"] = "projected_empty"
            report.update({
                "projected_nodes": 0,
                "projected_roots": 0,
                "projected_edges": 0,
                "projection_removed_nodes": removed_nodes,
            })
            _atomic_report(report_path, report)
            return {"roots": [], "nodes": {}}, report
        report["status"] = "pass_through_empty_projection_guard"
        report.update({
            "projected_nodes": len(original_nodes),
            "projected_roots": len(original_roots),
            "projected_edges": report["original_edges"],
            "projection_removed_nodes": removed_nodes,
        })
        _atomic_report(report_path, report)
        return graph_data, report

    report.update({
        "status": "projected",
        "projected_nodes": len(projected_nodes),
        "projected_roots": len(projected_roots),
        "projected_edges": _edge_count(projected_graph),
        "catalog_only_variants": catalog_variants,
        "catalog_and_observed_variants": catalog_observed_variants,
        "empirical_only_variants": empirical_only_variants,
        "unknown_exploration_variants": unknown_exploration_variants,
        "split_column_terminals": split_column_terminals,
        "exact_column_terminal_variants": exact_column_terminal_variants,
        "reachable_type_buckets": sorted(reachable_buckets),
        "fallback_node_count": len(fallback_nodes),
        "fallback_node_sample": fallback_nodes[:20],
        "equivalent_variants_merged": merged_variants,
        "projection_removed_nodes": removed_nodes,
    })
    if normalized_db == "mysql":
        report.update({
            "mysql_partition_blocked_combinations": mysql_partition_blocked_combinations,
            "mysql_partition_observed_variants": mysql_partition_observed_variants,
            "mysql_partition_unobserved_variants": mysql_partition_unobserved_variants,
            "mysql_variants_before_limit": mysql_variants_before_limit,
            "mysql_variants_after_limit": mysql_variants_after_limit,
            "mysql_variants_pruned_by_limit": mysql_variants_pruned_by_limit,
            "mysql_safe_merge_reductions": mysql_safe_merge_reductions,
        })
    elif normalized_db in {"oracle", "oracledb", "oracle_ai26"}:
        report.update({
            "oracle_variants_before_limit": oracle_variants_before_limit,
            "oracle_variants_after_limit": oracle_variants_after_limit,
            "oracle_variants_pruned_by_limit": oracle_variants_pruned_by_limit,
            "oracle_safe_merge_reductions": oracle_safe_merge_reductions,
            "oracle_observed_variants_after_limit": (
                oracle_observed_variants_after_limit
            ),
            "oracle_unknown_variants_after_limit": (
                oracle_unknown_variants_after_limit
            ),
            "oracle_catalog_incompatible_failures_pruned": (
                oracle_catalog_incompatible_failures_pruned
            ),
            "oracle_mixed_type_evidence_pruned": oracle_mixed_type_evidence_pruned,
            "oracle_empirical_overrides_capped": oracle_empirical_overrides_capped,
        })
    _atomic_report(report_path, report)
    _PROJECTION_CACHE[cache_key] = (projected_graph, report)
    return projected_graph, report
