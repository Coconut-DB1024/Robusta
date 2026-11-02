"""Preserve seeded WHERE column choices after removing unused UPDATE generation.

The former generator consumed random draws even when data changes were disabled.
Replay only those draws; this module never constructs SQL or connects to a DB.
"""
import random
import re
from datetime import date
from typing import Any, Dict, Optional

from BugFinding.db_initializer import RandomDataFactory
from RL.config import DB_CONFIG
from RL.sqlserver_evidence_types import BINARY, NUMERIC, STRINGS, TEMPORAL, base_type, column_type

def _value_category(column: Dict[str, Any]) -> str:
    """Map a live table column to a category understood by RandomDataFactory."""
    col_name = str(column.get("name", "") or "").lower()
    name_categories = (
        ("col_integer_", "INTEGER"),
        ("col_float_", "FLOAT"),
        ("col_string_", "STRING"),
        ("col_datetime_", "DATETIME"),
        ("col_binary_", "BINARY"),
        ("col_json_", "JSON"),
    )
    for prefix, category in name_categories:
        if col_name.startswith(prefix):
            return category

    # Column names are authoritative for the generated wide table. Real SQL
    # types provide a DBMS-neutral fallback for manually deployed schemas.
    col_type = str(column.get("type", "") or "").upper()
    numeric_match = re.search(
        r"\b(?:NUMERIC|DECIMAL)\s*\(\s*\d+\s*,\s*(\d+)\s*\)",
        col_type,
    )
    if numeric_match:
        return "INTEGER" if int(numeric_match.group(1)) == 0 else "FLOAT"
    if "JSON" in col_type:
        return "JSON"
    if any(token in col_type for token in ("BYTEA", "BINARY", "BLOB")):
        return "BINARY"
    if any(token in col_type for token in ("DATE", "TIME")):
        return "DATETIME"
    if any(token in col_type for token in ("FLOAT", "REAL", "DOUBLE", "NUMERIC", "DECIMAL")):
        return "FLOAT"
    if any(token in col_type for token in ("INT", "SERIAL", "BOOL", "BIT", "YEAR")):
        return "INTEGER"
    return "STRING"




def _oracle_column_kind(column: Dict[str, Any]) -> str:
    """Return a historical random-draw category for an Oracle column."""
    col_type = " ".join(str(column.get("type", "") or "").upper().split())

    if "BOOLEAN" in col_type or col_type == "BOOL":
        return "BOOLEAN"
    if "XMLTYPE" in col_type:
        return "XML"
    if "VECTOR" in col_type:
        return "VECTOR"
    if col_type == "JSON" or col_type.startswith("JSON "):
        return "JSON"
    if "BLOB" in col_type or "RAW" in col_type:
        return "BINARY"
    if any(token in col_type for token in ("DATE", "TIMESTAMP", "INTERVAL")):
        return "DATETIME"
    if any(token in col_type for token in ("BINARY_FLOAT", "BINARY_DOUBLE", "FLOAT", "REAL")):
        return "FLOAT"
    if any(token in col_type for token in ("NUMBER", "INTEGER", "INT", "DECIMAL", "NUMERIC", "SMALLINT", "BIGINT")):
        return "INTEGER"
    return "STRING"


def _oracle_condition_kind(column: Dict[str, Any]) -> Optional[str]:
    """Return a safe Oracle WHERE condition kind, or None for unsupported types."""
    col_type = " ".join(str(column.get("type", "") or "").upper().split())
    if any(token in col_type for token in ("BLOB", "CLOB", "NCLOB", "JSON", "XMLTYPE", "VECTOR")):
        return None
    if "BOOLEAN" in col_type or col_type == "BOOL":
        return "BOOLEAN"
    if "RAW" in col_type or "ROWID" in col_type or "UROWID" in col_type:
        return "BINARY"
    if any(token in col_type for token in ("DATE", "TIMESTAMP", "INTERVAL")):
        return "DATETIME"
    if any(token in col_type for token in ("BINARY_FLOAT", "BINARY_DOUBLE", "FLOAT", "REAL")):
        return "FLOAT"
    if any(token in col_type for token in ("NUMBER", "INTEGER", "INT", "DECIMAL", "NUMERIC", "SMALLINT", "BIGINT")):
        return "NUMERIC"
    if any(token in col_type for token in ("CHAR", "VARCHAR", "VARCHAR2", "NCHAR", "NVARCHAR2")):
        return "STRING"
    return None


class _SQLServerDraws:
    SUPPORTED = NUMERIC | STRINGS | BINARY | TEMPORAL | {"uniqueidentifier"}
    INTEGER_RANGES = {
        "bit": (0, 1), "tinyint": (0, 255), "smallint": (-32768, 32767),
        "int": (-(2**31), 2**31 - 1), "bigint": (-(2**63), 2**63 - 1),
    }

    @staticmethod
    def _metadata(column):
        return {str(key).lower(): value for key, value in column.items()}

    @staticmethod
    def _flag(value):
        return str(value).strip().lower() in {"1", "true", "yes"}

    @classmethod
    def is_nullable(cls, column):
        row = cls._metadata(column)
        # Missing nullability is not permission to assign NULL.
        return cls._flag(row.get("is_nullable", row.get("null")))

    @classmethod
    def _type_spec(cls, column):
        dtype = column_type(column)
        base = base_type(dtype)
        if base not in cls.SUPPORTED:
            raise ValueError(f"Unsupported SQL Server random-draw type: {dtype or column.get('type')!r}")
        match = re.search(r"\(([^)]+)\)$", dtype)
        args = match.group(1).split(",") if match else []
        if base in {"char", "varchar", "nchar", "nvarchar", "binary", "varbinary"}:
            width = args[0] if args else "1"
            if width == "max" and base in {"varchar", "nvarchar", "varbinary"}:
                return base, (32,)  # Keep random-draw values bounded; this is not the column's capacity.
            width = int(width)
            limit = 4000 if base in {"nchar", "nvarchar"} else 8000
            if not 1 <= width <= limit:
                raise ValueError(f"Invalid SQL Server column width: {dtype}")
            return base, (min(width, 32),)
        if base == "decimal":
            precision, scale = (int(args[0]), int(args[1]) if len(args) > 1 else 0) if args else (18, 0)
            if not 1 <= precision <= 38 or not 0 <= scale <= precision:
                raise ValueError(f"Invalid SQL Server decimal declaration: {dtype}")
            return base, (precision, scale)
        if base in {"time", "datetime2", "datetimeoffset"}:
            scale = int(args[0]) if args else 7
            if not 0 <= scale <= 7:
                raise ValueError(f"Invalid SQL Server temporal scale: {dtype}")
            return base, (scale,)
        if base == "float":
            precision = int(args[0]) if args else int(cls._metadata(column).get("precision") or 53)
            if not 1 <= precision <= 53:
                raise ValueError(f"Invalid SQL Server float precision: {dtype}")
            base = "real" if precision <= 24 else "float"
        return base, ()

    @classmethod
    def can_generate(cls, column):
        try:
            cls._type_spec(column)
            return True
        except (ValueError, TypeError, IndexError):
            return False

    @classmethod
    def eligible_target(cls, column):
        row = cls._metadata(column)
        name = str(row.get("name") or row.get("field") or "")
        if not name or name.lower() == "id" or not cls.can_generate(column):
            return False
        if base_type(column_type(column)) == "rowversion":
            return False
        if any(cls._flag(row.get(key)) for key in ("is_identity", "is_computed", "is_hidden")):
            return False
        # A physical nvarchar type does not reveal an ISJSON CHECK (or another
        # value constraint). Until a constraint-specific generator exists, exclude
        # constrained columns from the historical target draw.
        if cls._flag(row.get("has_check_constraint")):
            return False
        if int(row.get("generated_always_type") or 0) != 0:
            return False
        return "is_updatable" not in row or cls._flag(row["is_updatable"])


def _oracle_value_draws(category, column_type):
    if random.random() < 0.3:
        return True
    category = str(category or "").upper()
    dtype = " ".join(str(column_type or "").upper().split())
    if category == "BOOLEAN" or "BOOLEAN" in dtype:
        random.choice(range(2))
    elif category in {"XML", "VECTOR"} or "XMLTYPE" in dtype or "VECTOR" in dtype:
        pass
    elif category == "JSON" or dtype == "JSON":
        random.randint(1, 100); random.random()
    elif category == "BINARY":
        for _ in range(16): random.choice(range(16))
    elif "INTERVAL YEAR" in dtype:
        random.randint(0, 10); random.randint(0, 11)
    elif "INTERVAL DAY" in dtype:
        random.randint(0, 30); random.randint(0, 23)
        random.randint(0, 59); random.randint(0, 59)
    elif category == "DATETIME" or any(t in dtype for t in ("DATE", "TIMESTAMP")):
        random.random()
        if "WITH TIME ZONE" in dtype: random.choice(range(6))
    elif category in {"FLOAT", "REAL", "BINARY_FLOAT", "BINARY_DOUBLE"} or any(t in dtype for t in ("BINARY_FLOAT", "BINARY_DOUBLE", "FLOAT", "REAL")):
        random.uniform(-10000.0, 10000.0)
    elif category in {"INTEGER", "NUMERIC"} or any(t in dtype for t in ("NUMBER", "INTEGER", "SMALLINT", "BIGINT")):
        random.randint(-10000, 10000)
    elif not any(t in dtype for t in ("NCHAR", "NVARCHAR2", "NCLOB")):
        for _ in range(random.randint(1, 50)): random.choice(range(70))
    return False


def _sqlserver_value_draws(column):
    base, args = _SQLServerDraws._type_spec(column)
    if _SQLServerDraws.is_nullable(column) and random.random() < 0.3:
        return True
    if base in _SQLServerDraws.INTEGER_RANGES or base in {"decimal", "money", "smallmoney"}:
        if base in _SQLServerDraws.INTEGER_RANGES:
            lower, upper = _SQLServerDraws.INTEGER_RANGES[base]
        elif base == "decimal":
            upper = 10**args[0] - 1; lower = -upper
        else:
            bits = 64 if base == "money" else 32
            lower, upper = -(2**(bits-1)), 2**(bits-1)-1
        random.randint(lower, upper); random.choice(range(6))
    elif base in {"real", "float"}:
        random.uniform(-1000000, 1000000); random.choice(range(7))
    elif base in STRINGS:
        width = args[0] if args else 32
        size = random.choice([0, 1, width, random.randint(0, width)])
        # random.choices consumes one random() per character, regardless of alphabet.
        for _ in range(size): random.random()
    elif base in BINARY:
        width = 8 if base == "rowversion" else (args[0] if args else 32)
        size = width if base == "rowversion" else random.choice([0, 1, width, random.randint(0, width)])
        for _ in range(size): random.getrandbits(8)
    elif base == "uniqueidentifier":
        random.getrandbits(128)
    else:
        lower = date(1753, 1, 1) if base == "datetime" else date(1, 1, 1)
        upper = date(9999, 12, 31)
        if base == "smalldatetime": lower, upper = date(1900, 1, 1), date(2079, 6, 6)
        day = random.choice([lower, upper, date(2000, 2, 29), date.fromordinal(random.randint(lower.toordinal(), upper.toordinal()))])
        if base != "date":
            random.randrange(24); random.randrange(60); random.randrange(60)
            if base == "datetime": random.choice(range(4))
            elif base != "smalldatetime" and args[0]: random.randrange(10**args[0])
            if base == "datetimeoffset" and day.year not in {1, 9999}: random.randint(-840, 840)
    return False


def advance_instantiation_rng(instantiator, table_name):
    """Advance the shared RNG exactly as before, without generating a statement."""
    db_type = str(getattr(DB_CONFIG, "db_type", "") or "").strip().lower()
    if db_type == "sqlserver":
        cached = instantiator._sqlserver_table_cache(table_name)
        if "tlp_rng_targets" not in cached:
            cached["tlp_rng_targets"] = [c for c in cached["source"] if _SQLServerDraws.eligible_target(c)]
        targets = cached["tlp_rng_targets"]
        if not targets: return
        _sqlserver_value_draws(random.choice(targets))
        if "tlp_rng_conditions" not in cached:
            cached["tlp_rng_conditions"] = [c for c, _, capabilities in cached["facts"] if c.get("name") and not c.get("is_hidden") and "comparison" in capabilities and _SQLServerDraws.can_generate(c)]
        conditions = cached["tlp_rng_conditions"]
        if conditions:
            column = random.choice(conditions)
            if not _sqlserver_value_draws(column):
                dtype = cached["base_types"][id(column)]
                size = 2 if dtype in BINARY | {"bit", "uniqueidentifier"} else 4 if dtype in {"char", "varchar", "nchar", "nvarchar"} else 7
                random.choice(range(size))
        return
    columns = instantiator.loader.get_columns(table_name)
    if db_type in {"oracle", "oracledb", "oracle_ai26"}:
        columns = [c for c in columns if c.get("name") and not instantiator._oracle_is_id_column(c)]
        if not columns: return
        column = random.choice(columns)
        _oracle_value_draws(_oracle_column_kind(column), column.get("type", ""))
        conditions = [(c, _oracle_condition_kind(c)) for c in columns if _oracle_condition_kind(c)]
        if conditions:
            column, kind = random.choice(conditions)
            if kind == "BOOLEAN": random.choice(range(2))
            elif not _oracle_value_draws(kind, column.get("type", "")):
                random.choice(range(4 if kind == "STRING" else 2 if kind == "BINARY" else 7))
        return
    columns = [c for c in columns if c['name'] != 'id']
    if not columns: return
    column = random.choice(columns)
    if random.random() >= 0.3:
        RandomDataFactory.generate_value(_value_category(column))
    category = _value_category(random.choice(columns))
    random.choice(range(3 if category in {'STRING', 'BINARY', 'JSON'} else 7))
    RandomDataFactory.generate_value(category)
