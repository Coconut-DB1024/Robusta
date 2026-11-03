"""The fixed table used by the main experiment's metadata and SQL paths."""

from typing import Any, Optional, Tuple


TABLE_NAME = "test_wide_table"


def get_target_table(db_type: str, config: Any = None) -> Tuple[str, Optional[str], str]:
    """Return the catalog name, schema, and SQL identifier for the test table."""
    db_type = str(db_type or "").lower().strip()
    if db_type in {"postgresql", "postgres", "pg"}:
        return TABLE_NAME, "public", '"public"."test_wide_table"'
    if db_type in {"oracle", "oracledb", "oracle_ai26"}:
        # Oracle's initializer creates an unquoted name, stored in uppercase.
        return TABLE_NAME.upper(), None, '"TEST_WIDE_TABLE"'
    if db_type == "sqlserver":
        schema = str(getattr(config, "schema", "dbo") or "dbo")
        quoted_schema = "[" + schema.replace("]", "]]") + "]"
        return TABLE_NAME, schema, f"{quoted_schema}.[test_wide_table]"
    return TABLE_NAME, None, "`test_wide_table`"
