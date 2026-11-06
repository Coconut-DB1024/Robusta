"""Schema context, read-only predicate validation and lossless DSL formatting."""

from dataclasses import dataclass
import json
import re


@dataclass
class SeedContext:
    executor: object
    db_type: str
    dialect: str
    table_sql: str
    columns: list
    nodes: list


def open_context():
    from RL.config import DB_CONFIG, CANDIDATE_BITMAP_CONFIG, get_target_table
    from RL.candidate_bitmap import collect_token_signatures
    from RL.db_executor import create_db_executor, normalize_db_type

    db_type = normalize_db_type(DB_CONFIG.db_type)
    dialect = {"postgresql": "postgres", "mysql": "mysql", "oracle": "oracle",
               "sqlserver": "tsql"}[db_type]
    rules = json.loads(CANDIDATE_BITMAP_CONFIG.rules_path.read_text(encoding="utf-8-sig"))
    nodes = [dict(token=s.token, return_type=s.return_type, argument_types=s.arg_types,
                  terminal=s.is_terminal) for s in collect_token_signatures(rules).values()]
    table, schema, table_sql = get_target_table(db_type, DB_CONFIG)
    executor = create_db_executor(db_type=db_type, config=DB_CONFIG,
                                  query_timeout_seconds=DB_CONFIG.query_timeout)
    try:
        result = executor.list_columns(table, schema=schema)
        if not result.success or not result.rows:
            raise ValueError("The configured table has no readable columns.")
        columns = []
        for row in result.rows:
            row = {str(key).lower(): value for key, value in row.items()}
            name, data_type = row.get("field", row.get("column_name")), row.get("type", row.get("data_type"))
            if name:
                columns.append({"name": str(name), "type": str(data_type or "")})
        if not columns or not nodes:
            raise ValueError("No columns or operator nodes are available.")
        return SeedContext(executor, db_type, dialect, table_sql, columns, nodes)
    except Exception:
        executor.close()
        raise


def _parse_predicate(predicate, context):
    import sqlglot
    from sqlglot import exp

    if not predicate or len(predicate) > 4096 or any(c in predicate for c in "\r\n\x00"):
        raise ValueError("Expected one short, single-line predicate.")
    tokens = sqlglot.Dialect.get_or_raise(context.dialect).tokenize(predicate)
    functions = {node["token"][:-2].upper() for node in context.nodes if node["token"].endswith("()")}
    for index, token in enumerate(tokens):
        if token.comments or token.token_type.name in {"SEMICOLON", "DCOLON", "COLON_EQ"}:
            raise ValueError("Comments, statement separators, casts and assignments are not allowed.")
        if token.token_type.name == "L_PAREN" and index:
            previous = tokens[index - 1]
            if re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*", previous.text):
                if previous.text.upper() not in functions | {"AND", "OR", "NOT", "IN"}:
                    raise ValueError("Function is outside the configured Stage-2 nodes.")
                if index > 1 and tokens[index - 2].token_type.name == "DOT":
                    raise ValueError("Qualified function calls are not allowed.")

    prefix = f"SELECT 1 FROM {context.table_sql} WHERE ("
    statements = sqlglot.parse(prefix + predicate + ")", read=context.dialect)
    if len(statements) != 1 or not isinstance(statements[0], exp.Select):
        raise ValueError("Expected a single SELECT predicate.")
    statement = statements[0]
    base = statement.copy()
    base.set("where", None)
    if base != sqlglot.parse_one(f"SELECT 1 FROM {context.table_sql}", read=context.dialect):
        raise ValueError("Predicate changed the fixed SELECT template.")
    if len(list(statement.find_all(exp.Select))) != 1 or len(list(statement.find_all(exp.Table))) != 1:
        raise ValueError("Subqueries are not allowed.")
    if any(type(node).__name__ in {"NextValueFor", "PropertyEQ", "Into", "Command"}
           for node in statement.walk()):
        raise ValueError("Side-effecting expressions are not allowed.")
    names = {column["name"].casefold() for column in context.columns}
    column_spans = set()
    for column in statement.find_all(exp.Column):
        if column.table or column.db or column.catalog or column.name.casefold() not in names:
            raise ValueError("Only actual, unqualified table columns are allowed.")
        start, end = column.this.meta.get("start"), column.this.meta.get("end")
        if start is None or end is None:
            raise ValueError("Column source positions are unavailable.")
        column_spans.add((start - len(prefix), end - len(prefix)))
    return predicate, tokens, column_spans


def validate_predicate(predicate, context):
    import sqlglot

    try:
        parsed = _parse_predicate(predicate, context)
    except (ValueError, sqlglot.errors.SqlglotError):
        return None
    if context.db_type == "sqlserver":
        sql = f"SELECT TOP (1) 1 FROM {context.table_sql} WHERE ({predicate})"
    elif context.db_type == "oracle":
        sql = f"SELECT 1 FROM {context.table_sql} WHERE ({predicate}) FETCH FIRST 1 ROWS ONLY"
    else:
        sql = f"SELECT 1 FROM {context.table_sql} WHERE ({predicate}) LIMIT 1"
    result = context.executor.execute_query(sql, fetch="one")
    # Executability only: a successful query returning zero rows is accepted.
    return parsed if result.success else None


def predicate_template(parsed, context):
    from .sql_type_utils import detect_literal_type

    predicate, tokens, columns = parsed
    pieces, offset = [], 0
    for token in tokens:
        raw = predicate[token.start:token.end + 1]
        if (token.start, token.end) in columns:
            value = f"col({raw})"
        elif token.token_type.name in {"STRING", "NATIONAL_STRING", "NUMBER", "TRUE", "FALSE", "NULL"}:
            kind = detect_literal_type(raw, db_type=context.db_type)
            if kind is None and token.token_type.name == "NUMBER":
                kind = "float"
            if context.db_type == "sqlserver" and kind in {"null", "bool"}:
                value = raw.upper()
            else:
                value = f"{kind}({raw})" if kind else raw
        else:
            value = raw.upper() if re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*", raw) else raw
        pieces.append(predicate[offset:token.start])
        pieces.append(value)
        offset = token.end + 1
    pieces.append(predicate[offset:])
    return "".join(pieces)
