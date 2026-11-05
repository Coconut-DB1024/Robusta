from pglast import ast, parse_sql
from pglast.parser import ParseError
from pglast.visitors import Visitor


class InvalidQueryError(ValueError):
    pass


_SIDE_EFFECT_FUNCTIONS = frozenset(
    {
        "set_config",
        "nextval",
        "setval",
        "pg_notify",
        "lo_open",
        "lo_close",
        "lo_creat",
        "lo_create",
        "lo_import",
        "lo_export",
        "lo_put",
        "lo_unlink",
        "lowrite",
        "lo_truncate",
        "lo_truncate64",
    }
)
_SIDE_EFFECT_PREFIXES = ("pg_advisory_", "pg_try_advisory_", "dblink")


class _ReadOnlySelectVisitor(Visitor):
    def visit(self, ancestors, node):
        if type(node).__name__.endswith("Stmt") and not isinstance(node, ast.SelectStmt):
            raise InvalidQueryError("Only SELECT statements are permitted, including inside CTEs.")

        if isinstance(node, ast.SelectStmt):
            if node.intoClause is not None:
                raise InvalidQueryError("SELECT INTO is not permitted.")
            if node.lockingClause:
                raise InvalidQueryError("SELECT locking clauses (FOR UPDATE/SHARE) are not permitted.")

        if isinstance(node, ast.FuncCall):
            function_name = node.funcname[-1].sval.lower()
            if function_name in _SIDE_EFFECT_FUNCTIONS or function_name.startswith(
                _SIDE_EFFECT_PREFIXES
            ):
                raise InvalidQueryError(
                    f"Function {function_name!r} has possible side effects and is not permitted."
                )


def validate_select(sql: str) -> None:
    if not isinstance(sql, str) or not sql.strip():
        raise InvalidQueryError("SQL must be a non-empty string containing one SELECT.")
    if "\x00" in sql:
        raise InvalidQueryError("SQL must not contain NUL characters.")

    try:
        statements = parse_sql(sql)
    except ParseError as exc:
        raise InvalidQueryError(f"Invalid PostgreSQL SQL: {exc}") from exc

    if len(statements) != 1:
        raise InvalidQueryError("Exactly one SQL statement is required.")

    statement = statements[0].stmt
    if not isinstance(statement, ast.SelectStmt):
        raise InvalidQueryError("The top-level statement must be SELECT (optionally WITH).")

    first_operand = statement
    while first_operand.larg is not None:
        first_operand = first_operand.larg
    if first_operand.valuesLists:
        raise InvalidQueryError("The query must start with SELECT, not VALUES.")

    _ReadOnlySelectVisitor()(statement)
