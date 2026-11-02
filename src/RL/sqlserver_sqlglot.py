"""Local T-SQL parsing compatibility; never rewrite the SQL sent to the server."""


def parse_sqlserver_expression(sql: str):
    import sqlglot
    from sqlglot import exp

    dialect = sqlglot.Dialect.get_or_raise("tsql")
    parser = dialect.parser()
    # sqlglot 28.4.1 evaluates PARSENAME's literal index in Python. A string
    # index such as '1' raises TypeError although SQL Server accepts it through
    # implicit conversion. Keep the call/arguments intact for server validation.
    # Copy onto this parser instance only; other dialects/parsers are unaffected.
    parser.FUNCTIONS = {
        **parser.FUNCTIONS,
        "PARSENAME": lambda args: exp.Anonymous(this="PARSENAME", expressions=args),
    }
    expressions = parser.parse(dialect.tokenize(sql), sql)
    if not expressions or expressions[0] is None:
        raise sqlglot.errors.ParseError("Empty SQL Server expression.")
    return expressions[0]
