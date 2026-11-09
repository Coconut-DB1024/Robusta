def parse_sqlserver_expression(sql: str):
    import sqlglot
    from sqlglot import exp

    dialect = sqlglot.Dialect.get_or_raise("tsql")
    parser = dialect.parser()

    parser.FUNCTIONS = {
        **parser.FUNCTIONS,
        "PARSENAME": lambda args: exp.Anonymous(this="PARSENAME", expressions=args),
    }
    expressions = parser.parse(dialect.tokenize(sql), sql)
    if not expressions or expressions[0] is None:
        raise sqlglot.errors.ParseError("Empty SQL Server expression.")
    return expressions[0]
