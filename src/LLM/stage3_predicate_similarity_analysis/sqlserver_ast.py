from __future__ import annotations

import re

from sqlglot import exp
from sqlglot.dialects.tsql import TSQL
from sqlglot.tokens import Token, TokenType


class SqlServerNotLessThan(exp.Expression):
    arg_types = {"this": True, "expression": True}


class SqlServerNotGreaterThan(exp.Expression):
    arg_types = {"this": True, "expression": True}


_PROJECT_FUNCTIONS = frozenset("""
ABS ACOS APP_NAME ASCII ASIN ATAN ATN2 BINARY_CHECKSUM BIT_COUNT CEILING
CHARINDEX CHECKSUM CHOOSE COALESCE CONCAT CONCAT_WS COS COT DATALENGTH
DATEFROMPARTS DAY DB_ID DB_NAME DEGREES DIFFERENCE EOMONTH EXP FLOOR
GET_BIT GREATEST HOST_NAME IIF ISDATE ISJSON ISNULL ISNUMERIC LEAST LEFT
LEFT_SHIFT LEN LOG LOG10 LOWER LTRIM MONTH NCHAR NULLIF ORIGINAL_LOGIN
PARSENAME PATINDEX PI POWER QUOTENAME RADIANS REPLACE REPLICATE REVERSE
RIGHT RIGHT_SHIFT ROUND RTRIM SESSION_USER SET_BIT SIGN SIN SOUNDEX SPACE
SQRT SQUARE STR STUFF SUBSTRING SUSER_NAME SUSER_SNAME TAN TRANSLATE TRIM
UNICODE UPPER USER_NAME XACT_STATE YEAR
""".split())


def _source_function(name):
    def build(arguments):
        return exp.Anonymous(this=name, expressions=arguments)

    return build


_PARAMETERIZED_TYPE = re.compile(
    r"(?:decimal|numeric)\(\d+(?:,\d+)?\)"
    r"|(?:nvarchar|varchar|varbinary)\((?:max|\d+)\)"
    r"|(?:nchar|char|binary|float|time|datetime2|datetimeoffset)\(\d+\)",
    re.IGNORECASE,
)
_SYSTEM_VARIABLES = frozenset({"SPID", "TRANCOUNT", "VERSION"})


def _project_type_tokens(tokens):
    merged = []
    cursor = 0
    while cursor < len(tokens):
        if (
            cursor + 5 < len(tokens)
            and tokens[cursor].token_type == TokenType.VAR
            and tokens[cursor].text.lower() == "col"
            and tokens[cursor + 1].token_type == TokenType.L_PAREN
            and tokens[cursor + 2].token_type == TokenType.VAR
            and tokens[cursor + 2].text.startswith("__TYPE__")
            and tokens[cursor + 3].token_type == TokenType.L_PAREN
        ):
            start = cursor + 2
            end = start + 2
            while end < len(tokens) and tokens[end].token_type in (
                TokenType.NUMBER, TokenType.COMMA, TokenType.VAR,
            ):
                end += 1
            if (
                end + 1 < len(tokens)
                and tokens[end].token_type == TokenType.R_PAREN
                and tokens[end + 1].token_type == TokenType.R_PAREN
            ):
                name = "".join(token.text for token in tokens[start:end + 1])
                if _PARAMETERIZED_TYPE.fullmatch(name[len("__TYPE__"):]):
                    first, last = tokens[start], tokens[end]
                    merged.extend(tokens[cursor:start])
                    merged.append(Token(
                        TokenType.VAR, name,
                        line=first.line, col=first.col, start=first.start, end=last.end,
                        comments=[
                            comment for token in tokens[start:end + 1]
                            for comment in token.comments
                        ],
                    ))
                    cursor = end + 1
                    continue
        merged.append(tokens[cursor])
        cursor += 1
    return merged


class PredicateTSQL(TSQL):
    class Tokenizer(TSQL.Tokenizer):
        KEYWORDS = {**TSQL.Tokenizer.KEYWORDS, "!<": TokenType.GTE, "!>": TokenType.LTE}

        def tokenize(self, sql):
            return _project_type_tokens(super().tokenize(sql))

    class Parser(TSQL.Parser):
        FUNCTIONS = {
            **TSQL.Parser.FUNCTIONS,
            **{name: _source_function(name) for name in _PROJECT_FUNCTIONS},
        }

        def _parse_comparison(self):
            this = self._parse_range()
            while self._match_set(self.COMPARISON):
                comments = self._prev_comments
                operator = self._prev.text
                kind = (
                    SqlServerNotLessThan if operator == "!<" else
                    SqlServerNotGreaterThan if operator == "!>" else
                    self.COMPARISON[self._prev.token_type]
                )

                this = self.validate_expression(kind(this=this, expression=self._parse_range()))
                this.add_comments(comments)
            return this

        def _parse_parameter(self):
            if (
                self._curr and self._curr.token_type == TokenType.PARAMETER
                and self._next and self._next.token_type == TokenType.VAR
                and self._next.text.upper() in _SYSTEM_VARIABLES
            ):
                name = "@@" + self._next.text.upper()
                self._advance(2)
                return exp.Var(this=name)
            return super()._parse_parameter()
