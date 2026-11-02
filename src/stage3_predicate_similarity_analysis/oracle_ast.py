"""Oracle-only parsing adapters for the project's semantic predicate trees.

The source predicate files are never changed. These subclasses are loaded only
by AST mode; SQLGlot's global Oracle dialect and the other DBMS branches remain
untouched.
"""
from __future__ import annotations

from sqlglot import exp
from sqlglot.dialects.oracle import Oracle
from sqlglot.tokens import Token, TokenType


# Ordinary comma-separated function calls emitted by oracle_rules.json. Keep
# their original names and explicit operands: transpilation-oriented builders
# otherwise turn NVL into COALESCE, change date format values, or manufacture
# JSONPath nodes from one string literal. SYS_GUID is the graph's zero-arg call.
_GRAPH_FUNCTIONS = frozenset("""
    LNNVL REGEXP_LIKE JSON_EXISTS ABS CEIL FLOOR ROUND TRUNC MOD POWER SQRT EXP
    LN LOG SIGN REMAINDER GREATEST LEAST BITAND WIDTH_BUCKET ATAN2 ATAN COS COSH
    SIN SINH TAN TANH LOWER UPPER INITCAP NLS_LOWER NLS_UPPER NLS_INITCAP LENGTH
    LENGTHB LENGTHC LENGTH2 LENGTH4 INSTR INSTRB INSTRC INSTR2 INSTR4 SUBSTR
    SUBSTRB SUBSTRC SUBSTR2 SUBSTR4 REPLACE TRANSLATE TRIM LTRIM RTRIM LPAD RPAD
    CONCAT CHR NCHR ASCII ASCIISTR UNISTR COMPOSE DECOMPOSE TO_MULTI_BYTE
    TO_SINGLE_BYTE SOUNDEX NLSSORT REGEXP_COUNT REGEXP_INSTR REGEXP_REPLACE
    REGEXP_SUBSTR TO_CHAR TO_NUMBER TO_DATE TO_TIMESTAMP TO_TIMESTAMP_TZ NVL NVL2
    COALESCE NULLIF DECODE TO_BOOLEAN ADD_MONTHS MONTHS_BETWEEN LAST_DAY
    NUMTODSINTERVAL NUMTOYMINTERVAL FROM_TZ SYS_EXTRACT_UTC VSIZE ORA_HASH
    COLLATION RAWTOHEX DUMP TO_NCHAR JSON_VALUE JSON_QUERY JSON_SERIALIZE
    STANDARD_HASH SYS_GUID
""".split())

# Only these functions also have a keyword-based Oracle form. Do not route
# TRIM(... FROM ...) or JSON ... RETURNING/PASSING/... through a comma-call
# parser. Nested expressions and quoted text are not treated as such keywords.
_SPECIAL_CALL_SYNTAX = {
    "TRIM": frozenset({"FROM"}),
    "JSON_EXISTS": frozenset({"PASSING", "ON", "RETURNING"}),
    "JSON_QUERY": frozenset({"PASSING", "RETURNING", "WITH", "WITHOUT", "ON"}),
    "JSON_VALUE": frozenset({"PASSING", "RETURNING", "ON"}),
    "JSON_SERIALIZE": frozenset({"RETURNING", "PRETTY", "ASCII", "ON"}),
}

# Reserved type hints can contain spaces although they are not SQL identifier
# syntax. Limit this accommodation to Oracle graph types inside col(__TYPE__
# ...); never absorb an expression or quoted SQL text into a fake column name.
_MULTIWORD_GRAPH_TYPES = frozenset({
    "double precision", "long raw", "timestamp with time zone",
    "timestamp with local time zone", "timestamp without time zone",
    "interval year to month", "interval day to second",
})


def _project_type_tokens(tokens):
    merged = []
    cursor = 0
    while cursor < len(tokens):
        if (
            cursor + 3 < len(tokens)
            and tokens[cursor].token_type == TokenType.VAR
            and tokens[cursor].text.lower() == "col"
            and tokens[cursor + 1].token_type == TokenType.L_PAREN
            and tokens[cursor + 2].token_type == TokenType.VAR
            and tokens[cursor + 2].text.startswith("__TYPE__")
        ):
            start = cursor + 2
            end = start + 1
            while end < len(tokens) and tokens[end].text.isidentifier() and (
                tokens[end].token_type not in (TokenType.STRING, TokenType.IDENTIFIER)
            ):
                end += 1
            name = " ".join(token.text for token in tokens[start:end])
            if (
                end > start + 1 and end < len(tokens)
                and tokens[end].token_type == TokenType.R_PAREN
                and name[len("__TYPE__"):].lower() in _MULTIWORD_GRAPH_TYPES
            ):
                first, last = tokens[start], tokens[end - 1]
                merged.extend(tokens[cursor:start])
                merged.append(Token(
                    TokenType.VAR, name,
                    line=first.line, col=first.col, start=first.start, end=last.end,
                    comments=[comment for token in tokens[start:end]
                              for comment in token.comments],
                ))
                cursor = end
                continue
        merged.append(tokens[cursor])
        cursor += 1
    return merged


class OracleLike(exp.Expression):
    # LIKEC / LIKE2 / LIKE4 are distinct operators, not a LIKE alias. Preserve
    # the spelling as a property of ONE operator node, with ordered operands.
    arg_types = {"this": True, "expression": True, "operator": True}


class PredicateOracle(Oracle):
    def to_json_path(self, path):
        # A literal JSON path is still one original string parameter. Do not
        # add JSONPathRoot/JSONPathKey scoring nodes, even in native JSON forms.
        return path

    class Tokenizer(Oracle.Tokenizer):
        KEYWORDS = {
            **Oracle.Tokenizer.KEYWORDS,
            "LIKEC": TokenType.LIKE,
            "LIKE2": TokenType.LIKE,
            "LIKE4": TokenType.LIKE,
        }

        def tokenize(self, sql):
            return _project_type_tokens(super().tokenize(sql))

    class Parser(Oracle.Parser):
        RANGE_PARSERS = {
            **Oracle.Parser.RANGE_PARSERS,
            TokenType.LIKE: lambda self, this: self._parse_oracle_like(this),
        }

        def _parse_oracle_like(self, this):
            operator = self._prev.text.upper()
            if operator == "LIKE":
                # Ordinary LIKE retains its previous SQLGlot representation.
                return Oracle.Parser.RANGE_PARSERS[TokenType.LIKE](self, this)
            expression = self._parse_bitwise()
            node = self.validate_expression(OracleLike(
                this=this, expression=expression, operator=operator,
            ))
            # The surrounding range parser handles NOT and precedence; the
            # standard escape parser retains an explicit ESCAPE argument.
            return self._parse_escape(node)

        def _plain_graph_call(self, name):
            keywords = _SPECIAL_CALL_SYNTAX.get(name)
            if not keywords:
                return True
            depth = 0
            for index in range(self._index + 1, len(self._tokens)):
                token = self._tokens[index]
                if token.token_type == TokenType.L_PAREN:
                    depth += 1
                elif token.token_type == TokenType.R_PAREN:
                    depth -= 1
                    if depth == 0:
                        break
                elif (
                    depth == 1
                    and token.token_type not in (TokenType.STRING, TokenType.IDENTIFIER)
                    and token.text.upper() in keywords
                ):
                    return False
            return True

        def _parse_function_call(
            self, functions=None, anonymous=False, optional_parens=True, any_token=False,
        ):
            if (
                self._curr
                and self._next
                and self._next.token_type == TokenType.L_PAREN
                and self._curr.text.upper() in _GRAPH_FUNCTIONS
                and self._plain_graph_call(self._curr.text.upper())
            ):
                # SQLGlot's generic call parser retains the original name,
                # explicit argument order and literal values without a rewrite.
                anonymous = True
            return super()._parse_function_call(
                functions, anonymous, optional_parens, any_token,
            )
