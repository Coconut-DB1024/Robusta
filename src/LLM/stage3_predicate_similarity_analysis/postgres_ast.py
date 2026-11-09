from __future__ import annotations

from sqlglot import exp
from sqlglot.dialects.postgres import Postgres
from sqlglot.tokens import Token, TokenType


_MULTIWORD_GRAPH_TYPES = {"double precision", "character varying"}
_GEOMETRIC_LITERAL_TYPES = {"point", "line", "lseg", "box", "path", "polygon", "circle"}


class IsNormalized(exp.Expression):
    arg_types = {"this": True, "form": False}


class PostgresTypedLiteral(exp.Expression):
    arg_types = {"this": True, "kind": True}


class PostgresPrefix(exp.Expression):
    arg_types = {"this": True, "operator": True}


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
            if name[len("__TYPE__"):].lower() in _MULTIWORD_GRAPH_TYPES and (
                end > start + 1 and end < len(tokens)
            ) and (
                tokens[end].token_type == TokenType.R_PAREN
            ):
                first, last = tokens[start], tokens[end - 1]
                merged.extend(tokens[cursor:start])
                merged.append(Token(
                    TokenType.VAR, name,
                    line=first.line, col=first.col, start=first.start, end=last.end,
                    comments=[comment for token in tokens[start:end] for comment in token.comments],
                ))
                cursor = end
                continue
        merged.append(tokens[cursor])
        cursor += 1
    return merged


class PredicatePostgres(Postgres):
    class Tokenizer(Postgres.Tokenizer):
        def tokenize(self, sql):
            return _project_type_tokens(super().tokenize(sql))

    class Parser(Postgres.Parser):
        UNARY_PARSERS = {
            **Postgres.Parser.UNARY_PARSERS,

            TokenType.HASH: lambda self: self._parse_hash_prefix(),
        }

        def _parse_hash_prefix(self):
            return self.validate_expression(PostgresPrefix(this=self._parse_unary(), operator="#"))

        def _parse_type(self, parse_interval=True, fallback_to_identifier=False):
            if not fallback_to_identifier and self._curr and (
                self._curr.text.lower() in _GEOMETRIC_LITERAL_TYPES
                and self._curr.token_type in (TokenType.VAR, TokenType.POINT)
                and self._next.token_type == TokenType.STRING
            ):
                kind, value = self._curr.text.upper(), self._next.text
                self._advance(2)
                return self._parse_column_ops(PostgresTypedLiteral(this=value, kind=kind))
            return super()._parse_type(parse_interval, fallback_to_identifier)

        def _parse_is(self, this):
            start = self._index
            negate = self._match(TokenType.NOT)
            form = None
            if self._curr and self._curr.token_type == TokenType.VAR and (
                self._curr.text.upper() in {"NFC", "NFD", "NFKC", "NFKD"}
            ):
                form = self._curr.text.upper()
                self._advance()
            if self._curr and self._curr.token_type == TokenType.VAR and (
                self._curr.text.upper() == "NORMALIZED"
            ):
                self._advance()
                node = IsNormalized(this=this, form=form)
                if negate:
                    node = exp.Not(this=node)
                return self._parse_column_ops(node)

            self._retreat(start)
            negate = self._match(TokenType.NOT)
            if self._match(TokenType.UNKNOWN):
                node = exp.Is(this=this, expression=exp.Var(this="UNKNOWN"))
                if negate:
                    node = exp.Not(this=node)
                return self._parse_column_ops(node)
            self._retreat(start)
            return super()._parse_is(this)

        def _parse_normalize(self):
            value = self._parse_bitwise()
            form = False
            if self._match(TokenType.COMMA):
                form = self._parse_string() or self._parse_var()
                if form is None:
                    self.raise_error("Expected a NORMALIZE form")
            return self.validate_expression(exp.Normalize(this=value, form=form))
