"""Bounded Oracle-only value repair of a *whole* TLP validation group.

No table writes, raw-predicate writes, or per-case output files. This module
is loaded only by the Oracle TLP worker after an execution failure.
"""
from __future__ import annotations

import hashlib
import json
import re
import time
from functools import lru_cache
from pathlib import Path

from pkg.parser import SimpleParser, ASTNode
from RL.oracle_evidence_types import normalize_oracle_type, ORACLE_FIXED_RESULTS


@lru_cache(maxsize=1)
def _rules():
    path = Path(__file__).resolve().parents[1] / "pkg/configuration/oracle_rules.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _family(dtype):
    dtype = normalize_oracle_type(dtype)
    if dtype in {"number", "binary_float", "binary_double", "float", "real"}:
        return "number"
    if dtype in {"char", "varchar2", "nchar", "nvarchar2", "clob", "nclob"}:
        return "string"
    if dtype == "boolean":
        return "boolean"
    if dtype == "date" or dtype.startswith("timestamp"):
        return "date"
    return dtype


class _Parser(SimpleParser):
    """Private source-preserving parser; shared Stage-1/2 parser unchanged."""
    def tokenize(self, text):
        operators = set(self.precedence) | set(self.unary_operators) | set(self.operator_templates)
        operators = [o for o in operators if o != "UNARY_HIGH" and not o.endswith("()")]
        patterns = [r"\b" + r"\s+".join(re.escape(p) for p in o.split()) + r"\b"
                    if re.fullmatch(r"[A-Za-z_ ]+", o) else re.escape(o)
                    for o in sorted(operators, key=len, reverse=True)]
        identifier = r'(?:"(?:[^"]|"")+"|[A-Za-z_][\w$#]*)'
        pattern = re.compile(
            r"N?'(?:[^']|'')*'|(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?|"
            + identifier + r"\s*\.\s*" + identifier + "|"
            + "|".join(patterns) + "|" + identifier + r"\s*\(|" + identifier + r"|[(),+*/<>=!%|&^-]", re.I)
        matches = list(pattern.finditer(text))
        end = 0
        for match in matches:
            if text[end:match.start()].strip():
                raise ValueError("Unsupported Oracle syntax")
            end = match.end()
        if text[end:].strip():
            raise ValueError("Unsupported Oracle suffix")
        self.spans = [(m.start(), m.end()) for m in matches]
        return [m.group().strip() for m in matches]

    def _infer_literal_node(self, token):
        return ASTNode(token, "literal")

    def _span(self, start, node):
        node.start = self.spans[start][0]
        node.end = self.spans[self.pos - 1][1]
        return node

    def parse_atom(self):
        start = self.pos
        # Standard typed literals remain one leaf, including their type tag.
        if (str(self.current()).upper() in {"DATE", "TIMESTAMP"}
                and self.pos + 1 < len(self.tokens) and self.tokens[self.pos + 1].startswith("'")):
            token = self.consume() + " " + self.consume()
            return self._span(start, ASTNode(token, "literal"))
        node = super().parse_atom()
        if str(node.token).endswith("()"):
            node.token = re.sub(r"\s+\(\)$", "()", str(node.token))
            node.node_type = "function"
        return self._span(start, node)

    def parse_unary(self):
        start = self.pos
        return self._span(start, super().parse_unary())

    def parse_expression(self, min_prec):
        start = self.pos
        return self._span(start, super().parse_expression(min_prec))


def _predicate_span(sql):
    match = re.search(r"\bWHERE\s*\(", sql, re.I)
    if not match:
        return None
    start, depth, quote, i = match.end(), 1, "", match.end()
    while i < len(sql):
        c = sql[i]
        if quote:
            if c == quote:
                if i + 1 < len(sql) and sql[i + 1] == quote:
                    i += 2
                    continue
                quote = ""
        elif c in "'\"":
            quote = c
        elif c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
            if depth == 0:
                return start, i
        i += 1
    return None


def conversion_domain(error):
    match = re.search(r"(?:ORA-)?(\d{3,5})", str(error.get("error_code", "")))
    if not match:
        match = re.search(r"ORA-(\d{5})", str(error.get("error", "")))
    code = int(match[1]) if match else 0
    if code in {1722, 1481}:
        return "number"
    if code == 61800:
        return "boolean"
    if 1800 <= code <= 1899:
        return "date"
    if code in {40441, 40597}:
        return "json"
    if code == 1465:
        return "raw"
    return ""


class OraclePredicateRepair:
    MAX_SQL_CHARS = 32768
    MAX_TOKENS = 512
    MAX_DESCRIPTIONS = 64
    MAX_METADATA_SECONDS = 2.0

    def __init__(self, executor=None, columns=None):
        self.executor = executor
        self.columns = None if columns is None else {str(k).strip('"').upper(): _family(v) for k, v in columns.items()}
        self.descriptions = {}
        self.metadata_seconds = 0.0

    def _load_columns(self, table):
        if self.columns is not None:
            return
        self.columns = {}  # one attempt per worker, including failures
        if self.executor is not None:
            self._mark("columns")
            result = self.executor.list_columns(table.strip('"'))
            if result.success:
                for row in result.rows or []:
                    row = {str(k).lower(): v for k, v in row.items()}
                    name = row.get("field", row.get("column_name", ""))
                    dtype = row.get("type", row.get("data_type", ""))
                    self.columns[str(name).upper()] = _family(dtype)

    def _mark(self, phase, sql=""):
        progress = getattr(self.executor, "_stage3_worker_progress", None)
        if progress is not None:
            progress.mark(phase, sql)

    @staticmethod
    def _connection_healthy(conn):
        # python-oracledb's local check performs no network round trip. In Thin
        # 4.0.2 a Connection can still hold its implementation after the protocol
        # transport has been cleared; setting call_timeout then enters native
        # set_call_timeout with a missing transport and can crash the process.
        try:
            return conn is not None and bool(conn.is_healthy())
        except Exception:
            return False

    def _describe(self, token, child_types, expression, table):
        key = (token, tuple(child_types))
        if key in self.descriptions:
            return self.descriptions[key]
        if (self.executor is None or len(self.descriptions) >= self.MAX_DESCRIPTIONS
                or self.metadata_seconds >= self.MAX_METADATA_SECONDS):
            return ""
        self.descriptions[key] = ""
        started = time.monotonic()
        conn = getattr(self.executor, "_conn", None)
        old_timeout = None
        sql = f"SELECT {expression} FROM {table} WHERE 1=0"
        try:
            if not self._connection_healthy(conn):
                return ""
            old_timeout = getattr(conn, "call_timeout", None)
            if old_timeout is not None:
                self._mark("describe_set_timeout", sql)
                conn.call_timeout = min(old_timeout or 250, 250)
            with conn.cursor() as cursor:
                self._mark("describe", sql)
                cursor.parse(sql)
                if cursor.description:
                    self.descriptions[key] = _family(cursor.description[0][1].name)
        except Exception:
            pass
        finally:
            # A failed/expired parse can disconnect the transport. Do not touch
            # call_timeout on that connection, including in exception cleanup.
            # The executor's existing reconnect path handles the next query.
            if old_timeout is not None and self._connection_healthy(conn):
                self._mark("describe_restore_timeout", sql)
                try:
                    conn.call_timeout = old_timeout
                except Exception:
                    pass
            self.metadata_seconds += time.monotonic() - started
        return self.descriptions[key]

    def repair(self, predicate, error, *, table="test_wide_table", fallback=False):
        domain = conversion_domain(error)
        if not domain or len(predicate) > self.MAX_SQL_CHARS:
            return predicate, 0
        parser = _Parser(_rules())
        parser.unary_operators |= {"+", "-"}
        try:
            parser.tokens = parser.tokenize(predicate)
            if not parser.tokens or len(parser.tokens) > self.MAX_TOKENS:
                return predicate, 0
            parser.pos = 0
            root = parser.parse_expression(0)
            if parser.pos != len(parser.tokens):
                return predicate, 0
        except (ValueError, IndexError, RecursionError):
            return predicate, 0
        self._load_columns(table)
        seed = int(hashlib.sha1(predicate.encode()).hexdigest()[:8], 16)
        safe = {"number": str(1 + seed % 9), "boolean": "TRUE",
                "date": f"DATE '2024-01-{1 + seed % 28:02d}'", "json": "'{\"probe\":1}'", "raw": "'31'"}
        date_text = f"2024-01-{1 + seed % 28:02d}"
        fixed = _rules().get("candidate_bitmap", {}).get("non_target_slot_fixed_sql", {})
        definitions = _rules().get("definitions", {})
        families = {}
        edits = {}

        def token(node):
            return str(node.token).upper().removesuffix("()")

        def text(node):
            return predicate[node.start:node.end]

        def infer(node):
            if id(node) in families:
                return families[id(node)]
            name, raw = token(node), text(node).strip()
            children = node.children
            if not children:
                key = raw.rsplit(".", 1)[-1].strip().strip('"').upper()
                result = (self.columns or {}).get(key, "")
                if not result:
                    if raw.upper() in {"TRUE", "FALSE"}:
                        result = "boolean"
                    elif raw.upper() == "NULL":
                        result = "null"
                    elif re.fullmatch(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?", raw):
                        result = "number"
                    elif re.fullmatch(r"N?'(?:[^']|'')*'", raw, re.I | re.S):
                        result = "string"
                    elif raw.upper().startswith(("DATE '", "TIMESTAMP '")):
                        result = "date"
            else:
                child_types = [infer(c) for c in children]
                definition = definitions.get(name + "()", definitions.get(name, []))
                coarse = definition[0] if definition else ""
                result = ("boolean" if coarse == "T_BOOL" else "string" if coarse == "T_STR" else "")
                result = ORACLE_FIXED_RESULTS.get(name, result)
                if name in {"TO_DATE", "TO_TIMESTAMP", "TO_TIMESTAMP_TZ"}:
                    result = "date"
                if not result and name in {"+", "-", "*", "/"} and "number" in child_types:
                    result = "number"
                if not result and node.node_type == "function":
                    result = self._describe(name, child_types, raw, table)
            families[id(node)] = result
            return result

        def replace(node, value):
            if text(node) != value:
                # Outer replacement supersedes any pending inner edits.
                for span in list(edits):
                    if node.start <= span[0] and span[1] <= node.end:
                        del edits[span]
                edits[(node.start, node.end)] = value

        def force(node, wanted):
            current = infer(node)
            if current == "null":
                return
            if not node.children or fallback:
                value = safe.get(wanted)
                if value is not None:
                    if current == "string" and wanted in {"number", "boolean"}:
                        value = "'" + value + "'"
                    if current == "string" and wanted == "date":
                        value = "'" + date_text + "'"
                    replace(node, value)
                return
            # First attempt preserves functions and works on their arguments.
            protected_slots = fixed.get(token(node) + "()", {})
            for i, child in enumerate(node.children):
                if str(i) in protected_slots:
                    continue
                if infer(child) not in {"null", wanted}:
                    force(child, wanted)

        comparisons = {"=", "!=", "<>", "<", ">", "<=", ">=", "IN", "NOT IN", "BETWEEN", "NOT BETWEEN"}

        def visit(node):
            for child in node.children:
                visit(child)
            name, children = token(node), node.children
            if not children:
                return
            types = [infer(c) for c in children]
            if name in comparisons:
                wanted = ("boolean" if "boolean" in types else "number" if "number" in types
                          else "date" if "date" in types else "")
                if wanted == domain:
                    for child, dtype in zip(children, types):
                        if dtype not in {wanted, "null"}:
                            force(child, wanted)
            if domain == "boolean" and name in {"AND", "OR", "NOT", "IS TRUE", "IS FALSE", "IS NOT TRUE", "IS NOT FALSE", "LNNVL"}:
                for child, dtype in zip(children, types):
                    if dtype not in {"boolean", "number", "null"}:
                        force(child, "boolean")
            if node.node_type != "function":
                if domain == "number" and name in {"+", "-", "*", "/"}:
                    for child, dtype in zip(children, types):
                        if dtype == "string":
                            force(child, "number")
                return
            # Reuse Stage-2 safe sibling values, not its graph-building flow.
            slots = fixed.get(name + "()", {})
            if domain in {"number", "date", "json", "raw"}:
                for slot, value in slots.items():
                    if int(slot) < len(children):
                        replace(children[int(slot)], value)
            if name in {"TO_DATE", "TO_TIMESTAMP", "TO_TIMESTAMP_TZ"} and domain == "date":
                value, fmt = date_text, "YYYY-MM-DD"
                if name != "TO_DATE":
                    value += " 12:00:00"
                    fmt += " HH24:MI:SS"
                if name == "TO_TIMESTAMP_TZ":
                    value += " +00:00"
                    fmt += " TZH:TZM"
                replace(children[0], "'" + value + "'")
                if len(children) > 1:
                    replace(children[1], "'" + fmt + "'")
                return
            if name == "TO_CHAR" and domain in {"number", "date"} and len(children) > 1:
                replace(children[1], "'YYYY-MM-DD'" if types[0] == "date" else "'FM9990'")
            if domain == "json" and name.startswith("JSON"):
                replace(children[0], safe["json"])
            if domain == "raw" and name == "HEXTORAW":
                replace(children[0], safe["raw"])
            if name == "TO_BOOLEAN" and domain == "boolean":
                force(children[0], "boolean")
            if domain == "number" and (name == "TO_NUMBER" or infer(node) == "number"):
                for i, child in enumerate(children):
                    if str(i) not in slots and infer(child) == "string":
                        force(child, "number")

        visit(root)
        if domain == "boolean" and infer(root) not in {"boolean", "number", "null"}:
            force(root, "boolean")
        rewritten = predicate
        for (start, end), value in sorted(edits.items(), reverse=True):
            rewritten = rewritten[:start] + value + rewritten[end:]
        return rewritten, len(edits)

    def repair_group(self, group, error, *, fallback=False):
        if len(group) != 4:
            return group, 0
        span = _predicate_span(group[1])
        table_match = re.search(r'\bFROM\s+("[^"]+"|[A-Za-z_][\w$#]*)', group[1], re.I)
        if not span or not table_match:
            return group, 0
        original = group[1][span[0]:span[1]]
        wrapped = "(" + original + ")"
        if any(sql.count(wrapped) != 1 for sql in group[1:]):
            return group, 0
        repaired, count = self.repair(original, error, table=table_match[1], fallback=fallback)
        if not count or repaired == original:
            return group, 0
        return list(group[:1]) + [sql.replace(wrapped, "(" + repaired + ")", 1) for sql in group[1:]], count


def execute_oracle_group(executor, group, *, cleaner=lambda sql: sql):
    """At most two repairs; discard partial counts before every full retry."""
    current = list(group)
    summary = {"attempts": 0, "successes": 0, "rewritten_arguments": 0}
    repairer = getattr(executor, "_stage3_value_repairer", None)
    progress = getattr(executor, "_stage3_worker_progress", None)
    for attempt in range(3):
        values, failure = {}, None
        for role, sql in zip(("total", "true", "false", "null"), current):
            sql = cleaner(sql)
            if progress is not None:
                progress.mark(role, sql, attempt=attempt)
            result = executor.execute_query(sql, fetch="one")
            if not result.success:
                failure = {"role": role, "sql": sql, "error": result.error,
                           "error_code": result.error_code, "execution_time": result.execution_time,
                           "is_timeout": bool(result.timed_out)}
                break
            values[role] = int(next(iter(result.rows[0].values()))) if result.rows else 0
        if failure is None:
            summary["successes"] = int(summary["attempts"] > 0)
            return current, values, None, summary
        if attempt >= 2 or failure["is_timeout"] or not conversion_domain(failure):
            return current, None, failure, summary
        if repairer is None:
            repairer = OraclePredicateRepair(executor)
            executor._stage3_value_repairer = repairer
        try:
            if progress is not None:
                progress.mark("repair", failure["sql"], attempt=attempt)
            repaired, count = repairer.repair_group(current, failure, fallback=attempt > 0)
            if not count and attempt == 0:
                # E.g. CHR(number) cannot become numeric by fixing a string
                # argument. Only then replace the offending operand itself;
                # do not spend a database retry on an unchanged predicate.
                repaired, count = repairer.repair_group(current, failure, fallback=True)
        except Exception:
            # Repair is auxiliary: unsupported SQL/metadata must remain a
            # normal validation failure, never crash a worker or pass a case.
            return current, None, failure, summary
        if not count:
            return current, None, failure, summary
        current = repaired
        summary["attempts"] += 1
        summary["rewritten_arguments"] += count
    raise AssertionError("unreachable")
