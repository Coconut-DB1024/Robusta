"""Oracle-only, bounded metadata enrichment of Stage-2 evidence copies.

No graph/parser/reward state is changed.  Columns come from the executed SQL,
not from COL_* guesses.  Result types are described by Oracle (parse only,
never execute/fetch), cached by the local typed call, not by its descendants.
"""
from __future__ import annotations

import copy
import re
import time
from collections import Counter

from pkg.parser import SimpleParser


def normalize_oracle_type(value):
    text = str(value or "").strip().lower()
    text = re.sub(r"\([^()]*\)", "", text)
    text = " ".join(text.split()).removeprefix("db_type_")
    return {
        "varchar": "varchar2", "nvarchar": "nvarchar2", "integer": "number",
        "int": "number", "decimal": "number", "numeric": "number",
        "timestamp": "timestamp without time zone",
        "timestamp_tz": "timestamp with time zone",
        "timestamp_ltz": "timestamp with local time zone",
        "interval_ds": "interval day to second", "interval_ym": "interval year to month",
    }.get(text, text)


# Fixed result types, not merge rules. Other functions require observed
# complete physical inputs and an unambiguous described output.
ORACLE_FIXED_RESULTS = {"LENGTH": "number", "TO_NUMBER": "number"}


def complete_observed_types(evidence_by_outer):
    """Enrich private Stage-3 copies, without merging tuples or changing IDs.

    O(8 * evidence entries * arity); no database calls or file writes. A
    conflicting description is never resolved by guessing a coarse family.
    """
    result = {key: [dict(r) for r in records]
              for key, records in evidence_by_outer.items()}
    records = [r for group in result.values() for r in group]
    stats = Counter()
    for record in records:
        record["oracle_identity_before_enrichment"] = {
            "physical_type": record.get("physical_type", ""),
            "slot_child_physical_types": list(record.get("slot_child_physical_types") or []),
        }
        record["slot_child_physical_types"] = list(record.get("slot_child_physical_types") or [])

    def call_key(record):
        inputs = record.get("slot_child_physical_types") or []
        if inputs and all(inputs) and len(inputs) == len(record.get("slot_child_node_ids") or []):
            return (str(record.get("outer_token", "")).upper(), tuple(inputs))
        return None

    def add_unique(index, key, dtype):
        if key is not None and dtype:
            if key not in index:
                index[key] = dtype
            elif index[key] != dtype:
                index[key] = ""  # permanent conflict for this pass

    for _ in range(8):
        calls, witnesses = {}, {}
        for record in records:
            dtype = normalize_oracle_type(record.get("physical_type"))
            add_unique(calls, call_key(record), dtype)
            fp = record.get("subtree_fingerprint")
            if fp:
                add_unique(witnesses, (record.get("outer_node_id"), fp), dtype)
        changed = 0
        for record in records:
            inputs = record["slot_child_physical_types"]
            children = record.get("slot_child_node_ids") or []
            if len(inputs) != len(children):
                continue
            for i, fp in enumerate(record.get("slot_child_fingerprints") or []):
                if i < len(inputs) and not inputs[i] and fp:
                    dtype = witnesses.get((children[i], fp), "")
                    if dtype:
                        inputs[i] = dtype
                        stats["filled_input_slots"] += 1
                        changed += 1
            if not record.get("physical_type"):
                dtype = calls.get(call_key(record), "")
                source = "cached_outputs"
                if not dtype:
                    token = str(record.get("outer_token", "")).upper().removesuffix("()")
                    dtype = ORACLE_FIXED_RESULTS.get(token, "")
                    source = "fixed_outputs"
                if dtype:
                    record["physical_type"] = dtype
                    stats[source] += 1
                    changed += 1
        stats["passes"] += 1
        if not changed:
            break
    return result, dict(stats)


class _SourceParser(SimpleParser):
    """Preserve source spans only on a private, evidence-only AST."""
    def __init__(self, config):
        super().__init__(config)
        # Graph literals hide their sign inside int(-1); the executed SQL
        # contains a normal unary sign. Only this private parser accepts it.
        self.unary_operators = set(self.unary_operators) | {"+", "-"}

    def parse_atom(self):
        start = self.pos
        node = super().parse_atom()
        node.oracle_source_sql = " ".join(self.tokens[start:self.pos])
        return node

    def parse_unary(self):
        start = self.pos
        node = super().parse_unary()
        node.oracle_source_sql = " ".join(self.tokens[start:self.pos])
        return node

    def parse_expression(self, min_prec):
        start = self.pos
        node = super().parse_expression(min_prec)
        node.oracle_source_sql = " ".join(self.tokens[start:self.pos])
        return node


class OracleEvidenceTyper:
    # Per process, over its whole lifetime; failed descriptions are cached too.
    MAX_CALL_TYPES = 8192
    MAX_TABLES = 32
    MAX_AST_NODES = 256
    MAX_METADATA_SECONDS = 30.0

    def __init__(self, executor=None):
        self.executor = executor
        self.owns_executor = executor is None
        self.tables = {}
        self.call_types = {}
        self.stats = Counter()
        self.unavailable = False
        self.metadata_seconds = 0.0

    def close(self):
        if self.owns_executor and self.executor is not None:
            self.executor.close()

    def _executor(self):
        if self.unavailable:
            return None
        if self.executor is None:
            try:
                from RL.db_executor import create_db_executor
                self.executor = create_db_executor(autoconnect=True)
                # Dedicated metadata connection: bounded failure latency.
                self.executor._conn.call_timeout = 1000
            except Exception:
                self.unavailable = True
                self.stats["metadata_unavailable"] += 1
                return None
        return self.executor

    def _columns(self, table):
        if table not in self.tables:
            if len(self.tables) >= self.MAX_TABLES:
                return {}
            self.tables[table] = {}
            executor = self._executor()
            if executor is not None:
                result = executor.list_columns(table.strip('"'))
                self.stats["schema_queries"] += 1
                if result.success:
                    for row in result.rows or []:
                        row = {str(k).lower(): v for k, v in row.items()}
                        name = str(row.get("field", row.get("column_name", "")))
                        dtype = normalize_oracle_type(row.get("type", row.get("data_type", "")))
                        self.tables[table][name] = dtype
        return self.tables[table]

    def _describe(self, key, expression, table):
        if key in self.call_types:
            self.stats["description_cache_hits"] += 1
            return self.call_types[key]
        if len(self.call_types) >= self.MAX_CALL_TYPES:
            self.stats["description_budget_skips"] += 1
            return ""
        if self.metadata_seconds >= self.MAX_METADATA_SECONDS:
            self.stats["description_time_budget_skips"] += 1
            return ""
        self.call_types[key] = ""
        executor = self._executor()
        if executor is None:
            return ""
        started = time.monotonic()
        try:
            with executor._conn.cursor() as cursor:
                # cursor.parse describes a SELECT without evaluating values.
                cursor.parse(f"SELECT {expression} FROM {table} WHERE 1=0")
                self.stats["description_queries"] += 1
                info = cursor.description or []
                if info:
                    self.call_types[key] = normalize_oracle_type(
                        getattr(info[0][1], "name", info[0][1])
                    )
                    # A generic driver OBJECT tag does not identify the SQL
                    # object type. Never use it to join unrelated object types.
                    if self.call_types[key] == "object":
                        self.call_types[key] = ""
        except Exception:
            self.stats["description_failures"] += 1
        finally:
            self.metadata_seconds += time.monotonic() - started
        return self.call_types[key]
    def enrich(self, parse_result):
        sql = str(getattr(parse_result, "oracle_executed_sql", "") or "")
        # This is the exact SELECT envelope emitted by HeuristicCompletionClient.
        # Other envelopes stay legacy evidence; never align guessed SQL to an AST.
        match = re.fullmatch(
            r'\s*SELECT\s+1\s+FROM\s+("(?:[^"]|"")+"|[A-Za-z_][\w$#]*)'
            r'\s+WHERE\s+(.+)\s+FETCH\s+FIRST\s+1\s+ROW\s+ONLY\s*;?\s*',
            sql, re.I | re.S,
        )
        if not match or not getattr(parse_result, "success", False):
            self.stats["missing_sql_context"] += 1
            return parse_result
        from RL.predicate_structure import load_pkg_config, canonicalize_ast_tokens
        parser = _SourceParser(load_pkg_config())
        source = parser.parse(match[2])
        if source is None or parser.pos != len(parser.tokens):
            self.stats["unaligned_sql"] += 1
            return parse_result
        canonicalize_ast_tokens(source)
        original = parse_result.ast_root
        # Non-Boolean probes use the documented IS NOT NULL envelope.
        if (str(original.return_type).upper() != "T_BOOL"
                and source.token == "IS NOT NULL" and len(source.children) == 1):
            source = source.children[0]
        pairs = []

        def align(node, raw):
            children = list(getattr(node, "children", []) or [])
            raw_children = list(getattr(raw, "children", []) or [])
            # The graph abstracts int(-1) to one literal, whereas the SQL
            # parser represents its sign as a unary node. Preserve this one
            # leaf's real numeric type without changing either parser.
            if (not children and node.node_type != "column"
                    and re.fullmatch(r"[+-]?\s*\d+(?:\.\d*)?", raw.oracle_source_sql.strip())):
                raw_children = []
            if len(pairs) >= self.MAX_AST_NODES or len(children) != len(raw_children):
                return False
            if children and str(node.token).upper() != str(raw.token).upper():
                return False
            pairs.append((node, raw))
            return all(align(a, b) for a, b in zip(children, raw_children))

        result = copy.copy(parse_result)
        result.ast_root = copy.deepcopy(original)
        if not align(result.ast_root, source):
            self.stats["unaligned_sql"] += 1
            return parse_result
        columns = self._columns(match[1])
        for node, raw in reversed(pairs):
            expression = raw.oracle_source_sql
            text = expression.strip()
            dtype = ""
            children = node.children or []
            if not children:
                if str(node.token).startswith("COL_") or node.node_type == "column":
                    name = text[1:-1].replace('""', '"') if text.startswith('"') and text.endswith('"') else text.upper()
                    dtype = columns.get(name, "")
                elif text.upper() == "NULL":
                    dtype = "null"
                elif text.upper() in {"TRUE", "FALSE"}:
                    dtype = "boolean"
                elif re.fullmatch(r"[+-]?\s*\d+(?:\.\d*)?(?:[eE][+-]?\d+)?", text):
                    dtype = "number"
                elif re.fullmatch(r"'(?:[^']|'')*'", text, re.S):
                    dtype = "char"
                else:
                    dtype = self._describe((str(node.token), ()), expression, match[1])
            else:
                inputs = tuple(getattr(c, "oracle_physical_type", "") for c in children)
                if all(inputs):
                    dtype = self._describe((str(node.token), inputs), expression, match[1])
            node.oracle_physical_type = dtype
            node.oracle_local_binding = True
            self.stats["typed_nodes" if dtype else "untyped_nodes"] += 1
        self.stats["enriched_predicates"] += 1
        return result
