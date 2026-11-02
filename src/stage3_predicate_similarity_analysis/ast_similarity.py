from __future__ import annotations

import hashlib
import json
import zlib
from array import array
from collections import Counter, OrderedDict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


_AST_BLOCK_PREDICATES = 1_024
_AST_BLOCK_CACHE_BYTES = 32 * 1024 * 1024
_AST_TREE_CACHE_NODES = 50_000
_MAX_AST_NODES_PER_PREDICATE = 4_096
# Compacting parser wrappers can give one parent >255 children. The temporary
# index uses a two-byte count so otherwise valid wide expressions are not lost.
_NODE_RECORD_SIZE = 18  # 128-bit label fingerprint + uint16 little-endian count.
_DEPENDENCIES = None
# Index caches are bounded independently of the input count. They only reuse
# identical text/labels; literal values and column types are never anonymized.
_PARSE_CACHE_BYTES = 8 * 1024 * 1024
_PARSE_CACHE_ENTRIES = 8_192
# Reuse the same encoder instead of constructing one for every AST node.
# The JSON representation/fingerprint bytes remain exactly those of v3.
_JSON_ENCODE = json.JSONEncoder(ensure_ascii=False, separators=(",", ":")).encode


class AstDependencyError(RuntimeError):
    pass


def database_dialect_from_rl() -> str:
    """Read only the DBMS name; similarity analysis never connects to the DB."""
    from RL.config import DB_CONFIG

    dialect = str(getattr(DB_CONFIG, "db_type", "") or "").strip().lower()
    # SQL Server's SQLGlot dialect is named "tsql". Keep the project's
    # canonical DBMS names in manifests and select its local adapter below.
    if dialect == "postgres":
        dialect = "postgresql"
    if dialect not in {"mysql", "postgresql", "oracle", "sqlserver"}:
        raise NotImplementedError(
            "AST mode supports MySQL, PostgreSQL, Oracle and SQL Server; "
            f"RL.config.DB_CONFIG.db_type is {dialect!r}"
        )
    return dialect


def _load_dependencies():
    global _DEPENDENCIES
    if _DEPENDENCIES is not None:
        return _DEPENDENCIES
    try:
        import sqlglot
        from apted import APTED
        from apted.helpers import Tree
    except ImportError as exc:
        raise AstDependencyError(
            "AST mode requires the packages listed in "
            "stage3_predicate_similarity_analysis/requirements-ast.txt"
        ) from exc
    _DEPENDENCIES = (sqlglot, APTED, Tree)
    return _DEPENDENCIES


def _canonical_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return [type(value).__name__, value]
    if isinstance(value, (list, tuple)):
        return [type(value).__name__, [_canonical_value(item) for item in value]]
    if isinstance(value, dict):
        return [
            "dict",
            [
                [str(key), _canonical_value(item)]
                for key, item in sorted(value.items(), key=lambda entry: str(entry[0]))
            ],
        ]
    return [type(value).__name__, str(value)]


@lru_cache(maxsize=8_192)
def _label_fingerprint(label: str) -> bytes:
    return hashlib.blake2b(
        label.encode("utf-8"),
        digest_size=16,
        person=b"s3ast-label-v1",
    ).digest()


def _serialize_expression(expression: Any) -> bytes:
    sqlglot, _apted, _tree = _load_dependencies()
    expression_type = sqlglot.exp.Expression
    encoded = bytearray()

    # Keep the existing pre-compaction size/width guards. Even when a terminal
    # becomes one semantic node, a huge parser tree must not bypass the limits.
    source_nodes = 0
    pending = [expression]
    while pending:
        node = pending.pop()
        if not isinstance(node, expression_type):
            raise ValueError("unexpected non-expression in SQL AST")
        arguments = [value for value in node.args.values() if value is not None]
        if len(arguments) > 255:
            raise ValueError("AST node has more than 255 ordered children")
        source_nodes += 1 + len(arguments)
        for value in arguments:
            if isinstance(value, expression_type):
                pending.append(value)
            elif isinstance(value, (list, tuple)):
                if len(value) > 255:
                    raise ValueError("AST node has more than 255 ordered children")
                for item in value:
                    if isinstance(item, expression_type):
                        pending.append(item)
                    else:
                        source_nodes += 1
        if source_nodes > _MAX_AST_NODES_PER_PREDICATE:
            raise ValueError("AST node limit exceeded")

    def emit(label: str, child_count: int) -> None:
        if child_count > 65_535:
            raise ValueError("AST node has more than 65535 ordered children")
        fingerprint = _label_fingerprint(label)
        encoded.extend(fingerprint)
        encoded.extend(child_count.to_bytes(2, "little"))

    def atom_value(value: Any) -> Any:
        # A column/type/literal is ONE node. Its spelling, qualification,
        # quoting and type/value are data on that node, never extra TED nodes.
        if isinstance(value, expression_type):
            return [str(value.key), [
                [str(key), atom_value(item)]
                for key, item in value.args.items() if item is not None
            ]]
        if isinstance(value, (list, tuple)):
            return [atom_value(item) for item in value]
        return _canonical_value(value)

    def column_placeholder(node: Any) -> bool:
        # col(__TYPE__int)/col(INT_mock) is the project's column placeholder,
        # not a real function whose internals should contribute extra matches.
        # Only recognize its reserved type spellings; keep ordinary col(expr).
        if str(node.key) != "anonymous" or str(node.this).lower() != "col":
            return False
        values = node.args.get("expressions") or []
        if len(values) != 1:
            return False
        value = values[0]
        name = str(getattr(value, "name", "") or "")
        return (name.startswith("__TYPE__") or name.lower().endswith("_mock")
                or name.startswith(("COL_", "COLUMN_")))

    atomic_keys = {
        "column", "identifier", "literal", "datatype", "boolean", "null",
        "var", "hexstring", "bitstring", "bytestring", "rawstring", "national",
    }

    def visit(node: Any) -> None:
        if str(node.key) in atomic_keys or column_placeholder(node):
            emit(_JSON_ENCODE(atom_value(node)), 0)
            return

        # Operators/functions are the original AST nodes with ordered operand
        # children. Scalar properties (including an anonymous function's name)
        # belong to that node's label. Do not add ARG/ROLE nodes or child labels.
        attributes = []
        children = []
        for key, value in node.args.items():
            if value is None:
                continue
            if isinstance(value, expression_type):
                children.append(value)
            elif isinstance(value, (list, tuple)):
                scalar_items = []
                for item in value:
                    if isinstance(item, expression_type):
                        children.append(item)
                        scalar_items.append(None)
                    else:
                        scalar_items.append(_canonical_value(item))
                if not value or any(item is not None for item in scalar_items):
                    attributes.append([str(key), scalar_items])
            else:
                attributes.append([str(key), _canonical_value(value)])
        label = _JSON_ENCODE([str(node.key), attributes])
        emit(label, len(children))
        for child in children:
            visit(child)

    visit(expression)
    return bytes(encoded)


def _parser_dialect(dialect: str):
    sqlglot, _apted, _tree = _load_dependencies()
    normalized_dialect = str(dialect or "").strip().lower()
    if normalized_dialect == "mysql":
        return sqlglot.Dialect.get_or_raise("mysql")
    if normalized_dialect in {"postgresql", "postgres"}:
        # Only this branch loads the local PostgreSQL adapter.
        from .postgres_ast import PredicatePostgres

        return PredicatePostgres()
    if normalized_dialect == "oracle":
        from .oracle_ast import PredicateOracle

        return PredicateOracle()
    if normalized_dialect in {"sqlserver", "tsql"}:
        from .sqlserver_ast import PredicateTSQL

        return PredicateTSQL()
    raise NotImplementedError(
        f"AST parser dialect is not implemented yet: {normalized_dialect}"
    )


def _parse_and_serialize(predicate: str, dialect: str) -> bytes:
    sqlglot, _apted, _tree = _load_dependencies()
    expression = sqlglot.parse_one(
        predicate,
        read=_parser_dialect(dialect),
    )
    if expression is None:
        raise ValueError("SQL parser returned an empty expression")
    return _serialize_expression(expression)


class _IndexParser:
    """Reuse parser/tokenizer allocations within ONE sequential index builder.

    SQLGlot resets parser and tokenizer state for each parse/tokenize call. Keep
    parse_one for empty/multiple statements (its behavior varies by version).
    After any failed call, discard these objects before processing the next SQL.
    Only the most recent parser state is retained; input-sized token/tree lists
    never accumulate. finish() releases this state along with the parse cache.
    """
    def __init__(self, dialect: str):
        self.dialect_name = dialect
        self._dialect = self._parser = self._tokenizer = None

    def serialize(self, predicate: str) -> bytes:
        sqlglot, _apted, _tree = _load_dependencies()
        try:
            if self._parser is None:
                self._dialect = _parser_dialect(self.dialect_name)
                self._parser = self._dialect.parser()
                # SQLGlot 26 exposes a tokenizer instance; 28/30 expose a
                # factory. Both local DBMS adapters use the same index path.
                tokenizer = self._dialect.tokenizer
                self._tokenizer = tokenizer() if callable(tokenizer) else tokenizer
            expressions = self._parser.parse(self._tokenizer.tokenize(predicate), predicate)
            if len(expressions) != 1 or expressions[0] is None:
                return _parse_and_serialize(predicate, self.dialect_name)
            return _serialize_expression(expressions[0])
        except Exception:
            self.clear()
            raise

    def clear(self):
        self._dialect = self._parser = self._tokenizer = None


@dataclass(frozen=True)
class AstTreeStore:
    path: Path
    record_offsets: array
    node_counts: array
    parse_status: array
    tree_fingerprints: array
    block_index: Tuple[Tuple[int, int, int], ...]
    parse_error_types: Dict[str, int]

    def __post_init__(self) -> None:
        object.__setattr__(self, "_file", None)
        object.__setattr__(self, "_block_cache", OrderedDict())
        object.__setattr__(self, "_block_cache_bytes", 0)
        object.__setattr__(self, "_tree_cache", OrderedDict())
        object.__setattr__(self, "_tree_cache_nodes", 0)
        object.__setattr__(self, "_prepared_cache", OrderedDict())
        object.__setattr__(self, "_prepared_cache_nodes", 0)

    def close(self) -> None:
        handle = getattr(self, "_file", None)
        if handle is not None:
            handle.close()
            object.__setattr__(self, "_file", None)
        getattr(self, "_block_cache").clear()
        getattr(self, "_tree_cache").clear()
        getattr(self, "_prepared_cache").clear()
        object.__setattr__(self, "_block_cache_bytes", 0)
        object.__setattr__(self, "_tree_cache_nodes", 0)
        object.__setattr__(self, "_prepared_cache_nodes", 0)

    def _load_block(self, block_id: int) -> bytes:
        cache = getattr(self, "_block_cache")
        if block_id in cache:
            raw = cache.pop(block_id)
            cache[block_id] = raw
            return raw
        file_offset, compressed_size, raw_size = self.block_index[block_id]
        handle = getattr(self, "_file", None)
        if handle is None:
            handle = open(self.path, "rb")
            object.__setattr__(self, "_file", handle)
        handle.seek(file_offset)
        compressed = handle.read(compressed_size)
        raw = zlib.decompress(compressed)
        if len(raw) != raw_size:
            raise ValueError("corrupt compressed AST block")
        if raw_size <= _AST_BLOCK_CACHE_BYTES:
            cache[block_id] = raw
            object.__setattr__(
                self,
                "_block_cache_bytes",
                int(getattr(self, "_block_cache_bytes")) + raw_size,
            )
            while int(getattr(self, "_block_cache_bytes")) > _AST_BLOCK_CACHE_BYTES:
                _old_id, old_raw = cache.popitem(last=False)
                object.__setattr__(
                    self,
                    "_block_cache_bytes",
                    int(getattr(self, "_block_cache_bytes")) - len(old_raw),
                )
        return raw

    def encoded_for(self, predicate_id: int) -> bytes:
        block_id, local_id = divmod(predicate_id, _AST_BLOCK_PREDICATES)
        raw = self._load_block(block_id)
        start = int(self.record_offsets[predicate_id])
        end = (
            int(self.record_offsets[predicate_id + 1])
            if local_id + 1 < _AST_BLOCK_PREDICATES and predicate_id + 1 < len(self.record_offsets)
            else int(self.block_index[block_id][2])
        )
        return raw[start:end]

    def prepared_for(self, predicate_id: int):
        """Decode each active tree once for repeated window/LSH comparisons."""
        from .ast_distance import PreparedTree

        if not self.parse_status[predicate_id]:
            return None
        cache = self._prepared_cache
        if predicate_id in cache:
            cache.move_to_end(predicate_id)
            return cache[predicate_id]
        prepared = PreparedTree(self.encoded_for(predicate_id))
        cache[predicate_id] = prepared
        size = self._prepared_cache_nodes + prepared.size
        while size > _AST_TREE_CACHE_NODES:
            _, old = cache.popitem(last=False)
            size -= old.size
        object.__setattr__(self, "_prepared_cache_nodes", size)
        return prepared

    def tree_for(self, predicate_id: int):
        predicate_id = int(predicate_id)
        if not self.parse_status[predicate_id]:
            return None
        tree_cache = getattr(self, "_tree_cache")
        if predicate_id in tree_cache:
            cached_tree, node_count = tree_cache.pop(predicate_id)
            tree_cache[predicate_id] = (cached_tree, node_count)
            return cached_tree

        _sqlglot, _apted, Tree = _load_dependencies()
        block_id, local_id = divmod(predicate_id, _AST_BLOCK_PREDICATES)
        raw = self._load_block(block_id)
        start = int(self.record_offsets[predicate_id])
        if local_id + 1 < _AST_BLOCK_PREDICATES and predicate_id + 1 < len(self.record_offsets):
            end = int(self.record_offsets[predicate_id + 1])
        else:
            end = int(self.block_index[block_id][2])
        view = memoryview(raw)
        cursor = start

        def read_node():
            nonlocal cursor
            if cursor + _NODE_RECORD_SIZE > end:
                raise ValueError("corrupt serialized AST record")
            label = bytes(view[cursor : cursor + 16])
            child_count = int.from_bytes(view[cursor + 16:cursor + 18], "little")
            cursor += _NODE_RECORD_SIZE
            children = [read_node() for _ in range(child_count)]
            return Tree(label, *children)

        try:
            tree = read_node()
            if cursor != end:
                raise ValueError("serialized AST record has trailing data")
        finally:
            view.release()

        node_count = int(self.node_counts[predicate_id])
        if node_count <= _AST_TREE_CACHE_NODES:
            tree_cache[predicate_id] = (tree, node_count)
            object.__setattr__(
                self,
                "_tree_cache_nodes",
                int(getattr(self, "_tree_cache_nodes")) + node_count,
            )
            while int(getattr(self, "_tree_cache_nodes")) > _AST_TREE_CACHE_NODES:
                _old_id, (_old_tree, old_count) = tree_cache.popitem(last=False)
                object.__setattr__(
                    self,
                    "_tree_cache_nodes",
                    int(getattr(self, "_tree_cache_nodes")) - old_count,
                )
        return tree


class AstTreeIndexBuilder:
    def __init__(self, path: Path, dialect: str):
        self.path = Path(path)
        self.dialect = str(dialect)
        self._buffer = bytearray()
        self._record_offsets = array("Q")
        self._node_counts = array("I")
        self._parse_status = array("B")
        self._tree_fingerprints = array("Q")
        self._block_index: List[Tuple[int, int, int]] = []
        self._block_predicate_count = 0
        self._predicate_count = 0
        self._error_types: Counter[str] = Counter()
        self._finished = False
        self._parse_cache = OrderedDict()
        self._parse_cache_bytes = 0
        self._parser = _IndexParser(self.dialect)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_bytes(b"")

    def add(self, predicate: str) -> bool:
        try:
            cached = self._parse_cache.get(predicate)
            if cached is not None:
                self._parse_cache.move_to_end(predicate)
                encoded = cached[0]
            else:
                encoded = self._parser.serialize(predicate)
                cost = 4 * len(predicate) + len(encoded) + 256
                if cost <= _PARSE_CACHE_BYTES:
                    self._parse_cache[predicate] = (encoded, cost)
                    self._parse_cache_bytes += cost
                    while (self._parse_cache_bytes > _PARSE_CACHE_BYTES
                           or len(self._parse_cache) > _PARSE_CACHE_ENTRIES):
                        _, (_, old_cost) = self._parse_cache.popitem(last=False)
                        self._parse_cache_bytes -= old_cost
        except Exception as exc:
            return self.add_encoded(None, type(exc).__name__)
        return self.add_encoded(encoded)

    def add_encoded(self, encoded: Optional[bytes], error_type: Optional[str] = None) -> bool:
        """Append one ordered worker result; parser failures keep their slot.

        Only the parent owns the compressed file and predicate numbering. Worker
        processes return bytes/labels, never write index or source files.
        """
        self._record_offsets.append(len(self._buffer))
        try:
            if encoded is None:
                raise ValueError("missing parsed AST record")
            if len(encoded) % _NODE_RECORD_SIZE:
                raise ValueError("serialized AST has invalid record length")
            node_count = len(encoded) // _NODE_RECORD_SIZE
            if not node_count:
                raise ValueError("serialized AST is empty")
            self._buffer.extend(encoded)
            self._node_counts.append(node_count)
            self._parse_status.append(1)
            self._tree_fingerprints.append(
                int.from_bytes(
                    hashlib.blake2b(
                        encoded,
                        digest_size=8,
                        person=b"s3ast-tree-v1",
                    ).digest(),
                    "little",
                )
            )
            success = True
        except Exception as exc:
            self._node_counts.append(0)
            self._parse_status.append(0)
            self._tree_fingerprints.append(0)
            self._error_types[error_type or type(exc).__name__] += 1
            success = False

        self._predicate_count += 1
        self._block_predicate_count += 1
        if self._block_predicate_count >= _AST_BLOCK_PREDICATES:
            self._flush_block()
        return success

    def _flush_block(self) -> None:
        if not self._block_predicate_count:
            return
        raw = bytes(self._buffer)
        compressed = zlib.compress(raw, level=1)
        with open(self.path, "ab") as handle:
            file_offset = handle.tell()
            handle.write(compressed)
        self._block_index.append((file_offset, len(compressed), len(raw)))
        self._buffer.clear()
        self._block_predicate_count = 0

    def finish(self) -> AstTreeStore:
        if self._finished:
            raise RuntimeError("AST tree index has already been finalized")
        self._flush_block()
        self._finished = True
        self._parse_cache.clear()
        self._parse_cache_bytes = 0
        self._parser.clear()
        return AstTreeStore(
            path=self.path,
            record_offsets=self._record_offsets,
            node_counts=self._node_counts,
            parse_status=self._parse_status,
            tree_fingerprints=self._tree_fingerprints,
            block_index=tuple(self._block_index),
            parse_error_types=dict(self._error_types),
        )


def tree_edit_distance(left, right):
    from .ast_distance import apted_calculator

    calculator = apted_calculator(left, right)
    return int(calculator.compute_edit_distance())


def dependency_versions() -> Dict[str, str]:
    sqlglot, _APTED, _Tree = _load_dependencies()
    try:
        from importlib.metadata import version

        apted_version = version("apted")
    except Exception:
        apted_version = "unknown"
    return {"sqlglot": str(sqlglot.__version__), "apted": apted_version}
