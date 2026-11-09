from __future__ import annotations


from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
import json
import re

from RL.config import DB_CONFIG
from pkg.parser import (
    ASTNode, SimpleParser, tokenize_sqlserver, _TSQL_DSL_TYPES, _TSQL_IDENTIFIER,
)
from pkg.type_inference import TypeInferencer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PKG_DIR = PROJECT_ROOT / "pkg"


@dataclass(frozen=True)
class ExtractedEdge:
    source_node_id: str
    source_token: str
    source_return_type: str
    slot_id: int

    target_node_id: str
    target_token: str
    target_return_type: str
    target_node_type: str
    target_is_terminal: bool


@dataclass
class ParsedPredicateResult:
    original_predicate: str
    normalized_predicate: str
    abstract_predicate: str

    success: bool
    ast_root: Optional[Any] = None

    edges: List[ExtractedEdge] = field(default_factory=list)
    operators: Set[str] = field(default_factory=set)
    node_ids: Set[str] = field(default_factory=set)

    error: Optional[str] = None
    warnings: List[str] = field(default_factory=list)


_DATATYPE_MAPPING_CACHE: Optional[Dict[str, List[str]]] = None
_PKG_CONFIG_CACHE: Optional[Dict[str, Any]] = None


def load_pkg_config() -> Optional[Dict[str, Any]]:
    global _PKG_CONFIG_CACHE

    if _PKG_CONFIG_CACHE is not None:
        return _PKG_CONFIG_CACHE

    db_type = getattr(DB_CONFIG, "db_type", "mysql").lower()

    candidate_paths = [
        PKG_DIR / "configuration" / f"{db_type}_rules.json",
        PKG_DIR / "configuration" / "mysql_rules.json",
    ]

    for config_path in candidate_paths:
        if config_path.exists():
            try:
                with config_path.open("r", encoding="utf-8") as f:
                    _PKG_CONFIG_CACHE = json.load(f)
                return _PKG_CONFIG_CACHE
            except Exception as exc:
                print(
                    f'[predicate_structure] Failed to parse configuration: {config_path} | {exc}',
                    flush=True,
                )
                _PKG_CONFIG_CACHE = None
                return None

    _PKG_CONFIG_CACHE = None
    return None


def get_non_function_keywords(config: Optional[Dict[str, Any]] = None) -> Set[str]:
    cfg = config if config is not None else load_pkg_config()

    default_keywords = {
        "NOT",
        "AND",
        "OR",
        "XOR",
        "BETWEEN",
        "LIKE",
        "IN",
        "IS",
        "REGEXP",
    }

    if not isinstance(cfg, dict):
        return default_keywords

    configured = cfg.get("non_function_keywords")
    if not configured:
        return default_keywords

    if isinstance(configured, list):
        return {str(item).upper() for item in configured}

    return default_keywords


def _get_datatype_mapping() -> Dict[str, List[str]]:
    global _DATATYPE_MAPPING_CACHE

    if _DATATYPE_MAPPING_CACHE is not None:
        return _DATATYPE_MAPPING_CACHE

    db_type = getattr(DB_CONFIG, "db_type", "mysql").lower()
    config_path = PROJECT_ROOT / "RL" / "datatype" / f"{db_type}.json"

    if config_path.exists():
        try:
            with config_path.open("r", encoding="utf-8") as f:
                data = json.load(f)

            mapping: Dict[str, List[str]] = {}
            for key, value in data.items():
                if isinstance(value, list):
                    mapping[str(key).lower()] = [str(v).lower() for v in value]
                else:
                    mapping[str(key).lower()] = [str(value).lower()]
            _DATATYPE_MAPPING_CACHE = mapping
            return mapping
        except Exception as exc:
            print(
                f'[predicate_structure] Failed to read type mappings: {config_path} | {exc}',
                flush=True,
            )

    _DATATYPE_MAPPING_CACHE = {
        "int": ["int", "integer", "hex", "bit"],
        "float": ["float", "double", "decimal", "numeric"],
        "string": ["char", "varchar", "text", "string"],
        "date": ["date", "time", "timestamp"],
    }
    return _DATATYPE_MAPPING_CACHE


def _uses_sqlserver(config=None) -> bool:
    if config is not None:
        return str((config.get('sql_render', {}) or {}).get('version', '')).startswith('sqlserver_')
    return str(getattr(DB_CONFIG, 'db_type', '') or '').strip().lower() == 'sqlserver'


def normalize_predicate(predicate: str, config=None) -> str:
    if predicate is None:
        return ""

    text = str(predicate).strip()
    if _uses_sqlserver(config):
        try:
            return ' '.join(tokenize_sqlserver(text))
        except ValueError:
            return text

    text = re.sub(r"\s+", " ", text)

    def _compact_function_space(match: re.Match) -> str:
        name = match.group(1)
        keyword_blacklist = get_non_function_keywords()
        if name.upper() in keyword_blacklist:
            return f"{name} ("
        return f"{name}("

    text = re.sub(
        r"\b([A-Za-z_][A-Za-z0-9_]*)\s+\(",
        _compact_function_space,
        text,
    )

    text = re.sub(r"(?i)\bNOT\s*\(", "NOT (", text)

    return text.strip()


def _abstract_sqlserver_predicate(predicate: str, config: dict) -> str:
    tokens = tokenize_sqlserver(predicate)
    pairs, stack = {}, []
    for i, token in enumerate(tokens):
        if token == '(':
            stack.append(i)
        elif token == ')':
            if not stack:
                raise ValueError('Unbalanced SQL Server predicate')
            pairs[stack.pop()] = i
    if stack:
        raise ValueError('Unbalanced SQL Server predicate')

    wrapper_types = {
        _TSQL_DSL_TYPES[t.lower()] for i, t in enumerate(tokens[:-1])
        if tokens[i + 1] == '(' and t.lower() in _TSQL_DSL_TYPES
    }
    req_type = next((t for t in ('INT', 'FLOAT', 'STRING', 'BINARY', 'DATE', 'UNIQUEIDENTIFIER')
                     if t in wrapper_types), None)
    if req_type is None:
        req_type = 'INT' if any(t in {'+', '-', '*', '/', '%', '>>', '<<', '&', '|', '^'} for t in tokens) else 'STRING'
    reserved = set(config.get('forbidden_clauses', [])) | set(config.get('non_function_keywords', []))
    reserved.update(config.get('precedence', {}))
    reserved.update((config.get('candidate_bitmap', {}) or {}).get('terminal_tokens', {}))
    functions = set(config.get('definitions', {}))
    functions.update((config.get('candidate_bitmap', {}) or {}).get('callable_terminal_tokens', []))

    def render(lo, hi):
        pieces, i = [], lo
        while i < hi:
            token = tokens[i]
            name = token.lower()
            is_call = i + 1 < hi and tokens[i + 1] == '('
            if (is_call and re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', token)
                    and token.upper() not in reserved and name != 'col' and name not in _TSQL_DSL_TYPES
                    and token.upper() + '()' not in functions):
                close = pairs[i + 1]
                pieces.extend(tokens[i:close + 1])
                i = close + 1
                continue
            if i + 1 < hi and tokens[i + 1] == '(' and (name == 'col' or name in _TSQL_DSL_TYPES):
                close = pairs[i + 1]
                if name == 'col':
                    if close != i + 3 or not _TSQL_IDENTIFIER.fullmatch(tokens[i + 2]):
                        raise ValueError('col() requires one SQL Server column identifier')
                    pieces.append('COL_' + req_type)
                else:
                    inner = render(i + 2, close)
                    if not inner:
                        raise ValueError('Empty SQL Server DSL wrapper')

                    pieces.append(token + '(' + inner + ')')
                i = close + 1
                continue
            if (not is_call and _TSQL_IDENTIFIER.fullmatch(token) and not token.startswith('@')
                    and token.upper() not in reserved
                    and not re.fullmatch(r'(?:COL_|VALUE_|TYPICAL_|SINGULAR_|HOLE_)[A-Z0-9_]+', token.upper())):
                token = 'COL_' + req_type
            pieces.append(token)
            i += 1
        return ' '.join(pieces)

    return render(0, len(tokens))


def abstract_predicate_for_graph(predicate: str, config=None) -> str:
    abs_str = normalize_predicate(predicate, config=config)
    if _uses_sqlserver(config):
        return _abstract_sqlserver_predicate(abs_str, config if config is not None else load_pkg_config())
    lower_pred = abs_str.lower()
    mapping = _get_datatype_mapping()

    req_type = "string"
    explicit_type_found = False

    for category in mapping.keys():
        if f"{category}(" in lower_pred:
            req_type = category
            explicit_type_found = True
            break

    if not explicit_type_found:
        lower_without_strings = re.sub(r"(['\"]).*?\1", "", lower_pred)

        if any(
                op in lower_without_strings
                for op in ["+", "-", "*", "/", "%", ">>", "<<", "&", "|", "^"]
        ):
            req_type = "int"

    if req_type == "string" and ("decimal(" in lower_pred or "numeric(" in lower_pred):
        req_type = "float" if "float" in mapping else req_type

    abs_str = re.sub(
        r"(?i)\bcol\([^)]+\)",
        f"col_{req_type}",
        abs_str,
    )

    for category, keywords in mapping.items():
        kw_list = sorted(set([category] + list(keywords)), key=len, reverse=True)
        pattern_str = "|".join(re.escape(kw) for kw in kw_list)

        abs_str = re.sub(
            rf"(?i)\b(?:{pattern_str})\([^)]+\)",
            f"typical_{category}",
            abs_str,
        )

    abs_str = re.sub(
        r"(?i)\b(?:decimal|numeric)\([^)]+\)",
        "typical_float",
        abs_str,
    )

    return abs_str


def _node_return_type(node: Any) -> str:
    ret = getattr(node, "return_type", None)
    return str(ret) if ret else "UNKNOWN"


def _node_id(node: Any) -> str:
    token = str(getattr(node, "token", "UNKNOWN"))
    return_type = _node_return_type(node)
    return f"{token}::{return_type}"


def extract_edges_from_ast(ast_root: Any) -> List[ExtractedEdge]:
    edges: List[ExtractedEdge] = []

    def visit(node: Any) -> None:
        children = getattr(node, "children", []) or []

        source_token = str(getattr(node, "token", "UNKNOWN"))
        source_return_type = _node_return_type(node)
        source_node_id = _node_id(node)

        for child_index, child in enumerate(children):
            slot_id = 1 if source_token.upper() in {"IN", "NOT IN"} and child_index >= 1 else child_index
            target_token = str(getattr(child, "token", "UNKNOWN"))
            target_return_type = _node_return_type(child)
            target_node_type = str(getattr(child, "node_type", "unknown"))
            target_node_id = _node_id(child)

            edge = ExtractedEdge(
                source_node_id=source_node_id,
                source_token=source_token,
                source_return_type=source_return_type,
                slot_id=int(slot_id),
                target_node_id=target_node_id,
                target_token=target_token,
                target_return_type=target_return_type,
                target_node_type=target_node_type,
                target_is_terminal=(target_node_type != "operator"),
            )
            edges.append(edge)

            visit(child)

    if ast_root is not None:
        visit(ast_root)

    return edges


def canonicalize_ast_tokens(ast_root: Any) -> None:
    if ast_root is None:
        return

    def visit(node: Any) -> None:
        token = str(getattr(node, "token", ""))

        normalized = re.sub(r"\s+", "", token).upper()

        if normalized == "NOT()":
            node.token = "NOT"
        elif re.match(r"^[A-Z_][A-Z0-9_]*\(\)$", normalized):
            node.token = normalized

        for child in getattr(node, "children", []) or []:
            visit(child)

    visit(ast_root)


def canonicalize_mysql_is_syntax(ast_root: Any) -> None:
    if ast_root is None or str(getattr(DB_CONFIG, "db_type", "")).lower() != "mysql":
        return

    is_rhs_tokens = {
        "IS TRUE": "TRUE",
        "IS FALSE": "FALSE",
        "IS NULL": "NULL",
    }

    def visit(node: Any) -> None:
        children = list(getattr(node, "children", []) or [])
        for child in children:
            visit(child)

        token = re.sub(r"\s+", " ", str(getattr(node, "token", "")).strip()).upper()
        if len(children) != 1:
            return

        if token in is_rhs_tokens:
            node.token = "IS"
            node.node_type = "operator"
            node.children = [children[0], ASTNode(is_rhs_tokens[token], "literal")]
        elif token == "IS NOT NULL":
            is_node = ASTNode("IS", "operator", [children[0], ASTNode("NULL", "literal")])
            node.token = "NOT"
            node.node_type = "operator"
            node.children = [is_node]

    visit(ast_root)


def collect_operators_from_ast(ast_root: Any) -> Set[str]:
    operators: Set[str] = set()

    def visit(node: Any) -> None:
        if node is None:
            return

        node_type = str(getattr(node, "node_type", ""))
        if node_type == "operator":
            operators.add(_node_id(node))

        for child in getattr(node, "children", []) or []:
            visit(child)

    visit(ast_root)
    return operators


def collect_node_ids_from_ast(ast_root: Any) -> Set[str]:
    node_ids: Set[str] = set()

    def visit(node: Any) -> None:
        if node is None:
            return

        node_ids.add(_node_id(node))

        for child in getattr(node, "children", []) or []:
            visit(child)

    visit(ast_root)
    return node_ids


def validate_ast_for_graph(ast_root: Any) -> Optional[str]:
    if ast_root is None:
        return "empty AST"

    def visit(node: Any) -> Optional[str]:
        if node is None:
            return None

        token = str(getattr(node, "token", ""))
        token_upper = token.upper()
        if "??" in token or "HOLE" in token_upper:
            return "forbidden placeholder node"
        if token_upper.startswith("VALUE_") and "." in token:
            return "abnormal value node"

        for child in getattr(node, "children", []) or []:
            reason = visit(child)
            if reason:
                return reason
        return None

    return visit(ast_root)


class PredicateStructureAnalyzer:
    def __init__(self, config: Optional[Dict[str, Any]] = None, strict_consumption: bool = False):
        self.config = config if config is not None else load_pkg_config()
        self.parser = SimpleParser(self.config)
        self.inferencer = TypeInferencer(self.config)
        self.strict_consumption = strict_consumption

    def analyze(self, predicate: str, expected_type: str = "T_BOOL") -> ParsedPredicateResult:
        original = "" if predicate is None else str(predicate)
        normalized = normalize_predicate(original, config=self.config) if self.parser.is_sqlserver else normalize_predicate(original)
        abstracted = ""

        result = ParsedPredicateResult(
            original_predicate=original,
            normalized_predicate=normalized,
            abstract_predicate="",
            success=False,
        )

        if not normalized:
            result.error = "empty predicate after normalization"
            return result

        try:
            abstracted = (abstract_predicate_for_graph(normalized, config=self.config)
                          if self.parser.is_sqlserver else abstract_predicate_for_graph(normalized))
            result.abstract_predicate = abstracted

            ast_root = self.parser.parse(abstracted)
            if ast_root is None:
                result.error = f"parser returned None for abstract predicate: {abstracted}"
                return result

            canonicalize_ast_tokens(ast_root)

            canonicalize_mysql_is_syntax(ast_root)

            remaining_tokens = []
            try:
                tokens = getattr(self.parser, "tokens", [])
                pos = int(getattr(self.parser, "pos", 0))
                if pos < len(tokens):
                    remaining_tokens = tokens[pos:]
            except Exception:
                remaining_tokens = []

            if remaining_tokens:
                msg = f"parser did not consume all tokens: {remaining_tokens}"
                if self.strict_consumption:
                    result.error = msg
                    return result
                result.warnings.append(msg)

            self.inferencer.infer(ast_root, expected_type=expected_type)

            invalid_reason = validate_ast_for_graph(ast_root)
            if invalid_reason:
                result.error = invalid_reason
                result.success = False
                return result

            result.ast_root = ast_root
            result.edges = extract_edges_from_ast(ast_root)
            result.operators = collect_operators_from_ast(ast_root)
            result.node_ids = collect_node_ids_from_ast(ast_root)
            result.success = True
            return result

        except Exception as exc:
            result.abstract_predicate = abstracted
            result.error = str(exc)
            result.success = False
            return result


def analyze_predicate_structure(
    predicate: str,
    expected_type: str = "T_BOOL",
    config: Optional[Dict[str, Any]] = None,
    strict_consumption: bool = False,
) -> ParsedPredicateResult:
    analyzer = PredicateStructureAnalyzer(
        config=config,
        strict_consumption=strict_consumption,
    )
    return analyzer.analyze(predicate, expected_type=expected_type)


if __name__ == "__main__":
    samples = [
        "col(age) > int(10)",
        "(col(age) + int(1)) > int(10)",
        "NOT (col(name) LIKE string('%abc%'))",
        "NOT(col(name) LIKE string('%abc%'))",
        "col(age) BETWEEN int(18) AND int(60)",
    ]

    analyzer = PredicateStructureAnalyzer()

    for sample in samples:
        print("=" * 80)
        print("INPUT:", sample)
        res = analyzer.analyze(sample)
        print("SUCCESS:", res.success)
        print("NORMALIZED:", res.normalized_predicate)
        print("ABSTRACT:", res.abstract_predicate)
        print("ERROR:", res.error)
        print("WARNINGS:", res.warnings)
        print("OPERATORS:", sorted(res.operators))
        print("EDGES:")
        for e in res.edges:
            print(f"  {e.source_node_id} slot{e.slot_id} -> {e.target_node_id}")

    print("=" * 80)
    print("[SelfTest Hint]")
    print('Please check:')
    print('1. Check whether NOT (...) is parsed as NOT::T_BOOL instead of NOT()::T_BOOL.')
    print("2. Check whether col(name) LIKE string('%abc%') is abstracted as COL_STRING::STRING instead of COL_INT::INT.")
