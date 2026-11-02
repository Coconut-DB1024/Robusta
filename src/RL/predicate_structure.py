from __future__ import annotations

"""
predicate_structure.py

本模块是第二阶段 RL 与谓词知识图之间的统一结构分析接口。

它只负责：
1. 规范化谓词字符串；
2. 将类型标记谓词抽象成图构建器可理解的形式；
3. 调用现有 SimpleParser 解析 AST；
4. 调用现有 TypeInferencer 推断节点返回类型；
5. 从 Typed AST 中抽取“父节点 slot -> 子节点”的边；
6. 返回统一的 ParsedPredicateResult。

它不负责：
1. 判断 SQL 是否能在数据库执行；
2. 判断数据库返回结果是否正确；
3. 推导某个数据库的理论候选边全集；
4. 替 RL 生成或修复谓词。
"""

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
    """
    从 Typed AST 中抽取出来的一条结构边。

    含义：
        source_node_id 的第 slot_id 个参数位，连接到 target_node_id。

    例如：
        (col(age) + int(1)) > int(10)

    可能抽出：
        >::T_BOOL slot0 -> +::T_CMP
        +::T_CMP slot0 -> COL_INT::INT
        +::T_CMP slot1 -> VALUE_TYPICAL_INT::INT
        >::T_BOOL slot1 -> VALUE_TYPICAL_INT::INT
    """

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
    """
    谓词结构分析的统一返回结果。
    """

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
    """
    加载 pkg/configuration/<db_type>_rules.json。

    如果当前数据库没有专门配置，则回退到 mysql_rules.json。
    当前项目主要是 MySQL，因此这一步与旧版 pkg_manager.load_pkg_config() 行为兼容。
    """
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
                    f"[predicate_structure] 配置文件解析失败: {config_path} | {exc}",
                    flush=True,
                )
                _PKG_CONFIG_CACHE = None
                return None

    _PKG_CONFIG_CACHE = None
    return None


def get_non_function_keywords(config: Optional[Dict[str, Any]] = None) -> Set[str]:
    """
    获取“不能被当成普通函数名压缩空格”的 SQL 关键字集合。

    背景：
        normalize_predicate() 会把 LENGTH (x) 规范化成 LENGTH(x)，
        但不能把 NOT (x) 规范化成 NOT(x)，否则 parser 会把 NOT 当成函数调用。

    设计：
        1. 优先从 configuration/<db>_rules.json 中读取 non_function_keywords；
        2. 如果配置文件没有该字段，则使用保守兜底集合；
        3. 这不是候选边规则，也不是数据库弱类型规则，只是 parser 前的格式规范化辅助。
    """
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
    """
    读取 RL/datatype/<db_type>.json 中的数据类型映射。

    这个映射只用于把 int(...), string(...), float(...) 等类型标记
    抽象成 typical_int / typical_string / typical_float 等内部图节点。

    注意：
        这不是候选边规则；
        也不是数据库弱类型规则；
        只是项目内部 DSL 的类型 wrapper 映射。
    """
    global _DATATYPE_MAPPING_CACHE

    if _DATATYPE_MAPPING_CACHE is not None:
        return _DATATYPE_MAPPING_CACHE

    db_type = getattr(DB_CONFIG, "db_type", "mysql").lower()
    config_path = PROJECT_ROOT / "RL" / "datatype" / f"{db_type}.json"

    if config_path.exists():
        try:
            with config_path.open("r", encoding="utf-8") as f:
                data = json.load(f)

            # 防御：确保 value 都是 list[str]
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
                f"[predicate_structure] 类型映射读取失败: {config_path} | {exc}",
                flush=True,
            )

    # 兜底映射，保持与当前 pkg_manager.py 大体兼容。
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
    """
    对 RL decoder 生成的谓词做轻量规范化。

    当前只做低风险处理：
    1. 去掉首尾空白；
    2. 合并多余空白；
    3. 统一普通函数括号前的空白；
    4. 保留 NOT 与后续括号之间的空格，避免 NOT 被 parser 当成函数调用。

    不做强结构修复，不替 RL 构造谓词。
    """
    if predicate is None:
        return ""

    text = str(predicate).strip()
    if _uses_sqlserver(config):
        try:
            # 只规范 token 之间的空白，保留 Unicode 字符串、转义引号及限定名称。
            return ' '.join(tokenize_sqlserver(text))
        except ValueError:
            # 不修复错误输入；由后续解析入口统一返回失败。
            return text

    # 合并连续空白。
    text = re.sub(r"\s+", " ", text)

    # 统一普通函数括号前的空白：LENGTH (x) -> LENGTH(x)
    # 但必须排除 NOT / AND / OR / BETWEEN / LIKE / IN 等 SQL 关键字，
    # 否则 NOT (x) 会被改成 NOT(x)，进而被 parser 当成函数 NOT()。
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

    # 最后再强制把 NOT( / NOT   ( 统一成 NOT (。
    # 这样 parser 会把 NOT 当作一元运算符，而不是函数名。
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
                # 未纳入的特殊函数保持参数原文，由 parser 拒绝；不抽象类型/日期单位。
                close = pairs[i + 1]
                pieces.extend(tokens[i:close + 1])
                i = close + 1
                continue
            if i + 1 < hi and tokens[i + 1] == '(' and (name == 'col' or name in _TSQL_DSL_TYPES):
                close = pairs[i + 1]
                if name == 'col':
                    # col 的内容必须是一个完整标识符，不能由右括号正则截断。
                    if close != i + 3 or not _TSQL_IDENTIFIER.fullmatch(tokens[i + 2]):
                        raise ValueError('col() requires one SQL Server column identifier')
                    pieces.append('COL_' + req_type)
                else:
                    inner = render(i + 2, close)
                    if not inner:
                        raise ValueError('Empty SQL Server DSL wrapper')
                    # 让配置驱动的 parser 处理包装器；不以正则抹掉 ABS 等内部子树。
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
    """
    将类型标记谓词抽象成知识图 parser 更容易处理的形式。

    示例：
        col(salary) > int(1000)
    可能变成：
        col_int > typical_int

    说明：
        这里沿用现有 pkg_manager.py 的思想，但放到独立模块中，
        供 env_sql_dependency.py 和 pkg_manager.py 共同调用。
    """
    abs_str = normalize_predicate(predicate, config=config)
    if _uses_sqlserver(config):
        return _abstract_sqlserver_predicate(abs_str, config if config is not None else load_pkg_config())
    lower_pred = abs_str.lower()
    mapping = _get_datatype_mapping()

    # 1. 粗略推断 col(...) 应抽象成哪类列。
    #
    # 当前项目的 col(...) 本身不带类型，如 col(salary)，因此旧代码根据
    # 谓词里出现的 int(...), string(...), float(...) 等 wrapper 来猜列类型。
    # 这个策略不是完美的，但先保持兼容，后续再逐步增强。
    req_type = "string"
    explicit_type_found = False

    for category in mapping.keys():
        if f"{category}(" in lower_pred:
            req_type = category
            explicit_type_found = True
            break

    # 如果没有显式类型 wrapper，但出现算术 / 位运算，倾向于数值列。
    # 注意：
    #   只有在没有发现 int(...), string(...), float(...) 等显式 wrapper 时，
    #   才启用这个兜底逻辑。
    #   否则 string('%abc%') 里的 % 会被误判成取模运算符。
    if not explicit_type_found:
        lower_without_strings = re.sub(r"(['\"]).*?\1", "", lower_pred)

        if any(
                op in lower_without_strings
                for op in ["+", "-", "*", "/", "%", ">>", "<<", "&", "|", "^"]
        ):
            req_type = "int"

    # decimal / numeric 兜底到 float。
    if req_type == "string" and ("decimal(" in lower_pred or "numeric(" in lower_pred):
        req_type = "float" if "float" in mapping else req_type

    # 2. 替换 col(...)。
    abs_str = re.sub(
        r"(?i)\bcol\([^)]+\)",
        f"col_{req_type}",
        abs_str,
    )

    # 3. 替换各种类型 wrapper。
    for category, keywords in mapping.items():
        kw_list = sorted(set([category] + list(keywords)), key=len, reverse=True)
        pattern_str = "|".join(re.escape(kw) for kw in kw_list)

        abs_str = re.sub(
            rf"(?i)\b(?:{pattern_str})\([^)]+\)",
            f"typical_{category}",
            abs_str,
        )

    # 4. decimal / numeric 兜底。
    abs_str = re.sub(
        r"(?i)\b(?:decimal|numeric)\([^)]+\)",
        "typical_float",
        abs_str,
    )

    return abs_str


def _node_return_type(node: Any) -> str:
    """
    获取 AST 节点 return_type 的安全函数。
    """
    ret = getattr(node, "return_type", None)
    return str(ret) if ret else "UNKNOWN"


def _node_id(node: Any) -> str:
    """
    与 KnowledgeGraphBuilder 保持一致的节点 ID 规则：
        token::return_type
    """
    token = str(getattr(node, "token", "UNKNOWN"))
    return_type = _node_return_type(node)
    return f"{token}::{return_type}"


def extract_edges_from_ast(ast_root: Any) -> List[ExtractedEdge]:
    """
    从已经完成类型推断的 AST 中抽取所有父子边。

    注意：
        这个函数只读 AST，不修改 KnowledgeGraphBuilder。
        是否是“新边”，由外部 builder.add_predicate(ast_root) 判断。
    """
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
    """
    对 parser 生成的 AST 做轻量 token 规范化。

    目的：
        统一表面形式不同、但语义相同的算子节点，避免知识图中出现重复节点。

    示例：
        NOT()
        NOT ()
        NOT

    统一为：
        NOT

    注意：
        这里只改 AST 节点 token，不改变 AST 结构；
        不做数据库语义判断；
        不替 RL 生成或修复谓词。
    """
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
    """Map MySQL postfix IS syntax onto the configured binary IS graph node."""
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
    """
    收集 AST 中出现过的算子节点 ID。
    """
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
    """
    收集 AST 中所有节点 ID。
    """
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
    """
    谓词结构分析器。

    env_sql_dependency.py 和 pkg_manager.py 后续都应该使用这个类，
    避免各自维护一套 abstract / parse / infer / extract 逻辑。
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None, strict_consumption: bool = False):
        """
        :param config:
            parser / type inference 使用的配置。
            如果为 None，则自动加载当前数据库对应配置。

        :param strict_consumption:
            是否严格要求 parser 消费完所有 token。
            当前先默认 False，避免一上来改变旧行为。
            后续如果 parser 稳定，可以切换为 True。
        """
        self.config = config if config is not None else load_pkg_config()
        self.parser = SimpleParser(self.config)
        self.inferencer = TypeInferencer(self.config)
        self.strict_consumption = strict_consumption

    def analyze(self, predicate: str, expected_type: str = "T_BOOL") -> ParsedPredicateResult:
        """
        分析单条谓词。

        成功条件：
            1. normalize 后非空；
            2. abstract 后 parser 能返回 AST；
            3. type inference 不抛异常；
            4. 能完成 edge / operator 抽取。

        注意：
            SQL 是否执行成功不在这里判断。
        """
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

            # 对 AST token 做轻量规范化。
            # 例如把 NOT() / NOT () 统一为 NOT，避免知识图出现重复算子节点。
            canonicalize_ast_tokens(ast_root)
            # MySQL rules use one binary IS node; postfix spellings are syntax sugar.
            canonicalize_mysql_is_syntax(ast_root)

            # 当前 SimpleParser.parse() 不强制检查剩余 token。
            # 这里先记录 warning，不默认判失败，避免改变旧流程行为。
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
    """
    函数式便捷入口。

    适合一次性调用。
    如果在循环中大量调用，建议复用 PredicateStructureAnalyzer 实例。
    """
    analyzer = PredicateStructureAnalyzer(
        config=config,
        strict_consumption=strict_consumption,
    )
    return analyzer.analyze(predicate, expected_type=expected_type)


if __name__ == "__main__":
    # 简单自测，不依赖数据库。
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
    print("请重点检查：")
    print("1. NOT (...) 是否被解析为 NOT::T_BOOL，而不是 NOT()::T_BOOL。")
    print("2. col(name) LIKE string('%abc%') 是否抽象为 COL_STRING::STRING，而不是 COL_INT::INT。")
