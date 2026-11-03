"""
文件名: sql_generator.py
路径: BugFinding/sql_generator.py
描述:
    [最终增强版] TLP 测试用例生成器。

    该模块负责读取上一阶段生成的谓词，并将其转化为具体的 TLP 测试 SQL 语句。

"""

# ================= 导入 Python 标准库 =================
import os        # 用于文件路径操作和操作系统交互
import re        # 用于正则表达式匹配，提取 col(xxx) 格式的占位符
import random    # 用于随机选择列、生成随机值和决定概率
import glob      # 用于模式匹配查找文件 (如 *.txt)
import json      # 用于读取 JSON 格式的配置文件
import time      # 用于第三阶段截止时间和吞吐统计
import gzip      # 第三阶段完整测试用例的标准 gzip 无损压缩
import io        # 在 gzip 二进制流上提供 UTF-8 文本写入
from contextlib import contextmanager
from pathlib import Path  # 现代化的面向对象路径库
from typing import Dict, List, Tuple, Any, Optional # 用于类型注解，提高代码可读性

# ================= 导入项目底层依赖 =================
import sys  # 用于修改 Python 搜索路径
# 获取当前文件的父目录的父目录，即项目的根目录路径
project_root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# 将项目根目录添加到系统路径中，确保可以导入 RL 模块
sys.path.append(project_root_path)

# 1. 复用原先的数据库配置 (从中获取 db_type, host 等关键信息)
from RL.config import PROJECT_ROOT, DB_CONFIG, STAGE3_CONFIG
# 2. 复用原先的数据库执行器 (提供统一的 execute_query 接口，处理连接细节)
from RL.db_executor import create_db_executor
from RL.target_table import get_target_table
# 3. 保持原有 WHERE 谓词实例化的随机数序列
from BugFinding.tlp_random_compat import advance_instantiation_rng
from pkg.stage3_runtime import (
    Stage3DeadlineReached,
    check_deadline,
    write_status_json,
)

# ==================== [修改 1: 在这里插入清洗逻辑定义] ====================
# 定义需要剥离的 DSL 包装器 (注意：包含 col，但我们会在列名处理完之后再清洗它)
DSL_WRAPPERS = [
    "int", "float", "double", "decimal",
    "string", "char", "varchar",
    "date", "datetime", "timestamp", "time",
    "col", "hex", "bin", "bit"
]

# 预编译正则，用于将 int(18) -> 18
DSL_CLEAN_PATTERN = re.compile(
    r"\b(" + "|".join(DSL_WRAPPERS) + r")\s*\(\s*([^()]+)\s*\)",
    re.IGNORECASE
)


def cleanup_dsl(text: str) -> str:
    """
    递归清洗 DSL 包装器。
    例如: col_integer_0 > int(10)  ->  col_integer_0 > 10
    """
    if not text:
        return text
    if _is_sqlserver_db():
        return _rewrite_sqlserver_dsl(text, strip_wrappers=True)

    curr_text = text
    # 循环 5 次以处理嵌套结构
    for _ in range(5):
        # 将匹配到的 int(val) 替换为 val
        new_text = DSL_CLEAN_PATTERN.sub(lambda m: m.group(2).strip(), curr_text)
        if new_text == curr_text:
            break
        curr_text = new_text
    return curr_text

# ========================================================================

# ================= 全局路径与常量配置 =================
# 输入文件夹：存放上一阶段 (图遍历) 生成的谓词 txt 文件
INPUT_DIR = PROJECT_ROOT / "pkg" / "generated_sqls"
# 输出文件夹：存放生成的完整 SQL 测试用例文件 (本模块产物)
OUTPUT_DIR = PROJECT_ROOT / "BugFinding" / "generated_test_cases"
# 运算符配置文件所在目录 (存放 mysql.json, common.json 等)
OPERATOR_CONFIG_DIR = PROJECT_ROOT / "BugFinding" / "db_operators"
EXACT_COLUMN_TYPE_HINT_PREFIX = "__TYPE__"

ORACLE_DB_TYPES = {"oracle", "oracledb", "oracle_ai26"}


def _is_oracle_db() -> bool:
    """Return whether the current Stage-3 executor is an Oracle executor."""
    return str(getattr(DB_CONFIG, "db_type", "") or "").strip().lower() in ORACLE_DB_TYPES


def _is_sqlserver_db() -> bool:
    return str(getattr(DB_CONFIG, "db_type", "") or "").strip().lower() == "sqlserver"


def _sqlserver_quote_identifier(name: str) -> str:
    """Quote one raw identifier component, including embedded dots/brackets."""
    return "[" + str(name).replace("]", "]]") + "]"


def _sqlserver_column_type(column: Dict[str, Any]) -> str:
    from RL.sqlserver_evidence_types import column_type
    return column_type(column)


def _sqlserver_column_capabilities(column: Dict[str, Any]) -> set:
    from RL.sqlserver_evidence_types import base_type, NUMERIC, STRINGS, BINARY, TEMPORAL
    dtype = base_type(_sqlserver_column_type(column))
    if dtype not in NUMERIC | STRINGS | BINARY | TEMPORAL | {"uniqueidentifier"}:
        return set()
    capabilities = {"scalar"}
    # Legacy LOBs cannot be compared/grouped directly. XML, spatial and
    # sql_variant are outside this experiment's supported physical types.
    if dtype in {"text", "ntext", "image"}:
        return capabilities
    capabilities.add("comparison")
    return capabilities


_SQLSERVER_SIMPLE_DSL_TOKENS = re.compile(
    r"(?P<quoted>[Nn]?'(?:''|[^'])*'|\"(?:\"\"|[^\"])*\"|\[(?:\]\]|[^\]])*\])"
    r"|(?P<word>[A-Za-z_][A-Za-z0-9_$]*)"
    r"|(?:0[xX][0-9A-Fa-f]*(?![A-Za-z0-9_$])"
    r"|(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?(?![A-Za-z0-9_$]))"
    r"|(?P<punct>[(),.@+*/%<>=!~&|^;\-])"
    r"|(?P<space>\s+)|(?P<other>.)",
    re.DOTALL,
)
_SQLSERVER_DSL_WRAPPERS = (frozenset(DSL_WRAPPERS) - {"char", "col"}) | {
    "integer", "numeric", "nvarchar", "binary", "varbinary", "datetime2",
    "datetimeoffset", "uniqueidentifier",
}
_SQLSERVER_TYPE_ARGUMENT_CALLS = frozenset({
    "cast", "try_cast", "parse", "try_parse", "convert", "try_convert",
})


def _try_rewrite_simple_sqlserver_dsl(text, replace_column):
    """Fast path for ordinary generated expressions containing only col() DSL.

    Quoted contents are opaque. Complex syntax and actual DSL value wrappers
    still use the existing sqlglot-based rewrite below. Scan and check the whole
    input before calling replace_column, so fallback never consumes randomness.
    No expressions, replacement values or per-predicate plans are cached.
    """
    if "--" in text or "/*" in text:
        return None
    tokens, pairs, stack = [], {}, []
    for match in _SQLSERVER_SIMPLE_DSL_TOKENS.finditer(text):
        kind = match.lastgroup
        if kind == "other":
            return None
        if kind == "space":
            continue
        index = len(tokens)
        raw = match.group()
        tokens.append((match.start(), match.end(), raw, kind))
        if kind == "punct":
            if raw == "(":
                stack.append(index)
            elif raw == ")":
                if not stack:
                    return None
                pairs[stack.pop()] = index
    if stack:
        return None

    columns, i = [], 0
    while i + 1 < len(tokens):
        start, end, raw, kind = tokens[i]
        if kind != "word" or tokens[i + 1][2] != "(":
            i += 1
            continue
        name = raw.lower()
        if name in _SQLSERVER_DSL_WRAPPERS or name in _SQLSERVER_TYPE_ARGUMENT_CALLS:
            return None
        if name == "col":
            if i > 0 and tokens[i - 1][2] in {".", "@"}:
                return None
            close = pairs[i + 1]
            columns.append((start, tokens[close][1],
                            text[tokens[i + 1][1]:tokens[close][0]].strip()))
            i = close + 1
        else:
            i += 1
    if not columns:
        return text
    if replace_column is None:
        raise ValueError("Unresolved SQL Server col() placeholder.")
    pieces, cursor = [], 0
    for start, end, hint in columns:
        pieces.extend((text[cursor:start], replace_column(hint)))
        cursor = end
    pieces.append(text[cursor:])
    return "".join(pieces)


def _rewrite_sqlserver_dsl(text: str, replace_column=None, *, strip_wrappers=False) -> str:
    """Rewrite template calls without touching quoted data or SQL type arguments."""
    simple = _try_rewrite_simple_sqlserver_dsl(text, replace_column)
    if simple is not None:
        return simple
    import sqlglot
    tokens = sqlglot.Dialect.get_or_raise("tsql").tokenize(text)
    pairs, stack = {}, []
    for i, token in enumerate(tokens):
        if token.token_type.name == "L_PAREN":
            stack.append(i)
        elif token.token_type.name == "R_PAREN":
            if not stack:
                raise ValueError("Unbalanced SQL Server template parentheses.")
            pairs[stack.pop()] = i
    if stack:
        raise ValueError("Unbalanced SQL Server template parentheses.")
    wrappers = _SQLSERVER_DSL_WRAPPERS

    def bare_name(i):
        raw = text[tokens[i].start:tokens[i].end + 1]
        return raw.lower() if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", raw) else ""

    def top_level(lo, hi):
        while lo < hi:
            yield lo
            lo = pairs[lo] + 1 if lo in pairs else lo + 1

    def render(lo, hi, start, end):
        pieces, cursor, i = [], start, lo
        while i < hi:
            name = bare_name(i)
            qualified = (i > 0 and tokens[i - 1].token_type.name in {"DOT", "PARAMETER"})
            if i + 1 >= hi or tokens[i + 1].token_type.name != "L_PAREN":
                i += 1
                continue
            close = pairs[i + 1]
            inner_lo, inner_start, inner_end = i + 2, tokens[i + 1].end + 1, tokens[close].start
            replacement = None
            if name == "col" and not qualified:
                if replace_column is None:
                    raise ValueError("Unresolved SQL Server col() placeholder.")
                replacement = replace_column(text[inner_start:inner_end].strip())
            elif name in wrappers and not qualified and strip_wrappers:
                if inner_lo == close or any(tokens[j].token_type.name == "COMMA"
                                            for j in top_level(inner_lo, close)):
                    raise ValueError(f"SQL Server DSL wrapper {name} requires one expression.")
                replacement = "(" + render(inner_lo, close, inner_start, inner_end) + ")"
            else:
                # CAST/CONVERT declarations such as decimal(18,2) are real SQL.
                separator = next((j for j in top_level(inner_lo, close)
                                  if not qualified and (
                                      (name in {"cast", "try_cast", "parse", "try_parse"} and bare_name(j) == "as")
                                      or (name in {"convert", "try_convert"} and tokens[j].token_type.name == "COMMA"))), None)
                if separator is None:
                    inner = render(inner_lo, close, inner_start, inner_end)
                elif name in {"convert", "try_convert"}:
                    split = tokens[separator].end + 1
                    inner = text[inner_start:split] + render(separator + 1, close, split, inner_end)
                else:
                    split = tokens[separator].start
                    inner = render(inner_lo, separator, inner_start, split) + text[split:inner_end]
                replacement = text[tokens[i].start:inner_start] + inner + ")"
            pieces.extend((text[cursor:tokens[i].start], replacement))
            cursor, i = tokens[close].end + 1, close + 1
        pieces.append(text[cursor:end])
        return "".join(pieces)

    return render(0, len(tokens), 0, len(text))


# Local Oracle-only fallback. It is deliberately not common/MySQL/PostgreSQL.
# A dedicated BugFinding/db_operators/oracle.json can replace it later without
# changing the instantiation flow.
ORACLE_OPERATOR_MAP = {
    "NUMERIC": [
        "+", "-", "*", "/", "MOD", "ABS", "CEIL", "FLOOR", "ROUND", "TRUNC",
        "POWER", "SQRT", "EXP", "LN", "LOG", "SIGN", "GREATEST", "LEAST",
    ],
    "STRING": [
        "LIKE", "NOT LIKE", "LIKEC", "LIKE2", "LIKE4", "||", "LOWER", "UPPER",
        "INITCAP", "LENGTH", "LENGTHB", "INSTR", "SUBSTR", "REPLACE", "TRANSLATE",
        "TRIM", "LTRIM", "RTRIM", "LPAD", "RPAD", "CONCAT", "REGEXP_LIKE",
    ],
    "DATETIME": [
        "ADD_MONTHS", "MONTHS_BETWEEN", "LAST_DAY", "FROM_TZ", "SYSDATE",
        "SYSTIMESTAMP", "CURRENT_DATE", "CURRENT_TIMESTAMP",
    ],
    "LOGICAL": ["AND", "OR", "NOT"],
    "COMPARISON": [
        "=", "!=", "<>", "<", "<=", ">", ">=", "IS NULL", "IS NOT NULL",
        "IS TRUE", "IS FALSE", "IS NOT TRUE", "IS NOT FALSE", "IN", "NOT IN",
        "BETWEEN", "NOT BETWEEN",
    ],
    "JSON": ["IS JSON", "IS NOT JSON", "JSON_EXISTS", "JSON_VALUE", "JSON_QUERY"],
    "FLOW_CONTROL": ["NVL", "NVL2", "COALESCE", "NULLIF", "DECODE", "CASE"],
}

_COLUMN_TYPE_ALIASES = {
    "int2": "smallint",
    "int4": "integer",
    "int8": "bigint",
    "float4": "real",
    "float8": "double precision",
    "bool": "boolean",
    "bpchar": "character",
    "varchar": "character varying",
    "timestamp": "timestamp without time zone",
    "timestamptz": "timestamp with time zone",
    "time": "time without time zone",
    "timetz": "time with time zone",
}


def _normalize_column_type(value: Any) -> str:
    """Normalize live schema types to the names carried by Stage-3 hints."""
    text = " ".join(str(value or "").strip().lower().split())
    if text.startswith("pg_catalog."):
        text = text[len("pg_catalog.") :]
    if "(" in text:
        text = " ".join(re.sub(r"\([^()]*\)", "", text).split())
    return _COLUMN_TYPE_ALIASES.get(text, text)

# 确保输出目录存在，如果不存在则自动创建，避免写入时报错
os.makedirs(OUTPUT_DIR, exist_ok=True)


@contextmanager
def _open_test_case_writer(
    output_path: Path,
    *,
    compressed: bool,
    compression_level: int,
    buffer_bytes: int,
):
    """
    流式打开第三阶段完整测试用例输出。

    gzip 只改变无损存储格式；解压后的文本、SQL 顺序和四条 SELECT 一组的结构与 .sql
    完全一致。底层仍使用大缓冲，避免长期运行时产生大量小块磁盘写入。
    """
    if not compressed:
        with open(
            output_path,
            "w",
            encoding="utf-8",
            buffering=buffer_bytes,
        ) as text_file:
            yield text_file
        return

    with open(output_path, "wb", buffering=buffer_bytes) as raw_file:
        with gzip.GzipFile(
            filename="",
            mode="wb",
            fileobj=raw_file,
            compresslevel=compression_level,
            mtime=0,
        ) as gzip_file:
            with io.TextIOWrapper(
                gzip_file,
                encoding="utf-8",
                newline="",
            ) as text_file:
                yield text_file


def _is_predicate_file(path: str) -> bool:
    lower_path = str(path).lower()
    return lower_path.endswith(".txt") or lower_path.endswith(".txt.gz")


def _logical_predicate_name(path: str) -> str:
    """返回与未压缩格式一致的逻辑文件名，用于稳定输出名和追溯注释。"""
    file_name = os.path.basename(str(path))
    if file_name.lower().endswith(".txt.gz"):
        return file_name[:-3]
    return file_name


@contextmanager
def _open_predicate_reader(file_path: str, buffer_bytes: int):
    """流式读取 .txt 或 .txt.gz；两种格式向生成器提供完全相同的文本行。"""
    if str(file_path).lower().endswith(".gz"):
        with open(file_path, "rb", buffering=buffer_bytes) as raw_file:
            with gzip.GzipFile(fileobj=raw_file, mode="rb") as gzip_file:
                with io.TextIOWrapper(
                    gzip_file,
                    encoding="utf-8",
                    newline=None,
                ) as text_file:
                    yield text_file
        return

    with open(
        file_path,
        "r",
        encoding="utf-8",
        buffering=buffer_bytes,
    ) as text_file:
        yield text_file


def _discover_predicate_files(input_dir: Path) -> List[str]:
    """发现谓词文件；同一逻辑文件同时存在时优先读取新的 gzip 版本。"""
    candidates = glob.glob(str(input_dir / "output_sql_*.txt"))
    candidates.extend(glob.glob(str(input_dir / "output_sql_*.txt.gz")))
    selected = {}
    for path in sorted(candidates):
        if not _is_predicate_file(path):
            continue
        logical_name = _logical_predicate_name(path)
        previous = selected.get(logical_name)
        path_mtime = os.path.getmtime(path)
        previous_mtime = os.path.getmtime(previous) if previous is not None else -1.0
        if (
            previous is None
            or path_mtime > previous_mtime
            or (
                path_mtime == previous_mtime
                and str(path).lower().endswith(".txt.gz")
            )
        ):
            selected[logical_name] = str(path)
    return [selected[name] for name in sorted(selected)]


class DBMetaDataLoader:
    """
    类说明: 元数据加载器。
    功能: 连接数据库，实时查询 `test_wide_table` 表的结构信息 (列名、类型)。
    """
    def __init__(self):
        # 初始化一个字典，用于存储加载到的列信息，key为表名
        self.columns = {}
        self.table_name, self.table_schema, self.table_sql = get_target_table(
            DB_CONFIG.db_type, DB_CONFIG
        )
        # 实例化时立即执行加载逻辑，确保对象创建后即刻可用
        self._load_schema()

    def _load_schema(self):
        """
        方法说明: 核心加载逻辑。
        使用 SQL 的 DESCRIBE 语句向数据库查询表结构。
        """
        print("[SchemaLoader] 正在连接数据库加载元数据...")

        # === 复用点：使用 RL.db_executor 统一执行器 ===
        # autoconnect=True 表示初始化时自动建立数据库连接
        executor = create_db_executor(autoconnect=True)

        self.table_name, self.table_schema, self.table_sql = get_target_table(
            DB_CONFIG.db_type, getattr(executor, "config", DB_CONFIG)
        )
        # 目标表名 (固定为我们之前初始化的大宽表，不从其他表读取)
        table_name = self.table_name

        # 执行查询，返回 DBQueryResult 对象
        try:
            if self.table_schema is not None:
                res = executor.list_columns(table_name, schema=self.table_schema)
            else:
                res = executor.list_columns(table_name)
        finally:
            # Also release the connection when metadata loading raises.
            executor.close()

        # 检查查询是否成功 (使用 DBQueryResult 的 success 属性)
        if not res.success or not res.rows:
            raise RuntimeError(
                f"Cannot load required table {self.table_sql}: "
                f"{res.error or 'table missing or no readable columns'}."
            )

        # 解析查询结果，准备存储列信息
        cols = []

        # === 关键适配点：使用 res.rows 获取数据 ===
        # 根据 db_executor 的定义，结果存储在 .rows 中，且是 List[Dict] (DictCursor模式)
        if res.rows:
            for row in res.rows:
                # 尝试获取列名，兼容不同驱动的大小写 ('Field' 或 'field')
                col_name = row.get('Field') or row.get('field')
                # 尝试获取类型 ('Type' 或 'type')
                col_type = row.get('Type') or row.get('type')

                # 只有当列名和类型都有效时才保存
                if col_name and col_type:
                    column = dict(row) if _is_sqlserver_db() else {}
                    column.update(name=col_name, type=col_type)
                    if _is_sqlserver_db():
                        column.setdefault("table_schema", self.table_schema)
                        column.setdefault("table_name", table_name)
                    cols.append(column)

        # 将解析好的列列表保存到字典中，以表名为 Key
        self.columns[table_name] = cols
        self.columns[self.table_sql] = cols
        print(f"[SchemaLoader] 加载完成: {table_name} 包含 {len(cols)} 列。")

    def get_columns(self, table_name: str):
        """
        方法说明: 获取指定表的列信息列表。
        返回: List[Dict]，包含列名和类型。
        """
        return self.columns.get(table_name, [])


class SmartInstantiator:
    """
    类说明: 智能实例化器。
    功能:
    1. 读取配置文件 (db_operators) 了解运算符分类。
    2. 将抽象谓词 col(xxx) 智能替换为真实的列名 col_integer_5 等。
    """
    def __init__(self, schema_loader: DBMetaDataLoader):
        # 持有元数据加载器的引用，以便查询列信息
        self.loader = schema_loader
        # 用于存储运算符配置的字典 (Key: 类别, Value: 运算符列表)
        self.operator_map = {}

        # Exact guidance changes only Stage-3 placeholder instantiation. A small
        # arbitrary-column channel preserves discovery of undocumented coercions.
        self.exact_column_type_guidance_enabled = bool(
            getattr(STAGE3_CONFIG, "stage3_exact_column_type_guidance_enabled", True)
        )
        self.column_type_exploration_ratio = min(
            1.0,
            max(
                0.0,
                float(
                    getattr(
                        STAGE3_CONFIG,
                        "stage3_column_type_exploration_ratio",
                        0.05,
                    )
                ),
            ),
        )

        # 硬编码的类型归类映射 (用于判断列是否匹配算子)
        # 作用: 只有列的类型字符串包含这些关键字时，才认为该列属于这个类别
        if _is_sqlserver_db():
            from RL.sqlserver_evidence_types import NUMERIC, STRINGS, BINARY, TEMPORAL
            self.type_category_mapping = {
                "NUMERIC": sorted(NUMERIC), "STRING": sorted(STRINGS),
                "DATETIME": sorted(TEMPORAL), "BINARY": sorted(BINARY),
                "GUID": ["uniqueidentifier"],
            }
        elif _is_oracle_db():
            # Oracle 的类型分类只用于第三阶段占位符实例化；其他数据库继续
            # 使用原有映射，避免改变第一、二阶段或 MySQL/PostgreSQL 行为。
            self.type_category_mapping = {
                "NUMERIC": [
                    "NUMBER", "DECIMAL", "NUMERIC", "INTEGER", "INT",
                    "SMALLINT", "BIGINT", "FLOAT", "REAL", "DOUBLE",
                    "BINARY_FLOAT", "BINARY_DOUBLE",
                ],
                "STRING": [
                    "CHAR", "NCHAR", "VARCHAR", "VARCHAR2", "NVARCHAR2",
                    "CLOB", "NCLOB", "LONG", "ROWID", "UROWID",
                ],
                "DATETIME": [
                    "DATE", "TIMESTAMP", "INTERVAL YEAR", "INTERVAL DAY",
                ],
                "BINARY": ["RAW", "LONG RAW", "BLOB"],
                "JSON": ["JSON"],
                "BOOLEAN": ["BOOLEAN", "BOOL"],
                "XML": ["XMLTYPE"],
                "VECTOR": ["VECTOR"],
            }
        else:
            self.type_category_mapping = {
                'NUMERIC': ['INT', 'DECIMAL', 'FLOAT', 'DOUBLE', 'NUMERIC', 'BIT', 'YEAR', 'BIGINT', 'TINYINT', 'HEX'],
                'STRING': ['CHAR', 'TEXT', 'BLOB', 'BINARY', 'ENUM', 'VARCHAR', 'VARBINARY'],
                'DATETIME': ['DATE', 'TIME', 'TIMESTAMP', 'DATETIME'],
                'JSON': ['JSON', 'LONGTEXT'] # 适配你的 MySQL 5.5 LONGTEXT 存 JSON 方案
            }

        # 初始化时加载运算符配置
        self._load_operator_config()

    def _load_operator_config(self):
        """
        方法说明: 加载运算符配置文件。
        逻辑: 普通数据库优先找 db_type.json -> common.json -> 硬编码保底；
        Oracle 只使用 Oracle 专属文件或本地 Oracle profile。
        """
        # 从全局配置中读取当前数据库类型 (如 'mysql')
        db_type = str(getattr(DB_CONFIG, "db_type", "") or "").strip().lower()
        # Oracle 别名统一读取 oracle.json；其他数据库仍使用原文件名。
        config_db_type = "oracle" if db_type in ORACLE_DB_TYPES else db_type

        # 构造专用配置文件路径 (例如 BugFinding/db_operators/mysql.json)
        specific_config = OPERATOR_CONFIG_DIR / f"{config_db_type}.json"
        # 构造通用配置文件路径
        common_config = OPERATOR_CONFIG_DIR / "common.json"

        # 标记当前使用的是哪种配置，用于日志输出
        source_type = "None"

        # Oracle 必须始终使用 Oracle 专属算子集合。即使 oracle.json 尚未
        # 提供，也不能回退到 common/MySQL/PostgreSQL 配置。
        if db_type in ORACLE_DB_TYPES:
            if os.path.exists(specific_config):
                try:
                    with open(specific_config, 'r', encoding='utf-8') as f:
                        self.operator_map = json.load(f)
                    source_type = f"Specific ({config_db_type}.json)"
                except Exception as e:
                    print(f"[Config] Oracle 专用配置加载失败: {e}")
            if not self.operator_map:
                self.operator_map = dict(ORACLE_OPERATOR_MAP)
                source_type = "Oracle Stage-3 built-in profile"

            print(f"[Instantiator] 运算符配置加载完成。来源: [{source_type}]")
            print(f"[Instantiator] 支持的类型分类: {list(self.operator_map.keys())}")
            return

        # 1. 尝试加载专用配置
        if os.path.exists(specific_config):
            try:
                with open(specific_config, 'r', encoding='utf-8') as f:
                    self.operator_map = json.load(f)
                source_type = f"Specific ({db_type}.json)"
            except Exception as e:
                print(f"[Config] 加载专用配置失败: {e}")

        # 2. 如果专用配置没加载到，尝试通用配置
        if db_type == "sqlserver" and not self.operator_map:
            raise ValueError(f"SQL Server operator configuration unavailable: {specific_config}")
        if not self.operator_map and os.path.exists(common_config):
            try:
                with open(common_config, 'r', encoding='utf-8') as f:
                    self.operator_map = json.load(f)
                source_type = "Common (common.json)"
            except Exception as e:
                print(f"[Config] 加载通用配置失败: {e}")

        # 3. 如果文件都失败，使用硬编码保底 (Expanded Hardcoded Map)
        if not self.operator_map:
            source_type = "Hardcoded Fallback"
            # 这是一个扩充后的保底配置，包含了绝大多数 DBMS 支持的标准算子
            self.operator_map = {
                "NUMERIC": ["=", ">", "<", ">=", "<=", "!=", "<>"],
                "STRING": ["=", "!=", "<>", "LIKE", "NOT LIKE"],
                "DATETIME": ["=", ">", "<", ">=", "<=", "!="],
                "JSON": ["="] # JSON 通常作为字符串处理，用 = 比较
            }

        # 打印最终使用的配置来源，方便用户确认
        print(f"[Instantiator] 运算符配置加载完成。来源: [{source_type}]")
        # 打印一下加载到的 Key，确认内容
        print(f"[Instantiator] 支持的类型分类: {list(self.operator_map.keys())}")

    def _guess_operator_type(self, predicate: str) -> str:
        """
        方法说明: 根据谓词字符串内容，猜测它包含的算子属于哪一类 (NUMERIC, STRING...)。
        """
        if _is_sqlserver_db():
            import sqlglot
            tokens = sqlglot.Dialect.get_or_raise("tsql").tokenize(predicate)
            # Ignore quoted strings/identifiers: their contents are not operators.
            code = " ".join(
                predicate[t.start:t.end + 1].upper()
                if t.token_type.name not in {"STRING", "NATIONAL_STRING", "IDENTIFIER"} else "?"
                for t in tokens
            )
            for category, ops in self.operator_map.items():
                if category not in self.type_category_mapping:
                    continue
                for op in ops:
                    pattern = re.escape(op.rstrip("(").strip()).replace(r"\ ", r"\s+")
                    if op.endswith("("):
                        pattern += r"\s*\("
                    if re.search(r"(?<![\w])" + pattern + r"(?![\w])", code):
                        return category
            return "ANY"
        # 遍历配置中的所有分类
        for category, ops in self.operator_map.items():
            for op in ops:
                # 如果算子符号出现在谓词字符串中
                if op in predicate:
                    return category
        # 如果没匹配到，返回 'ANY' (表示不限制类型)
        return 'ANY'

    def _resolve_hint_operator_type(self, hint_name: str, predicate_context: str) -> str:
        """
        根据谓词字符串和 col(...) 中的类型线索，推断当前列占位符更适合的列类型类别。
        该方法用于 WHERE 谓词实例化的列类型选择。
        """
        op_type = self._guess_operator_type(predicate_context)

        if hint_name.endswith("_mock"):
            exact_type = hint_name.replace("_mock", "").upper()
            # 排除 CMP 和 T 这种泛型，防止把 'T' 误判为 TIMESTAMP。
            if exact_type not in ["CMP", "T"]:
                for cat, keywords in self.type_category_mapping.items():
                    if exact_type == cat or exact_type in keywords:
                        op_type = cat
                        break

        return op_type

    @staticmethod
    def _exact_type_from_hint(hint_name: str) -> Optional[str]:
        text = str(hint_name or "").strip()
        if not text.startswith(EXACT_COLUMN_TYPE_HINT_PREFIX):
            return None
        if _is_sqlserver_db():
            from RL.sqlserver_evidence_types import normalize_sqlserver_type
            return normalize_sqlserver_type(text[len(EXACT_COLUMN_TYPE_HINT_PREFIX):]) or None
        exact_type = _normalize_column_type(text[len(EXACT_COLUMN_TYPE_HINT_PREFIX) :])
        return exact_type or None

    @staticmethod
    def _oracle_is_id_column(column: Dict[str, Any]) -> bool:
        """Oracle exposes unquoted dictionary column names in upper case."""
        name = str(column.get("name", "") or "").strip().strip('"')
        return name.lower() == "id"


    def _choose_column_metadata(
        self,
        table_name: str,
        hint_name: str,
        predicate_context: str,
        *,
        capability: str = "comparison",
        binding_candidates=None,
    ) -> Optional[Dict[str, Any]]:
        if _is_sqlserver_db():
            return self._choose_sqlserver_column(
                table_name, hint_name, predicate_context, capability,
                binding_candidates=binding_candidates,
            )
        columns = [
            column
            for column in self.loader.get_columns(table_name)
            if not (
                self._oracle_is_id_column(column)
                if _is_oracle_db()
                else str(column.get("name", "")) == "id"
            )
        ]
        if not columns:
            return None

        exact_type = (
            self._exact_type_from_hint(hint_name)
            if self.exact_column_type_guidance_enabled
            else None
        )
        if exact_type:
            # Exploration is deliberately decided before exact matching, so its
            # configured fraction remains measurable across long experiments.
            if random.random() < self.column_type_exploration_ratio:
                return random.choice(columns)
            exact_candidates = [
                column
                for column in columns
                if _normalize_column_type(column.get("type")) == exact_type
            ]
            if exact_candidates:
                if binding_candidates is not None:
                    binding_candidates.extend(exact_candidates)
                return random.choice(exact_candidates)

        op_type = self._resolve_hint_operator_type(hint_name, predicate_context)
        candidates = []
        for column in columns:
            col_type = str(column.get("type", "") or "").upper()
            is_match = False
            if op_type in self.type_category_mapping:
                valid_keywords = self.type_category_mapping[op_type]
                is_match = any(keyword in col_type for keyword in valid_keywords)
            if is_match:
                score = 50
            elif op_type == "ANY":
                score = 20
            else:
                score = 10
            candidates.append((column, score))

        candidates.sort(key=lambda item: item[1], reverse=True)
        top_k = min(len(candidates), 3)
        # Preserve legacy behavior for old coarse placeholders and DBMS providers
        # that do not yet expose concrete column-type variants.
        if random.random() < 0.8:
            if binding_candidates is not None:
                binding_candidates.extend(c for c, score in candidates[:top_k] if score == 50)
            return random.choice(candidates[:top_k])[0]
        return random.choice(candidates)[0]

    def _sqlserver_table_cache(self, table_name):
        """Cache deterministic column facts for this loader's schema snapshot."""
        from RL.sqlserver_evidence_types import base_type

        columns = self.loader.get_columns(table_name)
        caches = getattr(self, "_sqlserver_table_caches", None)
        if caches is None:
            caches = self._sqlserver_table_caches = {}
        cached = caches.get(table_name)
        # DBMetaDataLoader replaces the column list when it reloads a schema.
        # Keep caches local to the instantiator; never change shared columns.
        if (cached is not None and cached["source"] is columns
                and cached["column_count"] == len(columns)):
            return cached

        facts = [
            (column, _sqlserver_column_type(column), _sqlserver_column_capabilities(column))
            for column in columns
        ]
        cached = {
            "source": columns,
            "column_count": len(columns),
            "facts": facts,
            "base_types": {id(column): base_type(dtype) for column, dtype, _ in facts},
            "capability_columns": {},
            "candidate_columns": {},
        }
        caches[table_name] = cached
        return cached

    def _choose_sqlserver_column(
        self, table_name, hint_name, predicate_context, capability, *, binding_candidates=None,
    ):
        from RL.sqlserver_evidence_types import base_type, NUMERIC, STRINGS, BINARY, TEMPORAL
        hint = str(hint_name).strip()
        exact_type = self._exact_type_from_hint(hint)
        if capability == "comparison" and exact_type in {"text", "ntext", "image"}:
            capability = self._sqlserver_lob_capability(hint, predicate_context, exact_type)
        cached = self._sqlserver_table_cache(table_name)
        if capability not in cached["capability_columns"]:
            cached["capability_columns"][capability] = [
                c for c, _, capabilities in cached["facts"]
                if c.get("name") and str(c["name"]).lower() != "id"
                and not c.get("is_hidden") and capability in capabilities
            ]
        columns = cached["capability_columns"][capability]
        if hint.startswith(EXACT_COLUMN_TYPE_HINT_PREFIX) and not exact_type:
            raise ValueError(f"Unsupported SQL Server column type hint: {hint!r}")
        if exact_type and self.exact_column_type_guidance_enabled:
            # Only cache matches for types present in the fixed schema. Unknown
            # hints return an empty pool without growing a cache per predicate.
            exact_key = (capability, "exact" if "(" in exact_type else "base")
            if exact_key not in cached["candidate_columns"]:
                by_type = {}
                for c, dtype, capabilities in cached["facts"]:
                    if (c.get("name") and str(c["name"]).lower() != "id"
                            and not c.get("is_hidden") and capability in capabilities):
                        key = dtype if "(" in exact_type else cached["base_types"][id(c)]
                        by_type.setdefault(key, []).append(c)
                cached["candidate_columns"][exact_key] = by_type
            candidates = cached["candidate_columns"][exact_key].get(exact_type, [])
        else:
            # Coarse graph terminals retain their own family; missing families
            # must not silently turn into arbitrary columns from the whole table.
            coarse = (hint[:-5] if hint.lower().endswith("_mock") else hint).upper()
            families = {
                "INT": {"bit", "tinyint", "smallint", "int", "bigint"},
                "INTEGER": {"bit", "tinyint", "smallint", "int", "bigint"},
                "FLOAT": {"real", "float", "decimal", "money", "smallmoney"},
                "NUMERIC": NUMERIC, "T_NUM": NUMERIC,
                "STRING": STRINGS, "STR": STRINGS, "T_STR": STRINGS,
                "DATETIME": TEMPORAL, "BINARY": BINARY, "GUID": {"uniqueidentifier"},
                "BOOL": set(), "BOOLEAN": set(), "T_BOOL": set(),
            }
            if exact_type:
                category = self._classify_column_category(exact_type)
                types = set(self.type_category_mapping.get(category, []))
            elif coarse in families:
                types = families[coarse]
            elif base_type(coarse):
                types = {base_type(coarse)}
            else:
                category = self._guess_operator_type(predicate_context)
                types = set(self.type_category_mapping.get(category, [])) if category != "ANY" else None
            if types is None:
                candidates = columns
            else:
                family_key = (capability, "family", frozenset(types))
                if family_key not in cached["candidate_columns"]:
                    cached["candidate_columns"][family_key] = [
                        c for c in columns if cached["base_types"][id(c)] in types
                    ]
                candidates = cached["candidate_columns"][family_key]
        if not candidates:
            raise ValueError(f"No SQL Server {capability} column for {hint!r} in {table_name}.")
        # Keep the existing configurable exploration channel, restricted to
        # supported columns with the capability required by the WHERE predicate.
        if random.random() < self.column_type_exploration_ratio:
            return random.choice(columns)
        if binding_candidates is not None:
            binding_candidates.extend(candidates)
        return random.choice(candidates)

    @staticmethod
    def _sqlserver_lob_capability(hint, predicate, dtype):
        """Keep observed LOB function inputs/NULL checks, but reject direct comparisons."""
        import sqlglot
        from sqlglot import exp
        markers = set()

        def replace(value):
            if value != hint:
                return "1"
            marker = f"__stage3_lob_{len(markers)}"
            markers.add(marker)
            return _sqlserver_quote_identifier(marker)

        # Only legacy LOB hints need this context check. Ordinary scalar columns
        # use the inexpensive metadata capability filter above.
        rewritten = _rewrite_sqlserver_dsl(predicate, replace, strip_wrappers=True)
        if not markers:
            return "comparison"
        expression = sqlglot.parse_one(rewritten, read="tsql")
        seen = set()
        for column in expression.find_all(exp.Column) if expression is not None else []:
            if column.name not in markers:
                continue
            seen.add(column.name)
            node = column
            while isinstance(node.parent, exp.Paren):
                node = node.parent
            parent = node.parent
            if isinstance(parent, exp.Is) and isinstance(parent.expression, exp.Null):
                continue
            if isinstance(parent, exp.Like) and dtype in {"text", "ntext"}:
                continue
            if isinstance(parent, exp.Func):
                continue  # Exact function/argument combinations come from projection evidence.
            return "comparison"
        return "scalar" if markers and seen == markers else "comparison"

    def _instantiate_sqlserver_columns(self, predicate, table_name):
        chosen = {}

        def replace(hint):
            if hint not in chosen:
                chosen[hint] = self.find_best_column(table_name, hint, predicate)
            return chosen[hint]

        return _rewrite_sqlserver_dsl(predicate, replace)

    def find_best_column(self, table_name: str, hint_name: str, predicate_context: str) -> str:
        """
        方法说明: 为谓词寻找最合适的列。
        逻辑: 根据算子类型给每一列打分，分数高的优先。
        """
        column = self._choose_column_metadata(table_name, hint_name, predicate_context)
        if _is_sqlserver_db():
            return _sqlserver_quote_identifier(column["name"])
        return str(column.get("name")) if column is not None else "col_unknown"

    def _classify_column_category(self, col_type: str) -> str:
        """
        根据数据库列类型字符串，粗略判断列所属类别。
        返回 NUMERIC / STRING / DATETIME / JSON / ANY。
        """
        if _is_sqlserver_db():
            from RL.sqlserver_evidence_types import base_type
            dtype = base_type(col_type)
            return next((category for category, types in self.type_category_mapping.items() if dtype in types), "ANY")
        col_type = str(col_type or "").upper()

        for category, keywords in self.type_category_mapping.items():
            if any(k in col_type for k in keywords):
                return category

        return "ANY"


    @staticmethod
    def _column_binding_key(column):
        # Retain declared lengths/precision and available collation metadata.
        # The pool has already passed the existing DBMS capability checks.
        dtype = (_sqlserver_column_type(column) if _is_sqlserver_db()
                 else str(column.get("type", "")))
        declared_type = str(column.get("type", "")) if column.get("is_user_defined") else ""
        return (
            re.sub(r"\s+", "", dtype.lower()),
            declared_type,
            column.get("collation_name", column.get("collation")),
            column.get("character_set_name"),
        )

    def _instantiate_independent_where_columns(self, raw_predicate, table_name):
        bindings = {}

        def render(column):
            if column is None:
                return "col_unknown"
            name = str(column["name"])
            return _sqlserver_quote_identifier(name) if _is_sqlserver_db() else name

        def replace(hint):
            if hint not in bindings:
                candidates = []
                selected = self._choose_column_metadata(
                    table_name, hint, raw_predicate, binding_candidates=candidates,
                )
                # Exploration deliberately supplies no candidates: preserve its
                # single choice for every occurrence, without another decision.
                compatible = []
                if selected is not None and selected in candidates:
                    key = self._column_binding_key(selected)
                    compatible = [c for c in candidates if self._column_binding_key(c) == key]
                bindings[hint] = (selected, compatible)
                return render(selected)
            selected, compatible = bindings[hint]
            return render(random.choice(compatible) if len(compatible) > 1 else selected)

        if _is_sqlserver_db():
            return _rewrite_sqlserver_dsl(raw_predicate, replace)
        # Quoted data/identifiers are opaque; only real column type markers vary.
        pattern = re.compile(
            r"'(?:''|[^'])*'|\"(?:\"\"|[^\"])*\"|`(?:``|[^`])*`|"
            r"\bcol\((?P<hint>(?:[^()]|\([^()]*\))*)\)"
        )
        return pattern.sub(
            lambda match: replace(match.group("hint"))
            if match.group("hint") is not None else match.group(0),
            raw_predicate,
        )

    def instantiate_predicate(
        self, raw_predicate: str, table_name: str, *, independent_columns: bool = False,
    ) -> str:
        """
        方法说明: 核心方法，将抽象的 col(xxx) 替换为真实列名。
        """
        if independent_columns:
            return self._instantiate_independent_where_columns(raw_predicate, table_name)
        if _is_sqlserver_db():
            return self._instantiate_sqlserver_columns(raw_predicate, table_name)
        # 使用正则提取 col(...) 中的内容，例如 'age'
        pattern = re.compile(r"col\(([^)]+)\)")
        matches = pattern.findall(raw_predicate)
        current_predicate = raw_predicate

        # 对每个提取到的 hint 进行替换
        for hint_name in set(matches):
            # 调用 find_best_column 找真实列名
            real_col = self.find_best_column(table_name, hint_name, raw_predicate)
            # 构造要被替换的字符串
            target_str = f"col({hint_name})"
            # 执行字符串替换
            current_predicate = current_predicate.replace(target_str, real_col)

        # 返回替换完成的谓词
        return current_predicate


class TLPQueryBuilder:
    """Construct the four WHERE-TLP SELECTs: total, true, false, null."""

    def build_queries(self, real_predicate: str, table_name: str) -> List[str]:
        return self._build_where_tlp_queries(real_predicate, table_name)

    def _build_where_tlp_queries(self, real_predicate: str, table_name: str) -> List[str]:
        """
        构造旧 WHERE-TLP 查询。

        必须与阶段1修改前 TestCaseGenerator.process_files() 中直接拼接的
        4 条 SELECT COUNT(*) 查询保持等价。
        """
        unknown_condition = f"({real_predicate}) IS NULL"
        if _is_sqlserver_db():
            # T-SQL conditions are not scalar values. CASE maps TRUE/FALSE to
            # non-NULL scalars and leaves only UNKNOWN in the NULL partition.
            # Keep one line per query for the existing four-statement consumer.
            unknown_condition = (
                f"CASE WHEN ({real_predicate}) THEN 1 "
                f"WHEN NOT ({real_predicate}) THEN 0 ELSE NULL END IS NULL"
            )
        return [
            f"SELECT COUNT(*) FROM {table_name};",
            f"SELECT COUNT(*) FROM {table_name} WHERE ({real_predicate});",
            f"SELECT COUNT(*) FROM {table_name} WHERE NOT ({real_predicate});",
            f"SELECT COUNT(*) FROM {table_name} WHERE {unknown_condition};",
        ]


class TestCaseGenerator:
    """
    类说明: 测试用例生成器主控类。
    流程: 扫描文件 -> 读取谓词 -> 生成 WHERE-TLP SQL -> 写入文件。
    """
    def __init__(self):
        # 初始化元数据加载器
        self.meta_loader = DBMetaDataLoader()
        # 初始化实例化器
        self.instantiator = SmartInstantiator(self.meta_loader)
        # 固定使用的大宽表表名
        self.table_name = self.meta_loader.table_sql

        self.tlp_mode = "where"  # Output metadata; no alternate detection mode.
        self.query_builder = TLPQueryBuilder()
        # 只控制第三阶段完整测试用例的无损存储格式，不改变生成内容。
        self.compress_test_cases = bool(
            getattr(STAGE3_CONFIG, "stage3_compress_test_cases", True)
        )
        self.test_case_compression_level = min(
            9,
            max(
                1,
                int(
                    getattr(
                        STAGE3_CONFIG,
                        "stage3_test_case_compression_level",
                        1,
                    )
                ),
            ),
        )
        print(f"[Generator] TLP mode: {self.tlp_mode}")
        print(
            "[Generator] Test-case storage: "
            f"{'gzip' if self.compress_test_cases else 'plain SQL'}"
            + (
                f" (level={self.test_case_compression_level})"
                if self.compress_test_cases
                else ""
            )
        )

    def build_stratified_case(self, raw_predicate):
        """限时分层支线逐组构造；与 process_files 的四条 SELECT 及文本格式一致。"""
        advance_instantiation_rng(self.instantiator, self.table_name)
        real_predicate = self.instantiator.instantiate_predicate(
            raw_predicate, self.table_name,
            independent_columns=bool(getattr(STAGE3_CONFIG, "stage3_tlp_independent_columns_enabled", False)),
        )
        queries = self.query_builder.build_queries(cleanup_dsl(real_predicate), self.table_name)
        text = (f"-- [Validation] TLP Check for: {raw_predicate}\n"
                + "".join(f"{sql}\n" for sql in queries) + "\n")
        return [sql.strip() for sql in queries], text

    def process_files(self, input_files=None, deadline=None):
        """
        方法说明: 扫描 INPUT_DIR 下的所有谓词文件并逐个流式处理。

        每条谓词实例化后立即写入测试用例文件，不再使用 readlines() 或保存完整
        generated_sqls 列表。生成内容和顺序保持原样，但内存占用只与单条谓词有关。
        """
        generation_start = time.monotonic()
        # 主流程显式传入本轮采样文件，避免失败后误读上一轮或其他残留文件。
        if input_files is None:
            files = _discover_predicate_files(INPUT_DIR)
        else:
            files = [str(path) for path in input_files if _is_predicate_file(path)]

        # 如果没找到文件，打印警告并退出
        if not files:
            print(f"[Warning] 在 {INPUT_DIR} 下没有找到任何 input 文件。")
            return {
                "success": False,
                "round_complete": False,
                "phase": "tlp_generation",
                "error": "No predicate input files found.",
                "predicates_read": 0,
                "output_files": [],
                "elapsed_seconds": time.monotonic() - generation_start,
            }

        print(f"[Generator] 发现 {len(files)} 个谓词文件，准备处理...")
        output_buffer_bytes = max(
            4096,
            int(getattr(STAGE3_CONFIG, "stage3_output_buffer_bytes", 1048576)),
        )

        output_files = []
        source_output_files = {}
        output_text_characters = {}
        source_sizes_bytes = {}
        source_text_characters = {}
        total_predicates = 0
        round_complete = True
        error_message = None

        # 遍历每一个文件
        for file_path in sorted(files):
            file_name = os.path.basename(file_path)
            logical_source_name = _logical_predicate_name(file_path)
            partial_path = None
            print(f"  -> 正在处理文件: {file_name}")

            try:
                check_deadline(deadline, "TLP test-case generation")
                # 默认写入无损压缩 .sql.gz；关闭开关时保持旧 .sql 文件名。
                plain_output_name = logical_source_name.replace(
                    "output_sql",
                    "tlp_test_case",
                ).replace(
                    ".txt", ".sql"
                )
                output_name = (
                    f"{plain_output_name}.gz"
                    if self.compress_test_cases
                    else plain_output_name
                )
                output_path = OUTPUT_DIR / output_name
                partial_path = Path(str(output_path) + ".partial")
                file_predicates = 0
                file_complete = True
                file_text_characters = 0
                file_source_characters = 0

                # 上次强制终止可能留下同名临时文件；正式输出从不使用该文件，
                # 新一轮开始前删除可立即释放磁盘，正式 .sql/.sql.gz 不受影响。
                if partial_path.exists():
                    partial_path.unlink()

                # 写入 .partial；只有完整文件或截止时间留下的有效完整测试组才提交。
                with _open_predicate_reader(
                    file_path,
                    output_buffer_bytes,
                ) as input_file, _open_test_case_writer(
                    partial_path,
                    compressed=self.compress_test_cases,
                    compression_level=self.test_case_compression_level,
                    buffer_bytes=output_buffer_bytes,
                ) as out_f:
                    # 写入文件头部说明
                    header = (
                        f"-- Source: {logical_source_name}\n"
                        "-- Type: TLP Logic Checking (WHERE Partition Check)\n"
                        f"-- TLP Mode: {self.tlp_mode}\n"
                        f"-- Target Table: {self.table_name}\n\n"
                    )
                    out_f.write(header)
                    file_text_characters += len(header)

                    for line in input_file:
                        if deadline is not None and time.monotonic() >= deadline:
                            file_complete = False
                            round_complete = False
                            break

                        file_source_characters += len(line)

                        raw_predicate = line.strip()
                        if not raw_predicate:
                            continue

                        # 保持历史随机序列，避免移除无执行作用的随机抽样后改变列选择。
                        advance_instantiation_rng(self.instantiator, self.table_name)
                        # 步骤 2: 将抽象谓词实例化为真实 SQL 条件。
                        real_predicate = self.instantiator.instantiate_predicate(
                            raw_predicate,
                            self.table_name,
                            independent_columns=bool(getattr(STAGE3_CONFIG, "stage3_tlp_independent_columns_enabled", False)),
                        )

                        # DSL 清洗与原流程一致。
                        real_predicate = cleanup_dsl(real_predicate)

                        # 生成并立即写入 TLP 四条核心语句，顺序保持不变。
                        validation_sqls = self.query_builder.build_queries(
                            real_predicate,
                            self.table_name,
                        )
                        # 每个谓词合并为一次缓冲写入，减少数百万次 Python write 调用；
                        # 文本内容、顺序和完整追溯注释保持不变。
                        block_parts = [
                            f"-- [Validation] TLP Check for: {raw_predicate}\n",
                        ]
                        block_parts.extend(f"{sql}\n" for sql in validation_sqls)
                        block_parts.append("\n")
                        block_text = "".join(block_parts)
                        out_f.write(block_text)
                        file_text_characters += len(block_text)
                        file_predicates += 1
                        total_predicates += 1

                os.replace(partial_path, output_path)
                output_files.append(str(output_path))
                source_output_files[file_name] = str(output_path)
                output_text_characters[output_name] = file_text_characters
                source_sizes_bytes[file_name] = os.path.getsize(file_path)
                source_text_characters[file_name] = file_source_characters
                file_size = output_path.stat().st_size
                elapsed = max(time.monotonic() - generation_start, 1e-9)
                storage_ratio = file_size / max(file_text_characters, 1)
                print(
                    f"     [Success] 已生成: {output_name}; "
                    f"predicates={file_predicates}, stored_bytes={file_size}, "
                    f"text_characters={file_text_characters}, "
                    f"storage_ratio={storage_ratio:.4f}, "
                    f"rate={total_predicates / elapsed:.2f} predicates/s, "
                    f"complete={file_complete}"
                )
                if not file_complete:
                    break

            except Stage3DeadlineReached as e:
                round_complete = False
                print(f"     [Stage3] Deadline reached: {e.phase}")
                # 异常式截止没有经过完整关闭和原子提交，临时文件不可作为正式追溯结果。
                if partial_path is not None and partial_path.exists():
                    try:
                        partial_path.unlink()
                    except OSError as cleanup_error:
                        print(
                            "     [Warning] 无法清理截止时未完成测试用例文件 "
                            f"{partial_path}: {cleanup_error}"
                        )
                break
            except Exception as e:
                # 捕获处理过程中的异常，打印文件名和错误信息
                print(f"     [Error] 处理文件 {file_name} 时出错: {e}")
                # 打印详细堆栈跟踪，方便调试
                import traceback
                traceback.print_exc()
                error_message = str(e)
                round_complete = False
                # 异常产生的临时文件不是完整可追溯输出，及时清理；旧正式文件保持不变。
                if partial_path is not None and partial_path.exists():
                    try:
                        partial_path.unlink()
                    except OSError as cleanup_error:
                        print(
                            "     [Warning] 无法清理未完成测试用例文件 "
                            f"{partial_path}: {cleanup_error}"
                        )
                break

        report = {
            "success": error_message is None,
            "round_complete": round_complete,
            "phase": "tlp_generation",
            "error": error_message,
            "predicates_read": total_predicates,
            "output_files": output_files,
            "source_output_files": source_output_files,
            "output_sizes_bytes": {
                os.path.basename(path): os.path.getsize(path)
                for path in output_files
                if os.path.exists(path)
            },
            "output_text_characters": output_text_characters,
            "source_sizes_bytes": source_sizes_bytes,
            "source_text_characters": source_text_characters,
            "test_case_storage": {
                "format": "gzip" if self.compress_test_cases else "plain_sql",
                "compression_level": (
                    self.test_case_compression_level
                    if self.compress_test_cases
                    else None
                ),
                "lossless_complete_output": True,
            },
            "elapsed_seconds": time.monotonic() - generation_start,
        }
        try:
            write_status_json(
                report,
                OUTPUT_DIR / "stage3_tlp_generation_status.json",
            )
        except Exception as exc:
            print(f"[Generator] Warning: failed to write compact status: {exc}")
        return report

# ================= 程序入口 =================
if __name__ == "__main__":
    # 实例化生成器并运行处理流程
    generator = TestCaseGenerator()
    generator.process_files()
