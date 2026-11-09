import os        
import re        
import random    
import glob      
import json      
import time      
import gzip      
import io        
from contextlib import contextmanager
from pathlib import Path  
from typing import Dict, List, Tuple, Any, Optional 


import sys  

project_root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

sys.path.append(project_root_path)


from RL.config import PROJECT_ROOT, DB_CONFIG, STAGE3_CONFIG, get_target_table

from RL.db_executor import create_db_executor

from BugFinding.tlp_random_compat import advance_instantiation_rng
from pkg.stage3_runtime import (
    Stage3DeadlineReached,
    check_deadline,
    write_status_json,
)


DSL_WRAPPERS = [
    "int", "float", "double", "decimal",
    "string", "char", "varchar",
    "date", "datetime", "timestamp", "time",
    "col", "hex", "bin", "bit"
]


DSL_CLEAN_PATTERN = re.compile(
    r"\b(" + "|".join(DSL_WRAPPERS) + r")\s*\(\s*([^()]+)\s*\)",
    re.IGNORECASE
)


def cleanup_dsl(text: str) -> str:
    if not text:
        return text
    if _is_sqlserver_db():
        return _rewrite_sqlserver_dsl(text, strip_wrappers=True)

    curr_text = text

    for _ in range(5):
        new_text = DSL_CLEAN_PATTERN.sub(lambda m: m.group(2).strip(), curr_text)
        if new_text == curr_text:
            break
        curr_text = new_text
    return curr_text


INPUT_DIR = PROJECT_ROOT / "pkg" / "generated_sqls"

OUTPUT_DIR = PROJECT_ROOT / "BugFinding" / "generated_test_cases"

OPERATOR_CONFIG_DIR = PROJECT_ROOT / "BugFinding" / "db_operators"
EXACT_COLUMN_TYPE_HINT_PREFIX = "__TYPE__"

ORACLE_DB_TYPES = {"oracle", "oracledb", "oracle_ai26"}


def _is_oracle_db() -> bool:
    return str(getattr(DB_CONFIG, "db_type", "") or "").strip().lower() in ORACLE_DB_TYPES


def _is_sqlserver_db() -> bool:
    return str(getattr(DB_CONFIG, "db_type", "") or "").strip().lower() == "sqlserver"


def _sqlserver_quote_identifier(name: str) -> str:
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
    text = " ".join(str(value or "").strip().lower().split())
    if text.startswith("pg_catalog."):
        text = text[len("pg_catalog.") :]
    if "(" in text:
        text = " ".join(re.sub(r"\([^()]*\)", "", text).split())
    return _COLUMN_TYPE_ALIASES.get(text, text)


os.makedirs(OUTPUT_DIR, exist_ok=True)


@contextmanager
def _open_test_case_writer(
    output_path: Path,
    *,
    compressed: bool,
    compression_level: int,
    buffer_bytes: int,
):
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
    file_name = os.path.basename(str(path))
    if file_name.lower().endswith(".txt.gz"):
        return file_name[:-3]
    return file_name


@contextmanager
def _open_predicate_reader(file_path: str, buffer_bytes: int):
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
    def __init__(self):
        self.columns = {}
        self.table_name, self.table_schema, self.table_sql = get_target_table(
            DB_CONFIG.db_type, DB_CONFIG
        )

        self._load_schema()

    def _load_schema(self):
        print('[SchemaLoader] Connecting to the database to load metadata...')

        executor = create_db_executor(autoconnect=True)

        self.table_name, self.table_schema, self.table_sql = get_target_table(
            DB_CONFIG.db_type, getattr(executor, "config", DB_CONFIG)
        )

        table_name = self.table_name

        try:
            if self.table_schema is not None:
                res = executor.list_columns(table_name, schema=self.table_schema)
            else:
                res = executor.list_columns(table_name)
        finally:
            executor.close()

        if not res.success or not res.rows:
            raise RuntimeError(
                f"Cannot load required table {self.table_sql}: "
                f"{res.error or 'table missing or no readable columns'}."
            )

        cols = []

        if res.rows:
            for row in res.rows:
                col_name = row.get('Field') or row.get('field')

                col_type = row.get('Type') or row.get('type')

                if col_name and col_type:
                    column = dict(row) if _is_sqlserver_db() else {}
                    column.update(name=col_name, type=col_type)
                    if _is_sqlserver_db():
                        column.setdefault("table_schema", self.table_schema)
                        column.setdefault("table_name", table_name)
                    cols.append(column)

        self.columns[table_name] = cols
        self.columns[self.table_sql] = cols
        print(f'[SchemaLoader] Metadata loaded: {table_name} has {len(cols)} columns.')

    def get_columns(self, table_name: str):
        return self.columns.get(table_name, [])


class SmartInstantiator:
    def __init__(self, schema_loader: DBMetaDataLoader):
        self.loader = schema_loader

        self.operator_map = {}

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

        if _is_sqlserver_db():
            from RL.sqlserver_evidence_types import NUMERIC, STRINGS, BINARY, TEMPORAL
            self.type_category_mapping = {
                "NUMERIC": sorted(NUMERIC), "STRING": sorted(STRINGS),
                "DATETIME": sorted(TEMPORAL), "BINARY": sorted(BINARY),
                "GUID": ["uniqueidentifier"],
            }
        elif _is_oracle_db():
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
                'JSON': ['JSON', 'LONGTEXT'] 
            }

        self._load_operator_config()

    def _load_operator_config(self):
        db_type = str(getattr(DB_CONFIG, "db_type", "") or "").strip().lower()

        config_db_type = "oracle" if db_type in ORACLE_DB_TYPES else db_type

        specific_config = OPERATOR_CONFIG_DIR / f"{config_db_type}.json"

        common_config = OPERATOR_CONFIG_DIR / "common.json"

        source_type = "None"

        if db_type in ORACLE_DB_TYPES:
            if os.path.exists(specific_config):
                try:
                    with open(specific_config, 'r', encoding='utf-8') as f:
                        self.operator_map = json.load(f)
                    source_type = f"Specific ({config_db_type}.json)"
                except Exception as e:
                    print(f'[Config] Failed to load the Oracle-specific configuration: {e}')
            if not self.operator_map:
                self.operator_map = dict(ORACLE_OPERATOR_MAP)
                source_type = "Oracle Stage-3 built-in profile"

            return

        if os.path.exists(specific_config):
            try:
                with open(specific_config, 'r', encoding='utf-8') as f:
                    self.operator_map = json.load(f)
                source_type = f"Specific ({db_type}.json)"
            except Exception as e:
                print(f'[Config] Failed to load the database-specific configuration: {e}')

        if db_type == "sqlserver" and not self.operator_map:
            raise ValueError(f"SQL Server operator configuration unavailable: {specific_config}")
        if not self.operator_map and os.path.exists(common_config):
            try:
                with open(common_config, 'r', encoding='utf-8') as f:
                    self.operator_map = json.load(f)
                source_type = "Common (common.json)"
            except Exception as e:
                print(f'[Config] Failed to load the common configuration: {e}')

        if not self.operator_map:
            source_type = "Hardcoded Fallback"

            self.operator_map = {
                "NUMERIC": ["=", ">", "<", ">=", "<=", "!=", "<>"],
                "STRING": ["=", "!=", "<>", "LIKE", "NOT LIKE"],
                "DATETIME": ["=", ">", "<", ">=", "<=", "!="],
                "JSON": ["="] 
            }

    def _guess_operator_type(self, predicate: str) -> str:
        if _is_sqlserver_db():
            import sqlglot
            tokens = sqlglot.Dialect.get_or_raise("tsql").tokenize(predicate)

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

        for category, ops in self.operator_map.items():
            for op in ops:
                if op in predicate:
                    return category

        return 'ANY'

    def _resolve_hint_operator_type(self, hint_name: str, predicate_context: str) -> str:
        op_type = self._guess_operator_type(predicate_context)

        if hint_name.endswith("_mock"):
            exact_type = hint_name.replace("_mock", "").upper()

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
        name = str(column.get("name", "") or "").strip().strip('"')
        return name.lower() == "id"

    def _table_column_cache(self, table_name):
        source = self.loader.get_columns(table_name)
        db_type = getattr(DB_CONFIG, "db_type", None)
        caches = getattr(self, "_table_column_caches", None)
        if caches is None:
            caches = self._table_column_caches = {}
        cached = caches.get(table_name)
        if (cached is not None and cached["source"] is source
                and cached["column_count"] == len(source)
                and cached["db_type"] == db_type):
            return cached
        cached = {
            "source": source,
            "column_count": len(source),
            "db_type": db_type,
            "columns": None,
            "exact_candidates": None,
            "ranked_candidates": {},
            "binding_keys": {},
            "binding_groups": {},
        }
        caches[table_name] = cached
        return cached

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
        cached = self._table_column_cache(table_name)
        if cached["columns"] is None:
            oracle_db = _is_oracle_db()
            cached["columns"] = [
                column
                for column in cached["source"]
                if not (
                    self._oracle_is_id_column(column)
                    if oracle_db
                    else str(column.get("name", "")) == "id"
                )
            ]
        columns = cached["columns"]
        if not columns:
            return None

        exact_type = (
            self._exact_type_from_hint(hint_name)
            if self.exact_column_type_guidance_enabled
            else None
        )
        if exact_type:
            if random.random() < self.column_type_exploration_ratio:
                return random.choice(columns)
            if cached["exact_candidates"] is None:
                by_type = {}
                for column in columns:
                    dtype = _normalize_column_type(column.get("type"))
                    by_type.setdefault(dtype, []).append(column)
                cached["exact_candidates"] = by_type
            exact_candidates = cached["exact_candidates"].get(exact_type, [])
            if exact_candidates:
                if binding_candidates is not None:
                    binding_candidates.extend(exact_candidates)
                return random.choice(exact_candidates)

        op_type = self._resolve_hint_operator_type(hint_name, predicate_context)
        valid_keywords = tuple(self.type_category_mapping.get(op_type, ()))
        ranking_key = (op_type, valid_keywords)
        candidates = cached["ranked_candidates"].get(ranking_key)
        if candidates is None:
            candidates = []
            for column in columns:
                col_type = str(column.get("type", "") or "").upper()
                is_match = any(keyword in col_type for keyword in valid_keywords)
                if is_match:
                    score = 50
                elif op_type == "ANY":
                    score = 20
                else:
                    score = 10
                candidates.append((column, score))
            candidates.sort(key=lambda item: item[1], reverse=True)
            cached["ranked_candidates"][ranking_key] = candidates
        top_k = min(len(candidates), 3)

        if random.random() < 0.8:
            if binding_candidates is not None:
                binding_candidates.extend(c for c, score in candidates[:top_k] if score == 50)
            return random.choice(candidates[:top_k])[0]
        return random.choice(candidates)[0]

    def _sqlserver_table_cache(self, table_name):
        from RL.sqlserver_evidence_types import base_type

        columns = self.loader.get_columns(table_name)
        caches = getattr(self, "_sqlserver_table_caches", None)
        if caches is None:
            caches = self._sqlserver_table_caches = {}
        cached = caches.get(table_name)

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

        if random.random() < self.column_type_exploration_ratio:
            return random.choice(columns)
        if binding_candidates is not None:
            binding_candidates.extend(candidates)
        return random.choice(candidates)

    @staticmethod
    def _sqlserver_lob_capability(hint, predicate, dtype):
        import sqlglot
        from sqlglot import exp
        markers = set()

        def replace(value):
            if value != hint:
                return "1"
            marker = f"__stage3_lob_{len(markers)}"
            markers.add(marker)
            return _sqlserver_quote_identifier(marker)

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
                continue  
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
        column = self._choose_column_metadata(table_name, hint_name, predicate_context)
        if _is_sqlserver_db():
            return _sqlserver_quote_identifier(column["name"])
        return str(column.get("name")) if column is not None else "col_unknown"

    def _classify_column_category(self, col_type: str) -> str:
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
        dtype = (_sqlserver_column_type(column) if _is_sqlserver_db()
                 else str(column.get("type", "")))
        declared_type = str(column.get("type", "")) if column.get("is_user_defined") else ""
        return (
            re.sub(r"\s+", "", dtype.lower()),
            declared_type,
            column.get("collation_name", column.get("collation")),
            column.get("character_set_name"),
        )

    def _compatible_columns(self, table_name, selected, candidates):
        cached = self._table_column_cache(table_name)
        keys = cached["binding_keys"]
        selected_id = id(selected)
        if selected_id not in keys:
            keys[selected_id] = self._column_binding_key(selected)
        pool_key = tuple(id(column) for column in candidates)
        groups = cached["binding_groups"].get(pool_key)
        if groups is None:
            groups = {}
            for column in candidates:
                column_id = id(column)
                if column_id not in keys:
                    keys[column_id] = self._column_binding_key(column)
                groups.setdefault(keys[column_id], []).append(column)
            cached["binding_groups"][pool_key] = groups
        return groups.get(keys[selected_id], [])

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

                compatible = []
                if selected is not None and selected in candidates:
                    compatible = self._compatible_columns(table_name, selected, candidates)
                bindings[hint] = (selected, compatible)
                return render(selected)
            selected, compatible = bindings[hint]
            return render(random.choice(compatible) if len(compatible) > 1 else selected)

        if _is_sqlserver_db():
            return _rewrite_sqlserver_dsl(raw_predicate, replace)

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
        if independent_columns:
            return self._instantiate_independent_where_columns(raw_predicate, table_name)
        if _is_sqlserver_db():
            return self._instantiate_sqlserver_columns(raw_predicate, table_name)

        pattern = re.compile(r"col\(([^)]+)\)")
        matches = pattern.findall(raw_predicate)
        current_predicate = raw_predicate

        for hint_name in set(matches):
            real_col = self.find_best_column(table_name, hint_name, raw_predicate)

            target_str = f"col({hint_name})"

            current_predicate = current_predicate.replace(target_str, real_col)

        return current_predicate


class TLPQueryBuilder:
    def build_queries(self, real_predicate: str, table_name: str) -> List[str]:
        return self._build_where_tlp_queries(real_predicate, table_name)

    def _build_where_tlp_queries(self, real_predicate: str, table_name: str) -> List[str]:
        unknown_condition = f"({real_predicate}) IS NULL"
        if _is_sqlserver_db():
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
    def __init__(self):
        self.meta_loader = DBMetaDataLoader()

        self.instantiator = SmartInstantiator(self.meta_loader)

        self.table_name = self.meta_loader.table_sql

        self.tlp_mode = "where"  
        self.query_builder = TLPQueryBuilder()

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

    def build_stratified_case(self, raw_predicate):
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
        generation_start = time.monotonic()

        if input_files is None:
            files = _discover_predicate_files(INPUT_DIR)
        else:
            files = [str(path) for path in input_files if _is_predicate_file(path)]

        if not files:
            print(f'[Warning] No input files found in {INPUT_DIR}.')
            return {
                "success": False,
                "round_complete": False,
                "phase": "tlp_generation",
                "error": "No predicate input files found.",
                "predicates_read": 0,
                "output_files": [],
                "elapsed_seconds": time.monotonic() - generation_start,
            }

        print(f'[Generator] Found {len(files)} predicate files; starting processing...')
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

        for file_path in sorted(files):
            file_name = os.path.basename(file_path)
            logical_source_name = _logical_predicate_name(file_path)
            partial_path = None

            try:
                check_deadline(deadline, "TLP test-case generation")

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

                if partial_path.exists():
                    partial_path.unlink()

                with _open_predicate_reader(
                    file_path,
                    output_buffer_bytes,
                ) as input_file, _open_test_case_writer(
                    partial_path,
                    compressed=self.compress_test_cases,
                    compression_level=self.test_case_compression_level,
                    buffer_bytes=output_buffer_bytes,
                ) as out_f:
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

                        advance_instantiation_rng(self.instantiator, self.table_name)

                        real_predicate = self.instantiator.instantiate_predicate(
                            raw_predicate,
                            self.table_name,
                            independent_columns=bool(getattr(STAGE3_CONFIG, "stage3_tlp_independent_columns_enabled", False)),
                        )

                        real_predicate = cleanup_dsl(real_predicate)

                        validation_sqls = self.query_builder.build_queries(
                            real_predicate,
                            self.table_name,
                        )

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
                if not file_complete:
                    break

            except Stage3DeadlineReached as e:
                round_complete = False
                print(f"     [Stage3] Deadline reached: {e.phase}")

                if partial_path is not None and partial_path.exists():
                    try:
                        partial_path.unlink()
                    except OSError as cleanup_error:
                        print(
                            f'     [Warning] Failed to remove the test-case file left incomplete at the deadline {partial_path}: {cleanup_error}'
                        )
                break
            except Exception as e:
                print(f'     [Error] Failed to process file {file_name}: {e}')

                import traceback
                traceback.print_exc()
                error_message = str(e)
                round_complete = False

                if partial_path is not None and partial_path.exists():
                    try:
                        partial_path.unlink()
                    except OSError as cleanup_error:
                        print(
                            f'     [Warning] Failed to remove incomplete test-case file {partial_path}: {cleanup_error}'
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


if __name__ == "__main__":
    generator = TestCaseGenerator()
    generator.process_files()
