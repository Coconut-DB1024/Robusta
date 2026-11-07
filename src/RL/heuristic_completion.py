"""
作用：
- 替代原先极其耗时的 LLM 补全逻辑。
- 根据不同 DBMS 加载外部 JSON 数据类型配置文件 (RL/datatype/<db_type>.json)。
- 动态将表结构中的真实列名按照配置归类。
- 利用动态生成的启发式规则，将 col(xxx) 替换为同类型的真实列，并剥离类型外壳。
"""
import re
import random
import json
import os
from RL.db_executor import BaseDBExecutor
from RL.config import DB_CONFIG, PROJECT_ROOT, get_target_table


class HeuristicCompletionClient:
    def __init__(self, db_exec: BaseDBExecutor):
        self.db_exec = db_exec
        self.table_name, self.table_schema, self.table_sql = get_target_table(
            self._current_db_type(), getattr(db_exec, "config", DB_CONFIG)
        )

        # 1. 动态加载对应数据库的数据类型映射配置
        self.type_mapping = self._load_datatype_config()

        # 2. 动态初始化列分类字典 (e.g., {"int": [], "float": [], "string": [], ...})
        self.col_by_type = {k: [] for k in self.type_mapping.keys()}
        self.all_cols = []

        # 3. 读取表结构并归类
        self._load_schema()

    def _load_datatype_config(self) -> dict:
        """根据配置的 db_type 加载对应的 JSON 映射文件"""
        db_type = getattr(DB_CONFIG, "db_type", "mysql").lower()
        config_path = os.path.join(PROJECT_ROOT, "RL", "datatype", f"{db_type}.json")

        if not os.path.exists(config_path):
            print(f'[Heuristic] Warning: {db_type}.json not found; using the minimal default mapping.')
            # 兜底默认映射
            return {
                "int": ["int"], "float": ["float", "decimal"],
                "string": ["char", "varchar", "text"], "date": ["date", "time"]
            }

        with open(config_path, "r", encoding="utf-8") as f:
            mapping = json.load(f)
        if self._current_db_type() == "sqlserver":
            # operator_profile 是探索配置，unsupported 不参与普通列选择。
            return {k: v for k, v in mapping.items() if isinstance(v, list) and k != "unsupported"}
        return mapping

    def _current_db_type(self) -> str:
        return str(getattr(DB_CONFIG, "db_type", "mysql") or "mysql").lower().strip()

    def _is_oracle(self) -> bool:
        return self._current_db_type() in {"oracle", "oracledb", "oracle_ai26"}

    def _quote_identifier(self, identifier: str) -> str:
        if self._current_db_type() == "sqlserver":
            return "[" + str(identifier).replace("]", "]]") + "]"
        if self._current_db_type() in {"postgresql", "postgres", "pg"}:
            return '"' + str(identifier).replace('"', '""') + '"'
        if self._is_oracle():
            return '"' + str(identifier).replace('"', '""') + '"'
        return "`" + str(identifier).replace("`", "``") + "`"

    @staticmethod
    def _row_value(row, *names, default=""):
        if not isinstance(row, dict):
            return default
        for name in names:
            if name in row:
                return row[name]
        lowered = {str(key).lower(): value for key, value in row.items()}
        for name in names:
            if str(name).lower() in lowered:
                return lowered[str(name).lower()]
        return default

    def _list_columns(self, table_name: str):
        if self.table_schema is not None:
            return self.db_exec.list_columns(table_name, schema=self.table_schema)
        return self.db_exec.list_columns(table_name)

    def _load_schema(self):
        """利用 DB 执行器读取表和列属性，并根据 JSON 配置动态分桶"""
        if not self.db_exec.ping():
            raise RuntimeError(f"Cannot load required table {self.table_sql}: database connection failed.")

        if self._current_db_type() == "sqlserver":
            self._load_sqlserver_schema()
            return

        # Read only the fixed experiment table, regardless of other tables.
        res_cols = self._list_columns(self.table_name)
        if not res_cols.success or not res_cols.rows:
            raise RuntimeError(
                f"Cannot load required table {self.table_sql}: "
                f"{res_cols.error or 'table missing or no readable columns'}."
            )

        for row in res_cols.rows:
            cname = self._row_value(row, "Field", "FIELD", "column_name", "COLUMN_NAME")
            ctype = str(self._row_value(row, "Type", "TYPE", "data_type", "DATA_TYPE")).lower()
            if not cname:
                continue
            self.all_cols.append(cname)

            # 根据加载的 json 配置进行动态归类匹配
            matched = False
            for category, keywords in self.type_mapping.items():
                # 只要数据库返回的列类型中包含关键字，就归入该类别
                if any(kw in ctype for kw in keywords):
                    self.col_by_type[category].append(cname)
                    matched = True
                    break

            # PostgreSQL 的未知类型不能兜底为字符串，避免 COL_STRING 误选 bytea 等非字符串列；
            # MySQL 继续保留原有的宽松兜底行为。
            if (
                not matched
                and "string" in self.col_by_type
                and self._current_db_type() not in {"postgresql", "postgres", "pg"}
                and not self._is_oracle()
            ):
                self.col_by_type["string"].append(cname)

        # 兜底机制：如果当前表里缺少某种数据类型的列，则拿全部列作为备选（防止程序崩溃）
        if not self._is_oracle():
            for k in self.col_by_type:
                if not self.col_by_type[k]:
                    self.col_by_type[k] = self.all_cols

    def _choose_col_by_category(self, category: str) -> str:
        """
        阶段17-5：根据抽象列类型选择真实列。
        例如 COL_INT -> int 类列，COL_STRING -> string 类列。
        """
        category = str(category or "").lower()
        if self._current_db_type() == "sqlserver":
            candidates = self.col_by_type.get(category, [])
            if not candidates:
                raise ValueError(f"SQL Server schema has no column for category {category!r}.")
            return random.choice(candidates)
        candidates = self.col_by_type.get(category) or (
            [] if self._is_oracle() else self.all_cols
        )
        if not candidates:
            if self._is_oracle():
                raise ValueError(f"Oracle schema has no column for category {category!r}.")
            return "unknown_col"
        return random.choice(candidates)

    def _abstract_col_category(self, suffix: str) -> str:
        """
        阶段17-5：将 COL_* 的后缀映射到 heuristic_completion 的列类型桶。
        """
        text = str(suffix or "").lower()
        if self._current_db_type() == "sqlserver":
            return self._sqlserver_type_category(text)

        if text in self.col_by_type:
            return text

        if any(k in text for k in ["int", "tinyint", "smallint", "mediumint", "bigint", "bool", "boolean", "bit"]):
            return "int" if "int" in self.col_by_type else next(iter(self.col_by_type.keys()), "string")

        if any(k in text for k in ["float", "double", "decimal", "numeric", "real"]):
            return "float" if "float" in self.col_by_type else next(iter(self.col_by_type.keys()), "string")

        if any(k in text for k in ["str", "string", "char", "varchar", "text", "blob", "json"]):
            return "string" if "string" in self.col_by_type else next(iter(self.col_by_type.keys()), "string")

        if any(k in text for k in ["date", "time", "timestamp", "year"]):
            return "date" if "date" in self.col_by_type else next(iter(self.col_by_type.keys()), "string")

        if self._is_oracle():
            # 未知 COL_* 后缀不强行归为字符串，避免 Oracle 类型错配。
            return "__unknown__"
        return "string" if "string" in self.col_by_type else next(iter(self.col_by_type.keys()), "string")

    def _replace_abstract_column_tokens(self, predicate: str) -> str:
        """
        阶段17-5：替换裸露的 COL_INT / COL_FLOAT / COL_STRING 等抽象列 token。
        """
        def repl(match):
            suffix = match.group(1)
            category = self._abstract_col_category(suffix)
            return self._choose_col_by_category(category)

        return re.sub(r"(?i)\bCOL_([A-Z0-9_]+)\b", repl, predicate)

    def _replace_hole_tokens(self, predicate: str) -> str:
        """
        阶段17-5：兜底替换 HOLE_T_*，避免送入数据库时残留抽象坑位。
        """
        text = predicate
        text = re.sub(r"(?i)\bHOLE_T_BOOL\b", "TRUE", text)
        text = re.sub(r"(?i)\bHOLE_T_STR\b", "'probe_str'", text)
        text = re.sub(r"(?i)\bHOLE_T_CMP\b", "1", text)
        text = re.sub(r"(?i)\bHOLE_T_ANY\b", "1", text)
        return text

    def _sqlserver_type_category(self, type_name: str) -> str:
        # 按完整基础类型匹配，避免 timestamp 被 time、sql_variant 被 int 命中。
        base = str(type_name or "").lower().split("(", 1)[0].strip()
        aliases = {"nchar": "string", "str": "string", "bool": "int", "boolean": "int",
                   "double": "float", "hex": "binary", "bin": "binary"}
        if base in aliases:
            return aliases[base]
        for category, keywords in self.type_mapping.items():
            if base == category or base in keywords:
                return category
        return "__unknown__"

    def _load_sqlserver_schema(self):
        result = self._list_columns(self.table_name)
        self._sqlserver_columns = {}
        if not result.success or not result.rows:
            raise RuntimeError(
                f"Cannot load required table {self.table_sql}: "
                f"{result.error or 'table missing or no readable columns'}."
            )
        for row in result.rows:
            name = self._row_value(row, "Field", "column_name")
            if not name or self._row_value(row, "is_hidden", default=False):
                continue
            # SQL Server 别名类型按执行器提供的 base_type 归类，不猜测别名中的单词。
            dtype = (self._row_value(row, "base_type") or self._row_value(row, "data_type")
                     or self._row_value(row, "Type"))
            category = self._sqlserver_type_category(dtype)
            self._sqlserver_columns[name] = category
            # Legacy LOBs are not interchangeable with ordinary scalar strings
            # (e.g. text equality and LOWER(ntext) fail). Keep explicit column
            # references available for deliberate probes, but never pick these
            # columns for a generic placeholder.
            base = str(dtype or "").lower().split("(", 1)[0].strip()
            if category in self.col_by_type and base not in {"text", "ntext"}:
                self.col_by_type[category].append(name)
                self.all_cols.append(name)

    def _clean_sqlserver_predicate(self, predicate: str) -> str:
        # 复用主线已有的 sqlglot 依赖，仅使用 token 位置改写 DSL，绝不重新生成 T-SQL。
        # 因此 N 字符串、转义标识符、!< / !> 及真实函数的原始写法都能保留。
        import sqlglot

        tokens = sqlglot.Dialect.get_or_raise("tsql").tokenize(predicate)
        if not tokens:
            raise ValueError("SQL Server predicate is empty.")
        pairs, stack = {}, []
        for i, token in enumerate(tokens):
            if token.token_type.name == "L_PAREN":
                stack.append(i)
            elif token.token_type.name == "R_PAREN":
                if not stack:
                    raise ValueError("Unbalanced SQL Server predicate parentheses.")
                pairs[stack.pop()] = i
        if stack:
            raise ValueError("Unbalanced SQL Server predicate parentheses.")

        # 与预处理 DSL 对齐；CHAR/NCHAR 是真实函数，不属于可删除的类型外壳。
        wrappers = {
            "int", "integer", "float", "double", "decimal", "numeric", "string",
            "date", "datetime", "datetime2", "datetimeoffset", "time", "timestamp",
            "hex", "bin", "bit", "binary", "varbinary", "varchar", "nvarchar", "uniqueidentifier",
        }

        def bare_name(i):
            raw = predicate[tokens[i].start:tokens[i].end + 1]
            if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", raw):
                return raw.lower()
            return ""  # 字符串、[COL_INT]、"TRUE" 等不是 DSL token。

        def qualified(i):
            return ((i > 0 and tokens[i - 1].token_type.name in {"DOT", "PARAMETER"})
                    or (i + 1 < len(tokens) and tokens[i + 1].token_type.name == "DOT"))

        def top_level(lo, hi):
            while lo < hi:
                yield lo
                lo = pairs[lo] + 1 if lo in pairs else lo + 1

        req_type = "string"
        for category in self.type_mapping:
            if any(bare_name(i) in wrappers and not qualified(i)
                   and self._sqlserver_type_category(bare_name(i)) == category
                   and tokens[i + 1].token_type.name == "L_PAREN"
                   for i in range(len(tokens) - 1)):
                req_type = category
                break
        chosen_hints = {}

        def choose_hint(lo, hi):
            hint = tuple(t.text for t in tokens[lo:hi])
            if not hint:
                raise ValueError("SQL Server col() requires a column hint.")
            if hint not in chosen_hints:
                # 已有真实列优先；限定名只取列部分，FROM 已锁定目标表。
                name = tokens[hi - 1].text
                category = self._sqlserver_columns.get(name)
                if category is not None:
                    if category not in self.col_by_type:
                        raise ValueError(f"Unsupported SQL Server column type: {name!r}.")
                    chosen_hints[hint] = name
                else:
                    chosen_hints[hint] = self._choose_col_by_category(req_type)
            return self._quote_identifier(chosen_hints[hint])

        def render(lo, hi):
            if lo >= hi:
                return ""
            pieces, cursor, i = [], tokens[lo].start, lo
            while i < hi:
                token = tokens[i]
                end, next_i = token.end + 1, i + 1
                name = bare_name(i) if not qualified(i) else ""
                replacement = predicate[token.start:end]
                is_call = i + 1 < hi and tokens[i + 1].token_type.name == "L_PAREN"
                if name and is_call:
                    close = pairs[i + 1]
                    end, next_i = tokens[close].end + 1, close + 1
                    inner_lo = i + 2
                    if name == "col":
                        replacement = choose_hint(inner_lo, close)
                    elif name in wrappers:
                        if inner_lo == close or any(tokens[j].token_type.name == "COMMA"
                                                    for j in top_level(inner_lo, close)):
                            raise ValueError(f"SQL Server DSL wrapper {name} requires one expression.")
                        replacement = "(" + render(inner_lo, close) + ")"
                    else:
                        # CAST 的 AS 之后、CONVERT 的首参数是类型声明，必须保留长度/精度。
                        separator = next((j for j in top_level(inner_lo, close)
                                          if (name in {"cast", "try_cast", "parse", "try_parse"}
                                              and bare_name(j) == "as")
                                          or (name in {"convert", "try_convert"}
                                              and tokens[j].token_type.name == "COMMA")), None)
                        if separator is None:
                            inner = render(inner_lo, close)
                        elif name in {"convert", "try_convert"}:
                            inner = predicate[tokens[i + 1].end + 1:tokens[separator].end + 1]
                            inner += render(separator + 1, close)
                        else:
                            inner = render(inner_lo, separator) + " "
                            inner += predicate[tokens[separator].start:tokens[close].start]
                        replacement = predicate[token.start:tokens[i + 1].end + 1] + inner + ")"
                elif name.startswith("col_"):
                    replacement = self._quote_identifier(self._choose_col_by_category(
                        self._abstract_col_category(name[4:])))
                elif name in {"true", "false", "hole_t_bool"}:
                    replacement = "(1 = 0)" if name == "false" else "(1 = 1)"
                elif name in {"hole_t_str", "hole_t_cmp", "hole_t_any"}:
                    replacement = "N'probe_str'" if name == "hole_t_str" else "1"
                pieces.extend((predicate[cursor:token.start], replacement))
                cursor, i = end, next_i
            pieces.append(predicate[cursor:tokens[hi - 1].end + 1])
            return "".join(pieces)

        return render(0, len(tokens))

    def _complete_sqlserver(self, predicate: str, expected_return_type=None) -> str:
        from sqlglot import exp
        from RL.sqlserver_sqlglot import parse_sqlserver_expression

        if not self.table_name:
            raise ValueError(f"SQL Server schema is unavailable: {DB_CONFIG.table_name} has no usable columns.")
        clean_pred = self._clean_sqlserver_predicate(predicate)
        # The outer node's category selects only the validation wrapper.
        # Candidate operand compatibility is learned from database execution,
        # so do not reject or coerce condition/scalar mixtures in the subtree.
        expected = str(expected_return_type or "").upper().strip()
        if expected:
            condition = expected in {"T_BOOL", "BOOL", "BOOLEAN"}
        else:
            expression = parse_sqlserver_expression(clean_pred)
            while isinstance(expression, exp.Paren):
                expression = expression.this
            condition = isinstance(expression, (exp.Predicate, exp.And, exp.Or, exp.Not))
        where_pred = clean_pred if condition else f"({clean_pred}\n) IS NOT NULL"
        table = self.table_sql
        return f"SELECT TOP (1) 1 FROM {table} WHERE {where_pred};"

    def complete_sql(self, predicate: str, expected_return_type=None) -> str:
        """
        启发式实例化与 SQL 拼接 (终极加固版)：
        1. 采用 5 层循环正则清洗，剥离所有嵌套的包裹器。
        2. 使用 SELECT 1 ... LIMIT 1，极速验证语法，保护数据库性能。
        """
        if self._current_db_type() == "sqlserver":
            return self._complete_sqlserver(predicate, expected_return_type)

        # 1. 动态类型推断
        req_type = "string"  # 默认类型
        lower_pred = predicate.lower()

        if "decimal(" in lower_pred or "numeric(" in lower_pred:
            req_type = "float" if "float" in self.col_by_type else req_type

        for category in self.type_mapping.keys():
            if f"{category}(" in lower_pred:
                req_type = category
                break

        # 2. 随机选取真实列
        # Oracle 不允许用不存在的 unknown_col 继续执行。
        has_generic_col = bool(re.search(r"(?i)\bcol\s*\(", predicate))
        if self._is_oracle() and not self.table_name:
            raise ValueError("Oracle schema is unavailable: no table was discovered.")
        if self._is_oracle() and has_generic_col and not self.col_by_type.get(req_type):
            raise ValueError(f"Oracle schema has no column for category {req_type!r}.")
        chosen_col = random.choice(self.col_by_type[req_type]) if self.col_by_type.get(req_type) else "unknown_col"

        # 3. 替换列名
        clean_pred = self._replace_abstract_column_tokens(predicate)
        clean_pred = self._replace_hole_tokens(clean_pred)
        clean_pred = re.sub(r"(?i)\bcol\s*\(\s*[^)]*\)", chosen_col, clean_pred)

        # 4. 【深度清洗】循环 5 次剥离数值类型包装 (防止 int(string(10)) 等嵌套怪胎)
        wrapper_keys = list(self.type_mapping.keys()) + ["decimal", "numeric", "hex", "bin", "bit"]
        wrappers_pattern = "|".join(wrapper_keys)
        regex_pattern = re.compile(rf"(?i)\b({wrappers_pattern})\s*\(\s*([^()]+)\s*\)")

        curr_sql = clean_pred
        for _ in range(5):
            new_sql = regex_pattern.sub(lambda m: m.group(2).strip(), curr_sql)
            if new_sql == curr_sql:
                break
            curr_sql = new_sql
        clean_pred = curr_sql

        # 5. 【极速执行保护】使用 SELECT 1 和 LIMIT 1，我们只关心语法对不对，不关心真实数据！
        table = self.table_sql

        # 如果清理后的谓词是空的，或者全是不合法符号，直接抛出异常走兜底
        if not clean_pred or len(clean_pred.strip()) < 2:
            raise ValueError("Predicate is too short or empty after cleanup.")

        # Candidate bitmap 验证非布尔表达式时，不能直接把表达式放到 WHERE 后面。
        expected_type = str(expected_return_type or "").upper().strip()
        where_pred = clean_pred
        if expected_type and expected_type not in {"T_BOOL", "BOOL", "BOOLEAN"}:
            where_pred = f"({clean_pred}) IS NOT NULL"

        if self._is_oracle():
            # Oracle 12c+/AI26 支持 FETCH FIRST；python-oracledb 不接受尾部分号。
            return (
                f"SELECT 1 FROM {table} "
                f"WHERE {where_pred} FETCH FIRST 1 ROW ONLY"
            )
        sql = f"SELECT 1 FROM {table} WHERE {where_pred} LIMIT 1;"
        return sql
