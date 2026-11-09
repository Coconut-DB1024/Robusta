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

        self.type_mapping = self._load_datatype_config()

        self.col_by_type = {k: [] for k in self.type_mapping.keys()}
        self.all_cols = []

        self._load_schema()

    def _load_datatype_config(self) -> dict:
        db_type = getattr(DB_CONFIG, "db_type", "mysql").lower()
        config_path = os.path.join(PROJECT_ROOT, "RL", "datatype", f"{db_type}.json")

        if not os.path.exists(config_path):
            print(f'[Heuristic] Warning: {db_type}.json not found; using the minimal default mapping.')

            return {
                "int": ["int"], "float": ["float", "decimal"],
                "string": ["char", "varchar", "text"], "date": ["date", "time"]
            }

        with open(config_path, "r", encoding="utf-8") as f:
            mapping = json.load(f)
        if self._current_db_type() == "sqlserver":
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
        if not self.db_exec.ping():
            raise RuntimeError(f"Cannot load required table {self.table_sql}: database connection failed.")

        if self._current_db_type() == "sqlserver":
            self._load_sqlserver_schema()
            return

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

            matched = False
            for category, keywords in self.type_mapping.items():
                if any(kw in ctype for kw in keywords):
                    self.col_by_type[category].append(cname)
                    matched = True
                    break

            if (
                not matched
                and "string" in self.col_by_type
                and self._current_db_type() not in {"postgresql", "postgres", "pg"}
                and not self._is_oracle()
            ):
                self.col_by_type["string"].append(cname)

        if not self._is_oracle():
            for k in self.col_by_type:
                if not self.col_by_type[k]:
                    self.col_by_type[k] = self.all_cols

    def _choose_col_by_category(self, category: str) -> str:
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
            return "__unknown__"
        return "string" if "string" in self.col_by_type else next(iter(self.col_by_type.keys()), "string")

    def _replace_abstract_column_tokens(self, predicate: str) -> str:
        def repl(match):
            suffix = match.group(1)
            category = self._abstract_col_category(suffix)
            return self._choose_col_by_category(category)

        return re.sub(r"(?i)\bCOL_([A-Z0-9_]+)\b", repl, predicate)

    def _replace_hole_tokens(self, predicate: str) -> str:
        text = predicate
        text = re.sub(r"(?i)\bHOLE_T_BOOL\b", "TRUE", text)
        text = re.sub(r"(?i)\bHOLE_T_STR\b", "'probe_str'", text)
        text = re.sub(r"(?i)\bHOLE_T_CMP\b", "1", text)
        text = re.sub(r"(?i)\bHOLE_T_ANY\b", "1", text)
        return text

    def _sqlserver_type_category(self, type_name: str) -> str:
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

            dtype = (self._row_value(row, "base_type") or self._row_value(row, "data_type")
                     or self._row_value(row, "Type"))
            category = self._sqlserver_type_category(dtype)
            self._sqlserver_columns[name] = category

            base = str(dtype or "").lower().split("(", 1)[0].strip()
            if category in self.col_by_type and base not in {"text", "ntext"}:
                self.col_by_type[category].append(name)
                self.all_cols.append(name)

    def _clean_sqlserver_predicate(self, predicate: str) -> str:
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

        wrappers = {
            "int", "integer", "float", "double", "decimal", "numeric", "string",
            "date", "datetime", "datetime2", "datetimeoffset", "time", "timestamp",
            "hex", "bin", "bit", "binary", "varbinary", "varchar", "nvarchar", "uniqueidentifier",
        }

        def bare_name(i):
            raw = predicate[tokens[i].start:tokens[i].end + 1]
            if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", raw):
                return raw.lower()
            return ""  

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
        if self._current_db_type() == "sqlserver":
            return self._complete_sqlserver(predicate, expected_return_type)

        req_type = "string"  
        lower_pred = predicate.lower()

        if "decimal(" in lower_pred or "numeric(" in lower_pred:
            req_type = "float" if "float" in self.col_by_type else req_type

        for category in self.type_mapping.keys():
            if f"{category}(" in lower_pred:
                req_type = category
                break

        has_generic_col = bool(re.search(r"(?i)\bcol\s*\(", predicate))
        if self._is_oracle() and not self.table_name:
            raise ValueError("Oracle schema is unavailable: no table was discovered.")
        if self._is_oracle() and has_generic_col and not self.col_by_type.get(req_type):
            raise ValueError(f"Oracle schema has no column for category {req_type!r}.")
        chosen_col = random.choice(self.col_by_type[req_type]) if self.col_by_type.get(req_type) else "unknown_col"

        clean_pred = self._replace_abstract_column_tokens(predicate)
        clean_pred = self._replace_hole_tokens(clean_pred)
        clean_pred = re.sub(r"(?i)\bcol\s*\(\s*[^)]*\)", chosen_col, clean_pred)

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

        table = self.table_sql

        if not clean_pred or len(clean_pred.strip()) < 2:
            raise ValueError("Predicate is too short or empty after cleanup.")

        expected_type = str(expected_return_type or "").upper().strip()
        where_pred = clean_pred
        if expected_type and expected_type not in {"T_BOOL", "BOOL", "BOOLEAN"}:
            where_pred = f"({clean_pred}) IS NOT NULL"

        if self._is_oracle():
            return (
                f"SELECT 1 FROM {table} "
                f"WHERE {where_pred} FETCH FIRST 1 ROW ONLY"
            )
        sql = f"SELECT 1 FROM {table} WHERE {where_pred} LIMIT 1;"
        return sql
