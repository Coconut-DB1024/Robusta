import re
from typing import Optional


def _is_sqlserver() -> bool:
    # 延迟读取，避免 pre_load 包初始化时改变其他数据库的导入流程。
    from RL.config import DB_CONFIG

    return str(getattr(DB_CONFIG, "db_type", "") or "").strip().lower() == "sqlserver"


def detect_literal_type(token: str, db_type: Optional[str] = None) -> Optional[str]: #根据一个 token（通常是右侧的字面量或常量表达式）推断数据类型。

    t = token.strip()  # 去掉首尾空白

    # 空字符串直接返回 None
    if not t:
        return None

    sqlserver = _is_sqlserver() if db_type is None else str(db_type).strip().lower() == "sqlserver"
    if sqlserver:
        if t.upper() == "NULL":
            return "null"
        if t.upper() in {"TRUE", "FALSE"}:
            return "bool"  # 项目内部条件终端；不在这里改写成可执行 T-SQL。
        if re.fullmatch(r"[Nn]?'(?:[^']|'')*'", t):
            # T-SQL 的带引号日期也是字符常量，转换由上下文决定。
            return "string"
        if re.fullmatch(r"0[xX][0-9a-fA-F]*", t):
            return "binary"  # 包含空二进制 0x；不沿用 MySQL 的 hex/int 桶。
        if re.fullmatch(r"[+-]?[0-9]+", t):
            return "int"
        if re.fullmatch(r"[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?", t):
            return "float"
        # QUOTED_IDENTIFIER ON 下双引号表示标识符，不当成字符串。
        return None

    # NULL（不区分大小写）
    if re.fullmatch(r"NULL", t, flags=re.IGNORECASE):
        return "null"

    # 布尔字面量 TRUE / FALSE（不区分大小写）
    if re.fullmatch(r"TRUE|FALSE", t, flags=re.IGNORECASE):
        return "bool"

    # 字符串字面量：单引号或双引号包裹
    if (t.startswith("'") and t.endswith("'")) or (t.startswith('"') and t.endswith('"')):
        inner = t[1:-1]  # 去掉首尾引号

        # 简单按常见格式识别 date / datetime，其余当作 string
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", inner):
            return "date"
        if re.fullmatch(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}", inner):
            return "datetime"

        return "string"

    # 纯整数（允许负号），例如 123, -10
    if re.fullmatch(r"-?\d+", t):
        return "int"

    # ================== 【新增：十六进制常量识别】 ==================
    # 支持 0xFF, 0x1a, 甚至负数 -0xFF
    if re.fullmatch(r"-?0x[0-9a-f]+", t, flags=re.IGNORECASE):
        return "hex"
    # ==========================================================

    # 浮点数（允许负号），例如 3.14, -0.5, 10.
    if re.fullmatch(r"-?\d+\.\d*", t):
        return "float"

    # 未加引号的日期格式 YYYY-MM-DD
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", t):
        return "date"

    # 未加引号的日期时间格式
    if re.fullmatch(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}", t):
        return "datetime"

    # TODO: 如果需要，可在此处扩展更多类型（TIME, YEAR, 十六进制等）
    # 当前未识别则返回 None，由上层处理（可能是列名/表达式/运算符）
    return None
