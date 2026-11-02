from typing import List
import re
from .sql_type_utils import detect_literal_type, _is_sqlserver


# SQL Server 的定界标识符可以包含空格、点、括号以及转义的 ] / "。
_SQLSERVER_NAME_PART = r'(?:\[(?:[^\]]|\]\])+\]|"(?:[^"]|"")+"|(?:[^\W\d]|[_#@])[\w$#@]*)'
_SQLSERVER_IDENTIFIER = re.compile(
    _SQLSERVER_NAME_PART + r'(?:\s*\.\s*' + _SQLSERVER_NAME_PART + r')*'
)
_SQLSERVER_NUMBER = re.compile(
    r'(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?'
)
_SQLSERVER_KEYWORDS = set(
    "AND OR NOT LIKE IN BETWEEN IS NULL TRUE FALSE DISTINCT FROM AS COLLATE ESCAPE "
    "CASE WHEN THEN ELSE END SELECT WHERE EXISTS ALL ANY SOME "
    "CURRENT_USER SESSION_USER SYSTEM_USER USER".split()
)
# 保留已有 DSL；NCHAR/CHAR 是函数，不能按类型包装器吞掉参数。
_SQLSERVER_DSL_WRAPPERS = {
    "col", "int", "integer", "float", "double", "decimal", "numeric",
    "string", "date", "datetime", "datetime2", "datetimeoffset", "time",
    "timestamp", "hex", "bin", "bit", "binary", "varbinary", "varchar", "nvarchar",
}
_SQLSERVER_DATEPART_FUNCTIONS = {"DATEADD", "DATEDIFF", "DATEDIFF_BIG", "DATEPART", "DATENAME"}
_SQLSERVER_CAST_FUNCTIONS = {"CAST", "TRY_CAST"}
_SQLSERVER_CONVERT_FUNCTIONS = {"CONVERT", "TRY_CONVERT"}


def _tokenize_sqlserver(expr: str) -> List[str]:
    tokens: List[str] = []
    i = 0
    while i < len(expr):
        if expr[i].isspace():
            i += 1
            continue
        # 字符串单独扫描，保留 N 前缀、双单引号及内部空白。
        start = i
        quote_at = i + 1 if expr[i:i + 2].upper() == "N'" else i
        if expr[quote_at:quote_at + 1] == "'":
            i = quote_at + 1
            while i < len(expr):
                if expr[i] == "'":
                    if expr[i:i + 2] == "''":
                        i += 2
                        continue
                    i += 1
                    break
                i += 1
            else:
                raise ValueError("Unterminated SQL Server string literal.")
            tokens.append(expr[start:i])
            continue
        # 只在 token 之间跳过注释，不触碰字符串/标识符中的注释符号。
        if expr.startswith("--", i):
            end = expr.find("\n", i + 2)
            i = len(expr) if end < 0 else end + 1
            continue
        if expr.startswith("/*", i):
            depth = 1
            i += 2
            while i < len(expr) and depth:
                if expr.startswith("/*", i):
                    depth += 1
                    i += 2
                elif expr.startswith("*/", i):
                    depth -= 1
                    i += 2
                else:
                    i += 1
            if depth:
                raise ValueError("Unterminated SQL Server block comment.")
            continue
        identifier = _SQLSERVER_IDENTIFIER.match(expr, i)
        if identifier:
            tokens.append(identifier.group())
            i = identifier.end()
            continue
        if expr[i] in '["':
            raise ValueError("Unterminated or empty SQL Server delimited identifier.")
        if expr[i:i + 2].lower() == "0x":
            binary = re.match(r"0[xX][0-9A-Za-z]*", expr[i:]).group()
            tokens.append(binary)
            i += len(binary)
            continue
        if expr[i] in "+-" and (
            not tokens or tokens[-1].upper() in {
                "(", ",", "=", "<>", "!=", "<", ">", "<=", ">=", "!<", "!>",
                "+", "-", "*", "/", "%", "&", "|", "^", "~", "<<", ">>",
                "AND", "OR", "NOT", "BETWEEN", "THEN", "ELSE", "FROM",
            }
        ):
            number = _SQLSERVER_NUMBER.match(expr, i + 1)
            if number and expr[i + 1:i + 3].lower() != "0x":
                tokens.append(expr[i:number.end()])
                i = number.end()
                continue
        number = _SQLSERVER_NUMBER.match(expr, i)
        if number:
            tokens.append(number.group())
            i = number.end()
            continue
        operator = expr[i:i + 2]
        if operator in {"<=", ">=", "<>", "!=", "!<", "!>", "<<", ">>", "??"}:
            tokens.append(operator)
            i += 2
        else:
            tokens.append(expr[i])
            i += 1
    return tokens


def _classify_sqlserver_token(token: str, role: str = "") -> str:
    if role == "raw":
        return token
    if role in {"function", "syntax"}:
        # 内置函数/关键字统一大小写；定界及限定名称保留原拼写。
        return token if any(c in token for c in '.[\"') else token.upper()
    literal_type = detect_literal_type(token, db_type="sqlserver")
    if literal_type in {"null", "bool"}:
        return token.upper()
    if literal_type is not None:
        if literal_type == "string" and token[:2].lower() == "n'":
            token = "N" + token[1:]
        return f"{literal_type}({token})"
    if token.upper() in _SQLSERVER_KEYWORDS or token.startswith("@@"):
        return token.upper()
    if token.startswith("@"):
        return token  # 参数/局部变量不是列；名称可能区分大小写。
    if _SQLSERVER_IDENTIFIER.fullmatch(token):
        return f"col({token})"
    return token


def _parse_sqlserver_predicate(example: str) -> str:
    tokens = _tokenize_sqlserver(example)
    classified: List[str] = []
    call_positions = set()
    frames = []
    for index, token in enumerate(tokens):
        previous = tokens[index - 1] if index else ""
        upper = token.upper()
        frame = frames[-1] if frames else None
        raw = bool(frame and frame["raw"])
        type_argument = bool(
            frame and (
                frame["type"]
                or (frame["name"] in _SQLSERVER_CONVERT_FUNCTIONS and frame["argument"] == 0)
            )
        )
        if token == "(":
            name = previous.upper() if index - 1 in call_positions else ""
            frames.append({
                "name": name, "argument": 0, "type": False,
                "raw": raw or type_argument or previous.lower() in _SQLSERVER_DSL_WRAPPERS,
            })
            classified.append(token)
            continue
        if token == ")":
            if frames:
                frames.pop()
            classified.append(token)
            continue
        if token == ",":
            if frame:
                frame["argument"] += 1
            classified.append(token)
            continue
        if raw:
            classified.append(token)
            continue
        if frame and frame["name"] in _SQLSERVER_CAST_FUNCTIONS and upper == "AS":
            frame["type"] = True
            classified.append("AS")
            continue
        next_is_open = index + 1 < len(tokens) and tokens[index + 1] == "("
        is_call = (
            next_is_open and _SQLSERVER_IDENTIFIER.fullmatch(token)
            and upper not in _SQLSERVER_KEYWORDS
        )
        if type_argument:
            classified.append(_classify_sqlserver_token(token, role="syntax"))
        elif (
            frame and frame["name"] in _SQLSERVER_DATEPART_FUNCTIONS
            and frame["argument"] == 0
        ) or previous.upper() == "COLLATE":
            classified.append(_classify_sqlserver_token(token, role="syntax"))
        elif is_call:
            call_positions.add(index)
            classified.append(
                token.lower() if token.lower() in _SQLSERVER_DSL_WRAPPERS
                else _classify_sqlserver_token(token, role="function")
            )
        else:
            classified.append(_classify_sqlserver_token(token))

    # 函数调用/DSL 的左括号紧跟名称，其余保持旧模板的空格与逗号风格。
    pieces: List[str] = []
    for index, token in enumerate(classified):
        attach = (
            token in {")", ","}
            or (index > 0 and classified[index - 1] in {"(", ","})
            or (token == "(" and index - 1 in call_positions)
        )
        if pieces and not attach:
            pieces.append(" ")
        pieces.append(token)
    return "".join(pieces)


def _normalize_space(s: str) -> str: #将字符串中的多余空白压缩为单个空格
    # s.strip() 去掉首尾空白
    # s.strip().split() 按任意空白分割成单词列表
    # " ".join(...) 再用单个空格把它们拼回去
    return " ".join(s.strip().split())


def _tokenize(expr: str) -> List[str]: #把表达式拆成 token 列表
    if _is_sqlserver():
        return _tokenize_sqlserver(expr)

    tokens: List[str] = []  # 存放最终拆分好的 token
    buf: List[str] = []     # 用来积累当前正在构建的 token 的字符
    i = 0                   # 当前扫描位置
    n = len(expr)           # 表达式长度

    # 主循环：逐字符扫描整个表达式
    while i < n:
        ch = expr[i]  # 当前字符

        #处理空白字符：结束当前 token
        if ch.isspace():
            if buf:                       # 如果 buffer 里有积累的内容
                tokens.append("".join(buf))  # 把它们组合成一个 token
                buf = []                 # 然后清空 buffer
            i += 1                       # 跳过这个空白
            continue                     # 处理下一个字符

        # 处理字符串字面量：以单引号或双引号开头
        if ch == "'" or ch == '"':
            if buf:
                # 在遇到引号前，如果 buffer 中已经有内容，先作为一个 token 输出
                tokens.append("".join(buf))
                buf = []
            quote = ch                   # 记录当前引号类型（' 或 "）
            j = i + 1                    # 从下一个字符开始查找匹配的结束引号
            literal_chars = [quote]      # 字面量字符列表，先放入开头引号

            # 向后扫描，直到找到同类引号或字符串结束
            while j < n:
                literal_chars.append(expr[j])  # 把当前字符加入字面量
                if expr[j] == quote:          # 如果遇到同类引号，认为是结束
                    j += 1                    # 将这个引号也包含进去
                    break                     # 结束字面量扫描
                j += 1                        # 否则继续向后扫描

            # 把字面量字符列表拼成一个完整 token，例如 "'abc'" 或 "\"2024-01-01\""
            literal = "".join(literal_chars)
            tokens.append(literal)            # 加入 token 列表
            i = j                             # 更新扫描位置到字面量结束处
            continue                          # 继续处理下一个字符

        # 括号和逗号：作为单独 token
        if ch in "(),":
            if buf:
                # 在遇到括号或逗号之前，如果 buffer 中有内容，先输出一个 token
                tokens.append("".join(buf))
                buf = []
            tokens.append(ch)  # 括号或逗号本身作为独立 token
            i += 1             # 前进一位
            continue           # 处理下一个字符

        # 一元运算符 +、-、~：作为单独 token
        if ch in "+-~":
            if buf:
                # 在遇到一元运算符之前，如果 buffer 中有内容，也先输出一个 token
                tokens.append("".join(buf))
                buf = []
            tokens.append(ch)  # 运算符本身作为独立 token
            i += 1             # 前进一位
            continue           # 处理下一个字符

        # 其他字符：统一加入 buf 进行积累（包括字母、数字、运算符符号等）
        buf.append(ch)
        i += 1

    # 循环结束后，如果 buf 中还有内容，把它作为最后一个 token 输出
    if buf:
        tokens.append("".join(buf))

    # 返回拆分好的 token 列表
    return tokens


def _looks_like_identifier(token: str) -> bool: #非严格规则：启发式判断,判断一个 token 是否“看起来像列名/标识符”。

    t = token.strip()  # 去掉首尾空白

    # 空字符串直接返回 False
    if not t:
        return False

    # 如果 token 内部还有空白，比如 "NOT LIKE"，视为运算符短语，不当成列名
    if any(ch.isspace() for ch in t):
        return False

    # 含有引号、括号、逗号的，一般就不是简单的列名
    if any(ch in "()'," for ch in t):
        return False

    import re
    # 使用正则匹配一个类似标识符的模式：字母/下划线开头，后面跟字母/数字/下划线/点
    # 点号用来支持 "table.column" 这种形式
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_\.]*", t):
        return False

    # 如果是全大写并且不含点，很可能是关键字（AND/OR/CASE/BETWEEN 等），
    # 为了降低误判概率，这里也先排除掉（根据实际情况你也可以调整）
    if t.isupper() and "." not in t:
        return False

    # 其它情况就认为“看起来像”列名/标识符
    return True


def _classify_token(token: str) -> str: #把原始 token 转换为“模板 token”
    if _is_sqlserver():
        return _classify_sqlserver_token(token)


    # 括号和逗号直接保持原样返回
    if token in ("(", ")", ","):
        return token

    # 尝试识别字面量类型（数字/字符串/日期等）
    lit_type = detect_literal_type(token)  # 例如返回 "int" / "float" / "string" / "date" 等
    if lit_type is not None:
        # 根据新需求：保留原始字面量内容
        # 例如 token="1000"，lit_type="int" -> "int(1000)"
        return f"{lit_type}({token})"

    # 判断是否看起来像标识符/列名
    if _looks_like_identifier(token):
        # 关键修改：保留原始列名内容
        # 例如 token="salary" -> "col(salary)"
        return f"col({token})"

    # 其它情况（各种运算符/关键字/自创运算符）保持原样
    return token


def _compress_parenthesis_types(tokens: List[str]) -> List[str]:
    # 直接返回原始 tokens，不做任何修改
    return tokens


def parse_predicate_example(example: str) -> str: #将一条 example 字符串转换为抽象谓词模板（同时保留具体值信息）
    if _is_sqlserver():
        return _parse_sqlserver_predicate(example)

    # 规范空白：把多余空白压成一个空格
    expr = _normalize_space(example)

    # 拆成 token 列表
    raw_tokens = _tokenize(expr)

    # 对每个 token 进行分类替换：
    # - 列名 -> column(列名)
    # - 字面量 -> 类型(原始字面量)
    # - 其它 -> 原样
    classified_tokens = [_classify_token(t) for t in raw_tokens]

    # 对括号中的内容做进一步处理,不再压缩，原样返回
    compressed_tokens = _compress_parenthesis_types(classified_tokens)

    # 把 token 列表重新组合成字符串，同时处理好标点前后的空格
    pieces: List[str] = []  # 存放最终拼接的字符串片段

    for idx, tok in enumerate(compressed_tokens):
        # 情况 1：右括号或逗号本身前面不额外加空格
        if tok in (")", ","):
            pieces.append(tok)
            continue

        # 情况 2：如果前一个 token 是左括号 "(" 或逗号 ","，
        #         则当前 token 直接接在后面，不加空格
        if pieces and pieces[-1] in ("(", ","):
            pieces.append(tok)
        else:
            # 情况 3：其他情况下，在前面加一个空格再拼接
            if pieces:
                pieces.append(" ")
            pieces.append(tok)

    # 把片段列表合并成最终字符串
    return "".join(pieces)
