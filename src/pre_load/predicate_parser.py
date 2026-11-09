from typing import List
import re
from .sql_type_utils import detect_literal_type, _is_sqlserver


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
        return token  
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


def _normalize_space(s: str) -> str: 
    return " ".join(s.strip().split())


def _tokenize(expr: str) -> List[str]: 
    if _is_sqlserver():
        return _tokenize_sqlserver(expr)

    tokens: List[str] = []  
    buf: List[str] = []     
    i = 0                   
    n = len(expr)           

    while i < n:
        ch = expr[i]  

        if ch.isspace():
            if buf:                       
                tokens.append("".join(buf))  
                buf = []                 
            i += 1                       
            continue                     

        if ch == "'" or ch == '"':
            if buf:
                tokens.append("".join(buf))
                buf = []
            quote = ch                   
            j = i + 1                    
            literal_chars = [quote]      

            while j < n:
                literal_chars.append(expr[j])  
                if expr[j] == quote:          
                    j += 1                    
                    break                     
                j += 1                        

            literal = "".join(literal_chars)
            tokens.append(literal)            
            i = j                             
            continue                          

        if ch in "(),":
            if buf:
                tokens.append("".join(buf))
                buf = []
            tokens.append(ch)  
            i += 1             
            continue           

        if ch in "+-~":
            if buf:
                tokens.append("".join(buf))
                buf = []
            tokens.append(ch)  
            i += 1             
            continue           

        buf.append(ch)
        i += 1

    if buf:
        tokens.append("".join(buf))

    return tokens


def _looks_like_identifier(token: str) -> bool: 
    t = token.strip()  

    if not t:
        return False

    if any(ch.isspace() for ch in t):
        return False

    if any(ch in "()'," for ch in t):
        return False

    import re

    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_\.]*", t):
        return False

    if t.isupper() and "." not in t:
        return False

    return True


def _classify_token(token: str) -> str: 
    if _is_sqlserver():
        return _classify_sqlserver_token(token)

    if token in ("(", ")", ","):
        return token

    lit_type = detect_literal_type(token)  
    if lit_type is not None:
        return f"{lit_type}({token})"

    if _looks_like_identifier(token):
        return f"col({token})"

    return token


def _compress_parenthesis_types(tokens: List[str]) -> List[str]:
    return tokens


def parse_predicate_example(example: str) -> str: 
    if _is_sqlserver():
        return _parse_sqlserver_predicate(example)

    expr = _normalize_space(example)

    raw_tokens = _tokenize(expr)

    classified_tokens = [_classify_token(t) for t in raw_tokens]

    compressed_tokens = _compress_parenthesis_types(classified_tokens)

    pieces: List[str] = []  

    for idx, tok in enumerate(compressed_tokens):
        if tok in (")", ","):
            pieces.append(tok)
            continue

        if pieces and pieces[-1] in ("(", ","):
            pieces.append(tok)
        else:
            if pieces:
                pieces.append(" ")
            pieces.append(tok)

    return "".join(pieces)
