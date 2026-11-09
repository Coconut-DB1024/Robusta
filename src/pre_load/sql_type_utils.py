import re
from typing import Optional


def _is_sqlserver() -> bool:
    from RL.config import DB_CONFIG

    return str(getattr(DB_CONFIG, "db_type", "") or "").strip().lower() == "sqlserver"


def detect_literal_type(token: str, db_type: Optional[str] = None) -> Optional[str]: 
    t = token.strip()  

    if not t:
        return None

    sqlserver = _is_sqlserver() if db_type is None else str(db_type).strip().lower() == "sqlserver"
    if sqlserver:
        if t.upper() == "NULL":
            return "null"
        if t.upper() in {"TRUE", "FALSE"}:
            return "bool"  
        if re.fullmatch(r"[Nn]?'(?:[^']|'')*'", t):
            return "string"
        if re.fullmatch(r"0[xX][0-9a-fA-F]*", t):
            return "binary"  
        if re.fullmatch(r"[+-]?[0-9]+", t):
            return "int"
        if re.fullmatch(r"[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?", t):
            return "float"

        return None

    if re.fullmatch(r"NULL", t, flags=re.IGNORECASE):
        return "null"

    if re.fullmatch(r"TRUE|FALSE", t, flags=re.IGNORECASE):
        return "bool"

    if (t.startswith("'") and t.endswith("'")) or (t.startswith('"') and t.endswith('"')):
        inner = t[1:-1]  

        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", inner):
            return "date"
        if re.fullmatch(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}", inner):
            return "datetime"

        return "string"

    if re.fullmatch(r"-?\d+", t):
        return "int"

    if re.fullmatch(r"-?0x[0-9a-f]+", t, flags=re.IGNORECASE):
        return "hex"

    if re.fullmatch(r"-?\d+\.\d*", t):
        return "float"

    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", t):
        return "date"

    if re.fullmatch(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}", t):
        return "datetime"

    return None
