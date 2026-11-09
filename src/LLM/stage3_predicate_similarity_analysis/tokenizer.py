from __future__ import annotations

import hashlib
import re
from typing import Iterable, List, Sequence

_TOKEN_RE = re.compile(
    r"""
    '(?:''|\\.|[^'])*'                         # single-quoted string
    |"(?:""|\\.|[^"])*"                    # quoted identifier/string
    |(?:!~~\*|~~\*|!~\*|~\*|!~~|~~|!~|::|->>|\#>>|->|\#>|<=|>=|<>|!=|\|\||&&|<<|>>)
    |(?:[A-Za-z_][A-Za-z0-9_$]*)               # word/identifier
    |(?:\d+(?:\.\d*)?(?:[eE][+-]?\d+)?|\.\d+(?:[eE][+-]?\d+)?)
    |(?:[^\s])                                  # punctuation/operator fallback
    """,
    re.VERBOSE,
)

_SQLSERVER_TOKEN_RE = re.compile(
    r"""
    [nN]?'(?:''|[^'])*'                       # T-SQL: backslash is literal
    |"(?:""|[^"])*"                         # quoted identifier/string
    |\[(?:\]\]|[^\]])*\]                    # bracket identifier; ]] escapes ]
    |0[xX][0-9A-Fa-f]*(?![A-Za-z0-9_$])       # binary literal, including empty 0x
    |(?:!<|!>|<=|>=|<>|!=|<<|>>)
    |(?:[A-Za-z_][A-Za-z0-9_$]*)
    |(?:\d+(?:\.\d*)?(?:[eE][+-]?\d+)?|\.\d+(?:[eE][+-]?\d+)?)
    |(?:[^\s])
    """,
    re.VERBOSE,
)

_SHINGLE_PERSON = b"s3pred-v1"


def tokenize_predicate(
    predicate: str,
    *,
    normalize_unquoted_tokens: bool = True,
) -> List[str]:
    is_sqlserver = False  
    token_re = _SQLSERVER_TOKEN_RE if is_sqlserver else _TOKEN_RE
    tokens = token_re.findall(str(predicate or ""))
    if not normalize_unquoted_tokens:
        return tokens

    normalized = []
    for token in tokens:
        if is_sqlserver and token.startswith(("N'", "n'")):
            normalized.append(token)
        elif token and (token[0].isalpha() or token[0] == "_"):
            normalized.append(token.upper())
        else:
            normalized.append(token)
    return normalized


def _hash_token_sequence(tokens: Sequence[str]) -> int:
    digest = hashlib.blake2b(digest_size=8, person=_SHINGLE_PERSON)
    for token in tokens:
        encoded = token.encode("utf-8")
        digest.update(len(encoded).to_bytes(4, "little"))
        digest.update(encoded)
    return int.from_bytes(digest.digest(), "little", signed=False)


def build_shingle_hashes(tokens: Sequence[str], size: int = 3) -> List[int]:
    size = int(size)
    if size <= 0:
        raise ValueError("shingle size must be greater than zero")
    if not tokens:
        return []

    if len(tokens) < size:
        return [_hash_token_sequence(tokens)]

    hashes = {
        _hash_token_sequence(tokens[index : index + size])
        for index in range(len(tokens) - size + 1)
    }
    return sorted(hashes)


def predicate_digest(predicate: str) -> bytes:
    return hashlib.blake2b(
        str(predicate).encode("utf-8"),
        digest_size=16,
        person=b"s3-line-v1",
    ).digest()
