#!/usr/bin/env python3
"""Independent PostgreSQL and local/online-LLM semantic edge discovery experiment."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import math
import os
import re
import sys
import time
from collections import Counter, defaultdict
from contextlib import redirect_stdout
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional


ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "experiment_config.json"
SHARED_CONFIG_PATH = ROOT.parents[1] / "configuration.json"
RESULTS_ROOT = ROOT / "results"
SATISFIABLE_PREDICATES_FILENAME = "satisfiable.txt"

MAX_PREDICATE_CHARS = 10_000
DEFAULT_MAX_PREDICATES_PER_TARGET = 5

WORD_RE = re.compile(r"[A-Za-z_][A-Za-z_0-9]*|[0-9]+")

MOCK_COLUMNS = [
    {
        "column_name": "col_integer_0",
        "data_type": "integer",
        "udt_name": "int4",
        "is_nullable": "YES",
        "ordinal_position": 1,
    },
    {
        "column_name": "col_float_0",
        "data_type": "double precision",
        "udt_name": "float8",
        "is_nullable": "YES",
        "ordinal_position": 2,
    },
    {
        "column_name": "col_string_0",
        "data_type": "text",
        "udt_name": "text",
        "is_nullable": "YES",
        "ordinal_position": 3,
    },
]

FIXTURE_DOCUMENT = (
    "PostgreSQL expressions may be used in a WHERE clause. "
    "Boolean expressions include AND, OR, and NOT. "
    "Function calls use a function name followed by a parenthesized argument list. "
    "Pattern matching operators include LIKE, ILIKE, SIMILAR TO, and regular expression operators. "
    "Arithmetic and bitwise operators accept left and right expressions."
)



def require_internal_path(path: Path) -> Path:
    resolved = path.resolve()
    try:
        resolved.relative_to(ROOT)
    except ValueError as exc:
        raise ValueError(f"Path must be inside {ROOT}: {path}") from exc
    return resolved


def resolve_output_path(path: Path) -> Path:
    if not path.is_absolute():
        if path.parts and path.parts[0] == "results":
            path = ROOT / path
        else:
            path = RESULTS_ROOT / path

    resolved = require_internal_path(path)

    try:
        resolved.relative_to(RESULTS_ROOT)
    except ValueError as exc:
        raise ValueError(
            f"Output path must be inside {RESULTS_ROOT}: {path}"
        ) from exc

    return resolved


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)

    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")

    return value


def load_experiment_config(config_path: Path) -> dict[str, Any]:
    shared = json.loads(SHARED_CONFIG_PATH.read_text(encoding="utf-8-sig"))
    if not isinstance(shared, dict):
        raise ValueError(f"Expected a JSON object in {SHARED_CONFIG_PATH}")
    if shared.get("db_type") != "postgresql":
        raise ValueError("当前仅支持postgresql数据库")

    config = read_json(config_path)
    online_model = shared.get("online_model")
    if not isinstance(online_model, dict):
        raise ValueError(f"{SHARED_CONFIG_PATH}: online_model must be a JSON object")
    config["online_model"] = dict(online_model)

    database = dict(config.get("database", {}))
    for target, source in (
        ("host", "host"), ("port", "port"), ("dbname", "database"),
        ("user", "user"), ("password", "password"), ("table", "table_name"),
        ("connect_timeout", "connect_timeout"),
    ):
        database[target] = shared[source]
    if shared.get("unix_socket"):
        database["host"] = shared["unix_socket"]
    for field in ("port", "connect_timeout", "query_timeout"):
        if type(shared[field]) is not int:
            raise ValueError(f"{SHARED_CONFIG_PATH}: {field} must be a JSON integer")
    database["statement_timeout_ms"] = shared["query_timeout"] * 1000
    config["database"] = database
    return config


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)

    return digest.hexdigest()


def atomic_json_write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    temporary = path.with_name(path.name + ".tmp")

    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(
            value,
            handle,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())

    os.replace(temporary, path)


def append_jsonl(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(
            json.dumps(
                value,
                ensure_ascii=False,
                sort_keys=True,
            )
            + "\n"
        )
        handle.flush()
        os.fsync(handle.fileno())


def read_jsonl(
    path: Path,
    logger: logging.Logger,
) -> list[dict[str, Any]]:
    if not path.is_file():
        return []

    records: list[dict[str, Any]] = []

    for number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue

        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            logger.warning(
                "Ignoring malformed JSONL record at %s:%s",
                path.name,
                number,
            )
            continue

        if isinstance(record, dict):
            records.append(record)

    return records



def rewrite_generated_predicates_txt(
    path: Path,
    interactions: Iterable[dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")

    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for record in interactions:
            predicates = record.get("parsed_predicates")
            if not isinstance(predicates, list):
                continue

            for predicate in predicates:
                if not isinstance(predicate, str):
                    continue

                one_line = predicate.replace("\r", " ").replace("\n", " ")
                handle.write(one_line + "\n")

        handle.flush()
        os.fsync(handle.fileno())

    os.replace(temporary, path)


def append_generated_predicates_txt(
    path: Path,
    predicates: Iterable[str],
) -> None:
    values = [
        predicate.replace("\r", " ").replace("\n", " ")
        for predicate in predicates
        if isinstance(predicate, str)
    ]

    if not values:
        return

    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("a", encoding="utf-8", newline="\n") as handle:
        for predicate in values:
            handle.write(predicate + "\n")

        handle.flush()
        os.fsync(handle.fileno())


def rewrite_satisfiable_predicates_txt(
    path: Path,
    validations: Iterable[dict[str, Any]],
) -> None:
    rewrite_generated_predicates_txt(
        path,
        (
            {"parsed_predicates": [record.get("predicate")]}
            for record in validations
            if record.get("db_satisfiable") is True
        ),
    )


def load_tokens(path: Path) -> list[str]:
    tokens = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    if not tokens:
        raise ValueError("tokens.txt has no non-empty tokens")

    if len(tokens) != len(set(tokens)):
        raise ValueError("tokens.txt contains duplicate tokens")

    return tokens


def load_slots(
    path: Path,
    tokens: list[str],
) -> dict[str, list[str]]:
    raw = read_json(path)

    token_set = set(tokens)

    if set(raw) != token_set:
        missing = sorted(token_set - set(raw))
        extra = sorted(set(raw) - token_set)

        raise ValueError(
            "slots.json keys must exactly match tokens.txt; "
            f"missing={missing}, extra={extra}"
        )

    slots: dict[str, list[str]] = {}

    for token in tokens:
        values = raw[token]

        if not isinstance(values, list):
            raise ValueError(f"Invalid slots for {token!r}")

        if not all(
            isinstance(item, str) and item
            for item in values
        ):
            raise ValueError(f"Invalid slots for {token!r}")

        if len(values) != len(set(values)):
            raise ValueError(f"Duplicate slots for {token!r}")

        slots[token] = values

    return slots


def load_examples(path: Path) -> list[str]:
    examples = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    if not examples:
        raise ValueError("examples.txt has no non-empty predicates")

    return examples



@dataclass(frozen=True)
class Target:
    source_operator: str
    source_slot: str
    target_operator: str

    @property
    def edge(self) -> str:
        return (
            f"{self.source_operator}."
            f"{self.source_slot} -> "
            f"{self.target_operator}"
        )

    @property
    def interaction_id(self) -> str:
        key = (
            f"{self.source_operator}\x1f"
            f"{self.source_slot}\x1f"
            f"{self.target_operator}"
        )

        return hashlib.sha256(
            key.encode("utf-8")
        ).hexdigest()[:24]


def enumerate_targets(
    tokens: list[str],
    slots: dict[str, list[str]],
) -> Iterable[Target]:
    for source_operator in tokens:
        for source_slot in slots[source_operator]:
            for target_operator in tokens:
                yield Target(
                    source_operator,
                    source_slot,
                    target_operator,
                )



def strict_predicates_json(
    raw_output: str,
    max_predicates: int,
) -> tuple[
    bool,
    list[str],
    bool,
    Optional[str],
]:

    try:
        decoder = json.JSONDecoder()
        parsed, end = decoder.raw_decode(raw_output)

        if raw_output[end:].strip():
            return (
                False,
                [],
                False,
                "trailing non-JSON content",
            )

    except json.JSONDecodeError as exc:
        return (
            False,
            [],
            False,
            f"invalid JSON: {exc.msg}",
        )

    if (
        not isinstance(parsed, dict)
        or set(parsed) != {"predicates"}
    ):
        return (
            False,
            [],
            False,
            'expected exactly {"predicates":[...]}',
        )

    values = parsed["predicates"]

    if not isinstance(values, list):
        return (
            False,
            [],
            False,
            "predicates must be a JSON array",
        )

    if len(values) > max_predicates:
        return (
            False,
            [],
            False,
            f"too many predicates; maximum is {max_predicates}",
        )

    predicates: list[str] = []

    for index, value in enumerate(values):
        if not isinstance(value, str):
            return (
                False,
                [],
                False,
                f"predicates[{index}] must be a string",
            )

        predicate = value.strip()

        if not predicate:
            return (
                False,
                [],
                False,
                f"predicates[{index}] must not be empty",
            )

        if len(predicate) > MAX_PREDICATE_CHARS:
            return (
                False,
                [],
                False,
                f"predicates[{index}] exceeds "
                f"{MAX_PREDICATE_CHARS} characters",
            )

        predicates.append(predicate)

    return (
        True,
        predicates,
        len(predicates) == 0,
        None,
    )


def predicate_is_safe_for_validation(
    predicate: str,
) -> tuple[bool, Optional[str]]:
    if "\x00" in predicate:
        return False, "NUL byte is not allowed"

    for prohibited in (";", "--", "/*", "*/"):
        if prohibited in predicate:
            return (
                False,
                f"unsafe SQL sequence {prohibited!r}",
            )

    return True, None



@dataclass
class DocumentChunk:
    chunk_id: str
    page: int
    text: str


def word_counts(text: str) -> Counter[str]:
    return Counter(
        match.group(0).lower()
        for match in WORD_RE.finditer(text)
    )


def char_signature(
    text: str,
    bits: int = 4096,
) -> int:
    lowered = text.lower()
    signature = 0

    if len(lowered) < 3:
        return signature

    for index in range(len(lowered) - 2):
        a = ord(lowered[index])
        b = ord(lowered[index + 1])
        c = ord(lowered[index + 2])

        position = (
            a * 65_537
            + b * 257
            + c
        ) % bits

        signature |= 1 << position

    return signature


class LexicalRAG:

    def __init__(
        self,
        chunks: list[DocumentChunk],
        cached: Optional[dict[str, Any]] = None,
    ):
        if not chunks:
            raise ValueError("RAG has no document chunks")

        self.chunks = chunks

        self._lower_texts = [
            chunk.text.lower()
            for chunk in chunks
        ]

        if cached is None:
            self.doc_terms = [
                dict(word_counts(chunk.text))
                for chunk in chunks
            ]

            self.char_signatures = [
                char_signature(chunk.text)
                for chunk in chunks
            ]

        else:
            self.doc_terms = cached["doc_terms"]

            self.char_signatures = [
                int(value, 16)
                for value in cached["char_signatures"]
            ]

        if (
            len(self.doc_terms) != len(chunks)
            or len(self.char_signatures) != len(chunks)
        ):
            raise ValueError("Corrupt RAG cache")

        self.postings: dict[
            str,
            list[tuple[int, int]]
        ] = defaultdict(list)

        document_frequency: Counter[str] = Counter()

        self.doc_lengths: list[int] = []

        for index, terms in enumerate(self.doc_terms):
            self.doc_lengths.append(
                sum(int(value) for value in terms.values())
            )

            for term, frequency in terms.items():
                self.postings[term].append(
                    (index, int(frequency))
                )
                document_frequency[term] += 1

        count = len(chunks)

        self.idf = {
            term: (
                math.log(
                    (count + 1)
                    / (frequency + 1)
                )
                + 1.0
            )
            for term, frequency
            in document_frequency.items()
        }

    def cache_payload(self) -> dict[str, Any]:
        return {
            "chunks": [
                {
                    "chunk_id": item.chunk_id,
                    "page": item.page,
                    "text": item.text,
                }
                for item in self.chunks
            ],
            "doc_terms": self.doc_terms,
            "char_signatures": [
                format(value, "x")
                for value in self.char_signatures
            ],
        }

    @classmethod
    def from_cache_payload(
        cls,
        payload: dict[str, Any],
    ) -> "LexicalRAG":
        chunks = [
            DocumentChunk(**entry)
            for entry in payload["chunks"]
        ]

        return cls(
            chunks,
            cached=payload,
        )

    def retrieve(
        self,
        source: str,
        slot: str,
        target: str,
        top_k: int,
    ) -> list[DocumentChunk]:
        query = f"{source} {slot} {target}"

        query_terms = word_counts(query)
        query_signature = char_signature(query)
        query_bits = query_signature.bit_count()

        scores: dict[int, float] = defaultdict(float)
        candidates: set[int] = set()

        for term, query_frequency in query_terms.items():
            for (
                document_index,
                document_term_frequency,
            ) in self.postings.get(term, []):
                candidates.add(document_index)

                scores[document_index] += (
                    self.idf[term]
                    * min(
                        query_frequency,
                        document_term_frequency,
                    )
                )

        if not candidates:
            candidates = set(
                range(len(self.chunks))
            )

        exact_terms = [
            value.lower()
            for value in (source, target)
            if len(value.strip()) >= 2
        ]

        for document_index in candidates:
            text = self._lower_texts[document_index]

            for value in exact_terms:
                if value in text:
                    scores[document_index] += 3.0

            if query_bits:
                common = (
                    self.char_signatures[document_index]
                    & query_signature
                ).bit_count()

                scores[document_index] += (
                    0.75
                    * common
                    / query_bits
                )

        ordered = sorted(
            candidates,
            key=lambda index: (
                -scores[index],
                self.chunks[index].chunk_id,
            ),
        )

        return [
            self.chunks[index]
            for index in ordered[:top_k]
        ]


def extract_pdf_chunks(
    pdf_path: Path,
    chunk_chars: int,
    overlap_chars: int,
) -> list[DocumentChunk]:
    try:
        from pypdf import PdfReader

    except ImportError as exc:
        raise RuntimeError(
            "pypdf is required for the PostgreSQL PDF."
        ) from exc

    if not pdf_path.is_file():
        raise FileNotFoundError(
            f"PostgreSQL PDF not found: {pdf_path}"
        )

    if (
        chunk_chars <= overlap_chars
        or overlap_chars < 0
    ):
        raise ValueError(
            "chunk_chars must be greater than "
            "overlap_chars >= 0"
        )

    reader = PdfReader(str(pdf_path))

    chunks: list[DocumentChunk] = []

    for page_number, page in enumerate(
        reader.pages,
        start=1,
    ):
        text = (
            page.extract_text()
            or ""
        ).strip()

        if not text:
            continue

        start = 0
        part = 1

        while start < len(text):
            end = min(
                len(text),
                start + chunk_chars,
            )

            chunks.append(
                DocumentChunk(
                    f"p{page_number:04d}_c{part:03d}",
                    page_number,
                    text[start:end],
                )
            )

            if end == len(text):
                break

            start = end - overlap_chars
            part += 1

    return chunks


def load_or_build_rag(
    pdf_path: Path,
    cache_dir: Path,
    chunk_chars: int,
    overlap_chars: int,
) -> tuple[LexicalRAG, float, bool]:
    cache_dir = resolve_output_path(cache_dir)

    if cache_dir != RESULTS_ROOT:
        raise ValueError("RAG cache_directory must be 'results'")

    cache_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    cache_file = (
        cache_dir
        / "rag_index.json"
    )

    stat = pdf_path.stat()

    fingerprint = {
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "chunk_chars": chunk_chars,
        "overlap_chars": overlap_chars,
    }

    if cache_file.is_file():
        try:
            cached = read_json(cache_file)

            if (
                cached.get("fingerprint")
                == fingerprint
            ):
                return (
                    LexicalRAG.from_cache_payload(
                        cached["index"]
                    ),
                    0.0,
                    True,
                )

        except (
            OSError,
            ValueError,
            KeyError,
            TypeError,
            json.JSONDecodeError,
        ):
            pass

    started = time.perf_counter()

    rag = LexicalRAG(
        extract_pdf_chunks(
            pdf_path,
            chunk_chars,
            overlap_chars,
        )
    )

    payload = {
        "fingerprint": fingerprint,
        "index": rag.cache_payload(),
    }

    atomic_json_write(
        cache_file,
        payload,
    )

    return (
        rag,
        time.perf_counter() - started,
        False,
    )


def fixture_rag() -> LexicalRAG:
    return LexicalRAG(
        [
            DocumentChunk(
                "fixture_001",
                0,
                FIXTURE_DOCUMENT,
            )
        ]
    )



def build_prompt(
    target: Target,
    columns: list[dict[str, Any]],
    chunks: list[DocumentChunk],
    examples: list[str],
    max_predicates: int,
) -> str:
    context = "\n\n".join(
        (
            "[Official PostgreSQL documentation: "
            f"{chunk.chunk_id}, page {chunk.page}]\n"
            f"{chunk.text}"
        )
        for chunk in chunks
    )

    few_shot = "\n".join(
        f"- {example}"
        for example in examples
    )

    column_json = json.dumps(
        columns,
        ensure_ascii=False,
        separators=(",", ":"),
    )

    return (
        "You are generating PostgreSQL WHERE predicates for a "
        "semantic dependency experiment.\n\n"

        "The required dependency is:\n"
        "source_operator.source_slot -> target_operator\n\n"

        "The target operator must directly fill the specified "
        "source slot as the immediate syntactic/AST child of the "
        "source operator. Merely placing both operators somewhere "
        "in the same predicate does not satisfy the requirement.\n\n"

        f"Source operator: {target.source_operator}\n"
        f"Source slot: {target.source_slot}\n"
        f"Target operator: {target.target_operator}\n"
        f"Required edge: {target.edge}\n\n"

        "The following are the only real table columns available. "
        "Do not invent additional columns and do not assume their "
        "runtime values.\n"
        f"{column_json}\n\n"

        "Fixed correct PostgreSQL predicate examples are provided "
        "only as context. They are not discovered edges:\n"
        f"{few_shot}\n\n"

        "Automatically retrieved excerpts from the official "
        "PostgreSQL documentation:\n"
        f"{context}\n\n"

        f"If the requested dependency is legal, generate as many valid "
        f"PostgreSQL WHERE predicates as possible, up to {max_predicates}. "
        f"Strongly prefer returning the full {max_predicates} predicates "
        "whenever they can be constructed legally.\n"

        "If you believe no legal PostgreSQL predicate can satisfy "
        "the requested dependency, return an empty list instead of "
        "inventing an invalid predicate.\n\n"

        "Return ONLY strict JSON with exactly one key named "
        "\"predicates\". The array should contain as many legal "
        f"predicates as you can construct, up to {max_predicates}.\n\n"

        "Valid refusal example:\n"
        "{\"predicates\":[]}\n\n"

        "Do not return Markdown, explanations, comments, or any "
        "additional keys."
    )



@dataclass
class BackendResponse:
    raw_output: str

    input_tokens: Optional[int]
    output_tokens: Optional[int]
    total_tokens: Optional[int]

    request_elapsed_seconds: float

    generate_called: bool

    error: Optional[str] = None

    stop_requested: bool = False
    stop_reason: Optional[str] = None

    tokenization_seconds: float = 0.0
    llm_generate_seconds: float = 0.0
    decode_seconds: float = 0.0
    api_call_seconds: float = 0.0


class QwenBackend:

    def __init__(
        self,
        model_config: dict[str, Any],
    ):
        try:
            import torch

            from transformers import (
                AutoModelForCausalLM,
                AutoTokenizer,
            )

        except ImportError as exc:
            raise RuntimeError(
                "torch and transformers are required "
                "for the local backend."
            ) from exc

        self.torch = torch

        model_path = model_config["path"]

        self.tokenizer = (
            AutoTokenizer.from_pretrained(
                model_path,
                local_files_only=True,
                trust_remote_code=True,
            )
        )

        self.model = (
            AutoModelForCausalLM.from_pretrained(
                model_path,
                local_files_only=True,
                trust_remote_code=True,
                device_map="auto",
                low_cpu_mem_usage=True,
            )
        )

        self.model.eval()

        self.max_new_tokens = int(
            model_config.get(
                "max_new_tokens",
                256,
            )
        )

    def generate(
        self,
        prompt: str,
    ) -> BackendResponse:
        request_started = time.perf_counter()

        messages = [
            {
                "role": "system",
                "content": (
                    "You generate only strict JSON "
                    "for PostgreSQL predicates."
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ]

        tokenization_started = (
            time.perf_counter()
        )

        try:
            rendered = (
                self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
            )

            encoded = self.tokenizer(
                rendered,
                return_tensors="pt",
                add_special_tokens=False,
            )

            device = getattr(
                self.model,
                "device",
                None,
            )

            if device is None:
                device = next(
                    self.model.parameters()
                ).device

            encoded = {
                name: value.to(device)
                for name, value
                in encoded.items()
            }

            input_ids = encoded["input_ids"]

            attention_mask = encoded.get(
                "attention_mask"
            )

            input_tokens = (
                int(attention_mask.sum().item())
                if attention_mask is not None
                else int(input_ids.shape[-1])
            )

        except Exception as exc:
            return BackendResponse(
                raw_output="",
                input_tokens=None,
                output_tokens=None,
                total_tokens=None,
                request_elapsed_seconds=(
                    time.perf_counter()
                    - request_started
                ),
                generate_called=False,
                error=(
                    "tokenization_error: "
                    f"{exc}"
                ),
                tokenization_seconds=(
                    time.perf_counter()
                    - tokenization_started
                ),
            )

        tokenization_seconds = (
            time.perf_counter()
            - tokenization_started
        )

        generate_started = (
            time.perf_counter()
        )

        try:
            kwargs: dict[str, Any] = {
                "max_new_tokens":
                    self.max_new_tokens,
                "do_sample": False,
            }

            if (
                self.tokenizer.eos_token_id
                is not None
            ):
                kwargs["pad_token_id"] = (
                    self.tokenizer.eos_token_id
                )

            with self.torch.inference_mode():
                generated = self.model.generate(
                    **encoded,
                    **kwargs,
                )

        except Exception as exc:
            generate_seconds = (
                time.perf_counter()
                - generate_started
            )

            return BackendResponse(
                raw_output="",
                input_tokens=input_tokens,
                output_tokens=None,
                total_tokens=None,
                request_elapsed_seconds=(
                    time.perf_counter()
                    - request_started
                ),
                generate_called=True,
                error=(
                    "generate_error: "
                    f"{exc}"
                ),
                tokenization_seconds=(
                    tokenization_seconds
                ),
                llm_generate_seconds=(
                    generate_seconds
                ),
            )

        generate_seconds = (
            time.perf_counter()
            - generate_started
        )

        decode_started = (
            time.perf_counter()
        )

        try:
            input_length = int(
                input_ids.shape[-1]
            )

            output_tokens = int(
                generated.shape[-1]
                - input_length
            )

            new_tokens = generated[
                0,
                input_length:
            ]

            raw_output = self.tokenizer.decode(
                new_tokens,
                skip_special_tokens=True,
            )

        except Exception as exc:
            decode_seconds = (
                time.perf_counter()
                - decode_started
            )

            return BackendResponse(
                raw_output="",
                input_tokens=input_tokens,
                output_tokens=None,
                total_tokens=None,
                request_elapsed_seconds=(
                    time.perf_counter()
                    - request_started
                ),
                generate_called=True,
                error=(
                    "decode_error: "
                    f"{exc}"
                ),
                tokenization_seconds=(
                    tokenization_seconds
                ),
                llm_generate_seconds=(
                    generate_seconds
                ),
                decode_seconds=(
                    decode_seconds
                ),
            )

        decode_seconds = (
            time.perf_counter()
            - decode_started
        )

        total_tokens = (
            input_tokens
            + output_tokens
        )

        return BackendResponse(
            raw_output=raw_output,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            request_elapsed_seconds=(
                time.perf_counter()
                - request_started
            ),
            generate_called=True,
            tokenization_seconds=(
                tokenization_seconds
            ),
            llm_generate_seconds=(
                generate_seconds
            ),
            decode_seconds=(
                decode_seconds
            ),
        )


def terminal_online_error(
    exc: Exception,
) -> bool:
    status_code = getattr(
        exc,
        "status_code",
        None,
    )

    if status_code in {
        401,
        403,
        429,
    }:
        return True

    error_code = str(
        getattr(exc, "code", "")
        or ""
    ).lower()

    message = str(exc).lower()

    terminal_markers = (
        "quota",
        "insufficient_quota",
        "arrearage",
        "balance",
        "invalid api key",
        "authentication",
        "permission denied",
    )

    return any(
        marker in error_code
        or marker in message
        for marker in terminal_markers
    )


class OnlineQwenBackend:

    def __init__(
        self,
        model_config: dict[str, Any],
        client: Any = None,
    ):
        self.model_name = str(
            model_config["name"]
        )

        self.max_tokens = int(
            model_config.get(
                "max_tokens",
                256,
            )
        )

        self.temperature = float(
            model_config.get(
                "temperature",
                0,
            )
        )

        self.stream = model_config.get("stream", False)

        if not isinstance(self.stream, bool):
            raise ValueError(
                "online_model.stream must be a JSON boolean"
            )

        self.extra_body: dict[str, Any] = {}

        if "enable_thinking" in model_config:
            enable_thinking = model_config["enable_thinking"]

            if not isinstance(enable_thinking, bool):
                raise ValueError(
                    "online_model.enable_thinking "
                    "must be a JSON boolean"
                )

            self.extra_body["enable_thinking"] = enable_thinking

        if client is not None:
            self.client = client
            return

        api_key = str(
            model_config.get(
                "api_key",
                "",
            )
            or ""
        ).strip()

        environment_name = str(
            model_config.get(
                "api_key_env",
                "DASHSCOPE_API_KEY",
            )
        )

        api_key = (
            api_key
            or os.environ.get(
                environment_name,
                "",
            ).strip()
        )

        if not api_key:
            raise ValueError(
                "Online API key is empty; "
                f"set {environment_name} "
                "or online_model.api_key"
            )

        try:
            from openai import OpenAI

        except ImportError as exc:
            raise RuntimeError(
                "openai is required for "
                "the online backend."
            ) from exc

        self.client = OpenAI(
            api_key=api_key,
            base_url=str(
                model_config["base_url"]
            ),
            timeout=float(
                model_config.get(
                    "timeout_seconds",
                    180,
                )
            ),
            max_retries=0,
        )

    def generate(
        self,
        prompt: str,
    ) -> BackendResponse:
        request_started = (
            time.perf_counter()
        )

        output_parts: list[str] = []
        usage = None
        has_choices = False
        response_error = None
        stop_requested = False

        request_options: dict[str, Any] = {}

        if self.stream:
            request_options = {
                "stream": True,
                "stream_options": {"include_usage": True},
            }

        try:
            response = (
                self.client
                .chat
                .completions
                .create(
                    model=self.model_name,
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You generate only "
                                "strict JSON for "
                                "PostgreSQL predicates."
                            ),
                        },
                        {
                            "role": "user",
                            "content": prompt,
                        },
                    ],
                    temperature=(
                        self.temperature
                    ),
                    max_tokens=(
                        self.max_tokens
                    ),
                    extra_body=self.extra_body,
                    **request_options,
                )
            )

            if self.stream:
                try:
                    for chunk in response:
                        chunk_usage = getattr(chunk, "usage", None)

                        if chunk_usage is not None:
                            usage = chunk_usage

                        choices = getattr(chunk, "choices", None) or []

                        if choices:
                            has_choices = True

                        for choice in choices:
                            if getattr(choice, "index", 0) != 0:
                                continue

                            delta = getattr(choice, "delta", None)
                            content = getattr(delta, "content", None)

                            if isinstance(content, str):
                                output_parts.append(content)
                finally:
                    close = getattr(response, "close", None)

                    if callable(close):
                        close()

            else:
                choices = getattr(response, "choices", None) or []
                has_choices = bool(choices)

                if choices:
                    message = getattr(choices[0], "message", None)
                    content = getattr(message, "content", None)

                    if isinstance(content, str):
                        output_parts.append(content)

                usage = getattr(response, "usage", None)

        except Exception as exc:
            stop_requested = (
                terminal_online_error(exc)
            )

            response_error = (
                "online_api_error: "
                f"{type(exc).__name__}: "
                f"{exc}"
            )

        elapsed = (
            time.perf_counter()
            - request_started
        )

        raw_output = "".join(output_parts)

        input_tokens = (
            getattr(
                usage,
                "prompt_tokens",
                None,
            )
            if usage is not None
            else None
        )

        output_tokens = (
            getattr(
                usage,
                "completion_tokens",
                None,
            )
            if usage is not None
            else None
        )

        total_tokens = (
            getattr(
                usage,
                "total_tokens",
                None,
            )
            if usage is not None
            else None
        )

        input_tokens = (
            int(input_tokens)
            if isinstance(
                input_tokens,
                int,
            )
            else None
        )

        output_tokens = (
            int(output_tokens)
            if isinstance(
                output_tokens,
                int,
            )
            else None
        )

        total_tokens = (
            int(total_tokens)
            if isinstance(
                total_tokens,
                int,
            )
            else None
        )

        if (
            total_tokens is None
            and input_tokens is not None
            and output_tokens is not None
        ):
            total_tokens = (
                input_tokens
                + output_tokens
            )

        if response_error is None and not has_choices:
            response_error = (
                "online_api_error: "
                "response has no choices"
            )

        return BackendResponse(
            raw_output=raw_output,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            request_elapsed_seconds=elapsed,
            generate_called=True,
            error=response_error,
            stop_requested=stop_requested,
            stop_reason=(
                response_error
                if stop_requested
                else None
            ),
            api_call_seconds=elapsed,
        )


class MockBackend:

    def __init__(self) -> None:
        self.calls = 0

    @staticmethod
    def _tokens(
        value: str,
    ) -> list[str]:
        return re.findall(
            (
                r"[A-Za-z_]"
                r"[A-Za-z_0-9]*"
                r"|[0-9]+"
                r"|[^\sA-Za-z_0-9]"
            ),
            value,
        )

    def generate(
        self,
        prompt: str,
    ) -> BackendResponse:
        self.calls += 1

        started = time.perf_counter()

        rendered = (
            "<system>"
            "You generate only strict JSON."
            "</system>"
            "<user>"
            + prompt
            + "</user>"
            "<assistant>"
        )

        raw_output = (
            '{"predicates":'
            '["TRUE","FALSE"]}'
        )

        input_tokens = len(
            self._tokens(rendered)
        )

        output_tokens = len(
            self._tokens(raw_output)
        )

        return BackendResponse(
            raw_output=raw_output,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=(
                input_tokens
                + output_tokens
            ),
            request_elapsed_seconds=(
                time.perf_counter()
                - started
            ),
            generate_called=True,
        )



def load_ast_map(
    path: Path,
) -> dict[str, Any]:
    mapping = read_json(path)

    if not isinstance(
        mapping.get("explicit_tokens"),
        dict,
    ):
        raise ValueError(
            "token_ast_map.json must contain "
            "explicit_tokens"
        )

    if not isinstance(
        mapping.get("function_rule"),
        dict,
    ):
        raise ValueError(
            "token_ast_map.json must contain "
            "function_rule"
        )

    return mapping


class StructureChecker:

    def __init__(
        self,
        mapping: dict[str, Any],
        columns: list[dict[str, Any]],
    ):
        self.mapping = mapping
        self.columns = columns

        self.integer_columns: set[str] = set()
        self.float_columns: set[str] = set()
        self.string_columns: set[str] = set()

        for column in columns:
            name = str(
                column.get(
                    "column_name",
                    "",
                )
            )

            data_type = str(
                column.get(
                    "data_type",
                    "",
                )
            ).lower()

            udt_name = str(
                column.get(
                    "udt_name",
                    "",
                )
            ).lower()

            if (
                data_type
                in {
                    "smallint",
                    "integer",
                    "bigint",
                }
                or udt_name
                in {
                    "int2",
                    "int4",
                    "int8",
                }
            ):
                self.integer_columns.add(
                    name
                )

            elif (
                data_type
                in {
                    "real",
                    "double precision",
                    "numeric",
                    "decimal",
                }
                or udt_name
                in {
                    "float4",
                    "float8",
                    "numeric",
                }
            ):
                self.float_columns.add(
                    name
                )

            elif (
                "char" in data_type
                or "text" in data_type
                or udt_name
                in {
                    "text",
                    "varchar",
                    "bpchar",
                }
            ):
                self.string_columns.add(
                    name
                )

        try:
            from pglast.parser import (
                parse_sql_json,
            )

        except ImportError:
            self.parse_sql_json = None

        else:
            self.parse_sql_json = (
                parse_sql_json
            )

    def resolve_mapping(
        self,
        token: str,
    ) -> tuple[
        Optional[dict[str, Any]],
        Optional[str],
    ]:
        explicit = (
            self.mapping[
                "explicit_tokens"
            ].get(token)
        )

        if explicit is not None:
            if (
                explicit.get("mode")
                == "unresolved"
            ):
                return (
                    None,
                    str(
                        explicit.get(
                            "reason",
                            "mapping marked unresolved",
                        )
                    ),
                )

            return explicit, None

        function_rule = (
            self.mapping[
                "function_rule"
            ]
        )

        if (
            function_rule.get("enabled")
            and token.endswith("()")
        ):
            return (
                {
                    "ast_node":
                        function_rule[
                            "ast_node"
                        ],
                    "function_name":
                        token[:-2].lower(),
                    "slot_field":
                        function_rule[
                            "slot_field"
                        ],
                },
                None,
            )

        return (
            None,
            (
                "No verified AST mapping "
                f"for token {token!r}"
            ),
        )

    @staticmethod
    def _node_type(
        node: Any,
    ) -> tuple[
        Optional[str],
        Optional[dict[str, Any]],
    ]:
        if (
            isinstance(node, dict)
            and len(node) == 1
        ):
            name, value = next(
                iter(node.items())
            )

            if isinstance(
                value,
                dict,
            ):
                return name, value

        return None, None

    @staticmethod
    def _string_value(
        value: Any,
    ) -> Optional[str]:
        if isinstance(value, str):
            return value

        if isinstance(value, dict):
            for key in (
                "str",
                "sval",
                "val",
            ):
                if isinstance(
                    value.get(key),
                    str,
                ):
                    return value[key]

            for child in value.values():
                result = (
                    StructureChecker
                    ._string_value(child)
                )

                if result is not None:
                    return result

        return None

    @classmethod
    def _name_list(
        cls,
        value: Any,
    ) -> list[str]:
        values = (
            value
            if isinstance(value, list)
            else []
        )

        names: list[str] = []

        for item in values:
            _, payload = (
                cls._node_type(item)
            )

            value_name = (
                cls._string_value(
                    payload
                    if payload is not None
                    else item
                )
            )

            if value_name is not None:
                names.append(
                    value_name
                )

        return names

    @classmethod
    def _walk_nodes(
        cls,
        value: Any,
    ) -> Iterable[Any]:
        node_type, _ = (
            cls._node_type(value)
        )

        if node_type is not None:
            yield value

        if isinstance(value, dict):
            for child in value.values():
                yield from cls._walk_nodes(
                    child
                )

        elif isinstance(value, list):
            for child in value:
                yield from cls._walk_nodes(
                    child
                )

    @staticmethod
    def _slot_index(
        slot: str,
    ) -> Optional[int]:
        match = re.fullmatch(
            r"slot([1-9][0-9]*)",
            slot,
        )

        if not match:
            return None

        return (
            int(match.group(1))
            - 1
        )

    def _match_column_terminal(
        self,
        payload: dict[str, Any],
        rule: dict[str, Any],
    ) -> bool:
        fields = payload.get("fields")

        names = self._name_list(
            fields
        )

        if not names:
            return False

        column_name = names[-1]

        group = rule.get(
            "column_type_group"
        )

        if group == "integer":
            return (
                column_name
                in self.integer_columns
            )

        if group == "float":
            return (
                column_name
                in self.float_columns
            )

        if group == "string":
            return (
                column_name
                in self.string_columns
            )

        return False

    @staticmethod
    def _extract_const_payload(
        payload: dict[str, Any],
    ) -> Any:
        if "val" in payload:
            return payload["val"]

        return payload

    def _match_literal_terminal(
        self,
        payload: dict[str, Any],
        rule: dict[str, Any],
    ) -> bool:
        literal_kind = rule.get(
            "literal_kind"
        )

        value = (
            self._extract_const_payload(
                payload
            )
        )

        text = json.dumps(
            value,
            ensure_ascii=False,
        ).lower()

        if literal_kind == "string":
            return (
                "sval" in text
                or '"str"' in text
            )

        if literal_kind == "integer":
            if (
                "ival" in text
                or "integer" in text
            ):
                return True

            if (
                "fval" in text
                or "float" in text
                or "sval" in text
            ):
                return False

            return bool(
                re.fullmatch(
                    r"-?[0-9]+",
                    str(value),
                )
            )

        if literal_kind == "float":
            if (
                "fval" in text
                or "float" in text
            ):
                return True

            if "sval" in text:
                return False

            return bool(
                re.fullmatch(
                    (
                        r"-?"
                        r"(?:"
                        r"[0-9]+\.[0-9]*"
                        r"|"
                        r"[0-9]*\.[0-9]+"
                        r")"
                        r"(?:[eE][+-]?[0-9]+)?"
                    ),
                    str(value),
                )
            )

        return False

    def _matches(
        self,
        node: Any,
        token: str,
        rule: dict[str, Any],
    ) -> bool:
        node_type, payload = (
            self._node_type(node)
        )

        if (
            node_type
            != rule.get("ast_node")
            or payload is None
        ):
            return False

        mode = rule.get("mode")

        if mode == "terminal_column":
            return (
                self._match_column_terminal(
                    payload,
                    rule,
                )
            )

        if mode == "terminal_literal":
            return (
                self._match_literal_terminal(
                    payload,
                    rule,
                )
            )

        if mode == "terminal_null":
            if payload.get("isnull") is True:
                return True

            value = payload.get("val")

            return value is None

        if mode == "terminal_boolean":
            expected = bool(
                rule.get("bool_value")
            )

            text = json.dumps(
                payload,
                ensure_ascii=False,
            ).lower()

            if expected:
                return (
                    '"boolval": true'
                    in text
                    or '"boolval":true'
                    in text
                )

            return (
                '"boolval": false'
                in text
                or '"boolval":false'
                in text
            )

        if (
            mode
            == "sql_value_function"
        ):
            enum_field = rule.get(
                "enum_field"
            )

            actual = str(
                payload.get(
                    enum_field,
                    "",
                )
            )

            return (
                actual
                in rule.get(
                    "enum_values",
                    [],
                )
            )

        if (
            mode
            in {
                "terminal_function",
                "normalized_test",
            }
            or "function_name" in rule
        ):
            names = self._name_list(
                payload.get("funcname")
            )

            function_name = (
                rule.get(
                    "function_name"
                )
            )

            if function_name:
                if not names:
                    return False

                if (
                    names[-1].lower()
                    != str(
                        function_name
                    ).lower()
                ):
                    return False

            if (
                mode
                == "normalized_test"
                and rule.get("negated")
            ):
                return False

            return True

        if "operator_names" in rule:
            names = self._name_list(
                payload.get("name")
            )

            if (
                not names
                or names[-1]
                not in rule[
                    "operator_names"
                ]
            ):
                return False

        enum_field = rule.get(
            "enum_field"
        )

        if enum_field:
            actual = str(
                payload.get(
                    enum_field,
                    "",
                )
            )

            if (
                actual
                not in rule.get(
                    "enum_values",
                    [],
                )
            ):
                return False

        return True

    def _matches_normalized_negated(
        self,
        node: Any,
        rule: dict[str, Any],
    ) -> bool:
        if (
            rule.get("mode")
            != "normalized_test"
            or not rule.get("negated")
        ):
            return False

        node_type, payload = (
            self._node_type(node)
        )

        if (
            node_type
            != "BoolExpr"
            or payload is None
        ):
            return False

        if (
            str(
                payload.get(
                    "boolop",
                    "",
                )
            )
            != "NOT_EXPR"
        ):
            return False

        args = payload.get("args")

        if (
            not isinstance(args, list)
            or not args
        ):
            return False

        inner = args[0]

        inner_type, inner_payload = (
            self._node_type(inner)
        )

        if (
            inner_type != "FuncCall"
            or inner_payload is None
        ):
            return False

        names = self._name_list(
            inner_payload.get("funcname")
        )

        return (
            bool(names)
            and names[-1].lower()
            == "is_normalized"
        )

    def _token_matches(
        self,
        node: Any,
        token: str,
        rule: dict[str, Any],
    ) -> bool:
        if self._matches(
            node,
            token,
            rule,
        ):
            return True

        return (
            self._matches_normalized_negated(
                node,
                rule,
            )
        )

    def _payload_for_rule(
        self,
        node: Any,
        rule: dict[str, Any],
    ) -> Optional[dict[str, Any]]:
        node_type, payload = (
            self._node_type(node)
        )

        if payload is None:
            return None

        if (
            rule.get("mode")
            == "normalized_test"
            and rule.get("negated")
        ):
            args = payload.get("args")

            if (
                not isinstance(args, list)
                or not args
            ):
                return None

            _, inner_payload = (
                self._node_type(
                    args[0]
                )
            )

            return inner_payload

        return payload

    def _direct_slot_children(
        self,
        node: Any,
        slot: str,
        rule: dict[str, Any],
    ) -> tuple[
        list[Any],
        Optional[str],
    ]:
        payload = self._payload_for_rule(
            node,
            rule,
        )

        if payload is None:
            return (
                [],
                "invalid AST node",
            )

        index = self._slot_index(
            slot
        )

        if index is None:
            return (
                [],
                f"Unrecognized slot name {slot!r}",
            )

        if "slot_selectors" in rule:
            selectors = rule[
                "slot_selectors"
            ]

            if index >= len(selectors):
                return (
                    [],
                    f"slot {slot} has no mapped selector",
                )

            selector = selectors[index]

            field = selector.get(
                "field"
            )

            if not field:
                return (
                    [],
                    "slot selector has no field",
                )

            value = payload.get(field)

            if (
                selector.get("list_mode")
                == "any"
            ):
                if not isinstance(
                    value,
                    list,
                ):
                    return (
                        [],
                        f"AST field {field} is not a list",
                    )

                return list(value), None

            if "list_index" in selector:
                list_index = int(
                    selector["list_index"]
                )

                if (
                    not isinstance(
                        value,
                        list,
                    )
                    or list_index
                    >= len(value)
                ):
                    return (
                        [],
                        (
                            f"AST field {field} "
                            f"index {list_index} "
                            "is absent"
                        ),
                    )

                return [
                    value[list_index]
                ], None

            if value is None:
                return (
                    [],
                    f"AST field {field} is absent",
                )

            return [value], None

        if "slot_fields" in rule:
            fields = rule[
                "slot_fields"
            ]

            if index >= len(fields):
                return (
                    [],
                    f"slot {slot} has no mapped AST field",
                )

            field = fields[index]

            child = payload.get(
                field
            )

            if child is None:
                return (
                    [],
                    f"AST field {field} is absent",
                )

            if (
                index == 1
                and rule.get(
                    "semantic_rhs_wrapper"
                )
            ):
                child_type, child_payload = (
                    self._node_type(child)
                )

                if (
                    child_type == "FuncCall"
                    and child_payload is not None
                ):
                    names = self._name_list(
                        child_payload.get(
                            "funcname"
                        )
                    )

                    if (
                        names
                        and names[-1].lower()
                        == "similar_to_escape"
                    ):
                        args = child_payload.get(
                            "args"
                        )

                        if (
                            isinstance(args, list)
                            and args
                        ):
                            child = args[0]

            return [child], None

        field = rule.get(
            "slot_field"
        )

        if not field:
            return (
                [],
                "mapping has no slot field",
            )

        value = payload.get(
            field
        )

        if isinstance(value, list):
            if index >= len(value):
                return (
                    [],
                    (
                        "AST argument index "
                        f"{index} is absent"
                    ),
                )

            return [
                value[index]
            ], None

        if index != 0:
            return (
                [],
                (
                    f"AST field {field} "
                    "supports only slot1"
                ),
            )

        if value is None:
            return (
                [],
                f"AST field {field} is absent",
            )

        return [value], None

    def check(
        self,
        predicate: str,
        target: Target,
    ) -> tuple[
        str,
        bool,
        str,
    ]:
        source_rule, source_problem = (
            self.resolve_mapping(
                target.source_operator
            )
        )

        target_rule, target_problem = (
            self.resolve_mapping(
                target.target_operator
            )
        )

        if (
            source_problem
            or target_problem
        ):
            return (
                "unresolved",
                False,
                (
                    source_problem
                    or target_problem
                    or "unresolved mapping"
                ),
            )

        if self.parse_sql_json is None:
            return (
                "unresolved",
                False,
                "pglast is not installed",
            )

        try:
            parsed = json.loads(
                self.parse_sql_json(
                    (
                        "SELECT 1 WHERE ("
                        f"{predicate}"
                        ")"
                    )
                )
            )

        except Exception as exc:
            return (
                "parse_error",
                False,
                (
                    "pglast parse failure: "
                    f"{exc}"
                ),
            )

        for node in self._walk_nodes(
            parsed
        ):
            if not self._token_matches(
                node,
                target.source_operator,
                source_rule,
            ):
                continue

            children, problem = (
                self._direct_slot_children(
                    node,
                    target.source_slot,
                    source_rule,
                )
            )

            if problem:
                continue

            for child in children:
                if self._token_matches(
                    child,
                    target.target_operator,
                    target_rule,
                ):
                    return (
                        "verified",
                        True,
                        (
                            "target operator is the "
                            "direct slot child"
                        ),
                    )

        return (
            "not_present",
            False,
            (
                "target operator is not the direct "
                "child of the requested source slot"
            ),
        )



class PostgreSQLAdapter:

    def __init__(
        self,
        config: dict[str, Any],
    ):
        self.config = config

    def _connect(self):
        try:
            import psycopg

        except ImportError as exc:
            raise RuntimeError(
                "psycopg is required for PostgreSQL validation."
            ) from exc

        connection_config = {
            key: self.config[key]
            for key in (
                "host",
                "port",
                "dbname",
                "user",
                "password",
            )
        }
        if "connect_timeout" in self.config:
            connection_config["connect_timeout"] = self.config["connect_timeout"]

        options = (
            "-c default_transaction_read_only=on "
            "-c statement_timeout="
            f"{int(self.config['statement_timeout_ms'])}"
        )

        return psycopg.connect(
            **connection_config,
            autocommit=True,
            options=options,
        )

    def get_columns(
        self,
    ) -> list[dict[str, Any]]:
        query = (
            "SELECT "
            "column_name, "
            "data_type, "
            "udt_name, "
            "is_nullable, "
            "ordinal_position "
            "FROM information_schema.columns "
            "WHERE table_schema = %s "
            "AND table_name = %s "
            "ORDER BY ordinal_position"
        )

        with (
            self._connect() as connection,
            connection.cursor() as cursor,
        ):
            cursor.execute(
                query,
                (
                    self.config["schema"],
                    self.config["table"],
                ),
            )

            field_names = [
                column.name
                for column in cursor.description
            ]

            columns = [
                dict(
                    zip(
                        field_names,
                        row,
                    )
                )
                for row in cursor.fetchall()
            ]

        if not columns:
            raise RuntimeError(
                "No columns found for "
                f"{self.config['schema']}."
                f"{self.config['table']}"
            )

        return columns

    def validate(
        self,
        predicate: str,
    ) -> dict[str, Any]:
        safe, reason = (
            predicate_is_safe_for_validation(
                predicate
            )
        )

        if not safe:
            return {
                "db_executable": False,
                "db_satisfiable": False,
                "db_status":
                    "rejected_unsafe_predicate",
                "db_elapsed_seconds": 0.0,
                "sqlstate": None,
                "db_error": reason,
            }

        started = time.perf_counter()

        try:
            from psycopg import sql

            with (
                self._connect() as connection,
                connection.cursor() as cursor,
            ):
                query = sql.SQL(
                    (
                        "SELECT 1 "
                        "FROM {}.{} "
                        "WHERE ({}) "
                        "LIMIT 1"
                    )
                ).format(
                    sql.Identifier(
                        self.config["schema"]
                    ),
                    sql.Identifier(
                        self.config["table"]
                    ),
                    sql.SQL(predicate),
                )

                cursor.execute(query)

                row = cursor.fetchone()

            elapsed = (
                time.perf_counter()
                - started
            )

            return {
                "db_executable": True,
                "db_satisfiable":
                    row is not None,
                "db_status":
                    (
                        "executed_satisfiable"
                        if row is not None
                        else "executed_unsatisfied"
                    ),
                "db_elapsed_seconds":
                    elapsed,
                "sqlstate": None,
                "db_error": None,
            }

        except Exception as exc:
            return {
                "db_executable": False,
                "db_satisfiable": False,
                "db_status":
                    "execution_error",
                "db_elapsed_seconds": (
                    time.perf_counter()
                    - started
                ),
                "sqlstate":
                    getattr(
                        exc,
                        "sqlstate",
                        None,
                    ),
                "db_error": str(exc),
            }


class FakeDatabaseAdapter:

    def get_columns(
        self,
    ) -> list[dict[str, Any]]:
        return list(
            MOCK_COLUMNS
        )

    def validate(
        self,
        predicate: str,
    ) -> dict[str, Any]:
        safe, reason = (
            predicate_is_safe_for_validation(
                predicate
            )
        )

        return {
            "db_executable": bool(safe),
            "db_satisfiable": bool(safe),
            "db_status": (
                "dry_run_mock_satisfiable"
                if safe
                else "rejected_unsafe_predicate"
            ),
            "db_elapsed_seconds": 0.0,
            "sqlstate": None,
            "db_error": reason,
        }



@dataclass
class ResultPaths:
    root: Path

    @property
    def interactions(self) -> Path:
        return (
            self.root
            / "interactions.jsonl"
        )

    @property
    def validations(self) -> Path:
        return (
            self.root
            / "validation.jsonl"
        )

    @property
    def discovered(self) -> Path:
        return (
            self.root
            / "discovered_edges.jsonl"
        )

    @property
    def progress(self) -> Path:
        return (
            self.root
            / "progress.csv"
        )

    @property
    def summary(self) -> Path:
        return (
            self.root
            / "summary.json"
        )

    @property
    def manifest(self) -> Path:
        return (
            self.root
            / "run_manifest.json"
        )

    @property
    def log(self) -> Path:
        return (
            self.root
            / "experiment.log"
        )

    @property
    def initialization(self) -> Path:
        return (
            self.root
            / "initialization.jsonl"
        )



def configure_logger(
    path: Path,
) -> logging.Logger:
    logger = logging.getLogger(
        f"llm_pg_edge_discovery.{path}"
    )

    logger.setLevel(
        logging.INFO
    )

    logger.propagate = False

    for existing in list(
        logger.handlers
    ):
        logger.removeHandler(
            existing
        )
        existing.close()

    handler = logging.FileHandler(
        path,
        encoding="utf-8",
    )

    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s "
            "%(levelname)s "
            "%(message)s"
        )
    )

    logger.addHandler(
        handler
    )

    return logger


def close_logger(
    logger: logging.Logger,
) -> None:
    for handler in list(
        logger.handlers
    ):
        logger.removeHandler(
            handler
        )
        handler.close()



class ProgressState:
    def __init__(
        self,
        interactions: Iterable[
            dict[str, Any]
        ],
        validations: Iterable[
            dict[str, Any]
        ],
    ):
        self.interactions = 0
        self.generate_calls = 0

        self.input_tokens = 0
        self.output_tokens = 0
        self.total_tokens = 0

        self.token_usage_interactions = 0

        self.request_elapsed_seconds = 0.0

        self.parsed_interactions = 0
        self.refused_targets = 0
        self.generated_predicates = 0

        self.validated_predicates = 0
        self.structure_verified = 0
        self.db_executable = 0
        self.db_satisfiable = 0

        self.successful_edges: set[str] = set()

        for record in interactions:
            self.add_interaction(
                record
            )

        for record in validations:
            self.add_validation(
                record
            )

    def add_interaction(
        self,
        record: dict[str, Any],
    ) -> None:
        self.interactions += 1

        if record.get(
            "generate_called"
        ):
            self.generate_calls += 1

        input_tokens = record.get(
            "input_tokens"
        )

        output_tokens = record.get(
            "output_tokens"
        )

        total_tokens = record.get(
            "total_tokens"
        )

        if (
            isinstance(
                input_tokens,
                int,
            )
            and isinstance(
                output_tokens,
                int,
            )
            and isinstance(
                total_tokens,
                int,
            )
        ):
            self.input_tokens += (
                input_tokens
            )

            self.output_tokens += (
                output_tokens
            )

            self.total_tokens += (
                total_tokens
            )

            self.token_usage_interactions += 1

        self.request_elapsed_seconds += float(
            record.get(
                "request_elapsed_seconds",
                0.0,
            )
            or 0.0
        )

        if record.get(
            "parse_success"
        ):
            self.parsed_interactions += 1

        if record.get(
            "refused"
        ):
            self.refused_targets += 1

        predicates = record.get(
            "parsed_predicates"
        )

        if isinstance(
            predicates,
            list,
        ):
            self.generated_predicates += (
                len(predicates)
            )

    def add_validation(
        self,
        record: dict[str, Any],
    ) -> None:
        self.validated_predicates += 1

        if (
            record.get(
                "structure_status"
            )
            == "verified"
        ):
            self.structure_verified += 1

        if record.get(
            "db_executable"
        ):
            self.db_executable += 1

        if record.get(
            "db_satisfiable"
        ):
            self.db_satisfiable += 1

        if record.get(
            "final_edge_success"
        ):
            edge = record.get(
                "target_edge"
            )

            if isinstance(edge, str):
                self.successful_edges.add(
                    edge
                )


def append_progress(
    path: Path,
    state: ProgressState,
    phase: str,
) -> None:
    columns = [
        "timestamp_utc",
        "phase",
        "interactions",
        "generate_calls",
        "input_tokens",
        "output_tokens",
        "total_tokens",
        "token_usage_interactions",
        "request_elapsed_seconds",
        "parsed_interactions",
        "refused_targets",
        "generated_predicates",
        "validated_predicates",
        "structure_verified",
        "db_executable",
        "db_satisfiable",
        "successful_edges",
    ]

    is_new = not path.exists()

    with path.open(
        "a",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=columns,
        )

        if is_new:
            writer.writeheader()

        writer.writerow(
            {
                "timestamp_utc":
                    time.strftime(
                        "%Y-%m-%dT%H:%M:%SZ",
                        time.gmtime(),
                    ),

                "phase":
                    phase,

                "interactions":
                    state.interactions,

                "generate_calls":
                    state.generate_calls,

                "input_tokens":
                    state.input_tokens,

                "output_tokens":
                    state.output_tokens,

                "total_tokens":
                    state.total_tokens,

                "token_usage_interactions":
                    state.token_usage_interactions,

                "request_elapsed_seconds":
                    (
                        f"{state.request_elapsed_seconds:.9f}"
                    ),

                "parsed_interactions":
                    state.parsed_interactions,

                "refused_targets":
                    state.refused_targets,

                "generated_predicates":
                    state.generated_predicates,

                "validated_predicates":
                    state.validated_predicates,

                "structure_verified":
                    state.structure_verified,

                "db_executable":
                    state.db_executable,

                "db_satisfiable":
                    state.db_satisfiable,

                "successful_edges":
                    len(
                        state.successful_edges
                    ),
            }
        )

        handle.flush()
        os.fsync(
            handle.fileno()
        )



def plan_hash(
    targets: list[Target],
) -> str:
    joined = "\n".join(
        target.interaction_id
        for target in targets
    )

    return hashlib.sha256(
        joined.encode("ascii")
    ).hexdigest()


def prepare_results(
    result_root: Path,
    run_mode: str,
    backend_name: str,
    model_name: str,
    targets: list[Target],
    input_paths: dict[str, Path],
    max_predicates_per_target: int,
) -> tuple[
    ResultPaths,
    logging.Logger,
]:
    result_root = resolve_output_path(
        result_root
    )

    if result_root != RESULTS_ROOT:
        raise ValueError("All experiment outputs must be directly in results/")

    result_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    paths = ResultPaths(
        result_root
    )

    logger = configure_logger(
        paths.log
    )

    manifest = {
        "schema_version": 3,
        "run_mode": run_mode,
        "backend": backend_name,
        "model_name": model_name,
        "planned_targets": len(targets),
        "plan_hash":
            plan_hash(targets),
        "max_predicates_per_target":
            max_predicates_per_target,
        "input_sha256": {
            name:
                file_digest(path)
            for name, path
            in input_paths.items()
        },
    }

    if paths.manifest.exists():
        existing = read_json(
            paths.manifest
        )

        keys = (
            "schema_version",
            "run_mode",
            "backend",
            "model_name",
            "planned_targets",
            "plan_hash",
            "max_predicates_per_target",
            "input_sha256",
        )

        if any(
            existing.get(key)
            != manifest[key]
            for key in keys
        ):
            raise RuntimeError(
                "Existing result directory belongs "
                "to a different plan or input set. "
                "Back up and remove results/ before starting a new experiment."
            )

    else:
        atomic_json_write(
            paths.manifest,
            manifest,
        )

    for jsonl_path in (
        paths.interactions,
        paths.validations,
        paths.discovered,
    ):
        jsonl_path.touch(
            exist_ok=True
        )

    return paths, logger


def index_interactions(
    records: Iterable[
        dict[str, Any]
    ],
) -> dict[str, dict[str, Any]]:
    indexed: dict[
        str,
        dict[str, Any]
    ] = {}

    for record in records:
        interaction_id = (
            record.get(
                "interaction_id"
            )
        )

        if isinstance(
            interaction_id,
            str,
        ):
            indexed[
                interaction_id
            ] = record

    return indexed


def predicate_id_for(
    interaction_id: str,
    predicate_index: int,
) -> str:
    return (
        f"{interaction_id}:"
        f"{predicate_index:03d}"
    )


def index_validations(
    records: Iterable[
        dict[str, Any]
    ],
) -> dict[str, dict[str, Any]]:
    indexed: dict[
        str,
        dict[str, Any]
    ] = {}

    for record in records:
        predicate_id = (
            record.get(
                "predicate_id"
            )
        )

        if isinstance(
            predicate_id,
            str,
        ):
            indexed[
                predicate_id
            ] = record

    return indexed



def load_reference_edges(
    path: Path,
) -> Optional[set[str]]:
    if not path.is_file():
        return None

    values: set[str] = set()

    for number, line in enumerate(
        path.read_text(
            encoding="utf-8"
        ).splitlines(),
        start=1,
    ):
        if not line.strip():
            continue

        try:
            value = json.loads(
                line
            )

        except json.JSONDecodeError as exc:
            raise ValueError(
                "Invalid reference_edges.jsonl "
                f"record at line {number}: "
                f"{exc}"
            ) from exc

        if not isinstance(
            value,
            dict,
        ):
            raise ValueError(
                "reference_edges.jsonl "
                f"line {number} must be an object"
            )

        if isinstance(
            value.get("edge"),
            str,
        ):
            values.add(
                value["edge"]
            )
            continue

        keys = (
            "source_operator",
            "source_slot",
            "target_operator",
        )

        if not all(
            isinstance(
                value.get(key),
                str,
            )
            for key in keys
        ):
            raise ValueError(
                "reference_edges.jsonl line "
                f"{number} needs edge or "
                "source_operator/source_slot/"
                "target_operator"
            )

        values.add(
            f"{value['source_operator']}."
            f"{value['source_slot']} -> "
            f"{value['target_operator']}"
        )

    return values



def build_summary(
    tokens: list[str],
    slots: dict[str, list[str]],
    targets: list[Target],
    interactions: list[dict[str, Any]],
    validations: list[dict[str, Any]],
    initialization: list[dict[str, Any]],
    reference_edges: Optional[set[str]],
    run_mode: str,
    backend_name: str,
    model_name: str,
    run_status: str,
    stop_reason: Optional[str],
) -> dict[str, Any]:
    state = ProgressState(
        interactions,
        validations,
    )

    generation_seconds = (
        state.request_elapsed_seconds
    )

    interactions_per_minute = (
        (
            state.generate_calls
            / generation_seconds
            * 60.0
        )
        if generation_seconds > 0
        else None
    )

    predicates_per_minute = (
        (
            state.generated_predicates
            / generation_seconds
            * 60.0
        )
        if generation_seconds > 0
        else None
    )

    generated_count = (
        state.generated_predicates
    )

    executable_count = (
        state.db_executable
    )

    satisfiable_count = (
        state.db_satisfiable
    )

    executable_rate = (
        executable_count
        / generated_count
        * 100.0
        if generated_count
        else 0.0
    )

    satisfiable_rate = (
        satisfiable_count
        / generated_count
        * 100.0
        if generated_count
        else 0.0
    )

    executable_conditional_satisfiable_rate = (
        satisfiable_count
        / executable_count
        * 100.0
        if executable_count
        else 0.0
    )

    db_total = sum(
        float(
            record.get(
                "db_elapsed_seconds",
                0.0,
            )
            or 0.0
        )
        for record in validations
    )

    rag_total = sum(
        float(
            record.get(
                "rag_retrieval_seconds",
                0.0,
            )
            or 0.0
        )
        for record in interactions
    )

    successful_edges = (
        state.successful_edges
    )

    actual_calls = (
        state.generate_calls
    )

    usage_count = (
        state.token_usage_interactions
    )

    summary: dict[str, Any] = {
        "run_mode":
            run_mode,

        "run_status":
            run_status,

        "stop_reason":
            stop_reason,

        "backend":
            backend_name,

        "model_name":
            model_name,

        "tokens_count":
            len(tokens),

        "source_slots_count":
            sum(
                len(value)
                for value in slots.values()
            ),

        "planned_targets":
            len(targets),

        "llm_interactions":
            actual_calls,

        "interaction_records":
            state.interactions,

        "completed_targets":
            state.interactions,

        "remaining_targets":
            max(
                0,
                len(targets)
                - state.interactions,
            ),

        "parsed_interactions":
            state.parsed_interactions,

        "parse_failed_interactions":
            (
                state.interactions
                - state.parsed_interactions
            ),

        "refused_targets":
            state.refused_targets,

        "generated_predicates":
            generated_count,

        "token_usage_interactions":
            usage_count,

        "token_usage_missing_interactions":
            max(
                0,
                actual_calls
                - usage_count,
            ),

        "token_usage_complete":
            usage_count
            == actual_calls,

        "total_input_tokens":
            state.input_tokens,

        "total_output_tokens":
            state.output_tokens,

        "total_tokens":
            state.total_tokens,

        "avg_input_tokens_per_interaction":
            (
                state.input_tokens
                / usage_count
                if usage_count
                else None
            ),

        "avg_output_tokens_per_interaction":
            (
                state.output_tokens
                / usage_count
                if usage_count
                else None
            ),

        "avg_total_tokens_per_interaction":
            (
                state.total_tokens
                / usage_count
                if usage_count
                else None
            ),

        "llm_request_total_seconds":
            generation_seconds,

        "llm_request_total_hours":
            generation_seconds
            / 3600.0,

        "avg_llm_request_seconds":
            (
                generation_seconds
                / actual_calls
                if actual_calls
                else None
            ),

        "interactions_per_minute":
            interactions_per_minute,

        "generated_predicates_per_minute":
            predicates_per_minute,

        "rag_retrieval_total_seconds":
            rag_total,

        "db_validation_total_seconds":
            db_total,

        "validated_predicates":
            state.validated_predicates,

        "structure_verified_count":
            state.structure_verified,

        "db_executable_count":
            executable_count,

        "db_satisfiable_count":
            satisfiable_count,

        "predicate_executable_rate_percent":
            executable_rate,

        "predicate_satisfiable_rate_percent":
            satisfiable_rate,

        "executable_conditional_satisfiable_rate_percent":
            executable_conditional_satisfiable_rate,

        "discovered_unique_edges":
            len(successful_edges),

        "target_success_rate_percent":
            (
                len(successful_edges)
                / len(targets)
                * 100.0
                if targets
                else 0.0
            ),

        "rag_index_build_seconds":
            sum(
                float(
                    record.get(
                        "rag_index_build_seconds",
                        0.0,
                    )
                    or 0.0
                )
                for record
                in initialization
            ),
    }

    if reference_edges is not None:
        discovered_reference_edges = (
            successful_edges
            & reference_edges
        )

        summary.update(
            {
                "reference_edges_count":
                    len(reference_edges),

                "discovered_reference_edges":
                    len(
                        discovered_reference_edges
                    ),

                "reference_edge_coverage_percent":
                    (
                        len(
                            discovered_reference_edges
                        )
                        / len(reference_edges)
                        * 100.0
                        if reference_edges
                        else 0.0
                    ),
            }
        )

    return summary


def write_summary_snapshot(
    paths: ResultPaths,
    tokens: list[str],
    slots: dict[str, list[str]],
    targets: list[Target],
    interactions: list[dict[str, Any]],
    validations: list[dict[str, Any]],
    initialization: list[dict[str, Any]],
    reference_edges: Optional[set[str]],
    run_mode: str,
    backend_name: str,
    model_name: str,
    run_status: str,
    stop_reason: Optional[str],
    rag_loaded_from_cache: Optional[bool],
    token_count_method: str,
    max_predicates_per_target: int,
) -> dict[str, Any]:
    summary = build_summary(
        tokens,
        slots,
        targets,
        interactions,
        validations,
        initialization,
        reference_edges,
        run_mode,
        backend_name,
        model_name,
        run_status,
        stop_reason,
    )

    summary.update(
        {
            "rag_index_loaded_from_cache":
                rag_loaded_from_cache,

            "result_directory":
                str(paths.root),

            "token_count_method":
                token_count_method,

            "max_predicates_per_target":
                max_predicates_per_target,
        }
    )

    atomic_json_write(
        paths.summary,
        summary,
    )

    return summary



def run_experiment(
    arguments: argparse.Namespace,
) -> dict[str, Any]:
    config_path = require_internal_path(
        Path(arguments.config)
        if arguments.config
        else DEFAULT_CONFIG
    )

    config = load_experiment_config(
        config_path
    )

    token_path = require_internal_path(
        ROOT / "tokens.txt"
    )

    slot_path = require_internal_path(
        ROOT / "slots.json"
    )

    example_path = require_internal_path(
        ROOT / "examples.txt"
    )

    ast_map_path = require_internal_path(
        ROOT / "token_ast_map.json"
    )

    tokens = load_tokens(
        token_path
    )

    slots = load_slots(
        slot_path,
        tokens,
    )

    examples = load_examples(
        example_path
    )

    targets = list(
        enumerate_targets(
            tokens,
            slots,
        )
    )

    if arguments.limit is not None:
        if arguments.limit < 0:
            raise ValueError(
                "--limit must be non-negative"
            )

        targets = targets[
            :arguments.limit
        ]

    generation_config = config.get(
        "generation",
        {},
    )

    if not isinstance(
        generation_config,
        dict,
    ):
        generation_config = {}

    max_predicates_per_target = int(
        generation_config.get(
            "max_predicates_per_target",
            DEFAULT_MAX_PREDICATES_PER_TARGET,
        )
    )

    if max_predicates_per_target <= 0:
        raise ValueError(
            "generation.max_predicates_per_target "
            "must be positive"
        )

    run_mode = (
        "dry_run"
        if arguments.dry_run
        else "production"
    )

    configured_backend = str(
        config.get(
            "backend",
            "local",
        )
    )

    selected_backend = (
        arguments.backend
        or configured_backend
    )

    if selected_backend not in {
        "local",
        "online",
    }:
        raise ValueError(
            "backend must be "
            "'local' or 'online'"
        )

    backend_name = (
        "mock"
        if arguments.dry_run
        else selected_backend
    )

    model_name = (
        "deterministic-mock"
        if arguments.dry_run
        else str(
            config["model"]["name"]
            if selected_backend
            == "local"
            else config[
                "online_model"
            ]["name"]
        )
    )

    if arguments.dry_run:
        generated_predicates_name = "generated_predicates-dry-run.txt"
    elif selected_backend == "local":
        generated_predicates_name = "generated_predicates-32b.txt"
    else:
        generated_predicates_name = "generated_predicates-235b.txt"

    result_root = resolve_output_path(RESULTS_ROOT)

    if arguments.results_dir:
        requested_root = resolve_output_path(Path(arguments.results_dir))
        if requested_root != result_root:
            raise ValueError(
                "All outputs are fixed to results/ beside the script; "
                "omit --results-dir or use --results-dir results"
            )

    paths, logger = prepare_results(
        result_root,
        run_mode,
        backend_name,
        model_name,
        targets,
        {
            "tokens.txt":
                token_path,

            "slots.json":
                slot_path,

            "examples.txt":
                example_path,

            "token_ast_map.json":
                ast_map_path,

            "experiment_config.json":
                config_path,

            "configuration.json":
                SHARED_CONFIG_PATH,
        },
        max_predicates_per_target,
    )

    generated_predicates_path = resolve_output_path(
        paths.root / generated_predicates_name
    )

    satisfiable_predicates_path = resolve_output_path(
        paths.root / SATISFIABLE_PREDICATES_FILENAME
    )

    previous_interactions = read_jsonl(
        paths.interactions,
        logger,
    )

    previous_validations = read_jsonl(
        paths.validations,
        logger,
    )

    initialization_records = read_jsonl(
        paths.initialization,
        logger,
    )

    interaction_by_id = index_interactions(
        previous_interactions
    )

    validation_by_predicate_id = (
        index_validations(
            previous_validations
        )
    )

    selected_ids = {
        target.interaction_id
        for target in targets
    }

    retained_interactions = [
        record
        for record in previous_interactions
        if record.get(
            "interaction_id"
        )
        in selected_ids
    ]

    retained_validations = [
        record
        for record in previous_validations
        if record.get(
            "interaction_id"
        )
        in selected_ids
    ]

    state = ProgressState(
        retained_interactions,
        retained_validations,
    )

    rewrite_generated_predicates_txt(
        generated_predicates_path,
        retained_interactions,
    )

    rewrite_satisfiable_predicates_txt(
        satisfiable_predicates_path,
        retained_validations,
    )

    known_discovered = {
        record.get(
            "target_edge"
        )
        for record in read_jsonl(
            paths.discovered,
            logger,
        )
        if isinstance(
            record.get(
                "target_edge"
            ),
            str,
        )
    }

    reference_edges = load_reference_edges(
        require_internal_path(
            ROOT
            / "reference_edges.jsonl"
        )
    )

    token_count_method = (
        "deterministic mock tokenizer "
        "for dry-run only"
        if arguments.dry_run

        else (
            "Qwen chat-template input_ids "
            "and generated-token delta"
            if selected_backend
            == "local"

            else (
                "online API response usage "
                "fields when supplied; "
                "unavailable values remain null"
            )
        )
    )

    rag_loaded_from_cache: Optional[
        bool
    ] = None

    run_status = "initializing"

    stop_reason: Optional[str] = None

    summary = write_summary_snapshot(
        paths,
        tokens,
        slots,
        targets,
        retained_interactions,
        retained_validations,
        initialization_records,
        reference_edges,
        run_mode,
        backend_name,
        model_name,
        run_status,
        stop_reason,
        rag_loaded_from_cache,
        token_count_method,
        max_predicates_per_target,
    )

    logger.info(
        (
            "Starting %s run with backend=%s "
            "model=%s and %s planned targets"
        ),
        run_mode,
        backend_name,
        model_name,
        len(targets),
    )

    try:

        if arguments.dry_run:
            rag = fixture_rag()

            rag_index_build_seconds = 0.0

            rag_loaded_from_cache = False

            database: Any = (
                FakeDatabaseAdapter()
            )

            backend: Any = (
                MockBackend()
            )

        else:
            rag_config = config[
                "rag"
            ]

            pdf_path = (
                require_internal_path(
                    ROOT
                    / rag_config[
                        "pdf_filename"
                    ]
                )
            )

            cache_dir = (
                resolve_output_path(
                    Path(rag_config[
                        "cache_directory"
                    ])
                )
            )

            (
                rag,
                rag_index_build_seconds,
                rag_loaded_from_cache,
            ) = load_or_build_rag(
                pdf_path,
                cache_dir,
                int(
                    rag_config[
                        "chunk_chars"
                    ]
                ),
                int(
                    rag_config[
                        "overlap_chars"
                    ]
                ),
            )

            database = (
                PostgreSQLAdapter(
                    config["database"]
                )
            )

            backend = (
                QwenBackend(
                    config["model"]
                )
                if selected_backend
                == "local"

                else OnlineQwenBackend(
                    config[
                        "online_model"
                    ]
                )
            )

        initialization_record = {
            "timestamp_utc":
                time.strftime(
                    "%Y-%m-%dT%H:%M:%SZ",
                    time.gmtime(),
                ),

            "run_mode":
                run_mode,

            "backend":
                backend_name,

            "rag_index_build_seconds":
                rag_index_build_seconds,

            "rag_index_loaded_from_cache":
                rag_loaded_from_cache,
        }

        append_jsonl(
            paths.initialization,
            initialization_record,
        )

        initialization_records.append(
            initialization_record
        )

        columns = database.get_columns()

        top_k = int(
            config["rag"].get(
                "top_k",
                5,
            )
        )


        run_status = (
            "running_generation"
        )

        for target in targets:
            if (
                target.interaction_id
                in interaction_by_id
            ):
                continue

            retrieval_started = (
                time.perf_counter()
            )

            chunks = rag.retrieve(
                target.source_operator,
                target.source_slot,
                target.target_operator,
                top_k,
            )

            rag_seconds = (
                time.perf_counter()
                - retrieval_started
            )

            prompt = build_prompt(
                target,
                columns,
                chunks,
                examples,
                max_predicates_per_target,
            )

            response = backend.generate(
                prompt
            )

            if response.error:
                parse_success = False
                predicates: list[str] = []
                refused = False
                parse_error = (
                    response.error
                )

            else:
                (
                    parse_success,
                    predicates,
                    refused,
                    parse_error,
                ) = strict_predicates_json(
                    response.raw_output,
                    max_predicates_per_target,
                )

            record = {
                "interaction_id":
                    target.interaction_id,

                "source_operator":
                    target.source_operator,

                "source_slot":
                    target.source_slot,

                "target_operator":
                    target.target_operator,

                "target_edge":
                    target.edge,

                "backend":
                    backend_name,

                "model_name":
                    model_name,

                "rag_chunk_ids":
                    [
                        chunk.chunk_id
                        for chunk in chunks
                    ],

                "rag_retrieval_seconds":
                    rag_seconds,

                "input_tokens":
                    response.input_tokens,

                "output_tokens":
                    response.output_tokens,

                "total_tokens":
                    response.total_tokens,

                "request_elapsed_seconds":
                    response.request_elapsed_seconds,

                "tokenization_seconds":
                    response.tokenization_seconds,

                "llm_generate_seconds":
                    response.llm_generate_seconds,

                "decode_seconds":
                    response.decode_seconds,

                "api_call_seconds":
                    response.api_call_seconds,

                "raw_output":
                    response.raw_output,

                "parse_success":
                    parse_success,

                "parse_error":
                    parse_error,

                "refused":
                    refused,

                "parsed_predicates":
                    predicates,

                "predicate_count":
                    len(predicates),

                "generate_called":
                    response.generate_called,

                "run_mode":
                    run_mode,
            }

            append_jsonl(
                paths.interactions,
                record,
            )

            interaction_by_id[
                target.interaction_id
            ] = record

            retained_interactions.append(
                record
            )

            append_generated_predicates_txt(
                generated_predicates_path,
                predicates,
            )

            state.add_interaction(
                record
            )

            append_progress(
                paths.progress,
                state,
                "generation",
            )

            if response.stop_requested:
                stop_reason = (
                    response.stop_reason
                    or (
                        "online backend "
                        "requested a stop"
                    )
                )

                run_status = (
                    "stopping_after_api_error"
                )

            write_summary_snapshot(
                paths,
                tokens,
                slots,
                targets,
                retained_interactions,
                retained_validations,
                initialization_records,
                reference_edges,
                run_mode,
                backend_name,
                model_name,
                run_status,
                stop_reason,
                rag_loaded_from_cache,
                token_count_method,
                max_predicates_per_target,
            )

            print(f"Completed interactions: {len(interaction_by_id)}/{len(targets)}", flush=True)

            if response.stop_requested:
                logger.warning(
                    "Stopping new API requests: %s",
                    stop_reason,
                )
                break

        print("Validating predicates in the database...", flush=True)
        logger.info(
            (
                "Generation phase stopped or completed. "
                "Beginning validation phase."
            )
        )


        run_status = (
            "running_validation"
        )

        ast_mapping = load_ast_map(
            ast_map_path
        )

        checker = StructureChecker(
            ast_mapping,
            columns,
        )

        target_by_id = {
            target.interaction_id:
                target
            for target in targets
        }

        for interaction in retained_interactions:
            interaction_id = (
                interaction.get(
                    "interaction_id"
                )
            )

            if not isinstance(
                interaction_id,
                str,
            ):
                continue

            target = target_by_id.get(
                interaction_id
            )

            if target is None:
                continue

            if not interaction.get(
                "parse_success"
            ):
                continue

            predicates = interaction.get(
                "parsed_predicates"
            )

            if not isinstance(
                predicates,
                list,
            ):
                continue

            for predicate_index, predicate in enumerate(
                predicates,
                start=1,
            ):
                if not isinstance(
                    predicate,
                    str,
                ):
                    continue

                predicate_id = predicate_id_for(
                    interaction_id,
                    predicate_index,
                )

                if (
                    predicate_id
                    in validation_by_predicate_id
                ):
                    continue

                (
                    structure_status,
                    target_present,
                    structure_reason,
                ) = checker.check(
                    predicate,
                    target,
                )

                db_result = database.validate(
                    predicate
                )

                final_edge_success = bool(
                    structure_status
                    == "verified"
                    and target_present
                    and db_result[
                        "db_executable"
                    ]
                )

                validation = {
                    "predicate_id":
                        predicate_id,

                    "predicate_index":
                        predicate_index,

                    "interaction_id":
                        interaction_id,

                    "source_operator":
                        target.source_operator,

                    "source_slot":
                        target.source_slot,

                    "target_operator":
                        target.target_operator,

                    "target_edge":
                        target.edge,

                    "predicate":
                        predicate,

                    "structure_status":
                        structure_status,

                    "structure_reason":
                        structure_reason,

                    "target_edge_present":
                        target_present,

                    **db_result,

                    "final_edge_success":
                        final_edge_success,

                    "run_mode":
                        run_mode,
                }

                append_jsonl(
                    paths.validations,
                    validation,
                )

                validation_by_predicate_id[
                    predicate_id
                ] = validation

                retained_validations.append(
                    validation
                )

                state.add_validation(
                    validation
                )

                if validation.get("db_satisfiable") is True:
                    append_generated_predicates_txt(
                        satisfiable_predicates_path,
                        [predicate],
                    )

                if (
                    final_edge_success
                    and target.edge
                    not in known_discovered
                ):
                    discovered_record = {
                        "interaction_id":
                            interaction_id,

                        "predicate_id":
                            predicate_id,

                        "source_operator":
                            target.source_operator,

                        "source_slot":
                            target.source_slot,

                        "target_operator":
                            target.target_operator,

                        "target_edge":
                            target.edge,

                        "predicate":
                            predicate,
                    }

                    append_jsonl(
                        paths.discovered,
                        discovered_record,
                    )

                    known_discovered.add(
                        target.edge
                    )

                if state.validated_predicates % 500 == 0:
                    append_progress(
                        paths.progress,
                        state,
                        "validation",
                    )

                    write_summary_snapshot(
                        paths,
                        tokens,
                        slots,
                        targets,
                        retained_interactions,
                        retained_validations,
                        initialization_records,
                        reference_edges,
                        run_mode,
                        backend_name,
                        model_name,
                        run_status,
                        stop_reason,
                        rag_loaded_from_cache,
                        token_count_method,
                        max_predicates_per_target,
                    )

        if (
            len(interaction_by_id)
            >= len(targets)
            and stop_reason is None
        ):
            run_status = "complete"

        else:
            run_status = "partial"

        summary = write_summary_snapshot(
            paths,
            tokens,
            slots,
            targets,
            retained_interactions,
            retained_validations,
            initialization_records,
            reference_edges,
            run_mode,
            backend_name,
            model_name,
            run_status,
            stop_reason,
            rag_loaded_from_cache,
            token_count_method,
            max_predicates_per_target,
        )

        logger.info(
            (
                "Run %s: interactions=%s, "
                "predicates=%s, "
                "executable=%s, "
                "satisfiable=%s, "
                "successful_edges=%s, "
                "remaining_targets=%s"
            ),
            run_status,
            summary[
                "llm_interactions"
            ],
            summary[
                "generated_predicates"
            ],
            summary[
                "db_executable_count"
            ],
            summary[
                "db_satisfiable_count"
            ],
            summary[
                "discovered_unique_edges"
            ],
            summary[
                "remaining_targets"
            ],
        )

        return summary

    except BaseException as exc:
        run_status = (
            "interrupted"
            if isinstance(
                exc,
                KeyboardInterrupt,
            )
            else "failed"
        )

        stop_reason = (
            f"{type(exc).__name__}: {exc}"
        )

        logger.exception(
            "Run stopped before normal completion"
        )

        raise

    finally:
        write_summary_snapshot(
            paths,
            tokens,
            slots,
            targets,
            retained_interactions,
            retained_validations,
            initialization_records,
            reference_edges,
            run_mode,
            backend_name,
            model_name,
            run_status,
            stop_reason,
            rag_loaded_from_cache,
            token_count_method,
            max_predicates_per_target,
        )

        close_logger(
            logger
        )



def parse_arguments(
    argv: Optional[list[str]] = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__
    )

    parser.add_argument(
        "--config",
        help=(
            "Path inside LLM/ to "
            "experiment_config.json"
        ),
    )

    parser.add_argument(
        "--results-dir",
        help=(
            "Outputs are fixed to results/ beside the script; "
            "only results or its absolute path is accepted"
        ),
    )

    parser.add_argument(
        "--backend",
        choices=(
            "local",
            "online",
        ),
        help=(
            "Override the configured "
            "model backend"
        ),
    )

    parser.add_argument(
        "--limit",
        type=int,
        help=(
            "Only the first N deterministic "
            "targets"
        ),
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Use fixture RAG, mock LLM, "
            "and fake database"
        ),
    )

    return parser.parse_args(
        argv
    )


def main(
    argv: Optional[list[str]] = None,
) -> int:
    try:
        summary = run_experiment(
            parse_arguments(argv)
        )

        if __package__:
            from .stage3_predicate_similarity_analysis import run_analysis
            from .result_archive import archive_results
        else:
            from stage3_predicate_similarity_analysis import run_analysis
            from result_archive import archive_results
        satisfiable_path = Path(summary["result_directory"]) / SATISFIABLE_PREDICATES_FILENAME
        print("Calculating predicate similarity...", flush=True)
        with (Path(summary["result_directory"]) / "experiment.log").open(
            "a", encoding="utf-8", buffering=1,
        ) as detail_log, redirect_stdout(detail_log):
            similarity_result = run_analysis(
                project_root=ROOT,
                input_path=satisfiable_path,
                output_dir=Path(summary["result_directory"]) / "similarity_result",
            )
            archive_results(
                project_dir=ROOT.parents[1],
                predicate_path=satisfiable_path,
                experiment=summary,
                similarity_result=similarity_result,
            )

    except Exception as exc:
        print(
            f"error: {exc}",
            file=sys.stderr,
        )

        return 1

    print("Experiment finished.", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
