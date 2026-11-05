from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

from .tokenizer import predicate_digest


class PredicateLookupError(RuntimeError):
    pass


def _open_text(path: Path):
    if path.name.lower().endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", newline=None)
    return open(path, "r", encoding="utf-8", newline=None)


def find_detail_record(run_dir: Path, predicate_id: int) -> Dict[str, Any]:
    """Find one locator record in the compressed detail output."""
    predicate_id = int(predicate_id)
    if predicate_id < 0:
        raise PredicateLookupError("predicate_id must be >= 0")

    detail_path = Path(run_dir).resolve() / "predicate_similarity_details.jsonl.gz"
    if not detail_path.is_file():
        raise PredicateLookupError(f"detail output does not exist: {detail_path}")

    with gzip.open(detail_path, "rt", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            current_id = int(record.get("predicate_id", -1))
            if current_id == predicate_id:
                return record
            if current_id > predicate_id:
                break
    raise PredicateLookupError(f"predicate_id was not found: {predicate_id}")


def lookup_predicate(
    *,
    project_root: Path,
    locator: Mapping[str, Any],
) -> Dict[str, Any]:
    """Read exactly one source line and verify its optional stored digest."""
    project_root = Path(project_root).resolve()
    raw_source = str(locator.get("source_file", "") or "")
    if not raw_source:
        raise PredicateLookupError("locator has no source_file")
    source_line = int(locator.get("source_line", 0) or 0)
    if source_line <= 0:
        raise PredicateLookupError("source_line must be a 1-based positive integer")

    candidate = Path(raw_source)
    source_path = (
        candidate.resolve()
        if candidate.is_absolute()
        else (project_root / candidate).resolve()
    )
    try:
        relative_source = source_path.relative_to(project_root).as_posix()
    except ValueError as exc:
        raise PredicateLookupError(
            f"source file is outside the project root: {source_path}"
        ) from exc
    if not source_path.is_file():
        raise PredicateLookupError(f"source file does not exist: {source_path}")

    predicate = None
    with _open_text(source_path) as handle:
        for current_line, line in enumerate(handle, start=1):
            if current_line == source_line:
                predicate = line.strip()
                break
    if predicate is None:
        raise PredicateLookupError(
            f"source line {source_line} does not exist in {source_path}"
        )

    actual_hash = predicate_digest(predicate).hex()
    expected_hash = str(locator.get("predicate_hash", "") or "").lower()
    if expected_hash and actual_hash != expected_hash:
        raise PredicateLookupError(
            "predicate hash mismatch; the Stage-3 source file may have changed: "
            f"expected={expected_hash}, actual={actual_hash}"
        )
    return {
        "predicate_id": locator.get("predicate_id"),
        "source_file": relative_source,
        "source_line": source_line,
        "predicate_hash": actual_hash,
        "predicate": predicate,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Retrieve one Stage-3 predicate from a similarity locator."
    )
    parser.add_argument("--project-root", type=Path, default=None)
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--predicate-id", type=int, default=None)
    parser.add_argument(
        "--nearest",
        action="store_true",
        help="Retrieve the selected record's nearest predicate instead.",
    )
    parser.add_argument("--source-file", default=None)
    parser.add_argument("--source-line", type=int, default=None)
    parser.add_argument("--predicate-hash", default=None)
    args = parser.parse_args(argv)

    project_root = (
        args.project_root.resolve()
        if args.project_root is not None
        else Path(__file__).resolve().parent.parent
    )
    try:
        if args.run_dir is not None:
            if args.predicate_id is None:
                raise PredicateLookupError(
                    "--predicate-id is required when --run-dir is used"
                )
            record = find_detail_record(args.run_dir, args.predicate_id)
            locator = record.get("nearest") if args.nearest else record
            if locator is None:
                raise PredicateLookupError("the selected predicate has no nearest match")
        else:
            if args.source_file is None or args.source_line is None:
                raise PredicateLookupError(
                    "use --run-dir with --predicate-id, or provide both "
                    "--source-file and --source-line"
                )
            locator = {
                "source_file": args.source_file,
                "source_line": args.source_line,
                "predicate_hash": args.predicate_hash,
            }

        result = lookup_predicate(project_root=project_root, locator=locator)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except (OSError, ValueError, json.JSONDecodeError, PredicateLookupError) as exc:
        print(f"[PredicateLookup] failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
