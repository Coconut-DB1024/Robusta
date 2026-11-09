import gzip
import json
import os
from pathlib import Path
import re
import shutil
import tempfile


_IDENTIFIER = r'(?:"(?:""|[^"])*"|`(?:``|[^`])*`|\[(?:\]\]|[^\]])*\]|[A-Za-z_][\w$#]*)'
_TLP_SELECT = re.compile(
    rf"\A\s*SELECT\s+COUNT\s*\(\s*\*\s*\)\s+FROM\s+"
    rf"{_IDENTIFIER}(?:\s*\.\s*{_IDENTIFIER})*\s+WHERE\s+(?P<predicate>.+)\Z",
    re.IGNORECASE | re.DOTALL,
)


def extract_tlp_predicate(sql):
    text = str(sql).strip()
    if text.endswith(";"):
        text = text[:-1].rstrip()
    match = _TLP_SELECT.fullmatch(text)
    if match is None:
        raise ValueError("Expected a generated SELECT COUNT(*) FROM table WHERE predicate query.")
    predicate = match.group("predicate").strip()
    if "\n" in predicate or "\r" in predicate:
        raise ValueError("Expected a single-line TLP predicate; refusing to change its SQL text.")
    return predicate


class SatisfiablePredicateReport:
    def __init__(self, output_path):
        self.output_path = Path(output_path)
        self._body = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self._body is not None:
            self._body.close()

    def append_cardinality_results(self, path):
        if not path:
            return

        with gzip.open(path, "rt", encoding="utf-8") as source:
            for line in source:
                record = json.loads(line)
                for variant in record["variants"]:
                    if variant["role"] not in {"true", "false", "null"}:
                        continue
                    if int(variant["cardinality"]) < 1:
                        continue
                    predicate = extract_tlp_predicate(variant["sql"])
                    if self._body is None:
                        self.output_path.parent.mkdir(parents=True, exist_ok=True)
                        self._body = tempfile.TemporaryFile(
                            mode="w+t", encoding="utf-8", newline="\n",
                            dir=self.output_path.parent,
                        )
                    self._body.write(predicate + "\n")

    def finish(self, rate_lines):
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        partial_path = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", newline="\n", delete=False,
                prefix=self.output_path.name + ".", suffix=".partial",
                dir=self.output_path.parent,
            ) as output:
                partial_path = Path(output.name)
                for line in rate_lines:
                    output.write(line + "\n")
                output.write("Satisfiable predicates:\n")
                if self._body is not None:
                    self._body.seek(0)
                    shutil.copyfileobj(self._body, output, length=1024 * 1024)
            os.replace(partial_path, self.output_path)
        finally:
            if partial_path is not None:
                partial_path.unlink(missing_ok=True)
