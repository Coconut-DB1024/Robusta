from __future__ import annotations

import csv
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional


COUNT_HEX_PREFIX = "hex:"
COUNT_DECIMAL_SAFE_BITS = 12000

SIGNATURE_FEEDBACK_COUNTERS = (
    "success_count",
    "failure_count",
    "type_failure_count",
    "value_failure_count",
    "other_failure_count",
    "timeout_count",
)


def merge_signature_feedback(
    current: Optional[dict],
    incoming: Optional[dict],
) -> dict:
    counters = SIGNATURE_FEEDBACK_COUNTERS + tuple(
        counter for counter in ("execution_failure_count", "environment_failure_count")
        if any(isinstance(value, dict) and counter in value
               for payload in (current or {}, incoming or {}) for value in payload.values())
    )
    merged = {
        str(key): {
            counter: int((value or {}).get(counter, 0) or 0)
            for counter in counters
        }
        for key, value in (current or {}).items()
        if isinstance(value, dict)
    }
    for key, value in (incoming or {}).items():
        if not isinstance(value, dict):
            continue
        item = merged.setdefault(
            str(key),
            {counter: 0 for counter in counters},
        )
        for counter in counters:
            item[counter] += int(value.get(counter, 0) or 0)
    return merged


class Stage3MetricsRecorder:
    CSV_FIELDS = (
        "timestamp",
        "elapsed_seconds",
        "depth",
        "phase",
        "event",
        "generated",
        "tested",
        "validated",
        "skipped",
        "sql_errors",
        "query_timeouts",
        "success_rate",
        "interval_tested",
        "interval_validated",
        "interval_success_rate",
        "statistics_overhead_seconds",
        "database_validation_raw_seconds",
        "database_validation_net_seconds",
        "net_non_database_processing_seconds",
        "net_processing_rate_predicates_per_second",
        "raw_end_to_end_rate_predicates_per_second",
        "interval_net_processing_rate_predicates_per_second",
        "interval_end_to_end_rate_predicates_per_second",
    )

    def __init__(
        self,
        output_dir: os.PathLike | str,
        *,
        start_monotonic: Optional[float] = None,
        interval_seconds: float = 600.0,
        enabled: bool = True,
        oracle_fault_exclusion: bool = False,
    ) -> None:
        self.enabled = bool(enabled)
        self.oracle_fault_exclusion = bool(oracle_fault_exclusion)
        self.oracle_excluded_cases = 0
        self.oracle_excluded_wall_seconds = 0.0
        if self.oracle_fault_exclusion:
            self.CSV_FIELDS = self.CSV_FIELDS + (
                "generated_raw", "elapsed_seconds_raw", "oracle_excluded_cases",
                "oracle_excluded_wall_seconds",
            )
        self.output_dir = Path(output_dir)
        self.csv_path = self.output_dir / "stage3_time_series.csv"
        self.summary_path = self.output_dir / "stage3_timing_summary.json"
        self.start_monotonic = (
            time.monotonic() if start_monotonic is None else float(start_monotonic)
        )
        self.interval_seconds = max(1.0, float(interval_seconds))
        self.next_snapshot_elapsed = self.interval_seconds
        self.statistics_overhead_seconds = 0.0
        self.statistics_overhead_by_phase: Dict[str, float] = {}
        self.phase_raw_seconds: Dict[str, float] = {}
        self.phase_detail_seconds: Dict[str, float] = {}
        self.active_phase: Optional[str] = None
        self.active_phase_start: Optional[float] = None
        self.current_depth = 0
        self.counters = {
            "generated": 0,
            "tested": 0,
            "validated": 0,
            "skipped": 0,
            "sql_errors": 0,
            "query_timeouts": 0,
        }
        self.rows_written = 0
        self.last_snapshot: Optional[dict] = None

        if self.enabled:
            started = time.perf_counter()
            self.output_dir.mkdir(parents=True, exist_ok=True)
            with self.csv_path.open("w", encoding="utf-8", newline="") as handle:
                csv.DictWriter(handle, fieldnames=self.CSV_FIELDS).writeheader()
            self._add_statistics_overhead(
                time.perf_counter() - started,
                phase="orchestration",
            )

    def _add_statistics_overhead(
        self,
        seconds: float,
        *,
        phase: Optional[str] = None,
    ) -> None:
        duration = max(0.0, float(seconds))
        self.statistics_overhead_seconds += duration
        phase_name = str(phase or self.active_phase or "orchestration")
        self.statistics_overhead_by_phase[phase_name] = (
            self.statistics_overhead_by_phase.get(phase_name, 0.0) + duration
        )

    def start_phase(self, phase: str, depth: Optional[int] = None) -> None:
        if not self.enabled:
            return
        started = time.perf_counter()
        if self.active_phase is not None:
            raise RuntimeError(
                f"Stage-3 metrics phase already active: {self.active_phase}"
            )
        if depth is not None:
            self.current_depth = int(depth)
        self.active_phase = str(phase)
        self.active_phase_start = time.monotonic()
        self._add_statistics_overhead(
            time.perf_counter() - started,
            phase=self.active_phase,
        )

    def finish_phase(self, phase: Optional[str] = None) -> float:
        if not self.enabled or self.active_phase is None:
            return 0.0
        started = time.perf_counter()
        now = time.monotonic()
        active_name = self.active_phase
        if phase is not None and str(phase) != active_name:
            raise RuntimeError(
                f"Stage-3 metrics phase mismatch: {active_name} != {phase}"
            )
        duration = max(0.0, now - float(self.active_phase_start or now))
        self.phase_raw_seconds[active_name] = (
            self.phase_raw_seconds.get(active_name, 0.0) + duration
        )
        self.active_phase = None
        self.active_phase_start = None
        self._add_statistics_overhead(
            time.perf_counter() - started,
            phase="orchestration",
        )
        return duration

    def update_counts(self, **counts: int) -> None:
        if not self.enabled:
            return
        started = time.perf_counter()
        for key in self.counters:
            if key in counts and counts[key] is not None:
                self.counters[key] = max(0, int(counts[key]))
        self._add_statistics_overhead(time.perf_counter() - started)

    def update_oracle_exclusions(self, current: dict, base: Optional[dict] = None) -> None:
        if not self.oracle_fault_exclusion:
            return
        base = base or {}
        self.oracle_excluded_cases = int(base.get("oracle_excluded_cases", 0)) + int(
            current.get("oracle_excluded_cases", 0))
        self.oracle_excluded_wall_seconds = float(base.get("oracle_excluded_wall_seconds", 0)) + float(
            current.get("oracle_excluded_wall_seconds", 0))

    def _effective_counters(self) -> dict:
        counters = dict(self.counters)
        if self.oracle_fault_exclusion:
            counters["generated"] = max(0, counters["generated"] - self.oracle_excluded_cases)
        return counters

    def add_phase_detail(self, phase: str, seconds: float) -> None:
        if not self.enabled:
            return
        started = time.perf_counter()
        phase_name = str(phase)
        self.phase_detail_seconds[phase_name] = (
            self.phase_detail_seconds.get(phase_name, 0.0)
            + max(0.0, float(seconds))
        )
        self._add_statistics_overhead(time.perf_counter() - started)

    def _phase_raw_snapshot(self, now: float) -> Dict[str, float]:
        values = dict(self.phase_raw_seconds)
        if self.active_phase is not None and self.active_phase_start is not None:
            values[self.active_phase] = values.get(self.active_phase, 0.0) + max(
                0.0,
                now - self.active_phase_start,
            )
        return values

    def _snapshot_values(self, now: float) -> dict:
        elapsed = max(0.0, now - self.start_monotonic)
        phase_raw = self._phase_raw_snapshot(now)
        db_raw = max(0.0, phase_raw.get("database_validation", 0.0))
        db_stats = max(
            0.0,
            self.statistics_overhead_by_phase.get("database_validation", 0.0),
        )
        db_net = max(0.0, db_raw - db_stats)

        net_non_database = max(
            0.0,
            elapsed - db_net - self.statistics_overhead_seconds,
        )
        tested = self.counters["tested"]
        validated = self.counters["validated"]
        generated = self._effective_counters()["generated"]
        extra = {}
        if self.oracle_fault_exclusion:
            extra = dict(generated_raw=self.counters["generated"], elapsed_seconds_raw=elapsed,
                         oracle_excluded_cases=self.oracle_excluded_cases,
                         oracle_excluded_wall_seconds=self.oracle_excluded_wall_seconds)

            elapsed = max(0.0, elapsed - self.oracle_excluded_wall_seconds)
            db_net = max(0.0, db_net - self.oracle_excluded_wall_seconds)
        return {
            **extra,
            "elapsed_seconds": elapsed,
            "phase_raw_seconds": phase_raw,
            "database_validation_raw_seconds": db_raw,
            "database_validation_net_seconds": db_net,
            "statistics_overhead_seconds": self.statistics_overhead_seconds,
            "net_non_database_processing_seconds": net_non_database,
            "success_rate": validated / tested if tested > 0 else 0.0,
            "net_processing_rate_predicates_per_second": (
                generated / net_non_database if net_non_database > 0.0 else 0.0
            ),
            "raw_end_to_end_rate_predicates_per_second": (
                tested / elapsed if elapsed > 0.0 else 0.0
            ),
        }

    def record(
        self,
        *,
        event: str,
        force: bool = False,
        phase: Optional[str] = None,
        depth: Optional[int] = None,
    ) -> bool:
        if not self.enabled:
            return False
        started = time.perf_counter()
        now = time.monotonic()
        elapsed = now - self.start_monotonic
        if not force and elapsed < self.next_snapshot_elapsed:
            self._add_statistics_overhead(time.perf_counter() - started)
            return False

        if depth is not None:
            self.current_depth = int(depth)
        values = self._snapshot_values(now)
        counters = self._effective_counters()
        previous = self.last_snapshot
        interval_elapsed = (
            values["elapsed_seconds"]
            - float((previous or {}).get("elapsed_seconds", 0.0))
        )
        interval_net = (
            values["net_non_database_processing_seconds"]
            - float((previous or {}).get("net_non_database_processing_seconds", 0.0))
        )
        interval_tested = self.counters["tested"] - int(
            (previous or {}).get("tested", 0)
        )
        interval_validated = self.counters["validated"] - int(
            (previous or {}).get("validated", 0)
        )
        interval_generated = counters["generated"] - int(
            (previous or {}).get("generated", 0)
        )
        row = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            "elapsed_seconds": f"{values['elapsed_seconds']:.6f}",
            "depth": self.current_depth,
            "phase": str(phase or self.active_phase or "orchestration"),
            "event": str(event),
            **counters,
            "success_rate": f"{values['success_rate']:.10f}",
            "interval_tested": interval_tested,
            "interval_validated": interval_validated,
            "interval_success_rate": f"{(interval_validated / interval_tested if interval_tested > 0 else 0.0):.10f}",
            "statistics_overhead_seconds": f"{values['statistics_overhead_seconds']:.9f}",
            "database_validation_raw_seconds": f"{values['database_validation_raw_seconds']:.6f}",
            "database_validation_net_seconds": f"{values['database_validation_net_seconds']:.6f}",
            "net_non_database_processing_seconds": f"{values['net_non_database_processing_seconds']:.6f}",
            "net_processing_rate_predicates_per_second": f"{values['net_processing_rate_predicates_per_second']:.6f}",
            "raw_end_to_end_rate_predicates_per_second": f"{values['raw_end_to_end_rate_predicates_per_second']:.6f}",
            "interval_net_processing_rate_predicates_per_second": f"{(interval_generated / interval_net if interval_net > 0.0 else 0.0):.6f}",
            "interval_end_to_end_rate_predicates_per_second": f"{(interval_tested / interval_elapsed if interval_elapsed > 0.0 else 0.0):.6f}",
        }
        if self.oracle_fault_exclusion:
            for key in ("generated_raw", "elapsed_seconds_raw", "oracle_excluded_cases",
                        "oracle_excluded_wall_seconds"):
                row[key] = values[key]

            if self.oracle_excluded_cases != (previous or {}).get("oracle_excluded_cases", 0):
                row["interval_net_processing_rate_predicates_per_second"] = ""
            if self.oracle_excluded_wall_seconds != (previous or {}).get("oracle_excluded_wall_seconds", 0):
                row["interval_end_to_end_rate_predicates_per_second"] = ""
        with self.csv_path.open("a", encoding="utf-8", newline="") as handle:
            csv.DictWriter(handle, fieldnames=self.CSV_FIELDS).writerow(row)
        self.rows_written += 1
        self.last_snapshot = {
            **counters,
            "elapsed_seconds": values["elapsed_seconds"],
            "net_non_database_processing_seconds": values[
                "net_non_database_processing_seconds"
            ],
        }
        if self.oracle_fault_exclusion:
            self.last_snapshot.update(oracle_excluded_cases=self.oracle_excluded_cases,
                                      oracle_excluded_wall_seconds=self.oracle_excluded_wall_seconds)
        while self.next_snapshot_elapsed <= elapsed:
            self.next_snapshot_elapsed += self.interval_seconds
        self._add_statistics_overhead(time.perf_counter() - started)
        return True

    def summary(self) -> dict:
        now = time.monotonic()
        values = self._snapshot_values(now)
        phase_raw = values.pop("phase_raw_seconds")
        phase_net = {
            phase: max(
                0.0,
                seconds - self.statistics_overhead_by_phase.get(phase, 0.0),
            )
            for phase, seconds in phase_raw.items()
        }
        if self.oracle_fault_exclusion and "database_validation" in phase_net:
            phase_net["database_validation"] = values["database_validation_net_seconds"]
        return {
            "enabled": self.enabled,
            "interval_seconds": self.interval_seconds,
            "time_series_path": str(self.csv_path),
            "rows_written": self.rows_written,
            "counters": self._effective_counters(),
            **values,
            "phase_raw_seconds": phase_raw,
            "phase_net_seconds": phase_net,
            "phase_detail_seconds": dict(self.phase_detail_seconds),
            "statistics_overhead_by_phase": dict(
                self.statistics_overhead_by_phase
            ),
            "net_speed_denominator": (
                "wall_clock - net_database_validation - statistics_overhead"
            ),
        }

    def write_measured_status(
        self,
        data: Any,
        output_path: os.PathLike | str,
    ) -> Path:
        started = time.perf_counter()
        path = write_status_json(data, output_path)
        self._add_statistics_overhead(
            time.perf_counter() - started,
            phase="orchestration",
        )
        return path

    def finish(self) -> dict:
        if not self.enabled:
            return self.summary()
        self.record(event="stage3_finished", force=True, phase="orchestration")
        report = self.summary()
        self.write_measured_status(report, self.summary_path)

        return self.summary()


class Stage3DeadlineReached(RuntimeError):
    def __init__(self, phase: str):
        super().__init__(f"Stage-3 deadline reached during {phase}")
        self.phase = phase


class Stage3ResourceLimitReached(RuntimeError):
    def __init__(self, phase: str, detail: str):
        super().__init__(f"Stage-3 resource limit reached during {phase}: {detail}")
        self.phase = phase
        self.detail = detail


def deadline_reached(deadline: Optional[float]) -> bool:
    return deadline is not None and time.monotonic() >= deadline


def check_deadline(deadline: Optional[float], phase: str) -> None:
    if deadline_reached(deadline):
        raise Stage3DeadlineReached(phase)


def remaining_seconds(deadline: Optional[float]) -> Optional[float]:
    if deadline is None:
        return None
    return max(0.0, deadline - time.monotonic())


def estimate_downstream_reserve(
    cost_model: Optional[dict],
    *,
    predicate_count: int,
    predicate_text_bytes: int,
    minimum_seconds: float,
    safety_factor: float,
    fixed_seconds: float,
) -> dict:
    model = cost_model or {}
    generation_seconds_per_byte = max(
        0.0,
        float(model.get("generation_seconds_per_source_byte", 0.0) or 0.0),
    )
    scan_seconds_per_case = max(
        0.0,
        float(model.get("scan_seconds_per_case", 0.0) or 0.0),
    )
    estimated_generation_seconds = (
        max(0, int(predicate_text_bytes)) * generation_seconds_per_byte
    )
    estimated_scan_seconds = max(0, int(predicate_count)) * scan_seconds_per_case
    learned_seconds = estimated_generation_seconds + estimated_scan_seconds
    has_learned_costs = (
        generation_seconds_per_byte > 0.0 or scan_seconds_per_case > 0.0
    )
    if has_learned_costs:
        reserve_seconds = max(
            max(0.0, float(minimum_seconds)),
            max(0.0, float(fixed_seconds))
            + max(1.0, float(safety_factor)) * learned_seconds,
        )
    else:
        reserve_seconds = max(0.0, float(minimum_seconds))

    return {
        "reserve_seconds": reserve_seconds,
        "estimated_generation_seconds": estimated_generation_seconds,
        "estimated_scan_seconds": estimated_scan_seconds,
        "estimated_downstream_seconds": learned_seconds,
        "has_learned_costs": has_learned_costs,
        "generation_seconds_per_source_byte": generation_seconds_per_byte,
        "scan_seconds_per_case": scan_seconds_per_case,
    }


def format_large_int(value: Any, max_bits: int = 256) -> str:
    number = decode_count(value)
    bit_length = number.bit_length()
    if bit_length <= max_bits:
        return str(number)
    approximate_digits = max(1, int(bit_length * 0.30103))
    return f"<{bit_length} bits, approximately {approximate_digits} decimal digits>"


def encode_count(value: Any) -> Any:
    if value is None:
        return None
    number = int(value)
    if number.bit_length() <= COUNT_DECIMAL_SAFE_BITS:
        return number
    return COUNT_HEX_PREFIX + format(number, "x")


def decode_count(value: Any) -> int:
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        if value.startswith(COUNT_HEX_PREFIX):
            return int(value[len(COUNT_HEX_PREFIX):], 16)
        return int(value)
    if value is None:
        return 0
    return int(value)


def encode_counts_in_place(dag_data: dict, deadline: Optional[float] = None) -> None:
    for node_id, node in dag_data.get("nodes", {}).items():
        check_deadline(deadline, f"count encoding at node {node_id}")
        if "count" in node:
            node["count"] = encode_count(node["count"])
    dag_data["_stage3_count_encoding"] = {
        "version": 1,
        "large_integer": "hexadecimal string with hex: prefix",
        "decimal_safe_bits": COUNT_DECIMAL_SAFE_BITS,
    }


def decode_counts_in_place(dag_data: dict, deadline: Optional[float] = None) -> None:
    for node_id, node in dag_data.get("nodes", {}).items():
        check_deadline(deadline, f"count decoding at node {node_id}")
        if "count" in node:
            node["count"] = decode_count(node["count"])


def atomic_json_dump(
    data: Any,
    output_path: os.PathLike | str,
    *,
    deadline: Optional[float] = None,
    phase: str = "json output",
    indent: Optional[int] = 2,
    ensure_ascii: bool = False,
    buffering: int = 1024 * 1024,
) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = Path(str(output) + ".partial")
    encoder = json.JSONEncoder(indent=indent, ensure_ascii=ensure_ascii)

    with open(
        partial,
        "w",
        encoding="utf-8",
        buffering=max(4096, int(buffering)),
    ) as handle:
        for chunk in encoder.iterencode(data):
            check_deadline(deadline, phase)
            handle.write(chunk)
        handle.write("\n")

    os.replace(partial, output)
    return output


def write_status_json(data: Any, output_path: os.PathLike | str) -> Path:
    return atomic_json_dump(
        data,
        output_path,
        deadline=None,
        phase="status output",
        indent=2,
        ensure_ascii=False,
    )
