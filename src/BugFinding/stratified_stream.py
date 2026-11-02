"""枚举/随机的分层后处理：有界批次轮转，不改变四条 TLP SELECT 或验证规则。"""

import os
import time
from collections import deque
from contextlib import ExitStack
from pathlib import Path

from RL.config import STAGE3_CONFIG
from pkg.stage3_runtime import write_status_json


class StratifiedTaskSource:
    """每批各层取近似等量谓词；SQL 落盘与验证共用同一组，避免重复读写。"""

    def __init__(self, generator, predicate_files, layers, *, deadline, batch_size,
                 output_dir, metrics_recorder=None, depth=None):
        self.generator = generator
        self.deadline = deadline
        self.batch_size = max(1, int(batch_size))
        self.output_dir = Path(output_dir)
        self.metrics = metrics_recorder
        self.depth = depth
        self.stack = ExitStack()
        self.states = []
        self.layers = {}
        self.active = deque()
        self.by_level = {}
        self.by_tlp_name = {}
        self.predicates_read = 0
        self.elapsed_seconds = 0.0
        self.complete = False
        self.input_files = list(predicate_files)
        self.layer_manifest = layers

    def __enter__(self):
        from .sql_generator import _open_predicate_reader, _open_test_case_writer, _logical_predicate_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        buffer_bytes = max(4096, int(getattr(STAGE3_CONFIG, "stage3_output_buffer_bytes", 1048576)))
        try:
            for raw in self.input_files:
                path = Path(raw)
                info = self.layer_manifest[path.name]
                level = int(info["dag_level"])
                logical = _logical_predicate_name(path)
                name = logical.replace("output_sql", "tlp_test_case").replace(".txt", ".sql")
                if self.generator.compress_test_cases:
                    name += ".gz"
                target = self.output_dir / name
                # 先登记目标，初始化失败时也能准确清理本次 partial。
                state = dict(source=path, target=target, level=level, count=0,
                             text_characters=0, source_characters=0)
                self.states.append(state)
                reader = self.stack.enter_context(_open_predicate_reader(path, buffer_bytes))
                writer = self.stack.enter_context(_open_test_case_writer(
                    str(target) + ".partial", compressed=self.generator.compress_test_cases,
                    compression_level=self.generator.test_case_compression_level,
                    buffer_bytes=buffer_bytes))
                header = (f"-- Source: {logical}\n"
                          "-- Type: TLP Logic Checking (WHERE Partition Check)\n"
                          f"-- TLP Mode: {self.generator.tlp_mode}\n"
                          f"-- Target Table: {self.generator.table_name}\n\n")
                writer.write(header)
                state.update(reader=reader, writer=writer, text_characters=len(header))
                self.by_tlp_name[name] = state
                self.by_level.setdefault(level, deque()).append(state)
                self.layers.setdefault(str(level), dict(dag_level=level, generated=0,
                    tlp_generated=0, tested=0, validated=0, skipped=0, tlp_generation_seconds=0.0))
                self.layers[str(level)]["generated"] += int(info.get("generated", 0))
            self.active = deque(sorted(self.by_level, reverse=True))
            return self
        except BaseException:
            self.stack.close()
            for state in self.states:
                Path(str(state["target"]) + ".partial").unlink(missing_ok=True)
            raise

    @property
    def output_files(self):
        return [str(s["target"]) for s in self.states]

    @property
    def source_output_files(self):
        return {s["source"].name: str(s["target"]) for s in self.states}

    def _fill_batch(self):
        # 切换计时只发生在批次边界，禁止把 TLP 构造时间记入数据库验证。
        if self.metrics is not None:
            self.metrics.finish_phase("database_validation")
            self.metrics.start_phase("tlp_generation", self.depth)
        started = time.monotonic()
        batch = []
        try:
            while self.active and len(batch) < self.batch_size:
                if self.deadline is not None and time.monotonic() >= self.deadline:
                    break
                level = self.active[0]
                queue = self.by_level[level]
                state = queue[0]
                item_started = time.monotonic()
                line = next(state["reader"], None)
                if line is None:
                    queue.popleft()
                    if not queue:
                        self.active.popleft()
                    continue
                state["source_characters"] += len(line)
                predicate = line.strip()
                if not predicate:
                    continue
                group, block = self.generator.build_stratified_case(predicate)
                state["writer"].write(block)
                state["text_characters"] += len(block)
                state["count"] += 1
                self.predicates_read += 1
                stats = self.layers[str(level)]
                stats["tlp_generated"] += 1
                stats["tlp_generation_seconds"] += time.monotonic() - item_started
                batch.append((state["target"].name, state["count"], group, self.generator.tlp_mode))
                self.active.rotate(-1)
            self.complete = not self.active
            return batch
        finally:
            self.elapsed_seconds += time.monotonic() - started
            if self.metrics is not None:
                self.metrics.finish_phase("tlp_generation")
                self.metrics.start_phase("database_validation", self.depth)

    def __iter__(self):
        while self.active:
            batch = self._fill_batch()
            if not batch:
                return
            yield from batch

    def observe_result(self, file_name, validated):
        stats = self.layers[str(self.by_tlp_name[file_name]["level"])]
        stats["tested"] += 1
        stats["validated" if validated else "skipped"] += 1

    def observe_excluded(self, file_name):
        stats = self.layers[str(self.by_tlp_name[file_name]["level"])]
        stats["oracle_excluded_cases"] = stats.get("oracle_excluded_cases", 0) + 1

    def __exit__(self, exc_type, exc, tb):
        started = time.monotonic()
        try:
            self.stack.close()
        finally:
            for state in self.states:
                partial = Path(str(state["target"]) + ".partial")
                if exc_type is None:
                    os.replace(partial, state["target"])
                else:
                    partial.unlink(missing_ok=True)
            self.elapsed_seconds += time.monotonic() - started

    def report(self):
        return dict(success=True, round_complete=self.complete, phase="tlp_generation", error=None,
            predicates_read=self.predicates_read, output_files=self.output_files,
            source_output_files=self.source_output_files, elapsed_seconds=self.elapsed_seconds,
            output_sizes_bytes={s["target"].name: s["target"].stat().st_size for s in self.states},
            output_text_characters={s["target"].name: s["text_characters"] for s in self.states},
            source_text_characters={s["source"].name: s["source_characters"] for s in self.states},
            source_sizes_bytes={s["source"].name: s["source"].stat().st_size for s in self.states},
            stratified_layers=self.layers,
            test_case_storage=dict(format="gzip" if self.generator.compress_test_cases else "plain_sql",
                                   lossless_complete_output=True))


def run_stratified_tlp(*, predicate_files, layers, signature_spans_by_file, deadline,
                       metrics_recorder, depth, cumulative_generated, cumulative_tlp_base):
    from .sql_generator import TestCaseGenerator, OUTPUT_DIR
    from .bug_finder import TLPBugFinder, VALIDATION_BATCH_SIZE
    base = cumulative_tlp_base or {}
    started = time.monotonic()
    if metrics_recorder is not None:
        metrics_recorder.start_phase("tlp_generation", depth)
    generator = TestCaseGenerator()
    source = StratifiedTaskSource(generator, predicate_files, layers, deadline=deadline,
        batch_size=VALIDATION_BATCH_SIZE, output_dir=OUTPUT_DIR,
        metrics_recorder=metrics_recorder, depth=depth)
    with source:
        initialization_seconds = time.monotonic() - started
        if metrics_recorder is not None:
            metrics_recorder.finish_phase("tlp_generation")
            metrics_recorder.start_phase("database_validation", depth)
        def progress(current):
            if metrics_recorder is not None:
                metrics_recorder.update_oracle_exclusions(current, base)
                metrics_recorder.update_counts(generated=cumulative_generated, **{
                    target: int(base.get(key, 0)) + int(current.get(key, 0))
                    for target, key in (("tested", "cases_run"), ("validated", "cases_validated"),
                        ("skipped", "cases_skipped"), ("sql_errors", "sql_errors"),
                        ("query_timeouts", "query_timeouts"))})
                metrics_recorder.record(event="periodic", phase="database_validation", depth=depth)
        spans = {Path(target).name: (signature_spans_by_file or {}).get(name, [])
                 for name, target in source.source_output_files.items()}
        validation_started = time.monotonic()
        finder = TLPBugFinder(deadline=deadline, input_files=source.output_files,
            predicate_source_files=source.source_output_files, signature_spans_by_file=spans,
            progress_callback=progress, stratified_task_source=source)
        scan = finder.run()
        validation_seconds = max(0.0, time.monotonic() - validation_started - source.elapsed_seconds)
        if metrics_recorder is not None:
            metrics_recorder.finish_phase("database_validation")
            metrics_recorder.start_phase("tlp_generation", depth)
    if metrics_recorder is not None:
        metrics_recorder.finish_phase("tlp_generation")
    source.elapsed_seconds += initialization_seconds
    generation = source.report()
    scan["elapsed_seconds"] = validation_seconds
    if "oracle_excluded_cases" in scan:
        scan["elapsed_seconds_raw"] = validation_seconds
        validation_seconds = max(0.0, validation_seconds - scan["oracle_excluded_wall_seconds"])
        scan["elapsed_seconds"] = validation_seconds
        scan["cases_per_second"] = scan["cases_run"] / validation_seconds if validation_seconds > 0 else None
    for info in source.layers.values():
        if "oracle_excluded_cases" in scan:
            excluded = info.setdefault("oracle_excluded_cases", 0)
            for key in ("generated", "tlp_generated"):
                info[key + "_raw"] = info[key]
                info[key] = max(0, info[key] - excluded)
        n = info["tested"]
        info["success_rate"] = info["validated"] / n if n else None
        print(f"[Stratified TLP] L{info['dag_level']}: generated={info['generated']}, "
              f"tlp={info['tlp_generated']}, tested={n}, validated={info['validated']}")
    write_status_json(generation, OUTPUT_DIR / "stage3_tlp_generation_status.json")
    return dict(success=bool(scan.get("success")), round_complete=bool(scan.get("round_complete")) and source.complete,
        stop_phase=None if source.complete and scan.get("round_complete") else "tlp_scan",
        phase_timings=dict(tlp_generation_seconds=source.elapsed_seconds,
                           database_validation_seconds=validation_seconds),
        tlp_generation=generation, tlp_scan=scan, stratified_layers=source.layers,
        tlp={key: scan.get(key, 0) for key in ("cases_loaded", "cases_run", "cases_validated",
             "cases_skipped", "sql_errors", "query_timeouts", "bugs") +
             (("oracle_excluded_cases", "oracle_excluded_wall_seconds", "oracle_fault_path",
               "oracle_worker_restarts", "cases_loaded_raw", "success_rate", "cases_per_second")
              if "oracle_excluded_cases" in scan else ())},
        tlp_signature_feedback=scan.get("signature_feedback", {}),
        tlp_signature_feedback_covered_cases=scan.get("signature_feedback_covered_cases", 0))
