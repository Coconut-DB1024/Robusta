from __future__ import annotations

import faulthandler
import multiprocessing
from multiprocessing.connection import wait
import time
from collections import deque


PHASES = ("starting", "connect", "idle", "case", "total", "true", "false",
          "null", "repair", "columns", "describe", "return_result",
          "describe_set_timeout", "describe_restore_timeout")


class OracleWorkerFailure(RuntimeError):
    def __init__(self, details):
        self.details = details
        super().__init__(
            f"Oracle worker {details.get('pid')} {details['kind']}; "
            f"case={details.get('case_idx')}, phase={details.get('phase')}, "
            f"exitcode={details.get('exitcode')}"
        )


class OracleValidationDeadline(RuntimeError):
    pass


class WorkerProgress:
    SQL_BYTES = 32768

    def __init__(self, context):
        self.state = context.RawArray("q", 7)
        self.sql = context.RawArray("c", self.SQL_BYTES)
        self.mark("starting")

    def mark(self, phase, sql="", *, case_idx=None, attempt=None):
        data = str(sql).encode("utf-8")[:self.SQL_BYTES]
        self.state[0] += 1
        if case_idx is not None:
            self.state[1] = int(case_idx)
            self.state[5] = time.monotonic_ns()
            self.state[3] = 0
        self.state[2] = PHASES.index(phase)
        if attempt is not None:
            self.state[3] = int(attempt)
        self.state[4] = time.monotonic_ns()
        self.sql[:len(data)] = data
        self.state[6] = len(data)
        self.state[0] += 1

    def snapshot(self):
        before = self.state[0]
        state = list(self.state)
        sql = bytes(self.sql[:state[6]]).decode("utf-8", "replace")
        stable = before == self.state[0] and before % 2 == 0
        now = time.monotonic_ns()
        return {
            "case_idx": state[1], "phase": PHASES[state[2]],
            "attempt": state[3], "sql": sql,
            "sql_truncated": state[6] == self.SQL_BYTES,
            "snapshot_stable": stable,
            "operation_seconds": (now - state[4]) / 1e9,
            "case_seconds": (now - state[5]) / 1e9 if state[5] else 0.0,
        }


def _worker_main(pipe, initializer, task_function, progress):
    try:
        faulthandler.enable()
        progress.mark("connect")
        initializer(progress)
        while True:
            progress.mark("idle")
            pipe.send(("ready", None))
            chunk = pipe.recv()
            if chunk is None:
                return
            for task in chunk:
                started = time.monotonic()

                pipe.send(("started", (task[0], task[1], started)))
                progress.mark("case", case_idx=task[1])
                result = task_function(task)
                progress.mark("return_result")
                pipe.send(("result", (result, started, time.monotonic())))
    except EOFError:
        pass
    except BaseException as exc:
        try:
            pipe.send(("error", type(exc).__name__))
        except (BrokenPipeError, EOFError, OSError):
            pass
    finally:
        pipe.close()


def _merged_intervals(intervals):
    merged = []
    for start, end in sorted(intervals):
        if end <= start:
            continue
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(end, merged[-1][1])
        else:
            merged.append([start, end])
    return merged


def excluded_wall_time(fault_intervals, productive_intervals):
    faults = _merged_intervals(fault_intervals)
    useful = _merged_intervals(productive_intervals)
    total = sum(end - start for start, end in faults)
    i = j = 0
    while i < len(faults) and j < len(useful):
        a, b = faults[i]
        c, d = useful[j]
        total -= max(0.0, min(b, d) - max(a, c))
        if b <= d:
            i += 1
        else:
            j += 1
    return max(0.0, total)


class OracleValidationPool:
    def __init__(self, processes, initializer, task_function, *, query_timeout=10,
                 deadline=None, on_poll=None, operation_timeout=None,
                 case_timeout=None, startup_timeout=60.0, max_startup_failures=3):
        self.deadline = deadline
        self.on_poll = on_poll
        self.operation_timeout = (max(30.0, query_timeout * 3.0)
                                  if operation_timeout is None else operation_timeout)
        self.case_timeout = (max(180.0, query_timeout * 15.0)
                             if case_timeout is None else case_timeout)
        self.startup_timeout = startup_timeout
        self.max_startup_failures = max(1, int(max_startup_failures))
        self.initializer = initializer
        self.task_function = task_function
        self.context = multiprocessing.get_context("spawn")
        self.excluded_wall_seconds = 0.0
        self.restarts = 0
        self.workers = []
        self.closed = False
        try:
            for _ in range(processes):
                worker = {"unattributed_failures": 0, "replacement": False}
                self.workers.append(worker)
                self._spawn(worker)
        except BaseException:
            self.terminate()
            raise

    def _spawn(self, worker):
        parent, child = self.context.Pipe()
        progress = WorkerProgress(self.context)
        process = self.context.Process(
            target=_worker_main,
            args=(child, self.initializer, self.task_function, progress), daemon=True,
        )
        worker.update(process=process, pipe=parent, progress=progress, ready=False,
                      pending=[], current=None, started=time.monotonic())
        try:
            process.start()
        finally:
            child.close()

    def _failure(self, worker, kind, **extra):
        snapshot = worker["progress"].snapshot()
        process = worker["process"]
        current = worker.get("current")
        details = {**snapshot, "kind": kind, "pid": process.pid,
                   "exitcode": process.exitcode,
                   "file": current[0] if current else None,
                   "pending_case_indices": [task[1] for task in worker["pending"]],
                   **extra}

        if current is not None:
            details.update(file=current[0], case_idx=current[1], tlp_mode=current[3])
        else:
            details.update(file=None, case_idx=None)
        return OracleWorkerFailure(details)

    def _check_deadline(self):
        if self.deadline is not None and time.monotonic() >= self.deadline:
            raise OracleValidationDeadline()

    def _find_fault(self):
        for worker in self.workers:
            if worker.get("process") is None:
                continue

            if worker["pipe"].poll():
                continue
            if worker["process"].exitcode is not None:
                return worker, self._failure(worker, "unexpected_exit")
            snapshot = worker["progress"].snapshot()

            elapsed = time.monotonic() - worker["started"]
            if not worker["ready"] and not worker["pending"]:
                if elapsed > self.startup_timeout:
                    return worker, self._failure(worker, "startup_timeout")
            if worker["pending"]:
                if elapsed > len(worker["pending"]) * self.case_timeout + self.startup_timeout:
                    return worker, self._failure(worker, "dispatch_timeout")
                if snapshot["snapshot_stable"] and snapshot["phase"] != "idle":
                    if snapshot["operation_seconds"] > self.operation_timeout:
                        return worker, self._failure(worker, "operation_timeout")
                    if snapshot["case_seconds"] > self.case_timeout:
                        return worker, self._failure(worker, "case_timeout")
        return None

    def _recover(self, worker, failure, queue, fault_intervals):
        details = failure.details
        current = worker.get("current")
        started = (worker.get("case_started", worker["started"])
                   if current is not None else worker["started"])
        remaining_tasks = list(worker["pending"])
        if current is not None:
            remaining_tasks = [task for task in remaining_tasks
                               if (task[0], task[1]) != (current[0], current[1])]
            worker["unattributed_failures"] = 0
        else:
            worker["unattributed_failures"] += 1
        if details["kind"] in {"connection_closed", "worker_exception"}:
            worker["process"].join(0.05)
        exitcode = self._retire(worker)
        if details.get("exitcode") is None:
            details["exitcode"] = exitcode
        fault_intervals.append((started, time.monotonic()))
        queue.extendleft(reversed(remaining_tasks))
        worker.update(pending=[], current=None, ready=False, replacement=True,
                      recovery_started=time.monotonic())
        if current is None and worker["unattributed_failures"] >= self.max_startup_failures:
            worker["disabled"] = True
            details["excluded_from_statistics"] = False
            if all(slot.get("disabled", False) for slot in self.workers):
                details["kind"] = "repeated_unattributed_worker_failure"
                raise OracleWorkerFailure(details)
            return "worker_unavailable", details
        details["excluded_from_statistics"] = current is not None
        return ("excluded" if current is not None else "worker_restarted"), details

    def imap_unordered(self, task_function, tasks, chunksize=1):
        batch_started = time.monotonic()
        pending = deque(tasks)
        remaining = len(tasks)
        useful, faults = [], []  
        try:
            while remaining:
                self._check_deadline()
                if self.on_poll is not None:
                    self.on_poll()
                fault = self._find_fault()
                if fault is not None:
                    status, info = self._recover(*fault, pending, faults)
                    remaining -= int(status == "excluded")
                    yield status, info
                    continue
                for worker in self.workers:
                    if pending and worker.get("process") is None and not worker.get("disabled"):
                        self._spawn(worker)
                        self.restarts += 1
                    if pending and worker["ready"] and worker["process"].exitcode is None:
                        chunk = [pending.popleft() for _ in range(min(len(pending), max(1, chunksize)))]
                        worker.update(pending=chunk, ready=False, started=time.monotonic())
                        try:
                            worker["pipe"].send(chunk)
                        except (BrokenPipeError, EOFError, OSError):
                            fault = (worker, self._failure(worker, "dispatch_failed"))
                            break
                if fault is not None:
                    status, info = self._recover(*fault, pending, faults)
                    remaining -= int(status == "excluded")
                    yield status, info
                    continue
                active = [w for w in self.workers if w.get("process") is not None]
                available = wait([w["pipe"] for w in active] +
                                 [w["process"].sentinel for w in active], timeout=0.25)
                for worker in active:
                    if worker["pipe"] not in available:
                        continue
                    try:
                        kind, payload = worker["pipe"].recv()
                    except (EOFError, OSError):
                        kind, payload = "closed", None
                    if kind in ("closed", "error"):
                        failure = self._failure(worker, "connection_closed" if kind == "closed"
                                                else "worker_exception", exception_type=payload)
                        status, info = self._recover(worker, failure, pending, faults)
                        remaining -= int(status == "excluded")
                        yield status, info
                    elif kind == "ready":
                        if worker["replacement"]:
                            faults.append((worker["recovery_started"], time.monotonic()))
                            worker["replacement"] = False
                        worker.update(ready=True, pending=[], current=None)
                    elif kind == "started":
                        name, idx, started = payload
                        current = worker["pending"][0]
                        if (name, idx) != (current[0], current[1]):
                            raise self._failure(worker, "invalid_worker_case_identity")
                        worker.update(current=current, case_started=started)
                    elif kind == "result":
                        result, started, finished = payload
                        useful.append((started, finished))
                        remaining -= 1
                        worker["pending"].pop(0)
                        worker.update(current=None, started=time.monotonic(), unattributed_failures=0)
                        yield result
        finally:
            self.excluded_wall_seconds += excluded_wall_time(
                [(max(start, batch_started), end) for start, end in faults], useful)

    def _retire(self, worker):
        process = worker.get("process")
        if process is None:
            return
        if process.is_alive():
            process.terminate()
        process.join(2.0)
        if process.is_alive():
            process.kill()
            process.join(2.0)
        if process.is_alive():
            raise self._failure(worker, "worker_could_not_be_stopped")
        worker["pipe"].close()
        exitcode = process.exitcode
        process.close()
        worker["process"] = None
        return exitcode

    def _join_until(self, seconds):
        deadline = time.monotonic() + seconds
        for worker in self.workers:
            if worker.get("process") is not None and worker["process"].pid is not None:
                worker["process"].join(max(0, deadline - time.monotonic()))

    def _shutdown(self, abort):
        if self.closed:
            return
        self.closed = True
        if not abort:
            for worker in self.workers:
                if worker.get("process") is None:
                    continue
                try:
                    worker["pipe"].send(None)
                except (BrokenPipeError, EOFError, OSError):
                    pass
            self._join_until(2.0)
        for worker in self.workers:
            if worker.get("process") is not None and worker["process"].is_alive():
                worker["process"].terminate()
        self._join_until(2.0)
        for worker in self.workers:
            if worker.get("process") is not None and worker["process"].is_alive():
                worker["process"].kill()
        self._join_until(2.0)
        for worker in self.workers:
            if worker.get("process") is None:
                continue
            worker["pipe"].close()
            if not worker["process"].is_alive():
                worker["process"].close()

    def close(self):
        self._shutdown(abort=False)

    def terminate(self):
        self._shutdown(abort=True)

    def join(self):
        pass
