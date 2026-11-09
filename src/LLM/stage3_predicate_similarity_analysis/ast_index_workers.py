from __future__ import annotations

import multiprocessing
import sys
from collections import OrderedDict, deque
from concurrent.futures import ProcessPoolExecutor
from typing import Iterable, Iterator, Optional


_CACHE_BYTES = 8 * 1024 * 1024
_CACHE_ENTRIES = 8_192
_WORKER_STATE = None


class _ParserState:
    def __init__(self, dialect: str):
        from .ast_similarity import _IndexParser

        self.parser = _IndexParser(dialect)
        self.cache = OrderedDict()
        self.cache_bytes = 0

    def parse(self, predicates: list[str]):
        records = []
        for predicate in predicates:
            try:
                cached = self.cache.get(predicate)
                if cached is not None:
                    self.cache.move_to_end(predicate)
                    encoded = cached[0]
                else:
                    encoded = self.parser.serialize(predicate)

                    cost = 4 * len(predicate) + len(encoded) + 256
                    if cost <= _CACHE_BYTES:
                        self.cache[predicate] = (encoded, cost)
                        self.cache_bytes += cost
                        while (self.cache_bytes > _CACHE_BYTES
                               or len(self.cache) > _CACHE_ENTRIES):
                            _, (_, old_cost) = self.cache.popitem(last=False)
                            self.cache_bytes -= old_cost
                records.append((encoded, None))
            except Exception as exc:
                records.append((None, type(exc).__name__))
        return records

    def close(self):
        self.parser.clear()
        self.cache.clear()
        self.cache_bytes = 0


def _initialize_worker(dialect: str) -> None:
    global _WORKER_STATE

    sys.dont_write_bytecode = True
    _WORKER_STATE = _ParserState(dialect)


def _parse_worker_batch(predicates: list[str]):
    if _WORKER_STATE is None:
        raise RuntimeError("AST parser worker was not initialized")
    return _WORKER_STATE.parse(predicates)


class OrderedAstParser:
    def __init__(
        self,
        dialect: str,
        workers: int = 1,
        batch_size: int = 256,
    ):
        self.dialect = str(dialect)
        self.workers = int(workers)
        self.batch_size = int(batch_size)
        if self.workers < 1 or self.batch_size < 1:
            raise ValueError("AST workers and batch_size must both be positive")
        self._pool: Optional[ProcessPoolExecutor] = None
        self._local_state: Optional[_ParserState] = None
        self._pending = deque()
        self._entered = False
        self._closed = False
        self._iterating = False
        self._previous_bytecode_setting = None

    def __enter__(self):
        if self._entered or self._closed:
            raise RuntimeError("AST batch parser cannot be entered again")

        self._previous_bytecode_setting = sys.dont_write_bytecode
        sys.dont_write_bytecode = True
        try:
            if self.workers == 1:
                self._local_state = _ParserState(self.dialect)
            else:
                self._pool = ProcessPoolExecutor(
                    max_workers=self.workers,
                    mp_context=multiprocessing.get_context("spawn"),
                    initializer=_initialize_worker,
                    initargs=(self.dialect,),
                )
            self._entered = True
            return self
        except BaseException:
            self.close()
            raise

    def parse_batches(self, batches: Iterable[list[str]]) -> Iterator[list[tuple]]:
        if not self._entered or self._closed:
            raise RuntimeError("Use OrderedAstParser inside its context manager")
        if self._iterating:
            raise RuntimeError("AST batch parser only supports one active iterator")
        self._iterating = True
        source = iter(batches)

        def next_batch():
            batch = next(source)
            if not isinstance(batch, (list, tuple)):
                raise TypeError("An AST input batch must be a list or tuple")
            if len(batch) > self.batch_size:
                raise ValueError("AST input batch exceeds configured batch_size")
            return batch

        try:
            if self._local_state is not None:
                while True:
                    try:
                        batch = next_batch()
                    except StopIteration:
                        break
                    records = self._local_state.parse(batch)
                    yield records
                return

            exhausted = False
            limit = 2 * self.workers
            while not exhausted or self._pending:
                while not exhausted and len(self._pending) < limit:
                    try:
                        batch = next_batch()
                    except StopIteration:
                        exhausted = True
                        break
                    self._pending.append(self._pool.submit(_parse_worker_batch, batch))
                if self._pending:
                    records = self._pending.popleft().result()
                    yield records
        finally:
            for future in self._pending:
                future.cancel()
            self._pending.clear()
            self._iterating = False

    map_batches = parse_batches

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            for future in self._pending:
                future.cancel()
            self._pending.clear()
            if self._pool is not None:
                self._pool.shutdown(wait=True, cancel_futures=True)
                self._pool = None
            if self._local_state is not None:
                self._local_state.close()
                self._local_state = None
        finally:
            if self._previous_bytecode_setting is not None:
                sys.dont_write_bytecode = self._previous_bytecode_setting
                self._previous_bytecode_setting = None

    def __exit__(self, exc_type, exc, traceback):
        self.close()
        return False
