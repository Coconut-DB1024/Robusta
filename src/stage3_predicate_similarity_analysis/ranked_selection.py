from __future__ import annotations

import sqlite3
from fractions import Fraction
from functools import cmp_to_key
from pathlib import Path


def skip_ratio(value) -> Fraction:
    try:
        ratio = Fraction(str(value))
    except (ValueError, ZeroDivisionError) as exc:
        raise ValueError("top_skip_fraction must be a number in [0, 1)") from exc
    if not 0 <= ratio < 1:
        raise ValueError("top_skip_fraction must be in [0, 1)")
    return ratio


def selected_rank(count: int, ratio: Fraction) -> int:
    return count * ratio.numerator // ratio.denominator + 1 if count else 0


def _compare_rows(a, b):
    delta = a[0] * b[1] - b[0] * a[1]
    if delta:
        return -1 if delta > 0 else 1
    return (a[2] > b[2]) - (a[2] < b[2])


def _compare_score_text(a, b):
    ai, au = map(int, a.split('/'))
    bi, bu = map(int, b.split('/'))
    delta = ai * bu - bi * au
    return (delta > 0) - (delta < 0)


class RankedCandidates:
    def __init__(self, fraction, disk_path: Path | None = None):
        self.ratio = skip_ratio(fraction)
        self.pending = {}
        self.connection = None
        if disk_path is not None:
            try:
                self.connection = sqlite3.connect(disk_path)
                self.connection.create_collation('exact_score', _compare_score_text)
                self.connection.execute('PRAGMA journal_mode=OFF')
                self.connection.execute('PRAGMA synchronous=OFF')
                self.connection.execute('PRAGMA cache_size=-8192')
                self.connection.execute('PRAGMA temp_store=FILE')
                self.connection.execute('CREATE TABLE scores(owner INTEGER, candidate INTEGER, score TEXT, source INTEGER)')
                self.connection.execute('CREATE INDEX scores_owner ON scores(owner)')
            except Exception:
                self.close()
                raise

    def add(self, owner: int, candidate: int, intersection: int, union: int, source: int):
        if self.connection is None:
            self.pending.setdefault(owner, []).append((intersection, union, candidate, source))
        else:
            self.connection.execute('INSERT INTO scores VALUES (?, ?, ?, ?)',
                                    (owner, candidate, f'{intersection}/{union}', source))

    def finish(self, owner: int, count: int):
        rank = selected_rank(count, self.ratio)
        if self.connection is None:
            rows = self.pending.pop(owner, [])
            if len(rows) != count:
                raise ValueError('ranked candidate count mismatch')
            if not count:
                return None
            rows.sort(key=cmp_to_key(_compare_rows))
            return rows[rank - 1]
        if not count:
            return None
        row = self.connection.execute(
            'SELECT score, candidate, source FROM scores WHERE owner=? '
            'ORDER BY score COLLATE exact_score DESC, candidate ASC LIMIT 1 OFFSET ?',
            (owner, rank - 1),
        ).fetchone()
        if row is None:
            raise ValueError('ranked candidate count mismatch')
        self.connection.execute('DELETE FROM scores WHERE owner=?', (owner,))
        if (owner + 1) % 2000 == 0:
            self.connection.commit()
        intersection, union = map(int, row[0].split('/'))
        return intersection, union, int(row[1]), int(row[2])

    def close(self):
        self.pending.clear()
        if self.connection is not None:
            self.connection.close()
            self.connection = None
