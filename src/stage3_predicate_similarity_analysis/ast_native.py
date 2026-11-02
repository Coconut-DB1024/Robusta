"""Compiled exact ordered TED with unit insert/delete and fixed relabel cost 2.

Imported lazily by AST mode only. No fastmath, approximate scores, reduced labels,
parallel candidate scheduling, or persistent compilation files are involved.
"""
from __future__ import annotations


import numpy as np
import numba
from numba import njit
from .ast_distance import RELABEL_COST


# At most two int32 matrices (8 MiB combined). The operation budget also avoids
# sending extremely bushy trees into an unsuitable Zhang--Shasha decomposition.
# These limits select an exact implementation; they never drop a candidate.
MAX_MATRIX_CELLS = 1_048_576
MAX_FOREST_WORK = 32_000_000


@njit(cache=False, nogil=True)
def forest_distance(a, la, ka, b, lb, kb):
    n, m = len(la) - 1, len(lb) - 1
    subtree = np.zeros((n + 1, m + 1), dtype=np.int32)
    forest = np.zeros((n + 1, m + 1), dtype=np.int32)
    for i in ka:
        i0 = la[i] - 1
        for j in kb:
            j0 = lb[j] - 1
            for y in range(j0, j + 1):
                forest[i0, y] = y - j0
            for x in range(i0 + 1, i + 1):
                forest[x, j0] = x - i0
                lx0 = la[x] - 1
                for y in range(j0 + 1, j + 1):
                    delete = forest[x - 1, y] + 1
                    insert = forest[x, y - 1] + 1
                    value = delete if delete < insert else insert
                    if lx0 == i0 and lb[y] - 1 == j0:
                        # Compare BOTH halves of the original 128-bit label.
                        cost = RELABEL_COST * int(a[x, 0] != b[y, 0] or a[x, 1] != b[y, 1])
                        rename = forest[x - 1, y - 1] + cost
                        if rename < value:
                            value = rename
                        subtree[x, y] = value
                    else:
                        join = forest[lx0, lb[y] - 1] + subtree[x, y]
                        if join < value:
                            value = join
                    forest[x, y] = value
    return subtree[n, m]


def prepare(tree):
    """Arrays live in the existing bounded tree cache, not a global tree list."""
    # Bytes are compared unchanged, without a smaller hash or label interning.
    labels = np.frombuffer(b'\0' * 16 + b''.join(tree.labels[1:]), dtype=np.uint64).reshape(-1, 2)
    arrays = (labels, np.asarray(tree.leftmost, dtype=np.int32),
              np.asarray(tree.keyroots, dtype=np.int32))
    return arrays


def warmup():
    """Compile once per process, including tiny result checks before selecting it."""
    if numba.config.DISABLE_JIT:
        raise RuntimeError('Numba JIT is disabled by the environment')
    a = np.frombuffer(b'\0' * 32, dtype=np.uint64).reshape(-1, 2)
    b = np.frombuffer(b'\0' * 16 + b'\0' * 8 + b'\1' * 8, dtype=np.uint64).reshape(-1, 2)
    leaves = np.asarray([0, 1], dtype=np.int32)
    roots = np.asarray([1], dtype=np.int32)
    if (forest_distance(a, leaves, roots, a, leaves, roots) != 0
            or forest_distance(a, leaves, roots, b, leaves, roots) != RELABEL_COST):
        raise RuntimeError('compiled AST distance self-check failed')


def versions():
    return {'numba': numba.__version__, 'numpy': np.__version__}
