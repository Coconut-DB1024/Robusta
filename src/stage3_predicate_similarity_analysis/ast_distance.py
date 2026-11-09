from __future__ import annotations

from collections import Counter
from functools import lru_cache


_MAX_DP_CELLS = 65_536
_NODE_RECORD_SIZE = 18  
_NATIVE_CHECKED = False
_NATIVE = None
_NATIVE_ERROR = None


RELABEL_COST = 2


@lru_cache(maxsize=4)
def apted_config():
    from apted import Config

    class LabelReplacementConfig(Config):
        def rename(self, node1, node2):
            return RELABEL_COST * int(node1.name != node2.name)

    return LabelReplacementConfig()


def apted_calculator(left_tree, right_tree):
    from apted import APTED
    return APTED(left_tree, right_tree, config=apted_config())


def _native_engine():
    global _NATIVE_CHECKED, _NATIVE, _NATIVE_ERROR
    if not _NATIVE_CHECKED:
        _NATIVE_CHECKED = True
        try:
            from . import ast_native

            ast_native.warmup()
            _NATIVE = ast_native
        except Exception as exc:
            _NATIVE_ERROR = f"{type(exc).__name__}: {exc}"
    return _NATIVE


def distance_backend_info():
    engine = _native_engine()
    return {
        'engine': ('Numba exact Zhang-Shasha; bounded APTED fallback' if engine
                   else 'Python exact Zhang-Shasha; APTED fallback'),
        'accelerated': engine is not None,
        'versions': engine.versions() if engine else {},
        'fallback_reason': _NATIVE_ERROR,
        'edit_costs': {
            'insert': 1, 'delete': 1,
            'relabel': RELABEL_COST,
        },
        'apted_strategy': 'adaptive paths',
        'tree_representation': 'semantic-predicate-ast-v3',
    }


class PreparedTree:
    __slots__ = ("encoded", "labels", "leftmost", "keyroots", "histogram",
                 "forest_work", "_native_arrays")

    def __init__(self, encoded: bytes):
        self.encoded = encoded
        labels = [b""]  
        leftmost = [0]
        cursor = 0

        def visit():
            nonlocal cursor
            if cursor + _NODE_RECORD_SIZE > len(encoded):
                raise ValueError("corrupt serialized AST record")
            label = encoded[cursor:cursor + 16]
            child_count = int.from_bytes(encoded[cursor + 16:cursor + 18], "little")
            cursor += _NODE_RECORD_SIZE
            first_leaf = len(labels)
            for _ in range(child_count):
                visit()
            labels.append(label)
            leftmost.append(first_leaf)

        visit()
        if cursor != len(encoded):
            raise ValueError("serialized AST record has trailing data")
        self.labels = tuple(labels)
        self.leftmost = tuple(leftmost)
        last_root = {leaf: i for i, leaf in enumerate(leftmost) if i}
        self.keyroots = tuple(sorted(last_root.values()))
        self.histogram = dict(Counter(labels[1:]))
        self.forest_work = sum(i - leftmost[i] + 1 for i in self.keyroots)
        self._native_arrays = None

    @property
    def size(self):
        return len(self.labels) - 1


def distance_lower_bound(left: PreparedTree, right: PreparedTree) -> int:
    a, b = left.histogram, right.histogram
    if len(a) > len(b):
        a, b = b, a
    common = sum(min(count, b.get(label, 0)) for label, count in a.items())
    return left.size + right.size - 2 * common


def exact_distance(left: PreparedTree, right: PreparedTree,
                   lower_bound: int | None = None) -> int:
    if left.encoded == right.encoded:
        return 0
    bound = distance_lower_bound(left, right) if lower_bound is None else lower_bound

    if left.leftmost == right.leftmost:
        upper = RELABEL_COST * sum(a != b for a, b in zip(left.labels, right.labels))
        if upper == bound:
            return upper
    engine = _native_engine()
    if engine is not None:
        if ((left.size + 1) * (right.size + 1) > engine.MAX_MATRIX_CELLS
                or left.forest_work * right.forest_work > engine.MAX_FOREST_WORK):
            return _apted_distance(left, right)
        if left._native_arrays is None:
            left._native_arrays = engine.prepare(left)
        if right._native_arrays is None:
            right._native_arrays = engine.prepare(right)
        return int(engine.forest_distance(*left._native_arrays, *right._native_arrays))
    if (left.size + 1) * (right.size + 1) > _MAX_DP_CELLS:
        return _apted_distance(left, right)
    return _forest_distance(left, right)


def _forest_distance(left: PreparedTree, right: PreparedTree) -> int:
    n, m = left.size, right.size
    a, b = left.labels, right.labels
    la, lb = left.leftmost, right.leftmost
    subtree = [[0] * (m + 1) for _ in range(n + 1)]
    forest = [[0] * (m + 1) for _ in range(n + 1)]
    for i in left.keyroots:
        i0 = la[i] - 1
        for j in right.keyroots:
            j0 = lb[j] - 1
            initial = forest[i0]
            for y in range(j0, j + 1):
                initial[y] = y - j0
            for x in range(i0 + 1, i + 1):
                row, prev = forest[x], forest[x - 1]
                row[j0] = x - i0
                tx = subtree[x]
                lx0 = la[x] - 1
                prefix = forest[lx0]
                is_root = lx0 == i0
                ax = a[x]
                for y in range(j0 + 1, j + 1):
                    delete = prev[y] + 1
                    insert = row[y - 1] + 1
                    value = delete if delete < insert else insert
                    if is_root and lb[y] - 1 == j0:
                        cost = RELABEL_COST * (ax != b[y])
                        rename = prev[y - 1] + cost
                        if rename < value:
                            value = rename
                        tx[y] = value
                    else:
                        join = prefix[lb[y] - 1] + tx[y]
                        if join < value:
                            value = join
                    row[y] = value
    return subtree[n][m]


def _apted_distance(left: PreparedTree, right: PreparedTree) -> int:
    from .ast_similarity import _load_dependencies

    _sqlglot, APTED, Tree = _load_dependencies()

    def decode(encoded):
        cursor = 0

        def visit():
            nonlocal cursor
            label = encoded[cursor:cursor + 16]
            count = int.from_bytes(encoded[cursor + 16:cursor + 18], "little")
            cursor += _NODE_RECORD_SIZE
            return Tree(label, *(visit() for _ in range(count)))

        return visit()

    calculator = apted_calculator(decode(left.encoded), decode(right.encoded))
    result = calculator.compute_edit_distance()
    return int(result)
