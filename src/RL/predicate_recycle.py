from __future__ import annotations

"""
predicate_recycle.py

阶段 6A / 6B：Correct 谓词回炉候选筛选与安全应用工具。

默认情况下，本模块只负责“筛选哪些 Correct 谓词值得回炉到 predicate.txt”，
只生成 recycle_candidates.json，不直接修改 predicate.txt。

只有显式传入 --apply 时，才会进入阶段 6B：
1. 备份 predicate.txt；
2. 去重追加候选谓词；
3. 写入 recycle_apply_log.json；
4. 标记 predicate_contribution.json 中的 added_to_predicate=True。

本模块不会触发第一阶段训练，不重建 HNSW，不启动 RL。

输入：
1. output/graph_runtime_stats/predicate_contribution.json
   阶段 3 记录的谓词贡献统计。

2. output/graph_runtime_stats/operator_stats.json
   阶段 3 记录的算子统计。

3. output/predicate.txt
   第一阶段当前使用的谓词训练文本。

输出：
1. output/graph_runtime_stats/recycle_candidates.json
   按 score 排序后的回炉候选列表。

2. output/graph_runtime_stats/recycle_apply_log.json
   仅在 --apply 或 --apply --dry-run 时生成，用于记录本次应用结果。

设计原则：
- 只使用系统真实运行统计；
- 不维护人工候选边全集；
- 不使用数据库弱类型规则；
- 不修改 Correct_Collection；
- 默认不修改 predicate.txt，只有显式 --apply 时才安全追加；
- 不影响现有 main.py / pre_load / RL 训练流程。
"""

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
import argparse
import json
import re
import shutil
import time

from RL.config import PROJECT_ROOT, OUTPUT_DIR, RL_ENV_CONFIG, RECYCLE_CONFIG, get_database_data_path, load_db_operator_profile, load_db_coverage_profile
from RL.graph_runtime_stats import get_default_runtime_stats


DEFAULT_PREDICATE_TXT_PATH = OUTPUT_DIR / "predicate.txt"
DEFAULT_CORRECT_COLLECTION_PATH = get_database_data_path("Correct_Collection.txt")
DEFAULT_RECYCLE_OUTPUT_PATH = OUTPUT_DIR / "graph_runtime_stats" / "recycle_candidates.json"

# 阶段 6B：真正应用回炉候选时使用的日志与备份路径。
DEFAULT_RECYCLE_APPLY_LOG_PATH = OUTPUT_DIR / "graph_runtime_stats" / "recycle_apply_log.json"
DEFAULT_PREDICATE_BACKUP_DIR = OUTPUT_DIR / "graph_runtime_stats" / "predicate_backups"


@dataclass
class RecycleCandidate:
    """
    一个可回炉谓词候选。

    predicate:
        原始谓词文本，例如 col(age) > int(10)。

    score:
        回炉优先级分数。越高越值得优先加入 predicate.txt。

    reasons:
        为什么它被选中，例如 has_new_edges / has_new_operators / has_low_freq_operators。

    注意：
        这里只保存候选信息，不直接写 predicate.txt。
    """
    predicate_key: str
    predicate: str
    score: float

    # 阶段 6A 补丁：候选类别。
    # strong_contribution：贡献过新边 / 新算子 / graph_new_edges；
    # low_freq_structure：包含低频算子；
    # diverse_structure：结构较丰富，适合作为 decoder 学习样本；
    # stable_success：DB 成功、parser 成功、结构正常，但未必贡献新边。
    category: str

    reasons: List[str]

    parse_success: bool
    success_count: int

    graph_new_edges: int
    graph_new_nodes: int
    fixed_base_structure: bool
    fixed_base_new_edges: int
    frontier_new_edges: int
    new_edge_count: int
    new_operator_count: int
    low_freq_operator_count: int

    edge_count: int
    operator_count: int

    first_seen_source: Optional[str]
    last_seen_source: Optional[str]
    first_seen_step: Optional[int]
    last_seen_step: Optional[int]

    normalized_predicate: str
    abstract_predicate: str

    new_edges: List[str]
    new_operators: List[str]
    operators: List[str]


@dataclass
class RecycleApplyResult:
    """
    阶段 6B：候选谓词应用结果。

    注意：
        这个结果会写入 recycle_apply_log.json，方便检查本次到底追加了哪些谓词。
    """
    applied_count: int
    skipped_duplicate_count: int
    skipped_invalid_count: int
    dry_run: bool

    predicate_txt_path: str
    backup_path: Optional[str]
    apply_log_path: str

    applied_predicate_keys: List[str]
    applied_predicates: List[str]
    skipped_duplicates: List[str]
    skipped_invalid_items: List[str]


def _safe_int(value: Any, default: int = 0) -> int:
    """安全转 int，避免 JSON 中的异常值影响筛选流程。"""
    try:
        return int(value)
    except Exception:
        return default


def _safe_float(value: Any, default: float = 0.0) -> float:
    """安全转 float。"""
    try:
        return float(value)
    except Exception:
        return default


def _timestamp_for_filename() -> str:
    """
    生成适合放进文件名的时间戳。

    用于 predicate.txt 备份文件名，避免多次 apply 时互相覆盖。
    """
    return time.strftime("%Y%m%d_%H%M%S", time.localtime())


def _load_json(path: Path, default: Any) -> Any:
    """安全读取 JSON。文件不存在或损坏时返回 default。"""
    path = Path(path)
    if not path.exists():
        return default

    try:
        if path.stat().st_size == 0:
            return default
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"[PredicateRecycle] WARN: JSON 读取失败: {path} | {e}")
        return default


def _write_json(path: Path, data: Any) -> None:
    """写 JSON，保证父目录存在。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def _read_text_lines(path: Path) -> List[str]:
    """读取文本文件的非空行。文件不存在时返回空列表。"""
    path = Path(path)
    if not path.exists():
        return []

    lines: List[str] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                lines.append(s)
    return lines


def _normalize_predicate_text(text: str) -> str:
    """
    用于去重的轻量规范化。

    注意：
        这里不做语义解析，不改变谓词内容，只压缩首尾空白。
    """
    return "" if text is None else str(text).strip()


def _operator_frequency(op_id: str, operator_stats: Dict[str, Any]) -> int:
    """
    计算某个算子的出现频率。

    使用 success_count + failure_count，和阶段 5 reward 中低频算子的判断保持一致。
    """
    item = operator_stats.get(op_id)
    if not isinstance(item, dict):
        return 0

    return (
        _safe_int(item.get("success_count", 0))
        + _safe_int(item.get("failure_count", 0))
    )


def _operator_profile() -> dict:
    return load_db_operator_profile()


def _target_sparse_operators() -> List[str]:
    ops = _operator_profile().get("target_sparse_operators", [])
    if ops:
        return [str(x) for x in ops]
    return [str(x) for x in getattr(RL_ENV_CONFIG, "target_sparse_operators", ())]


def _high_freq_operators() -> List[str]:
    ops = _operator_profile().get("high_freq_operators", [])
    if ops:
        return [str(x) for x in ops]
    return [str(x) for x in getattr(RL_ENV_CONFIG, "high_freq_operators", ())]


def _coverage_mode() -> str:
    return str(getattr(RL_ENV_CONFIG, "coverage_mode", "unrestricted")).lower()


def _coverage_profile() -> dict:
    return load_db_coverage_profile()


def _coverage_base_node_ids() -> Set[str]:
    return {str(x) for x in _coverage_profile().get("base_node_ids", [])}


def _active_target_sparse_operators() -> List[str]:
    if _coverage_mode() == "fixed_base":
        fixed_ops = _coverage_profile().get("fixed_base_target_sparse_operators", [])
        if fixed_ops:
            return [str(x) for x in fixed_ops]
    return _target_sparse_operators()


def _candidate_fixed_base_structure_from_item(item: Dict[str, Any]) -> bool:
    if "fixed_base_structure" in item:
        return bool(item.get("fixed_base_structure", False))

    base_nodes = _coverage_base_node_ids()
    node_ids = set(str(x) for x in item.get("node_ids", []) or [])
    if not base_nodes or not node_ids:
        return False
    return node_ids.issubset(base_nodes)


def _candidate_fixed_base_new_edges_from_item(item: Dict[str, Any]) -> int:
    return _safe_int(item.get("fixed_base_new_edges", 0))


def _candidate_frontier_new_edges_from_item(item: Dict[str, Any]) -> int:
    return _safe_int(item.get("frontier_new_edges", 0))


def _is_value_candidate(cand: RecycleCandidate) -> bool:
    mode = _coverage_mode()

    if cand.category == "strong_contribution":
        return True

    if cand.fixed_base_new_edges > 0:
        return True

    if (
        bool(getattr(RECYCLE_CONFIG, "recycle_frontier_new_edges_as_strong", True))
        and cand.frontier_new_edges > 0
    ):
        return True

    if mode != "fixed_base" and (
        cand.graph_new_edges > 0
        or cand.new_edge_count > 0
        or cand.new_operator_count > 0
    ):
        return True

    if (
        bool(getattr(RECYCLE_CONFIG, "recycle_low_freq_structure_in_value_channel", False))
        and cand.category == "low_freq_structure"
    ):
        return True

    return False


def _is_novel_structure_candidate(cand: RecycleCandidate) -> bool:
    if _is_value_candidate(cand):
        return False

    if cand.category == "low_freq_structure":
        return bool(getattr(RECYCLE_CONFIG, "recycle_low_freq_structure_in_novel_channel", True))

    return cand.category in {"diverse_structure", "stable_success", "valid_success"}


def _candidate_has_strong_signal(cand_or_fields: Any) -> bool:
    if isinstance(cand_or_fields, RecycleCandidate):
        return (
            cand_or_fields.graph_new_edges > 0
            or cand_or_fields.fixed_base_new_edges > 0
            or cand_or_fields.frontier_new_edges > 0
            or cand_or_fields.new_edge_count > 0
            or cand_or_fields.new_operator_count > 0
        )
    if not isinstance(cand_or_fields, dict):
        return False
    new_edges = cand_or_fields.get("new_edges", []) or []
    new_operators = cand_or_fields.get("new_operators", []) or []
    return (
        _safe_int(cand_or_fields.get("graph_new_edges", 0)) > 0
        or _safe_int(cand_or_fields.get("fixed_base_new_edges", 0)) > 0
        or _safe_int(cand_or_fields.get("frontier_new_edges", 0)) > 0
        or _safe_int(cand_or_fields.get("new_edge_count", 0)) > 0
        or (isinstance(new_edges, list) and len(new_edges) > 0)
        or _safe_int(cand_or_fields.get("new_operator_count", 0)) > 0
        or (isinstance(new_operators, list) and len(new_operators) > 0)
    )


def _operator_tokens(op: str) -> List[str]:
    raw = str(op or "").strip()
    base = raw.split("::", 1)[0]
    tokens = {raw.upper(), base.upper()}
    if base.endswith("()"):
        tokens.add(base[:-2].upper())
    return [t for t in tokens if t]


def _operator_matches_target_sparse(op: str) -> bool:
    op_tokens = set(_operator_tokens(op))
    for target in _active_target_sparse_operators():
        if op_tokens.intersection(_operator_tokens(str(target))):
            return True
    return False


def _candidate_has_sparse_target_operator(operators: List[str]) -> bool:
    return any(_operator_matches_target_sparse(str(op)) for op in operators or [])


def _is_high_freq_zero_signal(
    *,
    graph_new_edges: int,
    new_edges: List[str],
    new_operators: List[str],
    low_freq_ops: List[str],
    operators: List[str],
) -> bool:
    if graph_new_edges > 0 or len(new_edges) > 0 or len(new_operators) > 0:
        return False
    if low_freq_ops or _candidate_has_sparse_target_operator(operators):
        return False
    if not operators:
        return False
    high_freq = set(_high_freq_operators())
    return all(str(op) in high_freq for op in operators)


def _candidate_abstract_key(cand: RecycleCandidate) -> str:
    if cand.abstract_predicate:
        return str(cand.abstract_predicate)
    s = str(cand.predicate or "")
    s = re.sub(r"\bcol\s*\([^)]*\)", "col(X)", s, flags=re.IGNORECASE)
    s = re.sub(r"\bstring\s*\([^)]*\)", "string(S)", s, flags=re.IGNORECASE)
    s = re.sub(r"'(?:''|[^'])*'", "string(S)", s)
    s = re.sub(r"\b(?:int|float)\s*\([^)]*\)", "N", s, flags=re.IGNORECASE)
    s = re.sub(r"\b0x[0-9a-fA-F]+\b", "N", s)
    s = re.sub(r"(?<![\w.])-?\d+(?:\.\d+)?(?![\w.])", "N", s)
    return re.sub(r"\s+", " ", s).strip()


def _category_priority(category: str) -> int:
    """
    回炉候选类别优先级。

    分数 score 仍然是主排序依据；
    category_priority 只在分数接近时辅助排序。
    """
    priority = {
        "strong_contribution": 4,
        "low_freq_structure": 3,
        "diverse_structure": 2,
        "stable_success": 1,
        "valid_success": 0,
    }
    return priority.get(str(category), 0)


class PredicateRecycler:
    """
    回炉候选筛选与安全应用器。

    阶段 6A：
        只筛选候选，生成 recycle_candidates.json，不修改 predicate.txt。

    阶段 6B：
        只有显式调用 apply_candidates_to_predicate_txt() 或命令行传入 --apply 时，
        才会备份并追加写入 predicate.txt。
    """

    def __init__(
        self,
        predicate_txt_path: Optional[Path] = None,
        correct_collection_path: Optional[Path] = None,
        output_path: Optional[Path] = None,
        low_freq_threshold: Optional[int] = None,
    ) -> None:
        self.stats = get_default_runtime_stats()

        self.predicate_txt_path = Path(predicate_txt_path) if predicate_txt_path else DEFAULT_PREDICATE_TXT_PATH
        self.correct_collection_path = (
            Path(correct_collection_path) if correct_collection_path else DEFAULT_CORRECT_COLLECTION_PATH
        )
        self.output_path = Path(output_path) if output_path else DEFAULT_RECYCLE_OUTPUT_PATH

        # 阶段 6B：apply 日志与 predicate.txt 备份目录。
        self.apply_log_path = DEFAULT_RECYCLE_APPLY_LOG_PATH
        self.predicate_backup_dir = DEFAULT_PREDICATE_BACKUP_DIR

        self.low_freq_threshold = (
            int(low_freq_threshold)
            if low_freq_threshold is not None
            else int(RL_ENV_CONFIG.reward_low_freq_operator_threshold)
        )

    def load_existing_predicates(self) -> Set[str]:
        """
        读取当前 predicate.txt 中已有谓词，用于避免重复回炉。

        当前 pre_load/runner.py 会生成 output/predicate.txt，因此默认读取 output/predicate.txt。
        """
        existing = set()

        for line in _read_text_lines(self.predicate_txt_path):
            norm = _normalize_predicate_text(line)
            if norm:
                existing.add(norm)

        return existing

    def load_correct_collection(self) -> Set[str]:
        """
        读取 Correct_Collection。

        阶段 6A 默认不强制候选必须存在于 Correct_Collection，
        因为 predicate_contribution.json 本身就是由 SQL 成功谓词统计得来。
        这个函数先保留，后续 6B/6C 可用于额外校验。
        """
        correct = set()

        for line in _read_text_lines(self.correct_collection_path):
            norm = _normalize_predicate_text(line)
            if norm:
                correct.add(norm)

        return correct

    def _backup_predicate_txt(self) -> Optional[Path]:
        """
        备份当前 predicate.txt。

        返回：
            备份文件路径；如果 predicate.txt 当前不存在，则返回 None。

        说明：
            6B 会真正修改 predicate.txt，因此应用前必须备份。
        """
        if not self.predicate_txt_path.exists():
            return None

        self.predicate_backup_dir.mkdir(parents=True, exist_ok=True)

        backup_name = f"predicate_{_timestamp_for_filename()}.txt"
        backup_path = self.predicate_backup_dir / backup_name

        shutil.copy2(str(self.predicate_txt_path), str(backup_path))
        return backup_path

    def _mark_added_to_predicate(
        self,
        applied_predicate_keys: List[str],
        *,
        dry_run: bool,
    ) -> int:
        """
        将 predicate_contribution.json 中对应谓词标记为 added_to_predicate=True。

        dry_run=True 时不写文件，只返回理论会更新的数量。
        """
        if not applied_predicate_keys:
            return 0

        contrib_path = self.stats.predicate_contribution_path
        contrib = _load_json(contrib_path, {})

        if not isinstance(contrib, dict):
            return 0

        updated = 0
        now = time.time()

        for key in applied_predicate_keys:
            item = contrib.get(key)
            if not isinstance(item, dict):
                continue

            item["added_to_predicate"] = True
            item["added_to_predicate_time"] = now
            item["added_to_predicate_source"] = "predicate_recycle_stage_6B"
            updated += 1

        if not dry_run and updated > 0:
            _write_json(contrib_path, contrib)

        return updated

    def _score_one(
        self,
        predicate_key: str,
        item: Dict[str, Any],
        operator_stats: Dict[str, Any],
        existing_predicates: Set[str],
        *,
        include_initial: bool = False,
        require_new_contribution: bool = True,
    ) -> Optional[RecycleCandidate]:
        """
        对一条 predicate_contribution 记录打分。

        返回 None 表示该谓词不适合作为回炉候选。
        """
        if not isinstance(item, dict):
            return None

        predicate = _normalize_predicate_text(item.get("predicate", ""))
        if not predicate:
            return None

        if predicate in existing_predicates:
            return None

        if bool(item.get("added_to_predicate", False)):
            return None

        if not bool(item.get("parse_success", False)):
            return None

        first_seen_source = item.get("first_seen_source")
        last_seen_source = item.get("last_seen_source")

        if not include_initial:
            if first_seen_source == "pkg_initial" and last_seen_source == "pkg_initial":
                return None

        new_edges = item.get("new_edges", []) or []
        new_operators = item.get("new_operators", []) or []
        operators = item.get("operators", []) or []
        edges = item.get("edges", []) or []

        if not isinstance(new_edges, list):
            new_edges = []
        if not isinstance(new_operators, list):
            new_operators = []
        if not isinstance(operators, list):
            operators = []
        if not isinstance(edges, list):
            edges = []

        graph_new_edges = _safe_int(item.get("graph_new_edges", 0))
        graph_new_nodes = _safe_int(item.get("graph_new_nodes", 0))
        success_count = _safe_int(item.get("success_count", 0))
        mode = _coverage_mode()
        fixed_base_structure = _candidate_fixed_base_structure_from_item(item)
        fixed_base_new_edges = _candidate_fixed_base_new_edges_from_item(item)
        frontier_new_edges = _candidate_frontier_new_edges_from_item(item)

        low_freq_ops: List[str] = []
        for op_id in operators:
            freq = _operator_frequency(str(op_id), operator_stats)
            if freq <= self.low_freq_threshold or _operator_matches_target_sparse(str(op_id)):
                low_freq_ops.append(str(op_id))

        # ================== 阶段 6A 补丁：分层价值信号 ==================
        # strong_contribution：
        #   真正贡献过新边 / 新算子 / graph_new_edges 的谓词，优先级最高。
        frontier_as_strong = bool(getattr(RECYCLE_CONFIG, "recycle_frontier_new_edges_as_strong", True))
        if mode == "fixed_base":
            has_strong_contribution = (
                fixed_base_new_edges > 0
                or (frontier_as_strong and frontier_new_edges > 0)
            )
        else:
            has_strong_contribution = (
                graph_new_edges > 0
                or len(new_edges) > 0
                or len(new_operators) > 0
            )

        has_target_sparse_operator = _candidate_has_sparse_target_operator([str(x) for x in operators])

        # low_freq_structure：
        #   没有新边也没关系，只要包含低频算子，就有助于扩大训练分布。
        has_low_freq_structure = len(low_freq_ops) > 0

        # diverse_structure：
        #   结构稍复杂、边数或算子数较多的可执行谓词，可以帮助 decoder 学到更多组合形式。
        has_diverse_structure = (
            len(edges) >= 4
            or len(operators) >= 2
        )

        # stable_success：
        #   DB 成功、parser 成功、有结构边、有算子，即使没有新边，也可以作为稳定正确样本回炉。
        has_stable_success = (
            success_count >= 1
            and len(edges) > 0
            and len(operators) > 0
        )

        is_high_freq_zero_signal = _is_high_freq_zero_signal(
            graph_new_edges=graph_new_edges,
            new_edges=[str(x) for x in new_edges],
            new_operators=[str(x) for x in new_operators],
            low_freq_ops=low_freq_ops,
            operators=[str(x) for x in operators],
        )

        if is_high_freq_zero_signal and RECYCLE_CONFIG.recycle_disallow_high_freq_zero_signal:
            return None

        allow_frontier_non_fixed_base = (
            bool(getattr(RECYCLE_CONFIG, "recycle_allow_frontier_non_fixed_base_apply", True))
            and frontier_new_edges > 0
        )

        # 历史参数名仍保留为 require_new_contribution，避免影响旧调用。
        # 但这里的“contribution”已经扩展为“有价值信号”，不再只等价于新边/新算子。
        if mode == "fixed_base" and bool(getattr(RECYCLE_CONFIG, "recycle_fixed_base_require_base_signal", True)):
            if not fixed_base_structure and not allow_frontier_non_fixed_base:
                return None

        if require_new_contribution:
            has_value = (
                has_strong_contribution
                or has_low_freq_structure
                or has_target_sparse_operator
                or (has_stable_success and not is_high_freq_zero_signal)
            )
            if not has_value:
                return None

        if has_strong_contribution:
            category = "strong_contribution"
        elif has_low_freq_structure or has_target_sparse_operator:
            category = "low_freq_structure"
        elif has_diverse_structure:
            category = "diverse_structure"
        elif has_stable_success:
            category = "stable_success"
        else:
            category = "valid_success"

        score = 0.0
        reasons: List[str] = []

        if mode == "fixed_base":
            score_fixed_base_new_edges = int(fixed_base_new_edges)
            score_frontier_new_edges = int(frontier_new_edges) if frontier_as_strong else 0
            score_graph_new_edges = score_fixed_base_new_edges + score_frontier_new_edges
        else:
            score_fixed_base_new_edges = 0
            score_frontier_new_edges = 0
            score_graph_new_edges = int(graph_new_edges)

        # 1. 强贡献信号：仍然是最高价值。
        if score_fixed_base_new_edges > 0:
            fixed_weight = float(getattr(RECYCLE_CONFIG, "recycle_fixed_base_new_edges_score_weight", 5.0))
            score += fixed_weight * score_fixed_base_new_edges
            reasons.append(f"fixed_base_new_edges={score_fixed_base_new_edges}")

        if score_frontier_new_edges > 0:
            frontier_weight = float(getattr(RECYCLE_CONFIG, "recycle_frontier_new_edges_score_weight", 5.0))
            score += frontier_weight * score_frontier_new_edges
            reasons.append(f"frontier_new_edges={score_frontier_new_edges}")

        if mode != "fixed_base" and score_graph_new_edges > 0:
            score += 5.0 * score_graph_new_edges
            reasons.append(f"score_graph_new_edges={score_graph_new_edges}")

        if len(new_edges) > 0:
            score += 4.0 * len(new_edges)
            reasons.append(f"new_edges={len(new_edges)}")

        if len(new_operators) > 0 and mode != "fixed_base":
            score += 3.0 * len(new_operators)
            reasons.append(f"new_operators={len(new_operators)}")

        # 2. 低频结构信号。
        if len(low_freq_ops) > 0:
            score += 1.0 * len(low_freq_ops)
            reasons.append(f"low_freq_operators={len(low_freq_ops)}")

        if has_target_sparse_operator:
            score += 3.0
            reasons.append("target_sparse_operator")

        # 3. 结构多样性信号。
        if has_diverse_structure:
            diverse_bonus = 0.0

            if len(operators) >= 2:
                diverse_bonus += 1.0
                reasons.append(f"operator_count={len(operators)}")

            if len(edges) >= 4:
                diverse_bonus += 0.5
                reasons.append(f"edge_count={len(edges)}")

            score += diverse_bonus

        # 4. 稳定正确样本信号。
        # 这部分是本次补丁重点：没有新边也可以进入候选，避免几十条新边样本被 5000+ 原始 predicate 淹没。
        if has_stable_success:
            stable_bonus = 0.1 * min(success_count, 5)
            score += stable_bonus
            reasons.append(f"stable_success_count={success_count}")

        if is_high_freq_zero_signal:
            score *= 0.1
            reasons.append("high_freq_zero_signal_penalty")

        # 5. 边数量的小额平滑加分。
        if len(edges) > 0:
            score += 0.05 * min(len(edges), 20)

        if score <= 0.0:
            return None

        return RecycleCandidate(
            predicate_key=str(item.get("predicate_key", predicate_key)),
            predicate=predicate,
            score=float(score),
            category=category,
            reasons=reasons,

            parse_success=bool(item.get("parse_success", False)),
            success_count=success_count,

            graph_new_edges=graph_new_edges,
            graph_new_nodes=graph_new_nodes,
            fixed_base_structure=bool(fixed_base_structure),
            fixed_base_new_edges=int(fixed_base_new_edges),
            frontier_new_edges=int(frontier_new_edges),
            new_edge_count=len(new_edges),
            new_operator_count=len(new_operators),
            low_freq_operator_count=len(low_freq_ops),

            edge_count=len(edges),
            operator_count=len(operators),

            first_seen_source=first_seen_source,
            last_seen_source=last_seen_source,
            first_seen_step=item.get("first_seen_step"),
            last_seen_step=item.get("last_seen_step"),

            normalized_predicate=str(item.get("normalized_predicate", "")),
            abstract_predicate=str(item.get("abstract_predicate", "")),

            new_edges=[str(x) for x in new_edges],
            new_operators=[str(x) for x in new_operators],
            operators=[str(x) for x in operators],
        )

    def select_candidates(
        self,
        *,
        top_k: int = 100,
        min_score: float = 1.0,
        include_initial: bool = False,
        require_new_contribution: bool = True,
    ) -> List[RecycleCandidate]:
        predicate_contribution = self.stats.load_predicate_contribution()
        operator_stats = self.stats.load_operator_stats()
        existing_predicates = self.load_existing_predicates()

        candidates: List[RecycleCandidate] = []

        if not isinstance(predicate_contribution, dict):
            return candidates

        for pred_key, item in predicate_contribution.items():
            cand = self._score_one(
                predicate_key=str(pred_key),
                item=item,
                operator_stats=operator_stats if isinstance(operator_stats, dict) else {},
                existing_predicates=existing_predicates,
                include_initial=include_initial,
                require_new_contribution=require_new_contribution,
            )
            if cand is None:
                continue

            if cand.score < float(min_score):
                continue

            candidates.append(cand)

        candidates.sort(
            key=lambda c: (
                c.score,
                _category_priority(c.category),
                c.fixed_base_new_edges,
                c.frontier_new_edges,
                c.graph_new_edges,
                c.new_edge_count,
                c.new_operator_count,
                c.low_freq_operator_count,
                c.success_count,
            ),
            reverse=True,
        )

        capped: List[RecycleCandidate] = []
        abstract_counts: Dict[str, int] = {}
        abstract_cap = int(RECYCLE_CONFIG.recycle_abstract_cap)
        for cand in candidates:
            key = _candidate_abstract_key(cand)
            if abstract_cap > 0 and abstract_counts.get(key, 0) >= abstract_cap:
                continue
            abstract_counts[key] = abstract_counts.get(key, 0) + 1
            capped.append(cand)
            if len(capped) >= int(top_k):
                break

        return capped

    def write_candidates(
        self,
        candidates: List[RecycleCandidate],
        *,
        top_k: int,
        min_score: float,
        include_initial: bool,
        require_new_contribution: bool,
    ) -> Path:
        category_counts: Dict[str, int] = {}
        for cand in candidates:
            category_counts[cand.category] = category_counts.get(cand.category, 0) + 1

        payload = {
            "generated_at": time.time(),
            "stage": "6A_select_recycle_candidates",
            "project_root": str(PROJECT_ROOT),
            "predicate_txt_path": str(self.predicate_txt_path),
            "correct_collection_path": str(self.correct_collection_path),
            "output_path": str(self.output_path),
            "top_k": int(top_k),
            "min_score": float(min_score),
            "include_initial": bool(include_initial),
            "require_new_contribution": bool(require_new_contribution),
            "candidate_count": len(candidates),
            "category_counts": category_counts,
            "candidates": [asdict(c) for c in candidates],
        }

        _write_json(self.output_path, payload)
        return self.output_path

    def apply_candidates_to_predicate_txt(
        self,
        candidates: List[RecycleCandidate],
        *,
        apply_top_k: int = 800,
        novel_structure_target_count: Optional[int] = None,
        dry_run: bool = False,
        create_backup: bool = True,
    ) -> RecycleApplyResult:
        """
        阶段 6B：将筛选出的候选谓词安全追加到 predicate.txt。

        重要规则：
        - 只追加，不覆盖；
        - 自动去重；
        - 默认先备份 predicate.txt；
        - dry_run=True 时只生成日志，不修改 predicate.txt；
        - 应用成功后，将 predicate_contribution.json 中对应记录标记为 added_to_predicate=True。
        """
        existing_lines = _read_text_lines(self.predicate_txt_path)
        existing_set = {
            _normalize_predicate_text(line)
            for line in existing_lines
            if _normalize_predicate_text(line)
        }

        applied_predicates: List[str] = []
        applied_predicate_keys: List[str] = []
        skipped_duplicates: List[str] = []
        skipped_invalid_items: List[str] = []

        abstract_counts: Dict[str, int] = {}
        apply_limit = int(apply_top_k)
        enable_novel_fill = bool(getattr(RECYCLE_CONFIG, "recycle_enable_novel_structure_fill", True))

        if novel_structure_target_count is None:
            novel_target = int(getattr(RECYCLE_CONFIG, "recycle_novel_structure_target_count", 120))
        else:
            novel_target = int(novel_structure_target_count)

        if not enable_novel_fill:
            novel_target = 0
        novel_target = max(0, min(novel_target, apply_limit))
        value_limit = max(0, apply_limit - novel_target)

        value_candidates: List[RecycleCandidate] = []
        novel_candidates: List[RecycleCandidate] = []

        for cand in candidates:
            if _is_value_candidate(cand):
                value_candidates.append(cand)
            elif _is_novel_structure_candidate(cand):
                novel_candidates.append(cand)

        value_candidate_category_counts: Dict[str, int] = {}
        for cand in value_candidates:
            value_candidate_category_counts[cand.category] = (
                value_candidate_category_counts.get(cand.category, 0) + 1
            )

        novel_candidate_category_counts: Dict[str, int] = {}
        for cand in novel_candidates:
            novel_candidate_category_counts[cand.category] = (
                novel_candidate_category_counts.get(cand.category, 0) + 1
            )

        value_applied_count = 0
        novel_applied_count = 0
        applied_category_counts: Dict[str, int] = {}
        applied_group_counts: Dict[str, int] = {}

        def _try_apply_candidate(cand: RecycleCandidate, group_name: str) -> bool:
            predicate = _normalize_predicate_text(cand.predicate)
            predicate_key = str(cand.predicate_key).strip()

            if not predicate or not predicate_key:
                skipped_invalid_items.append(f"{predicate or '<EMPTY>'} | reason=invalid")
                return False

            allow_frontier_non_fixed_base = (
                bool(getattr(RECYCLE_CONFIG, "recycle_allow_frontier_non_fixed_base_apply", True))
                and int(cand.frontier_new_edges) > 0
            )

            if (
                _coverage_mode() == "fixed_base"
                and not bool(cand.fixed_base_structure)
                and not allow_frontier_non_fixed_base
            ):
                skipped_invalid_items.append(f"{predicate} | reason=non_fixed_base_{group_name}")
                return False

            if predicate in existing_set:
                skipped_duplicates.append(predicate)
                return False

            effective_apply_graph_new_edges = (
                cand.fixed_base_new_edges
                if _coverage_mode() == "fixed_base"
                else cand.graph_new_edges
            )
            if (
                _coverage_mode() == "fixed_base"
                and bool(getattr(RECYCLE_CONFIG, "recycle_frontier_new_edges_as_strong", True))
                and cand.frontier_new_edges > 0
            ):
                effective_apply_graph_new_edges = max(
                    int(effective_apply_graph_new_edges),
                    int(cand.frontier_new_edges),
                )

            if RECYCLE_CONFIG.recycle_apply_require_value_signal and _is_high_freq_zero_signal(
                graph_new_edges=effective_apply_graph_new_edges,
                new_edges=cand.new_edges,
                new_operators=([] if _coverage_mode() == "fixed_base" else cand.new_operators),
                low_freq_ops=(cand.operators if cand.low_freq_operator_count > 0 else []),
                operators=cand.operators,
            ):
                skipped_invalid_items.append(f"{predicate} | reason=high_freq_zero_signal")
                return False

            abstract_key = _candidate_abstract_key(cand)
            if (
                RECYCLE_CONFIG.recycle_abstract_cap > 0
                and abstract_counts.get(abstract_key, 0) >= RECYCLE_CONFIG.recycle_abstract_cap
            ):
                skipped_invalid_items.append(f"{predicate} | reason=abstract_cap")
                return False

            existing_set.add(predicate)
            abstract_counts[abstract_key] = abstract_counts.get(abstract_key, 0) + 1
            applied_predicates.append(predicate)
            applied_predicate_keys.append(predicate_key)
            applied_category_counts[cand.category] = (
                applied_category_counts.get(cand.category, 0) + 1
            )
            applied_group_counts[group_name] = (
                applied_group_counts.get(group_name, 0) + 1
            )
            return True

        for cand in value_candidates:
            if value_applied_count >= value_limit or len(applied_predicates) >= apply_limit:
                break
            if _try_apply_candidate(cand, "value"):
                value_applied_count += 1

        for cand in novel_candidates:
            if novel_applied_count >= novel_target or len(applied_predicates) >= apply_limit:
                break
            if _try_apply_candidate(cand, "novel_structure"):
                novel_applied_count += 1

        backup_path: Optional[Path] = None

        if not dry_run and applied_predicates:
            if create_backup:
                backup_path = self._backup_predicate_txt()

            self.predicate_txt_path.parent.mkdir(parents=True, exist_ok=True)

            merged_lines = list(existing_lines)
            merged_lines.extend(applied_predicates)

            with self.predicate_txt_path.open("w", encoding="utf-8") as f:
                for line in merged_lines:
                    s = _normalize_predicate_text(line)
                    if s:
                        f.write(s + "\n")

        self._mark_added_to_predicate(
            applied_predicate_keys,
            dry_run=dry_run,
        )

        result = RecycleApplyResult(
            applied_count=len(applied_predicates),
            skipped_duplicate_count=len(skipped_duplicates),
            skipped_invalid_count=len(skipped_invalid_items),
            dry_run=bool(dry_run),

            predicate_txt_path=str(self.predicate_txt_path),
            backup_path=str(backup_path) if backup_path is not None else None,
            apply_log_path=str(self.apply_log_path),

            applied_predicate_keys=applied_predicate_keys,
            applied_predicates=applied_predicates,
            skipped_duplicates=skipped_duplicates,
            skipped_invalid_items=skipped_invalid_items,
        )

        log_payload = {
            "generated_at": time.time(),
            "stage": "6B_apply_recycle_candidates",
            "value_apply_limit": value_limit,
            "novel_structure_target_count": novel_target,
            "value_candidate_count": len(value_candidates),
            "novel_structure_candidate_count": len(novel_candidates),
            "value_candidate_category_counts": value_candidate_category_counts,
            "novel_structure_candidate_category_counts": novel_candidate_category_counts,
            "value_applied_count": value_applied_count,
            "novel_structure_applied_count": novel_applied_count,
            "applied_category_counts": applied_category_counts,
            "applied_group_counts": applied_group_counts,
            "result": asdict(result),
        }
        _write_json(self.apply_log_path, log_payload)

        return result

    def run(
        self,
        *,
        top_k: int = 100,
        min_score: float = 1.0,
        include_initial: bool = False,
        require_new_contribution: bool = True,
    ) -> Tuple[List[RecycleCandidate], Path]:
        candidates = self.select_candidates(
            top_k=top_k,
            min_score=min_score,
            include_initial=include_initial,
            require_new_contribution=require_new_contribution,
        )

        output_path = self.write_candidates(
            candidates,
            top_k=top_k,
            min_score=min_score,
            include_initial=include_initial,
            require_new_contribution=require_new_contribution,
        )

        return candidates, output_path


def run_select_recycle_candidates(
    *,
    top_k: int = 100,
    min_score: float = 1.0,
    include_initial: bool = False,
    require_new_contribution: bool = True,
    predicate_txt_path: Optional[Path] = None,
    output_path: Optional[Path] = None,
) -> Tuple[List[RecycleCandidate], Path]:
    recycler = PredicateRecycler(
        predicate_txt_path=predicate_txt_path,
        output_path=output_path,
    )

    return recycler.run(
        top_k=top_k,
        min_score=min_score,
        include_initial=include_initial,
        require_new_contribution=require_new_contribution,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "阶段 6A/6B：从 predicate_contribution.json 中筛选高价值谓词候选；"
            "默认只筛选，传入 --apply 时才追加到 predicate.txt。"
        )
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=100,
        help="最多输出多少条候选。",
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=1.0,
        help="最低候选分数。",
    )
    parser.add_argument(
        "--include-initial",
        action="store_true",
        help="是否允许纯 pkg_initial 的初始谓词进入候选。",
    )
    parser.add_argument(
        "--allow-no-new-contribution",
        action="store_true",
        help="允许没有新边/新算子/低频算子的谓词进入候选。",
    )
    parser.add_argument(
        "--predicate-txt",
        type=str,
        default=str(DEFAULT_PREDICATE_TXT_PATH),
        help="当前 predicate.txt 路径，默认 output/predicate.txt。",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(DEFAULT_RECYCLE_OUTPUT_PATH),
        help="候选输出 JSON 路径。",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="阶段 6B：将筛选出的候选追加写入 predicate.txt。默认不启用。",
    )
    parser.add_argument(
        "--apply-top-k",
        type=int,
        default=800,
        help="阶段 6B：最多追加多少条候选到 predicate.txt。",
    )
    parser.add_argument(
        "--novel-structure-target-count",
        type=int,
        default=None,
        help=(
            "阶段 6B：novel_structure 通道目标数量。"
            "不传时使用 RECYCLE_CONFIG.recycle_novel_structure_target_count。"
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="阶段 6B：只生成 apply 日志，不实际修改 predicate.txt。",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="阶段 6B：应用前不备份 predicate.txt。默认会备份。",
    )

    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    recycler = PredicateRecycler(
        predicate_txt_path=Path(args.predicate_txt),
        output_path=Path(args.output),
    )

    candidates, output_path = recycler.run(
        top_k=args.top_k,
        min_score=args.min_score,
        include_initial=bool(args.include_initial),
        require_new_contribution=not bool(args.allow_no_new_contribution),
    )

    print("=== Predicate Recycle Stage 6A ===")
    print(f"Output: {output_path}")
    print(f"Candidates: {len(candidates)}")

    for idx, cand in enumerate(candidates[:10], start=1):
        print(
            f"[{idx}] score={cand.score:.2f} "
            f"category={cand.category} "
            f"new_edges={cand.new_edge_count} "
            f"new_ops={cand.new_operator_count} "
            f"low_freq_ops={cand.low_freq_operator_count} "
            f"predicate={cand.predicate}"
        )

    if bool(args.apply):
        print("\n=== Predicate Recycle Stage 6B Apply ===")
        apply_result = recycler.apply_candidates_to_predicate_txt(
            candidates,
            apply_top_k=int(args.apply_top_k),
            novel_structure_target_count=(
                None
                if args.novel_structure_target_count is None
                else int(args.novel_structure_target_count)
            ),
            dry_run=bool(args.dry_run),
            create_backup=not bool(args.no_backup),
        )

        print(f"Applied: {apply_result.applied_count}")
        print(f"Skipped duplicates: {apply_result.skipped_duplicate_count}")
        print(f"Skipped invalid: {apply_result.skipped_invalid_count}")
        print(f"Dry run: {apply_result.dry_run}")
        print(f"Predicate txt: {apply_result.predicate_txt_path}")
        print(f"Backup: {apply_result.backup_path}")
        print(f"Apply log: {apply_result.apply_log_path}")


if __name__ == "__main__":
    main()
