"""Candidate-edge reinforcement learning environment and database validation."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import json
import hashlib
import re
import numpy as np
# import torch  # <--- 【新增】导入 torch，用于后续手动清理显存
# 【修改 1】：导入 sqlglot 库，用于专业的 SQL 语法解析
# 请确保环境中已安装：pip install sqlglot
import sqlglot
import time   # <--- 【新增】：引入 time 用于性能统计
import os     # <--- 【新增】：引入 os 用于路径和多进程处理

# 兼容 gymnasium / gym 两种接口
try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:  # 如果本机没装 gymnasium，就退回到 gym
    import gym
    from gym import spaces  # type: ignore

# 项目内模块
from RL.config import (
    RL_ENV_CONFIG,
    RL_ALGO_CONFIG,
    RECYCLE_CONFIG,
    DB_CONFIG,
    CANDIDATE_BITMAP_CONFIG,
    STAGE2_MODE,
)
class SpaceLabel(str, Enum):
    A_CORRECT = "a"
    B_INCORRECT = "b"


@dataclass
class SigmaPair:
    sigma_a: float
    sigma_b: float


from RL.db_executor import BaseDBExecutor, DBQueryResult, create_db_executor
#from RL.llm_completion import LLMCompletionClient, get_database_schema_text
from RL.heuristic_completion import HeuristicCompletionClient # 【修改】引入极速启发式替换模块
from RL.reward_scheme import compute_reward
from RL.predicate_structure import PredicateStructureAnalyzer
from RL.graph_runtime_stats import get_default_runtime_stats
from RL import candidate_bitmap as candidate_bitmap_tools
from RL.candidate_edge_organization import (
    compute_structural_entropies,
    compute_neighbor_success_tendencies,
    load_candidate_edge_neighbors,
)
from RL.candidate_shared_state import (
    CandidateSharedState, normalize_failure_signature,
    advance_completion_variant, completion_retry_needed,
)

# ====== 导入 pkg_manager 的图谱工具 ======
from pkg.graph_builder import KnowledgeGraphBuilder
# ===============================================


# 【修改 3】：重写语法检查函数，使用 sqlglot 解析器
def _check_predicate_syntax_simple(predicate_str: str, dialect: str = "mysql") -> Tuple[bool, float]:
    """
    使用 sqlglot 解析器检查谓词语法。
    返回:
        Tuple[bool, float]: (是否语法合法, 抽象语法树节点数)
    """
    if not predicate_str or not predicate_str.strip():
        return False, 0.0

    normalized_dialect = str(dialect or "").lower().strip()
    # T-SQL 的字符串和 [标识符] 中也可以包含括号，交由解析器检查括号配对。
    if normalized_dialect != "sqlserver" and predicate_str.count('(') != predicate_str.count(')'):
        return False, 0.0

    # Oracle 分支以数据库实际执行结果作为语法权威。sqlglot 对 Oracle 专有
    # 语法（例如 LIKEC/LIKE2/LIKE4）支持不完整，因此 candidate_bitmap 模式
    # 不再进行 sqlglot 预检；其他数据库保持原有预检逻辑。
    if (
        normalized_dialect in {"oracle", "oracledb", "oracle_ai26"}
    ):
        return True, 0.0

    test_str = predicate_str.replace("??", "1")
    dummy_sql = f"SELECT * FROM t WHERE {test_str}"

    try:
        # 项目内部保留 postgresql/sqlserver 标识，解析时映射为 sqlglot 的方言名。
        sqlglot_dialect = {"postgresql": "postgres", "sqlserver": "tsql"}.get(
            str(dialect).lower(), dialect
        )
        if normalized_dialect == "sqlserver":
            from RL.sqlserver_sqlglot import parse_sqlserver_expression
            parsed = parse_sqlserver_expression(dummy_sql)
        else:
            parsed = sqlglot.parse_one(dummy_sql, read=sqlglot_dialect)
        if not parsed:
            return False, 0.0

        # ================== 【极度优雅的复杂度计算】 ==================
        # 遍历解析出的 AST 树，计算所有节点的总数。
        # 嵌套越深、逻辑越复杂，walk() 出来的节点就越多。完美替代正则！
        node_count = sum(1 for _ in parsed.walk())
        return True, float(node_count)
        # ==========================================================

    except Exception:
        return False, 0.0

# 执行结果 -> Va / Vb / Vc 分类的简单规则
_FORBIDDEN_PLACEHOLDER_RE = re.compile(
    r"(\?\?|\bHOLE(?:_[A-Z0-9_]+)?\b)",
    re.IGNORECASE,
)


def _count_forbidden_placeholders(text: str) -> int:
    if text is None:
        return 0
    return len(_FORBIDDEN_PLACEHOLDER_RE.findall(str(text)))


@dataclass
class SqlOutcome: # 用于记录一次 SQL 执行在“语义三集合”上的归类。
    label: SpaceLabel          # A_CORRECT / B_UNCERTAIN / C_INCORRECT
    reason: str                # 简单说明


def classify_sql_outcome(result: DBQueryResult) -> SqlOutcome: # 根据 DBExecutor 返回的 DBQueryResult，将一次 SQL 执行结果归类为 Correct / Uncertain / Incorrect
    # 执行失败的情况
    if not result.success:
        return SqlOutcome(
            label=SpaceLabel.B_INCORRECT,  # 【修改】统一归入 B_INCORRECT
            reason=f"DB query failed: {result.error or 'unknown error'}",
        )
    # 只要执行成功 (result.success is True)，无论 rowcount 是多少，都视为 Correct。
    if result.success:
        return SqlOutcome(
            label=SpaceLabel.A_CORRECT,
            reason=f"Success (rowcount={result.rowcount})",
        )
# 环境：SqlDependencyEnv
class SqlDependencyEnv(gym.Env):#
    """Select a candidate edge, construct a predicate and validate its SQL."""

    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        max_steps_per_episode: Optional[int] = None,
        db_executor: Optional[BaseDBExecutor] = None,
        #llm_client: Optional[LLMCompletionClient] = None,
        worker_rank: int = 0,
        worker_count: int = 1,
        shared_queue: Optional[Any] = None,  # 【新增】接收跨进程通信队列
        shared_reservations: Optional[Any] = None,
        reservation_lock: Optional[Any] = None,
        candidate_shared_state: Optional[CandidateSharedState] = None,
        enumeration_cycle_index: int = 1,
    ) -> None:
        super().__init__()
        # worker位置用于PPO分片或枚举模式的无锁批次分配；不改变公共构造流程。
        self.worker_count = max(1, int(worker_count))
        self.worker_rank = int(worker_rank) % self.worker_count
        self.shared_queue = shared_queue  # 【新增】保存队列实例
        self.candidate_shared_state = candidate_shared_state
        self.candidate_bitmap_enumeration_enabled = bool(
            getattr(RL_ENV_CONFIG, "candidate_bitmap_enumeration_enabled", False)
        )
        self.enumeration_cycle_index = int(enumeration_cycle_index)
        # 旧参数只用于兼容独立环境测试；正式 candidate_bitmap 训练统一使用
        # CandidateSharedState 中的 reservations/lock。
        self.shared_reservations = (
            candidate_shared_state.reservations
            if candidate_shared_state is not None
            else shared_reservations
        )
        self.reservation_lock = (
            candidate_shared_state.lock
            if candidate_shared_state is not None
            else reservation_lock
        )
        self.reservation_claimed_count = 0
        self.reservation_conflict_count = 0
        self.reservation_timeout_release_count = 0
        self.candidate_shared_state_version = -1
        self.current_candidate_bitmap_selected_reservation_token = ""
        # 每步只统计 PPO 最终目标边的单次预占；base 固定为0。
        self.current_candidate_bitmap_reservation_attempts = 0

        # Episode 步数上限
        self.max_steps: int = (
            max_steps_per_episode
            if max_steps_per_episode is not None
            else RL_ENV_CONFIG.max_steps_per_trajectory
        )

        self.candidate_bitmap_pool_size = int(RL_ENV_CONFIG.candidate_bitmap_pool_size)
        self.candidate_bitmap_feature_dim = int(RL_ENV_CONFIG.candidate_bitmap_feature_dim)
        self.action_space = spaces.Discrete(self.candidate_bitmap_pool_size)
        self.obs_dim = self.candidate_bitmap_pool_size * self.candidate_bitmap_feature_dim
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.obs_dim,),
            dtype=np.float32,
        )

        # 组件初始化
        # DB 执行器
        self.db_exec: BaseDBExecutor = db_executor or create_db_executor(autoconnect=True)
        if not hasattr(self.db_exec, "ping"):
            self.db_exec.ping = lambda: False

        """
        # LLM 客户端
        self.llm_client: LLMCompletionClient = llm_client or LLMCompletionClient()

        # 读取一次数据库 schema，作为后续 LLM 补全的上下文
        self.schema_text: str = get_database_schema_text(self.db_exec)
        """

        # 【修改】启发式替换客户端（彻底替代 LLM）
        self.heuristic_client = HeuristicCompletionClient(self.db_exec)

        # 【本地局域图谱校验器】
        self.structure_analyzer = PredicateStructureAnalyzer(strict_consumption=False)
        self.local_builder = KnowledgeGraphBuilder()

        self.runtime_stats = get_default_runtime_stats()
        self.runtime_stats_source = f"env_pid_{os.getpid()}"


        # ================== 阶段17-6：candidate bitmap 候选边状态 ==================
        self.candidate_bitmap_edge_records: list[Dict[str, Any]] = []
        self.candidate_bitmap_bitmaps: Dict[str, np.ndarray] = {}
        self.candidate_bitmap_formal_node_ids: set[str] = set()
        self.candidate_bitmap_formal_graph_data: Dict[str, Any] = {}
        self.candidate_bitmap_last_reload_step: int = -1
        self.current_candidate_bitmap_pool: list[Optional[Dict[str, Any]]] = []
        self.current_candidate_bitmap_features: Optional[np.ndarray] = None
        self.candidate_bitmap_tested_edges: set[str] = set()
        self.candidate_bitmap_success_edges: set[str] = set()
        self.candidate_bitmap_failed_edges: set[str] = set()
        # 由 observed_bitmap | failed_bitmap 缓存得到的全局探索结束状态。
        # 只有确认成功或确认失败的边为 1；失败次数未达到上限时不会在这里标记为结束。
        self.candidate_bitmap_exploration_done = np.zeros((0,), dtype=np.uint8)
        self.candidate_bitmap_probing = np.zeros((0,), dtype=np.uint8)
        self.candidate_bitmap_candidate_union_indices = np.zeros((0,), dtype=np.int32)
        self.candidate_bitmap_unfinished_indices = np.zeros((0,), dtype=np.int32)
        self.candidate_bitmap_edge_id_to_index: Dict[str, int] = {}
        self.current_candidate_bitmap_pool_unfinished_count: int = 0
        # PPO 分片在一个 worker 生命周期内保持稳定；正式图/HNSW刷新不会重新分配目标边。
        self.candidate_bitmap_ppo_shard_indices = np.zeros((0,), dtype=np.int32)
        self.candidate_bitmap_ppo_shard_index_set: set[int] = set()
        self.candidate_bitmap_ppo_shard_signature = ""
        self.candidate_bitmap_ppo_shard_report: Dict[str, Any] = {}
        self.candidate_bitmap_enumeration_order = np.zeros((0,), dtype=np.int32)
        self.candidate_bitmap_enumeration_batch_count: int = 0
        self.candidate_bitmap_enumeration_idle: bool = False
        # 每个 env 独立统计当前全局步数窗口内的变体耗尽次数；不在 Gym reset() 中清空。
        # 达到配置阈值后只对候选排序软降权，不改变 bitmap 或全局失败签名。
        self.candidate_bitmap_probe_variant_exhausted_counts: Dict[str, int] = {}
        self.candidate_bitmap_probe_variant_exhausted_window_index: int = -1
        self.candidate_bitmap_probe_template_cache: Dict[str, Dict[str, Any]] = {}
        self.candidate_bitmap_probe_template_cache_order: list[str] = []

        # 阶段17-14：candidate bitmap 静态结构缓存。
        # required_slot_count / support_rank / inner_ready 只依赖 candidate_edge_index
        # 和正式图，不需要在每次候选池采样时反复扫描全部 edge_records。
        self.candidate_bitmap_required_slot_count_cache: Dict[str, int] = {}
        self.candidate_bitmap_support_rank_cache: Dict[str, int] = {}
        self.candidate_bitmap_inner_ready_cache: Dict[Tuple[str, int], bool] = {}

        # 候选边组织缓存：近邻关系是每轮RL前预计算的静态数据；
        # 成功倾向会在每次Bitmap reload后根据最新 observed/failed 状态重算。
        self.candidate_edge_neighbor_indices = np.full((0, 0), -1, dtype=np.int32)
        self.candidate_edge_neighbor_distances = np.full((0, 0), np.inf, dtype=np.float32)
        self.candidate_edge_hnsw_tendencies = np.full((0,), 0.5, dtype=np.float32)

        self.candidate_bitmap_master_enqueued_edges: set[str] = set()
        self.candidate_bitmap_soft_retry_counts: Dict[str, int] = {}
        # ======================================================================

        # ================== 阶段 11：frontier-aware 覆盖反馈状态 ==================
        self.frontier_mode_active: bool = False
        self.frontier_window_steps: int = 0
        self.frontier_window_db_successes: int = 0
        self.frontier_window_parse_successes: int = 0
        self.frontier_window_new_edges: int = 0
        self.frontier_steps_since_new_edge: int = 0


        # 候选执行窗口内 predicted_new_edge 的实际命中统计。
        self.frontier_window_predicted_positive_steps: int = 0
        self.frontier_window_predicted_new_edges: int = 0
        self.frontier_window_prediction_hit_steps: int = 0
        self.frontier_window_prediction_miss_steps: int = 0


        # env 生命周期内累计 step，不在 reset() 中清零。
        # current_step 是 episode 内步数，会随 reset() 归零；
        self.env_total_steps: int = 0
        # ======================================================================
        self.mutation_rng = np.random.default_rng(
            int(RL_ALGO_CONFIG.seed) + (os.getpid() % 100000)
        )


        self._reload_candidate_bitmap_state(force=True)
        self._rebuild_candidate_bitmap_enumeration_order()
        # =====================================================================

        # Episode 内部状态
        self.current_step: int = 0           # 当前步编号（从 0 开始）
        self.prev_reward: float = 0.0  # 上一步的 reward（用于 obs[3]）
        self.last_reward: float = 0.0        # 上一次 step 的 reward
        self.last_sigmas: SigmaPair = SigmaPair(0.0, 0.0)  # 上一步的 sigma
        self.curr_level = 0
        self.curr_cluster = 0
        self.stagnation_count = 0
        self.level_stagnation_count = 0  # 监控整个 Level 的全局停滞情况
        self.last_diversity: float = 0.0     # 上一步的多样性得分
        self.last_used_uncertain: bool = False
        self.last_is_repeated: bool = False
        self.last_forced_non_novel: bool = False
        # --- 【新增：性能分析计时器】 ---
        self.profiling_stats = {
            "T_Vec_Select": 0.0,
            "T_Decode": 0.0,
            "T_Syntax_Check": 0.0,
            "T_LLM": 0.0,
            "T_DB_Exec": 0.0,
            "T_Reward": 0.0,
            "Total_Steps": 0
        }
        # ---------------------------------
        # --- 【语法错误日志路径】 ---
        # 获取当前运行目录 (即 RL 文件夹)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.error_log_path = os.path.join(current_dir, f"syntax_errors_pid_{os.getpid()}.log")
        # ----------------------------------

    # 构造观测向量

    def _build_obs(self) -> np.ndarray:
        return self._build_obs_candidate_bitmap()


    def _candidate_bitmap_load_json(self, path: Path) -> Dict[str, Any]:
        with Path(path).open("r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError(f"JSON root must be object: {path}")
        return data

    def _reload_candidate_edge_organization(self, edge_count: int) -> None:
        """加载静态近邻，并用最新Bitmap计算每条边的历史成功倾向。"""
        neutral = np.full((edge_count,), 0.5, dtype=np.float32)
        organization_enabled = bool(
            getattr(CANDIDATE_BITMAP_CONFIG, "candidate_organization_enabled", True)
        )
        hnsw_enabled = bool(
            getattr(CANDIDATE_BITMAP_CONFIG, "candidate_hnsw_enabled", True)
        )
        if not organization_enabled or not hnsw_enabled:
            self.candidate_edge_hnsw_tendencies = neutral
            return

        if self.candidate_edge_neighbor_indices.shape[0] != int(edge_count):
            neighbors_path = Path(
                getattr(CANDIDATE_BITMAP_CONFIG, "candidate_edge_neighbors_path")
            )
            indices, distances = load_candidate_edge_neighbors(
                neighbors_path,
                expected_edge_count=edge_count,
            )
            self.candidate_edge_neighbor_indices = indices
            self.candidate_edge_neighbor_distances = distances

        observed = self.candidate_bitmap_bitmaps.get(
            "observed_bitmap",
            np.zeros((edge_count,), dtype=np.uint8),
        )
        failed = self.candidate_bitmap_bitmaps.get(
            "failed_bitmap",
            np.zeros((edge_count,), dtype=np.uint8),
        )
        self.candidate_edge_hnsw_tendencies = compute_neighbor_success_tendencies(
            self.candidate_edge_neighbor_indices,
            self.candidate_edge_neighbor_distances,
            observed,
            failed,
            min_known_neighbors=int(
                getattr(CANDIDATE_BITMAP_CONFIG, "candidate_hnsw_min_known_neighbors", 4)
            ),
        )

    def _candidate_bitmap_hnsw_tendency(self, record: Dict[str, Any]) -> float:
        if not bool(getattr(CANDIDATE_BITMAP_CONFIG, "candidate_organization_enabled", True)):
            return 0.5
        if not bool(getattr(CANDIDATE_BITMAP_CONFIG, "candidate_hnsw_enabled", True)):
            return 0.5
        try:
            edge_index = int(record.get("edge_index", -1))
        except Exception:
            edge_index = -1
        if edge_index < 0 or edge_index >= len(self.candidate_edge_hnsw_tendencies):
            return 0.5
        return float(np.clip(self.candidate_edge_hnsw_tendencies[edge_index], 0.0, 1.0))

    def _candidate_bitmap_structural_entropy(self, record: Dict[str, Any]) -> float:
        if not bool(getattr(CANDIDATE_BITMAP_CONFIG, "candidate_organization_enabled", True)):
            return 0.0
        if not bool(getattr(CANDIDATE_BITMAP_CONFIG, "candidate_entropy_enabled", True)):
            return 0.0
        try:
            value = float(record.get("structural_entropy", 0.0) or 0.0)
        except Exception:
            value = 0.0
        return float(np.clip(value, 0.0, 1.0))

    def _refresh_candidate_bitmap_exploration_state_cache(self, edge_count: int) -> None:
        """Cache candidate-universe, exploration-done and probing states after bitmap reload."""
        n = max(0, int(edge_count))

        def _bitmap(name: str) -> np.ndarray:
            arr = np.asarray(
                self.candidate_bitmap_bitmaps.get(name, np.zeros((n,), dtype=np.uint8)),
                dtype=np.uint8,
            )
            if arr.shape[0] != n:
                return np.zeros((n,), dtype=np.uint8)
            return arr

        strict = _bitmap("strict_candidate_bitmap")
        weak = _bitmap("weak_candidate_bitmap")
        try_bits = _bitmap("try_candidate_bitmap")
        observed = _bitmap("observed_bitmap")
        failed = _bitmap("failed_bitmap")
        probing = _bitmap("probing_bitmap")

        candidate_union = (strict > 0) | (weak > 0) | (try_bits > 0)
        exploration_done = (observed > 0) | (failed > 0)
        probing_mask = probing > 0

        self.candidate_bitmap_exploration_done = exploration_done.astype(np.uint8)
        self.candidate_bitmap_probing = probing_mask.astype(np.uint8)
        self.candidate_bitmap_candidate_union_indices = np.where(candidate_union)[0].astype(np.int32)
        self.candidate_bitmap_unfinished_indices = np.where(
            candidate_union & (~exploration_done) & (~probing_mask)
        )[0].astype(np.int32)

    def _candidate_bitmap_apply_incremental_graph_edge(self, edge_index: int) -> bool:
        """Apply one DB-verified target edge to this worker's in-memory formal graph."""
        if edge_index < 0 or edge_index >= len(self.candidate_bitmap_edge_records):
            return False
        record = self.candidate_bitmap_edge_records[int(edge_index)]
        outer_node_id = str(record.get("outer_node_id", "") or "")
        inner_node_id = str(record.get("inner_node_id", "") or "")
        try:
            slot_index = int(record.get("slot_index", -1))
        except Exception:
            return False
        if not outer_node_id or not inner_node_id or slot_index < 0:
            return False

        graph_data = self.candidate_bitmap_formal_graph_data
        nodes = graph_data.get("nodes", {}) if isinstance(graph_data, dict) else {}
        if not isinstance(nodes, dict):
            return False
        outer_payload = nodes.get(outer_node_id)
        if not isinstance(outer_payload, dict) or inner_node_id not in nodes:
            return False
        slots = outer_payload.setdefault("slots", [])
        if not isinstance(slots, list):
            slots = []
            outer_payload["slots"] = slots
        while len(slots) <= slot_index:
            slots.append([])
        if not isinstance(slots[slot_index], list):
            slots[slot_index] = []
        if inner_node_id in slots[slot_index]:
            return False
        slots[slot_index].append(inner_node_id)
        return True

    def _candidate_bitmap_refresh_changed_hnsw_rows(self, changed_indices: set[int]) -> None:
        """Incrementally refresh only HNSW rows that reference changed labels."""
        if not changed_indices:
            return
        neighbor_indices = np.asarray(self.candidate_edge_neighbor_indices, dtype=np.int32)
        neighbor_distances = np.asarray(self.candidate_edge_neighbor_distances, dtype=np.float32)
        if neighbor_indices.ndim != 2 or neighbor_distances.shape != neighbor_indices.shape:
            return
        if neighbor_indices.shape[0] != len(self.candidate_bitmap_edge_records):
            return

        changed = np.asarray(sorted(changed_indices), dtype=np.int32)
        affected_rows = np.where(np.isin(neighbor_indices, changed).any(axis=1))[0]
        if affected_rows.size == 0:
            return

        observed = np.asarray(
            self.candidate_bitmap_bitmaps.get("observed_bitmap", []), dtype=np.uint8
        )
        failed = np.asarray(
            self.candidate_bitmap_bitmaps.get("failed_bitmap", []), dtype=np.uint8
        )
        minimum = max(
            1,
            int(getattr(CANDIDATE_BITMAP_CONFIG, "candidate_hnsw_min_known_neighbors", 4)),
        )
        tendencies = np.asarray(self.candidate_edge_hnsw_tendencies, dtype=np.float32)

        for row_index in affected_rows.tolist():
            indices = neighbor_indices[int(row_index)]
            distances = neighbor_distances[int(row_index)]
            valid = (
                (indices >= 0)
                & (indices < len(observed))
                & np.isfinite(distances)
            )
            valid_indices = indices[valid]
            valid_distances = distances[valid]
            if valid_indices.size == 0:
                tendencies[int(row_index)] = 0.5
                continue
            known = (observed[valid_indices] > 0) | (failed[valid_indices] > 0)
            if int(known.sum()) < minimum:
                tendencies[int(row_index)] = 0.5
                continue
            known_indices = valid_indices[known]
            weights = 1.0 / (1.0 + np.maximum(0.0, valid_distances[known]))
            total_weight = float(weights.sum())
            if total_weight <= 0.0:
                tendencies[int(row_index)] = 0.5
                continue
            success_rate = float(
                (weights * (observed[known_indices] > 0).astype(np.float32)).sum()
                / total_weight
            )
            confidence = min(1.0, float(len(known_indices)) / float(len(valid_indices)))
            tendencies[int(row_index)] = float(
                np.clip(0.5 + (success_rate - 0.5) * confidence, 0.0, 1.0)
            )

        self.candidate_edge_hnsw_tendencies = tendencies

    def _sync_candidate_shared_state(self, force: bool = False) -> None:
        """Merge immediate cross-worker completion state without disk reload."""
        shared_state = getattr(self, "candidate_shared_state", None)
        if shared_state is None or not self.candidate_bitmap_edge_records:
            return
        snapshot = shared_state.snapshot_if_changed(
            -1 if force else int(self.candidate_shared_state_version)
        )
        if not isinstance(snapshot, dict):
            return

        n = len(self.candidate_bitmap_edge_records)
        observed = np.asarray(
            self.candidate_bitmap_bitmaps.get("observed_bitmap", np.zeros(n)),
            dtype=np.uint8,
        ).copy()
        failed = np.asarray(
            self.candidate_bitmap_bitmaps.get("failed_bitmap", np.zeros(n)),
            dtype=np.uint8,
        ).copy()
        old_observed = observed.copy()
        old_failed = failed.copy()

        incremental_indices = []
        if bool(snapshot.get("full_snapshot", False)):
            observed.fill(0)
            failed.fill(0)
            for edge_id, raw_index in dict(snapshot.get("observed", {})).items():
                try:
                    edge_index = int(raw_index)
                except Exception:
                    edge_index = self.candidate_bitmap_edge_id_to_index.get(str(edge_id), -1)
                if 0 <= edge_index < n:
                    observed[edge_index] = 1
            for edge_id, raw_index in dict(snapshot.get("confirmed_failed", {})).items():
                try:
                    edge_index = int(raw_index)
                except Exception:
                    edge_index = self.candidate_bitmap_edge_id_to_index.get(str(edge_id), -1)
                if 0 <= edge_index < n and observed[edge_index] <= 0:
                    failed[edge_index] = 1
            incremental_indices.extend(
                dict(snapshot.get("incremental_graph_edges", {})).values()
            )
        else:
            for event in list(snapshot.get("events", []) or []):
                if not isinstance(event, dict):
                    continue
                try:
                    edge_index = int(event.get("edge_index", -1))
                except Exception:
                    continue
                if edge_index < 0 or edge_index >= n:
                    continue
                if str(event.get("kind", "")) == "observed":
                    observed[edge_index] = 1
                    failed[edge_index] = 0
                    if bool(event.get("incremental_graph_edge", False)):
                        incremental_indices.append(edge_index)
                elif str(event.get("kind", "")) == "confirmed_failed":
                    if observed[edge_index] <= 0:
                        failed[edge_index] = 1

        self.candidate_bitmap_bitmaps["observed_bitmap"] = observed
        self.candidate_bitmap_bitmaps["failed_bitmap"] = failed
        changed_indices = set(
            np.where((old_observed != observed) | (old_failed != failed))[0].tolist()
        )

        graph_changed = False
        for raw_index in incremental_indices:
            try:
                edge_index = int(raw_index)
            except Exception:
                continue
            graph_changed = self._candidate_bitmap_apply_incremental_graph_edge(edge_index) or graph_changed

        self._refresh_candidate_bitmap_exploration_state_cache(n)
        self._candidate_bitmap_refresh_changed_hnsw_rows(changed_indices)
        if graph_changed:
            self.candidate_bitmap_inner_ready_cache.clear()
        self.candidate_shared_state_version = int(snapshot.get("state_version", 0))

    def _candidate_bitmap_exploration_done_for_index(
        self,
        edge_index: int,
        edge_id: str = "",
    ) -> bool:
        """Return confirmed global completion; local success is also immediately treated as done."""
        if edge_id and edge_id in getattr(self, "candidate_bitmap_success_edges", set()):
            return True

        try:
            idx = int(edge_index)
        except Exception:
            return False

        cached = np.asarray(
            getattr(self, "candidate_bitmap_exploration_done", np.zeros((0,), dtype=np.uint8)),
            dtype=np.uint8,
        )
        if 0 <= idx < len(cached):
            return bool(int(cached[idx]) > 0)

        n = len(getattr(self, "candidate_bitmap_edge_records", []))
        observed = np.asarray(
            self.candidate_bitmap_bitmaps.get("observed_bitmap", np.zeros((n,), dtype=np.uint8)),
            dtype=np.uint8,
        )
        failed = np.asarray(
            self.candidate_bitmap_bitmaps.get("failed_bitmap", np.zeros((n,), dtype=np.uint8)),
            dtype=np.uint8,
        )
        return bool(
            0 <= idx < len(observed)
            and 0 <= idx < len(failed)
            and (int(observed[idx]) > 0 or int(failed[idx]) > 0)
        )

    def _candidate_bitmap_probing_for_index(self, edge_index: int) -> bool:
        try:
            idx = int(edge_index)
        except Exception:
            return False

        cached = np.asarray(
            getattr(self, "candidate_bitmap_probing", np.zeros((0,), dtype=np.uint8)),
            dtype=np.uint8,
        )
        if 0 <= idx < len(cached):
            return bool(int(cached[idx]) > 0)

        n = len(getattr(self, "candidate_bitmap_edge_records", []))
        probing = np.asarray(
            self.candidate_bitmap_bitmaps.get("probing_bitmap", np.zeros((n,), dtype=np.uint8)),
            dtype=np.uint8,
        )
        return bool(0 <= idx < len(probing) and int(probing[idx]) > 0)

    def _reload_candidate_bitmap_state(self, force: bool = False) -> None:

        current_step = int(getattr(self, "env_total_steps", getattr(self, "current_step", 0)))
        reload_interval = int(getattr(RL_ENV_CONFIG, "candidate_bitmap_reload_interval_steps", 200))

        if (
            not force
            and self.candidate_bitmap_edge_records
            and reload_interval > 0
            and current_step - int(self.candidate_bitmap_last_reload_step) < reload_interval
        ):
            return

        edge_index_path = Path(getattr(CANDIDATE_BITMAP_CONFIG, "edge_index_path"))
        bitmap_npz_path = Path(getattr(CANDIDATE_BITMAP_CONFIG, "bitmap_npz_path"))
        graph_path = Path(getattr(CANDIDATE_BITMAP_CONFIG, "global_knowledge_graph_path"))

        try:
            edge_index_data = self._candidate_bitmap_load_json(edge_index_path)
            edge_records = edge_index_data.get("edges", [])
            if not isinstance(edge_records, list):
                raise ValueError(f"candidate_edge_index.json 中 edges 字段必须是 list: {edge_index_path}")
            self.candidate_bitmap_edge_records = [x for x in edge_records if isinstance(x, dict)]
            self.candidate_bitmap_edge_id_to_index = {
                str(record.get("edge_id", "")): int(record.get("edge_index", -1))
                for record in self.candidate_bitmap_edge_records
                if str(record.get("edge_id", ""))
            }

            n = len(self.candidate_bitmap_edge_records)
            loaded = np.load(bitmap_npz_path)
            bitmaps: Dict[str, np.ndarray] = {}
            for name in (
                "strict_candidate_bitmap",
                "weak_candidate_bitmap",
                "try_candidate_bitmap",
                "observed_bitmap",
                "failed_bitmap",
                "probing_bitmap",
                "missing_candidate_bitmap",
                "available_initial_bitmap",
            ):
                if name in loaded:
                    arr = np.asarray(loaded[name], dtype=np.uint8)
                    if arr.shape[0] != n:
                        arr = np.zeros(n, dtype=np.uint8)
                else:
                    arr = np.zeros(n, dtype=np.uint8)
                bitmaps[name] = arr.copy()
            self.candidate_bitmap_bitmaps = bitmaps
            self._refresh_candidate_bitmap_exploration_state_cache(n)

            if self._candidate_bitmap_enumeration_active():
                self.candidate_edge_neighbor_indices = np.full((0, 0), -1, dtype=np.int32)
                self.candidate_edge_neighbor_distances = np.full((0, 0), np.inf, dtype=np.float32)
                self.candidate_edge_hnsw_tendencies = np.full((n,), 0.5, dtype=np.float32)
            else:
                self._reload_candidate_edge_organization(n)

            graph_data = self._candidate_bitmap_load_json(graph_path)
            self.candidate_bitmap_formal_graph_data = graph_data if isinstance(graph_data, dict) else {}
            nodes = graph_data.get("nodes", {}) if isinstance(graph_data, dict) else {}
            if isinstance(nodes, dict):
                self.candidate_bitmap_formal_node_ids = {str(x) for x in nodes.keys()}
            else:
                self.candidate_bitmap_formal_node_ids = set()

            if self._candidate_bitmap_enumeration_active():
                self.candidate_bitmap_required_slot_count_cache = {}
                self.candidate_bitmap_support_rank_cache = {}
                self.candidate_bitmap_inner_ready_cache = {}
                self._rebuild_candidate_bitmap_ppo_shard_cache()
            else:
                self._rebuild_candidate_bitmap_static_caches()
            self._sync_candidate_shared_state(force=True)

            self.candidate_bitmap_last_reload_step = current_step

            self.candidate_bitmap_probe_template_cache.clear()
            self.candidate_bitmap_probe_template_cache_order.clear()
            self.candidate_bitmap_master_enqueued_edges.clear()

            print(
                f"[CandidateBitmapEnv] reloaded candidate bitmap: "
                f"edges={len(self.candidate_bitmap_edge_records)}, "
                f"available={int(bitmaps.get('available_initial_bitmap', np.zeros(n)).sum())}, "
                f"missing={int(bitmaps.get('missing_candidate_bitmap', np.zeros(n)).sum())}",
                flush=True,
            )
        except Exception as exc:
            print(f"[CandidateBitmapEnv] reload failed: {exc}", flush=True)
            if not self.candidate_bitmap_edge_records:
                self.candidate_bitmap_edge_records = []
                self.candidate_bitmap_bitmaps = {}
                self.candidate_bitmap_formal_node_ids = set()
                self.candidate_bitmap_formal_graph_data = {}
                self.candidate_bitmap_required_slot_count_cache = {}
                self.candidate_bitmap_support_rank_cache = {}
                self.candidate_bitmap_inner_ready_cache = {}
                self.candidate_bitmap_exploration_done = np.zeros((0,), dtype=np.uint8)
                self.candidate_bitmap_probing = np.zeros((0,), dtype=np.uint8)
                self.candidate_bitmap_candidate_union_indices = np.zeros((0,), dtype=np.int32)
                self.candidate_bitmap_unfinished_indices = np.zeros((0,), dtype=np.int32)
                self.candidate_bitmap_edge_id_to_index = {}
                self.candidate_edge_neighbor_indices = np.full((0, 0), -1, dtype=np.int32)
                self.candidate_edge_neighbor_distances = np.full((0, 0), np.inf, dtype=np.float32)
                self.candidate_edge_hnsw_tendencies = np.full((0,), 0.5, dtype=np.float32)

    def _candidate_bitmap_level_value(self, level: str) -> float:
        text = str(level or "").lower()
        if text == "strict":
            return 1.0
        if text == "weak":
            return 0.5
        if text == "try":
            return 0.25
        return 0.0

    def _candidate_bitmap_representation_guidance_enabled(self) -> bool:
        """只控制pattern transfer对第二阶段的帮助，不控制HNSW或结构熵。"""
        return bool(
            getattr(
                CANDIDATE_BITMAP_CONFIG,
                "candidate_representation_guidance_enabled",
                True,
            )
        )

    def _candidate_bitmap_effective_pattern_transfer_score(
        self,
        record: Dict[str, Any],
    ) -> float:
        if not self._candidate_bitmap_representation_guidance_enabled():
            return 0.0
        try:
            value = float(record.get("pattern_transfer_score", 0.0) or 0.0)
        except Exception:
            value = 0.0
        return float(np.clip(value, 0.0, 1.0))

    def _candidate_bitmap_effective_edge_score(self, record: Dict[str, Any]) -> float:
        if self._candidate_bitmap_representation_guidance_enabled():
            try:
                return float(record.get("edge_score", 0.0) or 0.0)
            except Exception:
                return 0.0

        # 关闭表示迁移帮助时，忽略文件中可能残留的旧加分，只保留层级基础分。
        level = str(record.get("candidate_level", "try") or "try").lower()
        if level == "strict":
            return float(getattr(CANDIDATE_BITMAP_CONFIG, "strict_base_score", 1.0))
        if level == "weak":
            return float(getattr(CANDIDATE_BITMAP_CONFIG, "weak_base_score", 0.45))
        return float(getattr(CANDIDATE_BITMAP_CONFIG, "try_base_score", 0.15))

    def _candidate_bitmap_enumeration_active(self) -> bool:
        """Enumeration owns target selection and never uses PPO/scored candidate pools."""
        return bool(
            getattr(self, "candidate_bitmap_enumeration_enabled", False)
        )

    def _rebuild_candidate_bitmap_enumeration_order(self) -> None:
        """Build the configured fixed full-universe enumeration order."""
        if not self._candidate_bitmap_enumeration_active():
            self.candidate_bitmap_enumeration_order = np.zeros((0,), dtype=np.int32)
            self.candidate_bitmap_enumeration_batch_count = 0
            return

        records = list(getattr(self, "candidate_bitmap_edge_records", []))
        edge_count = len(records)
        if edge_count <= 0:
            self.candidate_bitmap_enumeration_order = np.zeros((0,), dtype=np.int32)
            self.candidate_bitmap_enumeration_batch_count = 0
            return

        base_seed = int(getattr(RL_ENV_CONFIG, "candidate_bitmap_enumeration_seed", 42))
        order_mode = str(
            getattr(
                RL_ENV_CONFIG,
                "candidate_bitmap_enumeration_order_mode",
                "independent_random",
            )
            or "independent_random"
        ).strip().lower()
        independent_worker_order = bool(
            getattr(
                RL_ENV_CONFIG,
                "candidate_bitmap_enumeration_independent_worker_order_enabled",
                False,
            )
        )
        worker_count = max(1, int(getattr(self, "worker_count", 1)))
        worker_rank = int(getattr(self, "worker_rank", 0)) % worker_count
        order_group_count = min(
            worker_count,
            max(
                1,
                int(
                    getattr(
                        RL_ENV_CONFIG,
                        "candidate_bitmap_enumeration_order_group_count",
                        8,
                    )
                ),
            ),
        )
        order_group_index = worker_rank % order_group_count
        entropy_bias = float(np.clip(
            float(
                getattr(
                    RL_ENV_CONFIG,
                    "candidate_bitmap_enumeration_entropy_bias",
                    0.0,
                )
            ),
            0.0,
            1.0,
        ))
        random_scores = np.zeros((edge_count,), dtype=np.float64)
        hash_denominator = float((1 << 64) - 1)
        for idx, record in enumerate(records):
            edge_id = str(record.get("edge_id", "") or f"edge_index:{idx}")
            digest_input = (
                f"{base_seed}:{worker_rank}:{edge_id}"
                if independent_worker_order
                else f"{base_seed}:{edge_id}"
            )
            digest = self._candidate_bitmap_stable_digest(digest_input)
            random_scores[idx] = int(digest[:16], 16) / hash_denominator

        random_order = np.argsort(-random_scores, kind="stable")
        if order_mode == "cycle_stratified_hard_random":
            # Enumeration-only scheduling: each worker refreshes an independent
            # full permutation per RL cycle. The prefix it is expected to execute
            # interleaves complexity-ordered structural edges and random edges
            # without reading scores, bitmaps, graph outcomes, reservations, or
            # PPO actions.
            cycle_index = int(getattr(self, "enumeration_cycle_index", 1))
            structural_ratio = float(np.clip(
                float(
                    getattr(
                        RL_ENV_CONFIG,
                        "candidate_bitmap_enumeration_structural_ratio",
                        0.70,
                    )
                ),
                0.0,
                1.0,
            ))
            per_worker_prefix_count = min(
                edge_count,
                int(np.ceil(
                    max(1, int(RECYCLE_CONFIG.recycle_interval_steps))
                    / float(worker_count)
                )),
            )
            outer_slot_counts: Dict[str, int] = {}
            for record in records:
                outer_node_id = str(record.get("outer_node_id", "") or "")
                if not outer_node_id:
                    continue
                try:
                    slot_index = int(record.get("slot_index", -1))
                except Exception:
                    slot_index = -1
                outer_slot_counts[outer_node_id] = max(
                    int(outer_slot_counts.get(outer_node_id, 0)),
                    slot_index + 1,
                )

            structural_indices: list[int] = []
            random_indices: list[int] = []
            cycle_worker_seed = f"{base_seed}:{cycle_index}:{worker_rank}"
            for idx, record in enumerate(records):
                outer_node_id = str(record.get("outer_node_id", "") or "")
                inner_node_id = str(record.get("inner_node_id", "") or "")
                outer_slot_count = int(outer_slot_counts.get(outer_node_id, 0))
                inner_slot_count = int(outer_slot_counts.get(inner_node_id, 0))
                if outer_slot_count >= 2 or inner_slot_count > 0:
                    structural_indices.append(idx)
                else:
                    random_indices.append(idx)

            def _structural_complexity_order_key(
                idx: int,
            ) -> Tuple[int, int, int, int]:
                record = records[idx]
                outer_node_id = str(record.get("outer_node_id", "") or "")
                inner_node_id = str(record.get("inner_node_id", "") or "")
                outer_slot_count = int(outer_slot_counts.get(outer_node_id, 0))
                inner_slot_count = int(outer_slot_counts.get(inner_node_id, 0))
                edge_id = str(record.get("edge_id", "") or f"edge_index:{idx}")
                # Enumeration-only ordering: prefer multi-slot outer operators,
                # then upstream operator-valued inners and their slot complexity.
                # The digest only breaks equal-complexity ties reproducibly for
                # each worker and RL cycle; no score or execution result is read.
                digest = self._candidate_bitmap_stable_digest(
                    f"{cycle_worker_seed}:structural_random:{edge_id}"
                )
                return (
                    -outer_slot_count,
                    -int(inner_slot_count > 0),
                    -inner_slot_count,
                    -int(digest[:16], 16),
                )

            structural_indices.sort(key=_structural_complexity_order_key)
            random_indices.sort(
                key=lambda idx: -int(
                    self._candidate_bitmap_stable_digest(
                        f"{cycle_worker_seed}:random:"
                        f"{records[idx].get('edge_id', '') or f'edge_index:{idx}'}"
                    )[:16],
                    16,
                )
            )

            structural_target_count = int(round(
                per_worker_prefix_count * structural_ratio
            ))
            structural_target_count = min(
                len(structural_indices),
                max(0, structural_target_count),
            )
            random_target_count = min(
                len(random_indices),
                per_worker_prefix_count - structural_target_count,
            )
            prefix_structural = structural_indices[:structural_target_count]
            prefix_random = random_indices[:random_target_count]

            # If either class is too small, retain a complete prefix by taking
            # deterministic remaining edges; the normal 20000+ universe has
            # enough of both classes to preserve the configured structural/random split.
            prefix_selected = set(prefix_structural)
            prefix_selected.update(prefix_random)
            prefix_missing = per_worker_prefix_count - len(prefix_selected)
            if prefix_missing > 0:
                fallback_indices = [
                    idx
                    for idx in range(edge_count)
                    if idx not in prefix_selected
                ]
                fallback_indices.sort(
                    key=lambda idx: -int(
                        self._candidate_bitmap_stable_digest(
                            f"{cycle_worker_seed}:prefix_fallback:"
                            f"{records[idx].get('edge_id', '') or f'edge_index:{idx}'}"
                        )[:16],
                        16,
                    )
                )
                prefix_random.extend(fallback_indices[:prefix_missing])

            structural_block_size = int(round(structural_ratio * 10.0))
            structural_block_size = min(10, max(0, structural_block_size))
            random_block_size = 10 - structural_block_size
            prefix_order: list[int] = []
            structural_position = 0
            random_position = 0
            while (
                structural_position < len(prefix_structural)
                or random_position < len(prefix_random)
            ):
                next_structural_position = min(
                    len(prefix_structural),
                    structural_position + structural_block_size,
                )
                prefix_order.extend(
                    prefix_structural[structural_position:next_structural_position]
                )
                structural_position = next_structural_position
                next_random_position = min(
                    len(prefix_random),
                    random_position + random_block_size,
                )
                prefix_order.extend(
                    prefix_random[random_position:next_random_position]
                )
                random_position = next_random_position
                if structural_block_size == 0:
                    prefix_order.extend(prefix_random[random_position:])
                    random_position = len(prefix_random)
                if random_block_size == 0:
                    prefix_order.extend(prefix_structural[structural_position:])
                    structural_position = len(prefix_structural)

            prefix_selected = set(prefix_order)
            remaining_indices = [
                idx for idx in range(edge_count) if idx not in prefix_selected
            ]
            remaining_indices.sort(
                key=lambda idx: -int(
                    self._candidate_bitmap_stable_digest(
                        f"{cycle_worker_seed}:tail:"
                        f"{records[idx].get('edge_id', '') or f'edge_index:{idx}'}"
                    )[:16],
                    16,
                )
            )
            self.candidate_bitmap_enumeration_order = np.asarray(
                prefix_order + remaining_indices,
                dtype=np.int32,
            )
            self.candidate_bitmap_enumeration_batch_count = edge_count
            if worker_rank == 0:
                print(
                    "[CandidateBitmap Enumeration] "
                    f"cycle={cycle_index}, seed={base_seed}, "
                    "fixed_order=False, "
                    "order_mode=cycle_stratified_hard_random, "
                    f"structural_ratio={structural_ratio:.3f}, "
                    f"worker_prefix_steps={per_worker_prefix_count}, "
                    f"structural_prefix_count={len(prefix_structural)}, "
                    f"random_prefix_count={len(prefix_order) - len(prefix_structural)}, "
                    f"edges={edge_count}, workers={worker_count}, "
                    f"steps_per_worker_order={edge_count}, tail_idle=0",
                    flush=True,
                )
            return

        if order_mode == "top_down_operator_block":
            # Enumeration-only scheduling: workers in one group visit the same
            # outer-operator blocks, while each worker keeps its own fixed order
            # inside a block. No score, entropy, bitmap outcome, lock, or shared
            # cursor participates in this order.
            outer_node_ids = {
                str(record.get("outer_node_id", "") or "")
                for record in records
                if str(record.get("outer_node_id", "") or "")
            }
            block_indices: Dict[str, list[int]] = {}
            required_slot_count: Dict[str, int] = {}
            nonterminal_inner_count: Dict[str, int] = {}
            self_loop_count: Dict[str, int] = {}

            for idx, record in enumerate(records):
                outer_node_id = str(record.get("outer_node_id", "") or "")
                inner_node_id = str(record.get("inner_node_id", "") or "")
                block_key = outer_node_id if outer_node_id else f"__edge_block_{idx}"
                block_indices.setdefault(block_key, []).append(idx)
                try:
                    slot_index = int(record.get("slot_index", -1))
                except Exception:
                    slot_index = -1
                required_slot_count[block_key] = max(
                    int(required_slot_count.get(block_key, 0)),
                    slot_index + 1,
                )
                if inner_node_id in outer_node_ids:
                    nonterminal_inner_count[block_key] = (
                        int(nonterminal_inner_count.get(block_key, 0)) + 1
                    )
                if outer_node_id and outer_node_id == inner_node_id:
                    self_loop_count[block_key] = int(self_loop_count.get(block_key, 0)) + 1

            def _block_order_key(block_key: str) -> Tuple[float, float, float, int]:
                block_size = max(1, len(block_indices.get(block_key, [])))
                nonterminal_ratio = float(
                    nonterminal_inner_count.get(block_key, 0)
                ) / float(block_size)
                self_loop_ratio = float(self_loop_count.get(block_key, 0)) / float(block_size)
                group_digest = self._candidate_bitmap_stable_digest(
                    f"{base_seed}:group:{order_group_index}:outer:{block_key}"
                )
                return (
                    -float(required_slot_count.get(block_key, 0)),
                    -nonterminal_ratio,
                    -self_loop_ratio,
                    -int(group_digest[:16], 16),
                )

            ordered_blocks = sorted(block_indices.keys(), key=_block_order_key)
            ordered_indices: list[int] = []
            for block_key in ordered_blocks:
                def _edge_order_key(idx: int) -> Tuple[int, int, int]:
                    record = records[idx]
                    outer_node_id = str(record.get("outer_node_id", "") or "")
                    inner_node_id = str(record.get("inner_node_id", "") or "")
                    edge_id = str(record.get("edge_id", "") or f"edge_index:{idx}")
                    worker_digest = self._candidate_bitmap_stable_digest(
                        f"{base_seed}:worker:{worker_rank}:edge:{edge_id}"
                    )
                    return (
                        0 if inner_node_id in outer_node_ids else 1,
                        0 if outer_node_id and outer_node_id == inner_node_id else 1,
                        -int(worker_digest[:16], 16),
                    )

                ordered_indices.extend(
                    sorted(block_indices[block_key], key=_edge_order_key)
                )

            self.candidate_bitmap_enumeration_order = np.asarray(
                ordered_indices,
                dtype=np.int32,
            )
            self.candidate_bitmap_enumeration_batch_count = edge_count
            if worker_rank == 0:
                workers_per_group = int(np.ceil(worker_count / float(order_group_count)))
                print(
                    "[CandidateBitmap Enumeration] "
                    f"cycle={self.enumeration_cycle_index}, seed={base_seed}, "
                    "fixed_order=True, order_mode=top_down_operator_block, "
                    f"order_groups={order_group_count}, "
                    f"workers_per_group_max={workers_per_group}, "
                    f"operator_blocks={len(ordered_blocks)}, entropy_enabled=False, "
                    f"edges={edge_count}, workers={worker_count}, "
                    f"steps_per_worker_order={edge_count}, tail_idle=0",
                    flush=True,
                )
            return

        if independent_worker_order:
            # Every worker owns a complete deterministic permutation. Cross-worker
            # overlap is intentional for the uncoordinated enumeration baseline.
            self.candidate_bitmap_enumeration_order = random_order.astype(
                np.int32,
                copy=False,
            )
            self.candidate_bitmap_enumeration_batch_count = edge_count
            if worker_rank == 0:
                print(
                    "[CandidateBitmap Enumeration] "
                    f"cycle={self.enumeration_cycle_index}, seed={base_seed}, "
                    "fixed_order=True, order_mode=independent_worker_random, "
                    "independent_worker_order=True, entropy_bias=0.000, "
                    f"edges={edge_count}, workers={worker_count}, "
                    f"steps_per_worker_order={edge_count}, tail_idle=0",
                    flush=True,
                )
            return

        # Coordinated mode preserves the existing global non-overlapping batches.
        # Within equal entropy values and the remaining region, stable hash order
        # keeps the complete enumeration deterministic across workers and cycles.
        structural_entropies = compute_structural_entropies(records).astype(
            np.float64,
            copy=False,
        )
        priority_count = min(
            edge_count,
            int(np.ceil(entropy_bias * float(edge_count))),
        )
        if priority_count > 0:
            entropy_order = np.lexsort((-random_scores, -structural_entropies))
            priority_indices = entropy_order[:priority_count]
            priority_mask = np.zeros((edge_count,), dtype=bool)
            priority_mask[priority_indices] = True
            remaining_indices = random_order[~priority_mask[random_order]]
            self.candidate_bitmap_enumeration_order = np.concatenate(
                (priority_indices, remaining_indices),
            ).astype(np.int32, copy=False)
        else:
            self.candidate_bitmap_enumeration_order = random_order.astype(
                np.int32,
                copy=False,
            )
        self.candidate_bitmap_enumeration_batch_count = (
            edge_count + worker_count - 1
        ) // worker_count
        if worker_rank == 0:
            tail_idle_count = (
                self.candidate_bitmap_enumeration_batch_count * worker_count
                - edge_count
            )
            print(
                "[CandidateBitmap Enumeration] "
                f"cycle={self.enumeration_cycle_index}, seed={base_seed}, "
                "fixed_order=True, order_mode=entropy_priority, "
                f"entropy_bias={entropy_bias:.3f}, "
                f"entropy_priority_count={priority_count}, "
                f"entropy_min={float(np.min(structural_entropies)):.3f}, "
                f"entropy_median={float(np.median(structural_entropies)):.3f}, "
                f"entropy_max={float(np.max(structural_entropies)):.3f}, "
                f"edges={edge_count}, workers={worker_count}, "
                f"batches_per_pass={self.candidate_bitmap_enumeration_batch_count}, "
                f"tail_idle={tail_idle_count}",
                flush=True,
            )

    def _candidate_bitmap_enumeration_assignment(
        self,
    ) -> Tuple[Optional[Dict[str, Any]], bool, int]:
        """Return this worker's assigned edge, idle flag, and shuffled position."""
        if not self._candidate_bitmap_enumeration_active():
            return None, False, -1

        order = np.asarray(
            getattr(self, "candidate_bitmap_enumeration_order", np.zeros((0,), dtype=np.int32)),
            dtype=np.int32,
        )
        batch_count = int(getattr(self, "candidate_bitmap_enumeration_batch_count", 0))
        if order.size <= 0 or batch_count <= 0:
            return None, True, -1

        independent_worker_order = bool(
            getattr(
                RL_ENV_CONFIG,
                "candidate_bitmap_enumeration_independent_worker_order_enabled",
                False,
            )
        )
        order_mode = str(
            getattr(
                RL_ENV_CONFIG,
                "candidate_bitmap_enumeration_order_mode",
                "independent_random",
            )
            or "independent_random"
        ).strip().lower()
        if order_mode in {
            "cycle_stratified_hard_random",
            "top_down_operator_block",
        } or independent_worker_order:
            shuffled_position = int(getattr(self, "env_total_steps", 0)) % int(order.size)
            record_index = int(order[shuffled_position])
            records = getattr(self, "candidate_bitmap_edge_records", [])
            if record_index < 0 or record_index >= len(records):
                return None, True, shuffled_position
            return records[record_index], False, shuffled_position

        batch_index = int(getattr(self, "env_total_steps", 0)) % batch_count
        worker_count = max(1, int(getattr(self, "worker_count", 1)))
        worker_rank = int(getattr(self, "worker_rank", 0)) % worker_count
        shuffled_position = batch_index * worker_count + worker_rank
        if shuffled_position >= int(order.size):
            return None, True, shuffled_position

        record_index = int(order[shuffled_position])
        records = getattr(self, "candidate_bitmap_edge_records", [])
        if record_index < 0 or record_index >= len(records):
            return None, True, shuffled_position
        return records[record_index], False, shuffled_position

    def _candidate_bitmap_ppo_sharding_active(self) -> bool:
        """PPO uses balanced target-edge shards; base always keeps the full universe."""
        return bool(
            not self._candidate_bitmap_enumeration_active()
            and int(getattr(self, "worker_count", 1)) > 1
            and getattr(RL_ENV_CONFIG, "candidate_bitmap_ppo_selection_enabled", True)
        )

    @staticmethod
    def _candidate_bitmap_stable_digest(text: str) -> str:
        """Return a process-independent digest; Python hash() is randomized per worker."""
        return hashlib.sha256(str(text).encode("utf-8")).hexdigest()

    def _rebuild_candidate_bitmap_ppo_shard_cache(self) -> None:
        """
        Build one stable, balanced target-edge shard for this PPO worker.

        Only known candidate metadata is used. Success/failure outcomes never
        participate, so the partition cannot leak experiment answers.
        """
        if not self._candidate_bitmap_ppo_sharding_active():
            self.candidate_bitmap_ppo_shard_indices = np.zeros((0,), dtype=np.int32)
            self.candidate_bitmap_ppo_shard_index_set = set()
            self.candidate_bitmap_ppo_shard_signature = ""
            self.candidate_bitmap_ppo_shard_report = {
                "active": False,
                "worker_rank": int(getattr(self, "worker_rank", 0)),
                "worker_count": int(getattr(self, "worker_count", 1)),
                "shard_size": 0,
            }
            return

        records = self.candidate_bitmap_edge_records
        signature_text = "\n".join(str(record.get("edge_id", "")) for record in records)
        signature = self._candidate_bitmap_stable_digest(signature_text)
        if signature and signature == self.candidate_bitmap_ppo_shard_signature:
            return

        strata: Dict[Tuple[str, str, str, str, int, int], list[Tuple[str, int]]] = {}
        for fallback_index, record in enumerate(records):
            try:
                edge_index = int(record.get("edge_index", fallback_index))
            except Exception:
                edge_index = int(fallback_index)
            if edge_index < 0 or edge_index >= len(records):
                continue

            edge_id = str(record.get("edge_id", "") or "")
            if not edge_id:
                continue
            outer_node_id = str(record.get("outer_node_id", "") or "")
            inner_node_id = str(record.get("inner_node_id", "") or "")
            try:
                slot_index = int(record.get("slot_index", -1))
            except Exception:
                slot_index = -1
            try:
                initial_edge_score = float(record.get("edge_score", 0.0) or 0.0)
            except Exception:
                initial_edge_score = 0.0

            # A 0.25-wide initial score band balances score ranges without using
            # any runtime DB success/failure result. The cached assignment is not
            # rebuilt when representation or HNSW scores are refreshed.
            score_band = int(np.floor(initial_edge_score * 4.0))
            inner_kind = (
                "terminal"
                if self._candidate_bitmap_required_slot_count(inner_node_id) <= 0
                else "operator"
            )
            stratum_key = (
                self._candidate_bitmap_record_level(record),
                self._candidate_bitmap_record_group(record),
                inner_kind,
                outer_node_id,
                int(slot_index),
                int(score_band),
            )
            strata.setdefault(stratum_key, []).append(
                (self._candidate_bitmap_stable_digest(edge_id), edge_index)
            )

        owned_indices: list[int] = []
        worker_count = int(getattr(self, "worker_count", 1))
        worker_loads = [0 for _ in range(worker_count)]
        for stratum_key in sorted(strata):
            items = sorted(strata[stratum_key], key=lambda item: (item[0], int(item[1])))
            start_worker = int(
                self._candidate_bitmap_stable_digest(repr(stratum_key))[:16],
                16,
            ) % worker_count
            worker_order = [
                (start_worker + offset) % worker_count
                for offset in range(worker_count)
            ]
            for _digest, edge_index in items:
                min_load = min(worker_loads)
                # 每个结构层保持稳定轮转；总负载优先保证9324条边均匀落到32个环境。
                owner = next(worker for worker in worker_order if worker_loads[worker] == min_load)
                worker_loads[owner] += 1
                if owner == int(getattr(self, "worker_rank", 0)):
                    owned_indices.append(int(edge_index))

        owned_indices = sorted(set(owned_indices))
        self.candidate_bitmap_ppo_shard_indices = np.asarray(owned_indices, dtype=np.int32)
        self.candidate_bitmap_ppo_shard_index_set = set(owned_indices)
        self.candidate_bitmap_ppo_shard_signature = signature

        level_counts: Dict[str, int] = {}
        group_counts: Dict[str, int] = {}
        kind_counts = {"terminal": 0, "operator": 0}
        for edge_index in owned_indices:
            record = records[edge_index]
            level = self._candidate_bitmap_record_level(record)
            group = self._candidate_bitmap_record_group(record)
            inner_node_id = str(record.get("inner_node_id", "") or "")
            inner_kind = (
                "terminal"
                if self._candidate_bitmap_required_slot_count(inner_node_id) <= 0
                else "operator"
            )
            level_counts[level] = int(level_counts.get(level, 0)) + 1
            group_counts[group] = int(group_counts.get(group, 0)) + 1
            kind_counts[inner_kind] = int(kind_counts.get(inner_kind, 0)) + 1

        self.candidate_bitmap_ppo_shard_report = {
            "active": True,
            "worker_rank": int(getattr(self, "worker_rank", 0)),
            "worker_count": int(getattr(self, "worker_count", 1)),
            "shard_size": int(len(owned_indices)),
            "level_counts": level_counts,
            "group_counts": group_counts,
            "inner_kind_counts": kind_counts,
        }
        print(
            "[CandidateBitmap PPO Shard] "
            f"worker={getattr(self, 'worker_rank', 0)}/{getattr(self, 'worker_count', 1)}, "
            f"edges={len(owned_indices)}, levels={level_counts}, "
            f"groups={group_counts}, inner={kind_counts}",
            flush=True,
        )

    def _candidate_bitmap_apply_ppo_shard(self, indices: list[int]) -> list[int]:
        """Filter target candidates only; graph traversal remains globally shared."""
        if not self._candidate_bitmap_ppo_sharding_active():
            return [int(idx) for idx in indices]
        if not self.candidate_bitmap_ppo_shard_signature:
            self._rebuild_candidate_bitmap_ppo_shard_cache()
        owned = self.candidate_bitmap_ppo_shard_index_set
        return [int(idx) for idx in indices if int(idx) in owned]

    def _rebuild_candidate_bitmap_static_caches(self) -> None:
        """
        阶段17-14：重建 candidate bitmap 静态结构缓存。

        这些缓存只依赖：
        1. candidate_bitmap_edge_records；
        2. candidate_bitmap_formal_node_ids。

        不依赖当前 action、reward、DB 执行结果或 parser 结果，
        因此不会改变探索语义，只减少每步候选池打分时的重复扫描。
        """
        required_slot_count: Dict[str, int] = {}

        for record in self.candidate_bitmap_edge_records:
            outer_node_id = str(record.get("outer_node_id", ""))
            if not outer_node_id:
                continue

            try:
                slot_index = int(record.get("slot_index", -1))
            except Exception:
                continue

            if slot_index < 0:
                continue

            required_slot_count[outer_node_id] = max(
                int(required_slot_count.get(outer_node_id, 0)),
                int(slot_index) + 1,
            )

        self.candidate_bitmap_required_slot_count_cache = required_slot_count
        self.candidate_bitmap_support_rank_cache = {}
        self.candidate_bitmap_inner_ready_cache = {}

        support_rank_cache: Dict[str, int] = {}
        for record in self.candidate_bitmap_edge_records:
            edge_id = str(record.get("edge_id", ""))
            if not edge_id:
                continue
            support_rank_cache[edge_id] = int(self._candidate_bitmap_support_rank(record))

        self.candidate_bitmap_support_rank_cache = support_rank_cache

        # inner_ready 是节点级属性：终端节点直接就绪；非终端节点的每个 slot
        # 都至少存在一条可完整构造的正式图下游时才就绪。环路本身不作为就绪证明。
        nodes = self.candidate_bitmap_formal_graph_data.get("nodes", {})
        if isinstance(nodes, dict):
            inner_depth_limit = max(
                0,
                int(getattr(CANDIDATE_BITMAP_CONFIG, "probe_predicate_max_depth", 3)) - 1,
            )
            for node_id in nodes:
                self._candidate_bitmap_node_downstream_ready(
                    str(node_id),
                    inner_depth_limit,
                    set(),
                )

        self._rebuild_candidate_bitmap_ppo_shard_cache()

    def _candidate_bitmap_required_slot_count(self, node_id: str) -> int:
        node_key = str(node_id)
        if node_key in self.candidate_bitmap_required_slot_count_cache:
            return int(self.candidate_bitmap_required_slot_count_cache.get(node_key, 0))

        max_slot = -1
        for record in self.candidate_bitmap_edge_records:
            if str(record.get("outer_node_id", "")) != node_key:
                continue
            try:
                slot_index = int(record.get("slot_index", -1))
            except Exception:
                continue
            max_slot = max(max_slot, slot_index)
        return max_slot + 1 if max_slot >= 0 else 0

    def _candidate_bitmap_support_rank(self, record: Dict[str, Any]) -> int:
        edge_id = str(record.get("edge_id", ""))
        if edge_id and edge_id in self.candidate_bitmap_support_rank_cache:
            return int(self.candidate_bitmap_support_rank_cache.get(edge_id, 0))

        outer_node_id = str(record.get("outer_node_id", ""))
        inner_node_id = str(record.get("inner_node_id", ""))

        outer_preexisting = outer_node_id in self.candidate_bitmap_formal_node_ids
        inner_preexisting = inner_node_id in self.candidate_bitmap_formal_node_ids
        is_self_loop = outer_node_id == inner_node_id

        inner_terminal_like = self._candidate_bitmap_required_slot_count(inner_node_id) == 0

        if outer_preexisting and inner_preexisting and not is_self_loop:
            return 60
        if outer_preexisting and inner_preexisting:
            return 50
        if outer_preexisting and inner_terminal_like:
            return 45
        if outer_preexisting:
            return 35
        if inner_preexisting and not is_self_loop:
            return 25
        if not is_self_loop:
            return 10
        return 0

    def _candidate_bitmap_node_downstream_ready(
        self,
        node_id: str,
        depth_remaining: int,
        visiting: set[str],
    ) -> bool:
        """Return whether one complete subtree already exists in the formal graph."""
        node_key = str(node_id)
        depth_remaining = max(0, int(depth_remaining))
        cache_key = (node_key, depth_remaining)
        cache = getattr(self, "candidate_bitmap_inner_ready_cache", None)
        if not isinstance(cache, dict):
            cache = {}
            self.candidate_bitmap_inner_ready_cache = cache
        if cache_key in cache:
            return bool(cache[cache_key])
        if node_key in visiting:
            return False

        formal_graph_data = getattr(self, "candidate_bitmap_formal_graph_data", {})
        nodes = formal_graph_data.get("nodes", {}) if isinstance(formal_graph_data, dict) else {}
        payload = nodes.get(node_key) if isinstance(nodes, dict) else None
        if not isinstance(payload, dict):
            cache[cache_key] = False
            return False

        required_slot_count = self._candidate_bitmap_required_slot_count(node_key)
        if bool(payload.get("is_terminal", False)) or required_slot_count <= 0:
            cache[cache_key] = True
            return True
        if depth_remaining <= 0:
            cache[cache_key] = False
            return False

        slots = payload.get("slots", [])
        if not isinstance(slots, list) or len(slots) < required_slot_count:
            cache[cache_key] = False
            return False

        next_visiting = set(visiting)
        next_visiting.add(node_key)

        for slot_index in range(required_slot_count):
            slot_items = slots[slot_index]
            if not isinstance(slot_items, list) or not slot_items:
                cache[cache_key] = False
                return False

            slot_ready = any(
                self._candidate_bitmap_node_downstream_ready(
                    str(child_id),
                    depth_remaining - 1,
                    next_visiting,
                )
                for child_id in slot_items
            )
            if not slot_ready:
                cache[cache_key] = False
                return False

        cache[cache_key] = True
        return True

    def _candidate_bitmap_inner_ready(self, record: Dict[str, Any]) -> bool:
        """Return the bottom-up readiness signal for one candidate target inner."""
        outer_node_id = str(record.get("outer_node_id", ""))
        inner_node_id = str(record.get("inner_node_id", ""))
        if not inner_node_id or outer_node_id == inner_node_id:
            # 自环仍可被探索，但不获得“下游已经完整”的优先级和 PPO 特征。
            return False
        inner_depth_limit = max(
            0,
            int(getattr(CANDIDATE_BITMAP_CONFIG, "probe_predicate_max_depth", 3)) - 1,
        )
        return self._candidate_bitmap_node_downstream_ready(
            inner_node_id,
            inner_depth_limit,
            set(),
        )

    def _candidate_bitmap_record_level(self, record: Dict[str, Any]) -> str:
        return str(record.get("candidate_level", "") or "").lower()

    def _candidate_bitmap_record_group(self, record: Dict[str, Any]) -> str:
        group = str(record.get("edge_group", "") or "").strip().lower()
        return group if group else "other"

    def _candidate_bitmap_is_core_to_extension(self, record: Dict[str, Any]) -> bool:
        if bool(record.get("is_core_to_extension", False)):
            return True
        return self._candidate_bitmap_record_group(record) == "core_to_extension"

    def _candidate_bitmap_is_core_internal(self, record: Dict[str, Any]) -> bool:
        if bool(record.get("is_core_internal", False)):
            return True
        return self._candidate_bitmap_record_group(record) == "core_internal"

    def _candidate_bitmap_is_extension_outer(self, record: Dict[str, Any]) -> bool:
        if bool(record.get("is_extension_outer", False)):
            return True
        return self._candidate_bitmap_record_group(record) == "extension_outer"

    def _candidate_bitmap_group_fair_sampling_active(self) -> bool:
        if not bool(getattr(RL_ENV_CONFIG, "candidate_bitmap_group_fair_sampling_enabled", False)):
            return False

        current_step = int(getattr(self, "env_total_steps", getattr(self, "current_step", 0)))
        warmup_steps = int(getattr(RL_ENV_CONFIG, "candidate_bitmap_group_fair_sampling_warmup_steps", 0))
        return current_step >= max(0, warmup_steps)

    def _candidate_bitmap_group_fair_target_counts(self, pool_size: int) -> Dict[str, int]:
        raw_ratios = getattr(
            RL_ENV_CONFIG,
            "candidate_bitmap_group_fair_target_ratios",
            (
                ("core_internal", 0.34),
                ("core_to_extension", 0.33),
                ("extension_outer", 0.33),
            ),
        )

        ratio_items = []
        if isinstance(raw_ratios, dict):
            ratio_items = list(raw_ratios.items())
        else:
            try:
                ratio_items = list(raw_ratios)
            except Exception:
                ratio_items = []

        cleaned = []
        for item in ratio_items:
            try:
                group_name = str(item[0]).strip().lower()
                ratio = float(item[1])
            except Exception:
                continue

            if not group_name or ratio <= 0:
                continue

            cleaned.append((group_name, ratio))

        if not cleaned:
            cleaned = [
                ("core_internal", 0.34),
                ("core_to_extension", 0.33),
                ("extension_outer", 0.33),
            ]

        total_ratio = sum(ratio for _, ratio in cleaned)
        if total_ratio <= 0:
            return {}

        pool_size = max(0, int(pool_size))
        counts: Dict[str, int] = {}
        remainders = []

        used = 0
        for group_name, ratio in cleaned:
            exact = pool_size * (ratio / total_ratio)
            base = int(np.floor(exact))
            counts[group_name] = base
            used += base
            remainders.append((exact - base, group_name))

        remaining = max(0, pool_size - used)
        remainders.sort(key=lambda x: (-float(x[0]), str(x[1])))

        for _, group_name in remainders[:remaining]:
            counts[group_name] = int(counts.get(group_name, 0)) + 1

        return counts

    def _candidate_bitmap_core_extension_soft_priority_active(self) -> bool:
        if not bool(getattr(RL_ENV_CONFIG, "candidate_bitmap_core_extension_soft_priority_enabled", True)):
            return False

        current_step = int(getattr(self, "env_total_steps", getattr(self, "current_step", 0)))
        warmup_steps = int(getattr(RL_ENV_CONFIG, "candidate_bitmap_core_extension_warmup_steps", 2000))

        if current_step >= max(0, warmup_steps):
            return True

        if (
            bool(getattr(RL_ENV_CONFIG, "candidate_bitmap_core_extension_activate_on_frontier", True))
            and bool(getattr(RL_ENV_CONFIG, "frontier_mode_enabled", True))
            and bool(self.frontier_mode_active)
        ):
            return True

        return False

    def _candidate_bitmap_core_extension_priority_bonus(self, record: Dict[str, Any]) -> float:
        if not self._candidate_bitmap_core_extension_soft_priority_active():
            return 0.0

        if not self._candidate_bitmap_is_core_to_extension(record):
            return 0.0

        edge_id = str(record.get("edge_id", "") or "")
        if edge_id in self.candidate_bitmap_success_edges:
            return 0.0

        base_bonus = float(
            getattr(RL_ENV_CONFIG, "candidate_bitmap_core_extension_priority_bonus", 0.35)
        )
        pressure_bonus = float(
            getattr(RL_ENV_CONFIG, "candidate_bitmap_core_extension_pressure_bonus", 0.35)
        ) * float(np.clip(self._frontier_stagnation_pressure(), 0.0, 1.0))

        return float(max(0.0, base_bonus + pressure_bonus))

    def _candidate_bitmap_frontier_open_flags(self) -> Dict[str, Any]:
        enabled = bool(getattr(RL_ENV_CONFIG, "candidate_bitmap_frontier_open_enabled", True))
        current_step = int(getattr(self, "env_total_steps", getattr(self, "current_step", 0)))
        warmup_steps = int(getattr(RL_ENV_CONFIG, "candidate_bitmap_frontier_open_warmup_steps", 500))
        pressure = float(self._frontier_stagnation_pressure())

        weak_threshold = float(getattr(RL_ENV_CONFIG, "candidate_bitmap_frontier_open_weak_pressure", 0.50))
        try_threshold = float(getattr(RL_ENV_CONFIG, "candidate_bitmap_frontier_open_try_pressure", 0.85))

        allow_frontier_open = (
            enabled
            and bool(getattr(RL_ENV_CONFIG, "frontier_mode_enabled", True))
            and bool(self.frontier_mode_active)
            and current_step >= max(0, warmup_steps)
        )

        fair_enabled = bool(getattr(RL_ENV_CONFIG, "candidate_bitmap_fair_exploration_open_enabled", False))
        fair_warmup_steps = int(getattr(RL_ENV_CONFIG, "candidate_bitmap_fair_exploration_warmup_steps", 0))
        fair_try_warmup_steps = int(getattr(RL_ENV_CONFIG, "candidate_bitmap_fair_exploration_try_warmup_steps", 2000))

        fair_exploration_active = bool(
            fair_enabled
            and current_step >= max(0, fair_warmup_steps)
        )

        fair_weak_open = bool(
            fair_exploration_active
            and bool(getattr(RL_ENV_CONFIG, "candidate_bitmap_fair_exploration_open_weak", True))
        )

        fair_try_open = bool(
            fair_exploration_active
            and bool(getattr(RL_ENV_CONFIG, "candidate_bitmap_fair_exploration_open_try", True))
            and current_step >= max(0, fair_try_warmup_steps)
        )

        weak_open = bool(
            (allow_frontier_open and pressure >= weak_threshold)
            or fair_weak_open
        )
        try_open = bool(
            (allow_frontier_open and pressure >= try_threshold)
            or fair_try_open
        )

        allow_frontier_open = bool(allow_frontier_open or fair_exploration_active)

        return {
            "enabled": bool(enabled),
            "allow_frontier_open": bool(allow_frontier_open),
            "pressure": float(pressure),
            "weak_open": bool(weak_open),
            "try_open": bool(try_open),
            "fair_exploration_active": bool(fair_exploration_active),
            "fair_weak_open": bool(fair_weak_open),
            "fair_try_open": bool(fair_try_open),
            "weak_threshold": float(weak_threshold),
            "try_threshold": float(try_threshold),
            "warmup_steps": int(warmup_steps),
            "current_step": int(current_step),
        }

    def _candidate_bitmap_frontier_level_min_counts(self, pool_size: int) -> Dict[str, int]:
        flags = self._candidate_bitmap_frontier_open_flags()
        if not bool(flags.get("allow_frontier_open", False)):
            return {"weak": 0, "try": 0}

        if bool(flags.get("fair_exploration_active", False)):
            weak_ratio = float(
                getattr(RL_ENV_CONFIG, "candidate_bitmap_fair_exploration_weak_min_ratio", 0.20)
            )
            try_ratio = float(
                getattr(RL_ENV_CONFIG, "candidate_bitmap_fair_exploration_try_min_ratio", 0.05)
            )
        else:
            weak_ratio = float(getattr(RL_ENV_CONFIG, "candidate_bitmap_frontier_weak_min_ratio", 0.25))
            try_ratio = float(getattr(RL_ENV_CONFIG, "candidate_bitmap_frontier_try_min_ratio", 0.10))

        weak_count = int(round(max(0, pool_size) * max(0.0, min(1.0, weak_ratio)))) if flags.get("weak_open") else 0
        try_count = int(round(max(0, pool_size) * max(0.0, min(1.0, try_ratio)))) if flags.get("try_open") else 0

        if weak_count + try_count > pool_size:
            overflow = weak_count + try_count - pool_size
            weak_count = max(0, weak_count - overflow)

        return {
            "weak": int(weak_count),
            "try": int(try_count),
        }

    def _candidate_bitmap_runtime_universe_indices(self) -> list[int]:
        """Return the strict/weak/try universe, excluding only edges currently being probed."""
        n = len(self.candidate_bitmap_edge_records)
        if n <= 0:
            return []

        cached_done = np.asarray(
            getattr(self, "candidate_bitmap_exploration_done", np.zeros((0,), dtype=np.uint8)),
            dtype=np.uint8,
        )
        cached_probing = np.asarray(
            getattr(self, "candidate_bitmap_probing", np.zeros((0,), dtype=np.uint8)),
            dtype=np.uint8,
        )
        if len(cached_done) != n or len(cached_probing) != n:
            self._refresh_candidate_bitmap_exploration_state_cache(n)

        candidate_indices = np.asarray(
            getattr(self, "candidate_bitmap_candidate_union_indices", np.zeros((0,), dtype=np.int32)),
            dtype=np.int32,
        )
        probing = np.asarray(self.candidate_bitmap_probing, dtype=np.uint8)
        return [
            int(idx)
            for idx in candidate_indices.tolist()
            if 0 <= int(idx) < n and int(probing[int(idx)]) == 0
        ]

    def _candidate_bitmap_available_indices(self) -> list[int]:
        # 评分模式不再硬删除 observed/failed；统一通过 exploration_done 软降权。
        return self._candidate_bitmap_runtime_universe_indices()

    def _candidate_bitmap_random_available_indices(self) -> list[int]:
        """Return the full strict/weak/try universe for the pure-random base ablation."""
        return self._candidate_bitmap_runtime_universe_indices()

    def _candidate_bitmap_can_retry_failed_core_extension(self, record: Dict[str, Any]) -> bool:
        if not bool(getattr(RL_ENV_CONFIG, "candidate_bitmap_core_extension_retry_failed_enabled", True)):
            return False

        if not self._candidate_bitmap_core_extension_soft_priority_active():
            return False

        if not self._candidate_bitmap_is_core_to_extension(record):
            return False

        edge_id = str(record.get("edge_id", "") or "")
        if not edge_id:
            return False

        local_count = int(self.candidate_bitmap_soft_retry_counts.get(edge_id, 0))
        max_local = int(
            getattr(RL_ENV_CONFIG, "candidate_bitmap_core_extension_retry_failed_max_local_attempts", 1)
        )
        if local_count >= max(0, max_local):
            return False

        last_error = str(record.get("last_error", "") or "").lower()
        if not last_error:
            return False

        keywords = getattr(
            RL_ENV_CONFIG,
            "candidate_bitmap_core_extension_retry_error_keywords",
            (),
        )
        keyword_texts = [str(x).lower() for x in keywords]
        return any(keyword in last_error for keyword in keyword_texts if keyword)

    def _candidate_bitmap_can_retry_failed_fair(self, record: Dict[str, Any]) -> bool:
        if not bool(getattr(RL_ENV_CONFIG, "candidate_bitmap_fair_failed_retry_enabled", False)):
            return False

        edge_id = str(record.get("edge_id", "") or "")
        if not edge_id:
            return False

        local_count = int(self.candidate_bitmap_soft_retry_counts.get(edge_id, 0))
        max_local = int(
            getattr(RL_ENV_CONFIG, "candidate_bitmap_fair_failed_retry_max_local_attempts", 1)
        )
        if local_count >= max(0, max_local):
            return False

        groups = getattr(
            RL_ENV_CONFIG,
            "candidate_bitmap_fair_failed_retry_groups",
            ("core_internal", "core_to_extension", "extension_outer"),
        )
        allowed_groups = {str(x).strip().lower() for x in groups}
        if self._candidate_bitmap_record_group(record) not in allowed_groups:
            return False

        last_error = str(record.get("last_error", "") or "").lower()
        if not last_error:
            return False

        keywords = getattr(
            RL_ENV_CONFIG,
            "candidate_bitmap_fair_failed_retry_error_keywords",
            (),
        )
        keyword_texts = [str(x).lower() for x in keywords]
        return any(keyword in last_error for keyword in keyword_texts if keyword)

    def _candidate_bitmap_is_selected_failed_retry(
        self,
        record: Dict[str, Any],
        edge_index: int,
    ) -> bool:
        try:
            idx = int(edge_index)
        except Exception:
            return False

        failed_bitmap = np.asarray(
            self.candidate_bitmap_bitmaps.get(
                "failed_bitmap",
                np.zeros(len(self.candidate_bitmap_edge_records), dtype=np.uint8),
            ),
            dtype=np.uint8,
        )

        if idx < 0 or idx >= len(failed_bitmap):
            return False

        if int(failed_bitmap[idx]) <= 0:
            return False

        if bool(getattr(RL_ENV_CONFIG, "candidate_bitmap_fair_failed_retry_enabled", False)):
            return self._candidate_bitmap_can_retry_failed_fair(record)

        return self._candidate_bitmap_can_retry_failed_core_extension(record)

    def _candidate_bitmap_retry_failed_indices(self, pool_size: int) -> list[int]:
        fair_retry_enabled = bool(
            getattr(RL_ENV_CONFIG, "candidate_bitmap_fair_failed_retry_enabled", False)
        )

        if not fair_retry_enabled:
            if not bool(getattr(RL_ENV_CONFIG, "candidate_bitmap_core_extension_retry_failed_enabled", True)):
                return []

            if not self._candidate_bitmap_core_extension_soft_priority_active():
                return []

        records = self.candidate_bitmap_edge_records
        n = len(records)
        if n <= 0:
            return []

        strict = np.asarray(self.candidate_bitmap_bitmaps.get("strict_candidate_bitmap", np.zeros(n)), dtype=np.uint8)
        weak = np.asarray(self.candidate_bitmap_bitmaps.get("weak_candidate_bitmap", np.zeros(n)), dtype=np.uint8)
        try_bits = np.asarray(self.candidate_bitmap_bitmaps.get("try_candidate_bitmap", np.zeros(n)), dtype=np.uint8)
        observed = np.asarray(self.candidate_bitmap_bitmaps.get("observed_bitmap", np.zeros(n)), dtype=np.uint8)
        failed = np.asarray(self.candidate_bitmap_bitmaps.get("failed_bitmap", np.zeros(n)), dtype=np.uint8)
        probing = np.asarray(self.candidate_bitmap_bitmaps.get("probing_bitmap", np.zeros(n)), dtype=np.uint8)

        candidate_union = (strict > 0) | (weak > 0) | (try_bits > 0)
        if fair_retry_enabled:
            max_retry = int(
                getattr(RL_ENV_CONFIG, "candidate_bitmap_fair_failed_retry_max_per_pool", 6)
            )
        else:
            max_retry = int(
                getattr(RL_ENV_CONFIG, "candidate_bitmap_core_extension_retry_failed_max_per_pool", 4)
            )
        max_retry = max(0, min(max_retry, max(0, pool_size)))

        if max_retry <= 0:
            return []

        retry_items = []
        for idx in np.where((candidate_union > 0) & (observed == 0) & (failed > 0) & (probing == 0))[0].tolist():
            idx = int(idx)
            if idx < 0 or idx >= n:
                continue

            record = records[idx]
            if not isinstance(record, dict):
                continue

            if fair_retry_enabled:
                if not self._candidate_bitmap_can_retry_failed_fair(record):
                    continue
            else:
                if not self._candidate_bitmap_can_retry_failed_core_extension(record):
                    continue

            retry_items.append((self._candidate_bitmap_edge_priority(record), idx))

        retry_items.sort(key=lambda item: (-float(item[0]), int(item[1])))
        return [int(idx) for _, idx in retry_items[:max_retry]]

    def _candidate_bitmap_edge_priority(self, record: Dict[str, Any]) -> float:
        edge_id = str(record.get("edge_id", ""))
        support_rank = self._candidate_bitmap_support_rank(record)

        edge_score = self._candidate_bitmap_effective_edge_score(record)
        pattern_transfer_score = self._candidate_bitmap_effective_pattern_transfer_score(record)

        level_text = self._candidate_bitmap_record_level(record)
        level_value = self._candidate_bitmap_level_value(level_text)

        score = 0.0
        score += 2.0 * (support_rank / 60.0)
        score += 1.0 * level_value
        score += 0.6 * edge_score
        score += 0.2 * pattern_transfer_score
        score += self._candidate_bitmap_core_extension_priority_bonus(record)

        frontier_flags = self._candidate_bitmap_frontier_open_flags()
        if bool(frontier_flags.get("allow_frontier_open", False)):
            pressure = float(frontier_flags.get("pressure", 0.0))
            level_boost = float(
                getattr(RL_ENV_CONFIG, "candidate_bitmap_frontier_level_priority_boost", 1.20)
            )
            strict_penalty = float(
                getattr(RL_ENV_CONFIG, "candidate_bitmap_frontier_strict_pressure_penalty", 0.25)
            )

            if level_text == "weak" and bool(frontier_flags.get("weak_open", False)):
                score += level_boost * max(0.25, pressure)

            if level_text == "try" and bool(frontier_flags.get("try_open", False)):
                score += level_boost * max(0.25, pressure)

            if level_text == "strict" and (
                bool(frontier_flags.get("weak_open", False))
                or bool(frontier_flags.get("try_open", False))
            ):
                score -= strict_penalty * pressure

        if bool(getattr(CANDIDATE_BITMAP_CONFIG, "candidate_organization_enabled", True)):
            if bool(getattr(CANDIDATE_BITMAP_CONFIG, "candidate_hnsw_enabled", True)):
                # 0.5为中性；成功邻居多则加分，失败邻居多则减分。
                hnsw_tendency = self._candidate_bitmap_hnsw_tendency(record)
                hnsw_weight = float(
                    getattr(CANDIDATE_BITMAP_CONFIG, "candidate_hnsw_priority_weight", 1.0)
                )
                score += hnsw_weight * (2.0 * hnsw_tendency - 1.0)

            if bool(getattr(CANDIDATE_BITMAP_CONFIG, "candidate_entropy_enabled", True)):
                # 前期优先低熵简单边；停滞压力升高后逐渐放开复杂边。
                entropy = self._candidate_bitmap_structural_entropy(record)
                entropy_weight = float(
                    getattr(CANDIDATE_BITMAP_CONFIG, "candidate_entropy_priority_weight", 1.0)
                )
                pressure = float(
                    frontier_flags.get("pressure", self._frontier_stagnation_pressure())
                )
                score -= entropy_weight * (1.0 - np.clip(pressure, 0.0, 1.0)) * entropy

        # 自底向上信号只影响评分模式。纯随机 base 不会调用本评分函数。
        if self._candidate_bitmap_inner_ready(record):
            score += float(
                getattr(RL_ENV_CONFIG, "candidate_bitmap_inner_ready_priority_weight", 1.0)
            )

        try:
            edge_index = int(record.get("edge_index", -1))
        except Exception:
            edge_index = -1

        # 成功和确认失败统一视为探索结束，只做软降权，不从候选池硬排除。
        if self._candidate_bitmap_exploration_done_for_index(edge_index, edge_id):
            score += float(
                getattr(RL_ENV_CONFIG, "candidate_bitmap_exploration_done_penalty", -1.0)
            )

        # 本地测试过只轻微降权；失败未达到确认阈值时仍然可以继续探索。
        if edge_id in self.candidate_bitmap_tested_edges:
            score += float(
                getattr(RL_ENV_CONFIG, "candidate_bitmap_local_tested_penalty", -0.2)
            )

        score += self._candidate_bitmap_probe_variant_exhausted_priority_penalty(edge_id)

        jitter = float(getattr(RL_ENV_CONFIG, "mutation_pool_priority_jitter", 0.02))
        if jitter > 0:
            score += float(self.mutation_rng.normal(0.0, jitter))

        return float(score)

    def _candidate_bitmap_refresh_probe_variant_exhausted_window(self) -> None:
        """Reset local exhaustion counts when the configured global-step window changes."""
        reset_steps = max(
            1,
            int(
                getattr(
                    RL_ENV_CONFIG,
                    "candidate_bitmap_probe_variant_exhausted_reset_steps",
                    40000,
                )
            ),
        )
        worker_count = max(1, int(getattr(self, "worker_count", 1)))
        local_steps = max(0, int(getattr(self, "env_total_steps", 0)))
        window_index = (local_steps * worker_count) // reset_steps
        if window_index == int(
            getattr(self, "candidate_bitmap_probe_variant_exhausted_window_index", -1)
        ):
            return

        self.candidate_bitmap_probe_variant_exhausted_counts = {}
        self.candidate_bitmap_probe_variant_exhausted_window_index = int(window_index)

    def _candidate_bitmap_record_probe_variant_exhausted(
        self,
        edge_id: str,
        construct_error: str,
    ) -> None:
        """Count only the exact no-new-predicate condition in the current local window."""
        if str(construct_error or "").strip().lower() != "probe_variant_exhausted":
            return

        edge_key = str(edge_id or "")
        if not edge_key:
            return

        self._candidate_bitmap_refresh_probe_variant_exhausted_window()
        counts = getattr(self, "candidate_bitmap_probe_variant_exhausted_counts", {})
        counts[edge_key] = int(counts.get(edge_key, 0)) + 1
        self.candidate_bitmap_probe_variant_exhausted_counts = counts

    def _candidate_bitmap_probe_variant_exhausted_priority_penalty(
        self,
        edge_id: str,
    ) -> float:
        """Return one fixed soft penalty after the local exhaustion threshold is reached."""
        self._candidate_bitmap_refresh_probe_variant_exhausted_window()
        threshold = max(
            1,
            int(
                getattr(
                    RL_ENV_CONFIG,
                    "candidate_bitmap_probe_variant_exhausted_threshold",
                    15,
                )
            ),
        )
        counts = getattr(self, "candidate_bitmap_probe_variant_exhausted_counts", {})
        if int(counts.get(str(edge_id or ""), 0)) < threshold:
            return 0.0
        return float(
            getattr(
                RL_ENV_CONFIG,
                "candidate_bitmap_probe_variant_exhausted_penalty",
                -1.5,
            )
        )

    def _sample_candidate_bitmap_pool(
        self,
        excluded_edge_ids: Optional[set[str]] = None,
    ) -> list[Optional[Dict[str, Any]]]:
        pool_size = int(self.candidate_bitmap_pool_size)
        excluded = {str(edge_id) for edge_id in (excluded_edge_ids or set()) if str(edge_id)}
        if not self.candidate_bitmap_edge_records:
            self.current_candidate_bitmap_pool_unfinished_count = 0
            return [None for _ in range(pool_size)]

        def _is_unfinished(record: Optional[Dict[str, Any]]) -> bool:
            if not isinstance(record, dict):
                return False
            try:
                edge_index = int(record.get("edge_index", -1))
            except Exception:
                edge_index = -1
            edge_id = str(record.get("edge_id", "") or "")
            return not self._candidate_bitmap_exploration_done_for_index(edge_index, edge_id)

        if bool(getattr(RL_ENV_CONFIG, "candidate_bitmap_random_pool_enabled", False)):
            indices = self._candidate_bitmap_apply_ppo_shard([
                idx
                for idx in self._candidate_bitmap_random_available_indices()
                if str(self.candidate_bitmap_edge_records[int(idx)].get("edge_id", ""))
                not in excluded
            ])
            if STAGE2_MODE == "base" and float(self.mutation_rng.random()) < 0.75:
                done = np.asarray(self.candidate_bitmap_exploration_done, dtype=bool).copy()
                for edge_id in self.candidate_bitmap_success_edges:
                    edge_index = self.candidate_bitmap_edge_id_to_index.get(edge_id, -1)
                    if 0 <= edge_index < len(done):
                        done[edge_index] = True
                finished_indices = [idx for idx in indices if done[idx]]
                if finished_indices:
                    indices = finished_indices
            sample_count = min(pool_size, len(indices))
            if sample_count > 0:
                sampled_indices = self.mutation_rng.choice(
                    np.asarray(indices, dtype=np.int64),
                    size=sample_count,
                    replace=False,
                ).tolist()
                selected = [
                    self.candidate_bitmap_edge_records[int(idx)]
                    for idx in sampled_indices
                ]
            else:
                selected = []

            if len(selected) < pool_size:
                if selected and bool(getattr(RL_ENV_CONFIG, "candidate_bitmap_allow_pool_padding", True)):
                    cursor = 0
                    while len(selected) < pool_size:
                        selected.append(selected[cursor % len(selected)])
                        cursor += 1
                else:
                    selected.extend([None] * (pool_size - len(selected)))

            self.current_candidate_bitmap_pool_unfinished_count = sum(
                1 for record in selected[:pool_size] if _is_unfinished(record)
            )
            return selected[:pool_size]

        indices = self._candidate_bitmap_apply_ppo_shard([
            idx
            for idx in self._candidate_bitmap_available_indices()
            if str(self.candidate_bitmap_edge_records[int(idx)].get("edge_id", ""))
            not in excluded
        ])
        configured_scored_count = max(
            0,
            int(getattr(RL_ENV_CONFIG, "candidate_bitmap_scored_pool_count", 48)),
        )
        configured_random_count = max(
            0,
            int(getattr(RL_ENV_CONFIG, "candidate_bitmap_random_pool_count", 16)),
        )
        scored_pool_count = min(pool_size, configured_scored_count)
        random_pool_count = min(
            max(0, pool_size - scored_pool_count),
            configured_random_count,
        )
        if scored_pool_count + random_pool_count < pool_size:
            random_pool_count += pool_size - scored_pool_count - random_pool_count

        scored = []
        for idx in indices:
            if idx < 0 or idx >= len(self.candidate_bitmap_edge_records):
                continue
            record = self.candidate_bitmap_edge_records[idx]
            scored.append((self._candidate_bitmap_edge_priority(record), idx, record))

        scored.sort(
            key=lambda item: (
                -float(item[0]),
                int(item[1]),
            )
        )

        selected_items = []
        selected_indices = set()

        level_min_counts = self._candidate_bitmap_frontier_level_min_counts(scored_pool_count)
        if any(int(v) > 0 for v in level_min_counts.values()):
            for level_name in ("try", "weak"):
                target_count = int(level_min_counts.get(level_name, 0))
                if target_count <= 0:
                    continue

                level_items = [
                    item for item in scored
                    if self._candidate_bitmap_record_level(item[2]) == level_name
                ]

                for item in level_items[:target_count]:
                    if len(selected_items) >= scored_pool_count:
                        break
                    idx = int(item[1])
                    if idx in selected_indices:
                        continue
                    selected_items.append(item)
                    selected_indices.add(idx)

        def _add_group_items(group_name: str, target_count: int, *, max_count: Optional[int] = None) -> None:
            if target_count <= 0 and (max_count is None or max_count <= 0):
                return

            group_name = str(group_name).strip().lower()
            existing_count = sum(
                1 for item in selected_items
                if self._candidate_bitmap_record_group(item[2]) == group_name
            )

            added = 0
            group_items = [
                item for item in scored
                if self._candidate_bitmap_record_group(item[2]) == group_name
            ]

            limit = max(0, int(target_count) - int(existing_count))
            if max_count is not None:
                limit = min(limit, int(max_count))

            if limit <= 0:
                return

            for item in group_items:
                if len(selected_items) >= scored_pool_count:
                    break
                if added >= limit:
                    break

                idx = int(item[1])
                if idx in selected_indices:
                    continue

                selected_items.append(item)
                selected_indices.add(idx)
                added += 1

        if self._candidate_bitmap_group_fair_sampling_active():
            group_target_counts = self._candidate_bitmap_group_fair_target_counts(scored_pool_count)
            for group_name, target_count in group_target_counts.items():
                _add_group_items(group_name, int(target_count))

        elif self._candidate_bitmap_core_extension_soft_priority_active():
            core_min_ratio = float(
                getattr(RL_ENV_CONFIG, "candidate_bitmap_core_internal_min_ratio", 0.60)
            )
            core_ext_ratio = float(
                getattr(RL_ENV_CONFIG, "candidate_bitmap_core_to_extension_target_ratio", 0.25)
            )
            ext_outer_max_ratio = float(
                getattr(RL_ENV_CONFIG, "candidate_bitmap_extension_outer_max_ratio", 0.15)
            )

            core_min_count = int(round(scored_pool_count * max(0.0, min(1.0, core_min_ratio))))
            core_ext_target_count = int(round(scored_pool_count * max(0.0, min(1.0, core_ext_ratio))))
            ext_outer_max_count = int(round(scored_pool_count * max(0.0, min(1.0, ext_outer_max_ratio))))

            _add_group_items("core_internal", core_min_count)
            _add_group_items("core_to_extension", core_ext_target_count)
            _add_group_items("extension_outer", ext_outer_max_count)

        for item in scored:
            if len(selected_items) >= scored_pool_count:
                break
            idx = int(item[1])
            if idx in selected_indices:
                continue
            selected_items.append(item)
            selected_indices.add(idx)

        selected_items = selected_items[:scored_pool_count]

        # 16 条随机候选从评分部分尚未选择的全集中等概率补入。
        remaining_indices = [idx for idx in indices if idx not in selected_indices]
        random_sample_count = min(random_pool_count, len(remaining_indices))
        if random_sample_count > 0:
            sampled_random_indices = self.mutation_rng.choice(
                np.asarray(remaining_indices, dtype=np.int64),
                size=random_sample_count,
                replace=False,
            ).tolist()
            for idx in sampled_random_indices:
                idx = int(idx)
                record = self.candidate_bitmap_edge_records[idx]
                selected_items.append((self._candidate_bitmap_edge_priority(record), idx, record))
                selected_indices.add(idx)

        # PPO 至少看到约 8 条尚未探索结束的边；全局不足 8 条时使用全部可用未结束边。
        configured_min_unfinished = max(
            0,
            int(getattr(RL_ENV_CONFIG, "candidate_bitmap_min_unfinished_per_pool", 8)),
        )
        eligible_indices = set(indices)
        cached_unfinished_indices = np.asarray(
            getattr(self, "candidate_bitmap_unfinished_indices", np.zeros((0,), dtype=np.int32)),
            dtype=np.int32,
        )
        unfinished_available = [
            int(idx)
            for idx in cached_unfinished_indices.tolist()
            if int(idx) in eligible_indices
            and _is_unfinished(self.candidate_bitmap_edge_records[int(idx)])
        ]
        unfinished_target = min(pool_size, configured_min_unfinished, len(unfinished_available))
        selected_unfinished_count = sum(1 for item in selected_items if _is_unfinished(item[2]))

        if selected_unfinished_count < unfinished_target:
            replacement_candidates = [
                (
                    self._candidate_bitmap_edge_priority(self.candidate_bitmap_edge_records[idx]),
                    idx,
                    self.candidate_bitmap_edge_records[idx],
                )
                for idx in unfinished_available
                if idx not in selected_indices
            ]
            replacement_candidates.sort(key=lambda item: (-float(item[0]), int(item[1])))

            replace_positions = sorted(
                [
                    pos
                    for pos, item in enumerate(selected_items)
                    if not _is_unfinished(item[2])
                ],
                key=lambda pos: (float(selected_items[pos][0]), int(selected_items[pos][1])),
            )
            needed = unfinished_target - selected_unfinished_count
            for position, replacement in zip(
                replace_positions[:needed],
                replacement_candidates[:needed],
            ):
                selected_indices.discard(int(selected_items[position][1]))
                selected_items[position] = replacement
                selected_indices.add(int(replacement[1]))

        # 打乱 64 个位置，避免评分候选和随机候选形成固定位置偏差。
        if len(selected_items) > 1:
            order = self.mutation_rng.permutation(len(selected_items)).tolist()
            selected_items = [selected_items[int(pos)] for pos in order]

        selected = [item[2] for item in selected_items[:pool_size]]

        if len(selected) < pool_size:
            if selected and bool(getattr(RL_ENV_CONFIG, "candidate_bitmap_allow_pool_padding", True)):
                cursor = 0
                while len(selected) < pool_size:
                    selected.append(selected[cursor % len(selected)])
                    cursor += 1
            else:
                selected.extend([None] * (pool_size - len(selected)))

        self.current_candidate_bitmap_pool_unfinished_count = sum(
            1 for record in selected[:pool_size] if _is_unfinished(record)
        )
        return selected[:pool_size]

    def _candidate_bitmap_edge_features(self, record: Dict[str, Any]) -> np.ndarray:
        dim = int(self.candidate_bitmap_feature_dim)
        feat = np.zeros((dim,), dtype=np.float32)

        edge_id = str(record.get("edge_id", ""))
        outer_node_id = str(record.get("outer_node_id", ""))
        inner_node_id = str(record.get("inner_node_id", ""))

        edge_score = self._candidate_bitmap_effective_edge_score(record)
        pattern_transfer_score = self._candidate_bitmap_effective_pattern_transfer_score(record)

        try:
            slot_index = int(record.get("slot_index", 0))
        except Exception:
            slot_index = 0

        support_rank = self._candidate_bitmap_support_rank(record)

        edge_score_norm = float(
            getattr(CANDIDATE_BITMAP_CONFIG, "strict_base_score", 1.0)
            + getattr(CANDIDATE_BITMAP_CONFIG, "pattern_transfer_score_weight", 0.30)
        )
        edge_score_norm = max(edge_score_norm, 1e-6)

        if dim > 0:
            feat[0] = float(np.clip(edge_score / edge_score_norm, 0.0, 1.0))
        if dim > 1:
            feat[1] = float(np.clip(pattern_transfer_score, 0.0, 1.0))
        if dim > 2:
            feat[2] = float(self._candidate_bitmap_level_value(str(record.get("candidate_level", ""))))
        if dim > 3:
            feat[3] = 1.0 if outer_node_id in self.candidate_bitmap_formal_node_ids else 0.0
        if dim > 4:
            feat[4] = 1.0 if inner_node_id in self.candidate_bitmap_formal_node_ids else 0.0
        if dim > 5:
            feat[5] = 1.0 if outer_node_id == inner_node_id else 0.0
        if dim > 6:
            feat[6] = float(np.clip(slot_index / 4.0, 0.0, 1.0))
        if dim > 7:
            feat[7] = float(np.clip(support_rank / 60.0, 0.0, 1.0))
        if dim > 8:
            try:
                edge_index = int(record.get("edge_index", -1))
            except Exception:
                edge_index = -1
            feat[8] = (
                1.0
                if self._candidate_bitmap_exploration_done_for_index(edge_index, edge_id)
                else 0.0
            )
        if dim > 9:
            feat[9] = 1.0 if self._candidate_bitmap_probing_for_index(edge_index) else 0.0
        if dim > 10:
            feat[10] = 1.0 if edge_id in self.candidate_bitmap_tested_edges else 0.0
        if dim > 11:
            frontier_flags = self._candidate_bitmap_frontier_open_flags()
            pressure = float(frontier_flags.get("pressure", self._frontier_stagnation_pressure()))
            if bool(frontier_flags.get("try_open", False)):
                feat[11] = float(np.clip(0.75 + 0.25 * pressure, 0.0, 1.0))
            elif bool(frontier_flags.get("weak_open", False)):
                feat[11] = float(np.clip(0.50 + 0.25 * pressure, 0.0, 1.0))
            else:
                feat[11] = float(np.clip(pressure, 0.0, 1.0))
        if dim > 12:
            feat[12] = self._candidate_bitmap_hnsw_tendency(record)
        if dim > 13:
            feat[13] = self._candidate_bitmap_structural_entropy(record)
        if dim > 14:
            feat[14] = 1.0 if self._candidate_bitmap_inner_ready(record) else 0.0

        return feat.astype(np.float32)

    def _build_obs_candidate_bitmap(self) -> np.ndarray:
        step_clock = int(getattr(self, "env_total_steps", getattr(self, "current_step", 0)))
        reload_interval = int(getattr(RL_ENV_CONFIG, "candidate_bitmap_reload_interval_steps", 200))

        if (
            not self.candidate_bitmap_edge_records
            or (
                reload_interval > 0
                and step_clock - int(self.candidate_bitmap_last_reload_step) >= reload_interval
            )
        ):
            self._reload_candidate_bitmap_state(force=True)

        self._sync_candidate_shared_state()

        if self._candidate_bitmap_enumeration_active():
            assigned_edge, enumeration_idle, _ = (
                self._candidate_bitmap_enumeration_assignment()
            )
            self.candidate_bitmap_enumeration_idle = bool(enumeration_idle)
            self.current_candidate_bitmap_pool = [assigned_edge] + [
                None
                for _ in range(max(0, self.candidate_bitmap_pool_size - 1))
            ]
            self.current_candidate_bitmap_pool_unfinished_count = 0
            if isinstance(assigned_edge, dict):
                try:
                    assigned_index = int(assigned_edge.get("edge_index", -1))
                except Exception:
                    assigned_index = -1
                assigned_id = str(assigned_edge.get("edge_id", "") or "")
                self.current_candidate_bitmap_pool_unfinished_count = int(
                    not self._candidate_bitmap_exploration_done_for_index(
                        assigned_index,
                        assigned_id,
                    )
                )

            # PPO外壳仍要求固定64xfeature_dim观测；枚举动作被忽略，因此使用全零占位，
            # 不调用任何候选评分、表示学习、HNSW或结构熵特征。
            features = np.zeros(
                (
                    self.candidate_bitmap_pool_size,
                    self.candidate_bitmap_feature_dim,
                ),
                dtype=np.float32,
            )
            self.current_candidate_bitmap_features = features
            return features.reshape(-1).astype(np.float32)

        # 每个环境每一步只生成一次64条候选。PPO目标边分片已经避免整池冲突，
        # 不再为预占失败反复遍历、评分和排序候选全集。
        self.current_candidate_bitmap_pool = self._sample_candidate_bitmap_pool()

        features = np.zeros(
            (
                self.candidate_bitmap_pool_size,
                self.candidate_bitmap_feature_dim,
            ),
            dtype=np.float32,
        )

        for i, record in enumerate(self.current_candidate_bitmap_pool):
            if record is None:
                continue
            features[i] = self._candidate_bitmap_edge_features(record)

        self.current_candidate_bitmap_features = features
        return features.reshape(-1).astype(np.float32)

    def _select_candidate_bitmap_edge_from_action(self, action: Any) -> Optional[Dict[str, Any]]:
        if not self.current_candidate_bitmap_pool:
            return None

        try:
            idx = int(np.asarray(action).reshape(-1)[0])
        except Exception:
            idx = 0

        if idx < 0 or idx >= len(self.current_candidate_bitmap_pool):
            return None

        return self.current_candidate_bitmap_pool[idx]

    def _candidate_bitmap_try_reserve_edge(self, edge_id: str, edge_index: int) -> Dict[str, Any]:
        """Legacy fallback: atomically reserve only the PPO-selected target edge."""
        reservations = getattr(self, "shared_reservations", None)
        lock = getattr(self, "reservation_lock", None)
        edge_key = str(edge_id or "")
        if not edge_key or reservations is None or lock is None:
            return {"token": "", "conflict_count": 0, "expired_count": 0}

        timeout = max(
            1.0,
            float(getattr(CANDIDATE_BITMAP_CONFIG, "candidate_bitmap_probe_reservation_timeout_seconds", 300.0)),
        )
        now = float(time.time())
        token = f"pid={os.getpid()}:{time.time_ns()}"
        expired_count = 0
        lock.acquire()
        try:
            existing = dict(reservations.get(edge_key, {}) or {})
            if existing:
                try:
                    claim_time = float(existing.get("claim_time", 0.0))
                except Exception:
                    claim_time = 0.0
                if claim_time > 0.0 and now - claim_time < timeout:
                    return {"token": "", "conflict_count": 1, "expired_count": 0}
                reservations.pop(edge_key, None)
                expired_count = 1

            reservations[edge_key] = {
                "token": token,
                "owner": f"pid={os.getpid()}",
                "edge_index": int(edge_index),
                "claim_time": now,
                "env_step": int(getattr(self, "env_total_steps", getattr(self, "current_step", 0))),
            }
            return {"token": token, "conflict_count": 0, "expired_count": expired_count}
        finally:
            lock.release()

    def _candidate_bitmap_select_and_reserve_edge(
        self,
        action_index: int,
        selected_edge: Optional[Dict[str, Any]],
    ) -> Tuple[Optional[Dict[str, Any]], int, bool, int, bool]:
        """Reserve only PPO's final target edge; base deliberately remains unreserved."""
        self.current_candidate_bitmap_selected_reservation_token = ""
        self.current_candidate_bitmap_reservation_attempts = 0

        if self._candidate_bitmap_enumeration_active():
            return selected_edge, int(action_index), False, 0, False

        # PPO=False is the full-graph random base: no sharding and no hard coordination.
        if not bool(getattr(RL_ENV_CONFIG, "candidate_bitmap_ppo_selection_enabled", True)):
            return selected_edge, int(action_index), False, 0, False
        if not bool(getattr(CANDIDATE_BITMAP_CONFIG, "candidate_bitmap_shared_reservation_enabled", True)):
            return selected_edge, int(action_index), False, 0, False
        if not isinstance(selected_edge, dict):
            return selected_edge, int(action_index), False, 0, False
        if (
            getattr(self, "candidate_shared_state", None) is None
            and getattr(self, "shared_reservations", None) is None
        ):
            return selected_edge, int(action_index), False, 0, False

        selected_edge_id = str(selected_edge.get("edge_id", "") or "")
        try:
            selected_edge_index = int(selected_edge.get("edge_index", -1))
        except Exception:
            selected_edge_index = -1
        if not selected_edge_id or selected_edge_index < 0:
            return selected_edge, int(action_index), False, 0, False

        self.current_candidate_bitmap_reservation_attempts = 1
        timeout = max(
            1.0,
            float(getattr(CANDIDATE_BITMAP_CONFIG, "candidate_bitmap_probe_reservation_timeout_seconds", 300.0)),
        )
        shared_state = getattr(self, "candidate_shared_state", None)
        if shared_state is not None:
            reserve_report = shared_state.try_reserve_edge(
                selected_edge_id,
                selected_edge_index,
                owner=f"worker={self.worker_rank},pid={os.getpid()}",
                timeout_seconds=timeout,
            )
        else:
            reserve_report = self._candidate_bitmap_try_reserve_edge(
                selected_edge_id,
                selected_edge_index,
            )

        selected_token = str(reserve_report.get("token", "") or "")
        conflicts = int(reserve_report.get("conflict_count", 0) or 0)
        expired_count = int(reserve_report.get("expired_count", 0) or 0)
        self.reservation_timeout_release_count += expired_count
        self.reservation_conflict_count += conflicts
        if selected_token:
            self.reservation_claimed_count += 1
        self.current_candidate_bitmap_selected_reservation_token = selected_token

        # The policy action and executed edge must stay identical. A missing
        # ownership token is a neutral skipped action; never execute B while PPO
        # is trained as if it selected A, and never count the collision as failure.
        if not selected_token:
            return None, int(action_index), False, conflicts, False

        return selected_edge, int(action_index), False, conflicts, True

    def _resolve_candidate_bitmap_action_index(self, action: Any) -> Tuple[int, int, str]:
        """Return PPO-proposed index, executed index, and the action source."""
        try:
            policy_action_index = int(np.asarray(action).reshape(-1)[0])
        except Exception:
            policy_action_index = 0

        if self._candidate_bitmap_enumeration_active():
            if bool(getattr(self, "candidate_bitmap_enumeration_idle", False)):
                return policy_action_index, -1, "enumeration_idle"
            return policy_action_index, 0, "enumeration"

        if bool(getattr(RL_ENV_CONFIG, "candidate_bitmap_ppo_selection_enabled", True)):
            return policy_action_index, policy_action_index, "ppo"

        # PPO 关闭时按 edge_id 去除池填充产生的重复项，保证每条有效边机会相同。
        valid_indices = []
        seen_edge_ids = set()
        for idx, record in enumerate(self.current_candidate_bitmap_pool or []):
            if not isinstance(record, dict):
                continue

            edge_id = str(record.get("edge_id", "") or "")
            dedupe_key = edge_id if edge_id else f"__pool_index_{idx}"
            if dedupe_key in seen_edge_ids:
                continue

            seen_edge_ids.add(dedupe_key)
            valid_indices.append(int(idx))

        if not valid_indices:
            return policy_action_index, -1, "random"

        valid_indices_array = np.asarray(valid_indices, dtype=np.int64)

        # 纯随机 base 的64条候选和最终64选1都保持均匀随机，不读取任何评分引导开关。
        if bool(getattr(RL_ENV_CONFIG, "candidate_bitmap_random_pool_enabled", False)):
            executed_action_index = int(self.mutation_rng.choice(valid_indices_array))
            return policy_action_index, executed_action_index, "random"

        representation_active = bool(
            getattr(
                CANDIDATE_BITMAP_CONFIG,
                "candidate_representation_guidance_enabled",
                True,
            )
        )
        organization_active = bool(
            getattr(CANDIDATE_BITMAP_CONFIG, "candidate_organization_enabled", True)
        ) and (
            bool(getattr(CANDIDATE_BITMAP_CONFIG, "candidate_hnsw_enabled", True))
            or bool(getattr(CANDIDATE_BITMAP_CONFIG, "candidate_entropy_enabled", True))
        )

        # 所有指导信号关闭时保留均匀随机兜底。
        if not representation_active and not organization_active:
            executed_action_index = int(self.mutation_rng.choice(valid_indices_array))
            return policy_action_index, executed_action_index, "random"

        try:
            guided_probability = float(
                getattr(
                    RL_ENV_CONFIG,
                    "candidate_bitmap_guided_selection_probability",
                    0.70,
                )
            )
        except Exception:
            guided_probability = 0.70
        if not np.isfinite(guided_probability):
            guided_probability = 0.0
        guided_probability = float(np.clip(guided_probability, 0.0, 1.0))

        # 引导模式仍保留均匀随机探索，避免候选长期集中在少数高分边。
        if float(self.mutation_rng.random()) >= guided_probability:
            executed_action_index = int(self.mutation_rng.choice(valid_indices_array))
            return policy_action_index, executed_action_index, "guided_random"

        try:
            scores = np.asarray(
                [
                    self._candidate_bitmap_edge_priority(
                        self.current_candidate_bitmap_pool[int(idx)]
                    )
                    for idx in valid_indices
                ],
                dtype=np.float64,
            )
            score_min = float(np.min(scores))
            score_max = float(np.max(scores))
            score_range = score_max - score_min
            temperature = float(
                getattr(
                    RL_ENV_CONFIG,
                    "candidate_bitmap_guided_selection_temperature",
                    0.50,
                )
            )

            if (
                not np.all(np.isfinite(scores))
                or not np.isfinite(score_range)
                or score_range <= 0.0
                or not np.isfinite(temperature)
                or temperature <= 0.0
            ):
                raise ValueError("guided selection scores or temperature are invalid")

            normalized_scores = (scores - score_min) / score_range
            logits = normalized_scores / temperature
            logits -= float(np.max(logits))
            weights = np.exp(logits)
            weight_sum = float(np.sum(weights))
            if not np.all(np.isfinite(weights)) or not np.isfinite(weight_sum) or weight_sum <= 0.0:
                raise ValueError("guided selection softmax weights are invalid")

            probabilities = weights / weight_sum
            executed_action_index = int(
                self.mutation_rng.choice(valid_indices_array, p=probabilities)
            )
            return policy_action_index, executed_action_index, "guided_softmax"
        except Exception:
            # 同分、NaN、非法温度或 softmax 异常均退回当前64条中的均匀随机。
            executed_action_index = int(self.mutation_rng.choice(valid_indices_array))
            return policy_action_index, executed_action_index, "guided_random"

    def _candidate_bitmap_runtime_paths(self, edge_index: int) -> Dict[str, Path]:
        base_dir = Path(
            getattr(
                RL_ENV_CONFIG,
                "candidate_bitmap_runtime_probe_dir",
                Path(getattr(CANDIDATE_BITMAP_CONFIG, "bitmap_dir")) / "runtime_probe",
            )
        )

        step_id = int(getattr(self, "env_total_steps", getattr(self, "current_step", 0)))
        run_dir = base_dir / f"pid_{os.getpid()}" / f"step_{step_id}_edge_{int(edge_index)}"
        run_dir.mkdir(parents=True, exist_ok=True)

        return {
            "dir": run_dir,
            "probe_graph": run_dir / "probe_graph.json",
            "probe_graph_report": run_dir / "probe_graph_report.json",
            "probe_predicate": run_dir / "probe_predicate.txt",
            "probe_predicate_report": run_dir / "probe_predicate_report.json",
        }

    def _candidate_bitmap_should_persist_probe_artifacts(self) -> bool:
        """
        Stage17-11: formal training does not persist per-step probe artifacts by default.
        """
        enabled = bool(
            getattr(RL_ENV_CONFIG, "candidate_bitmap_runtime_persist_probe_artifacts", False)
        )
        if not enabled:
            return False

        sample_every = int(
            getattr(RL_ENV_CONFIG, "candidate_bitmap_runtime_probe_artifact_sample_every", 0)
        )

        if sample_every <= 1:
            return True

        step_id = int(getattr(self, "env_total_steps", getattr(self, "current_step", 0)))
        return step_id % sample_every == 0

    def _candidate_bitmap_probe_cache_get(self, edge_id: str) -> Optional[Dict[str, Any]]:
        key = str(edge_id or "")
        if not key:
            return None
        item = self.candidate_bitmap_probe_template_cache.get(key)
        if not isinstance(item, dict):
            return None
        return dict(item)

    def _candidate_bitmap_sync_probe_variants(self, selected_edge: Dict[str, Any]) -> None:
        edge_id = str(selected_edge.get("edge_id", ""))
        shared_state = getattr(self, "candidate_shared_state", None)
        if shared_state is not None:
            selected_edge["tried_probe_signatures"] = shared_state.get_failure_signatures(edge_id)
            selected_edge["completion_variant_index"] = max(
                int(selected_edge.get("completion_variant_index", 0) or 0),
                shared_state.get_completion_variant(edge_id),
            )
        cached_item = self.candidate_bitmap_probe_template_cache.get(edge_id)
        if not isinstance(cached_item, dict):
            return
        signature = normalize_failure_signature(cached_item.get("predicate_template", ""))
        cached_variant = cached_item.get("probe_predicate_report", {}).get("probe_variant", {})
        if (
            signature in set(selected_edge.get("tried_probe_signatures", []) or [])
            or int(cached_variant.get("completion_variant_index", 0) or 0)
            != int(selected_edge.get("completion_variant_index", 0) or 0)
        ):
            self._candidate_bitmap_probe_cache_discard(edge_id)

    def _candidate_bitmap_probe_cache_discard(self, edge_id: str) -> None:
        self.candidate_bitmap_probe_template_cache.pop(edge_id, None)
        try:
            self.candidate_bitmap_probe_template_cache_order.remove(edge_id)
        except ValueError:
            pass

    def _candidate_bitmap_record_completion_retry(
        self, selected_edge: Dict[str, Any], result: Dict[str, Any],
    ) -> None:
        if completion_retry_needed(result):
            # Local progress also works without shared state. The callback advances
            # the shared/disk counters once per result using its existing batch flow.
            advance_completion_variant(selected_edge)
            self._candidate_bitmap_probe_cache_discard(str(selected_edge.get("edge_id", "")))

    def _candidate_bitmap_probe_cache_put(
        self,
        edge_id: str,
        *,
        predicate_template: str,
        probe_graph_report: Dict[str, Any],
        probe_predicate_report: Dict[str, Any],
    ) -> None:
        key = str(edge_id or "")
        if not key:
            return

        limit = int(getattr(RL_ENV_CONFIG, "candidate_bitmap_probe_template_cache_size", 5000))
        if limit <= 0:
            return

        if key not in self.candidate_bitmap_probe_template_cache:
            self.candidate_bitmap_probe_template_cache_order.append(key)

        self.candidate_bitmap_probe_template_cache[key] = {
            "predicate_template": str(predicate_template or ""),
            "probe_graph_report": dict(probe_graph_report or {}),
            "probe_predicate_report": dict(probe_predicate_report or {}),
        }

        while len(self.candidate_bitmap_probe_template_cache_order) > limit:
            old_key = self.candidate_bitmap_probe_template_cache_order.pop(0)
            self.candidate_bitmap_probe_template_cache.pop(old_key, None)

    def _candidate_bitmap_should_enqueue_to_master(
        self,
        *,
        edge_id: str,
        db_success: bool,
        target_edge_hit: bool,
        target_edge_new_signal: bool,
    ) -> bool:
        # mark_success() returns True exactly once for the first global success.
        # Local retry history still affects PPO scoring/reward, but must never
        # prevent that first successful target edge from reaching the Master.
        if (
            not bool(db_success)
            or not bool(target_edge_hit)
            or not bool(target_edge_new_signal)
        ):
            return False

        edge_key = str(edge_id or "")
        if not edge_key:
            return False

        if edge_key in self.candidate_bitmap_master_enqueued_edges:
            return False

        return True

    def _candidate_bitmap_edge_id_from_parse_edge(self, edge: Any) -> str:
        source_node_id = str(getattr(edge, "source_node_id", ""))
        target_node_id = str(getattr(edge, "target_node_id", ""))

        raw_slot = getattr(edge, "slot_id", 0)
        try:
            slot_index = int(raw_slot)
        except Exception:
            m = re.search(r"\d+", str(raw_slot))
            slot_index = int(m.group(0)) if m else 0

        return f"{source_node_id}::slot_{slot_index}->{target_node_id}"


    def _frontier_stagnation_pressure(self) -> float:
        threshold = max(1, int(getattr(RL_ENV_CONFIG, "frontier_stagnation_window_steps", 200)))
        return float(np.clip(self.frontier_steps_since_new_edge / threshold, 0.0, 1.0))



    def _update_frontier_state(
        self,
        *,
        db_success: bool,
        parse_success: bool,
        actual_new_edges: int,
        predicted_new_edges: int = 0,
    ) -> None:
        if not bool(getattr(RL_ENV_CONFIG, "frontier_mode_enabled", True)):
            self.frontier_mode_active = False
            return

        actual_new_edges = int(actual_new_edges or 0)
        predicted_new_edges = int(predicted_new_edges or 0)

        self.frontier_window_steps += 1
        self.frontier_window_db_successes += int(bool(db_success))
        self.frontier_window_parse_successes += int(bool(parse_success))
        self.frontier_window_new_edges += max(0, actual_new_edges)

        if predicted_new_edges > 0:
            self.frontier_window_predicted_positive_steps += 1
            self.frontier_window_predicted_new_edges += predicted_new_edges
            if actual_new_edges > 0:
                self.frontier_window_prediction_hit_steps += 1
            else:
                self.frontier_window_prediction_miss_steps += 1

        if actual_new_edges > 0:
            self.frontier_steps_since_new_edge = 0
        else:
            self.frontier_steps_since_new_edge += 1

        window = max(1, int(getattr(RL_ENV_CONFIG, "frontier_stagnation_window_steps", 200)))
        if self.frontier_window_steps < window:
            return

        min_db_success = int(getattr(RL_ENV_CONFIG, "frontier_min_db_success_in_window", 30))
        min_parse_success = int(getattr(RL_ENV_CONFIG, "frontier_min_parse_success_in_window", 20))
        max_new_edges = int(getattr(RL_ENV_CONFIG, "frontier_max_new_edges_in_window", 0))
        exit_new_edges = int(getattr(RL_ENV_CONFIG, "frontier_exit_new_edges_in_window", 2))

        should_enter = (
            self.frontier_window_db_successes >= min_db_success
            and self.frontier_window_parse_successes >= min_parse_success
            and self.frontier_window_new_edges <= max_new_edges
        )

        should_exit = self.frontier_window_new_edges >= exit_new_edges

        if should_enter:
            self.frontier_mode_active = True
        elif should_exit:
            self.frontier_mode_active = False

        self.frontier_last_window_predicted_positive_steps = int(
            self.frontier_window_predicted_positive_steps
        )
        self.frontier_last_window_actual_new_edges = int(
            self.frontier_window_new_edges
        )
        self.frontier_last_window_new_edge_hit_rate = float(
            self.frontier_window_new_edges / max(1, self.frontier_window_steps)
        )
        self.frontier_last_window_prediction_hit_rate = float(
            self.frontier_window_prediction_hit_steps
            / max(1, self.frontier_window_predicted_positive_steps)
        )

        self.frontier_window_steps = 0
        self.frontier_window_db_successes = 0
        self.frontier_window_parse_successes = 0
        self.frontier_window_new_edges = 0
        self.frontier_window_predicted_positive_steps = 0
        self.frontier_window_predicted_new_edges = 0
        self.frontier_window_prediction_hit_steps = 0
        self.frontier_window_prediction_miss_steps = 0


    # Gym / Gymnasium 接口：reset
    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]: # 重置环境：重置 Episode 步数,清零上一动作的一切信息,返回初始观测和 info
        super().reset(seed=seed)

        self.current_step = 0
        self.prev_reward = 0.0
        self.last_reward = 0.0
        self.last_sigmas = SigmaPair(0.0, 0.0)
        self.last_diversity = 0.0
        self.stagnation_count = 0  # <---重置停滞计时
        self.level_stagnation_count = 0  # <---重置停滞计时
        self.last_used_uncertain = False
        self.last_is_repeated = False
        self.last_forced_non_novel = False

        obs = self._build_obs()
        info: Dict[str, Any] = {
            "msg": "env reset",
        }
        return obs, info

    # ==================== [新增辅助方法] ====================
    # =======================================================

    # Gym / Gymnasium 接口：step
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        return self._step_candidate_bitmap(action)

    def _step_candidate_bitmap(
        self,
        action: np.ndarray,
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        policy_action_index, action_index, action_source = (
            self._resolve_candidate_bitmap_action_index(action)
        )
        selected_edge = self._select_candidate_bitmap_edge_from_action(action_index)
        proposed_edge = selected_edge
        (
            selected_edge,
            action_index,
            reservation_fallback,
            reservation_conflicts,
            reservation_claimed,
        ) = self._candidate_bitmap_select_and_reserve_edge(action_index, selected_edge)

        sigmas = SigmaPair(0.0, 0.0)

        if selected_edge is None:
            reservation_conflict_skip = bool(
                reservation_conflicts > 0 and isinstance(proposed_edge, dict)
            )
            enumeration_idle = action_source == "enumeration_idle"
            neutral_skip = bool(reservation_conflict_skip or enumeration_idle)
            attempted_edge_id = (
                str(proposed_edge.get("edge_id", "") or "")
                if isinstance(proposed_edge, dict)
                else ""
            )
            self.current_step += 1
            self.env_total_steps += 1
            self.profiling_stats["Total_Steps"] += 1
            # PPO并发冲突和枚举尾批空位都不是边本身的失败证据，使用中性奖励。
            penalty = 0.0 if neutral_skip else float(RL_ENV_CONFIG.reward_execution_error)
            skip_reason = (
                "PPO target edge reservation conflict"
                if reservation_conflict_skip
                else (
                    "enumeration tail batch idle slot"
                    if enumeration_idle
                    else "empty candidate bitmap edge pool"
                )
            )
            self.prev_reward = float(self.last_reward)
            self.last_reward = penalty
            self.last_sigmas = sigmas

            obs = self._build_obs()
            info = {
                "msg": skip_reason,
                "predicate_template": "",
                "completed_sql": None,
                "sql_outcome_label": None if neutral_skip else SpaceLabel.B_INCORRECT.value,
                "sql_outcome_reason": skip_reason,
                "db_success": False,
                "db_error": None if neutral_skip else skip_reason,
                "parse_success": False,
                "parse_error": None if neutral_skip else skip_reason,
                "local_new_nodes": 0,
                "local_new_edges": 0,
                "structure_edge_count": 0,
                "structure_operator_count": 0,
                "new_operator_count": 0,
                "low_freq_operator_count": 0,
                "generation_mode": "candidate_bitmap",
                "candidate_bitmap_edge": None,
                "candidate_bitmap_edge_id": "",
                "candidate_bitmap_action_source": action_source,
                "candidate_bitmap_policy_action_index": int(policy_action_index),
                "candidate_bitmap_executed_action_index": int(action_index),
                "candidate_bitmap_reservation_claimed": False,
                "candidate_bitmap_reservation_conflict": bool(reservation_conflicts > 0),
                "candidate_bitmap_reservation_conflict_count": int(reservation_conflicts),
                "candidate_bitmap_reservation_fallback": False,
                "candidate_bitmap_reservation_conflict_skip": bool(reservation_conflict_skip),
                "candidate_bitmap_reservation_attempt_count": int(
                    self.current_candidate_bitmap_reservation_attempts
                ),
                "candidate_bitmap_reservation_edge_id": attempted_edge_id,
                "candidate_bitmap_reservation_timeout_count": int(self.reservation_timeout_release_count),
                "candidate_bitmap_ppo_sharding_active": bool(
                    self._candidate_bitmap_ppo_sharding_active()
                ),
                "candidate_bitmap_worker_rank": int(self.worker_rank),
                "candidate_bitmap_worker_count": int(self.worker_count),
                "candidate_bitmap_ppo_shard_size": int(
                    len(self.candidate_bitmap_ppo_shard_indices)
                    if self._candidate_bitmap_ppo_sharding_active()
                    else 0
                ),
                "candidate_bitmap_target_edge_hit": False,
                "reward_info": {
                    "reward_total": penalty,
                    "reward_reason": skip_reason,
                },
                "runtime_stats": {
                    "skipped": True,
                    "reason": skip_reason,
                },
            }
            return obs, penalty, self.current_step >= self.max_steps, False, info

        edge_id = str(selected_edge.get("edge_id", ""))
        edge_index = int(selected_edge.get("edge_index", -1))
        enumeration_active = self._candidate_bitmap_enumeration_active()
        if enumeration_active:
            edge_group = str(selected_edge.get("edge_group", "") or "")
            core_to_extension_edge = bool(selected_edge.get("is_core_to_extension", False))
            soft_priority_active = False
            soft_priority_bonus = 0.0
            candidate_failed_retry = False
        else:
            edge_group = self._candidate_bitmap_record_group(selected_edge)
            core_to_extension_edge = self._candidate_bitmap_is_core_to_extension(selected_edge)
            soft_priority_active = self._candidate_bitmap_core_extension_soft_priority_active()
            soft_priority_bonus = self._candidate_bitmap_core_extension_priority_bonus(selected_edge)

            try:
                candidate_failed_retry = bool(
                    self._candidate_bitmap_is_selected_failed_retry(
                        selected_edge,
                        edge_index,
                    )
                )
            except Exception:
                candidate_failed_retry = False

        if candidate_failed_retry:
            self.candidate_bitmap_soft_retry_counts[edge_id] = (
                int(self.candidate_bitmap_soft_retry_counts.get(edge_id, 0)) + 1
            )

        candidate_local_tested = edge_id in self.candidate_bitmap_tested_edges
        is_repeated = bool(candidate_local_tested)
        self.candidate_bitmap_tested_edges.add(edge_id)

        candidate_level = str(selected_edge.get("candidate_level", ""))
        if enumeration_active:
            candidate_edge_score = 0.0
            candidate_pattern_transfer_score = 0.0
            candidate_support_rank = 0
            candidate_hnsw_tendency = 0.5
            candidate_structural_entropy = 0.0
            candidate_inner_ready = False
        else:
            candidate_edge_score = self._candidate_bitmap_effective_edge_score(selected_edge)
            candidate_pattern_transfer_score = (
                self._candidate_bitmap_effective_pattern_transfer_score(selected_edge)
            )
            candidate_support_rank = int(self._candidate_bitmap_support_rank(selected_edge))
            candidate_hnsw_tendency = self._candidate_bitmap_hnsw_tendency(selected_edge)
            candidate_structural_entropy = self._candidate_bitmap_structural_entropy(selected_edge)
            candidate_inner_ready = bool(self._candidate_bitmap_inner_ready(selected_edge))

        observed_bitmap = np.asarray(
            self.candidate_bitmap_bitmaps.get(
                "observed_bitmap",
                np.zeros(len(self.candidate_bitmap_edge_records), dtype=np.uint8),
            ),
            dtype=np.uint8,
        )
        failed_bitmap = np.asarray(
            self.candidate_bitmap_bitmaps.get(
                "failed_bitmap",
                np.zeros(len(self.candidate_bitmap_edge_records), dtype=np.uint8),
            ),
            dtype=np.uint8,
        )
        candidate_observed_before = bool(
            0 <= edge_index < len(observed_bitmap)
            and int(observed_bitmap[edge_index]) > 0
        )
        candidate_failed_before = bool(
            0 <= edge_index < len(failed_bitmap)
            and int(failed_bitmap[edge_index]) > 0
        )
        candidate_exploration_done = bool(
            self._candidate_bitmap_exploration_done_for_index(edge_index, edge_id)
        )
        candidate_probing = bool(self._candidate_bitmap_probing_for_index(edge_index))

        candidate_pool_records = [
            record
            for record in (self.current_candidate_bitmap_pool or [])
            if isinstance(record, dict)
        ]
        candidate_pool_done_count = sum(
            1
            for record in candidate_pool_records
            if self._candidate_bitmap_exploration_done_for_index(
                int(record.get("edge_index", -1)),
                str(record.get("edge_id", "") or ""),
            )
        )
        candidate_pool_valid_count = len(candidate_pool_records)
        candidate_pool_done_rate = (
            float(candidate_pool_done_count) / float(candidate_pool_valid_count)
            if candidate_pool_valid_count > 0
            else 0.0
        )

        # 三条返回路径共享同一组候选状态，只用于奖励解释和消融统计。
        # exploration_done 表示全局确认成功/失败；local_tested 仅表示当前 env 测试过。
        candidate_bitmap_selection_state_info = {
            "candidate_bitmap_observed_before": bool(candidate_observed_before),
            "candidate_bitmap_failed_before": bool(candidate_failed_before),
            "candidate_bitmap_exploration_done": bool(candidate_exploration_done),
            "candidate_bitmap_probing": bool(candidate_probing),
            "candidate_bitmap_local_tested": bool(candidate_local_tested),
            "candidate_bitmap_pool_size": int(len(self.current_candidate_bitmap_pool)),
            "candidate_bitmap_pool_unfinished_count": int(
                self.current_candidate_bitmap_pool_unfinished_count
            ),
            "candidate_bitmap_pool_valid_count": int(candidate_pool_valid_count),
            "candidate_bitmap_pool_done_count": int(candidate_pool_done_count),
            "candidate_bitmap_pool_done_rate": float(candidate_pool_done_rate),
            "candidate_bitmap_reservation_attempt_count": int(
                self.current_candidate_bitmap_reservation_attempts
            ),
            "candidate_bitmap_reservation_conflict_skip": False,
            "candidate_bitmap_ppo_sharding_active": bool(
                self._candidate_bitmap_ppo_sharding_active()
            ),
            "candidate_bitmap_worker_rank": int(self.worker_rank),
            "candidate_bitmap_worker_count": int(self.worker_count),
            "candidate_bitmap_ppo_shard_size": int(
                len(self.candidate_bitmap_ppo_shard_indices)
                if self._candidate_bitmap_ppo_sharding_active()
                else 0
            ),
            "candidate_bitmap_inner_ready": bool(candidate_inner_ready),
        }

        persist_probe_artifacts = self._candidate_bitmap_should_persist_probe_artifacts()
        paths: Dict[str, Path] = self._candidate_bitmap_runtime_paths(edge_index) if persist_probe_artifacts else {}

        predicate_template = ""
        probe_graph_report: Dict[str, Any] = {}
        probe_predicate_report: Dict[str, Any] = {}
        construct_error: Optional[str] = None
        probe_cache_hit = False
        safe_closure_used = False

        shared_state = getattr(self, "candidate_shared_state", None)
        self._candidate_bitmap_sync_probe_variants(selected_edge)

        try:
            if not self.candidate_bitmap_formal_graph_data:
                self._reload_candidate_bitmap_state(force=True)

            if persist_probe_artifacts:
                probe_graph_report = candidate_bitmap_tools.build_and_save_probe_graph(
                    Path(getattr(CANDIDATE_BITMAP_CONFIG, "global_knowledge_graph_path")),
                    probe_edge_index=edge_index,
                    probe_graph_path=paths["probe_graph"],
                    probe_report_path=paths["probe_graph_report"],
                    # True：允许 observed/confirmed failed 边重复构造；
                    # probing 边仍由 candidate_bitmap.py 无条件拒绝。
                    allow_observed=True,
                    mark_probing=bool(getattr(RL_ENV_CONFIG, "candidate_bitmap_update_npz_in_env", False)),
                )

                probe_predicate_report = candidate_bitmap_tools.build_and_save_probe_predicate(
                    paths["probe_graph"],
                    predicate_output_path=paths["probe_predicate"],
                    predicate_report_path=paths["probe_predicate_report"],
                    max_depth=int(getattr(CANDIDATE_BITMAP_CONFIG, "probe_predicate_max_depth", 3)),
                    edge_records=self.candidate_bitmap_edge_records,
                )

                predicate_template = paths["probe_predicate"].read_text(encoding="utf-8").strip()
            else:
                cached_probe = self._candidate_bitmap_probe_cache_get(edge_id)
                if cached_probe is not None:
                    probe_graph_report = cached_probe.get("probe_graph_report", {})
                    probe_predicate_report = cached_probe.get("probe_predicate_report", {})
                    predicate_template = str(cached_probe.get("predicate_template", "") or "").strip()
                    probe_cache_hit = True
                else:
                    probe_report = candidate_bitmap_tools.build_probe_predicate_for_edge(
                        formal_graph_data=self.candidate_bitmap_formal_graph_data,
                        edge_records=self.candidate_bitmap_edge_records,
                        bitmaps=self.candidate_bitmap_bitmaps,
                        probe_edge_index=edge_index,
                        # True：允许 observed/confirmed failed 边重复构造；
                        # probing 边仍由 candidate_bitmap.py 无条件拒绝。
                        allow_observed=True,
                        max_depth=int(getattr(CANDIDATE_BITMAP_CONFIG, "probe_predicate_max_depth", 3)),
                        graph_path=Path(getattr(CANDIDATE_BITMAP_CONFIG, "global_knowledge_graph_path")),
                    )
                    probe_graph_report = probe_report.get("probe_graph_report", {})
                    probe_predicate_report = probe_report.get("probe_predicate_report", {})
                    predicate_template = str(
                        probe_predicate_report.get("predicate", {}).get("sql", "") or ""
                    ).strip()

                    self._candidate_bitmap_probe_cache_put(
                        edge_id,
                        predicate_template=predicate_template,
                        probe_graph_report=probe_graph_report,
                        probe_predicate_report=probe_predicate_report,
                    )

            if str(probe_predicate_report.get("status", "")) != "success":
                construct_error = str(
                    probe_predicate_report.get("blocked_reason", "probe predicate construct blocked")
                )

            if not predicate_template:
                construct_error = construct_error or "probe predicate is empty"

            predicate_meta = probe_predicate_report.get("predicate", {})
            if isinstance(predicate_meta, dict):
                safe_closure_used = bool(predicate_meta.get("safe_closure_used", False))

        except Exception as exc:
            construct_error = str(exc)

        construct_environment_failure = bool(
            str(getattr(DB_CONFIG, "db_type", "") or "").lower().strip()
            in {"oracle", "oracledb", "oracle_ai26", "sqlserver"}
            and construct_error
            and candidate_bitmap_tools.candidate_bitmap_environment_failure(
                {"db_success": False, "db_error": construct_error}
            )
        )
        if construct_error:
            self._candidate_bitmap_record_probe_variant_exhausted(
                edge_id,
                construct_error,
            )
            self.current_step += 1
            self.env_total_steps += 1
            self.profiling_stats["Total_Steps"] += 1
            penalty, reward_info = compute_reward(
                is_repeated_action=is_repeated,
                db_success=False,
                parse_success=False,
                candidate_bitmap_target_edge_hit=False,
                candidate_bitmap_candidate_level=candidate_level,
                candidate_bitmap_edge_score=candidate_edge_score,
                candidate_bitmap_support_rank=candidate_support_rank,
                candidate_bitmap_observed_before=bool(candidate_observed_before),
                candidate_bitmap_failed_before=bool(candidate_failed_before),
                candidate_bitmap_exploration_done=bool(candidate_exploration_done),
                candidate_bitmap_local_tested=bool(candidate_local_tested),
                candidate_bitmap_failure_kind=(
                    "environment_error" if construct_environment_failure else "downstream_construct_failed"
                ),
                candidate_bitmap_safe_closure_used=False,
            )
            self.prev_reward = float(self.last_reward)
            self.last_reward = penalty
            self.last_sigmas = sigmas
            self.last_is_repeated = bool(is_repeated)
            if not construct_environment_failure:
                self.candidate_bitmap_failed_edges.add(edge_id)

            obs = self._build_obs()
            info = {
                "msg": f"candidate bitmap probe construction failed: {construct_error}",
                "predicate_template": predicate_template,
                "completed_sql": None,
                "sql_outcome_label": SpaceLabel.B_INCORRECT.value,
                "sql_outcome_reason": construct_error,
                "db_success": False,
                "db_error": construct_error,
                "parse_success": False,
                "parse_error": construct_error,
                "local_new_nodes": 0,
                "local_new_edges": 0,
                "structure_edge_count": 0,
                "structure_operator_count": 0,
                "new_operator_count": 0,
                "low_freq_operator_count": 0,
                "generation_mode": "candidate_bitmap",
                "candidate_bitmap_edge": selected_edge,
                "candidate_bitmap_edge_id": edge_id,
                "candidate_bitmap_edge_index": edge_index,
                "candidate_bitmap_action_source": action_source,
                "candidate_bitmap_policy_action_index": int(policy_action_index),
                "candidate_bitmap_executed_action_index": int(action_index),
                "candidate_bitmap_reservation_claimed": bool(reservation_claimed),
                "candidate_bitmap_reservation_conflict": bool(reservation_conflicts > 0),
                "candidate_bitmap_reservation_conflict_count": int(reservation_conflicts),
                "candidate_bitmap_reservation_fallback": bool(reservation_fallback),
                "candidate_bitmap_reservation_edge_id": edge_id,
                "candidate_bitmap_reservation_token": str(
                    self.current_candidate_bitmap_selected_reservation_token
                ),
                "candidate_bitmap_reservation_timeout_count": int(self.reservation_timeout_release_count),
                **candidate_bitmap_selection_state_info,
                "candidate_bitmap_edge_group": edge_group,
                "candidate_bitmap_core_to_extension": bool(core_to_extension_edge),
                "candidate_bitmap_soft_priority_active": bool(soft_priority_active),
                "candidate_bitmap_soft_priority_bonus": float(soft_priority_bonus),
                "candidate_bitmap_failed_retry": bool(candidate_failed_retry),
                "candidate_bitmap_representation_guidance_enabled": bool(
                    self._candidate_bitmap_representation_guidance_enabled()
                ),
                "candidate_bitmap_hnsw_success_tendency": float(candidate_hnsw_tendency),
                "candidate_bitmap_structural_entropy": float(candidate_structural_entropy),
                "candidate_bitmap_target_edge_hit": False,
                # 仅表示 probe 谓词在进入 DB 前构造失败，供消融实验统计使用。
                "candidate_bitmap_construct_failed": True,
                "failure_kind": (
                    "environment_error" if construct_environment_failure else "downstream_construct_failed"
                ),
                "environment_failure": bool(construct_environment_failure),
                "downstream_failure_reason": str(construct_error),
                "safe_closure_used": False,
                "safe_fallback_used": False,
                "candidate_bitmap_probe_graph_report": (
                    probe_graph_report
                    if bool(getattr(RL_ENV_CONFIG, "candidate_bitmap_include_probe_reports_in_info", False))
                    else {}
                ),
                "candidate_bitmap_probe_predicate_report": (
                    probe_predicate_report
                    if bool(getattr(RL_ENV_CONFIG, "candidate_bitmap_include_probe_reports_in_info", False))
                    else {}
                ),
                "reward_info": reward_info,
                "runtime_stats": {
                    "skipped": True,
                    "reason": "candidate bitmap probe construction failed",
                    "error": construct_error,
                },
            }
            return obs, penalty, self.current_step >= self.max_steps, False, info

        num_placeholders = _count_forbidden_placeholders(predicate_template)
        if num_placeholders > 0:
            fast_penalty, reward_info = compute_reward(
                is_repeated_action=is_repeated,
                is_syntax_error=False,
                db_success=False,
                parse_success=False,
                candidate_bitmap_target_edge_hit=False,
                candidate_bitmap_candidate_level=candidate_level,
                candidate_bitmap_edge_score=candidate_edge_score,
                candidate_bitmap_support_rank=candidate_support_rank,
                candidate_bitmap_observed_before=bool(candidate_observed_before),
                candidate_bitmap_failed_before=bool(candidate_failed_before),
                candidate_bitmap_exploration_done=bool(candidate_exploration_done),
                candidate_bitmap_local_tested=bool(candidate_local_tested),
                candidate_bitmap_failure_kind="placeholder_or_hole",
                candidate_bitmap_safe_closure_used=bool(safe_closure_used),
            )

            self.current_step += 1
            self.env_total_steps += 1
            self.profiling_stats["Total_Steps"] += 1
            self.prev_reward = float(self.last_reward)
            self.last_reward = float(fast_penalty)
            self.last_sigmas = sigmas
            self.last_is_repeated = bool(is_repeated)
            self.candidate_bitmap_failed_edges.add(edge_id)

            obs = self._build_obs()
            info = {
                "msg": f"Fail-Fast: placeholder_count={num_placeholders}",
                "predicate_template": predicate_template,
                "completed_sql": None,
                "sql_outcome_label": SpaceLabel.B_INCORRECT.value,
                "reward_info": reward_info,
                "runtime_stats": {
                    "skipped": True,
                    "reason": "candidate bitmap predicate placeholder",
                    "placeholder_count": int(num_placeholders),
                },
                "parse_success": False,
                "parse_error": "placeholder or HOLE detected",
                "local_new_nodes": 0,
                "local_new_edges": 0,
                "structure_edge_count": 0,
                "structure_operator_count": 0,
                "generation_mode": "candidate_bitmap",
                "candidate_bitmap_edge": selected_edge,
                "candidate_bitmap_edge_id": edge_id,
                "candidate_bitmap_edge_index": edge_index,
                "candidate_bitmap_action_source": action_source,
                "candidate_bitmap_policy_action_index": int(policy_action_index),
                "candidate_bitmap_executed_action_index": int(action_index),
                "candidate_bitmap_reservation_claimed": bool(reservation_claimed),
                "candidate_bitmap_reservation_conflict": bool(reservation_conflicts > 0),
                "candidate_bitmap_reservation_conflict_count": int(reservation_conflicts),
                "candidate_bitmap_reservation_fallback": bool(reservation_fallback),
                "candidate_bitmap_reservation_edge_id": edge_id,
                "candidate_bitmap_reservation_token": str(
                    self.current_candidate_bitmap_selected_reservation_token
                ),
                "candidate_bitmap_reservation_timeout_count": int(self.reservation_timeout_release_count),
                **candidate_bitmap_selection_state_info,
                "candidate_bitmap_edge_group": edge_group,
                "candidate_bitmap_core_to_extension": bool(core_to_extension_edge),
                "candidate_bitmap_soft_priority_active": bool(soft_priority_active),
                "candidate_bitmap_soft_priority_bonus": float(soft_priority_bonus),
                "candidate_bitmap_failed_retry": bool(candidate_failed_retry),
                "candidate_bitmap_target_edge_hit": False,
                "failure_kind": "placeholder_or_hole",
                "downstream_failure_reason": "",
                "safe_closure_used": bool(safe_closure_used),
                "safe_fallback_used": bool(safe_closure_used),
            }
            self._candidate_bitmap_record_completion_retry(selected_edge, info)
            return obs, float(fast_penalty), self.current_step >= self.max_steps, False, info

        current_dialect = DB_CONFIG.db_type
        t_syntax_start = time.time()
        is_syntax_valid, ast_node_count = _check_predicate_syntax_simple(
            predicate_str=predicate_template,
            dialect=current_dialect,
        )
        is_syntax_error = not is_syntax_valid
        t_syntax_end = time.time()
        self.profiling_stats["T_Syntax_Check"] += (t_syntax_end - t_syntax_start)

        completed_sql: Optional[str] = None
        heuristic_error: Optional[str] = None
        t_llm_start = time.time()
        try:
            completed_sql = self.heuristic_client.complete_sql(
                predicate_template,
                expected_return_type=selected_edge.get("outer_return_type"),
            )
        except Exception as exc:
            completed_sql = None
            heuristic_error = str(exc)
            print(f"[CandidateBitmapEnv] heuristic completion failed: {exc}", flush=True)
        t_llm_end = time.time()
        self.profiling_stats["T_LLM"] += (t_llm_end - t_llm_start)

        t_db_start = time.time()
        if completed_sql is not None:
            db_result: DBQueryResult = self.db_exec.execute_query(
                sql=completed_sql,
                params=None,
                fetch="all",
                autocommit=True,
            )
        else:
            db_result = DBQueryResult(
                sql=completed_sql or "",
                success=False,
                rows=None,
                rowcount=0,
                error=heuristic_error or "Heuristic failed to generate SQL.",
                error_code=None,
                execution_time=0.0,
                warning_count=0,
            )
        t_db_end = time.time()
        self.profiling_stats["T_DB_Exec"] += (t_db_end - t_db_start)

        if is_syntax_error:
            outcome = SqlOutcome(
                label=SpaceLabel.B_INCORRECT,
                reason=f"Syntax Check Failed (Pre-execution). Original Predicate: {predicate_template}",
            )
        else:
            outcome = classify_sql_outcome(db_result)

        parse_success = False
        parse_error = (
            f"sqlglot syntax check failed before PredicateStructureAnalyzer (dialect={current_dialect})"
            if is_syntax_error
            else None
        )
        parse_result = None
        structure_edge_count = 0
        structure_operator_count = 0
        parsed_edge_ids: set[str] = set()
        target_edge_hit = False

        if not is_syntax_error:
            parse_result = self.structure_analyzer.analyze(
                predicate_template,
                expected_type="T_BOOL",
            )
            parse_success = bool(parse_result.success)
            parse_error = parse_result.error
            structure_edge_count = len(parse_result.edges)
            structure_operator_count = len(parse_result.operators)

            if parse_success:
                parsed_edge_ids = {
                    self._candidate_bitmap_edge_id_from_parse_edge(edge)
                    for edge in parse_result.edges
                }
                target_edge_hit = edge_id in parsed_edge_ids

        target_edge_success = bool(db_result.success and parse_success and target_edge_hit)
        if target_edge_success and shared_state is not None:
            target_edge_new_signal = bool(shared_state.mark_success(edge_id, edge_index))
        else:
            target_edge_new_signal = bool(
                target_edge_success
                and not candidate_observed_before
                and edge_id not in self.candidate_bitmap_success_edges
            )
        local_new_edges = 1 if target_edge_new_signal else 0
        local_new_nodes = 0
        is_new_edge = bool(target_edge_new_signal)

        current_db_type = str(getattr(DB_CONFIG, "db_type", "") or "").lower().strip()
        environment_failure = bool(
            current_db_type in {"oracle", "oracledb", "oracle_ai26", "sqlserver"}
            and candidate_bitmap_tools.candidate_bitmap_environment_failure({
                "db_success": bool(db_result.success),
                "db_error": db_result.error,
                "db_error_code": db_result.error_code,
                "timed_out": bool(getattr(db_result, "timed_out", False)),
                "sqlstate": getattr(db_result, "sqlstate", None),
                "cancelled": bool(getattr(db_result, "cancelled", False)),
            })
            and (current_db_type == "sqlserver" or not safe_closure_used)
        )

        if target_edge_success:
            self.candidate_bitmap_success_edges.add(edge_id)
            self.candidate_bitmap_failed_edges.discard(edge_id)
        elif (not db_result.success) and not environment_failure:
            self.candidate_bitmap_failed_edges.add(edge_id)
        elif parse_success and not target_edge_hit and not environment_failure:
            self.candidate_bitmap_failed_edges.add(edge_id)

        valid_incorrect_for_collection = (
            outcome.label == SpaceLabel.B_INCORRECT
            and _count_forbidden_placeholders(predicate_template) == 0
            and (not is_syntax_error)
            and completed_sql is not None
            and (not db_result.success)
            and target_edge_hit
            and not environment_failure
        )

        # Evidence only: no reward, bitmap, graph, or queue decision reads it.
        # The DBMS-specific classifier records definite type/value failures;
        # MySQL also retains ordinary execution errors as weak evidence.
        # Timeouts and environment failures are always excluded.
        if (not db_result.success) and parse_success and target_edge_hit:
            try:
                if current_db_type in {"oracle", "oracledb", "oracle_ai26"}:
                    parse_result.oracle_executed_sql = completed_sql
                if current_db_type == "sqlserver":
                    parse_result.sqlserver_executed_sql = completed_sql
                self.runtime_stats.record_joint_signature_type_failure(
                    parse_result,
                    target_edge=selected_edge,
                    target_edge_hit=bool(target_edge_hit),
                    error_code=db_result.error_code,
                    error=db_result.error,
                    timed_out=bool(getattr(db_result, "timed_out", False)),
                    **({"sqlstate": getattr(db_result, "sqlstate", None),
                        "cancelled": bool(getattr(db_result, "cancelled", False))}
                       if current_db_type == "sqlserver" else {}),
                )
            except Exception as exc:
                print(
                    f"[RuntimeStats Warning] joint signature type failure skipped: {exc}",
                    flush=True,
                )

        should_enqueue_to_master = self._candidate_bitmap_should_enqueue_to_master(
            edge_id=edge_id,
            db_success=bool(outcome.label == SpaceLabel.A_CORRECT and db_result.success),
            target_edge_hit=bool(target_edge_hit),
            target_edge_new_signal=bool(target_edge_new_signal),
        )

        if self.shared_queue is not None:
            if should_enqueue_to_master:
                # Correct 文本仍交给第一阶段向量数据流程；target_edge 元数据只供
                # PKG 把本次验证成功的目标边写入正式图，不能把补全边一并入图。
                self.shared_queue.put((
                    "a",
                    {
                        "predicate": predicate_template,
                        **({"oracle_executed_sql": completed_sql}
                           if current_db_type in {"oracle", "oracledb", "oracle_ai26"}
                           else {}),
                        **({"sqlserver_executed_sql": completed_sql}
                           if current_db_type == "sqlserver" else {}),
                        "target_edge": {
                            "outer_node_id": str(selected_edge.get("outer_node_id", "")),
                            "outer_token": str(selected_edge.get("outer_token", "")),
                            "outer_return_type": str(selected_edge.get("outer_return_type", "")),
                            "slot_index": int(selected_edge.get("slot_index", -1)),
                            "inner_node_id": str(selected_edge.get("inner_node_id", "")),
                            "inner_token": str(selected_edge.get("inner_token", "")),
                            "inner_return_type": str(selected_edge.get("inner_return_type", "")),
                        },
                    },
                ))
                self.candidate_bitmap_master_enqueued_edges.add(edge_id)
            elif valid_incorrect_for_collection and target_edge_hit:
                self.shared_queue.put(("b", predicate_template))

        complexity_score = max(0.0, ast_node_count - 8.0) if not is_syntax_error else 0.0

        new_operator_count = 0
        low_freq_operator_count = 0
        if parse_success and parse_result is not None:
            try:
                operator_stats = self.runtime_stats.load_operator_stats()
                for op_id in sorted(list(parse_result.operators or [])):
                    op_item = operator_stats.get(op_id)
                    if not isinstance(op_item, dict):
                        new_operator_count += 1
                        low_freq_operator_count += 1
                        continue
                    op_freq = (
                        int(op_item.get("success_count", 0))
                        + int(op_item.get("failure_count", 0))
                    )
                    if op_freq <= RL_ENV_CONFIG.reward_low_freq_operator_threshold:
                        low_freq_operator_count += 1
            except Exception:
                new_operator_count = 0
                low_freq_operator_count = 0

        if is_syntax_error:
            failure_kind = "syntax_error"
        elif environment_failure:
            failure_kind = "environment_error"
        elif not db_result.success and safe_closure_used:
            # Temporary sibling completion only proves that this context failed;
            # it is not reliable evidence that the selected target edge is invalid.
            failure_kind = "context_inconclusive"
        elif not db_result.success and parse_success and target_edge_hit:
            failure_kind = "target_db_invalid"
        elif db_result.success and not target_edge_hit:
            failure_kind = "target_miss"
        elif not db_result.success:
            failure_kind = "db_invalid_no_target"
        elif not parse_success:
            failure_kind = "parse_failed"
        else:
            failure_kind = ""

        t_reward_start = time.time()
        reward, reward_info = compute_reward(
            is_repeated_action=is_repeated,
            is_syntax_error=is_syntax_error,
            db_success=db_result.success,
            is_new_edge=is_new_edge,
            complexity_score=complexity_score,
            parse_success=parse_success,
            local_new_edges=local_new_edges,
            local_new_nodes=local_new_nodes,
            structure_operator_count=structure_operator_count,
            new_operator_count=new_operator_count,
            low_freq_operator_count=low_freq_operator_count,
            sigma_a=sigmas.sigma_a,
            sigma_b=sigmas.sigma_b,
            coverage_mode="candidate_bitmap",
            fixed_base_structure=True,
            fixed_base_new_edges=local_new_edges,
            frontier_new_edges=0,
            frontier_new_nodes=0,
            mutation_operator_similarity=0.0,
            mutation_kind_success_rate=0.5,
            mutation_local_failed=bool(edge_id in self.candidate_bitmap_failed_edges),
            mutation_local_success=bool(edge_id in self.candidate_bitmap_success_edges),
            frontier_mode=bool(self.frontier_mode_active),
            frontier_predicted_new_edges=1,
            frontier_actual_new_edges=int(local_new_edges),
            frontier_stagnation_pressure=float(self._frontier_stagnation_pressure()),
            candidate_bitmap_target_edge_hit=bool(target_edge_hit),
            candidate_bitmap_candidate_level=candidate_level,
            candidate_bitmap_edge_score=candidate_edge_score,
            candidate_bitmap_support_rank=candidate_support_rank,
            candidate_bitmap_observed_before=bool(candidate_observed_before),
            candidate_bitmap_failed_before=bool(candidate_failed_before),
            candidate_bitmap_exploration_done=bool(candidate_exploration_done),
            candidate_bitmap_local_tested=bool(candidate_local_tested),
            candidate_bitmap_global_new_edge=bool(target_edge_new_signal),
            candidate_bitmap_failure_kind=failure_kind,
            candidate_bitmap_safe_closure_used=bool(safe_closure_used),
        )
        t_reward_end = time.time()
        self.profiling_stats["T_Reward"] += (t_reward_end - t_reward_start)

        runtime_stats_info = {}
        try:
            if outcome.label == SpaceLabel.A_CORRECT and db_result.success and target_edge_hit:
                if self.shared_queue is None:
                    runtime_stats_info = self.runtime_stats.record_sql_success(
                        predicate=predicate_template,
                        parse_result=parse_result,
                        step=self.current_step,
                        source=self.runtime_stats_source,
                        graph_new_nodes=local_new_nodes,
                        graph_new_edges=local_new_edges,
                    )
                else:
                    runtime_stats_info = {
                        "delegated_to_pkg_manager": True,
                        "reason": "success candidate bitmap predicate will be recorded by pkg_manager",
                    }
            elif RL_ENV_CONFIG.record_env_step_runtime_stats:
                runtime_stats_info = self.runtime_stats.record_sql_failure(
                    predicate=predicate_template,
                    parse_result=parse_result,
                    error=db_result.error or heuristic_error or outcome.reason,
                    step=self.current_step,
                    source=self.runtime_stats_source,
                )
            else:
                runtime_stats_info = {
                    "skipped": True,
                    "reason": "env step runtime stats disabled",
                    "db_success": bool(db_result.success),
                    "parse_success": bool(parse_success),
                    "target_edge_hit": bool(target_edge_hit),
                    "error": db_result.error or heuristic_error or outcome.reason,
                }
        except Exception as exc:
            runtime_stats_info = {"runtime_stats_error": str(exc)}

        success_signal = 1.0 if target_edge_new_signal else 0.0
        if success_signal > 0.0:
            self.stagnation_count = 0
            self.level_stagnation_count = 0
        else:
            self.stagnation_count += 1
            self.level_stagnation_count += 1

        self._update_frontier_state(
            db_success=bool(db_result.success),
            parse_success=bool(parse_success),
            actual_new_edges=int(local_new_edges),
            predicted_new_edges=1,
        )

        self.current_step += 1
        self.env_total_steps += 1
        self.profiling_stats["Total_Steps"] += 1
        self.prev_reward = float(self.last_reward)
        self.last_reward = float(reward)
        self.last_sigmas = sigmas
        self.last_is_repeated = bool(is_repeated)

        terminated: bool = self.current_step >= self.max_steps
        truncated: bool = False
        obs = self._build_obs()

        info: Dict[str, Any] = {
            "reward_info": reward_info,
            "sigma_a": sigmas.sigma_a,
            "sigma_b": sigmas.sigma_b,
            "is_repeated_action": is_repeated,
            "predicate_template": predicate_template,
            "completed_sql": completed_sql,
            "sql_outcome_label": outcome.label.value,
            "sql_outcome_reason": outcome.reason,
            "db_success": db_result.success,
            "db_error": db_result.error,
            "db_error_code": db_result.error_code,
            "db_timed_out": bool(getattr(db_result, "timed_out", False)),
            **({"db_sqlstate": getattr(db_result, "sqlstate", None),
                "db_cancelled": bool(getattr(db_result, "cancelled", False))}
               if current_db_type == "sqlserver" else {}),
            "db_rowcount": db_result.rowcount,
            "db_execution_time": db_result.execution_time,
            "db_warning_count": db_result.warning_count,
            "base_space_label": "candidate_bitmap",
            "base_index_in_space": int(action_index),
            "curr_level": self.curr_level,
            "curr_cluster": self.curr_cluster,
            "success_signal": float(success_signal),
            "coverage_mode": "candidate_bitmap",
            "fixed_base_structure": True,
            "fixed_base_new_edges": int(local_new_edges),
            "frontier_new_edges": 0,
            "frontier_new_nodes": 0,
            "parse_success": parse_success,
            "parse_error": parse_error,
            "local_new_nodes": local_new_nodes,
            "local_new_edges": local_new_edges,
            "structure_edge_count": structure_edge_count,
            "structure_operator_count": structure_operator_count,
            "new_operator_count": new_operator_count,
            "low_freq_operator_count": low_freq_operator_count,
            "runtime_stats": runtime_stats_info,
            "generation_mode": "candidate_bitmap",
            "candidate_bitmap_edge": selected_edge,
            "candidate_bitmap_edge_id": edge_id,
            "candidate_bitmap_edge_index": edge_index,
            "candidate_bitmap_action_source": action_source,
            "candidate_bitmap_policy_action_index": int(policy_action_index),
            "candidate_bitmap_executed_action_index": int(action_index),
            "candidate_bitmap_reservation_claimed": bool(reservation_claimed),
            "candidate_bitmap_reservation_conflict": bool(reservation_conflicts > 0),
            "candidate_bitmap_reservation_conflict_count": int(reservation_conflicts),
            "candidate_bitmap_reservation_fallback": bool(reservation_fallback),
            "candidate_bitmap_reservation_edge_id": edge_id,
            "candidate_bitmap_reservation_token": str(
                self.current_candidate_bitmap_selected_reservation_token
            ),
            "candidate_bitmap_reservation_timeout_count": int(self.reservation_timeout_release_count),
            **candidate_bitmap_selection_state_info,
            "candidate_bitmap_edge_group": edge_group,
            "candidate_bitmap_core_to_extension": bool(core_to_extension_edge),
            "candidate_bitmap_soft_priority_active": bool(soft_priority_active),
            "candidate_bitmap_soft_priority_bonus": float(soft_priority_bonus),
            "candidate_bitmap_failed_retry": bool(candidate_failed_retry),
            "candidate_bitmap_representation_guidance_enabled": bool(
                self._candidate_bitmap_representation_guidance_enabled()
            ),
            "candidate_bitmap_candidate_level": candidate_level,
            "candidate_bitmap_edge_score": float(candidate_edge_score),
            "candidate_bitmap_pattern_transfer_score": float(candidate_pattern_transfer_score),
            "candidate_bitmap_support_rank": int(candidate_support_rank),
            "candidate_bitmap_hnsw_success_tendency": float(candidate_hnsw_tendency),
            "candidate_bitmap_structural_entropy": float(candidate_structural_entropy),
            "candidate_bitmap_target_edge_hit": bool(target_edge_hit),
            "candidate_bitmap_global_new_edge": bool(target_edge_new_signal),
            "failure_kind": failure_kind,
            "environment_failure": bool(environment_failure),
            "downstream_failure_reason": "",
            "safe_closure_used": bool(safe_closure_used),
            "safe_fallback_used": bool(safe_closure_used),
            "candidate_bitmap_parsed_edge_count": int(len(parsed_edge_ids)),
            "candidate_bitmap_parsed_edges_sample": sorted(parsed_edge_ids)[:20],
            "candidate_bitmap_total_edges": int(len(self.candidate_bitmap_edge_records)),
            "candidate_bitmap_tested_edge_count": int(len(self.candidate_bitmap_tested_edges)),
            "candidate_bitmap_success_edge_count": int(len(self.candidate_bitmap_success_edges)),
            "candidate_bitmap_failed_edge_count": int(len(self.candidate_bitmap_failed_edges)),
            "candidate_bitmap_probe_dir": str(paths.get("dir", "")) if paths else "",
            "candidate_bitmap_probe_cache_hit": bool(probe_cache_hit),
            "candidate_bitmap_probe_graph_report": (
                probe_graph_report
                if bool(getattr(RL_ENV_CONFIG, "candidate_bitmap_include_probe_reports_in_info", False))
                else {}
            ),
            "candidate_bitmap_probe_predicate_report": (
                probe_predicate_report
                if bool(getattr(RL_ENV_CONFIG, "candidate_bitmap_include_probe_reports_in_info", False))
                else {}
            ),
            "candidate_bitmap_update_npz_in_env": bool(
                getattr(RL_ENV_CONFIG, "candidate_bitmap_update_npz_in_env", False)
            ),
            "candidate_bitmap_frontier_open_flags": self._candidate_bitmap_frontier_open_flags(),
            "frontier_mode_active": bool(self.frontier_mode_active),
            "frontier_stagnation_pressure": float(self._frontier_stagnation_pressure()),
        }

        self._candidate_bitmap_record_completion_retry(selected_edge, info)
        return obs, float(reward), terminated, truncated, info


    def render(self, mode: str = "human") -> None: # 简单的 render：打印当前步的一些关键信息。
        if mode != "human":
            return
        print(
            f"[Render] step={self.current_step}, "
            f"sigma=({self.last_sigmas.sigma_a:.3f}, "
            f"{self.last_sigmas.sigma_b:.3f}, "
            #f"{self.last_sigmas.sigma_c:.3f}), "
            f"last_reward={self.last_reward:.3f}"
        )

    # ==================== [供 Callback 远程调用的辅助方法] ====================
    # ===========================================================================

    # ==================== [供 Callback 拉取 UCB 数据的接口] ====================
    # ===========================================================================

    # ==================== [接收 Master 广播的同步接口] ====================
    # ===========================================================================

    def close(self) -> None: # 关闭 DBExecutor
        # --- 【新增：将性能统计数据写入 txt 文档】 ---
        try:
            steps = self.profiling_stats["Total_Steps"]
            if steps > 0:
                # 【修正】直接用 PID 作为标识，不再依赖 WORKER_GPU_ID
                pid = os.getpid()
                worker_id = str(pid)

                # 获取当前代码运行所在的文件夹路径 (即 RL 目录)
                current_dir = os.path.dirname(os.path.abspath(__file__))

                # 【修改】定义新的目标子文件夹路径: RL/time_collect
                time_collect_dir = os.path.join(current_dir, "time_collect")

                # 【修改】如果文件夹不存在，则创建它
                # exist_ok=True 非常重要，防止多个进程同时创建文件夹时报错
                os.makedirs(time_collect_dir, exist_ok=True)

                # 【修改】将文件名拼接在新的文件夹路径下
                txt_filename = os.path.join(time_collect_dir, f"profiling_stats_worker_{worker_id}_pid_{pid}.txt")

                # 写入统计报告
                with open(txt_filename, "w", encoding="utf-8") as f:
                    f.write(f"=== 强化学习环境性能统计报告 ===\n")
                    f.write(f"Worker GPU ID: {worker_id} | Process ID: {pid}\n")
                    f.write(f"总运行步数 (Total Steps): {steps}\n\n")

                    f.write("--- 1. 各阶段总耗时 ---\n")
                    f.write(f"选出向量总耗时:     {self.profiling_stats['T_Vec_Select']:.3f} 秒\n")  # <---【新增】
                    f.write(f"VAE 解码总耗时:     {self.profiling_stats['T_Decode']:.3f} 秒\n")
                    f.write(f"语法检查总耗时:     {self.profiling_stats['T_Syntax_Check']:.3f} 秒\n")
                    f.write(f"LLM 补全总耗时:     {self.profiling_stats['T_LLM']:.3f} 秒\n")
                    f.write(f"数据库执行总耗时:   {self.profiling_stats['T_DB_Exec']:.3f} 秒\n")
                    f.write(f"奖励计算总耗时:     {self.profiling_stats['T_Reward']:.3f} 秒\n\n")

                    f.write("--- 2. 生成单个谓词的平均耗时 ---\n")
                    f.write(f"平均 选出向量耗时:   {(self.profiling_stats['T_Vec_Select'] / steps):.4f} 秒/次\n")  # <---【新增】
                    f.write(f"平均 VAE 解码耗时:   {(self.profiling_stats['T_Decode'] / steps):.4f} 秒/次\n")
                    f.write(f"平均 语法检查耗时:   {(self.profiling_stats['T_Syntax_Check'] / steps):.4f} 秒/次\n")
                    f.write(f"平均 LLM 补全耗时:   {(self.profiling_stats['T_LLM'] / steps):.4f} 秒/次\n")
                    f.write(f"平均 数据库执行耗时: {(self.profiling_stats['T_DB_Exec'] / steps):.4f} 秒/次\n")
                    f.write(f"平均 奖励计算耗时:   {(self.profiling_stats['T_Reward'] / steps):.4f} 秒/次\n")
        except Exception as e:
            print(f"[Profiling Error] 写入性能统计数据失败: {e}")
        # -------------------------------------------------------------
        try:
            self.runtime_stats.flush_joint_signature_evidence()
        except Exception as e:
            print(f"[RuntimeStats Warning] joint signature flush failed: {e}")
        try:
            if self.db_exec is not None:
                self.db_exec.close()
        except Exception:
            pass


# 简单自测：随机动作跑一个 Episode
if __name__ == "__main__":
    print("=== SqlDependencyEnv SelfTest ===")

    env = SqlDependencyEnv()

    obs, info = env.reset()
    print("[SelfTest] initial obs:", obs)
    print("[SelfTest] reset info:", info)

    done = False
    step_idx = 0

    while not done and step_idx < 5:  # 只跑前 5 步
        # 随机采样一个动作（真实训练时由 PPO 给出）
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        print(f"\n[Step {step_idx}]")
        print("  action:", action)
        print("  reward:", reward)
        print("  obs   :", obs)
        print("  terminated:", terminated, "truncated:", truncated)
        print("  sql_outcome:", info.get("sql_outcome_label"), "-", info.get("sql_outcome_reason"))
        print("  completed_sql:", info.get("completed_sql"))

        step_idx += 1
        done = terminated or truncated

    env.close()
    print("=== SqlDependencyEnv SelfTest Done ===")
