"""
PKG Manager (Master Process)
负责在强化学习过程中，后台异步接收 Correct 谓词，
调用第三阶段的 Parser 和 Builder 进行增量建图，记录发现的新语义依赖（新边），
并批量刷入 Correct_Collection.txt 和监控日志。
"""
import sys
import os
import time
import json
import csv
import math
from pathlib import Path
from random import Random
from concurrent.futures import ProcessPoolExecutor

# ================= 跨文件夹调用设置 =================
# 获取项目根目录
PROJECT_ROOT = Path(__file__).resolve().parents[1]
# 第三阶段的文件夹路径
PKG_DIR = PROJECT_ROOT / "pkg"
# 将其加入系统路径，使当前文件可以 import 第三阶段的代码
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

# 导入第三阶段的核心建图工具
from pkg.graph_builder import KnowledgeGraphBuilder
from RL.predicate_structure import (
    PredicateStructureAnalyzer,
    load_pkg_config as _structure_load_pkg_config,
    abstract_predicate_for_graph as _structure_abstract_predicate_for_graph,
)
from RL.graph_runtime_stats import get_default_runtime_stats
from RL.progress import print_rl_progress
from RL.runtime_logging import report_joint_signature_flush_failure

import re  # 【新增】引入正则表达式模块
from RL.config import (
    CANDIDATE_BITMAP_CONFIG,
    DB_CONFIG,
    MONITOR_CONFIG,
    RL_ALGO_CONFIG,
    STAGE2_MODE,
)  # 【新增】引入数据库配置，用于动态获取类型映射
# =================================================

_DATATYPE_MAPPING = None

def _get_datatype_mapping() -> dict:
    """【新增】全局单例/懒加载：读取数据类型配置文件"""
    global _DATATYPE_MAPPING
    if _DATATYPE_MAPPING is None:
        db_type = getattr(DB_CONFIG, "db_type", "mysql").lower()
        config_path = PROJECT_ROOT / "RL" / "datatype" / f"{db_type}.json"
        if config_path.exists():
            with open(config_path, "r", encoding="utf-8") as f:
                _DATATYPE_MAPPING = json.load(f)
        else:
            # 兜底映射
            _DATATYPE_MAPPING = {
                "int": ["int", "integer", "hex"], "float": ["float", "decimal", "numeric"],
                "string": ["char", "varchar", "text", "string"], "date": ["date", "time", "timestamp"]
            }
    return _DATATYPE_MAPPING


def abstract_predicate_for_graph(predicate: str) -> str:
    """
    Compatibility wrapper for predicate abstraction.
    """
    return _structure_abstract_predicate_for_graph(predicate)

def load_pkg_config():
    """
    Compatibility wrapper for loading PKG parser configuration.
    """
    return _structure_load_pkg_config()


def _load_known_texts_from_txt(path: Path) -> set:
    known = set()
    path = Path(path)
    if not path.exists():
        return known
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                known.add(s)
    return known


def _append_text_lines(path: Path, items: list) -> None:
    if not items:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.writelines([f"{p}\n" for p in items])


def _candidate_initial_graph_mode() -> str:
    return "sampled_llm_seeded"


def _load_candidate_records_for_nodes_only() -> list:
    edge_index_path = Path(getattr(CANDIDATE_BITMAP_CONFIG, "edge_index_path"))
    with edge_index_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    records = payload.get("edges", []) if isinstance(payload, dict) else []
    if not isinstance(records, list):
        raise ValueError(f"candidate edge index 的 edges 必须为 list: {edge_index_path}")
    return [record for record in records if isinstance(record, dict)]


def _build_sampled_initial_graph(
    *,
    correct_path: Path,
    analyzer: PredicateStructureAnalyzer,
    candidate_records: list,
    edge_ratio: float,
    log_prefix: str,
):
    """Build a node-complete graph from verified predicate/subtree scaffolds."""
    builder = KnowledgeGraphBuilder()
    builder.initialize_nodes_from_candidate_records(candidate_records)

    # Initial-graph policy is part of the Stage-2 mode, not user configuration.
    # Random/enumeration keep ratio-based seeding without mandatory scaffolds.
    # Other modes retain one verified scaffold per multi-slot operator and the
    # existing permission to exceed the seed budget when necessary.
    require_multislot_scaffold = STAGE2_MODE not in {"base", "enumeration"}
    min_scaffold_per_operator = 1
    allow_scaffold_over_budget = True

    record_by_key = {}
    slot_counts = {}
    for record in candidate_records:
        try:
            key = (
                str(record.get("outer_node_id", "")),
                int(record.get("slot_index", -1)),
                str(record.get("inner_node_id", "")),
            )
        except Exception:
            continue
        if key[0] and key[1] >= 0 and key[2]:
            record_by_key[key] = record
            slot_counts[key[0]] = max(int(slot_counts.get(key[0], 0)), key[1] + 1)

    # A scaffold is the complete set of candidate edges parsed from one known
    # correct predicate. Selecting the whole set preserves recursive support to
    # terminals instead of inserting unrelated independent seed edges.
    known_texts = set()
    scaffolds = []
    if correct_path.exists():
        with correct_path.open("r", encoding="utf-8") as f:
            for line in f:
                predicate = line.strip()
                if not predicate:
                    continue
                known_texts.add(predicate)
                try:
                    parse_result = analyzer.analyze(predicate, expected_type="T_BOOL")
                except Exception:
                    continue
                if parse_result.success and parse_result.ast_root is not None:
                    scaffold_records = []
                    scaffold_ids = set()
                    for edge in parse_result.edges:
                        edge_key = (
                            str(edge.source_node_id),
                            int(edge.slot_id),
                            str(edge.target_node_id),
                        )
                        record = record_by_key.get(edge_key)
                        if not isinstance(record, dict):
                            continue
                        edge_id = str(record.get("edge_id", "") or "")
                        if edge_id and edge_id not in scaffold_ids:
                            scaffold_records.append(record)
                            scaffold_ids.add(edge_id)
                    if scaffold_records:
                        scaffolds.append({
                            "predicate": predicate,
                            "records": scaffold_records,
                            "edge_ids": scaffold_ids,
                            "outer_node_ids": {
                                str(record.get("outer_node_id", ""))
                                for record in scaffold_records
                            },
                            "slot_keys": {
                                (
                                    str(record.get("outer_node_id", "")),
                                    int(record.get("slot_index", -1)),
                                )
                                for record in scaffold_records
                            },
                        })

    verified_records = {
        str(record.get("edge_id", "")): record
        for scaffold in scaffolds
        for record in scaffold["records"]
        if str(record.get("edge_id", ""))
    }

    ratio = max(0.0, min(1.0, float(edge_ratio)))
    target_count = min(
        len(verified_records),
        max(0, int(math.ceil(len(candidate_records) * ratio))),
    )
    selected = []
    selected_ids = set()
    selected_scaffold_predicates = []
    selected_scaffold_outer_counts = {}

    def _add_scaffold(scaffold) -> int:
        added = 0
        for record in scaffold["records"]:
            edge_id = str(record.get("edge_id", "") or "")
            if not edge_id or edge_id in selected_ids:
                continue
            selected.append(record)
            selected_ids.add(edge_id)
            added += 1
        if added > 0:
            selected_scaffold_predicates.append(str(scaffold.get("predicate", "")))
            for outer_node_id in scaffold.get("outer_node_ids", set()):
                outer_key = str(outer_node_id or "")
                if outer_key:
                    selected_scaffold_outer_counts[outer_key] = (
                        int(selected_scaffold_outer_counts.get(outer_key, 0)) + 1
                    )
        return added

    multi_slot_outers = sorted(
        outer_node_id
        for outer_node_id, slot_count in slot_counts.items()
        if int(slot_count) > 1
    )
    missing_scaffold_outers = []

    # Mandatory phase: optionally give each multi-slot operator the configured
    # number of complete verified scaffolds. A scaffold is added as a whole so
    # its recursive downstream support enters the formal graph together.
    if require_multislot_scaffold and min_scaffold_per_operator > 0:
        for outer_node_id in multi_slot_outers:
            choices = [
                scaffold
                for scaffold in scaffolds
                if outer_node_id in scaffold["outer_node_ids"]
            ]
            choices.sort(
                key=lambda scaffold: (
                    len(scaffold["edge_ids"]),
                    str(scaffold.get("predicate", "")),
                )
            )

            for scaffold in choices:
                if (
                    int(selected_scaffold_outer_counts.get(outer_node_id, 0))
                    >= min_scaffold_per_operator
                ):
                    break
                new_count = len(scaffold["edge_ids"] - selected_ids)
                if new_count <= 0:
                    continue
                if (
                    not allow_scaffold_over_budget
                    and len(selected_ids) + new_count > target_count
                ):
                    continue
                _add_scaffold(scaffold)

            if (
                int(selected_scaffold_outer_counts.get(outer_node_id, 0))
                < min_scaffold_per_operator
            ):
                missing_scaffold_outers.append(outer_node_id)

    if STAGE2_MODE == "base":
        target_count = min(len(verified_records), len(candidate_records) // 20)
        shuffled_scaffolds = list(scaffolds)
        Random().shuffle(shuffled_scaffolds)
        for scaffold in shuffled_scaffolds:
            if len(selected_ids) >= target_count:
                break
            new_count = len(scaffold["edge_ids"] - selected_ids)
            if 0 < new_count <= target_count - len(selected_ids):
                _add_scaffold(scaffold)

    # Remaining budget: prefer verified scaffolds that add new operator/slot
    # coverage, then smaller predicates and stable text order.
    while STAGE2_MODE != "base" and len(selected_ids) < target_count:
        selected_slot_keys = {
            (str(record.get("outer_node_id", "")), int(record.get("slot_index", -1)))
            for record in selected
        }
        remaining = [
            scaffold
            for scaffold in scaffolds
            if scaffold["edge_ids"] - selected_ids
        ]
        if not remaining:
            break
        remaining.sort(
            key=lambda scaffold: (
                -len(scaffold["slot_keys"] - selected_slot_keys),
                -len(scaffold["edge_ids"] - selected_ids),
                len(scaffold["edge_ids"]),
                str(scaffold.get("predicate", "")),
            )
        )
        scaffold = remaining[0]
        new_count = len(scaffold["edge_ids"] - selected_ids)
        if len(selected_ids) > 0 and len(selected_ids) + new_count > target_count:
            fitting = [
                item
                for item in remaining
                if len(selected_ids) + len(item["edge_ids"] - selected_ids) <= target_count
            ]
            if not fitting:
                break
            scaffold = fitting[0]
        _add_scaffold(scaffold)

    for record in selected:
        try:
            builder.add_candidate_target_edge(
                outer_node_id=record.get("outer_node_id", ""),
                outer_token=record.get("outer_token", ""),
                outer_return_type=record.get("outer_return_type", ""),
                slot_index=record.get("slot_index", -1),
                inner_node_id=record.get("inner_node_id", ""),
                inner_token=record.get("inner_token", ""),
                inner_return_type=record.get("inner_return_type", ""),
            )
        except Exception as exc:
            print(f"{log_prefix} sampled initial edge skipped: {exc}")

    # Validate that every operator can start from the selected formal graph. A
    # missing verified scaffold is acceptable only when at least one target edge
    # has a ready inner subtree and every sibling slot is already supported.
    def _node_ready(node_id, depth_remaining, visiting):
        node = builder.nodes.get(str(node_id))
        if node is None:
            return False
        required = int(slot_counts.get(str(node_id), 0))
        if bool(node.is_terminal) or required <= 0:
            return True
        if depth_remaining <= 0 or str(node_id) in visiting:
            return False
        next_visiting = set(visiting)
        next_visiting.add(str(node_id))
        if len(node.slots) < required:
            return False
        return all(
            bool(node.slots[slot_index])
            and any(
                _node_ready(child_id, depth_remaining - 1, next_visiting)
                for child_id in node.slots[slot_index]
            )
            for slot_index in range(required)
        )

    depth_limit = max(
        0,
        int(getattr(CANDIDATE_BITMAP_CONFIG, "probe_predicate_max_depth", 3)) - 1,
    )
    bootstrap_outers = set()
    for record in candidate_records:
        outer_node_id = str(record.get("outer_node_id", "") or "")
        inner_node_id = str(record.get("inner_node_id", "") or "")
        try:
            target_slot = int(record.get("slot_index", -1))
        except Exception:
            continue
        outer_node = builder.nodes.get(outer_node_id)
        if outer_node is None or target_slot < 0:
            continue
        siblings_ready = all(
            slot_index == target_slot
            or (
                slot_index < len(outer_node.slots)
                and bool(outer_node.slots[slot_index])
                and any(
                    _node_ready(child_id, depth_limit, {outer_node_id})
                    for child_id in outer_node.slots[slot_index]
                )
            )
            for slot_index in range(int(slot_counts.get(outer_node_id, 0)))
        )
        if siblings_ready and _node_ready(inner_node_id, depth_limit, set()):
            bootstrap_outers.add(outer_node_id)

    selected_scaffold_outers = {
        str(record.get("outer_node_id", "")) for record in selected
    }
    blocked_outers = sorted(
        outer_node_id
        for outer_node_id in slot_counts
        if outer_node_id not in selected_scaffold_outers
        and outer_node_id not in bootstrap_outers
    )
    seed_report = {
        "mode": "sampled_llm_seeded",
        "candidate_edge_count": int(len(candidate_records)),
        "verified_edge_count": int(len(verified_records)),
        "target_edge_count": int(target_count),
        "selected_edge_count": int(builder.total_edges),
        "mandatory_budget_overflow": bool(builder.total_edges > target_count),
        "require_multislot_scaffold": bool(require_multislot_scaffold),
        "min_scaffold_per_operator": int(min_scaffold_per_operator),
        "allow_scaffold_over_budget": bool(allow_scaffold_over_budget),
        "selected_scaffold_predicate_count": int(len(selected_scaffold_predicates)),
        "multi_slot_operator_count": int(len(multi_slot_outers)),
        "missing_scaffold_operators": missing_scaffold_outers,
        "bootstrap_operator_count": int(len(bootstrap_outers)),
        "blocked_operator_count": int(len(blocked_outers)),
        "blocked_operators": blocked_outers,
    }
    builder.initial_seed_report = seed_report

    if STAGE2_MODE == "base":
        print("普通随机实验")
    else:
        print(
            f"{log_prefix} sampled_llm_seeded initial graph: "
            f"candidate_edges={len(candidate_records)}, verified_edges={len(verified_records)}, "
            f"seed_edges={builder.total_edges}, ratio={ratio:.4f}, "
            f"missing_scaffolds={missing_scaffold_outers}, blocked={blocked_outers}"
        )
    if blocked_outers and require_multislot_scaffold:
        raise ValueError(
            "sampled_llm_seeded preflight failed: operators have neither a verified "
            f"scaffold nor a valid bootstrap path: {blocked_outers}"
        )
    return builder, known_texts


def _flush_runtime_stats_batch(runtime_stats, stats_buffer: list) -> None:
    if not stats_buffer:
        return
    try:
        if hasattr(runtime_stats, "record_sql_success_batch"):
            runtime_stats.record_sql_success_batch(stats_buffer)
        else:
            for item in stats_buffer:
                runtime_stats.record_sql_success(
                    predicate=item["predicate"],
                    parse_result=item["parse_result"],
                    step=item.get("step"),
                    source=item.get("source", "pkg_global_graph"),
                    graph_new_nodes=item.get("graph_new_nodes", 0),
                    graph_new_edges=item.get("graph_new_edges", 0),
                )
    finally:
        stats_buffer.clear()


_PARSE_WORKER_ANALYZER = None


def _parse_predicate_worker(predicate: str):
    """
    多进程解析 worker。

    注意：
        worker 只做 analyzer.analyze()；
        不修改 KnowledgeGraphBuilder；
        不写 runtime stats；
        不访问数据库。
    """
    global _PARSE_WORKER_ANALYZER

    try:
        if _PARSE_WORKER_ANALYZER is None:
            pkg_config = load_pkg_config()
            _PARSE_WORKER_ANALYZER = PredicateStructureAnalyzer(
                config=pkg_config,
                strict_consumption=True,
            )

        parse_result = _PARSE_WORKER_ANALYZER.analyze(
            predicate,
            expected_type="T_BOOL",
        )
        return predicate, parse_result, None

    except Exception as e:
        return predicate, None, str(e)


def _prepare_initial_global_graph_builder(
    correct_txt_path,
    graph_path=None,
    log_prefix: str = "[Master Process]",
):
    """
    复用原先 global_graph_manager 中的初始图谱构建逻辑。

    逻辑保持原样：
        1. 如果 global_knowledge_graph.json 存在，优先加载缓存图谱；
        2. 如果没有加载到图谱节点，则读取 Correct_Collection.txt；
        3. 使用 PredicateStructureAnalyzer 解析谓词；
        4. 解析成功后调用 KnowledgeGraphBuilder.add_predicate；
        5. 保存到 global_knowledge_graph.json。
    """
    pkg_config = load_pkg_config()
    analyzer = PredicateStructureAnalyzer(config=pkg_config, strict_consumption=True)
    builder = KnowledgeGraphBuilder()

    kg_output_dir = PKG_DIR / "knowledge_graph"
    kg_output_dir.mkdir(parents=True, exist_ok=True)

    resolved_graph_path = (
        Path(graph_path)
        if graph_path is not None
        else kg_output_dir / "global_knowledge_graph.json"
    )
    resolved_graph_path.parent.mkdir(parents=True, exist_ok=True)

    correct_path = Path(correct_txt_path)
    initial_graph_mode = _candidate_initial_graph_mode()

    known_texts = set()
    if resolved_graph_path.exists():
        try:
            builder.load_from_json(str(resolved_graph_path))
            known_texts = _load_known_texts_from_txt(correct_path)
            if STAGE2_MODE == "base":
                print("普通随机实验")
            else:
                print(
                    f"{log_prefix} 已从缓存图谱加载: {resolved_graph_path}。"
                    f"已读取 {len(known_texts)} 条 Correct 文本用于去重。"
                )
        except Exception as e:
            print(f"{log_prefix} 缓存图谱加载失败，将执行一次性基础图谱构建: {e}")

    if not builder.nodes:
        if initial_graph_mode == "nodes_only":
            candidate_records = _load_candidate_records_for_nodes_only()
            builder.initialize_nodes_from_candidate_records(candidate_records)
            print(
                f"{log_prefix} nodes_only 初始图谱构建完成："
                f"nodes={len(builder.nodes)}，edges={builder.total_edges}。"
            )
            try:
                builder.save_to_json(str(resolved_graph_path))
                print(f"{log_prefix} nodes_only 初始图谱已成功保存至: {resolved_graph_path}")
            except Exception as e:
                print(f"{log_prefix} nodes_only 初始图谱保存失败: {e}")
        elif initial_graph_mode == "sampled_llm_seeded":
            candidate_records = _load_candidate_records_for_nodes_only()
            builder, known_texts = _build_sampled_initial_graph(
                correct_path=correct_path,
                analyzer=analyzer,
                candidate_records=candidate_records,
                edge_ratio=0.05,
                log_prefix=log_prefix,
            )
            try:
                builder.save_to_json(str(resolved_graph_path))
                if STAGE2_MODE != "base":
                    print(f"{log_prefix} sampled initial graph saved: {resolved_graph_path}")
            except Exception as e:
                print(f"{log_prefix} sampled initial graph save failed: {e}")
        else:
            print(f"{log_prefix} 未加载到缓存图谱，执行一次性基础图谱构建。")
            if correct_path.exists():
                with correct_path.open("r", encoding="utf-8") as f:
                    for line in f:
                        s = line.strip()
                        if not s or s in known_texts:
                            continue
                        known_texts.add(s)
                        try:
                            parse_result = analyzer.analyze(s, expected_type="T_BOOL")
                            if parse_result.success and parse_result.ast_root is not None:
                                builder.add_predicate(parse_result.ast_root)
                            else:
                                print(
                                    f"[Debug] Syntax rejected, cannot parse AST: "
                                    f"{parse_result.abstract_predicate or s} | "
                                    f"error: {parse_result.error}"
                                )
                        except Exception as e:
                            print(f"{log_prefix} warning: initial base predicate skipped '{s}' | error: {e}")

                print(
                    f"{log_prefix} 基础图谱构建完成！共加载 {len(known_texts)} 条初始谓词。"
                    f"初始总边数: {builder.total_edges}"
                )
                try:
                    builder.save_to_json(str(resolved_graph_path))
                    print(f"{log_prefix} 初始基础图谱已成功保存至: {resolved_graph_path}")
                except Exception as e:
                    print(f"{log_prefix} 初始图谱保存失败: {e}")
            else:
                print(f"{log_prefix} 未找到初始集合，将从零开始构建。")

    return builder, analyzer, known_texts, resolved_graph_path, correct_path


def ensure_initial_global_knowledge_graph(correct_txt_path=None, graph_path=None):
    """
    确保正式 global_knowledge_graph.json 存在。

    注意：
        这里不重新实现建图逻辑；
        只是调用从 global_graph_manager 原始逻辑中抽出的
        _prepare_initial_global_graph_builder()。
    """
    from RL.config import RL_ENV_CONFIG

    resolved_correct_path = Path(
        correct_txt_path
        or getattr(RL_ENV_CONFIG, "init_correct_predicates_path")
    )

    builder, _analyzer, known_texts, resolved_graph_path, correct_path = _prepare_initial_global_graph_builder(
        correct_txt_path=resolved_correct_path,
        graph_path=graph_path,
        log_prefix="[GlobalGraphBootstrap]",
    )

    return {
        "graph_path": str(resolved_graph_path),
        "correct_path": str(correct_path),
        "known_text_count": int(len(known_texts)),
        "node_count": int(len(builder.nodes)),
        "edge_count": int(builder.total_edges),
        "graph_exists": bool(resolved_graph_path.exists()),
        "initial_seed_report": getattr(builder, "initial_seed_report", {}),
    }


def global_graph_manager(queue, batch_size, correct_txt_path, progress_total_steps=None):
    """
    Master 进程的核心循环函数。
    由 train_actor_critic.py 在独立进程中启动。
    """
    print("[Master Process] PKG 增量建图管理器已启动。")

    builder, analyzer, known_texts, graph_path, correct_path = _prepare_initial_global_graph_builder(
        correct_txt_path=correct_txt_path,
        graph_path=None,
        log_prefix="[Master Process]",
    )

    runtime_stats = get_default_runtime_stats()

    # 本次训练启动时的图边数是所有进度记录的固定基线。
    initial_edge_count = int(builder.total_edges)
    progress_initialized_runs = set()
    progress_last_step_by_run = {}

    def _append_progress_snapshot(step: int, payload: dict) -> None:
        """由持有正式 builder 的图谱进程统一追加消融实验进度 CSV。"""
        if not bool(getattr(RL_ALGO_CONFIG, "progress_log_enabled", True)):
            return
        if not isinstance(payload, dict):
            return

        try:
            run_id = str(payload.get("run_id", "") or "").strip()
            if not run_id:
                return

            step = int(step)
            progress_dir = Path(
                getattr(
                    RL_ALGO_CONFIG,
                    "progress_log_dir",
                    PROJECT_ROOT / "RL" / "rl_logs" / "ablation_progress",
                )
            )
            progress_dir.mkdir(parents=True, exist_ok=True)
            progress_path = progress_dir / f"{run_id}_progress.csv"

            if run_id not in progress_initialized_runs:
                with progress_path.open("w", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        "run_id",
                        "seed",
                        "global_step",
                        "elapsed_seconds",
                        "initial_edges",
                        "current_edges",
                        "new_edges",
                        "construct_failures",
                        "construct_failure_rate",
                        "selected_done_count",
                        "selected_done_rate",
                        "selected_probing_count",
                        "selected_probing_rate",
                        "selected_local_tested_count",
                        "selected_local_tested_rate",
                        "avg_pool_unfinished_count",
                        "rewarded_new_edge_count",
                        "pool_done_rate",
                        "safe_fallback_used_count",
                        "reservation_attempt_count",
                        "reservation_conflict_count",
                        "reservation_conflict_rate",
                        "reservation_action_substitution_count",
                    ])
                    writer.writerow([
                        run_id,
                        int(payload.get("seed", 42)),
                        0,
                        "0.000",
                        initial_edge_count,
                        initial_edge_count,
                        0,
                        0,
                        "0.000000",
                        0,
                        "0.000000",
                        0,
                        "0.000000",
                        0,
                        "0.000000",
                        "0.000",
                        0,
                        "0.000000",
                        0,
                        0,
                        0,
                        "0.000000",
                        0,
                    ])
                progress_initialized_runs.add(run_id)
                progress_last_step_by_run[run_id] = 0

            # 同一步只记录一次；训练结束恰逢周期打点时不会产生重复行。
            if step <= int(progress_last_step_by_run.get(run_id, -1)):
                return

            current_edges = int(builder.total_edges)
            new_edges = current_edges - initial_edge_count
            elapsed_seconds = float(payload.get("elapsed_seconds", 0.0) or 0.0)
            construct_failures = int(payload.get("construct_failures", 0) or 0)
            construct_failure_rate = float(
                payload.get("construct_failure_rate", 0.0) or 0.0
            )
            selected_done_count = int(payload.get("selected_done_count", 0) or 0)
            selected_done_rate = float(payload.get("selected_done_rate", 0.0) or 0.0)
            selected_probing_count = int(payload.get("selected_probing_count", 0) or 0)
            selected_probing_rate = float(payload.get("selected_probing_rate", 0.0) or 0.0)
            selected_local_tested_count = int(
                payload.get("selected_local_tested_count", 0) or 0
            )
            selected_local_tested_rate = float(
                payload.get("selected_local_tested_rate", 0.0) or 0.0
            )
            avg_pool_unfinished_count = float(
                payload.get("avg_pool_unfinished_count", 0.0) or 0.0
            )
            rewarded_new_edge_count = int(
                payload.get("rewarded_new_edge_count", 0) or 0
            )
            pool_done_rate = float(payload.get("pool_done_rate", 0.0) or 0.0)
            safe_fallback_used_count = int(
                payload.get("safe_fallback_used_count", 0) or 0
            )
            reservation_attempt_count = int(
                payload.get("reservation_attempt_count", 0) or 0
            )
            reservation_conflict_count = int(
                payload.get("reservation_conflict_count", 0) or 0
            )
            reservation_conflict_rate = float(
                payload.get("reservation_conflict_rate", 0.0) or 0.0
            )
            reservation_action_substitution_count = int(
                payload.get("reservation_action_substitution_count", 0) or 0
            )

            with progress_path.open("a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    run_id,
                    int(payload.get("seed", 42)),
                    step,
                    f"{elapsed_seconds:.3f}",
                    initial_edge_count,
                    current_edges,
                    new_edges,
                    construct_failures,
                    f"{construct_failure_rate:.6f}",
                    selected_done_count,
                    f"{selected_done_rate:.6f}",
                    selected_probing_count,
                    f"{selected_probing_rate:.6f}",
                    selected_local_tested_count,
                    f"{selected_local_tested_rate:.6f}",
                    f"{avg_pool_unfinished_count:.3f}",
                    rewarded_new_edge_count,
                    f"{pool_done_rate:.6f}",
                    safe_fallback_used_count,
                    reservation_attempt_count,
                    reservation_conflict_count,
                    f"{reservation_conflict_rate:.6f}",
                    reservation_action_substitution_count,
                ])

            progress_last_step_by_run[run_id] = step
        except Exception as exc:
            # 统计写入与正式建图隔离，CSV 失败只告警，不中断图谱和训练流程。
            print(f"[Progress Log Warning] Failed to append progress CSV: {exc}", flush=True)

    buffer = []
    stats_buffer = []
    graph_buffer = []
    queued_predicates = set()
    start_time = time.time()
    # 最近一次 SYNC_PRINT 确认的全局 PPO 步数；首次同步前保持 -1。
    # 该值只写入 monitor_log.csv，不参与正式图解析、加边或奖励计算。
    latest_rl_step = -1

    parse_workers = int(os.environ.get(
        "PKG_PARSE_WORKERS",
        getattr(MONITOR_CONFIG, "pkg_parse_workers", 4),
    ))
    parse_workers = max(1, parse_workers)

    parse_min_batch = int(os.environ.get(
        "PKG_PARSE_MIN_BATCH",
        getattr(MONITOR_CONFIG, "pkg_parse_min_batch", 64),
    ))
    parse_min_batch = max(1, parse_min_batch)

    parse_pool = None
    if parse_workers > 1:
        parse_pool = ProcessPoolExecutor(max_workers=parse_workers)
        print(
            f"[Master Process] PKG parallel parser enabled: "
            f"workers={parse_workers}, min_batch={parse_min_batch}"
        )
    else:
        print("[Master Process] PKG parallel parser disabled; using serial parser.")

    # 监控日志路径 (从配置或者手动指定)
    monitor_csv = PROJECT_ROOT / "output" / "monitor_log.csv"
    monitor_csv.parent.mkdir(parents=True, exist_ok=True)

    # 如果是全新运行，写入 CSV 表头
    if not monitor_csv.exists():
        with open(monitor_csv, 'w', encoding='utf-8') as f:
            f.write("Time(s),RL_Step,New_Nodes,New_Edges,Total_Edges\n")

    # 统计信息
    total_received = 0
    total_new_edges_session = 0

    def _event_predicate(event) -> str:
        if isinstance(event, dict):
            return str(event.get("predicate", "") or "")
        return str(event or "")

    def _event_target_edge(event):
        if not isinstance(event, dict):
            return None
        target_edge = event.get("target_edge")
        return target_edge if isinstance(target_edge, dict) else None

    def _append_predicate_once(predicate_str: str) -> None:
        queued_predicates.discard(predicate_str)
        if predicate_str and predicate_str not in known_texts:
            known_texts.add(predicate_str)
            buffer.append(predicate_str)

    def _parse_batch(events: list):
        """
        批量解析图谱事件中的谓词，并保留事件携带的目标边元数据。
        """
        nonlocal parse_pool

        if not events:
            return []

        predicates = [_event_predicate(event) for event in events]

        if parse_pool is None or len(events) < parse_min_batch:
            results = []
            for event, pred in zip(events, predicates):
                try:
                    pr = analyzer.analyze(pred, expected_type="T_BOOL")
                    results.append((event, pred, pr, None))
                except Exception as e:
                    results.append((event, pred, None, str(e)))
            return results

        try:
            parsed = list(parse_pool.map(_parse_predicate_worker, predicates))
            return [
                (event, predicate_str, parse_result, parse_error)
                for event, (predicate_str, parse_result, parse_error) in zip(events, parsed)
            ]
        except Exception as e:
            print(
                f"[Master Process Warning] parallel parse failed, "
                f"fallback to serial parse: {e}"
            )
            results = []
            for event, pred in zip(events, predicates):
                try:
                    pr = analyzer.analyze(pred, expected_type="T_BOOL")
                    results.append((event, pred, pr, None))
                except Exception as se:
                    results.append((event, pred, None, str(se)))
            return results

    def _process_graph_batch(events: list, rl_step: int) -> None:
        """
        处理一批本轮新增 Correct 图谱事件。

        analyzer.analyze() 可以并行；
        正式图写入、stats_buffer、文件写入仍保持主进程串行。
        rl_step 仅用于 monitor_log.csv，表示最近一次确认的全局 PPO 步数。
        """
        nonlocal total_received, total_new_edges_session

        if not events:
            return

        parse_outputs = _parse_batch(events)

        for event, predicate_str, parse_result, parse_error in parse_outputs:
            total_received += 1

            new_nodes = 0
            new_edges = 0

            if parse_error is not None:
                print(
                    f"[Master Process] warning: predicate parse worker failed "
                    f"'{predicate_str}' | error: {parse_error}"
                )
                try:
                    runtime_stats.record_unknown_success(
                        predicate=predicate_str,
                        parse_error=str(parse_error),
                        abstract_predicate="",
                        step=None,
                        source="pkg_global_graph",
                    )
                except Exception as stat_e:
                    print(
                        f"[RuntimeStats Warning] exceptional success predicate stats failed "
                        f"'{predicate_str}' | error: {stat_e}"
                    )

                _append_predicate_once(predicate_str)
                continue

            try:
                if parse_result.success and parse_result.ast_root is not None:
                    target_edge = _event_target_edge(event)
                    if target_edge is None:
                        new_nodes, new_edges = builder.add_predicate(parse_result.ast_root)
                    else:
                        new_nodes, new_edges = builder.add_candidate_target_edge(**target_edge)

                    if new_edges > 0:
                        total_new_edges_session += new_edges
                        elapsed = time.time() - start_time
                        with open(monitor_csv, "a", encoding="utf-8") as f:
                            f.write(
                                f"{elapsed:.1f},{int(rl_step)},{new_nodes},{new_edges},{builder.total_edges}\n"
                            )
                else:
                    print(
                        f"[Master Process] warning: executable predicate parse failed, skipped graph insert "
                        f"'{predicate_str}' | abstract predicate: {parse_result.abstract_predicate} | "
                        f"error: {parse_result.error}"
                    )

                if (isinstance(event, dict)
                        and str(getattr(DB_CONFIG, "db_type", "")).lower()
                        in {"oracle", "oracledb", "oracle_ai26"}):
                    parse_result.oracle_executed_sql = event.get("oracle_executed_sql", "")
                    parse_result.oracle_target_edge = _event_target_edge(event)
                if (isinstance(event, dict)
                        and str(getattr(DB_CONFIG, "db_type", "")).lower() == "sqlserver"):
                    parse_result.sqlserver_executed_sql = event.get("sqlserver_executed_sql", "")
                stats_buffer.append({
                    "predicate": predicate_str,
                    "parse_result": parse_result,
                    "step": None,
                    "source": "pkg_global_graph",
                    "graph_new_nodes": new_nodes,
                    "graph_new_edges": new_edges,
                })

            except Exception as e:
                print(
                    f"[Master Process] warning: predicate graph update skipped "
                    f"'{predicate_str}' | error: {e}"
                )
                try:
                    runtime_stats.record_unknown_success(
                        predicate=predicate_str,
                        parse_error=str(e),
                        abstract_predicate="",
                        step=None,
                        source="pkg_global_graph",
                    )
                except Exception as stat_e:
                    print(
                        f"[RuntimeStats Warning] exceptional success predicate stats failed "
                        f"'{predicate_str}' | error: {stat_e}"
                    )

            _append_predicate_once(predicate_str)

        if len(buffer) >= batch_size:
            _append_text_lines(correct_path, buffer)
            buffer.clear()
            _flush_runtime_stats_batch(runtime_stats, stats_buffer)

    while True:
        data = queue.get()

        # 1. 终止信号
        if data == "DONE":
            break

        # 2. 同步打印信号 (来自 Callbacks)
        if isinstance(data, tuple) and data[0] == "SYNC_PRINT":
            step = data[1]
            progress_payload = data[2] if len(data) > 2 and isinstance(data[2], dict) else None
            # 二元和三元 SYNC_PRINT 都更新监控步数；三元消息额外写进度 CSV。
            latest_rl_step = int(step)

            if graph_buffer:
                pending = list(graph_buffer)
                graph_buffer.clear()
                _process_graph_batch(pending, latest_rl_step)

            if buffer:
                _append_text_lines(correct_path, buffer)
                buffer.clear()

            _flush_runtime_stats_batch(runtime_stats, stats_buffer)

            try:
                builder.save_to_json(str(graph_path))
            except Exception:
                pass

            # 必须在待处理谓词入图并保存之后记录，确保 current_edges 使用最新正式图。
            if progress_payload is not None:
                _append_progress_snapshot(step, progress_payload)

            # 打印对齐日志！
            print_rl_progress(step, progress_total_steps)
            continue

        # 3. 正常接收到谓词字符串或 candidate_bitmap 目标边事件。
        # 文本去重只控制 Correct_Collection 写入，不能阻止另一条目标边进入正式图。
        is_target_event = isinstance(data, dict) and isinstance(data.get("target_edge"), dict)
        predicate_str = _event_predicate(data)
        if not predicate_str:
            continue
        if not is_target_event and (
            predicate_str in known_texts or predicate_str in queued_predicates
        ):
            continue
        if not is_target_event:
            queued_predicates.add(predicate_str)

        # --- 【新增：极其轻量的实时打点，供外部平滑监控】 ---
        # 无论 batch 缓冲是否满，只要收到全新的正确谓词，立即追加一个 "1"
        with open(PROJECT_ROOT / "output" / "monitor_correct.log", 'a', encoding='utf-8') as f:
            f.write("1\n")
        # --------------------------------------------------

        graph_buffer.append(data)

        if len(graph_buffer) >= batch_size:
            pending = list(graph_buffer)
            graph_buffer.clear()
            _process_graph_batch(pending, latest_rl_step)

        continue

    # 循环结束，先处理尚未解析的 graph_buffer，再刷文本和统计。
    if graph_buffer:
        pending = list(graph_buffer)
        graph_buffer.clear()
        _process_graph_batch(pending, latest_rl_step)

    if buffer:
        _append_text_lines(correct_path, buffer)
        buffer.clear()

    _flush_runtime_stats_batch(runtime_stats, stats_buffer)
    try:
        runtime_stats.flush_joint_signature_evidence()
    except Exception as e:
        report_joint_signature_flush_failure(e, final=True)

    # 将全局唯一大图保存为 JSON，第三阶段可以直接读取这个大图，跳过 parse 阶段！
    try:
        # 【关键修改】：强制将 Path 对象转换为字符串 str()，防止第三方构建器内部报错
        builder.save_to_json(str(graph_path))
        print("[Master Process] Exiting. Global knowledge graph saved.", flush=True)
    except Exception as e:
        import traceback
        print(f"[Master Process 致命错误] 保存全局大图时发生异常，未能成功写入磁盘: {e}", flush=True)
        traceback.print_exc()

    if parse_pool is not None:
        parse_pool.shutdown(wait=True)
