import os
import sys
import multiprocessing as mp
import subprocess
import json
import re
import time
import datetime

from RL.runtime_logging import runtime_warning_scope

os.environ["OMP_NUM_THREADS"] = "1"

if __name__ == '__main__':
    try:
        mp.set_start_method('spawn', force=True)
        print("[Main] Multiprocessing start method set to 'spawn'")
    except RuntimeError:
        pass

def transformer_encoder():
    print("开始执行运算符编码流程...")
    from transformer_encoder.encode_operators import run_encoding
    run_encoding()
    print("运算符编码流程执行完毕。")


def run_operator_embedding_extraction():
    print("开始生成算子向量与相似算子表...")
    from transformer_encoder.operator_embedding_extractor import run_extraction
    run_extraction()
    print("算子向量与相似算子表生成完毕。")


def run_contrastive_learning_step(iter_n):
    from Contrastive_Learning import run_contrastive_learning
    run_contrastive_learning(
        iter_n=iter_n,
        model_dir_name="codet5-small",
        batch_size=12,
        num_epochs=8,
        learning_rate=1e-5,
        temperature=0.07,
    )


def run_candidate_bitmap_bootstrap():
    from pathlib import Path

    from RL.config import CANDIDATE_BITMAP_CONFIG, RL_ENV_CONFIG, STAGE2_MODE

    def _bootstrap_log(*args, **kwargs):
        if STAGE2_MODE != "base":
            print(*args, **kwargs)

    if not bool(getattr(CANDIDATE_BITMAP_CONFIG, "enabled", True)):
        _bootstrap_log("[CandidateBitmapBootstrap] skipped: CANDIDATE_BITMAP_CONFIG.enabled=False")
        return

    edge_index_path = Path(getattr(CANDIDATE_BITMAP_CONFIG, "edge_index_path"))
    bitmap_npz_path = Path(getattr(CANDIDATE_BITMAP_CONFIG, "bitmap_npz_path"))
    graph_path = Path(getattr(CANDIDATE_BITMAP_CONFIG, "global_knowledge_graph_path"))
    rules_path = Path(getattr(CANDIDATE_BITMAP_CONFIG, "rules_path"))

    _bootstrap_log("\n>>> [Stage 17-9] Candidate bitmap bootstrap before RL")
    _bootstrap_log(f"[CandidateBitmapBootstrap] edge_index_path = {edge_index_path}")
    _bootstrap_log(f"[CandidateBitmapBootstrap] bitmap_npz_path = {bitmap_npz_path}")
    _bootstrap_log(f"[CandidateBitmapBootstrap] graph_path = {graph_path}")

    from RL.candidate_bitmap import (
        build_and_save_candidate_bitmap,
        build_and_save_observed_bitmap,
        refresh_candidate_edge_scores_by_pattern_transfer,
    )

    need_build_universe = (
        not edge_index_path.exists()
        or not bitmap_npz_path.exists()
    )

    initial_graph_mode = "sampled_llm_seeded"

    if STAGE2_MODE == "base" and graph_path.exists():
        print("普通随机实验")

    if (
        not graph_path.exists()
        and initial_graph_mode in {"nodes_only", "sampled_llm_seeded"}
        and need_build_universe
    ):
        _bootstrap_log(
            "[CandidateBitmapBootstrap] initial graph mode requires the candidate edge universe, "
            "building candidate edge universe..."
        )
        candidate_report = build_and_save_candidate_bitmap(rules_path)
        _bootstrap_log(
            "[CandidateBitmapBootstrap] candidate edge universe 构建完成："
            f"num_edges={candidate_report.get('num_edges')}, "
            f"by_level={candidate_report.get('by_level')}"
        )
        need_build_universe = False

    if not graph_path.exists():
        _bootstrap_log(
            "[CandidateBitmapBootstrap] global_knowledge_graph.json 不存在，"
            f"按 initial_graph_mode={initial_graph_mode!r} 创建初始图。"
        )
        from RL.pkg_manager import ensure_initial_global_knowledge_graph

        graph_report = ensure_initial_global_knowledge_graph(
            correct_txt_path=getattr(RL_ENV_CONFIG, "init_correct_predicates_path"),
            graph_path=graph_path,
        )
        _bootstrap_log(
            "[CandidateBitmapBootstrap] 初始正式谓词知识图准备完成："
            f"nodes={graph_report.get('node_count')}, "
            f"edges={graph_report.get('edge_count')}, "
            f"path={graph_report.get('graph_path')}"
        )

    if not graph_path.exists():
        raise FileNotFoundError(
            "[CandidateBitmapBootstrap] global_knowledge_graph.json 仍然不存在，"
            "无法根据正式知识图刷新 observed_bitmap。正确路径应为："
            f"{graph_path}"
        )

    if need_build_universe:
        _bootstrap_log(
            "[CandidateBitmapBootstrap] candidate bitmap 主文件缺失，"
            "开始构建 candidate edge universe..."
        )
        candidate_report = build_and_save_candidate_bitmap(rules_path)
        _bootstrap_log(
            "[CandidateBitmapBootstrap] candidate edge universe 构建完成："
            f"num_edges={candidate_report.get('num_edges')}, "
            f"by_level={candidate_report.get('by_level')}"
        )
    else:
        _bootstrap_log("[CandidateBitmapBootstrap] candidate bitmap 主文件已存在，跳过全集重建。")

    _bootstrap_log("[CandidateBitmapBootstrap] 开始刷新 observed/missing/available bitmap...")
    observed_report = build_and_save_observed_bitmap(graph_path)
    candidate_info = observed_report.get("candidate", {}) if isinstance(observed_report, dict) else {}
    graph_info = observed_report.get("graph", {}) if isinstance(observed_report, dict) else {}

    _bootstrap_log(
        "[CandidateBitmapBootstrap] observed bitmap 刷新完成："
        f"graph_edges={graph_info.get('edge_count')}, "
        f"matched={graph_info.get('matched_candidate_edge_count')}, "
        f"candidate_edges={candidate_info.get('candidate_edge_count')}, "
        f"observed={candidate_info.get('observed_edge_count')}, "
        f"missing={candidate_info.get('missing_edge_count')}, "
        f"available={candidate_info.get('available_initial_edge_count')}"
    )

    if bool(getattr(RL_ENV_CONFIG, "candidate_bitmap_enumeration_enabled", False)):
        order_mode = str(
            getattr(
                RL_ENV_CONFIG,
                "candidate_bitmap_enumeration_order_mode",
                "independent_random",
            )
        )
        order_group_count = int(
            getattr(RL_ENV_CONFIG, "candidate_bitmap_enumeration_order_group_count", 8)
        )
        entropy_bias = float(
            getattr(RL_ENV_CONFIG, "candidate_bitmap_enumeration_entropy_bias", 0.35)
        )
        structural_ratio = float(
            getattr(
                RL_ENV_CONFIG,
                "candidate_bitmap_enumeration_structural_ratio",
                0.70,
            )
        )
        if order_mode.strip().lower() == "cycle_stratified_hard_random":
            _bootstrap_log(
                "[CandidateBitmapBootstrap] enumeration mode enabled: "
                "skip representation/HNSW/entropy score refresh; "
                "use per-cycle independent worker orders "
                f"(structural_ratio={structural_ratio:.3f}, random_ratio="
                f"{1.0 - structural_ratio:.3f})."
            )
        elif order_mode.strip().lower() == "top_down_operator_block":
            _bootstrap_log(
                "[CandidateBitmapBootstrap] enumeration mode enabled: "
                "skip representation/HNSW/entropy score refresh; "
                "use fixed top-down outer-operator blocks "
                f"(order_groups={order_group_count}, entropy_enabled=False)."
            )
        else:
            _bootstrap_log(
                "[CandidateBitmapBootstrap] enumeration mode enabled: "
                "skip representation/HNSW/entropy score refresh; "
                f"use configured fixed full candidate order (bias={entropy_bias:.3f})."
            )
        return

    representation_guidance_enabled = bool(
        getattr(
            CANDIDATE_BITMAP_CONFIG,
            "candidate_representation_guidance_enabled",
            True,
        )
    )
    _bootstrap_log(
        "[CandidateBitmapBootstrap] representation guidance = "
        f"{representation_guidance_enabled}; "
        "HNSW / entropy remain controlled by their own switches."
    )

    if bool(getattr(CANDIDATE_BITMAP_CONFIG, "refresh_pattern_transfer_before_rl", True)):
        if representation_guidance_enabled:
            _bootstrap_log(
                "[CandidateBitmapBootstrap] 开始刷新 pattern_transfer_score / edge_score "
                "（阶段18-3，RL 启动前强制刷新）..."
            )
        else:
            _bootstrap_log(
                "[CandidateBitmapBootstrap] 表示迁移帮助已关闭："
                "pattern_transfer_score置0，并继续刷新HNSW/熵组织数据..."
            )
        score_report = refresh_candidate_edge_scores_by_pattern_transfer(graph_path=graph_path)

        _bootstrap_log(
            "[CandidateBitmapBootstrap] pattern_transfer_score 刷新完成："
            f"edges={score_report.get('edge_count')}, "
            f"positive={score_report.get('positive_pattern_transfer_score_count')}, "
            f"zero={score_report.get('zero_pattern_transfer_score_count')}, "
            f"max={score_report.get('max_pattern_transfer_score')}, "
            f"mean={score_report.get('mean_pattern_transfer_score')}, "
            f"representation_guidance_enabled={score_report.get('representation_guidance_enabled')}, "
            f"operator_similarity_loaded={score_report.get('operator_similarity_loaded')}"
        )
    else:
        _bootstrap_log(
            "[CandidateBitmapBootstrap] 跳过 pattern_transfer_score 刷新："
            "CANDIDATE_BITMAP_CONFIG.refresh_pattern_transfer_before_rl=False"
        )


def run_rl_training(disable_periodic_trigger: bool = False):
    run_candidate_bitmap_bootstrap()

    if not disable_periodic_trigger:
        from RL.train_actor_critic import run_training
        run_training()
        return

    from RL.config import RECYCLE_CONFIG

    old_value = RECYCLE_CONFIG.recycle_enable_periodic_trigger
    RECYCLE_CONFIG.recycle_enable_periodic_trigger = False

    try:
        print("[Main] Periodic recycle trigger disabled for this RL run.")
        from RL.train_actor_critic import run_training
        run_training()
    finally:
        RECYCLE_CONFIG.recycle_enable_periodic_trigger = old_value
        print("[Main] Periodic recycle trigger restored after RL run.")


_STAGE3_SAMPLING_TIMING_SECONDS = (
    "sampler_setup_seconds",
    "counted_dag_load_seconds",
    "counted_dag_decode_seconds",
    "successor_ordering_seconds",
    "decode_cache_preparation_seconds",
    "graph_preparation_seconds",
    "predicate_traversal_render_seconds",
    "predicate_output_write_seconds",
    "sampling_overhead_seconds",
)
_STAGE3_SAMPLING_TIMING_COUNTERS = (
    "predicate_traversal_attempts",
    "predicate_output_write_calls",
)


def _stage3_sampling_phase_timings(sampling_report):
    raw = (
        sampling_report.get("sampling_timings", {})
        if isinstance(sampling_report, dict)
        else {}
    )
    return {
        name: max(0.0, float(raw.get(name, 0.0) or 0.0))
        for name in _STAGE3_SAMPLING_TIMING_SECONDS
    }


def run_predicate_knowledge_graph_traverse(
    max_depth=None,
    deadline=None,
    downstream_cost_model=None,
    runtime_signature_feedback=None,
    metrics_recorder=None,
    cumulative_generated_base=0,
    enumerate_all=False,
    localized_random=False,
    output_label=None,
):
    print("开始执行谓词知识图处理阶段...")
    from pkg.unfolder import unfolder
    from pkg.counter import counter
    from pkg.sampler import sampler

    print(">>> (跳过建图，直接读取 RL 阶段产出的全局大图)")
    if metrics_recorder is not None:
        metrics_recorder.start_phase("dag_unfolding", max_depth)
    unfold_started = time.monotonic()
    unfold_report = unfolder(max_depth=max_depth, deadline=deadline)
    unfold_elapsed = time.monotonic() - unfold_started
    unfold_detail_timings = dict(unfold_report.get("phase_timings", {}) or {})
    if metrics_recorder is not None:
        metrics_recorder.finish_phase("dag_unfolding")
        for phase_name in (
            "type_projection_seconds",
            "dag_expansion_seconds",
            "dag_output_seconds",
        ):
            metrics_recorder.add_phase_detail(
                phase_name,
                unfold_detail_timings.get(phase_name, 0.0),
            )
        metrics_recorder.record(
            event="dag_unfolding_finished",
            force=True,
            phase="dag_unfolding",
            depth=max_depth,
        )
    if not unfold_report.get("success", False) or not unfold_report.get(
        "round_complete", False
    ):
        print("谓词知识图处理阶段在 DAG 展开阶段停止。")
        return {
            "success": bool(unfold_report.get("success", False)),
            "round_complete": False,
            "phase": "unfolding",
            "error": unfold_report.get("error"),
            "total_generated": 0,
            "output_files": [],
            "sampling_elapsed_seconds": 0.0,
            "phase_timings": {
                "dag_unfolding_seconds": unfold_elapsed,
                **unfold_detail_timings,
                "combination_counting_seconds": 0.0,
                "predicate_sampling_seconds": 0.0,
                **_stage3_sampling_phase_timings({}),
            },
        }

    if metrics_recorder is not None:
        metrics_recorder.start_phase("combination_counting", max_depth)
    count_started = time.monotonic()
    count_report = counter(
        deadline=deadline,
        input_files=unfold_report.get("output_files"),
    )
    count_elapsed = time.monotonic() - count_started
    if metrics_recorder is not None:
        metrics_recorder.finish_phase("combination_counting")
        metrics_recorder.record(
            event="combination_counting_finished",
            force=True,
            phase="combination_counting",
            depth=max_depth,
        )
    if not count_report.get("success", False) or not count_report.get(
        "round_complete", False
    ):
        print("谓词知识图处理阶段在组合计数阶段停止。")
        return {
            "success": bool(count_report.get("success", False)),
            "round_complete": False,
            "phase": "counting",
            "error": count_report.get("error"),
            "resource_limited": bool(count_report.get("resource_limited", False)),
            "resource_detail": count_report.get("resource_detail"),
            "count_stats": count_report.get("count_stats", {}),
            "total_generated": 0,
            "output_files": [],
            "sampling_elapsed_seconds": 0.0,
            "phase_timings": {
                "dag_unfolding_seconds": unfold_elapsed,
                **unfold_detail_timings,
                "combination_counting_seconds": count_elapsed,
                "predicate_sampling_seconds": 0.0,
                **_stage3_sampling_phase_timings({}),
            },
        }

    def sampling_progress(generated_in_round):
        if metrics_recorder is None:
            return
        metrics_recorder.update_counts(
            generated=(
                int(cumulative_generated_base) + int(generated_in_round)
            )
        )
        metrics_recorder.record(
            event="periodic",
            phase="predicate_sampling",
            depth=max_depth,
        )

    if metrics_recorder is not None:
        metrics_recorder.start_phase("predicate_sampling", max_depth)
    sampling_report = sampler(
        deadline=deadline,
        input_files=count_report.get("output_files"),
        max_depth=max_depth,
        downstream_cost_model=downstream_cost_model,
        runtime_signature_feedback=runtime_signature_feedback,
        progress_callback=sampling_progress,
        enumerate_all=enumerate_all,
        localized_random=localized_random,
        output_label=output_label,
    )
    if metrics_recorder is not None:
        metrics_recorder.finish_phase("predicate_sampling")
        metrics_recorder.update_counts(
            generated=(
                int(cumulative_generated_base)
                + int(sampling_report.get("total_generated", 0) or 0)
            )
        )
        metrics_recorder.record(
            event="predicate_sampling_finished",
            force=True,
            phase="predicate_sampling",
            depth=max_depth,
        )
    sampling_phase_timings = _stage3_sampling_phase_timings(sampling_report)
    if metrics_recorder is not None:
        for phase_name, seconds in sampling_phase_timings.items():
            metrics_recorder.add_phase_detail(phase_name, seconds)
    sampling_report["phase_timings"] = {
        "dag_unfolding_seconds": unfold_elapsed,
        **unfold_detail_timings,
        "combination_counting_seconds": count_elapsed,
        "predicate_sampling_seconds": float(
            sampling_report.get("sampling_elapsed_seconds", 0.0) or 0.0
        ),
        **sampling_phase_timings,
    }
    print("谓词知识图处理阶段执行完毕。")
    return sampling_report if isinstance(sampling_report, dict) else {
        "success": False,
        "round_complete": False,
        "total_generated": 0,
        "output_files": [],
    }


def run_database_builder():
    print(">>> 正在启动数据库初始化模块...")
    from BugFinding.db_initializer import DatabaseInitializer
    initializer = DatabaseInitializer()
    initializer.setup_database()
    print(">>> 数据库初始化成功！")


def run_BugFinding(
    deadline=None,
    predicate_files=None,
    signature_spans_by_file=None,
    metrics_recorder=None,
    current_depth=None,
    cumulative_generated=0,
    cumulative_tlp_base=None,
    stratified_layers_by_file=None,
):
    print("开始第三阶段流程...")

    round_summary = {
        "phase_timings": {
            "tlp_generation_seconds": 0.0,
            "database_validation_seconds": 0.0,
        }
    }

    from RL.config import STAGE3_CONFIG

    print(">>> Stage3: TLP-Mode")

    if stratified_layers_by_file:
        from BugFinding.stratified_stream import run_stratified_tlp
        round_summary = run_stratified_tlp(
            predicate_files=predicate_files, layers=stratified_layers_by_file,
            signature_spans_by_file=signature_spans_by_file, deadline=deadline,
            metrics_recorder=metrics_recorder, depth=current_depth,
            cumulative_generated=cumulative_generated,
            cumulative_tlp_base=cumulative_tlp_base,
        )

    if not stratified_layers_by_file:
        if deadline is not None and time.monotonic() >= deadline:
            print(">>> [TLP] 已达到第三阶段截止时间，跳过新的测试用例生成。")
            round_summary["round_complete"] = False
            round_summary["stop_phase"] = "tlp_generation"
            return round_summary

        print(">>> [TLP] 正在启动测试用例生成模块...")
        from BugFinding.sql_generator import TestCaseGenerator
        generator = TestCaseGenerator()
        if metrics_recorder is not None:
            metrics_recorder.start_phase("tlp_generation", current_depth)
        generator_report = generator.process_files(
            input_files=predicate_files,
            deadline=deadline,
        )
        round_summary["phase_timings"]["tlp_generation_seconds"] = float(
            generator_report.get("elapsed_seconds", 0.0) or 0.0
        )
        if metrics_recorder is not None:
            metrics_recorder.finish_phase("tlp_generation")
            metrics_recorder.record(
                event="tlp_generation_finished",
                force=True,
                phase="tlp_generation",
                depth=current_depth,
            )
        print(">>> [TLP] 测试用例生成完成！")

        round_summary["tlp_generation"] = generator_report
        if not generator_report.get("success", False):
            round_summary["round_complete"] = False
            round_summary["stop_phase"] = "tlp_generation"
            print(">>> [TLP] 测试用例生成失败，本轮不扫描旧测试文件。")
            return round_summary
        if not generator_report.get("round_complete", True):
            round_summary["round_complete"] = False
            round_summary["stop_phase"] = "tlp_generation"
            print(">>> [TLP] 截止时间已到，已保留本轮完整生成的部分测试用例。")
            return round_summary

        print(">>> [TLP] 正在启动 Bug 搜寻模块...")
        from BugFinding.bug_finder import TLPBugFinder
        test_case_signature_spans = {}
        source_output_files = generator_report.get("source_output_files", {}) or {}
        for source_name, output_path in source_output_files.items():
            spans = (signature_spans_by_file or {}).get(source_name, [])
            if spans:
                test_case_signature_spans[os.path.basename(output_path)] = spans

        base_tlp = dict(cumulative_tlp_base or {})

        def scan_progress(current):
            if metrics_recorder is None:
                return
            metrics_recorder.update_oracle_exclusions(current, base_tlp)
            metrics_recorder.update_counts(
                generated=int(cumulative_generated),
                tested=int(base_tlp.get("cases_run", 0) or 0)
                + int(current.get("cases_run", 0) or 0),
                validated=int(base_tlp.get("cases_validated", 0) or 0)
                + int(current.get("cases_validated", 0) or 0),
                skipped=int(base_tlp.get("cases_skipped", 0) or 0)
                + int(current.get("cases_skipped", 0) or 0),
                sql_errors=int(base_tlp.get("sql_errors", 0) or 0)
                + int(current.get("sql_errors", 0) or 0),
                query_timeouts=int(base_tlp.get("query_timeouts", 0) or 0)
                + int(current.get("query_timeouts", 0) or 0),
            )
            metrics_recorder.record(
                event="periodic",
                phase="database_validation",
                depth=current_depth,
            )

        if metrics_recorder is not None:
            metrics_recorder.start_phase("database_validation", current_depth)
        finder = TLPBugFinder(
            deadline=deadline,
            input_files=generator_report.get("output_files"),
            predicate_source_files=generator_report.get("source_output_files", {}),
            signature_spans_by_file=test_case_signature_spans,
            progress_callback=scan_progress,
        )
        scan_report = finder.run()
        round_summary["phase_timings"]["database_validation_seconds"] = float(
            scan_report.get("elapsed_seconds", 0.0) or 0.0
        )
        if metrics_recorder is not None:
            metrics_recorder.finish_phase("database_validation")
            scan_progress(scan_report)
            metrics_recorder.record(
                event="database_validation_finished",
                force=True,
                phase="database_validation",
                depth=current_depth,
            )
        print(f">>> [TLP] 本次扫描发现 Bug 数: {finder.total_bugs_found}")
        round_summary["tlp"] = {
            "cases_loaded": scan_report.get("cases_loaded", finder.total_cases_loaded),
            "cases_run": finder.total_cases_run,
            "cases_validated": finder.total_cases_validated,
            "cases_satisfiable": finder.total_cases_satisfiable,
            "cases_skipped": finder.total_cases_skipped,
            "sql_errors": finder.total_sql_errors,
            "query_timeouts": finder.total_query_timeouts,
            "bugs": finder.total_bugs_found,
            "validation_status_path": scan_report.get("validation_status_path"),
            "error_summary_path": scan_report.get("error_summary_path"),
            "mysql_numeric_overflow_retry": dict(
                scan_report.get("mysql_numeric_overflow_retry", {}) or {}
            ),
        }
        if "oracle_excluded_cases" in scan_report:
            round_summary["tlp"].update({key: scan_report[key] for key in (
                "oracle_excluded_cases", "oracle_excluded_wall_seconds", "oracle_fault_path",
                "oracle_worker_restarts", "cases_loaded_raw", "success_rate", "cases_per_second")})
        round_summary["tlp_scan"] = scan_report
        round_summary["tlp_signature_feedback"] = dict(
            scan_report.get("signature_feedback", {}) or {}
        )
        round_summary["tlp_signature_feedback_covered_cases"] = int(
            scan_report.get("signature_feedback_covered_cases", 0) or 0
        )
        if isinstance(scan_report, dict) and not scan_report.get(
            "round_complete", True
        ):
            round_summary["round_complete"] = False
            round_summary["stop_phase"] = "tlp_scan"
        if scan_report.get("worker_failure"):
            round_summary["success"] = False
            round_summary["error"] = scan_report.get("error")
            round_summary["worker_failure"] = scan_report["worker_failure"]
            return round_summary

    round_summary.setdefault("round_complete", True)
    round_summary.setdefault("stop_phase", None)
    print("第三阶段流程执行完毕。")
    return round_summary


def _stage3_signature_feedback_path(project_root=None, *, db_type=None):
    if db_type is None:
        from RL.config import DB_CONFIG
        db_type = DB_CONFIG.db_type
    normalized_db = str(db_type or "").strip().lower()
    feedback_db = {
        "pg": "postgresql",
        "postgres": "postgresql",
        "oracledb": "oracle",
        "oracle_ai26": "oracle",
    }.get(normalized_db, normalized_db)
    root = project_root or os.path.dirname(os.path.abspath(__file__))
    return os.path.join(
        root,
        "pkg",
        "generated_sqls",
        f"stage3_signature_feedback-{feedback_db}.json",
    )


def _load_stage3_signature_feedback(
    db_type, project_root=None, *, allow_postgresql=False
):
    path = _stage3_signature_feedback_path(project_root, db_type=db_type)
    expected_db = str(db_type or "").strip().lower()
    oracle_aliases = {"oracle", "oracledb", "oracle_ai26"}
    persistent_db_types = {"mysql", "sqlserver", *oracle_aliases}
    if allow_postgresql:
        persistent_db_types.update({"postgresql", "postgres", "pg"})
    if expected_db not in persistent_db_types:
        return {}, path
    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, ValueError, TypeError):
        return {}, path
    recorded_db = str(payload.get("db_type", "") or "").strip().lower()
    postgres_aliases = {"postgresql", "postgres", "pg"}
    same_oracle = (
        expected_db in oracle_aliases
        and recorded_db in oracle_aliases
    )
    same_postgresql = (
        allow_postgresql
        and expected_db in postgres_aliases
        and recorded_db in postgres_aliases
    )
    if not recorded_db or (
        recorded_db != expected_db
        and not same_postgresql
        and not same_oracle
    ):
        return {}, path
    counters = {}
    for key, item in (payload.get("signatures", {}) or {}).items():
        if not isinstance(item, dict):
            continue
        counters[str(key)] = {
            name: int(item.get(name, 0) or 0)
            for name in (
                "success_count",
                "failure_count",
                "type_failure_count",
                "value_failure_count",
                "other_failure_count",
                "timeout_count",
            )
        }
        if expected_db == "sqlserver":
            counters[str(key)].update({
                name: int(item.get(name, 0) or 0)
                for name in ("execution_failure_count", "environment_failure_count")
                if name in item
            })
    return counters, path


def _write_stage3_signature_feedback(
    feedback,
    signature_catalog,
    covered_cases,
    db_type,
    path=None,
):
    entries = {}
    for signature_key, counters in sorted((feedback or {}).items()):
        if not isinstance(counters, dict):
            continue
        success_count = int(counters.get("success_count", 0) or 0)
        failure_count = int(counters.get("failure_count", 0) or 0)
        observations = success_count + failure_count
        entries[str(signature_key)] = {
            **dict((signature_catalog or {}).get(signature_key, {}) or {}),
            **dict(counters),
            "observations": observations,
            "success_rate": (
                success_count / observations if observations > 0 else None
            ),
        }
    payload = {
        "version": 1,
        "db_type": str(db_type or ""),
        "scope": "stage3_runtime_signature_feedback",
        "signature_count": len(entries),
        "covered_cases": int(covered_cases or 0),
        "signature_observations": sum(
            int(item.get("observations", 0) or 0)
            for item in entries.values()
        ),
        "signatures": entries,
    }
    output_path = path or _stage3_signature_feedback_path(db_type=db_type)
    from pkg.stage3_runtime import write_status_json
    write_status_json(payload, output_path)
    return output_path, payload


def _print_stage3_validation_rates(generated_count, tlp_report):
    """Report execution coverage and satisfiability among executable predicates."""
    generated = int(generated_count or 0)
    executable = int(tlp_report.get("cases_validated", 0) or 0)
    satisfiable = int(tlp_report.get("cases_satisfiable", 0) or 0)
    executability_rate = f"{100.0 * executable / generated:.2f}%" if generated else "N/A"
    satisfiability_rate = f"{100.0 * satisfiable / executable:.2f}%" if executable else "N/A"
    lines = (
        f"[Stage3] Executability rate: {executability_rate} ({executable}/{generated} generated predicates)",
        f"[Stage3] Satisfiability rate: {satisfiability_rate} ({satisfiable}/{executable} executable predicates)",
    )
    for line in lines:
        print(line, flush=True)
    return lines


def run_stage3():
    from pathlib import Path
    from BugFinding.validation_report import SatisfiablePredicateReport

    report_path = (
        Path(__file__).resolve().parent / "BugFinding" / "generated_test_cases"
        / "satisfiable_predicates.txt"
    )
    with SatisfiablePredicateReport(report_path) as predicate_report:
        return _run_stage3(predicate_report)


def _run_stage3(predicate_report):
    from RL.config import DB_CONFIG, STAGE3_CONFIG
    from pkg.stage3_runtime import (
        Stage3MetricsRecorder,
        format_large_int,
        merge_signature_feedback,
    )

    timed_enumeration_enabled = bool(
        getattr(STAGE3_CONFIG, "stage3_timed_enumeration_enabled", False)
    )
    timed_random_enabled = bool(
        getattr(STAGE3_CONFIG, "stage3_timed_random_enabled", False)
    )
    if timed_enumeration_enabled and timed_random_enabled:
        raise ValueError(
            "Stage-3 timed enumeration and localized-random modes are mutually exclusive."
        )
    timed_experiment_enabled = timed_enumeration_enabled or timed_random_enabled

    if not timed_experiment_enabled:
        project_root = os.path.dirname(os.path.abspath(__file__))
        prior_feedback, feedback_path = _load_stage3_signature_feedback(
            getattr(DB_CONFIG, "db_type", ""),
            project_root,
            allow_postgresql=True,
        )
        sampling_report = run_predicate_knowledge_graph_traverse(
            runtime_signature_feedback=prior_feedback,
        )
        bug_report = {}
        if sampling_report.get("success", False) and sampling_report.get(
            "output_files"
        ):
            bug_report = run_BugFinding(
                predicate_files=sampling_report.get("output_files"),
                signature_spans_by_file=sampling_report.get(
                    "signature_spans_by_file", {}
                ),
            ) or {}
        else:
            print("[Stage3] 谓词生成未成功提交，本轮不扫描旧输出文件。")
        rate_lines = _print_stage3_validation_rates(
            sampling_report.get("total_generated", 0),
            bug_report.get("tlp", {}) or {},
        )
        predicate_report.append_cardinality_results(
            (bug_report.get("tlp_scan", {}) or {}).get("cardinality_results_path")
        )
        predicate_report.finish(rate_lines)
        print(
            "[Stage3] Oracle worker failure; the fixed-depth round is incomplete."
            if isinstance(bug_report, dict) and bug_report.get("worker_failure")
            else "[Stage3] Completed the fixed depth-3 round."
        )
        from pkg.unfolder import MAX_DEPTH
        fixed_depth = int(MAX_DEPTH)
        tlp_scan = bug_report.get("tlp_scan", {}) if isinstance(bug_report, dict) else {}
        round_manifest = {
            "depth": fixed_depth,
            "total_generated": int(sampling_report.get("total_generated", 0) or 0),
            "round_complete": bool(
                sampling_report.get("round_complete", False)
                and tlp_scan.get("round_complete", True)
            ),
            "output_files": [
                os.path.relpath(os.path.abspath(str(path)), project_root).replace("\\", "/")
                for path in sampling_report.get("output_files", []) or []
            ],
            "sampling_timings": dict(
                sampling_report.get("sampling_timings", {}) or {}
            ),
        }
        if tlp_scan.get("validation_status_path"):
            round_manifest["validation_status_path"] = os.path.relpath(
                os.path.abspath(str(tlp_scan["validation_status_path"])), project_root
            ).replace("\\", "/")
        if "oracle_excluded_cases" in tlp_scan:
            round_manifest["oracle_excluded_cases"] = tlp_scan["oracle_excluded_cases"]
            round_manifest["oracle_fault_path"] = tlp_scan.get("oracle_fault_path")
        sampling_phase_timings = _stage3_sampling_phase_timings(sampling_report)
        fixed_report = {
            "mode": "fixed_depth",
            "rounds": 1,
            "cumulative_generated": round_manifest["total_generated"],
            "last_depth": fixed_depth,
            "predicate_rounds": [round_manifest],
            "tlp": dict(bug_report.get("tlp", {}) or {}) if isinstance(bug_report, dict) else {},
            "sampling_timings": dict(
                sampling_report.get("sampling_timings", {}) or {}
            ),
            "phase_timings": {
                "predicate_sampling_seconds": float(sampling_report.get("sampling_elapsed_seconds", 0.0) or 0.0),
                "tlp_generation_seconds": float(
                    (bug_report.get("phase_timings", {}) or {}).get("tlp_generation_seconds", 0.0) or 0.0
                ),
                "database_validation_seconds": float(tlp_scan.get("elapsed_seconds", 0.0) or 0.0),
                **sampling_phase_timings,
            },
        }
        if "oracle_excluded_cases" in tlp_scan:
            fixed_report["cumulative_generated_raw"] = fixed_report["cumulative_generated"]
            fixed_report["cumulative_generated"] = max(
                0, fixed_report["cumulative_generated"] - tlp_scan["oracle_excluded_cases"])
        if tlp_scan.get("worker_failure"):
            fixed_report.update(
                success=False, failed=True, round_complete=False,
                stop_phase="tlp_scan", error=tlp_scan.get("error"),
                worker_failure=tlp_scan["worker_failure"],
            )
        feedback_report = bug_report if isinstance(bug_report, dict) else {}
        feedback_data = feedback_report.get("tlp_signature_feedback", {}) or {}
        combined_feedback = merge_signature_feedback(
            prior_feedback,
            feedback_data,
        )
        feedback_covered = int(
            feedback_report.get("tlp_signature_feedback_covered_cases", 0) or 0
        )
        try:
            written_feedback_path, written_feedback = _write_stage3_signature_feedback(
                combined_feedback,
                sampling_report.get("signature_catalog", {}) or {},
                feedback_covered,
                getattr(DB_CONFIG, "db_type", ""),
                feedback_path,
            )
            fixed_report["signature_feedback"] = {
                "path": os.path.relpath(
                    os.path.abspath(str(written_feedback_path)), project_root
                ).replace("\\", "/"),
                "signature_count": int(written_feedback.get("signature_count", 0) or 0),
                "covered_cases": feedback_covered,
            }
        except Exception as exc:
            print(f"[Stage3] Warning: failed to write signature feedback: {exc}")
        status_path = os.path.join(
            project_root, "pkg", "generated_sqls", "stage3_run_status.json"
        )
        try:
            from pkg.stage3_runtime import write_status_json
            write_status_json(fixed_report, status_path)
            print(f"[Stage3] Compact run status: {status_path}")
        except Exception as exc:
            print(f"[Stage3] Warning: failed to write compact run status: {exc}")
        return fixed_report

    if timed_enumeration_enabled:
        depth_config_name = "stage3_timed_enumeration_depth"
        total_limit_config_name = "stage3_timed_enumeration_total_limit_seconds"
        generation_limit_config_name = "stage3_timed_enumeration_generation_limit_seconds"
        experiment_display_name = "Timed enumeration"
    else:
        depth_config_name = "stage3_timed_random_depth"
        total_limit_config_name = "stage3_timed_random_total_limit_seconds"
        generation_limit_config_name = "stage3_timed_random_generation_limit_seconds"
        experiment_display_name = "Timed localized random"
    configured_start_depth = int(getattr(STAGE3_CONFIG, depth_config_name, 1))
    current_depth = max(1, configured_start_depth)
    time_limit = float(
        getattr(STAGE3_CONFIG, total_limit_config_name, 0.0)
    )
    generation_time_limit = None
    generation_time_limit = float(
        getattr(
            STAGE3_CONFIG,
            generation_limit_config_name,
            3600.0,
        )
    )
    if time_limit <= 0.0 or generation_time_limit <= 0.0:
        raise ValueError(
            f"{experiment_display_name} requires positive generation and total time limits."
        )
    if generation_time_limit >= time_limit:
        raise ValueError(
            f"{experiment_display_name} generation limit must be shorter than its total limit."
        )
    stage3_start = time.monotonic()
    deadline = stage3_start + time_limit if time_limit > 0 else None
    generation_deadline = (
        stage3_start + generation_time_limit
        if generation_time_limit is not None
        else deadline
    )
    project_root = os.path.dirname(os.path.abspath(__file__))
    metrics_output_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "pkg",
        "generated_sqls",
    )
    metrics_recorder = Stage3MetricsRecorder(
        metrics_output_dir,
        start_monotonic=stage3_start,
        interval_seconds=float(
            getattr(STAGE3_CONFIG, "stage3_metrics_interval_seconds", 600.0)
        ),
        enabled=bool(getattr(STAGE3_CONFIG, "stage3_metrics_enabled", True)),
        oracle_fault_exclusion=str(getattr(DB_CONFIG, "db_type", "")).strip().lower()
        in {"oracle", "oracledb", "oracle_ai26"},
    )
    round_count = 0
    cumulative_generated = 0
    cumulative_sampling_seconds = 0.0
    cumulative_tlp = {
        "cases_loaded": 0,
        "cases_run": 0,
        "cases_validated": 0,
        "cases_satisfiable": 0,
        "cases_skipped": 0,
        "sql_errors": 0,
        "query_timeouts": 0,
    }
    if metrics_recorder.oracle_fault_exclusion:
        cumulative_tlp.update(oracle_excluded_cases=0, oracle_excluded_wall_seconds=0.0,
                              oracle_worker_restarts=0, cases_loaded_raw=0)
    stop_phase = None
    failed = False
    oracle_worker_failure = None
    downstream_cost_model = {}
    last_downstream_reserve = {}
    timed_generation_complete = None
    timed_output_label = None
    timed_random_sampling_metadata = {}
    runtime_signature_feedback, prior_feedback_path = _load_stage3_signature_feedback(
        getattr(DB_CONFIG, "db_type", ""),
        project_root,
    )
    runtime_signature_feedback_covered_cases = 0
    signature_catalog = {}
    predicate_rounds = []
    cumulative_phase_timings = {
        "dag_unfolding_seconds": 0.0,
        "type_projection_seconds": 0.0,
        "dag_expansion_seconds": 0.0,
        "dag_output_seconds": 0.0,
        "combination_counting_seconds": 0.0,
        "predicate_sampling_seconds": 0.0,
        **{name: 0.0 for name in _STAGE3_SAMPLING_TIMING_SECONDS},
        "tlp_generation_seconds": 0.0,
        "database_validation_seconds": 0.0,
    }
    cumulative_sampling_timings = {
        **{name: 0.0 for name in _STAGE3_SAMPLING_TIMING_SECONDS},
        **{name: 0 for name in _STAGE3_SAMPLING_TIMING_COUNTERS},
    }

    metrics_recorder.update_counts()
    metrics_recorder.record(
        event="stage3_started",
        force=True,
        phase="orchestration",
        depth=current_depth,
    )

    if timed_enumeration_enabled:
        print(
            "[Stage3] Timed enumeration enabled: "
            f"depth={current_depth}, fixed_step=1, "
            f"generation_time_limit_seconds={generation_time_limit}, "
            f"total_time_limit_seconds={time_limit}"
        )
    elif timed_random_enabled:
        print(
            "[Stage3] Timed localized random enabled: "
            f"depth={current_depth}, "
            f"generation_time_limit_seconds={generation_time_limit}, "
            f"total_time_limit_seconds={time_limit}, "
            f"roots_per_layer={int(getattr(STAGE3_CONFIG, 'stage3_random_root_pool_size', 6))}, "
            f"allow_duplicates=True"
        )
    for _ in range(1):

        round_count += 1
        round_kind = (
            "enumeration"
            if timed_enumeration_enabled
            else "localized-random"
        )
        print(f"\n>>> [Stage3] Start {round_kind} round {current_depth}")
        timed_output_label = (
            f"enumeration-d{current_depth}"
            if timed_enumeration_enabled
            else f"random-d{current_depth}"
        )
        sampling_report = run_predicate_knowledge_graph_traverse(
            max_depth=current_depth,
            deadline=generation_deadline,
            downstream_cost_model=None,
            runtime_signature_feedback=runtime_signature_feedback,
            metrics_recorder=metrics_recorder,
            cumulative_generated_base=cumulative_generated,
            enumerate_all=timed_enumeration_enabled,
            localized_random=timed_random_enabled,
            output_label=timed_output_label,
        )
        sampling_success = bool(sampling_report.get("success", False))
        sampling_complete = bool(sampling_report.get("round_complete", False))
        generated = int(sampling_report.get("total_generated", 0) or 0)
        raw_sampling_timings = dict(
            sampling_report.get("sampling_timings", {}) or {}
        )
        for timing_name in _STAGE3_SAMPLING_TIMING_SECONDS:
            cumulative_sampling_timings[timing_name] += max(
                0.0,
                float(raw_sampling_timings.get(timing_name, 0.0) or 0.0),
            )
        for counter_name in _STAGE3_SAMPLING_TIMING_COUNTERS:
            cumulative_sampling_timings[counter_name] += max(
                0,
                int(raw_sampling_timings.get(counter_name, 0) or 0),
            )
        round_output_files = []
        for output_file in sampling_report.get("output_files", []) or []:
            absolute_output = os.path.abspath(str(output_file))
            relative_output = os.path.relpath(absolute_output, project_root)
            round_output_files.append(relative_output.replace("\\", "/"))
        round_manifest = {
            "depth": current_depth,
            "total_generated": generated,
            "round_complete": sampling_complete,
            "output_files": round_output_files,
            "sampling_timings": raw_sampling_timings,
            "mode": (
                "enumeration"
                if timed_enumeration_enabled
                else "localized_random"
            ),
        }
        round_manifest["stratified_layers_by_file"] = dict(
            sampling_report.get("stratified_layers_by_file", {}) or {})
        if timed_random_enabled:
            timed_random_sampling_metadata = dict(
                sampling_report.get("random_sampling_by_file", {}) or {}
            )
            round_manifest["random_sampling_by_file"] = (
                timed_random_sampling_metadata
            )
        predicate_rounds.append(round_manifest)
        timed_generation_complete = sampling_complete
        last_downstream_reserve = dict(
            sampling_report.get("downstream_reserve", {}) or {}
        )

        bug_report = {}
        if not sampling_success:
            failed = True
            stop_phase = sampling_report.get("phase", "predicate_generation")
            print(
                f"[Stage3] Depth {current_depth} failed in {stop_phase}: "
                f"{sampling_report.get('error')}"
            )
        elif generated > 0 and sampling_report.get("output_files"):
            if deadline is not None and time.monotonic() >= deadline:
                stop_phase = sampling_report.get("phase", "sampling")
                print(
                    "[Stage3] Deadline reached after predicate generation; "
                    "full generated output is retained and no stale SQL is scanned."
                )
            else:
                bug_report = run_BugFinding(
                    deadline=deadline,
                    predicate_files=sampling_report.get("output_files"),
                    signature_spans_by_file=sampling_report.get(
                        "signature_spans_by_file", {}
                    ),
                    metrics_recorder=metrics_recorder,
                    current_depth=current_depth,
                    cumulative_generated=cumulative_generated + generated,
                    cumulative_tlp_base=cumulative_tlp,
                    stratified_layers_by_file=(
                        sampling_report.get("stratified_layers_by_file")
                    ),
                )
        elif sampling_complete:
            print(
                f"[Stage3] Depth {current_depth} generated no predicates; "
                "skipping database scan for this round."
            )

        if not sampling_complete and stop_phase is None:
            stop_phase = (
                "enumeration_generation_time_limit"
                if timed_enumeration_enabled
                else (
                    "random_generation_time_limit"
                    if timed_random_enabled
                    else sampling_report.get("phase", "predicate_generation")
                )
            )
        if sampling_report.get("resource_limited", False):
            print(
                "[Stage3] Exact-count safety budget reached: "
                f"{sampling_report.get('resource_detail')}"
            )

        cumulative_generated += generated
        cumulative_sampling_seconds += float(
            sampling_report.get("sampling_elapsed_seconds", 0.0) or 0.0
        )
        signature_catalog.update(
            sampling_report.get("signature_catalog", {}) or {}
        )
        tlp_report = bug_report.get("tlp", {}) if isinstance(bug_report, dict) else {}
        if isinstance(bug_report, dict) and (bug_report.get("tlp_scan", {}) or {}).get("worker_failure"):
            failed = True
            oracle_worker_failure = bug_report["tlp_scan"]["worker_failure"]
        if bug_report:
            if bug_report.get("success") is False:
                failed = True
            round_manifest["stratified_processing"] = bug_report.get("stratified_layers", {})
            round_manifest["tlp_generated"] = int(
                (bug_report.get("tlp_generation", {}) or {}).get("predicates_read", 0))
        runtime_signature_feedback = merge_signature_feedback(
            runtime_signature_feedback,
            bug_report.get("tlp_signature_feedback", {})
            if isinstance(bug_report, dict)
            else {},
        )
        runtime_signature_feedback_covered_cases += int(
            bug_report.get("tlp_signature_feedback_covered_cases", 0) or 0
        )
        for key in cumulative_tlp:
            converter = float if key == "oracle_excluded_wall_seconds" else int
            cumulative_tlp[key] += converter(tlp_report.get(key, 0) or 0)
        round_phase_timings = dict(
            sampling_report.get("phase_timings", {}) or {}
        )
        round_phase_timings.update(
            bug_report.get("phase_timings", {})
            if isinstance(bug_report, dict)
            else {}
        )
        for phase_name in cumulative_phase_timings:
            cumulative_phase_timings[phase_name] += float(
                round_phase_timings.get(phase_name, 0.0) or 0.0
            )
        metrics_recorder.update_oracle_exclusions(cumulative_tlp)
        metrics_recorder.update_counts(
            generated=cumulative_generated,
            tested=cumulative_tlp["cases_run"],
            validated=cumulative_tlp["cases_validated"],
            skipped=cumulative_tlp["cases_skipped"],
            sql_errors=cumulative_tlp["sql_errors"],
            query_timeouts=cumulative_tlp["query_timeouts"],
        )
        metrics_recorder.record(
            event="depth_round_finished",
            force=True,
            phase="orchestration",
            depth=current_depth,
        )
        validation_complete = bool(
            bug_report.get("round_complete", True)
            if isinstance(bug_report, dict)
            else True
        )
        round_complete = sampling_complete and validation_complete and not failed

        scan_report = (
            bug_report.get("tlp_scan", {})
            if isinstance(bug_report, dict)
            else {}
        )
        predicate_report.append_cardinality_results(scan_report.get("cardinality_results_path"))
        validation_status_path = scan_report.get("validation_status_path")
        if validation_status_path:
            round_manifest["validation_status_path"] = validation_status_path
        if "oracle_excluded_cases" in scan_report:
            round_manifest["oracle_excluded_cases"] = scan_report["oracle_excluded_cases"]
            round_manifest["oracle_fault_path"] = scan_report.get("oracle_fault_path")
        if not validation_complete:
            stop_phase = bug_report.get("stop_phase", "validation")

        stop_index = sampling_report.get("stop_index")
        stop_step = sampling_report.get("stop_step")
        print(
            f"[Stage3] Depth {current_depth} round finished: "
            f"generated={generated}, round_complete={round_complete}, "
            f"stop_root={sampling_report.get('stop_root_id')}, "
            f"stop_index={format_large_int(stop_index) if stop_index is not None else None}, "
            f"stop_step={format_large_int(stop_step) if stop_step is not None else None}, "
            f"stop_phase={stop_phase}"
        )

        if not round_complete:
            break
        if deadline is not None and time.monotonic() >= deadline:
            print("[Stage3] Time limit reached after completing the current round.")
            break
        print(
            f"[Stage3] {experiment_display_name} uses one depth only; "
            "not starting a second depth round."
        )
        break


    elapsed = max(time.monotonic() - stage3_start, 1e-9)
    effective_generated = max(0, cumulative_generated - cumulative_tlp.get("oracle_excluded_cases", 0))
    generation_rate = effective_generated / max(cumulative_sampling_seconds, 1e-9)
    cumulative_tested = int(cumulative_tlp.get("cases_run", 0) or 0)
    cumulative_untested = max(0, effective_generated - cumulative_tested)
    tested_coverage_percentage = (
        100.0 * cumulative_tested / effective_generated
        if effective_generated > 0
        else 100.0
    )
    print(
        f"[Stage3] {experiment_display_name} run finished: "
        f"rounds={round_count}, cumulative_generated={effective_generated}, "
        f"generation_rate={generation_rate:.2f} predicates/s, "
        f"elapsed_seconds={elapsed:.2f}, failed={failed}, stop_phase={stop_phase}"
    )
    print(
        "[Stage3] Cumulative TLP: "
        f"loaded={cumulative_tlp['cases_loaded']}, "
        f"run={cumulative_tlp['cases_run']}, "
        f"validated={cumulative_tlp['cases_validated']}, "
        f"skipped={cumulative_tlp['cases_skipped']}, "
        f"sql_errors={cumulative_tlp['sql_errors']}, "
        f"query_timeouts={cumulative_tlp['query_timeouts']}"
    )
    print(
        "[Stage3] Database-test coverage: "
        f"generated={effective_generated}, tested={cumulative_tested}, "
        f"untested={cumulative_untested}, "
        f"coverage={tested_coverage_percentage:.2f}%"
    )
    rate_lines = _print_stage3_validation_rates(cumulative_generated, cumulative_tlp)
    predicate_report.finish(rate_lines)
    if metrics_recorder.oracle_fault_exclusion:
        print(f"[Stage3] Oracle excluded={cumulative_tlp['oracle_excluded_cases']}, "
              f"generated_raw={cumulative_generated}, elapsed_raw={elapsed:.2f}s, "
              f"excluded_wall={cumulative_tlp['oracle_excluded_wall_seconds']:.2f}s")
    signature_feedback_entries = {}
    for signature_key, counters in sorted(runtime_signature_feedback.items()):
        success_count = int(counters.get("success_count", 0) or 0)
        failure_count = int(counters.get("failure_count", 0) or 0)
        observations = success_count + failure_count
        signature_feedback_entries[signature_key] = {
            **dict(signature_catalog.get(signature_key, {}) or {}),
            **dict(counters),
            "observations": observations,
            "success_rate": (
                success_count / observations if observations > 0 else None
            ),
        }
    signature_feedback_report = {
        "db_type": str(getattr(DB_CONFIG, "db_type", "") or ""),
        "signature_count": len(signature_feedback_entries),
        "covered_cases": runtime_signature_feedback_covered_cases,
        "signature_observations": sum(
            int(item.get("observations", 0) or 0)
            for item in signature_feedback_entries.values()
        ),
        "signatures": signature_feedback_entries,
    }
    signature_feedback_path = prior_feedback_path
    try:
        metrics_recorder.write_measured_status(
            signature_feedback_report,
            signature_feedback_path,
        )
        print(f"[Stage3] Compact signature feedback: {signature_feedback_path}")
    except Exception as exc:
        print(f"[Stage3] Warning: failed to write signature feedback: {exc}")

    timing_summary = metrics_recorder.finish()
    final_report = {
        "mode": (
            "timed_enumeration"
            if timed_enumeration_enabled
            else "timed_localized_random"
        ),
        "rounds": round_count,
        "cumulative_generated": cumulative_generated,
        "elapsed_seconds": elapsed,
        "sampling_elapsed_seconds": cumulative_sampling_seconds,
        "generation_rate_predicates_per_second": generation_rate,
        "time_limit_seconds": time_limit,
        "start_depth": configured_start_depth,
        "last_depth": current_depth,
        "failed": failed,
        "stop_phase": stop_phase,
        "sampling_timings": cumulative_sampling_timings,
        "database_test_coverage": {
            "generated": effective_generated,
            "tested": cumulative_tested,
            "untested": cumulative_untested,
            "percentage": tested_coverage_percentage,
        },
        "downstream_cost_model": downstream_cost_model,
        "last_downstream_reserve": last_downstream_reserve,
        "predicate_rounds": predicate_rounds,
        "phase_timings": cumulative_phase_timings,
        "timing": timing_summary,
        "signature_feedback": {
            "path": signature_feedback_path,
            "signature_count": len(signature_feedback_entries),
            "covered_cases": signature_feedback_report["covered_cases"],
        },
        "tlp": cumulative_tlp,
    }
    if metrics_recorder.oracle_fault_exclusion:
        final_report.update(cumulative_generated_raw=cumulative_generated,
                            cumulative_generated=effective_generated, elapsed_seconds_raw=elapsed,
                            elapsed_seconds=max(0.0, elapsed - cumulative_tlp["oracle_excluded_wall_seconds"]))
    if timed_enumeration_enabled:
        final_report["timed_enumeration"] = {
            "depth": configured_start_depth,
            "fixed_step": 1,
            "generation_time_limit_seconds": generation_time_limit,
            "total_time_limit_seconds": time_limit,
            "generation_complete": timed_generation_complete,
            "output_label": timed_output_label,
        }
    elif timed_random_enabled:
        final_report["timed_localized_random"] = {
            "depth": configured_start_depth,
            "generation_time_limit_seconds": generation_time_limit,
            "total_time_limit_seconds": time_limit,
            "generation_complete": timed_generation_complete,
            "output_label": timed_output_label,
            "allow_duplicates": True,
            "seed": int(getattr(STAGE3_CONFIG, "stage3_random_seed", 42)),
            "root_pool_size": int(
                getattr(STAGE3_CONFIG, "stage3_random_root_pool_size", 16)
            ),
            "root_pool_scope": "per_dag_level",
            "random_sampling_by_file": timed_random_sampling_metadata,
        }
    final_report["tlp_generated"] = sum(
        int(item.get("tlp_generated", 0)) for item in predicate_rounds)
    final_report["stratified_schedule"] = {
        "generation": "equal_time_per_layer" if timed_enumeration_enabled else "equal_count_per_layer",
        "validation": "equal_attempted_count_per_active_layer",
        "exhausted_layer_policy": "redistribute_to_remaining_layers",
    }
    if oracle_worker_failure:
        final_report.update(success=False, round_complete=False,
                            worker_failure=oracle_worker_failure)
    status_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "pkg",
        "generated_sqls",
        "stage3_run_status.json",
    )
    try:
        metrics_recorder.write_measured_status(final_report, status_path)
        print(f"[Stage3] Compact run status: {status_path}")
    except Exception as exc:
        print(f"[Stage3] Warning: failed to write compact run status: {exc}")
    return final_report


def run_stage3_predicate_similarity_post_analysis(stage3_report=None):
    if isinstance(stage3_report, dict) and stage3_report.get("worker_failure"):
        print("[PredicateSimilarity] skipped: Oracle worker failure; Stage 3 incomplete.")
        return None
    from stage3_predicate_similarity_analysis.config import CONFIG

    if not CONFIG.enabled:
        print(
            "[PredicateSimilarity] skipped: "
            "stage3_predicate_similarity_analysis.config.CONFIG.enabled=False"
        )
        return None

    print("\n>>> [Post Analysis] Start Stage-3 predicate similarity analysis...")
    try:
        from stage3_predicate_similarity_analysis.analyze import run_analysis

        result = run_analysis(
            project_root=os.path.dirname(os.path.abspath(__file__)),
            stage3_report=stage3_report,
        )
        return result
    except Exception as exc:
        print(f"[PredicateSimilarity] analysis failed: {exc}")
        if CONFIG.fail_main_on_error:
            raise
        return {"success": False, "error": str(exc)}


def _run_subprocess_step(step_name: str, cmd: list, cwd: str, stop_on_failure: bool = True) -> bool:
    print(f"\n{'=' * 80}")
    print(f"[AutoCycle] START STEP: {step_name}")
    if not cmd or cmd[-1] not in {"pre_rl", "rl_cycle"}:
        print(f"[AutoCycle] CMD: {' '.join(str(x) for x in cmd)}")
    print(f"{'=' * 80}", flush=True)
    sys.stderr.flush()

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    cycle_match = re.search(r"cycle_(\d+)_rl_cycle", str(step_name))
    if cycle_match:
        env["RL_ENUMERATION_CYCLE_INDEX"] = cycle_match.group(1)

    result = subprocess.run(
        cmd,
        cwd=cwd,
        env=env,
    )

    if result.returncode != 0:
        msg = (
            f"[AutoCycle] STEP FAILED: {step_name}, "
            f"returncode={result.returncode}"
        )
        print(msg)

        if stop_on_failure:
            raise RuntimeError(msg)

        return False

    print(f"[AutoCycle] STEP FINISHED: {step_name}", flush=True)
    return True


def _read_last_recycle_apply_count(project_root: str):
    log_path = os.path.join(
        project_root,
        "output",
        "graph_runtime_stats",
        "recycle_apply_log.json",
    )

    if not os.path.exists(log_path):
        return None

    try:
        with open(log_path, "r", encoding="utf-8") as f:
            payload = json.load(f)

        result = payload.get("result", {})
        return int(result.get("applied_count", 0))
    except Exception as e:
        print(f"[AutoCycle Warning] Failed to read apply log: {log_path} | {e}")
        return None


def _verify_operator_similarity_outputs(project_root: str, *, context: str) -> bool:
    required_paths = {
        "operator_similarity": os.path.join(project_root, "output", "operator_similarity.json"),
        "operator_prototypes": os.path.join(project_root, "output", "operator_prototypes.json"),
        "operator_embedding_metadata": os.path.join(project_root, "output", "operator_embedding_metadata.json"),
    }

    missing_or_empty = []
    for name, path in required_paths.items():
        if not os.path.exists(path):
            missing_or_empty.append(f"{name}: missing ({path})")
            continue
        if os.path.getsize(path) <= 0:
            missing_or_empty.append(f"{name}: empty ({path})")

    if missing_or_empty:
        print(f"[AutoCycle Verify] {context}: operator similarity outputs invalid:")
        for item in missing_or_empty:
            print(f"  - {item}")
        return False

    try:
        with open(required_paths["operator_similarity"], "r", encoding="utf-8") as f:
            similarity = json.load(f)
        if not isinstance(similarity, dict) or not similarity:
            print(
                f"[AutoCycle Verify] {context}: "
                "operator_similarity.json is not a non-empty object."
            )
            return False
    except Exception as e:
        print(f"[AutoCycle Verify] {context}: failed to read operator_similarity.json | {e}")
        return False

    print(f"[AutoCycle Verify] {context}: operator similarity outputs OK.")
    return True


def _resolve_auto_cycle_recycle_schedule(cycle_idx: int):
    from RL.config import AUTO_CYCLE_CONFIG

    if not bool(getattr(AUTO_CYCLE_CONFIG, "auto_cycle_recycle_dynamic_schedule_enabled", True)):
        apply_top_k = int(getattr(AUTO_CYCLE_CONFIG, "auto_cycle_recycle_apply_top_k", 500))
        novel_target = 0
        try:
            from RL.config import RECYCLE_CONFIG
            novel_target = int(getattr(RECYCLE_CONFIG, "recycle_novel_structure_target_count", 120))
        except Exception:
            novel_target = 120
        novel_target = max(0, min(novel_target, apply_top_k))
        value_limit = max(0, apply_top_k - novel_target)
        return {
            "value_limit": int(value_limit),
            "novel_target": int(novel_target),
            "apply_top_k": int(apply_top_k),
            "schedule_source": "static_config",
        }

    schedule = getattr(AUTO_CYCLE_CONFIG, "auto_cycle_recycle_value_novel_schedule", ())
    idx = max(1, int(cycle_idx)) - 1

    if idx < len(schedule):
        pair = schedule[idx]
        value_limit = int(pair[0])
        novel_target = int(pair[1])
        schedule_source = f"dynamic_schedule_cycle_{cycle_idx}"
    else:
        value_limit = int(
            getattr(AUTO_CYCLE_CONFIG, "auto_cycle_recycle_default_value_after_schedule", 400)
        )
        novel_target = int(
            getattr(AUTO_CYCLE_CONFIG, "auto_cycle_recycle_default_novel_after_schedule", 120)
        )
        schedule_source = "dynamic_schedule_default_after_schedule"

    value_limit = max(0, value_limit)
    novel_target = max(0, novel_target)
    apply_top_k = value_limit + novel_target

    return {
        "value_limit": int(value_limit),
        "novel_target": int(novel_target),
        "apply_top_k": int(apply_top_k),
        "schedule_source": schedule_source,
    }


@runtime_warning_scope()
def run_auto_cycle(num_cycles_override=None):
    from RL.config import (
        AUTO_CYCLE_CONFIG,
        OUTPUT_DIR,
        ENCODER_STATE_PATH,
        RL_ALGO_CONFIG,
        RECYCLE_CONFIG,
        CANDIDATE_BITMAP_CONFIG,
        RL_ENV_CONFIG,
    )

    project_root = os.path.dirname(os.path.abspath(__file__))
    main_py = os.path.join(project_root, "main.py")

    if num_cycles_override is not None:
        num_cycles = int(num_cycles_override)
        num_cycle_source = "command_line"
    elif AUTO_CYCLE_CONFIG.auto_cycle_auto_compute_num_cycles:
        total_timesteps = int(RL_ALGO_CONFIG.total_timesteps)
        interval_steps = int(RECYCLE_CONFIG.recycle_interval_steps)

        if not RECYCLE_CONFIG.recycle_enable_periodic_trigger:
            num_cycles = 1
            num_cycle_source = "auto_periodic_disabled"
        else:
            if interval_steps <= 0:
                raise ValueError(
                    "[AutoCycle] recycle_interval_steps 必须大于 0，"
                    "否则无法自动计算循环轮数。"
                )

            num_cycles = (total_timesteps + interval_steps - 1) // interval_steps
            num_cycle_source = "auto_from_total_timesteps"

            if total_timesteps % interval_steps != 0:
                print(
                    "[AutoCycle Warning] total_timesteps 不能被 recycle_interval_steps 整除，"
                    f"将向上取整为 {num_cycles} 轮。"
                )
    else:
        num_cycles = int(AUTO_CYCLE_CONFIG.auto_cycle_num_cycles)
        num_cycle_source = "config_auto_cycle_num_cycles"

    stop_on_failure = bool(AUTO_CYCLE_CONFIG.auto_cycle_stop_on_failure)

    if num_cycles <= 0:
        raise ValueError("[AutoCycle] num_cycles 必须大于 0。")

    print(f"\n{'=' * 80}")
    print("[AutoCycle] 自动循环启动")
    print(f"[AutoCycle] project_root = {project_root}")
    print(f"[AutoCycle] num_cycles = {num_cycles}")
    print(f"[AutoCycle] num_cycle_source = {num_cycle_source}")
    print(f"[AutoCycle] total_timesteps = {getattr(RL_ALGO_CONFIG, 'total_timesteps', None)}")
    print(f"[AutoCycle] n_envs = {getattr(RL_ALGO_CONFIG, 'n_envs', None)}")
    print(f"[AutoCycle] recycle_interval_steps = {getattr(RECYCLE_CONFIG, 'recycle_interval_steps', None)}")
    print(f"[AutoCycle] stop_on_failure = {stop_on_failure}")
    print(
        "[AutoCycle] run_stage3_after_cycles = "
        f"{getattr(AUTO_CYCLE_CONFIG, 'auto_cycle_run_stage3_after_cycles', False)}"
    )
    print(f"{'=' * 80}")

    predicate_txt_path = OUTPUT_DIR / "predicate.txt"
    need_initial_pre_rl = (
        not predicate_txt_path.exists()
        or not ENCODER_STATE_PATH.exists()
    )

    if need_initial_pre_rl:
        if AUTO_CYCLE_CONFIG.auto_cycle_run_pre_rl_if_missing:
            print(
                "[AutoCycle] 检测到 output/predicate.txt 或 encoder_state.pt 不存在，"
                "先自动运行一次 pre_rl。"
            )
            ok = _run_subprocess_step(
                step_name="initial_pre_rl",
                cmd=[sys.executable, main_py, "pre_rl"],
                cwd=project_root,
                stop_on_failure=stop_on_failure,
            )
            if not ok:
                print("[AutoCycle] initial_pre_rl 失败，跳过后续自动循环。")
                return
            if bool(getattr(AUTO_CYCLE_CONFIG, "auto_cycle_verify_operator_similarity_after_retrain", True)):
                verified = _verify_operator_similarity_outputs(
                    project_root,
                    context="initial_pre_rl",
                )
                if not verified:
                    msg = "[AutoCycle] initial_pre_rl 后未得到有效 operator_similarity 输出。"
                    if stop_on_failure:
                        raise RuntimeError(msg)
                    print(msg)
                    return
        else:
            raise FileNotFoundError(
                "[AutoCycle] 缺少 output/predicate.txt 或 encoder_state.pt，"
                "且 auto_cycle_run_pre_rl_if_missing=False。请先运行 python main.py pre_rl。"
            )

    if (
        not need_initial_pre_rl
        and bool(getattr(AUTO_CYCLE_CONFIG, "auto_cycle_verify_operator_similarity_after_retrain", True))
    ):
        verified = _verify_operator_similarity_outputs(
            project_root,
            context="before_auto_cycle",
        )
        if not verified:
            msg = (
                "[AutoCycle] 已存在 predicate.txt 和 encoder_state.pt，"
                "但 operator_similarity 输出缺失或无效。请先运行 python main.py pre_rl。"
            )
            if stop_on_failure:
                raise RuntimeError(msg)
            print(msg)
            return

    for cycle_idx in range(1, num_cycles + 1):
        print(f"\n{'#' * 80}")
        print(f"[AutoCycle] Cycle {cycle_idx}/{num_cycles} START")
        print(f"{'#' * 80}")

        ok = _run_subprocess_step(
            step_name=f"cycle_{cycle_idx}_rl_cycle",
            cmd=[sys.executable, main_py, "rl_cycle"],
            cwd=project_root,
            stop_on_failure=stop_on_failure,
        )
        if not ok:
            print(f"[AutoCycle] cycle_{cycle_idx}_rl_cycle 失败，跳过本轮剩余步骤。")
            continue

        recycle_schedule = _resolve_auto_cycle_recycle_schedule(cycle_idx)

        print(
            "[AutoCycle] recycle schedule: "
            f"cycle={cycle_idx}, "
            f"value_limit={recycle_schedule['value_limit']}, "
            f"novel_target={recycle_schedule['novel_target']}, "
            f"apply_top_k={recycle_schedule['apply_top_k']}, "
            f"source={recycle_schedule['schedule_source']}"
        )

        recycle_cmd = [
            sys.executable,
            "-m",
            "RL.predicate_recycle",
            "--top-k",
            str(AUTO_CYCLE_CONFIG.auto_cycle_recycle_top_k),
            "--min-score",
            str(AUTO_CYCLE_CONFIG.auto_cycle_recycle_min_score),
            "--apply",
            "--apply-top-k",
            str(recycle_schedule["apply_top_k"]),
            "--novel-structure-target-count",
            str(recycle_schedule["novel_target"]),
        ]

        if AUTO_CYCLE_CONFIG.auto_cycle_recycle_include_initial:
            recycle_cmd.append("--include-initial")

        if AUTO_CYCLE_CONFIG.auto_cycle_recycle_allow_no_new_contribution:
            recycle_cmd.append("--allow-no-new-contribution")

        if AUTO_CYCLE_CONFIG.auto_cycle_recycle_dry_run:
            recycle_cmd.append("--dry-run")

        if AUTO_CYCLE_CONFIG.auto_cycle_recycle_no_backup:
            recycle_cmd.append("--no-backup")

        ok = _run_subprocess_step(
            step_name=f"cycle_{cycle_idx}_recycle_apply",
            cmd=recycle_cmd,
            cwd=project_root,
            stop_on_failure=stop_on_failure,
        )
        if not ok:
            print(f"[AutoCycle] cycle_{cycle_idx}_recycle_apply 失败，跳过本轮重训。")
            continue

        applied_count = _read_last_recycle_apply_count(project_root)
        print(f"[AutoCycle] cycle_{cycle_idx} applied_count = {applied_count}")

        if applied_count is None:
            msg = (
                f"[AutoCycle] 无法读取 cycle_{cycle_idx} 的 recycle_apply_log.json，"
                "不能确认是否有新谓词写入 predicate.txt。"
            )

            if stop_on_failure:
                raise RuntimeError(msg)

            print(msg)
            print("[AutoCycle] stop_on_failure=False，跳过本轮 recycle_pre_rl。")
            continue

        if AUTO_CYCLE_CONFIG.auto_cycle_recycle_dry_run:
            print("[AutoCycle] 当前 recycle 是 dry-run，跳过 recycle_pre_rl。")
            continue

        if (
            AUTO_CYCLE_CONFIG.auto_cycle_skip_retrain_if_no_apply
            and applied_count == 0
        ):
            print("[AutoCycle] 本轮没有新谓词写入 predicate.txt，跳过 recycle_pre_rl。")
            continue

        if AUTO_CYCLE_CONFIG.auto_cycle_run_recycle_pre_rl:
            ok = _run_subprocess_step(
                step_name=f"cycle_{cycle_idx}_recycle_pre_rl",
                cmd=[sys.executable, main_py, "recycle_pre_rl"],
                cwd=project_root,
                stop_on_failure=stop_on_failure,
            )
            if not ok:
                print(f"[AutoCycle] cycle_{cycle_idx}_recycle_pre_rl 失败。")
                continue

            if bool(getattr(AUTO_CYCLE_CONFIG, "auto_cycle_verify_operator_similarity_after_retrain", True)):
                verified = _verify_operator_similarity_outputs(
                    project_root,
                    context=f"cycle_{cycle_idx}_recycle_pre_rl",
                )
                if not verified:
                    msg = (
                        f"[AutoCycle] cycle_{cycle_idx}_recycle_pre_rl 后 "
                        "operator_similarity 输出缺失或无效。"
                    )
                    if stop_on_failure:
                        raise RuntimeError(msg)
                    print(msg)
                    continue
        else:
            print("[AutoCycle] auto_cycle_run_recycle_pre_rl=False，跳过第一阶段重训。")

        print(f"\n[AutoCycle] Cycle {cycle_idx}/{num_cycles} FINISHED")

    print(f"\n{'=' * 80}")
    print("[AutoCycle] 自动循环全部结束")
    print(f"{'=' * 80}")

    if bool(getattr(CANDIDATE_BITMAP_CONFIG, "enabled", True)):
        try:
            from RL.candidate_bitmap import (
                build_and_save_observed_bitmap,
                check_observed_bitmap_graph_consistency,
            )

            consistency_report = check_observed_bitmap_graph_consistency()
            consistency_message = (
                "[AutoCycle] final candidate bitmap / graph consistency: "
                f"consistent={consistency_report.get('is_consistent')}, "
                f"observed_bitmap={consistency_report.get('observed_bitmap_count')}, "
                f"graph_candidate_edges={consistency_report.get('graph_candidate_edge_count')}, "
                f"bitmap_only={consistency_report.get('bitmap_only_count')}, "
                f"graph_only={consistency_report.get('graph_only_count')}, "
                f"report={consistency_report.get('report_path')}"
            )
            if consistency_report.get("is_consistent"):
                print(consistency_message)
                try:
                    observed_report = build_and_save_observed_bitmap()
                    candidate_info = observed_report.get("candidate", {})
                    graph_info = observed_report.get("graph", {})
                    print(
                        "[AutoCycle] final candidate observed report refreshed: "
                        f"graph_edges={graph_info.get('edge_count')}, "
                        f"observed={candidate_info.get('observed_edge_count')}, "
                        f"missing={candidate_info.get('missing_edge_count')}, "
                        f"coverage={float(candidate_info.get('coverage_rate', 0.0) or 0.0):.6f}, "
                        f"report={observed_report.get('observed_report_path')}"
                    )
                except Exception as observed_report_exc:
                    print(
                        "[AutoCycle Warning] final candidate observed report refresh "
                        f"failed; continuing without changing the formal graph: {observed_report_exc}"
                    )
            else:
                print(f"[AutoCycle Warning] {consistency_message}")
        except Exception as consistency_exc:
            print(
                "[AutoCycle Warning] final candidate bitmap / graph consistency "
                f"check failed; continuing with the current formal graph: {consistency_exc}"
            )

    if bool(getattr(AUTO_CYCLE_CONFIG, "auto_cycle_run_stage3_after_cycles", True)):
        ok = _run_subprocess_step(
            step_name="final_stage3",
            cmd=[sys.executable, main_py, "stage3"],
            cwd=project_root,
            stop_on_failure=stop_on_failure,
        )
        if not ok:
            print("[AutoCycle] final_stage3 失败。")
    else:
        print("[AutoCycle] auto_cycle_run_stage3_after_cycles=False，跳过第三阶段。")



if __name__ == "__main__":
    mode = "auto_cycle"
    if len(sys.argv) > 1:
        mode = sys.argv[1]

    if mode == "auto_cycle":
        num_cycles_override = None
        if len(sys.argv) > 2:
            try:
                num_cycles_override = int(sys.argv[2])
            except ValueError as e:
                raise ValueError(
                    "[AutoCycle] 用法: python main.py auto_cycle [num_cycles]，"
                    "其中 num_cycles 必须是整数。"
                ) from e

        run_auto_cycle(num_cycles_override=num_cycles_override)
        sys.exit(0)

    if mode == "stage3":
        stage3_start = time.time()
        print(f"{'=' * 50}")
        print(f"Stage 3 Only Started. PID: {os.getpid()}.")
        print(f"{'=' * 50}")

        stage3_report = run_stage3()

        stage3_end = time.time()
        print(f"\n{'=' * 50}")
        print(
            f"[Timer] Stage 3 finished. Total Time: "
            f"{datetime.timedelta(seconds=int(stage3_end - stage3_start))}"
        )
        print(f"{'=' * 50}")
        run_stage3_predicate_similarity_post_analysis(stage3_report)
        sys.exit(1 if isinstance(stage3_report, dict) and stage3_report.get("worker_failure") else 0)

    import torch
    import gc
    from pre_load import pre_load
    from Cluster.cluster import clustering
    from Cluster.convergence import is_converged, save_final_result
    from RL.config import RL_ENV_CONFIG

    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    print(f"{'=' * 50}")
    print(f"Main Process Started. PID: {os.getpid()}. Mode: {mode}")
    if torch.cuda.is_available():
        print(f"INFO: GPU Available! Device Count: {torch.cuda.device_count()}")
    else:
        print("WARNING: Running in CPU Mode.")
    print(f"{'=' * 50}")

    global_start_time = time.time()
    stage_times = {
        "Stage 1 (Pre/Code/Cluster/Contrastive)": 0.0,
        "Stage 2 (RL Training)": 0.0,
        "Stage 3 (Graph/Bug Detection)": 0.0
    }

    if mode in ["run_all", "pre_rl", "recycle_pre_rl"]:
        stage1_start = time.time()
        print("\n>>> [Stage 1] Start Pre-processing & Contrastive Learning...")

        if mode == "recycle_pre_rl":
            predicate_txt_path = os.path.join("output", "predicate.txt")

            if not os.path.exists(predicate_txt_path):
                raise FileNotFoundError(
                    "[Stage 6D] recycle_pre_rl 模式需要已有 output/predicate.txt，"
                    "但当前未找到。请先运行 pre_load 或完成 predicate_recycle apply。"
                )

            with open(predicate_txt_path, "r", encoding="utf-8") as f:
                existing_predicates = [line.strip() for line in f if line.strip()]

            if not existing_predicates:
                raise RuntimeError(
                    "[Stage 6D] output/predicate.txt 存在但为空，无法进行回炉重训。"
                )

            print(
                f"[Stage 6D] recycle_pre_rl 模式：跳过 pre_load，"
                f"直接使用已有 {predicate_txt_path}，共 {len(existing_predicates)} 条谓词。"
            )
        else:
            print("开始执行预处理...")
            result = pre_load()
            print(f"处理结束")

        i = 0
        while i < 20:
            print(f"开始第{i + 1}轮迭代")

            print(f">>> 开始编码阶段")
            transformer_encoder()
            print(f">>> 编码阶段结束")

            print(f">>> 开始聚类阶段")
            clustering(iteration=i)

            if is_converged(i):
                save_final_result(i)
                print(f"\n{'=' * 50}")
                print(f"第{i + 1}轮迭代收敛，结果已保存到: data/cluster_final_output")
                print(f"{'=' * 50}")
                break
            print(f">>> 聚类阶段结束")

            print(f">>> 开始对比学习阶段")
            run_contrastive_learning_step(i)

            i += 1

        if i >= 20:
            save_final_result(19)
            print(f"达到迭代上限，输出最后一次迭代结果，结果已保存到: data/cluster_final_output")

        print(">>> 开始生成算子向量与相似算子表")
        run_operator_embedding_extraction()
        print(">>> 算子向量与相似算子表生成结束")

        stage1_end = time.time()
        stage_times["Stage 1 (Pre/Code/Cluster/Contrastive)"] = stage1_end - stage1_start
    if mode in ["pre_rl", "recycle_pre_rl"]:
        global_end_time = time.time()
        print(f"\n{'=' * 50}")
        print(
            f"[Timer] Exiting after Stage 1. Total Time: {datetime.timedelta(seconds=int(global_end_time - global_start_time))}")
        if mode == "recycle_pre_rl":
            print("[Stage 6D] recycle_pre_rl finished. Updated transformer is ready for HNSW rebuild.")
        print(f"{'=' * 50}")
        sys.exit(0)
    if mode in ["run_all", "post_rl", "rl_cycle"]:
        stage2_start = time.time()
        print("\n>>> [Stage 2] Start RL & Downstream Tasks...")
        print(">>> [System] Cleaning up GPU memory before RL training...")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(">>> [System] Memory cleanup done.")

        print(">>> 开始强化学习阶段")

        disable_periodic_trigger_for_this_mode = mode in ["run_all", "post_rl"]

        run_rl_training(
            disable_periodic_trigger=disable_periodic_trigger_for_this_mode
        )

        stage2_end = time.time()
        stage_times["Stage 2 (RL Training)"] = stage2_end - stage2_start

        if mode == "rl_cycle":
            global_end_time = time.time()
            print(f"\n{'=' * 50}")
            print("[Stage 6E] rl_cycle finished.")
            print(
                "[Stage 6E] 本轮流程已完成：重建 HNSW -> 运行 RL cycle -> 执行 recycle 收尾。"
            )
            print(
                "[Stage 6E] 不进入第三阶段。下一步可根据 recycle 结果执行 apply / recycle_pre_rl / 下一轮 rl_cycle。"
            )
            print(
                f"[Timer] Exiting after RL cycle. Total Time: "
                f"{datetime.timedelta(seconds=int(global_end_time - global_start_time))}"
            )
            print(f"{'=' * 50}")
            sys.exit(0)

        stage3_start = time.time()
        print("\n>>> [Stage 3] Start Downstream Tasks (Graph Traverse & Bug Finding)...")
        print(">>> 开始第三阶段固定深度调度")
        stage3_report = run_stage3()

        stage3_end = time.time()
        stage_times["Stage 3 (Graph/Bug Detection)"] = stage3_end - stage3_start

        global_end_time = time.time()
        total_time_seconds = global_end_time - global_start_time

        print(f"\n{'=' * 50}")
        print("====== 系统运行完毕 ======")
        print(f"{'=' * 50}")
        print("用时统计报告:")
        if mode in ["run_all", "pre_rl"]:
            s1_time = datetime.timedelta(seconds=int(stage_times['Stage 1 (Pre/Code/Cluster/Contrastive)']))
            print(f"  Stage 1 (预处理/编码/聚类/对比学习): {s1_time}")

        if mode in ["run_all", "post_rl"]:
            s2_time = datetime.timedelta(seconds=int(stage_times['Stage 2 (RL Training)']))
            print(f"  Stage 2 (强化学习/HNSW构建):         {s2_time}")
            s3_time = datetime.timedelta(seconds=int(stage_times['Stage 3 (Graph/Bug Detection)']))
            print(f"  Stage 3 (图遍历/Bug检测):            {s3_time}")

        print("-" * 50)
        total_time = datetime.timedelta(seconds=int(total_time_seconds))
        print(f"  项目总计用时:                     {total_time}")
        print(f"{'=' * 50}")

        run_stage3_predicate_similarity_post_analysis(stage3_report)

