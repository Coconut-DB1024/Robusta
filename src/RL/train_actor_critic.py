from __future__ import annotations  # 启用未来版本的类型注解特性

import re
import sys  # 导入系统模块
import time  # 导入时间模块
import os
from pathlib import Path  # 导入 pathlib
from typing import Any, Dict, List, Optional  # 导入类型提示

# ==================== [路径与环境设置] ====================
# 获取当前脚本文件的绝对路径
CURRENT_FILE = Path(__file__).resolve()
# 获取项目的根目录
PROJECT_ROOT = CURRENT_FILE.parents[1]
# 检查并添加项目根目录到 sys.path
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

_FORBIDDEN_PLACEHOLDER_RE = re.compile(
    r"(\?\?|\bHOLE(?:_[A-Z0-9_]+)?\b)",
    re.IGNORECASE,
)


def _contains_forbidden_placeholder(text: str) -> bool:
    if text is None:
        return False
    return bool(_FORBIDDEN_PLACEHOLDER_RE.search(str(text)))

# ==================== [项目模块导入] ====================
# 只导入纯配置模块，这些模块不包含 torch 操作，是安全的。
from RL.config import (
    CANDIDATE_BITMAP_CONFIG,
    RL_ALGO_CONFIG,
    RL_ENV_CONFIG,
    RL_DIR,
    RECYCLE_CONFIG,
)
from RL.progress import get_rl_progress_total_steps
from RL.runtime_logging import runtime_warning_scope

# 【重要提醒】
# 此时此刻，绝对不要导入 RL.env_sql_dependency, stable_baselines3, torch 等库。
# 所有的重型导入都必须移入 run_training 函数内部。

# ==================== [Active candidate bitmap master processor] ====================

def master_data_processor(shared_queue, graph_queue, io_batch_size, global_correct_txt_path):
    """将验证成功的候选边事件和进度消息转发给图谱管理器。"""
    print(
        "[Master Worker] 候选边事件转发器已启动。",
        flush=True,
    )
    while True:
        try:
            msg = shared_queue.get()
            if msg == "DONE":
                print("[Master Worker] 收到 DONE，转发给图谱管理器。", flush=True)
                graph_queue.put("DONE")
                break

            if isinstance(msg, tuple) and msg and msg[0] == "SYNC_PRINT":
                # 保持同步打印消息顺序和 flush 语义。
                graph_queue.put(msg)
                continue

            if isinstance(msg, tuple) and msg and msg[0] in {"a", "b"}:
                label, payload = msg
                predicate = (
                    str(payload.get("predicate", "") or "")
                    if isinstance(payload, dict)
                    else str(payload or "")
                )
                if _contains_forbidden_placeholder(predicate):
                    print(f"[Master Worker] 跳过 placeholder 谓词: {predicate}", flush=True)
                    continue
                if label == "a":
                    # target_edge 元数据原样交给 PKG，确保成功边仍进入正式知识图。
                    graph_queue.put(
                        payload if isinstance(payload, dict) else {"predicate": predicate}
                    )
                # 失败事件不进入正式知识图。
                continue

            # 兼容旧版直接发送的 Correct 字符串，保留图谱消息通道。
            if isinstance(msg, str) and msg and not _contains_forbidden_placeholder(msg):
                graph_queue.put(msg)
        except (EOFError, BrokenPipeError):
            print("[Master Worker] 主管道已断开，后台进程安全退出。", flush=True)
            break
        except Exception as exc:
            print(f"[Master Worker Warning] 事件转发失败: {exc}", flush=True)
            time.sleep(1)


# ==================== [训练主流程] ====================

def _finalize_joint_signature_evidence() -> None:
    """Consolidate SQL Server/Oracle deltas after every producer has stopped."""
    from RL.config import DB_CONFIG
    from RL.joint_signature_evidence import SHARDED_EVIDENCE_DB_TYPES

    db_type = str(getattr(DB_CONFIG, "db_type", "") or "").strip().lower()
    if (db_type not in SHARDED_EVIDENCE_DB_TYPES
            or not getattr(RL_ENV_CONFIG, "stage2_joint_signature_evidence_enabled", True)):
        return
    from RL.graph_runtime_stats import get_default_runtime_stats

    store = get_default_runtime_stats().joint_signature_evidence
    try:
        merged = store.merge_pending_shards()
    except Exception as exc:
        # Keep all unfinished deltas/checkpoints for retry; do not let Stage 3
        # silently consume an incomplete snapshot after a persistence failure.
        raise RuntimeError(
            f"Joint signature evidence consolidation failed; pending evidence "
            f"was retained next to {store.path}: {exc}"
        ) from exc
    if merged:
        print(f"[Joint Signature Evidence] Consolidated {merged} delta files into {store.path}")


def _run_predicate_recycle_after_training(trigger_reason: str = "training_end") -> None:
    """
    阶段 6C：在一次 RL 训练结束后，半自动触发谓词回炉流程。

    trigger_reason:
        training_end:
            model.learn() 正常跑到 total_timesteps 或手动/异常结束后触发。

        periodic_stop:
            RecycleIntervalStopCallback 达到 recycle_interval_steps 后，
            返回 False 优雅停止当前 RL cycle，然后在 finally 中触发。

    设计原则：
        1. 只在训练结束、后台进程关闭后运行；
        2. 默认只筛选候选并执行 dry-run，不真实修改 predicate.txt；
        3. 不触发第一阶段重新训练；
        4. 不重建 HNSW；
        5. 任何异常都不影响 PPO 模型保存和训练收尾。
    """
    if trigger_reason == "periodic_stop" and not RECYCLE_CONFIG.recycle_run_after_periodic_stop:
        print("[Recycle] Skipped because recycle_run_after_periodic_stop=False.")
        return

    if trigger_reason != "periodic_stop" and not RECYCLE_CONFIG.enable_after_training:
        print("[Recycle] Disabled by RECYCLE_CONFIG.enable_after_training=False.")
        return

    try:
        from RL.predicate_recycle import PredicateRecycler

        print(f"\n[Recycle] Stage 6C: selecting recycle candidates after RL training... reason={trigger_reason}")

        recycler = PredicateRecycler()

        candidates, output_path = recycler.run(
            top_k=RECYCLE_CONFIG.recycle_top_k,
            min_score=RECYCLE_CONFIG.recycle_min_score,
            include_initial=RECYCLE_CONFIG.recycle_include_initial,
            require_new_contribution=RECYCLE_CONFIG.recycle_require_new_contribution,
        )

        category_counts: Dict[str, int] = {}
        for cand in candidates:
            category_counts[cand.category] = category_counts.get(cand.category, 0) + 1

        print(f"[Recycle] Candidate output: {output_path}")
        print(f"[Recycle] Candidate count: {len(candidates)}")
        print(f"[Recycle] Category counts: {category_counts}")

        # 默认策略：
        #   recycle_apply_after_training=False
        #   recycle_dry_run_after_training=True
        # 这表示训练结束后只生成 dry-run apply 日志，不真实写 predicate.txt。
        should_run_apply_stage = (
            RECYCLE_CONFIG.recycle_apply_after_training
            or RECYCLE_CONFIG.recycle_dry_run_after_training
        )

        if not should_run_apply_stage:
            print("[Recycle] Apply stage skipped. Candidates were generated only.")
            return

        dry_run = (
            RECYCLE_CONFIG.recycle_dry_run_after_training
            or not RECYCLE_CONFIG.recycle_apply_after_training
        )

        apply_result = recycler.apply_candidates_to_predicate_txt(
            candidates,
            apply_top_k=RECYCLE_CONFIG.recycle_apply_top_k,
            dry_run=dry_run,
            create_backup=RECYCLE_CONFIG.recycle_create_backup,
        )

        print("[Recycle] Apply stage finished.")
        print(f"[Recycle] Applied count: {apply_result.applied_count}")
        print(f"[Recycle] Skipped duplicates: {apply_result.skipped_duplicate_count}")
        print(f"[Recycle] Skipped invalid: {apply_result.skipped_invalid_count}")
        print(f"[Recycle] Dry run: {apply_result.dry_run}")
        print(f"[Recycle] Predicate txt: {apply_result.predicate_txt_path}")
        print(f"[Recycle] Backup: {apply_result.backup_path}")
        print(f"[Recycle] Apply log: {apply_result.apply_log_path}")

    except Exception as e:
        print(f"[Recycle Warning] Stage 6C failed, but training shutdown will continue: {e}")
        import traceback
        traceback.print_exc()

@runtime_warning_scope()
def run_training():
    """Run training; abort its child processes if startup or execution fails."""
    import multiprocessing as mp

    existing_children = set(mp.active_children())
    try:
        return _run_training()
    except Exception:
        # Failed workers may leave peers and pipeline consumers waiting on IPC.
        # Only stop children created by this invocation, then propagate the error.
        children = [p for p in mp.active_children() if p not in existing_children]
        for process in children:
            if process.is_alive():
                process.terminate()
        for process in children:
            process.join(timeout=5)
            if process.is_alive():
                process.kill()
                process.join(timeout=5)
        raise


def _run_training():
    """
    PPO 训练主入口函数。
    【设计核心】：所有的重型导入（Torch, SB3, Environment）都延迟到这个函数内部。
    """
    # SB3 PPO 的策略网络设备由 RL_ALGO_CONFIG.sb3_device 控制。
    print(f"[Main Process] SB3 policy device = {RL_ALGO_CONFIG.sb3_device}")
    enumeration_enabled = bool(
        getattr(RL_ENV_CONFIG, "candidate_bitmap_enumeration_enabled", False)
    )
    try:
        enumeration_cycle_index = int(os.environ.get("RL_ENUMERATION_CYCLE_INDEX", "1"))
    except ValueError:
        enumeration_cycle_index = 1
    if enumeration_enabled:
        configured_order_mode = str(
            getattr(
                RL_ENV_CONFIG,
                "candidate_bitmap_enumeration_order_mode",
                "independent_random",
            )
            or "independent_random"
        ).strip().lower()
        order_group_count = min(
            max(1, int(RL_ALGO_CONFIG.n_envs)),
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
        entropy_bias = float(
            getattr(RL_ENV_CONFIG, "candidate_bitmap_enumeration_entropy_bias", 0.0)
        )
        independent_worker_order = bool(
            getattr(
                RL_ENV_CONFIG,
                "candidate_bitmap_enumeration_independent_worker_order_enabled",
                False,
            )
        )
        structural_ratio = float(
            getattr(
                RL_ENV_CONFIG,
                "candidate_bitmap_enumeration_structural_ratio",
                0.70,
            )
        )
        order_mode = configured_order_mode
        if order_mode not in {
            "cycle_stratified_hard_random",
            "top_down_operator_block",
        }:
            order_mode = (
                "independent_worker_random"
                if independent_worker_order
                else "coordinated_entropy_priority"
            )
        order_lifetime = (
            "per-cycle independent worker order"
            if order_mode == "cycle_stratified_hard_random"
            else "fixed full-universe order across cycles"
        )
        print(
            f"[Enumeration] enabled: {order_lifetime}; "
            f"order_mode={order_mode}, "
            f"independent_worker_order={independent_worker_order}, "
            f"order_groups={order_group_count}, "
            f"structural_ratio={structural_ratio:.3f}, "
            f"entropy_bias={entropy_bias:.3f}, cycle={enumeration_cycle_index}, "
            f"workers={RL_ALGO_CONFIG.n_envs}"
        )

    # --- 【启动跨进程通信 Queue 和流水线进程】 ---
    import multiprocessing as mp

    # 使用 Manager().Queue() 保证在 SubprocVecEnv 中可序列化传递。
    # master 转发候选边事件，由图谱管理器统一处理。
    mp_manager = mp.Manager()
    shared_queue = mp_manager.Queue()  # Worker -> 轻量事件转发器
    graph_queue = mp_manager.Queue()  # 事件转发器 -> PKG 图谱管理器
    candidate_shared_state = None
    from RL.candidate_shared_state import create_candidate_shared_state_from_files
    # Load candidate artifacts once in the parent. Workers then receive
    # immediate observed/failed/incremental-edge state through Manager
    # proxies, while callback disk persistence remains batched.
    candidate_shared_state = create_candidate_shared_state_from_files(
        mp_manager,
        Path(getattr(CANDIDATE_BITMAP_CONFIG, "edge_index_path")),
        Path(getattr(CANDIDATE_BITMAP_CONFIG, "bitmap_npz_path")),
    )
    shared_edge_reservations = candidate_shared_state.reservations
    shared_edge_reservation_lock = candidate_shared_state.lock

    # 阶段 6C 补丁：周期性回炉触发 callback 引用。
    # 先定义为 None，避免训练早期异常时 finally 中变量未定义。
    recycle_trigger_callback = None

    from RL.config import MONITOR_CONFIG
    from RL.pkg_manager import global_graph_manager

    # 1. 启动 active 事件转发守护进程
    master_data_p = mp.Process(
        target=master_data_processor,
        args=(shared_queue, graph_queue, MONITOR_CONFIG.io_batch_size, MONITOR_CONFIG.global_correct_txt_path)
    )
    master_data_p.start()
    print("[Main Process] Master Data Processor (lightweight event forwarder) started successfully.")

    # 2. 启动原本的 PKG 图谱守护进程
    master_graph_p = mp.Process(
        target=global_graph_manager,
        args=(graph_queue, MONITOR_CONFIG.io_batch_size, MONITOR_CONFIG.global_correct_txt_path),
        kwargs={"progress_total_steps": get_rl_progress_total_steps()},
    )
    master_graph_p.start()
    print("[Main Process] Global PKG Manager (Pipeline Stage 2) started successfully.")
    # ----------------------------------------------------

    # ==================== [延迟导入区域] ====================
    # 现在可以安全地导入 heavy libraries 了
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
        from stable_baselines3.common.env_checker import check_env
        from stable_baselines3.common.utils import set_random_seed
        from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
        from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, BaseCallback
        from RL.candidate_policy import CandidateScoringPolicy
        # 从新文件导入我们的 Callback
        from RL.callbacks import (
            PercentageLogCallback,
            RecycleIntervalStopCallback,
            CandidateBitmapUpdateCallback,
            CandidateBitmapProgressCallback,
        )
    except ImportError as e:
        raise RuntimeError("stable-baselines3 import failed inside run_training.") from e
    # ==========================================================

    # 1. 准备目录路径
    log_dir = RL_DIR / "rl_logs"
    model_save_path = RL_DIR / "ppo_sql_model"
    ckpt_dir = RL_DIR / "checkpoints"

    # 2. 创建目录（如果不存在）
    log_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # 3. 设置全局随机种子
    seed = RL_ALGO_CONFIG.seed
    set_random_seed(seed)

    # 4. 打印训练信息
    print("=== RL Training Setup ===")
    print(f"Algorithm: {RL_ALGO_CONFIG.algo_name}")
    print(f"Device: {RL_ALGO_CONFIG.sb3_device}")
    print(f"Total Timesteps: {RL_ALGO_CONFIG.total_timesteps}")

    # 5. 环境兼容性检查 (在主进程中进行)
    if bool(getattr(RL_ENV_CONFIG, "enable_env_compliance_check", False)):
        print("\n[Env Check] Initializing environment for compliance check...")
        try:
            # 【局部导入】仅在此处导入环境类
            from RL.env_sql_dependency import SqlDependencyEnv

            test_env = SqlDependencyEnv()
            check_env(test_env, warn=True)
            test_env.close()

            # 【关键修正】强制清理显存，为后面的 PPO Learner 腾出空间
            del test_env  # 删除对象引用
            import gc
            gc.collect()  # 强制 Python 垃圾回收
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()  # 清空 PyTorch 缓存

            print("[Env Check] Passed and GPU memory cleaned.")

        except Exception as e:
            print(f"\n[Env Check] FAILED: {e}")
            print("请检查 gymnasium 版本或 env_sql_dependency.py 中的 reset/step 返回值格式。")
            raise
    else:
        print("[Env Check] Skipped by RL_ENV_CONFIG.enable_env_compliance_check=False.")

    # ==================== [多进程环境配置] ====================
    target_n_envs = RL_ALGO_CONFIG.n_envs
    print(f"\n[Parallel] Launching {target_n_envs} environments with SubprocVecEnv (spawn)...")

    # 从 env_runner 导入工厂函数
    # 这里的 env_runner 是我们之前创建的那个纯净文件
    from RL.env_runner import make_env_isolated_with_reservation
    from RL.progress import EnvironmentProgress

    environment_progress = EnvironmentProgress(
        target_n_envs, state=mp_manager.dict(), lock=mp_manager.Lock()
    )

    # 使用 SubprocVecEnv 启动多进程环境
    vec_env = SubprocVecEnv(
        [
            make_env_isolated_with_reservation(
                i,
                seed,
                target_n_envs,
                shared_queue,
                shared_edge_reservations,
                shared_edge_reservation_lock,
                candidate_shared_state,
                enumeration_cycle_index,
                environment_progress=environment_progress,
            )
            for i in range(target_n_envs)
        ],
        start_method='spawn'
    )

    # 添加 Monitor Wrapper，用于记录 TensorBoard 所需的 Reward 曲线
    vec_env = VecMonitor(vec_env, filename=str(log_dir))
    # ====================================================================

    # 6. 修正 PPO 参数
    # 确保 n_steps 足够大，避免频繁更新导致效率低下
    req_n_steps = RL_ALGO_CONFIG.n_steps
    req_batch_size = RL_ALGO_CONFIG.batch_size

    if req_n_steps < req_batch_size:
        print(f"\n[Config Warning] n_steps ({req_n_steps}) < batch_size ({req_batch_size}).")
        req_n_steps = max(256, req_batch_size)
        print(f"[Config Fix] Overriding n_steps to {req_n_steps}")

    # 7. 初始化 PPO 模型
    final_model_path = model_save_path.with_suffix(".zip")
    model = None
    expected_policy_class = CandidateScoringPolicy

    if final_model_path.exists():
        print(f"\n[PPO] Loading existing model: {final_model_path}")
        try:
            model = PPO.load(
                final_model_path,
                env=vec_env,
                device=RL_ALGO_CONFIG.sb3_device,
                learning_rate=RL_ALGO_CONFIG.learning_rate,
            )
            if expected_policy_class is not None and not isinstance(
                model.policy, expected_policy_class
            ):
                print(
                    "[PPO Warning] Existing model uses the old flat candidate policy; "
                    "creating a new shared edge-scoring policy."
                )
                model = None
        except Exception as e:
            print(
                f"\n[PPO Warning] Existing model is incompatible with current env/action space: {e}"
            )
            print("[PPO Warning] A new PPO model will be created. Old zip is kept untouched.")

    if model is None:
        print(f"\n[PPO] Creating NEW model...")
        import torch.nn as nn
        policy = CandidateScoringPolicy
        policy_kwargs = dict(
            activation_fn=nn.Tanh,
            candidate_pool_size=int(RL_ENV_CONFIG.candidate_bitmap_pool_size),
            candidate_feature_dim=int(RL_ENV_CONFIG.candidate_bitmap_feature_dim),
        )

        model = PPO(
            policy,
            vec_env,
            policy_kwargs=policy_kwargs,
            learning_rate=RL_ALGO_CONFIG.learning_rate,
            n_steps=req_n_steps,
            batch_size=req_batch_size,
            n_epochs=RL_ALGO_CONFIG.n_epochs,
            gamma=RL_ALGO_CONFIG.gamma,
            clip_range=RL_ALGO_CONFIG.clip_range,
            target_kl=0.05,
            verbose=1,
            tensorboard_log=str(log_dir / "tensorboard"),
            device=RL_ALGO_CONFIG.sb3_device,
            seed=seed
    )

    # 8. 设置回调列表
    checkpoint_save_freq = 200
    checkpoint_save_freq = max(
        200,
        int(getattr(RL_ENV_CONFIG, "candidate_bitmap_checkpoint_save_freq", 10000)),
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=checkpoint_save_freq,
        save_path=str(ckpt_dir),
        name_prefix="ppo_sql"
    )


    callback_items = [PercentageLogCallback(), checkpoint_callback]

    if bool(getattr(RL_ENV_CONFIG, "candidate_bitmap_callback_update_enabled", True)):
        candidate_bitmap_update_callback = CandidateBitmapUpdateCallback(
            verbose=1,
            shared_reservations=shared_edge_reservations,
            reservation_lock=shared_edge_reservation_lock,
            candidate_shared_state=candidate_shared_state,
        )
        callback_items.append(candidate_bitmap_update_callback)

    # progress_log_enabled=True 时，candidate_bitmap 独立记录消融进度；
    # False 时不创建统计 callback，也不发送周期 SYNC_PRINT。该开关不依赖 bitmap 更新开关。
    if bool(getattr(RL_ALGO_CONFIG, "progress_log_enabled", True)):
        candidate_bitmap_progress_callback = CandidateBitmapProgressCallback(
            shared_queue=shared_queue,
            verbose=1,
        )
        callback_items.append(candidate_bitmap_progress_callback)

    if RECYCLE_CONFIG.recycle_enable_periodic_trigger:
        recycle_trigger_callback = RecycleIntervalStopCallback(
            interval_steps=RECYCLE_CONFIG.recycle_interval_steps,
            stop_training_after_trigger=RECYCLE_CONFIG.recycle_stop_training_after_trigger,
            verbose=1,
            shared_queue=shared_queue,
        )
        callback_items.append(recycle_trigger_callback)

    callbacks = CallbackList(callback_items)

    # 9. 开始训练
    print("\n[PPO] Start Learning...")
    training_failed = False
    try:
        model.learn(
            total_timesteps=RL_ALGO_CONFIG.total_timesteps,
            callback=callbacks,
            progress_bar=False
        )
        print("\n[PPO] Training Completed.")
    except KeyboardInterrupt:
        print("\n[PPO] Training interrupted manually.")
    except Exception as e:
        training_failed = True
        print(f"\n[PPO] Training Error: {e}")
        import traceback
        traceback.print_exc()
        raise
    finally:
        # Fatal failures are cleaned up by run_training; do not run recycling.
        if not training_failed:
            # 10. 保存模型并关闭环境
            model.save(final_model_path)

            vec_env.close()

            # --- 【通知并关闭流水线进程】 ---
            print("[Main Process] Sending DONE signal to Pipeline...")
            shared_queue.put("DONE")

            master_data_p.join()
            if master_data_p.is_alive():
                print("[Main Process] Warning: Master Data Process 卡死，强制终止...")
                master_data_p.terminate()

            master_graph_p.join()
            if master_graph_p.is_alive():
                print("[Main Process] Warning: Master Graph Process 卡死，强制终止...")
                master_graph_p.terminate()

            print("[Main Process] All background processes closed.")
            try:
                mp_manager.shutdown()
            except Exception as manager_exc:
                print(f"[Main Process] Shared reservation manager shutdown warning: {manager_exc}")
            _finalize_joint_signature_evidence()
            # ----------------------------------------

            # ================== 阶段 6C：训练结束后半自动回炉 ==================
            # 注意：
            #   1. 此处只调用 predicate_recycle.py；
            #   2. 默认只筛选候选并执行 dry-run；
            #   3. 不自动触发第一阶段重新训练；
            #   4. 不重建 HNSW。
            recycle_triggered_by_callback = (
                recycle_trigger_callback is not None
                and getattr(recycle_trigger_callback, "has_triggered", False)
            )

            trigger_reason = "periodic_stop" if recycle_triggered_by_callback else "training_end"

            _run_predicate_recycle_after_training(trigger_reason=trigger_reason)
            # ==================================================================

# 程序入口
if __name__ == "__main__":
    run_training()
