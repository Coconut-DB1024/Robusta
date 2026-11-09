from __future__ import annotations  

import re
import sys  
import time  
import os
from pathlib import Path  
from typing import Any, Dict, List, Optional  


CURRENT_FILE = Path(__file__).resolve()

PROJECT_ROOT = CURRENT_FILE.parents[1]

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


from RL.config import (
    CANDIDATE_BITMAP_CONFIG,
    RL_ALGO_CONFIG,
    RL_ENV_CONFIG,
    RL_DIR,
    RECYCLE_CONFIG,
)
from RL.progress import get_rl_progress_total_steps
from RL.runtime_logging import runtime_warning_scope, suppress_sb3_logging_destination


def master_data_processor(shared_queue, graph_queue, io_batch_size, global_correct_txt_path):
    print(
        '[Master Worker] Candidate edge event forwarder started.',
        flush=True,
    )
    while True:
        try:
            msg = shared_queue.get()
            if msg == "DONE":
                print('[Master Worker] Received DONE; forwarding it to the graph manager.', flush=True)
                graph_queue.put("DONE")
                break

            if isinstance(msg, tuple) and msg and msg[0] == "SYNC_PRINT":
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
                    print(f'[Master Worker] Skipping placeholder predicate: {predicate}', flush=True)
                    continue
                if label == "a":
                    graph_queue.put(
                        payload if isinstance(payload, dict) else {"predicate": predicate}
                    )

                continue

            if isinstance(msg, str) and msg and not _contains_forbidden_placeholder(msg):
                graph_queue.put(msg)
        except (EOFError, BrokenPipeError):
            print('[Master Worker] Main pipe disconnected; background process exiting safely.', flush=True)
            break
        except Exception as exc:
            print(f'[Master Worker Warning] Event forwarding failed: {exc}', flush=True)
            time.sleep(1)


def _finalize_joint_signature_evidence() -> None:
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
        raise RuntimeError(
            f"Joint signature evidence consolidation failed; pending evidence "
            f"was retained next to {store.path}: {exc}"
        ) from exc


def _run_predicate_recycle_after_training(trigger_reason: str = "training_end") -> None:
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

    except Exception as e:
        print(f"[Recycle Warning] Stage 6C failed, but training shutdown will continue: {e}")
        import traceback
        traceback.print_exc()

@runtime_warning_scope()
def run_training():
    import multiprocessing as mp

    existing_children = set(mp.active_children())
    try:
        return _run_training()
    except Exception:
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

    import multiprocessing as mp

    mp_manager = mp.Manager()
    shared_queue = mp_manager.Queue()  
    graph_queue = mp_manager.Queue()  
    candidate_shared_state = None
    from RL.candidate_shared_state import create_candidate_shared_state_from_files

    candidate_shared_state = create_candidate_shared_state_from_files(
        mp_manager,
        Path(getattr(CANDIDATE_BITMAP_CONFIG, "edge_index_path")),
        Path(getattr(CANDIDATE_BITMAP_CONFIG, "bitmap_npz_path")),
    )
    shared_edge_reservations = candidate_shared_state.reservations
    shared_edge_reservation_lock = candidate_shared_state.lock

    recycle_trigger_callback = None

    from RL.config import MONITOR_CONFIG
    from RL.pkg_manager import global_graph_manager

    master_data_p = mp.Process(
        target=master_data_processor,
        args=(shared_queue, graph_queue, MONITOR_CONFIG.io_batch_size, MONITOR_CONFIG.global_correct_txt_path)
    )
    master_data_p.start()
    print("[Main Process] Master Data Processor (lightweight event forwarder) started successfully.")

    master_graph_p = mp.Process(
        target=global_graph_manager,
        args=(graph_queue, MONITOR_CONFIG.io_batch_size, MONITOR_CONFIG.global_correct_txt_path),
        kwargs={"progress_total_steps": get_rl_progress_total_steps()},
    )
    master_graph_p.start()
    print("[Main Process] Global PKG Manager (Pipeline Stage 2) started successfully.")

    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
        from stable_baselines3.common.env_checker import check_env
        from stable_baselines3.common.utils import set_random_seed
        from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
        from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, BaseCallback
        from RL.candidate_policy import CandidateScoringPolicy

        from RL.callbacks import (
            PercentageLogCallback,
            RecycleIntervalStopCallback,
            CandidateBitmapUpdateCallback,
            CandidateBitmapProgressCallback,
        )
    except ImportError as e:
        raise RuntimeError("stable-baselines3 import failed inside run_training.") from e

    log_dir = RL_DIR / "rl_logs"
    model_save_path = RL_DIR / "ppo_sql_model"
    ckpt_dir = RL_DIR / "checkpoints"

    log_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    seed = RL_ALGO_CONFIG.seed
    set_random_seed(seed)

    print("=== RL Training Setup ===")
    print(f"Algorithm: {RL_ALGO_CONFIG.algo_name}")
    if RL_ALGO_CONFIG.experiment_name.endswith(("_WCL", "_Pruning")):
        print(f"Experiment: {RL_ALGO_CONFIG.experiment_name}")
    print(f"Device: {RL_ALGO_CONFIG.sb3_device}")
    print(f"Total Timesteps: {RL_ALGO_CONFIG.total_timesteps}")

    if bool(getattr(RL_ENV_CONFIG, "enable_env_compliance_check", False)):
        print("\n[Env Check] Initializing environment for compliance check...")
        try:
            from RL.env_sql_dependency import SqlDependencyEnv

            test_env = SqlDependencyEnv()
            check_env(test_env, warn=True)
            test_env.close()

            del test_env  
            import gc
            gc.collect()  
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()  

            print("[Env Check] Passed and GPU memory cleaned.")

        except Exception as e:
            print(f"\n[Env Check] FAILED: {e}")
            print('Check the gymnasium version and the reset/step return-value formats in env_sql_dependency.py.')
            raise

    target_n_envs = RL_ALGO_CONFIG.n_envs

    from RL.env_runner import make_env_isolated_with_reservation
    from RL.progress import EnvironmentProgress

    environment_progress = EnvironmentProgress(
        target_n_envs, state=mp_manager.dict(), lock=mp_manager.Lock()
    )

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

    vec_env = VecMonitor(vec_env, filename=str(log_dir))

    req_n_steps = RL_ALGO_CONFIG.n_steps
    req_batch_size = RL_ALGO_CONFIG.batch_size

    if req_n_steps < req_batch_size:
        print(f"\n[Config Warning] n_steps ({req_n_steps}) < batch_size ({req_batch_size}).")
        req_n_steps = max(256, req_batch_size)
        print(f"[Config Fix] Overriding n_steps to {req_n_steps}")

    final_model_path = model_save_path.with_suffix(".zip")
    model = None
    expected_policy_class = CandidateScoringPolicy

    if final_model_path.exists():
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

    print("\n[PPO] Start Learning...")
    training_failed = False
    try:
        with suppress_sb3_logging_destination():
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
        if not training_failed:
            model.save(final_model_path)

            vec_env.close()

            print("[Main Process] Sending DONE signal to Pipeline...")
            shared_queue.put("DONE")

            master_data_p.join()
            if master_data_p.is_alive():
                print('[Main Process] Warning: Master Data Process is unresponsive; terminating...')
                master_data_p.terminate()

            master_graph_p.join()
            if master_graph_p.is_alive():
                print('[Main Process] Warning: Master Graph Process is unresponsive; terminating...')
                master_graph_p.terminate()

            try:
                mp_manager.shutdown()
            except Exception as manager_exc:
                print(f"[Main Process] Shared reservation manager shutdown warning: {manager_exc}")
            _finalize_joint_signature_evidence()

            recycle_triggered_by_callback = (
                recycle_trigger_callback is not None
                and getattr(recycle_trigger_callback, "has_triggered", False)
            )

            trigger_reason = "periodic_stop" if recycle_triggered_by_callback else "training_end"

            _run_predicate_recycle_after_training(trigger_reason=trigger_reason)


if __name__ == "__main__":
    run_training()
