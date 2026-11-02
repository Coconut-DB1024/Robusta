import sys
import time
from typing import Any, Dict, List, Optional

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import HumanOutputFormat
from RL.config import RL_ALGO_CONFIG, RL_ENV_CONFIG, RECYCLE_CONFIG
from RL import candidate_bitmap as candidate_bitmap_tools
from RL.progress import print_rl_progress


class PercentageOutputFormat(HumanOutputFormat):
    def write(self, key_values, key_excluded, step=0):
        print_rl_progress(step)


class PercentageLogCallback(BaseCallback):
    def _on_training_start(self) -> None:
        self.logger.output_formats = [
            PercentageOutputFormat(output.file)
            if isinstance(output, HumanOutputFormat) and output.file is sys.stdout
            else output
            for output in self.logger.output_formats
        ]

    def _on_step(self) -> bool:
        return True


# ==================== [自定义回调函数类 (从原文件剪切过来)] ====================
class RecycleIntervalStopCallback(BaseCallback):
    """
    阶段 6C 补丁：周期性回炉触发器。

    注意：
        这个 callback 不直接修改 predicate.txt，也不直接重训第一阶段。
        它只负责在达到 recycle_interval_steps 后优雅停止当前 RL cycle。

    为什么不在 callback 内直接运行 predicate_recycle？
        因为 callback 触发时，后台 Master Data / PKG Graph 进程可能还有队列数据没刷完。
        更安全的做法是：
            callback 返回 False 停止 learn()
            -> train_actor_critic.py 的 finally 保存模型
            -> 关闭 env 和后台进程
            -> 后台统计文件完全落盘
            -> 再调用 predicate_recycle
    """

    def __init__(
            self,
            interval_steps: Optional[int] = None,
            stop_training_after_trigger: Optional[bool] = None,
            verbose: int = 0,
            shared_queue: Optional[Any] = None,
    ) -> None:
        super().__init__(verbose=verbose)

        self.interval_steps = int(
            interval_steps
            if interval_steps is not None
            else RECYCLE_CONFIG.recycle_interval_steps
        )

        self.stop_training_after_trigger = bool(
            stop_training_after_trigger
            if stop_training_after_trigger is not None
            else RECYCLE_CONFIG.recycle_stop_training_after_trigger
        )

        self.shared_queue = shared_queue
        self.has_triggered: bool = False
        self.trigger_step: Optional[int] = None

    def _on_step(self) -> bool:
        """
        SB3 每次环境交互后调用。

        返回：
            True  -> 继续训练；
            False -> 优雅停止 model.learn()。
        """
        if not RECYCLE_CONFIG.recycle_enable_periodic_trigger:
            return True

        if self.has_triggered:
            return True

        if self.num_timesteps < self.interval_steps:
            return True

        self.has_triggered = True
        self.trigger_step = int(self.num_timesteps)

        # 通知后台进程尽量做一次同步打印/刷盘。
        # 真正完整刷盘仍然依赖 train_actor_critic.py finally 中发送 DONE 并 join 后台进程。
        if self.shared_queue is not None:
            try:
                self.shared_queue.put(("SYNC_PRINT", self.trigger_step))
            except Exception as e:
                print(f"[Recycle Trigger Warning] Failed to send SYNC_PRINT: {e}")

        if self.stop_training_after_trigger:
            print("触发回流重训。", flush=True)
            return False

        print("已记录回流触发，继续训练。", flush=True)
        return True


class CandidateBitmapUpdateCallback(BaseCallback):
    """
    阶段17-6B：主进程集中回写 candidate bitmap。

    该 callback 从 SB3 locals["infos"] 中收集每个 env step 的 candidate edge 结果，
    然后集中调用 candidate_bitmap.update_candidate_bitmaps_from_results()。

    这样可以避免 32 个 env worker 同时写 candidate_bitmaps.npz。
    """

    def __init__(
        self,
        flush_every_steps: Optional[int] = None,
        min_batch: Optional[int] = None,
        verbose: int = 0,
        shared_reservations: Optional[Any] = None,
        reservation_lock: Optional[Any] = None,
        candidate_shared_state: Optional[Any] = None,
    ) -> None:
        super().__init__(verbose=verbose)

        self.flush_every_steps = int(
            flush_every_steps
            if flush_every_steps is not None
            else getattr(RL_ENV_CONFIG, "candidate_bitmap_callback_flush_steps", 200)
        )
        self.flush_every_steps = max(1, self.flush_every_steps)

        self.min_batch = int(
            min_batch
            if min_batch is not None
            else getattr(RL_ENV_CONFIG, "candidate_bitmap_callback_min_batch", 1)
        )
        self.min_batch = max(1, self.min_batch)

        self.buffer: List[Dict[str, Any]] = []
        self.last_flush_step: int = 0
        self.total_flushed: int = 0
        self.last_report: Dict[str, Any] = {}
        self.candidate_shared_state = candidate_shared_state
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
        self.reservation_release_count = 0
        self.reservation_conflict_count = 0
        self.reservation_attempt_count = 0
        self.reservation_conflict_skip_count = 0
        self.reservation_timeout_by_worker: Dict[int, int] = {}
        self.ppo_shard_sizes: Dict[int, int] = {}

        # 阶段18-4：运行中低频刷新 pattern_transfer_score。
        # RL 开始前已经由 main.run_candidate_bitmap_bootstrap() 强制刷新一次；
        # 这里仅在运行中累计足够多新 observed edge 后低频刷新。
        self.last_pattern_transfer_refresh_step: int = 0
        self.observed_added_since_pattern_refresh: int = 0
        self.last_pattern_transfer_refresh_report: Dict[str, Any] = {}

    def _extract_candidate_bitmap_result(self, info: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if not isinstance(info, dict):
            return None

        if str(info.get("generation_mode", "")).lower() != "candidate_bitmap":
            return None

        edge_id = str(info.get("candidate_bitmap_edge_id", "") or "")
        edge_index = info.get("candidate_bitmap_edge_index", None)

        if not edge_id:
            return None

        try:
            edge_index_int = int(edge_index)
        except Exception:
            edge_index_int = -1

        if edge_index_int < 0:
            return None

        reward_info = info.get("reward_info", {})
        if not isinstance(reward_info, dict):
            reward_info = {}

        # 下游无法沿正式图完成构造时，env 会在进入数据库前标记构造失败。
        # 这些字段仅随现有集中回写结果传递，供日志和后续实验分析使用。
        downstream_construct_failed = bool(
            info.get("candidate_bitmap_construct_failed", False)
        )
        failure_kind = str(info.get("failure_kind", "") or "")
        if not failure_kind and downstream_construct_failed:
            failure_kind = "downstream_construct_failed"
        downstream_failure_reason = str(
            info.get("downstream_failure_reason", "")
            or (info.get("sql_outcome_reason", "") if downstream_construct_failed else "")
            or ""
        )
        global_new_edge = bool(info.get("candidate_bitmap_global_new_edge", False))
        safe_fallback_used = bool(
            info.get("safe_fallback_used", info.get("safe_closure_used", False))
        )
        reservation_conflict_count = int(
            info.get("candidate_bitmap_reservation_conflict_count", 0) or 0
        )
        reservation_attempt_count = int(
            info.get("candidate_bitmap_reservation_attempt_count", 0) or 0
        )
        reservation_conflict_rate = (
            float(reservation_conflict_count) / float(reservation_attempt_count)
            if reservation_attempt_count > 0
            else 0.0
        )
        pool_done_rate = float(
            info.get("candidate_bitmap_pool_done_rate", 0.0) or 0.0
        )
        reservation_action_substitution_count = int(
            bool(info.get("candidate_bitmap_reservation_fallback", False))
        )

        return {
            "edge_id": edge_id,
            "edge_index": edge_index_int,
            "candidate_level": str(info.get("candidate_bitmap_candidate_level", "")),
            "edge_group": str(info.get("candidate_bitmap_edge_group", "")),
            "is_core_to_extension": bool(info.get("candidate_bitmap_core_to_extension", False)),
            "soft_priority_active": bool(info.get("candidate_bitmap_soft_priority_active", False)),
            "soft_priority_bonus": float(info.get("candidate_bitmap_soft_priority_bonus", 0.0) or 0.0),
            "failed_retry": bool(info.get("candidate_bitmap_failed_retry", False)),
            "db_success": bool(info.get("db_success", False)),
            "parse_success": bool(info.get("parse_success", False)),
            "target_edge_hit": bool(info.get("candidate_bitmap_target_edge_hit", False)),
            "predicate": str(info.get("predicate_template", "") or ""),
            "completed_sql": str(info.get("completed_sql", "") or ""),
            "sql_outcome_label": info.get("sql_outcome_label"),
            "sql_outcome_reason": info.get("sql_outcome_reason"),
            "db_error": info.get("db_error"),
            "parse_error": info.get("parse_error"),
            "failure_kind": failure_kind,
            "downstream_construct_failed": downstream_construct_failed,
            "downstream_failure_reason": downstream_failure_reason,
            "inner_ready": bool(info.get("candidate_bitmap_inner_ready", False)),
            "safe_closure_used": bool(info.get("safe_closure_used", False)),
            "safe_fallback_used": safe_fallback_used,
            "global_new_edge": global_new_edge,
            "rewarded_new_edge_count": int(global_new_edge),
            "pool_done_rate": pool_done_rate,
            "reservation_claimed": bool(info.get("candidate_bitmap_reservation_claimed", False)),
            "reservation_conflict": bool(info.get("candidate_bitmap_reservation_conflict", False)),
            "reservation_conflict_count": reservation_conflict_count,
            "reservation_attempt_count": reservation_attempt_count,
            "reservation_conflict_rate": reservation_conflict_rate,
            "reservation_action_substitution_count": reservation_action_substitution_count,
            "reservation_conflict_skip": bool(
                info.get("candidate_bitmap_reservation_conflict_skip", False)
            ),
            "reservation_timeout_count": int(
                info.get("candidate_bitmap_reservation_timeout_count", 0) or 0
            ),
            "reservation_edge_id": str(
                info.get("candidate_bitmap_reservation_edge_id", edge_id) or edge_id
            ),
            "reservation_token": str(
                info.get("candidate_bitmap_reservation_token", "") or ""
            ),
            "ppo_sharding_active": bool(
                info.get("candidate_bitmap_ppo_sharding_active", False)
            ),
            "worker_rank": int(info.get("candidate_bitmap_worker_rank", 0) or 0),
            "worker_count": int(info.get("candidate_bitmap_worker_count", 1) or 1),
            "ppo_shard_size": int(info.get("candidate_bitmap_ppo_shard_size", 0) or 0),
            "reward_total": reward_info.get("reward_total"),
            "local_new_edges": int(info.get("local_new_edges", 0) or 0),
            "local_new_nodes": int(info.get("local_new_nodes", 0) or 0),
            "step": int(self.num_timesteps),
            "source": "candidate_bitmap_callback",
            "status": "success" if (
                bool(info.get("db_success", False))
                and bool(info.get("parse_success", False))
                and bool(info.get("candidate_bitmap_target_edge_hit", False))
            ) else "attempt_failed",
        }

    def _release_result_reservations(self, results: List[Dict[str, Any]]) -> None:
        shared_state = self.candidate_shared_state
        if shared_state is not None:
            claims = {
                str(item.get("reservation_edge_id", item.get("edge_id", "")) or ""):
                str(item.get("reservation_token", "") or "")
                for item in results
                if bool(item.get("reservation_claimed", False))
                and str(item.get("reservation_token", "") or "")
            }
            claims.pop("", None)
            self.reservation_release_count += int(
                shared_state.release_reservations(claims)
            )
            self.reservation_conflict_count += sum(
                int(item.get("reservation_conflict_count", 0) or 0)
                for item in results
            )
            return

        reservations = self.shared_reservations
        lock = self.reservation_lock
        if reservations is None or lock is None:
            return

        edge_ids = {
            str(item.get("reservation_edge_id", item.get("edge_id", "")) or "")
            for item in results
            if bool(item.get("reservation_claimed", False))
        }
        edge_ids.discard("")
        conflicts = sum(int(item.get("reservation_conflict_count", 0) or 0) for item in results)
        lock.acquire()
        try:
            for edge_id in edge_ids:
                if edge_id in reservations:
                    reservations.pop(edge_id, None)
                    self.reservation_release_count += 1
        finally:
            lock.release()
        self.reservation_conflict_count += int(conflicts)

    def _maybe_refresh_pattern_transfer_scores(self, observed_added_count: int) -> bool:
        """
        阶段18-4：运行中低频刷新 pattern_transfer_score / edge_score。

        刷新条件：
            1. 距离上次刷新 >= CANDIDATE_BITMAP_CONFIG.pattern_transfer_refresh_steps；
            2. 期间新增 observed edge >= CANDIDATE_BITMAP_CONFIG.pattern_transfer_refresh_min_observed_added。

        注意：
            - 每轮 RL 开始前的强制刷新由 main.run_candidate_bitmap_bootstrap() 负责；
            - 这里不重建 candidate edge universe；
            - 这里不修改 observed/failed/probing bitmap；
            - observed / confirmed failed / 新增正式图边由共享状态立即同步；
            - 只有表示迁移/HNSW 文件刷新或显式调试开关开启时，worker 才执行全量 reload。
        """
        if not bool(
            getattr(
                candidate_bitmap_tools.CANDIDATE_BITMAP_CONFIG,
                "candidate_representation_guidance_enabled",
                True,
            )
        ):
            return False

        try:
            added = int(observed_added_count or 0)
        except Exception:
            added = 0

        if added > 0:
            self.observed_added_since_pattern_refresh += added

        refresh_steps = int(
            getattr(candidate_bitmap_tools.CANDIDATE_BITMAP_CONFIG, "pattern_transfer_refresh_steps", 20000)
        )
        refresh_min_observed = int(
            getattr(candidate_bitmap_tools.CANDIDATE_BITMAP_CONFIG, "pattern_transfer_refresh_min_observed_added", 128)
        )

        refresh_steps = max(1, refresh_steps)
        refresh_min_observed = max(1, refresh_min_observed)

        steps_since_refresh = int(self.num_timesteps) - int(self.last_pattern_transfer_refresh_step)

        if steps_since_refresh < refresh_steps:
            return False

        if int(self.observed_added_since_pattern_refresh) < refresh_min_observed:
            return False

        try:
            score_report = candidate_bitmap_tools.refresh_candidate_edge_scores_by_pattern_transfer()
            self.last_pattern_transfer_refresh_step = int(self.num_timesteps)
            self.observed_added_since_pattern_refresh = 0
            self.last_pattern_transfer_refresh_report = (
                score_report if isinstance(score_report, dict) else {}
            )

            if self.verbose > 0:
                print_rl_progress(self.num_timesteps)

            return True

        except Exception as refresh_exc:
            if self.verbose > 0:
                print(
                    f"[CandidateBitmap Callback Warning] pattern_transfer_score refresh failed: {refresh_exc}",
                    flush=True,
                )
            return False

    def _flush_candidate_bitmap_results(self, *, force: bool = False) -> None:
        if not self.buffer:
            return

        if (not force) and len(self.buffer) < self.min_batch:
            return

        try:
            pending_results = list(self.buffer)
            report = candidate_bitmap_tools.update_candidate_bitmaps_from_results(
                pending_results
            )
            flushed_count = len(self.buffer)
            self.buffer.clear()

            self.total_flushed += flushed_count
            self.last_flush_step = int(self.num_timesteps)
            self.last_report = report if isinstance(report, dict) else {}

            if self.verbose > 0:
                print_rl_progress(self.num_timesteps)

            refreshed_pattern_scores = self._maybe_refresh_pattern_transfer_scores(
                int(self.last_report.get("observed_added_count", 0) or 0)
            )

            # Shared completion state is immediate. A full JSON/NPZ reload is
            # event-driven only when representation/HNSW artifacts changed, or
            # when the explicit debug switch requests every-flush synchronization.
            should_sync_workers = bool(refreshed_pattern_scores) or bool(
                getattr(RL_ENV_CONFIG, "candidate_bitmap_callback_sync_workers_after_flush", False)
            )
            if should_sync_workers:
                try:
                    self.training_env.env_method("_reload_candidate_bitmap_state", force=True)
                    if self.verbose > 0:
                        print_rl_progress(self.num_timesteps)
                except Exception as sync_exc:
                    if self.verbose > 0:
                        print(f"[CandidateBitmap Callback Warning] worker reload failed: {sync_exc}", flush=True)

        except Exception as exc:
            if self.verbose > 0:
                print(f"[CandidateBitmap Callback Error] bitmap update failed: {exc}", flush=True)

    def _on_step(self) -> bool:
        if not bool(getattr(RL_ENV_CONFIG, "candidate_bitmap_callback_update_enabled", True)):
            return True

        infos = self.locals.get("infos", [])
        completed_results: List[Dict[str, Any]] = []
        for info in infos:
            if str(info.get("generation_mode", "")).lower() == "candidate_bitmap":
                worker_rank = int(info.get("candidate_bitmap_worker_rank", 0) or 0)
                self.reservation_attempt_count += max(
                    0,
                    int(info.get("candidate_bitmap_reservation_attempt_count", 0) or 0),
                )
                self.reservation_conflict_skip_count += int(
                    bool(info.get("candidate_bitmap_reservation_conflict_skip", False))
                )
                if bool(info.get("candidate_bitmap_reservation_conflict_skip", False)):
                    self.reservation_conflict_count += max(
                        0,
                        int(info.get("candidate_bitmap_reservation_conflict_count", 0) or 0),
                    )
                self.reservation_timeout_by_worker[worker_rank] = max(
                    int(self.reservation_timeout_by_worker.get(worker_rank, 0)),
                    int(info.get("candidate_bitmap_reservation_timeout_count", 0) or 0),
                )
                if bool(info.get("candidate_bitmap_ppo_sharding_active", False)):
                    self.ppo_shard_sizes[worker_rank] = max(
                        0,
                        int(info.get("candidate_bitmap_ppo_shard_size", 0) or 0),
                    )

            result = self._extract_candidate_bitmap_result(info)
            if result is not None:
                if self.candidate_shared_state is not None:
                    threshold = int(
                        getattr(
                            candidate_bitmap_tools.CANDIDATE_BITMAP_CONFIG,
                            "target_db_failure_signature_threshold",
                            10,
                        )
                    )
                    result["shared_state_update"] = self.candidate_shared_state.record_result(
                        result,
                        threshold=max(1, threshold),
                    )
                self.buffer.append(result)
                completed_results.append(result)

        # 单边预占只覆盖实际 probe 的执行窗口。内存共享状态更新后立即释放，
        # 正式 bitmap 仍保留原来的集中批量回写频率，不增加磁盘I/O。
        if completed_results:
            try:
                self._release_result_reservations(completed_results)
            except Exception as release_exc:
                # 释放异常不改变边结果；300秒单边超时负责异常兜底。
                if self.verbose > 0:
                    print(
                        f"[CandidateBitmap Callback Warning] immediate reservation release failed: {release_exc}",
                        flush=True,
                    )

        should_flush_by_step = (
            int(self.num_timesteps) - int(self.last_flush_step)
            >= int(self.flush_every_steps)
        )

        if should_flush_by_step:
            self._flush_candidate_bitmap_results(force=False)

        return True

    def _on_training_end(self) -> None:
        self._flush_candidate_bitmap_results(force=True)


class CandidateBitmapProgressCallback(BaseCallback):
    """
    candidate_bitmap 模式的独立消融进度统计 callback。

    该 callback 只读取每步 infos 并发送轻量 SYNC_PRINT 消息：
    - progress_log_enabled=True：按 progress_log_interval_steps 记录进度；
    - progress_log_enabled=False：不发送消息，也不生成进度 CSV；
    - progress_log_interval_steps 是 SB3 汇总所有并行环境后的全局步数间隔。

    它不保存 Va/Vb/Vc，不同步旧 HNSW/UCB，也不参与 PPO 选边、奖励、
    candidate bitmap 更新或正式图写入。
    """

    def __init__(
            self,
            shared_queue: Optional[Any] = None,
            verbose: int = 0,
    ) -> None:
        super().__init__(verbose=verbose)
        self.shared_queue = shared_queue

        # True 启用周期统计，False 完全关闭统计消息和进度 CSV。
        self.progress_log_enabled = bool(
            getattr(RL_ALGO_CONFIG, "progress_log_enabled", True)
        )
        # 该间隔使用 SB3 全局步数，不是单个并行环境的局部步数。
        self.progress_log_interval_steps = max(
            1,
            int(getattr(RL_ALGO_CONFIG, "progress_log_interval_steps", 2048)),
        )
        self.progress_experiment_name = str(
            getattr(RL_ALGO_CONFIG, "experiment_name", "full") or "full"
        ).strip()
        self.progress_seed = int(getattr(RL_ALGO_CONFIG, "seed", 42))
        self.progress_run_id = ""
        self.progress_start_time: Optional[float] = None
        self.progress_construct_failures: int = 0
        self.progress_candidate_selection_count: int = 0
        self.progress_selected_done_count: int = 0
        self.progress_selected_probing_count: int = 0
        self.progress_selected_local_tested_count: int = 0
        self.progress_pool_unfinished_total: int = 0
        self.progress_pool_snapshot_count: int = 0
        self.progress_rewarded_new_edge_count: int = 0
        self.progress_pool_done_rate_total: float = 0.0
        self.progress_safe_fallback_count: int = 0
        self.progress_reservation_attempt_count: int = 0
        self.progress_reservation_conflict_count: int = 0
        self.progress_reservation_action_substitution_count: int = 0
        self.progress_reservation_conflict_skip_count: int = 0
        self.progress_ppo_shard_sizes: Dict[int, int] = {}
        self.last_progress_step: int = -1

    def _on_training_start(self) -> None:
        """初始化本次运行的唯一标识和墙钟计时起点。"""
        self.progress_start_time = time.perf_counter()
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.progress_run_id = (
            f"{self.progress_experiment_name}_seed{self.progress_seed}_{timestamp}"
        )
        if self.progress_log_enabled:
            self._send_progress_sync(0)

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            # 单边预占统计也覆盖“冲突后中性跳过”的无 edge 结果。
            self.progress_reservation_attempt_count += max(
                0,
                int(info.get("candidate_bitmap_reservation_attempt_count", 0) or 0),
            )
            self.progress_reservation_conflict_count += max(
                0,
                int(info.get("candidate_bitmap_reservation_conflict_count", 0) or 0),
            )
            self.progress_reservation_action_substitution_count += int(
                bool(info.get("candidate_bitmap_reservation_fallback", False))
            )
            self.progress_reservation_conflict_skip_count += int(
                bool(info.get("candidate_bitmap_reservation_conflict_skip", False))
            )
            if bool(info.get("candidate_bitmap_ppo_sharding_active", False)):
                worker_rank = int(info.get("candidate_bitmap_worker_rank", 0) or 0)
                self.progress_ppo_shard_sizes[worker_rank] = max(
                    0,
                    int(info.get("candidate_bitmap_ppo_shard_size", 0) or 0),
                )

            # 只统计 env 明确标记的 probe 谓词构造失败，其他失败不计入。
            if bool(info.get("candidate_bitmap_construct_failed", False)):
                self.progress_construct_failures += 1

            # 只统计实际选中了 candidate bitmap edge 的步骤。
            if bool(info.get("candidate_bitmap_edge_id", "")):
                self.progress_candidate_selection_count += 1
                if bool(info.get("candidate_bitmap_exploration_done", False)):
                    self.progress_selected_done_count += 1
                if bool(info.get("candidate_bitmap_probing", False)):
                    self.progress_selected_probing_count += 1
                if bool(info.get("candidate_bitmap_local_tested", False)):
                    self.progress_selected_local_tested_count += 1

                self.progress_pool_unfinished_total += max(
                    0,
                    int(info.get("candidate_bitmap_pool_unfinished_count", 0) or 0),
                )
                self.progress_pool_snapshot_count += 1
                self.progress_rewarded_new_edge_count += int(
                    bool(info.get("candidate_bitmap_global_new_edge", False))
                )
                self.progress_pool_done_rate_total += float(
                    info.get("candidate_bitmap_pool_done_rate", 0.0) or 0.0
                )
                self.progress_safe_fallback_count += int(
                    bool(
                        info.get(
                            "safe_fallback_used",
                            info.get("safe_closure_used", False),
                        )
                    )
                )

        progress_due = (
            self.progress_log_enabled
            and int(self.num_timesteps) - int(self.last_progress_step)
            >= self.progress_log_interval_steps
        )
        if progress_due:
            self._send_progress_sync(self.num_timesteps)

        return True

    def _send_progress_sync(self, step: int) -> None:
        """将统计快照送入现有主进程管道，由图谱进程统一写 CSV。"""
        if not self.progress_log_enabled or self.shared_queue is None:
            return

        step = int(step)
        if step == self.last_progress_step:
            return

        start_time = self.progress_start_time
        elapsed_seconds = (
            max(0.0, time.perf_counter() - start_time)
            if start_time is not None
            else 0.0
        )
        failure_rate = (
            float(self.progress_construct_failures) / float(step)
            if step > 0
            else 0.0
        )
        selection_count = int(self.progress_candidate_selection_count)
        selected_done_rate = (
            float(self.progress_selected_done_count) / float(selection_count)
            if selection_count > 0
            else 0.0
        )
        selected_probing_rate = (
            float(self.progress_selected_probing_count) / float(selection_count)
            if selection_count > 0
            else 0.0
        )
        selected_local_tested_rate = (
            float(self.progress_selected_local_tested_count) / float(selection_count)
            if selection_count > 0
            else 0.0
        )
        avg_pool_unfinished_count = (
            float(self.progress_pool_unfinished_total)
            / float(self.progress_pool_snapshot_count)
            if self.progress_pool_snapshot_count > 0
            else 0.0
        )
        pool_done_rate = (
            float(self.progress_pool_done_rate_total)
            / float(self.progress_pool_snapshot_count)
            if self.progress_pool_snapshot_count > 0
            else 0.0
        )
        reservation_conflict_rate = (
            float(self.progress_reservation_conflict_count)
            / float(self.progress_reservation_attempt_count)
            if self.progress_reservation_attempt_count > 0
            else 0.0
        )

        payload = {
            "run_id": self.progress_run_id,
            "experiment_name": self.progress_experiment_name,
            "seed": self.progress_seed,
            "elapsed_seconds": elapsed_seconds,
            "construct_failures": int(self.progress_construct_failures),
            "construct_failure_rate": failure_rate,
            "selected_done_count": int(self.progress_selected_done_count),
            "selected_done_rate": selected_done_rate,
            "selected_probing_count": int(self.progress_selected_probing_count),
            "selected_probing_rate": selected_probing_rate,
            "selected_local_tested_count": int(
                self.progress_selected_local_tested_count
            ),
            "selected_local_tested_rate": selected_local_tested_rate,
            "avg_pool_unfinished_count": avg_pool_unfinished_count,
            "rewarded_new_edge_count": int(self.progress_rewarded_new_edge_count),
            "pool_done_rate": pool_done_rate,
            "safe_fallback_used_count": int(self.progress_safe_fallback_count),
            "reservation_attempt_count": int(self.progress_reservation_attempt_count),
            "reservation_conflict_count": int(self.progress_reservation_conflict_count),
            "reservation_conflict_rate": reservation_conflict_rate,
            "reservation_action_substitution_count": int(
                self.progress_reservation_action_substitution_count
            ),
            # 单独保留冲突跳过数，便于控制台和后续诊断读取。
            "reservation_conflict_skip_count": int(
                self.progress_reservation_conflict_skip_count
            ),
            "ppo_sharding_active": bool(self.progress_ppo_shard_sizes),
            "ppo_shard_worker_count": int(len(self.progress_ppo_shard_sizes)),
            "ppo_shard_min_size": int(
                min(self.progress_ppo_shard_sizes.values())
                if self.progress_ppo_shard_sizes
                else 0
            ),
            "ppo_shard_max_size": int(
                max(self.progress_ppo_shard_sizes.values())
                if self.progress_ppo_shard_sizes
                else 0
            ),
        }

        try:
            self.shared_queue.put(("SYNC_PRINT", step, payload))
            self.last_progress_step = step
        except Exception as exc:
            # 统计失败不能中断训练；保留旧 step 以便后续步骤重试。
            if self.verbose > 0:
                print(f"[Progress Log Warning] Failed to send progress snapshot: {exc}")

    def _on_training_end(self) -> None:
        """补记非周期边界上的最终训练步，周期边界不重复发送。"""
        final_step = int(self.num_timesteps)
        if self.progress_log_enabled and final_step != int(self.last_progress_step):
            self._send_progress_sync(final_step)


