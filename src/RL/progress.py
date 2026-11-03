from threading import Lock


class EnvironmentReloadError(RuntimeError):
    """The environment cannot continue with an incomplete bitmap reload."""


class EnvironmentProgress:
    """Report completion milestones without synchronizing worker execution."""

    _labels = {
        "initialized": "[Parallel] 初始化进度",
        "bitmap_reloaded": "[CandidateBitmapEnv] reloaded candidate bitmap",
    }

    def __init__(self, worker_count=1, state=None, lock=None):
        self.worker_count = max(1, int(worker_count))
        self.state = {} if state is None else state
        self.lock = Lock() if lock is None else lock

    def completed(self, stage, worker_rank):
        label = self._labels[stage]
        rank = int(worker_rank)
        if not 0 <= rank < self.worker_count:
            raise ValueError(f"Invalid progress worker rank: {rank}")
        with self.lock:
            counts, finished_rounds, half_reported = self.state.get(
                stage, ([0] * self.worker_count, 0, False)
            )
            counts = list(counts)
            if stage == "initialized" and counts[rank]:
                return
            counts[rank] += 1
            # Each round counts one completion per worker. A fast worker cannot
            # count as several workers; later reloads never wait for slower ones.
            while True:
                done = sum(count > finished_rounds for count in counts)
                if (
                    self.worker_count > 1
                    and done >= (self.worker_count + 1) // 2
                    and not half_reported
                ):
                    print(f"{label}: 50% ({done}/{self.worker_count})", flush=True)
                    half_reported = True
                if done < self.worker_count:
                    break
                print(f"{label}: 100% ({done}/{self.worker_count})", flush=True)
                finished_rounds += 1
                half_reported = False
            self.state[stage] = (counts, finished_rounds, half_reported)


def get_rl_progress_total_steps():
    from RL.config import RECYCLE_CONFIG, RL_ALGO_CONFIG

    total_steps = max(1, int(RL_ALGO_CONFIG.total_timesteps))
    if (
        RECYCLE_CONFIG.recycle_enable_periodic_trigger
        and RECYCLE_CONFIG.recycle_stop_training_after_trigger
    ):
        interval_steps = int(RECYCLE_CONFIG.recycle_interval_steps)
        if interval_steps > 0:
            total_steps = min(total_steps, interval_steps)
    return total_steps


def print_rl_progress(step, total_steps=None):
    if total_steps is None:
        total_steps = get_rl_progress_total_steps()
    total_steps = max(1, int(total_steps))
    completed_steps = min(max(0, int(step)), total_steps)
    fraction = completed_steps / total_steps
    filled = int(fraction * 40)
    bar = "#" * filled + "-" * (40 - filled)
    print(f"[{bar}] {fraction * 100:.2f}%", flush=True)
