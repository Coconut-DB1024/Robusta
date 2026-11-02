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
