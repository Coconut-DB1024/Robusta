import os
import time

# 注意：这里千万不要直接 import torch 或 RL.env_sql_dependency
# 我们要把导入推迟到函数内部，确保环境变量设置生效后再导入

def make_env_isolated(rank: int, seed: int = 0, shared_queue=None):# 【新增 shared_queue 参数】
    """
    这是一个纯净的环境工厂函数。
    它会在子进程中被执行。
    """

    # 当前 candidate_bitmap 主线不再在 env worker 内加载本地 LLM，
    # 也不再为每个 worker 设置 CUDA_VISIBLE_DEVICES / WORKER_GPU_ID。
    # stable-baselines3 在主进程中导入 torch 是正常现象，不再进行旧的多卡隔离 warning 检查。

    def _init():
        # 当前 env worker 不再设置 worker 专属 GPU；
        # candidate_bitmap 环境只负责构造/执行 SQL 与回传结果。
        print(f"[Process {rank}] Environment initialized (Client Mode).")

        # 错峰启动保留，避免 32 个 worker 同时连接 DB / 初始化环境。
        time.sleep(rank * 0.1)

        # 延迟导入环境类，避免父进程提前加载环境依赖。
        from RL.env_sql_dependency import SqlDependencyEnv

        # 实例化环境，并将队列透传给 worker。
        env = SqlDependencyEnv(shared_queue=shared_queue)
        return env

    return _init


def make_env_isolated_with_reservation(
    rank: int,
    seed: int = 0,
    worker_count: int = 1,
    shared_queue=None,
    shared_reservations=None,
    reservation_lock=None,
    candidate_shared_state=None,
    enumeration_cycle_index: int = 1,
):
    """Factory variant that also passes candidate completion/reservation state."""
    def _init():
        print(f"[Process {rank}] Environment initialized (Client Mode).")
        time.sleep(rank * 0.1)
        from RL.env_sql_dependency import SqlDependencyEnv

        return SqlDependencyEnv(
            # rank/worker_count 用于 PPO 目标边分片或枚举模式的无锁批次位置；
            # 两者均只改变目标边选择，不改变后续构造、执行和回写流程。
            worker_rank=rank,
            worker_count=worker_count,
            shared_queue=shared_queue,
            shared_reservations=shared_reservations,
            reservation_lock=reservation_lock,
            candidate_shared_state=candidate_shared_state,
            enumeration_cycle_index=enumeration_cycle_index,
        )

    return _init
