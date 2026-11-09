import os
import time
from RL.progress import EnvironmentReloadError


def make_env_isolated(rank: int, seed: int = 0, shared_queue=None):
    def _init():
        time.sleep(rank * 0.1)

        try:
            from RL.env_sql_dependency import SqlDependencyEnv

            env = SqlDependencyEnv(shared_queue=shared_queue)
        except EnvironmentReloadError:
            raise
        except Exception as exc:
            print(f"[Process {rank}] Environment initialization failed: {type(exc).__name__}: {exc}", flush=True)
            raise
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
    environment_progress=None,
):
    def _init():
        time.sleep(rank * 0.1)

        try:
            from RL.env_sql_dependency import SqlDependencyEnv

            return SqlDependencyEnv(
                worker_rank=rank,
                worker_count=worker_count,
                shared_queue=shared_queue,
                shared_reservations=shared_reservations,
                reservation_lock=reservation_lock,
                candidate_shared_state=candidate_shared_state,
                enumeration_cycle_index=enumeration_cycle_index,
                environment_progress=environment_progress,
            )
        except EnvironmentReloadError:
            raise
        except Exception as exc:
            print(f"[Process {rank}] Environment initialization failed: {type(exc).__name__}: {exc}", flush=True)
            raise

    return _init
