from contextlib import contextmanager
import os
from pathlib import Path
import tempfile
from threading import Lock


_WARNING_DIRECTORY_ENV = "ROBUSTA_RUNTIME_WARNING_DIRECTORY"
_JOINT_SIGNATURE_TIMEOUT = "joint signature evidence lock timeout"
_seen_warning_scopes = set()
_warning_lock = Lock()


@contextmanager
def suppress_sb3_logging_destination():
    from stable_baselines3.common.logger import Logger

    original_log = Logger.log

    def log_without_destination(logger, *args, **kwargs):
        if len(args) == 1 and args[0] == f"Logging to {logger.get_dir()}":
            return
        return original_log(logger, *args, **kwargs)

    Logger.log = log_without_destination
    try:
        yield
    finally:
        Logger.log = original_log


@contextmanager
def runtime_warning_scope():
    if os.environ.get(_WARNING_DIRECTORY_ENV):
        yield
        return
    try:
        temporary = tempfile.TemporaryDirectory(
            prefix="robusta-runtime-log-", ignore_cleanup_errors=True,
        )
    except OSError:
        yield
        return
    with temporary as directory:
        os.environ[_WARNING_DIRECTORY_ENV] = directory
        try:
            yield
        finally:
            os.environ.pop(_WARNING_DIRECTORY_ENV, None)


def report_joint_signature_flush_failure(error, *, final=False):
    detail = str(error)
    if isinstance(error, TimeoutError) and (
        detail == _JOINT_SIGNATURE_TIMEOUT
        or detail.startswith(_JOINT_SIGNATURE_TIMEOUT + ":")
    ):
        directory = os.environ.get(_WARNING_DIRECTORY_ENV)
        with _warning_lock:
            if directory in _seen_warning_scopes:
                return
            _seen_warning_scopes.add(directory)
            if directory:
                try:
                    with (Path(directory) / "joint-signature-timeout.logged").open("x"):
                        pass
                except FileExistsError:
                    return
                except OSError:
                    pass
        print(
            f"[RuntimeStats Warning] joint signature flush failed: {_JOINT_SIGNATURE_TIMEOUT}",
            flush=True,
        )
        return
    stage = "final joint signature" if final else "joint signature"
    print(f"[RuntimeStats Warning] {stage} flush failed: {detail}", flush=True)
