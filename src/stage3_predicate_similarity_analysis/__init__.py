def run_analysis(*args, **kwargs):
    from .analyze import run_analysis as _run_analysis

    return _run_analysis(*args, **kwargs)

__all__ = ["run_analysis"]
