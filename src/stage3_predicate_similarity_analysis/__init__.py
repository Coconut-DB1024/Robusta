"""Post-run similarity analysis for Stage-3 predicate outputs."""


def run_analysis(*args, **kwargs):
    """Import the analyzer lazily so ``python -m ...analyze`` stays warning-free."""
    from .analyze import run_analysis as _run_analysis

    return _run_analysis(*args, **kwargs)

__all__ = ["run_analysis"]
