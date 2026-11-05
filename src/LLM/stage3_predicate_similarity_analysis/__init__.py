"""Post-run similarity analysis of the independent LLM's satisfiable TXT."""


def run_analysis(*args, **kwargs):
    """Import the analyzer lazily so ``python -m ...analyze`` stays warning-free."""
    from .analyze import run_analysis as _run_analysis

    return _run_analysis(*args, **kwargs)

__all__ = ["run_analysis"]
