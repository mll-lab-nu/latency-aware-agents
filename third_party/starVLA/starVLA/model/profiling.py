from contextlib import nullcontext


def stage_timer(profiler, name: str):
    """Time a stage on an optional duck-typed profiler (``.time(name)``).

    Returns a ``nullcontext`` when no profiler is passed, so inference paths that
    do not profile pay nothing and stay decoupled from the eval harness.
    """
    return profiler.time(name) if profiler is not None else nullcontext()
