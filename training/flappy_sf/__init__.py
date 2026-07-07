"""Flappy Bird sample-factory integration package.

Intentionally a thin/empty package init: ``integration.py`` pulls in
``sample_factory`` (a heavy, optional dependency that is absent in the starVLA
training env), so it must NOT be imported eagerly here — otherwise importing a
lightweight submodule like ``flappy_sf.gpu_rendering`` (used by latency_bench's
flappy env) would transitively require sample_factory and fail. Import what you
need explicitly, e.g. ``from training.flappy_sf.integration import run_from_config``.
"""
