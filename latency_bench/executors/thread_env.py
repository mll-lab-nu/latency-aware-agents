from __future__ import annotations

import os


THREAD_ENV_DEFAULTS = {
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}


def set_thread_env_defaults() -> None:
    for key, value in THREAD_ENV_DEFAULTS.items():
        os.environ.setdefault(key, value)
