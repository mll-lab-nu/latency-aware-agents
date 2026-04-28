from __future__ import annotations

import time
from typing import Any

from latency_bench.logging.wandb import WandbMetricLogger, init_wandb


class MetricLogger(WandbMetricLogger):
    def __init__(self, wandb_module: Any) -> None:
        super().__init__(wandb_module)


def setup_wandb(args: Any) -> Any:
    return init_wandb(args, default_run_name=f"flappy_gpu_render_{int(time.time())}")
