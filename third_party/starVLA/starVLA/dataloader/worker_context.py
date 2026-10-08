import multiprocessing.context
import os
from typing import Any


_DISTRIBUTED_ENV_KEYS = (
    "LOCAL_RANK",
    "RANK",
    "WORLD_SIZE",
    "MASTER_ADDR",
    "MASTER_PORT",
    "GROUP_RANK",
    "LOCAL_WORLD_SIZE",
    "ROLE_RANK",
    "ROLE_WORLD_SIZE",
    "TORCHELASTIC_RUN_ID",
    "TORCHELASTIC_RESTART_COUNT",
    "TORCHELASTIC_MAX_RESTARTS",
)


class _CpuOnlyWorkerSpawnProcess(multiprocessing.context.SpawnProcess):
    def start(self):
        had_cuda_visible_devices = "CUDA_VISIBLE_DEVICES" in os.environ
        if had_cuda_visible_devices:
            cuda_visible_devices = os.environ["CUDA_VISIBLE_DEVICES"]
        distributed_env = {key: os.environ[key] for key in _DISTRIBUTED_ENV_KEYS if key in os.environ}

        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        for key in distributed_env:
            del os.environ[key]
        try:
            return super().start()
        finally:
            if had_cuda_visible_devices:
                os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
            else:
                del os.environ["CUDA_VISIBLE_DEVICES"]
            for key, value in distributed_env.items():
                os.environ[key] = value


class CpuOnlyWorkerSpawnContext(multiprocessing.context.SpawnContext):
    Process = _CpuOnlyWorkerSpawnProcess


CPU_ONLY_WORKER_CONTEXT = CpuOnlyWorkerSpawnContext()


def _as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() not in {"false", "0", "no", "off"}
    return bool(value)


def build_cpu_only_dataloader_kwargs(
    num_workers: int,
    *,
    pin_memory: Any = False,
    persistent_workers: Any = False,
    prefetch_factor: Any = None,
) -> dict[str, Any]:
    dataloader_kwargs: dict[str, Any] = {
        "pin_memory": _as_bool(pin_memory),
    }
    if int(num_workers) <= 0:
        return dataloader_kwargs

    dataloader_kwargs["multiprocessing_context"] = CPU_ONLY_WORKER_CONTEXT
    dataloader_kwargs["persistent_workers"] = _as_bool(persistent_workers)
    if prefetch_factor is not None:
        dataloader_kwargs["prefetch_factor"] = int(prefetch_factor)
    return dataloader_kwargs
