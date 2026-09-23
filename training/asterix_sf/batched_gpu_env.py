from __future__ import annotations

from training.asterix_sf.gpu_core import CudaAsterixCore
from training.demon_attack_sf.batched_gpu_env import DemonAttackBatchedGpuEnv


class AsterixBatchedGpuEnv(DemonAttackBatchedGpuEnv):
    """Run batched Asterix emulation, preprocessing, stacking, and stepping on CUDA."""

    core_cls = CudaAsterixCore
