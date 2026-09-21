from __future__ import annotations

from training.atlantis_sf.gpu_core import CudaAtlantisCore
from training.demon_attack_sf.batched_gpu_env import DemonAttackBatchedGpuEnv


class AtlantisBatchedGpuEnv(DemonAttackBatchedGpuEnv):
    """Run batched Atlantis emulation, preprocessing, stacking, and stepping on CUDA."""

    core_cls = CudaAtlantisCore
