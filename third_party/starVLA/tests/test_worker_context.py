import os
from pathlib import Path
import sys

import torch
from torch.utils.data import DataLoader, Dataset


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from starVLA.dataloader.worker_context import (
    CPU_ONLY_WORKER_CONTEXT,
    _DISTRIBUTED_ENV_KEYS,
    build_cpu_only_dataloader_kwargs,
)


class _WorkerEnvDataset(Dataset):
    def __len__(self):
        return 1

    def __getitem__(self, index):
        return {
            "CUDA_VISIBLE_DEVICES": os.environ["CUDA_VISIBLE_DEVICES"],
            **{key: os.environ.get(key, "<unset>") for key in _DISTRIBUTED_ENV_KEYS},
        }


class _RefreshGenerationDataset(Dataset):
    def __init__(self) -> None:
        self.shared_generation = torch.zeros((), dtype=torch.int64).share_memory_()
        self.local_generation = 0

    def __len__(self):
        return 16

    def __getitem__(self, index):
        generation = int(self.shared_generation.item())
        if generation != self.local_generation:
            self.local_generation = generation
        return os.getpid(), self.local_generation

    def refresh(self):
        self.shared_generation.add_(1)


def test_cpu_only_worker_context_cleans_spawn_environment_and_restores_parent(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    for index, key in enumerate(_DISTRIBUTED_ENV_KEYS):
        monkeypatch.setenv(key, f"value-{index}")

    loader = DataLoader(
        _WorkerEnvDataset(),
        batch_size=1,
        num_workers=1,
        multiprocessing_context=CPU_ONLY_WORKER_CONTEXT,
    )
    worker_env = next(iter(loader))

    assert worker_env["CUDA_VISIBLE_DEVICES"] == [""]
    for key in _DISTRIBUTED_ENV_KEYS:
        assert worker_env[key] == ["<unset>"]

    assert os.environ["CUDA_VISIBLE_DEVICES"] == "0,1"
    for index, key in enumerate(_DISTRIBUTED_ENV_KEYS):
        assert os.environ[key] == f"value-{index}"


def test_cpu_only_worker_context_survives_dataloader_reconstruction(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")

    loader = DataLoader(
        _WorkerEnvDataset(),
        batch_size=1,
        num_workers=1,
        multiprocessing_context=CPU_ONLY_WORKER_CONTEXT,
    )
    reconstructed = DataLoader(
        loader.dataset,
        batch_size=loader.batch_size,
        num_workers=loader.num_workers,
        multiprocessing_context=loader.multiprocessing_context,
    )

    worker_env = next(iter(reconstructed))

    assert worker_env["CUDA_VISIBLE_DEVICES"] == [""]
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "0,1"


def test_cpu_only_dataloader_kwargs_skips_context_without_workers():
    kwargs = build_cpu_only_dataloader_kwargs(
        0,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,
    )

    assert kwargs == {"pin_memory": True}


def test_cpu_only_dataloader_kwargs_supports_persistent_workers(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")

    loader = DataLoader(
        _WorkerEnvDataset(),
        batch_size=1,
        num_workers=1,
        **build_cpu_only_dataloader_kwargs(
            1,
            pin_memory=False,
            persistent_workers=True,
            prefetch_factor=2,
        ),
    )
    worker_env = next(iter(loader))

    assert loader.persistent_workers is True
    assert worker_env["CUDA_VISIBLE_DEVICES"] == [""]
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "0,1"


def test_persistent_workers_keep_pids_and_drop_prefetched_previous_generation():
    dataset = _RefreshGenerationDataset()
    loader = DataLoader(
        dataset,
        batch_size=1,
        num_workers=2,
        **build_cpu_only_dataloader_kwargs(
            2,
            pin_memory=False,
            persistent_workers=True,
            prefetch_factor=2,
        ),
    )
    first = iter(loader)
    before = [next(first) for _ in range(4)]
    dataset.refresh()
    second = iter(loader)
    after = [next(second) for _ in range(4)]

    before_pids = {int(pid.item()) for pid, _generation in before}
    after_pids = {int(pid.item()) for pid, _generation in after}
    assert before_pids == after_pids
    assert {int(generation.item()) for _pid, generation in before} == {0}
    assert {int(generation.item()) for _pid, generation in after} == {1}
