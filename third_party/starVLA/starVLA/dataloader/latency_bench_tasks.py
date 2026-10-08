"""StarVLA loader for latency-bench H1 task contracts."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from starVLA.dataloader.gr00t_lerobot.data_config import BaseDataConfig
from starVLA.dataloader.gr00t_lerobot.datasets import LeRobotMixtureDataset, ModalityConfig
from starVLA.dataloader.gr00t_lerobot.embodiment_tags import EmbodimentTag
from starVLA.dataloader.gr00t_lerobot.registry import (
    ROBOT_TYPE_CONFIG_MAP,
    ROBOT_TYPE_TO_EMBODIMENT_TAG,
    load_custom_mixtures,
)
from starVLA.dataloader.gr00t_lerobot.transform.base import ComposedModalityTransform
from starVLA.dataloader.gr00t_lerobot.transform.state_action import (
    StateActionToTensor,
    StateActionTransform,
)
from starVLA.dataloader.worker_context import build_cpu_only_dataloader_kwargs
from starVLA.dataloader.lerobot_datasets import (
    collate_fn,
    get_vla_dataset as _get_vla_dataset,
    make_LeRobotSingleDataset,
)


class _H1StateActionTransform(StateActionTransform):
    """Match H1 deployment's bounded state normalization."""

    def apply(self, data: dict[str, Any]) -> dict[str, Any]:
        data = super().apply(data)
        data["state.proprio"] = data["state.proprio"].clamp(-1, 1)
        return data


class LatencyBenchH1DataConfig(BaseDataConfig):
    def __init__(self, contract: dict[str, Any]):
        self.robot_type = contract["robot_type"]
        self.video_keys = contract["video_keys"]
        self.state_keys = ["state.proprio"]
        self.action_keys = ["action.target"]
        self.language_keys = ["annotation.human.task_description"]

    def modality_config(self):
        return {
            "video": ModalityConfig(delta_indices=[0], modality_keys=self.video_keys),
            "state": ModalityConfig(delta_indices=[0], modality_keys=self.state_keys),
            "action": ModalityConfig(delta_indices=[0], modality_keys=self.action_keys),
            "language": ModalityConfig(delta_indices=[0], modality_keys=self.language_keys),
        }

    def transform(self):
        keys = [*self.state_keys, *self.action_keys]
        return ComposedModalityTransform(
            transforms=[
                StateActionToTensor(apply_to=keys),
                _H1StateActionTransform(
                    apply_to=keys,
                    normalization_modes={key: "min_max" for key in keys},
                ),
            ]
        )


def _register_contract(contract: dict[str, Any]) -> None:
    config = LatencyBenchH1DataConfig(contract)
    ROBOT_TYPE_CONFIG_MAP[config.robot_type] = config
    ROBOT_TYPE_TO_EMBODIMENT_TAG[config.robot_type] = EmbodimentTag.NEW_EMBODIMENT


def _configured_data_cfg(data_cfg: Any) -> Any:
    cfg = copy.deepcopy(data_cfg)
    load_custom_mixtures(cfg["custom_mixtures_path"])
    contract = json.loads(Path(cfg["task_contract_path"]).expanduser().read_text(encoding="utf-8"))
    _register_contract(contract)
    cfg.include_state = True
    cfg.auxiliary_fields = {
        name: value["column"]
        for name, value in contract["auxiliary"].items()
    }
    return cfg


class _RefreshableLatencyBenchMixture(LeRobotMixtureDataset):
    """Own an incrementally refreshable DAgger mixture inside persistent workers."""

    def __init__(
        self,
        data_cfg: Any,
        *,
        mode: str,
        balance_dataset_weights: bool,
        balance_trajectory_weights: bool,
        seed: int,
    ) -> None:
        self._refresh_data_cfg = copy.deepcopy(data_cfg)
        self._refresh_data_root = Path(self._refresh_data_cfg["data_root_dir"])
        self._refresh_mixture_name = self._refresh_data_cfg["data_mix"]
        self._refresh_mixture_path = Path(self._refresh_data_cfg["custom_mixtures_path"])
        self._refresh_contract = json.loads(
            Path(self._refresh_data_cfg["task_contract_path"]).read_text(encoding="utf-8")
        )
        self._refresh_source_cache: dict[tuple[str, str], Any] = {}
        super().__init__(
            self._materialize_current_mixture(None),
            mode=mode,
            balance_dataset_weights=balance_dataset_weights,
            balance_trajectory_weights=balance_trajectory_weights,
            seed=seed,
            data_cfg=self._refresh_data_cfg,
        )
        self._shared_mixture_generation = torch.zeros(
            (), dtype=torch.int64, device="cpu"
        ).share_memory_()
        self._local_mixture_generation = 0

    def _current_mixture_spec(self) -> list[list[Any]]:
        payload = json.loads(self._refresh_mixture_path.read_text(encoding="utf-8"))
        return payload[self._refresh_mixture_name]

    def _materialize_current_mixture(
        self,
        transform_metadata: dict[str, Any] | None,
    ) -> list[tuple[Any, float]]:
        _register_contract(self._refresh_contract)
        mixture = []
        for data_name, weight, robot_type in self._current_mixture_spec():
            key = (data_name, robot_type)
            if key not in self._refresh_source_cache:
                source = make_LeRobotSingleDataset(
                    self._refresh_data_root,
                    key[0],
                    key[1],
                    delete_pause_frame=False,
                    data_cfg=self._refresh_data_cfg,
                )
                if transform_metadata is not None:
                    # DAgger deltas use the normalization frozen by the initial mixture.
                    source.set_transforms_metadata(transform_metadata[source.tag])
                self._refresh_source_cache[key] = source
            mixture.append((self._refresh_source_cache[key], weight))
        return mixture

    def refresh_custom_mixture(self) -> None:
        """Publish the current mixture file to this process and its worker copies."""
        generation = self._shared_mixture_generation.item() + 1
        self._replace_data_mixture(self._materialize_current_mixture(self.merged_metadata))
        self._local_mixture_generation = generation
        self._shared_mixture_generation.fill_(generation)

    def __getitem__(self, index: int) -> dict:
        generation = self._shared_mixture_generation.item()
        if self._local_mixture_generation != generation:
            self._replace_data_mixture(self._materialize_current_mixture(self.merged_metadata))
            self._local_mixture_generation = generation
        return super().__getitem__(index)


def get_vla_dataset(
    data_cfg: Any,
    mode: str = "train",
    balance_dataset_weights: bool = False,
    balance_trajectory_weights: bool = False,
    seed: int = 42,
    **kwargs: Any,
):
    """Build the standard StarVLA mixture while attaching task auxiliaries."""
    cfg = _configured_data_cfg(data_cfg)
    return _get_vla_dataset(
        data_cfg=cfg,
        mode=mode,
        balance_dataset_weights=balance_dataset_weights,
        balance_trajectory_weights=balance_trajectory_weights,
        seed=seed,
        **kwargs,
    )


def build_dataloader(
    cfg: Any,
    *,
    data_mix: str | None = None,
    mode: str = "train",
    save_statistics_filename: str | None = "dataset_statistics.json",
) -> DataLoader:
    data_cfg = copy.deepcopy(cfg.datasets.vla_data)
    if data_mix is not None:
        data_cfg.data_mix = data_mix
    if mode == "eval":
        eval_sequential = data_cfg.get("eval_sequential_step_sampling")
        if eval_sequential is not None:
            data_cfg.sequential_step_sampling = eval_sequential
    if mode == "train" and "dagger_control_dir" in cfg.trainer:
        dataset = _RefreshableLatencyBenchMixture(
            _configured_data_cfg(data_cfg),
            mode=mode,
            balance_dataset_weights=False,
            balance_trajectory_weights=False,
            seed=42,
        )
    else:
        dataset = get_vla_dataset(data_cfg, mode=mode)
    num_workers = data_cfg["num_workers"]
    if mode == "eval":
        eval_num_workers = data_cfg.get("eval_num_workers")
        if eval_num_workers is not None:
            num_workers = eval_num_workers
        if save_statistics_filename == "dataset_statistics.json":
            save_statistics_filename = "eval_dataset_statistics.json"
    dataloader_kwargs = build_cpu_only_dataloader_kwargs(
        num_workers,
        pin_memory=data_cfg.pin_memory,
        persistent_workers=data_cfg.persistent_workers,
        prefetch_factor=data_cfg.prefetch_factor,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.datasets.vla_data.per_device_batch_size,
        collate_fn=collate_fn,
        num_workers=num_workers,
        **dataloader_kwargs,
    )
    if save_statistics_filename:
        output_dir = Path(cfg.output_dir)
        dataset.save_dataset_statistics(output_dir / save_statistics_filename)
    return dataloader


__all__ = ["LatencyBenchH1DataConfig", "build_dataloader", "collate_fn", "get_vla_dataset"]
