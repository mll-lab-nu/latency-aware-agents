import json
import os
import copy
from accelerate.logging import get_logger
import numpy as np
from torch.utils.data import DataLoader
import numpy as np
import torch.distributed as dist
from pathlib import Path
from starVLA.dataloader.worker_context import build_cpu_only_dataloader_kwargs
from starVLA.dataloader.vlm_datasets import make_vlm_dataloader

logger = get_logger(__name__)


def save_dataset_statistics(dataset_statistics, run_dir):
    """Saves a `dataset_statistics.json` file."""
    out_path = run_dir / "dataset_statistics.json"
    with open(out_path, "w") as f_json:
        for _, stats in dataset_statistics.items():
            for k in stats["action"].keys():
                if isinstance(stats["action"][k], np.ndarray):
                    stats["action"][k] = stats["action"][k].tolist()
            if "proprio" in stats:
                for k in stats["proprio"].keys():
                    if isinstance(stats["proprio"][k], np.ndarray):
                        stats["proprio"][k] = stats["proprio"][k].tolist()
            if "num_trajectories" in stats:
                if isinstance(stats["num_trajectories"], np.ndarray):
                    stats["num_trajectories"] = stats["num_trajectories"].item()
            if "num_transitions" in stats:
                if isinstance(stats["num_transitions"], np.ndarray):
                    stats["num_transitions"] = stats["num_transitions"].item()
        json.dump(dataset_statistics, f_json, indent=2)
    logger.info(f"Saved dataset statistics file at path {out_path}")



def build_dataloader(
    cfg,
    dataset_py="lerobot_datasets_oxe",
    *,
    data_mix: str | None = None,
    mode: str = "train",
    save_statistics_filename: str | None = "dataset_statistics.json",
): # TODO now here only is get dataset, we need mv dataloader to here

    if dataset_py == "lerobot_datasets":
        from starVLA.dataloader.lerobot_datasets import get_vla_dataset, collate_fn
        vla_dataset_cfg = copy.deepcopy(cfg.datasets.vla_data)
        if data_mix:
            vla_dataset_cfg.data_mix = data_mix
        if mode == "eval":
            eval_sequential = vla_dataset_cfg.get("eval_sequential_step_sampling", None)
            if eval_sequential is not None:
                vla_dataset_cfg.sequential_step_sampling = eval_sequential

        vla_dataset = get_vla_dataset(data_cfg=vla_dataset_cfg, mode=mode)
        num_workers_value = vla_dataset_cfg.get("num_workers", 4)
        num_workers = int(4 if num_workers_value is None else num_workers_value)
        if mode == "eval":
            eval_num_workers = vla_dataset_cfg.get("eval_num_workers", None)
            if eval_num_workers is not None:
                num_workers = int(eval_num_workers)
        dataloader_kwargs = build_cpu_only_dataloader_kwargs(
            num_workers,
            pin_memory=vla_dataset_cfg.pin_memory,
            persistent_workers=vla_dataset_cfg.persistent_workers,
            prefetch_factor=vla_dataset_cfg.prefetch_factor if "prefetch_factor" in vla_dataset_cfg else None,
        )
        if not dist.is_initialized() or dist.get_rank() == 0:
            logger.info(
                "Created %s VLA dataloader with num_workers=%s, persistent_workers=%s, cpu_only_workers=%s",
                mode,
                num_workers,
                dataloader_kwargs.get("persistent_workers", False),
                "multiprocessing_context" in dataloader_kwargs,
            )

        vla_train_dataloader = DataLoader(
            vla_dataset,
            batch_size=cfg.datasets.vla_data.per_device_batch_size,
            collate_fn=collate_fn,
            num_workers=num_workers,
            **dataloader_kwargs,
            # shuffle=True
        )        
        if save_statistics_filename and (not dist.is_initialized() or dist.get_rank() == 0): 
            
            output_dir = Path(cfg.output_dir)
            vla_dataset.save_dataset_statistics(output_dir / save_statistics_filename)
        return vla_train_dataloader
    elif dataset_py == "latency_bench_tasks":
        from starVLA.dataloader.latency_bench_tasks import build_dataloader as build_latency_bench_dataloader

        return build_latency_bench_dataloader(
            cfg,
            data_mix=data_mix,
            mode=mode,
            save_statistics_filename=save_statistics_filename,
        )
    elif dataset_py == "vlm_datasets":
        vlm_data_module = make_vlm_dataloader(cfg)
        vlm_train_dataloader = vlm_data_module["train_dataloader"]
        
        return vlm_train_dataloader
