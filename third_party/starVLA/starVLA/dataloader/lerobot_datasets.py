# Copyright 2025 NVIDIA Corp. and affiliates. All rights reserved.
# Modified by [Fangjing Wang/ SUST University] in [2025]. 
# Modification: [return raw data and suport multi-dataset mixture].
# Modified by [Jinhui YE/ HKUST University] in [2025]. 
# Modification: [suport topdowm processing, suport param from config].

import json
from pathlib import Path
from typing import Sequence
from omegaconf import OmegaConf

from starVLA.dataloader.gr00t_lerobot.datasets import LeRobotSingleDataset, LeRobotMixtureDataset
from starVLA.dataloader.gr00t_lerobot.registry import (
    ROBOT_TYPE_CONFIG_MAP,
    ROBOT_TYPE_TO_EMBODIMENT_TAG,
    EmbodimentTag,
    get_dataset_named_mixture,
    load_custom_mixtures,
)
from starVLA.training.rl_games.temporal_clip import resolve_modality_indices

RL_GAMES_TASK_METADATA = {
    "rl_games_flappy": ("flappy", 2),
    "rl_games_demon_attack": ("demon_attack", 6),
    "rl_games_defend_the_line": ("defend_the_line", 6),
    "rl_games_deadly_corridor": ("deadly_corridor", 7),
    "rl_games_asterix": ("asterix", 9),
    "rl_games_atlantis": ("atlantis", 4),
}
RL_GAMES_GYMNASIUM_ROBOT_TYPES = {
    "rl_games_gymnasium",
    "rl_games_gymnasium_discrete",
    "rl_games_gymnasium_native",
}
def _gymnasium_task_contract(manifest: dict) -> dict:
    return dict(manifest["gymnasium_task"])


def _configured_gymnasium_task_contract(data_cfg: dict) -> dict:
    return dict(data_cfg["gymnasium_task_contract"])


def _generic_gymnasium_metadata(
    dataset_path: Path,
    data_cfg: dict,
) -> tuple[dict, int]:
    manifest = json.loads(
        (dataset_path / "manifest.json").read_text(encoding="utf-8")
    )
    manifest_contract = _gymnasium_task_contract(manifest)
    configured_contract = _configured_gymnasium_task_contract(data_cfg)
    mismatched_fields = [
        key
        for key in configured_contract
        if manifest_contract[key] != configured_contract[key]
    ]
    if mismatched_fields:
        mismatch_details = "; ".join(
            f"{key}: manifest={manifest_contract[key]!r}, config={configured_contract[key]!r}"
            for key in mismatched_fields
        )
        raise ValueError(
            f"Gymnasium task contract mismatch for dataset {dataset_path}: {mismatch_details}"
        )

    active_action_dim = manifest["active_action_dim"]
    configured_action_dim = data_cfg["active_action_dim"]
    if active_action_dim != configured_action_dim:
        raise ValueError(
            f"Gymnasium active_action_dim mismatch for dataset {dataset_path}: "
            f"manifest={active_action_dim!r}, config={configured_action_dim!r}"
        )
    return manifest_contract, active_action_dim


def _manifest_action_dim(dataset_path: Path, fallback: int) -> int:
    manifest_path = dataset_path / "manifest.json"
    if not manifest_path.exists():
        return int(fallback)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return int(manifest.get("active_action_dim", fallback))


def collate_fn(batch):
    return batch


def _modality_config_with_dataset_indices(data_config, data_cfg: dict | None):
    modality_config = data_config.modality_config()
    if data_cfg is None:
        return modality_config

    video_cfg = modality_config.get("video")
    state_cfg = modality_config.get("state")
    action_cfg = modality_config.get("action")
    language_cfg = modality_config.get("language")
    if video_cfg is None or state_cfg is None or action_cfg is None:
        return modality_config

    resolved = resolve_modality_indices(
        default_observation_indices=list(video_cfg.delta_indices),
        default_state_indices=list(state_cfg.delta_indices),
        default_action_indices=list(action_cfg.delta_indices),
        data_cfg=data_cfg,
    )
    video_cfg.delta_indices = resolved.observation_indices
    state_cfg.delta_indices = resolved.state_indices
    action_cfg.delta_indices = resolved.action_indices
    if language_cfg is not None:
        language_cfg.delta_indices = resolved.language_indices
    return modality_config

def make_LeRobotSingleDataset(
    data_root_dir: Path | str,
    data_name: str,
    robot_type: str,
    delete_pause_frame: bool = False,
    data_cfg: dict | None = None,
) -> LeRobotSingleDataset:
    """
    Make a LeRobotSingleDataset object.

    :param data_root_dir: The root directory of the dataset.
    :param data_name: The name of the dataset.
    :param robot_type: The robot type config to use.
    :param crop_obs_camera: Whether to crop the observation camera images.
    :return: A LeRobotSingleDataset object.
    """
    
    data_config = ROBOT_TYPE_CONFIG_MAP[robot_type]
    modality_config = _modality_config_with_dataset_indices(data_config=data_config, data_cfg=data_cfg)
    transforms = data_config.transform()

    # Temporal observation window: expand the video modality's delta indices to the
    # last N frames so `_pack_sample` can emit a chronological frame sequence.
    # Driven centrally here (one place) rather than editing every DataConfig.
    num_obs_frames = int(data_cfg.get("num_obs_frames", 1) or 1) if data_cfg else 1
    if num_obs_frames > 1:
        modality_config["video"].delta_indices = list(range(-(num_obs_frames - 1), 1))
        # KV-memory per-frame supervision (scheme B): each of the R observation frames
        # needs its OWN action chunk, not just the base-index one. Widen the action delta
        # to cover the rollout window so `_pack_sample` can slice a chunk per frame.
        # Assumes the base action_indices are a contiguous horizon range(0, H).
        if data_cfg and bool(data_cfg.get("kv_memory", False)):
            action_cfg = modality_config["action"]
            horizon = len(list(action_cfg.delta_indices))
            action_cfg.delta_indices = list(range(-(num_obs_frames - 1), horizon))
    dataset_path = data_root_dir / data_name
    if robot_type not in ROBOT_TYPE_TO_EMBODIMENT_TAG:
        print(f"Warning: Robot type {robot_type} not found in ROBOT_TYPE_TO_EMBODIMENT_TAG, using {EmbodimentTag.NEW_EMBODIMENT} as default")
        embodiment_tag = EmbodimentTag.NEW_EMBODIMENT
    else:
        embodiment_tag = ROBOT_TYPE_TO_EMBODIMENT_TAG[robot_type]
    
    video_backend = data_cfg.get("video_backend", "decord") if data_cfg else "torchvision_av"
    dataset = LeRobotSingleDataset(
        dataset_path=dataset_path,
        modality_configs=modality_config,
        transforms=transforms,
        embodiment_tag=embodiment_tag,
        video_backend=video_backend, # decord is more efficiency | torchvision_av for video.av1
        delete_pause_frame=delete_pause_frame,
        data_cfg=data_cfg,
    )
    if robot_type in RL_GAMES_GYMNASIUM_ROBOT_TYPES:
        task_contract, active_action_dim = _generic_gymnasium_metadata(
            dataset_path,
            data_cfg,
        )
        dataset.rl_games_gymnasium_task_contract = task_contract
        dataset.rl_games_task = task_contract["task_name"]
        dataset.rl_games_action_env_dim = active_action_dim
    elif robot_type in RL_GAMES_TASK_METADATA:
        task_name, fallback_action_dim = RL_GAMES_TASK_METADATA[robot_type]
        dataset.rl_games_task = task_name
        dataset.rl_games_action_env_dim = _manifest_action_dim(dataset_path, fallback_action_dim)
    return dataset

def get_vla_dataset(
    data_cfg: dict,
    mode: str = "train",
    balance_dataset_weights: bool = False,
    balance_trajectory_weights: bool = False,
    seed: int = 42,
    **kwargs: dict,
) -> LeRobotMixtureDataset:
    """
    Get a LeRobotMixtureDataset object.
    """
    data_root_dir = data_cfg.data_root_dir
    data_mix = data_cfg.data_mix
    load_custom_mixtures(data_cfg.get("custom_mixtures_path", None))
    delete_pause_frame = data_cfg.get("delete_pause_frame", False)
    mixture_spec = get_dataset_named_mixture(data_mix)
    included_datasets, filtered_mixture_spec = set(), []
    for d_name, d_weight, robot_type in mixture_spec:  
        dataset_key = (d_name, robot_type)  
        if dataset_key in included_datasets:
            print(f"Skipping Duplicate Dataset: `{(d_name, d_weight, robot_type)}`")
            continue

        included_datasets.add(dataset_key)
        filtered_mixture_spec.append((d_name, d_weight, robot_type))

    dataset_mixture = []
    for d_name, d_weight, robot_type in filtered_mixture_spec:
        dataset_mixture.append((make_LeRobotSingleDataset(Path(data_root_dir), d_name, robot_type, delete_pause_frame=delete_pause_frame, data_cfg=data_cfg), d_weight))

    return LeRobotMixtureDataset(
        dataset_mixture,
        mode=mode,
        balance_dataset_weights=balance_dataset_weights,
        balance_trajectory_weights=balance_trajectory_weights,
        seed=seed,
        data_cfg=data_cfg,
        **kwargs,
    )



if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser()
    parser.add_argument("--config_yaml", type=str, default="examples/LIBERO/train_files/starvla_cotrain_libero.yaml", help="Path to YAML config")
    args, clipargs = parser.parse_known_args()

    if os.getenv("DEBUGPY_ENABLE", "0") == "1":
        import debugpy
        debugpy.listen(("0.0.0.0", 10092))
        print("Rank 0 waiting for debugger attach on port 10092...")
        debugpy.wait_for_client()

    cfg = OmegaConf.load(args.config_yaml)
    vla_dataset_cfg = cfg.datasets.vla_data
    for task_id in ["all"]:
        vla_dataset_cfg.task_id = task_id
        print(f"Testing Task ID: {task_id}")
        dataset = get_vla_dataset(data_cfg=vla_dataset_cfg)
    from torch.utils.data import DataLoader
    train_dataloader = DataLoader(
        dataset,
        batch_size=2,
        num_workers=1, # For Debug
        collate_fn=collate_fn,
    )

    cfg.output_dir = "./results/debug"
    output_dir = Path(cfg.output_dir)
    dataset.save_dataset_statistics(output_dir / "dataset_statistics.json")

    from tqdm import tqdm
    count = 0
    for batch in tqdm(train_dataloader, desc="Processing Batches"):
        if count > 100:
            break
        count += 1
        pass
