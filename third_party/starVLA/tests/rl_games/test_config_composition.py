from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf

from starVLA.training.rl_games.action_spec import apply_action_spec
from starVLA.training.rl_games.alias import apply_model_alias
from starVLA.training.rl_games.config_validation import validate_rl_games_config


REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "examples" / "rl_games" / "config"


@dataclass(frozen=True)
class ExpectedComposition:
    model: str
    env: str
    init: str
    mode: str
    model_alias: str
    framework_name: str
    task: str
    action_carrier: str
    latency_values: tuple[int, ...]
    data_mix: str
    source_hf: str
    action_env_dim: int
    base_model_repo_id: str
    initialization_hf_repo_id: str | None


SUPPORTED_COMPOSITIONS: tuple[ExpectedComposition, ...] = (
    ExpectedComposition(
        model="openvla",
        env="flappy",
        init="scratch",
        mode="single",
        model_alias="openvla",
        framework_name="QwenOFT",
        task="flappy",
        action_carrier="native",
        latency_values=(0,),
        data_mix="flappy_train",
        source_hf="",
        action_env_dim=2,
        base_model_repo_id="StarVLA/Qwen3-VL-4B-Instruct-Action",
        initialization_hf_repo_id=None,
    ),
    ExpectedComposition(
        model="openvla",
        env="flappy",
        init="bridge",
        mode="single",
        model_alias="openvla",
        framework_name="QwenOFT",
        task="flappy",
        action_carrier="bridge",
        latency_values=(0,),
        data_mix="flappy_train",
        source_hf="",
        action_env_dim=2,
        base_model_repo_id="Qwen/Qwen3-VL-4B-Instruct",
        initialization_hf_repo_id="StarVLA/Qwen3VL-OFT-Bridge-RT-1",
    ),
    ExpectedComposition(
        model="pi0",
        env="demon_attack",
        init="bridge",
        mode="single",
        model_alias="pi-0",
        framework_name="QwenPI",
        task="demon_attack",
        action_carrier="bridge",
        latency_values=(0,),
        data_mix="demon_attack_train",
        source_hf="",
        action_env_dim=6,
        base_model_repo_id="StarVLA/Qwen2.5-VL-3B-Instruct-Action",
        initialization_hf_repo_id="StarVLA/Qwen-PI-Bridge-RT-1",
    ),
    ExpectedComposition(
        model="openvla",
        env="air_raid",
        init="bridge",
        mode="single",
        model_alias="openvla",
        framework_name="QwenOFT",
        task="air_raid",
        action_carrier="bridge",
        latency_values=(0,),
        data_mix="air_raid_train",
        source_hf="",
        action_env_dim=6,
        base_model_repo_id="Qwen/Qwen3-VL-4B-Instruct",
        initialization_hf_repo_id="StarVLA/Qwen3VL-OFT-Bridge-RT-1",
    ),
    ExpectedComposition(
        model="openvla",
        env="defend_the_line",
        init="bridge",
        mode="single",
        model_alias="openvla",
        framework_name="QwenOFT",
        task="defend_the_line",
        action_carrier="bridge",
        latency_values=(0,),
        data_mix="defend_the_line_train",
        source_hf="latency-sensitive-bench/benchmark-datasets",
        action_env_dim=6,
        base_model_repo_id="Qwen/Qwen3-VL-4B-Instruct",
        initialization_hf_repo_id="StarVLA/Qwen3VL-OFT-Bridge-RT-1",
    ),
    ExpectedComposition(
        model="pi05",
        env="deadly_corridor",
        init="bridge",
        mode="mixed_latency",
        model_alias="pi-0.5",
        framework_name="QwenPI_v3",
        task="deadly_corridor",
        action_carrier="bridge",
        latency_values=(0, 1, 2, 3, 4, 5),
        data_mix="deadly_corridor_mixed_latency_train",
        source_hf="",
        action_env_dim=7,
        base_model_repo_id="Qwen/Qwen3-VL-4B-Instruct",
        initialization_hf_repo_id="StarVLA/Qwen3VL-PI_v3-Bridge-RT_1",
    ),
    ExpectedComposition(
        model="gr00t",
        env="deadly_corridor",
        init="scratch",
        mode="mixed_latency",
        model_alias="gr00t",
        framework_name="QwenGR00T",
        task="deadly_corridor",
        action_carrier="native",
        latency_values=(0, 1, 2, 3, 4, 5),
        data_mix="deadly_corridor_mixed_latency_train",
        source_hf="",
        action_env_dim=7,
        base_model_repo_id="StarVLA/Qwen3-VL-4B-Instruct-Action",
        initialization_hf_repo_id=None,
    ),
)


def _compose_cfg(expected: ExpectedComposition) -> DictConfig:
    with initialize_config_dir(version_base="1.1", config_dir=str(CONFIG_DIR)):
        cfg = compose(
            config_name="train",
            overrides=[
                f"model={expected.model}",
                f"env={expected.env}",
                f"init={expected.init}",
                f"mode={expected.mode}",
            ],
        )
    validate_rl_games_config(cfg)
    apply_model_alias(cfg)
    apply_action_spec(cfg)
    return cfg


@pytest.mark.parametrize("expected", SUPPORTED_COMPOSITIONS)
def test_supported_rl_games_config_composes(expected: ExpectedComposition) -> None:
    cfg = _compose_cfg(expected)
    latency_values = OmegaConf.select(cfg, "rl_games.env_eval.latency.values")
    post_train_latencies = OmegaConf.select(cfg, "rl_games.env_eval.post_train.latencies")

    assert cfg.rl_games.model_alias == expected.model_alias
    assert cfg.framework.name == expected.framework_name
    assert cfg.rl_games.task == expected.task
    assert cfg.rl_games.action_carrier == expected.action_carrier
    assert OmegaConf.select(cfg, "framework.qwenvl.base_vlm") not in (None, "")
    assert tuple(OmegaConf.to_container(latency_values, resolve=True)) == expected.latency_values
    assert tuple(OmegaConf.to_container(post_train_latencies, resolve=True)) == (0, 1, 2, 3, 4)
    assert OmegaConf.select(cfg, "rl_games.env_eval.interval_steps") is None
    assert OmegaConf.select(cfg, "rl_games.env_eval.num_episodes") is None
    assert OmegaConf.select(cfg, "rl_games.env_eval.max_episode_steps") is None
    assert cfg.rl_games.env_eval.image_size == 224
    assert cfg.datasets.vla_data.data_mix == expected.data_mix
    assert cfg.datasets.vla_data.obs_image_size is None
    assert cfg.dataset.source_hf == expected.source_hf
    assert cfg.framework.action_model.action_env_dim == expected.action_env_dim
    assert cfg.base_model.repo_id == expected.base_model_repo_id
    assert cfg.initialization.checkpoint_hf_repo_id == expected.initialization_hf_repo_id


def test_openvla_bridge_accepts_base_vlm_override() -> None:
    cfg = _compose_cfg(
        ExpectedComposition(
            model="openvla",
            env="flappy",
            init="bridge",
            mode="single",
            model_alias="openvla",
            framework_name="QwenOFT",
            task="flappy",
            action_carrier="bridge",
            latency_values=(0,),
            data_mix="flappy_train",
            source_hf="",
            action_env_dim=2,
            base_model_repo_id="Qwen/Qwen3-VL-4B-Instruct",
            initialization_hf_repo_id="StarVLA/Qwen3VL-OFT-Bridge-RT-1",
        )
    )

    OmegaConf.update(cfg, "framework.qwenvl.base_vlm", "/tmp/Qwen3-VL-4B-Instruct", merge=False)

    assert cfg.framework.qwenvl.base_vlm == "/tmp/Qwen3-VL-4B-Instruct"


@pytest.mark.parametrize("init", ("scratch", "bridge"))
def test_openvla_uses_discrete_ce_loss(init: str) -> None:
    cfg = _compose_cfg(
        ExpectedComposition(
            model="openvla",
            env="flappy",
            init=init,
            mode="single",
            model_alias="openvla",
            framework_name="QwenOFT",
            task="flappy",
            action_carrier="native" if init == "scratch" else "bridge",
            latency_values=(0,),
            data_mix="flappy_train",
            source_hf="",
            action_env_dim=2,
            base_model_repo_id=(
                "StarVLA/Qwen3-VL-4B-Instruct-Action" if init == "scratch" else "Qwen/Qwen3-VL-4B-Instruct"
            ),
            initialization_hf_repo_id=None if init == "scratch" else "StarVLA/Qwen3VL-OFT-Bridge-RT-1",
        )
    )

    assert cfg.framework.action_model.loss_type == "discrete_ce"


def test_deadly_corridor_single_mode_does_not_require_manifest_latency_filter() -> None:
    cfg = _compose_cfg(
        ExpectedComposition(
            model="pi05",
            env="deadly_corridor",
            init="bridge",
            mode="single",
            model_alias="pi-0.5",
            framework_name="QwenPI_v3",
            task="deadly_corridor",
            action_carrier="bridge",
            latency_values=(0,),
            data_mix="deadly_corridor_train",
            source_hf="",
            action_env_dim=7,
            base_model_repo_id="Qwen/Qwen3-VL-4B-Instruct",
            initialization_hf_repo_id="StarVLA/Qwen3VL-PI_v3-Bridge-RT_1",
        )
    )

    assert cfg.dataset.latency_filter is None
