from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
GENERIC_ROBOT_TYPE = "rl_games_gymnasium"
GYMNASIUM_TASK_CONTRACT_KEYS = (
    "task_name",
    "env_id",
    "make_kwargs",
    "registration_imports",
    "action_labels",
    "action_values",
    "noop_action_id",
    "base_prompt",
    "env_fps",
    "obs_fps",
    "frame_stack",
)


class AttrDict(dict):
    __getattr__ = dict.__getitem__


def _gymnasium_task_contract(task_name: str = "custom_balance") -> dict:
    return {
        "task_name": task_name,
        "env_id": "CustomBalance-v0",
        "make_kwargs": {"render_mode": "rgb_array"},
        "registration_imports": ["custom_balance.envs"],
        "action_labels": ["left", "coast", "right"],
        "action_values": [10, 20, 30],
        "noop_action_id": 1,
        "base_prompt": "Balance the custom system.",
        "env_fps": 30,
        "obs_fps": 10,
        "frame_stack": 1,
    }


def _write_manifest(
    dataset_path: Path,
    *,
    task_contract: dict,
    active_action_dim: int = 3,
) -> None:
    dataset_path.mkdir()
    (dataset_path / "manifest.json").write_text(
        json.dumps(
            {
                "integration_name": "gymnasium",
                "task_name": task_contract["task_name"],
                "gymnasium_task": task_contract,
                "robot_type": GENERIC_ROBOT_TYPE,
                "active_action_dim": active_action_dim,
                "action_dim": active_action_dim,
                "action_carrier": "native",
            }
        ),
        encoding="utf-8",
    )


def _data_cfg(task_contract: dict, active_action_dim: int = 3) -> AttrDict:
    return AttrDict(
        gymnasium_task_contract=task_contract,
        active_action_dim=active_action_dim,
        custom_mixtures_path=None,
    )


def _load_rl_games_data_config(monkeypatch: pytest.MonkeyPatch):
    embodiment_module = ModuleType(
        "starVLA.dataloader.gr00t_lerobot.embodiment_tags"
    )
    embodiment_module.EmbodimentTag = SimpleNamespace(NEW_EMBODIMENT="new")
    monkeypatch.setitem(sys.modules, embodiment_module.__name__, embodiment_module)

    path = (
        REPO_ROOT
        / "examples"
        / "rl_games"
        / "train_files"
        / "data_registry"
        / "data_config.py"
    )
    spec = importlib.util.spec_from_file_location("_rl_games_data_config_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_lerobot_datasets(monkeypatch: pytest.MonkeyPatch):
    class FakeLeRobotSingleDataset:
        def __init__(self, **kwargs):
            self.init_kwargs = kwargs

    class FakeLeRobotMixtureDataset:
        def __init__(self, dataset_mixture, **kwargs):
            self.dataset_mixture = dataset_mixture
            self.kwargs = kwargs

    class FakeDataConfig:
        def modality_config(self):
            return {
                "video": SimpleNamespace(delta_indices=[0]),
                "state": SimpleNamespace(delta_indices=[0]),
                "action": SimpleNamespace(delta_indices=[0]),
                "language": SimpleNamespace(delta_indices=[0]),
            }

        def transform(self):
            return "transform"

    robot_types = {
        "rl_games_flappy",
        "rl_games_demon_attack",
        "rl_games_deadly_corridor",
        GENERIC_ROBOT_TYPE,
        "rl_games_gymnasium_native",
    }
    datasets_module = ModuleType(
        "starVLA.dataloader.gr00t_lerobot.datasets"
    )
    datasets_module.LeRobotSingleDataset = FakeLeRobotSingleDataset
    datasets_module.LeRobotMixtureDataset = FakeLeRobotMixtureDataset
    monkeypatch.setitem(sys.modules, datasets_module.__name__, datasets_module)

    registry_module = ModuleType(
        "starVLA.dataloader.gr00t_lerobot.registry"
    )
    registry_module.ROBOT_TYPE_CONFIG_MAP = {
        robot_type: FakeDataConfig() for robot_type in robot_types
    }
    registry_module.ROBOT_TYPE_TO_EMBODIMENT_TAG = {
        robot_type: "new" for robot_type in robot_types
    }
    registry_module.EmbodimentTag = SimpleNamespace(NEW_EMBODIMENT="new")
    registry_module.get_dataset_named_mixture = None
    registry_module.load_custom_mixtures = None
    monkeypatch.setitem(sys.modules, registry_module.__name__, registry_module)

    temporal_clip_module = ModuleType(
        "starVLA.training.rl_games.temporal_clip"
    )
    temporal_clip_module.resolve_modality_indices = lambda **kwargs: SimpleNamespace(
        observation_indices=kwargs["default_observation_indices"],
        state_indices=kwargs["default_state_indices"],
        action_indices=kwargs["default_action_indices"],
        language_indices=[0],
    )
    monkeypatch.setitem(sys.modules, temporal_clip_module.__name__, temporal_clip_module)

    path = REPO_ROOT / "starVLA" / "dataloader" / "lerobot_datasets.py"
    spec = importlib.util.spec_from_file_location("_lerobot_datasets_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_generic_robot_type_reuses_rl_games_modality_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_config = _load_rl_games_data_config(monkeypatch)

    generic = data_config.ROBOT_TYPE_CONFIG_MAP[GENERIC_ROBOT_TYPE]

    assert isinstance(generic, data_config.GymnasiumDataConfig)
    assert isinstance(generic, data_config.FlappyDataConfig)
    assert data_config.ROBOT_TYPE_TO_EMBODIMENT_TAG[GENERIC_ROBOT_TYPE] == "new"
    assert generic.video_keys == ["video.image"]
    assert generic.state_keys == ["state.game_state"]
    assert generic.action_keys == ["action.button"]

    native = data_config.ROBOT_TYPE_CONFIG_MAP["rl_games_gymnasium_native"]
    assert isinstance(native, data_config.GymnasiumNativeDataConfig)
    assert native.state_keys == ["state.native"]
    assert native.action_keys == ["action.native"]


@pytest.mark.parametrize("task_name", ["custom_balance", "flappy"])
@pytest.mark.parametrize(
    "robot_type",
    [GENERIC_ROBOT_TYPE, "rl_games_gymnasium_native"],
)
def test_generic_dataset_uses_manifest_task_and_active_action_dimension(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    task_name: str,
    robot_type: str,
) -> None:
    lerobot_datasets = _load_lerobot_datasets(monkeypatch)
    dataset_path = tmp_path / "custom_balance_train"
    task_contract = _gymnasium_task_contract(task_name)
    _write_manifest(dataset_path, task_contract=task_contract)

    dataset = lerobot_datasets.make_LeRobotSingleDataset(
        data_root_dir=tmp_path,
        data_name=dataset_path.name,
        robot_type=robot_type,
        data_cfg=_data_cfg(task_contract),
    )

    assert dataset.init_kwargs["dataset_path"] == dataset_path
    assert dataset.rl_games_task == task_name
    assert dataset.rl_games_action_env_dim == 3
    assert dataset.rl_games_gymnasium_task_contract == task_contract


@pytest.mark.parametrize(
    ("robot_type", "expected"),
    [
        ("rl_games_flappy", ("flappy", 2)),
        ("rl_games_demon_attack", ("demon_attack", 6)),
        ("rl_games_deadly_corridor", ("deadly_corridor", 7)),
    ],
)
def test_existing_rl_games_tasks_keep_static_metadata_without_manifests(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    robot_type: str,
    expected: tuple[str, int],
) -> None:
    lerobot_datasets = _load_lerobot_datasets(monkeypatch)

    dataset = lerobot_datasets.make_LeRobotSingleDataset(
        data_root_dir=tmp_path,
        data_name="dataset_without_manifest",
        robot_type=robot_type,
    )

    assert (dataset.rl_games_task, dataset.rl_games_action_env_dim) == expected


@pytest.mark.parametrize(
    "required_key",
    [*GYMNASIUM_TASK_CONTRACT_KEYS, "active_action_dim"],
)
def test_generic_dataset_requires_task_metadata_from_manifest(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    required_key: str,
) -> None:
    lerobot_datasets = _load_lerobot_datasets(monkeypatch)
    dataset_path = tmp_path / "generic_train"
    task_contract = _gymnasium_task_contract()
    manifest = {
        "integration_name": "gymnasium",
        "task_name": "custom_balance",
        "gymnasium_task": task_contract,
        "robot_type": GENERIC_ROBOT_TYPE,
        "active_action_dim": 3,
        "action_dim": 3,
        "action_carrier": "native",
    }
    if required_key == "active_action_dim":
        del manifest[required_key]
    else:
        del manifest["gymnasium_task"][required_key]
    dataset_path.mkdir()
    (dataset_path / "manifest.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )

    with pytest.raises(KeyError, match=required_key):
        lerobot_datasets.make_LeRobotSingleDataset(
            data_root_dir=tmp_path,
            data_name=dataset_path.name,
            robot_type=GENERIC_ROBOT_TYPE,
            data_cfg=_data_cfg(_gymnasium_task_contract()),
        )


@pytest.mark.parametrize(
    ("field", "configured_value"),
    [
        ("task_name", "other_task"),
        ("env_id", "OtherBalance-v0"),
        ("make_kwargs", {"render_mode": "human"}),
        ("registration_imports", ["other.envs"]),
        ("action_labels", ["negative", "zero", "positive"]),
        ("action_values", [-1, 0, 1]),
        ("noop_action_id", 0),
        ("base_prompt", "Use another task prompt."),
        ("env_fps", 60),
        ("obs_fps", 30),
        ("frame_stack", 2),
    ],
)
def test_generic_dataset_rejects_each_configured_contract_mismatch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    field: str,
    configured_value,
) -> None:
    lerobot_datasets = _load_lerobot_datasets(monkeypatch)
    dataset_path = tmp_path / "generic_train"
    manifest_contract = _gymnasium_task_contract()
    _write_manifest(dataset_path, task_contract=manifest_contract)
    configured_contract = _gymnasium_task_contract()
    configured_contract[field] = configured_value

    with pytest.raises(ValueError, match=field):
        lerobot_datasets.make_LeRobotSingleDataset(
            data_root_dir=tmp_path,
            data_name=dataset_path.name,
            robot_type=GENERIC_ROBOT_TYPE,
            data_cfg=_data_cfg(configured_contract),
        )


def test_generic_dataset_rejects_configured_active_action_dim_mismatch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    lerobot_datasets = _load_lerobot_datasets(monkeypatch)
    dataset_path = tmp_path / "generic_train"
    task_contract = _gymnasium_task_contract()
    _write_manifest(dataset_path, task_contract=task_contract)

    with pytest.raises(ValueError, match="active_action_dim"):
        lerobot_datasets.make_LeRobotSingleDataset(
            data_root_dir=tmp_path,
            data_name=dataset_path.name,
            robot_type=GENERIC_ROBOT_TYPE,
            data_cfg=_data_cfg(task_contract, active_action_dim=4),
        )


def test_train_and_validation_each_check_their_own_manifest(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    lerobot_datasets = _load_lerobot_datasets(monkeypatch)
    task_contract = _gymnasium_task_contract()
    train_name = "generic_train"
    val_name = f"{train_name}__val"
    _write_manifest(tmp_path / train_name, task_contract=task_contract)
    val_contract = _gymnasium_task_contract()
    val_contract["base_prompt"] = "A mismatched validation prompt."
    _write_manifest(tmp_path / val_name, task_contract=val_contract)
    lerobot_datasets.load_custom_mixtures = lambda path: None
    lerobot_datasets.get_dataset_named_mixture = lambda name: [
        (name, 1.0, GENERIC_ROBOT_TYPE)
    ]

    train_cfg = _data_cfg(task_contract)
    train_cfg.update(data_root_dir=tmp_path, data_mix=train_name)
    lerobot_datasets.get_vla_dataset(train_cfg)

    val_cfg = _data_cfg(task_contract)
    val_cfg.update(data_root_dir=tmp_path, data_mix=val_name)
    with pytest.raises(ValueError, match="base_prompt"):
        lerobot_datasets.get_vla_dataset(val_cfg, mode="eval")
