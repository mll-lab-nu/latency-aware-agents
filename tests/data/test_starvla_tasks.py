from __future__ import annotations

import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest


def test_air_raid_gr00t_configs_keep_appo_h1_and_realtime_modes(tmp_path):
    from scripts.gym_adapt.gr00t import ROOT, evaluation_config, teacher_config, vla_config

    source = ROOT / "configs/examples/gymnasium/air_raid/small_model_train.yaml"
    teacher = teacher_config("air_raid", tmp_path, None, source)
    assert teacher["experiment"]["name"] == "air_raid_zero_latency_appo_h1"
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    (dataset / "manifest.json").write_text(json.dumps({
        "robot_type": "rl_games_gymnasium", "action_dim": 6, "active_action_dim": 6,
        "uses_state": False, "state_dim": 6, "gymnasium_task": teacher["env"],
    }))
    config = vla_config("air_raid", dataset, tmp_path / "backbone", tmp_path / "vla",
                        num_gpus=1, micro_batch=8, condition="zero_latency", wandb_entity="test")
    assert config["framework"]["action_model"]["action_horizon"] == 1
    assert config["datasets"]["vla_data"]["action_type"] == "discrete"
    assert config["datasets"]["vla_data"]["include_state"] is False
    assert config["trainer"]["max_train_steps"] == 5000
    for mode in ("zero", "iid", "measured"):
        evaluation = evaluation_config("air_raid", tmp_path, tmp_path / "backbone", tmp_path,
            mode=mode, profile=tmp_path / "profile.json", episodes=20, profiling=False, prompt_key=0)
        assert evaluation["latency"]["method"] == mode
        assert evaluation["executor"]["mode"] == ("realtime" if mode == "measured" else "simulated")
        assert evaluation["evaluation"]["eval_max_steps"] == 3600
        assert evaluation["policy"]["task_manifest_path"] == str(tmp_path / "manifest.json")


@pytest.mark.parametrize("task,state_dim,action_dim", [
    ("inverted_pendulum", 4, 5), ("ant", 27, 8), ("half_cheetah", 17, 6),
    ("hopper", 11, 3), ("humanoid", 376, 17), ("walker2d", 17, 6),
])
def test_appo_gr00t_h1_manifest_and_four_gpu_config(tmp_path, task, state_dim, action_dim):
    from scripts.gym_adapt.gr00t import teacher_config, vla_config, evaluation_config
    from latency_bench.envs.tasks.contracts import gymnasium_task_contract
    teacher = teacher_config(task, tmp_path, None)
    contract = gymnasium_task_contract(teacher["env"])
    assert contract["env_fps"] == contract["obs_fps"] == 5
    assert teacher["backend"]["algo"] == "APPO"
    assert len(contract["state_space"]["labels"]) == state_dim
    if task == "inverted_pendulum":
        assert contract["env_id"] == "LatencyBench/InvertedPendulumDiscrete-v0"
        assert len(contract["action_labels"]) == 5
    else:
        assert len(contract["action_space"]["labels"]) == action_dim
    manifest = {"robot_type": "rl_games_gymnasium", "action_dim": action_dim,
                "active_action_dim": action_dim, "uses_state": True,
                "state_dim": state_dim, "gymnasium_task": contract}
    if task == "hopper":
        manifest["gymnasium_task"] = {
            **contract,
            "task_name": "hopper_rgb_state",
            "env_id": "LatencyBench/HopperRgbState-v0",
            "registration_imports": ["latency_bench.envs.gymnasium_hopper_rgb_state"],
        }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    config = vla_config(task, tmp_path, tmp_path, tmp_path / "out",
                       num_gpus=4, micro_batch=16, condition="zero_latency", wandb_entity="test")
    model = config["framework"]["action_model"]
    assert (model["state_dim"], model["action_dim"], model["action_horizon"]) == (state_dim, action_dim, 1)
    assert model["future_action_window_size"] == 0
    assert config["trainer"]["gradient_accumulation_steps"] == 1
    assert config["trainer"]["pretrained_checkpoint"] is None
    assert config["trainer"]["max_train_steps"] == (2000 if task == "inverted_pendulum" else 5000)
    assert config["datasets"]["vla_data"]["action_type"] == ("discrete" if task == "inverted_pendulum" else "continuous")
    for profiling in (False, True):
        evaluation = evaluation_config(task, tmp_path, tmp_path, tmp_path / "eval",
            mode="zero", profile=None, episodes=20, profiling=profiling, prompt_key=0)
        assert gymnasium_task_contract(evaluation["env"]) == contract
        assert evaluation["evaluation"]["eval_max_steps"] == (900 if profiling else 1000)
    if task == "inverted_pendulum":
        import yaml
        from scripts.gym_adapt.gr00t import export_bundle
        trained = tmp_path / "trained"
        trained.mkdir()
        (trained / "config.full.yaml").write_text(yaml.safe_dump(config))
        (trained / "dataset_statistics.json").write_text("{}")
        checkpoint = trained / "steps_2000_pytorch_model.pt"
        checkpoint.write_bytes(b"test weights")
        (tmp_path / "latency_prompt_map.json").write_text("{}")
        source = tmp_path / "source.json"
        source.write_text("{}")
        bundle = tmp_path / "bundle"
        export_bundle(task, trained, checkpoint, tmp_path, bundle, source_manifest=source)
        inference = yaml.safe_load((bundle / "config.yaml").read_text())
        assert inference["datasets"]["vla_data"]["include_state"] is True
        assert inference["datasets"]["vla_data"]["gymnasium_task_contract"] == contract
        assert inference["framework"]["action_model"]["state_dim"] == 4
        assert inference["framework"]["action_model"]["action_dim"] == 5
        assert json.loads((bundle / "manifest.json").read_text()) == manifest


def test_starvla_task_contract_writes_stats_and_inverts_min_max(tmp_path):
    from latency_bench.data.starvla_tasks import (
        denormalize,
        normalize,
        task_spec,
        write_task_contract,
    )

    dataset = tmp_path / "dataset"
    (dataset / "data/chunk-000").mkdir(parents=True)
    (dataset / "meta").mkdir()
    (dataset / "meta/info.json").write_text(json.dumps({"features": {
        "observation.state": {"names": None}, "issued_command_chunk": {"names": None}},
        "label_provenance": {"semantics": "expert_closed_loop_selected_absolute_requests_v2"}}))
    pq.write_table(
        pa.table(
            {
                "observation.state": [[0.0] * 198, [1.0] * 198],
                "action": [[-1.0] * 43, [1.0] * 43],
                "issued_command_chunk": [[[-1.0] * 43] + [[-100.0] * 43 for _ in range(7)], [[1.0] * 43 for _ in range(8)]],
                "action_loss_mask": [[True] + [False] * 7, [True] * 8],
            }
        ),
        dataset / "data/chunk-000/episode_000000.parquet",
    )

    path = write_task_contract(dataset, spec=task_spec("balance"))
    contract = json.loads(path.read_text())
    assert contract["format"] == "latency_bench_starvla_h8_v2"
    assert contract["auxiliary"]["action_loss_mask"]["shape"] == [8]
    assert contract["state_dim"] == 198
    assert contract["action_dim"] == 43
    assert contract["action_horizon"] == 8
    assert contract["action_key"] == "issued_command_chunk"
    assert "action_target_key" not in contract
    assert "image_columns" not in contract

    action = np.asarray([[-1.0] * 43, [1.0] * 43])
    normalized = normalize(action, contract["normalization"]["action"])
    np.testing.assert_allclose(denormalize(normalized, contract["normalization"]["action"]), action)
    stats = json.loads((dataset / "meta/stats_gr00t.json").read_text())
    assert stats["statistics"]["issued_command_chunk"]["min"] == [-1.0] * 43
    table = pq.read_table(dataset / "data/chunk-000/episode_000000.parquet").drop(["action_loss_mask"])
    pq.write_table(table, dataset / "data/chunk-000/episode_000000.parquet")
    with pytest.raises(KeyError, match="action_loss_mask"):
        write_task_contract(dataset, spec=task_spec("balance"))


def test_profile_task_contract_uses_prefix_and_loss_masks():
    from latency_bench.data.starvla_tasks import task_spec

    for name, horizon in (("balance_profile_h8", 8), ("balance_profile_h16", 16)):
        auxiliary = task_spec(name).auxiliary_shape_map()
        assert auxiliary == {
            "action_prefix_mask": (horizon,),
            "action_loss_mask": (horizon,),
        }


def test_normalize_matches_min_max_boundaries_and_constant_dimensions(monkeypatch):
    from latency_bench.data.starvla_tasks import normalize

    values = np.asarray([
        [-1.0, 2.0, 0.17784182727336884],
        [0.0, 2.0, 0.17787],
        [1.0, 2.0, 0.17788860201835632],
        [-4.0, 2.0, -1.4565205574035645],
        [4.0, 2.0, 10.0],
    ], dtype=np.float32)
    stats = {"min": [-1.0, 2.0, 0.17784182727336884],
             "max": [1.0, 2.0, 0.17788860201835632]}
    expected = normalize(values, stats)
    np.testing.assert_allclose(expected[[0, 2, 3, 4]], [
        [-1.0, 0.0, -1.0], [1.0, 0.0, 1.0],
        [-1.0, 0.0, -1.0], [1.0, 0.0, 1.0],
    ])

    pytest.importorskip("accelerate")
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "third_party/starVLA"))
    from starVLA.dataloader.latency_bench_tasks import LatencyBenchH1DataConfig
    from starVLA.dataloader.gr00t_lerobot.schema import DatasetMetadata
    from starVLA.dataloader.gr00t_lerobot.embodiment_tags import EmbodimentTag

    def metadata_stats(bounds):
        return {**bounds, "mean": np.zeros(len(bounds["min"])),
                "std": np.ones(len(bounds["min"])),
                "q01": bounds["min"], "q99": bounds["max"]}

    metadata = DatasetMetadata(
        statistics={"state": {"proprio": metadata_stats(stats)},
                    "action": {"target": metadata_stats({"min": [-1.0], "max": [1.0]})}},
        modalities={"video": {},
                    "state": {"proprio": {"absolute": True, "shape": (3,), "continuous": True}},
                    "action": {"target": {"absolute": True, "shape": (1,), "continuous": True}}},
        embodiment_tag=EmbodimentTag.NEW_EMBODIMENT,
    )
    transform = LatencyBenchH1DataConfig({"robot_type": "latency_mikasa_h1", "video_keys": []}).transform()
    transform.set_metadata(metadata)
    for training in (True, False):
        transform.train() if training else transform.eval()
        output = transform({"state.proprio": values.copy(),
                            "action.target": np.full((len(values), 1), 3.0, dtype=np.float32)})
        np.testing.assert_allclose(output["state.proprio"].numpy(), expected)
        np.testing.assert_allclose(output["action.target"].numpy(), 3.0)


def test_dagger_mixture_refresh_opens_only_new_sources_and_updates_weights(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    pytest.importorskip("accelerate")
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "third_party/starVLA"))
    from starVLA.dataloader import latency_bench_tasks

    class Source:
        def __init__(self, name):
            self.name = name
            self.tag = "robot"
            self.trajectory_lengths = np.asarray([8])
            self.transform_metadata = None

        def __len__(self):
            return 8

        def set_transforms_metadata(self, metadata):
            self.transform_metadata = metadata

    opened = []

    def open_source(_root, name, _robot_type, **_kwargs):
        opened.append(name)
        return Source(name)

    monkeypatch.setattr(latency_bench_tasks, "make_LeRobotSingleDataset", open_source)
    mixture_path = tmp_path / "mixture.json"
    mixture_path.write_text(json.dumps({"train": [["bootstrap", 1.0, "robot"]]}))
    dataset = object.__new__(latency_bench_tasks._RefreshableLatencyBenchMixture)
    dataset._refresh_data_cfg = {"data_root_dir": "/", "data_mix": "train",
                                 "custom_mixtures_path": str(mixture_path)}
    dataset._refresh_data_root = Path("/")
    dataset._refresh_mixture_name = "train"
    dataset._refresh_mixture_path = mixture_path
    dataset._refresh_contract = {"robot_type": "robot", "video_keys": []}
    dataset._refresh_source_cache = {}
    dataset._shared_mixture_generation = torch.zeros(
        (), dtype=torch.int64, device="cpu"
    ).share_memory_()
    dataset._local_mixture_generation = 0
    dataset.balance_dataset_weights = False
    dataset.balance_trajectory_weights = False
    dataset.epoch = 0
    dataset._sequential_step_sampling = False
    dataset._replace_data_mixture(dataset._materialize_current_mixture(None))
    dataset.merged_metadata = {"robot": "frozen-bootstrap-metadata"}
    bootstrap_source = dataset.datasets[0]

    mixture_path.write_text(json.dumps({"train": [
        ["bootstrap", 1.0, "robot"], ["delta_1", 1.0, "robot"],
    ]}))
    latency_bench_tasks.ROBOT_TYPE_CONFIG_MAP.pop("robot", None)
    dataset.refresh_custom_mixture()
    assert dataset._shared_mixture_generation.item() == 1
    assert "robot" in latency_bench_tasks.ROBOT_TYPE_CONFIG_MAP
    delta_1_source = dataset.datasets[1]
    mixture_path.write_text(json.dumps({"train": [
        ["bootstrap", 1.0, "robot"], ["delta_1", 0.5, "robot"],
        ["delta_2", 1.0, "robot"],
    ]}))
    dataset.refresh_custom_mixture()
    assert dataset._shared_mixture_generation.item() == 2

    assert opened == ["bootstrap", "delta_1", "delta_2"]
    assert dataset.datasets[:2] == [bootstrap_source, delta_1_source]
    assert dataset.datasets[2].transform_metadata == "frozen-bootstrap-metadata"
    np.testing.assert_allclose(dataset.dataset_sampling_weights, [0.4, 0.2, 0.4])


def test_lerobot_sample_preserves_finite_state_outside_float16_range(monkeypatch):
    pytest.importorskip("accelerate")
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "third_party/starVLA"))
    from starVLA.dataloader.gr00t_lerobot.datasets import LeRobotSingleDataset

    dataset = object.__new__(LeRobotSingleDataset)
    dataset.data_cfg = {"include_state": True}
    dataset._modality_keys = {"video": [], "state": ["state.proprio"],
                              "action": ["action.target"], "language": ["annotation.task"]}
    dataset.tag = "test"
    state = np.asarray([[-69883.2578125]], dtype=np.float32)
    sample = dataset._pack_sample({"state.proprio": state,
                                  "action.target": np.asarray([[0.25]], dtype=np.float32),
                                  "annotation.task": ["test"]})
    np.testing.assert_array_equal(sample["state"], state)
    assert sample["state"].dtype == np.float32
    np.testing.assert_array_equal(sample["action"], [[0.25]])
    assert sample["image"] == []
    assert sample["lang"] == "test"


@pytest.mark.parametrize("modality,nested", [("state", False), ("action", False), ("action", True)])
@pytest.mark.parametrize("absolute", [False, True])
def test_lerobot_state_action_preserves_vectors_chunks_and_padding(monkeypatch, modality, nested, absolute):
    pytest.importorskip("accelerate")
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "third_party/starVLA"))
    from starVLA.dataloader.gr00t_lerobot.datasets import LeRobotSingleDataset

    shape = (3, 8, 17) if nested else (3, 376 if modality == "state" else 17)
    values = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    rows = []
    for value in values:
        if nested:
            row = np.empty(8, dtype=object)
            row[:] = list(value)
            rows.append(row)
        else:
            rows.append(value)
    dataset = object.__new__(LeRobotSingleDataset)
    dataset._delta_indices = {f"{modality}.target": np.array([-1, 0, 3])}
    dataset._trajectory_lengths = np.array([3])
    dataset.get_trajectory_index = lambda _trajectory_id: 0
    dataset.curr_traj_data = pd.DataFrame({"values": rows})
    dataset._lerobot_modality_meta = SimpleNamespace(**{
        modality: {"target": SimpleNamespace(original_key="values", start=1, end=4)}})
    dataset._metadata = SimpleNamespace(modalities=SimpleNamespace(**{
        modality: {"target": SimpleNamespace(absolute=absolute)}}))

    actual = dataset.get_state_or_action(0, modality, f"{modality}.target", 0)
    expected = values[[0, 0, 2], ..., 1:4].copy()
    if not absolute:
        expected[[0, 2]] = 0
    np.testing.assert_array_equal(actual, expected)
    assert actual.dtype == np.float64  # Existing retrieve_data_and_pad output dtype.


def test_lerobot_video_reference_uses_video_decoder(tmp_path, monkeypatch):
    """A LeRobot video reference dict must follow the timestamp video path."""
    pytest.importorskip("accelerate")
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "third_party/starVLA"))
    from starVLA.dataloader.gr00t_lerobot import datasets

    dataset = object.__new__(datasets.LeRobotSingleDataset)
    dataset._delta_indices = {"video.camera": np.array([0])}
    dataset._trajectory_lengths = np.array([1])
    dataset.get_trajectory_index = lambda _trajectory_id: 0
    dataset._lerobot_modality_meta = SimpleNamespace(
        video={"camera": SimpleNamespace(original_key="observation.image")}
    )
    dataset._lerobot_info_meta = {
        "features": {"observation.image": {"dtype": "video"}}
    }
    dataset.data_cfg = None
    dataset.curr_traj_data = pd.DataFrame(
        {"observation.image": [{"path": "frames/frame-000000.png"}], "timestamp": [0.0]}
    )
    dataset._dataset_path = tmp_path
    dataset._lerobot_version = "v2.0"
    dataset.video_backend = "pyav"
    dataset.video_backend_kwargs = {}
    dataset.get_video_path = lambda _trajectory_id, _key: tmp_path / "camera.mp4"
    expected = np.zeros((1, 2, 2, 3), dtype=np.uint8)

    def decode_video(*args, **kwargs):
        assert args[0].endswith("camera.mp4")
        np.testing.assert_array_equal(args[1], [0.0])
        return expected

    monkeypatch.setattr(datasets, "get_frames_by_timestamps", decode_video)
    frames = dataset.get_video(0, "video.camera", 0)

    np.testing.assert_array_equal(frames, expected)


def test_balance_masked_l1_uses_equal_sample_weights_and_no_padding_gradient(monkeypatch):
    """Long-lived Balance training contract: masks govern loss and gradient."""
    pytest.importorskip("accelerate")
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "third_party/starVLA"))
    import torch
    from starVLA.model.framework.VLM4A.QwenOFT import Qwenvl_OFT

    model = object.__new__(Qwenvl_OFT)
    model.action_env_dim = 43
    pred = torch.zeros((2, 8, 43), requires_grad=True)
    target = torch.ones_like(pred)
    target[0, 1:] = 100
    target[1] = 3
    mask = torch.tensor([[True] + [False] * 7, [True] * 8])
    loss = model._compute_action_loss(pred, target, action_loss_mask=mask)
    assert loss.item() == 2
    loss.backward()
    assert torch.count_nonzero(pred.grad[0, 1:]).item() == 0
    assert pred.grad[0, 0].abs().sum().item() == pytest.approx(0.5)
    assert pred.grad[1].abs().sum().item() == pytest.approx(0.5)
    all_valid = torch.ones((2, 8), dtype=torch.bool)
    torch.testing.assert_close(model._compute_action_loss(pred, target, action_loss_mask=all_valid),
                               (pred - target).abs().mean())
