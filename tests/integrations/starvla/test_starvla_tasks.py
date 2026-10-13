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

from helpers import REPO_ROOT


def test_starvla_task_contract_writes_stats_and_inverts_min_max(tmp_path):
    from latency_bench.integrations.starvla.task_contracts import (
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
    from latency_bench.integrations.starvla.task_contracts import task_spec

    for name, horizon in (("balance_profile_h8", 8), ("balance_profile_h16", 16)):
        auxiliary = task_spec(name).auxiliary_shape_map()
        assert auxiliary == {
            "action_prefix_mask": (horizon,),
            "action_loss_mask": (horizon,),
        }


def test_normalize_matches_min_max_boundaries_and_constant_dimensions(monkeypatch):
    from latency_bench.integrations.starvla.task_contracts import normalize

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
    monkeypatch.syspath_prepend(str(REPO_ROOT / "third_party/starVLA"))
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
    monkeypatch.syspath_prepend(str(REPO_ROOT / "third_party/starVLA"))
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
    monkeypatch.syspath_prepend(str(REPO_ROOT / "third_party/starVLA"))
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
    monkeypatch.syspath_prepend(str(REPO_ROOT / "third_party/starVLA"))
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
    monkeypatch.syspath_prepend(str(REPO_ROOT / "third_party/starVLA"))
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
    monkeypatch.syspath_prepend(str(REPO_ROOT / "third_party/starVLA"))
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


def test_saved_statistics_restore_field_order_across_embodiments(tmp_path, monkeypatch):
    """Long-lived statistics format: save/restore preserves each named field and its metadata."""
    pytest.importorskip("accelerate")
    monkeypatch.syspath_prepend(str(REPO_ROOT / "third_party/starVLA"))
    from starVLA.dataloader.gr00t_lerobot.datasets import LeRobotMixtureDataset
    from starVLA.dataloader.gr00t_lerobot.schema import DatasetMetadata

    def metadata(tag, fields):
        stats = {key: {name: [value] for name in ("min", "max", "mean", "std", "q01", "q99")}
                 for key, value in fields.items()}
        return DatasetMetadata.model_validate({
            "embodiment_tag": tag, "statistics": {"state": stats, "action": {"command": stats["x"]}},
            "modalities": {"video": {},
                          "state": {key: {"absolute": False, "shape": [1], "continuous": True} for key in fields},
                          "action": {"command": {"absolute": True, "shape": [1], "continuous": True}}},
        })

    class Source:
        def __init__(self, tag, keys):
            self.tag = tag
            self.modality_keys = {"action": ["action.command"], "state": [f"state.{key}" for key in keys]}
            self.trajectory_ids = [0]

        def __len__(self):
            return 1

        def set_transforms_metadata(self, value):
            self.transform_metadata = value

    dataset = LeRobotMixtureDataset.__new__(LeRobotMixtureDataset)
    dataset.datasets = [Source("gr1", ["x"]), Source("new_embodiment", ["y", "x"])]
    dataset.merged_metadata = {"gr1": metadata("gr1", {"x": 10.0}),
                               "new_embodiment": metadata("new_embodiment", {"y": 20.0, "x": 30.0})}
    expected = {tag: value.model_dump(mode="json") for tag, value in dataset.merged_metadata.items()}
    path = tmp_path / "statistics.json"
    dataset.save_dataset_statistics(path)
    dataset.apply_cached_statistics(json.loads(path.read_text()))
    for source in dataset.datasets:
        assert source.transform_metadata.model_dump(mode="json") == expected[source.tag]
