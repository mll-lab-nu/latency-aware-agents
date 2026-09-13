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
    pq.write_table(
        pa.table(
            {
                "observation.state": [[0.0] * 198, [1.0] * 198],
                "action": [[-1.0] * 43, [1.0] * 43],
            }
        ),
        dataset / "data/chunk-000/episode_000000.parquet",
    )

    path = write_task_contract(dataset, spec=task_spec("balance"))
    contract = json.loads(path.read_text())
    assert contract["state_dim"] == 198
    assert contract["action_dim"] == 43
    assert contract["action_horizon"] == 1
    assert "action_target_key" not in contract
    assert "image_columns" not in contract

    action = np.asarray([[-1.0] * 43, [1.0] * 43])
    normalized = normalize(action, contract["normalization"]["action"])
    np.testing.assert_allclose(denormalize(normalized, contract["normalization"]["action"]), action)
    stats = json.loads((dataset / "meta/stats_gr00t.json").read_text())
    assert stats["statistics"]["action"]["min"] == [-1.0] * 43


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
    transform = LatencyBenchH1DataConfig({"robot_type": "latency_haic_h1", "video_keys": []}).transform()
    transform.set_metadata(metadata)
    for training in (True, False):
        transform.train() if training else transform.eval()
        output = transform({"state.proprio": values.copy(),
                            "action.target": np.full((len(values), 1), 3.0, dtype=np.float32)})
        np.testing.assert_allclose(output["state.proprio"].numpy(), expected)
        np.testing.assert_allclose(output["action.target"].numpy(), 3.0)


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
