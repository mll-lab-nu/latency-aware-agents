"""Successful native H1 packaging contracts owned by the MIKASA task scripts."""

import json
import sys
from types import ModuleType

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def test_package_only_preparation_preserves_collected_run_configs(tmp_path):
    """Long-lived H1 resume contract: packaging preserves the collection configuration."""
    from helpers import REPO_ROOT
    from scripts.tasks.mikasa_intercept_grab_fast.successful_h1_dataset import main

    args = ["--config", str(REPO_ROOT / "configs/examples/mikasa_intercept_grab_fast/collect_successful_h1.yaml"),
            "--stage", "prepare", "--checkpoint", str(tmp_path / "teacher.pt"),
            "--latency-frames", "2", "--output", str(tmp_path / "run"), "--repo-id", "org/data"]
    main(args)
    paths = [tmp_path / "run" / name for name in ("collect.yaml", "latency.yaml")]
    original = [path.read_bytes() for path in paths]
    main(args + ["--package-only"])
    assert [path.read_bytes() for path in paths] == original


def test_direct_h1_packaging_preserves_successful_episodes_cameras_and_labels(tmp_path, monkeypatch):
    """Long-lived H1 packaging contract: successful issued labels and training statistics."""
    from latency_bench.integrations.starvla.mikasa_h1_export import prepare

    datasets = []

    class FakeLeRobotDataset:
        @classmethod
        def create(cls, **kwargs):
            dataset = cls()
            dataset.root = kwargs["root"]
            dataset.features = kwargs["features"]
            dataset.episodes = []
            dataset.current = []
            datasets.append(dataset)
            return dataset

        def add_frame(self, values):
            self.current.append({key: value.copy() if isinstance(value, np.ndarray) else value
                                 for key, value in values.items()})

        def save_episode(self, *, parallel_encoding):
            self.episodes.append(self.current)
            self.current = []

        def finalize(self):
            (self.root / "data/chunk-000").mkdir(parents=True)
            (self.root / "meta").mkdir()
            rows = [
                {"observation.state": row["observation.state"].tolist(), "action": row["action"].tolist()}
                for episode in self.episodes for row in episode
            ]
            pq.write_table(pa.Table.from_pylist(rows), self.root / "data/chunk-000/file-000.parquet")
            (self.root / "meta/tasks.jsonl").write_text(json.dumps({"task": "Intercept and grasp the ball."}) + "\n")
            (self.root / "meta/info.json").write_text(json.dumps({"features": self.features}))

    module = ModuleType("lerobot.datasets.lerobot_dataset")
    module.LeRobotDataset = FakeLeRobotDataset
    monkeypatch.setitem(sys.modules, "lerobot", ModuleType("lerobot"))
    monkeypatch.setitem(sys.modules, "lerobot.datasets", ModuleType("lerobot.datasets"))
    monkeypatch.setitem(sys.modules, module.__name__, module)

    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "metadata.json").write_text(json.dumps({"success_once": [True, False, True, True]}))
    for episode_index, length in enumerate([1, 2, 11, 1]):
        rgb = np.empty((length, 128, 128, 6), dtype=np.uint8)
        rgb[..., :3] = 10 + episode_index
        rgb[..., 3:] = 20 + episode_index
        np.savez(raw / f"train_data_{episode_index}.npz",
                 rgb=rgb, proprio=np.full((length, 7), episode_index, dtype=np.float32),
                 action=np.full((length, 7), episode_index / 10, dtype=np.float32),
                 applied_command=np.full((length, 7), -1, dtype=np.float32),
                 reward=np.full(length, 9999 if episode_index == 1 else -1, dtype=np.float32),
                 language_instruction=np.array("Intercept and grasp the ball."))

    report = prepare(raw, tmp_path / "dataset/lerobot", train_episodes=2, val_episodes=1)
    assert report["train_episode_indices"] == [0, 2]
    assert report["validation_episode_indices"] == [3]
    assert report["counts"] == {"train": {"episodes": 2, "frames": 12},
                                "validation": {"episodes": 1, "frames": 1}}
    assert [len(episode) for episode in datasets[0].episodes] == [1, 11]
    for dataset, indices in zip(datasets, ([0, 2], [3])):
        for episode, index in zip(dataset.episodes, indices):
            for row in episode:
                np.testing.assert_array_equal(row["observation.images.top"], 10 + index)
                np.testing.assert_array_equal(row["observation.images.wrist"], 20 + index)
                np.testing.assert_array_equal(row["observation.state"], index)
                np.testing.assert_allclose(row["action"], index / 10)
                assert row["task"] == "Intercept and grasp the ball."
    train = json.loads((tmp_path / "dataset/lerobot/task_contract.json").read_text())
    val = json.loads((tmp_path / "dataset/lerobot__val/task_contract.json").read_text())
    assert train["action_horizon"] == 1
    assert train["state_dim"] == train["action_dim"] == 7
    assert val["normalization"] == train["normalization"]
    assert train["prompt"] == val["prompt"] == "Intercept and grasp the ball."
    np.testing.assert_allclose(train["normalization"]["state"]["min"], [0.0] * 7)
    np.testing.assert_allclose(train["normalization"]["state"]["max"], [2.0] * 7)
    for split in ("lerobot", "lerobot__val"):
        directory = tmp_path / "dataset" / split
        modality = json.loads((directory / "meta/modality.json").read_text())
        assert modality["action"]["target"]["original_key"] == "action"
        assert modality["state"]["proprio"]["end"] == 7
        assert modality["video"]["top"]["original_key"] == "observation.images.top"
        assert modality["video"]["wrist"]["original_key"] == "observation.images.wrist"
        manifest = json.loads((directory / "manifest.json").read_text())
        assert manifest["robot_type"] == "latency_mikasa_h1"
        assert manifest["action_horizon"] == 1
        assert manifest["normalization"] == train["normalization"]
        stats = json.loads((directory / "meta/stats_gr00t.json").read_text())
        assert stats["statistics"]["observation.state"] == train["normalization"]["state"]
