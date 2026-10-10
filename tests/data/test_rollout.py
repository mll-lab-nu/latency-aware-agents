"""Long-lived data contracts: source-time teacher labels, canonical handoff and failure recovery."""
import io
import json

import gymnasium as gym
import numpy as np
import pyarrow.parquet as pq
import pytest
from PIL import Image

from latency_bench.core.types import Action, PolicyOutput
from latency_bench.data.rollout import collect_rollouts
from latency_bench.integrations.starvla.lerobot_export import convert_canonical_rollout_to_lerobot
from latency_bench.envs.adapters.gymnasium import GymnasiumEnvAdapter
from latency_bench.policy.base import PolicyRunner


class CounterEnv(gym.Env):
    """Reuse state and render buffers while recording the actions actually applied."""
    metadata = {"render_modes": ["rgb_array"]}
    observation_space = gym.spaces.Box(0, 10, (1,), dtype=np.float32)
    action_space = gym.spaces.Box(-10, 10, (1,), dtype=np.float32)
    instances = []

    def __init__(self, render_mode=None):
        self.state = np.zeros(1, dtype=np.float32)
        self.rgb = np.zeros((4, 4, 3), dtype=np.uint8)
        self.applied = []
        self.closed = False
        self.instances.append(self)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.state[:] = 0
        return self.state, {}

    def step(self, action):
        self.applied.append(float(action[0]))
        self.state += 1
        return self.state, 1.0, False, bool(self.state[0] == 4), {}

    def render(self):
        self.rgb[:] = int(self.state[0])
        return self.rgb

    def close(self):
        self.closed = True


class StateTeacher(PolicyRunner):
    """Issue a command determined solely by the source state."""
    def predict(self, observation):
        return PolicyOutput(Action(np.asarray(observation.data + 1, dtype=np.float32)))

    def predict_batch(self, observations):
        return [self.predict(observation) for observation in observations]


def make_teacher(config, action_resolver):
    return StateTeacher()


@pytest.fixture
def config(tmp_path):
    gym.envs.registry.pop("CollectionCounter-v0", None)
    gym.register("CollectionCounter-v0", entry_point=CounterEnv, disable_env_checker=True)
    CounterEnv.instances = []
    return {
        "experiment": {"name": "canonical", "seed": 7},
        "env": {"name": "gymnasium", "env_id": "CollectionCounter-v0", "task_name": "counter",
                "registration_imports": [], "make_kwargs": {"render_mode": "rgb_array"},
                "env_fps": 20, "obs_fps": 10, "frame_stack": 1, "noop_action": [0.0],
                "base_prompt": "Issue the next count.", "state_space": {"labels": ["count"]},
                "action_space": {"type": "box", "shape": [1], "dtype": "float32",
                                 "low": [-10], "high": [10], "labels": ["command"]}},
        "policy": {"type": "custom", "factory": "test_rollout:make_teacher"},
        "executor": {"mode": "simulated"},
        "latency": {"method": "fixed", "fixed_latency_ms": 150},
        "scheduler": {"hold_policy": "hold", "ordering_policy": "latest_ready"},
        "evaluation": {"eval_episodes": 2, "eval_max_steps": 8},
        "logging": {"output_dir": str(tmp_path / "dataset"), "save_step_records": False,
                    "save_action_records": False, "save_latency_records": False},
        "collection": {"student_observation": {"state_key": "gymnasium_state"}, "val_fraction": 0.5},
    }


@pytest.mark.parametrize("latency_ms,parallel", [(0, 1), (150, 1), (150, 2), ("profile", 1)])
def test_continuous_source_snapshots_and_labels_survive_conversion(config, tmp_path, latency_ms, parallel):
    if latency_ms == "profile":
        from latency_bench.latency.burst_profile import build_worker_slot_distributions_from_sessions
        samples = [{"sample_index": i, "worker_id": i % 2, "observation_to_action_latency_ms": 150.0,
                    "worker_service_time_ms": 150.0, "worker_service_time_ms_bin": 150} for i in range(4)]
        distribution = build_worker_slot_distributions_from_sessions([samples], spike_threshold_ms_by_worker_slot={"0": 200.0, "1": 200.0})
        (tmp_path / "distribution.json").write_text(json.dumps(distribution))
        (tmp_path / "profile.json").write_text(json.dumps({
            "distribution_path": "distribution.json", "model_id": "test", "gpu_class": "cpu",
            "workload_id": "counter", "instance_id": "test", "source_run_id": "test",
        }))
        config["latency"] = {"method": "iid", "profile_path": str(tmp_path / "profile.json"), "seed": 7}
    else:
        config["latency"]["fixed_latency_ms"] = latency_ms
    config["evaluation"]["eval_parallel_envs"] = parallel
    result = collect_rollouts(config)
    output = tmp_path / "dataset"
    episodes = []
    for split in ("train", "val"):
        rows = pq.read_table(output / f"{split}.parquet").to_pylist()
        assert [row["action"] for row in rows] == [[1.0], [3.0]]
        assert [row["state"] for row in rows] == [[0.0], [2.0]]
        assert [row["raw_reward"] for row in rows] == [2.0, 2.0]
        for row in rows:
            image = np.asarray(Image.open(io.BytesIO(row["image"]["bytes"])))
            assert (image == row["state"][0]).all()
        episodes.append(rows[0]["episode_idx"])
    assert len(set(episodes)) == 2
    assert result["metadata"]["label_provenance"]["semantics"] == "teacher_issued_command"
    if latency_ms:
        assert all(env.applied[:4] == [0.0, 0.0, 0.0, 1.0] for env in CounterEnv.instances)
    assert all(env.closed for env in CounterEnv.instances)
    manifest = convert_canonical_rollout_to_lerobot(output, tmp_path / "lerobot")
    assert manifest["frames"] == 2
    converted = pq.read_table(next((tmp_path / "lerobot/data").rglob("*.parquet"))).to_pylist()
    assert [row["action"] for row in converted] == [[1.0], [3.0]]


def test_completed_episode_is_recoverable_after_image_write_failure(config, tmp_path, monkeypatch):
    from latency_bench.data import rollout_image_io
    from latency_bench.data.hf_parquet import completed_episode_indices_from_shards
    original = rollout_image_io.save_image_frame
    calls = 0

    def fail_second_episode(*args):
        nonlocal calls
        calls += 1
        if calls == 3:
            raise OSError("disk full")
        return original(*args)

    monkeypatch.setattr(rollout_image_io, "save_image_frame", fail_second_episode)
    with pytest.raises(OSError, match="disk full"):
        collect_rollouts(config)
    output = tmp_path / "dataset"
    assert completed_episode_indices_from_shards(output) == {0}
    assert not (output / "metadata.json").exists()
    assert all(env.closed for env in CounterEnv.instances)
    monkeypatch.setattr(rollout_image_io, "save_image_frame", original)
    saved_state = json.loads((output / "probe_state.json").read_text())
    legacy_state = {key: value for key, value in saved_state.items() if key != "label_semantics"}
    (output / "probe_state.json").write_text(json.dumps(legacy_state))
    with pytest.raises(ValueError, match="legacy labels"):
        collect_rollouts(config)
    (output / "probe_state.json").write_text(json.dumps(saved_state))
    result = collect_rollouts(config)
    assert result["metadata"]["total_episodes"] == 2
    with pytest.raises(FileExistsError, match="without resumable state"):
        collect_rollouts(config)


def test_realtime_collection_rejected_before_component_construction(config):
    config["executor"]["mode"] = "realtime"
    with pytest.raises(ValueError, match="paused or simulated"):
        collect_rollouts(config)
    assert not CounterEnv.instances


def test_dropped_observation_has_no_label_and_is_not_counted_as_row_filtering(config, tmp_path):
    """Long-lived collection contract: only actual teacher outputs count as candidate rows."""
    config["executor"]["simulated_worker_capacity"] = 1
    result = collect_rollouts(config)
    rows = pq.read_table(tmp_path / "dataset/train.parquet").to_pylist()
    assert [row["action"] for row in rows] == [[1.0]]
    report = result["metadata"]["filter_report"]
    assert report["train_rows_before_row_filter"] == report["train_rows_after_row_filter"] == 1
