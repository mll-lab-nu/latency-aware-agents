import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml

from scripts.starvla import balance_dagger, collect


def test_collect_bootstrap_uses_smoke_budget_and_returns_canonical_paths(tmp_path, monkeypatch):
    teacher = tmp_path / "teacher.pt"
    teacher.write_bytes(b"teacher")
    captured = {}

    def fake_collect(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(collect, "_collect_parkour", fake_collect)

    train, val = collect.collect_bootstrap(
        "extreme_parkour_go2",
        teacher,
        tmp_path / "bootstrap",
        "task-python",
        smoke=True,
        device="cuda:3",
    )

    assert train == (tmp_path / "bootstrap/lerobot").resolve()
    assert val == (tmp_path / "bootstrap/lerobot__val").resolve()
    assert captured["train_episodes"] == 2
    assert captured["val_episodes"] == 1
    assert captured["keep_failed"] is True
    assert captured["device"] == "cuda:3"


@pytest.mark.parametrize("delay", [0, 2])
def test_balance_dagger_labels_teacher_branch_without_changing_student(tmp_path, delay):
    """Long-lived collector contract: future expert frames, inherited-request and tail masks."""
    import copy
    import gymnasium as gym
    from training.gymnasium_sf.integration import GymnasiumActionLatencyWrapper
    from latency_bench.data.dagger import DaggerState

    class Native(gym.Env):
        action_space = gym.spaces.Box(-10000, 10000, (43,), dtype=np.float32)
        observation_space = gym.spaces.Box(-10000, 10000, (198,), dtype=np.float32)

        def reset(self, *, seed):
            super().reset(seed=seed)
            self.value = 0.0
            self.terminate = False
            return self._policy_observation(), {}

        def _policy_observation(self):
            return np.full(198, self.value, dtype=np.float32)

        def g1_absolute_request_from_raw_action(self, action):
            return action

        def copy_balance_state_from(self, source):
            self.__dict__.update(copy.deepcopy(source.__dict__))

        def step(self, action):
            self.value = action[0]
            return self._policy_observation(), 1.0, self.terminate, False, {}

    class Teacher:
        def predict_batch(self, observations):
            return [SimpleNamespace(raw_output=np.stack([
                np.full(43, obs.data[0] + 1 + head * 100, dtype=np.float32)
                for head in range(8)])) for obs in observations]

    def env():
        return GymnasiumActionLatencyWrapper(
            Native(), fixed_latency_raw_frames=delay, env_fps=50, obs_fps=50, seed=0,
            simulated_worker_capacity=1, add_latency_info=True, noop_action=[0.0] * 43,
            hold_policy="hold", ordering_policy="latest_ready", hold_last_chunk_action=True)

    main, branch = env(), env()
    main.reset(seed=0)
    for _ in range(5):
        main.step(np.full((8, 43), -5, dtype=np.float32))
    value, frame = main.unwrapped.value, main.env_step_idx
    config = {"env": {"task_name": "g1_sonic_balance_simple", "env_fps": 50, "obs_fps": 50,
                      "base_prompt": "balance"}, "evaluation": {"eval_max_steps": 1000}}
    adapter = balance_dagger.BalanceDaggerAdapter(config, Teacher(), tmp_path, [main], [branch])
    state = DaggerState(5, [5], [0], [0], [0], [SimpleNamespace(action_chunk=np.full((8, 43), -5))])
    snapshot = {"images": {0: np.zeros((2, 2, 3), dtype=np.uint8)},
                "states": {0: main.unwrapped._policy_observation().copy()}}
    row = adapter.start_records(snapshot, [0], state)[0]
    mask = np.asarray(row["action_valid_mask"])
    chunk = np.asarray(row["issued_command_chunk"])
    assert main.unwrapped.value == value
    assert main.env_step_idx == frame
    if delay == 0:
        np.testing.assert_array_equal(mask, np.ones(8, dtype=bool))
        np.testing.assert_array_equal(chunk[:, 0], np.arange(-4, 4))
    else:
        np.testing.assert_array_equal(mask, [False, False, False, True, True, True, True, True])
        np.testing.assert_array_equal(chunk[:, 0], [0, 0, 0, 196, 296, 196, 296, 497])
    adapter.step(snapshot, [0], state)
    assert main.unwrapped.value == -5
    assert main.env_step_idx == frame + 1
    main.unwrapped.terminate = True
    config["evaluation"]["eval_max_steps"] = 7
    state.episode_steps[0] = 6
    row = adapter.start_records(snapshot, [0], state)[0]
    if delay == 0:
        assert row["action_valid_mask"] == [True] + [False] * 7
        np.testing.assert_array_equal(np.asarray(row["issued_command_chunk"])[1:], 0)
    else:
        assert row is None
    main.close(); branch.close()


def test_balance_dagger_relabels_saved_absolute_actions_without_loading_models(tmp_path, monkeypatch):
    source = tmp_path / "source"
    output = tmp_path / "output"
    source.mkdir()
    state = np.linspace(-0.5, 0.5, 198)
    image_type = pa.struct([("bytes", pa.binary()), ("path", pa.string())])
    source_table = pa.table({
        "image": pa.array([{"bytes": b"png", "path": None}], type=image_type),
        "env_name": ["g1_sonic_balance_simple"],
        "episode_idx": [0],
        "decision_step": [0],
        "split": ["train"],
        "seed": [0],
        "prompt": ["balance"],
        "latency_raw_frames": [0],
        "latency_ms": [0.0],
        "raw_reward": [1.0],
        "action": [list(np.arange(43, dtype=np.float32))],
        "action_text": ["old"],
        "state": [state.tolist()],
    })
    pq.write_table(source_table, source / "train.parquet")
    pq.write_table(source_table.slice(0, 0), source / "val.parquet")
    (source / "metadata.json").write_text(
        json.dumps({"train_rows": 1, "val_rows": 0, "keep": "metadata"}),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        balance_dagger,
        "load_config",
        lambda path: (_ for _ in ()).throw(AssertionError("teacher model loaded")),
    )
    policy_config = tmp_path / "policy.yaml"
    policy_config.write_text(yaml.safe_dump({
        "env": {
            "task_name": "g1_sonic_balance_simple",
            "env_id": "Balance-v0",
            "registration_imports": [],
            "make_kwargs": {"scene_path": "relative/scene.xml"},
        },
    }))

    balance_dagger.collect_balance_dagger(
        policy_config,
        tmp_path / "unused-teacher.pth",
        output,
        tmp_path / "unused-teacher.yaml",
        row_budget=1,
        num_envs=1,
        round_number=0,
        relabel_from=source,
    )

    relabeled = pq.read_table(output / "train.parquet")
    expected = np.arange(43, dtype=np.float32)
    np.testing.assert_allclose(relabeled["action"][0].as_py(), expected)
    assert relabeled["action_text"][0].as_py().startswith("[0.000000, 1.000000, 2.000000")
    assert relabeled["image"][0].as_py() == source_table["image"][0].as_py()
    assert relabeled.schema == source_table.schema
    assert (output / "metadata.json").read_bytes() == (source / "metadata.json").read_bytes()


def test_haic_collection_runs_train_and_val_at_teacher_50hz(tmp_path, monkeypatch):
    teacher = tmp_path / "teacher.pt"
    teacher.write_bytes(b"teacher")
    commands = []
    monkeypatch.setattr(collect, "_run", lambda command, **kwargs: commands.append(command))
    monkeypatch.setattr(collect, "_prepare_converted", lambda task, raw, output: commands.append([task, str(raw), str(output)]))

    collect._collect_haic(
        teacher_checkpoint=teacher,
        output_root=tmp_path / "haic",
        task_python="task-python",
        train_episodes=250,
        val_episodes=25,
        keep_failed=False,
        device="cuda:1",
    )

    assert commands[0][2:4] == ["teacher-collect", "--teacher-checkpoint"]
    assert commands[0][commands[0].index("--split") + 1] == "train"
    assert commands[0][commands[0].index("--episode-budget") + 1] == "250"
    assert commands[0][commands[0].index("--seed") + 1] == "0"
    assert commands[1][commands[1].index("--split") + 1] == "val"
    assert commands[1][commands[1].index("--episode-budget") + 1] == "25"
    assert commands[1][commands[1].index("--seed") + 1] == "1"
    assert commands[-1][-2:] == [str(teacher.resolve()), str((tmp_path / "haic/actor_adapt.pt").resolve())]


def test_mikasa_collection_uses_converter_task_directory_name(tmp_path, monkeypatch):
    teacher = tmp_path / "teacher.pt"
    teacher.write_bytes(b"teacher")
    commands = []
    moves = []
    monkeypatch.setattr(collect, "_run", lambda command, **kwargs: commands.append(command))
    monkeypatch.setattr(collect.shutil, "move", lambda source, destination: moves.append((source, destination)))
    monkeypatch.setattr(collect, "_write_mikasa_contract", lambda train, val: None)

    collect._collect_mikasa(
        teacher_checkpoint=teacher,
        output_root=tmp_path / "mikasa",
        task_python="task-python",
        train_episodes=2,
        val_episodes=1,
        keep_failed=True,
        device="cuda:0",
    )

    assert commands[0][-1] == "--keep-failed"
    assert Path(moves[0][0]).name == "intercept_grab_fast_vla_v0_h1_train"
    assert Path(moves[1][0]).name == "intercept_grab_fast_vla_v0_h1_val"


def test_mikasa_contract_writes_loader_modality_metadata(tmp_path):
    dirs = []
    for split in ("train", "val"):
        dataset = tmp_path / split
        (dataset / "data/chunk-000").mkdir(parents=True)
        (dataset / "meta").mkdir()
        pq.write_table(
            pa.table({"observation.state": [[0.0] * 7], "action": [[0.0] * 7]}),
            dataset / "data/chunk-000/file-000.parquet",
        )
        dirs.append(dataset)

    collect._write_mikasa_contract(*dirs)

    modality = json.loads(
        (dirs[0] / "meta/modality.json").read_text(encoding="utf-8")
    )
    assert modality["state"]["proprio"]["original_key"] == "observation.state"
    assert modality["action"]["target"]["original_key"] == "action"
    assert list(modality["video"]) == ["top", "wrist"]
