import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from PIL import Image
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


def test_balance_collection_is_zero_latency_and_uses_state198(tmp_path, monkeypatch):
    teacher = tmp_path / "teacher.pt"
    teacher.write_bytes(b"teacher")
    commands = []
    monkeypatch.setattr(collect, "_run", lambda command, **kwargs: commands.append(command))
    monkeypatch.setattr(collect, "_prepare_converted", lambda task, raw, output: commands.append([task, str(raw), str(output)]))

    collect._collect_balance(
        teacher_checkpoint=teacher,
        output_root=tmp_path / "balance",
        task_python="task-python",
        train_episodes=90,
        val_episodes=10,
        keep_failed=False,
        device="cuda:0",
    )

    export = commands[0]
    assert export[export.index("--max-episodes") + 1] == "100"
    assert export[export.index("--val-fraction") + 1] == "0.1"
    assert "--balance-teacher-observation-state" in export
    assert "--latency-config" not in export
    prepare = commands[1]
    assert prepare[0] == "humanoidbench_balance_simple"
    assert prepare[-1].endswith("/balance/lerobot")


@pytest.mark.parametrize("row_budget,max_steps", [(4, 1000), (3, 1)])
def test_balance_dagger_executes_student_and_records_same_state_teacher_action(tmp_path, monkeypatch, row_budget, max_steps):
    environments = []
    written = {}

    class FakeEnv:
        def __init__(self):
            self.unwrapped = self
            self.state = 0
            self.actions = []
            self.states_at_step = []

        def reset(self, *, seed):
            self.state = seed
            return np.zeros(198, dtype=np.float32), {}

        def render(self):
            return np.full((2, 2, 3), self.state % 255, dtype=np.uint8)

        def _official_gr00t_proprio(self):
            return np.full(198, self.state, dtype=np.float32)

        def g1_absolute_request_from_raw_action(self, action):
            return np.asarray(action) + 10

        def step(self, action):
            self.states_at_step.append(self.state)
            self.actions.append(np.asarray(action))
            self.state += 1
            return None, 1.0, False, False, {}

        def close(self):
            pass

    class FakePool:
        def __init__(self, **kwargs):
            pass

        def reset_state(self, slot_id):
            pass

        def predict_batch(self, observations):
            return [
                SimpleNamespace(action_chunk=np.full((8, 43), -1.0))
                for _ in observations
            ]

        def close(self):
            pass

    class FakeTeacherPolicy:
        def predict_batch(self, observations):
            return [
                SimpleNamespace(raw_output=np.repeat(
                    (observation.data[0] + np.arange(8))[:, None], 43, axis=1,
                ))
                for observation in observations
            ]

    def make_env(*args, **kwargs):
        env = FakeEnv()
        environments.append(env)
        return env

    monkeypatch.setattr(balance_dagger, "make_collection_env", make_env)
    monkeypatch.setattr(balance_dagger, "ProcessInferencePool", FakePool)
    monkeypatch.setattr(
        balance_dagger,
        "load_teacher",
        lambda *args, **kwargs: FakeTeacherPolicy(),
    )
    monkeypatch.setattr(
        balance_dagger,
        "write_balance_dataset",
        lambda output, env_config, train, val: written.update(train=train, val=val),
    )
    policy_config = tmp_path / "policy.yaml"
    policy_config.write_text(yaml.safe_dump({
        "env": {
            "task_name": "g1_sonic_balance_simple",
            "env_id": "Balance-v0",
            "registration_imports": [],
            "make_kwargs": {},
            "simulator_device": "cuda:0",
            "env_fps": 50,
            "obs_fps": 50,
            "frame_stack": 1,
            "base_prompt": "balance",
        },
        "policy": {"device": "cuda:1"},
        "executor": {"inference_batch_size": 2},
        "evaluation": {"eval_max_steps": max_steps},
    }))

    balance_dagger.collect_balance_dagger(
        policy_config,
        tmp_path / "teacher.pth",
        tmp_path / "raw",
        tmp_path / "teacher.yaml",
        row_budget=row_budget,
        num_envs=2,
        round_number=0,
    )

    assert len(written["train"]) == row_budget
    assert written["val"] == []
    seed_base = 10_000_000
    assert written["train"][0]["state"] == [seed_base] * 198
    assert written["train"][0]["action"] == [seed_base + 10] * 43
    offsets = [0, 1, 1, 2] if max_steps == 1000 else [0, 1, 2]
    assert [row["state"][0] for row in written["train"]] == [seed_base + offset for offset in offsets]
    assert environments[0].states_at_step == [seed_base + offset for offset in ([0, 1] if max_steps == 1000 else [0, 2])]
    assert environments[1].states_at_step == [seed_base + offset for offset in ([1, 2] if max_steps == 1000 else [1])]
    for row in written["train"]:
        assert row["issued_command_chunk"] == [[row["state"][0] + 10 + head] * 43 for head in range(8)]
        with Image.open(row["image"]) as image:
            assert np.asarray(image)[0, 0, 0] == row["state"][0] % 255
    if max_steps == 1:
        assert [row["episode_idx"] for row in written["train"]] == [0, 1, 2]
        assert [row["decision_step"] for row in written["train"]] == [0, 0, 0]
    assert all(np.array_equal(action, np.full((8, 43), -1.0)) for env in environments for action in env.actions)


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
