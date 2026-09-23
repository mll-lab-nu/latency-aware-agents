from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
import yaml

from latency_bench.data.starvla_tasks import task_spec, write_task_contract
from latency_bench.core.types import Action, Observation
from latency_bench.policy.action_prefix import resolve_action_prefix, with_action_prefix
from scripts.mikasa.evaluate import _reset_seeded_slot, _reset_seeded_slots
from scripts.starvla import train_task
from scripts.tasks.mikasa_intercept_grab_fast.collect_conditioned_h8 import (
    _copy_branch_state,
    _prefix_settings,
    _save_episode,
    _selected_slots,
    _teacher_chunks,
)
from scripts.tasks.mikasa_intercept_grab_fast.prepare_conditioned_h8_dataset import (
    _features as conditioned_h8_features,
)
from scripts.tasks.mikasa_intercept_grab_fast.prepare_h8_dataset import prepare_mikasa_h8
from scripts.tasks.mikasa_intercept_grab_fast import dispatch_h8_naive_dagger_queue as naive_queue


def _write_h1_split(root: Path, episode_lengths: list[int], *, split_at: int | None = None) -> None:
    (root / "data/chunk-000").mkdir(parents=True)
    (root / "meta").mkdir()
    episodes = np.repeat(np.arange(len(episode_lengths)), episode_lengths)
    values = np.arange(1, len(episodes) + 1, dtype=np.float32)
    table = pa.table(
        {
            "episode_index": episodes,
            "observation.state": [[float(value)] * 7 for value in values],
            "action": [[float(value)] * 7 for value in values],
        }
    )
    cuts = [0, split_at, len(table)] if split_at is not None else [0, len(table)]
    for index, (start, end) in enumerate(zip(cuts, cuts[1:])):
        pq.write_table(
            table.slice(start, end - start),
            root / f"data/chunk-000/file-{index:03d}.parquet",
        )
    action_names = [f"action_{index}" for index in range(7)]
    (root / "meta/info.json").write_text(
        json.dumps(
            {
                "robot_type": "latency_mikasa_h1",
                "codebase_version": "v3.0",
                "total_episodes": len(episode_lengths),
                "total_frames": len(episodes),
                "features": {
                    "observation.state": {
                        "dtype": "float32",
                        "shape": [7],
                        "names": [f"state_{index}" for index in range(7)],
                    },
                    "action": {
                        "dtype": "float32",
                        "shape": [7],
                        "names": action_names,
                    },
                },
            }
        )
    )
    (root / "meta/modality.json").write_text(
        json.dumps(
            {
                "state": {"proprio": {"start": 0, "end": 7, "original_key": "observation.state"}},
                "action": {"target": {"start": 0, "end": 7, "original_key": "action"}},
                "video": {
                    "top": {"original_key": "video.top"},
                    "wrist": {"original_key": "video.wrist"},
                },
                "annotation": {},
            }
        )
    )


def test_mikasa_h8_conversion_preserves_episode_boundaries_and_masks_padding(tmp_path: Path):
    source_train = tmp_path / "source/lerobot"
    source_val = tmp_path / "source/lerobot__val"
    _write_h1_split(source_train, [35] * 249 + [196], split_at=4000)
    _write_h1_split(source_val, [37] * 24 + [60])

    result = prepare_mikasa_h8(source_train, source_val, tmp_path / "output/lerobot")

    assert result["splits"] == {
        "train": {"episodes": 250, "anchors": 8911, "valid_action_targets": 64288},
        "validation": {"episodes": 25, "anchors": 948, "valid_action_targets": 6884},
    }
    output = tmp_path / "output/lerobot"
    rows = []
    for path in sorted((output / "data").glob("*/*.parquet")):
        rows.extend(pq.read_table(path).to_pylist())
    np.testing.assert_array_equal(np.asarray(rows[0]["issued_command_chunk"])[:, 0], np.arange(1, 9))
    np.testing.assert_array_equal(np.asarray(rows[34]["issued_command_chunk"])[1:], 0)
    assert rows[34]["action_loss_mask"] == [True] + [False] * 7
    np.testing.assert_array_equal(
        np.asarray(rows[3999]["issued_command_chunk"])[:, 0], np.arange(4000, 4008)
    )

    contract = json.loads((output / "task_contract.json").read_text())
    assert contract["format"] == "latency_bench_starvla_h8_v1"
    assert contract["robot_type"] == "latency_mikasa_h8"
    assert contract["action_key"] == "issued_command_chunk"
    assert contract["action_horizon"] == 8
    assert contract["auxiliary"] == {
        "action_loss_mask": {"shape": [8], "column": "action_loss_mask"}
    }
    assert contract["normalization"]["action"]["min"] == [1.0] * 7
    assert contract["normalization"]["action"]["max"] == [8911.0] * 7
    val_contract = json.loads((tmp_path / "output/lerobot__val/task_contract.json").read_text())
    assert val_contract["normalization"] == contract["normalization"]
    info = json.loads((output / "meta/info.json").read_text())
    assert info["robot_type"] == "latency_mikasa_h8"
    assert info["features"]["issued_command_chunk"]["shape"] == [8, 7]
    assert info["features"]["action_loss_mask"]["shape"] == [8]


def test_mikasa_h8_and_h1_task_contracts_remain_distinct():
    h1 = task_spec("mikasa")
    h8 = task_spec("mikasa_h8")
    assert (h1.action_horizon, h1.action_key, h1.robot_type) == (1, "action", "latency_mikasa_h1")
    assert (h8.action_horizon, h8.action_key, h8.robot_type) == (
        8,
        "issued_command_chunk",
        "latency_mikasa_h8",
    )
    assert h8.auxiliary_shape_map() == {"action_loss_mask": (8,)}


def test_mikasa_conditioned_h8_contract_is_71d_and_exports_both_masks(tmp_path):
    spec = task_spec("mikasa_conditioned_h8")
    assert (spec.state_dim, spec.action_dim, spec.action_horizon) == (71, 7, 8)
    assert spec.auxiliary_shape_map() == {
        "action_prefix_mask": (8,),
        "action_loss_mask": (8,),
    }
    assert conditioned_h8_features()["observation.state"]["shape"] == (71,)
    assert conditioned_h8_features()["issued_command_chunk"]["shape"] == (8, 7)

    dataset = tmp_path / "lerobot"
    (dataset / "data/chunk-000").mkdir(parents=True)
    (dataset / "meta").mkdir()
    pq.write_table(
        pa.table(
            {
                "observation.state": [[0.0] * 71],
                "action": [[0.0] * 7],
                "issued_command_chunk": [[[0.0] * 7 for _ in range(8)]],
                "action_prefix_mask": [[True, True] + [False] * 6],
                "action_loss_mask": [[False, False] + [True] * 6],
            }
        ),
        dataset / "data/chunk-000/file-000.parquet",
    )
    (dataset / "meta/info.json").write_text(
        json.dumps(
            {
                "features": {
                    "observation.state": {"names": [f"state_{index}" for index in range(71)]},
                    "issued_command_chunk": {"names": [f"action_{index}" for index in range(7)]},
                }
            }
        )
    )
    write_task_contract(dataset, spec=spec)
    contract = json.loads((dataset / "task_contract.json").read_text())
    assert contract["format"] == "latency_bench_starvla_h8_v2"
    assert contract["state_dim"] == 71
    assert contract["auxiliary"] == {
        "action_prefix_mask": {"shape": [8], "column": "action_prefix_mask"},
        "action_loss_mask": {"shape": [8], "column": "action_loss_mask"},
    }


@pytest.mark.parametrize(
    ("p95_ms", "frames"),
    [(100.32732703685761, 3), (150.35263347625732, 4), (100.35134406089783, 3)],
)
def test_mikasa_conditioned_prefix_resolution_uses_profile_p95(tmp_path, p95_ms, frames):
    profile = tmp_path / "profile.json"
    profile.write_text(json.dumps({"summary": {"p95_ms": p95_ms}}))
    resolved_profile = resolve_action_prefix(
        {
            "mode": "profile_p95",
            "profile_path": str(profile),
            "profile_revision": "66414463f2a609316da63935941a79abf0a73ce9",
        },
        env_fps=20,
        action_horizon=8,
        action_dim=7,
    )
    assert resolved_profile["frames"] == frames
    assert resolved_profile["p95_ms"] == p95_ms
    assert resolved_profile["profile_revision"] == "66414463f2a609316da63935941a79abf0a73ce9"


def test_mikasa_fixed_prefix_resolution_uses_exact_two_frames():
    resolved_fixed = resolve_action_prefix(
        {"mode": "fixed_frames", "frames": 2},
        env_fps=20,
        action_horizon=8,
        action_dim=7,
    )
    assert resolved_fixed == {
        "mode": "fixed_frames",
        "frames": 2,
        "env_fps": 20,
        "action_horizon": 8,
        "action_dim": 7,
    }


def test_mikasa_batched_slots_use_their_scalar_seeds():
    class SeededEnv:
        def __init__(self, num_envs):
            self.unwrapped = self
            self.device = torch.device("cpu")
            self.observation = torch.zeros(num_envs, dtype=torch.int64)

        def reset(self, *, seed, options):
            slot_id = int(options["env_idx"].item())
            self.observation[slot_id] = seed
            return self.observation.clone(), {}

    seeds = [101, 203, 307]
    batched = SeededEnv(len(seeds))
    batched_observation = _reset_seeded_slots(batched, seeds)

    isolated_observations = []
    for seed in seeds:
        isolated = SeededEnv(1)
        observation, _ = _reset_seeded_slot(isolated, slot_id=0, seed=seed)
        isolated_observations.append(int(observation[0]))

    assert batched_observation.tolist() == isolated_observations == seeds
    assert len(set(batched_observation.tolist())) == len(seeds)


def test_conditioned_zero_eval_keeps_real_returned_chunk_context_without_explicit_prefix():
    class Scheduler:
        def scheduled_chunk_event(self, current_step, current_time_ms):
            del current_step, current_time_ms
            return SimpleNamespace(
                action_chunk=[Action(np.full(7, head + 1.0)) for head in range(8)]
            ), 2

    observation = Observation(
        data=None,
        env_step=2,
        sim_time_ms=100.0,
        metadata={
            "mikasa_proprio": np.zeros(7, dtype=np.float32),
            "returned_action_context": {
                "state_key": "mikasa_proprio",
                "horizon": 8,
                "frames": None,
                "order": np.arange(7),
                "low": -np.ones(7),
                "high": np.ones(7),
            },
        },
    )

    conditioned = with_action_prefix(observation, Scheduler(), {"mode": "none"})

    state = conditioned.metadata["mikasa_proprio"]
    assert state.shape == (71,)
    np.testing.assert_array_equal(state[7:21], 0.0)
    np.testing.assert_array_equal(state[21:28], 3.0)
    np.testing.assert_array_equal(state[-8:], [False, False, True, False, False, False, False, False])
    assert "action_prefix" not in conditioned.metadata
    assert "action_prefix_mask" not in conditioned.metadata


class _BranchEnv:
    def __init__(self, *, state, elapsed, success_after):
        import torch

        self.unwrapped = self
        self.device = torch.device("cpu")
        self.state = torch.as_tensor(state, dtype=torch.float32)
        self._elapsed_steps = torch.as_tensor(elapsed, dtype=torch.int64)
        self.oracle_info = self._elapsed_steps[:, None].to(torch.float32)
        self.reached_status = torch.zeros(len(state), dtype=torch.bool)
        self.success_after = torch.as_tensor(success_after, dtype=torch.int64)

    @property
    def elapsed_steps(self):
        return self._elapsed_steps

    def get_state_dict(self):
        return {"state": self.state.clone()}

    def set_state_dict(self, state):
        self.state = state["state"].clone()

    def get_obs(self):
        return self.state.clone()

    def step(self, action):
        import torch

        self.state += action
        self._elapsed_steps += 1
        success = self._elapsed_steps >= self.success_after
        self.oracle_info = self.state[:, :1].clone()
        self.reached_status.copy_(success)
        return (
            {"state": self.state.clone()},
            torch.zeros(len(self.state)),
            torch.zeros(len(self.state), dtype=torch.bool),
            torch.zeros(len(self.state), dtype=torch.bool),
            {"success": success},
        )


def test_conditioned_private_branch_copies_mikasa_state_and_matches_same_action():
    import torch

    main = _BranchEnv(state=[[1.0] * 7, [2.0] * 7], elapsed=[4, 6], success_after=[100, 100])
    private = _BranchEnv(
        state=[[9.0] * 7, [8.0] * 7],
        elapsed=[0, 0],
        success_after=[100, 100],
    )
    _copy_branch_state(main, private)

    np.testing.assert_array_equal(private.state, main.state)
    np.testing.assert_array_equal(private.elapsed_steps, main.elapsed_steps)
    np.testing.assert_array_equal(private.oracle_info, main.oracle_info)
    np.testing.assert_array_equal(private.reached_status, main.reached_status)

    action = torch.tensor([[0.25] * 7, [-0.5] * 7])
    main.step(action)
    private.step(action)
    np.testing.assert_array_equal(private.state, main.state)
    np.testing.assert_array_equal(private.get_obs(), main.get_obs())


def test_conditioned_teacher_batches_prefixes_and_masks_each_terminal_tail():
    import torch

    class Teacher:
        def __init__(self):
            self.calls = 0

        def forward(self, observation):
            self.calls += 1
            return torch.full(
                (len(observation["state"]), 7),
                (self.calls + 2.0) / 10.0,
            )

    main = _BranchEnv(
        state=[[0.0] * 7, [1.0] * 7],
        elapsed=[0, 0],
        success_after=[100, 100],
    )
    private = _BranchEnv(
        state=[[9.0] * 7, [8.0] * 7],
        elapsed=[0, 0],
        success_after=[5, 4],
    )
    prefixes = np.zeros((2, 8, 7), dtype=np.float32)
    prefixes[0, 0] = 1.0
    prefixes[0, 1] = 2.0
    prefixes[1, 0] = 6.0
    prefixes[1, 1] = 7.0
    prefix_masks = np.asarray([[True, True] + [False] * 6] * 2)

    chunks, loss_masks = _teacher_chunks(
        Teacher(),
        main,
        private,
        prefixes,
        prefix_masks,
        np.asarray([True, True]),
    )

    np.testing.assert_array_equal(chunks[:, :2], prefixes[:, :2])
    np.testing.assert_allclose(chunks[0, 2:5, 0], [0.3, 0.4, 0.5])
    np.testing.assert_allclose(chunks[1, 2:4, 0], [0.3, 0.4])
    np.testing.assert_array_equal(chunks[0, 5:], 0.0)
    np.testing.assert_array_equal(chunks[1, 4:], 0.0)
    np.testing.assert_array_equal(
        loss_masks,
        [
            [False, False, True, True, True, False, False, False],
            [False, False, True, True, False, False, False, False],
        ],
    )


def test_naive_dagger_labels_full_h8_without_prefix():
    class Teacher:
        def forward(self, observation):
            return torch.full((len(observation["state"]), 7), 0.25)

    main = _BranchEnv(state=[[0.0] * 7], elapsed=[0], success_after=[100])
    private = _BranchEnv(state=[[9.0] * 7], elapsed=[0], success_after=[100])
    prefixes = np.zeros((1, 8, 7), dtype=np.float32)
    mask = np.zeros((1, 8), dtype=bool)
    chunks, loss_masks = _teacher_chunks(
        Teacher(), main, private, prefixes, mask, np.asarray([True]),
    )

    assert _prefix_settings("zero-latency", {}) == {"mode": "none"}
    assert conditioned_h8_features("mikasa_h8")["observation.state"]["shape"] == (7,)
    assert "action_prefix_mask" not in conditioned_h8_features("mikasa_h8")
    np.testing.assert_array_equal(loss_masks, np.ones((1, 8), dtype=bool))
    np.testing.assert_allclose(chunks, 0.25)


def test_naive_queue_respects_gpu_claims_after_ssh_json(monkeypatch):
    host = naive_queue.HOSTS[0]
    payload = {
        "memory": {str(gpu): 0 for gpu in range(8)},
        "busy_gpus": [],
        "old_training_claims": list(range(6)),
        "old_simulator_claims": [6, 7],
        "shareable_simulators": [],
    }
    monkeypatch.setattr(naive_queue.subprocess, "check_output",
                        lambda *args, **kwargs: json.dumps(payload))
    snapshot = naive_queue._remote(host, "snapshot", "--host", host)
    assert naive_queue._slot(snapshot, {}, host) is None

    snapshot["shareable_simulators"] = [6, 7]
    assert naive_queue._slot(snapshot, {}, host) == ((6, 7), 6)

    snapshot["old_training_claims"] = [2, 3, 4, 5]
    assert naive_queue._slot(snapshot, {}, host) == ((0, 1), 6)


def test_conditioned_batch_selects_shortest_seed_prefix_reaching_target():
    assert _selected_slots([0, 5, 4, 7], collected=3, target=10) == [
        False,
        True,
        True,
        False,
    ]


def test_dagger_episode_keeps_student_control_separate_from_teacher_suffix(tmp_path):
    teacher = np.zeros((8, 7), dtype=np.float32)
    student = np.zeros_like(teacher)
    teacher[2] = 0.4
    student[2] = -0.2
    prefix_mask = np.asarray([True, True] + [False] * 6)
    loss_mask = np.asarray([False, False, True] + [False] * 5)
    path = tmp_path / "episode.npz"
    _save_episode(path, [{
        "top": np.zeros((128, 128, 3), dtype=np.uint8),
        "wrist": np.zeros((128, 128, 3), dtype=np.uint8),
        "state": np.zeros(71, dtype=np.float32),
        "chunk": teacher,
        "student_chunk": student,
        "prefix_mask": prefix_mask,
        "loss_mask": loss_mask,
        "frame": 0,
        "request_index": 1,
        "latency_raw_frames": 2,
        "latency_ms": 100.0,
    }], seed=11_000_000)
    with np.load(path) as episode:
        np.testing.assert_array_equal(episode["issued_command_chunk"][0], teacher)
        np.testing.assert_array_equal(episode["student_command_chunk"][0], student)
        np.testing.assert_array_equal(episode["action_loss_mask"][0], loss_mask)
        assert episode["request_index"].tolist() == [1]


@pytest.mark.parametrize(
    ("model", "micro_batch", "gradient_accumulation_steps"),
    [("qwenoft", 128, 1), ("qwenpi_v3", 128, 1), ("qwengr00t", 64, 2)],
)
def test_mikasa_h8_single_gpu_training_config(
    model, micro_batch, gradient_accumulation_steps, tmp_path, monkeypatch
):
    monkeypatch.setenv("WANDB_ENTITY", "test-entity")
    bootstrap = tmp_path / "bootstrap/lerobot"
    (bootstrap / "meta").mkdir(parents=True)
    (bootstrap / "meta/info.json").write_text(json.dumps({"codebase_version": "v3.0"}))
    (bootstrap / "task_contract.json").write_text(
        json.dumps(
            {
                "state_dim": 7,
                "action_dim": 7,
                "robot_type": "latency_mikasa_h8",
                "action_horizon": 8,
                "normalization": {"action": {"min": [0] * 7, "max": [1] * 7}},
            }
        )
    )
    args = SimpleNamespace(
        task="mikasa_h8",
        model=model,
        output_dir=tmp_path,
        smoke=False,
        num_gpus=1,
        micro_batch=micro_batch,
        num_workers=8,
        persistent_workers=True,
        pin_memory=True,
        prefetch_factor=3,
        condition="zero-latency",
        gradient_checkpointing=True,
    )

    config = train_task.training_config(
        args,
        bootstrap=bootstrap.parent,
        deltas=[],
        backbone=tmp_path / "Qwen3-VL-4B-Instruct",
        actor_checkpoint=None,
        run_id="bootstrap",
        steps=5000,
        stop_step=5000,
        pretrained=None,
        resume=False,
    )

    assert config["framework"]["action_model"]["action_horizon"] == 8
    assert config["framework"]["action_model"]["future_action_window_size"] == 7
    assert config["framework"]["qwenvl"]["enable_gradient_checkpointing"] is True
    assert config["datasets"]["vla_data"]["per_device_batch_size"] == micro_batch
    assert config["trainer"]["gradient_accumulation_steps"] == gradient_accumulation_steps
    assert config["trainer"]["max_train_steps"] == 5000
    assert config["wandb_tags"] == ["mikasa_h8", "h8", "formal", "zero-latency"]
    mixture = json.loads((tmp_path / "bootstrap_mixture.json").read_text())
    assert mixture["train"][0][2] == "latency_mikasa_h8"


def test_mikasa_h8_dataset_reference_uses_h8_artifact_path(tmp_path):
    reference = train_task.dataset_reference("mikasa_h8", tmp_path)
    assert reference["action_horizon"] == 8
    assert "teacher-rollouts-h8-state-v1" in reference["path"]


def test_mikasa_h8_execution_audit_covers_initial_noops_requests_and_chunk_age(tmp_path):
    (tmp_path / "eval_config.yaml").write_text(
        yaml.safe_dump(
            {
                "executor": {
                    "mode": "simulated",
                    "simulated_worker_capacity": 1,
                },
                "latency": {
                    "method": "temporal",
                    "seed": 0,
                    "profile_path": "/profiles/qwenoft/profile.json",
                },
                "env": {"obs_fps": 20},
            }
        )
    )
    (tmp_path / "episode_metrics.jsonl").write_text("{}\n{}\n")
    events = [
        {
            "obs_env_step": 0,
            "ready_raw_frame": 2,
            "apply_raw_frame": 2,
            "applied": True,
            "dropped": False,
            "metadata": {"episode_id": episode},
        }
        for episode in range(2)
    ]
    (tmp_path / "actions.jsonl").write_text(
        "".join(json.dumps(event) + "\n" for event in events)
    )
    steps = [
        {
            "episode_id": episode,
            "env_step": env_step,
            "active_action": {"is_noop": env_step < 2},
            "info": {
                "active_action_obs_env_step": None if env_step < 2 else 0,
                **({} if env_step < 2 else {"chunk_age_raw_frames": env_step}),
                "in_flight_count": 1 if env_step < 2 else 0,
                "observation_submitted": env_step == 0,
                "observation_dropped": env_step != 0,
            },
        }
        for episode in range(2)
        for env_step in range(4)
    ]
    (tmp_path / "steps.jsonl").write_text(
        "".join(json.dumps(step) + "\n" for step in steps)
    )
    audit = train_task.mikasa_h8_execution_audit(tmp_path)
    assert audit["passed"] is True
    assert audit["initial_noop_transitions"] == 4
    assert audit["first_action_apply_raw_frames"] == [2, 2]
    assert audit["executed_later_heads"] == 4
    zero_config = yaml.safe_load((tmp_path / "eval_config.yaml").read_text())
    zero_config["latency"] = {"method": "zero", "sync_cuda": False}
    (tmp_path / "eval_config.yaml").write_text(yaml.safe_dump(zero_config))
    zero_events = [
        {
            "obs_env_step": 0,
            "ready_raw_frame": 0,
            "apply_raw_frame": 0,
            "applied": True,
            "dropped": False,
            "metadata": {"episode_id": episode},
        }
        for episode in range(2)
    ]
    (tmp_path / "actions.jsonl").write_text(
        "".join(json.dumps(event) + "\n" for event in zero_events)
    )
    zero_steps = [
        {
            "episode_id": episode,
            "env_step": env_step,
            "active_action": {"is_noop": False},
            "info": {
                "active_action_obs_env_step": env_step,
                "chunk_age_raw_frames": 0,
                "in_flight_count": 0,
                "observation_submitted": True,
                "observation_dropped": False,
            },
        }
        for episode in range(2)
        for env_step in range(4)
    ]
    (tmp_path / "steps.jsonl").write_text(
        "".join(json.dumps(step) + "\n" for step in zero_steps)
    )
    zero_audit = train_task.mikasa_h8_execution_audit(tmp_path)
    assert zero_audit["latency_method"] == "zero"
    zero_steps[3]["info"]["chunk_age_raw_frames"] = 1
    (tmp_path / "steps.jsonl").write_text(
        "".join(json.dumps(step) + "\n" for step in zero_steps)
    )
    with pytest.raises(AssertionError, match="chunk execution mismatch"):
        train_task.mikasa_h8_execution_audit(tmp_path)


def test_mikasa_h8_execution_audit_requires_exact_fixed_two_frame_delay(tmp_path):
    (tmp_path / "eval_config.yaml").write_text(
        yaml.safe_dump(
            {
                "executor": {"mode": "simulated", "simulated_worker_capacity": 1},
                "latency": {"method": "fixed", "fixed_latency_ms": 100.0},
                "env": {"obs_fps": 20},
            }
        )
    )
    (tmp_path / "episode_metrics.jsonl").write_text("{}\n")
    action = {
        "obs_env_step": 0,
        "ready_raw_frame": 2,
        "apply_raw_frame": 2,
        "latency_raw_frames": 2,
        "applied": True,
        "dropped": False,
        "metadata": {"episode_id": 0},
    }
    (tmp_path / "actions.jsonl").write_text(json.dumps(action) + "\n")
    steps = [
        {
            "episode_id": 0,
            "env_step": env_step,
            "active_action": {"is_noop": env_step < 2},
            "info": {
                "active_action_obs_env_step": None if env_step < 2 else 0,
                **({} if env_step < 2 else {"chunk_age_raw_frames": env_step}),
                "in_flight_count": 1 if env_step < 2 else 0,
                "observation_submitted": env_step == 0,
                "observation_dropped": env_step != 0,
            },
        }
        for env_step in range(4)
    ]
    (tmp_path / "steps.jsonl").write_text(
        "".join(json.dumps(step) + "\n" for step in steps)
    )
    audit = train_task.mikasa_h8_execution_audit(tmp_path)
    assert audit["latency_raw_frames"] == 2

    action["latency_raw_frames"] = 3
    (tmp_path / "actions.jsonl").write_text(json.dumps(action) + "\n")
    with pytest.raises(AssertionError, match="non-2-frame"):
        train_task.mikasa_h8_execution_audit(tmp_path)


def test_mikasa_h8_eval_uses_20hz_rtx3090_profile_capacity_one(tmp_path, monkeypatch):
    args = SimpleNamespace(
        task="mikasa_h8",
        model="qwenoft",
        condition="zero-latency",
        eval_seed=4242424242,
        inference_device="cuda:1",
        simulator_device="cuda:0",
        smoke=False,
        vla_python=tmp_path / "vla-python",
    )
    profile_path = tmp_path / "profiles/qwenoft/profile.json"
    monkeypatch.setattr(
        train_task,
        "mikasa_h8_profile_latency",
        lambda model: {
            "method": "temporal",
            "seed": 0,
            "profile_path": str(profile_path),
        },
    )

    config = train_task.eval_config(
        args,
        tmp_path / "bundle",
        tmp_path / "teacher",
        tmp_path / "evaluation",
        eval_episodes=50,
    )

    assert config["env"]["env_fps"] == 20
    assert config["env"]["obs_fps"] == 20
    assert config["executor"] == {
        "mode": "simulated",
        "simulated_inference_pool": True,
        "simulated_worker_capacity": 1,
        "inference_devices": ["cuda:1"],
        "inference_batch_size": 8,
    }
    assert config["latency"] == {
        "method": "temporal",
        "seed": 0,
        "profile_path": str(profile_path),
    }
    assert config["evaluation"]["eval_parallel_envs"] == 10

    zero_config = train_task.eval_config(
        args,
        tmp_path / "bundle",
        tmp_path / "teacher",
        tmp_path / "evaluation-zero",
        eval_episodes=50,
        zero_latency=True,
    )
    assert zero_config["executor"]["simulated_worker_capacity"] == 1
    assert zero_config["latency"] == {"method": "zero", "sync_cuda": False}


def test_mikasa_conditioned_eval_keeps_causal_action_context_in_all_settings(
    tmp_path, monkeypatch
):
    args = SimpleNamespace(
        task="mikasa_conditioned_h8",
        model="qwenpi_v3",
        condition="profile-latency",
        eval_seed=4242424242,
        inference_device="cuda:1",
        simulator_device="cuda:0",
        smoke=False,
        vla_python=tmp_path / "vla-python",
    )
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    profile_prefix = {
        "mode": "profile_p95",
        "frames": 3,
        "p95_ms": 100.32732703685761,
        "profile_revision": "66414463f2a609316da63935941a79abf0a73ce9",
        "env_fps": 20,
        "action_horizon": 8,
        "action_dim": 7,
    }
    (bundle / "task_contract.json").write_text(
        json.dumps({"action_prefix": profile_prefix})
    )
    monkeypatch.setattr(
        train_task,
        "mikasa_h8_profile_latency",
        lambda model: {
            "method": "temporal",
            "seed": 0,
            "profile_path": "/profiles/qwenpi_v3/profile.json",
        },
    )

    profile = train_task.eval_config(
        args, bundle, tmp_path / "teacher", tmp_path / "profile", eval_episodes=50
    )
    zero = train_task.eval_config(
        args,
        bundle,
        tmp_path / "teacher",
        tmp_path / "zero",
        eval_episodes=50,
        zero_latency=True,
    )
    fixed = train_task.eval_config(
        args,
        bundle,
        tmp_path / "teacher",
        tmp_path / "fixed",
        eval_episodes=50,
        fixed_latency=True,
    )

    expected_context = {
        "state_key": "mikasa_proprio",
        "horizon": 8,
        "frames": None,
        "order": list(range(7)),
        "low": [-1.0] * 7,
        "high": [1.0] * 7,
    }
    assert profile["env"]["returned_action_context"] == expected_context
    assert zero["env"]["returned_action_context"] == expected_context
    assert fixed["env"]["returned_action_context"] == expected_context
    assert profile["policy"]["action_prefix"] == profile_prefix
    assert zero["policy"]["action_prefix"] == {"mode": "none"}
    assert fixed["latency"] == {"method": "fixed", "fixed_latency_ms": 100.0}


def test_mikasa_h8_eval_reorders_physical_simulator_and_inference_gpus(tmp_path, monkeypatch):
    args = SimpleNamespace(
        task="mikasa_h8",
        task_python=tmp_path / "mikasa-python",
        output_dir=tmp_path,
        eval_episodes=2,
        smoke=True,
        eval_cuda_visible_devices="0,3",
    )
    monkeypatch.setattr(train_task, "eval_config", lambda *args, **kwargs: {})
    calls = []
    monkeypatch.setattr(
        train_task,
        "run_command",
        lambda command, log, *, env: calls.append((command, log, env)),
    )

    train_task.evaluate(args, tmp_path / "bundle", tmp_path / "teacher", label="final")

    assert calls[0][2]["CUDA_VISIBLE_DEVICES"] == "0,3"
