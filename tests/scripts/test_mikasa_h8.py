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

from latency_bench.data.starvla_tasks import normalize, task_spec, write_task_contract
from latency_bench.core.types import Action, Observation
from latency_bench.policy.action_prefix import encode_prefix_state, resolve_action_prefix, with_action_prefix
from scripts.mikasa.evaluate import _reset_seeded_slot, _reset_seeded_slots
from scripts.starvla import train_task
from scripts.tasks.mikasa_intercept_grab_fast import fixed_latency_grid as grid
from scripts.tasks.mikasa_intercept_grab_fast.collect_conditioned_h8 import (
    _copy_branch_state,
    _prefix_settings,
    _save_episode,
    _selected_slots,
    _teacher_chunks,
)
from scripts.tasks.mikasa_intercept_grab_fast.diagnose_conditioned_h8 import suffix_errors
from scripts.tasks.mikasa_intercept_grab_fast.prepare_conditioned_h8_dataset import (
    _features as conditioned_h8_features,
)
from scripts.tasks.mikasa_intercept_grab_fast.prepare_h8_dataset import prepare_mikasa_h8
from scripts.tasks.mikasa_intercept_grab_fast import publish_h8_checkpoints as h8_publisher
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


def test_h8_diagnostic_uses_only_teacher_suffix_heads():
    target = np.zeros((8, 7), dtype=np.float32)
    prediction = np.zeros_like(target)
    prediction[:3] = 100
    prediction[3] = 2
    prediction[4] = 4
    prediction[5:] = 100
    mask = np.array([False, False, False, True, True, False, False, False])

    result = suffix_errors(prediction, target, mask)

    assert result["first_head"] == 3
    assert result["first_mae"] == 2
    assert result["mae"] == 3


def test_h8_publication_paths_keep_training_conditions_distinct():
    prefix = "mikasa-intercept-grab-fast/vla/starvla-qwenoft-h8/"
    dagger = "h8-conditioned-dagger-20260923"
    sft = "h8-conditioned-seed-reset-20260922"

    assert h8_publisher._publication_path("zero-latency", "qwenoft", "h8-naive-dagger-20260923", "dagger") == (
        "zero-latency/" + prefix + "h8-naive-dagger-20260923"
    )
    for condition, suffix in (("profile-latency", "profile"), ("fixed-2", "fixed-2")):
        assert h8_publisher._publication_path(condition, "qwenoft", f"{sft}-{suffix}", "sft") == (
            "latency-aware/" + prefix + f"{sft}-{suffix}"
        )
        assert h8_publisher._publication_path(condition, "qwenoft", dagger, "dagger") == (
            "latency-aware/" + prefix + f"{dagger}-{suffix}"
        )


def test_h8_publisher_resolves_historical_marker_to_migrated_bundle(tmp_path, monkeypatch):
    run = tmp_path / "run"
    run.mkdir()
    stage = tmp_path / "stage"
    stage.mkdir()
    (stage / "config.yaml").write_text("model: qwenoft\n")
    path = h8_publisher._publication_path(
        "profile-latency", "qwenoft", "h8-conditioned-dagger-20260923", "dagger"
    )
    historical = {"path": path.replace("latency-aware/", "profile-latency/").removesuffix("-profile"),
                  "revision": "original-revision"}
    (run / "publication_hf_dagger.json").write_text(json.dumps(historical))
    monkeypatch.setattr(h8_publisher, "_prepare_stage", lambda *_: (stage, run, path, 5000))
    api = SimpleNamespace(
        repo_info=lambda *_, **__: SimpleNamespace(sha="migrated-revision"),
        get_paths_info=lambda *_, **__: [
            SimpleNamespace(path=f"{path}/config.yaml", size=(stage / "config.yaml").stat().st_size)
        ],
    )

    published = h8_publisher._publish(api, "profile-latency", "qwenoft", "dagger", dry_run=False)

    assert published == {"path": path, "revision": "migrated-revision"}
    assert json.loads((run / "publication_hf_dagger.json").read_text()) == historical


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


def test_mikasa_conditioned_h8_contract_is_63d_and_exports_both_masks(tmp_path):
    spec = task_spec("mikasa_conditioned_h8")
    assert (spec.state_dim, spec.action_dim, spec.action_horizon) == (63, 7, 8)
    assert spec.auxiliary_shape_map() == {
        "action_prefix_mask": (8,),
        "action_loss_mask": (8,),
    }
    assert conditioned_h8_features()["observation.state"]["shape"] == (63,)
    assert conditioned_h8_features()["issued_command_chunk"]["shape"] == (8, 7)

    dataset = tmp_path / "lerobot"
    (dataset / "data/chunk-000").mkdir(parents=True)
    (dataset / "meta").mkdir()
    pq.write_table(
        pa.table(
            {
                "observation.state": [[0.0] * 63],
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
                    "observation.state": {"names": [f"state_{index}" for index in range(63)]},
                    "issued_command_chunk": {"names": [f"action_{index}" for index in range(7)]},
                }
            }
        )
    )
    write_task_contract(dataset, spec=spec)
    contract = json.loads((dataset / "task_contract.json").read_text())
    assert contract["format"] == "latency_bench_starvla_h8_v3"
    assert contract["state_dim"] == 63
    stats = json.loads((dataset / "meta/stats_gr00t.json").read_text())["statistics"]
    assert stats["observation.state"] == contract["normalization"]["state"]
    assert stats["observation.state"]["min"][7:] == [-1.0] * 56
    assert stats["observation.state"]["max"][7:] == [1.0] * 56
    state = np.zeros(63, dtype=np.float32)
    state[7:21] = np.linspace(-1, 1, 14)
    np.testing.assert_allclose(normalize(state, contract["normalization"]["state"]), state,
                               atol=1e-7)
    assert contract["auxiliary"] == {
        "action_prefix_mask": {"shape": [8], "column": "action_prefix_mask"},
        "action_loss_mask": {"shape": [8], "column": "action_loss_mask"},
    }


@pytest.mark.parametrize("frames", [1, 2, 3, 4])
def test_prefix_state_is_observation_relative_and_excludes_teacher_suffix(frames):
    """Long-lived MIKASA contract: offline labels and live requests encode identical causal input."""
    from scripts.tasks.mikasa_intercept_grab_fast.collect_conditioned_h8 import _conditioned_observation

    proprio = np.arange(7, dtype=np.float32)
    chunk = np.arange(56, dtype=np.float32).reshape(8, 7) / 56
    mask = np.arange(8) < frames
    offline = encode_prefix_state(proprio, chunk, mask)
    changed_suffix = chunk.copy()
    changed_suffix[frames:] = -100
    np.testing.assert_array_equal(encode_prefix_state(proprio, changed_suffix, mask), offline)
    changed_prefix = chunk.copy()
    changed_prefix[0] = -0.5
    assert not np.array_equal(encode_prefix_state(proprio, changed_prefix, mask), offline)
    prefix = {"mode": "fixed_frames", "frames": frames, "action_horizon": 8, "action_dim": 7}
    for old_head in (1, 2, 4, 7):
        # Different ages can expose the same projected prefix, including hold-last.
        scheduler = SimpleNamespace(project_actions=lambda frame, count: [Action(a) for a in chunk[:count]])
        live, actions, live_mask = _conditioned_observation(proprio, scheduler, prefix, old_head)
        np.testing.assert_array_equal(live, offline)
        np.testing.assert_array_equal(live_mask, mask)
        np.testing.assert_array_equal(actions[:frames], chunk[:frames])
    assert offline.shape == (63,)
    np.testing.assert_array_equal(offline[7 + frames * 7:], 0)


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


@pytest.mark.parametrize("frames", [1, 2, 3, 4])
def test_mikasa_fixed_prefix_resolution_uses_training_frames(frames):
    resolved_fixed = resolve_action_prefix(
        {"mode": "fixed_frames", "frames": frames},
        env_fps=20,
        action_horizon=8,
        action_dim=7,
    )
    assert resolved_fixed == {
        "mode": "fixed_frames",
        "frames": frames,
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


def test_returned_chunk_context_is_independent_of_explicit_prefix():
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


@pytest.mark.parametrize("frames", [1, 2, 3, 4])
def test_conditioned_teacher_batches_prefixes_and_masks_each_terminal_tail(frames):
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
        success_after=[frames + 3, frames + 2],
    )
    prefixes = np.zeros((2, 8, 7), dtype=np.float32)
    prefixes[0, :frames] = np.arange(1, frames + 1)[:, None]
    prefixes[1, :frames] = np.arange(6, frames + 6)[:, None]
    prefix_masks = np.tile(np.arange(8) < frames, (2, 1))

    chunks, loss_masks = _teacher_chunks(
        Teacher(),
        main,
        private,
        prefixes,
        prefix_masks,
        np.asarray([True, True]),
    )

    np.testing.assert_array_equal(chunks[:, :frames], prefixes[:, :frames])
    np.testing.assert_allclose(chunks[0, frames:frames + 3, 0], [0.3, 0.4, 0.5])
    np.testing.assert_allclose(chunks[1, frames:frames + 2, 0], [0.3, 0.4])
    np.testing.assert_array_equal(chunks[0, frames + 3:], 0.0)
    np.testing.assert_array_equal(chunks[1, frames + 2:], 0.0)
    np.testing.assert_array_equal(
        loss_masks,
        [
            (np.arange(8) >= frames) & (np.arange(8) < frames + 3),
            (np.arange(8) >= frames) & (np.arange(8) < frames + 2),
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

    snapshot["old_training_claims"] = [0, 1, 4, 5]
    running = {"qwenoft": {"host": host, "train_devices": "6,7",
                            "simulator_gpu": 6, "status": "running"}}
    assert naive_queue._slot(snapshot, running, host) == ((2, 3), 2)


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
    ("model", "micro_batch", "num_gpus", "condition", "checkpointing", "gradient_accumulation_steps"),
    [("qwenoft", 128, 1, "zero-latency", True, 1),
     ("qwenpi_v3", 128, 1, "zero-latency", True, 1),
     ("qwengr00t", 64, 1, "zero-latency", True, 2),
     *[("qwenoft", 64, 2, f"fixed-{k}", False, 1) for k in (1, 3, 4)]],
)
def test_mikasa_h8_training_config(
    model, micro_batch, num_gpus, condition, checkpointing, gradient_accumulation_steps, tmp_path, monkeypatch
):
    task = "mikasa_h8" if condition == "zero-latency" else "mikasa_conditioned_h8"
    monkeypatch.setenv("WANDB_ENTITY", "test-entity")
    bootstrap = tmp_path / "bootstrap/lerobot"
    (bootstrap / "meta").mkdir(parents=True)
    (bootstrap / "meta/info.json").write_text(json.dumps({"codebase_version": "v3.0"}))
    (bootstrap / "task_contract.json").write_text(
        json.dumps(
            {
                "state_dim": 7 if task == "mikasa_h8" else 63,
                "action_dim": 7,
                "robot_type": f"latency_{task}",
                "action_horizon": 8,
                "normalization": {"action": {"min": [0] * 7, "max": [1] * 7}},
            }
        )
    )
    args = SimpleNamespace(
        task=task,
        model=model,
        output_dir=tmp_path,
        smoke=False,
        num_gpus=num_gpus,
        micro_batch=micro_batch,
        num_workers=8,
        persistent_workers=True,
        pin_memory=True,
        prefetch_factor=3,
        condition=condition,
        gradient_checkpointing=checkpointing,
    )

    config = train_task.training_config(
        args,
        bootstrap=bootstrap.parent,
        deltas=[],
        backbone=tmp_path / "Qwen3-VL-4B-Instruct",
        run_id="bootstrap",
        steps=5000,
        stop_step=5000,
        pretrained=None,
        resume=False,
    )

    assert config["framework"]["action_model"]["action_horizon"] == 8
    assert config["framework"]["action_model"]["state_dim"] == (7 if task == "mikasa_h8" else 63)
    assert config["framework"]["action_model"]["future_action_window_size"] == 7
    assert config["framework"]["qwenvl"]["enable_gradient_checkpointing"] is checkpointing
    assert config["datasets"]["vla_data"]["per_device_batch_size"] == micro_batch
    assert config["trainer"]["gradient_accumulation_steps"] == gradient_accumulation_steps
    assert config["trainer"]["max_train_steps"] == 5000
    assert config["wandb_tags"] == [task, "h8", "formal", condition]
    mixture = json.loads((tmp_path / "bootstrap_mixture.json").read_text())
    assert mixture["train"][0][2] == f"latency_{task}"

    if condition != "zero-latency":
        from scripts.tasks.mikasa_intercept_grab_fast.run_dagger import _trainer_config

        (tmp_path / "bootstrap.yaml").write_text(yaml.safe_dump(config))
        dagger_args = SimpleNamespace(
            source_run=tmp_path, output=tmp_path, condition=condition, model=model,
            bootstrap=bootstrap.parent, total_steps=5000, round_steps=500, smoke=False,
            source_checkpoint=tmp_path / "sft-step-5000.pt",
        )
        dagger = yaml.safe_load(_trainer_config(dagger_args, None).read_text())
        assert dagger["wandb_entity"] == config["wandb_entity"]
        assert dagger["wandb_project"] == config["wandb_project"]
        assert dagger["datasets"]["vla_data"]["per_device_batch_size"] == 64
        assert dagger["trainer"]["gradient_accumulation_steps"] == 1
        assert dagger["framework"]["qwenvl"]["enable_gradient_checkpointing"] is False
        assert dagger["trainer"]["max_train_steps"] == 5000
        assert dagger["trainer"]["save_interval"] == 500
        assert dagger["trainer"]["is_resume"] is False
        resumed = yaml.safe_load(_trainer_config(dagger_args, tmp_path / "steps_500_state").read_text())
        assert resumed["trainer"]["is_resume"] is True
        assert resumed["trainer"]["pretrained_checkpoint"] == str(tmp_path / "steps_500_state")


def test_mikasa_h8_dataset_reference_uses_h8_artifact_path(tmp_path):
    reference = train_task.dataset_reference("mikasa_h8", tmp_path)
    assert reference["action_horizon"] == 8
    assert "teacher-rollouts-h8-state-v1" in reference["path"]


def test_dagger_rejects_a_different_source_condition_before_starting(tmp_path):
    """Long-lived artifact boundary: a different Fixed-K source cannot be relabeled."""
    from scripts.tasks.mikasa_intercept_grab_fast.run_dagger import run

    (tmp_path / "run.json").write_text(json.dumps({
        "task": "mikasa_conditioned_h8", "model": "qwenoft", "condition": "fixed-2",
    }))
    with pytest.raises(ValueError, match="source must match"):
        run(SimpleNamespace(source_run=tmp_path, condition="fixed-3", model="qwenoft"))


@pytest.mark.parametrize("selected_step,bundle_name", [(0, "source"), (4500, "best")])
@pytest.mark.parametrize("train_latency", [2, 3])
def test_grid_selects_one_completed_dagger_checkpoint(tmp_path, selected_step, bundle_name, train_latency):
    """Long-lived selection contract: all eval rows use the development-selected bundle."""
    sft = tmp_path / "sft"
    sft.mkdir()
    (sft / "run.json").write_text(json.dumps({"training": {"sft_updates": 5000}}))
    (tmp_path / "run.json").write_text(json.dumps({"source_run": str(sft), "condition": f"fixed-{train_latency}"}))
    (tmp_path / "result.json").write_text(json.dumps({"dagger_steps": 5000, "best_step": selected_step}))
    bundle, source = grid.selected_model(tmp_path)
    assert bundle == tmp_path / "bundles" / bundle_name
    assert source["selected_step"] == selected_step
    assert source["condition"] == f"fixed-{train_latency}"


@pytest.mark.parametrize("train_latency", [1, 2, 3, 4])
def test_dagger_latency_cells_resume_and_select_only_matched_results(tmp_path, monkeypatch, train_latency):
    """Long-lived controller contract: five cells resume independently; only matched scores select weights."""
    from scripts.tasks.mikasa_intercept_grab_fast import run_dagger as dagger

    args = SimpleNamespace(
        output=tmp_path, eval_episodes=100,
        condition=f"fixed-{train_latency}", task_python=Path("python"), model="qwenoft", source_run=tmp_path,
        latency_config=tmp_path / "latency.yaml", bootstrap=tmp_path / "bootstrap",
        teacher_checkpoint=tmp_path / "teacher.pt", head_budget=16000, eval_envs=10,
        simulator_gpu=0, inference_gpu=1, inference_batch_size=8, gpu_lock=tmp_path / "gpu.lock",
        source_policy_checkpoint=tmp_path / "sft.pt", eval_configs={},
    )
    (tmp_path / "run.json").write_text(json.dumps({"backbone": {"revision": "test"}}))
    prefix = {"mode": "fixed_frames", "frames": train_latency, "action_horizon": 8, "action_dim": 7}
    for latency in range(5):
        path = tmp_path / f"eval_{latency}.yaml"
        path.write_text(yaml.safe_dump({"policy": {"action_prefix": prefix}}))
        args.eval_configs[latency] = path
    calls = []
    fail_once = True

    def evaluate(command, log):
        nonlocal fail_once
        destination = command[command.index("--output") + 1]
        latency = int(destination.name.removeprefix("eval_"))
        step = int(destination.parent.name.removeprefix("step_"))
        if latency == 3 and fail_once:
            fail_once = False
            raise RuntimeError("interrupted evaluation")
        config = yaml.safe_load(command[command.index("--eval-config") + 1].read_text())
        assert config["policy"]["action_prefix"] == prefix
        assert command[command.index("--seed") + 1] == dagger.DEVELOPMENT_SEED
        assert command[command.index("--checkpoint") + 1] == tmp_path / f"step_{step}.pt"
        calls.append((step, latency))
        count = (80 if step != 500 else 70) if latency == train_latency else 100
        dagger._write_json(destination / "result.json", {
            "success_count": count, "success_rate": count / 100, "episodes": 100,
            "mean_return": 2 if step >= 1000 else 1,
        })

    monkeypatch.setattr(dagger, "_run", evaluate)
    monkeypatch.setattr(dagger, "inference_bundle", lambda *args, **kwargs: None)
    with pytest.raises(RuntimeError, match="interrupted"):
        dagger._eval(args, 0, tmp_path / "step_0.pt", resident=False)
    assert not (tmp_path / "best.json").exists()
    assert not (tmp_path / "evaluations/step_0000/result.json").exists()
    dagger._eval(args, 0, tmp_path / "step_0.pt", resident=False)
    assert calls == [(0, latency) for latency in range(5)]
    dagger._eval(args, 0, tmp_path / "step_0.pt", resident=False)
    assert calls == [(0, latency) for latency in range(5)]
    for step, best_step in ((500, 0), (1000, 1000), (1500, 1500)):
        result = dagger._eval(args, step, tmp_path / f"step_{step}.pt", resident=True)
        assert result["episodes"] == 100
        assert calls[-5:] == [(step, latency) for latency in range(5)]
        assert len(result["latencies"]) == 5
        assert json.loads((tmp_path / "best.json").read_text())["step"] == best_step


@pytest.mark.parametrize("train_latency", [1, 2, 3, 4])
def test_prefix63_report_keeps_matched_selection_and_fixed_prefix(tmp_path, train_latency):
    """Long-lived reporting contract: all five cells use the run's K and selected checkpoint."""
    from scripts.tasks.mikasa_intercept_grab_fast.prefix_repair_results import summarize

    (tmp_path / "run.json").write_text(json.dumps({"condition": f"fixed-{train_latency}"}))
    (tmp_path / "best.json").write_text(json.dumps({"step": 500, "checkpoint": "best.pt"}))
    cell = {"action_prefix": {"frames": train_latency}, "audit": {"passed": True},
            "seed": 4242424442, "episodes": 100, "success_rate": 0.4}
    for step in range(0, 5001, 500):
        for latency in range(5):
            path = tmp_path / f"evaluations/step_{step:04d}/eval_{latency}"
            path.mkdir(parents=True)
            (path / "result.json").write_text(json.dumps(cell))
    for latency in range(5):
        path = tmp_path / f"heldout/eval_{latency}"
        path.mkdir(parents=True)
        (path / "result.json").write_text(json.dumps({
            **cell, "seed": 4242424242, "episodes": 200, "checkpoint": "best.pt",
        }))
    result = summarize(tmp_path, None, tmp_path / "report")
    assert result["selection_latency"] == train_latency
    assert len(result["periodic"]) == 55
    assert len(result["heldout"]) == 5
    assert {item["checkpoint"] for item in result["heldout"]} == {"best.pt"}
    assert "comparison" not in result


@pytest.mark.parametrize("upload_result", ["ok", "missing-state", "network-error"])
def test_prefix63_completed_phase_publication_requires_complete_remote_state(tmp_path, monkeypatch, upload_result):
    """Long-lived publication contract: a model-only or failed upload cannot acknowledge recoverability."""
    from scripts.tasks.mikasa_intercept_grab_fast import publish_prefix63 as publish

    trained = tmp_path / "training"
    checkpoints = trained / "checkpoints"
    checkpoints.mkdir(parents=True)
    model = checkpoints / "steps_5000_pytorch_model.pt"
    model.write_bytes(b"test-model")
    state = checkpoints / "steps_5000_state"
    state.mkdir()
    (state / "optimizer.pt").write_bytes(b"test-optimizer")
    contract = tmp_path / "task_contract.json"
    contract.write_text(json.dumps({"state_dim": 63, "action_prefix": {"frames": 2}}))
    data = {"task_contract_path": str(contract), "include_state": True,
            "obs_image_size": [224, 224], "image_mode": "single", "stitch_grid": [1, 2],
            "num_obs_frames": 1}
    cfg = {"run_id": "dagger", "training_latency_condition": "fixed-2",
           "version_id": "0.21", "framework": {"qwenvl": {"base_vlm": "/remote/backbone"}},
           "datasets": {"vla_data": data}, "trainer": {"max_train_steps": 5000}}
    (trained / "config.full.yaml").write_text(yaml.safe_dump(cfg))
    (trained / "dataset_statistics.json").write_text("{}")
    uploaded = {}

    def upload_folder(**kwargs):
        if upload_result == "network-error":
            raise RuntimeError("upload failed")
        for path in Path(kwargs["folder_path"]).rglob("*"):
            if path.is_file():
                uploaded[kwargs["path_in_repo"] + "/" + path.relative_to(kwargs["folder_path"]).as_posix()] = path.read_bytes()
        return SimpleNamespace(oid="verified-revision")

    def paths_info(repo, paths, **kwargs):
        assert kwargs["revision"] == "verified-revision"
        return [SimpleNamespace(path=p, size=len(uploaded[p])) for p in paths
                if not (upload_result == "missing-state" and p.endswith("optimizer.pt"))]

    monkeypatch.setattr(publish, "HfApi", lambda **kwargs: SimpleNamespace(
        upload_folder=upload_folder, get_paths_info=paths_info))
    receipt = trained / "publication_step_5000.json"
    if upload_result != "ok":
        with pytest.raises(RuntimeError):
            publish.publish_training_endpoint(trained, "test/fixed-2")
        assert not receipt.exists()
    else:
        publish.publish_training_endpoint(trained, "test/fixed-2")
        assert json.loads(receipt.read_text())["revision"] == "verified-revision"
        prefix = "test/fixed-2/training/dagger/step_05000/"
        assert uploaded[prefix + "checkpoints/model.pt"] == b"test-model"
        assert uploaded[prefix + "trainer_state/optimizer.pt"] == b"test-optimizer"
        assert json.loads(uploaded[prefix + "task_contract.json"])["state_dim"] == 63


@pytest.mark.parametrize("train_latency", [1, 2, 3, 4])
def test_fixed_dagger_final_eval_uses_all_latencies_and_one_best(tmp_path, monkeypatch, train_latency):
    """Long-lived final evaluation contract: matched-only development still ends with five held-out cells."""
    from scripts.tasks.mikasa_intercept_grab_fast import run_dagger as dagger

    (tmp_path / "best.json").write_text(json.dumps({"step": 1500, "checkpoint": "selected.pt"}))
    args = SimpleNamespace(output=tmp_path, condition=f"fixed-{train_latency}",
                           heldout_episodes=200, total_steps=5000, source_checkpoint=Path("sft.pt"),
                           eval_configs={latency: Path(f"eval_{latency}.yaml") for latency in range(5)})
    calls = []

    def rollout(args, mode, destination, **kwargs):
        calls.append(kwargs)
        dagger._write_json(destination / "result.json", {"success_count": 1})

    monkeypatch.setattr(dagger, "_rollout", rollout)
    dagger._heldout(args)
    assert [call["eval_config"] for call in calls] == list(args.eval_configs.values())
    assert all(call["checkpoint"] == Path("selected.pt") and call["episodes"] == 200
               and call["seed"] == 4242424242 for call in calls)
    result = json.loads((tmp_path / "result.json").read_text())
    assert result["best_step"] == 1500 and len(result["heldout"]) == 5


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
                "env": {"obs_fps": 20, "env_fps": 20},
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


@pytest.mark.parametrize("frames", [1, 2, 3, 4])
def test_mikasa_h8_execution_audit_requires_configured_fixed_delay(tmp_path, frames):
    (tmp_path / "eval_config.yaml").write_text(
        yaml.safe_dump(
            {
                "executor": {"mode": "simulated", "simulated_worker_capacity": 1},
                "latency": {"method": "fixed", "fixed_latency_ms": frames * 50.0},
                "env": {"obs_fps": 20, "env_fps": 20},
            }
        )
    )
    (tmp_path / "episode_metrics.jsonl").write_text("{}\n")
    action = {
        "obs_env_step": 0,
        "ready_raw_frame": frames,
        "apply_raw_frame": frames,
        "latency_raw_frames": frames,
        "applied": True,
        "dropped": False,
        "metadata": {"episode_id": 0},
    }
    (tmp_path / "actions.jsonl").write_text(json.dumps(action) + "\n")
    steps = [
        {
            "episode_id": 0,
            "env_step": env_step,
            "active_action": {"is_noop": env_step < frames},
            "info": {
                "active_action_obs_env_step": None if env_step < frames else 0,
                **({} if env_step < frames else {"chunk_age_raw_frames": env_step}),
                "in_flight_count": 1 if env_step < frames else 0,
                "observation_submitted": env_step == 0,
                "observation_dropped": env_step != 0,
            },
        }
        for env_step in range(frames + 2)
    ]
    (tmp_path / "steps.jsonl").write_text(
        "".join(json.dumps(step) + "\n" for step in steps)
    )
    audit = train_task.mikasa_h8_execution_audit(tmp_path)
    assert audit["latency_raw_frames"] == frames

    action["latency_raw_frames"] = frames + 1
    (tmp_path / "actions.jsonl").write_text(json.dumps(action) + "\n")
    with pytest.raises(AssertionError, match=f"non-{frames}-frame"):
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
        backbone_path=tmp_path / "backbone",
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
        tmp_path / "evaluation-zero",
        eval_episodes=50,
        eval_latency_frames=0,
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
        backbone_path=tmp_path / "backbone",
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
        args, bundle, tmp_path / "profile", eval_episodes=50
    )
    zero = train_task.eval_config(
        args,
        bundle,
        tmp_path / "zero",
        eval_episodes=50,
        eval_latency_frames=0,
    )
    fixed = train_task.eval_config(
        args,
        bundle,
        tmp_path / "fixed",
        eval_episodes=50,
        eval_latency_frames=2,
    )

    assert profile["env"]["action_prefix_state_key"] == "mikasa_proprio"
    assert zero["env"]["action_prefix_state_key"] == "mikasa_proprio"
    assert fixed["env"]["action_prefix_state_key"] == "mikasa_proprio"
    assert profile["policy"]["action_prefix"] == profile_prefix
    assert zero["policy"]["action_prefix"] == profile_prefix
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

    train_task.evaluate(args, tmp_path / "bundle", label="final")

    assert calls[0][2]["CUDA_VISIBLE_DEVICES"] == "0,3"


@pytest.mark.parametrize("train_latency", [1, 2, 3, 4])
@pytest.mark.parametrize("eval_latency", [0, 1, 2, 3, 4])
def test_conditioned_grid_preserves_training_prefix(tmp_path, train_latency, eval_latency):
    """Long-lived grid contract: the row changes arrival, never the trained prefix."""
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    prefix = {"mode": "fixed_frames", "frames": train_latency, "env_fps": 20,
              "action_horizon": 8, "action_dim": 7}
    (bundle / "task_contract.json").write_text(json.dumps({"action_prefix": prefix}))
    args = SimpleNamespace(train_latency=train_latency, eval_latency=eval_latency,
                           vla_python=tmp_path / "python", backbone_path=tmp_path / "backbone")
    config = grid.build_config(args, bundle, tmp_path / "eval")
    assert config["policy"]["action_prefix"] == prefix
    assert config["env"]["action_prefix_state_key"] == "mikasa_proprio"
    assert config["executor"]["simulated_worker_capacity"] == 1
    assert config["evaluation"]["eval_episodes"] == 200
    assert config["experiment"]["seed"] == 4242424242
    assert config["latency"] == ({"method": "zero", "sync_cuda": False} if eval_latency == 0
                                  else {"method": "fixed", "fixed_latency_ms": eval_latency * 50.0})


def test_grid_aggregate_reuses_zero_column_and_requires_common_seeds(tmp_path, monkeypatch):
    """Long-lived reporting contract: 25 ordered cells share the formal seed protocol."""
    def cell(train, latency):
        return {
            "train_latency": train, "latency_frames": latency, "n_episodes": 200,
            "seed_start": grid.START_SEED, "seed_end": grid.START_SEED + 199,
            "success_count": train * 10 + latency, "success_rate": (train * 10 + latency) / 200,
            "mean_return": train + latency, "std_return": 1.0,
            "artifact": {"selected_step": 5000, "column": train},
            "action_prefix": {"frames": train}, "execution_audit": {"passed": True},
        }
    zero = tmp_path / "zero.json"
    zero.write_text(json.dumps({"curves": {"qwenoft_h8": {
        "artifact": {"revision": grid.ZERO_REVISION},
        "latencies": {str(latency): cell(0, latency) for latency in range(5)},
    }}}))
    for train in range(1, 5):
        for latency in range(5):
            directory = tmp_path / f"train_{train}" / f"eval_{latency}"
            directory.mkdir(parents=True)
            (directory / "manifest.json").write_text("{}")
            (directory / "summary.json").write_text(json.dumps(cell(train, latency)))
    monkeypatch.setattr(grid, "summarize_cell", lambda directory, _: json.loads((directory / "summary.json").read_text()))
    result = grid.aggregate(zero, tmp_path)
    assert [(item["train_latency"], item["latency_frames"]) for item in result["cells"]] == [
        (train, latency) for train in range(5) for latency in range(5)
    ]
    assert [item["success_count"] for item in result["cells"][:5]] == list(range(5))
    assert all(item["action_prefix"] == {"mode": "none"} for item in result["cells"][:5])
    bad = cell(3, 1)
    bad["seed_start"] += 1
    (tmp_path / "train_3/eval_1/summary.json").write_text(json.dumps(bad))
    with pytest.raises(AssertionError):
        grid.aggregate(zero, tmp_path)


def test_grid_cell_summarizes_raw_episodes_and_action_trace(tmp_path):
    """Long-lived grid evidence contract: success/return and latency come from raw records."""
    prefix = {"mode": "fixed_frames", "frames": 2, "env_fps": 20,
              "action_horizon": 8, "action_dim": 7}
    (tmp_path / "eval_config.yaml").write_text(yaml.safe_dump({
        "executor": {"mode": "simulated", "simulated_worker_capacity": 1},
        "env": {"env_fps": 20, "obs_fps": 20},
        "latency": {"method": "fixed", "fixed_latency_ms": 50.0},
        "policy": {"action_prefix": prefix},
        "logging": {"output_dir": "/original/evaluation/location"},
    }))
    episodes, steps, actions = [], [], []
    for episode in range(200):
        episodes.append({
            "episode_id": episode, "episode_return_env": 1.0 if episode % 2 else 3.0,
            "survival_steps": 60, "metadata": {"episode_seed": grid.START_SEED + episode,
                "task_metrics": {"success": episode % 2 == 0}},
        })
        for frame in range(60):
            actions.append({
                "obs_env_step": frame, "ready_raw_frame": frame + 1,
                "apply_raw_frame": frame + 1 if frame < 59 else None,
                "latency_raw_frames": 1, "applied": frame < 59, "dropped": False,
                "metadata": {"episode_id": episode},
            })
            steps.append({
                "episode_id": episode, "env_step": frame,
                "active_action": {"is_noop": frame == 0},
                "info": {"active_action_obs_env_step": frame - 1 if frame else None,
                         "chunk_age_raw_frames": 1, "in_flight_count": 1,
                         "observation_submitted": True, "observation_dropped": False},
            })
    for name, values in (("episode_metrics", episodes), ("steps", steps), ("actions", actions)):
        (tmp_path / f"{name}.jsonl").write_text("".join(json.dumps(row) + "\n" for row in values))
    manifest = {"train_latency": 2, "eval_latency": 1, "bundle": "fixed2",
                "artifact": {"revision": "795468e53947d26314e891cbc375775c6e0bd491"},
                "inference_code": {"benchmark": "test"}, "action_prefix": prefix}
    summary = grid.summarize_cell(tmp_path, manifest)
    assert summary["success_count"] == 100
    assert summary["success_rate"] == 0.5
    assert summary["mean_return"] == 2.0
    assert summary["std_return"] == 1.0
    assert summary["execution_audit"]["passed"] is True
    assert summary["evidence_path"] == "/original/evaluation/location"
    actions[0]["latency_raw_frames"] = 2
    (tmp_path / "actions.jsonl").write_text("".join(json.dumps(row) + "\n" for row in actions))
    with pytest.raises(AssertionError):
        grid.summarize_cell(tmp_path, manifest)
