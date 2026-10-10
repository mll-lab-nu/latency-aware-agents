import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml

from scripts.starvla import balance_dagger


def test_balance_dagger_model_geometry_and_latency_recipes(tmp_path):
    """Long-lived experiment contract: QwenPI is H16/profile; QwenOFT baseline is H8/zero."""
    from latency_bench.integrations.starvla.task_contracts import task_spec
    from scripts.tasks.humanoidbench_balance_simple import vla_profile_dagger as profile

    profile_path = tmp_path / "profile.json"
    profile_path.write_text(json.dumps({"summary": {"mean_ms": 99.176261}}))
    qwenpi = profile.vla_experiment_config("qwenpi_v3", "profile_conditioned", horizon=16, profile_path=profile_path)
    qwenoft = profile.vla_experiment_config("qwenoft", "zero_full", horizon=8)
    assert qwenpi["training"]["ppo"]["chunk_horizon"] == 16
    assert qwenpi["training"]["ppo"]["active_heads"] == [0, 16]
    assert qwenpi["latency"]["method"] == "temporal"
    assert qwenpi["latency"]["profile_worker_slot"] == 0
    assert qwenpi["latency"]["add_latency_info"] is True
    assert qwenpi["policy"]["action_prefix"]["frames"] == 5
    assert qwenoft["policy"]["action_prefix"] == {"mode": "none"}
    assert qwenoft["training"]["ppo"]["chunk_horizon"] == 8
    assert qwenoft["training"]["ppo"]["active_heads"] == [0, 8]
    assert qwenoft["latency"]["method"] == "zero"
    assert profile.TEACHER_QUERY_CADENCE_RAW_FRAMES == 1
    assert profile.TEACHER.name == "checkpoint.pth"
    assert profile.TEACHER_CONFIG.name == "fasttd3_train.yaml"
    assert profile.TEACHER_REVISION == "a8210caa0e7432f208b377d678ca5f10f5f1b879"
    assert (task_spec("balance_profile_h16").action_horizon,
            task_spec("balance_profile_h16").state_dim) == (16, 902)
    assert (task_spec("balance_profile_h8").action_horizon,
            task_spec("balance_profile_h8").state_dim) == (8, 550)


def test_profile_accepts_conditioned_qwenoft_with_explicit_profile(tmp_path):
    from scripts.tasks.humanoidbench_balance_simple import vla_profile_dagger as profile

    profile_path = tmp_path / "profile.json"
    profile_path.write_text(json.dumps({"summary": {"mean_ms": 37.880780}}))
    config = profile.vla_experiment_config(
        "qwenoft", "profile_conditioned", horizon=8, profile_path=profile_path
    )

    assert config["latency"]["profile_path"] == str(profile_path)
    assert config["training"]["ppo"]["chunk_horizon"] == 8




def test_profile_teacher_suffix_stops_loss_mask_at_termination():
    from collections import Counter

    import torch
    from scripts.tasks.humanoidbench_balance_simple.vla_profile_dagger import VLAProfileRollout

    class Physics:
        frames = 0

        def step(self, action):
            self.frames += 1
            return np.full(198, self.frames, dtype=np.float32), 0.0, self.frames == 2, False, {}

    class Teacher:
        queries = 0

        def empirical_obs_normalizer(self, observation, update_stats):
            return observation

        def actor(self, observation):
            self.queries += 1
            return torch.full((1, 43), float(self.queries))

    physics, teacher = Physics(), Teacher()
    rollout = object.__new__(VLAProfileRollout)
    rollout.prediction = SimpleNamespace(unwrapped=physics)
    rollout.chunk = SimpleNamespace(
        native_action_space=SimpleNamespace(low=-np.ones(43), high=np.ones(43)),
        env=SimpleNamespace(action=lambda action: action),
    )
    targets, valid, timings = np.zeros((16, 43)), np.zeros(16, dtype=bool), Counter()

    rollout._teacher_suffix(teacher, np.zeros(198), 100, 5, 11, targets, valid, timings)

    assert valid.tolist() == [False] * 5 + [True] * 2 + [False] * 9
    assert teacher.queries == timings["teacher_queries"] == 2
    assert physics.frames == timings["teacher_branch_frames"] == 2
    np.testing.assert_array_equal(targets[5:7, 0], [1, 2])


def test_balance_profile_row_reuses_the_async_image_path(tmp_path):
    from scripts.tasks.humanoidbench_balance_simple import vla_profile_dagger as profile

    config = profile.vla_experiment_config("qwenpi_v3", "zero_full", horizon=16)["env"]
    row = balance_dagger._balance_row_for_image_path(
        config, Path("/tmp/profile-frame.png"), np.zeros(198), np.zeros(43),
        episode=7, step=11, seed=17, split="train")

    assert row["image"] == "/tmp/profile-frame.png"
    assert row["episode_idx"] == 7
    assert row["decision_step"] == 11
    assert "rgb" not in row


def test_profile_inference_uses_the_explicit_local_backbone(tmp_path):
    from scripts.tasks.humanoidbench_balance_simple import vla_profile_dagger as profile

    config = profile.vla_experiment_config("qwenpi_v3", "zero_full", horizon=16)
    result = profile.policy_config(
        config, tmp_path / "bundle", tmp_path / "env/bin/python", 8,
        tmp_path / "Qwen3-VL-4B-Instruct")

    assert result["policy"]["backbone_path"] == str(tmp_path / "Qwen3-VL-4B-Instruct")


def test_profile_publish_records_the_selected_checkpoint_and_revision(tmp_path, monkeypatch):
    from scripts.tasks.humanoidbench_balance_simple import vla_profile_dagger as profile

    bundle = tmp_path / "bundle"
    (bundle / "checkpoints").mkdir(parents=True)
    published = {}
    monkeypatch.setattr(
        profile.artifacts, "publish_model",
        lambda source, **kwargs: published.update(source=source, kwargs=kwargs) or "revision-123")
    args = SimpleNamespace(
        condition="latency-aware", model="qwenpi_v3", run_id="formal-profile-conditioned",
        horizon=16, label_mode="profile_conditioned", teacher=tmp_path / "teacher.pth",
        teacher_config=tmp_path / "teacher.yaml", teacher_revision="teacher-revision",
        teacher_sha256="teacher-sha", output_dir=tmp_path,
    )
    evaluation = {"n": 64, "mean_length": 100.0, "mean_return": 80.0, "episodes": []}
    result = {"optimizer_steps": 10000}
    run_record = {"development_seeds": list(range(32)), "heldout_seeds": list(range(64)),
                  "wandb_runs": {"dagger": {"id": "run-dagger"}}}

    publication = profile.publish_result(args, bundle, evaluation, result, run_record)

    assert published["source"] == bundle
    assert published["kwargs"]["condition"] == "latency-aware"
    assert json.loads((bundle / "eval_profile.json").read_text())["n"] == 64
    assert json.loads((bundle / "provenance.json").read_text())["optimizer_steps"] == 10000
    assert publication["revision"] == "revision-123"
    assert json.loads((tmp_path / "publication.json").read_text()) == publication


def test_profile_smoke_rows_keep_train_and_validation_for_one_long_episode(tmp_path, monkeypatch):
    from scripts.tasks.humanoidbench_balance_simple import vla_profile_dagger as profile

    rows = [{"episode_idx": 7, "decision_step": step,
             "action_prefix_mask": [False], "action_loss_mask": [True]}
            for step in range(20)]
    captured = {}
    monkeypatch.setattr(profile, "_ProfileEnvPool", lambda *args, **kwargs: SimpleNamespace(
        num_envs=2, close=lambda: None, flush=lambda: {}))
    episodes = [{"seed": 7, "length": 20, "return": 1.0, "applied_heads": [0],
                 "action_ready_frame_counts": {}, "worker_service_frame_counts": {}}]
    monkeypatch.setattr(profile, "_collect_requests", lambda *args, **kwargs: (
        episodes, rows, {"steady_seconds": 0.0, "inference_seconds": 0.0,
                   "initial_requests_seconds": 0.0, "inference_batch_sizes": {},
                   "generated_labels": len(rows)}))
    def write_dataset(directory, config, train, val):
        captured.update(train=train, val=val)
        (directory / "metadata.json").write_text("{}")
    monkeypatch.setattr(profile, "write_balance_dataset", write_dataset)
    monkeypatch.setattr(profile, "write_json", lambda *args, **kwargs: None)
    args = SimpleNamespace(
        smoke=True, rollout_overlap=False, inference_batch_size=8, rollout_workers=16,
        label_mode="profile_conditioned", teacher=tmp_path / "teacher.pth",
        teacher_config=tmp_path / "teacher.yaml", teacher_revision="revision",
        teacher_sha256="sha", collect_bootstrap_only=True, collection_seed=0,
        horizon=1,
    )
    config = {"env": {}, "latency": {}, "policy": {"action_prefix": {"mode": "none"}}}

    profile.rollout_batch(args, config, tmp_path / "collection", bundle=None, run=None, count=20)

    assert len(captured["train"]) == 18
    assert len(captured["val"]) == 2


def test_profile_periodic_eval_is_logged_with_its_training_run(tmp_path, monkeypatch):
    from scripts.tasks.humanoidbench_balance_simple import vla_profile_dagger as profile

    captured = {}

    class Run:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def log(self, metrics):
            captured["metrics"] = metrics

    monkeypatch.setenv("WANDB_ENTITY", "entity")
    monkeypatch.setattr(profile.wandb, "init", lambda **kwargs: captured.update(kwargs=kwargs) or Run())
    args = SimpleNamespace(smoke=False, collect_bootstrap_only=False, run_id="formal", task="balance",
                           model="qwenoft", label_mode="zero_full", output_dir=tmp_path,
                           wandb_runs={"dagger": {"id": "formal-dagger"}})
    evaluation = {"mean_length": 40.0, "mean_return": 27.0, "n": 32, "ge100": 0, "ge500": 0}

    profile._log_periodic_eval(args, "dagger", 5000, 10000, evaluation, 56.5625)

    assert captured["kwargs"]["id"] == "formal-dagger"
    assert captured["kwargs"]["name"] == "balance-zero_full-qwenoft-dagger"
    assert captured["metrics"] == {"global_step": 5000, "optimizer_steps": 10000,
                                    "eval/mean_length": 40.0, "eval/mean_return": 27.0,
                                    "eval/best_mean_length": 56.5625, "eval/n": 32,
                                    "eval/ge100": 0, "eval/ge500": 0}


def test_profile_dagger_launch_uses_one_resumable_full_schedule(tmp_path, monkeypatch):
    from scripts.tasks.humanoidbench_balance_simple import vla_profile_dagger as profile

    bootstrap = tmp_path / "bootstrap"
    (bootstrap / "lerobot").mkdir(parents=True)
    (bootstrap / "lerobot/task_contract.json").write_text(
        json.dumps({"robot_type": "profile_h8"})
    )
    deltas = [tmp_path / "dagger_01", tmp_path / "dagger_02"]
    state = tmp_path / "training/dagger/checkpoints/steps_2_state"
    state.mkdir(parents=True)
    args = SimpleNamespace(
        output_dir=tmp_path,
        task="balance",
        label_mode="zero_full",
        model="qwenoft",
        horizon=8,
        smoke=True,
        run_id="smoke",
        backbone=tmp_path / "backbone",
        gradient_checkpointing=False,
        vla_python=tmp_path / "env/bin/python",
        port=29767,
        num_gpus=2,
        inference_batch_size=8,
        wandb_runs={"dagger": {"id": "smoke-dagger"}},
    )
    captured = {}

    def config(*_args, **kwargs):
        captured.update(kwargs)
        return {
            "trainer": {},
            "framework": {"qwenvl": {}},
            "datasets": {"vla_data": {
                "custom_mixtures_path": str(tmp_path / "dagger_mixture.json"),
            }},
        }

    class Process:
        pid = 123

    monkeypatch.setattr(profile, "training_config", config)
    monkeypatch.setattr(profile, "_resident_socket_dir", lambda control_dir: control_dir)
    monkeypatch.setattr(profile.subprocess, "Popen", lambda *args, **kwargs: Process())

    process, control_dir, mixture_path = profile.start_dagger_training(
        args,
        {"rounds": 2, "round_steps": 2},
        bootstrap,
        deltas,
        state,
        True,
        resume_step=500,
    )

    launch = yaml.safe_load((tmp_path / "dagger.yaml").read_text())
    assert isinstance(process, Process)
    assert control_dir == tmp_path / "dagger_control"
    assert mixture_path == tmp_path / "dagger_mixture.json"
    assert captured["steps"] == 4
    assert captured["stop_step"] is None
    assert captured["pretrained"] == state
    assert captured["resume"] is True
    assert launch["trainer"]["save_interval"] == 2
    assert launch["trainer"]["dagger_control_dir"] == str(control_dir)
    assert "stop_after_steps" not in launch["trainer"]
    assert json.loads((control_dir / "trainer.json").read_text())["pid"] == 123
    assert json.loads(mixture_path.read_text())["train"] == [
        [str(bootstrap / "lerobot"), 1.0, "profile_h8"],
        [str(deltas[0] / "lerobot"), 0.5, "profile_h8"],
        [str(deltas[1] / "lerobot"), 1.0, "profile_h8"],
    ]


def test_profile_dagger_starts_at_bootstrap_then_advances_latest_across_rounds(tmp_path, monkeypatch):
    """Bootstrap stays separate while all DAgger rounds share one trainer."""
    from scripts.tasks.humanoidbench_balance_simple import vla_profile_dagger as profile

    output = tmp_path / "output"
    collections, rollouts, training, dagger_launches, wandb_calls = [], [], [], [], []

    class Run:
        url = "https://wandb.invalid/run"

        def __init__(self):
            self.summary = {}

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def define_metric(self, *args, **kwargs):
            pass

        def log(self, metrics):
            pass

    def rollout(
        args, config, directory, *, bundle, run, count=None, seeds=None,
        resident_control_dir=None, resident_step=None,
    ):
        rollouts.append((directory, bundle, count))
        if count is not None:
            if directory == output / "bootstrap":
                (directory / "lerobot").mkdir(parents=True, exist_ok=True)
                (directory / "lerobot/task_contract.json").write_text(
                    json.dumps({"robot_type": "profile_h8"}))
            collections.append(bundle)
            return {"seconds": 1.0, "startup_seconds": 0.1, "steady_seconds": 0.8,
                    "conversion_seconds": 0.1, "labels_per_second": 1.0,
                    "generated_labels": count, "num_envs": 16, "inference_seconds": 0.1,
                    "n": 1, "mean_length": 1.0, "mean_return": 1.0,
                    "ge100": 0, "ge500": 0, "episodes": [],
                    "teacher_controlled": bundle is None}
        episodes = [{"seed": seed, "length": 1, "return": 1.0} for seed in seeds]
        return {"n": len(episodes), "mean_length": 1.0, "mean_return": 1.0,
                "ge100": 0, "ge500": 0, "episodes": episodes,
                "teacher_controlled": bundle is None}

    def train(args, bootstrap, deltas, phase, steps, boundary, pretrained, resume):
        training.append((phase, steps, boundary, pretrained, resume))
        return output / f"bundles/{phase}_{boundary}", output / f"states/{boundary}"

    class Process:
        args = ["accelerate"]

        def poll(self):
            return None

        def wait(self):
            return 0

    def start(args, budget, bootstrap, deltas, pretrained, resume, *, resume_step):
        dagger_launches.append((list(deltas), pretrained, resume))
        checkpoint_dir = output / "training/dagger/checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        for boundary in range(budget["round_steps"],
                              budget["rounds"] * budget["round_steps"] + 1,
                              budget["round_steps"]):
            (checkpoint_dir / f"steps_{boundary}_state").mkdir()
            (checkpoint_dir / f"steps_{boundary}_pytorch_model.pt").touch()
        return Process(), output / "dagger_control", output / "dagger_mixture.json"

    def bundle(_trained, _weights, destination, **_kwargs):
        (destination / "checkpoints").mkdir(parents=True, exist_ok=True)
        (destination / "checkpoints/model.pt").touch()
        return destination

    monkeypatch.setenv("WANDB_ENTITY", "entity")
    monkeypatch.setattr(sys, "argv", ["profile", "--output", str(output),
                        "--vla-python", "python", "--backbone", "qwen", "--model", "qwenoft", "--horizon", "8",
                        "--label-mode", "zero_full", "--gradient-checkpointing"])
    monkeypatch.setattr(
        profile,
        "vla_experiment_config",
        lambda model, label_mode, horizon, profile_path=None: {
            "env": {"make_kwargs": {}},
            "backend": {},
            "policy": {"action_prefix": {"mode": "none"}},
            "latency": {"method": "zero"},
        },
    )
    monkeypatch.setattr(profile, "rollout_batch", rollout)
    monkeypatch.setattr(profile, "train_segment", train)
    monkeypatch.setattr(profile, "start_dagger_training", start)
    monkeypatch.setattr(profile, "inference_bundle", bundle)
    monkeypatch.setattr(profile, "_wait_for_round_ready", lambda *args: None)
    monkeypatch.setattr(profile, "_log_periodic_eval", lambda *args: None)
    monkeypatch.setattr(profile.wandb, "init", lambda **kwargs: wandb_calls.append(kwargs) or Run())

    profile.main()

    checklist = json.loads((output / "semantic_checklist.json").read_text())
    assert checklist["model_geometry"] == "qwenoft_h8"
    assert checklist["gradient_checkpointing"] is True
    assert checklist["teacher"] == "zero50_fasttd3_h1_step_200000"
    budget = json.loads((output / "budget.json").read_text())
    assert budget["state_dim"] == 550
    assert budget["teacher_query_cadence_raw_frames"] == 1
    assert collections[:3] == [None, output / "bundles/bootstrap_5000",
                               output / "bundles/dagger_01_500"]
    assert rollouts[0] == (output / "teacher_oracle", None, None)
    assert training == [("bootstrap", 5000, 5000, None, False)]
    assert dagger_launches == [(
        [output / "dagger_01"],
        output / "bundles/bootstrap_5000/checkpoints/model.pt",
        False,
    )]
    assert json.loads((output / "result.json").read_text())["new_optimizer_steps"] == 10000
    assert [item[0] for item in rollouts[-2:]] == [
        output / "heldout/bootstrap", output / "heldout/latest"]
    assert json.loads((output / "heldout/best/episodes.json").read_text())["mean_length"] == 1.0
    assert {call["id"] for call in wandb_calls} == {
        f"{tmp_path.name}-output-bootstrap", f"{tmp_path.name}-output-dagger"}
    assert {call["name"] for call in wandb_calls} == {
        "balance-zero_full-qwenoft", "balance-zero_full-qwenoft-dagger"
    }
    assert json.loads((output / "run.json").read_text())["wandb_runs"] == {
        "bootstrap": {"id": f"{tmp_path.name}-output-bootstrap", "url": "https://wandb.invalid/run"},
        "dagger": {"id": f"{tmp_path.name}-output-dagger", "url": "https://wandb.invalid/run"},
    }
    assert (output / "dagger_control/step_500.continue.json").is_file()
    assert json.loads((output / "dagger_mixture.json").read_text())["train"][-1][0] == str(
        output / "dagger_10/lerobot")


def test_profile_train_segment_writes_random_40_40_20_and_checkpointing(tmp_path, monkeypatch):
    """Long-lived learner contract: replay weights and memory mode are explicit in generated config."""
    from scripts.tasks.humanoidbench_balance_simple import vla_profile_dagger as profile

    bootstrap = tmp_path / "bootstrap"
    (bootstrap / "lerobot").mkdir(parents=True)
    (bootstrap / "lerobot/task_contract.json").write_text(json.dumps({"robot_type": "profile_h16"}))
    mixture_path = tmp_path / "mixture.json"
    config = {
        "framework": {"qwenvl": {}},
        "datasets": {"vla_data": {"custom_mixtures_path": str(mixture_path)}},
        "trainer": {"save_interval": 2500},
    }
    monkeypatch.setenv("WANDB_ENTITY", "entity")
    monkeypatch.setattr(profile, "training_config", lambda *args, **kwargs: config)
    monkeypatch.setattr(profile.runtime, "environment", lambda: {})
    launched = {}
    monkeypatch.setattr(
        profile.runtime, "run_command",
        lambda *args, **kwargs: launched.update(env=kwargs["env"]))
    monkeypatch.setattr(profile, "inference_bundle", lambda trained, weights, destination, **kwargs: destination)
    args = SimpleNamespace(
        task="balance", model="qwenpi_v3", horizon=16, label_mode="profile_conditioned",
        smoke=False, gradient_checkpointing=False, output_dir=tmp_path, backbone=tmp_path / "backbone",
        vla_python=tmp_path / "qwenpi/bin/python", port=1234, run_id="run",
        num_gpus=2,
        wandb_runs={"dagger": {"id": "run-dagger"}},
    )
    deltas = [tmp_path / f"round_{index}" for index in range(3)]

    profile.train_segment(args, bootstrap, deltas, "dagger_03", 5000, 1500,
                          tmp_path / "state", True)

    assert config["framework"]["qwenvl"]["enable_gradient_checkpointing"] is False
    assert config["wandb_name"] == "balance-profile_conditioned-qwenpi_v3"
    assert launched["env"]["CUDA_HOME"] == str(tmp_path / "qwenpi")
    assert config["trainer"]["save_interval"] == 500
    assert config["datasets"]["vla_data"]["sequential_step_sampling"] is False
    assert json.loads(mixture_path.read_text())["train"] == [
        [str(bootstrap / "lerobot"), 1.0, "profile_h16"],
        [str(deltas[0] / "lerobot"), 0.25, "profile_h16"],
        [str(deltas[1] / "lerobot"), 0.25, "profile_h16"],
        [str(deltas[2] / "lerobot"), 1.0, "profile_h16"],
    ]


@pytest.mark.parametrize("cohort_size", [2, 4])
@pytest.mark.parametrize("collecting", [True, False])
def test_profile_cohorts_keep_action_ownership_and_finish_episode_budget(tmp_path, cohort_size, collecting):
    """Profile collector contract: overlap preserves slot actions and complete-episode selection."""
    from threading import Event
    from scripts.tasks.humanoidbench_balance_simple.vla_profile_dagger import _collect_requests

    inference_entered, simulation_entered = Event(), Event()

    class Envs:
        num_envs, directory = 4, tmp_path

        def __init__(self):
            self.state = {}

        def exchange(self, command, items):
            result = {}
            for slot, value in items:
                if command == "start":
                    self.state[slot] = (value, 0)
                else:
                    seed, step = self.state[slot]
                    assert value == (seed, step)
                    if cohort_size == 2 and seed == 100 and step == 0:
                        assert inference_entered.wait(5)
                        simulation_entered.set()
                    step += 1
                    if step == seed % 3 + 1:
                        episode = {"seed": seed, "length": step}
                        rows = [{"episode_idx": seed, "decision_step": i,
                                 "action_loss_mask": [True]} for i in range(step)] if collecting else []
                        result[slot] = ("episode", (episode, rows))
                        del self.state[slot]
                        continue
                    self.state[slot] = (seed, step)
                result[slot] = ("request", SimpleNamespace(data=self.state[slot], metadata={"slot_id": slot}))
            return result

    class Inference:
        worker_count = 2

        def predict_batch(self, observations):
            if cohort_size == 2 and observations[0].data == (102, 0):
                inference_entered.set()
                assert simulation_entered.wait(5)
            return [SimpleNamespace(action_chunk=observation.data) for observation in observations]

    envs = Envs()
    episodes, rows, metrics = _collect_requests(
        envs, Inference(), count=7 if collecting else None,
        seeds=None if collecting else list(range(100, 107)), seed_start=100,
        cohort_size=cohort_size, run=SimpleNamespace(log=lambda record: None))
    assert not envs.state
    assert len({episode["seed"] for episode in episodes}) == len(episodes)
    if collecting:
        expected = [(seed, step) for seed in range(100, 104) for step in range(seed % 3 + 1)][:7]
        assert [(row["episode_idx"], row["decision_step"]) for row in rows] == expected
        assert metrics["generated_labels"] == sum(episode["length"] for episode in episodes)
    else:
        assert sorted(episode["seed"] for episode in episodes) == list(range(100, 107))
        assert rows == []


def test_profile_worker_startup_failure_propagates_and_reaps_children(tmp_path, monkeypatch):
    """Profile process ownership: a failed child must not strand sibling samplers."""
    import multiprocessing as mp
    from scripts.tasks.humanoidbench_balance_simple.vla_profile_dagger import _ProfileEnvPool

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "6,7")
    before = {process.pid for process in mp.active_children()}
    args = SimpleNamespace(rollout_workers=2, teacher=tmp_path / "teacher.pth",
                           teacher_config=tmp_path / "teacher.yaml",
                           label_mode="profile_conditioned")
    with pytest.raises(RuntimeError, match="KeyError"):
        _ProfileEnvPool(args, {}, tmp_path, 2, collecting=False, oracle=False)
    assert {process.pid for process in mp.active_children()} == before


@pytest.mark.parametrize("delay", [0, 2])
def test_balance_dagger_labels_teacher_branch_without_changing_student(tmp_path, delay):
    """Long-lived collector contract: future expert frames, inherited-request and tail masks."""
    import copy
    import gymnasium as gym
    from latency_bench.envs.latency_wrapper import ActionLatencyWrapper

    class Native(gym.Env):
        action_space = gym.spaces.Box(-10000, 10000, (43,), dtype=np.float32)
        observation_space = gym.spaces.Box(-10000, 10000, (198,), dtype=np.float32)

        def reset(self, *, seed):
            super().reset(seed=seed)
            self.value = 0.0
            self.terminate = False
            return self._policy_observation(), {}

        def render(self):
            return np.zeros((2, 2, 3), dtype=np.uint8)

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
        return ActionLatencyWrapper(
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
    adapter = balance_dagger.BalanceDaggerCollector(config, Teacher(), tmp_path, [main], [branch])
    state = SimpleNamespace(episode_steps=[5], episode_ids=[0], seeds=[0],
                            outputs=[SimpleNamespace(action_chunk=np.full((8, 43), -5))])
    snapshot = {"images": {0: np.zeros((2, 2, 3), dtype=np.uint8)},
                "states": {0: main.unwrapped._policy_observation().copy()}}
    row = adapter._start_records(snapshot, [0], state)[0]
    mask = np.asarray(row["action_loss_mask"])
    chunk = np.asarray(row["issued_command_chunk"])
    assert main.unwrapped.value == value
    assert main.env_step_idx == frame
    if delay == 0:
        np.testing.assert_array_equal(mask, np.ones(8, dtype=bool))
        np.testing.assert_array_equal(chunk[:, 0], np.arange(-4, 4))
    else:
        np.testing.assert_array_equal(mask, [False, False, False, True, True, True, True, True])
        np.testing.assert_array_equal(chunk[:, 0], [0, 0, 0, 196, 296, 196, 296, 497])
    adapter._step([0], state)
    assert main.unwrapped.value == -5
    assert main.env_step_idx == frame + 1
    main.unwrapped.terminate = True
    config["evaluation"]["eval_max_steps"] = 7
    state.episode_steps[0] = 6
    row = adapter._start_records(snapshot, [0], state)[0]
    if delay == 0:
        assert row["action_loss_mask"] == [True] + [False] * 7
        np.testing.assert_array_equal(np.asarray(row["issued_command_chunk"])[1:], 0)
    else:
        assert row is None
    main.close(); branch.close()

    # Exercise the real collector's odd budget and episode seed progression.
    envs, branches = [env(), env()], [env(), env()]
    config["evaluation"]["eval_max_steps"] = 2 if delay == 0 else 12
    collector = balance_dagger.BalanceDaggerCollector(config, Teacher(), tmp_path, envs, branches)
    closed, resets = [], []
    pool = SimpleNamespace(
        reset_state=resets.append,
        predict_batch=lambda observations: [
            SimpleNamespace(action_chunk=np.full((8, 43), -5)) for _ in observations
        ],
        close=lambda: closed.append(True),
    )
    try:
        assert collector.collect(pool, row_budget=5, seed=900) == 5
        assert len(collector.train_rows) == 5
        assert collector.student_raw_frames == 5
        assert [row["seed"] for row in collector.train_rows] == [900, 901, 900, 901, 902 if delay == 0 else 900]
        assert [row["decision_step"] for row in collector.train_rows] == [0, 0, 1, 1, 0 if delay == 0 else 2]
        assert all(Path(row["image"]).exists() for row in collector.train_rows)
        assert closed == [True]
        assert resets == ([0, 1, 0, 1] if delay == 0 else [0, 1])
    finally:
        for item in (*envs, *branches):
            item.close()


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


@pytest.mark.parametrize("frames", [None, 3])
def test_eval_request_does_not_read_future_latency_or_results(frames):
    from collections import Counter
    from latency_bench.core.clock import EnvClock
    from latency_bench.core.types import Action, OrderingPolicy
    from latency_bench.scheduler.action_queue import ActionScheduler
    from latency_bench.scheduler.persistence import HoldPolicy
    from scripts.tasks.humanoidbench_balance_simple.vla_profile_dagger import VLAProfileRollout

    def forbidden(*args, **kwargs):
        pytest.fail("Evaluation read a future outcome")

    requests = []
    for hidden_future in [2, 6]:
        rollout = object.__new__(VLAProfileRollout)
        rollout.initial_sampler = None
        rollout.action_prefix = ({"mode": "none"} if frames is None else
                                 {"mode": "profile_mean", "frames": frames, "action_horizon": 8, "action_dim": 43})
        rollout.chunk = SimpleNamespace(horizon=8)
        rollout.latency = SimpleNamespace(
            env_step_idx=0,
            profile_sampler=hidden_future,
            result_timeline=SimpleNamespace(release_arrived=forbidden),
            _next_latency_sample=forbidden,
            worker_capacity=SimpleNamespace(can_submit=lambda: True),
            action_scheduler=ActionScheduler(
                EnvClock(env_fps=50, obs_fps=50), Action(np.ones(43)),
                HoldPolicy(), OrderingPolicy.LATEST_READY, hold_last_chunk_action=True,
            ),
        )
        observation = {"obs": np.zeros(198), "scheduled_chunk": np.zeros((8, 43)),
                       "scheduled_index": np.zeros(8)}
        rollout.env = SimpleNamespace(
            reset=lambda seed: (observation, {}),
            unwrapped=SimpleNamespace(render=lambda: np.zeros((8, 8, 3), dtype=np.uint8)),
        )
        generator = rollout.requests(None, 1, 0, None, {"policy": {"action_prefix": rollout.action_prefix}},
                                     collect=False, oracle=False, writer=None, timings=Counter())
        requests.append(next(generator))
        generator.close()
    for key in requests[0].metadata:
        np.testing.assert_array_equal(requests[0].metadata[key], requests[1].metadata[key])
    if frames is None:
        assert "action_prefix" not in requests[0].metadata
        assert "action_prefix_mask" not in requests[0].metadata
    else:
        np.testing.assert_array_equal(requests[0].metadata["action_prefix"][:frames], 1)
        assert requests[0].metadata["action_prefix_mask"].sum() == frames


@pytest.mark.parametrize("actual_delay", [1, 5])
def test_fixed_k_collection_aligns_student_teacher_and_export(tmp_path, monkeypatch, actual_delay):
    """Long-lived collection contract: fixed inputs/labels, sampled execution delay."""
    from collections import Counter
    import torch
    from latency_bench.core.clock import EnvClock
    from latency_bench.core.types import Action, OrderingPolicy
    from latency_bench.scheduler.action_queue import ActionScheduler
    from latency_bench.scheduler.persistence import HoldPolicy
    from scripts.tasks.humanoidbench_balance_simple import vla_profile_dagger as profile

    class Physics:
        steps = 0

        def copy_balance_state_from(self, source):
            self.steps = 0

        def step(self, action):
            self.steps += 1
            return np.zeros(198), 0.0, self.steps == 5, False, {}

    class Teacher:
        def empirical_obs_normalizer(self, physical, update_stats):
            return physical

        def actor(self, physical):
            return torch.full((1, 43), 0.5)

    rollout = object.__new__(profile.VLAProfileRollout)
    rollout.initial_sampler = None
    rollout.action_prefix = {"mode": "profile_mean", "frames": 3, "action_horizon": 8, "action_dim": 43}
    rollout.chunk = SimpleNamespace(
        horizon=8, native_action_space=SimpleNamespace(low=-np.ones(43), high=np.ones(43)),
        env=SimpleNamespace(action=lambda value: value, reverse_action=lambda value: value),
    )
    rollout.latency = SimpleNamespace(
        env_step_idx=0, worker_capacity=SimpleNamespace(can_submit=lambda: True),
        action_scheduler=ActionScheduler(EnvClock(env_fps=50, obs_fps=50), Action(np.ones(43)),
                                         HoldPolicy(), OrderingPolicy.LATEST_READY),
    )
    observation = {"obs": np.zeros(198), "scheduled_chunk": np.zeros((8, 43)),
                   "scheduled_index": np.zeros(8), "replay_clock": np.array([[0, 0, -1, -1]])}
    rollout.env = SimpleNamespace(
        reset=lambda seed: (observation, {}),
        unwrapped=SimpleNamespace(render=lambda: np.zeros((8, 8, 3), dtype=np.uint8)),
        step=lambda action: (observation, 0.0, True, False, {
            "command_admitted": True, "latency_raw_frames": actual_delay,
            "worker_service_raw_frames": actual_delay, "latency_ms": actual_delay * 20,
        }),
    )
    rollout.prediction = SimpleNamespace(unwrapped=Physics())
    monkeypatch.setattr(profile, "_balance_row_for_image_path", lambda *args: {})
    generator = rollout.requests(Teacher(), 1, 0, tmp_path, {"env": {}}, collect=True,
                                 oracle=False, writer=SimpleNamespace(write=lambda *args: None),
                                 timings=Counter())
    request = next(generator)
    with pytest.raises(StopIteration) as done:
        generator.send(np.zeros((8, 43)))
    episode, rows = done.value.value
    row = rows[0]
    np.testing.assert_array_equal(request.metadata["action_prefix"][:3], np.ones((3, 43)))
    np.testing.assert_array_equal(row["action_prefix_mask"], request.metadata["action_prefix_mask"])
    np.testing.assert_array_equal(np.asarray(row["issued_command_chunk"])[:3], request.metadata["action_prefix"][:3])
    assert row["action_loss_mask"] == [False, False, False, True, True, False, False, False]
    assert row["latency_raw_frames"] == actual_delay
    assert episode["request_audit"][0]["prefix_frames"] == 3


def test_legacy_training_run_cannot_reuse_cached_evaluations(tmp_path, monkeypatch):
    """Long-lived migration contract: old data/history cannot enter a fixed-K run."""
    from scripts.tasks.humanoidbench_balance_simple import vla_profile_dagger as profile

    (tmp_path / "environment.json").write_text(json.dumps({"policy": {}}))
    monkeypatch.setattr(sys, "argv", [
        "profile", "--output", str(tmp_path), "--vla-python", "python", "--backbone", "qwen",
        "--horizon", "16", "--label-mode", "profile_conditioned", "--resume", "--no-gradient-checkpointing",
    ])
    with pytest.raises(ValueError, match="Resume requires a fixed-K run"):
        profile.main()


def test_prepare_starvla_lerobot_dataset_converts_from_raw_export(monkeypatch, tmp_path):
    from scripts.rollout_data import prepare_starvla_lerobot_dataset as prepare

    raw_export = tmp_path / "raw"
    raw_export.mkdir()
    (raw_export / "metadata.json").write_text(json.dumps({"env_name": "demon_attack"}), encoding="utf-8")
    (raw_export / "train.parquet").write_bytes(b"train")
    (raw_export / "val.parquet").write_bytes(b"val")

    calls = []

    def fake_convert(raw_export_dir, output_dir, **kwargs):
        calls.append((raw_export_dir, output_dir, kwargs))
        (output_dir / "meta").mkdir(parents=True)
        (output_dir / "meta" / "info.json").write_text("{}", encoding="utf-8")
        (output_dir.with_name(f"{output_dir.name}__val") / "meta").mkdir(parents=True)
        return {"dataset_name": output_dir.name}

    monkeypatch.setattr(prepare, "convert_canonical_rollout_to_lerobot", fake_convert)

    result = prepare.main([
        str(raw_export),
        str(tmp_path / "lerobot"),
        "demon_attack_train",
    ])

    assert result == 0
    assert calls == [
        (
            raw_export,
            tmp_path / "lerobot" / "demon_attack_train",
            {"force": True, "action_carrier": "native"},
        )
    ]


def test_prepare_starvla_lerobot_dataset_uploads_converted_root(monkeypatch, tmp_path):
    from scripts.rollout_data import prepare_starvla_lerobot_dataset as prepare

    lerobot_root = tmp_path / "lerobot"
    (lerobot_root / "demon_attack_train").mkdir(parents=True)
    (lerobot_root / "demon_attack_train__val").mkdir()

    created = {}
    uploaded = {}

    class FakeApi:
        def create_repo(self, **kwargs):
            created.update(kwargs)

        def upload_folder(self, **kwargs):
            uploaded.update(kwargs)

    monkeypatch.setattr(prepare, "HfApi", lambda: FakeApi())
    monkeypatch.setenv("HF_TOKEN", "env-token")

    prepare.push_lerobot_dataset(
        lerobot_root=lerobot_root,
        data_mix="demon_attack_train",
        repo_id="user/demon_attack_train_lerobot",
        private=True,
    )

    for request in (created, uploaded):
        assert request["repo_id"] == "user/demon_attack_train_lerobot"
        assert request["repo_type"] == "dataset"
        assert request["token"] == "env-token"
    assert created["private"] is True
    assert uploaded["folder_path"] == str(lerobot_root)
    assert set(uploaded["allow_patterns"]) == {
        "demon_attack_train/**", "demon_attack_train__val/**",
    }
