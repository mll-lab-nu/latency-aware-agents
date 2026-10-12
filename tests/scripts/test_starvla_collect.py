import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from latency_bench.envs.tasks.humanoidbench import profile_collection


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
    from latency_bench.envs.tasks.humanoidbench.profile_collection import VLAProfileRollout

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
    row = profile_collection._balance_row_for_image_path(
        config, Path("/tmp/profile-frame.png"), np.zeros(198), np.zeros(43),
        episode=7, step=11, seed=17, split="train")

    assert row["image"] == "/tmp/profile-frame.png"
    assert row["episode_idx"] == 7
    assert row["decision_step"] == 11
    assert "rgb" not in row


def test_profile_inference_uses_the_explicit_local_backbone(tmp_path):
    from scripts.tasks.humanoidbench_balance_simple import vla_profile_dagger as profile

    config = profile.vla_experiment_config("qwenpi_v3", "zero_full", horizon=16)
    from latency_bench.envs.tasks.humanoidbench.profile_collection import policy_config
    result = policy_config(
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
        output_dir=tmp_path,
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
    from latency_bench.envs.tasks.humanoidbench import profile_collection as profile

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
        collect_bootstrap_only=True, collection_seed=0,
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
    from latency_bench.envs.tasks.humanoidbench.profile_collection import _collect_requests

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
    from latency_bench.envs.tasks.humanoidbench.profile_collection import _ProfileEnvPool

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "6,7")
    before = {process.pid for process in mp.active_children()}
    args = SimpleNamespace(rollout_workers=2, teacher=tmp_path / "teacher.pth",
                           teacher_config=tmp_path / "teacher.yaml",
                           label_mode="profile_conditioned")
    with pytest.raises(RuntimeError, match="KeyError"):
        _ProfileEnvPool(args, {}, tmp_path, 2, collecting=False, oracle=False)
    assert {process.pid for process in mp.active_children()} == before






@pytest.mark.parametrize("frames", [None, 3])
def test_eval_request_does_not_read_future_latency_or_results(frames):
    from collections import Counter
    from latency_bench.core.clock import EnvClock
    from latency_bench.core.types import Action, OrderingPolicy
    from latency_bench.scheduler.action_queue import ActionScheduler
    from latency_bench.scheduler.persistence import HoldPolicy
    from latency_bench.envs.tasks.humanoidbench.profile_collection import VLAProfileRollout

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
    from latency_bench.envs.tasks.humanoidbench import profile_collection as profile

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


@pytest.mark.parametrize("prefix_mode", ["fixed_frames", "profile_mean"])
def test_public_balance_collection_resolves_action_prefix(tmp_path, monkeypatch, prefix_mode):
    """Long-lived collection boundary: user prefix config fixes the H16 label contract."""
    from latency_bench.data.rollout import collect_rollouts
    from scripts.tasks.humanoidbench_balance_simple.vla_profile_dagger import vla_experiment_config

    config = vla_experiment_config("qwenoft", "zero_full", horizon=16)
    (tmp_path / "checkpoint.pth").write_bytes(b"teacher fixture")
    (tmp_path / "teacher.yaml").write_text("{}\n")
    profile = tmp_path / "profile.json"
    profile.write_text(json.dumps({"summary": {"mean_ms": 40}}))
    config["policy"] = {
        "type": "sample_factory", "source": "local", "root": str(tmp_path),
        "checkpoint": "checkpoint.pth", "model_config": "teacher.yaml",
        "action_prefix": ({"mode": "fixed_frames", "frames": 2} if prefix_mode == "fixed_frames"
                          else {"mode": "profile_mean", "profile_path": str(profile)}),
    }
    config["executor"].update(inference_devices=["cpu"], rollout_workers=1,
                               inference_batch_size=1, rollout_overlap=False)
    config["logging"]["output_dir"] = str(tmp_path / "dataset")
    config["collection"] = {"label_mode": "profile_conditioned", "action_horizon": 16,
                            "row_budget": 1, "teacher_revision": "fixture"}

    def collect_batch(args, resolved, output, **kwargs):
        prefix = resolved["policy"]["action_prefix"]
        assert (prefix["frames"], prefix["action_horizon"], prefix["action_dim"]) == (2, 16, 43)
        output.mkdir()
        (output / "metadata.json").write_text('{"train_rows": 1}')

    monkeypatch.setattr(profile_collection, "rollout_batch", collect_batch)
    assert collect_rollouts(config)["metadata"]["train_rows"] == 1
