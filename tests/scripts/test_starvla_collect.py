import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml

from scripts.starvla import balance_dagger, collect


def test_balance_dagger_model_geometry_and_latency_recipes():
    """Long-lived experiment contract: QwenPI is H16/profile; QwenOFT baseline is H8/zero."""
    from latency_bench.data.starvla_tasks import task_spec
    from scripts.tasks.humanoidbench_balance_simple import vla_profile_dagger as profile

    qwenpi = profile.vla_experiment_config("qwenpi_v3", "profile_executed")
    qwenoft = profile.vla_experiment_config("qwenoft", "zero_full")
    assert qwenpi["training"]["ppo"]["chunk_horizon"] == 16
    assert qwenpi["training"]["ppo"]["active_heads"] == [0, 16]
    assert qwenpi["latency"]["method"] == "temporal"
    assert qwenpi["latency"]["profile_worker_slot"] == 0
    assert qwenpi["latency"]["add_latency_info"] is True
    assert "/qwenpi_v3/" in qwenpi["latency"]["profile_path"]
    assert qwenoft["training"]["ppo"]["chunk_horizon"] == 8
    assert qwenoft["training"]["ppo"]["active_heads"] == [0, 8]
    assert qwenoft["latency"]["method"] == "zero"
    assert (task_spec("balance_profile_h16").action_horizon,
            task_spec("balance_profile_h16").state_dim) == (16, 902)
    assert (task_spec("balance_profile_h8").action_horizon,
            task_spec("balance_profile_h8").state_dim) == (8, 550)


def test_profile_teacher_suffix_stops_at_last_executed_head():
    """Executed-only relabeling does no work past its replay clock."""
    from collections import Counter

    import torch
    from scripts.tasks.humanoidbench_balance_simple.vla_profile_dagger import VLAProfileRollout

    class Physics:
        frames = 0

        def step(self, action):
            self.frames += 1
            return np.full(198, self.frames, dtype=np.float32), 0.0, False, False, {}

    class Teacher:
        queries = 0

        def empirical_obs_normalizer(self, observation, update_stats):
            return observation

        def actor(self, observation):
            self.queries += 1
            return torch.full((1, 16 * 43), float(self.queries))

    physics, teacher = Physics(), Teacher()
    rollout = object.__new__(VLAProfileRollout)
    rollout.prediction = SimpleNamespace(unwrapped=physics)
    rollout.chunk = SimpleNamespace(
        native_action_space=SimpleNamespace(low=-np.ones(43), high=np.ones(43)),
        env=SimpleNamespace(action=lambda action: action),
    )
    targets, valid, timings = np.zeros((16, 43)), np.zeros(16, dtype=bool), Counter()

    rollout._teacher_suffix(teacher, np.zeros(198), 100, 5, 11, targets, valid, timings)

    assert valid.tolist() == [False] * 5 + [True] * 6 + [False] * 5
    assert teacher.queries == timings["teacher_queries"] == 2
    assert physics.frames == timings["teacher_branch_frames"] == 6
    np.testing.assert_array_equal(targets[5:10], np.ones((5, 43)))
    np.testing.assert_array_equal(targets[10], np.full(43, 2.0))


def test_qwenpi_flow_loss_ignores_masked_teacher_heads(monkeypatch):
    """Long-lived QwenPI contract: unlabelled chunk heads affect neither loss nor gradient."""
    pytest.importorskip("accelerate")
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "third_party/starVLA"))
    import torch
    from omegaconf import OmegaConf
    from starVLA.model.modules.action_model.LayerwiseFM_ActionHeader import LayerwiseFlowmatchingActionHead

    head = LayerwiseFlowmatchingActionHead(OmegaConf.create({"framework": {"action_model": {
        "diffusion_model_cfg": {"num_layers": 1, "input_embedding_dim": 32,
                                "cross_attention_dim": 32, "num_attention_heads": 1,
                                "attention_head_dim": 32, "output_dim": 32},
        "action_hidden_dim": 32, "hidden_size": 32, "action_dim": 3, "action_env_dim": 2,
        "action_horizon": 4, "state_dim": 2, "num_inference_timesteps": 2,
        "num_target_vision_tokens": 1, "add_pos_embed": False,
        "noise_beta_alpha": 1.5, "noise_beta_beta": 1.0, "noise_s": 0.999,
        "num_timestep_buckets": 1000,
    }}})).eval()
    actions = torch.randn(2, 4, 3, requires_grad=True)
    state = torch.randn(2, 1, 2)
    hidden = [torch.randn(2, 3, 32)]
    mask = torch.tensor([[True, False, False, False], [False, True, True, True]])
    torch.manual_seed(77)
    loss = head(hidden, actions, state, action_valid_mask=mask)
    loss.backward()
    assert torch.count_nonzero(actions.grad[~mask]).item() == 0
    changed = actions.detach().clone()
    changed[~mask] = 10000
    torch.manual_seed(77)
    torch.testing.assert_close(head(hidden, changed, state, action_valid_mask=mask), loss)


def test_profile_queue_reserves_controllers_between_gpu_phases():
    """Long-lived Balance queue ownership: CPU-only controllers still own their GPUs."""
    from scripts.tasks.humanoidbench_balance_simple.queue_profile import reserved_gpus

    devices = [f'<gpu><uuid>GPU-{i}</uuid><processes>'
               + ('<process_info><pid>123</pid></process_info>' if i == 4 else '')
               + '</processes></gpu>' for i in range(8)]
    xml = '<nvidia_smi_log>' + ''.join(devices) + '</nvidia_smi_log>'
    assert reserved_gpus(xml, [b'CUDA_VISIBLE_DEVICES=6,7\0', b'CUDA_VISIBLE_DEVICES=GPU-5\0']) == {'4', '5', '6', '7'}
    assert reserved_gpus(xml, []) == {'4'}


def test_profile_queue_runs_qwenpi_h16_recipes_after_smoke(tmp_path, monkeypatch):
    """Long-lived contract: exact GPU recipes pass smoke and share profile bootstrap rows."""
    from scripts.tasks.humanoidbench_balance_simple import queue_profile as queue

    launched, environments = [], []

    class Process:
        pid = 123

        def __init__(self, command, **kwargs):
            self.mode = command[command.index("--label-mode") + 1]
            self.smoke = "--smoke" in command
            environments.append(kwargs["env"])
            launched.append((self.mode, self.smoke, "--no-gradient-checkpointing" in command,
                             command[command.index("--model") + 1],
                             "--collect-bootstrap-only" in command,
                             "--bootstrap-data" in command,
                             "--rollout-overlap" in command, kwargs["env"]["WANDB_MODE"],
                             kwargs["env"]["CUDA_VISIBLE_DEVICES"]))

        def poll(self):
            return 0

    monkeypatch.setattr(queue.sys, "argv", ["queue", "--output", str(tmp_path), "--vla-python", "python",
                                           "--qwen-backbone", "qwen"])
    monkeypatch.setattr(queue.subprocess, "Popen", Process)
    monkeypatch.setattr(queue, "busy_gpus", lambda: set())
    monkeypatch.setattr(queue.time, "sleep", lambda seconds: None)
    assert queue.main() == 0
    assert launched == [
        ("zero_full", True, True, "qwenpi_v3", False, False, True, "disabled", "2,3"),
        ("profile_suffix", True, True, "qwenpi_v3", False, False, True, "disabled", "4,5"),
        ("profile_executed", True, True, "qwenpi_v3", False, False, True, "disabled", "6,7"),
        ("zero_full", False, True, "qwenpi_v3", False, False, True, "online", "2,3"),
        ("profile_executed", False, True, "qwenpi_v3", True, False, True, "online", "4,5"),
        ("profile_suffix", False, True, "qwenpi_v3", False, True, True, "online", "4,5"),
        ("profile_executed", False, True, "qwenpi_v3", False, True, True, "online", "6,7"),
    ]
    assert all(str(queue.ROOT / "third_party/humanoid-bench") in env["PYTHONPATH"].split(":")
               for env in environments)
    jobs = json.loads((tmp_path / "queue.json").read_text())["jobs"]
    assert [jobs[mode]["status"] for mode, _ in queue.JOBS] == ["completed"] * 3


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


def test_profile_dagger_starts_at_bootstrap_then_advances_latest_across_rounds(tmp_path, monkeypatch):
    """Bootstrap is fresh; DAgger rollout and full training state use latest."""
    from scripts.tasks.humanoidbench_balance_simple import vla_profile_dagger as profile

    output = tmp_path / "output"
    collections, rollouts, training, wandb_calls = [], [], [], []

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

    def rollout(args, config, directory, *, bundle, run, count=None, seeds=None):
        rollouts.append((directory, bundle, count))
        if count is not None:
            collections.append(bundle)
            return {"seconds": 1.0, "startup_seconds": 0.1, "steady_seconds": 0.8,
                    "conversion_seconds": 0.1, "labels_per_second": 1.0,
                    "generated_labels": count, "num_envs": 16, "inference_seconds": 0.1}
        episodes = [{"seed": seed, "length": 1, "return": 1.0} for seed in seeds]
        return {"n": len(episodes), "mean_length": 1.0, "mean_return": 1.0,
                "ge100": 0, "ge500": 0, "episodes": episodes}

    def train(args, bootstrap, deltas, phase, steps, boundary, pretrained, resume):
        training.append((phase, steps, boundary, pretrained, resume))
        return output / f"bundles/{phase}_{boundary}", output / f"states/{boundary}"

    monkeypatch.setenv("WANDB_ENTITY", "entity")
    monkeypatch.setattr(sys, "argv", ["profile", "--output", str(output),
                        "--vla-python", "python", "--backbone", "qwen", "--model", "qwenoft",
                        "--label-mode", "zero_full", "--gradient-checkpointing"])
    monkeypatch.setattr(profile, "vla_experiment_config", lambda model, label_mode: {
        "env": {"make_kwargs": {}}, "backend": {}, "latency": {"method": "zero"}})
    monkeypatch.setattr(profile, "rollout_batch", rollout)
    monkeypatch.setattr(profile, "train_segment", train)
    monkeypatch.setattr(profile, "_log_periodic_eval", lambda *args: None)
    monkeypatch.setattr(profile.wandb, "init", lambda **kwargs: wandb_calls.append(kwargs) or Run())

    profile.main()

    checklist = json.loads((output / "semantic_checklist.json").read_text())
    assert checklist["model_geometry"] == "qwenoft_h8"
    assert checklist["gradient_checkpointing"] is True
    assert json.loads((output / "budget.json").read_text())["state_dim"] == 550
    assert collections[:3] == [None, output / "bundles/bootstrap_5000",
                               output / "bundles/dagger_01_500"]
    assert training[:3] == [
        ("bootstrap", 5000, 5000, None, False),
        ("dagger_01", 5000, 500, output / "bundles/bootstrap_5000/checkpoints/model.pt", False),
        ("dagger_02", 5000, 1000, output / "states/500", True),
    ]
    assert json.loads((output / "result.json").read_text())["new_optimizer_steps"] == 10000
    assert {call["id"] for call in wandb_calls} == {
        f"{tmp_path.name}-output-bootstrap", f"{tmp_path.name}-output-dagger"}
    assert {call["name"] for call in wandb_calls} == {
        "balance-zero_full-qwenoft", "balance-zero_full-qwenoft-dagger"
    }
    assert json.loads((output / "run.json").read_text())["wandb_runs"] == {
        "bootstrap": {"id": f"{tmp_path.name}-output-bootstrap", "url": "https://wandb.invalid/run"},
        "dagger": {"id": f"{tmp_path.name}-output-dagger", "url": "https://wandb.invalid/run"},
    }
    pending_bundle = output / "bundles/dagger_09_4500"
    pending_state = output / "training/dagger/checkpoints/steps_4500_state"
    pending_state.mkdir(parents=True)
    (pending_state.parent / "steps_4500_pytorch_model.pt").touch()
    (pending_bundle / "checkpoints").mkdir(parents=True)
    (pending_bundle / "checkpoints/model.pt").touch()
    pending_collection = output / "dagger_09/episodes.json"
    pending_collection.parent.mkdir()
    pending_collection.write_text(json.dumps({
        "seconds": 1.0, "startup_seconds": 0.1, "steady_seconds": 0.8,
        "conversion_seconds": 0.1, "labels_per_second": 1.0,
        "generated_labels": 16000, "num_envs": 16, "inference_seconds": 0.1,
    }))
    history_path = output / "history.json"
    profile.write_json(history_path, json.loads(history_path.read_text())[:9])
    rollouts.clear()
    training.clear()
    monkeypatch.setattr(sys, "argv", [*sys.argv, "--resume"])
    profile.main()

    assert rollouts[0] == (output / "evaluation/dagger_09_4500", pending_bundle, None)
    assert training == [("dagger_10", 5000, 5000, pending_state, True)]


def test_profile_bootstrap_reuse_changes_only_the_supervision_mask(tmp_path, monkeypatch):
    """Long-lived mask-ablation contract: rows and targets are shared, only the mask differs."""
    from scripts.tasks.humanoidbench_balance_simple import vla_profile_dagger as profile

    source = tmp_path / "source"
    source.mkdir()
    row = {
        "issued_command_chunk": [[[float(head)] * 43 for head in range(16)]],
        "teacher_valid_mask": [[False] * 4 + [True] * 12],
        "executed_action_mask": [[False] * 4 + [True] * 3 + [False] * 9],
        "action_valid_mask": [[False] * 4 + [True] * 3 + [False] * 9],
    }
    table = pa.table(row)
    pq.write_table(table, source / "train.parquet")
    pq.write_table(table, source / "val.parquet")
    (source / "metadata.json").write_text(json.dumps({
        "train_rows": 1, "val_rows": 1, "label_provenance": {
            "semantics": "profile_executed",
            "teacher_valid_head_counts": np.asarray(row["teacher_valid_mask"]).sum(0).tolist(),
            "executed_head_counts": np.asarray(row["executed_action_mask"]).sum(0).tolist(),
        }
    }))
    monkeypatch.setattr(profile, "convert_canonical_rollout_to_lerobot", lambda *args, **kwargs: None)
    monkeypatch.setattr(profile, "prepare_lerobot", lambda *args, **kwargs: None)

    outputs = {}
    for mode in ("profile_suffix", "profile_executed"):
        output = tmp_path / mode
        profile.reuse_bootstrap(SimpleNamespace(label_mode=mode, horizon=16,
                                normalization_source=None), {}, source, output)
        outputs[mode] = pq.read_table(output / "train.parquet").to_pylist()[0]

    assert outputs["profile_suffix"]["issued_command_chunk"] == outputs["profile_executed"]["issued_command_chunk"]
    assert outputs["profile_suffix"]["action_valid_mask"] == row["teacher_valid_mask"][0]
    assert outputs["profile_executed"]["action_valid_mask"] == row["executed_action_mask"][0]


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
    }
    monkeypatch.setenv("WANDB_ENTITY", "entity")
    monkeypatch.setattr(profile, "training_config", lambda *args, **kwargs: config)
    monkeypatch.setattr(profile, "environment", lambda: {})
    monkeypatch.setattr(profile, "run_command", lambda *args, **kwargs: None)
    monkeypatch.setattr(profile, "inference_bundle", lambda trained, weights, destination, **kwargs: destination)
    args = SimpleNamespace(
        task="balance", model="qwenpi_v3", horizon=16, label_mode="profile_suffix",
        smoke=False, gradient_checkpointing=False, output_dir=tmp_path, backbone=tmp_path / "backbone",
        vla_python="python", port=1234, run_id="run",
        wandb_runs={"dagger": {"id": "run-dagger"}},
    )
    deltas = [tmp_path / f"round_{index}" for index in range(3)]

    profile.train_segment(args, bootstrap, deltas, "dagger_03", 5000, 1500,
                          tmp_path / "state", True)

    assert config["framework"]["qwenvl"]["enable_gradient_checkpointing"] is False
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
                                 "action_valid_mask": [True]} for i in range(step)] if collecting else []
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
                           label_mode="profile_executed")
    with pytest.raises(RuntimeError, match="KeyError"):
        _ProfileEnvPool(args, {}, tmp_path, 2, collecting=False, oracle=False)
    assert {process.pid for process in mp.active_children()} == before



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
