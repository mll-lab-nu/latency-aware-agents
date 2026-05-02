from __future__ import annotations

import sys
import types
from argparse import Namespace
from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest
import torch

from modeling.demon_attack_ppo import (
    DemonAttackPpoObservationConfig,
    DemonAttackPpoTorchObservationPipeline,
    latency_channel_value,
)
from training.demon_attack_ppo.cli import build_parser
from training.demon_attack_ppo.env_batch import QueueLatencyWrapper, SyncDemonAttackBatch
from training.demon_attack_ppo.observation_pipeline import ObservationPipeline


class _FakeAtariEnv(gym.Env):
    metadata = {}

    def __init__(self):
        super().__init__()
        self.action_space = gym.spaces.Discrete(10)
        self.observation_space = gym.spaces.Box(0, 255, shape=(4, 4), dtype=np.uint8)
        self.applied_actions: list[int] = []
        self.step_count = 0

    def reset(self, **kwargs):
        self.step_count = 0
        return np.full((4, 4), 3, dtype=np.uint8), {}

    def step(self, action):
        self.applied_actions.append(int(action))
        self.step_count += 1
        obs = np.full((4, 4), self.step_count, dtype=np.uint8)
        return obs, 1.0, False, False, {"base_action": int(action)}


class _ScriptedDemonAttackEnv(gym.Env):
    metadata = {}

    def __init__(self, steps):
        super().__init__()
        self.action_space = gym.spaces.Discrete(10)
        self.observation_space = gym.spaces.Box(0, 255, shape=(4, 4), dtype=np.uint8)
        self.steps = list(steps)
        self.index = 0
        self.reset_count = 0

    def reset(self, **kwargs):
        self.reset_count += 1
        return np.full((4, 4), self.reset_count, dtype=np.uint8), {}

    def step(self, action):
        if self.index >= len(self.steps):
            raise AssertionError("script exhausted")
        reward, terminated, truncated, info = self.steps[self.index]
        self.index += 1
        obs = np.full((4, 4), self.index + 10, dtype=np.uint8)
        return obs, float(reward), bool(terminated), bool(truncated), dict(info)


def _args() -> Namespace:
    return Namespace(
        screen_size=4,
        stack_frames=3,
        obs_mode="latency_channel",
        frame_skip_values="0,4",
    )


def _to_hwc(obs: torch.Tensor) -> np.ndarray:
    return obs.permute(0, 2, 3, 1).cpu().numpy()


def test_cli_defaults_match_demon_attack_training_entrypoint():
    args = build_parser().parse_args([])

    assert args.env_id == "ALE/DemonAttack-v5"
    assert args.screen_size == 84
    assert args.noop_max == 30
    assert args.action_repeat == 1
    assert args.repeat_action_probability == 0.0
    assert args.obs_mode == "latency_channel"
    assert args.frame_skip_values == "0,1,2,3,4,5"
    assert args.full_action_space is False


def test_cli_accepts_positive_action_repeat_and_rejects_zero(capsys):
    args = build_parser().parse_args(["--action-repeat", "4"])

    assert args.action_repeat == 4

    with pytest.raises(SystemExit):
        build_parser().parse_args(["--action-repeat", "0"])

    assert ">= 1" in capsys.readouterr().err


def test_checkpoint_schedule_detects_rollout_boundary_crossing():
    from training.demon_attack_ppo.ppo_loop import (
        _advance_checkpoint_step,
        _checkpoint_due,
    )

    next_checkpoint_step = 50_000

    assert _checkpoint_due(global_step=32_768, next_checkpoint_step=next_checkpoint_step) is False
    assert _checkpoint_due(global_step=65_536, next_checkpoint_step=next_checkpoint_step) is True
    assert _advance_checkpoint_step(
        next_checkpoint_step=next_checkpoint_step,
        global_step=65_536,
        checkpoint_freq=50_000,
    ) == 100_000


def test_fs0_train_script_uses_lzj_learnable_regime():
    script = Path("scripts/bash_scripts/train_demon_attack_fs0.sh").read_text()

    assert 'N_ENVS="${N_ENVS:-8}"' in script
    assert 'N_STEPS="${N_STEPS:-128}"' in script
    assert "--batch-size 256" in script
    assert "--action-repeat \"${ACTION_REPEAT:-4}\"" in script
    assert "--terminal-on-life-loss" in script
    assert "--gamma 0.99" in script
    assert "--gae-lambda 0.95" in script
    assert "--clip-coef 0.1" in script
    assert "--clip-coef-final 0.1" in script
    assert "--ent-coef 0.01" in script
    assert "--checkpoint-freq 500000" in script


def test_make_demon_attack_env_uses_action_repeat_for_atari_wrapper(monkeypatch):
    from training.demon_attack_ppo.env_batch import make_demon_attack_env

    gym_make_kwargs = []
    atari_wrapper_kwargs = []

    class FakeAtariWrapper(gym.Wrapper):
        def __init__(self, env, **kwargs):
            super().__init__(env)
            atari_wrapper_kwargs.append(kwargs)

    def fake_make(*args, **kwargs):
        gym_make_kwargs.append(kwargs)
        return _FakeAtariEnv()

    atari_wrappers = types.ModuleType("stable_baselines3.common.atari_wrappers")
    atari_wrappers.AtariWrapper = FakeAtariWrapper
    common = types.ModuleType("stable_baselines3.common")
    stable_baselines3 = types.ModuleType("stable_baselines3")
    monkeypatch.setitem(sys.modules, "stable_baselines3", stable_baselines3)
    monkeypatch.setitem(sys.modules, "stable_baselines3.common", common)
    monkeypatch.setitem(
        sys.modules,
        "stable_baselines3.common.atari_wrappers",
        atari_wrappers,
    )
    monkeypatch.setattr("training.demon_attack_ppo.env_batch.gym.make", fake_make)

    env = make_demon_attack_env(
        seed=0,
        frame_skip_values=(0,),
        action_repeat=4,
    )

    assert isinstance(env, QueueLatencyWrapper)
    assert gym_make_kwargs[0]["frameskip"] == 1
    assert atari_wrapper_kwargs[0]["frame_skip"] == 4


def test_sync_batch_forwards_action_repeat_to_env_builder(monkeypatch):
    calls = []

    def fake_make_demon_attack_env(**kwargs):
        calls.append(kwargs)
        return _FakeAtariEnv()

    monkeypatch.setattr(
        "training.demon_attack_ppo.env_batch.make_demon_attack_env",
        fake_make_demon_attack_env,
    )

    envs = SyncDemonAttackBatch(
        n_envs=2,
        seed=10,
        frame_skip_values=(0,),
        device="cpu",
        action_repeat=4,
    )

    try:
        assert [call["action_repeat"] for call in calls] == [4, 4]
    finally:
        envs.close()


def test_ppo_loop_forwards_action_repeat_to_sync_batch(monkeypatch, tmp_path):
    from training.demon_attack_ppo import ppo_loop

    class StopAfterEnvConstruction(Exception):
        pass

    captured_kwargs = {}

    class FakeSyncDemonAttackBatch:
        def __init__(self, **kwargs):
            captured_kwargs.update(kwargs)

        def reset(self):
            raise StopAfterEnvConstruction

    monkeypatch.setattr(ppo_loop, "SyncDemonAttackBatch", FakeSyncDemonAttackBatch)
    args = build_parser().parse_args(
        [
            "--train-steps",
            "1",
            "--n-envs",
            "1",
            "--n-steps",
            "1",
            "--batch-size",
            "1",
            "--n-epochs",
            "1",
            "--device",
            "cpu",
            "--save",
            str(tmp_path / "model.pt"),
        ]
    )
    args.action_repeat = 4

    with pytest.raises(StopAfterEnvConstruction):
        ppo_loop.train(args, wandb_module=object())

    assert captured_kwargs["action_repeat"] == 4


def test_ppo_loop_train_start_event_records_training_semantics(monkeypatch, tmp_path):
    from training.demon_attack_ppo import ppo_loop

    events = []

    class FakeSyncDemonAttackBatch:
        n_actions = 2

        def __init__(self, **kwargs):
            self.n_envs = kwargs["n_envs"]

        def reset(self):
            return torch.zeros((self.n_envs, 1, 1), dtype=torch.uint8)

        def frame_skips(self):
            return np.zeros(self.n_envs, dtype=np.int64)

        def step(self, action_np, profiler=None):
            return (
                np.zeros(self.n_envs, dtype=np.float32),
                np.zeros(self.n_envs, dtype=np.bool_),
                [{} for _ in range(self.n_envs)],
                torch.zeros((self.n_envs, 1, 1), dtype=torch.uint8),
            )

        def close(self):
            return None

    class FakeObservationPipeline:
        in_channels = 1

        def __init__(self, args, n_envs, device):
            self.n_envs = n_envs
            self.device = torch.device(device)
            self.observation_config = types.SimpleNamespace(image_size=1)

        def reset(self, raw_frames, frame_skips):
            return torch.zeros(
                (self.n_envs, 1, 1, 1),
                dtype=torch.uint8,
                device=self.device,
            )

        def step(self, raw_frames, dones, frame_skips, profiler=None):
            return torch.zeros(
                (self.n_envs, 1, 1, 1),
                dtype=torch.uint8,
                device=self.device,
            )

    class FakeAgent(torch.nn.Module):
        def __init__(self, **kwargs):
            super().__init__()
            self.bias = torch.nn.Parameter(torch.zeros(()))

        def get_action_and_value(self, obs, action=None):
            batch_size = obs.shape[0]
            if action is None:
                action = torch.zeros(batch_size, dtype=torch.long, device=obs.device)
            logprob = self.bias.expand(batch_size)
            entropy = torch.ones(batch_size, device=obs.device)
            value = self.bias.expand(batch_size)
            return action, logprob, entropy, value

        def get_value(self, obs):
            return self.bias.expand(obs.shape[0])

    monkeypatch.setattr(ppo_loop, "SyncDemonAttackBatch", FakeSyncDemonAttackBatch)
    monkeypatch.setattr(ppo_loop, "ObservationPipeline", FakeObservationPipeline)
    monkeypatch.setattr(ppo_loop, "DemonAttackPpoAgent", FakeAgent)
    monkeypatch.setattr(
        ppo_loop,
        "save_checkpoint",
        lambda path, agent, optimizer, args, global_step: None,
    )
    monkeypatch.setattr(
        ppo_loop,
        "emit_event",
        lambda logger, event, formatter: events.append(event),
    )
    args = build_parser().parse_args(
        [
            "--train-steps",
            "1",
            "--n-envs",
            "1",
            "--n-steps",
            "1",
            "--batch-size",
            "1",
            "--n-epochs",
            "1",
            "--device",
            "cpu",
            "--save",
            str(tmp_path / "model.pt"),
            "--action-repeat",
            "4",
            "--terminal-on-life-loss",
            "--gamma",
            "0.99",
            "--gae-lambda",
            "0.95",
            "--checkpoint-freq",
            "500000",
            "--log-interval-seconds",
            "0",
            "--no-norm-adv",
        ]
    )
    wandb_module = types.SimpleNamespace(log=lambda *args, **kwargs: None)

    ppo_loop.train(args, wandb_module=wandb_module)

    train_start = next(event for event in events if event["event"] == "train_start")
    assert train_start["action_repeat"] == 4
    assert train_start["terminal_on_life_loss"] is True
    assert train_start["gamma"] == 0.99
    assert train_start["gae_lambda"] == 0.95
    assert train_start["checkpoint_freq"] == 500000


def test_training_observation_pipeline_matches_shared_torch_pipeline():
    args = _args()
    config = DemonAttackPpoObservationConfig.from_dict(
        {
            "image_size": args.screen_size,
            "stack_frames": args.stack_frames,
            "obs_mode": args.obs_mode,
            "frame_skip_values": args.frame_skip_values,
        }
    )
    training_pipeline = ObservationPipeline(args=args, n_envs=2, device="cpu")
    shared_pipeline = DemonAttackPpoTorchObservationPipeline(config, n_envs=2, device="cpu")
    first = torch.randint(0, 256, (2, 6, 5), dtype=torch.uint8)
    second = torch.randint(0, 256, (2, 6, 5), dtype=torch.uint8)
    frame_skips = np.asarray([0, 4], dtype=np.int64)
    dones = np.asarray([False, True], dtype=np.bool_)

    assert torch.equal(
        training_pipeline.reset(first, frame_skips),
        shared_pipeline.reset(first, frame_skips),
    )
    assert torch.equal(
        training_pipeline.step(second, dones, frame_skips),
        shared_pipeline.step(second, dones, frame_skips),
    )


def test_training_observation_pipeline_stack_done_reset_and_latency_channel():
    pipeline = ObservationPipeline(args=_args(), n_envs=2, device=torch.device("cpu"))
    raw = torch.stack(
        [
            torch.full((4, 4), 10, dtype=torch.uint8),
            torch.full((4, 4), 30, dtype=torch.uint8),
        ],
        dim=0,
    )

    reset_obs = _to_hwc(pipeline.reset(raw, np.asarray([0, 4], dtype=np.int64)))

    assert reset_obs.shape == (2, 4, 4, 4)
    assert np.all(reset_obs[0, :, :, :3] == 10)
    assert np.all(reset_obs[1, :, :, :3] == 30)
    assert np.all(reset_obs[0, :, :, -1] == latency_channel_value(0, 4))
    assert np.all(reset_obs[1, :, :, -1] == latency_channel_value(4, 4))

    next_raw = torch.stack(
        [
            torch.full((4, 4), 20, dtype=torch.uint8),
            torch.full((4, 4), 40, dtype=torch.uint8),
        ],
        dim=0,
    )

    step_obs = _to_hwc(
        pipeline.step(
            next_raw,
            dones=np.asarray([False, True], dtype=np.bool_),
            frame_skips=np.asarray([2, 4], dtype=np.int64),
        )
    )

    assert np.all(step_obs[0, :, :, :2] == 10)
    assert np.all(step_obs[0, :, :, 2:3] == 20)
    assert np.all(step_obs[0, :, :, -1] == latency_channel_value(2, 4))
    assert np.all(step_obs[1, :, :, :3] == 40)
    assert np.all(step_obs[1, :, :, -1] == latency_channel_value(4, 4))


def test_queue_latency_wrapper_schedules_actions_with_episode_fixed_latency():
    env = QueueLatencyWrapper(_FakeAtariEnv(), frame_skip_values=(2,), seed=0)
    _, reset_info = env.reset(seed=123)

    _, _, _, _, info0 = env.step(5)
    _, _, _, _, info1 = env.step(6)
    _, _, _, _, info2 = env.step(7)

    assert reset_info["frame_skip"] == 2
    assert info0["queued_action"] == 5
    assert info0["applied_action"] == 0
    assert info1["applied_action"] == 0
    assert info2["applied_action"] == 5
    assert info2["pending_queue_len"] == 2
    assert env.unwrapped.applied_actions == [0, 0, 5]


def test_sync_batch_accumulates_full_game_return_across_life_losses():
    envs = SyncDemonAttackBatch(
        n_envs=1,
        seed=0,
        frame_skip_values=(0,),
        device="cpu",
        env_fns=[
            lambda: _ScriptedDemonAttackEnv(
                [
                    (1.0, True, False, {"lives": 2}),
                    (2.0, True, False, {"lives": 1}),
                    (3.0, True, False, {"lives": 0}),
                ]
            )
        ],
    )

    try:
        envs.reset()
        rewards, dones_np, infos, _ = envs.step(np.asarray([0]))
        assert rewards[0] == 1.0
        assert bool(dones_np[0]) is True
        assert "episodic_return" not in infos[0]
        assert envs.episode_returns[0] == 1.0
        assert envs.episode_lengths[0] == 1

        _, dones_np, infos, _ = envs.step(np.asarray([0]))
        assert bool(dones_np[0]) is True
        assert "episodic_return" not in infos[0]
        assert envs.episode_returns[0] == 3.0
        assert envs.episode_lengths[0] == 2

        _, dones_np, infos, _ = envs.step(np.asarray([0]))
        assert bool(dones_np[0]) is True
        assert infos[0]["episodic_return"] == 6.0
        assert infos[0]["episodic_length"] == 3
        assert envs.episode_returns[0] == 0.0
        assert envs.episode_lengths[0] == 0
    finally:
        envs.close()


def test_sync_batch_vectorized_life_loss_and_game_over_metrics_are_separate():
    envs = SyncDemonAttackBatch(
        n_envs=2,
        seed=0,
        frame_skip_values=(0,),
        device="cpu",
        env_fns=[
            lambda: _ScriptedDemonAttackEnv([(4.0, True, False, {"lives": 2})]),
            lambda: _ScriptedDemonAttackEnv([(5.0, True, False, {"lives": 0})]),
        ],
    )

    try:
        envs.reset()
        _, dones_np, infos, _ = envs.step(np.asarray([0, 0]))

        assert dones_np.tolist() == [True, True]
        assert "episodic_return" not in infos[0]
        assert infos[1]["episodic_return"] == 5.0
        assert infos[1]["episodic_length"] == 1
        assert envs.episode_returns.tolist() == [4.0, 0.0]
        assert envs.episode_lengths.tolist() == [1, 0]
    finally:
        envs.close()


def test_optional_ale_smoke_can_build_sync_batch_when_roms_are_available():
    pytest.importorskip("ale_py")
    pytest.importorskip("stable_baselines3")
    from training.demon_attack_ppo.env_batch import SyncDemonAttackBatch

    try:
        envs = SyncDemonAttackBatch(
            n_envs=1,
            seed=0,
            frame_skip_values=(0,),
            device="cpu",
            env_id="ALE/DemonAttack-v5",
        )
    except Exception as exc:
        pytest.skip(f"ALE DemonAttack environment unavailable: {exc}")

    try:
        frames = envs.reset()
        assert frames.dtype == torch.uint8
        assert frames.device.type == "cpu"
        assert frames.shape[0] == 1
        assert envs.n_actions > 1
    finally:
        envs.close()
