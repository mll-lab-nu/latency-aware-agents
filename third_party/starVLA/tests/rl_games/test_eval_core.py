import numpy as np
import sys
import types
from omegaconf import OmegaConf

from starVLA.training.rl_games.eval_core import (
    ActionLatencyQueue,
    RlGamesEvalRunner,
    _TaskEvaluator,
    _apply_prompt_mode as apply_eval_prompt_mode,
    _resize_rgb,
    decode_deadly_factorized_11,
    decode_deadly_multibinary_7,
    decode_defend_the_line_multibinary,
    decode_defend_the_line_tuple,
    decode_discrete_argmax,
)
from starVLA.dataloader.gr00t_lerobot.datasets import _apply_prompt_mode as apply_train_prompt_mode


def test_decode_discrete_argmax():
    values = np.array([0.1, 0.7, 0.2], dtype=np.float32)
    assert decode_discrete_argmax(values, 3) == 1


def test_decode_discrete_argmax_ignores_bridge_padding():
    values = np.array([0.1, 0.2, 99.0, 99.0, 99.0, 99.0, 99.0], dtype=np.float32)
    assert decode_discrete_argmax(values, 2) == 1


def test_resize_rgb_preserves_source_resolution_when_image_size_is_none():
    image = np.zeros((288, 512, 3), dtype=np.uint8)

    resized = _resize_rgb(image, None, prefer_area=True)

    assert resized.shape == (288, 512, 3)


def test_task_evaluator_accepts_null_image_size_for_native_eval_resolution():
    cfg = OmegaConf.create(
        {
            "rl_games": {
                "env_eval": {
                    "image_size": None,
                },
            },
            "framework": {
                "action_model": {
                    "state_dim": 1,
                },
            },
        }
    )

    evaluator = _TaskEvaluator(task="flappy", cfg=cfg)

    assert evaluator.image_size is None


def test_decode_deadly_multibinary_7():
    values = np.array([0.2, 0.6, 0.51, 0.49, 0.8, 0.1, 0.9], dtype=np.float32)
    assert decode_deadly_multibinary_7(values) == [0, 1, 1, 0, 1, 0, 1]


def test_decode_deadly_multibinary_7_ignores_extra_padding():
    values = np.array([0.2, 0.6, 0.51, 0.49, 0.8, 0.1, 0.9, 99.0], dtype=np.float32)
    assert decode_deadly_multibinary_7(values) == [0, 1, 1, 0, 1, 0, 1]


def test_decode_deadly_factorized_11():
    values = np.array([
        0.1, 0.8, 0.1,
        0.2, 0.1, 0.7,
        0.3, 0.6, 0.1,
        0.4, 0.6,
    ], dtype=np.float32)
    assert decode_deadly_factorized_11(values) == [1, 2, 1, 1]


def test_decode_defend_the_line_tuple():
    values = np.array([0.1, 0.2, 0.3, 0.4, 0.8, 0.5, 99.0], dtype=np.float32)
    assert decode_defend_the_line_tuple(values) == [2, 0]
    assert decode_defend_the_line_multibinary(values) == [0, 1, 0]


def test_task_evaluator_decodes_defend_the_line_joint_action():
    evaluator = object.__new__(_TaskEvaluator)
    evaluator.task = "defend_the_line"

    action = evaluator._decode_action(np.asarray([0.1, 0.2, 0.3, 0.9, 0.8, 0.5], dtype=np.float32))

    assert action == [1, 0, 1]
    assert evaluator._queue_default_action() == [0, 0, 0]


def test_task_evaluator_uses_shared_deadly_multibinary_decode():
    cfg = OmegaConf.create(
        {
            "rl_games": {
                "env_eval": {
                    "deadly": {
                        "action_layout": "multibinary_7",
                        "multibinary_threshold": None,
                    }
                }
            },
            "framework": {"action_model": {"loss_type": "discrete_ce"}},
        }
    )
    evaluator = object.__new__(_TaskEvaluator)
    evaluator.task = "deadly_corridor"
    evaluator.cfg = cfg
    evaluator.env_eval_cfg = cfg.rl_games.env_eval

    action = evaluator._decode_action(
        np.asarray([0.8, 0.2, 0.7, 0.1, 0.9, 0.3, 0.6], dtype=np.float32)
    )

    assert action == [1, 0, 1, 0, 1, 0, 1]


def test_task_evaluator_decodes_air_raid_six_action_space():
    evaluator = object.__new__(_TaskEvaluator)
    evaluator.task = "air_raid"

    action = evaluator._decode_action(
        np.asarray([0.1, 0.2, 0.3, 0.9, 0.4, 0.5, 99.0], dtype=np.float32)
    )

    assert action == 3


def test_action_latency_queue():
    queue = ActionLatencyQueue(latency=2, default_action=0)
    queue.reset()
    outputs = [queue.schedule_and_get(a) for a in [1, 2, 3, 4]]
    assert outputs == [0, 0, 1, 2]


def test_prompt_mode_latency_neutral_strips_timing_suffix():
    prompt = (
        "You are playing Flappy Bird. Pass through the pipe gaps and stay alive. "
        "Choose the action: NOOP, FLAP. Current action latency is 2 raw frames (66.67 ms). "
        "The environment runs at 30 FPS and observations are emitted at 30 FPS. Choose the best next action."
    )

    assert apply_train_prompt_mode(prompt, "latency_neutral") == (
        "You are playing Flappy Bird. Pass through the pipe gaps and stay alive. Choose the action: NOOP, FLAP."
    )
    assert apply_eval_prompt_mode(prompt, "latency_neutral") == (
        "You are playing Flappy Bird. Pass through the pipe gaps and stay alive. Choose the action: NOOP, FLAP."
    )


def test_eval_runner_infers_latencies_from_prompt_map_when_values_empty(tmp_path):
    prompt_map = tmp_path / "latency_prompt_map.json"
    prompt_map.write_text(
        '{"0": {"latency": 0, "latency_ms": 0.0, "prompt": "zero"},'
        ' "2": {"latency": 2, "latency_ms": 66.6, "prompt": "two"}}',
        encoding="utf-8",
    )
    cfg = OmegaConf.create(
        {
            "rl_games": {
                "task": "demon_attack",
                "model_alias": "openvla",
                "env_eval": {
                    "enabled": True,
                    "latency": {
                        "values": [],
                        "prompt_map_path": str(prompt_map),
                    },
                },
            }
        }
    )

    runner = RlGamesEvalRunner(cfg=cfg, output_dir=str(tmp_path))

    assert runner._get_latency_values(stage="mid_train") == [0, 2]


def test_task_evaluator_latency_neutral_prompt_mode_strips_prompt_map_prompts(tmp_path):
    prompt_map = tmp_path / "latency_prompt_map.json"
    prompt_map.write_text(
        '{"0": {"latency": 0, "latency_ms": 0.0, "prompt": "zero Current action latency is 0 raw frames (0.00 ms)."}}',
        encoding="utf-8",
    )
    cfg = OmegaConf.create(
        {
            "seed": 42,
            "rl_games": {
                "env_eval": {
                    "prompt_mode": "latency_neutral",
                    "latency": {
                        "values": [0],
                        "prompt_map_path": str(prompt_map),
                    },
                },
            },
            "framework": {
                "action_model": {
                    "state_dim": 1,
                },
            },
        }
    )

    runner = RlGamesEvalRunner(cfg=cfg, output_dir=str(tmp_path))
    assert runner._resolve_prompt(latency=0, mapping={0: {"prompt": "zero Current action latency is 0 raw frames (0.00 ms)."}}, task="flappy") == "zero"


def test_cross_task_prompt_resolution_uses_task_description_for_unmapped_latency(tmp_path):
    prompt_map = tmp_path / "defend_prompt_map.json"
    prompt_map.write_text(
        '{"0": {"latency": 0, "latency_ms": 0.0, "prompt": "zero latency prompt"}}',
        encoding="utf-8",
    )
    cfg = OmegaConf.create(
        {
            "rl_games": {
                "task": "cross_task",
                "env_eval": {
                    "prompt_mode": "default",
                    "latency": {
                        "prompt_map_path": None,
                    },
                },
                "cross_task": {
                    "eval_tasks": {
                        "defend_the_line": {
                            "prompt_map_path": str(prompt_map),
                            "task_description": "Defend the line fallback prompt.",
                        },
                    },
                },
            },
            "framework": {
                "action_model": {
                    "state_dim": 1,
                },
            },
        }
    )

    runner = RlGamesEvalRunner(cfg=cfg, output_dir=str(tmp_path))

    assert runner._resolve_prompt(latency=2, mapping={0: {"prompt": "zero latency prompt"}}, task="defend_the_line") == "Defend the line fallback prompt."


def test_task_evaluator_reuses_episode_seeds_across_tasks_and_latencies_by_default():
    cfg = OmegaConf.create(
        {
            "seed": 42,
            "rl_games": {
                "env_eval": {
                    "fixed_episode_seeds": True,
                },
            },
            "framework": {
                "action_model": {
                    "state_dim": 1,
                },
            },
        }
    )

    for task in ("flappy", "demon_attack", "deadly_corridor"):
        evaluator = _TaskEvaluator(task=task, cfg=cfg)
        assert [evaluator._episode_seed(latency=0, episode=episode) for episode in range(3)] == [42, 43, 44]
        assert [evaluator._episode_seed(latency=8, episode=episode) for episode in range(3)] == [42, 43, 44]


def test_task_evaluator_respects_explicit_latency_seed_stride():
    cfg = OmegaConf.create(
        {
            "seed": 42,
            "rl_games": {
                "env_eval": {
                    "fixed_episode_seeds": True,
                    "latency_seed_stride": 1000,
                },
            },
            "framework": {
                "action_model": {
                    "state_dim": 1,
                },
            },
        }
    )

    evaluator = _TaskEvaluator(task="flappy", cfg=cfg)

    assert [evaluator._episode_seed(latency=8, episode=episode) for episode in range(3)] == [8042, 8043, 8044]


def test_task_evaluator_respects_explicit_task_seed_stride():
    cfg = OmegaConf.create(
        {
            "seed": 42,
            "rl_games": {
                "env_eval": {
                    "fixed_episode_seeds": True,
                    "task_seed_stride": 100000,
                },
            },
            "framework": {
                "action_model": {
                    "state_dim": 1,
                },
            },
        }
    )

    evaluator = _TaskEvaluator(task="demon_attack", cfg=cfg)

    assert [evaluator._episode_seed(latency=0, episode=episode) for episode in range(3)] == [100042, 100043, 100044]


def test_task_evaluator_saved_seed_overrides_take_precedence():
    cfg = OmegaConf.create(
        {
            "seed": 42,
            "rl_games": {
                "env_eval": {
                    "fixed_episode_seeds": True,
                    "latency_seed_stride": 1000,
                },
            },
            "framework": {
                "action_model": {
                    "state_dim": 1,
                },
            },
        }
    )

    evaluator = _TaskEvaluator(task="flappy", cfg=cfg)

    assert evaluator._episode_seed_for_run(latency=8, episode=1, seed_overrides={1: 123456}) == 123456
    assert evaluator._episode_seed_for_run(latency=8, episode=2, seed_overrides={1: 123456}) == 8044
    assert evaluator._episode_seed_for_run(latency=8, episode=3, seed_overrides={3: None}) is None


class _FixedRng:
    def __init__(self, value: int):
        self.value = value

    def integers(self, low: int, high: int) -> int:
        assert low == 1
        assert high == 31
        return self.value


class _FakeDemonAttackEnv:
    def __init__(self):
        self.np_random = _FixedRng(value=30)
        self.reset_calls = []
        self.step_actions = []

    def reset(self, **kwargs):
        self.reset_calls.append(kwargs)
        return "reset_obs", {"reset": True}

    def step(self, action):
        self.step_actions.append(action)
        return "noop_obs", 0.0, False, False, {"noop": True}


class _FakeAirRaidEnv:
    def __init__(self):
        self.np_random = _FixedRng(value=2)
        self.reset_calls = []
        self.step_actions = []

    def reset(self, **kwargs):
        self.reset_calls.append(kwargs)
        return np.zeros((250, 160, 3), dtype=np.uint8), {"reset": True}

    def step(self, action):
        self.step_actions.append(action)
        return np.zeros((250, 160, 3), dtype=np.uint8), 0.0, False, False, {"action": action}

    def render(self):
        return np.zeros((250, 160, 3), dtype=np.uint8)


def test_demon_attack_env_uses_noop_reset_max_30_by_default(monkeypatch):
    fake_env = _FakeDemonAttackEnv()
    fake_gym = types.SimpleNamespace(make=lambda *args, **kwargs: fake_env)
    monkeypatch.setitem(sys.modules, "gymnasium", fake_gym)
    monkeypatch.setitem(sys.modules, "ale_py", types.SimpleNamespace())
    cfg = OmegaConf.create(
        {
            "rl_games": {
                "env_eval": {
                    "frameskip": 4,
                },
            },
            "framework": {
                "action_model": {
                    "state_dim": 1,
                },
            },
        }
    )

    evaluator = _TaskEvaluator(task="demon_attack", cfg=cfg)
    env = evaluator._make_env()
    obs, info = env.reset(seed=42)

    assert obs == "noop_obs"
    assert info == {"noop": True}
    assert fake_env.reset_calls == [{"seed": 42}]
    assert fake_env.step_actions == [0] * 30


def test_air_raid_env_uses_pal_native_frameskip_and_fire_reset(monkeypatch):
    fake_env = _FakeAirRaidEnv()
    captured = {}

    def fake_make(env_id, **kwargs):
        captured["env_id"] = env_id
        captured["kwargs"] = kwargs
        return fake_env

    fake_gym = types.SimpleNamespace(make=fake_make)
    monkeypatch.setitem(sys.modules, "gymnasium", fake_gym)
    monkeypatch.setitem(sys.modules, "ale_py", types.SimpleNamespace())
    cfg = OmegaConf.create(
        {
            "rl_games": {
                "env_eval": {
                    "frameskip": 4,
                    "atari": {
                        "env_id": "ALE/AirRaid-v5",
                        "gym_frameskip": 1,
                        "noop_max": 30,
                        "fire_reset": True,
                        "mode": 1,
                        "difficulty": 0,
                        "repeat_action_probability": 0.0,
                        "full_action_space": False,
                        "max_num_frames_per_episode": 108000,
                    },
                },
            },
            "framework": {
                "action_model": {
                    "state_dim": 1,
                },
            },
        }
    )

    evaluator = _TaskEvaluator(task="air_raid", cfg=cfg)
    env = evaluator._make_env()
    env.reset(seed=7)

    assert captured["env_id"] == "ALE/AirRaid-v5"
    assert captured["kwargs"] == {
        "obs_type": "rgb",
        "frameskip": 1,
        "repeat_action_probability": 0.0,
        "full_action_space": False,
        "mode": 1,
        "difficulty": 0,
        "render_mode": "rgb_array",
        "max_num_frames_per_episode": 108000,
    }
    assert fake_env.reset_calls == [{"seed": 7}]
    assert fake_env.step_actions == [0, 0, 1, 2]


def test_air_raid_eval_steps_four_raw_frames_per_policy_decision():
    fake_env = _FakeAirRaidEnv()
    cfg = OmegaConf.create(
        {
            "rl_games": {
                "env_eval": {
                    "frameskip": 4,
                },
            },
            "framework": {
                "action_model": {
                    "state_dim": 1,
                },
            },
        }
    )
    evaluator = _TaskEvaluator(task="air_raid", cfg=cfg)

    evaluator._step_env_once(fake_env, action=5)

    assert fake_env.step_actions == [5, 5, 5, 5]


def test_air_raid_prompt_fallback_matches_latency_bench_contract():
    cfg = OmegaConf.create(
        {
            "rl_games": {
                "task": "air_raid",
                "env_eval": {
                    "prompt_mode": "default",
                    "latency": {
                        "prompt_map_path": None,
                    },
                },
            },
            "framework": {
                "action_model": {
                    "state_dim": 1,
                },
            },
        }
    )

    runner = RlGamesEvalRunner(cfg=cfg, output_dir="/tmp/air_raid_eval_test")

    assert runner._resolve_prompt(latency=0, mapping={}, task="air_raid") == (
        "Protect both buildings from flying saucers. Choose exactly one action from: "
        "noop, fire, right, left, rightfire, leftfire."
    )


class _OneStepEnv:
    def __init__(self):
        self.reset_calls = []
        self.close_calls = 0

    def reset(self, **kwargs):
        self.reset_calls.append(kwargs)
        return np.zeros((12, 16, 3), dtype=np.uint8), {}

    def step(self, action):
        return np.zeros((12, 16, 3), dtype=np.uint8), 0.0, True, False, {}

    def close(self):
        self.close_calls += 1


class _SlotTrackingModel:
    def __init__(self):
        self.reset_memory_calls = []
        self.example_slot_ids = []

    def reset_memory(self, slot_id=None):
        self.reset_memory_calls.append(slot_id)

    def predict_action(self, *, examples):
        self.example_slot_ids.append([example["slot_id"] for example in examples])
        return {"normalized_actions": np.zeros((len(examples), 1, 6), dtype=np.float32)}


def _single_step_eval_cfg(*, vectorized: bool = False):
    env_eval = {
        "frameskip": 1,
    }
    if vectorized:
        env_eval["vectorized"] = {
            "enabled": True,
            "batch_size": 2,
        }
    return OmegaConf.create(
        {
            "seed": 42,
            "rl_games": {
                "env_eval": env_eval,
            },
            "framework": {
                "action_model": {
                    "state_dim": 1,
                },
            },
        }
    )


def test_task_evaluator_resets_serial_model_memory_at_each_episode(monkeypatch):
    evaluator = _TaskEvaluator(task="demon_attack", cfg=_single_step_eval_cfg())
    model = _SlotTrackingModel()
    monkeypatch.setattr(evaluator, "_make_env", lambda: _OneStepEnv())

    evaluator.run_latency(
        model=model,
        latency=0,
        prompt="prompt",
        max_steps=1,
        num_episodes=2,
        episode_indices=[0, 1],
    )

    assert model.reset_memory_calls == [0, 0]
    assert model.example_slot_ids == [[0], [0]]


def test_task_evaluator_resets_vectorized_model_memory_per_slot(monkeypatch):
    evaluator = _TaskEvaluator(task="demon_attack", cfg=_single_step_eval_cfg(vectorized=True))
    model = _SlotTrackingModel()
    monkeypatch.setattr(evaluator, "_make_env", lambda: _OneStepEnv())

    evaluator.run_latency(
        model=model,
        latency=0,
        prompt="prompt",
        max_steps=1,
        num_episodes=3,
        episode_indices=[0, 1, 2],
    )

    assert model.reset_memory_calls == [0, 1, 0]
    assert model.example_slot_ids == [[0, 1], [0]]


def test_eval_runner_saves_result_after_each_episode(monkeypatch, tmp_path):
    cfg = OmegaConf.create(
        {
            "seed": 42,
            "rl_games": {
                "task": "demon_attack",
                "model_alias": "openvla",
                "env_eval": {
                    "enabled": True,
                    "seed": 42,
                    "post_train": {
                        "enabled": True,
                        "latencies": [0],
                        "num_episodes": 2,
                        "max_steps_per_episode": 1,
                    },
                },
            },
            "framework": {
                "action_model": {
                    "state_dim": 1,
                },
            },
        }
    )
    save_totals = []
    original_save = RlGamesEvalRunner._save

    def spy_save(self, result, step, stage):
        save_totals.append(int(result.aggregate["total_episodes"]))
        return original_save(self, result, step, stage)

    def fake_run_latency(self, **kwargs):
        first_episode_metrics = {
            "latency": 0,
            "num_episodes": 1,
            "mean_reward": 1.0,
            "mean_length": 1.0,
            "std_reward": 0.0,
            "std_length": 0.0,
            "episode_rewards": [1.0],
            "episode_lengths": [1],
            "decoded_action_hist": {"0": 1},
            "fixed_episode_seeds": True,
            "eval_seed": 42,
            "episode_seeds": [42],
        }
        second_episode_metrics = {
            "latency": 0,
            "num_episodes": 2,
            "mean_reward": 1.5,
            "mean_length": 1.0,
            "std_reward": 0.5,
            "std_length": 0.0,
            "episode_rewards": [1.0, 2.0],
            "episode_lengths": [1, 1],
            "decoded_action_hist": {"0": 2},
            "fixed_episode_seeds": True,
            "eval_seed": 42,
            "episode_seeds": [42, 43],
        }
        kwargs["on_episode_complete"](first_episode_metrics)
        kwargs["on_episode_complete"](second_episode_metrics)
        return second_episode_metrics

    monkeypatch.setattr(RlGamesEvalRunner, "_save", spy_save)
    monkeypatch.setattr(_TaskEvaluator, "run_latency", fake_run_latency)

    runner = RlGamesEvalRunner(cfg=cfg, output_dir=str(tmp_path))
    result = runner.run(model=object(), step=2500, stage="post_train")

    assert save_totals == [1, 2, 2]
    assert result.path == str(tmp_path / "eval" / "post_train" / "step_2500.json")


def test_eval_runner_passes_saved_seed_overrides_by_latency(monkeypatch, tmp_path):
    cfg = OmegaConf.create(
        {
            "seed": 42,
            "rl_games": {
                "task": "flappy",
                "model_alias": "openvla",
                "env_eval": {
                    "enabled": True,
                    "mid_train": {
                        "enabled": True,
                        "latencies": [0],
                        "num_episodes": 2,
                        "max_steps_per_episode": 1,
                    },
                },
            },
            "framework": {
                "action_model": {
                    "state_dim": 1,
                },
            },
        }
    )
    seen_seed_overrides = []
    seen_episode_indices = []

    def fake_run_latency(self, **kwargs):
        seen_seed_overrides.append(kwargs["seed_overrides"])
        seen_episode_indices.append(kwargs["episode_indices"])
        return {
            "latency": 0,
            "num_episodes": 2,
            "mean_reward": 0.0,
            "mean_length": 1.0,
            "std_reward": 0.0,
            "std_length": 0.0,
            "episode_rewards": [0.0, 0.0],
            "episode_lengths": [1, 1],
            "decoded_action_hist": {"0": 2},
            "fixed_episode_seeds": True,
            "eval_seed": 42,
            "episode_seeds": [9001, 9002],
            "episode_indices": [5, 7],
        }

    monkeypatch.setattr(_TaskEvaluator, "run_latency", fake_run_latency)

    runner = RlGamesEvalRunner(cfg=cfg, output_dir=str(tmp_path))
    result = runner.run(
        model=object(),
        step=1000,
        stage="mid_train",
        episode_seed_overrides={"flappy/latency_0": {5: 9001, 7: 9002}},
    )

    assert seen_seed_overrides == [{5: 9001, 7: 9002}]
    assert seen_episode_indices == [[5, 7]]
    assert result.per_latency["flappy/latency_0"]["episode_seeds"] == [9001, 9002]
