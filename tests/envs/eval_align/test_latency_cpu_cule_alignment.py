from __future__ import annotations

import numpy as np
import pytest

from latency_bench.core.types import Action
from latency_bench.envs.base import ENV_RAW_RGB_FRAME_STACK_INFO_KEY


FRAME_STACK = 4
SEED = 12345
ACTIONS = [0, 1, 2, 3, 4, 5, 0, 2]
# These seeds select each of the 30 supported NOOP counts in the ALE reset wrapper.
ALL_NOOP_SEEDS = (4, 29, 38, 28, 1, 30, 13, 27, 22, 32, 11, 25, 106, 7, 12,
                  23, 16, 57, 2, 67, 6, 0, 130, 21, 3, 73, 10, 14, 5, 19)


def test_cule_core_and_latency_backend_are_byte_identical():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    core_trace = _cule_core_trace()
    backend_trace = _cule_latency_backend_trace()

    assert len(core_trace) == len(backend_trace)
    for core_point, backend_point in zip(core_trace, backend_trace):
        assert set(core_point) == set(backend_point)
        for key, core_value in core_point.items():
            backend_value = backend_point[key]
            if isinstance(core_value, np.ndarray):
                assert np.array_equal(core_value, backend_value), key
            else:
                assert core_value == backend_value


@pytest.mark.parametrize(("configured_clip_reward", "num_slots"), [(False, 1), (True, 16)])
def test_cule_life_loss_soft_reset_matches_ale_and_continues_the_episode(configured_clip_reward, num_slots):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    pytest.importorskip("cv2")

    from latency_bench.envs.atari import TRUE_EPISODE_END_INFO_KEY
    from latency_bench.envs.adapters.demon_attack import DemonAttackEnvAdapter
    from latency_bench.executors.demon_attack_gpu_batched_backend import (
        DemonAttackGpuBatchedEnvStepBackend,
    )

    noop = Action(value=0, name="NOOP", is_noop=True)
    cpu = DemonAttackEnvAdapter(
        env_fps=60.0,
        noop_action=noop,
        screen_size=84,
        noop_max=1,
        reset_mode="fire",
        terminal_on_life_loss=True,
        clip_reward=False,
        continue_life_loss_terminal=True,
        frame_stack=FRAME_STACK,
        export_env_raw_rgb_frames=True,
    )
    backend = DemonAttackGpuBatchedEnvStepBackend(
        config={
            "env": {
                "name": "demon_attack",
                "env_fps": 60.0,
                "obs_fps": 15.0,
                "noop_max": 1,
                "reset_mode": "fire",
                "terminal_on_life_loss": True,
                "clip_reward": configured_clip_reward,
                "frame_stack": FRAME_STACK,
                "gpu_env_device": "cuda",
                "max_episode_steps": 25000,
            },
            "policy": {"type": "starvla"},
            "evaluation": {"eval_raw_reward": True},
        },
        noop_action=noop,
        num_slots=num_slots,
    )
    actions = np.random.default_rng(2026).integers(
        0,
        6,
        size=4000,
        dtype=np.int64,
    ).tolist()
    cpu_return = 0.0
    cule_return = 0.0
    soft_reset_count = 0
    compare_next_transition = False
    true_episode_end_count = 0
    peer_reset_count = 0

    try:
        cpu.reset(seed=SEED)
        for slot in range(num_slots):
            backend.reset_slot(slot, episode_id=slot, seed=SEED + slot)
        for raw_frame, action_value in enumerate(actions, start=1):
            action = Action(value=action_value, name=str(action_value), is_noop=action_value == 0)
            cpu_result = cpu.step(action)
            slot_actions = {
                slot: Action(value=(action_value + slot) % 6, name=str((action_value + slot) % 6))
                for slot in range(num_slots)
            }
            slot_actions[0] = action
            outcomes = backend.step_slots(slot_actions)
            cule_result = outcomes[0].result
            cpu_return += cpu_result.reward
            cule_return += cule_result.reward

            assert cule_result.reward == cpu_result.reward
            assert cule_result.done == cpu_result.done
            assert cule_result.truncated == cpu_result.truncated
            assert cule_result.info["lives"] == cpu_result.info["lives"]
            assert cule_result.info[TRUE_EPISODE_END_INFO_KEY] == cpu_result.info[TRUE_EPISODE_END_INFO_KEY]
            assert ("soft_reset" in cule_result.info) == ("soft_reset" in cpu_result.info)

            cpu_observation = cpu.observe()
            cule_observation = backend.observe_slots([0])[0]
            is_soft_reset = "soft_reset" in cpu_result.info
            if is_soft_reset or compare_next_transition:
                assert cule_observation.env_step == cpu_observation.env_step
                assert cule_observation.sim_time_ms == cpu_observation.sim_time_ms
                assert np.array_equal(cpu_observation.data, cule_observation.data)
                assert np.array_equal(
                    cpu_observation.metadata[ENV_RAW_RGB_FRAME_STACK_INFO_KEY],
                    cule_observation.metadata[ENV_RAW_RGB_FRAME_STACK_INFO_KEY],
                )
            compare_next_transition = is_soft_reset

            if is_soft_reset:
                soft_reset_count += 1
                assert cpu_result.done is False
                assert cule_result.done is False
                assert np.array_equal(
                    cpu.gym_env.unwrapped.ale.getRAM(),
                    backend.core.ram[0].detach().cpu().numpy(),
                )

            if cpu_result.done or cpu_result.truncated:
                true_episode_end_count += 1
                assert cpu_result.info["lives"] == 0
                assert cpu_result.info[TRUE_EPISODE_END_INFO_KEY] is True
                break
            # Other replicas may hit their episode budget while slot zero is alive.
            if num_slots > 1 and raw_frame in (128, 256):
                peer = raw_frame // 128
                backend.reset_slot(peer, episode_id=num_slots + peer, seed=SEED + raw_frame)
                peer_reset_count += 1
            for slot in range(1, num_slots):
                if outcomes[slot].result.done or outcomes[slot].result.truncated:
                    backend.reset_slot(slot, episode_id=slot, seed=SEED + slot)
                    peer_reset_count += 1
    finally:
        cpu.close()
        backend.close()

    assert soft_reset_count > 0
    assert true_episode_end_count == 1
    assert cule_return == cpu_return
    if num_slots > 1:
        assert peer_reset_count > 0


def test_frozen_h8_teacher_cpu_and_cule_continue_beyond_first_life_loss(tmp_path):
    """Regression for the September 23 first-life teacher scores (OFT best).

    DEMON_ATTACK_TEACHER_EVAL_CONFIG points to the restored, immutable OFT H8
    eval YAML described in docs/demon_attack.md. The artifact is
    supplied explicitly so the ordinary test suite never downloads weights.
    """
    import os
    from copy import deepcopy
    from types import SimpleNamespace

    if "DEMON_ATTACK_TEACHER_EVAL_CONFIG" not in os.environ:
        pytest.skip("Set DEMON_ATTACK_TEACHER_EVAL_CONFIG to run the frozen-teacher regression")
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    from latency_bench.core.actions import ActionResolver
    from latency_bench.core.config import load_config
    from latency_bench.integrations.sample_factory.teacher_config import load_rollout_training_config
    from latency_bench.integrations.sample_factory.teacher_runtime import build_sample_factory_rollout_cfg
    from latency_bench.eval.driver import run_from_config
    from latency_bench.executors.backend_selection import build_env_step_backend
    from latency_bench.policy.sample_factory import SampleFactoryPolicyRunner

    config = load_config(os.environ["DEMON_ATTACK_TEACHER_EVAL_CONFIG"])
    config["experiment"]["seed"] = 42
    config['evaluation'].update(episodes=16, max_env_steps=7000, episode_seed_offset=104729, eval_raw_reward=True)
    config["executor"]["parallel_envs"] = 16
    config["env"]["clip_reward"] = True
    config["logging"].update(save_step_records=False, save_action_records=False, save_latency_records=False)
    descriptor, _ = load_rollout_training_config(config["policy"]["checkpoint_path"], device_override="gpu")
    descriptor["latency_override"] = config["latency"]
    module, cfg = build_sample_factory_rollout_cfg(descriptor)
    cfg.simulator = "cpu"
    cfg.env_gpu_observations = False
    cfg.env_gpu_actions = False
    cfg.episode_metrics_path = ""
    integration = SimpleNamespace(
        register_components=module.register_demon_attack_envs,
        build_sf_cfg_from_config=lambda _: cfg,
        make_reference_env=module.make_reference_env,
    )
    resolver = ActionResolver(config["env"])
    policy = SampleFactoryPolicyRunner(config=config, action_resolver=resolver,
                                      integration_module=integration, policy_type="demon_attack_sf")
    results = []
    for backend_name in ("serial", "demon_attack_gpu_batched"):
        run_config = deepcopy(config)
        run_config["logging"]["output_dir"] = str(tmp_path / backend_name)
        backend = build_env_step_backend(backend_name, config=run_config,
                                         noop_action=resolver.noop_action, num_slots=16)
        episodes = run_from_config(run_config, policy=policy, env_backend=backend)
        assert all(episode.survival_steps == 7000 for episode in episodes)
        assert any(episode.metadata["soft_reset_count"] > 0 for episode in episodes)
        results.append([episode.episode_return for episode in episodes])
    assert results[0] == results[1]


@pytest.mark.gpu
@pytest.mark.parametrize("seed", [SEED, *ALL_NOOP_SEEDS])
def test_cule_reset_and_first_step_match_latency_cpu_ram_and_images(seed):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    pytest.importorskip("cv2")

    cpu = _make_latency_cpu_env()
    core = _make_cule_core()
    try:
        cpu.reset(seed=seed)
        cule_obs, _infos = core.reset_batch(seeds=[seed])
        torch.cuda.synchronize()

        cpu_raw = cpu.render_game_frame().copy()
        cule_raw = core.native_rgb_frames()[0].detach().cpu().numpy().copy()
        cpu_obs84 = _latest_latency_cpu_frame(cpu.observe().data)
        cule_obs84 = cule_obs[0, 0].detach().cpu().numpy().copy()

        assert np.array_equal(cpu.gym_env.unwrapped.ale.getRAM(), core.ram[0].detach().cpu().numpy())
        np.testing.assert_array_equal(cpu_raw, cule_raw)
        np.testing.assert_array_equal(cpu_obs84, cule_obs84)
        # A seeded reset must reproduce the same hardware state and observation.
        repeated_obs, _ = core.reset_batch(seeds=[seed])
        np.testing.assert_array_equal(cpu.gym_env.unwrapped.ale.getRAM(), core.ram[0].cpu().numpy())
        np.testing.assert_array_equal(cpu_obs84, repeated_obs[0, 0].cpu().numpy())
        # Reading the current frame must preserve it and the next transition.
        np.testing.assert_array_equal(cule_raw, core.native_rgb_frames()[0].cpu().numpy())
        np.testing.assert_array_equal(cule_raw, core.native_rgb_frames()[0].cpu().numpy())

        cpu.step(Action(value=0, name="0", is_noop=True))
        cule_step = core.step_batch([0])
        torch.cuda.synchronize()

        cpu_step_obs = _latest_latency_cpu_frame(cpu.observe().data)
        cule_step_obs = cule_step.observations[0, 0].detach().cpu().numpy().copy()
        assert np.array_equal(cpu.gym_env.unwrapped.ale.getRAM(), core.ram[0].detach().cpu().numpy())
        assert np.array_equal(cpu.render_game_frame(), core.native_rgb_frames()[0].detach().cpu().numpy())
        assert np.array_equal(cpu_step_obs, cule_step_obs)
    finally:
        cpu.close()


@pytest.mark.gpu
def test_cule_observation_matches_latency_cpu_after_action_trace():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    cv2 = pytest.importorskip("cv2")

    cpu = _make_latency_cpu_env()
    core = _make_cule_core()
    try:
        cpu.reset(seed=SEED)
        core.reset_batch(seeds=[SEED])
        torch.cuda.synchronize()

        for action_value in ACTIONS[:4]:
            cpu.step(Action(value=action_value, name=str(action_value), is_noop=action_value == 0))
            cule_step = core.step_batch([action_value])
            torch.cuda.synchronize()

        cpu_raw = cpu.render_game_frame().copy()
        cule_raw = core.native_rgb_frames()[0].detach().cpu().numpy().copy()
        cpu_obs84 = _latest_latency_cpu_frame(cpu.observe().data)
        cule_obs84 = cule_step.observations[0, 0].detach().cpu().numpy().copy()
        obs84_from_raw = cv2.resize(
            cv2.cvtColor(cule_raw, cv2.COLOR_RGB2GRAY), (84, 84), interpolation=cv2.INTER_AREA,
        )

        assert np.array_equal(cpu_raw, cule_raw)
        assert np.array_equal(cule_obs84, obs84_from_raw)
        np.testing.assert_array_equal(cpu_obs84, cule_obs84)
    finally:
        cpu.close()


def test_latency_cule_backend_rejects_non_fire_reset_mode():
    # CULE currently runs the hard-coded noop+fire reset sequence. Until raw/noop
    # are implemented there, non-fire latency_bench configs must fail loudly.
    from latency_bench.executors.demon_attack_gpu_batched_backend import DemonAttackGpuBatchedEnvStepBackend

    with pytest.raises(ValueError, match="reset_mode='fire'"):
        DemonAttackGpuBatchedEnvStepBackend(
            config={
                "env": {
                    "name": "demon_attack",
                    "env_fps": 60.0,
                    "obs_fps": 15.0,
                    "noop_max": 30,
                    "reset_mode": "noop",
                    "terminal_on_life_loss": False,
                    "clip_reward": False,
                    "frame_stack": FRAME_STACK,
                    "gpu_env_device": "cuda",
                    "max_episode_steps": 25000,
                },
                "policy": {"type": "starvla"},
            },
            noop_action=Action(value=0, name="NOOP", is_noop=True),
            num_slots=1,
        )


def _make_latency_cpu_env():
    from latency_bench.envs.adapters.demon_attack import DemonAttackEnvAdapter

    return DemonAttackEnvAdapter(
        env_fps=60.0,
        noop_action=Action(value=0, name="NOOP", is_noop=True),
        screen_size=84,
        noop_max=30,
        reset_mode="fire",
        terminal_on_life_loss=False,
        clip_reward=False,
        frame_stack=FRAME_STACK,
        export_env_raw_rgb_frames=True,
    )


def _latest_latency_cpu_frame(observation_data) -> np.ndarray:
    return np.asarray(observation_data, dtype=np.uint8)[-1].copy()


def _make_cule_core():
    from training.demon_attack_sf.gpu_core import CudaDemonAttackCore

    return CudaDemonAttackCore(
        batch_size=1,
        device="cuda",
        noop_max=30,
        terminal_on_life_loss=False,
        obs_stride=1,
        frame_stack=1,
    )


def _cule_core_trace() -> list[dict[str, object]]:
    torch = pytest.importorskip("torch")

    core = _make_cule_core()
    obs, infos = core.reset_batch(seeds=[SEED])
    torch.cuda.synchronize()
    rgb = core.native_rgb_frames()[0].detach().cpu().numpy().copy()
    obs84 = obs[0, 0].detach().cpu().numpy().copy()
    raw_stack = [rgb.copy() for _slot in range(FRAME_STACK)]
    obs_stack = [obs84.copy() for _slot in range(FRAME_STACK)]
    trace = [
        _trace_point(
            label="reset",
            action=-1,
            env_step=0,
            observation=np.stack(obs_stack, axis=0),
            raw_rgb_stack=np.stack(raw_stack, axis=0),
            reward=0.0,
            done=False,
            truncated=False,
            score=infos[0]["score"],
            lives=infos[0]["ale.lives"],
        )
    ]
    for env_step, action_value in enumerate(ACTIONS, start=1):
        result = core.step_batch([action_value])
        torch.cuda.synchronize()
        rgb = core.native_rgb_frames()[0].detach().cpu().numpy().copy()
        obs84 = result.observations[0, 0].detach().cpu().numpy().copy()
        done = bool(result.terminated[0].detach().cpu().item())
        truncated = bool(result.truncated[0].detach().cpu().item())
        if not bool(done or truncated):
            raw_stack = raw_stack[1:] + [rgb.copy()]
            obs_stack = obs_stack[1:] + [obs84.copy()]
        trace.append(
            _trace_point(
                label=f"step_{env_step}",
                action=action_value,
                env_step=env_step,
                observation=np.stack(obs_stack, axis=0),
                raw_rgb_stack=np.stack(raw_stack, axis=0),
                reward=float(result.rewards[0].detach().cpu().item()),
                done=done,
                truncated=truncated,
                score=int(result.scores[0].detach().cpu().item()),
                lives=int(result.lives[0].detach().cpu().item()),
            )
        )
    return trace


def _cule_latency_backend_trace() -> list[dict[str, object]]:
    from latency_bench.executors.demon_attack_gpu_batched_backend import DemonAttackGpuBatchedEnvStepBackend

    backend = DemonAttackGpuBatchedEnvStepBackend(
        config={
            "env": {
                "name": "demon_attack",
                "env_fps": 60.0,
                "obs_fps": 15.0,
                "noop_max": 30,
                "reset_mode": "fire",
                "terminal_on_life_loss": False,
                "clip_reward": False,
                "frame_stack": FRAME_STACK,
                "gpu_env_device": "cuda",
                "max_episode_steps": 25000,
            },
            "policy": {"type": "starvla"},
            "evaluation": {"eval_raw_reward": False},
        },
        noop_action=Action(value=0, name="NOOP", is_noop=True),
        num_slots=1,
    )
    try:
        reset_observation = backend.reset_slot(0, episode_id=0, seed=SEED)
        trace = [_trace_point_from_backend_observation("reset", -1, reset_observation, 0.0, False, False)]
        for env_step, action_value in enumerate(ACTIONS, start=1):
            result = backend.step_slots(
                {0: Action(value=action_value, name=str(action_value), is_noop=action_value == 0)}
            )[0].result
            observation = backend.observe_slots([0])[0]
            trace.append(
                _trace_point_from_backend_observation(
                    f"step_{env_step}",
                    action_value,
                    observation,
                    result.reward,
                    result.done,
                    result.truncated,
                )
            )
        return trace
    finally:
        backend.close()


def _trace_point_from_backend_observation(
    label: str,
    action: int,
    observation,
    reward: float,
    done: bool,
    truncated: bool,
) -> dict[str, object]:
    metadata = observation.metadata
    return _trace_point(
        label=label,
        action=action,
        env_step=int(observation.env_step),
        observation=np.asarray(observation.data, dtype=np.uint8).copy(),
        raw_rgb_stack=np.asarray(metadata[ENV_RAW_RGB_FRAME_STACK_INFO_KEY], dtype=np.uint8).copy(),
        reward=reward,
        done=done,
        truncated=truncated,
        score=metadata["score"],
        lives=metadata["ale.lives"],
    )


def _trace_point(
    *,
    label: str,
    action: int,
    env_step: int,
    observation: np.ndarray,
    raw_rgb_stack: np.ndarray,
    reward: float,
    done: bool,
    truncated: bool,
    score: int,
    lives: int,
) -> dict[str, object]:
    return {
        "label": label,
        "action": int(action),
        "env_step": np.asarray([int(env_step)], dtype=np.int64),
        "sim_time_ms": np.asarray([float(env_step) * (1000.0 / 60.0)], dtype=np.float64),
        "observation": np.asarray(observation, dtype=np.uint8).copy(),
        "raw_rgb_stack": np.asarray(raw_rgb_stack, dtype=np.uint8).copy(),
        "reward": np.asarray([float(reward)], dtype=np.float32),
        "done": np.asarray([bool(done)], dtype=np.bool_),
        "truncated": np.asarray([bool(truncated)], dtype=np.bool_),
        "score": np.asarray([int(score)], dtype=np.int64),
        "lives": np.asarray([int(lives)], dtype=np.int64),
    }
