"""CPU (pygame) adapter vs GPU batched backend equivalence for Flappy.

The simulated openvla-flappy runs historically used the CPU path
(`use_gpu_render: false` -> SerialEnvStepBackend -> pygame render). Switching
sim to the batched GPU render path is only safe if both paths are equivalent
where it matters for a starvla policy:

- the native raw RGB frame stack (`ENV_RAW_RGB_FRAME_STACK_INFO_KEY`) is the
  ONLY pixel input starvla consumes (`_build_example` resizes it itself), so
  it must be bit-exact;
- game dynamics (reward / done / truncated / score) must match step for step.

`obs.data` is intentionally NOT compared: the CPU adapter produces it with a
PIL bilinear resize while the GPU pipeline uses torch interpolation, and no
starvla code reads it.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np
import pytest

from latency_bench.core.types import Action
from latency_bench.envs.flappy import FlappyBirdEnvAdapter
from latency_bench.executors.env_step_backend import SerialEnvStepBackend
from latency_bench.executors.factory import build_executor
from training.common.raw_rgb_export import ENV_RAW_RGB_FRAME_STACK_INFO_KEY


def _noop() -> Action:
    return Action(value=0, name="noop", is_noop=True)


def _flap() -> Action:
    return Action(value=1, name="flap", is_oneshot=True)


def _gpu_backend_config(output_dir: str, *, frame_stack: int) -> dict[str, Any]:
    return {
        "experiment": {
            "name": "flappy_cpu_vs_gpu_render_equivalence_test",
            "seed": 0,
        },
        "executor": {
            "mode": "simulated",
        },
        "env": {
            "name": "flappy",
            "gym_id": "FlappyBird-v0",
            "env_fps": 30,
            "obs_fps": 30,
            "noop_action": "noop",
            "action_map": {
                "noop": 0,
                "flap": 1,
            },
            "oneshot_actions": ["flap"],
            "pipe_gap": 100,
            "bird_color": "yellow",
            "pipe_color": "green",
            "background": "day",
            "obs_resize": [84, 84],
            "frame_stack": int(frame_stack),
            "use_gpu_render": True,
            "simulator": "cpu",
            "gpu_render_device": "cpu",
            "gpu_render_profile": False,
            "gpu_render_profile_interval": 1000,
        },
        "latency": {
            "type": "zero",
        },
        "scheduler": {
            "hold_policy": "one_frame_then_noop",
            "ordering_policy": "latest_ready",
        },
        "policy": {
            "type": "starvla",
            "actions": ["noop", "flap"],
        },
        "evaluation": {
            "eval_episodes": 1,
            "eval_max_steps": 6,
            "eval_parallel_envs": 2,
        },
        "logging": {
            "output_dir": output_dir,
            "save_step_records": True,
            "save_action_records": True,
            "save_latency_records": True,
        },
    }


def _cpu_serial_backend(num_slots: int, *, frame_stack: int) -> SerialEnvStepBackend:
    return SerialEnvStepBackend(
        [
            FlappyBirdEnvAdapter(
                gym_id="FlappyBird-v0",
                env_fps=30,
                noop_action=_noop(),
                pipe_gap=100,
                bird_color="yellow",
                pipe_color="green",
                background="day",
                obs_resize=(84, 84),
                frame_stack=int(frame_stack),
                use_gpu_render=False,
                policy_observation_layout="hwc",
                export_env_raw_rgb_frames=True,
            )
            for _slot in range(num_slots)
        ]
    )


def _raw_stacks(backend, slot_ids: list[int]) -> dict[int, np.ndarray]:
    observations = backend.observe_slots(slot_ids)
    return {
        slot_id: np.asarray(observations[slot_id].metadata[ENV_RAW_RGB_FRAME_STACK_INFO_KEY])
        for slot_id in slot_ids
    }


def _assert_raw_stacks_equal(backend_a, backend_b, slot_ids: list[int]) -> None:
    stacks_a = _raw_stacks(backend_a, slot_ids)
    stacks_b = _raw_stacks(backend_b, slot_ids)
    for slot_id in slot_ids:
        np.testing.assert_array_equal(stacks_a[slot_id], stacks_b[slot_id])


@pytest.mark.parametrize("frame_stack", [1, 4])
def test_flappy_cpu_adapter_matches_gpu_batched_backend_raw_stack_and_dynamics(tmp_path, frame_stack):
    pytest.importorskip("torch")
    from latency_bench.executors.flappy_gpu_batched_backend import FlappyGpuBatchedEnvStepBackend

    slot_ids = [0, 1]
    config = _gpu_backend_config(str(tmp_path), frame_stack=frame_stack)
    cpu = _cpu_serial_backend(len(slot_ids), frame_stack=frame_stack)
    gpu = FlappyGpuBatchedEnvStepBackend(config=config, noop_action=_noop(), num_slots=len(slot_ids))
    try:
        for slot_id, seed in zip(slot_ids, [101, 202]):
            cpu.reset_slot(slot_id, episode_id=slot_id, seed=seed)
            gpu.reset_slot(slot_id, episode_id=slot_id, seed=seed)
        _assert_raw_stacks_equal(cpu, gpu, slot_ids)

        action_plan = [
            {0: _noop(), 1: _flap()},
            {0: _flap(), 1: _noop()},
            {0: _flap()},
            {1: _flap()},
            {0: _noop(), 1: _noop()},
            {0: _flap(), 1: _flap()},
        ]
        for actions_by_slot in action_plan:
            cpu_responses = cpu.step_slots(actions_by_slot)
            gpu_responses = gpu.step_slots(actions_by_slot)
            assert set(gpu_responses) == set(cpu_responses)
            for slot_id in cpu_responses:
                cpu_result = cpu_responses[slot_id].result
                gpu_result = gpu_responses[slot_id].result
                # The GPU backend rounds rewards through float32; the CPU
                # adapter keeps float64. Same value, different representation.
                assert np.float32(gpu_result.reward) == np.float32(cpu_result.reward)
                assert gpu_result.done is cpu_result.done
                assert gpu_result.truncated is cpu_result.truncated
                assert gpu_result.info.get("score") == cpu_result.info.get("score")
                # The done/auto-reset contract differs by design (the GPU
                # backend auto-resets in step); keep this test on the
                # alive-path so every compared frame has one provenance.
                assert cpu_result.done is False and cpu_result.truncated is False
            _assert_raw_stacks_equal(cpu, gpu, slot_ids)
    finally:
        gpu.close()
        cpu.close()


def _f32(value: Any) -> Any:
    value = np.float32(value)
    return None if np.isnan(value) else value


def _metrics_contract(metrics: list[Any]) -> list[tuple[Any, ...]]:
    return [
        (
            metric.episode_id,
            _f32(metric.episode_return),
            metric.survival_steps,
            _f32(metric.episode_return_env),
            _f32(metric.return_raw),
            metric.game_score,
            metric.num_actions,
            metric.num_dropped_actions,
            metric.num_invalid_actions,
        )
        for metric in metrics
    ]


def test_flappy_cpu_serial_executor_matches_gpu_batched_executor_episode_metrics(tmp_path):
    pytest.importorskip("torch")

    cpu_config = _gpu_backend_config(str(tmp_path / "cpu"), frame_stack=4)
    cpu_config["policy"] = {"type": "constant", "action": "flap", "actions": ["noop", "flap"]}
    cpu_config["evaluation"]["eval_episodes"] = 3
    cpu_config["evaluation"]["eval_parallel_envs"] = 3
    cpu_config["env"]["use_gpu_render"] = False
    cpu_config["evaluation"]["latency_bench_env_backend"] = "serial"

    gpu_config = deepcopy(cpu_config)
    gpu_config["logging"]["output_dir"] = str(tmp_path / "gpu")
    gpu_config["env"]["use_gpu_render"] = True
    gpu_config["evaluation"]["latency_bench_env_backend"] = "flappy_gpu_batched"

    cpu_executor = build_executor(cpu_config)
    gpu_executor = build_executor(gpu_config)
    try:
        episode_ids = [0, 1, 2]
        seeds = [11, 22, 33]
        cpu_metrics = cpu_executor.run_episodes(episode_ids=episode_ids, seeds=seeds, eval_max_steps=6)
        gpu_metrics = gpu_executor.run_episodes(episode_ids=episode_ids, seeds=seeds, eval_max_steps=6)
    finally:
        gpu_executor.close()
        cpu_executor.close()

    assert _metrics_contract(gpu_metrics) == _metrics_contract(cpu_metrics)
