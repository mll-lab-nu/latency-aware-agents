"""Collect Balance states visited by StarVLA and label them with FastTD3."""

from __future__ import annotations

import argparse
from importlib import import_module
import json
from pathlib import Path

import gymnasium as gym
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from latency_bench.core.actions import ActionResolver
from latency_bench.core.config import load_config
from latency_bench.core.types import Observation
from latency_bench.data.action_layouts import (
    G1_SONIC_GR00T_ACTION_SPEC,
    encode_action_for_env,
)
from latency_bench.data.hf_parquet import write_parquet_splits
from latency_bench.data.dagger import DaggerStep, collect_dagger
from latency_bench.data.rollout_image_io import save_image_frame
from latency_bench.data.sf_teacher_rows import teacher_dataset_row
from latency_bench.envs.humanoidbench_sonic_contract import (
    MUJOCO_INDEX_FOR_POLICY,
    POLICY_INDEX_FOR_MUJOCO,
)
from latency_bench.envs.raw_rgb import ENV_RAW_RGB_FRAME_STACK_INFO_KEY
from latency_bench.executors.realtime.pool import ProcessInferencePool
from latency_bench.policy.sample_factory import build_sample_factory_policy


REPO_ROOT = Path(__file__).resolve().parents[2]


def _student_observation(image: np.ndarray, state: np.ndarray, step: int, slot: int) -> Observation:
    return Observation(
        data=image,
        env_step=step,
        sim_time_ms=step * 20.0,
        metadata={
            ENV_RAW_RGB_FRAME_STACK_INFO_KEY: image[None],
            "official_gr00t_proprio": state,
            "slot_id": slot,
        },
    )


def _teacher_actions(policy, states: list[np.ndarray]) -> list:
    outputs = policy.predict_batch([
        Observation(data=state, env_step=0, sim_time_ms=0.0, metadata={"slot_id": slot})
        for slot, state in enumerate(states)
    ])
    return [output.action.value for output in outputs]


def _conditioned_policy_target(
    native,
    state: np.ndarray,
    policy_request: np.ndarray,
) -> np.ndarray:
    """Return bootstrap-compatible command labels without mutating the environment."""
    command_state = np.asarray(state, dtype=np.float64)[112:198]
    q_previous = native._joint_ranges[:, 0] + (command_state[:43] + 1.0) * 0.5 * (
        native._joint_ranges[:, 1] - native._joint_ranges[:, 0]
    )
    v_previous = command_state[43:] * native._target_velocity_limits
    request = np.asarray(policy_request, dtype=np.float64)
    request_mujoco = np.concatenate(
        [request[:29][POLICY_INDEX_FOR_MUJOCO], request[29:]]
    )
    conditioned, _ = native._condition_velocity(
        request_mujoco,
        q_previous,
        v_previous,
        native._joint_ranges,
        native._target_velocity_limits,
    )
    return np.concatenate([conditioned[:29][MUJOCO_INDEX_FOR_POLICY], conditioned[29:]])


def _relabel_saved_dataset(
    policy_config: dict,
    source_dir: Path,
    output_dir: Path,
) -> None:
    env_config = policy_config["env"]
    for module_name in env_config["registration_imports"]:
        import_module(module_name)
    make_kwargs = env_config["make_kwargs"] | {
        "scene_path": str(REPO_ROOT / env_config["make_kwargs"]["scene_path"]),
    }
    calibration_env = gym.make(env_config["env_id"], **make_kwargs)
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        native = calibration_env.unwrapped
        for split in ("train", "val"):
            source_path = source_dir / f"{split}.parquet"
            table = pq.read_table(source_path)
            states = table["state"].to_pylist()
            requests = table["action"].to_pylist()
            conditioned_actions = [
                _conditioned_policy_target(native, state, request)
                for state, request in zip(states, requests, strict=True)
            ]
            encoded_actions = [
                encode_action_for_env(
                    env_config["task_name"],
                    action,
                    action_spec=G1_SONIC_GR00T_ACTION_SPEC,
                )
                for action in conditioned_actions
            ]
            table = table.set_column(
                table.schema.get_field_index("action"),
                "action",
                pa.array(
                    [encoded["action"] for encoded in encoded_actions],
                    type=table["action"].type,
                ),
            )
            table = table.set_column(
                table.schema.get_field_index("action_text"),
                "action_text",
                pa.array(
                    [encoded["action_text"] for encoded in encoded_actions],
                    type=table["action_text"].type,
                ),
            )
            pq.write_table(table, output_dir / f"{split}.parquet")
    finally:
        calibration_env.close()
    (output_dir / "metadata.json").write_bytes(
        (source_dir / "metadata.json").read_bytes()
    )


class BalanceDaggerAdapter:
    """Keep independent Gym resets, conditioned labels, and canonical PNG rows."""

    inference_period = record_period = 1
    budget_mode = "step"
    shard_rows = None
    seed_mode = "global"

    def __init__(self, policy_config, teacher_policy, output_dir, envs):
        self.config = policy_config
        self.env_config = policy_config["env"]
        self.teacher_policy = teacher_policy
        self.output_dir, self.envs = output_dir, envs
        self.num_envs = len(envs)
        self.train_rows = []

    def reset(self, slot, state):
        self.envs[slot].reset(seed=state.seeds[slot])

    def snapshot(self, due, record_slots):
        return {
            "images": {slot: np.asarray(self.envs[slot].render()).copy() for slot in record_slots},
            "states": {slot: self.envs[slot].unwrapped._official_gr00t_proprio().copy()
                       for slot in record_slots},
        }

    def observations(self, snapshot, slots, state):
        return [
            _student_observation(snapshot["images"][slot], snapshot["states"][slot],
                                 state.episode_steps[slot], slot)
            for slot in slots
        ]

    def start_records(self, snapshot, slots, state):
        actions = _teacher_actions(self.teacher_policy, [snapshot["states"][slot] for slot in slots])
        rows = []
        for slot, action in zip(slots, actions, strict=True):
            native = self.envs[slot].unwrapped
            label = _conditioned_policy_target(
                native, snapshot["states"][slot], native.g1_absolute_request_from_raw_action(action),
            )
            image_path = (
                self.output_dir / "images/train" / f"episode_{state.episode_ids[slot]:06d}"
                / f"decision_step_{state.episode_steps[slot]:06d}.png"
            )
            row = teacher_dataset_row(
                env_name=self.env_config["task_name"],
                spec={"episode_idx": state.episode_ids[slot], "split": "train", "seed": state.seeds[slot]},
                decision_step=state.episode_steps[slot], image=str(image_path),
                raw_action=label, info={"latency_raw_frames": 0, "latency_ms": 0.0}, reward=0.0,
                env_fps=self.env_config["env_fps"], obs_fps=self.env_config["obs_fps"],
                action_spec=G1_SONIC_GR00T_ACTION_SPEC, base_prompt=self.env_config["base_prompt"],
                state=snapshot["states"][slot],
            )
            row["rgb"] = snapshot["images"][slot]
            rows.append(row)
        return rows

    def step(self, snapshot, active, state):
        done, rewards = [], {}
        for slot in active:
            _, rewards[slot], terminated, truncated, _ = self.envs[slot].step(
                state.outputs[slot].action.value,
            )
            if terminated or truncated or state.episode_steps[slot] + 1 == self.config["evaluation"]["eval_max_steps"]:
                done.append(slot)
        return DaggerStep(done, rewards)

    def finish_record(self, row, snapshot, result, slot, finished):
        row[G1_SONIC_GR00T_ACTION_SPEC.reward_field] = result.data[slot]

    def write_rows(self, shard_index, slot, rows, writer):
        for row in rows:
            writer.submit(save_image_frame, row.pop("rgb"), Path(row["image"]), None)
            self.train_rows.append(row)


def collect_balance_dagger(
    policy_config_path: Path,
    teacher_checkpoint: Path,
    output_dir: Path,
    *,
    row_budget: int,
    num_envs: int,
    round_number: int,
    relabel_from: Path | None = None,
) -> None:
    policy_config = yaml.safe_load(policy_config_path.read_text(encoding="utf-8"))
    env_config = policy_config["env"]
    if relabel_from is not None:
        _relabel_saved_dataset(policy_config, relabel_from, output_dir)
        return
    for module_name in env_config["registration_imports"]:
        import_module(module_name)

    teacher_config = load_config(
        REPO_ROOT / "configs/examples/gymnasium/g1_sonic_balance_simple/fasttd3_train.yaml"
    )
    teacher_config["backend"]["run_mode"] = "eval"
    teacher_config["env"]["make_kwargs"]["gr00t_balance_teacher_observation_state"] = True
    teacher_config["policy"] = {
        "type": "gymnasium_sf",
        "checkpoint_path": str(teacher_checkpoint),
        "device": policy_config["env"]["simulator_device"],
        "deterministic": True,
    }
    teacher_policy = build_sample_factory_policy(
        teacher_config,
        ActionResolver(teacher_config["env"]),
        policy_type="gymnasium_sf",
    )
    student_pool = ProcessInferencePool(
        config=policy_config,
        inference_devices=[policy_config["policy"]["device"]],
    )
    envs = [
        gym.make(env_config["env_id"], **env_config["make_kwargs"])
        for _ in range(num_envs)
    ]
    output_dir.mkdir(parents=True, exist_ok=True)
    adapter = BalanceDaggerAdapter(policy_config, teacher_policy, output_dir, envs)
    try:
        collect_dagger(
            adapter, student_pool, row_budget=row_budget, seed=round_number * 1_000_000,
            is_running=lambda: True,
        )
    finally:
        for env in envs:
            env.close()
    train_rows = adapter.train_rows

    write_parquet_splits(
        output_dir, train_rows, [], action_spec=G1_SONIC_GR00T_ACTION_SPEC, keep_dataset=False
    )
    gymnasium_task = {
        key: env_config[key]
        for key in (
            "task_name", "env_id", "registration_imports", "make_kwargs",
            "env_fps", "obs_fps", "frame_stack", "base_prompt",
        )
    }
    gymnasium_task["state_space"] = {
        "labels": [f"state_{index}" for index in range(len(train_rows[0]["state"]))]
    }
    gymnasium_task["action_space"] = {
        "type": "box",
        "labels": list(G1_SONIC_GR00T_ACTION_SPEC.labels),
        "dtype": "float32",
    }
    metadata = {
        "env_name": env_config["task_name"],
        "integration_name": "gymnasium",
        "action_layout": G1_SONIC_GR00T_ACTION_SPEC.layout,
        "action_labels": list(G1_SONIC_GR00T_ACTION_SPEC.labels),
        "reward_field": G1_SONIC_GR00T_ACTION_SPEC.reward_field,
        "rows_unit": "decision_step",
        "obs_fps": env_config["obs_fps"],
        "obs_stride_raw_frames": int(env_config["env_fps"] / env_config["obs_fps"]),
        "base_prompt": env_config["base_prompt"],
        "gymnasium_task": gymnasium_task,
        "train_rows": len(train_rows),
        "val_rows": 0,
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-config", type=Path, required=True)
    parser.add_argument("--teacher-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--row-budget", type=int, required=True)
    parser.add_argument("--num-envs", type=int, required=True)
    parser.add_argument("--round", type=int, required=True)
    parser.add_argument("--relabel-from", type=Path)
    args = parser.parse_args()
    collect_balance_dagger(
        args.policy_config,
        args.teacher_checkpoint,
        args.output_dir,
        row_budget=args.row_budget,
        num_envs=args.num_envs,
        round_number=args.round,
        relabel_from=args.relabel_from,
    )


if __name__ == "__main__":
    main()
