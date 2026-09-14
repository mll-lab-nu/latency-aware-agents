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
from latency_bench.data.hf_parquet import embed_image_rows
from latency_bench.data.dagger import DaggerState, DaggerStep, collect_dagger
from latency_bench.data.rollout_image_io import save_image_frame
from latency_bench.data.sf_teacher_rows import teacher_dataset_row
from latency_bench.envs.raw_rgb import ENV_RAW_RGB_FRAME_STACK_INFO_KEY
from latency_bench.executors.realtime.pool import ProcessInferencePool
from latency_bench.policy.sample_factory import build_sample_factory_policy
from training.gymnasium_sf.integration import GymnasiumActionLatencyWrapper


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
    return [np.asarray(output.raw_output, dtype=np.float32) for output in outputs]


def _relabel_saved_dataset(
    policy_config: dict,
    source_dir: Path,
    output_dir: Path,
) -> None:
    env_config = policy_config["env"]
    output_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val"):
        table = pq.read_table(source_dir / f"{split}.parquet")
        encoded_actions = [
            encode_action_for_env(
                env_config["task_name"], request,
                action_spec=G1_SONIC_GR00T_ACTION_SPEC,
            )
            for request in table["action"].to_pylist()
        ]
        for column in ("action", "action_text"):
            table = table.set_column(
                table.schema.get_field_index(column), column,
                pa.array([encoded[column] for encoded in encoded_actions], type=table[column].type),
            )
        pq.write_table(table, output_dir / f"{split}.parquet")
    (output_dir / "metadata.json").write_bytes(
        (source_dir / "metadata.json").read_bytes()
    )


class BalanceDaggerAdapter:
    """Keep independent Gym resets, direct request labels, and canonical PNG rows."""

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
            chunk = np.stack([native.g1_absolute_request_from_raw_action(head) for head in action])
            label = chunk[0]
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
            row["issued_command_chunk"] = chunk.tolist()
            row["rgb"] = snapshot["images"][slot]
            rows.append(row)
        return rows

    def step(self, snapshot, active, state):
        done, rewards = [], {}
        for slot in active:
            _, rewards[slot], terminated, truncated, _ = self.envs[slot].step(
                state.outputs[slot].action_chunk,
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


def make_collection_env(config: dict, *, seed: int):
    """Use the training arm's virtual-clock scheduler with absolute H8 requests."""
    env_config, latency = config["env"], config["latency"]
    for module in env_config["registration_imports"]:
        import_module(module)
    env = gym.make(env_config["env_id"], **env_config["make_kwargs"])
    profile = latency["method"] == "temporal"
    return GymnasiumActionLatencyWrapper(
        env, fixed_latency_raw_frames=0, env_fps=env_config["env_fps"],
        obs_fps=env_config["obs_fps"], seed=seed,
        profile_path=latency["profile_path"] if profile else None,
        profile_worker_slot=latency["profile_worker_slot"] if profile else None,
        profile_method="temporal" if profile else None,
        simulated_worker_capacity=1, add_latency_info=True,
        hold_policy=config["scheduler"]["hold_policy"],
        ordering_policy=config["scheduler"]["ordering_policy"],
        hold_last_chunk_action=config["scheduler"]["hold_last_chunk_action"],
        noop_action=env_config["noop_action"],
    )


def load_teacher(teacher_config_path: Path, teacher_checkpoint: Path, device: str):
    config = load_config(teacher_config_path)
    config["backend"]["run_mode"] = "eval"
    config["policy"] = {"type": "gymnasium_sf", "checkpoint_path": str(teacher_checkpoint),
                        "device": device, "deterministic": True}
    return build_sample_factory_policy(config, ActionResolver(config["env"]), policy_type="gymnasium_sf")


def collect_balance_dagger(
    policy_config_path: Path,
    teacher_checkpoint: Path,
    output_dir: Path,
    teacher_config_path: Path,
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

    teacher_policy = load_teacher(teacher_config_path, teacher_checkpoint, env_config["simulator_device"])
    student_pool = ProcessInferencePool(
        config=policy_config,
        inference_devices=[policy_config["policy"]["device"]],
    )
    envs = [
        make_collection_env(policy_config, seed=10_000_000 + round_number * 1_000_000 + slot)
        for slot in range(num_envs)
    ]
    output_dir.mkdir(parents=True, exist_ok=True)
    adapter = BalanceDaggerAdapter(policy_config, teacher_policy, output_dir, envs)
    try:
        collect_dagger(
            adapter, student_pool, row_budget=row_budget, seed=10_000_000 + round_number * 1_000_000,
            is_running=lambda: True,
        )
    finally:
        for env in envs:
            env.close()
    train_rows = adapter.train_rows

    write_balance_dataset(output_dir, env_config, train_rows, [])


def write_balance_dataset(output_dir, env_config, train_rows, val_rows):
    train = pa.Table.from_pylist(embed_image_rows(train_rows))
    pq.write_table(train, output_dir / "train.parquet")
    pq.write_table(pa.Table.from_pylist(embed_image_rows(val_rows), schema=train.schema), output_dir / "val.parquet")
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
        "val_rows": len(val_rows),
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def collect_balance_bootstrap(policy_config_path, teacher_config_path, teacher_checkpoint, output_dir, *, smoke):
    """Collect fresh train/validation episodes under the matching teacher arm."""
    config = load_config(policy_config_path)
    teacher = load_teacher(teacher_config_path, teacher_checkpoint, config["env"]["simulator_device"])
    output_dir.mkdir(parents=True, exist_ok=True)
    train_count, val_count = (2, 1) if smoke else (90, 10)
    train, val = [], []
    for episode in range(train_count + val_count):
        env = make_collection_env(config, seed=episode)
        env.reset(seed=episode)
        adapter = BalanceDaggerAdapter(config, teacher, output_dir, [env])
        state = DaggerState(0, [0], [0], [episode], [episode], [None])
        rows = []
        try:
            for step in range(8 if smoke else config["evaluation"]["eval_max_steps"]):
                state.episode_steps[0] = step
                snapshot = adapter.snapshot([0], [0])
                row = adapter.start_records(snapshot, [0], state)[0]
                _, reward, terminated, truncated, info = env.step(np.asarray(row["issued_command_chunk"], dtype=np.float32))
                row[G1_SONIC_GR00T_ACTION_SPEC.reward_field] = reward
                row["latency_raw_frames"] = info["latency_raw_frames"]
                row["latency_ms"] = info["latency_ms"]
                save_image_frame(row.pop("rgb"), Path(row["image"]), None)
                rows.append(row)
                if terminated or truncated:
                    break
        finally:
            env.close()
        (train if episode < train_count else val).extend(rows)
        print(f"Bootstrap episode {episode + 1}/{train_count + val_count}: {len(rows)} anchors", flush=True)
    write_balance_dataset(output_dir, config["env"], train, val)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-config", type=Path, required=True)
    parser.add_argument("--teacher-checkpoint", type=Path, required=True)
    parser.add_argument("--teacher-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--row-budget", type=int)
    parser.add_argument("--num-envs", type=int)
    parser.add_argument("--round", type=int)
    parser.add_argument("--relabel-from", type=Path)
    args = parser.parse_args()
    if args.bootstrap:
        collect_balance_bootstrap(args.policy_config, args.teacher_config, args.teacher_checkpoint, args.output_dir, smoke=args.smoke)
        return
    collect_balance_dagger(
        args.policy_config,
        args.teacher_checkpoint,
        args.output_dir,
        args.teacher_config,
        row_budget=args.row_budget,
        num_envs=args.num_envs,
        round_number=args.round,
        relabel_from=args.relabel_from,
    )


if __name__ == "__main__":
    main()
