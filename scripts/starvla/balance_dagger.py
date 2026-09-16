"""Collect Balance states visited by StarVLA and label them with FastTD3."""

from __future__ import annotations

import argparse
import copy
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
    G1_SONIC_ACTION_SPEC,
    encode_action_for_env,
)
from latency_bench.data.hf_parquet import embed_image_rows
from latency_bench.data.dagger import DaggerStep, collect_dagger
from latency_bench.data.rollout_image_io import save_image_frame
from latency_bench.data.sf_teacher_rows import teacher_dataset_row
from latency_bench.latency.samplers import HiddenRegimeSampler
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
            "gymnasium_state": state,
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
                action_spec=G1_SONIC_ACTION_SPEC,
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


def _copy_collection_env(source, target):
    """Copy a Balance branch without aliasing live physics or scheduler state."""
    memo = {id(source.clock): source.clock}
    sampler = source.profile_sampler
    # Fitted tables are immutable; sampler RNG and regime remain branch-owned.
    if isinstance(sampler, HiddenRegimeSampler):
        for table in (sampler.distributions, sampler.model, sampler.rank_processes, sampler.pre_worker_time):
            memo[id(table)] = table
    while source is not source.unwrapped:
        memo[id(source.env)] = target.env
        target.__dict__.update(copy.deepcopy(source.__dict__, memo))
        source, target = source.env, target.env
    target.copy_balance_state_from(source)


def _teacher_requests(policy, envs):
    actions = _teacher_actions(policy, [env.unwrapped._policy_observation() for env in envs])
    return [np.stack([env.unwrapped.g1_absolute_request_from_raw_action(head) for head in action])
            for env, action in zip(envs, actions, strict=True)]


def _teacher_rollout(policy, envs, remaining_steps, minimum_sources):
    """Label eight physical frames, excluding inherited student requests."""
    chunks = np.zeros((len(envs), 8, 43), dtype=np.float32)
    masks = np.zeros((len(envs), 8), dtype=bool)
    active = list(range(len(envs)))
    for index in range(8):
        active = [slot for slot in active if index < remaining_steps[slot]]
        if not active:
            break
        requests = _teacher_requests(policy, [envs[slot] for slot in active])
        done = []
        for slot, request in zip(active, requests, strict=True):
            env = envs[slot]
            _, _, terminated, truncated, info = env.step(request)
            if info["applied_source_raw_frame"] >= minimum_sources[slot]:
                chunks[slot, index] = env.current_applied_action
                masks[slot, index] = True
            if terminated or truncated:
                done.append(slot)
        active = [slot for slot in active if slot not in done]
    return chunks, masks


def _balance_row(config, output_dir, image, state, label, episode, step, seed, split):
    image_path = output_dir / f"images/{split}" / f"episode_{episode:06d}" / f"decision_step_{step:06d}.png"
    row = teacher_dataset_row(
        env_name=config["task_name"], spec={"episode_idx": episode, "split": split, "seed": seed},
        decision_step=step, image=str(image_path), raw_action=label,
        info={"latency_raw_frames": 0, "latency_ms": 0.0}, reward=0.0,
        env_fps=config["env_fps"], obs_fps=config["obs_fps"],
        action_spec=G1_SONIC_ACTION_SPEC, base_prompt=config["base_prompt"], state=state,
    )
    row["rgb"] = image
    return row


class BalanceDaggerAdapter:
    """Keep the student trajectory live while its own teacher labels independent branches."""

    inference_period = record_period = 1
    budget_mode = "step"
    shard_rows = None
    seed_mode = "global"

    def __init__(self, policy_config, teacher_policy, output_dir, envs, branches):
        self.config = policy_config
        self.env_config = policy_config["env"]
        self.teacher_policy = teacher_policy
        self.output_dir, self.envs, self.branches = output_dir, envs, branches
        self.num_envs = len(envs)
        self.train_rows = []
        self.teacher_branch_raw_frames = self.student_raw_frames = 0

    def reset(self, slot, state):
        self.envs[slot].reset(seed=state.seeds[slot])

    def snapshot(self, due, record_slots):
        return {
            "images": {slot: np.asarray(self.envs[slot].render()).copy() for slot in record_slots},
            "states": {slot: self.envs[slot].unwrapped._policy_observation().copy()
                       for slot in record_slots},
        }

    def observations(self, snapshot, slots, state):
        return [
            _student_observation(snapshot["images"][slot], snapshot["states"][slot],
                                 state.episode_steps[slot], slot)
            for slot in slots
        ]

    def start_records(self, snapshot, slots, state):
        for slot in slots:
            _copy_collection_env(self.envs[slot], self.branches[slot])
        chunks, masks = _teacher_rollout(
            self.teacher_policy, [self.branches[slot] for slot in slots],
            [self.config["evaluation"]["eval_max_steps"] - state.episode_steps[slot] for slot in slots],
            [self.envs[slot].env_step_idx for slot in slots],
        )
        self.teacher_branch_raw_frames += sum(self.branches[slot].env_step_idx - self.envs[slot].env_step_idx for slot in slots)
        rows = []
        for slot, chunk, mask in zip(slots, chunks, masks, strict=True):
            if not mask.any():
                rows.append(None)
                continue
            row = _balance_row(self.env_config, self.output_dir, snapshot["images"][slot],
                               snapshot["states"][slot], chunk[0], state.episode_ids[slot],
                               state.episode_steps[slot], state.seeds[slot], "train")
            row["issued_command_chunk"] = chunk.tolist()
            row["action_valid_mask"] = mask.tolist()
            rows.append(row)
        return rows

    def step(self, snapshot, active, state):
        done, rewards = [], {}
        self.student_raw_frames += len(active)
        for slot in active:
            _, rewards[slot], terminated, truncated, _ = self.envs[slot].step(
                state.outputs[slot].action_chunk,
            )
            if terminated or truncated or state.episode_steps[slot] + 1 == self.config["evaluation"]["eval_max_steps"]:
                done.append(slot)
        return DaggerStep(done, rewards)

    def finish_record(self, row, snapshot, result, slot, finished):
        row[G1_SONIC_ACTION_SPEC.reward_field] = result.data[slot]

    def write_rows(self, shard_index, slot, rows, writer):
        for row in rows:
            writer.submit(save_image_frame, row.pop("rgb"), Path(row["image"]), None)
            self.train_rows.append(row)


def make_collection_env(config: dict, *, seed: int, render: bool = True):
    """Use the training arm's virtual-clock scheduler with absolute H8 requests."""
    env_config, latency = config["env"], config["latency"]
    for module in env_config["registration_imports"]:
        import_module(module)
    kwargs = env_config["make_kwargs"] if render else env_config["make_kwargs"] | {"render_mode": None}
    env = gym.make(env_config["env_id"], **kwargs)
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
    branches = [make_collection_env(policy_config, seed=slot, render=False) for slot in range(num_envs)]
    adapter = BalanceDaggerAdapter(policy_config, teacher_policy, output_dir, envs, branches)
    try:
        collect_dagger(
            adapter, student_pool, row_budget=row_budget, seed=10_000_000 + round_number * 1_000_000,
            is_running=lambda: True,
        )
    finally:
        for env in (*envs, *branches):
            env.close()
    train_rows = adapter.train_rows

    write_balance_dataset(output_dir, env_config, train_rows, [])
    _record_label_provenance(output_dir, policy_config, teacher_config_path, teacher_checkpoint,
                             student_raw_frames=adapter.student_raw_frames,
                             teacher_branch_raw_frames=adapter.teacher_branch_raw_frames)


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
        "labels": list(G1_SONIC_ACTION_SPEC.labels),
        "dtype": "float32",
    }
    metadata = {
        "env_name": env_config["task_name"],
        "integration_name": "gymnasium",
        "action_layout": G1_SONIC_ACTION_SPEC.layout,
        "action_labels": list(G1_SONIC_ACTION_SPEC.labels),
        "reward_field": G1_SONIC_ACTION_SPEC.reward_field,
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


def _record_label_provenance(output_dir, config, teacher_config_path, teacher_checkpoint, **counts):
    path = output_dir / "metadata.json"
    metadata = json.loads(path.read_text())
    metadata["label_provenance"] = {
        "semantics": "expert_closed_loop_selected_absolute_requests_v2",
        "condition": "latency-aware" if config["latency"]["method"] == "temporal" else "zero-latency",
        "teacher_checkpoint": str(teacher_checkpoint.resolve()),
        "teacher_config": str(teacher_config_path.resolve()),
        "latency": config["latency"], "scheduler": config["scheduler"],
    }
    metadata.update(counts)
    path.write_text(json.dumps(metadata, indent=2) + "\n")


def collect_balance_bootstrap(policy_config_path, teacher_config_path, teacher_checkpoint, output_dir, *, smoke):
    """Slice physical-frame request windows from this arm's teacher-controlled episodes."""
    config = load_config(policy_config_path)
    teacher = load_teacher(teacher_config_path, teacher_checkpoint, config["env"]["simulator_device"])
    output_dir.mkdir(parents=True, exist_ok=True)
    train_count, val_count = (2, 1) if smoke else (90, 10)
    train, val = [], []
    teacher_raw_frames = 0
    for episode in range(train_count + val_count):
        env = make_collection_env(config, seed=episode)
        env.reset(seed=episode)
        rows, selected, valid = [], [], []
        split = "train" if episode < train_count else "val"
        try:
            for step in range(8 if smoke else config["evaluation"]["eval_max_steps"]):
                image = np.asarray(env.render()).copy()
                state = env.unwrapped._policy_observation().copy()
                request = _teacher_requests(teacher, [env])[0]
                row = _balance_row(config["env"], output_dir, image, state, request[0],
                                   episode, step, episode, split)
                _, reward, terminated, truncated, info = env.step(request)
                row[G1_SONIC_ACTION_SPEC.reward_field] = reward
                row["latency_raw_frames"] = info["latency_raw_frames"]
                row["latency_ms"] = info["latency_ms"]
                selected.append(np.asarray(env.current_applied_action).copy())
                valid.append(info["applied_source_raw_frame"] >= 0)
                rows.append(row)
                if terminated or truncated:
                    break
        finally:
            env.close()
        teacher_raw_frames += len(rows)
        labeled = []
        for index, row in enumerate(rows):
            length = min(8, len(rows) - index)
            chunk = np.zeros((8, 43), dtype=np.float32)
            mask = np.zeros(8, dtype=bool)
            mask[:length] = valid[index:index + length]
            chunk[:length] = selected[index:index + length]
            chunk[~mask] = 0
            if mask.any():
                row["issued_command_chunk"] = chunk.tolist()
                row["action_valid_mask"] = mask.tolist()
                save_image_frame(row.pop("rgb"), Path(row["image"]), None)
                labeled.append(row)
        (train if split == "train" else val).extend(labeled)
        print(f"Bootstrap episode {episode + 1}/{train_count + val_count}: {len(labeled)} anchors", flush=True)
    write_balance_dataset(output_dir, config["env"], train, val)
    _record_label_provenance(output_dir, config, teacher_config_path, teacher_checkpoint,
                             teacher_raw_frames=teacher_raw_frames)


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
