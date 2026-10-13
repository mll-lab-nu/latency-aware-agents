"""Execute Balance H8/H16 collection and evaluation with private teacher branches."""
from __future__ import annotations

import copy
import json
import multiprocessing as mp
import os
import time
import traceback
from collections import Counter, deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace

os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"

import gymnasium as gym
import numpy as np
import torch
import pyarrow as pa
import pyarrow.parquet as pq

from latency_bench.core.latency_distribution import derive_seed
from latency_bench.core.config import load_config
from latency_bench.integrations.starvla.lerobot_export import convert_canonical_rollout_to_lerobot
from latency_bench.executors.realtime.pool import ProcessInferencePool
from latency_bench.integrations.starvla.resident_dagger import ResidentInferencePool
from latency_bench.core.types import Observation
from latency_bench.data.action_layouts import G1_SONIC_ACTION_SPEC
from latency_bench.data.hf_parquet import embed_image_rows
from latency_bench.data.rollout_rows import teacher_dataset_row
from latency_bench.policy.action_prefix import action_prefix_arrays
from latency_bench.integrations.starvla.task_contracts import prepare_lerobot
from latency_bench.data.rollout_image_io import RolloutImageWriter
from latency_bench.integrations.sample_factory.actions import scale_box_action
from sample_factory.algo.fast_td3.models import FastTD3ActorCritic
from sample_factory.algo.mc_ppo.evaluation import episode_summary, write_json
from training.gymnasium_sf.integration import build_train_cfg_from_config, make_gymnasium_sf_env


from latency_bench.policy.model_source import model_file
from latency_bench.envs.base import ENV_RAW_RGB_FRAME_STACK_INFO_KEY

INFERENCE_SEED = 941731
TEACHER_QUERY_CADENCE_RAW_FRAMES = 1


def _student_observation(
    image: np.ndarray,
    state: np.ndarray,
    step: int,
    slot: int,
    action_prefix: np.ndarray | None = None,
    action_prefix_mask: np.ndarray | None = None,
) -> Observation:
    metadata = {
        ENV_RAW_RGB_FRAME_STACK_INFO_KEY: image[None],
        "gymnasium_state": state,
        "slot_id": slot,
    }
    if action_prefix is not None:
        metadata.update(
            action_prefix=action_prefix,
            action_prefix_mask=action_prefix_mask,
        )
    return Observation(
        data=image,
        env_step=step,
        sim_time_ms=step * 20.0,
        metadata=metadata,
    )


def _balance_row_for_image_path(config, image_path, state, label, episode, step, seed, split):
    row = teacher_dataset_row(
        env_name=config["name"], spec={"episode_idx": episode, "split": split, "seed": seed},
        decision_step=step, image=str(image_path), raw_action=label,
        info={"latency_raw_frames": 0, "latency_ms": 0.0}, reward=0.0,
        env_fps=config["env_fps"], obs_fps=config["obs_fps"],
        action_spec=G1_SONIC_ACTION_SPEC, base_prompt=config["base_prompt"], state=state,
    )
    return row


def write_balance_dataset(output_dir, env_config, train_rows, val_rows):
    train = pa.Table.from_pylist(embed_image_rows(train_rows))
    pq.write_table(train, output_dir / "train.parquet")
    pq.write_table(pa.Table.from_pylist(embed_image_rows(val_rows), schema=train.schema), output_dir / "val.parquet")
    gymnasium_task = {
        key: env_config[key]
        for key in (
            "env_id", "registration_imports", "make_kwargs",
            "env_fps", "obs_fps", "frame_stack", "base_prompt",
        )
    }
    gymnasium_task["task_name"] = env_config["name"]
    gymnasium_task["state_space"] = {
        "labels": [f"state_{index}" for index in range(len(train_rows[0]["state"]))]
    }
    gymnasium_task["action_space"] = {
        "type": "box",
        "labels": list(G1_SONIC_ACTION_SPEC.labels),
        "dtype": "float32",
    }
    metadata = {
        "env_name": env_config["name"],
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


def causal_state(observation):
    """Raw physical state plus normalized, returned-only action context."""
    return np.concatenate((observation["obs"], observation["scheduled_chunk"].reshape(-1),
                           observation["scheduled_index"])).astype(np.float32)


class VLAProfileRollout:
    """Expose admitted visual requests and the selected H8/H16 supervision mask."""

    def __init__(self, config):
        self.cfg = build_train_cfg_from_config(config)
        self.cfg.episode_metrics_path = ""
        self.env = make_gymnasium_sf_env(
            self.cfg.env, self.cfg, SimpleNamespace(worker_index=0, vector_index=0, env_id=0)).env
        self.chunk = self.env.env
        self.latency = self.chunk.env.env
        self.initial_sampler = copy.deepcopy(self.latency.profile_sampler)
        kwargs = {**config["env"]["make_kwargs"], "control_mode": "direct_absolute_position",
                  "render_mode": None}
        self.prediction = gym.make(config["env"]["env_id"], **kwargs)
        self.prediction.reset(seed=1)
        self.action_prefix = config["policy"]["action_prefix"]

    def close(self):
        self.prediction.close()
        self.env.close()

    def _teacher_suffix(self, teacher, observation, frame, start, stop, targets, valid, timings):
        """Roll the H1 teacher from the private branch with one query per frame."""
        physics = self.prediction.unwrapped
        for head in range(start, stop):
            started = time.perf_counter()
            physical = torch.as_tensor(observation, dtype=torch.float32)[None]
            with torch.inference_mode():
                command = teacher.actor(teacher.empirical_obs_normalizer(physical, update_stats=False))
            targets[head] = command[0].cpu().numpy()
            timings["teacher_forward_seconds"] += time.perf_counter() - started
            timings["teacher_queries"] += 1
            valid[head] = True
            raw = scale_box_action(targets[head], self.chunk.native_action_space.low,
                                   self.chunk.native_action_space.high)
            started = time.perf_counter()
            observation, _, terminated, truncated, _ = physics.step(self.chunk.env.action(raw))
            timings["branch_step_seconds"] += time.perf_counter() - started
            timings["teacher_branch_frames"] += 1
            done = terminated or truncated or frame + head + 1 == 1000
            if done:
                break

    def teacher_labels(self, teacher, physical_observation, timings, absolute_prefix, prefix_mask):
        """Replay the student's fixed prefix, then label its counterfactual suffix."""
        frame = self.latency.env_step_idx
        started = time.perf_counter()
        physics = self.prediction.unwrapped
        physics.copy_balance_state_from(self.env.unwrapped)
        timings["branch_setup_seconds"] += time.perf_counter() - started
        targets = np.zeros((self.chunk.horizon, 43), dtype=np.float32)
        valid = np.zeros(self.chunk.horizon, dtype=bool)
        mask = np.zeros(self.chunk.horizon, dtype=bool) if prefix_mask is None else prefix_mask
        prefix_length = mask.sum().item()
        obs = physical_observation
        started = time.perf_counter()
        for offset in range(prefix_length):
            targets[offset] = self.chunk.env.reverse_action(absolute_prefix[offset])
        for offset in range(prefix_length):
            obs, _, terminated, truncated, _ = physics.step(absolute_prefix[offset])
            timings["prefix_replay_frames"] += 1
            if terminated or truncated or frame + offset + 1 == 1000:
                timings["prefix_replay_seconds"] += time.perf_counter() - started
                return targets, mask, valid
        timings["prefix_replay_seconds"] += time.perf_counter() - started
        self._teacher_suffix(
            teacher, obs, frame, prefix_length, self.chunk.horizon, targets, valid, timings)
        return targets, mask, valid

    def requests(self, teacher, seed, slot, directory, config, *, collect, oracle, writer, timings):
        """Yield admitted RGB/state requests; receive absolute joint chunks."""
        observation, _ = self.env.reset(seed=seed)
        if self.initial_sampler is not None:
            self.latency.profile_sampler = copy.deepcopy(self.initial_sampler)
            self.latency.profile_sampler.rng.seed(derive_seed(271828, episode_idx=seed))
        rows = {}
        frames, total = 0, 0.0
        applied_heads = np.zeros(self.chunk.horizon, dtype=np.int64)
        action_ready_frames, worker_service_frames = Counter(), Counter()
        request_audit = []
        issued = np.zeros((self.chunk.horizon, 43), dtype=np.float32)

        while True:
            admit = self.latency.worker_capacity.can_submit()
            if admit:
                absolute_prefix, prefix_mask = action_prefix_arrays(
                    self.latency.action_scheduler, self.latency.env_step_idx, self.action_prefix
                )
                if oracle or collect:
                    targets, label_prefix_mask, valid = self.teacher_labels(
                        teacher, observation["obs"], timings, absolute_prefix, prefix_mask)
                timings["requests"] += 1
                if oracle and not collect:
                    request = None
                else:
                    started = time.perf_counter()
                    image = self.env.unwrapped.render()
                    timings["render_seconds"] += time.perf_counter() - started
                    state = causal_state(observation)
                    if collect:
                        image_path = directory / f"images/{seed}/{frames:06d}.png"
                        started = time.perf_counter()
                        writer.write(image, image_path)
                        timings["image_submit_seconds"] += time.perf_counter() - started
                        absolute = np.stack([self.chunk.env.action(head) for head in targets])
                        row = _balance_row_for_image_path(
                            config["env"], image_path, state, absolute[0],
                            seed, frames, seed, "train")
                        row.update(
                            issued_command_chunk=absolute.tolist(),
                            action_prefix_mask=label_prefix_mask.tolist(),
                            action_loss_mask=valid.tolist(),
                        )
                        rows[frames] = row
                    request = _student_observation(
                        image, state, frames, slot, absolute_prefix, prefix_mask
                    )
                absolute = yield request
                issued = targets if oracle else np.stack([self.chunk.env.reverse_action(head) for head in absolute])
            started = time.perf_counter()
            observation, reward, terminated, truncated, info = self.env.step(issued)
            timings["env_step_seconds"] += time.perf_counter() - started
            done = terminated or truncated
            assert info["command_admitted"] == admit
            if admit:
                action_ready_frames[info["latency_raw_frames"]] += 1
                worker_service_frames[info["worker_service_raw_frames"]] += 1
                request_audit.append({
                        "obs_frame": frames,
                        "prefix_frames": None if prefix_mask is None else prefix_mask.sum().item(),
                        "actual_delay_frames": info["latency_raw_frames"],
                    })
                if collect:
                    rows[frames]["latency_raw_frames"] = info["latency_raw_frames"]
                    rows[frames]["latency_ms"] = info["latency_ms"]
            source, head = observation["replay_clock"][0, 2:4]
            if source >= 0:
                applied_heads[head] += 1
            frames += 1
            total += reward
            if done:
                samples = [row for row in rows.values() if any(row["action_loss_mask"])]
                return {"seed": seed, "length": frames, "return": total,
                        "applied_heads": applied_heads.tolist(), "valid_labels": len(samples),
                        "action_ready_frame_counts": dict(action_ready_frames),
                        "worker_service_frame_counts": dict(worker_service_frames),
                        "underflow_count": self.latency.action_scheduler.underflow_count,
                        "underflow_frames": self.latency.action_scheduler.underflow_frames,
                        "request_audit": request_audit}, samples


def policy_config(config, bundle, vla_python, batch_size, backbone):
    result = copy.deepcopy(config)
    input_keys = {"state_info_key": "gymnasium_state", "image_views_info_key": ENV_RAW_RGB_FRAME_STACK_INFO_KEY}
    result["experiment"]["seed"] = INFERENCE_SEED
    result["executor"]["inference_batch_size"] = batch_size
    result['policy'] = {'type': 'starvla', 'action_prefix': config['policy']['action_prefix'], 'task_contract_path': str(bundle / 'task_contract.json'), 'backbone_path': str(backbone), 'state_info_key': input_keys['state_info_key'], 'image_views_info_key': input_keys['image_views_info_key'], 'worker_python_executable': str(vla_python), 'checkpoint': str(Path(str(bundle / 'checkpoints/model.pt')).absolute()).lstrip('/'), 'source': 'local', 'root': '/'}
    result["executor"]["inference_devices"] = ['cuda:0']
    return result


def _rollout_worker(connection, config, directory, teacher_path, teacher_config_path, slots,
                    render_device, collecting, oracle):
    """Own each assigned simulator, its private teacher branch and bounded image I/O."""
    os.environ["MUJOCO_EGL_DEVICE_ID"] = render_device
    torch.set_num_threads(1)
    try:
        with ExitStack() as cleanup:
            envs = {}
            for slot in slots:
                envs[slot] = VLAProfileRollout(config)
                cleanup.callback(envs[slot].close)
            writer = RolloutImageWriter(max_workers=1, row_buffer_size=8, export_image_size=None)
            cleanup.callback(writer.close)
            teacher, timings = None, Counter()
            if collecting or oracle:
                first = envs[slots[0]]
                teacher_cfg = build_train_cfg_from_config(load_config(teacher_config_path))
                teacher_actions = gym.spaces.Box(
                    -1.0, 1.0, (teacher_cfg.fasttd3_action_chunk_horizon * 43,), dtype=np.float32)
                teacher = FastTD3ActorCritic(
                    first.env.observation_space, teacher_actions, teacher_cfg).eval()
                teacher.load_state_dict(torch.load(teacher_path, map_location="cpu", weights_only=False)["model"])
                teacher.requires_grad_(False)
            generators = {}
            connection.send(("ready", None))
            while True:
                command, items = connection.recv()
                if command == "close":
                    break
                if command == "flush":
                    started = time.perf_counter()
                    writer.flush()
                    timings["image_flush_seconds"] += time.perf_counter() - started
                    connection.send(("result", timings))
                    continue
                results = []
                for slot, value in items:
                    if command == "start":
                        generators[slot] = envs[slot].requests(
                            teacher, value, slot, directory, config, collect=collecting, oracle=oracle,
                            writer=writer, timings=timings)
                        results.append((slot, ("request", next(generators[slot]))))
                    elif command == "advance":
                        try:
                            results.append((slot, ("request", generators[slot].send(value))))
                        except StopIteration as end:
                            results.append((slot, ("episode", end.value)))
                            del generators[slot]
                    else:
                        raise RuntimeError(f"Unknown rollout command: {command}")
                connection.send(("result", results))
    except BaseException:
        connection.send(("error", traceback.format_exc()))
    finally:
        connection.close()


class _ProfileEnvPool:
    """Preserve simulator ownership while advancing request-sized work in parallel."""

    def __init__(self, args, config, directory, num_envs, *, collecting, oracle):
        self.num_envs, self.directory = num_envs, directory
        self.connections, self.processes = [], []
        workers = min(args.rollout_workers, num_envs)
        devices = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
        context = mp.get_context("spawn")
        try:
            for worker in range(workers):
                parent, child = context.Pipe()
                process = context.Process(target=_rollout_worker, args=(
                    child, config, directory, args.teacher, args.teacher_config,
                    list(range(worker, num_envs, workers)), devices[worker % len(devices)],
                    collecting, oracle),
                    name=f"profile-rollout-{worker}")
                process.start()
                child.close()
                self.connections.append(parent)
                self.processes.append(process)
            self._receive(range(workers), "ready")
        except BaseException:
            self.close()
            raise

    def _receive(self, workers, expected):
        results = []
        for worker in workers:
            tag, payload = self.connections[worker].recv()
            if tag == "error":
                raise RuntimeError(payload)
            if tag != expected:
                raise RuntimeError(f"Rollout worker returned {tag}, expected {expected}")
            results.append(payload)
        return results

    def exchange(self, command, items):
        grouped = [[] for _ in self.connections]
        for slot, value in items:
            grouped[slot % len(grouped)].append((slot, value))
        active = [worker for worker, batch in enumerate(grouped) if batch]
        for worker in active:
            self.connections[worker].send((command, grouped[worker]))
        return dict(item for batch in self._receive(active, "result") for item in batch)

    def flush(self):
        for connection in self.connections:
            connection.send(("flush", []))
        timings = Counter()
        for result in self._receive(range(len(self.connections)), "result"):
            timings.update(result)
        return timings

    def close(self):
        for connection, process in zip(self.connections, self.processes, strict=True):
            if process.is_alive():
                try:
                    connection.send(("close", []))
                except (BrokenPipeError, EOFError, OSError):
                    pass
        for process in self.processes:
            process.join(timeout=5)
        for process in self.processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=1)
            if process.is_alive():
                process.kill()
                process.join()
        for connection in self.connections:
            connection.close()


def _collect_requests(envs, pool, *, count, seeds, seed_start, cohort_size, run):
    """Alternate fixed cohorts without racing inference pipes or choosing fast episodes."""
    collecting = count is not None
    episodes, rows, requests = [], [], {}
    next_seed = seed_start
    seed_iterator = iter(seeds) if seeds is not None else None

    def start(slots):
        nonlocal next_seed
        assignments = []
        for slot in slots:
            if collecting:
                if len(rows) >= count:
                    break
                seed = next_seed
                next_seed += 1
            else:
                seed = next(seed_iterator, None)
                if seed is None:
                    break
            assignments.append((slot, seed))
        for slot, (_, observation) in envs.exchange("start", assignments).items():
            requests[slot] = observation

    def predict(slots):
        started = time.perf_counter()
        predictions = [None] * len(slots) if pool is None else [
            result.action_chunk for result in pool.predict_batch([requests[slot] for slot in slots])]
        return predictions, time.perf_counter() - started

    started = time.perf_counter()
    start(range(envs.num_envs))
    initial_requests_seconds = time.perf_counter() - started
    ready = deque([list(range(begin, min(begin + cohort_size, envs.num_envs)))
                   for begin in range(0, envs.num_envs, cohort_size)])
    inference_seconds, batches = 0.0, Counter()
    started = time.perf_counter()
    # A single submitter owns all GPU pipe traffic; simulation stays on the main thread.
    with ThreadPoolExecutor(max_workers=1) as executor:
        group = ready.popleft()
        slots = [slot for slot in group if slot in requests]
        pending = executor.submit(predict, slots)
        while True:
            predictions, seconds = pending.result()
            if pool is not None:
                inference_seconds += seconds
                sizes = Counter(slot % pool.worker_count for slot in slots)
                batches.update(sizes.values())
            next_group = ready.popleft() if ready else None
            if next_group is not None:
                next_slots = [slot for slot in next_group if slot in requests]
                next_pending = executor.submit(predict, next_slots)
            results = envs.exchange("advance", zip(slots, predictions, strict=True))
            finished = []
            for slot in slots:
                event, value = results[slot]
                if event == "request":
                    requests[slot] = value
                else:
                    episode, samples = value
                    episodes.append(episode)
                    rows.extend(samples)
                    del requests[slot]
                    finished.append(slot)
                    print(json.dumps({"phase": envs.directory.name, "episodes": len(episodes),
                                      "labels": len(rows), "length": episode["length"]}), flush=True)
                    if collecting and run is not None:
                        run.log({"collection/valid_labels": min(len(rows), count),
                                 "collection/episodes": len(episodes)})
            start(finished)
            if any(slot in requests for slot in group):
                ready.append(group)
            if next_group is None:
                if not ready:
                    break
                next_group = ready.popleft()
                next_slots = [slot for slot in next_group if slot in requests]
                next_pending = executor.submit(predict, next_slots)
            group, slots, pending = next_group, next_slots, next_pending
    rows.sort(key=lambda row: (row["episode_idx"], row["decision_step"]))
    metrics = {"steady_seconds": time.perf_counter() - started, "inference_seconds": inference_seconds,
               "initial_requests_seconds": initial_requests_seconds,
               "inference_batch_sizes": dict(sorted(batches.items())), "generated_labels": len(rows)}
    return episodes, rows[:count] if collecting else rows, metrics


def rollout_batch(
    args,
    config,
    directory,
    *,
    bundle,
    run,
    count=None,
    seeds=None,
    resident_control_dir=None,
    resident_step=None,
    inference_config=None,
):
    """Collect/evaluate with private CPU branches and rank-owned GPU inference."""
    started = time.perf_counter()
    directory.mkdir(parents=True, exist_ok=True)
    collecting, oracle = count is not None, bundle is None
    cohort_size = 2 if args.smoke else 2 * args.inference_batch_size
    num_envs = cohort_size * (2 if args.rollout_overlap and not oracle else 1)
    if not collecting:
        num_envs = min(num_envs, len(seeds))
    with ExitStack() as cleanup:
        envs = _ProfileEnvPool(args, config, directory, num_envs, collecting=collecting, oracle=oracle)
        cleanup.callback(envs.close)
        pool = None
        if not oracle:
            if resident_control_dir is not None:
                resident_config = json.loads(
                    (resident_control_dir / "resident_policy.json").read_text(encoding="utf-8")
                )
                pool = ResidentInferencePool(
                    socket_dir=Path(resident_config["socket_dir"]),
                    step=resident_step,
                    worker_count=args.num_gpus,
                    batch_size=args.inference_batch_size,
                )
            else:
                pool = ProcessInferencePool(
                    config=inference_config if inference_config is not None else policy_config(
                        config, bundle, args.vla_python,
                        args.inference_batch_size, args.backbone,
                    ),
                    inference_devices=(inference_config["executor"]["inference_devices"]
                        if inference_config is not None else [f"cuda:{index}" for index in range(args.num_gpus)]),
                )
            cleanup.callback(pool.close)
        startup_seconds = time.perf_counter() - started
        episodes, rows, metrics = _collect_requests(
            envs, pool, count=count, seeds=seeds, seed_start=args.collection_seed,
            cohort_size=cohort_size, run=run)
        worker_timings = envs.flush()
    collection_seconds = time.perf_counter() - started
    conversion_started = time.perf_counter()
    label_counts = {}
    if collecting:
        if args.smoke:
            train = [row for index, row in enumerate(rows) if index % 10]
            val = [row for index, row in enumerate(rows) if not index % 10]
        else:
            episode_ids = list(dict.fromkeys(row["episode_idx"] for row in rows))
            validation = set(episode_ids[::10])
            train = [row for row in rows if row["episode_idx"] not in validation]
            val = [row for row in rows if row["episode_idx"] in validation]
        for row in val:
            row["split"] = "val"
        write_balance_dataset(directory, config["env"], train, val)
        metadata_path = directory / "metadata.json"
        metadata = json.loads(metadata_path.read_text())
        prefix_masks = np.asarray([row["action_prefix_mask"] for row in rows], dtype=np.int64)
        loss_masks = np.asarray([row["action_loss_mask"] for row in rows], dtype=np.int64)
        metadata["label_provenance"] = {
            "semantics": args.label_mode,
            "action_prefix": config["policy"]["action_prefix"],
            "teacher_checkpoint": str(args.teacher), "teacher_config": str(args.teacher_config),
            "teacher_query_cadence_raw_frames": TEACHER_QUERY_CADENCE_RAW_FRAMES,
            "teacher_revision": args.teacher_revision,

            "latency": config["latency"],
            "state": f"physical198+returned_normalized_chunk{args.horizon}x43+current_index{args.horizon}",
            "student_control": not oracle, "labels": len(rows),
            "prefix_head_counts": prefix_masks.sum(0).tolist(),
            "loss_head_counts": loss_masks.sum(0).tolist(),
            "student_checkpoint": (model_file(inference_config["policy"], "checkpoint") if inference_config is not None
                                   else str(bundle / "checkpoints/model.pt") if bundle is not None else None),
        }
        label_counts = {
            "prefix_head_counts": prefix_masks.sum(0).tolist(),
            "loss_head_counts": loss_masks.sum(0).tolist(),
        }
        write_json(metadata_path, metadata)
        if not args.collect_bootstrap_only:
            converted = directory / "converted"
            convert_canonical_rollout_to_lerobot(directory, converted, action_carrier="native",
                                                write_starvla_handoff=False)
            prepare_lerobot(f"balance_profile_h{args.horizon}", converted, directory / "converted__val",
                            directory / "lerobot", normalization_source=args.normalization_source)
            contract_path = directory / "lerobot/task_contract.json"
            contract = json.loads(contract_path.read_text())
            contract["action_prefix"] = config["policy"]["action_prefix"]
            write_json(contract_path, contract)
    finished = time.perf_counter()
    applied_head_counts = np.asarray([episode["applied_heads"] for episode in episodes]).sum(0).tolist()
    action_ready_frame_counts, worker_service_frame_counts = Counter(), Counter()
    for episode in episodes:
        action_ready_frame_counts.update(episode["action_ready_frame_counts"])
        worker_service_frame_counts.update(episode["worker_service_frame_counts"])
    result = {**episode_summary(episodes), **metrics, "seconds": finished - started,
              "startup_seconds": startup_seconds, "collection_seconds": collection_seconds,
              "conversion_seconds": finished - conversion_started,
              "worker_timings_sum": dict(worker_timings), "labels": len(rows),
              "labels_per_second": len(rows) / (finished - started),
              "inference_seed": INFERENCE_SEED,
              "action_prefix": config["policy"]["action_prefix"],
              "render_backend": os.environ["MUJOCO_GL"],
              "rollout_workers": min(args.rollout_workers, num_envs), "num_envs": num_envs,
              "inference_batch_size": args.inference_batch_size, "overlap": num_envs > cohort_size,
              "teacher_controlled": oracle, "applied_head_counts": applied_head_counts,
              "action_ready_frame_counts": dict(action_ready_frame_counts),
              "worker_service_frame_counts": dict(worker_service_frame_counts), **label_counts}
    write_json(directory / "episodes.json", result)
    return result
