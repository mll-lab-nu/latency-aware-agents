"""Zero-teacher → Balance VLA DAgger with explicit H8/H16 label contracts.

Owns the experiment schedule and parallel sampling. Physics/label branching,
export, distributed training and GPU inference reuse existing implementations.
"""
from __future__ import annotations

import argparse
import copy
import json
import multiprocessing as mp
import os
import shutil
import time
import traceback
from collections import Counter, deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pyarrow.parquet as pq
import torch
import wandb
import yaml

from latency_bench.core.latency_distribution import derive_seed
from latency_bench.core.config import load_config
from latency_bench.data.exporters.starvla_lerobot import convert_canonical_rollout_to_lerobot
from latency_bench.executors.realtime.pool import ProcessInferencePool
from latency_bench.envs.raw_rgb import ENV_RAW_RGB_FRAME_STACK_INFO_KEY
from scripts.starvla.balance_dagger import _balance_row, _student_observation, write_balance_dataset
from scripts.starvla.prepare_dataset import prepare_lerobot
from scripts.starvla.train_task import BACKBONES, STARVLA, environment, inference_bundle, run_command, training_config
from latency_bench.data.rollout_image_io import RolloutImageWriter
from latency_bench.integrations.sample_factory.actions import scale_box_action
from sample_factory.algo.fast_td3.models import FastTD3ActorCritic
from sample_factory.algo.mc_ppo.evaluation import comparison, episode_summary, write_json
from training.gymnasium_sf.integration import build_train_cfg_from_config, make_gymnasium_sf_env


ROOT = Path(__file__).resolve().parents[3]
INFERENCE_SEED = 941731
MODEL_HORIZONS = {"qwenoft": 8, "qwenpi_v3": 16}
LABEL_MODES = ("zero_full", "profile_suffix", "profile_executed")
TEACHER_BLOCK = 5
TEACHER = ROOT / "outputs/tasks/humanoidbench_balance_simple/h16_optimizer100k/20260915T124345Z/rl_checkpoints/balance-qwenpi_v3-h16-zero10-w128e1s1-replay6553600-100kopt-20260915T124345Z-s1-rl/checkpoint_p0/checkpoint_000100000_31400509.pth"
TEACHER_CONFIG = ROOT / "outputs/tasks/humanoidbench_balance_simple/h16_optimizer100k/20260915T124345Z/configs/qwenpi_h16_zero10_w128e1s1_optimizer100k.yaml"


def vla_experiment_config(model, label_mode):
    """Use the requested zero/profile deployment condition and model horizon."""
    config = load_config(ROOT / "configs/examples/gymnasium/g1_sonic_balance_simple/ppo_train.yaml")
    horizon = MODEL_HORIZONS[model]
    if label_mode == "zero_full":
        config["latency"] = {"method": "zero", "seed": 271828, "add_latency_info": True}
    else:
        config["latency"] = {
            "method": "temporal",
            "profile_path": str(
                ROOT / "results/profiling/profiles" / model / "1x-rtx3090/g1_sonic_balance_simple"
                / "instance_3a0d42681a03715c/profile.json"),
            "profile_worker_slot": 0,
            "seed": 271828,
            "add_latency_info": True,
        }
    config["training"]["ppo"]["chunk_horizon"] = horizon
    config["training"]["ppo"]["active_heads"] = [0, horizon]
    return config


def causal_state(observation):
    """Raw physical state plus normalized, returned-only action context."""
    return np.concatenate((observation["obs"], observation["scheduled_chunk"].reshape(-1),
                           observation["scheduled_index"])).astype(np.float32)


class VLAProfileRollout:
    """Expose admitted visual requests and the selected H8/H16 supervision mask."""

    def __init__(self, config, label_mode):
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
        self.label_mode = label_mode

    def close(self):
        self.prediction.close()
        self.env.close()

    def _teacher_suffix(self, teacher, observation, frame, start, stop, targets, valid, timings):
        """Roll the zero10 teacher from the private branch in five-frame blocks."""
        physics = self.prediction.unwrapped
        teacher_expected = {}
        commands = None
        for head in range(start, stop):
            if (head - start) % TEACHER_BLOCK == 0:
                started = time.perf_counter()
                physical = torch.as_tensor(observation, dtype=torch.float32)[None]
                with torch.no_grad():
                    commands = teacher.actor(teacher.empirical_obs_normalizer(physical, update_stats=False))
                commands = commands[0].reshape(16, 43)[:TEACHER_BLOCK].cpu().numpy()
                timings["teacher_forward_seconds"] += time.perf_counter() - started
                timings["teacher_queries"] += 1
            targets[head] = commands[(head - start) % TEACHER_BLOCK]
            valid[head] = True
            raw = scale_box_action(targets[head], self.chunk.native_action_space.low,
                                   self.chunk.native_action_space.high)
            started = time.perf_counter()
            observation, reward, terminated, _, _ = physics.step(self.chunk.env.action(raw))
            timings["branch_step_seconds"] += time.perf_counter() - started
            timings["teacher_branch_frames"] += 1
            done = terminated or frame + head + 1 == 1000
            teacher_expected[head] = (observation.copy(), reward, done)
            if done:
                break
        return teacher_expected

    def teacher_labels(self, teacher, physical_observation, timings):
        """Replay the committed prefix, then roll the teacher suffix."""
        frame = self.latency.env_step_idx
        started = time.perf_counter()
        shadow = copy.copy(self.latency)
        shadow.profile_sampler = copy.deepcopy(self.latency.profile_sampler)
        sample = shadow._next_latency_sample(worker_slot=0)
        scheduler = copy.deepcopy(self.latency.action_scheduler,
                                  {id(self.latency.action_scheduler._applied_events): []})
        timeline = copy.deepcopy(self.latency.result_timeline)
        physics = self.prediction.unwrapped
        physics.copy_balance_state_from(self.env.unwrapped)
        timings["branch_setup_seconds"] += time.perf_counter() - started
        targets = np.zeros((self.chunk.horizon, 43), dtype=np.float32)
        valid = np.zeros(self.chunk.horizon, dtype=bool)
        expected = {}
        obs = physical_observation

        started = time.perf_counter()
        prefix_length = min(sample.action_ready_raw_frames, self.chunk.horizon)
        for offset in range(prefix_length):
            current = frame + offset
            released, _ = timeline.release_arrived(current)
            for event in released:
                scheduler.enqueue(event)
            command = scheduler.update(current, self.latency.clock.step_to_time_ms(current))
            targets[offset] = self.chunk.env.reverse_action(command.value)
            obs, reward, terminated, _, _ = physics.step(command.value)
            done = terminated or current + 1 == 1000
            expected[current] = (obs.copy(), reward, done)
            if done:
                timings["prefix_replay_seconds"] += time.perf_counter() - started
                return targets, valid, expected, {}, sample
        timings["prefix_replay_seconds"] += time.perf_counter() - started
        teacher_expected = self._teacher_suffix(
            teacher, obs, frame, prefix_length, self.chunk.horizon, targets, valid, timings)
        return targets, valid, expected, teacher_expected, sample

    def requests(self, teacher, seed, slot, directory, config, *, collect, oracle, writer, timings):
        """Yield admitted RGB/state requests; receive absolute joint chunks."""
        observation, _ = self.env.reset(seed=seed)
        if self.initial_sampler is not None:
            self.latency.profile_sampler = copy.deepcopy(self.initial_sampler)
            self.latency.profile_sampler.rng.seed(derive_seed(271828, episode_idx=seed))
        rows, records, prefix = {}, {}, {}
        frames, total = 0, 0.0
        applied_heads = np.zeros(self.chunk.horizon, dtype=np.int64)
        issued = np.zeros((self.chunk.horizon, 43), dtype=np.float32)
        deferred = collect and not oracle and self.label_mode != "zero_full"
        active_source = None

        def finish(source, stop):
            record = records[source]
            targets = np.zeros((self.chunk.horizon, 43), dtype=np.float32)
            valid = np.zeros(self.chunk.horizon, dtype=bool)
            self._teacher_suffix(teacher, record["observation"], source, record["start"], stop,
                                 targets, valid, timings)
            absolute = np.stack([self.chunk.env.action(head) for head in targets])
            row = _balance_row(config["env"], record["image_path"], record["state"], absolute[0],
                               seed, source, seed, "train")
            executed = np.logical_and(record["executed"], valid)
            row.update(issued_command_chunk=absolute.tolist(),
                       teacher_valid_mask=valid.tolist(),
                       executed_action_mask=executed.tolist(),
                       action_valid_mask=(executed if self.label_mode == "profile_executed"
                                          else valid).tolist(),
                       latency_raw_frames=record["latency_raw_frames"],
                       latency_ms=record["latency_ms"])
            rows[source] = (row, valid, {})

        while True:
            if deferred:
                event, head = self.latency.action_scheduler.scheduled_chunk_event(
                    self.latency.env_step_idx,
                    self.latency.clock.step_to_time_ms(self.latency.env_step_idx))
                source = -1 if event is None else event.obs_env_step
                if source in records and source != active_source:
                    if active_source is not None and self.label_mode == "profile_executed":
                        finish(active_source, np.flatnonzero(records[active_source]["executed"])[-1] + 1)
                    started = time.perf_counter()
                    self.prediction.unwrapped.copy_balance_state_from(self.env.unwrapped)
                    timings["branch_setup_seconds"] += time.perf_counter() - started
                    records[source]["observation"] = observation["obs"].copy()
                    records[source]["start"] = head
                    active_source = source
                    if self.label_mode == "profile_suffix":
                        finish(source, self.chunk.horizon)
            admit = self.latency.worker_capacity.can_submit()
            sample = None
            if admit:
                if collect and not deferred:
                    targets, valid, expected, teacher_expected, sample = self.teacher_labels(
                        teacher, observation["obs"], timings)
                    prefix.update(expected)
                started = time.perf_counter()
                image = self.env.unwrapped.render()
                timings["render_seconds"] += time.perf_counter() - started
                timings["requests"] += 1
                state = causal_state(observation)
                if collect:
                    image_path = directory / f"images/{seed}/{frames:06d}.png"
                    started = time.perf_counter()
                    writer.write(image, image_path)
                    timings["image_submit_seconds"] += time.perf_counter() - started
                    if deferred:
                        records[frames] = {
                            "image_path": str(image_path), "state": state,
                            "executed": np.zeros(self.chunk.horizon, dtype=bool),
                        }
                    else:
                        absolute = np.stack([self.chunk.env.action(head) for head in targets])
                        row = _balance_row(config["env"], str(image_path), state, absolute[0],
                                           seed, frames, seed, "train")
                        suffix_mask = valid.tolist()
                        row.update(issued_command_chunk=absolute.tolist(),
                                   teacher_valid_mask=suffix_mask,
                                   executed_action_mask=[False] * self.chunk.horizon,
                                   action_valid_mask=suffix_mask if self.label_mode != "profile_executed"
                                   else [False] * self.chunk.horizon)
                        rows[frames] = (row, valid, teacher_expected)
                request = _student_observation(image, state, frames, slot)
                absolute = yield request
                issued = targets if oracle else np.stack([self.chunk.env.reverse_action(head) for head in absolute])
            started = time.perf_counter()
            observation, reward, terminated, truncated, info = self.env.step(issued)
            timings["env_step_seconds"] += time.perf_counter() - started
            done = terminated or truncated
            assert info["command_admitted"] == admit
            if sample is not None:
                assert (sample.action_ready_raw_frames, sample.worker_service_raw_frames) == (
                    info["latency_raw_frames"], info["worker_service_raw_frames"])
                rows[frames][0]["latency_raw_frames"] = info["latency_raw_frames"]
                rows[frames][0]["latency_ms"] = info["latency_ms"]
            elif deferred and admit:
                records[frames]["latency_raw_frames"] = info["latency_raw_frames"]
                records[frames]["latency_ms"] = info["latency_ms"]
            if frames in prefix:
                predicted, predicted_reward, predicted_done = prefix.pop(frames)
                np.testing.assert_array_equal(observation["obs"], predicted)
                assert reward == predicted_reward and done == predicted_done
            source, head = observation["replay_clock"][0, 2:4]
            if source >= 0:
                applied_heads[head] += 1
                if collect:
                    if deferred:
                        records[source]["executed"][head] = True
                        if source in rows:
                            row, valid, _ = rows[source]
                            row["executed_action_mask"][head] = bool(valid[head])
                    else:
                        row, valid, teacher_expected = rows[source]
                        row["executed_action_mask"][head] = bool(valid[head])
                        if self.label_mode == "profile_executed":
                            row["action_valid_mask"][head] = bool(valid[head])
                        if oracle and valid[head]:
                            predicted, predicted_reward, predicted_done = teacher_expected[head]
                            np.testing.assert_array_equal(observation["obs"], predicted)
                            assert reward == predicted_reward and done == predicted_done
            frames += 1
            total += reward
            if done:
                if deferred and active_source is not None and self.label_mode == "profile_executed":
                    finish(active_source, np.flatnonzero(records[active_source]["executed"])[-1] + 1)
                samples = [row for row, _, _ in rows.values() if any(row["action_valid_mask"])]
                return {"seed": seed, "length": frames, "return": total,
                        "applied_heads": applied_heads.tolist(), "valid_labels": len(samples)}, samples


def policy_config(config, bundle, vla_python, batch_size):
    result = copy.deepcopy(config)
    result["experiment"]["seed"] = INFERENCE_SEED
    result["executor"]["inference_batch_size"] = batch_size
    result["policy"] = {
        "type": "starvla", "checkpoint_path": str(bundle / "checkpoints/model.pt"),
        "task_contract_path": str(bundle / "task_contract.json"),
        "state_info_key": "gymnasium_state", "image_views_info_key": ENV_RAW_RGB_FRAME_STACK_INFO_KEY,
        "worker_python_executable": str(vla_python), "device": "cuda:0",
    }
    return result


def _audit_label_contract(rollout, teacher, directory):
    observation, _ = rollout.env.reset(seed=700001)
    sampler_state = (rollout.latency.profile_sampler.rng.getstate()
                     if rollout.latency.profile_sampler is not None else None)
    timings = Counter()
    _, valid, _, _, sample = rollout.teacher_labels(teacher, observation["obs"], timings)
    if sampler_state is not None:
        assert rollout.latency.profile_sampler.rng.getstate() == sampler_state
    prefix_length = min(sample.action_ready_raw_frames, rollout.chunk.horizon)
    assert not valid[:prefix_length].any()
    assert timings["teacher_queries"] == (valid.sum() + TEACHER_BLOCK - 1) // TEACHER_BLOCK
    write_json(directory / "label_contract_audit.json", {
        "live_sampler_unchanged": True,
        "prefix_masked": True,
        "teacher_block": TEACHER_BLOCK,
        "horizon": rollout.chunk.horizon,
        "teacher_queries": timings["teacher_queries"],
    })


def _rollout_worker(connection, config, directory, teacher_path, slots, render_device, collecting, oracle,
                    label_mode):
    """Own each assigned simulator, its private teacher branch and bounded image I/O."""
    os.environ["MUJOCO_EGL_DEVICE_ID"] = render_device
    torch.set_num_threads(1)
    try:
        with ExitStack() as cleanup:
            envs = {}
            for slot in slots:
                envs[slot] = VLAProfileRollout(config, label_mode)
                cleanup.callback(envs[slot].close)
            writer = RolloutImageWriter(max_workers=1, row_buffer_size=8, export_image_size=None)
            cleanup.callback(writer.close)
            teacher, timings = None, Counter()
            if collecting:
                first = envs[slots[0]]
                # The zero10 expert is H16, but only its trained heads 0..4 are consumed.
                teacher_actions = gym.spaces.Box(-1.0, 1.0, (16 * 43,), dtype=np.float32)
                teacher = FastTD3ActorCritic(first.env.observation_space, teacher_actions, first.cfg).eval()
                teacher.load_state_dict(torch.load(teacher_path, map_location="cpu", weights_only=False)["model"])
                teacher.requires_grad_(False)
                if 0 in slots:
                    _audit_label_contract(first, teacher, directory)
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
                    child, config, directory, args.teacher, list(range(worker, num_envs, workers)),
                    devices[worker % len(devices)], collecting, oracle, args.label_mode),
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
                    if collecting:
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


def rollout_batch(args, config, directory, *, bundle, run, count=None, seeds=None):
    """Collect/evaluate with private CPU branches and one inference worker per GPU."""
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
            pool = ProcessInferencePool(config=policy_config(config, bundle, args.vla_python, args.inference_batch_size),
                                        inference_devices=["cuda:0", "cuda:1"])
            cleanup.callback(pool.close)
        startup_seconds = time.perf_counter() - started
        episodes, rows, metrics = _collect_requests(
            envs, pool, count=count, seeds=seeds, seed_start=args.collection_seed,
            cohort_size=cohort_size, run=run)
        worker_timings = envs.flush()
    collection_seconds = time.perf_counter() - started
    conversion_started = time.perf_counter()
    if collecting:
        episode_ids = list(dict.fromkeys(row["episode_idx"] for row in rows))
        validation = set(episode_ids[::10])
        train = [row for row in rows if row["episode_idx"] not in validation]
        val = [row for row in rows if row["episode_idx"] in validation]
        for row in val:
            row["split"] = "val"
        write_balance_dataset(directory, config["env"], train, val)
        metadata_path = directory / "metadata.json"
        metadata = json.loads(metadata_path.read_text())
        teacher_valid = np.asarray([row["teacher_valid_mask"] for row in rows], dtype=np.int64)
        executed = np.asarray([row["executed_action_mask"] for row in rows], dtype=np.int64)
        supervised = np.asarray([row["action_valid_mask"] for row in rows], dtype=np.int64)
        metadata["label_provenance"] = {
            "semantics": args.label_mode,
            "teacher_checkpoint": str(args.teacher), "teacher_config": str(args.teacher_config),
            "teacher_block": TEACHER_BLOCK, "latency": config["latency"],
            "state": f"physical198+returned_normalized_chunk{args.horizon}x43+current_index{args.horizon}",
            "student_control": not oracle, "labels": len(rows),
            "teacher_valid_head_counts": teacher_valid.sum(0).tolist(),
            "executed_head_counts": executed.sum(0).tolist(),
            "supervised_head_counts": supervised.sum(0).tolist(),
            "student_checkpoint": str(bundle / "checkpoints/model.pt") if bundle is not None else None,
        }
        write_json(metadata_path, metadata)
        if not args.collect_bootstrap_only:
            converted = directory / "converted"
            convert_canonical_rollout_to_lerobot(directory, converted, action_carrier="native",
                                                normalize_state=False, write_starvla_handoff=False)
            prepare_lerobot(f"balance_profile_h{args.horizon}", converted, directory / "converted__val",
                            directory / "lerobot", normalization_source=args.normalization_source)
    finished = time.perf_counter()
    result = {**episode_summary(episodes), **metrics, "seconds": finished - started,
              "startup_seconds": startup_seconds, "collection_seconds": collection_seconds,
              "conversion_seconds": finished - conversion_started,
              "worker_timings_sum": dict(worker_timings), "labels": len(rows),
              "labels_per_second": len(rows) / (finished - started),
              "inference_seed": INFERENCE_SEED, "label_contract_audit": "passed" if collecting else None,
              "rollout_workers": min(args.rollout_workers, num_envs), "num_envs": num_envs,
              "inference_batch_size": args.inference_batch_size, "overlap": num_envs > cohort_size}
    write_json(directory / "episodes.json", result)
    return result


def reuse_bootstrap(args, config, source, directory):
    """Reuse identical profile targets/images while selecting this arm's requested mask."""
    started = time.perf_counter()
    directory.mkdir(parents=True)
    mask_column = {"profile_suffix": "teacher_valid_mask",
                   "profile_executed": "executed_action_mask"}[args.label_mode]
    for split in ("train", "val"):
        table = pq.read_table(source / f"{split}.parquet")
        table = table.set_column(table.schema.get_field_index("action_valid_mask"),
                                 "action_valid_mask", table[mask_column])
        pq.write_table(table, directory / f"{split}.parquet")
    shutil.copyfile(source / "metadata.json", directory / "metadata.json")
    metadata_path = directory / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["label_provenance"]["semantics"] = args.label_mode
    metadata["label_provenance"]["bootstrap_source"] = str(source)
    counts_key = {"profile_suffix": "teacher_valid_head_counts",
                  "profile_executed": "executed_head_counts"}[args.label_mode]
    metadata["label_provenance"]["supervised_head_counts"] = metadata["label_provenance"][counts_key]
    write_json(metadata_path, metadata)
    converted = directory / "converted"
    convert_canonical_rollout_to_lerobot(directory, converted, action_carrier="native",
                                        normalize_state=False, write_starvla_handoff=False)
    prepare_lerobot(f"balance_profile_h{args.horizon}", converted, directory / "converted__val",
                    directory / "lerobot", normalization_source=args.normalization_source)
    seconds = time.perf_counter() - started
    labels = metadata["train_rows"] + metadata["val_rows"]
    result = {"seconds": seconds, "startup_seconds": 0.0, "steady_seconds": 0.0,
              "conversion_seconds": seconds, "labels_per_second": labels / seconds,
              "generated_labels": labels, "num_envs": 0, "inference_seconds": 0.0,
              "bootstrap_source": str(source)}
    write_json(directory / "episodes.json", result)
    return result


def _wandb_stage_run(args, stage, config=None):
    suffix = "" if stage == "bootstrap" else f"-{stage}"
    disabled = args.smoke or args.collect_bootstrap_only
    return wandb.init(project="starvla_tasks", entity=None if disabled else os.environ["WANDB_ENTITY"],
                      id=args.wandb_runs[stage]["id"],
                      name=f"{args.task}-{args.label_mode}-{args.model}{suffix}",
                      group=args.run_id, mode="disabled" if disabled else "online", dir=str(args.output_dir),
                      config=config, resume="allow")


def _log_periodic_eval(args, stage, training_step, optimizer_steps, evaluation, best_score):
    with _wandb_stage_run(args, stage) as eval_run:
        eval_run.log({"global_step": training_step, "optimizer_steps": optimizer_steps,
                      "eval/mean_length": evaluation["mean_length"],
                      "eval/mean_return": evaluation["mean_return"],
                      "eval/best_mean_length": best_score, "eval/n": evaluation["n"],
                      "eval/ge100": evaluation["ge100"], "eval/ge500": evaluation["ge500"]})


def train_segment(args, bootstrap, deltas, phase, steps, boundary, pretrained, resume):
    """Native ZeRO-2 training; fail on OOM without changing the requested batch."""
    training_phase = "bootstrap" if phase == "bootstrap" else "dagger"
    config = training_config(args, bootstrap=bootstrap, deltas=deltas, backbone=args.backbone,
                             actor_checkpoint=None, run_id=training_phase, steps=steps, stop_step=boundary,
                             pretrained=pretrained, resume=resume)
    config["wandb_tags"] = ["balance", f"h{args.horizon}", args.label_mode, "zero10-teacher",
                            "smoke" if args.smoke else "formal"]
    config["wandb_entity"] = None if args.smoke else os.environ["WANDB_ENTITY"]
    config["wandb_group"] = args.run_id
    config["framework"]["qwenvl"]["enable_gradient_checkpointing"] = args.gradient_checkpointing
    config["datasets"]["vla_data"]["sequential_step_sampling"] = False
    # One held-out minibatch does not need another 32 persistent loader workers.
    config["datasets"]["vla_data"]["eval_num_workers"] = 0
    robot_type = json.loads((bootstrap / "lerobot/task_contract.json").read_text())["robot_type"]
    mixture = [[str(bootstrap / "lerobot"), 1.0, robot_type]]
    if deltas:
        mixture.extend([str(delta / "lerobot"), 0.5 / (len(deltas) - 1), robot_type]
                       for delta in deltas[:-1])
        mixture.append([str(deltas[-1] / "lerobot"), 1.0, robot_type])
    write_json(Path(config["datasets"]["vla_data"]["custom_mixtures_path"]), {
        "train": mixture,
        "validation": [[str(bootstrap / "lerobot__val"), 1.0, robot_type]],
    })
    path = args.output_dir / f"{phase}_{boundary}.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False))
    env = environment() | {"WANDB_MODE": "disabled" if args.smoke else "online",
                           "WANDB_RUN_ID": args.wandb_runs[training_phase]["id"], "WANDB_RESUME": "allow"}
    run_command([args.vla_python, "-m", "accelerate.commands.launch", "--config_file",
                 STARVLA / "starVLA/config/deepseeds/deepspeed_zero2.yaml", "--num_processes", 2,
                 "--main_process_port", args.port, STARVLA / "starVLA/training/train_starvla.py",
                 "--config_yaml", path], args.output_dir / f"{phase}_{boundary}.log", cwd=STARVLA, env=env)
    trained = args.output_dir / "training" / training_phase
    weights = trained / "checkpoints" / f"steps_{boundary}_pytorch_model.pt"
    bundle = inference_bundle(trained, weights, args.output_dir / f"bundles/{phase}_{boundary}",
                              bootstrap=bootstrap, backbone=BACKBONES["qwen"])
    state = trained / "checkpoints" / f"steps_{boundary}_state"
    return bundle, state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--vla-python", type=Path, required=True)
    parser.add_argument("--backbone", type=Path, required=True)
    parser.add_argument("--teacher", type=Path, default=TEACHER)
    parser.add_argument("--teacher-config", type=Path, default=TEACHER_CONFIG)
    parser.add_argument("--bootstrap-data", type=Path)
    parser.add_argument("--collect-bootstrap-only", action="store_true")
    parser.add_argument("--model", choices=MODEL_HORIZONS, default="qwenpi_v3")
    parser.add_argument("--label-mode", choices=LABEL_MODES, required=True)
    parser.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--port", type=int, default=29767)
    parser.add_argument("--rollout-workers", type=int, default=16)
    parser.add_argument("--inference-batch-size", type=int, default=8)
    parser.add_argument("--rollout-overlap", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    args.output_dir = args.output.resolve()
    if not args.resume:
        args.output_dir.mkdir(parents=True, exist_ok=False)
    args.run_id = f"{args.output_dir.parent.name}-{args.output_dir.name}"
    os.environ["WANDB_RUN_GROUP"] = args.run_id
    args.task = "balance"
    args.condition = "zero-latency" if args.label_mode == "zero_full" else "latency-aware"
    args.horizon = MODEL_HORIZONS[args.model]
    args.num_gpus, args.micro_batch, args.num_workers, args.prefetch_factor = 2, 64, 16, 3
    args.persistent_workers, args.pin_memory = True, True
    args.normalization_source = None
    torch.set_num_threads(1)
    torch.manual_seed(1)
    if args.resume:
        config = json.loads((args.output_dir / "environment.json").read_text())
        budget = json.loads((args.output_dir / "budget.json").read_text())
        run_record = json.loads((args.output_dir / "run.json").read_text())
        history = json.loads((args.output_dir / "history.json").read_text())
        bootstrap_data = (Path(run_record["bootstrap_data"])
                          if run_record["bootstrap_data"] is not None else None)
        development = run_record["development_seeds"]
        heldout = run_record["heldout_seeds"]
        args.wandb_runs = run_record["wandb_runs"]
    else:
        config = vla_experiment_config(args.model, args.label_mode)
        config["env"]["make_kwargs"]["render_mode"] = "rgb_array"
        config["env"]["make_kwargs"]["camera_name"] = "head_camera"
        config["backend"]["initial_model_path"] = str(args.teacher)
        write_json(args.output_dir / "environment.json", config)
        bootstrap_data = args.bootstrap_data.resolve() if args.bootstrap_data is not None else None
        profile_path = None if args.label_mode == "zero_full" else config["latency"]["profile_path"]
        budget = {"bootstrap_labels": 512 if args.smoke else 71000,
                  "bootstrap_steps": 2 if args.smoke else 5000,
                  "round_labels": 512 if args.smoke else 16000,
                  "round_steps": 2 if args.smoke else 500,
                  "rounds": 2 if args.smoke else 10, "micro_batch": 64,
                  "global_batch": 128, "gradient_checkpointing": args.gradient_checkpointing,
                  "workers_per_rank": 16, "prefetch": 3,
                  "inference_seed": INFERENCE_SEED, "profile_is_deployment_proxy": True,
                  "rollout_workers": args.rollout_workers,
                  "inference_batch_size": args.inference_batch_size,
                  "rollout_overlap": args.rollout_overlap, "model": args.model,
                  "action_horizon": args.horizon,
                  "state_dim": 198 + args.horizon * 43 + args.horizon,
                  "label_mode": args.label_mode, "teacher_block": TEACHER_BLOCK,
                  "profile_path": profile_path,
                  "bootstrap_data": str(bootstrap_data) if bootstrap_data is not None else None}
        write_json(args.output_dir / "budget.json", budget)
        write_json(args.output_dir / "semantic_checklist.json", {
            "bootstrap_training": ("data_source_only" if args.collect_bootstrap_only
                                   else "from_random_initialization"),
            "model_geometry": f"{args.model}_h{args.horizon}",
            "gradient_checkpointing": args.gradient_checkpointing,
            "teacher": "zero10_fasttd3_h16_heads_0_through_4",
            "teacher_chunking": f"closed_loop_requery_every_{TEACHER_BLOCK}_frames",
            "label_mode": args.label_mode,
            "profile_prefix": "committed_actions_in_target_but_masked",
            "profile_executed_mask": "teacher_valid_intersection_same_request_replay_clock",
            "replay": "round1_bootstrap_latest_50_50_then_bootstrap_latest_history_40_40_20",
            "dagger_optimizer": "single_continuous_optimizer_scheduler_rng_across_rounds",
            "rollout_checkpoint": "latest",
            "selection_checkpoint": "best_only_for_reporting",
            "development_eval_episodes": (0 if args.collect_bootstrap_only
                                            else (2 if args.smoke else 32)),
            "heldout_eval_episodes": (0 if args.collect_bootstrap_only
                                       else (2 if args.smoke else 32)),
        })
        development = list(range(810001, 810003 if args.smoke else 810033))
        heldout = list(range(820101, 820103)) if args.smoke else list(range(820001, 820033))
        history = []
        args.wandb_runs = {
            "bootstrap": {"id": f"{args.run_id}-bootstrap"},
            "dagger": {"id": f"{args.run_id}-dagger"},
        }
        run_record = {"wandb_runs": args.wandb_runs, "development_seeds": development,
                      "heldout_seeds": heldout, "teacher": str(args.teacher),
                      "teacher_config": str(args.teacher_config),
                      "bootstrap_data": str(bootstrap_data) if bootstrap_data is not None else None,
                      "bootstrap_bundle": None}
        write_json(args.output_dir / "run.json", run_record)
    bootstrap = args.output_dir / "bootstrap"
    args.normalization_source = bootstrap / "lerobot" if args.resume else None
    deltas = []
    best_entry = max(history, key=lambda entry: entry["evaluation"]["mean_length"]) if history else None
    best_bundle = Path(best_entry["bundle"]) if best_entry is not None else None
    best_score = best_entry["evaluation"]["mean_length"] if best_entry is not None else -float("inf")
    best_stage = ("bootstrap" if best_entry["round"] == 0 else "dagger") if best_entry else None
    optimizer_steps = history[-1]["optimizer_steps"] if history else 0
    bootstrap_best = (Path(run_record["bootstrap_bundle"])
                      if run_record["bootstrap_bundle"] is not None else None)
    rollout_bundle = Path(history[-1]["bundle"]) if history else None
    dagger_pretrained, dagger_resume = None, False
    completed = {entry["round"]: entry for entry in history}
    for round_number in range(budget["rounds"] + 1):
        stage = "bootstrap" if round_number == 0 else "dagger"
        phase = "bootstrap" if round_number == 0 else f"dagger_{round_number:02d}"
        directory = args.output_dir / phase
        if round_number in completed:
            latest = Path(completed[round_number]["bundle"])
            if round_number == 0:
                bootstrap_best = latest
                rollout_bundle = latest
                dagger_pretrained, dagger_resume = latest / "checkpoints/model.pt", False
            else:
                deltas.append(directory)
                boundary = round_number * budget["round_steps"]
                state = args.output_dir / f"training/dagger/checkpoints/steps_{boundary}_state"
                rollout_bundle = latest
                dagger_pretrained, dagger_resume = state, True
            continue
        args.collection_seed = 4100001 + round_number * 100000
        labels = budget["bootstrap_labels"] if round_number == 0 else budget["round_labels"]
        with _wandb_stage_run(args, stage, budget) as run:
            run.define_metric("eval/*", step_metric="optimizer_steps")
            run.log({"round": round_number})
            if (directory / "episodes.json").is_file():
                collection = json.loads((directory / "episodes.json").read_text())
            else:
                collection = (reuse_bootstrap(args, config, bootstrap_data, directory)
                              if round_number == 0 and bootstrap_data is not None
                              else rollout_batch(args, config, directory, bundle=rollout_bundle,
                                                 run=run, count=labels))
            run.log({"round": round_number, "collection/labels": labels,
                     **{f"collection/{key}": collection[key] for key in (
                         "seconds", "startup_seconds", "steady_seconds", "conversion_seconds",
                         "labels_per_second", "generated_labels", "num_envs", "inference_seconds")}})
            args.wandb_runs[stage]["url"] = run.url
            write_json(args.output_dir / "run.json", run_record)
        if args.collect_bootstrap_only:
            write_json(args.output_dir / "bootstrap_ready.json", {
                "bootstrap": str(directory), "labels": collection["generated_labels"],
                "model": args.model, "horizon": args.horizon, "label_mode": args.label_mode,
            })
            return
        if round_number == 0:
            args.normalization_source = bootstrap / "lerobot"
        else:
            deltas.append(directory)
        if round_number == 0:
            steps, start, end = budget["bootstrap_steps"], 0, budget["bootstrap_steps"]
            interval, pretrained, resume = steps, None, False
        else:
            steps = budget["rounds"] * budget["round_steps"]
            start, end = (round_number - 1) * budget["round_steps"], round_number * budget["round_steps"]
            interval, pretrained, resume = budget["round_steps"], dagger_pretrained, dagger_resume
        for boundary in range(start + interval, end + 1, interval):
            training_phase = "bootstrap" if round_number == 0 else "dagger"
            checkpoint_dir = args.output_dir / "training" / training_phase / "checkpoints"
            state = checkpoint_dir / f"steps_{boundary}_state"
            weights = checkpoint_dir / f"steps_{boundary}_pytorch_model.pt"
            latest = args.output_dir / f"bundles/{phase}_{boundary}"
            if not (args.resume and state.is_dir() and weights.is_file()
                    and (latest / "checkpoints/model.pt").is_file()):
                latest, state = train_segment(
                    args, bootstrap, deltas, phase, steps, boundary, pretrained, resume)
            evaluation = rollout_batch(args, config, args.output_dir / f"evaluation/{phase}_{boundary}",
                                       bundle=latest, run=None, seeds=development)
            optimizer_steps = boundary if round_number == 0 else budget["bootstrap_steps"] + boundary
            if evaluation["mean_length"] > best_score:
                best_bundle, best_score, best_stage = latest, evaluation["mean_length"], stage
            history.append({"round": round_number, "phase_step": boundary - start,
                            "optimizer_steps": optimizer_steps,
                            "bundle": str(latest), "best_bundle": str(best_bundle), "evaluation": evaluation})
            write_json(args.output_dir / "history.json", history)
            write_json(args.output_dir / "checkpoints.json", {"best": str(best_bundle), "latest": str(latest),
                       "best_mean_length": best_score, "optimizer_steps": optimizer_steps})
            _log_periodic_eval(args, stage, boundary, optimizer_steps, evaluation, best_score)
            pretrained, resume = state, True
        if round_number == 0:
            bootstrap_best = latest
            run_record["bootstrap_bundle"] = str(bootstrap_best)
            write_json(args.output_dir / "run.json", run_record)
            rollout_bundle = bootstrap_best
            dagger_pretrained, dagger_resume = bootstrap_best / "checkpoints/model.pt", False
        else:
            rollout_bundle = latest
            dagger_pretrained, dagger_resume = state, True
    results = {}
    for name, bundle in (("bootstrap", bootstrap_best), ("best", best_bundle), ("latest", latest)):
        results[name] = rollout_batch(args, config, args.output_dir / f"heldout/{name}",
                                      bundle=bundle, run=None, seeds=heldout)
    result = {"optimizer_steps": optimizer_steps,
              "new_optimizer_steps": optimizer_steps,
              "budget": budget, "heldout": results,
              "best_vs_bootstrap": comparison(results["best"]["episodes"], results["bootstrap"]["episodes"])}
    write_json(args.output_dir / "result.json", result)
    final_stage = "dagger"
    for name, stage in (("bootstrap", "bootstrap"), ("best", best_stage), ("latest", final_stage)):
        with _wandb_stage_run(args, stage) as run:
            run.log({f"heldout/{name}/mean_length": results[name]["mean_length"],
                     f"heldout/{name}/mean_return": results[name]["mean_return"],
                     f"heldout/{name}/n": results[name]["n"],
                     f"heldout/{name}/ge100": results[name]["ge100"],
                     f"heldout/{name}/ge500": results[name]["ge500"]})
            if name == "best":
                run.summary["heldout_best_mean_length"] = results[name]["mean_length"]
    with _wandb_stage_run(args, final_stage) as run:
        run.summary["completed_optimizer_steps"] = optimizer_steps
        run.summary["new_optimizer_steps"] = result["new_optimizer_steps"]
        run.summary["best_vs_bootstrap_mean_length_gain"] = result["best_vs_bootstrap"]["mean_length_gain"]
        run.summary["best_vs_bootstrap_paired_95_interval"] = result["best_vs_bootstrap"]["paired_95_interval"]
        run.summary["best_vs_bootstrap_accepted"] = result["best_vs_bootstrap"]["accepted"]


if __name__ == "__main__":
    main()
