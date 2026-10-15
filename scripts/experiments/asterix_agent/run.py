"""Preflight and evaluate a frozen Asterix protocol with existing executors."""

import argparse
from copy import deepcopy
import json
from pathlib import Path
import platform
import shutil
import time

import yaml

from latency_bench.core.actions import ActionResolver
from latency_bench.core.clock import EnvClock
from latency_bench.core.config import load_config, resolve_runtime_config
from latency_bench.envs.factory import build_env_adapter
from latency_bench.executors.factory import build_executor
from latency_bench.harness.codex import HarnessCodexRunner
from latency_bench.harness.history import PublicHistory
from latency_bench.latency.samplers import build_sampler
from latency_bench.logging.video import EpisodeVideoRecorder
from latency_bench.utils.io import write_json


FROZEN_POLICY_FILES = {"prompt_file": "prompt.txt", "model_instructions_file": "worker_instructions.txt"}


def load_run_config(path):
    config = load_config(path)
    resolve_runtime_config(config)
    for key in ("model_instructions_file", "prompt_file"):
        config["policy"][key] = str(Path(config["policy"][key]).resolve())
    for key in ("workspace_root", "runtime_root"):
        Path(config["policy"][key]).mkdir(parents=True, exist_ok=True)
    return config


def episode_config(config, directory, seed):
    run = deepcopy(config)
    run["logging"]["output_dir"] = str(directory)
    run["experiment"]["seed"] = seed
    run["latency"]["seed"] = 42 + seed
    run["evaluation"]["episodes"] = 1
    for field, name in FROZEN_POLICY_FILES.items():
        run["policy"][field] = str(Path(config["logging"]["output_dir"]) / "inputs" / name)
    return run


def freeze_config(config):
    """Claim a fresh output directory and snapshot the worker's text inputs."""
    root = Path(config["logging"]["output_dir"])
    root.mkdir(parents=True, exist_ok=True)
    with (root / "frozen_config.yaml").open("x") as stream:
        yaml.safe_dump(config, stream, sort_keys=False)
    (root / "inputs").mkdir()
    for field, name in FROZEN_POLICY_FILES.items():
        shutil.copyfile(config["policy"][field], root / "inputs" / name)


def preflight(config, *, reference_validation=True):
    freeze_config(config)
    root = Path(config["logging"]["output_dir"])
    directory = root / "preflight"
    run = episode_config(config, directory, config["evaluation"]["seeds"][0])
    resolver = ActionResolver(run["env"])
    env = build_env_adapter(run, resolver.noop_action)
    clock = EnvClock(env_fps=run["env"]["env_fps"], obs_fps=run["env"]["obs_fps"])
    history = PublicHistory(directory / "public", clock=clock,
                            image_offsets=run["harness"]["image_offsets"], action_history=8)
    policy = HarnessCodexRunner(run, resolver)
    summary = {"status": "started", "python": platform.python_version(), "actual_api_realtime_score": False}
    try:
        if reference_validation:
            validation = verify_reference(config)
            write_json(directory / "reference_validation.json", validation)
        env.reset(seed=run["evaluation"]["seeds"][0])
        history.start_episode(0)
        for frame in range(17):
            if frame % 4 == 0:
                history.observe(frame, frame, env.render_game_frame())
            if frame < 16:
                env.step(resolver.noop_action)
        observation = history.request(16, 16)
        output = policy.predict(observation)
        call = json.loads(Path(output.metadata["agent_call_dir"], "call.json").read_text())
        events = [json.loads(line) for line in Path(output.metadata["agent_call_dir"], "events.jsonl").read_text().splitlines()]
        item_types = sorted({event["params"]["item"]["type"] for event in events
                             if event.get("method") in {"item/started", "item/completed"}})
        if set(item_types) - {"userMessage", "agentMessage", "reasoning"}:
            raise ValueError(f"Unexpected native tool use: {item_types}")
        summary.update(status="passed", action=output.action.value,
                       image_frames=[item["frame"] for item in observation.metadata["harness"]["images"]],
                       token_usage=call["result"]["tokenUsage"], native_item_types=item_types,
                       effective_instructions=str(policy.episode_dir / "worker_instructions.txt"),
                       input_overhead="Total input tokens include remaining native protocol and tool definitions")
    except BaseException as error:
        summary.update(status="failed", error=repr(error))
        raise
    finally:
        write_json(directory / "summary.json", summary)
        policy.close()
        history.close()
        env.close()
    return summary


def verify_reference(config):
    """Compare the frozen strategy and actual archived IID draws before using quota."""
    reference = Path(config["provenance"]["reference_results"])
    clock = EnvClock(env_fps=config["env"]["env_fps"], obs_fps=config["env"]["obs_fps"])
    counts = {}
    for seed in config["evaluation"]["seeds"]:
        directory = reference / f"guard_intercept_mid_robust-seed{seed}" / "guard_intercept_mid_robust"
        if (directory / "prompt.txt").read_text() != Path(config["policy"]["prompt_file"]).read_text():
            raise ValueError("The frozen strategy differs from the archived reference prompt")
        run = episode_config(config, Path(config["logging"]["output_dir"]), seed)
        sampler = build_sampler(run["latency"], clock=clock)
        count = 0
        with (directory / "issued.jsonl").open() as stream:
            for line in stream:
                request = json.loads(line)
                sample = sampler.sample({"worker_slot": request["virtual_worker_slot"]})
                if sample.latency_ms != request["sampled_delay_ms"]:
                    raise ValueError(f"Archived latency mismatch: seed {seed}, request {count}")
                if request["obs_frame"] + sample.action_ready_raw_frames != request["action_ready_frame"]:
                    raise ValueError(f"Archived frame conversion mismatch: seed {seed}, request {count}")
                count += 1
        if count == 0:
            raise ValueError(f"No archived latency samples for seed {seed}")
        counts[str(seed)] = count
    return {"status": "passed", "reference": str(reference), "matching_draws_by_seed": counts,
            "strategy": "exact text match", "latency_seed_rule": "42 + seed", "profile_worker_slot": 0}


def evaluate(config):
    root = Path(config["logging"]["output_dir"])
    if yaml.safe_load((root / "frozen_config.yaml").read_text()) != config:
        raise ValueError("Configuration differs from the frozen preflight configuration")
    for field, name in FROZEN_POLICY_FILES.items():
        if Path(config["policy"][field]).read_bytes() != (root / "inputs" / name).read_bytes():
            raise ValueError(f"Frozen input changed after preflight: {field}")
    if json.loads((root / "preflight/summary.json").read_text())["status"] != "passed":
        raise ValueError("A successful subscription preflight is required")
    summary = {"status": "running", "episodes": [], "rounds": [], "actual_api_realtime_score": False,
               "historical_mean": config["provenance"]["historical_mean"], "wall_seconds": 0}
    with (root / "summary.json").open("x") as stream:
        json.dump(summary, stream, indent=2)
    started = time.perf_counter()
    try:
        for round_id in range(1, config["evaluation"]["rounds"] + 1):
            completed = []
            for seed in config["evaluation"]["seeds"]:
                directory = root / f"round_{round_id}" / f"seed_{seed}"
                directory.mkdir(parents=True)
                run = episode_config(config, directory, seed)
                write_json(directory / "config.json", run)
                episode = {"round": round_id, "seed": seed, "status": "started", "directory": str(directory)}
                summary["episodes"].append(episode)
                executor = None
                video = None
                episode_start = time.perf_counter()
                try:
                    executor = build_executor(run)
                    # The original experiment samples pooled slot 0 with random.Random(42 + seed).
                    # Bypass the normal factory's additional per-episode seed derivation.
                    executor.episode_latency_source_factory = None
                    executor.latency_source = build_sampler(run["latency"], clock=executor.clock)
                    if run["logging"]["video"]["enabled"]:
                        video = EpisodeVideoRecorder(directory / "videos", fps=config["provenance"]["normal_game_fps"])
                        executor.video_recorder = video
                        executor.video_episode_metadata = {0: {"seed": seed, "round": round_id,
                                                              "control_fps": executor.clock.env_fps}}
                    metrics = executor.run_episode(0, seed=seed, max_env_steps=run["evaluation"]["max_env_steps"])
                    episode.update(status="completed", score=metrics.episode_return, frames=metrics.survival_steps,
                                   game_seconds=metrics.survival_steps / config["provenance"]["normal_game_fps"])
                    completed.append(episode["score"])
                except BaseException as error:
                    episode.update(status="incomplete", error=repr(error))
                    if video is not None and video._writer is not None:
                        video.close_episode({"status": "incomplete"})
                    raise
                finally:
                    if executor is not None:
                        executor.close()
                        if executor.logger is not None:
                            executor.logger.flush()
                    episode["wall_seconds"] = time.perf_counter() - episode_start
                    episode.update(call_statistics(directory))
                    write_json(directory / "summary.json", episode)
                    write_json(root / "summary.json", summary)
                print(json.dumps(episode), flush=True)
            summary["rounds"].append({"round": round_id, "scores": completed,
                                      "mean": sum(completed) / len(completed)})
        summary["status"] = "completed"
    except BaseException as error:
        summary.update(status="stopped", error=repr(error))
        raise
    finally:
        summary["wall_seconds"] = time.perf_counter() - started
        write_json(root / "summary.json", summary)
        write_report(root, summary, config)
    return summary


def call_statistics(directory):
    calls = [json.loads(path.read_text()) for path in directory.glob("agent/worker_*/episode_*/decision_*/call.json")]
    tokens = {"inputTokens": 0, "cachedInputTokens": 0, "outputTokens": 0}
    measured = 0
    for call in calls:
        usage = call.get("result", {}).get("tokenUsage")
        if usage is not None:
            measured += 1
            for key in tokens:
                tokens[key] += usage["last"][key]
    event_path = directory / "public/episode_000/events.jsonl"
    events = [json.loads(line) for line in event_path.read_text().splitlines()] if event_path.exists() else []
    discarded = sum(event["kind"] == "discarded" for event in events)
    stale = sum(event["kind"] == "discarded" and event["reason"] in {"ignored_late_result", "latest_ready_discarded"}
                for event in events)
    requests = sum(event["kind"] == "request" for event in events)
    return {"calls": len(calls), "calls_with_usage": measured, "tokens": tokens,
            "discarded_results": discarded, "requests": requests,
            "discarded_fraction": discarded / requests if requests else None,
            "stale_results": stale, "stale_fraction": stale / requests if requests else None}


def write_report(root, summary, config):
    rows = ["# Asterix harness reproduction control", "",
            f"Status: {summary['status']}. Scores use sampled virtual latency and real subscription decisions.",
            "These scores are not measurements of API realtime gameplay.", "",
            "| Round | Seed | Status | Score | Calls | Input tokens | Cached tokens | Wall seconds | Stale fraction |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|"]
    for item in summary["episodes"]:
        rows.append(f"| {item['round']} | {item['seed']} | {item['status']} | {item.get('score', '—')} | "
                    f"{item['calls']} | {item['tokens']['inputTokens']} | {item['tokens']['cachedInputTokens']} | "
                    f"{item['wall_seconds']:.1f} | {item['stale_fraction']} |")
    rows.extend(["", *[f"Round {item['round']} mean: {item['mean']:.1f}." for item in summary["rounds"]], "",
                 f"Historical comparison mean: {config['provenance']['historical_mean']}. This value is not a test assertion.",
                 "Incomplete episodes have no score. Incomplete rounds have no mean.", "",
                 f"Protocol: gpt-6.1-sol, low effort, ChatGPT subscription; seeds {config['evaluation']['seeds']}; "
                 f"{config['evaluation']['rounds']} configured round(s); at most {config['evaluation']['max_env_steps']} raw frames.",
                 "Control clock: 7.5 FPS; normal game time: 60 FPS; observations every four raw frames.",
                 "Images: PIL bilinear RGB 224×224; up to three images at t−16, t−8, t.",
                 "Latency: original fitted IID pooled slot 0; seed 42 + episode seed; three real workers, virtual capacity eight.",
                 "Condition difference: minimal worker instructions replace built-in Codex coding instructions; unrelated tools are disabled.",
                 "The frozen game strategy and its supplied image/action-history prompt format match the reference.",
                 "Primary evidence: frozen_config.yaml, preflight/, round_*/seed_*/config.json, agent/, public/, steps.jsonl, actions.jsonl and videos/.",
                 "Source identity: ../deployment.json and ../source.tar.gz retain the repository revision and the reviewed source snapshot."])
    if "error" in summary:
        rows.extend(["", f"Stop reason: {summary['error']}"])
    (root / "REPORT.md").write_text("\n".join(rows) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiments/asterix_agent/reproduce.yaml")
    parser.add_argument("--phase", required=True, choices=("preflight", "evaluate", "report"))
    args = parser.parse_args()
    config = load_run_config(args.config)
    if args.phase == "preflight":
        print(json.dumps(preflight(config)), flush=True)
    elif args.phase == "evaluate":
        evaluate(config)
    else:
        root = Path(config["logging"]["output_dir"])
        frozen = yaml.safe_load((root / "frozen_config.yaml").read_text())
        write_report(root, json.loads((root / "summary.json").read_text()), frozen)


if __name__ == "__main__":
    main()
