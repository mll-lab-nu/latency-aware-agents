"""Run one bounded, frozen Asterix research and paired-evaluation cycle."""

import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

import numpy as np
from PIL import Image
import yaml

from latency_bench.core.actions import ActionResolver
from latency_bench.envs.factory import build_env_adapter
from latency_bench.executors.factory import build_executor
from latency_bench.harness.candidate import CandidateRuntime
from latency_bench.latency.samplers import build_sampler
from latency_bench.logging.video import EpisodeVideoRecorder
from latency_bench.utils.io import write_json
from scripts.experiments.asterix_agent.research_agent import propose
from scripts.experiments.asterix_agent.research_contract import (
    apply_proposal, candidate_config, candidate_diff, freeze_candidate, promotion_decision, read_candidate,
)
from scripts.experiments.asterix_agent.run import call_statistics, load_run_config, preflight


def records(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def public_snapshot(episode, request_path):
    snapshot = json.loads(Path(request_path).read_text())
    images = Path(episode) / "public/episode_000/images"
    for item in snapshot["images"]:
        item["path"] = str(images / Path(item["path"]).name)
    for event in snapshot["events"]:
        if event["kind"] == "observation":
            event["observation"]["path"] = str(images / Path(event["observation"]["path"]).name)
    return snapshot


def replay_episode(base, episode, seed):
    """Restore the recorded action prefix and verify public pixels and physical outcomes."""
    episode = Path(episode)
    steps = records(episode / "steps.jsonl")
    observations = {event["frame"]: event["observation"] for event in records(
        episode / "public/episode_000/events.jsonl") if event["kind"] == "observation"}
    if json.loads((episode / "summary.json").read_text())["status"] != "completed" or not steps:
        raise ValueError("Research requires a completed review episode")
    resolver = ActionResolver(base["env"])
    env = build_env_adapter(base, resolver.noop_action)
    matched_images = 0
    try:
        env.reset(seed=seed)
        for frame, step in enumerate(steps):
            if step["env_step"] != frame:
                raise ValueError("Review trajectory has a missing or repeated frame")
            if frame in observations:
                actual = Image.fromarray(env.render_game_frame()).resize((224, 224), Image.Resampling.BILINEAR)
                image_path = episode / "public/episode_000/images" / Path(observations[frame]["path"]).name
                with Image.open(image_path) as expected:
                    if not np.array_equal(np.asarray(actual), np.asarray(expected)):
                        raise ValueError(f"Review image differs at frame {frame}")
                matched_images += 1
            result = env.step(resolver.resolve(step["active_action"]["value"]))
            if (result.reward != step["reward"] or bool(result.done or result.truncated) != step["done"]
                    or result.info["lives"] != step["info"]["lives"]):
                raise ValueError(f"Review physical outcome differs at frame {frame}")
    finally:
        env.close()
    return {"status": "passed", "seed": seed, "frames": len(steps), "matching_images": matched_images}


def review_evidence(episode):
    episode = Path(episode)
    steps = records(episode / "steps.jsonl")
    events = records(episode / "public/episode_000/events.jsonl")
    outcomes = []
    previous_lives = steps[0]["info"]["lives"]
    interesting = {0, len(steps) - 1}
    for step in steps:
        lives = step["info"]["lives"]
        if step["reward"] or lives != previous_lives or step["done"]:
            outcomes.append({"frame": step["env_step"], "reward": step["reward"], "lives": lives, "done": step["done"]})
        if lives < previous_lives:
            interesting.update(max(0, step["env_step"] - age) for age in (32, 16, 4))
        previous_lives = lives
    observations = [event["observation"] for event in events if event["kind"] == "observation"]
    selected = {max((item["frame"] for item in observations if item["frame"] <= frame)) for frame in interesting}
    images = [{"frame": item["frame"], "path": str(episode / "public/episode_000/images" / Path(item["path"]).name)}
              for item in observations if item["frame"] in selected][:12]
    intervals = []
    for step in steps:
        action = step["active_action"]["value"]
        if intervals and intervals[-1]["action"] == action:
            intervals[-1]["end_frame"] = step["env_step"] + 1
        else:
            intervals.append({"start_frame": step["env_step"], "end_frame": step["env_step"] + 1, "action": action})
    evidence = {"summary": json.loads((episode / "summary.json").read_text()), "outcomes": outcomes,
                "executed_intervals": intervals,
                "decision_events": [{key: value for key, value in event.items()
                                     if key in {"kind", "frame", "request_id", "source_frame", "action", "response", "reason"}}
                                    for event in events if event["kind"] in {"returned", "accepted", "discarded"}]}
    return evidence, images


def check_candidate(base, root, snapshot):
    _, settings = read_candidate(root)
    runtime = CandidateRuntime(root, base["policy"]["bubblewrap_executable"])
    prompt = runtime.call(settings["prompt_function"], "prompt", snapshot, {"base": (Path(root) / "prompt.txt").read_text()})
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("Candidate formatter must produce a nonempty prompt")
    if settings["request_function"] is not None:
        if type(runtime.call(settings["request_function"], "request", snapshot, {})) is not bool:
            raise ValueError("Candidate request policy must return a boolean")
    if runtime.call("diagnostic:verify", "diagnostic", snapshot, {}) is not True:
        raise ValueError("Candidate diagnostic must pass")
    return {"status": "passed", "prompt_characters": len(prompt), "diagnostic": "passed"}


def regression_checks(root):
    """Run evaluator-owned contracts before spending subscription quota."""
    command = [sys.executable, "-m", "pytest", "tests/harness", "tests/executors", "tests/scheduler",
               "tests/policies/test_codex_policy.py", "tests/latency", "-q"]
    with (Path(root) / "regression.log").open("w") as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=300)


def run_game(config, version, seed):
    directory = Path(config["logging"]["output_dir"])
    directory.mkdir(parents=True)
    write_json(directory / "config.json", config)
    episode = {"version": version, "seed": seed, "status": "started", "directory": str(directory)}
    write_json(directory / "summary.json", episode)
    executor = None
    video = None
    started = time.perf_counter()
    try:
        executor = build_executor(config)
        executor.episode_latency_source_factory = None
        executor.latency_source = build_sampler(config["latency"], clock=executor.clock)
        if config["logging"]["video"]["enabled"]:
            video = EpisodeVideoRecorder(directory / "videos", fps=config["provenance"]["normal_game_fps"])
            executor.video_recorder = video
            executor.video_episode_metadata = {0: {"seed": seed, "version": version,
                                                   "control_fps": executor.clock.env_fps}}
        metrics = executor.run_episode(0, seed=seed, max_env_steps=config["evaluation"]["max_env_steps"])
        episode.update(status="completed", score=metrics.episode_return, frames=metrics.survival_steps)
    except BaseException as error:
        episode.update(status="incomplete", error=repr(error))
        if video is not None and video._writer is not None:
            video.close_episode({"status": "incomplete"})
        raise
    finally:
        try:
            if executor is not None:
                executor.close()
                if executor.logger is not None:
                    executor.logger.flush()
        finally:
            episode.update(wall_seconds=time.perf_counter() - started, **call_statistics(directory))
            write_json(directory / "summary.json", episode)
    return episode


def promote(active, previous, root, frozen_files):
    """Publish only the evaluated candidate, with an atomic default-version update."""
    active = Path(active)
    active.parent.mkdir(parents=True, exist_ok=True)
    with active.with_suffix(".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        current = json.loads(active.read_text()) if active.exists() else None
        if current != previous:
            raise RuntimeError("Default strategy changed during research; promotion stopped")
        if read_candidate(root)[0] != frozen_files:
            raise RuntimeError("Candidate changed after freezing")
        temporary = active.with_suffix(".tmp")
        write_json(temporary, {"candidate_root": str(Path(root).resolve()), "report": str(Path(root).parent / "REPORT.md")})
        os.replace(temporary, active)


def report(root, summary, settings):
    root = Path(root)
    completed = sum(item["status"] == "completed" for item in summary["episodes"])
    lines = ["# Asterix auto research", "", f"Status: {summary['status']}",
             f"Completed game episodes: {completed}/{2 * len(settings['seeds'])}.",
             "", "Evaluation protocol: paired virtual latency with real subscription decisions.",
             "This protocol is not an API realtime measurement or an independent holdout evaluation.",
             f"Planned seeds: {settings['seeds']}; at most {settings['max_env_steps']} raw frames per episode.",
             "Control: 7.5 FPS; legal observations every four raw frames; normal game time: 60 FPS.",
             "Model: gpt-6.1-sol, low; three real workers, virtual capacity eight; latest_ready.",
             "Latency: existing IID pooled slot 0; initial stream 42 + seed for both versions.",
             "Changed request timing can assign those same sequential draws to different observation frames."]
    if (root / "proposal.json").exists():
        diagnosis = json.loads((root / "proposal.json").read_text())["diagnosis"]
        lines.extend(["", f"Research diagnosis at raw frame {diagnosis['frame']}:",
                      "Researcher's observation: " + diagnosis["observation"],
                      "Researcher's hypothesis: " + diagnosis["hypothesis"],
                      "Expected effect: " + diagnosis["expected_effect"]])
    if summary["episodes"]:
        lines.extend(["", "| Version | Seed | Status | Score | Frames | Calls | Input | Cached | Wall seconds | Stale fraction |",
                      "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|"])
    for item in summary["episodes"]:
        tokens = item["tokens"]
        lines.append(f"| {item['version']} | {item['seed']} | {item['status']} | {item.get('score', '—')} | "
                     f"{item.get('frames', '—')} | {item['calls']} | {tokens['inputTokens']} | "
                     f"{tokens['cachedInputTokens']} | {item['wall_seconds']:.1f} | {item['stale_fraction']} |")
    if (root / "research/call.json").exists():
        call = json.loads((root / "research/call.json").read_text())
        lines.extend(["", f"Research turn status: {call['status']}; wall seconds: {call['wall_seconds']:.1f}."])
        usage = call["result"]["tokenUsage"] if "result" in call else None
        if usage is not None:
            lines.append("Research token usage: " + json.dumps(usage["last"]))
    evidence = [name for name in ("source/", "source_revision.txt", "working_tree.diff", "protocol.json", "run_config.json",
                "research_instructions.txt", "regression.log", "evidence.json", "replay.json", "proposal.json", "research/",
                "candidate.diff", "checks.json", "parent/", "candidate/", "evaluation/") if (root / name).exists()]
    lines.extend(["", "Promotion: " + json.dumps(summary["promotion"]),
                  f"Total wall seconds: {summary['wall_seconds']:.1f}.",
                  "Subscription preflights: " + ("recorded under preflight_*/preflight/." if list(root.glob("preflight_*")) else "not run."),
                  "Incomplete episodes have no score. Only a complete paired cohort has comparison means.",
                  "Evidence: " + ", ".join(evidence) + ", summary.json."])
    if "error" in summary:
        lines.extend(["", "Stop reason: " + summary["error"]])
    (root / "REPORT.md").write_text("\n".join(lines) + "\n")


def execute(root):
    root = Path(root).resolve()
    settings = json.loads((root / "run_config.json").read_text())
    base = json.loads((root / "protocol.json").read_text())
    summary = {"status": "started", "episodes": [], "promotion": {"promoted": False, "reason": "validation_incomplete"},
               "actual_api_realtime_score": False, "wall_seconds": 0}
    started = time.perf_counter()

    def expired(signum, stack):
        raise TimeoutError("Research total wall-time budget exhausted")

    previous_handler = signal.signal(signal.SIGALRM, expired)
    previous_term = signal.signal(signal.SIGTERM, expired)
    signal.setitimer(signal.ITIMER_REAL, settings["total_seconds"])
    try:
        write_json(root / "summary.json", summary)
        regression_checks(root)
        episode = Path(settings["review_episode"])
        write_json(root / "replay.json", replay_episode(base, episode, settings["review_seed"]))
        evidence, images = review_evidence(episode)
        write_json(root / "evidence.json", evidence)
        shutil.copyfile(Path(__file__).with_name("research_instructions.txt"), root / "research_instructions.txt")
        proposal = propose(base, read_candidate(root / "parent")[0], evidence, images, root / "research", settings["research_seconds"])
        write_json(root / "proposal.json", proposal)
        if not apply_proposal(root / "parent", root / "work", proposal):
            summary["status"] = proposal["status"]
            summary["promotion"] = {"promoted": False, "reason": proposal["status"]}
            return summary
        if proposal["diagnosis"]["frame"] >= evidence["summary"]["frames"]:
            raise ValueError("Diagnosis refers to a frame outside the completed review episode")
        frozen = freeze_candidate(root / "work", root / "candidate")
        (root / "candidate.diff").write_text(candidate_diff(root / "parent", root / "candidate"))
        request_paths = sorted((episode / "public/episode_000").glob("request_*.json"))
        candidates = [(path, json.loads(path.read_text())["frame"]) for path in request_paths]
        source = max((pair for pair in candidates if pair[1] <= proposal["diagnosis"]["frame"]), key=lambda pair: pair[1])[0]
        snapshot = public_snapshot(episode, source)
        checks = {version: check_candidate(base, root / version, snapshot) for version in ("parent", "candidate")}
        write_json(root / "checks.json", checks)
        # These calls validate credentials, actual instruction overrides, images and response parsing.
        for version in ("parent", "candidate"):
            run = candidate_config(base, root / version, root / f"preflight_{version}", settings["seeds"][0], settings["max_env_steps"])
            preflight(run, reference_validation=False)
        summary["status"] = "evaluating"
        write_json(root / "summary.json", summary)
        for seed in settings["seeds"]:
            for version in ("parent", "candidate"):
                if read_candidate(root / "candidate")[0] != frozen:
                    raise RuntimeError("Candidate changed after freezing")
                directory = root / "evaluation" / f"seed_{seed}" / version
                run = candidate_config(base, root / version, directory, seed, settings["max_env_steps"])
                try:
                    item = run_game(run, version, seed)
                finally:
                    if (directory / "summary.json").exists():
                        item = json.loads((directory / "summary.json").read_text())
                        summary["episodes"].append(item)
                        write_json(root / "summary.json", summary)
                print(json.dumps(item), flush=True)
        summary["promotion"] = promotion_decision(summary["episodes"], settings["seeds"], checks_passed=True)
        if summary["promotion"]["promoted"]:
            promote(settings["active_version_file"], json.loads((root / "previous_active.json").read_text()), root / "candidate", frozen)
        summary["status"] = "completed"
        return summary
    except BaseException as error:
        summary.update(status="stopped", error=repr(error))
        summary["promotion"] = {"promoted": False, "reason": "validation_incomplete"}
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)
        signal.signal(signal.SIGTERM, previous_term)
        summary["wall_seconds"] = time.perf_counter() - started
        write_json(root / "summary.json", summary)
        report(root, summary, settings)


def prepare(config_path):
    settings = yaml.safe_load(Path(config_path).read_text())
    seeds = settings["seeds"]
    if not seeds or len(set(seeds)) != len(seeds) or any(type(seed) is not int or seed < 0 for seed in seeds):
        raise ValueError("Research seeds must be distinct nonnegative integers")
    if not 0 < settings["research_seconds"] <= settings["total_seconds"] or settings["max_env_steps"] <= 0:
        raise ValueError("Research time and frame budgets must be positive")
    base = load_run_config(settings["base_config"])
    stride = base["env"]["env_fps"] / base["env"]["obs_fps"]
    if 2 * len(seeds) * math.ceil(settings["max_env_steps"] / stride) > settings["max_game_decisions"]:
        raise ValueError("Configured episodes exceed the total game-decision budget")
    root = Path(settings["output_dir"]).resolve()
    root.mkdir(parents=True, exist_ok=False)
    source = root / "source"
    source.mkdir()
    for name in ("latency_bench", "configs", "scripts", "training", "tests"):
        shutil.copytree(name, source / name, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    shutil.copyfile("pytest.ini", source / "pytest.ini")
    shutil.copyfile(settings["source_revision_file"], root / "source_revision.txt")
    shutil.copyfile(settings["source_diff_file"], root / "working_tree.diff")
    shutil.copyfile(base["policy"]["model_instructions_file"], root / "worker_instructions.txt")
    base["policy"]["model_instructions_file"] = str(root / "worker_instructions.txt")
    base["policy"]["stop_used_percent"] = None
    active = Path(settings["active_version_file"])
    previous = json.loads(active.read_text()) if active.exists() else None
    write_json(root / "previous_active.json", previous)
    if previous is not None:
        freeze_candidate(previous["candidate_root"], root / "parent")
    else:
        parent = root / "parent"
        parent.mkdir()
        shutil.copyfile(base["policy"]["prompt_file"], parent / "prompt.txt")
        shutil.copyfile(Path(__file__).with_name("prompt.py"), parent / "strategy.py")
        shutil.copyfile(Path(__file__).with_name("diagnostic.py"), parent / "diagnostic.py")
        write_json(parent / "settings.json", {**base["harness"], "request_function": None, "prompt_function": "strategy:format_prompt"})
        read_candidate(parent)
    write_json(root / "protocol.json", base)
    write_json(root / "run_config.json", settings)
    return root


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiments/asterix_agent/research.yaml")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--execute", type=Path, help=argparse.SUPPRESS)
    mode.add_argument("--report", type=Path, help="Regenerate a completed or stopped cycle's report")
    args = parser.parse_args()
    if args.report is not None:
        root = args.report
        report(root, json.loads((root / "summary.json").read_text()), json.loads((root / "run_config.json").read_text()))
    elif args.execute is not None:
        execute(args.execute)
    else:
        root = prepare(args.config)
        environment = {**os.environ, "PYTHONPATH": str(root / "source"), "PYTHONDONTWRITEBYTECODE": "1"}
        result = subprocess.run([sys.executable, "-m", "scripts.experiments.asterix_agent.research", "--execute", str(root)],
                                cwd=root / "source", env=environment)
        raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
