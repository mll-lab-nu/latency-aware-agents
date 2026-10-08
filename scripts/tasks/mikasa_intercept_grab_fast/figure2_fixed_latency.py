"""Run and summarize the Figure 2 MIKASA H1/H8 fixed-latency sweep."""

from __future__ import annotations

import argparse
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import threading
from typing import Any

import yaml


MODEL_REPO_ID = "MLL-Lab/LAGEN-models"
MODEL_REVISION = "56b2c095d76ea5b10dd304e6e03c2ca81801630b"
H8_MODEL_REVISION = "5b533937a72e9fa923ecf28b1c1be5a16bad488d"
BACKBONE_REPO_ID = "Qwen/Qwen3-VL-4B-Instruct"
BACKBONE_REVISION = "ebb281ec70b05090aa6165b016eac8ec08e71b17"
BENCHMARK_REVISION = "0ce3400b86f01252c563e76a9e6f12f1ffdedb8d"
STARVLA_REVISION = "3d2cd7fce72def48d9020d794392489031ba3f46"
MIKASA_REVISION = "34ece90941291dceaf942938b9f6c92f9dbae01f"
START_SEED = 4242424242
FORMAL_LATENCIES = (0, 1, 2, 3, 4)
FPS = 20
MAX_STEPS = 60

ARTIFACTS = (
    {
        "model": "qwenoft",
        "model_label": "QwenOFT",
        "horizon": 1,
        "bundle_relative": (
            "zero-latency/mikasa-intercept-grab-fast/vla/starvla-qwenoft-h1/"
            "mikasa_intercept_grab_fast_qwenoft_formal_latest_r8gpu8_20260907T082154Z"
        ),
        "source": {
            "type": "benchmark_models",
            "repo_id": MODEL_REPO_ID,
            "revision": MODEL_REVISION,
            "path": (
                "zero-latency/mikasa-intercept-grab-fast/vla/starvla-qwenoft-h1/"
                "mikasa_intercept_grab_fast_qwenoft_formal_latest_r8gpu8_20260907T082154Z"
            ),
        },
    },
    {
        "model": "qwenoft",
        "model_label": "QwenOFT",
        "horizon": 8,
        "bundle_relative": "qwenoft/bundles/best",
        "source": {
            "type": "benchmark_models",
            "repo_id": MODEL_REPO_ID,
            "revision": H8_MODEL_REVISION,
            "path": (
                "zero-latency/mikasa-intercept-grab-fast/vla/starvla-qwenoft-h8/"
                "h8-naive-dagger-20260923"
            ),
            "selected_step": 5000,
        },
    },
    {
        "model": "qwenpi_v3",
        "model_label": "QwenPI v3",
        "horizon": 1,
        "bundle_relative": (
            "zero-latency/mikasa-intercept-grab-fast/vla/starvla-qwenpi_v3-h1/"
            "mikasa_intercept_grab_fast_qwenpi_v3_formal_latest_r8gpu8_20260907T082154Z"
        ),
        "source": {
            "type": "benchmark_models",
            "repo_id": MODEL_REPO_ID,
            "revision": MODEL_REVISION,
            "path": (
                "zero-latency/mikasa-intercept-grab-fast/vla/starvla-qwenpi_v3-h1/"
                "mikasa_intercept_grab_fast_qwenpi_v3_formal_latest_r8gpu8_20260907T082154Z"
            ),
            "inference_config_override": {
                "field": "framework.action_model.diffusion_model_cfg.interleave_self_attention",
                "hub_value": True,
                "eval_value": False,
            },
        },
    },
    {
        "model": "qwenpi_v3",
        "model_label": "QwenPI v3",
        "horizon": 8,
        "bundle_relative": "qwenpi_v3/bundles/best",
        "source": {
            "type": "benchmark_models",
            "repo_id": MODEL_REPO_ID,
            "revision": H8_MODEL_REVISION,
            "path": (
                "zero-latency/mikasa-intercept-grab-fast/vla/starvla-qwenpi_v3-h8/"
                "h8-naive-dagger-20260923"
            ),
            "selected_step": 4000,
        },
    },
    {
        "model": "qwengr00t",
        "model_label": "QwenGR00T",
        "horizon": 1,
        "bundle_relative": (
            "zero-latency/mikasa-intercept-grab-fast/vla/starvla-qwengr00t-h1/"
            "mikasa_intercept_grab_fast_qwengr00t_formal_latest_r8gpu7_20260907T075003Z"
        ),
        "source": {
            "type": "benchmark_models",
            "repo_id": MODEL_REPO_ID,
            "revision": MODEL_REVISION,
            "path": (
                "zero-latency/mikasa-intercept-grab-fast/vla/starvla-qwengr00t-h1/"
                "mikasa_intercept_grab_fast_qwengr00t_formal_latest_r8gpu7_20260907T075003Z"
            ),
        },
    },
    {
        "model": "qwengr00t",
        "model_label": "QwenGR00T",
        "horizon": 8,
        "bundle_relative": "qwengr00t/bundles/best",
        "source": {
            "type": "benchmark_models",
            "repo_id": MODEL_REPO_ID,
            "revision": H8_MODEL_REVISION,
            "path": (
                "zero-latency/mikasa-intercept-grab-fast/vla/starvla-qwengr00t-h8/"
                "h8-naive-dagger-20260923"
            ),
            "selected_step": 3000,
        },
    },
)


def jobs(latencies: tuple[int, ...]) -> list[dict[str, Any]]:
    return [
        {**artifact, "latency_frames": latency_frames}
        for artifact in ARTIFACTS
        for latency_frames in latencies
    ]


def artifact_dir(
    h1_root: Path,
    h8_root: Path,
    artifact: dict[str, Any],
) -> Path:
    root = h1_root if artifact["horizon"] == 1 else h8_root
    return root / artifact["bundle_relative"]


def condition_dir(output_root: Path, job: dict[str, Any]) -> Path:
    return (
        output_root
        / job["model"]
        / f"h{job['horizon']}"
        / f"latency_{job['latency_frames']}"
    )


def inference_code() -> dict[str, Any]:
    return {
        "benchmark_revision": BENCHMARK_REVISION,
        "starvla_revision": STARVLA_REVISION,
        "mikasa_revision": MIKASA_REVISION,
        "working_tree": "local source synchronized before launch",
    }


def prepare(
    *,
    h1_root: Path,
    h8_root: Path,
    backbone_path: Path,
    manifest_path: Path,
) -> Path:
    curves = []
    for artifact in ARTIFACTS:
        directory = artifact_dir(h1_root, h8_root, artifact)
        contract = json.loads((directory / "task_contract.json").read_text(encoding="utf-8"))
        config = yaml.safe_load((directory / "config.yaml").read_text(encoding="utf-8"))
        if contract["action_horizon"] != artifact["horizon"]:
            raise ValueError(f"task contract horizon mismatch: {directory}")
        if config["framework"]["action_model"]["action_horizon"] != artifact["horizon"]:
            raise ValueError(f"model config horizon mismatch: {directory}")
        if artifact["model"] == "qwenpi_v3" and artifact["horizon"] == 1:
            if config["framework"]["action_model"]["diffusion_model_cfg"]["interleave_self_attention"] is not False:
                raise ValueError(f"QwenPI H1 inference config mismatch: {directory}")
        checkpoint = directory / "checkpoints/model.pt"
        curves.append(
            {
                **artifact,
                "bundle": str(directory),
                "checkpoint": str(checkpoint),
                "checkpoint_bytes": checkpoint.stat().st_size,
            }
        )
    manifest = {
        "backbone": {
            "repo_id": BACKBONE_REPO_ID,
            "revision": BACKBONE_REVISION,
            "path": str(backbone_path),
        },
        "inference_code": inference_code(),
        "curves": curves,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest_path


def build_eval_config(
    *,
    h1_root: Path,
    h8_root: Path,
    backbone_path: Path,
    output_dir: Path,
    job: dict[str, Any],
    vla_python: Path,
    episodes: int,
    seed: int,
) -> dict[str, Any]:
    bundle = artifact_dir(h1_root, h8_root, job)
    latency_frames = job["latency_frames"]
    latency = (
        {"method": "zero", "sync_cuda": False}
        if latency_frames == 0
        else {
            "method": "fixed",
            "fixed_latency_ms": latency_frames * 1000.0 / FPS,
            "sync_cuda": False,
        }
    )
    return {
        "experiment": {
            "name": (
                f"mikasa_intercept_grab_fast_{job['model']}_h{job['horizon']}_"
                f"fixed_{latency_frames}f"
            ),
            "seed": seed,
        },
        "executor": {
            "mode": "simulated",
            "simulated_inference_pool": True,
            "simulated_worker_capacity": 1,
            "inference_devices": ["cuda:0"],
            "inference_batch_size": 8,
        },
        "env": {
            "name": "mikasa_intercept_grab_fast",
            "env_fps": FPS,
            "obs_fps": FPS,
            "frame_stack": 1,
            "simulator_device": "cuda:0",
            "noop_action": [0.0] * 7,
            "base_prompt": "Intercept the rolling ball and grasp it to stop it.",
        },
        "latency": latency,
        "scheduler": {
            "hold_policy": "hold",
            "ordering_policy": "latest_ready",
            "hold_last_chunk_action": True,
        },
        "policy": {
            "action_prefix": {"mode": "none"},
            "type": "starvla",
            "checkpoint_path": str(bundle / "checkpoints/model.pt"),
            "model_config_path": str(bundle / "config.yaml"),
            "task_contract_path": str(bundle / "task_contract.json"),
            "backbone_path": str(backbone_path),
            "worker_python_executable": str(vla_python),
            "device": "cuda:0",
            "state_info_key": "mikasa_proprio",
            "image_views_info_key": "mikasa_image_views",
        },
        "evaluation": {
            "eval_episodes": episodes,
            "eval_parallel_envs": 10,
            "eval_max_steps": MAX_STEPS,
            "eval_deterministic": True,
        },
        "logging": {
            "output_dir": str(output_dir),
            "video": {"enabled": False, "num_bins": 10},
            "save_step_records": True,
            "save_action_records": True,
            "save_latency_records": True,
        },
    }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _execution_audit(
    run_dir: Path,
    *,
    horizon: int,
    latency_frames: int,
    episodes: list[dict[str, Any]],
) -> dict[str, Any]:
    actions = _read_jsonl(run_dir / "actions.jsonl")
    steps = _read_jsonl(run_dir / "steps.jsonl")
    if any(action["latency_raw_frames"] != latency_frames for action in actions):
        raise AssertionError(f"fixed latency mismatch: {run_dir}")

    steps_by_episode = {episode["episode_id"]: [] for episode in episodes}
    for step in steps:
        steps_by_episode[step["episode_id"]].append(step["env_step"])
    if any(indices != list(range(MAX_STEPS)) for indices in steps_by_episode.values()):
        raise AssertionError(f"step trace mismatch: {run_dir}")

    first_actions = [action for action in actions if action["obs_env_step"] == 0]
    first_apply_by_episode = {
        action["metadata"]["episode_id"]: action["apply_raw_frame"]
        for action in first_actions
    }
    if len(first_actions) != len(episodes) or set(first_apply_by_episode) != set(steps_by_episode):
        raise AssertionError(f"missing first action: {run_dir}")
    if any(
        action["ready_raw_frame"] != latency_frames
        or action["apply_raw_frame"] != latency_frames
        or not action["applied"]
        or action["dropped"]
        for action in first_actions
    ):
        raise AssertionError(f"first action timing mismatch: {run_dir}")

    initial_noop_transitions = 0
    executed_later_heads = 0
    for step in steps:
        first_apply = first_apply_by_episode[step["episode_id"]]
        info = step["info"]
        if info["in_flight_count"] > 1 or info["observation_submitted"] == info["observation_dropped"]:
            raise AssertionError(f"worker capacity mismatch: {run_dir}")
        if step["env_step"] < first_apply:
            if not step["active_action"]["is_noop"] or info["active_action_obs_env_step"] is not None:
                raise AssertionError(f"startup noop mismatch: {run_dir}")
            initial_noop_transitions += 1
        elif horizon == 8:
            source_step = info["active_action_obs_env_step"]
            age = info["chunk_age_raw_frames"]
            if step["active_action"]["is_noop"] or age != step["env_step"] - source_step:
                raise AssertionError(f"chunk execution mismatch: {run_dir}")
            if age > 0:
                executed_later_heads += 1
    if horizon == 8 and latency_frames > 0 and executed_later_heads == 0:
        raise AssertionError(f"H8 never executed a later chunk head: {run_dir}")
    return {
        "request_fps": FPS,
        "simulated_worker_capacity": 1,
        "startup_policy": "environment_starts_at_reset",
        "latency_raw_frames": latency_frames,
        "fixed_latency_ms": latency_frames * 1000.0 / FPS,
        "initial_noop_transitions": initial_noop_transitions,
        "first_action_apply_raw_frames": sorted(first_apply_by_episode.values()),
        "executed_later_heads": executed_later_heads if horizon == 8 else None,
        "passed": True,
    }


def summarize_condition(
    run_dir: Path,
    *,
    job: dict[str, Any],
    bundle: Path,
    episodes: int,
    seed: int,
    write_summary: bool = True,
) -> dict[str, Any]:
    records = _read_jsonl(run_dir / "episode_metrics.jsonl")
    expected_seeds = list(range(seed, seed + episodes))
    observed_seeds = [record["metadata"]["episode_seed"] for record in records]
    if observed_seeds != expected_seeds:
        raise AssertionError(f"episode seeds mismatch: {run_dir}")
    if [record["episode_id"] for record in records] != list(range(episodes)):
        raise AssertionError(f"episode ids mismatch: {run_dir}")

    returns = [float(record["episode_return_env"]) for record in records]
    lengths = [int(record["survival_steps"]) for record in records]
    successes = [float(record["metadata"]["task_metrics"]["success"]) for record in records]
    if not all(math.isfinite(value) for value in returns):
        raise AssertionError(f"non-finite return: {run_dir}")
    if lengths != [MAX_STEPS] * episodes:
        raise AssertionError(f"episode length mismatch: {run_dir}")

    success_count = int(sum(successes))
    summary = {
        "model": job["model"],
        "model_label": job["model_label"],
        "horizon": job["horizon"],
        "latency_frames": job["latency_frames"],
        "latency_ms": job["latency_frames"] * 1000.0 / FPS,
        "n_episodes": episodes,
        "seed_start": expected_seeds[0],
        "seed_end": expected_seeds[-1],
        "seeds_unique": len(set(observed_seeds)) == episodes,
        "success_count": success_count,
        "success_rate": success_count / episodes,
        "mean_return": statistics.fmean(returns),
        "std_return": statistics.pstdev(returns),
        "min_return": min(returns),
        "max_return": max(returns),
        "mean_length": statistics.fmean(lengths),
        "std_length": statistics.pstdev(lengths),
        "min_length": min(lengths),
        "max_length": max(lengths),
        "artifact": {**job["source"], "bundle_path": str(bundle)},
        "inference_code": inference_code(),
        "evidence_path": str(run_dir),
        "execution_audit": _execution_audit(
            run_dir,
            horizon=job["horizon"],
            latency_frames=job["latency_frames"],
            episodes=records,
        ),
    }
    if write_summary:
        (run_dir / "summary.json").write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )
    return summary


def _run_condition(
    *,
    h1_root: Path,
    h8_root: Path,
    backbone_path: Path,
    output_root: Path,
    job: dict[str, Any],
    gpu: str,
    task_python: Path,
    vla_python: Path,
    runtime_root: Path,
    episodes: int,
    seed: int,
) -> None:
    bundle = artifact_dir(h1_root, h8_root, job)
    output_dir = condition_dir(output_root, job)
    complete_path = output_dir / "complete.json"
    if complete_path.is_file():
        complete = json.loads(complete_path.read_text(encoding="utf-8"))
        if complete["episodes"] != episodes or complete["seed"] != seed:
            raise ValueError(f"completion protocol mismatch: {output_dir}")
        return
    if output_dir.exists():
        raise FileExistsError(f"incomplete condition directory already exists: {output_dir}")
    output_dir.mkdir(parents=True)
    config = build_eval_config(
        h1_root=h1_root,
        h8_root=h8_root,
        backbone_path=backbone_path,
        output_dir=output_dir,
        job=job,
        vla_python=vla_python,
        episodes=episodes,
        seed=seed,
    )
    config_path = output_dir / "requested_eval_config.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    command = [
        str(task_python),
        str(runtime_root / "scripts/mikasa/evaluate.py"),
        "latency-eval",
        "--eval-config",
        str(config_path),
    ]
    env = os.environ | {
        "CUDA_VISIBLE_DEVICES": gpu,
        "PYTHONPATH": os.pathsep.join(
            [str(runtime_root), str(runtime_root / "third_party/MIKASA-Robo")]
        ),
        "WANDB_MODE": "disabled",
        "TOKENIZERS_PARALLELISM": "false",
    }
    with (output_dir / "stdout.log").open("w", encoding="utf-8") as log:
        subprocess.run(
            command,
            cwd=runtime_root,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )
    summarize_condition(
        output_dir,
        job=job,
        bundle=bundle,
        episodes=episodes,
        seed=seed,
    )
    complete_path.write_text(
        json.dumps(
            {
                "model": job["model"],
                "horizon": job["horizon"],
                "latency_frames": job["latency_frames"],
                "episodes": episodes,
                "seed": seed,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def run_queue(
    *,
    h1_root: Path,
    h8_root: Path,
    backbone_path: Path,
    output_root: Path,
    task_python: Path,
    vla_python: Path,
    runtime_root: Path,
    episodes: int,
    seed: int,
    latencies: tuple[int, ...],
    gpus: tuple[str, ...],
) -> None:
    pending = deque(jobs(latencies))
    lock = threading.Lock()
    failures: list[dict[str, Any]] = []

    def worker(gpu: str) -> None:
        while True:
            with lock:
                if not pending:
                    return
                job = pending.popleft()
            print(
                f"[figure2-mikasa] gpu={gpu} model={job['model']} "
                f"h={job['horizon']} latency={job['latency_frames']}",
                flush=True,
            )
            try:
                _run_condition(
                    h1_root=h1_root,
                    h8_root=h8_root,
                    backbone_path=backbone_path,
                    output_root=output_root,
                    job=job,
                    gpu=gpu,
                    task_python=task_python,
                    vla_python=vla_python,
                    runtime_root=runtime_root,
                    episodes=episodes,
                    seed=seed,
                )
            except (subprocess.CalledProcessError, AssertionError, FileExistsError) as error:
                with lock:
                    failures.append({**job, "error": str(error)})

    with ThreadPoolExecutor(max_workers=len(gpus)) as executor:
        futures = [executor.submit(worker, gpu) for gpu in gpus]
        for future in futures:
            future.result()
    if failures:
        output_root.mkdir(parents=True, exist_ok=True)
        (output_root / "failures.json").write_text(
            json.dumps(failures, indent=2) + "\n", encoding="utf-8"
        )
        raise RuntimeError(f"{len(failures)} MIKASA conditions failed")


def aggregate(
    *,
    output_root: Path,
    revision: str,
    episodes: int,
    seed: int,
    latencies: tuple[int, ...],
) -> dict[str, Any]:
    if len(set(latencies)) != len(latencies):
        raise AssertionError("duplicate latency condition")
    curves: dict[str, dict[str, Any]] = {}
    for job in jobs(latencies):
        run_dir = condition_dir(output_root, job)
        complete = json.loads((run_dir / "complete.json").read_text(encoding="utf-8"))
        if complete != {
            "model": job["model"],
            "horizon": job["horizon"],
            "latency_frames": job["latency_frames"],
            "episodes": episodes,
            "seed": seed,
        }:
            raise AssertionError(f"completion protocol mismatch: {run_dir}")
        item = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
        recomputed = summarize_condition(
            run_dir,
            job=job,
            bundle=Path(item["artifact"]["bundle_path"]),
            episodes=episodes,
            seed=seed,
            write_summary=False,
        )
        if item != recomputed:
            raise AssertionError(f"summary does not match raw records: {run_dir}")
        key = f"{job['model']}_h{job['horizon']}"
        if key not in curves:
            curves[key] = {
                "model": job["model"],
                "model_label": job["model_label"],
                "horizon": job["horizon"],
                "artifact": item["artifact"],
                "latencies": {},
            }
        curves[key]["latencies"][str(job["latency_frames"])] = {
            name: item[name]
            for name in (
                "latency_frames",
                "latency_ms",
                "n_episodes",
                "seed_start",
                "seed_end",
                "seeds_unique",
                "success_count",
                "success_rate",
                "mean_return",
                "std_return",
                "min_return",
                "max_return",
                "mean_length",
                "std_length",
                "min_length",
                "max_length",
                "evidence_path",
                "inference_code",
                "execution_audit",
            )
        }
    result = {
        "protocol": {
            "task": "InterceptGrabFast-VLA-v0",
            "env_fps": FPS,
            "latencies_frames": list(latencies),
            "latencies_ms": [latency * 1000.0 / FPS for latency in latencies],
            "episodes_per_condition": episodes,
            "seed_start": seed,
            "seed_end": seed + episodes - 1,
            "eval_parallel_envs": 10,
            "eval_max_steps": MAX_STEPS,
            "simulated_worker_capacity": 1,
            "inference_batch_size": 8,
            "executor": "simulated",
        },
        "model_source": {
            "repo_id": MODEL_REPO_ID,
            "h1_revision": revision,
            "h8_revision": H8_MODEL_REVISION,
        },
        "curves": curves,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "combined_results.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    (output_root / "RESULTS.md").write_text(_results_markdown(result), encoding="utf-8")
    return result


def _results_markdown(result: dict[str, Any]) -> str:
    protocol = result["protocol"]
    curve_keys = [f"{artifact['model']}_h{artifact['horizon']}" for artifact in ARTIFACTS]
    latency_headers = [f"L{latency}" for latency in protocol["latencies_frames"]]
    lines = [
        "# MIKASA H1 / H8 DAgger fixed-latency sweep",
        "",
        (
            f"Each cell uses {protocol['episodes_per_condition']} episodes with seeds "
            f"`{protocol['seed_start']}–{protocol['seed_end']}`. "
            f"L{protocol['latencies_frames'][0]}–L{protocol['latencies_frames'][-1]} "
            f"correspond to {protocol['latencies_ms'][0]:g}–"
            f"{protocol['latencies_ms'][-1]:g} ms at 20 Hz."
        ),
        (
            "QwenPI v3 H1 uses the isolated inference config with "
            "`interleave_self_attention: false`; its Hub checkpoint is unchanged."
        ),
        "H8 uses zero-latency naive DAgger best checkpoints: OFT step 5000, PI v3 step 4000, GR00T step 3000.",
        "",
        "## Success",
        "",
        "| Model / horizon | " + " | ".join(latency_headers) + " |",
        "|---|" + "---:|" * len(latency_headers),
    ]
    for key in curve_keys:
        curve = result["curves"][key]
        cells = []
        for latency in protocol["latencies_frames"]:
            item = curve["latencies"][str(latency)]
            cells.append(
                f"{item['success_count']}/{item['n_episodes']} ({item['success_rate']:.1%})"
            )
        lines.append(
            f"| {curve['model_label']} H{curve['horizon']} | " + " | ".join(cells) + " |"
        )
    lines.extend(
        [
            "",
            "## Return mean ± population SD",
            "",
            "| Model / horizon | " + " | ".join(latency_headers) + " |",
            "|---|" + "---:|" * len(latency_headers),
        ]
    )
    for key in curve_keys:
        curve = result["curves"][key]
        cells = [
            (
                f"{curve['latencies'][str(latency)]['mean_return']:.5f} ± "
                f"{curve['latencies'][str(latency)]['std_return']:.5f}"
            )
            for latency in protocol["latencies_frames"]
        ]
        lines.append(
            f"| {curve['model_label']} H{curve['horizon']} | " + " | ".join(cells) + " |"
        )
    lines.extend(
        [
            "",
            "All episode lengths are 60 transitions. Every condition passed its fixed-latency "
            "execution audit; H8 conditions additionally passed the observation-aligned chunk audit.",
            "",
        ]
    )
    return "\n".join(lines)


def _path(value: str) -> Path:
    return Path(value).expanduser().absolute()


def _add_bundle_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--h1-root", type=_path, required=True)
    parser.add_argument("--h8-root", type=_path, required=True)
    parser.add_argument("--backbone-path", type=_path, required=True)


def _add_run_arguments(parser: argparse.ArgumentParser) -> None:
    _add_bundle_arguments(parser)
    parser.add_argument("--output-root", type=_path, required=True)
    parser.add_argument("--task-python", type=_path, required=True)
    parser.add_argument("--vla-python", type=_path, required=True)
    parser.add_argument("--runtime-root", type=_path, required=True)
    parser.add_argument("--gpus", nargs="+", required=True)
    parser.add_argument("--episodes", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--latencies", type=int, nargs="+", required=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare_parser = subparsers.add_parser("prepare")
    _add_bundle_arguments(prepare_parser)
    prepare_parser.add_argument("--manifest-path", type=_path, required=True)
    _add_run_arguments(subparsers.add_parser("run"))
    aggregate_parser = subparsers.add_parser("aggregate")
    aggregate_parser.add_argument("--output-root", type=_path, required=True)
    aggregate_parser.add_argument("--revision", default=MODEL_REVISION)
    aggregate_parser.add_argument("--episodes", type=int, required=True)
    aggregate_parser.add_argument("--seed", type=int, required=True)
    aggregate_parser.add_argument("--latencies", type=int, nargs="+", required=True)
    args = parser.parse_args()

    if args.command == "prepare":
        print(
            prepare(
                h1_root=args.h1_root,
                h8_root=args.h8_root,
                backbone_path=args.backbone_path,
                manifest_path=args.manifest_path,
            )
        )
        return
    latencies = tuple(args.latencies)
    if args.command == "run":
        run_queue(
            h1_root=args.h1_root,
            h8_root=args.h8_root,
            backbone_path=args.backbone_path,
            output_root=args.output_root,
            task_python=args.task_python,
            vla_python=args.vla_python,
            runtime_root=args.runtime_root,
            episodes=args.episodes,
            seed=args.seed,
            latencies=latencies,
            gpus=tuple(args.gpus),
        )
        return
    aggregate(
        output_root=args.output_root,
        revision=args.revision,
        episodes=args.episodes,
        seed=args.seed,
        latencies=latencies,
    )


if __name__ == "__main__":
    main()
