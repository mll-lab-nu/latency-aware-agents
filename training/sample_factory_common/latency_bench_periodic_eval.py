from __future__ import annotations

import json
import multiprocessing as mp
import time
from collections import deque
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from latency_bench.core.config import load_config, merge_dicts
from latency_bench.run import run_eval_suites_from_config
from training.common.checkpoints import checkpoint_dir
from training.sample_factory_common.checkpoint_io import patch_sample_factory_checkpoint_loader


REPO_ROOT = Path(__file__).resolve().parents[2]
BASE_CONFIGS = {
    "flappy": REPO_ROOT / "configs" / "base" / "flappy.yaml",
    "demon_attack": REPO_ROOT / "configs" / "base" / "demon_attack.yaml",
    "deadly_corridor": REPO_ROOT / "configs" / "base" / "deadly_corridor.yaml",
}
POLICY_TYPES = {
    "flappy": "flappy_sf",
    "demon_attack": "demon_attack_sf",
    "deadly_corridor": "deadly_corridor_sf",
}


def periodic_eval_enabled(config: dict[str, Any]) -> bool:
    evaluation_cfg = config.get("evaluation", {})
    return isinstance(evaluation_cfg, dict) and evaluation_cfg.get("eval_interval_steps") is not None


def build_latency_bench_config_from_training_config(
    training_config: dict[str, Any],
    *,
    checkpoint_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    env_name = str(training_config["env"]["name"]).strip().lower()
    base_config = load_config(BASE_CONFIGS[env_name])
    override = {
        key: deepcopy(training_config[key])
        for key in (
            "experiment",
            "backend",
            "env",
            "latency",
            "scheduler",
            "training",
            "evaluation",
            "logging",
        )
        if key in training_config
    }
    config = merge_dicts(base_config, override)
    config["backend"]["run_mode"] = "eval"
    config["experiment"]["num_episodes"] = int(config["evaluation"]["eval_episodes"])
    config["policy"] = merge_dicts(
        base_config.get("policy", {}),
        {
            "type": POLICY_TYPES[env_name],
            "checkpoint_path": str(Path(checkpoint_path).expanduser()),
            "deterministic": bool(config["evaluation"].get("eval_deterministic", True)),
        },
    )
    config["logging"]["output_dir"] = str(Path(output_dir).expanduser())
    return config


def run_sample_factory_training(cfg, training_config: dict[str, Any]):
    from sample_factory.algo.utils.misc import ExperimentStatus
    from sample_factory.train import make_runner, run_rl

    patch_sample_factory_checkpoint_loader()
    if not periodic_eval_enabled(training_config):
        return run_rl(cfg)

    cfg, runner = make_runner(cfg)
    observer = LatencyBenchPeriodicEvalObserver(training_config)
    runner.register_observer(observer)
    status = runner.init()
    if status == ExperimentStatus.SUCCESS:
        status = runner.run()
        observer.run_final_eval(runner)
    return status


EVAL_SCALAR_METRICS = ("mean_return", "std_return", "mean_length")


@dataclass
class LatencyBenchEvalRequest:
    config: dict[str, Any]
    metadata: dict[str, Any]
    result_path: Path
    env_steps: int
    phase: str


@dataclass
class LatencyBenchEvalJob:
    process: Any
    result_path: Path
    env_steps: int
    phase: str

    def is_done(self) -> bool:
        return self.process.exitcode is not None

    def wait(self) -> None:
        self.process.join()

    def read_summaries(self) -> list[dict[str, Any]]:
        result = json.loads(self.result_path.read_text(encoding="utf-8"))
        return list(result["summaries"])


class LatencyBenchPeriodicEvalObserver:
    def __init__(self, training_config: dict[str, Any], *, checkpoint_timeout_sec: float = 60.0):
        self.training_config = deepcopy(training_config)
        self.interval_steps = int(training_config["evaluation"]["eval_interval_steps"])
        self.next_eval_step = self.interval_steps
        self.checkpoint_timeout_sec = float(checkpoint_timeout_sec)
        self.last_eval_step: int | None = None
        self.active_job: LatencyBenchEvalJob | None = None
        self.pending_jobs: deque[LatencyBenchEvalRequest] = deque()

    def on_init(self, runner) -> None:
        pass

    def on_connect_components(self, runner) -> None:
        pass

    def on_start(self, runner) -> None:
        pass

    def extra_summaries(self, runner, policy_id, env_steps, writer) -> None:
        del policy_id, env_steps, writer
        self._poll_eval_job(runner)

    def on_training_step(self, runner, training_iteration_since_resume: int) -> None:
        del training_iteration_since_resume
        self._poll_eval_job(runner)
        env_steps = _runner_env_steps(runner)
        if env_steps >= self.next_eval_step:
            self._submit_eval_job(runner, env_steps=env_steps, phase="train_periodic")
            while self.next_eval_step <= env_steps:
                self.next_eval_step += self.interval_steps

    def on_stop(self, runner) -> None:
        pass

    def run_final_eval(self, runner) -> None:
        self._drain_eval_jobs(runner)
        env_steps = _runner_env_steps(runner)
        checkpoint_path = self._latest_checkpoint_for_eval()
        job = self._submit_eval_job(
            runner,
            env_steps=env_steps,
            phase="train_final",
            checkpoint_path=checkpoint_path,
        )
        job.wait()
        self._complete_eval_job(runner, job)

    def _submit_eval_job(
        self,
        runner,
        *,
        env_steps: int,
        phase: str,
        checkpoint_path: Path | None = None,
    ) -> LatencyBenchEvalJob | None:
        if checkpoint_path is None:
            checkpoint_path = self._save_checkpoint_for_eval(runner)
        step_dir = f"step_{int(env_steps)}"
        if phase == "train_final":
            step_dir = f"{step_dir}_final"
        output_dir = (
            Path(str(self.training_config["logging"]["output_dir"])).expanduser()
            / "latency_bench_train_eval"
            / step_dir
        )
        config = build_latency_bench_config_from_training_config(
            self.training_config,
            checkpoint_path=checkpoint_path,
            output_dir=output_dir,
        )
        metadata = {
            "eval_phase": phase,
            "train_env_steps": int(env_steps),
            "checkpoint_path": str(checkpoint_path),
        }
        result_path = output_dir / "eval_result.json"
        self.pending_jobs.append(
            LatencyBenchEvalRequest(
                config=config,
                metadata=metadata,
                result_path=result_path,
                env_steps=int(env_steps),
                phase=phase,
            )
        )
        return self._start_next_eval_job()

    def _start_next_eval_job(self) -> LatencyBenchEvalJob | None:
        if self.active_job is not None or not self.pending_jobs:
            return self.active_job
        request = self.pending_jobs.popleft()
        process = _launch_latency_bench_eval_process(request.config, request.metadata, request.result_path)
        self.active_job = LatencyBenchEvalJob(
            process=process,
            result_path=request.result_path,
            env_steps=request.env_steps,
            phase=request.phase,
        )
        return self.active_job

    def _poll_eval_job(self, runner) -> None:
        job = self.active_job
        if job is not None and job.is_done():
            self._complete_eval_job(runner, job)

    def _drain_eval_jobs(self, runner) -> None:
        while self.active_job is not None or self.pending_jobs:
            job = self.active_job
            if job is None:
                job = self._start_next_eval_job()
            job.wait()
            self._complete_eval_job(runner, job)

    def _complete_eval_job(self, runner, job: LatencyBenchEvalJob) -> None:
        summaries = job.read_summaries()
        _log_eval_summaries(
            runner,
            summaries,
            env_steps=job.env_steps,
        )
        self.last_eval_step = job.env_steps
        self.active_job = None
        self._start_next_eval_job()

    def _latest_checkpoint_for_eval(self) -> Path:
        root = checkpoint_dir(
            self.training_config["backend"]["train_dir"],
            self.training_config["experiment"]["name"],
            policy_id=0,
        )
        return _checkpoint_paths(root)[-1]

    def _save_checkpoint_for_eval(self, runner) -> Path:
        root = checkpoint_dir(
            self.training_config["backend"]["train_dir"],
            self.training_config["experiment"]["name"],
            policy_id=0,
        )
        before = _checkpoint_mtimes(root)
        runner.save_periodic.emit()
        deadline = time.monotonic() + self.checkpoint_timeout_sec
        while time.monotonic() <= deadline:
            current = _checkpoint_paths(root)
            updated_paths = [
                path
                for path in current
                if path not in before or path.stat().st_mtime_ns != before[path]
            ]
            if updated_paths:
                return updated_paths[-1]
            time.sleep(0.05)
        raise RuntimeError(f"No new checkpoint appeared under {root}")


def _latency_bench_eval_process_main(
    config: dict[str, Any],
    metadata: dict[str, Any],
    result_path: str,
) -> None:
    path = Path(result_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    summaries = run_eval_suites_from_config(config, progress=False, extra_metadata=metadata)
    path.write_text(json.dumps({"summaries": summaries}, indent=2), encoding="utf-8")


def _launch_latency_bench_eval_process(
    config: dict[str, Any],
    metadata: dict[str, Any],
    result_path: str | Path,
) -> Any:
    path = Path(result_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ctx = mp.get_context("spawn")
    process = ctx.Process(
        target=_latency_bench_eval_process_main,
        args=(config, metadata, str(path)),
    )
    process.start()
    return process


def _runner_env_steps(runner) -> int:
    return max(int(value) for value in runner.env_steps.values())


def _checkpoint_paths(root: Path) -> list[Path]:
    return sorted(root.glob("checkpoint_*.pth"))


def _checkpoint_mtimes(root: Path) -> dict[Path, int]:
    return {path: path.stat().st_mtime_ns for path in _checkpoint_paths(root)}


def _log_eval_summaries(
    runner,
    summaries: list[dict[str, Any]],
    *,
    env_steps: int,
) -> None:
    del runner
    _log_wandb(_eval_scalar_payload(summaries, env_steps=env_steps))


def _summary_metric_group(summary: dict[str, Any]) -> str:
    latency_type = str(summary.get("latency_type", ""))
    if latency_type == "fixed_sweep":
        return "fixed"
    if latency_type == "normal_distribution":
        return "normal"
    if latency_type == "uniform_distribution":
        return "uniform"
    suite_name = str(summary["suite_name"])
    if suite_name.startswith("fixed_"):
        return "fixed"
    if suite_name.startswith("normal_"):
        return "normal"
    if suite_name.startswith("uniform_"):
        return "uniform"
    return suite_name


def _summary_series_name(summary: dict[str, Any]) -> str:
    return _latency_fragment(summary["latency"])


def _eval_scalar_payload(summaries: list[dict[str, Any]], *, env_steps: int) -> dict[str, float | int]:
    payload: dict[str, float | int] = {
        "global_step": int(env_steps),
        "eval/global_steps": int(env_steps),
    }
    for summary in summaries:
        prefix = f"eval/{_summary_metric_group(summary)}/{_summary_series_name(summary)}"
        for metric in EVAL_SCALAR_METRICS:
            payload[f"{prefix}/{metric}"] = float(summary[metric])
        payload[f"{prefix}/completed_episodes"] = float(
            summary.get("completed_episodes", len(summary.get("returns", [])))
        )
    return payload


def _latency_fragment(latency: int | str) -> str:
    if isinstance(latency, int) and not isinstance(latency, bool):
        return f"latency_{int(latency)}"
    return "".join(
        char if char.isalnum() or char in {"-", "_", "."} else "_"
        for char in str(latency)
    ).strip("._-")


def _log_wandb(payload: dict[str, float | int]) -> None:
    try:
        import wandb
    except ImportError:
        return
    if wandb.run is not None:
        wandb.log(payload)
