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
from latency_bench.logging.wandb_eval import eval_scalar_payload
from latency_bench.eval.sweeps import run_eval_suites_from_config
from training.common.checkpoints import (
    CheckpointEvalLease,
    acquire_checkpoint_eval_lease,
    checkpoint_dir,
    checkpoint_mtimes,
    checkpoint_paths,
    mark_checkpoint_for_eval_retention,
    patch_sample_factory_checkpoint_loader,
    release_checkpoint_eval_lease,
    update_checkpoint_eval_lease,
)


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
    "gymnasium": "gymnasium_sf",
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
    base_config = (
        deepcopy(training_config)
        if env_name == "gymnasium"
        else load_config(BASE_CONFIGS[env_name])
    )
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
    config["policy"] = merge_dicts(
        base_config.get("policy", {}),
        {
            "type": POLICY_TYPES[env_name],
            "checkpoint_path": str(Path(checkpoint_path).expanduser()),
            "deterministic": bool(config["evaluation"].get("eval_deterministic", True)),
        },
    )
    config["logging"]["output_dir"] = str(Path(output_dir).expanduser())
    # Eval after training does not need visualization
    config["logging"]["video"]["enabled"] = False
    config["logging"]["save_step_records"] = False
    config["logging"]["save_action_records"] = False
    config["logging"]["save_latency_records"] = False
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
        if status == ExperimentStatus.SUCCESS:
            if not observer.run_final_eval(runner):
                status = ExperimentStatus.FAILURE
    return status


@dataclass
class LatencyBenchEvalRequest:
    config: dict[str, Any]
    metadata: dict[str, Any]
    result_path: Path
    checkpoint_path: Path
    env_steps: int
    phase: str


@dataclass
class LatencyBenchEvalJob:
    process: Any
    result_path: Path
    lease: CheckpointEvalLease
    env_steps: int
    phase: str

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
        del runner, policy_id, env_steps, writer

    def on_training_step(self, runner, training_iteration_since_resume: int) -> None:
        del training_iteration_since_resume
        env_steps = _runner_env_steps(runner)
        if env_steps >= self.next_eval_step:
            self._submit_eval_request(runner, env_steps=env_steps, phase="train_periodic")
            while self.next_eval_step <= env_steps:
                self.next_eval_step += self.interval_steps

    def on_stop(self, runner) -> None:
        pass

    def run_deferred_eval(self, runner) -> bool:
        while self.pending_jobs:
            request = self.pending_jobs.popleft()
            if not self._run_eval_request(runner, request):
                self.pending_jobs.clear()
                return False
        return True

    def run_final_eval(self, runner) -> bool:
        if not self.run_deferred_eval(runner):
            return False
        env_steps = _runner_env_steps(runner)
        checkpoint_path = self._latest_checkpoint_for_eval()
        self._submit_eval_request(
            runner,
            env_steps=env_steps,
            phase="train_final",
            checkpoint_path=checkpoint_path,
        )
        return self.run_deferred_eval(runner)

    def _submit_eval_request(
        self,
        runner,
        *,
        env_steps: int,
        phase: str,
        checkpoint_path: Path | None = None,
    ) -> LatencyBenchEvalRequest:
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
        mark_checkpoint_for_eval_retention(checkpoint_path, metadata=metadata)
        request = LatencyBenchEvalRequest(
            config=config,
            metadata=metadata,
            result_path=result_path,
            checkpoint_path=checkpoint_path,
            env_steps=int(env_steps),
            phase=phase,
        )
        self.pending_jobs.append(request)
        return request

    def _run_eval_request(self, runner, request: LatencyBenchEvalRequest) -> bool:
        lease = acquire_checkpoint_eval_lease(request.checkpoint_path, metadata=request.metadata)
        process = _launch_latency_bench_eval_process(request.config, request.metadata, request.result_path)
        update_checkpoint_eval_lease(lease, {"eval_pid": getattr(process, "pid", None)})
        job = LatencyBenchEvalJob(
            process=process,
            result_path=request.result_path,
            lease=lease,
            env_steps=request.env_steps,
            phase=request.phase,
        )
        self.active_job = job
        job.wait()
        return self._complete_eval_job(runner, job)

    def _complete_eval_job(self, runner, job: LatencyBenchEvalJob) -> bool:
        if job.process.exitcode != 0:
            return self._fail_eval_job(runner, job, f"process exitcode={job.process.exitcode}")
        try:
            summaries = job.read_summaries()
        except FileNotFoundError as exc:
            return self._fail_eval_job(runner, job, str(exc))
        release_checkpoint_eval_lease(job.lease)
        _log_eval_summaries(
            runner,
            summaries,
            env_steps=job.env_steps,
        )
        self.last_eval_step = job.env_steps
        self.active_job = None
        return True

    def _fail_eval_job(self, runner, job: LatencyBenchEvalJob, reason: str) -> bool:
        del runner
        from sample_factory.utils.utils import log

        log.error(
            "LatencyBench eval failed at phase=%s env_steps=%s result_path=%s: %s",
            job.phase,
            job.env_steps,
            job.result_path,
            reason,
        )
        release_checkpoint_eval_lease(job.lease)
        self.active_job = None
        return False

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
    return checkpoint_paths(root)


def _checkpoint_mtimes(root: Path) -> dict[Path, int]:
    return checkpoint_mtimes(root)


def _log_eval_summaries(
    runner,
    summaries: list[dict[str, Any]],
    *,
    env_steps: int,
) -> None:
    del runner
    _log_wandb(eval_scalar_payload(summaries, env_steps=env_steps, prefix="eval_corrected"))


def _log_wandb(payload: dict[str, float | int]) -> None:
    try:
        import wandb
    except ImportError:
        return
    if wandb.run is not None:
        wandb.log(payload)
