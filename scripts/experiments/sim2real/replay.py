"""Replay logical experiment sessions independently of resident executor slots."""
from collections import defaultdict
from copy import deepcopy
from datetime import datetime, timezone
import shutil
import time

from latency_bench.core.actions import ActionResolver
from latency_bench.core.latency_distribution import derive_seed
from latency_bench.latency.samplers import build_episode_sampler_factory, build_sampler
from latency_bench.logging.trajectory_logger import TrajectoryLogger
from latency_bench.executors.backend_selection import build_env_step_backend, resolve_latency_bench_env_backend
from latency_bench.executors.factory import build_executor
from latency_bench.utils.io import read_json, write_json, write_yaml
from scripts.experiments.sim2real import runtime


def grouped_inputs(latency_by_run, clock, *, protocol, replica_seed):
    """Build logical episodes and streams independently of slot scheduling."""
    env_seed, env_seed_stride = protocol["seed"], protocol["seed_stride"]
    groups, seeds, sources = [], [], {}
    for run_id, run_latency in latency_by_run.items():
        group = [env_seed_stride * run_id + episode for episode in range(protocol["episodes"])]
        groups.append(group)
        seeds.extend(env_seed + episode_id for episode_id in group)
        latency = run_latency.copy()
        factory = build_episode_sampler_factory(latency, clock)
        if factory is None:
            if replica_seed:
                latency["seed"] = derive_seed(latency["seed"], vector_index=0)
            sampler = build_sampler(latency, clock=clock)
            sources.update((episode_id, sampler) for episode_id in group)
        else:
            sources.update((episode_id, factory(local_id)) for local_id, episode_id in enumerate(group))
    return groups, seeds, sources


class RunTrajectoryLogger(TrajectoryLogger):
    """Persist complete logical runs and their traces before a slot is reused."""

    def __init__(self, output, *, run_paths, protocol,
                 episode_metadata=None):
        super().__init__(output / "stream", save_step_records=True,
                         save_action_records=True, save_latency_records=True)
        self.run_paths = run_paths
        self.env_seed = protocol["seed"]
        self.env_seed_stride = protocol["seed_stride"]
        self.episodes_per_run = protocol["episodes"]
        self.episode_metadata = episode_metadata
        self._run_metrics = defaultdict(list)

    def log_episode_metrics(self, metrics):
        run_key, local_id = divmod(metrics.episode_id, self.env_seed_stride)
        if self.episode_metadata is None:
            metrics.metadata.update(run_id=run_key, episode_in_run=local_id,
                                    episode_seed=self.env_seed + metrics.episode_id)
        else:
            identity = self.episode_metadata[metrics.episode_id]
            metrics.metadata.update(identity)
            metrics.episode_id = self.env_seed_stride * identity["run_id"] + identity["episode_in_run"]
            for record in self.step_records:
                record.episode_id = metrics.episode_id
            for event in self.action_records:
                event.metadata["episode_id"] = metrics.episode_id
                event.latency.metadata["episode_id"] = metrics.episode_id
            for record in self.latency_records:
                record.metadata["episode_id"] = metrics.episode_id
        self.output_dir = self.run_paths[run_key]
        self.episode_metrics = self._run_metrics[run_key]
        super().log_episode_metrics(metrics)
        if len(self.episode_metrics) == self.episodes_per_run:
            super().flush()
            (self.output_dir / "DONE").touch()
            del self._run_metrics[run_key]
        self.episode_metrics = []

    def flush(self):
        # Complete runs already flushed their metrics; do not overwrite them
        # with the now-empty in-memory list at executor shutdown.
        self._flush_episode_records()


def _run_grouped_episodes(executor, groups, seeds, sources, max_env_steps):
    """Run and flush prebuilt logical episode groups through a resident executor."""
    result = executor.run_episodes(
        episode_ids=[episode_id for group in groups for episode_id in group],
        seeds=seeds,
        episode_groups=groups,
        latency_sources_by_episode=sources,
        max_env_steps=max_env_steps,
    )
    executor.logger.flush()
    return result


def build_replay_executor(config, parallel_runs):
    """Use the task's existing environment backend, including for a one-slot check."""
    config = deepcopy(config)
    config['evaluation'].update(episodes=config['evaluation']['episodes'] * parallel_runs)
    config["executor"]["parallel_envs"] = parallel_runs
    config["executor"]["inference_batch_size"] = parallel_runs
    runtime.prepare_replay_runtime(config)
    if config["env"]["name"] == "mikasa_intercept_grab_fast":
        # The simulator interpreter owns this optional ManiSkill dependency.
        from latency_bench.envs.adapters.mikasa import MikasaEnvStepBackend
        backend = MikasaEnvStepBackend(config=config, num_slots=parallel_runs)
    else:
        backend = build_env_step_backend(
            resolve_latency_bench_env_backend(config, parallel_envs=parallel_runs),
            config=config, noop_action=ActionResolver(config["env"]).noop_action, num_slots=parallel_runs,
        )
    executor = build_executor(config, env_backend=backend,
                              inference_devices=config["executor"]["inference_devices"])
    return executor, config


def replay_runs(executor, configs, run_paths, runtime_dir, *, protocol, replica_seed):
    """Persist each missing run in its original run directory."""
    for run_id, path in run_paths.items():
        if path.exists():
            shutil.rmtree(path)  # The caller supplies only incomplete runs.
        write_yaml(path / "generated.yaml", configs[run_id])
    executor.logger = RunTrajectoryLogger(
        runtime_dir, run_paths=run_paths, protocol=protocol,
    )
    groups, seeds, sources = grouped_inputs(
        {run_id: config["latency"] for run_id, config in configs.items()}, executor.clock,
        protocol=protocol, replica_seed=replica_seed,
    )
    return _run_grouped_episodes(
        executor, groups, seeds, sources,
        next(iter(configs.values()))["evaluation"]["max_env_steps"],
    )


def simulate(base, output, manifest, *, inference_device, simulator_device, parallel_runs):
    """Resume only missing runs; retain original profile, clocks, seeds and result paths."""
    protocol = manifest["protocol"]
    fps = read_json(output / "calibration/calibration.json")["selected_env_fps"]
    profile = output / "profile/profile.json"
    base = deepcopy(base)
    base["policy"]["action_noise_seeded"] = manifest["model"] in {"qwenpi_v3", "qwengr00t"}
    replica_seed = manifest["task"] in {"flappy", "mikasa_intercept_grab_fast"}
    runtime_dir = output / "parallel_runtime"
    configs = {
        method: {
            run_id: runtime.run_config(
                base, output / "sim" / method / str(index), inference_device=inference_device,
                simulator_device=simulator_device, env_fps=fps, run_id=run_id,
                method=method, profile_path=profile, latency_seed=index,
                protocol=protocol,
            ) for index, run_id in enumerate(protocol["valid"])
        } for method in protocol["methods"]
    }
    executor, resolved = build_replay_executor(
        runtime.run_config(base, runtime_dir, inference_device=inference_device, simulator_device=simulator_device,
                         env_fps=fps, run_id=protocol["valid"][0], method="fixed", profile_path=profile,
                         latency_seed=0, protocol=protocol),
        parallel_runs,
    )
    with runtime.warmed_replay_executor(executor):
        write_json(runtime_dir / "config.json", resolved)
        context = output / "simulation_parallel.json"
        if not context.exists():
            write_json(context, {
                "parallel_runs": parallel_runs, "inference_batch_size": parallel_runs,
                "action_noise": "episode_frame" if base["policy"]["action_noise_seeded"] else "deterministic",
                "retained_runs": [str(path.parent.relative_to(output)) for path in sorted(output.glob("sim/*/*/DONE"))],
                "episode_id": f"{protocol['seed_stride']} * valid_run_id + episode_in_run",
                "temporal_replica_zero_seed": replica_seed,
            })
        for method in protocol["methods"]:
            paths = {run_id: output / "sim" / method / str(index)
                     for index, run_id in enumerate(protocol["valid"])}
            pending = {}
            for run_id, path in paths.items():
                if not (path / "DONE").exists():
                    pending[run_id] = path
            if pending:
                print(f"Parallel {method}: runs {list(pending)}, slots={parallel_runs}", flush=True)
                replay_runs(executor, {run_id: configs[method][run_id] for run_id in pending}, pending,
                            runtime_dir, protocol=protocol, replica_seed=replica_seed)


def replay_group(executor, config, path, run_ids, *, protocol, latency_seeds):
    """Replay missing runs through an already resident model and environment pool."""
    for run_id in run_ids:
        run_path = path / "runs" / str(run_id)
        if run_path.exists():
            shutil.rmtree(run_path)  # Only incomplete runs are passed by the caller.
    executor.logger = RunTrajectoryLogger(
        path, run_paths={run_id: path / "runs" / str(run_id) for run_id in run_ids},
        protocol=protocol,
    )
    groups, seeds, sources = grouped_inputs(
        {run_id: {**config["latency"], "seed": latency_seeds[str(run_id)]} for run_id in run_ids},
        executor.clock, protocol=protocol, replica_seed=True,
    )
    return _run_grouped_episodes(
        executor, groups, seeds, sources, config["evaluation"]["max_env_steps"],
    )


def replay_settings(executor, jobs, output, *, protocol):
    """Replay (run ID, path, config, latency seed) jobs independently of slot assignment."""
    groups, seeds, sources, identities, paths = [], [], {}, {}, {}
    for session_id, (run_id, path, config, run_latency_seed) in enumerate(jobs):
        if path.exists():
            shutil.rmtree(path)  # Only incomplete runs enter the job list.
        paths[session_id] = path
        logical, run_seeds, run_sources = grouped_inputs(
            {run_id: {**config["latency"], "seed": run_latency_seed}}, executor.clock,
            protocol=protocol, replica_seed=True,
        )
        group = [protocol["seed_stride"] * session_id + episode for episode in range(protocol["episodes"])]
        groups.append(group)
        seeds.extend(run_seeds)
        for episode, internal_id in enumerate(group):
            sources[internal_id] = run_sources[logical[0][episode]]
            identities[internal_id] = {"run_id": run_id, "episode_in_run": episode,
                                       "episode_seed": run_seeds[episode], "latency_seed": run_latency_seed}
    executor.logger = RunTrajectoryLogger(
        output / "runtime", run_paths=paths,
        protocol=protocol, episode_metadata=identities,
    )
    started = time.perf_counter()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    print(f"PARALLEL_RUNS={len(jobs)} SETTINGS={len({path.parent for path in paths.values()})}", flush=True)
    _run_grouped_episodes(
        executor, groups, seeds, sources, jobs[0][2]["evaluation"]["max_env_steps"],
    )
    timing = output / "runtime/groups" / f"{stamp}.json"
    write_json(timing, {"stage": "sim", "seconds": time.perf_counter() - started, "gpu_count": 1,
                        "runs": [path.relative_to(output).as_posix() for path in paths.values()]})
