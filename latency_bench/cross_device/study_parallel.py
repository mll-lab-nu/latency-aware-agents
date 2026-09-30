"""Replay independent held-out runs with a resident model and batched inference."""
from copy import deepcopy
import shutil

from latency_bench.core.actions import ActionResolver
from latency_bench.cross_device import study
from latency_bench.cross_device.grouped_replay import (
    RunTrajectoryLogger,
    _run_grouped_episodes,
    grouped_inputs,
)
from latency_bench.cross_device.runtime_configs import write_generated_config
from latency_bench.executors.backend_selection import build_env_step_backend, resolve_latency_bench_env_backend
from latency_bench.executors.factory import build_executor
from latency_bench.utils.io import read_json, write_json


def build_replay_executor(config, parallel_runs):
    """Use the task's existing environment backend, including for a one-slot check."""
    config = deepcopy(config)
    config["evaluation"].update(
        eval_parallel_envs=parallel_runs, eval_episodes=study.EPISODES * parallel_runs,
    )
    config["executor"]["inference_batch_size"] = parallel_runs
    config["logging"]["simulated_pipeline_profile"] = False
    study._prepare_replay_runtime(config)
    if config["env"]["name"] == "mikasa_intercept_grab_fast":
        # The simulator interpreter owns this optional ManiSkill dependency.
        from scripts.mikasa.evaluate import MikasaEnvStepBackend
        backend = MikasaEnvStepBackend(config=config, num_slots=parallel_runs)
    else:
        backend = build_env_step_backend(
            resolve_latency_bench_env_backend(config, eval_parallel_envs=parallel_runs),
            config=config, noop_action=ActionResolver(config["env"]).noop_action, num_slots=parallel_runs,
        )
    executor = build_executor(config, env_backend=backend,
                              inference_devices=config["executor"]["inference_devices"])
    return executor, config


def replay_runs(executor, configs, run_paths, runtime, on_run_complete, *, env_seed, replica_seed):
    """Persist each missing two-episode run in its original publication directory."""
    for run_id, path in run_paths.items():
        if path.exists():
            shutil.rmtree(path)  # The caller supplies only incomplete runs.
        write_generated_config(path / "generated.yaml", configs[run_id])
    executor.logger = RunTrajectoryLogger(
        runtime, on_run_complete, run_paths=run_paths,
        env_seed=env_seed, env_seed_stride=study.ENV_SEED_STRIDE,
    )
    groups, seeds, sources = grouped_inputs(
        {run_id: config["latency"] for run_id, config in configs.items()}, executor.clock,
        env_seed=env_seed, env_seed_stride=study.ENV_SEED_STRIDE, replica_seed=replica_seed,
    )
    return _run_grouped_episodes(
        executor, groups, seeds, sources,
        next(iter(configs.values()))["evaluation"]["eval_max_steps"],
    )


def simulate(base, output, manifest, publication, *, inference_device, simulator_device, parallel_runs):
    """Resume only missing runs; retain original profile, clocks, seeds and result paths."""
    protocol = manifest["protocol"]
    fps = read_json(output / "calibration/calibration.json")["selected_env_fps"]
    profile = output / "profile/profile.json"
    base = deepcopy(base)
    base["policy"]["action_noise_seeded"] = manifest["model"] in {"qwenpi_v3", "qwengr00t"}
    replica_seed = manifest["task"] in {"flappy", "mikasa_intercept_grab_fast"}
    runtime = output / "parallel_runtime"
    configs = {
        method: {
            run_id: study.run_config(
                base, output / "sim" / method / str(index), inference_device=inference_device,
                simulator_device=simulator_device, env_fps=fps, run_id=run_id,
                method=method, profile_path=profile, latency_seed=index,
                episodes=protocol["episodes"], env_seed=protocol["seed"],
            ) for index, run_id in enumerate(protocol["valid"])
        } for method in protocol["methods"]
    }
    executor, resolved = build_replay_executor(
        study.run_config(base, runtime, inference_device=inference_device, simulator_device=simulator_device,
                         env_fps=fps, run_id=protocol["valid"][0], method="fixed", profile_path=profile,
                         latency_seed=0, env_seed=protocol["seed"]),
        parallel_runs,
    )
    with study._warmed_replay_executor(executor):
        write_json(runtime / "config.json", resolved)
        context = output / "simulation_parallel.json"
        if not context.exists():
            write_json(context, {
                "parallel_runs": parallel_runs, "inference_batch_size": parallel_runs,
                "action_noise": "episode_frame" if base["policy"]["action_noise_seeded"] else "deterministic",
                "retained_runs": [str(path.parent.relative_to(output)) for path in sorted(output.glob("sim/*/*/DONE"))],
                "episode_id": "20 * valid_run_id + episode_in_run",
                "temporal_replica_zero_seed": replica_seed,
            })
        publication.publish("sim_parallel_context", [context])
        for method in protocol["methods"]:
            paths = {run_id: output / "sim" / method / str(index)
                     for index, run_id in enumerate(protocol["valid"])}
            def publish_run(run_id, path):
                publication.publish(str(path.relative_to(output)), [path])
            pending = {}
            for run_id, path in paths.items():
                if (path / "DONE").exists():
                    publish_run(run_id, path)
                else:
                    pending[run_id] = path
            if pending:
                print(f"Parallel {method}: runs {list(pending)}, slots={parallel_runs}", flush=True)
                replay_runs(executor, {run_id: configs[method][run_id] for run_id in pending}, pending,
                            runtime, publish_run, env_seed=protocol["seed"], replica_seed=replica_seed)
