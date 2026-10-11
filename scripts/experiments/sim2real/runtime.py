"""Runtime configuration, asset preparation and resource lifetime for sim2real experiments."""

from latency_bench.core.config import resolve_runtime_config
from contextlib import contextmanager
from copy import deepcopy
import os
import shutil
import subprocess
import sys

from latency_bench.profiling.runtime_configs import (
    set_env_clock_preserving_observation_stride, apply_profile_latency_summary,
    eval_command, write_generated_config,
)
from latency_bench.latency.hardware_info import collect_hardware_fingerprint
from latency_bench.utils.io import read_json, write_json, write_json_atomic
from latency_bench.utils.paths import REPO_ROOT
from scripts.experiments.sim2real.assets import prepare_model_assets


def run_config(base, output, *, inference_device, simulator_device, env_fps,
               run_id, method="measured", profile_path=None, latency_seed=None,
               protocol):
    """Change execution coordinates while retaining the archived task contract."""
    config = deepcopy(base)
    set_env_clock_preserving_observation_stride(config, env_fps)
    if simulator_device is not None:
        config["env"]["simulator_device"] = simulator_device
    config["executor"].update(
        mode="realtime" if method == "measured" else "simulated",
        inference_devices=[inference_device],
    )
    if method == "measured":
        for key in ("simulated_worker_capacity", "inference_pool", "inference_batch_size"):
            config["executor"].pop(key, None)
        config.pop("latency", None)
        config["executor"]["sync_cuda"] = True
    else:
        profile = read_json(profile_path)
        config["executor"].update(
            simulated_worker_capacity=1, inference_pool=True, inference_batch_size=1,
        )
        config["latency"] = {
            "method": method, "profile_path": str(profile_path), "seed": latency_seed,
            "profile_worker_slot": 0,
        }
        apply_profile_latency_summary(config["latency"], method, profile)
    if method != "measured" and "add_latency_info" in base.get("latency", {}):
        config["latency"]["add_latency_info"] = base["latency"]["add_latency_info"]
    config["experiment"]["seed"] = protocol["seed"] + protocol["seed_stride"] * run_id
    config["evaluation"].update(
        episodes=protocol["episodes"], suites=[],
        episode_seed_offset=0,
    )
    config["executor"]["parallel_envs"] = 1
    config["logging"].update(
        output_dir=str(output), save_step_records=True, save_action_records=True,
        save_latency_records=True,
    )
    return config


def prepare_replay_runtime(config):
    """Resolve model inputs and RNG before building a simulated replay executor."""
    # Real collection does not need to import the replay policy/model runtime.

    from latency_bench.policy.starvla import prepare_starvla_checkpoint_input_config
    from latency_bench.utils.seeding import seed_everything

    resolve_runtime_config(config)
    prepare_starvla_checkpoint_input_config(config)
    seed_everything(config["experiment"]["seed"])


@contextmanager
def warmed_replay_executor(executor):
    """Own warmup and closure, including a failed warmup, for resident replay."""
    try:
        executor.run_warmup()
        yield executor
    finally:
        executor.close()


def execute_stage(config, output, entry):
    """Resume completed protocol stages; restart an interrupted stage in full."""
    if (output / "DONE").exists():
        return
    if output.exists():
        shutil.rmtree(output)
    generated = output / "generated.yaml"
    write_generated_config(generated, config)
    print(f"Running {output}; log: {output / 'stdout.log'}", flush=True)
    command = eval_command(entry, generated, output)
    with (output / "stdout.log").open("w") as log:
        subprocess.run(command, cwd=REPO_ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
    (output / "DONE").touch()


def runtime_base(source, worker_python):
    manifest = read_json(source / "experiment.json")
    base = prepare_model_assets(manifest, manifest["assets"], eval_config_path=source / "eval_template.yaml")
    base["policy"]["worker_python_executable"] = worker_python
    fps = read_json(source / "calibration/calibration.json")["selected_env_fps"]
    return base, manifest, fps


def record_execution_contract(output, stage, config):
    """Freeze measurement identity before accepting completed runs on resume."""
    hardware = collect_hardware_fingerprint()
    selected = [gpu for gpu in hardware["gpus"]
                if gpu["uuid"] == os.environ["CUDA_VISIBLE_DEVICES"]][0]
    contract = {"config": config, "python": sys.executable, "gpu_uuid": selected["uuid"],
                "driver_version": hardware["gpu"]["driver_version"], "packages": hardware["packages"],
                "cpu": {key: value for key, value in hardware["cpu"].items() if key != "cpu_mhz"},
                "memory": hardware["memory"], "system": hardware["system"]}
    contract_path = output / f"{stage}_contract.json"
    if contract_path.exists():
        if read_json(contract_path) != contract:
            raise ValueError(f"{stage.title()} execution contract changed: {contract_path}")
    else:
        write_json_atomic(contract_path, contract)
        write_json(output / f"{stage}_hardware.json", {"gpu": selected, "fingerprint": hardware})
