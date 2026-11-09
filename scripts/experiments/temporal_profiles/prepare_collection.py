"""Write matched Flappy demonstration configs for an explicitly chosen teacher."""
from __future__ import annotations

import argparse
from pathlib import Path

from latency_bench.integrations.sample_factory.teacher_config import load_rollout_training_config
from latency_bench.utils.io import write_yaml


def prepare(checkpoint: Path, profile: Path, output: Path, *, episodes=200, max_steps=7200):
    if not checkpoint.is_file():
        raise ValueError("--checkpoint must identify the selected teacher checkpoint file")
    if episodes < 1 or max_steps < 1:
        raise ValueError("Episode count and raw-frame cap must be positive")
    descriptor, _ = load_rollout_training_config(checkpoint.absolute())
    flat = descriptor["flat_cfg"]
    method = {"profile_sample": "temporal", "hmm": "hmm", "block": "block",
              "ar1": "ar1", "arma": "arma"}[flat["latency_type"]]
    if (descriptor["env_name"], flat["env_fps"], flat["obs_fps"], flat["action_history_decisions"],
            flat["simulated_worker_capacity"], flat["latency_profile_worker_slot"]) != ("flappy", 10, 10, 8, 1, 0):
        raise ValueError("This comparison requires a Flappy 10/10 Hz H8 teacher with worker capacity 1, slot 0")
    # Validate that relocation points to the intended kind of fitted bundle.
    from latency_bench.core.clock import EnvClock
    from latency_bench.latency.samplers import build_sampler
    build_sampler({"method": method, "profile_path": str(profile)}, clock=EnvClock(env_fps=10, obs_fps=10))
    output = output.resolve()
    checkpoint_root = Path(descriptor["checkpoint_experiment_dir"])
    checkpoint_file = Path(descriptor["checkpoint_path"])
    config = {
        "experiment": {"name": f"flappy_{method}_collection_{max_steps}", "seed": 42},
        "env": {"name": "flappy", "env_fps": 10, "obs_fps": 10, "simulator": "gpu"},
        "executor": {"mode": "simulated", "parallel_envs": 256, "inference_devices": ["gpu"],
                     "simulated_worker_capacity": 1},
        "policy": {"type": "sample_factory", "source": "local", "root": str(checkpoint_root),
                   "checkpoint": str(checkpoint_file.relative_to(checkpoint_root))},
        "scheduler": {"hold_policy": flat["hold_policy"], "ordering_policy": flat["ordering_policy"]},
        "latency": {"method": method, "profile_path": str(profile.resolve()), "profile_worker_slot": 0,
                    "seed": 42, "add_latency_info": False},
        "evaluation": {"episodes": episodes, "max_env_steps": max_steps},
        "logging": {"output_dir": str(output / "raw")},
        "collection": {"two_pass": True, "max_attempts": episodes, "replay_num_envs": 256,
                       "val_fraction": .1, "split_seed": 0, "filter_preset": "none", "context_window": 1,
                       "row_buffer_size": 8192, "image_writer_workers": 32},
    }
    configs = {"collect.yaml": config, "convert.yaml": {
        "conversion": {"format": "lerobot", "input_dir": str(output / "raw"),
                       "output_dir": str(output / "lerobot"), "action_carrier": "native"},
    }}
    for name in configs:
        if (output / name).exists():
            raise FileExistsError(f"Config already exists: {output / name}")
    for name, data in configs.items():
        write_yaml(output / name, data)
    return config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--profile-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--max-steps", type=int, default=7200)
    args = parser.parse_args()
    prepare(args.checkpoint, args.profile_path, args.output_dir, episodes=args.episodes, max_steps=args.max_steps)
    print(f"Collection and conversion configs: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
