"""Show the L1 action queue transition for the Lunar Lander config."""

import json
from pathlib import Path
from types import SimpleNamespace

from latency_bench.core.config import load_config
from training.gymnasium_sf.integration import (
    SAMPLE_FACTORY_CONFIG_PARSER,
    build_cli_args_from_config,
    make_gymnasium_sf_env,
)


def main() -> None:
    config = load_config("configs/examples/gymnasium/lunar_lander/small_model_train.yaml")
    config["latency"]["add_latency_info"] = True
    cfg = SAMPLE_FACTORY_CONFIG_PARSER.parse_train(build_cli_args_from_config(config))
    env = make_gymnasium_sf_env(
        "latency_gymnasium",
        cfg=cfg,
        env_config=SimpleNamespace(worker_index=0, vector_index=0, env_id=0),
    )
    _, reset_info = env.reset(seed=7)
    trace = []
    for issued_action_id in [2, 1, 3]:
        _, reward, terminated, truncated, info = env.step(issued_action_id)
        trace.append(
            {
                "issued_action_id": issued_action_id,
                "applied_action_id": info["applied_action"],
                "ready_raw_frame": info["ready_raw_frame"],
                "env_step_idx": info["env_step_idx"],
                "latency_raw_frames": info["latency_raw_frames"],
                "queued_action": info["queued_action"],
                "reward": reward,
                "terminated": terminated,
                "truncated": truncated,
            }
        )
    payload = {
        "fixed_latency_ms": config["latency"]["fixed_latency_ms"],
        "env_fps": config["env"]["env_fps"],
        "expected_latency_raw_frames": 1,
        "reset_applied_action": reset_info["applied_action"],
        "trace": trace,
        "claim": (
            "With L1, issued action 2 is not applied on its issue tick "
            "(do_nothing=0); it is applied on the next raw tick."
        ),
    }
    output_path = Path("artifacts/lunar_lander/l1_scheduler_trace.json")
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    env.close()


if __name__ == "__main__":
    main()
