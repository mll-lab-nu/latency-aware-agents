#!/usr/bin/env python
"""Reconstruct config.yaml + dataset_statistics.json for an HF-synced checkpoint.

The training checkpoint sync only uploads the model state (steps_N_state/), not
the run's config.yaml or dataset_statistics.json. ``baseframework.from_pretrained``
needs both at the run dir (= the state dir's parents[1]). If the original run dir
is gone (e.g. a fresh server), this regenerates them:

  * config.yaml — recomposed from the *same* Hydra config the run used
    (model/env/init/mode + overrides). The architecture (QwenOFT, action_dim,
    Qwen3-VL backbone) is deterministic from ``model=...``, so it matches the
    weights. ``framework.qwenvl.base_vlm`` is pointed at --base-vlm on this box.
  * dataset_statistics.json — a stub by default (discrete-action models don't use
    norm stats at inference); pass --stats-from to copy a real one instead.

Place the outputs at the run dir so the checkpoint loads, e.g. for a checkpoint at
``<run>/checkpoints/steps_5000_state/model.safetensors`` write them to ``<run>/``.

Example (flappy mixed-latency openvla run):
    python examples/rl_games/scripts/reconstruct_checkpoint_config.py \
        --run-dir /workspace/outputs/_ckpt/openvla_flappy_mixed/_loadable \
        --model openvla --env flappy --init bridge --mode mixed_latency \
        --base-vlm /workspace/playground/Pretrained_models/Qwen3-VL-4B-Instruct-Action
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

CONFIG_DIR = REPO_ROOT / "examples" / "rl_games" / "config"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", required=True, help="Where to write config.yaml + dataset_statistics.json.")
    p.add_argument("--base-vlm", required=True, help="Local path to the base Qwen3-VL model on this box.")
    p.add_argument("--model", default="openvla")
    p.add_argument("--env", default="flappy")
    p.add_argument("--init", default="bridge")
    p.add_argument("--mode", default="mixed_latency")
    p.add_argument("--config-name", default="train")
    p.add_argument("--run-id", default="reconstructed_run", help="Dummy run_id to satisfy config validation.")
    p.add_argument("--workspace-dir", default="/workspace")
    p.add_argument(
        "--override",
        action="append",
        default=[],
        help="Extra Hydra override(s), repeatable, e.g. --override datasets.vla_data.prompt_mode=latency_neutral. "
        "Use the architecture-affecting overrides from the original launch if any.",
    )
    p.add_argument(
        "--stats-from",
        default="",
        help="Copy this dataset_statistics.json instead of writing a stub (e.g. from the converted dataset dir).",
    )
    return p.parse_args()


def build_cfg(args):
    from hydra import compose, initialize_config_dir
    from starVLA.training.rl_games import apply_action_spec, apply_model_alias, validate_rl_games_config

    overrides = [
        f"model={args.model}",
        f"env={args.env}",
        f"init={args.init}",
        f"mode={args.mode}",
        f"run_id={args.run_id}",
        f"workspace_dir={args.workspace_dir}",
        # Point the backbone at the base VLM available on this box (concrete path).
        f"paths.base_model_dir={args.base_vlm}",
        f"framework.qwenvl.base_vlm={args.base_vlm}",
        *args.override,
    ]
    with initialize_config_dir(version_base="1.1", config_dir=str(CONFIG_DIR)):
        cfg = compose(config_name=args.config_name, overrides=overrides)
    try:
        validate_rl_games_config(cfg)
    except Exception as exc:  # noqa: BLE001 - validation may require run-only fields we don't need
        print(f"[warn] validate_rl_games_config raised (continuing): {exc}")
    apply_model_alias(cfg)
    apply_action_spec(cfg)
    return cfg


def main() -> None:
    args = parse_args()
    from omegaconf import OmegaConf

    run_dir = Path(args.run_dir).expanduser().resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    cfg = build_cfg(args)

    # Resolve interpolations now and save a fully-concrete config.yaml, since
    # read_mode_config does to_container(resolve=True) at load time.
    config_path = run_dir / "config.yaml"
    try:
        container = OmegaConf.to_container(cfg, resolve=True)
        OmegaConf.save(config=OmegaConf.create(container), f=str(config_path))
    except Exception as exc:  # noqa: BLE001 - fall back to unresolved if some interpolation can't resolve
        print(f"[warn] could not fully resolve config ({exc}); saving unresolved.")
        OmegaConf.save(config=cfg, f=str(config_path))
    print(f"Wrote {config_path}")

    stats_path = run_dir / "dataset_statistics.json"
    if args.stats_from:
        import shutil

        shutil.copyfile(Path(args.stats_from).expanduser(), stats_path)
        print(f"Copied dataset_statistics.json from {args.stats_from} -> {stats_path}")
    else:
        # Discrete-action (discrete_ce) models decode by argmax and do not use norm
        # stats at inference, so an empty stub satisfies read_mode_config's check.
        stats_path.write_text(json.dumps({}), encoding="utf-8")
        print(f"Wrote stub {stats_path} (empty; fine for discrete-action inference).")

    print(
        "\nDone. Point the checkpoint loader's run dir here. For a state dir at "
        f"{run_dir}/checkpoints/steps_N_state/model.safetensors it will resolve this run dir automatically."
    )


if __name__ == "__main__":
    main()
