"""Evaluate a deterministic SONIC run/jump teacher on G1 Hurdle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import gymnasium as gym
import mujoco
import numpy as np
import onnxruntime as ort

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from latency_bench.envs.humanoidbench_sonic import (
    G1_CONTROL_MODE_SONIC,
    G1_SONIC_ENV_SPECS,
    SONIC_DECODER_MODE_ENVIRONMENT,
    g1_raw_reset_action,
)
from latency_bench.envs.humanoidbench_sonic_assets import (
    prepare_g1_sonic_assets,
    prepare_g1_sonic_encoder,
    prepare_g1_sonic_skill_references,
)
from latency_bench.envs.humanoidbench_sonic_contract import G1_BODY_JOINT_NAMES
from latency_bench.envs.humanoidbench_sonic_teacher import (
    DEFAULT_HURDLE_POSITIONS,
    DEFAULT_TRIGGER_DISTANCES,
    FOOT_CONTACT_HEIGHT,
    evaluate_trigger_distances,
)
from latency_bench.envs.humanoidbench_sonic_reference import (
    canonicalize_reference_heading,
    default_initial_qpos,
    encode_qpos_tokens,
    load_bones_seed_reference_qpos,
)


def _reference_foot_contacts(
    scene_path: Path,
    reference_qpos: np.ndarray,
) -> np.ndarray:
    model = mujoco.MjModel.from_xml_path(str(scene_path))
    data = mujoco.MjData(model)
    body_qpos_addresses = np.asarray(
        [model.jnt_qposadr[model.joint(name).id] for name in G1_BODY_JOINT_NAMES]
    )
    foot_site_ids = [model.site(name).id for name in ("left_foot", "right_foot")]
    contacts = []
    for reference_frame in reference_qpos:
        mujoco.mj_resetDataKeyframe(model, data, 0)
        data.qpos[:7] = reference_frame[:7]
        data.qpos[body_qpos_addresses] = reference_frame[7:]
        mujoco.mj_forward(model, data)
        contacts.append(data.site_xpos[foot_site_ids, 2] <= FOOT_CONTACT_HEIGHT)
    return np.asarray(contacts)


def evaluate(
    *,
    cache_dir: Path,
    trigger_distances: tuple[float, ...] = DEFAULT_TRIGGER_DISTANCES,
    jump_start_phase: int,
    seed: int = 0,
) -> dict[str, Any]:
    """Run the trigger sweep against the shared SONIC Hurdle environment."""

    asset_root, decoder_path = prepare_g1_sonic_assets(cache_dir)
    encoder_path = prepare_g1_sonic_encoder(cache_dir)
    skill_paths = prepare_g1_sonic_skill_references(cache_dir)
    scene_path = asset_root / G1_SONIC_ENV_SPECS["g1_sonic_hurdle"][
        "scene_filename"
    ].removeprefix("sonic_g1/")
    encoder = ort.InferenceSession(
        str(encoder_path), providers=["CPUExecutionProvider"]
    )

    run_qpos, run_num_tokens = load_bones_seed_reference_qpos(
        skill_paths["fast_run"]
    )
    jump_qpos, jump_num_tokens = load_bones_seed_reference_qpos(
        skill_paths["obstacle_jump"]
    )
    run_qpos = canonicalize_reference_heading(run_qpos)
    jump_qpos = canonicalize_reference_heading(jump_qpos)
    run_tokens = encode_qpos_tokens(encoder, run_qpos, run_num_tokens - 1)
    jump_tokens = encode_qpos_tokens(encoder, jump_qpos, jump_num_tokens - 1)
    run_reference_qpos = run_qpos[:run_num_tokens]
    jump_reference_qpos = jump_qpos[:jump_num_tokens]

    scene_path = scene_path.resolve()
    decoder_path = decoder_path.resolve()
    jump_tokens = jump_tokens[jump_start_phase:]
    jump_reference_qpos = jump_reference_qpos[jump_start_phase:]
    run_reference_foot_contacts = _reference_foot_contacts(
        scene_path, run_reference_qpos
    )
    jump_reference_foot_contacts = _reference_foot_contacts(
        scene_path, jump_reference_qpos
    )
    neutral_hand_action = np.asarray(
        g1_raw_reset_action(str(scene_path), learn_hands=True)[-14:],
        dtype=np.float32,
    )

    def env_factory():
        env = gym.make(
            G1_SONIC_ENV_SPECS["g1_sonic_hurdle"]["env_id"],
            scene_path=str(scene_path),
            control_mode=G1_CONTROL_MODE_SONIC,
            sonic_decoder_path=str(decoder_path),
            sonic_decoder_mode=SONIC_DECODER_MODE_ENVIRONMENT,
            sonic_initial_reference_qpos=default_initial_qpos(),
            render_mode=None,
            width=256,
            height=256,
        )
        env.unwrapped.randomness = 0.0
        return env

    result = evaluate_trigger_distances(
        env_factory,
        run_tokens,
        jump_tokens,
        run_reference_qpos,
        jump_reference_qpos,
        run_reference_foot_contacts,
        jump_reference_foot_contacts,
        neutral_hand_action,
        trigger_distances,
        seed=seed,
        hurdle_positions=DEFAULT_HURDLE_POSITIONS,
    )
    return {"jump_start_phase": jump_start_phase, **result}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--trigger-distance", type=float, action="append")
    parser.add_argument("--jump-start-phase", type=int, required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    result = evaluate(
        cache_dir=args.cache_dir,
        trigger_distances=(
            tuple(args.trigger_distance)
            if args.trigger_distance is not None
            else DEFAULT_TRIGGER_DISTANCES
        ),
        jump_start_phase=args.jump_start_phase,
        seed=args.seed,
    )
    print(json.dumps(result, indent=2, default=lambda value: value.item()))


if __name__ == "__main__":
    main()
