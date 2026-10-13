"""Compare real LeRobot examples with the corrected checkpoint evaluation path."""

from __future__ import annotations

import argparse
import io
from pathlib import Path
import sys

import numpy as np
from omegaconf import OmegaConf
from PIL import Image
import torch

from latency_bench.utils.io import write_json_atomic
from latency_bench.core.actions import ActionResolver
from latency_bench.core.types import Observation
from latency_bench.policy.starvla import build_starvla_policy
from scripts.tasks.decision_history.eval_vla_hist8 import evaluation_config
from starVLA.dataloader.lerobot_datasets import make_LeRobotSingleDataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--bundle-dir", type=Path, required=True)
    parser.add_argument("--backbone", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    train_cfg = OmegaConf.load(args.run_dir / "config.full.yaml")
    data_cfg = train_cfg.datasets.vla_data
    dataset = make_LeRobotSingleDataset(args.dataset_dir.parent, args.dataset_dir.name,
                                       "rl_games_demon_attack", data_cfg=data_cfg)
    config = evaluation_config("demon_attack", "qwenoft", args.bundle_dir, args.profile,
                               args.output_dir, args.backbone, Path(sys.executable), device="cuda:0", episodes=2)
    policy = build_starvla_policy(config, ActionResolver(config["env"]))
    indices = np.linspace(101, len(dataset) - 100, 64, dtype=int)
    examples = []
    observations = []
    for index in indices:
        example = dataset[index]
        trajectory_id, frame_index = dataset.all_steps[index]
        row = dataset.get_trajectory_data(trajectory_id).iloc[frame_index]
        raw_image = np.asarray(Image.open(io.BytesIO(row["observation.image"]["bytes"])).convert("RGB"))
        observation = Observation(data={"transport": example["state"].reshape(-1)},
                                  env_step=int(frame_index), sim_time_ms=0.,
                                  metadata={"env_raw_rgb_frame_stack": raw_image[None], "slot_id": 0})
        actual = policy._build_example(observation)
        np.testing.assert_array_equal(actual["state"], example["state"])
        np.testing.assert_array_equal(np.asarray(actual["image"][0]), np.asarray(example["image"][0]))
        assert actual["lang"] == example["lang"]
        observations.append(observation)
        examples.append(example)
    model = policy._wrapper._framework
    model_counts = np.zeros(6, dtype=np.int64)
    for start in range(0, len(examples), 8):
        with torch.no_grad():
            expected = model.predict_action(examples[start:start + 8])["normalized_actions"]
        actual = policy.predict_batch(observations[start:start + 8])
        scores = np.asarray([item.metadata["raw_action_scores"] for item in actual])
        np.testing.assert_allclose(scores, expected[:, 0], rtol=1e-4, atol=1e-4)
        np.testing.assert_array_equal([item.action.value for item in actual], expected[:, 0].argmax(-1))
        model_counts += np.bincount(expected[:, 0].argmax(-1), minlength=6)
    reset = observations[0]
    reset.data["transport"] = np.zeros(88, dtype=np.float32)
    np.testing.assert_array_equal(policy._build_example(reset)["state"], np.zeros((1, 88)))
    write_json_atomic(args.output_dir / "input_parity.json", {
        "passed": True, "checkpoint": str(args.bundle_dir / "checkpoints/model.pt"), "samples": len(examples),
        "history_encoding": "binary_0_1", "action_counts": model_counts.tolist(),
        "checked": ["image_pixels", "prompt", "state", "raw_scores", "decoded_action", "reset_history"],
    })
    print("HIST8_INPUT_PARITY_VERIFIED", args.bundle_dir, model_counts.tolist(), flush=True)


if __name__ == "__main__":
    main()
