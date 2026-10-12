"""Replay published ghost-trail episodes with raw RGB observations."""

import argparse
import json
from pathlib import Path

from datasets import load_dataset
from huggingface_hub import hf_hub_download

from latency_bench.core.config import load_config
from latency_bench.data.rollout import collect_rollouts
from latency_bench.data.rollout_probe import write_probe_state


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args(argv)
    config = load_config(args.config)
    source = config["replay_source"]
    output = Path(config["logging"]["output_dir"])
    state_path = output / "probe_state.json"
    if state_path.exists():
        state = json.loads(state_path.read_text())
        if state["replay_source"] != source:
            raise ValueError("Replay source does not match the existing probe state")
    else:
        if output.exists() and any(output.iterdir()):
            raise FileExistsError(output)
        repo, revision, directory = source["repo_id"], source["revision"], source["path"]
        dataset = load_dataset("parquet", data_files={
            split: f"hf://datasets/{repo}@{revision}/{directory}/{split}.parquet"
            for split in ("train", "val")
        })
        specs = {}
        for split in dataset.values():
            for row in split.select_columns(["episode_idx", "seed", "split", "decision_step"]):
                index = row["episode_idx"]
                spec = specs.setdefault(index, {
                    "attempt_idx": index, "episode_idx": index, "split": row["split"],
                    "seed": row["seed"], "episode_length": 0, "episode_raw_return": 0.0,
                })
                spec["episode_length"] = max(spec["episode_length"], row["decision_step"] + 1)
        metadata_path = hf_hub_download(repo, directory + "/metadata.json", repo_type="dataset", revision=revision)
        metadata = json.loads(Path(metadata_path).read_text())
        write_probe_state(output, {
            "next_attempt_idx": len(specs), "attempted_episodes": len(specs), "rejected_episodes": 0,
            "accepted_specs": [specs[index] for index in sorted(specs)],
            "runtime_metadata": {key: metadata[key] for key in (
                "checkpoint_train_step", "frame_stack", "env_fps", "obs_fps")},
            "replay_source": source,
        })
    collect_rollouts(config)


if __name__ == "__main__":
    main()
