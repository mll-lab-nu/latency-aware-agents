#!/usr/bin/env python3
"""Audit raw and LeRobot conditioned MIKASA H8 datasets before training."""

from __future__ import annotations

import argparse
import json
from scripts.tasks.mikasa_intercept_grab_fast.fixed_latency import FIXED_LATENCIES
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


PROFILE_REVISION = "66414463f2a609316da63935941a79abf0a73ce9"


def _parquet_rows(dataset: Path) -> list[dict]:
    rows = []
    for path in sorted((dataset / "data").glob("*/*.parquet")):
        rows.extend(pq.read_table(path).to_pylist())
    return rows


def _raw_rows(raw_root: Path, episodes: list[int]) -> list[dict[str, np.ndarray]]:
    rows = []
    for episode_index in episodes:
        episode = np.load(raw_root / f"episodes/episode_{episode_index:06d}.npz")
        prefix_masks = episode["action_prefix_mask"]
        loss_masks = episode["action_loss_mask"]
        chunks = episode["issued_command_chunk"]
        states = episode["state"]
        actions = episode["action"]
        latency_frames = episode["latency_raw_frames"]
        assert states.shape[1:] == (71,)
        assert chunks.shape[1:] == (8, 7)
        assert prefix_masks.shape[1:] == (8,)
        assert loss_masks.shape[1:] == (8,)
        assert np.isfinite(states).all()
        assert np.isfinite(chunks).all()
        np.testing.assert_array_equal(actions, chunks[:, 0])
        for state, chunk, prefix_mask, loss_mask, latency_frame in zip(
            states, chunks, prefix_masks, loss_masks, latency_frames
        ):
            rows.append(
                {
                    "state": state,
                    "chunk": chunk,
                    "prefix_mask": prefix_mask,
                    "loss_mask": loss_mask,
                    "latency_frame": latency_frame,
                }
            )
    return rows


def audit(
    raw_root: Path,
    dataset: Path,
    *,
    condition: str,
) -> dict:
    collection = json.loads((raw_root / "collection.json").read_text(encoding="utf-8"))
    attempts = [
        json.loads(line)
        for line in (raw_root / "attempts.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [row["seed"] for row in attempts] == list(range(collection["attempted_episodes"]))
    num_envs = collection["collection_num_envs"]
    assert collection["latency_stream_scope"] == "collector_slot"
    for batch_index, start in enumerate(range(0, len(attempts), num_envs)):
        batch = attempts[start : start + num_envs]
        assert [row["batch_index"] for row in batch] == [batch_index] * num_envs
        assert [row["slot_id"] for row in batch] == list(range(num_envs))
        assert len({row["split"] for row in batch}) == 1
    assert collection["format"] == "mikasa_conditioned_h8_raw_v2"
    saved_by_index = {
        row["episode_index"]: row for row in collection["saved_episodes"]
    }
    train_split = collection["splits"]["train"]
    validation_split = collection["splits"]["validation"]
    assert set(train_split["seeds"]).isdisjoint(validation_split["seeds"])
    for split in (train_split, validation_split):
        episodes = [saved_by_index[index] for index in split["episode_indices"]]
        assert split["seeds"] == [episode["seed"] for episode in episodes]
        assert split["rows"] == sum(episode["labels"] for episode in episodes)
        assert split["valid_actions"] == sum(episode["valid_actions"] for episode in episodes)
        assert split["valid_actions"] >= split["target_valid_actions"]
        assert sum(episode["valid_actions"] for episode in episodes[:-1]) < split[
            "target_valid_actions"
        ]
        assert split["successes"] == sum(episode["success"] for episode in episodes)
        assert split["failures"] == len(episodes) - split["successes"]

    selected_attempts = [row for row in attempts if row["selected"]]
    assert [row["seed"] for row in selected_attempts] == (
        train_split["seeds"] + validation_split["seeds"]
    )
    assert len(selected_attempts) == len(saved_by_index)
    for episode_index, attempt in enumerate(selected_attempts):
        assert saved_by_index[episode_index] == {
            "episode_index": episode_index,
            **attempt,
        }

    prefix = collection["action_prefix"]
    prefix_frames = prefix["frames"]
    expected_prefix_mask = np.arange(8) < prefix_frames
    if condition == "profile-latency":
        assert collection["latency_config"]["profile_source"]["revision"] == PROFILE_REVISION
        assert collection["latency_config"]["latency"]["seed"] == 0
        assert prefix["mode"] == "profile_p95"
        assert prefix["profile_revision"] == PROFILE_REVISION
    else:
        assert collection["latency_config"]["latency"] == {
            "method": "fixed",
            "fixed_latency_ms": FIXED_LATENCIES[condition] * 50.0,
        }
        assert prefix["mode"] == "fixed_frames"
        assert prefix_frames == FIXED_LATENCIES[condition]

    raw_train = _raw_rows(raw_root, train_split["episode_indices"])
    raw_validation = _raw_rows(raw_root, validation_split["episode_indices"])
    for row in raw_train + raw_validation:
        np.testing.assert_array_equal(row["prefix_mask"], expected_prefix_mask)
        assert row["loss_mask"].any()
        assert not np.logical_and(row["prefix_mask"], row["loss_mask"]).any()
        suffix_length = int(row["loss_mask"].sum())
        np.testing.assert_array_equal(
            row["loss_mask"],
            (np.arange(8) >= prefix_frames) & (np.arange(8) < prefix_frames + suffix_length),
        )
        if condition in FIXED_LATENCIES:
            assert row["latency_frame"] == FIXED_LATENCIES[condition]

    datasets = (
        (dataset, raw_train),
        (dataset.with_name(f"{dataset.name}__val"), raw_validation),
    )
    split_counts = {}
    for split, (dataset_dir, raw_rows) in zip(("train", "validation"), datasets):
        contract = json.loads((dataset_dir / "task_contract.json").read_text(encoding="utf-8"))
        assert contract["format"] == "latency_bench_starvla_h8_v2"
        assert contract["robot_type"] == "latency_mikasa_conditioned_h8"
        assert (contract["state_dim"], contract["action_dim"], contract["action_horizon"]) == (
            71,
            7,
            8,
        )
        assert contract["action_prefix"] == prefix
        modality = json.loads((dataset_dir / "meta/modality.json").read_text(encoding="utf-8"))
        assert [entry["original_key"] for entry in modality["video"].values()] == [
            "observation.images.top",
            "observation.images.wrist",
        ]
        assert modality["annotation"] == {
            "human.task_description": {"original_key": "task_index"}
        }
        packaged_rows = _parquet_rows(dataset_dir)
        assert len(packaged_rows) == len(raw_rows)
        for packaged, raw in zip(packaged_rows, raw_rows):
            np.testing.assert_array_equal(packaged["observation.state"], raw["state"])
            np.testing.assert_array_equal(packaged["issued_command_chunk"], raw["chunk"])
            np.testing.assert_array_equal(packaged["action_prefix_mask"], raw["prefix_mask"])
            np.testing.assert_array_equal(packaged["action_loss_mask"], raw["loss_mask"])
        source = collection["splits"][split]
        split_counts[split] = {
            "episodes": len(source["episode_indices"]),
            "rows": len(packaged_rows),
            "valid_actions": source["valid_actions"],
            "successes": source["successes"],
            "failures": source["failures"],
        }

    result = {
        "passed": True,
        "condition": condition,
        "model": collection["model"],
        "collection_num_envs": num_envs,
        "latency_stream_scope": collection["latency_stream_scope"],
        "attempted_episodes": collection["attempted_episodes"],
        "selected_episodes": len(selected_attempts),
        "split_seeds": {
            split: collection["splits"][split]["seeds"]
            for split in ("train", "validation")
        },
        "action_prefix": prefix,
        "latency_raw_frame_counts": collection["latency_raw_frame_counts"],
        "splits": split_counts,
    }
    (dataset.parent / "dataset_audit.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--condition", choices=("profile-latency", *FIXED_LATENCIES), required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            audit(
                args.raw_root,
                args.dataset,
                condition=args.condition,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
