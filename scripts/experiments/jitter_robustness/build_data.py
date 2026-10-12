from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Sequence



HERE = Path(__file__).resolve().parent
FIELDNAMES = (
    "record_type",
    "horizon_variant",
    "env_id",
    "env_title",
    "train_label",
    "eval_label",
    "seed",
    "episode_id",
    "value",
    "value_unit",
    "std_return",
    "seed_count",
    "seeds",
    "eval_max_steps",
    "source_kind",
    "source_id",
    "source_step",
)


HORIZON_ORDER = {"capped": 0, "uncapped": 1}
ENV_ORDER = {"flappy": 0, "demon-attack": 1, "deadly-corridor": 2}
TRAIN_ORDER = {
    "flappy": {
        label: index
        for index, label in enumerate(
            ("1", "2", "3", "N(2,1)", "U(1,3)", "Mix N(2,1)", "Mix U(1,3)")
        )
    },
    "demon-attack": {
        label: index
        for index, label in enumerate(
            ("2", "4", "6", "N(4,1)", "U(2,6)", "Mix N(4,1)", "Mix U(2,6)")
        )
    },
    "deadly-corridor": {
        label: index
        for index, label in enumerate(
            ("2", "4", "6", "N(4,1)", "U(2,6)", "Mix N(4,1)", "Mix U(2,6)")
        )
    },
}
EVAL_ORDER = {
    "flappy": {
        label: index for index, label in enumerate(("1", "2", "3", "N(2,1)", "U(1,3)"))
    },
    "demon-attack": {
        label: index for index, label in enumerate(("2", "4", "6", "N(4,1)", "U(2,6)"))
    },
    "deadly-corridor": {
        label: index for index, label in enumerate(("2", "4", "6", "N(4,1)", "U(2,6)"))
    },
}
RECORD_ORDER = {"cell_aggregate": 0, "seed_mean_return": 1, "episode_return": 2}

LOCAL_EXPERIMENTS = (
    ("flappy", "Flappy Bird", "flappy", "obs15", 3600, ("capped", "uncapped")),
    (
        "demon-attack",
        "Demon Attack",
        "demon_attack/capped",
        "obs15",
        3600,
        ("capped",),
    ),
    (
        "demon-attack",
        "Demon Attack",
        "demon_attack/extended",
        "obs15",
        36000,
        ("uncapped",),
    ),
    (
        "deadly-corridor",
        "Deadly Corridor",
        "deadly_corridor",
        "obs8p75",
        30000,
        ("capped", "uncapped"),
    ),
)

LOCAL_TRAIN_SETTINGS = {
    "flappy": (
        ("fixed_l1", "1"),
        ("fixed_l2", "2"),
        ("fixed_l3", "3"),
        ("normal_n2_s1", "N(2,1)"),
        ("uniform_u1_3", "U(1,3)"),
        ("mixed_normal_n2_s1", "Mix N(2,1)"),
        ("mixed_uniform_u1_3", "Mix U(1,3)"),
    ),
    "demon-attack": (
        ("fixed_l2", "2"),
        ("fixed_l4", "4"),
        ("fixed_l6", "6"),
        ("normal_n4_s1", "N(4,1)"),
        ("uniform_u2_6", "U(2,6)"),
        ("mixed_normal_n4_s1", "Mix N(4,1)"),
        ("mixed_uniform_u2_6", "Mix U(2,6)"),
    ),
    "deadly-corridor": (
        ("fixed_l2", "2"),
        ("fixed_l4", "4"),
        ("fixed_l6", "6"),
        ("normal_n4_s1", "N(4,1)"),
        ("uniform_u2_6", "U(2,6)"),
        ("mixed_normal_n4_s1", "Mix N(4,1)"),
        ("mixed_uniform_u2_6", "Mix U(2,6)"),
    ),
}

LOCAL_EVAL_PROFILES = {
    "flappy": (
        ("fixed_1_2_3/queue_eval_latency_1.json", "1"),
        ("fixed_1_2_3/queue_eval_latency_2.json", "2"),
        ("fixed_1_2_3/queue_eval_latency_3.json", "3"),
        (
            "normal_mean_2_std_1_min_0_max_4/"
            "queue_eval_latency_normal_mean_2_std_1_min_0_max_4.json",
            "N(2,1)",
        ),
        (
            "uniform_min_1_max_3/queue_eval_latency_uniform_min_1_max_3.json",
            "U(1,3)",
        ),
    ),
    "demon-attack": (
        ("fixed_2_4_6/queue_eval_latency_2.json", "2"),
        ("fixed_2_4_6/queue_eval_latency_4.json", "4"),
        ("fixed_2_4_6/queue_eval_latency_6.json", "6"),
        (
            "normal_mean_4_std_1_min_0_max_15/"
            "queue_eval_latency_normal_mean_4_std_1_min_0_max_15.json",
            "N(4,1)",
        ),
        (
            "uniform_min_2_max_6/queue_eval_latency_uniform_min_2_max_6.json",
            "U(2,6)",
        ),
    ),
    "deadly-corridor": (
        ("fixed_2_4_6/queue_eval_latency_2.json", "2"),
        ("fixed_2_4_6/queue_eval_latency_4.json", "4"),
        ("fixed_2_4_6/queue_eval_latency_6.json", "6"),
        (
            "normal_mean_4_std_1_min_0_max_15/"
            "queue_eval_latency_normal_mean_4_std_1_min_0_max_15.json",
            "N(4,1)",
        ),
        (
            "uniform_min_2_max_6/queue_eval_latency_uniform_min_2_max_6.json",
            "U(2,6)",
        ),
    ),
}

LOCAL_SEEDS = {
    "flappy": (911, 922, 933),
    "demon-attack": (0, 42, 114514),
    "deadly-corridor": (0, 42, 114514),
}








def local_data_rows(results_root: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for (
        env_id,
        env_title,
        root_fragment,
        obs_fragment,
        eval_max_steps,
        variants,
    ) in LOCAL_EXPERIMENTS:
        for train_setting, train_label in LOCAL_TRAIN_SETTINGS[env_id]:
            for seed in LOCAL_SEEDS[env_id]:
                summary_dir = (
                    results_root
                    / root_fragment
                    / "latency_bench"
                    / "jitter_robustness"
                    / train_setting
                    / obs_fragment
                    / f"seed_{seed}"
                )
                for filename, eval_label in LOCAL_EVAL_PROFILES[env_id]:
                    summary_path = summary_dir / filename
                    with summary_path.open(encoding="utf-8") as f:
                        summary = json.load(f)
                    for variant in variants:
                        for episode_id, value in enumerate(summary["returns"]):
                            rows.append(
                                {
                                    "record_type": "episode_return",
                                    "horizon_variant": variant,
                                    "env_id": env_id,
                                    "env_title": env_title,
                                    "train_label": train_label,
                                    "eval_label": eval_label,
                                    "seed": seed,
                                    "episode_id": episode_id,
                                    "value": value,
                                    "value_unit": "episode_return",
                                    "std_return": "",
                                    "seed_count": "",
                                    "seeds": "",
                                    "eval_max_steps": eval_max_steps,
                                    "source_kind": "local_json",
                                    "source_id": str(summary_path.relative_to(results_root)),
                                    "source_step": "",
                                }
                            )
    return rows


def row_order(row: dict[str, object]) -> tuple[object, ...]:
    seed = -1 if row["record_type"] == "cell_aggregate" else int(row["seed"])
    episode_id = (
        -1 if row["record_type"] != "episode_return" else int(row["episode_id"])
    )
    env_id = str(row["env_id"])
    return (
        HORIZON_ORDER[str(row["horizon_variant"])],
        ENV_ORDER[env_id],
        TRAIN_ORDER[env_id][str(row["train_label"])],
        EVAL_ORDER[env_id][str(row["eval_label"])],
        RECORD_ORDER[str(row["record_type"])],
        seed,
        episode_id,
    )


def write_rows(rows: Sequence[dict[str, object]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(sorted(rows, key=row_order))


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate completed local jitter evaluations")
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = local_data_rows(args.results_root)
    write_rows(rows, args.output)
    print(f"Wrote {args.output} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
