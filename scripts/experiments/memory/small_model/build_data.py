from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

FIELDNAMES = (
    "record_type",
    "horizon_variant",
    "env_id",
    "env_title",
    "env_fps",
    "context_window",
    "latency_regime",
    "latency_label",
    "seed",
    "episode_id",
    "value",
    "value_unit",
    "eval_max_steps",
    "eval_episode_count",
    "source_kind",
    "source_id",
    "source_step",
)


HERE = Path(__file__).resolve().parent
SEEDS = (10, 11, 12, 13, 14)


@dataclass(frozen=True)
class LocalProfile:
    regime: str
    label: str
    setting_dir: str
    suite_dir: str
    filename: str


@dataclass(frozen=True)
class LocalEnv:
    env_id: str
    title: str
    source_dir: str
    fps: int
    context_windows: tuple[int, ...]
    profiles: tuple[LocalProfile, ...]


@dataclass(frozen=True)
class LocalVariant:
    name: str
    output_dir: str | None
    eval_max_steps: int
    context_windows: tuple[int, ...]


DEMON_PROFILES = (
    LocalProfile("fixed_l4", "fixed δ=4", "fixed_l4", "fixed_4", "queue_eval_latency_4.json"),
    LocalProfile(
        "uniform_u2_6",
        "U(2,6)",
        "uniform_u2_6",
        "uniform_min_2_max_6",
        "queue_eval_latency_uniform_min_2_max_6.json",
    ),
)

FLAPPY_PROFILES = (
    LocalProfile("fixed_l2", "fixed δ=2", "fixed_l2", "fixed_2", "queue_eval_latency_2.json"),
    LocalProfile(
        "uniform_u1_3",
        "U(1,3)",
        "uniform_u1_3",
        "uniform_min_1_max_3",
        "queue_eval_latency_uniform_min_1_max_3.json",
    ),
)

LOCAL_ENVS = (
    LocalEnv("demon-attack", "Demon Attack", "demon_attack", 60, tuple(range(4, 9)), DEMON_PROFILES),
    LocalEnv("flappy", "Flappy Bird", "flappy", 30, tuple(range(4, 9)), FLAPPY_PROFILES),
    LocalEnv("deadly-corridor", "Deadly Corridor", "deadly_corridor", 35, (), DEMON_PROFILES),
)

LOCAL_VARIANTS = {
    "demon-attack": (
        LocalVariant("capped", "capped", 3600, tuple(range(4, 9))),
        LocalVariant("uncapped", "extended", 36000, tuple(range(4, 13))),
    ),
    "flappy": (
        LocalVariant("capped", "capped", 3600, tuple(range(4, 9))),
        LocalVariant("uncapped", "extended", 36000, tuple(range(1, 9))),
    ),
    "deadly-corridor": (
        LocalVariant("capped", None, 30000, (1, 2, 4, 6, 8, 10, 12, 14, 16)),
        LocalVariant("uncapped", None, 30000, (1, 2, 4, 6, 8, 10, 12, 14, 16)),
    ),
}


def local_result_path(
    results_root: Path,
    env: LocalEnv,
    profile: LocalProfile,
    context_window: int,
    seed: int,
    output_dir: str | None = "extended",
) -> Path:
    path = results_root / env.source_dir
    if output_dir is not None:
        path /= output_dir
    return (
        path
        / "latency_bench"
        / "frame_stack"
        / profile.setting_dir
        / f"fs{context_window}"
        / f"seed_{seed}"
        / profile.suite_dir
        / profile.filename
    )




def read_fresh_local_rows(results_root: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for env in LOCAL_ENVS:
        for variant in LOCAL_VARIANTS[env.env_id]:
            for context_window in variant.context_windows:
                for profile in env.profiles:
                    for seed in SEEDS:
                        path = local_result_path(
                            results_root,
                            env,
                            profile,
                            context_window,
                            seed,
                            variant.output_dir,
                        )
                        record = json.loads(path.read_text(encoding="utf-8"))
                        for episode_id, value in enumerate(record["returns"], start=1):
                            rows.append(
                                {
                                    "record_type": "episode_return",
                                    "horizon_variant": variant.name,
                                    "env_id": env.env_id,
                                    "env_title": env.title,
                                    "env_fps": env.fps,
                                    "context_window": context_window,
                                    "latency_regime": profile.regime,
                                    "latency_label": profile.label,
                                    "seed": seed,
                                    "episode_id": episode_id,
                                    "value": value,
                                    "value_unit": "episode_return",
                                    "eval_max_steps": variant.eval_max_steps,
                                    "eval_episode_count": len(record["returns"]),
                                    "source_kind": "local_json",
                                    "source_id": path.relative_to(results_root).as_posix(),
                                    "source_step": "",
                                }
                            )
    return rows


def write_rows(rows: Sequence[dict[str, object]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate completed local evaluations")
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = read_fresh_local_rows(args.results_root)
    write_rows(rows, args.output)
    print(f"Wrote {args.output} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
