from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


HEADER = [
    "game",
    "model",
    "train_latency",
    "eval_frames_advance",
    "mean_score",
    "mean_steps",
    "seed",
    "source_file",
    "raw_return_mean",
    "return_mean",
    "raw_frames_mean",
    "episode_length_mean",
]

EVAL_FILE_RE = re.compile(r"^queue_eval_latency_(-?\d+)\.json$")
LATENCY_DIR_RE = re.compile(r"^(?:train_)?latency_(-?\d+)$")
SEED_DIR_RE = re.compile(r"^seed_(-?\d+)$")

EPISODE_LIST_KEYS = ("episodes", "records", "results", "metrics", "episode_metrics")


def warn(message: str) -> None:
    print(f"warning: {message}", file=sys.stderr)


def is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def first_number(mapping: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        value = mapping.get(key)
        if is_number(value):
            return value
    return None


def mean(values: list[Any]) -> float | None:
    numeric = [float(value) for value in values if is_number(value)]
    if not numeric:
        return None
    return sum(numeric) / len(numeric)


def mean_from_records(records: list[dict[str, Any]], keys: tuple[str, ...]) -> float | None:
    values = []
    for record in records:
        value = first_number(record, keys)
        if value is not None:
            values.append(value)
    return mean(values)


def find_episode_records(payload: Any) -> list[dict[str, Any]] | None:
    if isinstance(payload, list):
        records = [item for item in payload if isinstance(item, dict)]
        return records if records else None

    if not isinstance(payload, dict):
        return None

    for key in EPISODE_LIST_KEYS:
        value = payload.get(key)
        if isinstance(value, list):
            records = [item for item in value if isinstance(item, dict)]
            if records:
                return records
    return None


def extract_metrics(payload: Any) -> dict[str, Any]:
    records = find_episode_records(payload)
    if records is not None:
        metrics = {
            "mean_score": mean_from_records(records, ("score", "return", "raw_return")),
            "mean_steps": mean_from_records(records, ("steps", "length", "raw_frames")),
            "raw_return_mean": mean_from_records(records, ("raw_return",)),
            "return_mean": mean_from_records(records, ("return",)),
            "raw_frames_mean": mean_from_records(records, ("raw_frames",)),
            "episode_length_mean": mean_from_records(records, ("length", "episode_length", "steps")),
        }
        if metrics["mean_score"] is None:
            metrics["mean_score"] = metrics["return_mean"] or metrics["raw_return_mean"]
        if metrics["mean_steps"] is None:
            metrics["mean_steps"] = metrics["episode_length_mean"] or metrics["raw_frames_mean"]
        return metrics

    if not isinstance(payload, dict):
        return {key: None for key in HEADER if key not in {"game", "model", "train_latency", "eval_frames_advance", "seed", "source_file"}}

    metrics = {
        "mean_score": first_number(payload, ("mean_score", "score_mean")),
        "mean_steps": first_number(payload, ("mean_steps", "steps_mean")),
        "raw_return_mean": first_number(payload, ("raw_return_mean", "mean_raw_return")),
        "return_mean": first_number(payload, ("return_mean", "mean_return")),
        "raw_frames_mean": first_number(payload, ("raw_frames_mean", "mean_raw_frames", "frames_mean")),
        "episode_length_mean": first_number(
            payload,
            ("episode_length_mean", "mean_episode_length", "length_mean", "mean_length"),
        ),
    }

    if metrics["return_mean"] is None:
        metrics["return_mean"] = mean(payload.get("returns", []))
    if metrics["episode_length_mean"] is None:
        metrics["episode_length_mean"] = mean(payload.get("lengths", []))
    if metrics["raw_return_mean"] is None:
        metrics["raw_return_mean"] = mean(payload.get("raw_returns", []))
    if metrics["raw_frames_mean"] is None:
        metrics["raw_frames_mean"] = mean(payload.get("raw_frames", []))

    if metrics["mean_score"] is None:
        metrics["mean_score"] = metrics["return_mean"] if metrics["return_mean"] is not None else metrics["raw_return_mean"]
    if metrics["mean_steps"] is None:
        metrics["mean_steps"] = (
            metrics["episode_length_mean"] if metrics["episode_length_mean"] is not None else metrics["raw_frames_mean"]
        )
    return metrics


def parse_context(root: Path, path: Path) -> dict[str, str] | None:
    match = EVAL_FILE_RE.match(path.name)
    if match is None:
        return None

    try:
        relative = path.relative_to(root)
    except ValueError:
        relative = path

    parts = relative.parts
    latency_index = None
    train_latency = None
    for index, part in enumerate(parts[:-1]):
        latency_match = LATENCY_DIR_RE.match(part)
        if latency_match is not None:
            latency_index = index
            train_latency = latency_match.group(1)

    seed = None
    for part in parts[:-1]:
        seed_match = SEED_DIR_RE.match(part)
        if seed_match is not None:
            seed = seed_match.group(1)

    if latency_index is None or train_latency is None or seed is None or latency_index == 0:
        warn(f"skipping {relative.as_posix()}: could not parse game/train latency/seed from path")
        return None

    return {
        "game": parts[latency_index - 1],
        "train_latency": train_latency,
        "eval_frames_advance": match.group(1),
        "seed": seed,
        "source_file": relative.as_posix(),
    }


def format_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and not math.isfinite(value):
        return ""
    return str(value)


def build_row(root: Path, path: Path) -> dict[str, str] | None:
    context = parse_context(root, path)
    if context is None:
        return None

    try:
        with path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
    except json.JSONDecodeError as exc:
        warn(f"skipping {context['source_file']}: invalid JSON ({exc})")
        return None
    except OSError as exc:
        warn(f"skipping {context['source_file']}: could not read file ({exc})")
        return None

    metrics = extract_metrics(payload)
    train_latency = context["train_latency"]
    row = {
        "game": context["game"],
        "model": f"Sample Factory PPO latency={train_latency}",
        "train_latency": train_latency,
        "eval_frames_advance": context["eval_frames_advance"],
        "mean_score": format_value(metrics.get("mean_score")),
        "mean_steps": format_value(metrics.get("mean_steps")),
        "seed": context["seed"],
        "source_file": context["source_file"],
        "raw_return_mean": format_value(metrics.get("raw_return_mean")),
        "return_mean": format_value(metrics.get("return_mean")),
        "raw_frames_mean": format_value(metrics.get("raw_frames_mean")),
        "episode_length_mean": format_value(metrics.get("episode_length_mean")),
    }
    return row


def int_sort_key(value: str) -> tuple[int, int | str]:
    try:
        return (0, int(value))
    except ValueError:
        return (1, value)


def row_sort_key(row: dict[str, str]) -> tuple[Any, ...]:
    return (
        row["game"],
        int_sort_key(row["train_latency"]),
        int_sort_key(row["eval_frames_advance"]),
        int_sort_key(row["seed"]),
        row["source_file"],
    )


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=HEADER)
        writer.writeheader()
        writer.writerows(rows)


def per_game_path(out: Path, game: str) -> Path:
    return out.with_name(f"{out.stem}_{game}{out.suffix}")


def collect_rows(root: Path) -> list[dict[str, str]]:
    candidates = sorted(root.rglob("queue_eval_latency_*.json"))
    if not candidates:
        warn(f"No queue_eval_latency_*.json files found under {root}")
        return []

    rows = []
    for path in candidates:
        if path.name.endswith("_actions.json"):
            continue
        row = build_row(root, path)
        if row is not None:
            rows.append(row)

    rows.sort(key=row_sort_key)
    if not rows:
        warn(f"No valid queue eval result rows collected under {root}")
    return rows


def write_outputs(out: Path, rows: list[dict[str, str]], per_game: bool) -> None:
    write_csv(out, rows)
    if not per_game:
        return

    rows_by_game: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        rows_by_game[row["game"]].append(row)

    for game, game_rows in sorted(rows_by_game.items()):
        write_csv(per_game_path(out, game), game_rows)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect queue evaluation JSON files into CSV summaries.")
    parser.add_argument("--root", required=True, type=Path, help="Local root directory containing downloaded outputs.")
    parser.add_argument("--out", required=True, type=Path, help="Output CSV path for the combined table.")
    parser.add_argument("--no-per-game", action="store_true", help="Do not write per-game CSV files next to --out.")
    args = parser.parse_args(argv)

    if not args.root.is_dir():
        parser.error(f"--root must be an existing local directory: {args.root}")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = args.root.resolve()
    out = args.out
    rows = collect_rows(root)
    write_outputs(out, rows, per_game=not args.no_per_game)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
