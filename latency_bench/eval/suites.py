from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from latency_bench.core.latency_distribution import (
    LatencyDistributionSpec,
    distribution_frames_to_ms_latency_config,
    latency_distribution_label,
)


@dataclass(frozen=True)
class EvalSuite:
    name: str
    fixed_latency_values: list[int] | None
    latency_config: dict[str, Any] | None
    latency_label: int | str
    metadata: dict[str, Any]


def has_eval_suites(config: dict[str, Any]) -> bool:
    evaluation_cfg = config.get("evaluation", {})
    return isinstance(evaluation_cfg, dict) and "eval_suites" in evaluation_cfg


def parse_eval_suites(config: dict[str, Any]) -> list[EvalSuite]:
    evaluation_cfg = config.get("evaluation", {})
    suites_cfg = evaluation_cfg.get("eval_suites") or {}
    env_fps = float(config["env"]["env_fps"])
    seed = config["experiment"].get("seed")
    suites: list[EvalSuite] = []

    for item in suites_cfg.get("fixed", []) or []:
        values = [int(value) for value in item]
        if len(values) == 2:
            start, stop = values
            values = list(range(start, stop + 1))
            name = f"fixed_{start}_to_{stop}"
        else:
            name = "fixed_" + "_".join(str(value) for value in values)
        suites.append(
            EvalSuite(
                name=name,
                fixed_latency_values=values,
                latency_config=None,
                latency_label=name,
                metadata={"suite_name": name, "latency_type": "fixed_sweep"},
            )
        )

    for item in suites_cfg.get("normal", []) or []:
        mean, std, min_frame, max_frame = item
        spec = LatencyDistributionSpec(
            type="normal",
            mean=float(mean),
            std=float(std),
            min=int(min_frame),
            max=int(max_frame),
        )
        label = latency_distribution_label(spec)
        suites.append(
            EvalSuite(
                name=label,
                fixed_latency_values=None,
                latency_config=distribution_frames_to_ms_latency_config(spec, env_fps=env_fps, seed=seed),
                latency_label=label,
                metadata={"suite_name": label, "latency_type": "normal_distribution"},
            )
        )

    for item in suites_cfg.get("uniform", []) or []:
        min_frame, max_frame = item
        spec = LatencyDistributionSpec(
            type="uniform",
            mean=None,
            std=None,
            min=int(min_frame),
            max=int(max_frame),
        )
        label = latency_distribution_label(spec)
        suites.append(
            EvalSuite(
                name=label,
                fixed_latency_values=None,
                latency_config=distribution_frames_to_ms_latency_config(spec, env_fps=env_fps, seed=seed),
                latency_label=label,
                metadata={"suite_name": label, "latency_type": "uniform_distribution"},
            )
        )

    for item in suites_cfg.get("iid", []) or []:
        name = item["name"]
        profile_path = item["profile_path"]
        latency_seed = item["seed"]
        suites.append(
            EvalSuite(
                name=name,
                fixed_latency_values=None,
                latency_config={
                    "method": "iid",
                    "profile_path": profile_path,
                    "seed": latency_seed,
                },
                latency_label=name,
                metadata={
                    "suite_name": name,
                    "latency_type": "profile_sample",
                    "profile_path": profile_path,
                    "latency_seed": latency_seed,
                },
            )
        )

    return suites
