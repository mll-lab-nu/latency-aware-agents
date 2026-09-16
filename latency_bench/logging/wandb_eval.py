from __future__ import annotations

from typing import Any


EVAL_SCALAR_METRICS = ("mean_return", "std_return", "mean_length")


def eval_scalar_payload(
    summaries: list[dict[str, Any]],
    *,
    env_steps: int,
    prefix: str,
) -> dict[str, float | int]:
    payload: dict[str, float | int] = {
        "global_step": int(env_steps),
        f"{prefix}/global_steps": int(env_steps),
    }
    for summary in summaries:
        metric_prefix = f"{prefix}/{summary_metric_group(summary)}/{summary_series_name(summary)}"
        for metric in EVAL_SCALAR_METRICS:
            payload[f"{metric_prefix}/{metric}"] = float(summary[metric])
        payload[f"{metric_prefix}/completed_episodes"] = float(len(summary["returns"]))
        if "task_metrics" in summary:
            for name, task_metric in summary["task_metrics"].items():
                task_metric_prefix = f"{metric_prefix}/task_metrics/{name}"
                payload[f"{task_metric_prefix}/mean"] = float(task_metric["mean"])
                payload[f"{task_metric_prefix}/std"] = float(task_metric["std"])
    return payload


def summary_metric_group(summary: dict[str, Any]) -> str:
    latency_type = str(summary.get("latency_type", ""))
    if latency_type == "fixed_sweep":
        return "fixed"
    if latency_type == "normal_distribution":
        return "normal"
    if latency_type == "uniform_distribution":
        return "uniform"
    suite_name = str(summary["suite_name"])
    if suite_name.startswith("fixed_"):
        return "fixed"
    if suite_name.startswith("normal_"):
        return "normal"
    if suite_name.startswith("uniform_"):
        return "uniform"
    return suite_name


def summary_series_name(summary: dict[str, Any]) -> str:
    return latency_fragment(summary["latency"])


def latency_fragment(latency: int | str) -> str:
    if isinstance(latency, int) and not isinstance(latency, bool):
        return f"latency_{int(latency)}"
    return "".join(
        char if char.isalnum() or char in {"-", "_", "."} else "_"
        for char in str(latency)
    ).strip("._-")


def normalize_wandb_tags(value: Any) -> list[str] | None:
    if value is None:
        return None
    if isinstance(value, str):
        tags = [item.strip() for item in value.split(",") if item.strip()]
        return tags or None
    tags = [
        tag
        for item in value
        for tag in (part.strip() for part in str(item).split(","))
        if tag
    ]
    return tags or None
