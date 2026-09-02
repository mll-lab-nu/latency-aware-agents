"""Build Humanoid result figures from published compact evidence."""

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]


def plot_training_curve() -> None:
    frames: list[int] = []
    rewards: list[float] = []
    current_frame = 0
    for line in (ROOT / "small_model" / "sf_log.txt").read_text().splitlines():
        frame_match = re.search(r"Total num frames: (\d+)", line)
        if frame_match:
            current_frame = int(frame_match.group(1))
        reward_match = re.search(r"Avg episode reward: \[\(0, '(-?[0-9.]+)'\)\]", line)
        if reward_match and current_frame:
            frames.append(current_frame)
            rewards.append(float(reward_match.group(1)))

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(frames, rewards, color="#147a80", linewidth=2)
    ax.set(
        xlabel="Environment frames",
        ylabel="Mean episode return",
        title="Humanoid · zero-latency APPO · 10M frames",
    )
    fig.tight_layout()
    fig.savefig(ROOT / "training_curve.png", dpi=180)
    plt.close(fig)


def plot_small_model_probe() -> None:
    result = json.loads((ROOT / "small_model_probe.json").read_text())
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    axes[0].bar(
        ["Mean", "Median"],
        [result["mean_return"], result["median_return"]],
        color=["#476f87", "#147a80"],
    )
    axes[0].axhline(
        result["return_gate"],
        color="#b46a2a",
        linestyle="--",
        linewidth=1.8,
        label="rollout gate 5,000",
    )
    axes[0].set(ylabel="Episode return", title="20-episode L0 probe")
    axes[0].legend(frameon=False)

    axes[1].bar(
        ["Return > 5,000", "Full 1,000 steps"],
        [result["episodes_above_gate"], result["full_episodes"]],
        color=["#147a80", "#476f87"],
    )
    axes[1].axhline(result["episodes"], color="#b46a2a", linestyle="--", linewidth=1.5)
    axes[1].set(ylim=(0, result["episodes"] + 1), ylabel="Episodes", title="Teacher quality gates")
    fig.suptitle("Humanoid · best small-model checkpoint")
    fig.tight_layout()
    fig.savefig(ROOT / "small_model_probe.png", dpi=180)
    plt.close(fig)


def plot_vla_eval() -> None:
    result = json.loads((ROOT / "vla_eval" / "step_5000.json").read_text())
    latencies = ["0", "2"]
    rows = [result["per_latency"][latency] for latency in latencies]
    labels = [f"L{latency}\n{row['latency_ms']} ms" for latency, row in zip(latencies, rows)]

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    axes[0].bar(
        labels,
        [row["mean_return"] for row in rows],
        yerr=[row["std_return"] for row in rows],
        capsize=6,
        color="#476f87",
    )
    axes[0].set(ylabel="Mean return ± std", title="Final-checkpoint return")
    axes[1].bar(labels, [row["mean_length"] for row in rows], color="#147a80")
    axes[1].axhline(result["max_steps"], color="#b46a2a", linestyle="--", linewidth=1.5)
    axes[1].set(ylim=(0, result["max_steps"] * 1.05), ylabel="Mean environment steps", title="Episode survival")
    fig.suptitle("Humanoid · 5k QwenVLA closed-loop evaluation")
    fig.tight_layout()
    fig.savefig(ROOT / "vla_eval.png", dpi=180)
    plt.close(fig)


def main() -> None:
    plot_training_curve()
    plot_small_model_probe()
    plot_vla_eval()


if __name__ == "__main__":
    main()
