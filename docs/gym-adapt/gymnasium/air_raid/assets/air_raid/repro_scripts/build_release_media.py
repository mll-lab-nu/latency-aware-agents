"""Build AirRaid release figures from synchronized experiment evidence."""

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
TRAIN_STDOUT = ROOT / "training_stdout.txt"
EVAL_METRICS = ROOT / "small_model_eval_random_noop" / "episode_metrics.jsonl"


def strip_ansi(text: str) -> str:
    return re.sub(r"\x1b\[[0-9;]*m", "", text)


def plot_training_curve() -> None:
    frames: list[int] = []
    rewards: list[float] = []
    current_frame = 0
    for line in strip_ansi(TRAIN_STDOUT.read_text()).splitlines():
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
        xlabel="Raw environment frames",
        ylabel="Mean episode return",
        title="AirRaid · zero-latency Atari APPO · 10M frames",
    )
    fig.tight_layout()
    fig.savefig(ROOT / "training_curve.png", dpi=180)
    plt.close(fig)


def plot_eval_returns() -> None:
    rows = [json.loads(line) for line in EVAL_METRICS.read_text().splitlines()]
    episode_ids = [row["episode_id"] for row in rows]
    returns = [row["episode_return"] for row in rows]
    mean_return = sum(returns) / len(returns)

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(episode_ids, returns, color="#476f87")
    ax.axhline(
        mean_return,
        color="#b46a2a",
        linestyle="--",
        linewidth=2,
        label=f"mean {mean_return:.0f}",
    )
    ax.set(
        xlabel="Evaluation episode",
        ylabel="Episode return",
        title="AirRaid · L0 random-noop + FireReset evaluation",
    )
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(ROOT / "eval_returns.png", dpi=180)
    plt.close(fig)


def main() -> None:
    plot_training_curve()
    plot_eval_returns()


if __name__ == "__main__":
    main()
