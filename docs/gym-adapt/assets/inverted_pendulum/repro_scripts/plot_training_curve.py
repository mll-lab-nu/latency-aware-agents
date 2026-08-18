"""Create a compact reward curve from Sample Factory's training stdout."""

import re
from pathlib import Path

import matplotlib.pyplot as plt


def main() -> None:
    text = Path("/workspace/gym-adapt-mujoco-logs/07_train_stdout.txt").read_text()
    frames = [int(value) for value in re.findall(r"Total num frames: (\d+)", text)]
    rewards = [float(value) for value in re.findall(r"Avg episode reward: \[\(0, '([0-9.]+)'\)\]", text)]
    count = min(len(frames), len(rewards))
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(frames[:count], rewards[:count], color="#1f5f99", linewidth=2)
    ax.set(xlabel="Environment frames", ylabel="Mean episode return", title="InvertedPendulum discrete catalog, L1 latency")
    fig.tight_layout()
    fig.savefig("artifacts/inverted_pendulum/training_curve.png", dpi=180)


if __name__ == "__main__":
    main()
