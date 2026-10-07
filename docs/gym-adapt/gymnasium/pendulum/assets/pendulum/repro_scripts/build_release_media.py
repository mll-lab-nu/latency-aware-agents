"""Build Pendulum release figures from synchronized experiment evidence."""

import json
import re
from pathlib import Path

import cv2
import gymnasium as gym
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

import latency_bench.envs.tasks.pendulum  # registers the task wrapper


ROOT = Path("artifacts/pendulum")
TRAIN_STDOUT = ROOT / "server/train_stdout.txt"
VIDEO = Path("outputs/examples/gymnasium_pendulum/eval/videos/bin_00_episode_000006.mp4")
EVAL_METRICS = Path("outputs/examples/gymnasium_pendulum/eval/episode_metrics.jsonl")
MAKE_KWARGS = {
    "base_env_id": "Pendulum-v1",
    "render_mode": "rgb_array",
    "torque_values": [-2.0, -1.0, 0.0, 1.0, 2.0],
    "base_make_kwargs": {"g": 10.0},
}


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
        xlabel="Environment frames",
        ylabel="Mean episode return",
        title="Pendulum discrete torque catalog · fixed L1 latency",
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
    ax.axhline(mean_return, color="#b46a2a", linestyle="--", linewidth=2, label=f"mean {mean_return:.1f}")
    ax.set(
        xlabel="Evaluation episode",
        ylabel="Episode return",
        title="Pendulum checkpoint evaluation · fixed L1 latency",
    )
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(ROOT / "eval_returns.png", dpi=180)
    plt.close(fig)


def render_task_media() -> None:
    env = gym.make("LatencyBench/PendulumDiscrete-v0", **MAKE_KWARGS)
    env.reset(seed=0)
    Image.fromarray(env.render()).save(ROOT / "reset.png")
    frames = []
    env.reset(seed=7)
    for action_id in ([4] * 20 + [0] * 20 + [2] * 10) * 2:
        env.step(action_id)
        frames.append(Image.fromarray(env.render()))
    frames[0].save(
        ROOT / "torque_catalog_trace.gif",
        save_all=True,
        append_images=frames[1:],
        duration=50,
        loop=0,
    )
    env.close()


def build_contact_sheet() -> None:
    capture = cv2.VideoCapture(str(VIDEO))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    selected = []
    for index in [round(i * (frame_count - 1) / 7) for i in range(8)]:
        capture.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = capture.read()
        if ok:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            selected.append(Image.fromarray(rgb).resize((250, 250)))
    capture.release()
    sheet = Image.new("RGB", (1000, 540), "#eef2f3")
    draw = ImageDraw.Draw(sheet)
    for i, frame in enumerate(selected):
        x = (i % 4) * 250
        y = (i // 4) * 270
        sheet.paste(frame, (x, y))
        draw.text((x + 8, y + 250), f"t={i * 199 / 7 / 20:.1f}s", fill="#17384b")
    sheet.save(ROOT / "eval_episode_contact_sheet.png")


def main() -> None:
    plot_training_curve()
    plot_eval_returns()
    render_task_media()
    build_contact_sheet()


if __name__ == "__main__":
    main()
