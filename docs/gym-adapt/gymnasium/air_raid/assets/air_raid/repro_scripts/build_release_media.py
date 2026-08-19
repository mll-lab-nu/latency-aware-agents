"""Build AirRaid release figures from synchronized experiment evidence."""

import json
import re
from pathlib import Path

import cv2
import gymnasium as gym
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

import latency_bench.envs.gymnasium_air_raid  # registers the task wrapper
from latency_bench.core.config import load_config


ROOT = Path("artifacts/air_raid")
TRAIN_STDOUT = ROOT / "server/train_stdout.txt"
VIDEO = Path("outputs/examples/gymnasium_air_raid/eval/videos/bin_00_episode_000006.mp4")
EVAL_METRICS = Path("outputs/examples/gymnasium_air_raid/eval/episode_metrics.jsonl")


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
    ax.set(xlabel="Policy decisions", ylabel="Mean episode return", title="AirRaid PAL raw-frame environment · fixed L1 latency")
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
    ax.axhline(mean_return, color="#b46a2a", linestyle="--", linewidth=2, label=f"mean {mean_return:.0f}")
    ax.set(xlabel="Evaluation episode", ylabel="Episode return", title="AirRaid checkpoint evaluation · fixed L1 latency")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(ROOT / "eval_returns.png", dpi=180)
    plt.close(fig)


def render_task_media() -> None:
    config = load_config("configs/examples/gymnasium/air_raid/small_model_train.yaml")
    env = gym.make(config["env"]["env_id"], **config["env"]["make_kwargs"])
    env.reset(seed=0)
    Image.fromarray(env.render()).save(ROOT / "reset.png")
    frames = []
    env.reset(seed=7)
    for action_id in ([0, 1, 2, 3, 4, 5] * 10):
        _, _, terminated, truncated, _ = env.step(action_id)
        frames.append(Image.fromarray(env.render()))
        if terminated or truncated:
            break
    frames[0].save(ROOT / "action_catalog_trace.gif", save_all=True, append_images=frames[1:], duration=80, loop=0)
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
            selected.append((index, Image.fromarray(rgb).resize((192, 300))))
    capture.release()
    sheet = Image.new("RGB", (768, 640), "#eef2f3")
    draw = ImageDraw.Draw(sheet)
    for i, (index, frame) in enumerate(selected):
        x = (i % 4) * 192
        y = (i // 4) * 320
        sheet.paste(frame, (x, y))
        draw.text((x + 6, y + 301), f"decision={index} · sim={index / 12.5:.1f}s", fill="#17384b")
    sheet.save(ROOT / "eval_episode_contact_sheet.png")


def main() -> None:
    plot_training_curve()
    plot_eval_returns()
    render_task_media()
    build_contact_sheet()


if __name__ == "__main__":
    main()
