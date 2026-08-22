"""Build Hopper release figures from synchronized experiment evidence."""

import json
import re
from pathlib import Path

import cv2
import gymnasium as gym
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

import latency_bench.envs.gymnasium_hopper  # registers the task wrapper
from latency_bench.core.config import load_config


ROOT = Path(__file__).resolve().parents[1]
TRAIN_STDOUT = ROOT / "small_model" / "sf_log.txt"
EVAL_ROOT = ROOT / "small_model_eval"
VIDEO_MANIFEST = EVAL_ROOT / "videos" / "manifest.jsonl"
EVAL_METRICS = EVAL_ROOT / "episode_metrics.jsonl"


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
    ax.set(xlabel="Environment frames", ylabel="Mean episode return", title="Hopper ternary torque catalog · zero latency")
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
    ax.set(xlabel="Evaluation episode", ylabel="Episode return", title="Hopper checkpoint evaluation · zero latency")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(ROOT / "eval_returns.png", dpi=180)
    plt.close(fig)


def render_task_media() -> None:
    config = load_config("configs/examples/gymnasium/hopper/small_model_train.yaml")
    env = gym.make(config["env"]["env_id"], **config["env"]["make_kwargs"])
    env.reset(seed=0)
    Image.fromarray(env.render()).save(ROOT / "reset.png")
    frames = []
    env.reset(seed=7)
    for action_id in [13] * 10 + [26] * 10 + [0] * 10 + [23] * 10:
        _, _, terminated, truncated, _ = env.step(action_id)
        frames.append(Image.fromarray(env.render()))
        if terminated or truncated:
            break
    frames[0].save(ROOT / "torque_catalog_trace.gif", save_all=True, append_images=frames[1:], duration=40, loop=0)
    env.close()


def build_contact_sheet() -> None:
    manifest = json.loads(VIDEO_MANIFEST.read_text().splitlines()[0])
    video = VIDEO_MANIFEST.parent / manifest["video_path"]
    capture = cv2.VideoCapture(str(video))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = capture.get(cv2.CAP_PROP_FPS)
    selected = []
    for index in [round(i * (frame_count - 1) / 7) for i in range(8)]:
        capture.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = capture.read()
        if ok:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            selected.append((index, Image.fromarray(rgb).resize((240, 240))))
    capture.release()
    sheet = Image.new("RGB", (960, 520), "#eef2f3")
    draw = ImageDraw.Draw(sheet)
    for i, (index, frame) in enumerate(selected):
        x = (i % 4) * 240
        y = (i // 4) * 260
        sheet.paste(frame, (x, y))
        draw.text((x + 8, y + 241), f"t={index / fps:.1f}s", fill="#17384b")
    sheet.save(ROOT / "eval_episode_contact_sheet.png")


def main() -> None:
    plot_training_curve()
    plot_eval_returns()
    build_contact_sheet()


if __name__ == "__main__":
    main()
