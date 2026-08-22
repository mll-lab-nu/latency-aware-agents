from __future__ import annotations

import json
import re
import subprocess
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]


def plot_training_curve() -> None:
    text = (ROOT / "small_model" / "sf_log.txt").read_text()
    steps = [int(value) for value in re.findall(r"Total num frames: (\d+)", text)]
    rewards = [float(value) for value in re.findall(r"Avg episode reward: \[\(0, '([^']+)'\)\]", text)]
    count = min(len(steps), len(rewards))
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=180)
    ax.plot(steps[:count], rewards[:count], color="#147a80", linewidth=2.2)
    ax.set(title="Hopper continuous APPO · zero latency", xlabel="Environment frames", ylabel="Mean episode return")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(ROOT / "training_curve.png")
    plt.close(fig)


def plot_eval_returns() -> None:
    rows = [json.loads(line) for line in (ROOT / "small_model_eval" / "episode_metrics.jsonl").read_text().splitlines()]
    returns = [row["episode_return"] for row in rows]
    mean_return = sum(returns) / len(returns)
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=180)
    ax.bar(range(len(returns)), returns, color="#476f87")
    ax.axhline(mean_return, color="#b46a2a", linestyle="--", linewidth=2, label=f"mean {mean_return:.1f}")
    ax.set(title="Hopper continuous best-checkpoint evaluation · L0", xlabel="Episode", ylabel="Return")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(ROOT / "eval_returns.png")
    plt.close(fig)


def build_contact_sheet() -> None:
    manifest_path = ROOT / "small_model_eval" / "videos" / "manifest.jsonl"
    manifest = json.loads(manifest_path.read_text().splitlines()[0])
    video = manifest_path.parent / manifest["video_path"]
    with tempfile.TemporaryDirectory() as tmp_dir:
        pattern = str(Path(tmp_dir) / "frame_%02d.png")
        subprocess.run(
            ["ffmpeg", "-loglevel", "error", "-i", str(video), "-vf", "fps=1,scale=240:240", "-frames:v", "8", pattern],
            check=True,
        )
        frames = [Image.open(path).convert("RGB") for path in sorted(Path(tmp_dir).glob("frame_*.png"))]
        sheet = Image.new("RGB", (960, 520), "#eef2f3")
        draw = ImageDraw.Draw(sheet)
        for index, frame in enumerate(frames):
            x = (index % 4) * 240
            y = (index // 4) * 260
            sheet.paste(frame, (x, y))
            draw.text((x + 8, y + 241), f"t={index}s", fill="#17384b")
        sheet.save(ROOT / "eval_episode_contact_sheet.png")


if __name__ == "__main__":
    plot_training_curve()
    plot_eval_returns()
    build_contact_sheet()
