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
    fig, ax = plt.subplots(figsize=(12, 6.75), dpi=120)
    ax.plot(steps[:count], rewards[:count], color="#18794e", linewidth=2.6)
    ax.set(title="Swimmer zero-latency APPO training", xlabel="Environment frames", ylabel="Mean episode return")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(ROOT / "training_curve.png")
    plt.close(fig)


def plot_latency_sweep() -> None:
    summaries = [
        json.loads((ROOT / "small_model_eval" / f"queue_eval_latency_{latency}.json").read_text())
        for latency in range(5)
    ]
    labels = [f"L{item['eval_latency_steps']}\n{item['configured_latency_ms']:.0f} ms" for item in summaries]
    fig, (return_ax, length_ax) = plt.subplots(1, 2, figsize=(12, 6.75), dpi=120)
    return_ax.errorbar(
        labels,
        [item["mean_return"] for item in summaries],
        yerr=[item["std_return"] for item in summaries],
        fmt="o-",
        color="#18794e",
        ecolor="#93a69a",
        capsize=6,
        linewidth=2.6,
        markersize=8,
    )
    return_ax.set(title="Return", ylabel="Mean ± std over 10 episodes")
    return_ax.grid(axis="y", alpha=0.2)
    length_ax.plot(labels, [item["mean_length"] for item in summaries], "o-", color="#476f87", linewidth=2.6)
    length_ax.set_ylim(0, 1050)
    length_ax.set(title="Episode length", ylabel="Steps")
    length_ax.grid(axis="y", alpha=0.2)
    fig.suptitle("Swimmer policy under fixed action latency")
    fig.tight_layout()
    fig.savefig(ROOT / "eval_latency_sweep.png")
    plt.close(fig)


def plot_l0_returns() -> None:
    summary = json.loads((ROOT / "small_model_eval" / "queue_eval_latency_0.json").read_text())
    fig, ax = plt.subplots(figsize=(12, 6.75), dpi=120)
    ax.bar(range(10), summary["returns"], color="#476f87")
    ax.axhline(
        summary["mean_return"],
        color="#b46a2a",
        linestyle="--",
        linewidth=2,
        label=f"mean {summary['mean_return']:.3f}",
    )
    ax.set(title="Swimmer best-checkpoint evaluation · L0", xlabel="Episode", ylabel="Return")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(ROOT / "eval_l0_returns.png")
    plt.close(fig)


def build_contact_sheet() -> None:
    manifest_path = ROOT / "small_model_eval" / "eval_latency_0" / "videos" / "manifest.jsonl"
    manifest = json.loads(manifest_path.read_text().splitlines()[0])
    video = manifest_path.parent / manifest["video_path"]
    with tempfile.TemporaryDirectory() as tmp_dir:
        pattern = str(Path(tmp_dir) / "frame_%02d.png")
        subprocess.run(
            ["ffmpeg", "-loglevel", "error", "-i", str(video), "-vf", "fps=8/40,scale=360:360", "-frames:v", "8", pattern],
            check=True,
        )
        frames = [Image.open(path).convert("RGB") for path in sorted(Path(tmp_dir).glob("frame_*.png"))]
        sheet = Image.new("RGB", (1440, 760), "#f4f2ea")
        draw = ImageDraw.Draw(sheet)
        for index, frame in enumerate(frames):
            x = (index % 4) * 360
            y = (index // 4) * 380
            sheet.paste(frame, (x, y))
            draw.rectangle((x, y + 350, x + 360, y + 380), fill="#182018")
            draw.text((x + 12, y + 357), f"frame {index + 1} / 8", fill="white")
        sheet.save(ROOT / "eval_episode_contact_sheet.png")


if __name__ == "__main__":
    plot_training_curve()
    plot_latency_sweep()
    plot_l0_returns()
    build_contact_sheet()
