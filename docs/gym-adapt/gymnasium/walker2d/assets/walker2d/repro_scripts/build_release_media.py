from __future__ import annotations

import json
import re
import subprocess
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw


ASSET_ROOT = Path(__file__).resolve().parents[1]


def plot_training_curve() -> None:
    text = (ASSET_ROOT / "continuous_small_model" / "sf_log.txt").read_text()
    steps = [int(value) for value in re.findall(r"Total num frames: (\d+)", text)]
    rewards = [float(value) for value in re.findall(r"Avg episode reward: \[\(0, '([^']+)'\)\]", text)]
    count = min(len(steps), len(rewards))
    fig, ax = plt.subplots(figsize=(12, 6.75), dpi=120)
    ax.plot(steps[:count], rewards[:count], color="#18794e", linewidth=2.6)
    ax.set(title="Walker2D continuous-action zero-latency APPO training", xlabel="Environment frames", ylabel="Mean episode return")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(ASSET_ROOT / "training_curve.png")
    plt.close(fig)


def plot_latency_sweep() -> None:
    summaries = [
        json.loads((ASSET_ROOT / "continuous_small_model_eval" / f"queue_eval_latency_{latency}.json").read_text())
        for latency in range(5)
    ]
    labels = [f"L{item['eval_latency_steps']}\n{item['configured_latency_ms']:.0f} ms" for item in summaries]
    means = [item["mean_return"] for item in summaries]
    stds = [item["std_return"] for item in summaries]
    fig, ax = plt.subplots(figsize=(12, 6.75), dpi=120)
    ax.errorbar(labels, means, yerr=stds, fmt="o-", color="#18794e", ecolor="#93a69a", capsize=6, linewidth=2.6, markersize=8)
    ax.set(title="Walker2D continuous policy under action latency", xlabel="Evaluation latency", ylabel="Return (mean ± std, 10 episodes)")
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(ASSET_ROOT / "eval_latency_sweep.png")
    plt.close(fig)


def plot_vla_l0_eval() -> None:
    small_rows = [
        json.loads(line)
        for line in (
            ASSET_ROOT
            / "continuous_small_model_eval"
            / "eval_latency_0"
            / "episode_metrics.jsonl"
        ).read_text().splitlines()
    ]
    vla_rows = [
        json.loads(line)
        for line in (
            ASSET_ROOT
            / "vla_eval_continuous_projector"
            / "eval_latency_0"
            / "episode_metrics.jsonl"
        ).read_text().splitlines()
    ]
    groups = [small_rows, vla_rows]
    labels = ["Small model\n10 episodes", "QwenVLA\n20 episodes"]
    fig, (return_ax, length_ax) = plt.subplots(1, 2, figsize=(12, 6.75), dpi=120)
    for ax, field, title, ylabel in (
        (return_ax, "episode_return", "Closed-loop return", "Episode return"),
        (length_ax, "survival_steps", "Episode survival", "Environment steps"),
    ):
        values = [[row[field] for row in group] for group in groups]
        boxes = ax.boxplot(
            values,
            labels=labels,
            patch_artist=True,
            widths=0.5,
            showfliers=False,
        )
        for patch, color in zip(boxes["boxes"], ["#93a69a", "#b46a2a"]):
            patch.set(facecolor=color, edgecolor="#182018", alpha=0.78)
        for index, points in enumerate(values, start=1):
            offsets = np.linspace(-0.12, 0.12, len(points))
            ax.scatter(index + offsets, points, color="#182018", s=18, alpha=0.72, zorder=3)
            ax.scatter(
                index,
                np.mean(points),
                marker="D",
                color="white",
                edgecolor="#182018",
                s=58,
                zorder=4,
            )
        ax.set(title=title, ylabel=ylabel)
        ax.grid(axis="y", alpha=0.2)
    length_ax.axhline(
        1000,
        color="#182018",
        linestyle="--",
        linewidth=1.5,
        label="1,000-step horizon",
    )
    length_ax.set_ylim(0, 1050)
    length_ax.legend(frameon=False)
    fig.suptitle("Walker2D · post-fix zero-latency closed-loop evaluation")
    fig.tight_layout()
    fig.savefig(ASSET_ROOT / "vla_l0_eval.png")
    plt.close(fig)


def build_contact_sheet() -> None:
    manifest_path = ASSET_ROOT / "continuous_small_model_eval" / "eval_latency_0" / "videos" / "manifest.jsonl"
    manifest = json.loads(manifest_path.read_text().splitlines()[0])
    video = manifest_path.parent / manifest["video_path"]
    with tempfile.TemporaryDirectory() as tmp_dir:
        pattern = str(Path(tmp_dir) / "frame_%02d.png")
        subprocess.run(
            ["ffmpeg", "-loglevel", "error", "-i", str(video), "-vf", "fps=8/5.9,scale=360:360", "-frames:v", "8", pattern],
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
        sheet.save(ASSET_ROOT / "eval_episode_contact_sheet.png")


if __name__ == "__main__":
    plot_training_curve()
    plot_latency_sweep()
    plot_vla_l0_eval()
    build_contact_sheet()
