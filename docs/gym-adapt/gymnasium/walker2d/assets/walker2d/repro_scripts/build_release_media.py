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


def plot_vla_validation_loss() -> None:
    rows = json.loads((ASSET_ROOT / "vla_loss_anchors.json").read_text())
    steps = [row["step"] for row in rows]
    fig, ax = plt.subplots(figsize=(12, 6.75), dpi=120)
    ax.plot(
        steps,
        [row["text_bin_validation_loss"] for row in rows],
        color="#476f87",
        marker="o",
        linewidth=2.5,
        label="256-bin text state",
    )
    ax.plot(
        steps,
        [row["continuous_validation_loss"] for row in rows],
        color="#b46a2a",
        marker="s",
        linestyle="--",
        linewidth=2.5,
        label="Continuous projector",
    )
    ax.set(
        title="Walker2D QwenOFT held-out action loss",
        xlabel="Training step",
        ylabel="Held-out L1 · 3,200 samples",
    )
    ax.legend(frameon=False)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(ASSET_ROOT / "vla_validation_loss_comparison.png")
    plt.close(fig)


def plot_vla_state_encoding_comparison() -> None:
    text_bins = [
        json.loads((ASSET_ROOT / "vla_eval_text_bins" / f"queue_eval_latency_{latency}.json").read_text())
        for latency in (0, 2)
    ]
    continuous = [
        json.loads(
            (ASSET_ROOT / "vla_eval_continuous_projector" / f"queue_eval_latency_{latency}.json").read_text()
        )
        for latency in (0, 2)
    ]
    labels = ["L0 · 0 ms", "L2 · 16 ms"]
    x = np.arange(len(labels))
    width = 0.34
    fig, (return_ax, length_ax) = plt.subplots(1, 2, figsize=(12, 6.75), dpi=120)
    for ax, field, title, ylabel in (
        (return_ax, "mean_return", "Closed-loop return", "Mean return · 20 episodes"),
        (length_ax, "mean_length", "Episode length", "Mean environment steps · 20 episodes"),
    ):
        text_values = [row[field] for row in text_bins]
        continuous_values = [row[field] for row in continuous]
        text_bars = ax.bar(
            x - width / 2,
            text_values,
            width,
            color="#93a69a",
            edgecolor="#182018",
            label="256-bin text state",
        )
        continuous_bars = ax.bar(
            x + width / 2,
            continuous_values,
            width,
            color="#b46a2a",
            edgecolor="#182018",
            hatch="//",
            label="Continuous projector",
        )
        ax.bar_label(text_bars, labels=[f"{value:.1f}" for value in text_values], padding=4)
        ax.bar_label(continuous_bars, labels=[f"{value:.1f}" for value in continuous_values], padding=4)
        ax.set_xticks(x, labels)
        ax.set(title=title, ylabel=ylabel)
        ax.grid(axis="y", alpha=0.2)
    return_ax.legend(frameon=False)
    fig.suptitle("Walker2D QwenOFT state encoding comparison · seed 0 and final 5k checkpoints")
    fig.tight_layout()
    fig.savefig(ASSET_ROOT / "vla_state_encoding_comparison.png")
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
    plot_vla_validation_loss()
    plot_vla_state_encoding_comparison()
    build_contact_sheet()
