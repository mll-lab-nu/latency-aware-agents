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
    ax.set(
        title="Ant zero-latency APPO training",
        xlabel="Environment frames",
        ylabel="Mean episode return",
    )
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
    length_ax.axhline(1000, color="#b46a2a", linestyle="--", linewidth=1.8)
    length_ax.set(title="Survival", ylabel="Mean episode length")
    length_ax.grid(axis="y", alpha=0.2)
    fig.suptitle("Ant policy under fixed action latency")
    fig.tight_layout()
    fig.savefig(ROOT / "eval_latency_sweep.png")
    plt.close(fig)


def plot_l0_returns() -> None:
    summary = json.loads((ROOT / "small_model_eval" / "queue_eval_latency_0.json").read_text())
    returns = summary["returns"]
    fig, ax = plt.subplots(figsize=(12, 6.75), dpi=120)
    ax.bar(range(len(returns)), returns, color="#476f87")
    ax.axhline(
        summary["mean_return"],
        color="#b46a2a",
        linestyle="--",
        linewidth=2,
        label=f"mean {summary['mean_return']:.1f}",
    )
    ax.set(title="Ant best-checkpoint evaluation · L0", xlabel="Episode", ylabel="Return")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(ROOT / "eval_l0_returns.png")
    plt.close(fig)


def plot_vla_training_curve() -> None:
    rows = {
        row["_step"]: row
        for row in (
            json.loads(line)
            for line in (ROOT / "vla_training" / "wandb_history.jsonl").read_text().splitlines()
        )
    }
    ordered = [rows[step] for step in sorted(rows)]
    steps = [row["_step"] for row in ordered]
    fig, (loss_ax, grad_ax) = plt.subplots(1, 2, figsize=(12, 6.75), dpi=120)
    loss_ax.plot(steps, [row["train/loss"] for row in ordered], "o-", color="#18794e", label="train L1")
    loss_ax.plot(steps, [row["eval/loss"] for row in ordered], "o-", color="#476f87", label="validation L1")
    loss_ax.set(title="Offline action loss", xlabel="Training step", ylabel="L1 loss")
    loss_ax.legend(frameon=False)
    loss_ax.grid(alpha=0.2)
    grad_ax.plot(steps, [row["train/grad_norm_pre_clip"] for row in ordered], "o-", color="#b46a2a")
    grad_ax.set(title="Gradient norm before clipping", xlabel="Training step", ylabel="L2 norm")
    grad_ax.grid(alpha=0.2)
    fig.suptitle("Ant RGB+state QwenOFT training")
    fig.tight_layout()
    fig.savefig(ROOT / "vla_training_curve.png")
    plt.close(fig)


def plot_vla_eval_returns() -> None:
    summaries = [
        json.loads((ROOT / "vla_eval" / f"queue_eval_latency_{latency}.json").read_text())
        for latency in (0, 2)
    ]
    fig, (return_ax, length_ax) = plt.subplots(1, 2, figsize=(12, 6.75), dpi=120)
    colors = ["#18794e", "#b46a2a"]
    for x, (summary, color) in enumerate(zip(summaries, colors, strict=True)):
        returns = summary["returns"]
        offsets = [(index - (len(returns) - 1) / 2) * 0.014 for index in range(len(returns))]
        return_ax.scatter([x + offset for offset in offsets], returns, color=color, alpha=0.65, s=34)
        return_ax.errorbar(
            x,
            summary["mean_return"],
            yerr=summary["std_return"],
            fmt="o",
            color="#182018",
            ecolor="#182018",
            capsize=7,
            linewidth=2.2,
            markersize=8,
        )
    labels = ["L0\n0 ms", "L2\n100 ms"]
    return_ax.set_xticks([0, 1], labels)
    return_ax.set(title="Return", ylabel="20 episodes; marker is mean ± std")
    return_ax.grid(axis="y", alpha=0.2)
    length_ax.bar(labels, [summary["mean_length"] for summary in summaries], color=colors)
    length_ax.axhline(1000, color="#182018", linestyle="--", linewidth=1.5)
    length_ax.set(title="Survival", ylabel="Mean episode length")
    length_ax.grid(axis="y", alpha=0.2)
    fig.suptitle("Ant RGB+state QwenVLA closed-loop evaluation")
    fig.tight_layout()
    fig.savefig(ROOT / "vla_eval_returns.png")
    plt.close(fig)


def build_contact_sheet() -> None:
    manifest_path = ROOT / "small_model_eval" / "eval_latency_0" / "videos" / "manifest.jsonl"
    manifest = json.loads(manifest_path.read_text().splitlines()[0])
    video = manifest_path.parent / manifest["video_path"]
    with tempfile.TemporaryDirectory() as tmp_dir:
        pattern = str(Path(tmp_dir) / "frame_%02d.png")
        subprocess.run(
            ["ffmpeg", "-loglevel", "error", "-i", str(video), "-vf", "fps=8/50,scale=360:360", "-frames:v", "8", pattern],
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
    plot_vla_training_curve()
    plot_vla_eval_returns()
    build_contact_sheet()
