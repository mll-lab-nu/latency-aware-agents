from __future__ import annotations

import json
import re
import subprocess
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
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
        title="Hopper RGB+state zero-latency APPO training",
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
    means = [item["mean_return"] for item in summaries]
    stds = [item["std_return"] for item in summaries]
    lengths = [item["mean_length"] for item in summaries]
    fig, (return_ax, length_ax) = plt.subplots(2, 1, figsize=(12, 9), dpi=120, sharex=True)
    bars = return_ax.bar(labels, means, yerr=stds, capsize=6, color="#18794e", edgecolor="#182018")
    return_ax.bar_label(bars, labels=[f"{value:.1f}" for value in means], padding=5)
    return_ax.set(title="Hopper RGB+state policy under fixed action latency", ylabel="Return (mean ± std, 10 episodes)")
    return_ax.grid(axis="y", alpha=0.2)
    length_bars = length_ax.bar(labels, lengths, color="#93a69a", edgecolor="#182018")
    length_ax.bar_label(length_bars, labels=[f"{value:.1f}" for value in lengths], padding=5)
    length_ax.set(xlabel="Evaluation latency", ylabel="Mean episode length")
    length_ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(ROOT / "eval_latency_sweep.png")
    plt.close(fig)


def plot_l0_episodes() -> None:
    rows = [
        json.loads(line)
        for line in (ROOT / "small_model_eval" / "eval_latency_0" / "episode_metrics.jsonl").read_text().splitlines()
    ]
    returns = [row["episode_return"] for row in rows]
    lengths = [row["survival_steps"] for row in rows]
    fig, ax = plt.subplots(figsize=(12, 6.75), dpi=120)
    bars = ax.bar(range(10), returns, color="#476f87", edgecolor="#182018")
    for bar, length in zip(bars, lengths, strict=True):
        if length < 1000:
            bar.set_hatch("//")
    ax.axhline(sum(returns) / len(returns), color="#b46a2a", linestyle="--", linewidth=2)
    ax.set(title="Hopper RGB+state best-checkpoint evaluation at L0", xlabel="Episode", ylabel="Return")
    ax.legend(
        handles=[
            Patch(facecolor="#476f87", edgecolor="#182018", label="1,000-step episode"),
            Patch(facecolor="#476f87", edgecolor="#182018", hatch="//", label="Early termination"),
        ],
        frameon=False,
    )
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(ROOT / "eval_l0_episodes.png")
    plt.close(fig)


def plot_vla_failure_diagnostic() -> None:
    diagnostic = json.loads((ROOT / "diagnostics" / "state_action_diagnostic.json").read_text())
    prefix = json.loads((ROOT / "diagnostics" / "teacher_prefix_1_3_10.json").read_text())
    prefix_50 = json.loads((ROOT / "diagnostics" / "teacher_prefix_50.json").read_text())
    teacher_summary = json.loads(
        (ROOT / "diagnostics" / "teacher_same_seed_summary.jsonl").read_text().splitlines()[0]
    )
    vla_summary = json.loads((ROOT / "vla_eval" / "queue_eval_latency_0.json").read_text())

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.8), dpi=120)
    distance = diagnostic["state_nearest_train_distance"]
    distance_keys = ["held_out_val", "teacher_online_full", "teacher_online_failed", "vla_online"]
    distance_labels = ["Held-out\nvalidation", "Teacher\nfull", "Teacher\nfailed", "VLA\nonline"]
    distance_colors = ["#93a69a", "#18794e", "#7aa58d", "#b46a2a"]
    p90 = [distance[key]["p90"] for key in distance_keys]
    median = [distance[key]["median"] for key in distance_keys]
    axes[0].bar(distance_labels, p90, color=distance_colors, edgecolor="#182018", alpha=0.55)
    axes[0].scatter(range(4), median, color="#182018", marker="_", s=360, linewidths=3, zorder=3)
    axes[0].set_yscale("log")
    axes[0].set(title="State distance to training manifold", ylabel="Normalized Euclidean distance (log scale)")
    axes[0].legend(
        handles=[Patch(facecolor="#93a69a", edgecolor="#182018", alpha=0.55, label="p90 bar"),
                 plt.Line2D([], [], color="#182018", marker="_", linestyle="None", markersize=14, markeredgewidth=3, label="median")],
        frameon=False,
    )
    axes[0].grid(axis="y", alpha=0.2)

    phases = ["0–9", "10–24", "25–49", "50+"]
    vla_error = diagnostic["actions"]["vla_vs_nearest_train_by_step"]
    teacher_error = diagnostic["actions"]["teacher_vs_nearest_train_by_step"]
    x = np.arange(len(phases))
    width = 0.36
    axes[1].bar(x - width / 2, [teacher_error[key]["mean"] for key in ("0_9", "10_24", "25_49", "50_plus")], width, color="#18794e", edgecolor="#182018", label="Teacher")
    axes[1].bar(x + width / 2, [vla_error[key]["mean"] for key in ("0_9", "10_24", "25_49", "50_plus")], width, color="#b46a2a", edgecolor="#182018", label="VLA")
    axes[1].set_yscale("log")
    axes[1].set_xticks(x, phases)
    axes[1].set(title="Action error by episode phase", xlabel="Environment steps", ylabel="MAE to nearest training-state action (log scale)")
    axes[1].legend(frameon=False)
    axes[1].grid(axis="y", alpha=0.2)

    length_labels = ["Teacher", "VLA", "Prefix 1", "Prefix 3", "Prefix 10", "Prefix 50"]
    lengths = [
        teacher_summary["mean_length"],
        vla_summary["mean_length"],
        prefix["1"]["mean_length"],
        prefix["3"]["mean_length"],
        prefix["10"]["mean_length"],
        prefix_50["50"]["mean_length"],
    ]
    length_bars = axes[2].bar(length_labels, lengths, color=["#18794e", *["#b46a2a"] * 5], edgecolor="#182018")
    axes[2].bar_label(length_bars, labels=[f"{value:.1f}" for value in lengths], padding=4)
    axes[2].tick_params(axis="x", rotation=35)
    axes[2].set(title="Teacher-prefix intervention", ylabel="Mean episode length (20 matched seeds)")
    axes[2].grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(ROOT / "vla_failure_diagnostic.png")
    plt.close(fig)


def plot_continuous_projector_training_loss() -> None:
    rows = json.loads((ROOT / "vla_training_continuous_projector" / "loss_anchors.json").read_text())
    steps = [row["step"] for row in rows]
    train_loss = [row["train_loss"] for row in rows]
    validation_loss = [row["validation_loss"] for row in rows]
    fig, ax = plt.subplots(figsize=(12, 6.75), dpi=120)
    ax.plot(steps, train_loss, color="#476f87", marker="o", linewidth=2.5, label="Train L1")
    ax.plot(
        steps,
        validation_loss,
        color="#b46a2a",
        marker="s",
        linestyle="--",
        linewidth=2.5,
        label="Held-out L1",
    )
    ax.set(
        title="Hopper continuous-projector QwenOFT loss",
        xlabel="Training step",
        ylabel="Action L1 loss",
    )
    ax.legend(frameon=False)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(ROOT / "vla_continuous_projector_training_loss.png")
    plt.close(fig)


def plot_vla_state_encoding_comparison() -> None:
    old = [
        json.loads((ROOT / "vla_eval" / f"queue_eval_latency_{latency}.json").read_text())
        for latency in (0, 2)
    ]
    continuous = [
        json.loads(
            (ROOT / "vla_eval_continuous_projector" / f"queue_eval_latency_{latency}.json").read_text()
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
        old_values = [row[field] for row in old]
        continuous_values = [row[field] for row in continuous]
        old_bars = ax.bar(
            x - width / 2,
            old_values,
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
        ax.bar_label(old_bars, labels=[f"{value:.1f}" for value in old_values], padding=4)
        ax.bar_label(continuous_bars, labels=[f"{value:.1f}" for value in continuous_values], padding=4)
        ax.set_xticks(x, labels)
        ax.set(title=title, ylabel=ylabel)
        ax.grid(axis="y", alpha=0.2)
    return_ax.legend(frameon=False)
    fig.suptitle("Hopper QwenOFT state encoding comparison · same seeds and checkpoint step")
    fig.tight_layout()
    fig.savefig(ROOT / "vla_state_encoding_comparison.png")
    plt.close(fig)


def build_contact_sheet() -> None:
    manifest_path = ROOT / "small_model_eval" / "eval_latency_0" / "videos" / "manifest.jsonl"
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
    plot_latency_sweep()
    plot_l0_episodes()
    plot_vla_failure_diagnostic()
    plot_continuous_projector_training_loss()
    plot_vla_state_encoding_comparison()
    build_contact_sheet()
