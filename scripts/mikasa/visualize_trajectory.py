#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


_FPS = 20
_STEP_FRACTIONS = (0.0, 0.25, 0.5, 0.85, 1.0)
_PHASES = ("Initial", "Approach", "Intercept", "Close & stabilize", "Success")


def _write_key_steps(
    output_path: Path,
    steps: list[int],
    action: np.ndarray,
    proprio: np.ndarray,
    reward: np.ndarray,
    success: np.ndarray,
    done: np.ndarray,
) -> None:
    action_names = ("dx", "dy", "dz", "droll", "dpitch", "dyaw", "gripper")
    proprio_names = ("x", "y", "z", "roll", "pitch", "yaw", "gripper_width")
    fieldnames = ["step", "time_s", "phase", "reward", "cumulative_reward"]
    fieldnames += [f"action_{name}" for name in action_names]
    fieldnames += [f"proprio_{name}" for name in proprio_names]
    fieldnames += ["success", "done"]
    cumulative_reward = np.cumsum(reward)

    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for step, phase in zip(steps, _PHASES, strict=True):
            row = {
                "step": step,
                "time_s": f"{step / _FPS:.2f}",
                "phase": phase,
                "reward": f"{reward[step]:.6f}",
                "cumulative_reward": f"{cumulative_reward[step]:.6f}",
                "success": success[step],
                "done": done[step],
            }
            row.update({f"action_{name}": f"{action[step, i]:.6f}" for i, name in enumerate(action_names)})
            row.update(
                {f"proprio_{name}": f"{proprio[step, i]:.6f}" for i, name in enumerate(proprio_names)}
            )
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render one MIKASA PPO-teacher trajectory for slides.")
    parser.add_argument("input_npz", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    with np.load(args.input_npz) as episode:
        rgb = episode["rgb"]
        proprio = episode["proprio"]
        action = episode["action"]
        reward = episode["reward"]
        success = episode["success"]
        done = episode["done"]
        instruction = episode["language_instruction"].item()
        seed = episode["episode_seed"].item()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.input_npz.stem
    png_path = args.output_dir / f"{stem}_data.png"
    pdf_path = args.output_dir / f"{stem}_data.pdf"
    csv_path = args.output_dir / f"{stem}_key_steps.csv"

    steps = [round(fraction * (len(reward) - 1)) for fraction in _STEP_FRACTIONS]
    _write_key_steps(csv_path, steps, action, proprio, reward, success, done)

    time_s = np.arange(len(reward)) / _FPS
    shown_step = steps[2]
    success_step = np.flatnonzero(success)[-1]

    figure = plt.figure(figsize=(16, 9), dpi=240, facecolor="white")
    grid = figure.add_gridspec(
        3,
        4,
        height_ratios=(1.35, 1.0, 1.0),
        left=0.06,
        right=0.95,
        bottom=0.08,
        top=0.82,
        hspace=0.58,
        wspace=0.58,
    )
    figure.suptitle("PPO-teacher demonstration as VLA training data", fontsize=22, fontweight="bold", y=0.96)
    figure.text(
        0.5,
        0.91,
        f"InterceptGrabFast-VLA-v0  |  seed {seed}  |  {len(reward)} steps / {len(reward) / _FPS:.2f} s"
        f"  |  return {reward.sum():.3f}  |  success at {success_step / _FPS:.2f} s",
        ha="center",
        fontsize=12,
    )
    figure.text(0.5, 0.87, f'Instruction: "{instruction}"', ha="center", fontsize=12)

    top_axis = figure.add_subplot(grid[0, 0])
    top_axis.imshow(rgb[shown_step, :, :, :3])
    top_axis.set_title(f"Top RGB input — t={shown_step / _FPS:.2f} s", fontsize=11)
    top_axis.axis("off")

    wrist_axis = figure.add_subplot(grid[0, 1])
    wrist_axis.imshow(rgb[shown_step, :, :, 3:6])
    wrist_axis.set_title(f"Wrist RGB input — t={shown_step / _FPS:.2f} s", fontsize=11)
    wrist_axis.axis("off")

    table_axis = figure.add_subplot(grid[0, 2:])
    table_axis.axis("off")
    table_axis.set_title("VLA training data contract", fontsize=12, pad=10)
    contract = (
        ("Top RGB", "VLA input", f"[T, {rgb.shape[1]}, {rgb.shape[2]}, 3]"),
        ("Wrist RGB", "VLA input", f"[T, {rgb.shape[1]}, {rgb.shape[2]}, 3]"),
        ("Instruction", "VLA input", "text"),
        ("Proprio", "recorded only", "[T, 7]"),
        ("Action", "training target", "[T, 7]"),
        ("Reward / success / done", "diagnostic only", "[T]"),
    )
    table = table_axis.table(
        cellText=contract,
        colLabels=("Field", "Role", "Shape"),
        cellLoc="left",
        colLoc="left",
        loc="center",
        colWidths=(0.43, 0.32, 0.25),
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.55)
    table_axis.text(
        0.0,
        -0.08,
        "The PPO teacher's 49D privileged state is not stored in this exported trajectory.",
        transform=table_axis.transAxes,
        fontsize=9,
        color="0.35",
    )

    translation_axis = figure.add_subplot(grid[1, :2])
    for index, label in enumerate(("dx", "dy", "dz")):
        translation_axis.plot(time_s, action[:, index], label=label, linewidth=1.8)
    translation_axis.set_title("Teacher action — translation")
    translation_axis.set_ylabel("normalized delta")
    translation_axis.legend(ncol=3, frameon=False)

    rotation_axis = figure.add_subplot(grid[1, 2:])
    for index, label in zip(range(3, 6), ("droll", "dpitch", "dyaw"), strict=True):
        rotation_axis.plot(time_s, action[:, index], label=label, linewidth=1.8)
    rotation_axis.set_title("Teacher action — rotation")
    rotation_axis.set_ylabel("normalized delta")
    rotation_axis.legend(ncol=3, frameon=False)

    gripper_axis = figure.add_subplot(grid[2, :2])
    command_line = gripper_axis.plot(time_s, action[:, 6], label="gripper command", linewidth=1.8)[0]
    width_axis = gripper_axis.twinx()
    width_line = width_axis.plot(
        time_s,
        proprio[:, 6],
        color="tab:orange",
        label="gripper width",
        linewidth=1.8,
    )[0]
    gripper_axis.set_title("Gripper command and observed width")
    gripper_axis.set_ylabel("command (-1 close, +1 open)")
    width_axis.set_ylabel("width [m]")
    gripper_axis.legend((command_line, width_line), (command_line.get_label(), width_line.get_label()), frameon=False)

    reward_axis = figure.add_subplot(grid[2, 2:])
    reward_line = reward_axis.plot(time_s, reward, label="normalized reward", linewidth=1.8)[0]
    return_axis = reward_axis.twinx()
    return_line = return_axis.plot(
        time_s,
        np.cumsum(reward),
        color="tab:green",
        label="cumulative return",
        linewidth=1.8,
    )[0]
    success_mark = reward_axis.scatter(
        time_s[success_step],
        reward[success_step],
        marker="*",
        s=120,
        color="tab:red",
        label="success",
    )
    reward_axis.set_title("Environment result")
    reward_axis.set_ylabel("normalized reward")
    return_axis.set_ylabel("cumulative return")
    reward_axis.legend(
        (reward_line, return_line, success_mark),
        (reward_line.get_label(), return_line.get_label(), success_mark.get_label()),
        frameon=False,
        loc="upper left",
    )

    for axis in (translation_axis, rotation_axis, gripper_axis, reward_axis):
        axis.axvline(time_s[shown_step], color="0.55", linestyle=":", linewidth=1.0)
        axis.axvline(time_s[success_step], color="tab:red", linestyle="--", linewidth=1.0)
        axis.set_xlim(time_s[0], len(reward) / _FPS)
        axis.set_xlabel("time [s]")
        axis.grid(alpha=0.22)

    figure.savefig(png_path, dpi=240, facecolor="white")
    figure.savefig(pdf_path, facecolor="white")
    plt.close(figure)


if __name__ == "__main__":
    main()
