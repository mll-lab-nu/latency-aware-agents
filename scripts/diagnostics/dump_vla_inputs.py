"""Visualize and sanity-check what a starVLA model actually sees.

Two things are easy to get wrong with temporal multi-image input: the frame order
fed to the model, and whether training and evaluation preprocess frames the same
way. This script dumps the images for each ``image_mode`` (single / multiframe /
stitch) from BOTH paths side by side so you can eyeball them, and asserts they
match where they should:

  - eval: the real ``StarVlaPolicyRunner._build_example`` (latency_bench/policy/starvla.py)
  - train: the same selection + ``stitch_frames`` as ``_pack_sample`` (gr00t datasets.py)

Frames are numbered so temporal order and stitch layout are obvious. Pass
``--frames-dir`` to use real PNG frames instead of synthetic ones.

With ``--checkpoint`` and ``--print-trainable`` it also loads the model, applies
``trainer.freeze_vit`` / ``trainer.freeze_llm_layers``, and prints the trainable-parameter table.

Run in the starVLA env (stitch + freeze import torch via trainer_tools):
  /home/lixinyuan/miniconda3/envs/starvla_rl_games_gr00t/bin/python \
    scripts/diagnostics/dump_vla_inputs.py --frames 4 --out playground/vla_inputs
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from latency_bench.core.types import Observation
from latency_bench.policy.starvla import (
    ENV_RAW_RGB_FRAME_STACK_INFO_KEY,
    StarVlaPolicyRunner,
    _get_stitch_frames,
)

IMAGE_MODES = ("single", "multiframe", "stitch")


def _synthetic_frames(n: int, size: tuple[int, int]) -> list[np.ndarray]:
    """n numbered raw frames (oldest..newest), each a distinct hue with a big index."""
    height, width = size
    frames = []
    for i in range(n):
        hue = int(180 * i / max(1, n - 1))
        rgb = tuple(int(c) for c in _hsv_to_rgb(hue, 160, 220))
        img = Image.new("RGB", (width, height), rgb)
        draw = ImageDraw.Draw(img)
        draw.text((width // 2 - 8, height // 2 - 12), str(i), fill=(0, 0, 0))
        draw.rectangle([1, 1, width - 2, height - 2], outline=(0, 0, 0))
        frames.append(np.asarray(img, dtype=np.uint8))
    return frames


def _hsv_to_rgb(h: int, s: int, v: int) -> tuple[int, int, int]:
    img = Image.new("HSV", (1, 1), (h, s, v)).convert("RGB")
    return img.getpixel((0, 0))


def _load_frames(frames_dir: Path) -> list[np.ndarray]:
    paths = sorted(frames_dir.glob("*.png")) + sorted(frames_dir.glob("*.jpg"))
    if not paths:
        raise SystemExit(f"no .png/.jpg frames found in {frames_dir}")
    return [np.asarray(Image.open(p).convert("RGB"), dtype=np.uint8) for p in paths]


def _eval_runner(image_mode: str, num_obs_frames: int, stitch_grid: tuple[int, int],
                 obs_resize: tuple[int, int] | None, env_name: str) -> StarVlaPolicyRunner:
    """A StarVlaPolicyRunner with only the attributes _build_example reads, so the
    REAL eval method runs without loading a model."""
    runner = object.__new__(StarVlaPolicyRunner)
    runner._image_mode = image_mode
    runner._num_obs_frames = num_obs_frames
    runner._stitch_grid = stitch_grid
    runner._obs_resize = obs_resize
    runner._env_name = env_name
    runner._latency_prompt_map = None
    runner._include_state = False
    runner._state_dim = None
    return runner


def _pack_sample_images(frames, image_mode, num_obs_frames, stitch_grid):
    """Mirror of datasets.py:_pack_sample image selection (shares stitch_frames)."""
    stitch_frames = _get_stitch_frames()
    if image_mode == "single":
        selected = [frames[-1]]
    else:
        n = min(num_obs_frames, len(frames))
        selected = [frames[i] for i in range(len(frames) - n, len(frames))]
    pil = [Image.fromarray(f) for f in selected]
    if image_mode == "stitch":
        return [stitch_frames(pil, grid=stitch_grid, size=(224, 224))]
    return [p.resize((224, 224)) for p in pil]


def _to_pil(image) -> Image.Image:
    return image if isinstance(image, Image.Image) else Image.fromarray(np.asarray(image))


def _dump_images(images, out_dir: Path, prefix: str) -> list[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    names = []
    for idx, image in enumerate(images):
        name = f"{prefix}_{idx}.png"
        _to_pil(image).save(out_dir / name)
        names.append(name)
    return names


def visualize(args: argparse.Namespace) -> None:
    if args.frames_dir:
        frames = _load_frames(Path(args.frames_dir))
    else:
        frames = _synthetic_frames(args.frames, size=(args.raw_height, args.raw_width))
    print(f"raw stack: {len(frames)} frames, shape {frames[0].shape} (oldest..newest)")

    out_dir = Path(args.out)
    obs_resize = (224, 224)  # flappy eval default; eval resizes single/multiframe to this
    for mode in IMAGE_MODES:
        obs = Observation(
            data=frames[-1],
            env_step=0,
            sim_time_ms=0.0,
            metadata={ENV_RAW_RGB_FRAME_STACK_INFO_KEY: np.stack(frames)},
        )
        runner = _eval_runner(mode, args.frames, tuple(args.stitch_grid), obs_resize, args.env_name)
        eval_images = runner._build_example(obs)["image"]
        train_images = _pack_sample_images(frames, mode, args.frames, tuple(args.stitch_grid))

        eval_names = _dump_images(eval_images, out_dir / mode, "eval")
        train_names = _dump_images(train_images, out_dir / mode, "train")
        sizes = [tuple(_to_pil(im).size) for im in eval_images]  # (w, h)
        print(f"\n[{mode}] eval={len(eval_images)} img sizes={sizes}")
        print(f"  eval  -> {out_dir / mode}/{eval_names}")
        print(f"  train -> {out_dir / mode}/{train_names}")

        match = len(eval_images) == len(train_images) and all(
            np.array_equal(np.asarray(_to_pil(a)), np.asarray(_to_pil(b)))
            for a, b in zip(eval_images, train_images)
        )
        flag = "OK train==eval" if match else "MISMATCH train vs eval"
        print(f"  consistency: {flag}")
        if not match:
            raise SystemExit(f"[{mode}] train/eval preprocessing diverged — see dumped PNGs")
    print(f"\nall modes consistent. images under {out_dir}/")


def print_trainable(args: argparse.Namespace) -> None:
    sys.path.insert(0, str(REPO_ROOT / "starVLA"))
    from omegaconf import OmegaConf
    from starVLA.model.framework.base_framework import baseframework
    from starVLA.training.trainer_utils.trainer_tools import TrainerUtils, resolve_freeze_llm_layer_indices

    print(f"loading framework from {args.checkpoint}")
    model = baseframework.from_pretrained(args.checkpoint).eval()
    for param in model.parameters():
        param.requires_grad = True
    print("before freeze:")
    TrainerUtils.print_trainable_parameters(model)

    cfg = OmegaConf.create({"trainer": {"freeze_vit": args.freeze_vit, "freeze_llm_layers": args.freeze_llm_layers}})
    indices, total = resolve_freeze_llm_layer_indices(model, cfg)
    print(f"freeze_vit={args.freeze_vit}, freeze_llm_layers={args.freeze_llm_layers} -> freezing {indices} / total={total}")
    TrainerUtils.freeze_vit_and_llm_layers(model, cfg)
    print("after freeze:")
    TrainerUtils.print_trainable_parameters(model)


def main() -> None:
    parser = argparse.ArgumentParser(description="Dump/verify starVLA temporal image inputs and freezing.")
    parser.add_argument("--frames", type=int, default=4, help="number of synthetic raw frames")
    parser.add_argument("--frames-dir", type=str, default=None, help="dir of real PNG/JPG frames (oldest first)")
    parser.add_argument("--stitch-grid", type=int, nargs=2, default=[2, 2])
    parser.add_argument("--raw-height", type=int, default=120)
    parser.add_argument("--raw-width", type=int, default=160)
    parser.add_argument("--env-name", type=str, default="flappy")
    parser.add_argument("--out", type=str, default="playground/vla_inputs")
    parser.add_argument("--print-trainable", action="store_true", help="load --checkpoint and show freeze effect")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--freeze-vit", action="store_true", help="freeze the Qwen-VL visual tower")
    parser.add_argument("--freeze-llm-layers", type=int, nargs=2, default=[], metavar=("START", "END"))
    args = parser.parse_args()

    if args.print_trainable:
        if not args.checkpoint:
            raise SystemExit("--print-trainable requires --checkpoint")
        print_trainable(args)
    else:
        visualize(args)


if __name__ == "__main__":
    main()
