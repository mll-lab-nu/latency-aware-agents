"""End-to-end equivalence check for the streaming KV memory on a REAL checkpoint.

The cache / rotation / position math is unit-proven on tiny models
(starVLA/tests/test_kv_memory_equivalence.py, test_kv_memory_rope_window.py). This
script validates the one piece those cannot: the real-model segment construction
(processor chat-template token boundaries, get_rope_index positions, pixel
handling) once the memory path is wired to an actual Qwen3-VL-4B checkpoint.

Invariant checked: with ``window >= num_frames`` (NO eviction), feeding frames one
at a time through the streaming KV memory is mathematically identical to a single
full forward over the same text-first ``[instruction][frame0..frameN-1][action]``
window. So the per-step action output after N frames must equal the full-window
forward to floating-point tolerance. This exercises exactly the fragile real-model
wiring end to end.

With eviction (``--window`` < frames) the two legitimately differ by the
StreamingLLM history bleed (surviving frames' deeper KV keep the evicted frame's
influence); that case is reported, not asserted.

Requires a GPU and a QwenOFT checkpoint. Run in the starVLA env:
  .../python scripts/diagnostics/check_kv_memory_e2e.py \
    --checkpoint <ckpt> --frames 3 --dtype float32
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[2]
for path in (REPO_ROOT, REPO_ROOT / "starVLA"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def _synthetic_frames(n: int, size: tuple[int, int]) -> list[Image.Image]:
    height, width = size
    frames = []
    for i in range(n):
        img = Image.new("RGB", (width, height), (40 + 30 * i, 90, 200 - 30 * i))
        ImageDraw.Draw(img).text((width // 2, height // 2), str(i), fill=(255, 255, 255))
        frames.append(img)
    return frames


@torch.inference_mode()
def _reference_actions(model, frames, instruction) -> np.ndarray:
    """One full text-first forward over all frames (no cache) -> action output.

    Uses the same text-first window builder the memory path uses, so the only
    difference under test is cache vs no-cache (which must be a no-op without
    eviction)."""
    batch = model._build_text_first_inputs(frames, instruction)
    backbone = model.qwen_vl_interface.model.model
    param_dtype = next(model.parameters()).dtype
    pixel_values = batch["pixel_values"].to(param_dtype)
    out = backbone(
        input_ids=batch["input_ids"],
        pixel_values=pixel_values,
        image_grid_thw=batch["image_grid_thw"],
        use_cache=False,
    )
    with torch.autocast("cuda", dtype=torch.float32):
        queries = model._gather_action_token_embeddings(
            out.last_hidden_state, batch["input_ids"], action_token_id=model.action_token_id
        )
        actions = model.action_model.predict_action(queries)
    return actions.float().cpu().numpy()


@torch.inference_mode()
def _memory_actions(model, frames, instruction, window) -> np.ndarray:
    """Stream frames one at a time through the per-slot KV memory; return the action
    output after the last frame."""
    model.kv_memory_enabled = True
    model.kv_window = window
    model.reset_memory()
    out = None
    for frame in frames:
        out = model.predict_action([{"image": [frame], "lang": instruction, "slot_id": 0}])
    return np.asarray(out["normalized_actions"])


def _check_training(model, args, dtype) -> None:
    """Batched training rollout must not change the loss: loss([e0,e1]) (one batched
    group) == mean(loss([e0]), loss([e1])) (per-sample). Confirms the grouped batching
    and no_grad on non-final steps in _forward_memory are value-neutral on the real
    model."""
    model.kv_memory_enabled = True
    model.kv_window = args.window if args.window is not None else max(2, args.frames // 2)

    train_obs = getattr(model.config.datasets.vla_data, "obs_image_size", None)
    size = tuple(train_obs[::-1]) if train_obs else (args.raw_height, args.raw_width)
    action_dim = int(model.config.framework.action_model.action_dim)
    horizon = int(model.config.framework.action_model.action_horizon)

    def make_example(seed):
        rng = np.random.default_rng(seed)
        frames = [
            Image.fromarray(rng.integers(0, 256, (size[0], size[1], 3), dtype=np.uint8))
            for _ in range(args.frames)
        ]
        action = rng.standard_normal((horizon, action_dim)).astype(np.float32)
        return {"image": frames, "lang": args.instruction, "action": action}

    e0, e1 = make_example(0), make_example(1)
    loss_batched = float(model.forward([e0, e1])["action_loss"].item())
    loss0 = float(model.forward([e0])["action_loss"].item())
    loss1 = float(model.forward([e1])["action_loss"].item())
    expected = 0.5 * (loss0 + loss1)
    atol = args.atol if args.atol is not None else (1e-4 if dtype == torch.float32 else 1e-2)
    diff = abs(loss_batched - expected)
    print(f"window={model.kv_window} frames={args.frames} dtype={args.dtype}")
    print(f"loss([e0,e1])={loss_batched:.6f}  mean(loss(e0),loss(e1))={expected:.6f}  diff={diff:.3e} (atol={atol})")
    if diff <= atol:
        print("PASS: batched training rollout is value-neutral on the real model.")
    else:
        raise SystemExit(f"FAIL: batched-vs-per-sample loss diff {diff:.3e} exceeds atol {atol}.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Real-checkpoint KV-memory equivalence check.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--frames", type=int, default=3)
    parser.add_argument("--window", type=int, default=None, help="memory window; default = frames (no eviction)")
    parser.add_argument("--instruction", type=str, default="You are playing Flappy Bird.")
    parser.add_argument("--dtype", choices=["float32", "bfloat16"], default="float32")
    parser.add_argument("--raw-height", type=int, default=224)
    parser.add_argument("--raw-width", type=int, default=224)
    parser.add_argument("--atol", type=float, default=None, help="default 1e-3 (fp32) / 5e-2 (bf16)")
    parser.add_argument("--check-training", action="store_true",
                        help="check batched training rollout (loss) instead of inference equivalence")
    args = parser.parse_args()

    dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16}[args.dtype]

    from starVLA.model.framework.base_framework import baseframework

    print(f"loading {args.checkpoint} as {args.dtype}")
    model = baseframework.from_pretrained(args.checkpoint).to(dtype).to("cuda").eval()

    if args.check_training:
        _check_training(model, args, dtype)
        return

    atol = args.atol if args.atol is not None else (1e-3 if dtype == torch.float32 else 5e-2)
    window = args.window if args.window is not None else args.frames
    evicts = window < args.frames

    train_obs = getattr(model.config.datasets.vla_data, "obs_image_size", None)
    size = tuple(train_obs[::-1]) if train_obs else (args.raw_height, args.raw_width)
    frames = _synthetic_frames(args.frames, size)  # pre-sized so both paths feed identical pixels

    ref = _reference_actions(model, frames, args.instruction)
    mem = _memory_actions(model, frames, args.instruction, window)

    if ref.shape != mem.shape:
        raise SystemExit(f"shape mismatch: reference {ref.shape} vs memory {mem.shape}")
    max_abs = float(np.abs(ref - mem).max())
    print(f"frames={args.frames} window={window} eviction={evicts} dtype={args.dtype}")
    print(f"max abs diff (memory vs full forward) = {max_abs:.3e} (atol={atol})")

    if evicts:
        print("eviction active: difference is the expected StreamingLLM history bleed (not asserted).")
        return
    if max_abs <= atol:
        print("PASS: streaming KV memory == full forward (no eviction) on the real model.")
    else:
        raise SystemExit(f"FAIL: diff {max_abs:.3e} exceeds atol {atol}. Real-model wiring is off.")


if __name__ == "__main__":
    main()
