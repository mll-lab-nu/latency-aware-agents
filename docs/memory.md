# Temporal images and KV memory

StarVLA QwenOFT supports temporal image inputs and optional KV memory.
Training and evaluation must use the same image mode and observation window.
The checkpoint configuration selects these settings.

## Image inputs

| Setting | Meaning | Default |
| --- | --- | --- |
| `datasets.vla_data.num_obs_frames` | Number of temporal frames | `1` |
| `datasets.vla_data.image_mode` | `single`, `multiframe`, or `stitch` | `single` |
| `datasets.vla_data.stitch_grid` | Rows and columns for the stitched image | `[2, 2]` |

`single` uses the latest frame. `multiframe` supplies separate images.
`stitch` places frames in one image, trading spatial detail for fewer image tokens.
For these image modes, configure the environment's frame stack to supply the
training window. The inference adapter reads the checkpoint's image settings.

## KV memory

Set `framework.kv_memory.enabled: true` to enable QwenOFT's memory path.
`window` controls the inference memory capacity. `rollout_len` controls the
training sequence length. Each inference slot retains its own memory and resets
it at episode boundaries and soft resets.

Memory inference processes the current image and retains previous frame keys
and values. It does not require the environment to resend the memory window.
Training uses the same recurrence, detaches state between steps, and computes
loss at the final step. A rollout longer than the window exposes training to eviction.

Cached deeper-layer states retain information from evicted frames. Their output
need not match a fresh forward pass over only the surviving frames.
Unit checks of cache and position operations do not establish complete behavior
with a pretrained model. Check the selected checkpoint's processor and
memory configuration through a full evaluation before interpreting its results.

## Commands

Training examples live in the pinned StarVLA checkout:

```bash
bash third_party/starVLA/commands/memory/train_flappy_openvla_multiframe.sh
bash third_party/starVLA/commands/memory/train_flappy_openvla_stitch.sh
bash third_party/starVLA/commands/memory/train_flappy_openvla_kv_memory.sh
```

Inspect each recipe's dataset, checkpoint, and output paths before running it.
These are experiment recipes, not installation commands.

Existing evaluation configurations:

```bash
python -m latency_bench.run --config configs/experiments/flappy/starvla_flappy_multiframe.yaml
python -m latency_bench.run --config configs/experiments/flappy/starvla_flappy_kv_memory.yaml
```

Supply the intended checkpoint and dataset metadata as described in the
[artifact guide](artifacts.md#prepare-a-backbone-for-evaluation).
The [visual-history catalogue](https://huggingface.co/datasets/MLL-Lab/LAGEN-datasets/tree/e0f4007f504978f93c32557ea4e9b97fdf62fcf5/experiments/visual-history)
records published comparisons and their experimental conditions.
