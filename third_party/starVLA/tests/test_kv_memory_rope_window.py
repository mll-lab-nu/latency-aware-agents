"""Gate for the KV-memory window construction (req3, milestones 4/5).

QwenOFT._kv_forward_step_batched preprocesses only the current frame, then rebuilds
the n_visible-frame window's token ids by REPLICATING the frame's token block and
lets ``Qwen3VLModel.get_rope_index`` assign the M-RoPE positions. Two properties
must hold for the streaming KV memory to be correct:

  1. Replication validity: positions over the replicated window equal positions over
     a genuine multi-image window (identical frames -> identical ids -> identical
     positions). This is what lets us skip preprocessing duplicate frames.
  2. Append-stability: appending a new frame leaves the text + earlier frames'
     positions unchanged, so a cached block can be re-rotated to the slot it occupies
     in the current window (the canonical-K assumption behind assemble_cache).

Run with the starVLA env:
  /home/lixinyuan/miniconda3/envs/starvla_rl_games_gr00t/bin/python -m pytest \
    starVLA/tests/test_kv_memory_rope_window.py -q
"""

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from transformers.models.qwen3_vl.configuration_qwen3_vl import (
    Qwen3VLConfig,
    Qwen3VLTextConfig,
    Qwen3VLVisionConfig,
)
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLModel


def _tiny_vl_model():
    text = Qwen3VLTextConfig(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        vocab_size=200,
        rope_scaling={"mrope_section": [2, 1, 1], "rope_type": "default"},
        _attn_implementation="eager",
    )
    vis = Qwen3VLVisionConfig(
        hidden_size=32,
        intermediate_size=64,
        depth=2,
        num_heads=4,
        out_hidden_size=64,
        patch_size=14,
        spatial_merge_size=2,
        temporal_patch_size=2,
    )
    cfg = Qwen3VLConfig(text_config=text.to_dict(), vision_config=vis.to_dict())
    torch.manual_seed(0)
    return Qwen3VLModel(cfg).to(torch.float32).eval()


def _window(model, n_frames, *, prefix_len, frame_grid, act_len):
    """Build (input_ids[1,L], grid[n_frames,3], prefix_len, frame_block_len) for a
    text-first window [text][vs img.. ve]*N [action] with one resolution per frame."""
    cfg = model.config
    vs, ve, img = cfg.vision_start_token_id, cfg.vision_end_token_id, cfg.image_token_id
    t, h, w = frame_grid
    spms = cfg.vision_config.spatial_merge_size
    pad = (t * h * w) // (spms * spms)
    block = [vs] + [img] * pad + [ve]
    ids = list(range(prefix_len)) + block * n_frames + [50 + i for i in range(act_len)]
    grid = torch.tensor([[t, h, w]] * n_frames)
    return torch.tensor([ids]), grid, prefix_len, len(block)


@torch.no_grad()
def test_replicated_block_matches_genuine_multiframe_positions():
    """Replicating one frame's token block N times yields the same get_rope_index
    output as a from-scratch N-frame window (the basis for single-frame preprocessing)."""
    model = _tiny_vl_model()
    prefix_len, frame_grid, act_len = 3, (1, 4, 4), 2

    ids1, grid1, p_len, blk = _window(model, 1, prefix_len=prefix_len, frame_grid=frame_grid, act_len=act_len)
    for n in (2, 3, 4):
        prefix = ids1[0, :p_len]
        block = ids1[0, p_len : p_len + blk]
        action = ids1[0, p_len + blk :]
        replicated = torch.cat([prefix] + [block] * n + [action]).unsqueeze(0)
        genuine, grid_n, _, _ = _window(model, n, prefix_len=prefix_len, frame_grid=frame_grid, act_len=act_len)

        assert torch.equal(replicated, genuine), f"replicated ids != genuine for N={n}"
        pos_rep, _ = model.get_rope_index(replicated, image_grid_thw=grid1.repeat(n, 1))
        pos_gen, _ = model.get_rope_index(genuine, image_grid_thw=grid_n)
        assert torch.equal(pos_rep, pos_gen), f"positions differ for N={n}"


@torch.no_grad()
def test_append_leaves_earlier_positions_stable():
    """Appending a frame must not move the positions of the text + earlier frames, so
    a cached block's slot positions are stable step to step (canonical-K assumption)."""
    model = _tiny_vl_model()
    prefix_len, frame_grid, act_len = 3, (1, 4, 4), 2

    for n in (1, 2, 3):
        ids_n, grid_n, p_len, blk = _window(model, n, prefix_len=prefix_len, frame_grid=frame_grid, act_len=act_len)
        ids_n1, grid_n1, _, _ = _window(model, n + 1, prefix_len=prefix_len, frame_grid=frame_grid, act_len=act_len)
        pos_n, _ = model.get_rope_index(ids_n, image_grid_thw=grid_n)
        pos_n1, _ = model.get_rope_index(ids_n1, image_grid_thw=grid_n1)

        shared = p_len + n * blk  # text + first n frame blocks
        assert torch.equal(pos_n[:, :, :shared], pos_n1[:, :, :shared]), (
            f"earlier positions shifted when going from {n} to {n + 1} frames"
        )
