"""Milestone-2 gate for the fixed-size KV memory (req3).

Proves the no-eviction invariant: feeding a Qwen3-VL text model one frame block at
a time through a growing KV cache (with sliced M-RoPE positions) produces the same
last hidden state as a single full forward. This is the correctness foundation the
streaming KV memory is built on.

Run with the starVLA env, e.g.:
  /home/lixinyuan/miniconda3/envs/starvla_rl_games_gr00t/bin/python -m pytest \
    starVLA/tests/test_kv_memory_equivalence.py -q
"""

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from transformers.cache_utils import DynamicCache
from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLTextConfig
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextModel

from starVLA.model.modules.vlm.kv_memory import FrameKVMemory, chunked_prefill_last_hidden


def _build_positions(block_specs):
    """M-RoPE [3,1,L] positions for an ordered list of (kind, length) blocks.

    Mirrors get_rope_index's layout: text/action advance all three components
    together; a frame block shares one temporal index with arange h/w grids,
    offset by the running max+1.
    """
    cols = []
    nxt = 0
    for kind, n in block_specs:
        if kind in ("text", "action"):
            rng = torch.arange(n) + nxt
            cols.append(rng.view(1, -1).expand(3, -1))
            nxt = int(rng.max()) + 1 if n > 0 else nxt
        else:  # frame
            base = nxt
            block = torch.stack(
                [torch.zeros(n, dtype=torch.long), torch.arange(n), torch.arange(n)]
            ) + base
            cols.append(block)
            nxt = int(block.max()) + 1
    return torch.cat(cols, dim=1).unsqueeze(1)  # [3, 1, L]


def _tiny_text_model(num_hidden_layers=2):
    config = Qwen3VLTextConfig(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        vocab_size=100,
        rope_scaling={"mrope_section": [2, 1, 1], "rope_type": "default"},
        _attn_implementation="eager",
    )
    torch.manual_seed(0)
    return Qwen3VLTextModel(config).to(torch.float32).eval()


def _mrope_positions(text_len, frame_tokens, n_frames, act_len):
    """Build [3, 1, L] M-RoPE positions for [text][frame*N][act], frames-after-text.

    Text/action advance all three components together; each frame block is a flat
    spatial grid sharing one temporal index, mimicking get_rope_index's layout
    closely enough to exercise distinct per-component positions.
    """
    cols = []
    nxt = 0

    def text_block(n):
        nonlocal nxt
        rng = torch.arange(n) + nxt
        nxt = int(rng.max()) + 1 if n > 0 else nxt
        return rng.view(1, -1).expand(3, -1)

    cols.append(text_block(text_len))
    for _ in range(n_frames):
        base = nxt
        t_idx = torch.zeros(frame_tokens, dtype=torch.long)
        h_idx = torch.arange(frame_tokens, dtype=torch.long)
        w_idx = torch.arange(frame_tokens, dtype=torch.long)
        block = torch.stack([t_idx, h_idx, w_idx]) + base
        cols.append(block)
        nxt = int(block.max()) + 1
    cols.append(text_block(act_len))
    return torch.cat(cols, dim=1).unsqueeze(1)  # [3, 1, L]


@torch.no_grad()
def test_chunked_prefill_matches_full_forward():
    model = _tiny_text_model()
    text_len, frame_tokens, n_frames, act_len = 5, 4, 3, 2
    seq_len = text_len + frame_tokens * n_frames + act_len

    embeds = torch.randn(1, seq_len, model.config.hidden_size, dtype=torch.float32)
    positions = _mrope_positions(text_len, frame_tokens, n_frames, act_len)

    full = model(
        inputs_embeds=embeds,
        position_ids=positions,
        attention_mask=torch.ones(1, seq_len, dtype=torch.long),
        use_cache=False,
    ).last_hidden_state

    # Chunk as [text][f0][f1][f2][act] -- the streaming order.
    chunk_sizes = [text_len] + [frame_tokens] * n_frames + [act_len]
    chunked = chunked_prefill_last_hidden(model, embeds, positions, chunk_sizes)

    assert chunked.shape == full.shape
    assert torch.allclose(full, chunked, atol=1e-5, rtol=1e-4), (
        f"max abs diff = {(full - chunked).abs().max().item():.2e}"
    )


@torch.no_grad()
def test_action_tail_matches_when_frames_cached_separately():
    """The action-query hidden states (what the OFT head reads) must match even
    when every frame is prefilled in its own chunk."""
    model = _tiny_text_model()
    text_len, frame_tokens, n_frames, act_len = 3, 4, 4, 6
    seq_len = text_len + frame_tokens * n_frames + act_len

    embeds = torch.randn(1, seq_len, model.config.hidden_size, dtype=torch.float32)
    positions = _mrope_positions(text_len, frame_tokens, n_frames, act_len)

    full = model(
        inputs_embeds=embeds,
        position_ids=positions,
        attention_mask=torch.ones(1, seq_len, dtype=torch.long),
        use_cache=False,
    ).last_hidden_state

    chunk_sizes = [text_len] + [frame_tokens] * n_frames + [act_len]
    chunked = chunked_prefill_last_hidden(model, embeds, positions, chunk_sizes)

    full_tail = full[:, -act_len:]
    chunked_tail = chunked[:, -act_len:]
    assert torch.allclose(full_tail, chunked_tail, atol=1e-5, rtol=1e-4), (
        f"action-tail max abs diff = {(full_tail - chunked_tail).abs().max().item():.2e}"
    )


def _stream_action_hidden(model, text_emb, frame_embs, act_emb, window):
    """Drive FrameKVMemory frame-by-frame (the eval recurrence); return the final
    step's action-query hidden states."""
    mem = FrameKVMemory(model.rotary_emb, window, len(model.layers), model.config)
    text_len = text_emb.shape[1]
    frame_len = frame_embs[0].shape[1]
    act_len = act_emb.shape[1]
    action_hidden = None
    for frame_emb in frame_embs:
        n_past = mem.num_past_frames()
        specs = [("text", text_len)] + [("frame", frame_len)] * (n_past + 1) + [("action", act_len)]
        full_pos = _build_positions(specs)
        if not mem.has_text():
            # prefill: [text][frame0][action], empty cache
            new_emb = torch.cat([text_emb, frame_emb, act_emb], dim=1)
            cache = DynamicCache(config=model.config)
            cache_position = torch.arange(0, new_emb.shape[1])
            out = model(
                inputs_embeds=new_emb,
                position_ids=full_pos,
                attention_mask=torch.ones(1, new_emb.shape[1], dtype=torch.long),
                past_key_values=cache,
                use_cache=True,
                cache_position=cache_position,
            )
            mem.set_text(mem.canonicalize_block(cache, 0, text_len, full_pos[:, :, :text_len]))
            mem.add_frame(
                mem.canonicalize_block(cache, text_len, frame_len, full_pos[:, :, text_len : text_len + frame_len])
            )
        else:
            past_len = text_len + n_past * frame_len
            cache = mem.assemble_cache(full_pos[:, :, :past_len])
            new_emb = torch.cat([frame_emb, act_emb], dim=1)
            cache_position = torch.arange(past_len, past_len + new_emb.shape[1])
            out = model(
                inputs_embeds=new_emb,
                position_ids=full_pos[:, :, past_len:],
                attention_mask=torch.ones(1, past_len + new_emb.shape[1], dtype=torch.long),
                past_key_values=cache,
                use_cache=True,
                cache_position=cache_position,
            )
            mem.add_frame(
                mem.canonicalize_block(cache, past_len, frame_len, full_pos[:, :, past_len : past_len + frame_len])
            )
        action_hidden = out.last_hidden_state[:, -act_len:]
    return action_hidden


@torch.no_grad()
def test_rotation_roundtrip_is_exact():
    """The M-RoPE re-rotation behind eviction must be exact: un-rotating a key by
    the position it was rotated to, then rotating it to a new position, equals
    rotating the raw key straight to the new position."""
    from starVLA.model.modules.vlm.kv_memory import rotate_keys, unrotate_keys

    model = _tiny_text_model(num_hidden_layers=1)
    rotary = model.rotary_emb
    b, n_kv, length, hd = 1, 2, 6, model.config.head_dim
    raw_k = torch.randn(b, n_kv, length, hd)
    pos_old = _build_positions([("text", 2), ("frame", length - 2)])  # cached-at positions
    pos_new = _build_positions([("frame", length - 2), ("text", 2)])  # shifted slot positions

    k_at_old = rotate_keys(rotary, raw_k, pos_old)
    canonical = unrotate_keys(rotary, k_at_old, pos_old)
    re_rotated = rotate_keys(rotary, canonical, pos_new)
    direct = rotate_keys(rotary, raw_k, pos_new)

    assert torch.allclose(canonical, raw_k, atol=1e-6), "un-rotate is not the inverse of rotate"
    assert torch.allclose(re_rotated, direct, atol=1e-6), (
        f"re-rotation mismatch: {(re_rotated - direct).abs().max().item():.2e}"
    )


def _stream_action_hidden_batched(model, streams, window):
    """Drive several same-layout streams together via FrameKVMemory.stack/slice and
    return the final step's action hidden states [B, act_len, H].

    Mirrors QwenOFT._kv_forward_step_batched at the text-model level: per-stream
    memories are stacked along the batch dim, stepped once, then sliced back. The
    result for stream i must equal running stream i alone (batching changes nothing).
    """
    B = len(streams)
    mems = [FrameKVMemory(model.rotary_emb, window, len(model.layers), model.config) for _ in range(B)]
    text_len = streams[0]["text"].shape[1]
    frame_len = streams[0]["frames"][0].shape[1]
    act_len = streams[0]["act"].shape[1]
    n_frames = len(streams[0]["frames"])
    action_hidden = None
    for fi in range(n_frames):
        n_past = mems[0].num_past_frames()
        specs = [("text", text_len)] + [("frame", frame_len)] * (n_past + 1) + [("action", act_len)]
        full_pos1 = _build_positions(specs)
        full_pos = full_pos1.expand(3, B, full_pos1.shape[-1]).contiguous()
        batched = FrameKVMemory.stack(mems) if B > 1 else mems[0]
        if not batched.has_text():
            new_emb = torch.cat(
                [torch.cat([s["text"], s["frames"][fi], s["act"]], dim=1) for s in streams], dim=0
            )
            cache = DynamicCache(config=model.config)
            out = model(
                inputs_embeds=new_emb,
                position_ids=full_pos,
                attention_mask=torch.ones(B, new_emb.shape[1], dtype=torch.long),
                past_key_values=cache,
                use_cache=True,
                cache_position=torch.arange(0, new_emb.shape[1]),
            )
            batched.set_text(batched.canonicalize_block(cache, 0, text_len, full_pos[:, :, :text_len]))
            batched.add_frame(
                batched.canonicalize_block(cache, text_len, frame_len, full_pos[:, :, text_len : text_len + frame_len])
            )
        else:
            past_len = text_len + n_past * frame_len
            cache = batched.assemble_cache(full_pos[:, :, :past_len])
            new_emb = torch.cat([torch.cat([s["frames"][fi], s["act"]], dim=1) for s in streams], dim=0)
            out = model(
                inputs_embeds=new_emb,
                position_ids=full_pos[:, :, past_len:],
                attention_mask=torch.ones(B, past_len + new_emb.shape[1], dtype=torch.long),
                past_key_values=cache,
                use_cache=True,
                cache_position=torch.arange(past_len, past_len + new_emb.shape[1]),
            )
            batched.add_frame(
                batched.canonicalize_block(cache, past_len, frame_len, full_pos[:, :, past_len : past_len + frame_len])
            )
        if B > 1:
            for i, mem in enumerate(mems):
                sliced = batched.slice(i)
                mem.text, mem.frames = sliced.text, sliced.frames
        action_hidden = out.last_hidden_state[:, -act_len:]
    return action_hidden


@torch.no_grad()
def test_batched_streaming_matches_per_sample():
    """Cross-stream batching (FrameKVMemory.stack/slice) must be a no-op on values:
    each stream in a batched forward gets the same action hidden as running it alone.
    Exercises the eviction path (n_frames > window) at batch size > 1."""
    model = _tiny_text_model(num_hidden_layers=2)
    window = 3
    text_len, frame_len, act_len = 4, 4, 2
    n_frames = window + 1  # one eviction

    torch.manual_seed(7)
    streams = []
    for _ in range(2):
        streams.append(
            {
                "text": torch.randn(1, text_len, model.config.hidden_size),
                "frames": [torch.randn(1, frame_len, model.config.hidden_size) for _ in range(n_frames)],
                "act": torch.randn(1, act_len, model.config.hidden_size),
            }
        )

    batched = _stream_action_hidden_batched(model, streams, window)
    assert batched.shape[0] == 2
    for i, stream in enumerate(streams):
        ref = _stream_action_hidden(model, stream["text"], stream["frames"], stream["act"], window)
        assert torch.allclose(batched[i : i + 1], ref, atol=1e-6), (
            f"stream {i} batched vs alone max abs diff = {(batched[i:i+1] - ref).abs().max().item():.2e}"
        )


@torch.no_grad()
def test_streaming_memory_with_eviction_matches_windowed_forward_single_layer():
    """With a single decoder layer, layer-0 keys are context-independent, so the
    streaming memory after eviction must match a fresh forward over the surviving
    window exactly. This isolates and verifies the eviction + re-rotation path.
    (With >1 layer the two legitimately differ by the StreamingLLM history bleed:
    surviving frames' deeper-layer KV retain the evicted frame's influence. Since
    training uses the same streaming path, that bleed is consistent train/infer.)
    """
    model = _tiny_text_model(num_hidden_layers=1)
    window = 3
    text_len, frame_len, act_len = 4, 4, 2
    n_frames = window + 1  # forces exactly one eviction (f0 dropped)

    text_emb = torch.randn(1, text_len, model.config.hidden_size)
    frame_embs = [torch.randn(1, frame_len, model.config.hidden_size) for _ in range(n_frames)]
    act_emb = torch.randn(1, act_len, model.config.hidden_size)

    streamed = _stream_action_hidden(model, text_emb, frame_embs, act_emb, window)

    kept = frame_embs[-window:]
    specs = [("text", text_len)] + [("frame", frame_len)] * window + [("action", act_len)]
    ref_pos = _build_positions(specs)
    ref_emb = torch.cat([text_emb, *kept, act_emb], dim=1)
    ref = model(
        inputs_embeds=ref_emb,
        position_ids=ref_pos,
        attention_mask=torch.ones(1, ref_emb.shape[1], dtype=torch.long),
        use_cache=False,
    ).last_hidden_state[:, -act_len:]

    assert streamed.shape == ref.shape
    assert torch.allclose(streamed, ref, atol=1e-5, rtol=1e-4), (
        f"post-eviction action-tail max abs diff = {(streamed - ref).abs().max().item():.2e}"
    )
