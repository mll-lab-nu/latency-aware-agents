# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License.

"""Fixed-size cross-step KV memory for the Qwen3-VL backbone (QwenOFT req3).

Goal: keep the visual KV of the last ``window`` frames as a memory that the
current frame and the action-query tokens attend to, so a streaming policy does
not re-encode old frames every step. Training uses the same recurrence, so there
is no train/inference gap.

Milestone status (see docs/memory.md):
  1/2 (this file, no eviction): an incremental, KV-cached forward fed one frame
      at a time is mathematically identical to a single forward over the whole
      sequence. ``chunked_prefill_last_hidden`` is that primitive, and
      ``tests/test_kv_memory_equivalence.py`` is the gate that proves it.
  3   (eviction): once the window is full the oldest frame is dropped and the
      remaining frames' M-RoPE temporal positions shift, so their cached (already
      RoPE-rotated) keys go stale. The fix is StreamingLLM-style slot-based
      re-rotation / pre-RoPE caching. Tracked separately; not implemented here.
"""

from __future__ import annotations

from typing import Callable, List, Optional

import torch
from transformers.cache_utils import DynamicCache
from transformers.models.qwen3_vl.modeling_qwen3_vl import rotate_half


def make_packed_sliding_mask_mod(kind: torch.Tensor, step: torch.Tensor, window: int):
    """FlexAttention ``mask_mod`` for the packed sliding-window + attention-sink layout.

    Used by the packed single-forward training path
    (``QwenOFT._forward_memory_packed``), which lays the whole rollout out as
    ``[text/sink] [frame_1][action_1] ... [frame_R][action_R]`` and runs ONE
    Qwen3-VL forward under this mask instead of a per-frame Python loop.

    ``kind`` (0=text/sink, 1=frame, 2=action) and ``step`` (frame/step index)
    are per-token 1D long tensors on the attention device. A text/sink key's
    ``step`` encodes which queries may see it:
      * ``-1`` = legacy single global sink (``rebased_sink=false``), visible to
        every query;
      * ``-2`` = dedup shared front sink (``rebased_sink=true``), visible only to
        the pre-eviction steps ``q < W`` — those steps' windows haven't evicted
        anything yet, so eval keeps one sink at the front for all of them;
      * ``>= W`` = per-step re-based sink, visible only to its own step ``q == k``
        (re-based so its text→frame distance matches eval's rolling-KV layout).
    ``window`` is the number of visible frames (``kv_window``). The predicate
    composes:
      * base causal (``kv <= q``);
      * a sink (text) key is visible per the ``step`` encoding above
        (``step==-1`` global, ``step==-2`` shared for ``q<W``, else own step);
      * a frame key is visible to any non-text query within the last ``window``
        frames (``step[q]-window < step[kv] <= step[q]``);
      * an action key is visible ONLY to its own step's action query (read-out
        isolation), so frames never attend any action and the frame KV stream is
        bit-identical to the streaming loop / eval.
    Every query sees at least its own sink and itself → no fully-masked row → no NaN.
    """
    W = int(window)

    def mask_mod(b, h, q_idx, kv_idx):
        causal = kv_idx <= q_idx
        kq = kind[q_idx]
        kk = kind[kv_idx]
        sq = step[q_idx]
        sk = step[kv_idx]
        sink = (kk == 0) & ((sk == -1) | ((sk == -2) & (sq < W)) | (sq == sk))
        frame_vis = (kk == 1) & (kq != 0) & (sk <= sq) & (sk > sq - W)
        action_vis = (kk == 2) & (kq == 2) & (sq == sk)
        return causal & (sink | frame_vis | action_vis)

    return mask_mod


def _key_cos_sin(rotary_emb, keys: torch.Tensor, position_ids: torch.Tensor):
    """cos/sin for `keys` [B, n_kv, L, hd] at `position_ids` [3, B, L] -> [B,1,L,hd]."""
    cos, sin = rotary_emb(keys, position_ids)  # [B, L, hd]
    return cos.unsqueeze(1), sin.unsqueeze(1)


def rotate_keys(rotary_emb, keys: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
    """Apply M-RoPE to keys at the given positions (same as the model's attention)."""
    cos, sin = _key_cos_sin(rotary_emb, keys, position_ids)
    return keys * cos + rotate_half(keys) * sin


def unrotate_keys(rotary_emb, keys: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
    """Inverse of :func:`rotate_keys`. Un-rotating a key by the position it was
    rotated to yields its canonical (position-0 / pre-RoPE) form, which can be
    re-rotated to any later position without accumulating error across evictions.
    """
    cos, sin = _key_cos_sin(rotary_emb, keys, position_ids)
    return keys * cos - rotate_half(keys) * sin


def chunked_prefill_last_hidden(
    text_model,
    inputs_embeds: torch.Tensor,
    position_ids: torch.Tensor,
    chunk_sizes: List[int],
    cache: Optional[DynamicCache] = None,
) -> torch.Tensor:
    """Run a Qwen3-VL text model over ``inputs_embeds`` in sequential chunks.

    All chunks share one growing ``DynamicCache``; each chunk only computes its
    own tokens and attends back to the cache. With append-only positions this
    equals a single full forward (verified by the equivalence test). This is the
    no-eviction core of the fixed-size KV memory.

    Args:
        text_model: a ``Qwen3VLTextModel`` (the ``language_model`` backbone).
        inputs_embeds: ``[B, L, H]`` token embeddings for the whole sequence.
        position_ids: ``[3, B, L]`` M-RoPE position ids for the whole sequence.
        chunk_sizes: token counts per chunk, in order, summing to ``L``.
        cache: optional pre-populated cache (e.g. carried static prefix).

    Returns:
        ``[B, L, H]`` last hidden state, concatenated across chunks.
    """
    if sum(chunk_sizes) != inputs_embeds.shape[1]:
        raise ValueError(f"chunk_sizes sum {sum(chunk_sizes)} != seq len {inputs_embeds.shape[1]}")

    if cache is None:
        cache = DynamicCache(config=text_model.config)

    batch_size = inputs_embeds.shape[0]
    device = inputs_embeds.device
    outputs: List[torch.Tensor] = []
    start = int(cache.get_seq_length())
    cursor = 0
    for size in chunk_sizes:
        end = start + size
        embeds_chunk = inputs_embeds[:, cursor : cursor + size]
        pos_chunk = position_ids[:, :, cursor : cursor + size]
        cache_position = torch.arange(start, end, device=device)
        # Full-length mask over [0, end): the chunk attends to everything cached so far.
        attention_mask = torch.ones(batch_size, end, dtype=torch.long, device=device)
        out = text_model(
            inputs_embeds=embeds_chunk,
            position_ids=pos_chunk,
            attention_mask=attention_mask,
            past_key_values=cache,
            use_cache=True,
            cache_position=cache_position,
        )
        outputs.append(out.last_hidden_state)
        start = end
        cursor += size
    return torch.cat(outputs, dim=1)


_Block = tuple  # (K0_list: List[Tensor], V_list: List[Tensor], length: int)


class FrameKVMemory:
    """Fixed-size streaming KV memory for one stream (one env slot / one rollout).

    Layout the memory reconstructs each step:
        [text (static)] [past frame blocks (ring buffer)] [current frame] [action]

    Keys are stored in canonical (pre-RoPE / position-0) form so a block can be
    re-rotated to whatever slot it currently occupies without accumulating error
    across evictions. The static text block is kept once; up to ``window - 1`` past
    frame blocks are kept so that, together with the current frame, the model
    attends to ``window`` frames.
    """

    def __init__(self, rotary_emb, window: int, num_layers: int, text_config=None):
        self.rotary_emb = rotary_emb
        self.window = int(window)
        self.num_layers = int(num_layers)
        self.text_config = text_config
        self.text: Optional[_Block] = None
        self.frames: List[_Block] = []  # oldest first

    def reset(self) -> None:
        self.text = None
        self.frames = []

    def has_text(self) -> bool:
        return self.text is not None

    def num_past_frames(self) -> int:
        return len(self.frames)

    def _blocks(self) -> List[_Block]:
        return ([self.text] if self.text is not None else []) + self.frames

    def past_length(self) -> int:
        return sum(block[2] for block in self._blocks())

    def assemble_cache(self, past_positions: torch.Tensor) -> DynamicCache:
        """Build a DynamicCache for the stored blocks, re-rotated to ``past_positions``.

        Args:
            past_positions: ``[3, B, past_length()]`` M-RoPE positions for the stored
                blocks, in order (text then frames), matching the current layout.
        """
        cache = DynamicCache(config=self.text_config)
        blocks = self._blocks()
        for layer in range(self.num_layers):
            key_parts, val_parts = [], []
            offset = 0
            for (k0_list, v_list, length) in blocks:
                pos = past_positions[:, :, offset : offset + length]
                key_parts.append(rotate_keys(self.rotary_emb, k0_list[layer], pos))
                val_parts.append(v_list[layer])
                offset += length
            cache.update(torch.cat(key_parts, dim=2), torch.cat(val_parts, dim=2), layer)
        return cache

    def canonicalize_block(self, cache: DynamicCache, start: int, length: int, positions: torch.Tensor) -> _Block:
        """Extract a freshly cached block (post-RoPE at ``positions``) and store it in
        canonical form. ``positions`` is ``[3, B, length]`` for that block."""
        k0_list, v_list = [], []
        for layer in range(self.num_layers):
            keys = cache.layers[layer].keys[:, :, start : start + length]
            vals = cache.layers[layer].values[:, :, start : start + length]
            k0_list.append(unrotate_keys(self.rotary_emb, keys, positions))
            v_list.append(vals)
        return (k0_list, v_list, length)

    def set_text(self, block: _Block) -> None:
        self.text = block

    def add_frame(self, block: _Block) -> None:
        """Append the current frame block, evicting the oldest to keep ``window - 1``."""
        self.frames.append(block)
        while len(self.frames) > self.window - 1:
            self.frames.pop(0)

    def detach_(self) -> None:
        """Detach all stored tensors (truncated BPTT: history is a fixed memory)."""
        def _d(block):
            k0, v, n = block
            return ([k.detach() for k in k0], [vv.detach() for vv in v], n)

        if self.text is not None:
            self.text = _d(self.text)
        self.frames = [_d(b) for b in self.frames]

    # ── Batching across streams (eval slots) ────────────────────────────
    # The rotation / cache math is already batch-general (positions carry B),
    # so several same-layout streams can share one forward. We stack their
    # per-layer key/value tensors along the batch dim, step once, then slice
    # the updated blocks back out to each stream.

    def layout_key(self):
        """Identity of the cache layout: streams with equal keys can be batched.

        Two streams batch only if their text presence and per-frame block lengths
        match exactly, so the stacked tensors are rectangular and positions align.
        """
        text_len = self.text[2] if self.text is not None else None
        return (text_len, tuple(block[2] for block in self.frames))

    @classmethod
    def stack(cls, memories: List["FrameKVMemory"]) -> "FrameKVMemory":
        """Stack same-layout per-stream memories into one batched memory (dim 0)."""
        head = memories[0]
        keys = {m.layout_key() for m in memories}
        if len(keys) != 1:
            raise ValueError(f"cannot batch streams with differing layouts: {keys}")

        def _cat(blocks):
            k0_list, v_list, length = blocks[0]
            num_layers = len(k0_list)
            k0 = [torch.cat([b[0][layer] for b in blocks], dim=0) for layer in range(num_layers)]
            v = [torch.cat([b[1][layer] for b in blocks], dim=0) for layer in range(num_layers)]
            return (k0, v, length)

        batched = cls(head.rotary_emb, head.window, head.num_layers, head.text_config)
        if head.text is not None:
            batched.text = _cat([m.text for m in memories])
        for fi in range(len(head.frames)):
            batched.frames.append(_cat([m.frames[fi] for m in memories]))
        return batched

    def slice(self, index: int) -> "FrameKVMemory":
        """Extract stream ``index`` from a batched memory as a fresh per-stream memory."""
        def _slice(block):
            k0_list, v_list, length = block
            k0 = [k[index : index + 1] for k in k0_list]
            v = [vv[index : index + 1] for vv in v_list]
            return (k0, v, length)

        out = FrameKVMemory(self.rotary_emb, self.window, self.num_layers, self.text_config)
        if self.text is not None:
            out.text = _slice(self.text)
        out.frames = [_slice(b) for b in self.frames]
        return out


def memory_step(
    qwen_model,
    memory: "FrameKVMemory",
    *,
    new_input_ids: torch.Tensor,
    pixel_values: Optional[torch.Tensor] = None,
    image_grid_thw: Optional[torch.Tensor] = None,
    full_position_ids: torch.Tensor,
    frame_start_in_new: int,
    frame_len: int,
    prefill_text_len: int = 0,
) -> torch.Tensor:
    """One streaming step over a ``Qwen3VLModel`` backbone, using/updating ``memory``.

    Mirrors the proven text-model driver in tests/test_kv_memory_equivalence.py,
    lifted to the multimodal backbone so the current frame's pixels are encoded
    fresh while past frames come from the canonical KV memory.

    NOTE: the per-step segment construction that produces ``new_input_ids`` /
    ``pixel_values`` / ``full_position_ids`` (text-first window via the processor +
    get_rope_index) is done by the caller (QwenOFT) and needs end-to-end validation
    against the real Qwen3-VL-4B checkpoint. The cache/rotation math used here is
    unit-proven.

    Args:
        qwen_model: a ``Qwen3VLModel`` (``interface.model.model``).
        memory: the per-stream :class:`FrameKVMemory`.
        new_input_ids: ``[B, new_len]`` ids for the new tokens (current frame
            placeholders + action; plus the static text prefix on prefill).
        pixel_values / image_grid_thw: for the CURRENT frame only.
        full_position_ids: ``[3, B, full_len]`` M-RoPE positions for the whole
            visible window (text + cached frames + current frame + action).
        frame_start_in_new / frame_len: where the current frame block sits inside
            the new tokens (so its KV can be canonicalized into the memory).
        prefill_text_len: on the prefill step, the static text length at the front
            of the new tokens (cached once as the static prefix); 0 afterwards.

    Returns:
        ``[B, new_len, H]`` last hidden state for the new tokens.
    """
    past_len = memory.past_length()
    new_len = new_input_ids.shape[1]
    device = new_input_ids.device

    cache = memory.assemble_cache(full_position_ids[:, :, :past_len]) if past_len > 0 else DynamicCache(
        config=memory.text_config
    )
    cache_position = torch.arange(past_len, past_len + new_len, device=device)
    attention_mask = torch.ones(new_input_ids.shape[0], past_len + new_len, dtype=torch.long, device=device)

    out = qwen_model(
        input_ids=new_input_ids,
        position_ids=full_position_ids[:, :, past_len:],
        attention_mask=attention_mask,
        past_key_values=cache,
        use_cache=True,
        cache_position=cache_position,
        pixel_values=pixel_values,
        image_grid_thw=image_grid_thw,
    )

    # Canonicalize the static text prefix (prefill only) then the current frame.
    if prefill_text_len > 0:
        memory.set_text(
            memory.canonicalize_block(cache, past_len, prefill_text_len, full_position_ids[:, :, :prefill_text_len])
        )
    frame_abs_start = past_len + frame_start_in_new
    frame_positions = full_position_ids[:, :, frame_abs_start : frame_abs_start + frame_len]
    memory.add_frame(memory.canonicalize_block(cache, frame_abs_start, frame_len, frame_positions))

    return out.last_hidden_state
