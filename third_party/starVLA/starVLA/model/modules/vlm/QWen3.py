# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# Implemented by [Jinhui YE / HKUST University] in [2025].

import time
from collections import OrderedDict
from functools import wraps
from inspect import signature
from typing import Optional

import torch
from starVLA.model.profiling import stage_timer as _stage
from starVLA.training.trainer_utils import initialize_overwatch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from transformers.modeling_outputs import CausalLMOutputWithPast

logger = initialize_overwatch(__name__)

IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = 151655
VIDEO_TOKEN_INDEX = 151656
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_VIDEO_TOKEN = "<video>"

_ACTION_TOKEN_MIN = 151669  # how can we know this range? check how you add fast tokens into VLM
_ACTION_TOKEN_MAX = (
    153716  # here only for fast_tokenizer, see starVLA/model/modules/vlm/tools/add_qwen_special_tokens/README.md
)


import torch.nn as nn


def _patch_qwen3vl_flash_attention_position_ids() -> None:
    from transformers.models.qwen3_vl import modeling_qwen3_vl as qwen3_vl

    text_attention_cls = qwen3_vl.Qwen3VLTextAttention
    if hasattr(text_attention_cls, "_starvla_flash_attention_position_ids_patched"):
        return

    original_forward = text_attention_cls.forward

    def forward_without_flash_attention_position_ids(self, *args, **kwargs):
        # Qwen3-VL already applies M-RoPE through position_embeddings before
        # attention. Passing the repeated temporal M-RoPE ids into HF's FA2
        # wrapper makes Transformers 4.57 mis-detect packed sequences and build
        # empty cu_seqlens on KV-memory cache steps.
        if self.config._attn_implementation == "flash_attention_2" and "position_ids" in kwargs:
            del kwargs["position_ids"]
        return original_forward(self, *args, **kwargs)

    text_attention_cls.forward = forward_without_flash_attention_position_ids
    text_attention_cls._starvla_flash_attention_position_ids_patched = True


# The fixed Triton stage counts keep the PyTorch 2.11 CUDA 12.8 kernel
# within the shared-memory budget on sm_120.
_FLEX_KERNEL_OPTIONS = {
    "BACKEND": "TRITON",
    "fwd_num_stages": 2,
    "bwd_num_stages": 1,
}


def _patch_qwen3vl_flex_attention_support() -> None:
    """Allow loading Qwen3-VL with ``attn_implementation='flex_attention'``.

    Qwen3-VL's attention already dispatches through the generic
    ``ALL_ATTENTION_FUNCTIONS`` interface (which includes the ``flex_attention``
    integration), but the model classes conservatively declare
    ``_supports_flex_attn = False``, so Transformers 4.57 refuses to load with
    flex at __init__. Flipping the flag lets the official dispatch path run; the
    packed KV-memory training (`QwenOFT._forward_memory_packed`) passes a
    BlockMask that FlexAttention consumes natively.

    Also wraps the registered ``flex_attention`` interface to inject the fixed
    Triton kernel options. Wrapping the dispatch entry — rather than threading
    ``kernel_options`` through the top-level forward kwargs — guarantees both the
    forward and the gradient-checkpointing backward recompile pick it up.
    """
    from transformers.models.qwen3_vl import modeling_qwen3_vl as qwen3_vl

    for cls_name in (
        "Qwen3VLForConditionalGeneration",
        "Qwen3VLModel",
        "Qwen3VLTextModel",
        "Qwen3VLPreTrainedModel",
    ):
        cls = getattr(qwen3_vl, cls_name, None)
        if cls is not None:
            cls._supports_flex_attn = True

    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    base_flex = ALL_ATTENTION_FUNCTIONS["flex_attention"]
    if getattr(base_flex, "_starvla_kernel_options_patched", False):
        return

    def flex_attention_with_kernel_options(*args, **kwargs):
        kwargs["kernel_options"] = _FLEX_KERNEL_OPTIONS
        return base_flex(*args, **kwargs)

    flex_attention_with_kernel_options._starvla_kernel_options_patched = True
    ALL_ATTENTION_FUNCTIONS["flex_attention"] = flex_attention_with_kernel_options


def _position_tensor_key(tensor):
    if tensor is None:
        return None
    return (tensor.device, tensor.dtype, tuple(tensor.shape),
            tuple(tensor.detach().cpu().reshape(-1).tolist()))


def _cache_position_method(module, name, weight):
    """Cache layout-only outputs on one native model instance, never image features."""
    original = getattr(module, name)
    parameters = signature(original)
    cache = OrderedDict()

    @wraps(original)
    def cached(*args, **kwargs):
        if module.training or torch.is_grad_enabled():
            cache.clear()
            return original(*args, **kwargs)
        arguments = parameters.bind(*args, **kwargs)
        arguments.apply_defaults()
        parameter = weight()
        key = (
            parameter.device, parameter.dtype, parameter._version,
            tuple(_position_tensor_key(value) for value in arguments.arguments.values()),
        )
        if key not in cache:
            cache[key] = original(*args, **kwargs)
            if len(cache) > 8:
                cache.popitem(last=False)
        cache.move_to_end(key)
        return cache[key]

    setattr(module, name, cached)
    return cache


class _QWen3_VL_Interface(nn.Module):
    """
    This exists because of the diversity of VLMs, so we encapsulate the changes here.
    Lightweight wrapper around Qwen3-VL (Qwen3VLForConditionalGeneration).

    Purpose:
        - Unify interface with other VLM backends (CausalLM-like usage).
        - Centralize preprocessing (tokenization + multimodal packing).
        - Provide consistent forward / generate signatures.

    """

    def __init__(self, config: Optional[dict] = None, **kwargs):
        """
        Initialize the Qwen3-VL wrapper.
        Following https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct

        """
        super().__init__()

        qwenvl_config = config.framework.get("qwenvl", {})
        model_id = qwenvl_config.get("base_vlm", "Qwen/Qwen3-VL-4B-Instruct")
        attn_implementation = qwenvl_config.get("attn_implementation", "sdpa")
        enable_grad_ckpt = qwenvl_config.get("enable_gradient_checkpointing", False)
        print(
            f"[QWen3] loading {model_id} with gradient_checkpointing={enable_grad_ckpt}",
            flush=True,
        )

        if attn_implementation == "flash_attention_2":
            import flash_attn_2_cuda  # noqa: F401

            _patch_qwen3vl_flash_attention_position_ids()

        if attn_implementation == "flex_attention":
            _patch_qwen3vl_flex_attention_support()

        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_id,
            attn_implementation=attn_implementation,
            dtype=torch.bfloat16,
            ignore_mismatched_sizes=True, # resize image no longer needed? @TODO check bug
        )
        processor = AutoProcessor.from_pretrained(model_id)
        processor.tokenizer.padding_side = "left"

        if enable_grad_ckpt:
            model.config.text_config.use_cache = False
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
            model.enable_input_require_grads()
            print(
                "[QWen3] gradient_checkpointing ENABLED "
                f"(use_reentrant=False, active={model.is_gradient_checkpointing}, "
                f"text_use_cache={model.config.text_config.use_cache})",
                flush=True,
            )

        self.model = model
        self.processor = processor
        self.config = config
        # align qwen3 with qwen2.5
        self.model.config.hidden_size = self.model.config.text_config.hidden_size
        self._last_build_timing = {}
        visual = model.model.visual
        self._position_caches = [
            _cache_position_method(model.model, "get_rope_index", lambda: visual.rotary_pos_emb.inv_freq),
            _cache_position_method(visual, "rot_pos_emb", lambda: visual.rotary_pos_emb.inv_freq),
            _cache_position_method(visual, "fast_pos_embed_interpolate", lambda: visual.pos_embed.weight),
        ]

        # only for fast base model
        if "-Action" in model_id:
            self._ACTION_TOKEN_MIN = _ACTION_TOKEN_MIN
            self._ACTION_TOKEN_MAX = _ACTION_TOKEN_MAX

    def _clear_position_caches(self):
        for cache in self._position_caches:
            cache.clear()

    def train(self, mode=True):
        self._clear_position_caches()
        return super().train(mode)

    def _apply(self, fn, recurse=True):
        self._clear_position_caches()
        return super()._apply(fn, recurse=recurse)

    def _load_from_state_dict(self, *args, **kwargs):
        self._clear_position_caches()
        return super()._load_from_state_dict(*args, **kwargs)

    def _profile_timing_enabled(self) -> bool:
        return self.config.trainer.profile_timing.enabled

    def _profile_sync(self) -> None:
        if self._profile_timing_enabled() and torch.cuda.is_available():
            torch.cuda.synchronize()

    def forward(
        self,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        """
        Forward pass delegating to underlying Qwen2.5-VL backbone.
        """

        with torch.autocast("cuda", dtype=torch.bfloat16):
            outputs = self.model(
                **kwargs,
            )

        return outputs

    def forward_last_hidden(self, **kwargs) -> torch.Tensor:
        """
        Run the Qwen3-VL backbone and return only the final hidden state.

        OFT-style action heads only consume the last hidden layer. Calling the
        full conditional-generation wrapper with ``output_hidden_states=True``
        returns every layer's activations, which can largely erase the memory
        benefit of gradient checkpointing.
        """

        kwargs = dict(kwargs)
        kwargs.pop("output_hidden_states", None)
        kwargs.pop("output_attentions", None)
        kwargs.setdefault("return_dict", True)
        kwargs.setdefault("use_cache", False)

        def _wrapper_forward_last_hidden() -> torch.Tensor:
            lm_head = getattr(self.model, "lm_head", None)
            captured = {}
            handle = None
            if lm_head is not None:
                def _capture_lm_head_input(_module, inputs):
                    captured["last_hidden"] = inputs[0]

                handle = lm_head.register_forward_pre_hook(_capture_lm_head_input)

            with torch.autocast("cuda", dtype=torch.bfloat16):
                try:
                    outputs = self.model(
                        **kwargs,
                        output_attentions=False,
                        output_hidden_states=False,
                        return_dict=True,
                    )
                finally:
                    if handle is not None:
                        handle.remove()

            if "last_hidden" in captured:
                return captured["last_hidden"]

            # Last-resort compatibility path for unusual model wrappers that do
            # not expose lm_head. This preserves correctness, but returns every
            # layer and therefore saves less memory.
            with torch.autocast("cuda", dtype=torch.bfloat16):
                outputs = self.model(
                    **kwargs,
                    output_attentions=False,
                    output_hidden_states=True,
                    return_dict=True,
                )
            return outputs.hidden_states[-1]

        backbone = getattr(self.model, "model", None)
        if backbone is None:
            return _wrapper_forward_last_hidden()

        try:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                outputs = backbone(**kwargs)
        except TypeError:
            # Some Transformers Qwen-VL wrappers inject visual embeddings before
            # calling the text backbone, so the backbone itself may not accept
            # pixel_values/image_grid_thw. Keep those versions working.
            return _wrapper_forward_last_hidden()

        if hasattr(outputs, "last_hidden_state"):
            return outputs.last_hidden_state
        return outputs[0]

    def generate(
        self,
        **kwargs,
    ):
        """
        High-level generation interface (auto-regressive decoding), optionally vision-conditioned.

        Args:
            **kwargs: fully follow raw model.generate() signature.
        Returns:
            GenerateOutput | Model-dependent generation return.
        """
        with torch.autocast("cuda", dtype=torch.float16):
            generation_output = self.model.generate(
                **kwargs,
            )
        return generation_output

    def build_qwenvl_inputs(self, images, instructions, solutions=None, profiler=None, **kwargs):
        """
        Build model inputs from raw data (images + instructions + optional solutions).
        Follow Oficial Qwen3-VL Instruct format: https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct
        """

        # Create messages: one message per sample
        with _stage(profiler, "starvla_qwen_message_build_ms"):
            messages = []
            assert len(images) == len(instructions), "Images and instructions must have the same length"
            for imgs, instruction in zip(images, instructions):
                content = [{"type": "image", "image": img} for img in imgs]

                if "CoT_prompt" in self.config.datasets.vla_data:  # If using a grounding prompt to task
                    CoT_prompt = self.config.datasets.vla_data.get("CoT_prompt", "")
                    prompt = CoT_prompt.replace("{instruction}", instruction)
                else:
                    prompt = instruction

                content.append({"type": "text", "text": prompt})
                msg = [{"role": "user", "content": content}]

                if solutions is not None:
                    solution = solutions[len(messages)]
                    msg.append({"role": "assistant", "content": [{"type": "text", "text": solution}]})
                messages.append(msg)

        profile_timing = self._profile_timing_enabled()
        build_timing = {}

        t_processor = time.perf_counter() if profile_timing else None
        with _stage(profiler, "starvla_qwen_processor_ms"):
            batch_inputs = self.processor.apply_chat_template(
                messages, tokenize=True, padding=True, add_generation_prompt=True, return_dict=True, return_tensors="pt"
            )
        if profile_timing:
            build_timing["timing/qwen_processor"] = time.perf_counter() - t_processor

        # if solutions, mask out the solution tokens in labels
        if solutions is not None:  #  here only for fast_tokenizer now.
            action_token_min = _ACTION_TOKEN_MIN  # how can we know this range? --> we has other way for this, but is slower see qwenhelix branch
            action_token_max = _ACTION_TOKEN_MAX  # here only for fast_tokenizer, see starVLA/model/modules/vlm/tools/add_qwen_special_tokens/README.md
            labels = batch_inputs["input_ids"].clone()
            # For each sequence in the batch, find the first occurrence of an action token.
            for i in range(labels.size(0)):
                seq = labels[i]
                # Create a mask for tokens within the action token range.
                mask_seq = (seq >= action_token_min) & (seq <= action_token_max)
                nonzero_indices = torch.nonzero(mask_seq, as_tuple=False)
                if nonzero_indices.numel() > 0:
                    first_action_index = nonzero_indices[0].item()
                    # Mask out all tokens before the first action token.
                    seq[:first_action_index] = IGNORE_INDEX
                else:
                    # If no action token is found, mask the entire sequence.
                    seq[:] = IGNORE_INDEX
                    RuntimeWarning(
                        "action token are on in yout tokenizer, plz see starVLA/model/modules/vlm/tools/add_qwen_special_tokens/README.md."
                    )

            labels[labels == self.processor.tokenizer.pad_token_id] = -100  ## mask out pad tokens as well
            batch_inputs["labels"] = labels

        t_h2d = time.perf_counter() if profile_timing else None
        with _stage(profiler, "starvla_qwen_h2d_ms"):
            batch_inputs = batch_inputs.to(self.model.device)
        if profile_timing:
            self._profile_sync()
            build_timing["timing/qwen_h2d"] = time.perf_counter() - t_h2d
        self._last_build_timing = build_timing
        return batch_inputs


if __name__ == "__main__":
    import argparse
    import os

    from omegaconf import OmegaConf

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_yaml",
        type=str,
        default="examples/SimplerEnv/train_files/starvla_cotrain_oxe.yaml",
        help="Path to YAML config",
    )
    args, clipargs = parser.parse_known_args()

    if os.getenv("DEBUGPY_ENABLE", "0") == "1":
        import debugpy
        debugpy.listen(("0.0.0.0", 10092))
        print("Rank 0 waiting for debugger attach on port 10092...")
        debugpy.wait_for_client()

    cfg = OmegaConf.load(args.config_yaml)

    cfg.framework.qwenvl.base_vlm = "./playground/Pretrained_models/Qwen3-VL-4B-Instruct"
    qwen_vl = _QWen3_VL_Interface(cfg)
    pass
