# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# Implemented by [Jinhui YE / HKUST University] in [2025].

"""
Qwen-OFT Framework

A lightweight implementation that uses an action special token to parallelly predict continuous actions
conditioned on multi-view images plus a language instruction (shares parameters with the VLM).
Inspired by OpenVLA-OFT
Key Points:
  - Qwen2.5 vision-language backbone
  - Injects an action special token into the VLM
  - Continuous action prediction via L1 regression over the action special token hidden states


Note: How to add special tokens to Qwen2.5:
  download our model checkpoint with special tokens added: https://huggingface.co/StarVLA/Qwen2.5-VL-3B-Instruct-Action
  or /starVLA/model/modules/vlm/tools/add_qwen_special_tokens/README.md （adpat a little code)

"""

from dataclasses import dataclass, field
import time
from typing import Any, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from deployment.model_server.tools.image_tools import to_pil_preserve
from starVLA.model.tools import FRAMEWORK_REGISTRY
from starVLA.training.trainer_utils import initialize_overwatch

logger = initialize_overwatch(__name__)

# HuggingFace Default / LLaMa-2 IGNORE_INDEX (for labels)
IGNORE_INDEX = -100

from starVLA.model.framework.base_framework import baseframework
from starVLA.model.framework.share_tools import add_discretized_state_to_instruction, merge_framework_config
from starVLA.model.modules.action_model.MLP_ActionHeader import get_action_model
from starVLA.model.modules.vlm import get_vlm_model
from starVLA.training.trainer_utils.trainer_tools import resize_images


# ──────────────────────────────────────────────────────────────────────
#  Default Config for QwenOFT
#  - Documents every framework-level parameter with type + description
#  - YAML values override these defaults; extra YAML keys are preserved
# ──────────────────────────────────────────────────────────────────────
@dataclass
class QwenOFTDefaultConfig:
    """QwenOFT framework default parameters.

    All fields can be overridden by the corresponding key in the YAML
    ``framework:`` section.  Extra YAML keys not listed here are kept
    as-is (Config-as-API flexibility).
    """

    # --- Registry identifier (must match @FRAMEWORK_REGISTRY.register) ---
    name: str = "QwenOFT"

    # === VLM backbone (Qwen2.5-VL / Qwen3-VL) ===
    qwenvl: dict = field(
        default_factory=lambda: {
            # Path to base VLM checkpoint (local or HF hub id)
            "base_vlm": "./playground/Pretrained_models/Qwen3-VL-4B-Instruct-Action",
            # Attention implementation: "flash_attention_2" | "eager" | "sdpa"
            "attn_implementation": "flash_attention_2",
        }
    )

    # === Action head (MLP regression over action special tokens) ===
    action_model: dict = field(
        default_factory=lambda: {
            # Action head architecture type
            "action_model_type": "MLP",
            # Dimensionality of each action vector (e.g., 7 for 6-DoF + gripper)
            "action_dim": 7,
            # Proprioceptive state dimension for continuous_projector mode
            "state_dim": 7,
            # Hidden dim for the action MLP (auto-set from VLM hidden_size at runtime)
            "action_hidden_dim": 2560,
            # How many future steps to predict
            "future_action_window_size": 8,
            # How many past steps included in action chunk (usually 0)
            "past_action_window_size": 0,
            # Loss over predicted actions. Supported: "l1", "discrete_ce",
            # "multibinary_bce", "asterix_factorized_ce".
            # "multibinary_ce" is accepted as a CLI alias for "multibinary_bce".
            "loss_type": "l1",
            # State conditioning: preserve the existing text-bin path by default.
            "state_encoding": "discretized_text",
        }
    )


@FRAMEWORK_REGISTRY.register("QwenOFT")
class Qwenvl_OFT(baseframework):
    """
    Multimodal vision-language-action model (OFT variant).

    Components:
      - Qwen2.5-VL / Qwen3-VL backbone for fused language/vision token embeddings
      - Action special token injected into the VLM sequence
      - MLP regression head over action token hidden states (L1 loss)

    Focus: Predict future continuous actions conditioned on images + instruction.
    """

    def __init__(
        self,
        config: Optional[dict] = None,
        **kwargs,
    ) -> None:
        """
        Construct all submodules and cache key configuration values.

        Args:
            config: Hierarchical configuration (OmegaConf/dict) containing framework + trainer sections.
            **kwargs: Reserved for future overrides (unused).
        """
        super().__init__()
        # Merge framework defaults with YAML config (YAML wins on conflicts)
        self.config = merge_framework_config(QwenOFTDefaultConfig, config)
        self.qwen_vl_interface = get_vlm_model(config=self.config)
        # align action_hidden_dim to VLM hidden_size at runtime
        self.config.framework.action_model.action_hidden_dim = self.qwen_vl_interface.model.config.hidden_size
        self.action_model = get_action_model(config=self.config)

        # `action_horizon` is the single source of truth for chunk length.
        # Legacy aliases (`future_action_window_size`, `past_action_window_size`)
        # are normalised upstream by `share_tools.apply_config_compat`, so we
        # only ever read `action_horizon` here.
        self.action_horizon = int(self.config.framework.action_model.action_horizon)
        self.chunk_len = self.action_horizon
        self.action_dim = int(self.config.framework.action_model.action_dim)
        self.action_env_dim = int(getattr(self.config.framework.action_model, "action_env_dim", self.action_dim))
        self.action_loss_type = str(getattr(self.config.framework.action_model, "loss_type", "l1")).lower()
        self.state_encoding = str(self.config.framework.action_model.state_encoding)
        if self.state_encoding not in {"discretized_text", "continuous_projector"}:
            raise ValueError(f"Unsupported state_encoding: {self.state_encoding}")
        if self.state_encoding == "continuous_projector":
            self.action_model.state_projector = nn.Linear(
                int(self.config.framework.action_model.state_dim),
                int(self.config.framework.action_model.action_hidden_dim),
            )
        cross_task_cfg = getattr(getattr(self.config, "rl_games", None), "cross_task", None)
        self.loss_by_task = self._to_plain_dict(getattr(cross_task_cfg, "loss_by_task", None))
        self.loss_weight_by_task = self._to_plain_dict(getattr(cross_task_cfg, "loss_weight_by_task", None))
        # self.hidden_dim = config.framework.action_model.action_hidden_dim

        self.action_token = "🔍"  # TODO also can add spacail token to Qwen, but too complex
        self.action_token_id = self.qwen_vl_interface.processor.tokenizer("🔍", add_special_tokens=False)["input_ids"][0]

        # --- Fixed-size KV memory; default OFF so existing paths are untouched. ---
        kv_cfg = getattr(self.config.framework, "kv_memory", None)
        self.kv_memory_enabled = bool(getattr(kv_cfg, "enabled", False)) if kv_cfg is not None else False
        self.kv_window = int(getattr(kv_cfg, "window", 4)) if kv_cfg is not None else 4
        self.kv_rollout_len = int(getattr(kv_cfg, "rollout_len", max(self.kv_window + 1, 2))) if kv_cfg is not None else self.kv_window + 1
        # Packed single-forward training path (StreamingLLM-style sliding-window+sink
        # mask via FlexAttention) — replaces the per-frame Python loop. Eval/live still
        # uses the streaming KV memory (`_predict_action_memory`). OFF by default so the
        # legacy loop stays the reference; enable per experiment via
        # `framework.kv_memory.packed_train: true`.
        self.kv_train_packed = bool(getattr(kv_cfg, "packed_train", False)) if kv_cfg is not None else False
        # Per-step re-based sink for the packed path: replicate the (short) text
        # prefix once per step and re-base its M-RoPE positions so the sink→frame
        # distance matches the eval rolling-KV layout exactly (StreamingLLM
        # "positions within the cache"), instead of one global sink whose distance
        # grows with the step index. ON by default — the growing-distance sink was
        # the train/eval forward mismatch behind the packed run's eval collapse.
        # Set false to fall back to the single global sink for A/B.
        self.kv_rebased_sink = bool(getattr(kv_cfg, "rebased_sink", True)) if kv_cfg is not None else True
        if self.state_encoding == "continuous_projector" and self.kv_memory_enabled:
            raise ValueError("continuous_projector requires framework.kv_memory.enabled=false")
        # Per-stream (eval slot) memories; rebuilt per sample during training.
        self._kv_memories: dict = {}

        # L1 loss
        self.l1_loss = nn.L1Loss()

    def _profile_timing_enabled(self) -> bool:
        return self.config.trainer.profile_timing.enabled

    def _profile_sync(self) -> None:
        if self._profile_timing_enabled() and torch.cuda.is_available():
            torch.cuda.synchronize()

    def _profile_start(self) -> float:
        self._profile_sync()
        return time.perf_counter()

    def _profile_elapsed(self, start: float) -> float:
        self._profile_sync()
        return time.perf_counter() - start

    @staticmethod
    def _to_plain_dict(value: Any) -> dict[str, Any]:
        if value is None or value == "":
            return {}
        if isinstance(value, dict):
            return {str(key): item for key, item in value.items()}
        if hasattr(value, "items"):
            return {str(key): item for key, item in value.items()}
        return {}

    def _prepare_action_array(self, action: Any) -> np.ndarray:
        values = np.asarray(action)
        if values.ndim == 1:
            values = values[None, :]
        if values.shape[-1] == self.action_dim:
            return values
        if values.shape[-1] > self.action_dim:
            return values[..., : self.action_dim]
        pad_width = [(0, 0)] * values.ndim
        pad_width[-1] = (0, self.action_dim - values.shape[-1])
        return np.pad(values, pad_width, mode="constant")

    def _compute_action_loss_for_type(
        self,
        pred_actions: torch.Tensor,
        actions_target: torch.Tensor,
        *,
        loss_type: str,
        action_env_dims: Optional[List[int]] = None,
    ) -> torch.Tensor:
        loss_type = str(loss_type).lower()
        effective_dim = min(self.action_env_dim, pred_actions.shape[-1], actions_target.shape[-1])
        if effective_dim <= 0:
            raise ValueError(
                f"Invalid action_env_dim={self.action_env_dim} for predicted shape={pred_actions.shape} "
                f"and target shape={actions_target.shape}"
            )

        if loss_type in {"discrete_ce", "ce", "cross_entropy"}:
            if action_env_dims is not None:
                dims = torch.as_tensor(action_env_dims, device=pred_actions.device, dtype=torch.long)
                if dims.numel() != pred_actions.shape[0]:
                    raise ValueError(f"Expected {pred_actions.shape[0]} action_env_dims, got {dims.numel()}")
                losses = []
                weights = []
                for dim_value in torch.unique(dims).tolist():
                    dim = int(min(dim_value, pred_actions.shape[-1], actions_target.shape[-1]))
                    if dim < 2:
                        raise ValueError(f"discrete_ce requires at least 2 classes, got action_env_dim={dim}")
                    mask = dims == int(dim_value)
                    logits = pred_actions[mask, ..., :dim]
                    target_class = actions_target[mask, ..., :dim].argmax(dim=-1).long()
                    losses.append(F.cross_entropy(logits.reshape(-1, dim), target_class.reshape(-1), reduction="sum"))
                    weights.append(target_class.numel())
                total_weight = max(1, int(sum(weights)))
                return sum(losses) / total_weight

            if effective_dim < 2:
                raise ValueError(
                    f"action_model.loss_type={loss_type!r} requires at least 2 action classes, "
                    f"got effective_dim={effective_dim}"
                )
            logits = pred_actions[..., :effective_dim]
            target_class = actions_target[..., :effective_dim].argmax(dim=-1).long()
            return F.cross_entropy(
                logits.reshape(-1, effective_dim),
                target_class.reshape(-1),
            )

        if loss_type in {"asterix_factorized_ce", "factorized_6_ce", "factorized_ce"}:
            if effective_dim < 6:
                raise ValueError(
                    f"action_model.loss_type={loss_type!r} requires at least 6 action logits, "
                    f"got effective_dim={effective_dim}"
                )
            vertical_logits = pred_actions[..., 0:3]
            vertical_target = actions_target[..., 0:3].argmax(dim=-1).long()
            horizontal_logits = pred_actions[..., 3:6]
            horizontal_target = actions_target[..., 3:6].argmax(dim=-1).long()
            vertical_loss = F.cross_entropy(
                vertical_logits.reshape(-1, 3),
                vertical_target.reshape(-1),
            )
            horizontal_loss = F.cross_entropy(
                horizontal_logits.reshape(-1, 3),
                horizontal_target.reshape(-1),
            )
            return 0.5 * (vertical_loss + horizontal_loss)

        if loss_type in {"multibinary_bce", "multibinary_ce", "bce", "binary_cross_entropy"}:
            logits = pred_actions[..., :effective_dim]
            targets = (actions_target[..., :effective_dim] > 0).to(dtype=logits.dtype)
            return F.binary_cross_entropy_with_logits(logits, targets)

        if loss_type in {"mse", "l2"}:
            return F.mse_loss(
                pred_actions[..., :effective_dim],
                actions_target[..., :effective_dim],
            )

        if loss_type not in {"l1", "mae"}:
            raise ValueError(
                f"Unsupported action_model.loss_type={loss_type!r}; "
                "expected one of: l1, discrete_ce, multibinary_bce, asterix_factorized_ce"
            )

        return self.l1_loss(
            pred_actions[..., :effective_dim],
            actions_target[..., :effective_dim],
        )

    def _compute_action_loss(
        self,
        pred_actions: torch.Tensor,
        actions_target: torch.Tensor,
        action_env_dims: Optional[List[int]] = None,
        rl_games_tasks: Optional[List[str]] = None,
        action_loss_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if action_loss_mask is not None:
            error = (pred_actions[..., :self.action_env_dim] - actions_target[..., :self.action_env_dim]).abs().mean(dim=-1)
            error = error.masked_fill(~action_loss_mask, 0)
            return (error.sum(dim=-1) / action_loss_mask.sum(dim=-1)).mean()
        if self.loss_by_task and rl_games_tasks is not None:
            if len(rl_games_tasks) != pred_actions.shape[0]:
                raise ValueError(f"Expected {pred_actions.shape[0]} task labels, got {len(rl_games_tasks)}")
            losses = []
            weights = []
            for task_name in sorted(set(str(task) for task in rl_games_tasks)):
                mask_values = [str(task) == task_name for task in rl_games_tasks]
                mask = torch.as_tensor(mask_values, device=pred_actions.device, dtype=torch.bool)
                if not bool(mask.any()):
                    continue
                task_loss_type = str(self.loss_by_task.get(task_name, self.action_loss_type)).lower()
                task_dims = (
                    [int(dim) for dim, keep in zip(action_env_dims, mask_values) if keep]
                    if action_env_dims is not None
                    else None
                )
                task_loss = self._compute_action_loss_for_type(
                    pred_actions[mask],
                    actions_target[mask],
                    loss_type=task_loss_type,
                    action_env_dims=task_dims,
                )
                task_weight = float(self.loss_weight_by_task.get(task_name, 1.0))
                sample_count = int(mask.sum().item())
                losses.append(task_loss * task_weight * sample_count)
                weights.append(task_weight * sample_count)
            if losses:
                return sum(losses) / max(1.0, float(sum(weights)))

        return self._compute_action_loss_for_type(
            pred_actions,
            actions_target,
            loss_type=self.action_loss_type,
            action_env_dims=action_env_dims,
        )

    def _forward_qwen_last_hidden(self, qwen_inputs: dict) -> torch.Tensor:
        """
        Return only the final VLM hidden state when the interface supports it.

        Requesting ``output_hidden_states=True`` from Hugging Face materializes
        every layer's activations. For OFT we only need the last layer, so the
        Qwen3 interface exposes a leaner path that preserves the practical memory
        savings from gradient checkpointing.
        """

        if hasattr(self.qwen_vl_interface, "forward_last_hidden"):
            return self.qwen_vl_interface.forward_last_hidden(**qwen_inputs)

        qwenvl_outputs = self.qwen_vl_interface(
            **qwen_inputs,
            output_attentions=False,
            output_hidden_states=True,
            return_dict=True,
            use_cache=False,
        )
        return qwenvl_outputs.hidden_states[-1]

    def _state_conditioned_instructions(self, instructions, state):
        if state is not None and self.state_encoding == "discretized_text":
            return self.add_discretized_state_to_instruction(instructions, state)
        return instructions

    def _condition_action_queries(self, action_queries, state):
        if self.state_encoding == "discretized_text":
            return action_queries
        state_tensor = torch.as_tensor(
            np.stack(state), device=action_queries.device, dtype=action_queries.dtype
        ).squeeze(1)
        return action_queries + self.action_model.state_projector(state_tensor).unsqueeze(1)

    def forward(
        self,
        examples: List[dict] = None,
        **kwargs,
    ) -> Tuple:
        """
        Training forward: directly regress future actions (no diffusion).

        Flow:
          1. Build QwenVL inputs (images + instruction tokens)
          2. Extract hidden states from configured layer range
          7. Predict action and compute L1 loss

        Args:
            examples: List[dict], each dict requires:
                - image: List[PIL.Image] (multi-view)
                - lang: str instruction
                - action: np.ndarray or list shaped [T, action_dim]
            **kwargs: Reserved.

        Returns:
            dict:
                action_loss (torch.Tensor): Scalar diffusion noise prediction loss.
        """
        if self.kv_memory_enabled:
            if self.kv_train_packed:
                return self._forward_memory_packed(examples)
            return self._forward_memory(examples)

        batch_images = [example["image"] for example in examples]  #  [B，[PLT]]
        instructions = [example["lang"] for example in examples]  # [B, str]
        actions = [self._prepare_action_array(example["action"]) for example in examples]  # label [B， len, action_dim]
        action_env_dims = [int(example.get("action_env_dim", self.action_env_dim)) for example in examples]
        rl_games_tasks = [str(example.get("rl_games_task", "")) for example in examples]
        if not any(rl_games_tasks):
            rl_games_tasks = None
        state = (
            [example["state"] for example in examples] if "state" in examples[0] else None
        )  # List[ndarray (1, state_dim)] or None

        instructions = self._state_conditioned_instructions(instructions, state)

        # step 0: add special action token to instruction
        action_tokens = (
            self.action_token * self.chunk_len
        )  # can't add " " between two tokens, otherwise will be tokenized to multiple tokens
        prompt_suffix = f" Please predict the next {self.chunk_len} robot actions: <action>{action_tokens}<action>."
        instructions = [instruction + prompt_suffix for instruction in instructions]

        profile_timing = self._profile_timing_enabled()
        timing_metrics = {}
        batch_stats = {}

        # Step 1: QWenVL input format
        t_inputs = self._profile_start() if profile_timing else None
        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(images=batch_images, instructions=instructions)
        if profile_timing:
            timing_metrics["timing/qwen_input_build_total"] = self._profile_elapsed(t_inputs)
            timing_metrics.update(self.qwen_vl_interface._last_build_timing)
            input_ids = qwen_inputs["input_ids"]
            attention_mask = qwen_inputs["attention_mask"]
            effective_tokens = attention_mask.sum()
            effective_token_count = effective_tokens.detach().float().item()
            batch_stats["batch/size"] = float(len(examples))
            batch_stats["batch/input_len_max"] = float(input_ids.shape[1])
            batch_stats["batch/input_len_mean"] = float(effective_token_count / input_ids.shape[0])
            batch_stats["batch/padding_ratio"] = float(1.0 - effective_token_count / float(attention_mask.numel()))
            batch_stats["batch/effective_tokens"] = float(effective_token_count)
            batch_stats["batch/image_count"] = float(sum(len(images) for images in batch_images))
            if "pixel_values" in qwen_inputs:
                batch_stats["batch/pixel_values_rows"] = float(qwen_inputs["pixel_values"].shape[0])

        t_vlm = self._profile_start() if profile_timing else None
        with torch.autocast("cuda", dtype=torch.bfloat16):
            last_hidden = self._forward_qwen_last_hidden(qwen_inputs)  # [B, L, H]
        if profile_timing:
            timing_metrics["timing/vlm_forward"] = self._profile_elapsed(t_vlm)

        # Step 4: Action Expert Forward and Loss
        t_action = self._profile_start() if profile_timing else None
        with torch.autocast("cuda", dtype=torch.float32):
            # Extract action token embeddings as action prediction queries
            input_ids = qwen_inputs["input_ids"]
            action_queries = self._gather_action_token_embeddings(
                last_hidden, input_ids, action_token_id=self.action_token_id
            )  # [B, chunk_len, H]
            action_queries = self._condition_action_queries(action_queries, state)
            pred_actions = self.action_model.predict_action(action_queries)  # (B, chunk_len, action_dim)

            # Label alignment: take the last chunk_len segment
            actions = torch.tensor(
                np.array(actions), device=pred_actions.device, dtype=pred_actions.dtype
            )  # [B, T_full, action_dim]
            actions_target = actions[:, -self.action_horizon :, :]  # (B, action_horizon, action_dim)

            action_loss_mask = None
            if "action_loss_mask" in examples[0]:
                action_loss_mask = torch.as_tensor(
                    np.asarray([example["action_loss_mask"] for example in examples]),
                    device=pred_actions.device,
                    dtype=torch.bool,
                )

            action_loss = self._compute_action_loss(
                pred_actions,
                actions_target,
                action_env_dims=action_env_dims,
                rl_games_tasks=rl_games_tasks,
                action_loss_mask=action_loss_mask,
            )
        if profile_timing:
            timing_metrics["timing/action_head_loss"] = self._profile_elapsed(t_action)

        return {"action_loss": action_loss, "loss_weight": float(len(examples)), "timing": timing_metrics, "batch_stats": batch_stats}

    @torch.inference_mode()
    def predict_action(
        self,
        examples: List[dict] = None,
        **kwargs: str,
    ) -> np.ndarray:
        """

        Steps:
          1. Resize images to training resolution (if specified)
          2. Encode with QwenVL (hidden states retained)
          6. Return normalized action trajectory

        Returns:
            dict:
                normalized_actions (np.ndarray): Shape [B, T, action_dim], diffusion-sampled normalized actions.
        """
        if type(examples) is not list:
            examples = [examples]
        if self.kv_memory_enabled:
            return self._predict_action_memory(examples)
        batch_images = [to_pil_preserve(example["image"]) for example in examples]  #  [B，[PLT]]
        instructions = [example["lang"] for example in examples]  # [B, str]
        state = (
            [example["state"] for example in examples] if "state" in examples[0] else None
        )  # List[ndarray (1, state_dim)] or None

        instructions = self._state_conditioned_instructions(instructions, state)

        train_obs_image_size = getattr(self.config.datasets.vla_data, "obs_image_size", None)
        if train_obs_image_size:
            batch_images = resize_images(batch_images, target_size=train_obs_image_size)

        # step 0: add special action token to instruction
        action_tokens = (
            self.action_token * self.chunk_len
        )  # can't add " " between two tokens, otherwise will be tokenized to multiple tokens
        prompt_suffix = f" Please predict the next {self.chunk_len} robot actions: <action>{action_tokens}<action>."
        instructions = [instruction + prompt_suffix for instruction in instructions]

        # Step 1: QWenVL input format
        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(images=batch_images, instructions=instructions)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            last_hidden = self._forward_qwen_last_hidden(qwen_inputs)  # [B, L, H]

        # Step 4: Action Expert Forward and Loss
        with torch.autocast("cuda", dtype=torch.float32):
            # Extract action token embeddings as action prediction queries
            input_ids = qwen_inputs["input_ids"]
            action_queries = self._gather_action_token_embeddings(
                last_hidden, input_ids, action_token_id=self.action_token_id
            )  # [B, chunk_len, H]
            action_queries = self._condition_action_queries(action_queries, state)
            pred_actions = self.action_model.predict_action(action_queries)  # (B, chunk_len, action_dim)

        normalized_actions = pred_actions.detach().cpu().numpy()
        return {"normalized_actions": normalized_actions}

    # ──────────────────────────────────────────────────────────────────
    #  Fixed-size KV memory. Default OFF. The cache/rotation core is
    #  unit-proven (tests/test_kv_memory_equivalence.py); the processor-driven
    #  segment construction below still needs end-to-end validation against the
    #  real Qwen3-VL-4B checkpoint on GPU.
    # ──────────────────────────────────────────────────────────────────
    def reset_memory(self, slot_id=None) -> None:
        """Clear the KV memory for one eval slot (episode boundary), or all."""
        if slot_id is None:
            self._kv_memories.clear()
        else:
            self._kv_memories.pop(slot_id, None)

    def _kv_components(self):
        hf = self.qwen_vl_interface.model  # Qwen3VLForConditionalGeneration
        backbone = hf.model  # Qwen3VLModel
        text_model = backbone.language_model  # Qwen3VLTextModel
        return backbone, text_model.rotary_emb, len(text_model.layers), text_model.config

    def _new_kv_memory(self):
        from starVLA.model.modules.vlm.kv_memory import FrameKVMemory

        _, rotary_emb, num_layers, text_config = self._kv_components()
        return FrameKVMemory(rotary_emb, self.kv_window, num_layers, text_config)

    def _build_text_first_inputs(self, frames, instruction):
        """Tokenize a text-first window: [instruction][frame imgs...][action suffix].

        Returns the processor batch (input_ids, pixel_values, image_grid_thw, ...).
        Static text comes first so its KV stays positionally stable across steps.
        """
        action_tokens = self.action_token * self.chunk_len
        action_suffix = f" Please predict the next {self.chunk_len} robot actions: <action>{action_tokens}<action>."
        content = [{"type": "text", "text": instruction}]
        content += [{"type": "image", "image": img} for img in frames]
        content += [{"type": "text", "text": action_suffix}]
        messages = [[{"role": "user", "content": content}]]
        batch = self.qwen_vl_interface.processor.apply_chat_template(
            messages, tokenize=True, padding=True, add_generation_prompt=True, return_dict=True, return_tensors="pt"
        )
        return batch.to(self.qwen_vl_interface.model.device)

    def _kv_segments(self, input_ids_1d):
        """Partition a text-first window's ids into (prefix_len, [frame_lens], action_len).

        A frame block spans its vision_start .. vision_end (inclusive); the prefix is
        everything before the first vision_start; the action suffix is everything
        after the last vision_end. Using the real special-token ids keeps eviction
        leak-free (each evicted frame removes exactly its own markers).
        """
        cfg = self.qwen_vl_interface.model.config
        vstart = int(getattr(cfg, "vision_start_token_id", 151652))
        vend = int(getattr(cfg, "vision_end_token_id", 151653))
        ids = input_ids_1d.tolist()
        starts = [i for i, t in enumerate(ids) if t == vstart]
        ends = [i for i, t in enumerate(ids) if t == vend]
        if not starts or not ends:
            raise RuntimeError("KV memory: window inputs have no vision_start/end tokens")
        action_start = ends[-1] + 1
        bounds = starts + [action_start]
        frame_lens = [bounds[i + 1] - bounds[i] for i in range(len(starts))]
        return starts[0], frame_lens, len(ids) - action_start

    def _kv_forward_step_batched(self, memories, frames, instruction):
        """Run ONE streaming KV-memory step for a group of same-layout streams and
        return ``(last_hidden_new [B, new_len, H], new_input_ids [B, new_len])``.

        ``memories`` and ``frames`` are aligned lists (one entry per stream / eval
        slot); all entries share ``instruction`` and history layout (grouped by the
        caller). Two efficiencies vs a naive per-frame, per-step encode:

        - Single-frame preprocessing: the image processor runs once per stream on its
          OWN current frame. The window's multi-frame token layout (needed only for
          M-RoPE positions via ``get_rope_index``) is reconstructed by replicating the
          frame's token block, so duplicate frames are never re-preprocessed.
        - Cross-stream batching: the streams' canonical KV blocks are stacked along the
          batch dim, so the whole group is one forward; results are sliced back to each
          stream's memory afterwards.

        Past frames come from each stream's memory; only the current frame's pixels are
        encoded fresh. ``B == 1`` (training rollout) reuses the same path with no
        stacking, so gradients flow through the in-place memory updates.
        """
        from starVLA.model.modules.vlm.kv_memory import FrameKVMemory, memory_step

        backbone = self.qwen_vl_interface.model.model  # Qwen3VLModel
        B = len(memories)
        n_past = memories[0].num_past_frames()
        n_visible = n_past + 1

        # One processor call per stream's current frame (no duplicate-frame work).
        per_stream = [self._build_text_first_inputs([frame], instruction) for frame in frames]
        ids1 = per_stream[0]["input_ids"][0]
        cur_grid = per_stream[0]["image_grid_thw"][:1]
        prefix_len, frame_lens, action_len = self._kv_segments(ids1)
        frame_len = frame_lens[0]

        # Reconstruct the n_visible-frame window ids by replicating the frame block,
        # then let get_rope_index assign the (content-independent) M-RoPE positions.
        prefix, action = ids1[:prefix_len], ids1[prefix_len + frame_len :]
        block = ids1[prefix_len : prefix_len + frame_len]
        window_ids = torch.cat([prefix] + [block] * n_visible + [action]).unsqueeze(0)
        full_pos1, _ = backbone.get_rope_index(window_ids, image_grid_thw=cur_grid.repeat(n_visible, 1))
        full_pos = full_pos1.expand(3, B, full_pos1.shape[-1]).contiguous()

        batched = FrameKVMemory.stack(memories) if B > 1 else memories[0]
        past_len = batched.past_length()
        is_prefill = not batched.has_text()
        expected_past = 0 if is_prefill else prefix_len + n_past * frame_len
        if past_len != expected_past:
            raise RuntimeError(
                f"KV memory slice misaligned: memory past_length={past_len} but window "
                f"expects {expected_past}. Token boundaries drifted across steps."
            )
        if is_prefill:
            new_start, frame_start_in_new, prefill_text_len = 0, prefix_len, prefix_len
        else:
            new_start, frame_start_in_new, prefill_text_len = past_len, 0, 0

        new_ids = window_ids[:, new_start:].expand(B, -1).contiguous()
        pixels = torch.cat([b["pixel_values"] for b in per_stream], dim=0)
        grid = cur_grid.repeat(B, 1)

        last_hidden = memory_step(
            backbone,
            batched,
            new_input_ids=new_ids,
            pixel_values=pixels,
            image_grid_thw=grid,
            full_position_ids=full_pos,
            frame_start_in_new=frame_start_in_new,
            frame_len=frame_len,
            prefill_text_len=prefill_text_len,
        )

        if B > 1:
            for i, memory in enumerate(memories):
                sliced = batched.slice(i)
                memory.text, memory.frames = sliced.text, sliced.frames
        return last_hidden, new_ids

    # ------------------------------------------------------------------ #
    # Packed single-forward training path (StreamingLLM-style mask).
    #
    # Instead of the per-frame Python loop (`_forward_memory`), pack the whole
    # rollout into ONE sequence
    #     [text/sink] [frame_1][action_1] ... [frame_R][action_R]
    # and run a SINGLE Qwen3-VL forward under a FlexAttention BlockMask that
    # reproduces the streaming visibility:
    #   * the text prefix is an always-visible attention sink (StreamingLLM);
    #   * each frame attends only the last `kv_window` frames (sliding window);
    #   * each action read-out block is isolated (seen only by its own step), so
    #     the frame KV stream is bit-identical to the streaming loop / eval.
    # Custom M-RoPE `position_ids` keep frame->frame and action->frame relative
    # distances exactly matching eval; only the text-sink->frame absolute distance
    # grows with step (accepted approximation — the sink is distance-robust).
    # Eval/live (`_predict_action_memory`) is untouched and keeps the rolling KV.
    # ------------------------------------------------------------------ #
    def _packed_window_layout(self, frames, instruction):
        """Build the packed [text][f1][a1]...[fR][aR] window for ONE stream.

        Token structure is content-independent (image placeholders are identical
        per frame), so the prefix / frame / action blocks are reconstructed by
        token-replication (matching `_kv_forward_step_batched`'s `window_ids`),
        which guarantees each action block equals the per-step layout exactly.

        Returns dict with: input_ids [1,L], pixel_values, image_grid_thw [R,3],
        position_ids [3,1,L] (custom M-RoPE), kind [L] (0=text,1=frame,2=action),
        step [L] (frame/step index for frame&action tokens, -1 for text),
        action_idx [R, chunk_len] (packed indices of each step's action tokens).
        """
        R = len(frames)
        backbone = self.qwen_vl_interface.model.model  # Qwen3VLModel
        device = self.qwen_vl_interface.model.device

        # One processor call per frame: identical token block, distinct pixels.
        per_frame = [self._build_text_first_inputs([f], instruction) for f in frames]
        ids1 = per_frame[0]["input_ids"][0]
        grid1 = per_frame[0]["image_grid_thw"][:1]  # [1,3]
        prefix_len, frame_lens, action_len = self._kv_segments(ids1)
        frame_len = frame_lens[0]
        for pf in per_frame:
            pl, fl, al = self._kv_segments(pf["input_ids"][0])
            if (pl, fl[0], al) != (prefix_len, frame_len, action_len):
                raise RuntimeError(
                    "Packed KV memory: frames have inconsistent token layout "
                    f"({(pl, fl[0], al)} vs {(prefix_len, frame_len, action_len)}); "
                    "all frames in a rollout must share resolution/grid."
                )

        prefix = ids1[:prefix_len]
        frame_block = ids1[prefix_len : prefix_len + frame_len]
        action_block = ids1[prefix_len + frame_len :]  # incl. assistant header
        block_len = frame_len + action_len

        pixel_values = torch.cat([pf["pixel_values"] for pf in per_frame], dim=0)
        grid = grid1.repeat(R, 1)  # [R, 3]

        # Frames-only M-RoPE positions: frames laid out consecutively (no action
        # in between), so frame→frame / action→frame relative distances match eval
        # ("positions within the cache"). These are the canonical frame positions
        # reused regardless of where the frame sits in the packed token sequence.
        frames_only_ids = torch.cat([prefix] + [frame_block] * R).unsqueeze(0)
        full_pos_f, _ = backbone.get_rope_index(frames_only_ids, image_grid_thw=grid)  # [3,1,Lf]
        text_pos = full_pos_f[:, :, :prefix_len]

        action_offsets = (action_block == self.action_token_id).nonzero(as_tuple=False).flatten()
        if action_offsets.numel() < self.chunk_len:
            raise RuntimeError(
                f"Packed KV memory: action block has {action_offsets.numel()} action "
                f"tokens (< chunk_len={self.chunk_len})."
            )
        action_offsets = action_offsets[-self.chunk_len :]  # last chunk_len, matches gather

        def _frame_pos(k):
            s = prefix_len + k * frame_len
            return full_pos_f[:, :, s : s + frame_len]

        def _action_pos(frame_pos_k):
            fk_max = int(frame_pos_k.max().item())
            ap = torch.arange(1, action_len + 1, device=device) + fk_max
            return ap.view(1, 1, -1).expand(3, 1, -1)

        W = self.kv_window
        parts, pos_parts, kind_parts, step_parts, action_idx_rows = [], [], [], [], []

        if self.kv_rebased_sink:
            # Per-step re-based sink, with the redundant pre-eviction sinks deduped.
            # Steps k < W all have j0=0 → o_k=0 → an identical sink at [0,P), so they
            # share ONE front sink (step=-2, visible to queries with step < W). This
            # is exactly eval before any eviction (sink pinned at the front, frames
            # growing from P). Steps k >= W each carry their own re-based sink
            # ([sink_k][frame_k][action_k]) at o_k so the sink→frame distance equals
            # eval's rolling-KV layout (bit-exact). Net: 1 + max(0,R-W) sinks instead
            # of R, saving (W-1) prefix copies while staying numerically identical.
            base0 = int(full_pos_f[0, 0, prefix_len].item())  # frame 0's temporal start

            # Shared front sink for the pre-eviction steps (step=-2).
            parts.append(prefix)
            pos_parts.append(text_pos)
            kind_parts.append(torch.zeros(prefix_len, dtype=torch.long, device=device))
            step_parts.append(torch.full((prefix_len,), -2, dtype=torch.long, device=device))
            cursor = prefix_len

            for k in range(R):
                if k >= W:
                    # Re-based sink_k sits right before step k's oldest visible frame.
                    j0 = k - W + 1
                    o_k = int(full_pos_f[0, 0, prefix_len + j0 * frame_len].item()) - base0
                    parts.append(prefix)
                    pos_parts.append(text_pos + o_k)  # shift all 3 M-RoPE dims by o_k
                    kind_parts.append(torch.zeros(prefix_len, dtype=torch.long, device=device))
                    step_parts.append(torch.full((prefix_len,), k, dtype=torch.long, device=device))
                    cursor += prefix_len
                frame_pos_k = _frame_pos(k)
                parts += [frame_block, action_block]
                pos_parts += [frame_pos_k, _action_pos(frame_pos_k)]
                kind_parts.append(
                    torch.cat([
                        torch.ones(frame_len, dtype=torch.long, device=device),
                        torch.full((action_len,), 2, dtype=torch.long, device=device),
                    ])
                )
                step_parts.append(torch.full((block_len,), k, dtype=torch.long, device=device))
                action_idx_rows.append(cursor + frame_len + action_offsets)
                cursor += block_len
        else:
            # Legacy single global sink: [sink][f0 a0]...[f_{R-1} a_{R-1}]. Sink at
            # the front (step=-1, always visible); sink→frame distance grows with k.
            parts.append(prefix)
            pos_parts.append(text_pos)
            kind_parts.append(torch.zeros(prefix_len, dtype=torch.long, device=device))
            step_parts.append(torch.full((prefix_len,), -1, dtype=torch.long, device=device))
            cursor = prefix_len
            for k in range(R):
                frame_pos_k = _frame_pos(k)
                parts += [frame_block, action_block]
                pos_parts += [frame_pos_k, _action_pos(frame_pos_k)]
                kind_parts.append(
                    torch.cat([
                        torch.ones(frame_len, dtype=torch.long, device=device),
                        torch.full((action_len,), 2, dtype=torch.long, device=device),
                    ])
                )
                step_parts.append(torch.full((block_len,), k, dtype=torch.long, device=device))
                action_idx_rows.append(cursor + frame_len + action_offsets)
                cursor += block_len

        input_ids = torch.cat(parts).unsqueeze(0)  # [1, L]
        position_ids = torch.cat(pos_parts, dim=2)  # [3,1,L]
        kind = torch.cat(kind_parts)
        step = torch.cat(step_parts)
        action_idx = torch.stack(action_idx_rows, dim=0)  # [R, chunk_len]

        return {
            "input_ids": input_ids,
            "pixel_values": pixel_values,
            "image_grid_thw": grid,
            "position_ids": position_ids,
            "kind": kind,
            "step": step,
            "action_idx": action_idx,
        }

    def _packed_block_mask(self, kind, step, L, batch):
        """Build a FlexAttention BlockMask for the packed sliding-window+sink layout."""
        from torch.nn.attention.flex_attention import create_block_mask
        from starVLA.model.modules.vlm.kv_memory import make_packed_sliding_mask_mod

        mask_mod = make_packed_sliding_mask_mod(kind, step, self.kv_window)
        # B=H=None → the mask is shared across batch and heads (it depends only on
        # token kind/step, not content), so one BlockMask covers the whole batch.
        return create_block_mask(mask_mod, B=None, H=None, Q_LEN=L, KV_LEN=L, device=str(kind.device))

    def _assert_flex_attention(self):
        """Packed training needs the decoder dispatched through FlexAttention.

        We pass a FlexAttention BlockMask straight to the model, so the decoder
        (forward AND gradient-checkpointing recompute during backward) must use
        the flex backend the whole time — a transient toggle around the forward
        would leave the recompute on sdpa and choke on the BlockMask. The eval
        streaming path also runs fine under flex, so the whole model is simply
        loaded with `attn_implementation: flex_attention`.
        """
        _, _, _, text_config = self._kv_components()
        impl = getattr(text_config, "_attn_implementation", None)
        if impl != "flex_attention":
            raise RuntimeError(
                "Packed KV-memory training requires the VLM loaded with "
                f"attn_implementation='flex_attention' (got {impl!r}). Set "
                "framework.qwenvl.attn_implementation: flex_attention."
            )

    def _forward_memory_packed(self, examples: List[dict]) -> dict:
        """Training forward via a single packed sliding-window+sink masked pass.

        Per-frame supervision identical to `_forward_memory` (every real frame is
        supervised at the window it naturally has), but the expensive VLM forward
        runs ONCE per rollout instead of R times. Full BPTT across the rollout
        (no per-step detach) — pair with gradient checkpointing for memory.
        """
        self._assert_flex_attention()
        profile = self._profile_timing_enabled()
        timing = {"timing/mem_nograd_fwd": 0.0, "timing/mem_grad_fwd": 0.0, "timing/mem_action_head": 0.0}

        instructions = [
            self._instruction_with_state(ex["lang"], ex.get("state") if "state" in ex else None)
            for ex in examples
        ]
        groups: dict = {}
        for idx, instruction in enumerate(instructions):
            valid = np.asarray(examples[idx]["valid"], dtype=bool)
            groups.setdefault((instruction, tuple(valid.tolist())), []).append(idx)

        total_weighted_loss = None
        total_weight = 0.0
        for (instruction, valid_key), indices in groups.items():
            valid = np.asarray(valid_key, dtype=bool)
            real_positions = [k for k in range(len(valid)) if valid[k]]
            R = len(real_positions)
            if R == 0:
                continue
            action_env_dims = [int(examples[i].get("action_env_dim", self.action_env_dim)) for i in indices]
            rl_games_tasks = [str(examples[i].get("rl_games_task", "")) for i in indices]
            if not any(rl_games_tasks):
                rl_games_tasks = None

            # Build per-sample packed layouts (identical token structure, distinct pixels).
            layouts = [
                self._packed_window_layout([examples[i]["image"][k] for k in real_positions], instruction)
                for i in indices
            ]
            base = layouts[0]
            L = base["input_ids"].shape[1]
            B = len(indices)
            input_ids = base["input_ids"].expand(B, -1).contiguous()
            position_ids = base["position_ids"].expand(3, B, -1).contiguous()
            pixel_values = torch.cat([lay["pixel_values"] for lay in layouts], dim=0)
            grid = torch.cat([lay["image_grid_thw"] for lay in layouts], dim=0)  # [B*R, 3]
            action_idx = base["action_idx"]  # [R, chunk_len]
            block_mask = self._packed_block_mask(base["kind"], base["step"], L, B)

            t_fwd = self._profile_start() if profile else None
            last_hidden = self._forward_qwen_last_hidden(
                {
                    "input_ids": input_ids,
                    "position_ids": position_ids,
                    "attention_mask": block_mask,
                    "pixel_values": pixel_values,
                    "image_grid_thw": grid,
                    "use_cache": False,
                }
            )  # [B, L, H]
            if profile:
                timing["timing/mem_grad_fwd"] += self._profile_elapsed(t_fwd)

            t_head = self._profile_start() if profile else None
            with torch.autocast("cuda", dtype=torch.float32):
                for j, k in enumerate(real_positions):
                    action_queries = last_hidden[:, action_idx[j], :]  # [B, chunk_len, H]
                    group_preds = self.action_model.predict_action(action_queries)  # [B, chunk, dim]
                    targets = [self._prepare_action_array(examples[i]["actions_per_frame"][k]) for i in indices]
                    targets = torch.tensor(np.array(targets), device=group_preds.device, dtype=group_preds.dtype)
                    targets = targets[:, -self.action_horizon :, :]
                    step_loss = self._compute_action_loss(
                        group_preds, targets, action_env_dims=action_env_dims, rl_games_tasks=rl_games_tasks
                    )
                    weight = float(sum(float(np.asarray(examples[i]["density_weight"])[k]) for i in indices))
                    contribution = step_loss * weight
                    total_weighted_loss = contribution if total_weighted_loss is None else total_weighted_loss + contribution
                    total_weight += weight
            if profile:
                timing["timing/mem_action_head"] += self._profile_elapsed(t_head)

        action_loss = total_weighted_loss / max(total_weight, 1e-6)
        return {"action_loss": action_loss, "loss_weight": total_weight, "timing": timing, "batch_stats": {}}

    def _instruction_with_state(self, instruction, state):
        if state is None:
            return instruction
        return self.add_discretized_state_to_instruction([instruction], [state])[0]

    def _forward_memory(self, examples: List[dict]) -> dict:
        """Training forward with the streaming KV memory: per-frame supervision.

        Each sample is an R-frame rollout `[t-R+1 .. t]`; the dataloader marks clamp
        padding via ``valid`` and supplies one action chunk per frame in
        ``actions_per_frame`` ([R, H, dim]). We skip padding frames (never feed memory,
        never supervise) and supervise EVERY real frame at the window it naturally has
        (1 -> window, then evict), which matches eval (`_predict_action_memory`) frame
        for frame, including the unfilled window at an episode's start. History KV is
        detached between steps (truncated BPTT depth 1), so each step's loss flows only
        through that step's forward; the per-step weighted losses are summed and the
        trainer backpropagates once.

        Same-instruction, same-`valid` rollouts share a token layout and window
        evolution, so they run each step in one batched forward via
        `_kv_forward_step_batched` (degrades to per-sample otherwise), mirroring
        `_predict_action_memory`.
        """
        profile = self._profile_timing_enabled()
        timing = {"timing/mem_nograd_fwd": 0.0, "timing/mem_grad_fwd": 0.0, "timing/mem_action_head": 0.0}

        instructions = [
            self._instruction_with_state(ex["lang"], ex.get("state") if "state" in ex else None)
            for ex in examples
        ]

        # Group rollouts that share the prompt AND the same real/padding pattern: those
        # march through identical window states each step, so they batch cleanly.
        groups: dict = {}
        for idx, instruction in enumerate(instructions):
            valid = np.asarray(examples[idx]["valid"], dtype=bool)
            groups.setdefault((instruction, tuple(valid.tolist())), []).append(idx)

        total_weighted_loss = None
        total_weight = 0.0
        for (instruction, valid_key), indices in groups.items():
            valid = np.asarray(valid_key, dtype=bool)
            real_positions = [k for k in range(len(valid)) if valid[k]]
            memories = [self._new_kv_memory() for _ in indices]
            rollout = [examples[i]["image"] for i in indices]
            action_env_dims = [int(examples[i].get("action_env_dim", self.action_env_dim)) for i in indices]
            rl_games_tasks = [str(examples[i].get("rl_games_task", "")) for i in indices]
            if not any(rl_games_tasks):
                rl_games_tasks = None

            for step_pos, k in enumerate(real_positions):
                if step_pos > 0:
                    for memory in memories:
                        memory.detach_()  # truncated BPTT: history is a fixed memory
                step_frames = [frames[k] for frames in rollout]

                t_fwd = self._profile_start() if profile else None
                last_hidden, new_ids = self._kv_forward_step_batched(memories, step_frames, instruction)
                if profile:
                    timing["timing/mem_grad_fwd"] += self._profile_elapsed(t_fwd)

                t_head = self._profile_start() if profile else None
                with torch.autocast("cuda", dtype=torch.float32):
                    action_queries = self._gather_action_token_embeddings(
                        last_hidden, new_ids, action_token_id=self.action_token_id
                    )
                    group_preds = self.action_model.predict_action(action_queries)  # [Bg, chunk, dim]
                    targets = [
                        self._prepare_action_array(examples[i]["actions_per_frame"][k]) for i in indices
                    ]
                    targets = torch.tensor(
                        np.array(targets), device=group_preds.device, dtype=group_preds.dtype
                    )  # [Bg, H, dim]
                    targets = targets[:, -self.action_horizon :, :]
                    step_loss = self._compute_action_loss(
                        group_preds,
                        targets,
                        action_env_dims=action_env_dims,
                        rl_games_tasks=rl_games_tasks,
                    )
                if profile:
                    timing["timing/mem_action_head"] += self._profile_elapsed(t_head)

                # Per-supervised-step density weight: sum of the group's per-frame
                # weights at frame k. Uniform weights reduce this to a sample-count
                # weighting, i.e. the exact mean over all supervised (sample, step).
                weight = float(
                    sum(float(np.asarray(examples[i]["density_weight"])[k]) for i in indices)
                )
                contribution = step_loss * weight
                total_weighted_loss = contribution if total_weighted_loss is None else total_weighted_loss + contribution
                total_weight += weight

        action_loss = total_weighted_loss / max(total_weight, 1e-6)
        return {"action_loss": action_loss, "loss_weight": total_weight, "timing": timing, "batch_stats": {}}

    @torch.inference_mode()
    def _predict_action_memory(self, examples: List[dict]) -> dict:
        """Eval-time streaming prediction with per-slot KV memory (req3, milestone 4).

        The newest frame (``example['image'][-1]``) is encoded fresh; past frames are
        served from the slot's memory. ``reset_memory(slot_id)`` (called from the
        policy at episode boundaries) clears a slot. Streams that currently share the
        same instruction and history layout are processed in one batched forward; the
        rest fall back to their own group (no left padding needed).
        """
        train_obs_image_size = getattr(self.config.datasets.vla_data, "obs_image_size", None)

        # Resolve each example to its (slot memory, current frame, instruction).
        entries = []
        for idx, example in enumerate(examples):
            slot_id = example.get("slot_id", idx)
            frames = [to_pil_preserve(f) for f in example["image"]]
            if train_obs_image_size:
                frames = resize_images(frames, target_size=train_obs_image_size)
            state = example.get("state") if "state" in example else None
            instruction = self._instruction_with_state(example["lang"], state)
            memory = self._kv_memories.get(slot_id)
            if memory is None:
                memory = self._new_kv_memory()
                self._kv_memories[slot_id] = memory
            entries.append({"idx": idx, "frame": frames[-1], "instruction": instruction, "memory": memory})

        # Group same-layout streams (same prompt + same cached layout) into one forward.
        groups: dict = {}
        for entry in entries:
            key = (entry["instruction"], entry["memory"].layout_key())
            groups.setdefault(key, []).append(entry)

        preds: List[Optional[torch.Tensor]] = [None] * len(examples)
        for group in groups.values():
            instruction = group[0]["instruction"]
            memories = [entry["memory"] for entry in group]
            frames = [entry["frame"] for entry in group]
            last_hidden, new_ids = self._kv_forward_step_batched(memories, frames, instruction)
            with torch.autocast("cuda", dtype=torch.float32):
                action_queries = self._gather_action_token_embeddings(
                    last_hidden, new_ids, action_token_id=self.action_token_id
                )
                pred_actions = self.action_model.predict_action(action_queries)
            for i, entry in enumerate(group):
                preds[entry["idx"]] = pred_actions[i : i + 1]

        normalized = torch.cat(preds, dim=0).detach().cpu().numpy()
        return {"normalized_actions": normalized}

    def _gather_action_token_embeddings(
        self,
        last_hidden: torch.Tensor,  # [B, L, H]
        input_ids: torch.Tensor,  # [B, L]
        action_token_id=None,  # Can be int or List[int]
    ) -> torch.Tensor:
        """
        Vectorized batch extraction of action token embeddings:
          - No per-sample for loop
          - Select the last chunk_len action placeholder tokens from each sample
        Args:
            last_hidden: [B, L, H]
            input_ids:   [B, L]
            action_token_id: int or List[int]
        Returns:
            action_queries: [B, chunk_len, H]
        """
        if action_token_id is None:
            raise ValueError("action_token_id must not be None")

        device = input_ids.device
        B, L, H = last_hidden.shape

        # Support multiple ids (e.g., multiple variants)
        if isinstance(action_token_id, (list, tuple, set)):
            id_list = torch.tensor(list(action_token_id), device=device, dtype=input_ids.dtype)
            # torch.isin requires PyTorch >=1.10
            mask = torch.isin(input_ids, id_list)
        else:
            mask = input_ids == action_token_id  # [B, L]

        counts = mask.sum(dim=1)  # [B]
        if (counts < self.chunk_len).any():
            insufficient = (counts < self.chunk_len).nonzero(as_tuple=False).flatten().tolist()
            raise RuntimeError(
                f"The following samples have insufficient action tokens (< {self.chunk_len}): {insufficient} |"
                f" counts={counts.tolist()}"
            )

        # Position indices
        idx = torch.arange(L, device=device).unsqueeze(0).expand(B, L)  # [B, L]
        masked_pos = torch.where(mask, idx, torch.full_like(idx, -1))  # Set non-action positions to -1

        # Take the last chunk_len positions (higher indices = later in sequence)
        # Note: count sufficiency already verified, so -1 won't be incorrectly selected
        topk_pos = masked_pos.topk(k=self.chunk_len, dim=-1).values  # [B, chunk_len] unsorted
        # Sort in temporal order
        selected_pos = topk_pos.sort(dim=-1).values  # [B, chunk_len]

        # Gather
        expanded_index = selected_pos.unsqueeze(-1).expand(-1, -1, H)  # [B, chunk_len, H]
        action_queries = last_hidden.gather(dim=1, index=expanded_index)  # [B, chunk_len, H]
        return action_queries

    # Discretised state → instruction prefix (π₀.5 style); shared with QwenPI_v3.
    add_discretized_state_to_instruction = staticmethod(add_discretized_state_to_instruction)


if __name__ == "__main__":
    import argparse
    import os

    from omegaconf import OmegaConf

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_yaml",
        type=str,
        default="examples/LIBERO/train_files/starvla_cotrain_libero.yaml",
        help="Path to YAML config",
    )
    args, clipargs = parser.parse_known_args()

    if os.getenv("DEBUGPY_ENABLE", "0") == "1":
        import debugpy

        debugpy.listen(("0.0.0.0", 10092))
        print("Rank 0 waiting for debugger attach on port 10092...")
        debugpy.wait_for_client()

    cfg = OmegaConf.load(args.config_yaml)

    model = Qwenvl_OFT(cfg)
    print(model)

    image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    sample = {
        "action": np.random.uniform(-1, 1, size=(16, 7)).astype(np.float16),
        "image": [image],
        "lang": "This is a fake instruction for testing.",
        "state": np.random.uniform(-1, 1, size=(1, 7)).astype(np.float16),  # chunk, state_dim
    }
    sample2 = sample.copy()
    sample2["lang"] = "Another fake instruction for testing."

    batch = [sample, sample2]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    forward_output = model(batch)
    action_loss = forward_output["action_loss"]
    print(f"[train] Action Loss (with state): {action_loss.item()}")

    predict_output = model.predict_action(examples=[batch[0]])
    normalized_actions = predict_output["normalized_actions"]
    print(f"[infer] Predicted Action shape: {normalized_actions.shape}")

    # Backward-compat: examples without `state` should still work.
    sample_no_state = {k: v for k, v in sample.items() if k != "state"}
    forward_no_state = model([sample_no_state, sample_no_state])
    print(f"[train] Action Loss (no state): {forward_no_state['action_loss'].item()}")
    predict_no_state = model.predict_action(examples=[sample_no_state])
    print(f"[infer] Predicted Action shape (no state): {predict_no_state['normalized_actions'].shape}")

    print("Finished")
