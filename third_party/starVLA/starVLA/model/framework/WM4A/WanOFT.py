# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
"""
WanOFT Framework — Wan2.2-TI2V World Model + MLP Regression for Action Prediction.

Uses Wan2.2-TI2V-5B (DiT-based Text+Image-to-Video model) as the perception
backbone with a lightweight MLP action head (L1 regression).

Architecture:
  UMT5 (text) + VAE (image→latent) → WanTransformer3D
    → hidden_states [B, N, 3072]
    → Global avg pool → [B, 3072]
    → Linear projection → [B, chunk_len, 3072]
    → MLP (L1 regression) → action predictions [B, chunk_len, action_dim]

Key differences from WanGR00T:
  - Action head: MLP L1 regression (not flow-matching diffusion)
  - No repeated_diffusion_steps (single forward pass)
  - Faster training & inference
"""

import sys
from pathlib import Path

_workspace_root = Path(__file__).parent.parent.parent.parent.parent
if str(_workspace_root) not in sys.path:
    sys.path.insert(0, str(_workspace_root))

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from deployment.model_server.tools.image_tools import to_pil_preserve
from starVLA.training.trainer_utils import initialize_overwatch

logger = initialize_overwatch(__name__)

IGNORE_INDEX = -100

from starVLA.model.framework.base_framework import baseframework
from starVLA.model.framework.share_tools import add_discretized_state_to_instruction, merge_framework_config
from starVLA.model.modules.action_model.MLP_ActionHeader import get_action_model
from starVLA.model.modules.world_model import get_world_model
from starVLA.model.tools import FRAMEWORK_REGISTRY
from starVLA.training.trainer_utils.trainer_tools import resize_images


class TokenAttentionActionQueryProjector(nn.Module):
    """Project full Wan token sequences into per-step action queries."""

    def __init__(self, hidden_dim: int, chunk_len: int, num_heads: int) -> None:
        super().__init__()
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}.")
        if chunk_len <= 0:
            raise ValueError(f"chunk_len must be positive, got {chunk_len}.")
        if num_heads <= 0:
            raise ValueError(f"num_heads must be positive, got {num_heads}.")
        if hidden_dim % num_heads != 0:
            raise ValueError(f"hidden_dim={hidden_dim} must be divisible by num_heads={num_heads}.")

        self.hidden_dim = int(hidden_dim)
        self.chunk_len = int(chunk_len)
        self.num_heads = int(num_heads)
        self.query_tokens = nn.Parameter(torch.empty(self.chunk_len, self.hidden_dim))
        self.key_value_norm = nn.LayerNorm(self.hidden_dim)
        self.attention = nn.MultiheadAttention(
            embed_dim=self.hidden_dim,
            num_heads=self.num_heads,
            batch_first=True,
        )
        self.output_norm = nn.LayerNorm(self.hidden_dim)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.normal_(self.query_tokens, mean=0.0, std=0.02)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if hidden_states.ndim != 3:
            raise ValueError(f"hidden_states must have shape [B, N, H], got {tuple(hidden_states.shape)}.")
        if hidden_states.shape[-1] != self.hidden_dim:
            raise ValueError(
                f"hidden_states hidden dim must be {self.hidden_dim}, got shape={tuple(hidden_states.shape)}."
            )
        target_dtype = self.query_tokens.dtype
        hidden_states = hidden_states.to(dtype=target_dtype)
        batch_size = int(hidden_states.shape[0])
        queries = self.query_tokens.unsqueeze(0).expand(batch_size, -1, -1)
        key_values = self.key_value_norm(hidden_states)
        action_queries, _attention_weights = self.attention(
            query=queries,
            key=key_values,
            value=key_values,
            need_weights=False,
        )
        return self.output_norm(action_queries + queries)


@dataclass
class WanOFTDefaultConfig:
    """WanOFT default parameters."""

    name: str = "WanOFT"

    # === World Model backbone (Wan2.2-TI2V-5B-Diffusers) ===
    world_model: dict = field(
        default_factory=lambda: {
            "base_wm": "./playground/Pretrained_models/Wan-AI/Wan2.2-TI2V-5B-Diffusers",
            "extract_layers": [-1],
            "num_frames": None,
        }
    )

    qwenvl: dict = field(
        default_factory=lambda: {
            "base_vlm": "./playground/Pretrained_models/Wan-AI/Wan2.2-TI2V-5B-Diffusers",
        }
    )

    # === Action head (MLP L1 regression) ===
    action_model: dict = field(
        default_factory=lambda: {
            "action_model_type": "MLP",
            "action_dim": 7,
            "state_dim": 7,
            "action_hidden_dim": 3072,
            "action_horizon": 8,
            "future_action_window_size": 7,
            "past_action_window_size": 0,
            "loss_type": "l1",
            "state_encoding": "discretized_text",
            "class_weights": None,
            "future_loss_weight": None,
            "action_query_source": "mean",
            "action_query_num_heads": 24,
        }
    )


@FRAMEWORK_REGISTRY.register("WanOFT")
class Wan_OFT(baseframework):
    """
    World-Model-for-Action framework using Wan2.2-TI2V + MLP regression.

    Components:
      - Wan2.2-TI2V DiT (UMT5 + VAE + WanTransformer3D) for features
      - Adaptive pooling + MLP regression head (L1 loss)
    """

    def __init__(self, config: Optional[dict] = None, **kwargs) -> None:
        super().__init__()
        self.config = merge_framework_config(WanOFTDefaultConfig, config)

        self.backbone = get_world_model(config=self.config)

        wm_hidden = self.backbone.model.config.hidden_size
        self.config.framework.action_model.action_hidden_dim = wm_hidden

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
        self.state_encoding = self.config.framework.action_model.state_encoding
        if self.state_encoding == "continuous_projector":
            self.action_model.state_projector = nn.Linear(
                self.config.framework.action_model.state_dim,
                wm_hidden,
            )
        self.action_query_source = str(self.config.framework.action_model.action_query_source).strip().lower()

        self.action_query_proj = nn.Linear(wm_hidden, self.chunk_len * wm_hidden)  # Project into a two-layer MLP
        self.token_action_query_proj: TokenAttentionActionQueryProjector | None = None
        if self.action_query_source == "token_attention":
            action_query_num_heads = int(self.config.framework.action_model.action_query_num_heads)
            self.token_action_query_proj = TokenAttentionActionQueryProjector(
                hidden_dim=wm_hidden,
                chunk_len=self.chunk_len,
                num_heads=action_query_num_heads,
            )
        elif self.action_query_source != "mean":
            raise ValueError(
                f"Unsupported action_model.action_query_source={self.action_query_source!r}; "
                "expected one of: mean, token_attention."
            )

        self.l1_loss = nn.L1Loss()

    def _prepare_action_array(self, action) -> np.ndarray:
        values = np.asarray(action)
        if values.ndim == 1:
            values = values[None, :]
        if values.shape[-1] != self.action_dim:
            raise ValueError(
                f"WanOFT expected action dim={self.action_dim}, got action shape={values.shape}. "
                "Set framework.action_model.action_dim to match the task action carrier."
            )
        return values

    def _state_conditioned_instructions(self, instructions, state):
        if state is not None and self.state_encoding == "discretized_text":
            return add_discretized_state_to_instruction(instructions, state)
        return instructions

    def _condition_action_queries(self, action_queries, state):
        if self.state_encoding == "discretized_text":
            return action_queries
        state_tensor = torch.as_tensor(
            np.stack(state), device=action_queries.device, dtype=action_queries.dtype
        ).squeeze(1)
        return action_queries + self.action_model.state_projector(state_tensor).unsqueeze(1)

    def _action_model_config_value(self, key: str):
        action_model_cfg = getattr(getattr(self.config, "framework", None), "action_model", None)
        if action_model_cfg is None:
            return None
        getter = getattr(action_model_cfg, "get", None)
        if callable(getter):
            return getter(key, None)
        return getattr(action_model_cfg, key, None)

    def _action_class_weight_tensor(
        self,
        effective_dim: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Optional[torch.Tensor]:
        class_weights = self._action_model_config_value("class_weights")
        if class_weights is None:
            return None
        weights = [float(value) for value in class_weights]
        if len(weights) != effective_dim:
            raise ValueError(
                f"action_model.class_weights must contain exactly {effective_dim} values for the active action "
                f"classes, got {len(weights)} values: {weights}"
            )
        return torch.tensor(weights, device=device, dtype=dtype)

    def _compute_action_loss(self, pred_actions: torch.Tensor, actions_target: torch.Tensor) -> torch.Tensor:
        if actions_target.shape[-2] != self.action_horizon:
            raise ValueError(
                f"WanOFT expected action target horizon={self.action_horizon}, got target shape={actions_target.shape}. "
                "Set datasets.vla_data.action_indices to the released checkpoint chunk length."
            )
        effective_dim = min(self.action_env_dim, pred_actions.shape[-1], actions_target.shape[-1])
        if effective_dim <= 0:
            raise ValueError(
                f"Invalid action_env_dim={self.action_env_dim} for predicted shape={pred_actions.shape} "
                f"and target shape={actions_target.shape}"
            )
        if self.action_loss_type in {"current_discrete_ce", "current_ce", "current_cross_entropy"}:
            if effective_dim < 2:
                raise ValueError(
                    f"action_model.loss_type={self.action_loss_type!r} requires at least 2 action classes, "
                    f"got effective_dim={effective_dim}"
                )
            logits = pred_actions[:, 0, :effective_dim]
            target_class = actions_target[:, 0, :effective_dim].argmax(dim=-1).long()
            class_weights = self._action_class_weight_tensor(effective_dim, logits.device, logits.dtype)
            return F.cross_entropy(logits, target_class, weight=class_weights)

        if self.action_loss_type in {
            "current_plus_future_discrete_ce",
            "current_plus_future_ce",
            "current_plus_future_cross_entropy",
        }:
            if effective_dim < 2:
                raise ValueError(
                    f"action_model.loss_type={self.action_loss_type!r} requires at least 2 action classes, "
                    f"got effective_dim={effective_dim}"
                )
            if self.action_horizon <= 1:
                raise ValueError(
                    f"action_model.loss_type={self.action_loss_type!r} requires action_horizon > 1, "
                    f"got action_horizon={self.action_horizon}"
                )
            future_loss_weight = self._action_model_config_value("future_loss_weight")
            if future_loss_weight is None:
                raise ValueError(
                    f"action_model.future_loss_weight is required when loss_type={self.action_loss_type!r}"
                )

            current_logits = pred_actions[:, 0, :effective_dim]
            current_target_class = actions_target[:, 0, :effective_dim].argmax(dim=-1).long()
            class_weights = self._action_class_weight_tensor(effective_dim, current_logits.device, current_logits.dtype)
            current_loss = F.cross_entropy(current_logits, current_target_class, weight=class_weights)

            future_logits = pred_actions[:, 1:, :effective_dim]
            future_target_class = actions_target[:, 1:, :effective_dim].argmax(dim=-1).long()
            future_loss = F.cross_entropy(
                future_logits.reshape(-1, effective_dim),
                future_target_class.reshape(-1),
                weight=class_weights,
            )
            return current_loss + float(future_loss_weight) * future_loss

        if self.action_loss_type in {"current_multibinary_bce", "current_bce"}:
            logits = pred_actions[:, 0, :effective_dim]
            targets = (actions_target[:, 0, :effective_dim] > 0).to(dtype=logits.dtype)
            return F.binary_cross_entropy_with_logits(logits, targets)

        if self.action_loss_type in {
            "multibinary_bce",
            "multibinary_ce",
            "bce",
            "binary_cross_entropy",
        }:
            logits = pred_actions[..., :effective_dim]
            targets = (actions_target[..., :effective_dim] > 0).to(dtype=logits.dtype)
            return F.binary_cross_entropy_with_logits(logits, targets)

        if self.action_loss_type in {"discrete_ce", "ce", "cross_entropy"}:
            if effective_dim < 2:
                raise ValueError(
                    f"action_model.loss_type={self.action_loss_type!r} requires at least 2 action classes, "
                    f"got effective_dim={effective_dim}"
                )
            logits = pred_actions[..., :effective_dim]
            target_class = actions_target[..., :effective_dim].argmax(dim=-1).long()
            class_weights = self._action_class_weight_tensor(effective_dim, logits.device, logits.dtype)
            return F.cross_entropy(
                logits.reshape(-1, effective_dim),
                target_class.reshape(-1),
                weight=class_weights,
            )

        if self.action_loss_type in {"asterix_factorized_ce", "factorized_6_ce", "factorized_ce"}:
            if effective_dim < 6:
                raise ValueError(
                    f"action_model.loss_type={self.action_loss_type!r} requires at least 6 action logits, "
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

        if self.action_loss_type not in {"l1", "mae"}:
            raise ValueError(
                f"Unsupported action_model.loss_type={self.action_loss_type!r}; "
                "expected one of: l1, discrete_ce, current_discrete_ce, "
                "current_plus_future_discrete_ce, multibinary_bce, current_multibinary_bce, "
                "asterix_factorized_ce"
            )

        return self.l1_loss(pred_actions[..., :effective_dim], actions_target[..., :effective_dim])

    def _pool_to_action_queries(self, hidden_states: torch.Tensor) -> torch.Tensor:
        B, N, H = hidden_states.shape
        if self.action_query_source == "token_attention":
            if self.token_action_query_proj is None:
                raise RuntimeError("token_action_query_proj is required when action_query_source=token_attention.")
            return self.token_action_query_proj(hidden_states)
        target_dtype = self.action_query_proj.weight.dtype
        pooled = hidden_states.mean(dim=1).to(dtype=target_dtype)
        queries = self.action_query_proj(pooled)
        action_queries = queries.view(B, self.chunk_len, H)
        return action_queries

    def forward(self, examples: List[dict] = None, **kwargs) -> Tuple:
        batch_images = [example["image"] for example in examples]
        instructions = [example["lang"] for example in examples]
        actions = [self._prepare_action_array(example["action"]) for example in examples]
        state = [example["state"] for example in examples] if "state" in examples[0] else None

        train_obs_image_size = getattr(self.config.datasets.vla_data, "obs_image_size", None)
        if train_obs_image_size:
            batch_images = resize_images(batch_images, target_size=train_obs_image_size)

        instructions = self._state_conditioned_instructions(instructions, state)

        wm_inputs = self.backbone.build_inputs(images=batch_images, instructions=instructions)

        with torch.autocast("cuda", dtype=torch.bfloat16):
            wm_outputs = self.backbone(
                **wm_inputs,
                output_hidden_states=True,
                return_dict=True,
            )
            last_hidden = wm_outputs.hidden_states[-1]

        with torch.autocast("cuda", dtype=torch.float32):
            action_queries = self._pool_to_action_queries(last_hidden)  # B, chunk_len, hidden_dim
            action_queries = self._condition_action_queries(action_queries, state)
            pred_actions = self.action_model.predict_action(action_queries)

            actions = torch.tensor(np.array(actions), device=pred_actions.device, dtype=pred_actions.dtype)
            actions_target = actions[:, -self.action_horizon :, :]

            action_loss = self._compute_action_loss(pred_actions, actions_target)

        return {"action_loss": action_loss, "loss_weight": float(len(examples))}

    @torch.inference_mode()
    def predict_action(self, examples: List[dict], **kwargs) -> np.ndarray:
        if type(examples) is not list:
            examples = [examples]
        batch_images = [to_pil_preserve(example["image"]) for example in examples]
        instructions = [example["lang"] for example in examples]
        state = [example["state"] for example in examples] if "state" in examples[0] else None

        instructions = self._state_conditioned_instructions(instructions, state)

        train_obs_image_size = getattr(self.config.datasets.vla_data, "obs_image_size", None)
        if train_obs_image_size:
            batch_images = resize_images(batch_images, target_size=train_obs_image_size)

        wm_inputs = self.backbone.build_inputs(images=batch_images, instructions=instructions)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            wm_outputs = self.backbone(
                **wm_inputs,
                output_hidden_states=True,
                return_dict=True,
            )
            last_hidden = wm_outputs.hidden_states[-1]

        with torch.autocast("cuda", dtype=torch.float32):
            action_queries = self._pool_to_action_queries(last_hidden)
            action_queries = self._condition_action_queries(action_queries, state)
            pred_actions = self.action_model.predict_action(action_queries)

        normalized_actions = pred_actions.detach().cpu().numpy()
        return {"normalized_actions": normalized_actions}


if __name__ == "__main__":
    import argparse
    import os

    from omegaconf import OmegaConf
    from PIL import Image

    if os.getenv("DEBUGPY_ENABLE", "0") == "1":
        import debugpy

        debugpy.listen(("0.0.0.0", 10092))
        print("Rank 0 waiting for debugger attach on port 10092...")
        debugpy.wait_for_client()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_yaml",
        type=str,
        default="examples/LIBERO/train_files/starvla_cotrain_libero.yaml",
        help="Path to YAML config",
    )
    args, clipargs = parser.parse_known_args()

    cfg = OmegaConf.load(args.config_yaml)

    cfg.framework.name = "WanOFT"
    cfg.framework.qwenvl.base_vlm = "./playground/Pretrained_models/Wan-AI/Wan2.2-TI2V-5B-Diffusers"
    cfg.framework.world_model = {
        "base_wm": "./playground/Pretrained_models/Wan-AI/Wan2.2-TI2V-5B-Diffusers",
        "extract_layers": [-1],
    }

    model: Wan_OFT = Wan_OFT(cfg)
    print(model)

    image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    sample = {
        "action": np.random.uniform(-1, 1, size=(16, 7)).astype(np.float16),
        "image": [image, image],
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

    predict_output = model.predict_action(examples=[sample])
    normalized_actions = predict_output["normalized_actions"]
    print(f"[infer] Predicted Action shape: {normalized_actions.shape}")

    # Backward-compat: examples without `state` should still work.
    sample_no_state = {k: v for k, v in sample.items() if k != "state"}
    forward_no_state = model([sample_no_state, sample_no_state])
    print(f"[train] Action Loss (no state): {forward_no_state['action_loss'].item()}")
    predict_no_state = model.predict_action(examples=[sample_no_state])
    print(f"[infer] Predicted Action shape (no state): {predict_no_state['normalized_actions'].shape}")

    print("Finished")
