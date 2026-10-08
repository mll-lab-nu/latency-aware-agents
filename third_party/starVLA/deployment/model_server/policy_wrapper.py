# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License.
"""Policy server wrapper.

Encapsulates a `baseframework` instance plus a :class:`PolicyNormProcessor`
that reuses the *training-time* :class:`ComposedModalityTransform` for action
un-normalization (no hand-rolled math). The websocket server returns
already-unnormalized actions.

Client-side responsibilities that REMAIN on the client:
  - environment-specific adapters (image_history, gripper sticky, action
    ensembling)
  - chunk-cache scheduling (`step % chunk_size == 0` triggers a new infer)

Exposed API:
  - ``metadata`` (dict, sent at handshake): ``action_chunk_size``,
    ``available_unnorm_keys``, ``action_keys``, ``state_keys``.
  - ``predict_action(examples, unnorm_key=None, **kwargs)`` returns
    ``{"actions": np.ndarray[B, T, action_dim]}`` in deployment mode.
  - ``action_output_mode="rl_games"`` returns decoded game actions plus
    ``raw_action_scores`` for diagnostics.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from starVLA.model.framework.base_framework import baseframework
from starVLA.model.framework.share_tools import read_mode_config
from starVLA.model.profiling import stage_timer as _stage

from deployment.model_server.policy_norm_processor import PolicyNormProcessor
from deployment.model_server.rl_games_action_decode import (
    decode_rl_games_actions,
    resolve_asterix_action_decode_spec,
    resolve_deadly_action_decode_spec,
)


class PolicyServerWrapper:
    _rl_games_gymnasium_action_space_type = "discrete"
    _rl_games_gymnasium_robot_type = "rl_games_gymnasium"
    """Wraps a `baseframework` for use as a websocket-server policy."""

    def __init__(
        self,
        ckpt_path: str,
        device: str = "cuda",
        use_bf16: bool = False,
        unnorm_key: Optional[str] = None,
        action_output_mode: str = "deployment",
        rl_games_env_name: Optional[str] = None,
        rl_games_action_layout: Optional[str] = None,
        rl_games_multibinary_threshold: Optional[float] = None,
        rl_games_action_env_dim: Optional[int] = None,
        rl_games_gymnasium_action_space_type: str = "discrete",
        rl_games_gymnasium_robot_type: str = "rl_games_gymnasium",
        backbone_path: Optional[str] = None,
    ) -> None:
        self._ckpt_path = str(ckpt_path)
        self._action_output_mode = action_output_mode
        self._rl_games_env_name = rl_games_env_name
        self._rl_games_action_env_dim = rl_games_action_env_dim
        self._rl_games_gymnasium_action_space_type = rl_games_gymnasium_action_space_type
        self._rl_games_gymnasium_robot_type = rl_games_gymnasium_robot_type
        self._rl_games_action_layout = rl_games_action_layout
        self._rl_games_multibinary_threshold = rl_games_multibinary_threshold

        logging.info("PolicyServerWrapper: loading framework from %s", self._ckpt_path)
        framework = baseframework.from_pretrained(
            self._ckpt_path,
            backbone_path=backbone_path,
        )
        if use_bf16:
            framework = framework.to(torch.bfloat16)
        framework = framework.to(device).eval()
        self._framework = framework

        # Co-located metadata.
        model_cfg, _ = read_mode_config(self._ckpt_path)
        self._model_cfg = model_cfg
        if self._rl_games_env_name == "deadly_corridor":
            (
                self._rl_games_action_layout,
                self._rl_games_multibinary_threshold,
            ) = resolve_deadly_action_decode_spec(
                model_cfg,
                action_layout=rl_games_action_layout,
                multibinary_threshold=rl_games_multibinary_threshold,
            )
        elif self._rl_games_env_name == "asterix":
            self._rl_games_action_layout = resolve_asterix_action_decode_spec(
                model_cfg,
                action_layout=rl_games_action_layout,
            )

        # action_chunk_size = future_action_window_size + 1 (matches old client).
        action_model_cfg = model_cfg["framework"]["action_model"]
        
        if "action_horizon" in action_model_cfg:
            self._action_chunk_size = int(action_model_cfg["action_horizon"])
        elif "future_action_window_size" in action_model_cfg:
            self._action_chunk_size = int(action_model_cfg["future_action_window_size"]) + 1
        else:
            raise ValueError(
                f"PolicyServerWrapper: no action_horizon or future_action_window_size found in model config for {self._ckpt_path}"
            )
        # Cache of PolicyNormProcessor instances per unnorm_key.
        # For single-dataset ckpts unnorm_key is auto-selected; for multi-dataset
        # ckpts clients must pass unnorm_key per request.
        self._default_unnorm_key = unnorm_key
        self._norm_processors: Dict[str, PolicyNormProcessor] = {}

        # Peek at available keys without building a full processor.
        _, _ns = read_mode_config(self._ckpt_path)
        self._available_unnorm_keys: List[str] = list(_ns.keys())

        # Eagerly build when unambiguous; defer for multi-key / no explicit key.
        if unnorm_key is not None or len(self._available_unnorm_keys) == 1:
            default_proc = self._get_processor(unnorm_key)
            self._default_unnorm_key = default_proc.unnorm_key
            logging.info(
                "PolicyServerWrapper ready: action_chunk_size=%d, default_unnorm_key=%s, "
                "available_unnorm_keys=%s, action_keys=%s, state_keys=%s",
                self._action_chunk_size,
                default_proc.unnorm_key,
                default_proc.available_unnorm_keys,
                default_proc.action_keys,
                default_proc.state_keys,
            )
        else:
            logging.info(
                "PolicyServerWrapper ready (multi-key): action_chunk_size=%d, "
                "available_unnorm_keys=%s — clients must pass unnorm_key per request.",
                self._action_chunk_size,
                self._available_unnorm_keys,
            )

    def _get_processor(self, unnorm_key: Optional[str]) -> PolicyNormProcessor:
        cache_key = unnorm_key if unnorm_key is not None else "__default__"
        if cache_key not in self._norm_processors:
            if self._rl_games_env_name == "gymnasium":
                processor = PolicyNormProcessor(
                    self._ckpt_path,
                    unnorm_key=unnorm_key,
                    robot_type=self._rl_games_gymnasium_robot_type,
                )
            else:
                processor = PolicyNormProcessor(
                    self._ckpt_path,
                    unnorm_key=unnorm_key,
                )
            self._norm_processors[cache_key] = processor
        return self._norm_processors[cache_key]

    @property
    def metadata(self) -> Dict[str, Any]:
        """Model-invariant metadata; sent to client at websocket handshake."""
        base = {
            "env": "starvla_policy_server",
            "ckpt_path": self._ckpt_path,
            "action_chunk_size": self._action_chunk_size,
            "available_unnorm_keys": self._available_unnorm_keys,
            "default_unnorm_key": self._default_unnorm_key,
            "action_output_mode": self._action_output_mode,
        }
        if self._rl_games_env_name is not None:
            base["rl_games_env_name"] = self._rl_games_env_name
        # Enrich with per-embodiment keys when a default processor already exists.
        if self._default_unnorm_key is not None:
            proc = self._get_processor(self._default_unnorm_key)
            base["action_keys"] = proc.action_keys
            base["state_keys"] = proc.state_keys
        return base

    def reset_memory(self, slot_id=None) -> None:
        """Clear the framework's streaming KV memory for one slot (or all).

        No-op for frameworks without KV memory.
        """
        reset = getattr(self._framework, "reset_memory", None)
        if callable(reset):
            reset(slot_id)

    def predict_action(
        self,
        examples: List[dict],
        unnorm_key: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, np.ndarray]:
        """Run the framework, then un-normalize via training-time transforms.

        Args:
            examples: list of dicts (each with ``image`` / ``lang`` / optional ``state``).
            unnorm_key: dataset key for un-normalization stats. ``None`` -->
                use the wrapper's default (auto-picked at startup).
            **kwargs: forwarded to the framework's ``predict_action``
                (``do_sample``, ``use_ddim``, ``num_ddim_steps``, ...).

        Returns:
            Deployment mode returns un-normalized ``actions``. RL-games mode
            returns decoded game ``actions`` and raw model scores.
        """
        effective_key = unnorm_key if unnorm_key is not None else self._default_unnorm_key
        if effective_key is None:
            if len(self._available_unnorm_keys) == 1:
                effective_key = self._available_unnorm_keys[0]
            else:
                raise ValueError(
                    f"predict_action: unnorm_key not specified and no default set. "
                    f"Pass one of {self._available_unnorm_keys}."
                )
        proc = self._get_processor(effective_key)
        if self._rl_games_env_name == "gymnasium":
            examples = [
                {**example, "state": proc.apply_state(example["state"])}
                if "state" in example
                else example
                for example in examples
            ]
        # Forwarded to the framework via **kwargs as well; both layers profile.
        profiler = kwargs.get("profiler")

        out = self._framework.predict_action(examples=examples, **kwargs)
        normalized = np.asarray(out["normalized_actions"])  # (B, T, D)

        if self._action_output_mode == "rl_games":
            with _stage(profiler, "starvla_wrapper_rl_games_decode_ms"):
                action_values = None
                if self._rl_games_env_name == "gymnasium":
                    if self._rl_games_gymnasium_action_space_type == "box":
                        action_values = np.stack(
                            [proc.unapply_actions(normalized[b]) for b in range(normalized.shape[0])],
                            axis=0,
                        )
                decode_kwargs = {}
                if self._rl_games_env_name == "gymnasium":
                    decode_kwargs = {
                        "action_env_dim": self._rl_games_action_env_dim,
                        "gymnasium_action_space_type": self._rl_games_gymnasium_action_space_type,
                        "action_values": action_values,
                    }
                return decode_rl_games_actions(
                    normalized_actions=normalized,
                    env_name=self._rl_games_env_name,
                    deadly_action_layout=(
                        self._rl_games_action_layout
                        if self._rl_games_env_name == "deadly_corridor"
                        else None
                    ),
                    deadly_multibinary_threshold=(
                        self._rl_games_multibinary_threshold
                        if self._rl_games_env_name == "deadly_corridor"
                        else None
                    ),
                    asterix_action_layout=(
                        self._rl_games_action_layout
                        if self._rl_games_env_name == "asterix"
                        else None
                    ),
                    **decode_kwargs,
                )

        with _stage(profiler, "starvla_wrapper_unnormalize_ms"):
            unnorm = np.stack(
                [proc.unapply_actions(normalized[b]) for b in range(normalized.shape[0])],
                axis=0,
            )
        return {"actions": unnorm}
