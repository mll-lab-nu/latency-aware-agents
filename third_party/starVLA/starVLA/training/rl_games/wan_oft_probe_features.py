from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from typing import Protocol, TypedDict

import torch
from tqdm import tqdm

from starVLA.model.framework.share_tools import add_discretized_state_to_instruction
from starVLA.training.rl_games.linear_probe import fork_seeded_torch_rng
from starVLA.training.rl_games.wan_oft_probe_data import PROBE_LABEL_NAMES, ProbeExample
from starVLA.training.trainer_utils.trainer_tools import resize_images

VAE_TEMPORAL_FEATURE = "vae_temporal_delta"
DIT_GLOBAL_FEATURE = "dit_global_mean"
DIT_TEMPORAL_FEATURE = "dit_temporal_delta"
ACTION_QUERY_FEATURE = "action_query"

PROBE_FEATURE_NAMES = (
    VAE_TEMPORAL_FEATURE,
    DIT_GLOBAL_FEATURE,
    DIT_TEMPORAL_FEATURE,
    ACTION_QUERY_FEATURE,
)


class WanTransformerConfig(Protocol):
    patch_size: Sequence[int]


class WanTransformer(Protocol):
    config: WanTransformerConfig


class WanBackboneOutput(Protocol):
    hidden_states: tuple[torch.Tensor, ...]


class WanBackbone(Protocol):
    transformer: WanTransformer

    def build_inputs(
        self,
        images: list[list[object]],
        instructions: list[str],
    ) -> dict[str, object]: ...

    def __call__(self, **kwargs: object) -> WanBackboneOutput: ...


class WanActionModel(Protocol):
    def predict_action(self, actions_hidden_states: torch.Tensor) -> torch.Tensor: ...


class WanOFTProbeModel(Protocol):
    config: object
    backbone: WanBackbone
    action_model: WanActionModel
    action_env_dim: int

    def parameters(self) -> Iterator[torch.nn.Parameter]: ...

    def eval(self) -> torch.nn.Module: ...

    def _pool_to_action_queries(self, hidden_states: torch.Tensor) -> torch.Tensor: ...


class WanProbeFeatureBatch(TypedDict):
    vae_temporal_delta: torch.Tensor
    dit_global_mean: torch.Tensor
    dit_temporal_delta: torch.Tensor
    action_query: torch.Tensor
    action_logits: torch.Tensor


class ExtractedProbeDataset(TypedDict):
    sample_ids: torch.Tensor
    labels: dict[str, torch.Tensor]
    features: dict[str, torch.Tensor]
    action_logits: torch.Tensor


class ProbeTargetSet(TypedDict):
    sample_ids: torch.Tensor
    labels: dict[str, torch.Tensor]


ProbeExampleTransform = Callable[[Sequence[ProbeExample], int], list[ProbeExample]]


def pool_vae_temporal_groups(latents: torch.Tensor, temporal_patch_size: int) -> torch.Tensor:
    if latents.ndim != 5:
        raise ValueError(f"Wan VAE latents must have shape [B,C,T,H,W], got {tuple(latents.shape)}")
    if temporal_patch_size <= 0:
        raise ValueError(f"temporal_patch_size must be positive, got {temporal_patch_size}")
    batch_size, channels, temporal_length, height, width = latents.shape
    if temporal_length % temporal_patch_size != 0:
        raise ValueError(
            f"VAE temporal length={temporal_length} is not divisible by temporal patch size={temporal_patch_size}"
        )
    temporal_groups = temporal_length // temporal_patch_size
    grouped = latents.reshape(
        batch_size,
        channels,
        temporal_groups,
        temporal_patch_size,
        height,
        width,
    )
    return grouped.mean(dim=(3, 4, 5)).permute(0, 2, 1)


def pool_dit_temporal_groups(
    hidden_states: torch.Tensor,
    latent_shape: Sequence[int],
    patch_size: Sequence[int],
) -> torch.Tensor:
    if hidden_states.ndim != 3:
        raise ValueError(f"Wan hidden states must have shape [B,N,H], got {tuple(hidden_states.shape)}")
    if len(latent_shape) != 5:
        raise ValueError(f"latent_shape must describe [B,C,T,H,W], got {tuple(latent_shape)}")
    if len(patch_size) != 3:
        raise ValueError(f"Wan patch_size must contain three values, got {tuple(patch_size)}")
    batch_size, _channels, temporal_length, height, width = [int(value) for value in latent_shape]
    temporal_patch, height_patch, width_patch = [int(value) for value in patch_size]
    if min(temporal_patch, height_patch, width_patch) <= 0:
        raise ValueError(f"Wan patch sizes must be positive, got {tuple(patch_size)}")
    if temporal_length % temporal_patch or height % height_patch or width % width_patch:
        raise ValueError(
            f"Latent shape={tuple(latent_shape)} is not divisible by patch_size={tuple(patch_size)}"
        )
    temporal_groups = temporal_length // temporal_patch
    spatial_tokens = (height // height_patch) * (width // width_patch)
    expected_tokens = temporal_groups * spatial_tokens
    if hidden_states.shape[0] != batch_size or hidden_states.shape[1] != expected_tokens:
        raise ValueError(
            f"Wan hidden shape={tuple(hidden_states.shape)} does not match latent_shape={tuple(latent_shape)} "
            f"and patch_size={tuple(patch_size)}; expected [B,{expected_tokens},H]"
        )
    # Wan Conv3d patch embedding flattens [T,H,W] in row-major order, so each
    # contiguous spatial block belongs to one temporal patch.
    return hidden_states.reshape(batch_size, temporal_groups, spatial_tokens, hidden_states.shape[-1]).mean(dim=2)


def two_group_delta_feature(temporal_groups: torch.Tensor, feature_name: str) -> torch.Tensor:
    if temporal_groups.ndim != 3:
        raise ValueError(
            f"{feature_name} temporal groups must have shape [B,T,H], got {tuple(temporal_groups.shape)}"
        )
    if temporal_groups.shape[1] != 2:
        raise ValueError(
            f"{feature_name} requires exactly two Wan temporal groups, got shape={tuple(temporal_groups.shape)}"
        )
    first = temporal_groups[:, 0]
    second = temporal_groups[:, 1]
    return torch.cat((first, second, second - first), dim=-1)


def _training_image_size(model: WanOFTProbeModel) -> object | None:
    datasets_config = getattr(model.config, "datasets", None)
    vla_data_config = getattr(datasets_config, "vla_data", None)
    return getattr(vla_data_config, "obs_image_size", None)


@torch.inference_mode()
def extract_wan_oft_probe_batch(
    model: WanOFTProbeModel,
    examples: Sequence[ProbeExample],
    vae_seed: int,
) -> WanProbeFeatureBatch:
    if not examples:
        raise ValueError("Cannot extract WanOFT probe features from an empty batch")
    device = next(model.parameters()).device
    batch_images: list[list[object]] = [list(example["frames"]) for example in examples]
    image_size = _training_image_size(model)
    if image_size is not None:
        batch_images = resize_images(batch_images, target_size=image_size)
    instructions = [example["prompt"] for example in examples]
    states = [example["state"] for example in examples]
    instructions = add_discretized_state_to_instruction(instructions, states)

    with fork_seeded_torch_rng(device, vae_seed):
        wm_inputs = model.backbone.build_inputs(images=batch_images, instructions=instructions)
        latent_value = wm_inputs.get("hidden_states")
        if not isinstance(latent_value, torch.Tensor):
            raise TypeError(f"Wan backbone hidden_states input must be a Tensor, got {type(latent_value)}")
        outputs = model.backbone(**wm_inputs, output_hidden_states=True, return_dict=True)

    if not outputs.hidden_states:
        raise RuntimeError("Wan backbone returned no hidden states for probing")
    last_hidden = outputs.hidden_states[-1]
    patch_size = tuple(int(value) for value in model.backbone.transformer.config.patch_size)
    temporal_patch_size = patch_size[0]

    vae_groups = pool_vae_temporal_groups(latent_value, temporal_patch_size=temporal_patch_size)
    dit_groups = pool_dit_temporal_groups(
        hidden_states=last_hidden,
        latent_shape=latent_value.shape,
        patch_size=patch_size,
    )
    action_queries = model._pool_to_action_queries(last_hidden)
    action_predictions = model.action_model.predict_action(action_queries)
    if action_predictions.ndim != 3 or action_predictions.shape[1] < 1:
        raise ValueError(f"WanOFT action predictions must have shape [B,T,A], got {tuple(action_predictions.shape)}")
    if model.action_env_dim <= 1 or model.action_env_dim > action_predictions.shape[-1]:
        raise ValueError(
            f"Invalid action_env_dim={model.action_env_dim} for predictions shape={tuple(action_predictions.shape)}"
        )

    return WanProbeFeatureBatch(
        vae_temporal_delta=two_group_delta_feature(vae_groups, VAE_TEMPORAL_FEATURE).detach().to("cpu", torch.float16),
        dit_global_mean=last_hidden.mean(dim=1).detach().to("cpu", torch.float16),
        dit_temporal_delta=two_group_delta_feature(dit_groups, DIT_TEMPORAL_FEATURE).detach().to("cpu", torch.float16),
        action_query=action_queries[:, 0].detach().to("cpu", torch.float16),
        action_logits=action_predictions[:, 0, : model.action_env_dim].detach().to("cpu", torch.float32),
    )


def extract_probe_dataset(
    model: WanOFTProbeModel,
    batches: Iterator[list[ProbeExample]],
    transform: ProbeExampleTransform,
    control_seed: int,
    vae_seed: int,
    description: str,
) -> ExtractedProbeDataset:
    sample_id_parts: list[torch.Tensor] = []
    label_parts: dict[str, list[torch.Tensor]] = {label_name: [] for label_name in PROBE_LABEL_NAMES}
    feature_parts: dict[str, list[torch.Tensor]] = {feature_name: [] for feature_name in PROBE_FEATURE_NAMES}
    action_logit_parts: list[torch.Tensor] = []

    for batch_index, examples in enumerate(tqdm(batches, desc=description, unit="batch")):
        transformed = transform(examples, control_seed)
        feature_batch = extract_wan_oft_probe_batch(
            model=model,
            examples=transformed,
            vae_seed=vae_seed + batch_index,
        )
        sample_id_parts.append(
            torch.tensor(
                [
                    [example["episode_index"], example["frame_index"], example["decision_step"]]
                    for example in examples
                ],
                dtype=torch.int64,
            )
        )
        for label_name in PROBE_LABEL_NAMES:
            label_parts[label_name].append(
                torch.tensor([example["labels"][label_name] for example in examples], dtype=torch.int64)
            )
        for feature_name in PROBE_FEATURE_NAMES:
            feature_parts[feature_name].append(feature_batch[feature_name])
        action_logit_parts.append(feature_batch["action_logits"])

    if not sample_id_parts:
        raise ValueError(f"No examples were extracted for {description}")
    return ExtractedProbeDataset(
        sample_ids=torch.cat(sample_id_parts, dim=0),
        labels={label_name: torch.cat(parts, dim=0) for label_name, parts in label_parts.items()},
        features={feature_name: torch.cat(parts, dim=0) for feature_name, parts in feature_parts.items()},
        action_logits=torch.cat(action_logit_parts, dim=0),
    )


def probe_target_set(dataset: ExtractedProbeDataset) -> ProbeTargetSet:
    return ProbeTargetSet(
        sample_ids=dataset["sample_ids"],
        labels={label_name: values for label_name, values in dataset["labels"].items()},
    )


def assert_matching_probe_targets(
    reference: ProbeTargetSet,
    candidate: ProbeTargetSet,
    context: str,
) -> None:
    if not torch.equal(reference["sample_ids"], candidate["sample_ids"]):
        raise ValueError(f"Probe sample identities changed across {context}")
    for label_name in PROBE_LABEL_NAMES:
        if not torch.equal(reference["labels"][label_name], candidate["labels"][label_name]):
            raise ValueError(f"Probe label {label_name!r} changed across {context}")
