#!/usr/bin/env python
from __future__ import annotations

import argparse
import gc
import json
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from typing import cast

import torch
import wandb
from omegaconf import DictConfig, OmegaConf
from wandb.sdk.wandb_run import Run

from starVLA.training.rl_games.linear_probe import (
    LinearProbeConfig,
    classification_metrics,
    current_action_class_spec,
    latency_class_spec,
    timing_class_spec,
    train_linear_probe,
    unavailable_latency_probe_report,
)
from starVLA.training.rl_games.wan_oft_probe_data import (
    CURRENT_ACTION_LABEL,
    LATENCY_ID_LABEL,
    PROBE_LABEL_NAMES,
    TIME_SINCE_LAST_FLAP_LABEL,
    TIME_TO_NEXT_FLAP_LABEL,
    ProbeExample,
    iter_probe_batches,
    latency_neutral_prompt_examples,
    load_task_prompts,
    normal_examples,
    repeated_last_frame_examples,
    select_episode_paths,
    shuffled_frame_examples,
)
from starVLA.training.rl_games.wan_oft_probe_features import (
    PROBE_FEATURE_NAMES,
    ExtractedProbeDataset,
    ProbeExampleTransform,
    ProbeTargetSet,
    WanOFTProbeModel,
    assert_matching_probe_targets,
    extract_probe_dataset,
    probe_target_set,
)

CHECKPOINTS = ("pre_sft", "post_sft")
CONDITIONS: tuple[tuple[str, ProbeExampleTransform], ...] = (
    ("normal", normal_examples),
    ("shuffled_frames", shuffled_frame_examples),
    ("repeated_last_frame", repeated_last_frame_examples),
    ("latency_neutral_prompt", latency_neutral_prompt_examples),
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Probe frozen WanOFT temporal representations before and after Flappy SFT."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--base-model-dir", type=Path, required=True)
    parser.add_argument("--pre-sft-checkpoint", type=Path, required=True)
    parser.add_argument("--post-sft-checkpoint", type=Path, required=True)
    parser.add_argument("--train-dataset-dir", type=Path, required=True)
    parser.add_argument("--validation-dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-device", type=str, required=True)
    parser.add_argument("--probe-device", type=str, required=True)
    parser.add_argument("--extraction-batch-size", type=int, required=True)
    parser.add_argument("--max-train-episodes", type=int, required=True)
    parser.add_argument("--max-validation-episodes", type=int, required=True)
    parser.add_argument("--image-sequence-length", type=int, required=True)
    parser.add_argument("--maximum-exact-distance", type=int, required=True)
    parser.add_argument("--flap-action-id", type=int, required=True)
    parser.add_argument("--selection-seed", type=int, required=True)
    parser.add_argument("--control-seed", type=int, required=True)
    parser.add_argument("--vae-seed", type=int, required=True)
    parser.add_argument("--probe-seed", type=int, required=True)
    parser.add_argument("--probe-epochs", type=int, required=True)
    parser.add_argument("--probe-batch-size", type=int, required=True)
    parser.add_argument("--probe-learning-rate", type=float, required=True)
    parser.add_argument("--probe-weight-decay", type=float, required=True)
    parser.add_argument("--wandb-entity", type=str, required=True)
    parser.add_argument("--wandb-project", type=str, required=True)
    parser.add_argument("--wandb-run-name", type=str, required=True)
    return parser.parse_args()


def _require_file(path: Path, label: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} does not exist or is not a file: {resolved}")
    return resolved


def _require_directory(path: Path, label: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_dir():
        raise FileNotFoundError(f"{label} does not exist or is not a directory: {resolved}")
    return resolved


def _prepare_output_directory(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    if resolved.exists():
        raise FileExistsError(
            f"Probe output directory already exists: {resolved}. Use a new output directory for an immutable experiment."
        )
    resolved.mkdir(parents=True)
    return resolved


def _load_probe_config(config_path: Path, base_model_dir: Path, image_sequence_length: int) -> DictConfig:
    from starVLA.model.framework.share_tools import apply_config_compat
    from starVLA.training.rl_games import apply_action_spec, apply_model_alias

    cfg = OmegaConf.load(str(config_path))
    OmegaConf.update(cfg, "framework.qwenvl.base_vlm", str(base_model_dir), merge=True)
    OmegaConf.update(cfg, "framework.world_model.base_wm", str(base_model_dir), merge=True)
    cfg = apply_config_compat(cfg)
    apply_model_alias(cfg)
    apply_action_spec(cfg)

    framework_name = str(OmegaConf.select(cfg, "framework.name", default=""))
    configured_frames = int(OmegaConf.select(cfg, "framework.world_model.num_frames", default=0) or 0)
    configured_sequence_length = int(OmegaConf.select(cfg, "datasets.vla_data.image_sequence_length", default=0) or 0)
    action_query_source = str(OmegaConf.select(cfg, "framework.action_model.action_query_source", default=""))
    if framework_name != "WanOFT":
        raise ValueError(f"Probe config must build framework.name=WanOFT, got {framework_name!r}")
    if configured_frames != image_sequence_length or configured_sequence_length != image_sequence_length:
        raise ValueError(
            f"Probe requires matching five-frame config, got world_model.num_frames={configured_frames}, "
            f"datasets.vla_data.image_sequence_length={configured_sequence_length}, requested={image_sequence_length}"
        )
    if action_query_source != "mean":
        raise ValueError(
            f"WanOFT v0 temporal probe is defined for action_query_source='mean', got {action_query_source!r}"
        )
    return cfg


def _load_checkpoint_state(checkpoint_path: Path) -> dict[str, torch.Tensor]:
    if checkpoint_path.suffix == ".safetensors":
        from safetensors.torch import load_file

        return load_file(str(checkpoint_path))
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or not all(isinstance(key, str) for key in state):
        raise TypeError(f"Checkpoint must contain a tensor state_dict, got {type(state)} from {checkpoint_path}")
    non_tensor_keys = [key for key, value in state.items() if not isinstance(value, torch.Tensor)]
    if non_tensor_keys:
        raise TypeError(f"Checkpoint {checkpoint_path} contains non-tensor state entries: {non_tensor_keys[:10]}")
    return cast(dict[str, torch.Tensor], state)


def _build_frozen_model(cfg: DictConfig, checkpoint_path: Path, device_name: str) -> WanOFTProbeModel:
    from starVLA.model.framework.base_framework import build_framework

    model_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    model = build_framework(model_cfg)
    state = _load_checkpoint_state(checkpoint_path)
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as error:
        raise RuntimeError(
            f"Checkpoint {checkpoint_path} does not exactly match the configured WanOFT architecture"
        ) from error
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.to(torch.device(device_name))
    model.eval()
    return cast(WanOFTProbeModel, model)


def _probe_config(args: argparse.Namespace) -> LinearProbeConfig:
    return LinearProbeConfig(
        epochs=int(args.probe_epochs),
        batch_size=int(args.probe_batch_size),
        learning_rate=float(args.probe_learning_rate),
        weight_decay=float(args.probe_weight_decay),
        seed=int(args.probe_seed),
        device=str(args.probe_device),
    )


def _dataset_batches(
    episode_paths: Sequence[Path],
    task_prompts: dict[int, str],
    args: argparse.Namespace,
) -> Iterator[list[ProbeExample]]:
    return iter_probe_batches(
        episode_paths=episode_paths,
        task_prompts=task_prompts,
        image_sequence_length=int(args.image_sequence_length),
        maximum_exact_distance=int(args.maximum_exact_distance),
        flap_action_id=int(args.flap_action_id),
        batch_size=int(args.extraction_batch_size),
    )


def _extract_split(
    model: WanOFTProbeModel,
    episode_paths: Sequence[Path],
    task_prompts: dict[int, str],
    transform: Callable[[Sequence[ProbeExample], int], list[ProbeExample]],
    description: str,
    args: argparse.Namespace,
) -> ExtractedProbeDataset:
    return extract_probe_dataset(
        model=model,
        batches=_dataset_batches(episode_paths, task_prompts, args),
        transform=transform,
        control_seed=int(args.control_seed),
        vae_seed=int(args.vae_seed),
        description=description,
    )


def _label_specs(
    train_data: ExtractedProbeDataset,
    validation_data: ExtractedProbeDataset,
    maximum_exact_distance: int,
) -> dict[str, tuple[list[int], list[str]]]:
    current_values, current_names = current_action_class_spec()
    timing_values, timing_names = timing_class_spec(maximum_exact_distance)
    latency_values, latency_names = latency_class_spec(
        train_data["labels"][LATENCY_ID_LABEL],
        validation_data["labels"][LATENCY_ID_LABEL],
    )
    return {
        CURRENT_ACTION_LABEL: (current_values, current_names),
        TIME_TO_NEXT_FLAP_LABEL: (timing_values, timing_names),
        TIME_SINCE_LAST_FLAP_LABEL: (timing_values, timing_names),
        LATENCY_ID_LABEL: (latency_values, latency_names),
    }


def _latency_probe_is_available(
    train_data: ExtractedProbeDataset,
    validation_data: ExtractedProbeDataset,
) -> bool:
    train_values = {int(value) for value in train_data["labels"][LATENCY_ID_LABEL].tolist()}
    validation_values = {int(value) for value in validation_data["labels"][LATENCY_ID_LABEL].tolist()}
    if len(train_values) < 2 or len(validation_values) < 2:
        return False
    if train_values != validation_values:
        raise ValueError(
            f"Latency classes differ between probe train and validation splits: {sorted(train_values)} vs "
            f"{sorted(validation_values)}"
        )
    return True


def _save_probe_state(
    output_dir: Path,
    checkpoint_name: str,
    condition_name: str,
    feature_name: str,
    label_name: str,
    state: dict[str, torch.Tensor],
) -> str:
    path = output_dir / "probe_states" / checkpoint_name / condition_name / feature_name / f"{label_name}.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)
    return str(path.relative_to(output_dir))


def _run_condition_probes(
    checkpoint_name: str,
    condition_name: str,
    train_data: ExtractedProbeDataset,
    validation_data: ExtractedProbeDataset,
    output_dir: Path,
    config: LinearProbeConfig,
    maximum_exact_distance: int,
) -> dict[str, object]:
    current_values, current_names = current_action_class_spec()
    direct_predictions = validation_data["action_logits"].argmax(dim=-1).to(torch.int64)
    direct_metrics = classification_metrics(
        targets=validation_data["labels"][CURRENT_ACTION_LABEL],
        predictions=direct_predictions,
        class_values=current_values,
        class_names=current_names,
    )
    specs = _label_specs(train_data, validation_data, maximum_exact_distance)
    latency_available = _latency_probe_is_available(train_data, validation_data)
    probe_reports: dict[str, object] = {}

    for feature_name in PROBE_FEATURE_NAMES:
        label_reports: dict[str, object] = {}
        for label_name in PROBE_LABEL_NAMES:
            if label_name == LATENCY_ID_LABEL and not latency_available:
                label_reports[label_name] = unavailable_latency_probe_report(
                    train_labels=train_data["labels"][label_name],
                    validation_labels=validation_data["labels"][label_name],
                )
                continue
            class_values, class_names = specs[label_name]
            probe_report, probe_state = train_linear_probe(
                train_features=train_data["features"][feature_name],
                train_labels=train_data["labels"][label_name],
                validation_features=validation_data["features"][feature_name],
                validation_labels=validation_data["labels"][label_name],
                class_values=class_values,
                class_names=class_names,
                config=config,
            )
            state_path = _save_probe_state(
                output_dir=output_dir,
                checkpoint_name=checkpoint_name,
                condition_name=condition_name,
                feature_name=feature_name,
                label_name=label_name,
                state=probe_state,
            )
            label_reports[label_name] = {**probe_report, "state_path": state_path}
        probe_reports[feature_name] = label_reports

    return {
        "train_samples": int(train_data["sample_ids"].shape[0]),
        "validation_samples": int(validation_data["sample_ids"].shape[0]),
        "feature_dimensions": {
            feature_name: int(train_data["features"][feature_name].shape[1])
            for feature_name in PROBE_FEATURE_NAMES
        },
        "direct_action_metrics": direct_metrics,
        "probes": probe_reports,
    }


def _macro_f1(
    results: dict[str, object],
    checkpoint_name: str,
    condition_name: str,
    feature_name: str,
    label_name: str,
) -> float | None:
    checkpoint_results = cast(dict[str, object], results[checkpoint_name])
    condition_results = cast(dict[str, object], checkpoint_results[condition_name])
    probes = cast(dict[str, object], condition_results["probes"])
    feature_reports = cast(dict[str, object], probes[feature_name])
    label_report = cast(dict[str, object], feature_reports[label_name])
    if label_report.get("status") != "evaluated":
        return None
    metrics = cast(dict[str, object], label_report["metrics"])
    return float(metrics["macro_f1"])


def _comparison_records(results: dict[str, object]) -> dict[str, list[dict[str, object]]]:
    pre_post: list[dict[str, object]] = []
    control_drops: list[dict[str, object]] = []
    for condition_name, _transform in CONDITIONS:
        for feature_name in PROBE_FEATURE_NAMES:
            for label_name in PROBE_LABEL_NAMES:
                pre_score = _macro_f1(results, "pre_sft", condition_name, feature_name, label_name)
                post_score = _macro_f1(results, "post_sft", condition_name, feature_name, label_name)
                if pre_score is not None and post_score is not None:
                    pre_post.append(
                        {
                            "condition": condition_name,
                            "feature": feature_name,
                            "label": label_name,
                            "pre_sft_macro_f1": pre_score,
                            "post_sft_macro_f1": post_score,
                            "post_minus_pre_macro_f1": post_score - pre_score,
                        }
                    )
    for checkpoint_name in CHECKPOINTS:
        for condition_name, _transform in CONDITIONS[1:]:
            for feature_name in PROBE_FEATURE_NAMES:
                for label_name in PROBE_LABEL_NAMES:
                    normal_score = _macro_f1(results, checkpoint_name, "normal", feature_name, label_name)
                    control_score = _macro_f1(results, checkpoint_name, condition_name, feature_name, label_name)
                    if normal_score is not None and control_score is not None:
                        control_drops.append(
                            {
                                "checkpoint": checkpoint_name,
                                "control": condition_name,
                                "feature": feature_name,
                                "label": label_name,
                                "normal_macro_f1": normal_score,
                                "control_macro_f1": control_score,
                                "normal_minus_control_macro_f1": normal_score - control_score,
                            }
                        )
    return {"pre_post": pre_post, "control_drops": control_drops}


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _wandb_condition_metrics(
    checkpoint_name: str,
    condition_name: str,
    condition_results: dict[str, object],
) -> dict[str, float | int]:
    prefix = f"probe/{checkpoint_name}/{condition_name}"
    direct_metrics = cast(dict[str, object], condition_results["direct_action_metrics"])
    payload: dict[str, float | int] = {
        f"{prefix}/train_samples": int(condition_results["train_samples"]),
        f"{prefix}/validation_samples": int(condition_results["validation_samples"]),
        f"{prefix}/direct_action/accuracy": float(direct_metrics["accuracy"]),
        f"{prefix}/direct_action/balanced_accuracy": float(direct_metrics["balanced_accuracy"]),
        f"{prefix}/direct_action/macro_f1": float(direct_metrics["macro_f1"]),
    }
    probes = cast(dict[str, object], condition_results["probes"])
    for feature_name in PROBE_FEATURE_NAMES:
        feature_reports = cast(dict[str, object], probes[feature_name])
        for label_name in PROBE_LABEL_NAMES:
            report = cast(dict[str, object], feature_reports[label_name])
            if report.get("status") != "evaluated":
                continue
            metrics = cast(dict[str, object], report["metrics"])
            report_prefix = f"{prefix}/{feature_name}/{label_name}"
            payload.update(
                {
                    f"{report_prefix}/accuracy": float(metrics["accuracy"]),
                    f"{report_prefix}/balanced_accuracy": float(metrics["balanced_accuracy"]),
                    f"{report_prefix}/macro_f1": float(metrics["macro_f1"]),
                    f"{report_prefix}/first_epoch_loss": float(report["first_epoch_loss"]),
                    f"{report_prefix}/final_epoch_loss": float(report["final_epoch_loss"]),
                }
            )
    return payload


def _wandb_comparison_table(records: list[dict[str, object]]) -> wandb.Table | None:
    if not records:
        return None
    columns = list(records[0])
    if any(list(record) != columns for record in records):
        raise ValueError("W&B comparison records must use one consistent ordered schema")
    return wandb.Table(columns=columns, data=[[record[column] for column in columns] for record in records])


def _log_wandb_comparisons(run: Run, comparisons: dict[str, list[dict[str, object]]]) -> None:
    payload: dict[str, wandb.Table] = {}
    for comparison_name, records in comparisons.items():
        table = _wandb_comparison_table(records)
        if table is not None:
            payload[f"comparisons/{comparison_name}"] = table
    if payload:
        run.log(payload)


def _validate_devices(model_device_name: str, probe_device_name: str) -> None:
    model_device = torch.device(model_device_name)
    probe_device = torch.device(probe_device_name)
    if model_device.type != "cuda":
        raise ValueError(f"WanOFT feature extraction requires a CUDA model device, got {model_device_name!r}")
    if not torch.cuda.is_available():
        raise RuntimeError("WanOFT feature extraction requires CUDA, but torch.cuda.is_available() is false")
    if probe_device.type not in {"cpu", "cuda"}:
        raise ValueError(f"Linear probing supports a CPU or CUDA probe device, got {probe_device_name!r}")
    device_count = torch.cuda.device_count()
    for label, device in (("model", model_device), ("probe", probe_device)):
        if device.type != "cuda" or device.index is None:
            continue
        if device.index < 0 or device.index >= device_count:
            raise ValueError(
                f"Configured {label} device={device} is unavailable; torch reports {device_count} CUDA device(s)"
            )


def main() -> int:
    args = _parse_args()
    _validate_devices(str(args.model_device), str(args.probe_device))
    config_path = _require_file(args.config, "config")
    base_model_dir = _require_directory(args.base_model_dir, "base model directory")
    checkpoint_paths = {
        "pre_sft": _require_file(args.pre_sft_checkpoint, "pre-SFT checkpoint"),
        "post_sft": _require_file(args.post_sft_checkpoint, "post-SFT checkpoint"),
    }
    train_dataset_dir = _require_directory(args.train_dataset_dir, "probe train dataset")
    validation_dataset_dir = _require_directory(args.validation_dataset_dir, "probe validation dataset")
    output_dir = _prepare_output_directory(args.output_dir)
    cfg = _load_probe_config(config_path, base_model_dir, int(args.image_sequence_length))
    train_episode_paths = select_episode_paths(
        dataset_dir=train_dataset_dir,
        max_episodes=int(args.max_train_episodes),
        seed=int(args.selection_seed),
    )
    validation_episode_paths = select_episode_paths(
        dataset_dir=validation_dataset_dir,
        max_episodes=int(args.max_validation_episodes),
        seed=int(args.selection_seed) + 1,
    )
    train_prompts = load_task_prompts(train_dataset_dir)
    validation_prompts = load_task_prompts(validation_dataset_dir)
    probe_config = _probe_config(args)
    wandb_run = wandb.init(
        entity=str(args.wandb_entity),
        project=str(args.wandb_project),
        name=str(args.wandb_run_name),
        tags=["WanOFT", "linear-probe", "flappy", "latency-2", "context5"],
        config={
            "objective": "WanOFT pre-SFT versus post-SFT temporal representation linear probing",
            "config_path": str(config_path),
            "base_model_dir": str(base_model_dir),
            "pre_sft_checkpoint": str(checkpoint_paths["pre_sft"]),
            "post_sft_checkpoint": str(checkpoint_paths["post_sft"]),
            "train_dataset_dir": str(train_dataset_dir),
            "validation_dataset_dir": str(validation_dataset_dir),
            "max_train_episodes": int(args.max_train_episodes),
            "max_validation_episodes": int(args.max_validation_episodes),
            "extraction_batch_size": int(args.extraction_batch_size),
            "image_sequence_length": int(args.image_sequence_length),
            "maximum_exact_distance": int(args.maximum_exact_distance),
            "selection_seed": int(args.selection_seed),
            "control_seed": int(args.control_seed),
            "vae_seed": int(args.vae_seed),
            "linear_probe": dict(probe_config),
        },
    )
    if wandb_run is None:
        raise RuntimeError("wandb.init returned no run for the WanOFT temporal probe")
    results: dict[str, object] = {}
    reference_train: ProbeTargetSet | None = None
    reference_validation: ProbeTargetSet | None = None
    wandb_step = 0

    for checkpoint_name in CHECKPOINTS:
        model = _build_frozen_model(cfg, checkpoint_paths[checkpoint_name], str(args.model_device))
        checkpoint_results: dict[str, object] = {}
        for condition_name, transform in CONDITIONS:
            train_data = _extract_split(
                model=model,
                episode_paths=train_episode_paths,
                task_prompts=train_prompts,
                transform=transform,
                description=f"{checkpoint_name}/{condition_name}/train",
                args=args,
            )
            validation_data = _extract_split(
                model=model,
                episode_paths=validation_episode_paths,
                task_prompts=validation_prompts,
                transform=transform,
                description=f"{checkpoint_name}/{condition_name}/validation",
                args=args,
            )
            if reference_train is None or reference_validation is None:
                reference_train = probe_target_set(train_data)
                reference_validation = probe_target_set(validation_data)
            else:
                assert_matching_probe_targets(
                    reference_train,
                    probe_target_set(train_data),
                    f"{checkpoint_name}/{condition_name}/train",
                )
                assert_matching_probe_targets(
                    reference_validation,
                    probe_target_set(validation_data),
                    f"{checkpoint_name}/{condition_name}/validation",
                )
            condition_results = _run_condition_probes(
                checkpoint_name=checkpoint_name,
                condition_name=condition_name,
                train_data=train_data,
                validation_data=validation_data,
                output_dir=output_dir,
                config=probe_config,
                maximum_exact_distance=int(args.maximum_exact_distance),
            )
            checkpoint_results[condition_name] = condition_results
            wandb_run.log(
                _wandb_condition_metrics(checkpoint_name, condition_name, condition_results),
                step=wandb_step,
            )
            wandb_step += 1
            del train_data
            del validation_data
            gc.collect()
        results[checkpoint_name] = checkpoint_results
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    comparisons = _comparison_records(results)
    report: dict[str, object] = {
        "schema_version": 1,
        "objective": "WanOFT pre-SFT versus post-SFT temporal representation linear probing",
        "config": str(config_path),
        "base_model_dir": str(base_model_dir),
        "checkpoints": {name: str(path) for name, path in checkpoint_paths.items()},
        "datasets": {
            "train": str(train_dataset_dir),
            "validation": str(validation_dataset_dir),
            "selected_train_episodes": [str(path.relative_to(train_dataset_dir)) for path in train_episode_paths],
            "selected_validation_episodes": [
                str(path.relative_to(validation_dataset_dir)) for path in validation_episode_paths
            ],
        },
        "seeds": {
            "episode_selection": int(args.selection_seed),
            "controls": int(args.control_seed),
            "vae": int(args.vae_seed),
            "linear_probe": int(args.probe_seed),
        },
        "linear_probe": dict(probe_config),
        "wandb": {
            "entity": str(args.wandb_entity),
            "project": str(args.wandb_project),
            "run_name": str(args.wandb_run_name),
            "run_id": wandb_run.id,
            "run_url": wandb_run.url,
        },
        "results": results,
        "comparisons": comparisons,
    }
    report_path = output_dir / "report.json"
    _write_json(report_path, report)
    _log_wandb_comparisons(wandb_run, comparisons)
    report_artifact = wandb.Artifact(
        name=f"{args.wandb_run_name}-report",
        type="wan-oft-linear-probe-report",
    )
    report_artifact.add_file(str(report_path))
    wandb_run.log_artifact(report_artifact)
    wandb_run.summary["report_path"] = str(report_path)
    wandb_run.finish()
    print(json.dumps(report["comparisons"], indent=2, sort_keys=True))
    print(f"WanOFT temporal probe report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
