from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import TypedDict

import torch
import torch.nn.functional as F


class LinearProbeConfig(TypedDict):
    epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    seed: int
    device: str


class PerClassMetrics(TypedDict):
    class_value: int
    precision: float
    recall: float
    f1: float
    support: int


class ClassificationMetrics(TypedDict):
    accuracy: float
    balanced_accuracy: float
    macro_f1: float
    confusion_matrix: list[list[int]]
    per_class: dict[str, PerClassMetrics]


class LinearProbeReport(TypedDict):
    status: str
    feature_dim: int
    class_values: list[int]
    class_names: list[str]
    train_samples: int
    validation_samples: int
    train_class_counts: dict[str, int]
    validation_class_counts: dict[str, int]
    first_epoch_loss: float
    final_epoch_loss: float
    metrics: ClassificationMetrics


class UnavailableProbeReport(TypedDict):
    status: str
    reason: str
    observed_train_values: list[int]
    observed_validation_values: list[int]


def current_action_class_spec() -> tuple[list[int], list[str]]:
    return [0, 1], ["NOOP", "FLAP"]


def timing_class_spec(maximum_exact_distance: int) -> tuple[list[int], list[str]]:
    if maximum_exact_distance <= 0:
        raise ValueError(f"maximum_exact_distance must be positive, got {maximum_exact_distance}")
    class_values = list(range(maximum_exact_distance + 1))
    class_names = [str(value) for value in range(maximum_exact_distance)] + [f"{maximum_exact_distance}+"]
    return class_values, class_names


def latency_class_spec(train_labels: torch.Tensor, validation_labels: torch.Tensor) -> tuple[list[int], list[str]]:
    observed = sorted(
        {int(value) for value in train_labels.tolist()} | {int(value) for value in validation_labels.tolist()}
    )
    return observed, [f"latency_{value}" for value in observed]


def unavailable_latency_probe_report(
    train_labels: torch.Tensor,
    validation_labels: torch.Tensor,
) -> UnavailableProbeReport:
    train_values = sorted({int(value) for value in train_labels.tolist()})
    validation_values = sorted({int(value) for value in validation_labels.tolist()})
    return UnavailableProbeReport(
        status="unavailable",
        reason="latency_id requires at least two latency classes in both the probe train and validation splits",
        observed_train_values=train_values,
        observed_validation_values=validation_values,
    )


def _validate_class_spec(class_values: list[int], class_names: list[str]) -> None:
    if len(class_values) < 2:
        raise ValueError(f"Linear classification requires at least two classes, got {class_values}")
    if len(class_values) != len(class_names):
        raise ValueError(f"class_values and class_names must align, got {class_values} and {class_names}")
    if len(set(class_values)) != len(class_values):
        raise ValueError(f"class_values must be unique, got {class_values}")
    if len(set(class_names)) != len(class_names):
        raise ValueError(f"class_names must be unique, got {class_names}")


def _encode_labels(labels: torch.Tensor, class_values: list[int]) -> torch.Tensor:
    if labels.ndim != 1:
        raise ValueError(f"Probe labels must be one-dimensional, got shape={tuple(labels.shape)}")
    value_to_index = {value: index for index, value in enumerate(class_values)}
    unknown = sorted({int(value) for value in labels.tolist()} - set(value_to_index))
    if unknown:
        raise ValueError(f"Probe labels contain values outside class_values={class_values}: {unknown}")
    return torch.tensor([value_to_index[int(value)] for value in labels.tolist()], dtype=torch.int64)


def class_counts(labels: torch.Tensor, class_values: list[int], class_names: list[str]) -> dict[str, int]:
    _validate_class_spec(class_values, class_names)
    return {
        class_name: int(torch.count_nonzero(labels == class_value).item())
        for class_value, class_name in zip(class_values, class_names, strict=True)
    }


def classification_metrics(
    targets: torch.Tensor,
    predictions: torch.Tensor,
    class_values: list[int],
    class_names: list[str],
) -> ClassificationMetrics:
    _validate_class_spec(class_values, class_names)
    if targets.shape != predictions.shape or targets.ndim != 1:
        raise ValueError(
            f"targets and predictions must be aligned one-dimensional tensors, got {tuple(targets.shape)} and "
            f"{tuple(predictions.shape)}"
        )
    if targets.numel() == 0:
        raise ValueError("Cannot compute classification metrics for an empty validation set")
    encoded_targets = _encode_labels(targets.to("cpu", torch.int64), class_values)
    encoded_predictions = _encode_labels(predictions.to("cpu", torch.int64), class_values)
    class_count = len(class_values)
    confusion = torch.zeros((class_count, class_count), dtype=torch.int64)
    flat_indices = encoded_targets * class_count + encoded_predictions
    confusion += torch.bincount(flat_indices, minlength=class_count * class_count).reshape(class_count, class_count)

    per_class: dict[str, PerClassMetrics] = {}
    recalls: list[float] = []
    f1_scores: list[float] = []
    for class_index, (class_value, class_name) in enumerate(zip(class_values, class_names, strict=True)):
        true_positive = int(confusion[class_index, class_index].item())
        false_positive = int(confusion[:, class_index].sum().item()) - true_positive
        false_negative = int(confusion[class_index, :].sum().item()) - true_positive
        support = int(confusion[class_index, :].sum().item())
        precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
        recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
        recalls.append(recall)
        f1_scores.append(f1)
        per_class[class_name] = PerClassMetrics(
            class_value=class_value,
            precision=float(precision),
            recall=float(recall),
            f1=float(f1),
            support=support,
        )

    accuracy = float(torch.count_nonzero(encoded_targets == encoded_predictions).item() / targets.numel())
    return ClassificationMetrics(
        accuracy=accuracy,
        balanced_accuracy=float(sum(recalls) / len(recalls)),
        macro_f1=float(sum(f1_scores) / len(f1_scores)),
        confusion_matrix=[[int(value) for value in row] for row in confusion.tolist()],
        per_class=per_class,
    )


def _validate_probe_config(config: LinearProbeConfig) -> None:
    if config["epochs"] <= 0:
        raise ValueError(f"epochs must be positive, got {config['epochs']}")
    if config["batch_size"] <= 0:
        raise ValueError(f"batch_size must be positive, got {config['batch_size']}")
    if config["learning_rate"] <= 0.0:
        raise ValueError(f"learning_rate must be positive, got {config['learning_rate']}")
    if config["weight_decay"] < 0.0:
        raise ValueError(f"weight_decay must be non-negative, got {config['weight_decay']}")


def _validate_feature_tensors(
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    validation_features: torch.Tensor,
    validation_labels: torch.Tensor,
) -> None:
    if train_features.ndim != 2 or validation_features.ndim != 2:
        raise ValueError(
            f"Probe features must have shape [N,D], got {tuple(train_features.shape)} and "
            f"{tuple(validation_features.shape)}"
        )
    if train_features.shape[0] != train_labels.shape[0]:
        raise ValueError(
            f"Train feature/label counts differ: {train_features.shape[0]} and {train_labels.shape[0]}"
        )
    if validation_features.shape[0] != validation_labels.shape[0]:
        raise ValueError(
            f"Validation feature/label counts differ: {validation_features.shape[0]} and "
            f"{validation_labels.shape[0]}"
        )
    if train_features.shape[1] != validation_features.shape[1]:
        raise ValueError(
            f"Train/validation feature dimensions differ: {train_features.shape[1]} and "
            f"{validation_features.shape[1]}"
        )
    if train_features.shape[0] == 0 or validation_features.shape[0] == 0:
        raise ValueError("Probe train and validation sets must both be non-empty")
    if not torch.isfinite(train_features).all() or not torch.isfinite(validation_features).all():
        raise ValueError("Probe features contain non-finite values")


@contextmanager
def fork_seeded_torch_rng(device: torch.device, seed: int) -> Iterator[None]:
    if device.type == "cuda":
        device_index = torch.cuda.current_device() if device.index is None else int(device.index)
        forked_devices = [device_index]
    elif device.type == "cpu":
        device_index = None
        forked_devices = []
    else:
        raise ValueError(f"Seeded probe RNG supports CPU or CUDA devices, got {device}")

    with torch.random.fork_rng(devices=forked_devices):
        if device_index is None:
            torch.random.default_generator.manual_seed(seed)
        else:
            with torch.cuda.device(device_index):
                torch.cuda.manual_seed(seed)
        yield


def _predict_linear_batches(
    classifier: torch.nn.Linear,
    features: torch.Tensor,
    feature_mean: torch.Tensor,
    feature_scale: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    predictions: list[torch.Tensor] = []
    classifier.eval()
    with torch.inference_mode():
        for start in range(0, features.shape[0], batch_size):
            end = min(start + batch_size, features.shape[0])
            batch = features[start:end].to(device=device, dtype=torch.float32)
            normalized = (batch - feature_mean) / feature_scale
            predictions.append(classifier(normalized).argmax(dim=-1).to("cpu"))
    return torch.cat(predictions, dim=0)


def train_linear_probe(
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    validation_features: torch.Tensor,
    validation_labels: torch.Tensor,
    class_values: list[int],
    class_names: list[str],
    config: LinearProbeConfig,
) -> tuple[LinearProbeReport, dict[str, torch.Tensor]]:
    _validate_probe_config(config)
    _validate_class_spec(class_values, class_names)
    _validate_feature_tensors(train_features, train_labels, validation_features, validation_labels)
    encoded_train = _encode_labels(train_labels.to("cpu", torch.int64), class_values)
    _encode_labels(validation_labels.to("cpu", torch.int64), class_values)
    train_counts = torch.bincount(encoded_train, minlength=len(class_values))
    missing_train_classes = [
        class_names[index] for index, count in enumerate(train_counts.tolist()) if int(count) == 0
    ]
    if missing_train_classes:
        raise ValueError(
            f"Probe training split is missing required classes {missing_train_classes}; select more episodes"
        )

    device = torch.device(config["device"])
    feature_mean_cpu = train_features.to(torch.float32).mean(dim=0)
    feature_std_cpu = train_features.to(torch.float32).std(dim=0, unbiased=False)
    feature_scale_cpu = torch.where(feature_std_cpu > 1.0e-6, feature_std_cpu, torch.ones_like(feature_std_cpu))
    feature_mean = feature_mean_cpu.to(device)
    feature_scale = feature_scale_cpu.to(device)
    class_weights = (encoded_train.numel() / (len(class_values) * train_counts.to(torch.float32))).to(device)

    with fork_seeded_torch_rng(device, config["seed"]):
        classifier = torch.nn.Linear(train_features.shape[1], len(class_values), device=device, dtype=torch.float32)

    optimizer = torch.optim.AdamW(
        classifier.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )
    permutation_generator = torch.Generator(device="cpu")
    permutation_generator.manual_seed(config["seed"])
    epoch_losses: list[float] = []
    classifier.train()
    for _epoch in range(config["epochs"]):
        permutation = torch.randperm(train_features.shape[0], generator=permutation_generator)
        epoch_loss_sum = 0.0
        for start in range(0, train_features.shape[0], config["batch_size"]):
            indices = permutation[start : start + config["batch_size"]]
            batch_features = train_features[indices].to(device=device, dtype=torch.float32)
            batch_targets = encoded_train[indices].to(device)
            normalized = (batch_features - feature_mean) / feature_scale
            logits = classifier(normalized)
            loss = F.cross_entropy(logits, batch_targets, weight=class_weights)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            epoch_loss_sum += float(loss.detach().item()) * int(indices.numel())
        epoch_losses.append(epoch_loss_sum / train_features.shape[0])

    encoded_predictions = _predict_linear_batches(
        classifier=classifier,
        features=validation_features,
        feature_mean=feature_mean,
        feature_scale=feature_scale,
        batch_size=config["batch_size"],
        device=device,
    )
    class_value_tensor = torch.tensor(class_values, dtype=torch.int64)
    raw_predictions = class_value_tensor[encoded_predictions]
    metrics = classification_metrics(
        targets=validation_labels.to("cpu", torch.int64),
        predictions=raw_predictions,
        class_values=class_values,
        class_names=class_names,
    )
    report = LinearProbeReport(
        status="evaluated",
        feature_dim=int(train_features.shape[1]),
        class_values=list(class_values),
        class_names=list(class_names),
        train_samples=int(train_features.shape[0]),
        validation_samples=int(validation_features.shape[0]),
        train_class_counts=class_counts(train_labels, class_values, class_names),
        validation_class_counts=class_counts(validation_labels, class_values, class_names),
        first_epoch_loss=float(epoch_losses[0]),
        final_epoch_loss=float(epoch_losses[-1]),
        metrics=metrics,
    )
    state = {
        "weight": classifier.weight.detach().to("cpu"),
        "bias": classifier.bias.detach().to("cpu"),
        "feature_mean": feature_mean_cpu,
        "feature_scale": feature_scale_cpu,
        "class_values": class_value_tensor,
    }
    return report, state
