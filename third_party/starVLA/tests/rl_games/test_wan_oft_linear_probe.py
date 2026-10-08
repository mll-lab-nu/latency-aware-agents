from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from starVLA.training.rl_games.wan_oft_probe_data import (
    ProbeExample,
    ProbeLabels,
    build_flap_timing_labels,
    latency_neutral_prompt,
    repeated_last_frame_examples,
    shuffled_frame_examples,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def _probe_example() -> ProbeExample:
    frames = tuple(
        Image.fromarray(np.full((2, 2, 3), fill_value=value, dtype=np.uint8))
        for value in (10, 20, 30, 40, 50)
    )
    return ProbeExample(
        frames=frames,
        prompt=(
            "You are playing Flappy Bird. Current action latency is 2 raw frames (33.33 ms). "
            "Choose the best next action."
        ),
        state=np.zeros((1, 7), dtype=np.float32),
        episode_index=3,
        frame_index=11,
        decision_step=17,
        labels=ProbeLabels(
            current_action=0,
            time_to_next_flap=2,
            time_since_last_flap=4,
            latency_id=2,
        ),
    )


def _frame_values(example: ProbeExample) -> list[int]:
    return [int(np.asarray(frame)[0, 0, 0]) for frame in example["frames"]]


def _png_bytes(value: int) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(np.full((2, 2, 3), fill_value=value, dtype=np.uint8)).save(buffer, format="PNG")
    return buffer.getvalue()


def test_timing_labels_use_true_decision_step_gaps() -> None:
    next_flap, since_flap = build_flap_timing_labels(
        decision_steps=[0, 1, 4, 5, 9],
        action_ids=[0, 1, 0, 0, 1],
        flap_action_id=1,
        maximum_exact_distance=4,
    )

    assert next_flap == [1, 0, 4, 4, 0]
    assert since_flap == [4, 0, 3, 4, 0]


def test_timing_labels_reject_reindexed_or_unsorted_steps() -> None:
    with pytest.raises(ValueError, match="strictly increasing"):
        build_flap_timing_labels(
            decision_steps=[0, 3, 2],
            action_ids=[0, 1, 0],
            flap_action_id=1,
            maximum_exact_distance=4,
        )


def test_frame_controls_are_deterministic_and_preserve_sample_metadata() -> None:
    example = _probe_example()

    first_shuffle = shuffled_frame_examples([example], seed=6201)[0]
    second_shuffle = shuffled_frame_examples([example], seed=6201)[0]
    repeated = repeated_last_frame_examples([example], seed=6201)[0]

    assert _frame_values(first_shuffle) == _frame_values(second_shuffle)
    assert sorted(_frame_values(first_shuffle)) == [10, 20, 30, 40, 50]
    assert _frame_values(first_shuffle) != [10, 20, 30, 40, 50]
    assert _frame_values(repeated) == [50, 50, 50, 50, 50]
    assert first_shuffle["labels"] == example["labels"]
    assert repeated["decision_step"] == example["decision_step"]


def test_latency_neutral_prompt_removes_only_explicit_latency_suffix() -> None:
    prompt = _probe_example()["prompt"]

    assert latency_neutral_prompt(prompt) == "You are playing Flappy Bird."
    assert latency_neutral_prompt("Play Flappy Bird.") == "Play Flappy Bird."


def test_probe_episode_loader_preserves_context_order_and_true_timing(tmp_path: Path) -> None:
    pyarrow = pytest.importorskip("pyarrow")
    parquet = pytest.importorskip("pyarrow.parquet")
    from starVLA.training.rl_games.wan_oft_probe_data import iter_episode_examples

    image_type = pyarrow.struct([("bytes", pyarrow.binary()), ("path", pyarrow.string())])
    rows = 2
    table = pyarrow.table(
        {
            "observation.context_images": pyarrow.array(
                [
                    [{"bytes": _png_bytes(value), "path": None} for value in (10, 20, 30, 40)],
                    [{"bytes": _png_bytes(value), "path": None} for value in (20, 30, 40, 50)],
                ],
                type=pyarrow.list_(image_type),
            ),
            "observation.image": pyarrow.array(
                [{"bytes": _png_bytes(50), "path": None}, {"bytes": _png_bytes(60), "path": None}],
                type=image_type,
            ),
            "observation.state": pyarrow.array([[0.0] * 7] * rows, type=pyarrow.list_(pyarrow.float32(), 7)),
            "episode_index": pyarrow.array([0, 0], type=pyarrow.int64()),
            "frame_index": pyarrow.array([0, 1], type=pyarrow.int64()),
            "decision_step": pyarrow.array([3, 8], type=pyarrow.int64()),
            "task_index": pyarrow.array([0, 0], type=pyarrow.int64()),
            "action_id": pyarrow.array([0, 1], type=pyarrow.int64()),
            "latency": pyarrow.array([0, 0], type=pyarrow.int64()),
        }
    )
    episode_path = tmp_path / "data" / "chunk-000" / "episode_000000.parquet"
    episode_path.parent.mkdir(parents=True)
    parquet.write_table(table, episode_path)

    examples = list(
        iter_episode_examples(
            episode_path=episode_path,
            task_prompts={0: "Play Flappy Bird."},
            image_sequence_length=5,
            maximum_exact_distance=4,
            flap_action_id=1,
        )
    )

    assert _frame_values(examples[0]) == [10, 20, 30, 40, 50]
    assert _frame_values(examples[1]) == [20, 30, 40, 50, 60]
    assert examples[0]["labels"]["time_to_next_flap"] == 4
    assert examples[1]["labels"]["time_to_next_flap"] == 0
    assert examples[0]["decision_step"] == 3


def test_temporal_group_pooling_and_delta_feature() -> None:
    torch = pytest.importorskip("torch")
    from starVLA.training.rl_games.wan_oft_probe_features import (
        pool_dit_temporal_groups,
        pool_vae_temporal_groups,
        two_group_delta_feature,
    )

    latents = torch.empty((1, 2, 2, 2, 2), dtype=torch.float32)
    latents[:, :, 0] = 1.0
    latents[:, :, 1] = 3.0
    vae_groups = pool_vae_temporal_groups(latents, temporal_patch_size=1)
    vae_feature = two_group_delta_feature(vae_groups, feature_name="vae")

    hidden = torch.empty((1, 8, 3), dtype=torch.float32)
    hidden[:, :4] = 2.0
    hidden[:, 4:] = 6.0
    dit_groups = pool_dit_temporal_groups(
        hidden_states=hidden,
        latent_shape=latents.shape,
        patch_size=(1, 1, 1),
    )
    dit_feature = two_group_delta_feature(dit_groups, feature_name="dit")

    assert vae_groups.shape == (1, 2, 2)
    assert torch.equal(vae_feature, torch.tensor([[1.0, 1.0, 3.0, 3.0, 2.0, 2.0]]))
    assert dit_groups.shape == (1, 2, 3)
    assert torch.equal(dit_feature, torch.tensor([[2.0, 2.0, 2.0, 6.0, 6.0, 6.0, 4.0, 4.0, 4.0]]))


def test_temporal_delta_rejects_non_two_group_features() -> None:
    torch = pytest.importorskip("torch")
    from starVLA.training.rl_games.wan_oft_probe_features import two_group_delta_feature

    with pytest.raises(ValueError, match="exactly two"):
        two_group_delta_feature(torch.zeros((2, 1, 8)), feature_name="dit")


def test_classification_metrics_reports_macro_f1_and_balanced_accuracy() -> None:
    torch = pytest.importorskip("torch")
    from starVLA.training.rl_games.linear_probe import classification_metrics

    metrics = classification_metrics(
        targets=torch.tensor([0, 0, 1, 1]),
        predictions=torch.tensor([0, 1, 1, 1]),
        class_values=[0, 1],
        class_names=["NOOP", "FLAP"],
    )

    assert metrics["confusion_matrix"] == [[1, 1], [0, 2]]
    assert metrics["balanced_accuracy"] == pytest.approx(0.75)
    assert metrics["macro_f1"] == pytest.approx((2.0 / 3.0 + 0.8) / 2.0)


def test_linear_probe_learns_separable_features_deterministically() -> None:
    torch = pytest.importorskip("torch")
    from starVLA.training.rl_games.linear_probe import LinearProbeConfig, train_linear_probe

    negative = torch.tensor([[-2.0, -1.0], [-1.5, -0.5], [-1.0, -2.0], [-2.5, -1.5]])
    positive = -negative
    train_features = torch.cat((negative, positive), dim=0)
    train_labels = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
    validation_features = torch.tensor([[-3.0, -1.0], [-1.0, -3.0], [3.0, 1.0], [1.0, 3.0]])
    validation_labels = torch.tensor([0, 0, 1, 1])
    config = LinearProbeConfig(
        epochs=30,
        batch_size=4,
        learning_rate=0.05,
        weight_decay=0.0,
        seed=7,
        device="cpu",
    )

    first_report, first_state = train_linear_probe(
        train_features=train_features,
        train_labels=train_labels,
        validation_features=validation_features,
        validation_labels=validation_labels,
        class_values=[0, 1],
        class_names=["NOOP", "FLAP"],
        config=config,
    )
    second_report, second_state = train_linear_probe(
        train_features=train_features,
        train_labels=train_labels,
        validation_features=validation_features,
        validation_labels=validation_labels,
        class_values=[0, 1],
        class_names=["NOOP", "FLAP"],
        config=config,
    )

    assert first_report["metrics"]["macro_f1"] == 1.0
    assert second_report["metrics"] == first_report["metrics"]
    assert torch.equal(first_state["weight"], second_state["weight"])
    assert torch.equal(first_state["bias"], second_state["bias"])


def test_flappy_converter_preserves_decision_step_for_probing() -> None:
    converter_source = (
        REPO_ROOT / "examples/rl_games/bash_scripts/gr00t/data_conversion/convert_flappy_to_starvla_lerobot.py"
    ).read_text(encoding="utf-8")

    assert '"decision_step": int(row[flappy_columns.frame])' in converter_source
    assert '"decision_step": pa.array([row["decision_step"] for row in rows]' in converter_source
    assert '"decision_step": {"dtype": "int64", "shape": [1]}' in converter_source
