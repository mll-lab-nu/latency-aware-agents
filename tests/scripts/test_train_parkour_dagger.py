from pathlib import Path

from scripts.official_gr00t.train_parkour_dagger import (
    COLLECTOR_RANKS,
    ENVS_PER_COLLECTOR_RANK,
    DAGGER_ROUNDS,
    GRADIENT_ACCUMULATION_STEPS,
    PARKOUR_DAGGER_ROWS_PER_ROUND,
    checkpoint_dir,
    collector_command,
    training_command,
)


def test_commands_preserve_paper_scale_and_round_resume_contract():
    output = Path("output")
    teacher = Path("teacher.pt")
    bootstrap = Path("bootstrap")

    first_collect = collector_command(
        Path("python"),
        mode="dagger-collect",
        teacher_checkpoint=teacher,
        policy_config=Path("policy.yaml"),
        output_dir=Path("raw"),
        inference_device="cuda:1",
        num_envs=ENVS_PER_COLLECTOR_RANK,
        seed=1,
        round_index=0,
        student_actor_checkpoint=None,
        dagger_row_budget=PARKOUR_DAGGER_ROWS_PER_ROUND // COLLECTOR_RANKS,
    )
    later_collect = collector_command(
        Path("python"),
        mode="dagger-collect",
        teacher_checkpoint=teacher,
        policy_config=Path("policy.yaml"),
        output_dir=Path("raw"),
        inference_device="cuda:1",
        num_envs=ENVS_PER_COLLECTOR_RANK,
        seed=2,
        round_index=1,
        student_actor_checkpoint=Path("actor.pt"),
    )
    first_train = training_command(
        bootstrap_checkpoint=bootstrap,
        teacher_checkpoint=teacher,
        dataset_dir=Path("dataset"),
        training_output=output,
        round_index=0,
    )
    later_train = training_command(
        bootstrap_checkpoint=bootstrap,
        teacher_checkpoint=teacher,
        dataset_dir=Path("dataset"),
        training_output=output,
        round_index=1,
    )

    assert first_collect[first_collect.index("--num_envs") + 1] == str(
        ENVS_PER_COLLECTOR_RANK
    )
    assert first_collect[first_collect.index("--seed") + 1] == "1"
    assert later_collect[later_collect.index("--seed") + 1] == "2"
    assert COLLECTOR_RANKS * ENVS_PER_COLLECTOR_RANK == 192
    assert DAGGER_ROUNDS * PARKOUR_DAGGER_ROWS_PER_ROUND == 23_040_000
    assert DAGGER_ROUNDS * PARKOUR_DAGGER_ROWS_PER_ROUND * 5 == 115_200_000
    assert first_collect[first_collect.index("--dagger_row_budget") + 1] == str(
        PARKOUR_DAGGER_ROWS_PER_ROUND // COLLECTOR_RANKS
    )
    assert "--student_actor_checkpoint" not in first_collect
    assert later_collect[-2:] == ["--student_actor_checkpoint", "actor.pt"]
    assert "--resume-checkpoint-path" not in first_train
    assert "--parkour-enabled" not in first_train
    assert "--state-history-length" not in first_train
    assert later_train[-2:] == [
        "--resume-checkpoint-path",
        "output/checkpoint-500",
    ]
    assert first_train[first_train.index("--gradient-accumulation-steps") + 1] == str(
        GRADIENT_ACCUMULATION_STEPS
    )
    assert first_train[first_train.index("--episode-sampling-rate") + 1] == "1.0"
    assert first_train[first_train.index("--dataloader-num-workers") + 1] == "0"
    assert "--experiment-name" not in first_train
    assert checkpoint_dir(output, 10) == output / "checkpoint-5000"
