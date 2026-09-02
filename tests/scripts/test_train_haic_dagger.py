from pathlib import Path

import scripts.official_gr00t.train_haic_dagger as train_haic_dagger
from scripts.official_gr00t.train_haic_dagger import (
    BOOTSTRAP_GRADIENT_ACCUMULATION_STEPS,
    BOOTSTRAP_TRAIN_EPISODES,
    BOOTSTRAP_UPDATES,
    BOOTSTRAP_VAL_EPISODES,
    COLLECTOR_RANKS,
    DAGGER_ROUNDS,
    ENVS_PER_COLLECTOR_RANK,
    GLOBAL_BATCH_SIZE,
    GRADIENT_ACCUMULATION_STEPS,
    HAIC_DAGGER_ROWS_PER_COLLECTOR,
    HAIC_DAGGER_ROWS_PER_ROUND,
    UPDATES_PER_ROUND,
    bootstrap_training_command,
    checkpoint_dir,
    collector_command,
    prepare_command,
    training_command,
    write_policy_config,
)


def test_main_resolves_bootstrap_paths_from_output_root(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        "sys.argv", ["train_haic_dagger.py", "--output-root", "/tmp/haic-custom"]
    )
    monkeypatch.setattr(
        train_haic_dagger,
        "run_pipeline",
        lambda args: captured.update(vars(args)),
    )

    train_haic_dagger.main()

    root = Path("/tmp/haic-custom")
    assert captured["bootstrap_dataset"] == root / "bootstrap/lerobot"
    assert captured["bootstrap_checkpoint"] == (
        root / "bootstrap/training/checkpoint-5000"
    )


def test_haic_pipeline_constants_are_single_gpu_and_paper_scale():
    assert BOOTSTRAP_TRAIN_EPISODES == 250
    assert BOOTSTRAP_VAL_EPISODES == 25
    assert BOOTSTRAP_UPDATES == 5_000
    assert GLOBAL_BATCH_SIZE == 64
    assert BOOTSTRAP_GRADIENT_ACCUMULATION_STEPS == 2
    assert DAGGER_ROUNDS == 10
    assert UPDATES_PER_ROUND == 500
    assert GRADIENT_ACCUMULATION_STEPS == 72
    assert COLLECTOR_RANKS * HAIC_DAGGER_ROWS_PER_COLLECTOR == HAIC_DAGGER_ROWS_PER_ROUND
    assert COLLECTOR_RANKS * ENVS_PER_COLLECTOR_RANK == 192


def test_haic_commands_encode_smoke_and_resume_contract():
    teacher = Path("teacher.pt")
    first = collector_command(
        Path("haic-python"),
        mode="dagger-collect",
        teacher_checkpoint=teacher,
        output_dir=Path("raw"),
        inference_device="cuda:0",
        num_envs=8,
        seed=2,
        round_index=0,
        policy_config=Path("policy.yaml"),
        row_budget=64,
    )
    assert first[first.index("--num-envs") + 1] == "8"
    assert first[first.index("--row-budget") + 1] == "64"
    assert first[first.index("--vla-cadence") + 1] == "5"
    assert first[first.index("--policy-config") + 1] == "policy.yaml"

    prep = prepare_command(
        Path("raw"),
        Path("dataset"),
        dagger=True,
        round_index=0,
        expected_rows=64,
        normalization_source=Path("bootstrap/lerobot"),
    )
    assert "--dagger" in prep
    assert prep[prep.index("--expected-rows") + 1] == "64"

    train = training_command(
        bootstrap_checkpoint=Path("bootstrap/checkpoint-1"),
        actor_checkpoint=teacher,
        dataset_dir=Path("dataset"),
        training_output=Path("output"),
        resume_checkpoint=Path("output/checkpoint-500"),
        max_steps=2,
        gradient_accumulation_steps=1,
    )
    assert train[train.index("--num-gpus") + 1] == "1"
    assert train[train.index("--max-state-dim") + 1] == "605"
    assert train[train.index("--max-action-dim") + 1] == "275"
    assert train[train.index("--gradient-accumulation-steps") + 1] == "1"
    assert train[-2:] == ["--resume-checkpoint-path", "output/checkpoint-500"]

    bootstrap = bootstrap_training_command(
        base_model=Path("base"),
        dataset_dir=Path("dataset"),
        training_output=Path("output"),
    )
    assert bootstrap[bootstrap.index("--max-steps") + 1] == str(BOOTSTRAP_UPDATES)
    assert bootstrap[
        bootstrap.index("--gradient-accumulation-steps") + 1
    ] == str(BOOTSTRAP_GRADIENT_ACCUMULATION_STEPS)
    assert "--haic-dagger" not in bootstrap
    assert checkpoint_dir(Path("output"), 5_000) == Path("output/checkpoint-5000")


def test_haic_policy_config_points_at_selected_vla(tmp_path):
    source = tmp_path / "source.yaml"
    source.write_text(
        "policy:\n  model_path: old\n  device: cuda:9\n  action_dim: 275\n"
    )
    destination = tmp_path / "runtime.yaml"
    write_policy_config(
        source,
        destination,
        model_path=Path("checkpoint-2"),
        inference_device="cuda:0",
    )
    contents = destination.read_text()
    assert "model_path: checkpoint-2" in contents
    assert "device: cuda:0" in contents
