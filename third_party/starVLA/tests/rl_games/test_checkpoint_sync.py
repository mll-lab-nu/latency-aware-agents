from pathlib import Path
from types import SimpleNamespace

from omegaconf import OmegaConf

from examples.rl_games.scripts.setup_training_assets import setup_assets
from starVLA.training.rl_games.checkpoint_sync import CheckpointSyncManager


def test_local_keep_last_n_prunes_old_files(tmp_path: Path):
    cfg = OmegaConf.create(
        {
            "checkpoint": {
                "local": {"keep_last_n": 2},
                "sync": {"enabled": False, "repo_id": None, "keep_last_n": 0},
            }
        }
    )
    manager = CheckpointSyncManager(cfg=cfg)

    files = []
    for step in [1, 2, 3]:
        path = tmp_path / f"steps_{step}_pytorch_model.pt"
        path.write_text("x", encoding="utf-8")
        files.append(path)
        manager.register_local_checkpoint(step=step, model_path=str(path))

    assert not files[0].exists()
    assert files[1].exists()
    assert files[2].exists()


def test_setup_assets_resumes_from_latest_checkpoint_when_best_model_exists(tmp_path: Path):
    base_model_dir = tmp_path / "base_model"
    base_model_dir.mkdir()
    (base_model_dir / "config.json").write_text("{}", encoding="utf-8")
    checkpoint_dir = tmp_path / "checkpoints"
    (checkpoint_dir / "steps_20_state").mkdir(parents=True)
    (checkpoint_dir / "best_state").mkdir()
    (checkpoint_dir / "best_model_metadata.json").write_text('{"best_step": 100}', encoding="utf-8")

    args = SimpleNamespace(
        model="unsupported",
        env="unsupported",
        mode="single",
        initialization_mode="scratch",
        action_carrier="",
        dataset_local_dir=str(tmp_path / "datasets"),
        base_model_dir=str(base_model_dir),
        base_model_repo_id=None,
        checkpoint_local_dir=str(checkpoint_dir),
        checkpoint_load="local",
        checkpoint_hf_repo_id="",
        checkpoint_save_best_model="true",
        checkpoint_sync_enabled="false",
        checkpoint_sync_repo_id="",
        hf_repo_id="",
        initialization_local_dir="",
        initialization_hf_repo_id="",
        initialization_checkpoint_filename="",
    )

    result = setup_assets(args)

    assert result["resume_source"] == "local"
    assert result["resume_checkpoint"] == str(checkpoint_dir / "steps_20_state")
    assert result["resume_step"] == 20


def test_local_keep_last_n_prunes_existing_checkpoints_on_new_save(tmp_path: Path):
    cfg = OmegaConf.create(
        {
            "checkpoint": {
                "local": {"keep_last_n": 1},
                "sync": {"enabled": False, "repo_id": None, "keep_last_n": 0},
            }
        }
    )
    manager = CheckpointSyncManager(cfg=cfg)
    old_state = tmp_path / "steps_1_state"
    old_state.mkdir()
    old_model = tmp_path / "steps_1_pytorch_model.pt"
    old_model.write_text("old", encoding="utf-8")
    new_state = tmp_path / "steps_2_state"
    new_state.mkdir()

    manager.register_local_checkpoint(step=2, state_path=str(new_state), model_path=None)

    assert not old_state.exists()
    assert not old_model.exists()
    assert new_state.exists()
