from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from latency_bench.core.config import load_config
from scripts.sweeps.generate_sweep_configs import (
    _apply_latency_setting,
    _build_latency_bench_eval_config,
    main as generate_sweep_configs,
)
from scripts.tasks.flappy.run_profile_controls import write_control_configs
from latency_bench.eval.config import _episode_seed

from helpers import REPO_ROOT


OFFICIAL_SEEDS = tuple(range(10, 15))


def _read_manifest(path: Path) -> list[dict[str, str]]:
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    header = lines[0].split("\t")
    return [dict(zip(header, line.split("\t"))) for line in lines[1:]]


def test_iid_latency_setting_preserves_profile_training_config():
    config = load_config("configs/base/deadly_corridor.yaml")
    _apply_latency_setting(
        config,
        {
            "kind": "iid",
            "profile_path": "results/profiling/profile.json",
            "profile_worker_slot": 1,
        },
        seed=7,
    )

    assert config["latency"] == {
        "method": "iid",
        "profile_path": "results/profiling/profile.json",
        "profile_worker_slot": 1,
        "seed": 7,
        "add_latency_info": False,
    }


def test_flappy_capacity_one_profile_spec_emits_three_temporal_best_eval_jobs(tmp_path):
    generated_root = tmp_path / "configs"
    output_root = tmp_path / "outputs"
    train_dir = tmp_path / "checkpoints"

    generate_sweep_configs(
        [
            "--spec",
            "configs/experiments/flappy/small_policy_qwen_profiles_10fps_25m_capacity1_seed42.yaml",
            "--generated-root",
            str(generated_root),
            "--output-root",
            str(output_root),
            "--train-dir",
            str(train_dir),
            "--eval-checkpoint-kind",
            "best",
        ]
    )

    rows = _read_manifest(generated_root / "manifest.tsv")
    assert len(rows) == 3
    assert {row["latency_label"] for row in rows} == {
        "qwenoft_rtx3090_temporal_25m_capacity1_20260923",
        "qwengr00t_rtx3090_temporal_25m_capacity1_20260923",
        "qwenpi_v3_rtx3090_temporal_25m_capacity1_20260923",
    }
    assert {row["seed"] for row in rows} == {"42"}
    assert {row["frame_stack"] for row in rows} == {"4"}
    assert {row["obs_fps"] for row in rows} == {"10"}
    assert {row["train_upload_repo_id"] for row in rows} == {""}
    assert {row["eval_upload_repo_id"] for row in rows} == {""}

    for row in rows:
        train_cfg = yaml.safe_load(Path(row["train_config"]).read_text(encoding="utf-8"))
        eval_cfg = yaml.safe_load(Path(row["eval_config"]).read_text(encoding="utf-8"))
        assert train_cfg["env"]["env_fps"] == 10
        assert train_cfg["env"]["obs_fps"] == 10
        assert train_cfg["training"]["train_for_env_steps"] == 25_000_000
        assert train_cfg["evaluation"]["eval_interval_steps"] == 5_000_000
        assert train_cfg["logging"]["wandb_project"] == "starvla_tasks"
        assert train_cfg["executor"]["simulated_worker_capacity"] == 1
        assert train_cfg["latency"]["method"] == "temporal"
        assert train_cfg["latency"]["profile_worker_slot"] == 0
        assert [suite["latency"] for suite in eval_cfg['evaluation']['suites']] == [{"method": 'temporal', 'profile_path': train_cfg['latency']['profile_path'], 'seed': 42}]
        assert eval_cfg["policy"]["checkpoint_kind"] == "best"
        assert eval_cfg["backend"]["extra_args"] == ["--load_checkpoint_kind", "best"]
        assert eval_cfg["executor"]["simulated_worker_capacity"] == 1
        assert eval_cfg["evaluation"]["episodes"] == 100
        assert eval_cfg["evaluation"]["max_env_steps"] == 18_000
        assert "eval_interval_steps" not in eval_cfg["evaluation"]

        control_root = tmp_path / "controls" / row["latency_label"]
        control_paths = write_control_configs(Path(row["eval_config"]), control_root)
        for name, path in control_paths.items():
            control = yaml.safe_load(path.read_text(encoding="utf-8"))
            assert control["policy"] == (
                {"type": "constant", "action": "noop"} if name == "noop"
                else {"type": "random", "actions": ["noop", "flap"]}
            )
            assert control["logging"]["output_dir"] == str(control_root / name)
            assert not any(key.startswith("wandb_") for key in control["logging"])
            assert control["evaluation"]["suites"] == eval_cfg["evaluation"]["suites"]
            assert control["evaluation"]["episodes"] == 100
            assert control["evaluation"]["max_env_steps"] == 18_000
            assert control["executor"]["parallel_envs"] == 100
            assert control["env"]["env_fps"] == control["env"]["obs_fps"] == 10
            assert control["executor"]["simulated_worker_capacity"] == 1
            assert [_episode_seed(control, episode_id) for episode_id in (0, 99)] == [
                _episode_seed(eval_cfg, episode_id) for episode_id in (0, 99)
            ]


def test_deadly_native35_history_spec_uses_game_clock_and_best_checkpoint(tmp_path):
    generated_root = tmp_path / "configs"
    generate_sweep_configs(
        [
            "--spec",
            "configs/experiments/deadly_corridor/decision_history_native35_temporal_25m_seed42.yaml",
            "--generated-root",
            str(generated_root),
            "--output-root",
            str(tmp_path / "outputs"),
            "--train-dir",
            str(tmp_path / "checkpoints"),
            "--eval-checkpoint-kind",
            "best",
        ]
    )

    rows = _read_manifest(generated_root / "manifest.tsv")
    assert len(rows) == 3
    assert {row["latency_label"] for row in rows} == {
        "qwenoft_rtx3090_temporal_hist8_capacity1_native35",
        "gr00t_rtx3090_temporal_hist8_capacity1_native35",
        "pi05_rtx3090_temporal_hist8_capacity1_native35",
    }
    for row in rows:
        train_cfg = yaml.safe_load(Path(row["train_config"]).read_text(encoding="utf-8"))
        eval_cfg = yaml.safe_load(Path(row["eval_config"]).read_text(encoding="utf-8"))
        assert row["observation_stride"] == "4"
        assert train_cfg["env"]["env_fps"] == eval_cfg["env"]["env_fps"] == 35
        assert train_cfg["env"]["obs_fps"] == eval_cfg["env"]["obs_fps"] == 8.75
        assert train_cfg["env"]["action_history_decisions"] == 8
        assert "decision_v1" in train_cfg["logging"]["wandb_tags"]
        assert train_cfg["executor"]["simulated_worker_capacity"] == 1
        assert train_cfg["latency"]["method"] == "temporal"
        assert train_cfg["training"]["train_for_env_steps"] == 25_000_000
        assert eval_cfg["policy"]["checkpoint_kind"] == "best"
        assert eval_cfg["backend"]["extra_args"] == ["--load_checkpoint_kind", "best"]
        assert eval_cfg["evaluation"]["episodes"] == 100
        assert row["train_upload_repo_id"] == row["eval_upload_repo_id"] == ""


@pytest.mark.parametrize(
    ("env_name", "latency_label", "latency_frames", "obs_fps", "obs_fps_label"),
    [
        ("flappy", "fixed_l3", 3, 13, "13"),
        ("demon_attack", "fixed_l6", 6, 15, "15"),
        ("deadly_corridor", "fixed_l6", 6, 8.75, "8p75"),
    ],
)
def test_memory_rollout_teacher_spec_emits_single_uploadable_job(
    tmp_path,
    env_name,
    latency_label,
    latency_frames,
    obs_fps,
    obs_fps_label,
):
    generated_root = tmp_path / "configs"
    output_root = tmp_path / "outputs"
    train_dir = tmp_path / "checkpoints"

    generate_sweep_configs(
        [
            "--spec",
            f"configs/experiments/memory/rollout_teacher/{env_name}.yaml",
            "--generated-root",
            str(generated_root),
            "--output-root",
            str(output_root),
            "--train-dir",
            str(train_dir),
        ]
    )

    rows = _read_manifest(generated_root / "manifest.tsv")
    assert len(rows) == 1
    row = rows[0]
    assert row["env"] == env_name
    assert row["experiment_family"] == "memory_rollout_teacher"
    assert row["sweep_name"] == "teacher"
    assert row["seed"] == "0"
    assert row["frame_stack"] == "4"
    assert row["obs_fps"] == obs_fps_label
    assert row["latency_label"] == latency_label
    assert row["latency_kind"] == "fixed"
    assert row["latency_fixed_frames"] == str(latency_frames)
    assert row["train_upload_repo_id"] == "MLL-Lab/LAGEN-models"
    assert row["train_upload_repo_type"] == "model"
    assert row["eval_upload_repo_id"] == ""
    assert row["train_artifact_repo_path"].startswith(
        f"latency-aware/{env_name.replace('_', '-')}/small-policy/memory_rollout_teacher:{env_name}:"
    )

    train_cfg = yaml.safe_load(Path(row["train_config"]).read_text(encoding="utf-8"))
    assert train_cfg["experiment"]["seed"] == 0
    assert train_cfg["env"]["frame_stack"] == 4
    assert train_cfg["env"]["obs_fps"] == pytest.approx(obs_fps)
    if env_name == "demon_attack":
        assert train_cfg["env"]["reset_mode"] == "fire"
    assert train_cfg["latency"]["method"] == "fixed"
    assert train_cfg["latency"]["fixed_latency_ms"] == pytest.approx(
        latency_frames * 1000 / train_cfg["env"]["env_fps"]
    )
    assert [suite["latency"] for suite in train_cfg['evaluation']['suites']] == [{"method": "fixed", "fixed_latency_ms": latency_frames * 1000.0 / train_cfg["env"]["env_fps"]}]


def test_latency_bench_eval_config_disables_full_trace_records(tmp_path):
    base_config = {
        "experiment": {"name": "test"},
        "executor": {"mode": "simulated"},
        "backend": {
            "train_dir": str(tmp_path / "checkpoints"),
            "extra_args": [],
        },
        "env": {
            "name": "flappy",
            "env_fps": 30,
            "obs_fps": 30,
            "frame_stack": 4,
        },
        "evaluation": {
            "eval_interval_steps": 1000,
            "episodes": 3,
            "suites": [{'name': 'fixed_1', 'latency': {'method': 'fixed', 'fixed_latency_ms': 33.333333333333336}}],
        },
        "logging": {
            "output_dir": str(tmp_path / "train_output"),
            "save_step_records": True,
            "save_action_records": True,
            "save_latency_records": True,
        },
        "policy": {},
    }
    full_train_config = {
        "experiment": {"name": "test"},
        "backend": {
            "train_dir": str(tmp_path / "checkpoints"),
            "extra_args": [],
        },
        "env": {
            "name": "flappy",
            "env_fps": 30,
            "obs_fps": 30,
            "frame_stack": 4,
        },
        "evaluation": {
            "eval_interval_steps": 1000,
            "episodes": 3,
            "suites": [{'name': 'fixed_1', 'latency': {'method': 'fixed', 'fixed_latency_ms': 33.333333333333336}}],
        },
        "logging": {
            "output_dir": str(tmp_path / "train_output"),
            "save_step_records": True,
            "save_action_records": True,
            "save_latency_records": True,
        },
    }

    config = _build_latency_bench_eval_config(
        base_config,
        full_train_config,
        sweep_name="frame_stack",
        latency_label="fixed_l1",
        frame_stack=4,
        obs_fps=30,
        observation_stride=1,
        seed=10,
        output_root=tmp_path / "outputs",
        latency_bench_policy_device="auto",
    )

    assert config["logging"]["save_step_records"] is False
    assert config["logging"]["save_action_records"] is False
    assert config["logging"]["save_latency_records"] is False


def test_gymnasium_sweep_preserves_task_contract_and_uses_task_identity(tmp_path):
    generated_root = tmp_path / "configs"
    output_root = tmp_path / "outputs"
    train_dir = tmp_path / "checkpoints"
    spec_path = tmp_path / "gymnasium_sweep.yaml"
    spec_path.write_text(
        yaml.safe_dump(
            {
                "__base__": str(
                    REPO_ROOT
                    / "configs/examples/gymnasium/inverted_pendulum/small_model_train.yaml"
                ),
                "experiment_family": "generic_gym_sweep",
                "generated_root": str(generated_root),
                "output_root": str(output_root),
                "train_dir": str(train_dir),
                "seeds": [3],
                "latency_settings": {
                    "fixed_l1": {"kind": "fixed", "frames": 1},
                },
                "sweeps": {
                    "observation_rate": {
                        "frame_stack": [1],
                        "obs_fps": [25],
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    generate_sweep_configs(["--spec", str(spec_path)])

    rows = _read_manifest(generated_root / "manifest.tsv")
    assert len(rows) == 1
    assert rows[0]["env"] == "inverted_pendulum"
    assert rows[0]["run_name"] == (
        "inverted_pendulum_observation_rate_fixed_l1_obs25_seed3"
    )

    train_cfg = yaml.safe_load(Path(rows[0]["train_config"]).read_text(encoding="utf-8"))
    eval_cfg = load_config(rows[0]["eval_config"])
    expected_env = load_config(
        REPO_ROOT
        / "configs/examples/gymnasium/inverted_pendulum/small_model_train.yaml"
    )["env"]

    assert train_cfg["env"] == expected_env
    assert eval_cfg["env"] == expected_env
    assert eval_cfg["policy"]["type"] == "sample_factory"


def test_memory_small_model_spec_emits_official_seed_matrix(tmp_path):
    generated_root = tmp_path / "configs"
    output_root = tmp_path / "outputs"
    train_dir = tmp_path / "checkpoints"

    generate_sweep_configs(
        [
            "--spec",
            "configs/experiments/memory/small_model/flappy_extended.yaml",
            "--generated-root",
            str(generated_root),
            "--output-root",
            str(output_root),
            "--train-dir",
            str(train_dir),
        ]
    )

    rows = _read_manifest(generated_root / "manifest.tsv")
    assert len(rows) == 80
    assert len({row["train_config"] for row in rows}) == 80
    assert {row["seed"] for row in rows} == {str(seed) for seed in OFFICIAL_SEEDS}
    assert {row["schema_version"] for row in rows} == {"v2"}
    assert {row["experiment_family"] for row in rows} == {"memory_small_model"}
    assert {row["sweep_name"] for row in rows} == {"frame_stack"}
    assert {row["env"] for row in rows} == {"flappy"}
    assert {row["frame_stack"] for row in rows} == {"1", "2", "3", "4", "5", "6", "7", "8"}
    assert {row["observation_stride"] for row in rows} == {"1"}
    assert {row["train_upload_repo_id"] for row in rows} == {
        "MLL-Lab/LAGEN-models"
    }
    assert {row["eval_upload_repo_id"] for row in rows} == {
        "MLL-Lab/LAGEN-datasets"
    }

    train_cfg = yaml.safe_load(
        (
            generated_root
            / "frame_stack"
            / "fixed_l2"
            / "fs8"
            / "seed_10"
            / "train.yaml"
        ).read_text(encoding="utf-8")
    )
    assert train_cfg["experiment"]["name"] == "flappy_frame_stack_fixed_l2_fs8_seed10"
    assert train_cfg["experiment"]["seed"] == 10
    assert train_cfg["env"]["frame_stack"] == 8
    assert train_cfg["env"]["obs_fps"] == pytest.approx(30)
    assert train_cfg["env"]["obs_resize"] == [84, 84]
    assert train_cfg["env"]["use_gpu_render"] is True
    assert train_cfg["env"]["simulator"] == "gpu"
    assert train_cfg["env"]["gpu_render_device"] == "auto"
    assert train_cfg["env"]["gpu_render_profile"] is False
    assert train_cfg["env"]["gpu_render_profile_interval"] == 200
    assert train_cfg["scheduler"] == {
        "hold_policy": "one_frame_then_noop",
        "ordering_policy": "latest_ready",
    }
    assert train_cfg["training"]["worker_num_splits"] == 1
    assert train_cfg["evaluation"]["episodes"] == 100
    assert train_cfg["latency"]["method"] == "fixed"
    assert train_cfg["logging"]["wandb_group"] == "flappy-fs8-fixed_l2"
    assert train_cfg["logging"]["wandb_name"] == "flappy-fs8-fixed_l2-s10"
    assert train_cfg["logging"]["wandb_tags"] == [
        "memory_small_model",
        "flappy",
        "frame_stack",
        "fixed",
        "fixed_l2",
        "fs8",
        "seed10",
    ]
    assert train_cfg["logging"]["output_dir"] == str(
        output_root / "train" / "frame_stack" / "fixed_l2" / "fs8" / "seed_10"
    )
    assert "eval_interval_steps" not in train_cfg["evaluation"]

    eval_cfg = load_config(
        generated_root / "frame_stack" / "fixed_l2" / "fs4" / "seed_10" / "latency_bench_eval.yaml"
    )
    assert eval_cfg["backend"]["run_mode"] == "eval"
    assert eval_cfg["backend"]["extra_args"] == ["--load_checkpoint_kind", "latest"]
    assert eval_cfg["policy"]["type"] == "sample_factory"
    assert eval_cfg["policy"]["checkpoint_kind"] == "latest"
    assert eval_cfg["env"]["obs_resize"] == [84, 84]
    assert eval_cfg["env"]["use_gpu_render"] is True
    assert eval_cfg["env"]["gpu_render_device"] == "auto"
    assert eval_cfg["scheduler"] == train_cfg["scheduler"]
    assert eval_cfg["training"]["worker_num_splits"] == 1
    assert eval_cfg["evaluation"]["episodes"] == 100
    assert "eval_interval_steps" not in eval_cfg["evaluation"]
    assert [suite["latency"] for suite in eval_cfg['evaluation']['suites']] == [{"method": "fixed", "fixed_latency_ms": 2 * 1000.0 / eval_cfg["env"]["env_fps"]}]
    assert eval_cfg["logging"]["save_step_records"] is False
    assert eval_cfg["logging"]["save_action_records"] is False
    assert eval_cfg["logging"]["save_latency_records"] is False

    uniform_eval_cfg = load_config(
        generated_root / "frame_stack" / "uniform_u1_3" / "fs4" / "seed_10" / "latency_bench_eval.yaml"
    )
    assert [suite["latency"] for suite in uniform_eval_cfg['evaluation']['suites']] == [{"method": 'uniform', "seed": uniform_eval_cfg["experiment"]["seed"], 'min_latency_ms': 1 * 1000.0 / uniform_eval_cfg["env"]["env_fps"], 'max_latency_ms': 3 * 1000.0 / uniform_eval_cfg["env"]["env_fps"]}]

    assert [suite["latency"] for suite in train_cfg['evaluation']['suites']] == [{"method": "fixed", "fixed_latency_ms": 2 * 1000.0 / train_cfg["env"]["env_fps"]}]
    assert "train_upload_repo_id" not in train_cfg
    assert "train_upload_repo_type" not in train_cfg
    assert "train_upload_path_prefix" not in train_cfg
    assert "eval_upload_repo_id" not in train_cfg
    assert "eval_upload_repo_type" not in train_cfg
    assert "eval_upload_path_prefix" not in train_cfg


def test_paper_sweep_specs_use_base_inheritance():
    spec_paths = [
        REPO_ROOT / "configs/experiments/observation_stride/flappy_capped.yaml",
        REPO_ROOT / "configs/experiments/observation_stride/flappy_extended.yaml",
        REPO_ROOT / "configs/experiments/observation_stride/demon_attack_capped.yaml",
        REPO_ROOT / "configs/experiments/observation_stride/demon_attack_extended.yaml",
        REPO_ROOT / "configs/experiments/observation_stride/deadly_corridor.yaml",
        REPO_ROOT / "configs/experiments/memory/small_model/flappy_capped.yaml",
        REPO_ROOT / "configs/experiments/memory/small_model/flappy_extended.yaml",
        REPO_ROOT / "configs/experiments/memory/small_model/demon_attack_capped.yaml",
        REPO_ROOT / "configs/experiments/memory/small_model/demon_attack_extended.yaml",
        REPO_ROOT / "configs/experiments/memory/small_model/deadly_corridor.yaml",
        REPO_ROOT / "configs/experiments/memory/small_model/asterix_zero_latency.yaml",
        REPO_ROOT / "configs/experiments/memory/small_model/atlantis_zero_latency.yaml",
    ]

    for spec_path in spec_paths:
        spec = yaml.safe_load(spec_path.read_text(encoding="utf-8"))
        assert "__base__" in spec
        assert "base_config" not in spec
        assert spec["experiment_family"] in {"observation_stride", "memory_small_model"}


@pytest.mark.parametrize("gpu_eval", [False, True])
def test_asterix_zero_latency_memory_spec_generates_gymnasium_policy(tmp_path, gpu_eval):
    generated_root = tmp_path / "configs"
    output_root = tmp_path / "outputs"
    train_dir = tmp_path / "checkpoints"
    spec = REPO_ROOT / "configs/experiments/memory/small_model/asterix_zero_latency.yaml"
    if gpu_eval:
        override = tmp_path / "gpu_eval.yaml"
        override.write_text(yaml.safe_dump({
            '__base__': str(spec),
            'env': {
                'action_history_decisions': 8,
            },
            'evaluation': {
                'max_env_steps': 7200,
            },
            'executor': {
                'parallel_envs': 32,
                'env_backend': 'asterix_gpu_batched',
            },
        }))
        spec = override

    generate_sweep_configs(
        [
            "--spec",
            str(spec),
            "--generated-root",
            str(generated_root),
            "--output-root",
            str(output_root),
            "--train-dir",
            str(train_dir),
        ]
    )

    rows = _read_manifest(generated_root / "manifest.tsv")
    assert len(rows) == 5
    assert {row["env"] for row in rows} == {"asterix"}
    assert {row["latency_kind"] for row in rows} == {"fixed"}
    assert {row["latency_fixed_frames"] for row in rows} == {"0"}
    assert {row["frame_stack"] for row in rows} == {"4"}
    assert {row["obs_fps"] for row in rows} == {"15"}

    train_cfg = yaml.safe_load(Path(rows[0]["train_config"]).read_text(encoding="utf-8"))
    resolved_train_cfg = load_config(rows[0]["train_config"])
    eval_cfg = load_config(rows[0]["eval_config"])

    assert train_cfg["experiment"]["name"].startswith("asterix_frame_stack_fixed_l0_fs4_seed")
    assert train_cfg["env"]["name"] == "asterix"
    assert train_cfg["env"]["name"] == "asterix"
    assert resolved_train_cfg["env"]["simulator"] == "cpu"
    assert eval_cfg["env"]["simulator"] == ("gpu" if gpu_eval else "cpu")
    if gpu_eval:
        assert train_cfg["env"]["action_history_decisions"] == 8
        assert eval_cfg["env"]["action_history_decisions"] == 8
        assert eval_cfg["executor"]["parallel_envs"] == 32
        assert eval_cfg["evaluation"]["max_env_steps"] == 7200
    assert train_cfg["latency"] == {
        "method": "fixed",
        "fixed_latency_ms": 0.0,
        "add_latency_info": False,
    }
    assert eval_cfg["policy"]["type"] == "sample_factory"
    assert [suite["latency"] for suite in eval_cfg['evaluation']['suites']] == [{"method": "zero"}]


def test_atlantis_zero_latency_memory_spec_generates_gymnasium_policy(tmp_path):
    generated_root = tmp_path / "configs"
    output_root = tmp_path / "outputs"
    train_dir = tmp_path / "checkpoints"

    generate_sweep_configs(
        [
            "--spec",
            "configs/experiments/memory/small_model/atlantis_zero_latency.yaml",
            "--generated-root",
            str(generated_root),
            "--output-root",
            str(output_root),
            "--train-dir",
            str(train_dir),
        ]
    )

    rows = _read_manifest(generated_root / "manifest.tsv")
    assert len(rows) == 5
    assert {row["env"] for row in rows} == {"atlantis"}
    assert {row["latency_kind"] for row in rows} == {"fixed"}
    assert {row["latency_fixed_frames"] for row in rows} == {"0"}
    assert {row["frame_stack"] for row in rows} == {"4"}
    assert {row["obs_fps"] for row in rows} == {"15"}

    train_cfg = yaml.safe_load(Path(rows[0]["train_config"]).read_text(encoding="utf-8"))
    resolved_train_cfg = load_config(rows[0]["train_config"])
    eval_cfg = load_config(rows[0]["eval_config"])

    assert train_cfg["experiment"]["name"].startswith("atlantis_frame_stack_fixed_l0_fs4_seed")
    assert train_cfg["env"]["name"] == "atlantis"
    assert train_cfg["env"]["name"] == "atlantis"
    assert resolved_train_cfg["env"]["simulator"] == "cpu"
    assert eval_cfg["env"]["simulator"] == "cpu"
    assert train_cfg["latency"] == {
        "method": "fixed",
        "fixed_latency_ms": 0.0,
        "add_latency_info": False,
    }
    assert eval_cfg["policy"]["type"] == "sample_factory"
    assert [suite["latency"] for suite in eval_cfg['evaluation']['suites']] == [{"method": "zero"}]


def test_observation_stride_spec_names_axis_by_stride_and_writes_obs_fps(tmp_path):
    generated_root = tmp_path / "configs"
    output_root = tmp_path / "outputs"
    train_dir = tmp_path / "checkpoints"

    generate_sweep_configs(
        [
            "--spec",
            "configs/experiments/observation_stride/demon_attack_extended.yaml",
            "--generated-root",
            str(generated_root),
            "--output-root",
            str(output_root),
            "--train-dir",
            str(train_dir),
        ]
    )

    rows = _read_manifest(generated_root / "manifest.tsv")
    assert len(rows) == 90
    assert {row["seed"] for row in rows} == {str(seed) for seed in OFFICIAL_SEEDS}
    assert {row["sweep_name"] for row in rows} == {"observation_stride"}
    assert {row["observation_stride"] for row in rows} == {"2", "3", "4", "5", "6", "7", "8", "9", "10"}

    train_cfg = yaml.safe_load(
        (
            generated_root
            / "observation_stride"
            / "fixed_l4"
            / "stride5"
            / "seed_10"
            / "train.yaml"
        ).read_text(encoding="utf-8")
    )
    assert train_cfg["experiment"]["name"] == "demon_attack_observation_stride_fixed_l4_stride5_seed10"
    assert train_cfg["env"]["frame_stack"] == 4
    assert train_cfg["env"]["obs_fps"] == pytest.approx(12)
    assert train_cfg["logging"]["wandb_group"] == "demon-attack-stride5-fixed_l4"
    assert train_cfg["logging"]["wandb_name"] == "demon-attack-stride5-fixed_l4-s10"
    assert train_cfg["logging"]["wandb_tags"] == [
        "observation_stride",
        "demon-attack",
        "observation_stride",
        "fixed",
        "fixed_l4",
        "stride5",
        "seed10",
    ]
    assert train_cfg["logging"]["output_dir"] == str(
        output_root / "train" / "observation_stride" / "fixed_l4" / "stride5" / "seed_10"
    )
    assert "eval_interval_steps" not in train_cfg["evaluation"]


def test_observation_stride_spec_supports_fractional_deadly_corridor_obs_fps(tmp_path):
    generated_root = tmp_path / "configs"
    output_root = tmp_path / "outputs"
    train_dir = tmp_path / "checkpoints"

    generate_sweep_configs(
        [
            "--spec",
            "configs/experiments/observation_stride/deadly_corridor.yaml",
            "--generated-root",
            str(generated_root),
            "--output-root",
            str(output_root),
            "--train-dir",
            str(train_dir),
        ]
    )

    train_cfg = yaml.safe_load(
        (
            generated_root
            / "observation_stride"
            / "fixed_l4"
            / "stride5"
            / "seed_10"
            / "train.yaml"
        ).read_text(encoding="utf-8")
    )
    assert train_cfg["env"]["obs_fps"] == pytest.approx(7)
    bench_cfg = load_config(
        generated_root / "observation_stride" / "fixed_l4" / "stride5" / "seed_10" / "latency_bench_eval.yaml"
    )
    assert bench_cfg["policy"]["type"] == "sample_factory"
    assert bench_cfg["env"]["noop_action"] == [0, 0, 0, 0]


def test_jitter_robustness_spec_includes_mixed_train_runs(tmp_path):
    generated_root = tmp_path / "configs"
    output_root = tmp_path / "outputs"
    train_dir = tmp_path / "checkpoints"

    generate_sweep_configs(
        [
            "--spec",
            "configs/experiments/jitter_robustness/flappy.yaml",
            "--generated-root",
            str(generated_root),
            "--output-root",
            str(output_root),
            "--train-dir",
            str(train_dir),
        ]
    )

    rows = _read_manifest(generated_root / "manifest.tsv")
    assert len(rows) == 21
    assert {row["latency_label"] for row in rows} == {
        "fixed_l1",
        "fixed_l2",
        "fixed_l3",
        "normal_n2_s1",
        "uniform_u1_3",
        "mixed_normal_n2_s1",
        "mixed_uniform_u1_3",
    }
    assert {row["seed"] for row in rows} == {"911", "922", "933"}

    train_cfg = yaml.safe_load(
        (
            generated_root
            / "jitter_robustness"
            / "mixed_normal_n2_s1"
            / "obs15"
            / "seed_911"
            / "train.yaml"
        ).read_text(encoding="utf-8")
    )
    assert train_cfg["latency"]["method"] == "normal"
    assert train_cfg["latency"]["per_episode"] is True
    assert [suite["latency"] for suite in train_cfg['evaluation']['suites']] == [{"method": "fixed", "fixed_latency_ms": 1 * 1000.0 / train_cfg["env"]["env_fps"]}, {"method": "fixed", "fixed_latency_ms": 2 * 1000.0 / train_cfg["env"]["env_fps"]}, {"method": "fixed", "fixed_latency_ms": 3 * 1000.0 / train_cfg["env"]["env_fps"]}, {"method": 'normal', "seed": train_cfg["experiment"]["seed"], 'min_latency_ms': 0 * 1000.0 / train_cfg["env"]["env_fps"], 'max_latency_ms': 4 * 1000.0 / train_cfg["env"]["env_fps"], 'mean_latency_ms': 2.0 * 1000.0 / train_cfg["env"]["env_fps"], 'std_latency_ms': 1.0 * 1000.0 / train_cfg["env"]["env_fps"]}, {"method": 'uniform', "seed": train_cfg["experiment"]["seed"], 'min_latency_ms': 1 * 1000.0 / train_cfg["env"]["env_fps"], 'max_latency_ms': 3 * 1000.0 / train_cfg["env"]["env_fps"]}]

    eval_cfg = load_config(
        generated_root
        / "jitter_robustness"
        / "mixed_uniform_u1_3"
        / "obs15"
        / "seed_911"
        / "latency_bench_eval.yaml"
    )
    assert eval_cfg["latency"]["method"] == "uniform"
    assert eval_cfg["latency"]["per_episode"] is True
    assert [suite["latency"] for suite in eval_cfg['evaluation']['suites']] == [{"method": "fixed", "fixed_latency_ms": 1 * 1000.0 / eval_cfg["env"]["env_fps"]}, {"method": "fixed", "fixed_latency_ms": 2 * 1000.0 / eval_cfg["env"]["env_fps"]}, {"method": "fixed", "fixed_latency_ms": 3 * 1000.0 / eval_cfg["env"]["env_fps"]}, {"method": 'normal', "seed": eval_cfg["experiment"]["seed"], 'min_latency_ms': 0 * 1000.0 / eval_cfg["env"]["env_fps"], 'max_latency_ms': 4 * 1000.0 / eval_cfg["env"]["env_fps"], 'mean_latency_ms': 2.0 * 1000.0 / eval_cfg["env"]["env_fps"], 'std_latency_ms': 1.0 * 1000.0 / eval_cfg["env"]["env_fps"]}, {"method": 'uniform', "seed": eval_cfg["experiment"]["seed"], 'min_latency_ms': 1 * 1000.0 / eval_cfg["env"]["env_fps"], 'max_latency_ms': 3 * 1000.0 / eval_cfg["env"]["env_fps"]}]
