from __future__ import annotations

import subprocess
import shlex

import pytest
from pathlib import Path

from omegaconf import OmegaConf

from examples.rl_games.scripts import launch_train, run_experiment
from examples.rl_games.scripts.setup_training_assets import _resolve_explicit_resume_checkpoint, _task_converter_and_verifier


REPO_ROOT = Path(__file__).resolve().parents[2]

ARCHIVED_COMMAND_MODELS = ("openvla", "pi0", "pi05")
ENVS = ("flappy", "demon_attack", "deadly_corridor")
OPENVLA_DEADLY_CROSS_TASK_SETUPS = (
    "flappy_zero_deadly_mixed",
    "deadly_zero_flappy_mixed",
    "demon_zero_deadly_mixed",
    "deadly_zero_demon_mixed",
    "flappy_demon_deadly_024",
    "defendtheline_zero_deadly_zero",
)
OPENVLA_DEADLY_CROSS_TASK_SCRIPTS = (
    "flappy_deadly/flappy_zero_deadly_mixed",
    "deadly_zero_flappy_mixed",
    "demon_zero_deadly_mixed",
    "deadly_zero_demon_mixed",
    "flappy_demon_deadly_024",
    "defendtheline_deadly/defendtheline_zero_deadly_zero",
)


def _command_path(model: str, env: str) -> Path:
    command_name = f"train_{env}_{model}.sh"
    archived_path = REPO_ROOT / "commands" / model / command_name
    if archived_path.exists():
        return archived_path
    return REPO_ROOT / "commands" / command_name


def _read_archived_openvla_command(env: str) -> str:
    command_name = f"train_{env}_openvla.sh"
    archived_path = REPO_ROOT / "commands" / "openvla" / command_name

    assert archived_path.exists()
    assert not (REPO_ROOT / "commands" / command_name).exists()
    return archived_path.read_text(encoding="utf-8")


def _read_archived_pi05_command(env: str) -> str:
    command_name = f"train_{env}_pi05.sh"
    archived_path = REPO_ROOT / "commands" / "pi05" / command_name

    assert archived_path.exists()
    assert not (REPO_ROOT / "commands" / command_name).exists()
    return archived_path.read_text(encoding="utf-8")


def _read_archived_pi0_command(env: str) -> str:
    command_name = f"train_{env}_pi0.sh"
    archived_path = REPO_ROOT / "commands" / "pi0" / command_name

    assert archived_path.exists()
    assert not (REPO_ROOT / "commands" / command_name).exists()
    return archived_path.read_text(encoding="utf-8")


def test_flappy_openvla_command_preserves_release_defaults() -> None:
    command_text = _read_archived_openvla_command("flappy")

    assert 'LATENCY="${1:-${LATENCY:-0}}"' in command_text
    assert 'MAX_EPISODES="${MAX_EPISODES:-200}"' in command_text
    assert 'MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-5000}"' in command_text
    assert 'PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-32}"' in command_text
    assert 'GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-4}"' in command_text
    assert 'SAVE_INTERVAL="${SAVE_INTERVAL:-100}"' in command_text
    assert 'DATASET_LOCAL_DIR="${DATASET_LOCAL_DIR:-data/flappy_fix_latency_${LATENCY}_${MAX_EPISODES}ep}"' in command_text
    assert 'RUN_ID="${RUN_ID:-flappy_fix_latency_${LATENCY}_${MAX_EPISODES}ep}"' in command_text
    assert 'PROMPT_MAP_PATH="${PROMPT_MAP_PATH:-prompt/flappy_latency_prompt_map.json}"' in command_text
    assert 'rl_games.env_eval.latency.prompt_map_path="${PROMPT_MAP_PATH}"' in command_text
    assert "rl_games.env_eval.mid_train.enabled=false" in command_text
    assert "rl_games.env_eval.post_train.enabled=false" in command_text
    assert "checkpoint.save_pt_file=false" in command_text
    assert "checkpoint.save_best_model=false" in command_text
    assert "checkpoint.local.keep_last_n=1" in command_text
    assert "checkpoint.load=none" in command_text


def test_flappy_pi05_command_matches_openvla_data_and_training_defaults() -> None:
    command_text = _read_archived_pi05_command("flappy")

    assert 'LATENCY="${1:-${LATENCY:-0}}"' in command_text
    assert 'MAX_EPISODES="${MAX_EPISODES:-200}"' in command_text
    assert 'MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-5000}"' in command_text
    assert 'PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-32}"' in command_text
    assert 'GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-4}"' in command_text
    assert 'SAVE_INTERVAL="${SAVE_INTERVAL:-100}"' in command_text
    assert 'DATASET_LOCAL_DIR="${DATASET_LOCAL_DIR:-data/flappy_fix_latency_${LATENCY}_${MAX_EPISODES}ep}"' in command_text
    assert 'RUN_ID="${RUN_ID:-pi05_flappy_fix_latency_${LATENCY}_${MAX_EPISODES}ep}"' in command_text
    assert 'PROMPT_MAP_PATH="${PROMPT_MAP_PATH:-prompt/flappy_latency_prompt_map.json}"' in command_text
    assert "model=pi05" in command_text
    assert "env=flappy" in command_text
    assert "init=bridge" in command_text
    assert 'rl_games.env_eval.latency.prompt_map_path="${PROMPT_MAP_PATH}"' in command_text
    assert "rl_games.env_eval.mid_train.enabled=false" in command_text
    assert "rl_games.env_eval.post_train.enabled=false" in command_text
    assert "checkpoint.save_pt_file=false" in command_text
    assert "checkpoint.save_best_model=false" in command_text
    assert "checkpoint.local.keep_last_n=1" in command_text
    assert "checkpoint.load=none" in command_text


def test_demon_attack_pi05_command_matches_openvla_data_and_training_defaults() -> None:
    command_text = _read_archived_pi05_command("demon_attack")

    assert 'LATENCY="${1:-${LATENCY:-0}}"' in command_text
    assert 'MAX_EPISODES="${MAX_EPISODES:-200}"' in command_text
    assert 'MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-7000}"' in command_text
    assert 'PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-32}"' in command_text
    assert 'GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-4}"' in command_text
    assert 'SAVE_INTERVAL="${SAVE_INTERVAL:-100}"' in command_text
    assert 'DATASET_LOCAL_DIR="${DATASET_LOCAL_DIR:-data/demon_attack_fix_latency_${LATENCY}_${MAX_EPISODES}ep}"' in command_text
    assert 'RUN_ID="${RUN_ID:-pi05_demon_attack_fix_latency_${LATENCY}_${MAX_EPISODES}ep}"' in command_text
    assert 'PROMPT_MAP_PATH="${PROMPT_MAP_PATH:-prompt/demon_attack_latency_prompt_map.json}"' in command_text
    assert "model=pi05" in command_text
    assert "env=demon_attack" in command_text
    assert "init=bridge" in command_text
    assert 'rl_games.env_eval.latency.prompt_map_path="${PROMPT_MAP_PATH}"' in command_text
    assert "rl_games.env_eval.mid_train.enabled=false" in command_text
    assert "rl_games.env_eval.post_train.enabled=false" in command_text


def test_deadly_corridor_pi05_command_preserves_pi05_loss_and_matches_openvla_training_defaults() -> None:
    command_text = _read_archived_pi05_command("deadly_corridor")

    assert 'LATENCY="${LATENCY:-0}"' in command_text
    assert 'MAX_EPISODES="${MAX_EPISODES:-1000}"' in command_text
    assert 'MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-500}"' in command_text
    assert 'PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-32}"' in command_text
    assert 'GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-4}"' in command_text
    assert 'SAVE_INTERVAL="${SAVE_INTERVAL:-100}"' in command_text
    assert 'DATASET_LOCAL_DIR="${DATASET_LOCAL_DIR:-data/deadly_corridor_fix_latency_${LATENCY}_${MAX_EPISODES}ep}"' in command_text
    assert 'RUN_ID="${RUN_ID:-pi05_deadly_corridor_fix_latency_${LATENCY}_${MAX_EPISODES}ep}"' in command_text
    assert 'PROMPT_MAP_PATH="${PROMPT_MAP_PATH:-prompt/deadly_corridor_latency_prompt_map.json}"' in command_text
    assert "model=pi05" in command_text
    assert "env=deadly_corridor" in command_text
    assert "init=bridge" in command_text
    assert "rl_games.env_eval.deadly.action_layout=multibinary_7" in command_text
    assert "rl_games.deadly_corridor_loss_type=" not in command_text
    assert 'rl_games.env_eval.latency.prompt_map_path="${PROMPT_MAP_PATH}"' in command_text
    assert "rl_games.env_eval.mid_train.enabled=false" in command_text
    assert "rl_games.env_eval.post_train.enabled=false" in command_text
    assert '"$@"' in command_text


def test_pi0_commands_match_openvla_data_and_training_defaults() -> None:
    env_specs = {
        "flappy": (200, 5000, "flappy"),
        "demon_attack": (200, 7000, "demon_attack"),
        "deadly_corridor": (1000, 500, "deadly_corridor"),
    }

    for env, (max_episodes, max_train_steps, prompt_name) in env_specs.items():
        command_text = _read_archived_pi0_command(env)

        assert f'MAX_EPISODES="${{MAX_EPISODES:-{max_episodes}}}"' in command_text
        assert f'MAX_TRAIN_STEPS="${{MAX_TRAIN_STEPS:-{max_train_steps}}}"' in command_text
        assert 'PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-32}"' in command_text
        assert 'GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-4}"' in command_text
        assert 'SAVE_INTERVAL="${SAVE_INTERVAL:-100}"' in command_text
        assert f'DATASET_LOCAL_DIR="${{DATASET_LOCAL_DIR:-data/{env}_fix_latency_${{LATENCY}}_${{MAX_EPISODES}}ep}}"' in command_text
        assert f'RUN_ID="${{RUN_ID:-pi0_{env}_fix_latency_${{LATENCY}}_${{MAX_EPISODES}}ep}}"' in command_text
        assert f'PROMPT_MAP_PATH="${{PROMPT_MAP_PATH:-prompt/{prompt_name}_latency_prompt_map.json}}"' in command_text
        assert "model=pi0" in command_text
        assert f"env={env}" in command_text
        assert "init=bridge" in command_text
        assert 'rl_games.env_eval.latency.prompt_map_path="${PROMPT_MAP_PATH}"' in command_text
        assert "rl_games.env_eval.mid_train.enabled=false" in command_text
        assert "rl_games.env_eval.post_train.enabled=false" in command_text
        assert "checkpoint.save_pt_file=false" in command_text
        assert "checkpoint.save_best_model=false" in command_text
        assert "checkpoint.local.keep_last_n=1" in command_text
        assert "checkpoint.load=none" in command_text

    deadly_command_text = _read_archived_pi0_command("deadly_corridor")
    assert "rl_games.env_eval.deadly.action_layout=multibinary_7" in deadly_command_text
    assert "rl_games.deadly_corridor_loss_type=" not in deadly_command_text
    assert '"$@"' in deadly_command_text


def test_demon_attack_openvla_command_preserves_release_defaults() -> None:
    command_text = _read_archived_openvla_command("demon_attack")

    assert 'LATENCY="${1:-${LATENCY:-0}}"' in command_text
    assert 'MAX_EPISODES="${MAX_EPISODES:-200}"' in command_text
    assert 'MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-7000}"' in command_text
    assert 'PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-32}"' in command_text
    assert 'GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-4}"' in command_text
    assert 'SAVE_INTERVAL="${SAVE_INTERVAL:-100}"' in command_text
    assert 'MID_TRAIN_INTERVAL="${MID_TRAIN_INTERVAL:-500}"' in command_text
    assert 'MID_TRAIN_LATENCIES="${MID_TRAIN_LATENCIES:-[${LATENCY}]}"' in command_text
    assert 'DATASET_LOCAL_DIR="${DATASET_LOCAL_DIR:-data/demon_attack_fix_latency_${LATENCY}_${MAX_EPISODES}ep}"' in command_text
    assert 'RUN_ID="${RUN_ID:-demon_attack_fix_latency_${LATENCY}_${MAX_EPISODES}ep}"' in command_text
    assert 'PROMPT_MAP_PATH="${PROMPT_MAP_PATH:-prompt/demon_attack_latency_prompt_map.json}"' in command_text
    assert 'rl_games.env_eval.latency.prompt_map_path="${PROMPT_MAP_PATH}"' in command_text
    assert "rl_games.env_eval.mid_train.enabled=true" in command_text
    assert 'rl_games.env_eval.mid_train.interval_steps="${MID_TRAIN_INTERVAL}"' in command_text
    assert '"rl_games.env_eval.mid_train.latencies=${MID_TRAIN_LATENCIES}"' in command_text
    assert "rl_games.env_eval.post_train.enabled=false" in command_text
    assert "checkpoint.save_pt_file=false" in command_text
    assert "checkpoint.save_best_model=false" in command_text
    assert "checkpoint.local.keep_last_n=1" in command_text
    assert "checkpoint.load=none" in command_text


def test_deadly_corridor_openvla_command_preserves_release_defaults() -> None:
    command_text = _read_archived_openvla_command("deadly_corridor")

    assert 'LATENCY="${LATENCY:-0}"' in command_text
    assert 'MAX_EPISODES="${MAX_EPISODES:-1000}"' in command_text
    assert 'MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-500}"' in command_text
    assert 'PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-32}"' in command_text
    assert 'GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-4}"' in command_text
    assert 'SAVE_INTERVAL="${SAVE_INTERVAL:-100}"' in command_text
    assert 'DATASET_LOCAL_DIR="${DATASET_LOCAL_DIR:-data/deadly_corridor_fix_latency_${LATENCY}_${MAX_EPISODES}ep}"' in command_text
    assert 'RUN_ID="${RUN_ID:-deadly_corridor_fix_latency_${LATENCY}_${MAX_EPISODES}ep}"' in command_text
    assert 'PROMPT_MAP_PATH="${PROMPT_MAP_PATH:-prompt/deadly_corridor_latency_prompt_map.json}"' in command_text
    assert "rl_games.deadly_corridor_loss_type=multibinary_bce" in command_text
    assert "rl_games.env_eval.deadly.action_layout=multibinary_7" in command_text
    assert 'rl_games.env_eval.latency.prompt_map_path="${PROMPT_MAP_PATH}"' in command_text
    assert "rl_games.env_eval.mid_train.enabled=false" in command_text
    assert "rl_games.env_eval.post_train.enabled=false" in command_text
    assert "checkpoint.save_pt_file=false" in command_text
    assert "checkpoint.save_best_model=false" in command_text
    assert "checkpoint.local.keep_last_n=1" in command_text
    assert "checkpoint.load=none" in command_text


def test_flappy_openvla_curriculum_commands_preserve_release_defaults() -> None:
    for strategy in ("cumulative", "exclusive"):
        command_text = _read_archived_openvla_command(f"flappy_curriculum_{strategy}")

        assert 'LATENCY_FILTER="${LATENCY_FILTER:-[0,1,2,3,4]}"' in command_text
        assert 'EPISODES_PER_LATENCY="${EPISODES_PER_LATENCY:-40}"' in command_text
        assert 'MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-5000}"' in command_text
        assert 'PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-32}"' in command_text
        assert 'GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-4}"' in command_text
        assert 'SAVE_INTERVAL="${SAVE_INTERVAL:-100}"' in command_text
        assert 'DATASET_LOCAL_DIR="${DATASET_LOCAL_DIR:-data/flappy_mixed_latency_${EPISODES_PER_LATENCY}ep_per_lat}"' in command_text
        assert f'RUN_ID="${{RUN_ID:-flappy_curriculum_{strategy}_${{EPISODES_PER_LATENCY}}ep_per_latency}}"' in command_text
        assert 'PROMPT_MAP_PATH="${PROMPT_MAP_PATH:-prompt/flappy_latency_prompt_map.json}"' in command_text
        assert f"mode=curriculum_{strategy}" in command_text
        assert '"dataset.latency_filter=${LATENCY_FILTER}"' in command_text
        assert 'dataset.episodes_per_latency="${EPISODES_PER_LATENCY}"' in command_text
        assert 'rl_games.env_eval.latency.prompt_map_path="${PROMPT_MAP_PATH}"' in command_text
        assert "rl_games.env_eval.mid_train.enabled=false" in command_text
        assert "rl_games.env_eval.post_train.enabled=false" in command_text
        assert "checkpoint.save_pt_file=false" in command_text
        assert "checkpoint.save_best_model=false" in command_text
        assert "checkpoint.local.keep_last_n=1" in command_text
        assert "checkpoint.load=none" in command_text


def test_training_command_matrix_targets_hydra_launcher() -> None:
    for model in ARCHIVED_COMMAND_MODELS:
        for env in ENVS:
            command_path = _command_path(model, env)

            assert command_path.exists(), f"Missing command wrapper: {command_path}"

            command_text = command_path.read_text(encoding="utf-8")
            assert "python examples/rl_games/scripts/launch_train.py" in command_text
            assert "examples/rl_games/experiments/" not in command_text
            assert f"model={model}" in command_text
            assert f"env={env}" in command_text
            assert "init=bridge" in command_text
            assert "mode=single" not in command_text
            assert "WANDB_ENTITY" not in command_text
            assert "wandb_entity=" not in command_text
            assert "rl_games.env_eval.post_train.latencies=" not in command_text
            assert "trainer.batch_size=" not in command_text
            assert "datasets.vla_data.per_device_batch_size=" in command_text
            assert "dataset.source_hf=data/" not in command_text
            assert "checkpoint.save_pt_file=true" not in command_text


def test_training_commands_are_valid_bash() -> None:
    command_paths = [str(_command_path(model, env)) for model in ARCHIVED_COMMAND_MODELS for env in ENVS]

    subprocess.run(["bash", "-n", *command_paths], check=True, cwd=REPO_ROOT)


def test_wan_oft_temporal_probe_command_resolves_repo_root_from_archive() -> None:
    command_name = "probe_flappy_wan_oft_temporal_latents.sh"
    command_path = REPO_ROOT / "commands" / "wanoft" / command_name

    assert command_path.exists()
    assert not (REPO_ROOT / "commands" / command_name).exists()
    command_text = command_path.read_text(encoding="utf-8")
    assert 'REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"' in command_text


def test_kv_memory_training_command_shards_mid_train_latency_bench_eval() -> None:
    command_text = (REPO_ROOT / "commands" / "memory" / "train_flappy_openvla_kv_memory.sh").read_text(
        encoding="utf-8"
    )

    assert "launch.num_processes=2" in command_text
    assert "rl_games.env_eval.distributed_mode=rank_sharded" in command_text


def test_memory_training_commands_save_bf16_safetensors_model_file() -> None:
    command_paths = sorted((REPO_ROOT / "commands" / "memory").glob("*.sh"))

    for command_path in command_paths:
        command_text = command_path.read_text(encoding="utf-8")
        assert "checkpoint.save_safetensors_file=true" in command_text, command_path
        assert "checkpoint.save_pt_file=false" in command_text, command_path


def test_wan_oft_commands_are_valid_bash() -> None:
    command_paths = [
        REPO_ROOT / "commands" / "wanoft" / "train_flappy_wan_oft.sh",
        REPO_ROOT / "commands" / "wanoft" / "train_flappy_wan_oft_mixed_latency.sh",
        REPO_ROOT / "commands" / "wanoft" / "train_flappy_wan_oft_curriculum_cumulative.sh",
        REPO_ROOT / "commands" / "wanoft" / "train_flappy_wan_oft_curriculum_exclusive.sh",
        REPO_ROOT / "commands" / "wanoft" / "train_demon_attack_wan_oft.sh",
        REPO_ROOT / "commands" / "wanoft" / "train_deadly_corridor_wan_oft.sh",
    ]

    subprocess.run(["bash", "-n", *[str(path) for path in command_paths]], check=True, cwd=REPO_ROOT)


def test_wan_oft_commands_explicitly_use_eval_core() -> None:
    command_paths = sorted((REPO_ROOT / "commands").rglob("train_*wan_oft*.sh"))

    assert command_paths
    for command_path in command_paths:
        command_text = command_path.read_text(encoding="utf-8")
        assert "rl_games.env_eval.eval_backend=eval_core" in command_text, command_path


def test_archived_release_commands_share_shell_structure() -> None:
    command_paths = sorted((REPO_ROOT / "commands" / "openvla").glob("train_*.sh"))
    command_paths += sorted((REPO_ROOT / "commands" / "pi0").glob("train_*.sh"))
    command_paths += sorted((REPO_ROOT / "commands" / "pi05").glob("train_*.sh"))
    command_paths += sorted((REPO_ROOT / "commands" / "wanoft").glob("train_*.sh"))

    for command_path in command_paths:
        command_text = command_path.read_text(encoding="utf-8")
        assert command_text.startswith("#!/usr/bin/env bash\n\nset -euo pipefail\n\n# Usage:\n"), command_path


def test_archived_wan_oft_commands_only_run_post_train_eval() -> None:
    script_names = (
        "train_deadly_corridor_wan_oft.sh",
        "train_demon_attack_wan_oft.sh",
        "train_flappy_wan_oft.sh",
        "train_flappy_wan_oft_mixed_latency.sh",
        "train_flappy_wan_oft_curriculum_cumulative.sh",
        "train_flappy_wan_oft_curriculum_exclusive.sh",
    )

    for script_name in script_names:
        command_path = REPO_ROOT / "commands" / "wanoft" / script_name
        assert command_path.exists()
        assert not (REPO_ROOT / "commands" / script_name).exists()
        command_text = command_path.read_text(encoding="utf-8")

        assert "datasets.vla_data.eval_data_mix=" not in command_text
        assert "trainer.eval_interval=" not in command_text
        assert "trainer.eval_num_batches=" not in command_text
        assert "rl_games.env_eval.mid_train.enabled=false" in command_text
        assert "rl_games.env_eval.mid_train.interval_steps=" not in command_text
        assert "rl_games.env_eval.mid_train.latencies=" not in command_text
        assert "rl_games.env_eval.mid_train.num_episodes=" not in command_text
        if script_name == "train_deadly_corridor_wan_oft.sh":
            assert "rl_games.env_eval.enabled=false" in command_text
            assert "rl_games.env_eval.post_train.enabled=false" in command_text
            assert "rl_games.env_eval.post_train.latencies=" not in command_text
            assert "rl_games.env_eval.post_train.num_episodes=" not in command_text
        else:
            assert "rl_games.env_eval.post_train.enabled=true" in command_text
            assert "rl_games.env_eval.post_train.latencies=" in command_text
            assert "rl_games.env_eval.post_train.num_episodes=" in command_text


def test_demon_attack_wan_oft_command_preserves_parameterized_post_train_eval() -> None:
    command_text = (
        REPO_ROOT / "commands" / "wanoft" / "train_demon_attack_wan_oft.sh"
    ).read_text(encoding="utf-8")

    assert 'POST_TRAIN_LATENCIES="${POST_TRAIN_LATENCIES:-[${LATENCY}]}"' in command_text
    assert 'POST_TRAIN_NUM_EPISODES="${POST_TRAIN_NUM_EPISODES:-20}"' in command_text
    assert '"rl_games.env_eval.post_train.latencies=${POST_TRAIN_LATENCIES}"' in command_text
    assert 'rl_games.env_eval.post_train.num_episodes="${POST_TRAIN_NUM_EPISODES}"' in command_text
    assert "if [[ $# -gt 0 ]]; then\n  shift\nfi" in command_text
    assert 'OBSERVATION_INDICES="["' in command_text
    assert "for ((offset = CONTEXT_WINDOW - 1; offset >= 1; offset--)); do" in command_text
    assert '"datasets.vla_data.observation_indices=${OBSERVATION_INDICES}"' in command_text
    assert '"$@"' in command_text


def test_deadly_corridor_wan_oft_command_preserves_context_and_action_contract() -> None:
    command_text = (
        REPO_ROOT / "commands" / "wanoft" / "train_deadly_corridor_wan_oft.sh"
    ).read_text(encoding="utf-8")

    assert 'LATENCY="${1:-${LATENCY:-6}}"' in command_text
    assert 'CONTEXT_WINDOW="${CONTEXT_WINDOW:-5}"' in command_text
    assert 'MAX_EPISODES="${MAX_EPISODES:-1000}"' in command_text
    assert "model=wan_oft" in command_text
    assert "env=deadly_corridor" in command_text
    assert "init=wan_oft_libero" in command_text
    assert "dataset.converted_name=deadly_corridor_train__bridge" in command_text
    assert "datasets.vla_data.data_mix=deadly_corridor_train__bridge" in command_text
    assert 'datasets.vla_data.image_sequence_length="${CONTEXT_WINDOW}"' in command_text
    assert '"datasets.vla_data.observation_indices=${OBSERVATION_INDICES}"' in command_text
    assert 'framework.world_model.num_frames="${CONTEXT_WINDOW}"' in command_text
    assert "rl_games.deadly_corridor_loss_type=current_multibinary_bce" in command_text
    assert "rl_games.env_eval.deadly.action_layout=multibinary_7" in command_text
    assert "rl_games.env_eval.deadly.multibinary_threshold=0.0" in command_text
    assert "rl_games.env_eval.enabled=false" in command_text
    assert "rl_games.env_eval.post_train.enabled=false" in command_text
    assert "rl_games.env_eval.post_train.latencies=" not in command_text
    assert "if [[ $# -gt 0 ]]; then\n  shift\nfi" in command_text
    assert '"$@"' in command_text


def test_flappy_wan_oft_command_preserves_parameterized_fix_latency_defaults() -> None:
    command_text = (
        REPO_ROOT / "commands" / "wanoft" / "train_flappy_wan_oft.sh"
    ).read_text(encoding="utf-8")

    assert 'LATENCY="${1:-${LATENCY:-0}}"' in command_text
    assert "if [[ $# -gt 0 ]]; then\n  shift\nfi" in command_text
    assert 'CONTEXT_WINDOW="${CONTEXT_WINDOW:-5}"' in command_text
    assert 'MAX_EPISODES="${MAX_EPISODES:-200}"' in command_text
    assert 'MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-2000}"' in command_text
    assert 'PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-4}"' in command_text
    assert 'GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-32}"' in command_text
    assert 'POST_TRAIN_NUM_EPISODES="${POST_TRAIN_NUM_EPISODES:-20}"' in command_text
    assert 'MAX_STEPS_PER_EPISODE="${MAX_STEPS_PER_EPISODE:-3600}"' in command_text
    assert 'POST_TRAIN_LATENCIES="${POST_TRAIN_LATENCIES:-[${LATENCY}]}"' in command_text
    assert 'DATASET_LOCAL_DIR="${DATASET_LOCAL_DIR:-data/flappy_fix_latency_${LATENCY}_${MAX_EPISODES}ep_context${CONTEXT_WINDOW}}"' in command_text
    assert 'run_id="${RUN_ID}"' in command_text
    assert 'paths.dataset_local_dir="${DATASET_LOCAL_DIR}"' in command_text
    assert 'rl_games.env_eval.latency.prompt_map_path="${PROMPT_MAP_PATH}"' in command_text
    assert '"rl_games.env_eval.post_train.latencies=${POST_TRAIN_LATENCIES}"' in command_text
    assert 'OBSERVATION_INDICES="["' in command_text
    assert "for ((offset = CONTEXT_WINDOW - 1; offset >= 1; offset--)); do" in command_text
    assert '"datasets.vla_data.observation_indices=${OBSERVATION_INDICES}"' in command_text
    assert '"$@"' in command_text


def test_wan_oft_flappy_mixed_latency_command_preserves_context5_baseline() -> None:
    command_path = REPO_ROOT / "commands" / "wanoft" / "train_flappy_wan_oft_mixed_latency.sh"
    command_text = command_path.read_text(encoding="utf-8")

    assert "model=wan_oft" in command_text
    assert "env=flappy" in command_text
    assert "init=wan_oft_libero" in command_text
    assert "mode=mixed_latency" in command_text
    assert "run_id=\"${RUN_ID}\"" in command_text
    assert "paths.dataset_local_dir=\"${DATASET_LOCAL_DIR}\"" in command_text
    assert "dataset.converted_name=flappy_mixed_latency_train__bridge" in command_text
    assert '"dataset.latency_filter=${LATENCY_FILTER}"' in command_text
    assert "dataset.episodes_per_latency=\"${EPISODES_PER_LATENCY}\"" in command_text
    assert "datasets.vla_data.data_mix=flappy_mixed_latency_train__bridge" in command_text
    assert "datasets.vla_data.eval_data_mix=" not in command_text
    assert "datasets.vla_data.obs_image_size=[224,224]" in command_text
    assert "datasets.vla_data.image_sequence_length=\"${CONTEXT_WINDOW}\"" in command_text
    assert "datasets.vla_data.observation_indices=[-4,-3,-2,-1,0]" in command_text
    assert "datasets.vla_data.sequential_step_sampling=false" in command_text
    assert "framework.world_model.num_frames=\"${CONTEXT_WINDOW}\"" in command_text
    assert "framework.action_model.loss_type=current_discrete_ce" in command_text
    assert "trainer.learning_rate.action_query_proj=1.0e-4" in command_text
    assert "+trainer.learning_rate.action_query_proj=1.0e-4" not in command_text
    assert "trainer.distributed_backend=none" in command_text
    assert "launch.use_accelerate=false" in command_text
    assert "trainer.gradient_accumulation_steps=\"${GRADIENT_ACCUMULATION_STEPS}\"" in command_text
    assert "datasets.vla_data.per_device_batch_size=\"${PER_DEVICE_BATCH_SIZE}\"" in command_text
    assert "rl_games.env_eval.latency.prompt_map_path=\"${PROMPT_MAP_PATH}\"" in command_text
    assert "rl_games.env_eval.mid_train.enabled=false" in command_text
    assert '"rl_games.env_eval.post_train.latencies=${POST_TRAIN_LATENCIES}"' in command_text
    assert "checkpoint.save_training_state=false" in command_text


def test_wan_oft_flappy_curriculum_commands_enable_sequential_sampling() -> None:
    for script_name, mode_name, strategy in (
        ("train_flappy_wan_oft_curriculum_cumulative.sh", "curriculum_cumulative", "cumulative"),
        ("train_flappy_wan_oft_curriculum_exclusive.sh", "curriculum_exclusive", "exclusive"),
    ):
        command_text = (REPO_ROOT / "commands" / "wanoft" / script_name).read_text(encoding="utf-8")

        assert "model=wan_oft" in command_text
        assert "env=flappy" in command_text
        assert "init=wan_oft_libero" in command_text
        assert f"mode={mode_name}" in command_text
        assert f"datasets.vla_data.latency_curriculum.strategy={strategy}" in command_text
        assert '"datasets.vla_data.latency_curriculum.latencies=${LATENCY_FILTER}"' in command_text
        assert "datasets.vla_data.sequential_step_sampling=true" in command_text
        assert "datasets.vla_data.action_balance.enabled=false" in command_text
        assert "dataset.converted_name=flappy_mixed_latency_train__bridge" in command_text
        assert "datasets.vla_data.data_mix=flappy_mixed_latency_train__bridge" in command_text
        assert "datasets.vla_data.latency_curriculum.eval_at_phase_end=false" in command_text


def test_flappy_wan_oft_fixed_latency_pipeline_uses_archived_memory_history_path() -> None:
    script_path = REPO_ROOT / "scripts" / "run_flappy_wan_oft_pipeline.sh"
    script_text = script_path.read_text(encoding="utf-8")

    assert "--mode <" not in script_text
    assert "\n    --mode)" not in script_text
    assert 'DATASET_REPO="latency-sensitive-bench/benchmark-datasets"' in script_text
    assert 'DATASET_CONFIG="flappy_fixed_latency_3_200ep_7k2steps"' in script_text
    assert "LATENCY=3" in script_text
    assert "examples/rl_games/install/bootstrap.sh" in script_text
    assert "--tier dev" in script_text
    assert "--model wan_oft" in script_text
    assert 'CONDA_ENV_NAME="starvla_rl_games_wan_oft"' in script_text
    assert "--conda-env" not in script_text
    assert "--python-version" not in script_text
    assert "convert_flappy_history_to_starvla_lerobot.py" in script_text
    assert "convert_flappy_to_starvla_lerobot.py" not in script_text
    assert '--dataset-name "${DATASET_REPO}"' in script_text
    assert '--dataset-config-name "${DATASET_CONFIG}"' in script_text
    assert '--max-episodes "${MAX_EPISODES}"' in script_text
    assert "--context-images-column" not in script_text
    assert '"dataset.source_hf=${DATASET_REPO}"' in script_text
    assert '"dataset.config_name=${DATASET_CONFIG}"' in script_text
    assert '"paths.run_root_dir=${RUN_ROOT_DIR}"' in script_text
    assert '"paths.base_model_dir=${BASE_MODEL_DIR}"' in script_text
    assert '"initialization.checkpoint_local_dir=${INIT_CHECKPOINT_DIR}"' in script_text
    assert 'POST_TRAIN_LATENCIES="[3]"' in script_text
    assert (
        'bash commands/wanoft/train_flappy_wan_oft.sh 3 "${TRAIN_OVERRIDES[@]}"'
        in script_text
    )
    assert "train_flappy_wan_oft_curriculum_cumulative.sh" not in script_text
    assert "train_flappy_wan_oft_curriculum_exclusive.sh" not in script_text
    assert "UPLOAD_REPO=\"${UPLOAD_REPO:-latency-sensitive-bench/benchmark-models}\"" in script_text
    assert "UPLOAD_PATH_IN_REPO=\"${UPLOAD_PATH_IN_REPO:-latency-aware/flappy/vla/starvla-wanoft-h8/${RUN_ID}}\"" in script_text
    assert "--image-sequence-length \"${CONTEXT_WINDOW}\"" in script_text
    assert "hf upload \"${UPLOAD_REPO}\" \"${RUN_DIR}\" \"${UPLOAD_PATH_IN_REPO}\"" in script_text
    subprocess.run(["bash", "-n", str(script_path)], check=True, cwd=REPO_ROOT)
    help_result = subprocess.run(
        ["bash", str(script_path), "--help"],
        check=True,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert "fixed-latency-3" in help_result.stdout
    assert "--mode" not in help_result.stdout
    assert "--accept-rom-license" not in help_result.stdout


def test_demon_attack_wan_oft_fixed_latency_pipeline_uses_archived_memory_history_path() -> None:
    script_path = REPO_ROOT / "scripts" / "run_demon_attack_wan_oft_pipeline.sh"
    training_command_path = REPO_ROOT / "commands" / "wanoft" / "train_demon_attack_wan_oft.sh"
    script_text = script_path.read_text(encoding="utf-8")

    assert training_command_path.exists()
    assert "--latency <" not in script_text
    assert "\n    --latency)" not in script_text
    assert 'DATASET_REPO="latency-sensitive-bench/benchmark-datasets"' in script_text
    assert 'DATASET_CONFIG="demon_attack_fixed_latency_6_200ep_7k2steps"' in script_text
    assert "LATENCY_RAW_FRAMES=6" in script_text
    assert 'CONDA_ENV_NAME="starvla_rl_games_wan_oft"' in script_text
    assert 'BENCHMARK_ROOT="${LATENCY_BENCH_ROOT:-}"' in script_text
    assert "--benchmark-root <path>" in script_text
    assert "rev-parse --show-superproject-working-tree" in script_text
    assert 'third_party/flappy-bird-gymnasium/setup.py' in script_text
    assert 'LATENCY_BENCH_ROOT="${BENCHMARK_ROOT}" "${BOOTSTRAP_ARGS[@]}"' in script_text
    assert "--tier dev" in script_text
    assert "--model wan_oft" in script_text
    assert "--conda-env" not in script_text
    assert "--python-version" not in script_text
    assert "convert_demon_attack_history_to_starvla_lerobot.py" in script_text
    assert "convert_demon_attack_to_starvla_lerobot.py" not in script_text
    assert '--dataset-name "${DATASET_REPO}"' in script_text
    assert '--dataset-config-name "${DATASET_CONFIG}"' in script_text
    assert '--max-episodes "${MAX_EPISODES}"' in script_text
    assert "--context-images-column" not in script_text
    assert '"dataset.source_hf=${DATASET_REPO}"' in script_text
    assert '"dataset.config_name=${DATASET_CONFIG}"' in script_text
    assert '"paths.run_root_dir=${RUN_ROOT_DIR}"' in script_text
    assert '"paths.base_model_dir=${BASE_MODEL_DIR}"' in script_text
    assert '"initialization.checkpoint_local_dir=${INIT_CHECKPOINT_DIR}"' in script_text
    assert '"rl_games.env_eval.post_train.enabled=false"' in script_text
    assert (
        'bash commands/wanoft/train_demon_attack_wan_oft.sh 6 "${TRAIN_OVERRIDES[@]}"'
        in script_text
    )
    assert 'bash commands/train_demon_attack_wan_oft.sh "${LATENCY}"' not in script_text
    assert "--image-sequence-length \"${CONTEXT_WINDOW}\"" in script_text
    assert 'manifest.get("latency_raw_frames") != [6]' in script_text
    assert 'manifest.get("latency_env_steps") != [1.5]' in script_text
    subprocess.run(["bash", "-n", str(script_path)], check=True, cwd=REPO_ROOT)
    help_result = subprocess.run(
        ["bash", str(script_path), "--help"],
        check=True,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert "fixed-raw-frame-latency-6" in help_result.stdout
    assert "1.5 policy decision steps" in help_result.stdout
    assert "--latency" not in help_result.stdout
    assert "--accept-rom-license" not in help_result.stdout


def test_deadly_corridor_wan_oft_pipeline_uses_fixed_latency_history_data_without_core_eval() -> None:
    script_path = REPO_ROOT / "scripts" / "run_deadly_corridor_wan_oft_pipeline.sh"
    training_command_path = REPO_ROOT / "commands" / "wanoft" / "train_deadly_corridor_wan_oft.sh"
    script_text = script_path.read_text(encoding="utf-8")

    assert training_command_path.exists()
    assert 'DATASET_REPO="latency-sensitive-bench/benchmark-datasets"' in script_text
    assert 'DATASET_CONFIG="deadly_corridor_fixed_latency_6_1000ep_7k2steps"' in script_text
    assert "LATENCY_RAW_FRAMES=6" in script_text
    assert 'CONDA_ENV_NAME="starvla_rl_games_wan_oft"' in script_text
    assert "--model wan_oft" in script_text
    assert "rev-parse --show-superproject-working-tree" in script_text
    assert 'LATENCY_BENCH_ROOT="${BENCHMARK_ROOT}" "${BOOTSTRAP_ARGS[@]}"' in script_text
    assert "--conda-env" not in script_text
    assert "--python-version" not in script_text
    assert "convert_deadly_corridor_history_to_starvla_lerobot.py" in script_text
    assert "--dataset-config-name \"${DATASET_CONFIG}\"" in script_text
    assert "--max-episodes \"${MAX_EPISODES}\"" in script_text
    assert '--image-sequence-length "${CONTEXT_WINDOW}"' in script_text
    assert "--context-images-column" not in script_text
    assert '"rl_games.env_eval.enabled=false"' in script_text
    assert '"rl_games.env_eval.post_train.enabled=false"' in script_text
    assert 'bash commands/wanoft/train_deadly_corridor_wan_oft.sh 6 "${TRAIN_OVERRIDES[@]}"' in script_text
    assert 'manifest.get("action_layout") != "multibinary_7"' in script_text
    assert 'manifest.get("source_action_layout") != "deadly_corridor_joint_54"' in script_text
    assert 'manifest.get("latency_raw_frames") != [6]' in script_text
    assert 'manifest.get("latency_env_steps") != [1.5]' in script_text
    subprocess.run(["bash", "-n", str(script_path)], check=True, cwd=REPO_ROOT)
    help_result = subprocess.run(
        ["bash", str(script_path), "--help"],
        check=True,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert "fixed-raw-frame-latency-6" in help_result.stdout
    assert "Core environment evaluation is disabled" in help_result.stdout
    assert "--latency" not in help_result.stdout
    assert "--accept-rom-license" not in help_result.stdout


def test_demon_attack_wan_oft_pipeline_validates_raw_frame_latency_contract() -> None:
    script_text = (REPO_ROOT / "scripts" / "run_demon_attack_wan_oft_pipeline.sh").read_text(
        encoding="utf-8"
    )

    assert 'manifest.get("latency_unit") != "raw_frames"' in script_text
    assert 'manifest.get("latency_raw_frames") != [6]' in script_text
    assert 'manifest.get("latency_env_steps") != [1.5]' in script_text


def test_openvla_deadly_cross_task_scripts_are_valid_bash() -> None:
    script_dir = REPO_ROOT / "examples" / "rl_games" / "bash_scripts" / "openvla" / "bridge" / "cross_task"
    command_paths = [str(script_dir / f"{script}.sh") for script in OPENVLA_DEADLY_CROSS_TASK_SCRIPTS]

    subprocess.run(["bash", "-n", *command_paths], check=True, cwd=REPO_ROOT)


def test_gr00t_launchers_use_the_gr00t_model_environment_and_run_names() -> None:
    script_dir = REPO_ROOT / "examples" / "rl_games" / "bash_scripts" / "gr00t"

    for script_path in script_dir.rglob("*.sh"):
        script_text = script_path.read_text(encoding="utf-8")
        assert "model=gr00t" in script_text
        assert "conda activate starvla_rl_games_gr00t" in script_text
        assert "model=openvla" not in script_text
        assert "openvla_bridge" not in script_text


@pytest.mark.parametrize("latency", [2, 4, 6, 8])
def test_deadly_fixed_training_preserves_latency_dataset(latency: int, tmp_path: Path) -> None:
    """Keep each fixed-delay command bound to its original training data selection."""
    script = REPO_ROOT / "examples/rl_games/bash_scripts/openvla/bridge/fixed/deadly_corridor" / f"latency_{latency}.sh"
    command = script.read_text().split("python examples/rl_games/scripts/launch_train.py", 1)[1]
    overrides = shlex.split(command.replace("\\\n", " "))
    cfg = launch_train.compose_training_config(
        config_name="train", model="openvla", env="deadly_corridor", init="bridge", mode="single",
        overrides=overrides,
    )
    setup = launch_train.setup_namespace_from_cfg(cfg, tmp_path, "results/Checkpoints")

    assert setup.source_dataset_hf == "latency-sensitive-bench/benchmark-datasets"
    assert setup.source_dataset_subdir == (
        f"latency-aware/deadly-corridor/deadly_1000ep/deadly_corridor_fix_latency_{latency}_1000ep"
    )


def test_cross_task_setup_supports_deadly_corridor_converter() -> None:
    _, _, robot_type = _task_converter_and_verifier("deadly_corridor")

    assert robot_type == "rl_games_deadly_corridor"


def test_setup_supports_defend_the_line_converter() -> None:
    _, _, robot_type = _task_converter_and_verifier("defend_the_line")

    assert robot_type == "rl_games_defend_the_line"


def test_openvla_deadly_cross_task_setups_compose() -> None:
    for setup_name in OPENVLA_DEADLY_CROSS_TASK_SETUPS:
        cfg = launch_train.compose_training_config(
            config_name="train",
            model="openvla",
            env="cross_task",
            init="bridge",
            mode="cross_task",
            overrides=[f"cross_task_setup={setup_name}"],
        )
        train_tasks = OmegaConf.to_container(cfg.rl_games.cross_task.train_tasks, resolve=True)
        task_names = {task["name"] for task in train_tasks}

        assert "deadly_corridor" in task_names
        assert cfg.framework.name == "QwenOFT"
        assert cfg.framework.action_model.action_dim == 7
        assert cfg.framework.action_model.action_env_dim == 7
        assert cfg.rl_games.cross_task.loss_by_task.deadly_corridor == "multibinary_bce"
        assert cfg.rl_games.env_eval.deadly.action_layout == "multibinary_7"


def test_openvla_three_env_cross_task_setup_uses_024_latencies() -> None:
    cfg = launch_train.compose_training_config(
        config_name="train",
        model="openvla",
        env="cross_task",
        init="bridge",
        mode="cross_task",
        overrides=["cross_task_setup=flappy_demon_deadly_024"],
    )
    train_tasks = OmegaConf.to_container(cfg.rl_games.cross_task.train_tasks, resolve=True)

    assert [task["name"] for task in train_tasks] == ["flappy", "demon_attack", "deadly_corridor"]
    for task in train_tasks:
        assert task["train_latency_filter"] == [0, 2, 4]
        assert task["eval_latency_filter"] == [0, 2, 4]
    assert cfg.rl_games.cross_task.loss_by_task.flappy == "discrete_ce"
    assert cfg.rl_games.cross_task.loss_by_task.demon_attack == "discrete_ce"
    assert cfg.rl_games.cross_task.loss_by_task.deadly_corridor == "multibinary_bce"
    assert cfg.rl_games.env_eval.deadly.multibinary_threshold == 0.0
    for task_name in ("flappy", "demon_attack", "deadly_corridor"):
        post_train = getattr(cfg.rl_games.cross_task.eval_tasks, task_name).post_train
        assert list(post_train.latencies) == [0, 2, 4]


def test_openvla_defendtheline_deadly_zero_setup_matches_competitor_budget() -> None:
    cfg = launch_train.compose_training_config(
        config_name="train",
        model="openvla",
        env="cross_task",
        init="bridge",
        mode="cross_task",
        overrides=["cross_task_setup=defendtheline_zero_deadly_zero"],
    )
    train_tasks = OmegaConf.to_container(cfg.rl_games.cross_task.train_tasks, resolve=True)

    assert [task["name"] for task in train_tasks] == ["defend_the_line", "deadly_corridor"]
    assert train_tasks[0]["train_source_hf"] == "latency-sensitive-bench/benchmark-datasets"
    assert train_tasks[0]["train_source_subdir"] == "zero-latency/defend-the-line/defend_the_line_fixed_latency_0_1000ep_7k2steps"
    assert train_tasks[0]["train_latency_filter"] == [0]
    assert train_tasks[0]["eval_latency_filter"] == [0]
    assert train_tasks[0]["episodes_per_latency"] == 40
    assert train_tasks[1]["train_source_hf"] == "latency-sensitive-bench/benchmark-datasets"
    assert train_tasks[1]["train_latency_filter"] == [0]
    assert train_tasks[1]["eval_latency_filter"] == [0, 2, 4]
    assert train_tasks[1]["episodes_per_latency"] == 1000
    assert cfg.rl_games.cross_task.loss_by_task.defend_the_line == "discrete_ce"
    assert cfg.rl_games.cross_task.loss_by_task.deadly_corridor == "multibinary_bce"
    assert cfg.rl_games.env_eval.deadly.multibinary_threshold == 0.0
    assert list(cfg.rl_games.cross_task.eval_tasks.defend_the_line.post_train.latencies) == [0, 2, 4]
    assert list(cfg.rl_games.cross_task.eval_tasks.deadly_corridor.post_train.latencies) == [0, 2, 4]


def test_default_training_config_disables_held_out_action_classification() -> None:
    cfg = launch_train.compose_training_config(
        config_name="train",
        model="openvla",
        env="flappy",
        init="bridge",
        mode="single",
        overrides=[],
    )

    assert cfg.trainer.eval_action_classification is False


def test_openvla_three_env_cross_task_forwards_deadly_threshold(tmp_path: Path) -> None:
    cfg = launch_train.compose_training_config(
        config_name="train",
        model="openvla",
        env="cross_task",
        init="bridge",
        mode="cross_task",
        overrides=["cross_task_setup=flappy_demon_deadly_024"],
    )
    setup = {
        "dataset_local_dir": str(tmp_path / "datasets"),
        "base_model_dir": str(tmp_path / "base_model"),
        "resume_found": False,
    }

    cmd = launch_train.build_trainer_command(cfg, setup, tmp_path, "results/Checkpoints")

    assert "++rl_games.env_eval.deadly.action_layout=multibinary_7" in cmd
    assert "++rl_games.env_eval.deadly.multibinary_threshold=0.0" in cmd


def test_launcher_does_not_translate_trainer_batch_size_alias() -> None:
    launcher_text = (REPO_ROOT / "examples" / "rl_games" / "scripts" / "launch_train.py").read_text(
        encoding="utf-8"
    )

    assert '"trainer.batch_size"' not in launcher_text


def test_launcher_forwards_canonical_per_device_batch_size_override(tmp_path: Path) -> None:
    cfg = launch_train.compose_training_config(
        config_name="train",
        model="openvla",
        env="flappy",
        init="bridge",
        mode="single",
        overrides=["datasets.vla_data.per_device_batch_size=16"],
    )
    setup = {
        "dataset_local_dir": str(tmp_path / "datasets"),
        "base_model_dir": str(tmp_path / "base_model"),
        "resume_found": False,
    }

    cmd = launch_train.build_trainer_command(cfg, setup, tmp_path, "results/Checkpoints")

    assert "++datasets.vla_data.per_device_batch_size=16" in cmd


def test_launcher_auto_forwards_new_nested_config_fields(tmp_path: Path) -> None:
    cfg = launch_train.compose_training_config(
        config_name="train",
        model="openvla",
        env="flappy",
        init="bridge",
        mode="single",
        overrides=[
            "trainer.optimizer.fused=false",
            "+datasets.vla_data.synthetic_cache.enabled=true",
        ],
    )
    setup = {
        "dataset_local_dir": str(tmp_path / "datasets"),
        "base_model_dir": str(tmp_path / "base_model"),
        "resume_found": False,
    }

    cmd = launch_train.build_trainer_command(cfg, setup, tmp_path, "results/Checkpoints")

    assert "++trainer.optimizer.fused=false" in cmd
    assert "++datasets.vla_data.synthetic_cache.enabled=true" in cmd


def test_launcher_preserves_declared_trainer_field_across_second_composition(tmp_path: Path) -> None:
    cfg = launch_train.compose_training_config(
        config_name="train",
        model="wan_oft",
        env="demon_attack",
        init="wan_oft_libero",
        mode="single",
        overrides=["trainer.learning_rate.action_query_proj=1.0e-4"],
    )
    setup = {
        "dataset_local_dir": str(tmp_path / "datasets"),
        "base_model_dir": str(tmp_path / "wan_base"),
        "resume_found": False,
    }

    cmd = launch_train.build_trainer_command(cfg, setup, tmp_path, "results/Checkpoints")
    forwarded_override = next(item for item in cmd if "action_query_proj" in item)
    recomposed_cfg = launch_train.compose_training_config(
        config_name="train",
        model="wan_oft",
        env="demon_attack",
        init="wan_oft_libero",
        mode="single",
        overrides=[forwarded_override],
    )

    assert forwarded_override == "++trainer.learning_rate.action_query_proj=0.0001"
    assert recomposed_cfg.trainer.learning_rate.action_query_proj == 1.0e-4


def test_launcher_runtime_overrides_are_last(tmp_path: Path) -> None:
    cfg = launch_train.compose_training_config(
        config_name="train",
        model="openvla",
        env="flappy",
        init="bridge",
        mode="single",
        overrides=[
            "datasets.vla_data.data_root_dir=config_root",
            "framework.qwenvl.base_vlm=config_base",
        ],
    )
    setup = {
        "dataset_local_dir": str(tmp_path / "resolved_datasets"),
        "base_model_dir": str(tmp_path / "resolved_base_model"),
        "resume_found": False,
    }

    cmd = launch_train.build_trainer_command(cfg, setup, tmp_path, "results/Checkpoints")

    data_root_overrides = [item for item in cmd if item.endswith("datasets.vla_data.data_root_dir=config_root") or "datasets.vla_data.data_root_dir=" in item]
    base_vlm_overrides = [item for item in cmd if item.endswith("framework.qwenvl.base_vlm=config_base") or "framework.qwenvl.base_vlm=" in item]

    assert data_root_overrides[-1] == f"++datasets.vla_data.data_root_dir={tmp_path / 'resolved_datasets'}"
    assert base_vlm_overrides[-1] == f"++framework.qwenvl.base_vlm={tmp_path / 'resolved_base_model'}"


def test_run_experiment_auto_forwards_canonical_nested_fields(tmp_path: Path) -> None:
    cfg = {
        "config_name": "train",
        "model": "openvla",
        "env": "flappy",
        "init": "bridge",
        "mode": "single",
        "run_id": "run_experiment_test",
        "seed": 42,
        "wandb_entity": "entity",
        "wandb_project": "project",
        "paths": {"dataset_local_dir": "datasets", "base_model_dir": "base_model"},
        "checkpoint": {
            "load": "none",
            "hf_repo_id": None,
            "sync": {"enabled": False, "keep_last_n": 0, "repo_id": None},
            "local": {"keep_last_n": 1},
            "save_best_model": False,
            "save_pt_file": False,
        },
        "trainer": {"optimizer": {"fused": False}},
        "datasets": {"vla_data": {"per_device_batch_size": 8, "synthetic_cache": {"enabled": True}}},
        "framework": {"qwenvl": {"base_vlm": "config_base"}},
        "rl_games": {"env_eval": {"enabled": False, "latency": {"values": [0]}}},
    }
    setup = {
        "dataset_local_dir": str(tmp_path / "resolved_datasets"),
        "base_model_dir": str(tmp_path / "resolved_base_model"),
        "resume_found": False,
    }

    cmd = run_experiment._trainer_command(cfg, setup, tmp_path, "results/Checkpoints")

    assert "trainer.optimizer.fused=false" in cmd
    assert "++datasets.vla_data.synthetic_cache.enabled=true" in cmd
    assert not any(item.startswith("++trainer.") for item in cmd)
    assert "trainer.batch_size=" not in " ".join(cmd)
    assert [item for item in cmd if "datasets.vla_data.data_root_dir=" in item][-1] == (
        f"++datasets.vla_data.data_root_dir={tmp_path / 'resolved_datasets'}"
    )
    assert [item for item in cmd if "framework.qwenvl.base_vlm=" in item][-1] == (
        f"++framework.qwenvl.base_vlm={tmp_path / 'resolved_base_model'}"
    )


def test_legacy_run_experiment_setup_namespace_forwards_latency_unit(tmp_path: Path) -> None:
    cfg = {
        "run_id": "legacy_setup",
        "model": "openvla",
        "env": "demon_attack",
        "mode": "single",
        "paths": {
            "dataset_local_dir": "datasets",
            "base_model_dir": "base_model",
        },
        "dataset": {
            "converted_name": "demon_attack_train",
        },
    }

    setup_args = run_experiment._setup_namespace(cfg, tmp_path, "results/Checkpoints")
    assert setup_args.target_latency_unit == "observation_steps"

    cfg["dataset"]["revision"] = "1234567890abcdef"
    cfg["dataset"]["target_latency_unit"] = "raw_frames"
    setup_args = run_experiment._setup_namespace(cfg, tmp_path, "results/Checkpoints")
    assert setup_args.target_latency_unit == "raw_frames"
    assert setup_args.source_dataset_revision == "1234567890abcdef"


def test_launcher_forwards_vit_and_llm_freeze_overrides(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("WANDB_ENTITY", "test")
    cfg = launch_train.compose_training_config(
        config_name="train",
        model="openvla",
        env="flappy",
        init="bridge",
        mode="single",
        overrides=["trainer.freeze_vit=true", "trainer.freeze_llm_layers=[0,27]"],
    )
    setup = {
        "dataset_local_dir": str(tmp_path / "datasets"),
        "base_model_dir": str(tmp_path / "base_model"),
        "resume_found": False,
    }

    cmd = launch_train.build_trainer_command(cfg, setup, tmp_path, "results/Checkpoints")

    assert "++trainer.freeze_vit=true" in cmd
    assert "++trainer.freeze_llm_layers=[0,27]" in cmd
    assert "trainer.freeze_llm_bottom_ratio=" not in cmd


def test_deadly_corridor_loss_selector_forwards_l1(tmp_path: Path) -> None:
    cfg = launch_train.compose_training_config(
        config_name="train",
        model="openvla",
        env="deadly_corridor",
        init="bridge",
        mode="single",
        overrides=["rl_games.deadly_corridor_loss_type=l1"],
    )
    setup = {
        "dataset_local_dir": str(tmp_path / "datasets"),
        "base_model_dir": str(tmp_path / "base_model"),
        "resume_found": False,
    }

    cmd = launch_train.build_trainer_command(cfg, setup, tmp_path, "results/Checkpoints")

    assert cfg.framework.action_model.loss_type == "l1"
    assert "++framework.action_model.loss_type=l1" in cmd


def test_launcher_setup_namespace_forwards_episodes_per_latency(tmp_path: Path) -> None:
    cfg = launch_train.compose_training_config(
        config_name="train",
        model="openvla",
        env="flappy",
        init="bridge",
        mode="mixed_latency",
        overrides=["dataset.episodes_per_latency=100"],
    )

    setup_args = launch_train.setup_namespace_from_cfg(cfg, tmp_path, "results/Checkpoints")

    assert setup_args.episodes_per_latency == 100


def test_setup_training_assets_exposes_episodes_per_latency_option() -> None:
    setup_text = (REPO_ROOT / "examples" / "rl_games" / "scripts" / "setup_training_assets.py").read_text(
        encoding="utf-8"
    )

    assert "--episodes-per-latency" in setup_text


def test_deadly_corridor_loss_selector_accepts_multibinary_ce_alias(tmp_path: Path) -> None:
    cfg = launch_train.compose_training_config(
        config_name="train",
        model="openvla",
        env="deadly_corridor",
        init="bridge",
        mode="single",
        overrides=["rl_games.deadly_corridor_loss_type=multibinary_ce"],
    )
    setup = {
        "dataset_local_dir": str(tmp_path / "datasets"),
        "base_model_dir": str(tmp_path / "base_model"),
        "resume_found": False,
    }

    cmd = launch_train.build_trainer_command(cfg, setup, tmp_path, "results/Checkpoints")

    assert cfg.framework.action_model.loss_type == "multibinary_bce"
    assert "++framework.action_model.loss_type=multibinary_bce" in cmd


def test_launcher_defaults_to_one_last_checkpoint_and_no_pt_file(tmp_path: Path) -> None:
    cfg = launch_train.compose_training_config(
        config_name="train",
        model="openvla",
        env="flappy",
        init="bridge",
        mode="single",
        overrides=[],
    )
    setup = {
        "dataset_local_dir": str(tmp_path / "datasets"),
        "base_model_dir": str(tmp_path / "base_model"),
        "resume_found": False,
    }

    cmd = launch_train.build_trainer_command(cfg, setup, tmp_path, "results/Checkpoints")

    assert "++checkpoint.local.keep_last_n=1" in cmd
    assert "++checkpoint.save_best_model=true" in cmd
    assert "++checkpoint.save_pt_file=false" in cmd


def test_run_train_exposes_deadly_loss_type_option() -> None:
    run_train_text = (REPO_ROOT / "examples" / "rl_games" / "scripts" / "run_train.sh").read_text(
        encoding="utf-8"
    )

    assert "--deadly-loss-type <l1|multibinary_bce|multibinary_ce>" in run_train_text
    assert "rl_games.deadly_corridor_loss_type=$DEADLY_LOSS_TYPE" in run_train_text


def test_run_train_exposes_distributed_eval_option() -> None:
    run_train_text = (REPO_ROOT / "examples" / "rl_games" / "scripts" / "run_train.sh").read_text(
        encoding="utf-8"
    )

    assert "--eval-distributed-mode <none|rank_sharded>" in run_train_text
    assert "rl_games.env_eval.distributed_mode=$EVAL_DISTRIBUTED_MODE" in run_train_text


def test_launch_train_exposes_distributed_eval_option() -> None:
    launcher_text = (REPO_ROOT / "examples" / "rl_games" / "scripts" / "launch_train.py").read_text(
        encoding="utf-8"
    )

    assert "--eval-distributed-mode" in launcher_text
    assert "rl_games.env_eval.distributed_mode={cli_args.eval_distributed_mode}" in launcher_text


def test_deadly_corridor_openvla_wrapper_passes_extra_args() -> None:
    command_text = _command_path("openvla", "deadly_corridor").read_text(encoding="utf-8")

    assert '"$@"' in command_text


def test_launcher_forwards_cross_task_setup_outputs(tmp_path: Path) -> None:
    cfg = OmegaConf.create({
        "config_name": "train",
        "model": "openvla",
        "env": "cross_task",
        "init": "bridge",
        "mode": "cross_task",
        "run_id": "cross_task_test",
        "seed": 42,
        "wandb_entity": "entity",
        "wandb_project": "project",
        "paths": {"dataset_local_dir": "datasets"},
        "checkpoint": {
            "sync": {"enabled": False, "keep_last_n": 0, "repo_id": ""},
            "local": {"keep_last_n": 3},
            "save_best_model": True,
        },
        "trainer": {"is_resume": False},
        "rl_games": {
            "task": "cross_task",
            "model_alias": "openvla",
            "initialization_mode": "bridge",
            "action_carrier": "bridge",
            "env_eval": {
                "enabled": True,
                "distributed_mode": "rank_sharded",
                "latency": {"mode": "mixed", "values": [0, 1]},
                "mid_train": {"interval_steps": 500},
                "post_train": {"enabled": True},
            },
            "cross_task": {
                "eval_tasks": {
                    "flappy": {
                        "frameskip": 1,
                        "image_size": 224,
                        "mid_train": {
                            "enabled": True,
                            "latencies": [0, 1],
                            "num_episodes": 2,
                            "max_steps_per_episode": 200,
                        },
                    },
                    "demon_attack": {
                        "frameskip": 4,
                        "image_size": 224,
                        "mid_train": {
                            "enabled": True,
                            "latencies": [0],
                            "num_episodes": 3,
                            "max_steps_per_episode": 300,
                        },
                    },
                }
            },
        },
        "datasets": {"vla_data": {"include_state": False}},
    })
    setup = {
        "dataset_local_dir": str(tmp_path / "datasets"),
        "data_mix": "cross__flappy__demon",
        "eval_data_mix": "cross__flappy__demon__val",
        "custom_mixtures_path": str(tmp_path / "datasets" / "_generated_mixtures" / "cross.json"),
        "base_model_dir": str(tmp_path / "base"),
        "resume_found": False,
        "pretrained_checkpoint": str(tmp_path / "init.pt"),
        "cross_task_prompt_maps": {
            "flappy": str(tmp_path / "flappy_prompt_map.json"),
            "demon_attack": str(tmp_path / "demon_prompt_map.json"),
        },
    }

    cmd = launch_train.build_trainer_command(cfg, setup, tmp_path, "results/Checkpoints")

    assert "++datasets.vla_data.data_mix=cross__flappy__demon" in cmd
    assert "++datasets.vla_data.eval_data_mix=cross__flappy__demon__val" in cmd
    assert f"++datasets.vla_data.custom_mixtures_path={setup['custom_mixtures_path']}" in cmd
    assert "++rl_games.env_eval.distributed_mode=rank_sharded" in cmd
    assert f"++rl_games.cross_task.eval_tasks.flappy.prompt_map_path={setup['cross_task_prompt_maps']['flappy']}" in cmd
    assert (
        f"++rl_games.cross_task.eval_tasks.demon_attack.prompt_map_path="
        f"{setup['cross_task_prompt_maps']['demon_attack']}"
    ) in cmd
    assert "++rl_games.cross_task.eval_tasks.flappy.mid_train.latencies=[0,1]" in cmd
    assert "++rl_games.cross_task.eval_tasks.flappy.mid_train.num_episodes=2" in cmd
    assert "++rl_games.cross_task.eval_tasks.demon_attack.mid_train.max_steps_per_episode=300" in cmd


def test_cross_task_mode_does_not_override_initialization_group() -> None:
    mode_text = (REPO_ROOT / "examples" / "rl_games" / "config" / "mode" / "cross_task.yaml").read_text(
        encoding="utf-8"
    )

    assert "initialization_mode:" not in mode_text
    assert "action_carrier:" not in mode_text


def test_launcher_defaults_workspace_to_repo_root_when_env_is_unset(monkeypatch) -> None:
    original_exists = Path.exists

    def fake_exists(path: Path) -> bool:
        if str(path) == "/workspace":
            return True
        return original_exists(path)

    monkeypatch.delenv("WORKSPACE_DIR", raising=False)
    monkeypatch.setattr(Path, "exists", fake_exists)

    cfg = OmegaConf.create({"workspace_dir": "WORKSPACE_DIR"})

    assert launch_train._workspace_dir(cfg) == launch_train.REPO_ROOT


def test_launcher_treats_checkpoint_load_path_as_explicit_checkpoint(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "runs" / "run_1" / "checkpoints" / "steps_400_state"
    cfg = OmegaConf.create(
        {
            "run_id": "run_1",
            "model": "openvla",
            "env": "flappy",
            "mode": "single",
            "rl_games": {"initialization_mode": "bridge", "action_carrier": "bridge"},
            "dataset": {"converted_name": "flappy_train"},
            "paths": {
                "dataset_local_dir": "datasets",
                "base_model_dir": "base_model",
            },
            "base_model": {"repo_id": "Qwen/Qwen3-VL-4B-Instruct"},
            "checkpoint": {
                "load": str(checkpoint_path),
                "hf_repo_id": "",
                "save_best_model": True,
                "sync": {"enabled": False, "repo_id": ""},
            },
            "initialization": {
                "checkpoint_local_dir": "",
                "checkpoint_hf_repo_id": "",
                "checkpoint_filename": "",
            },
        }
    )

    namespace = launch_train.setup_namespace_from_cfg(cfg, tmp_path, str(tmp_path / "runs"))

    assert namespace.checkpoint == str(checkpoint_path)
    assert namespace.checkpoint_load == "none"


def test_vla_trainer_saves_last_checkpoints_independently_from_best_model() -> None:
    trainer_text = (REPO_ROOT / "starVLA" / "training" / "train_starvla.py").read_text(encoding="utf-8")

    assert ") and not self._save_best_model_enabled:" not in trainer_text


def test_vla_trainer_treats_non_positive_save_interval_as_disabled() -> None:
    trainer_text = (REPO_ROOT / "starVLA" / "training" / "train_starvla.py").read_text(encoding="utf-8")
    checkpoint_block = trainer_text.split(
        "if self._save_periodic_checkpoints_enabled() and ",
        maxsplit=1,
    )[1].split("if self.completed_steps >=", maxsplit=1)[0]

    assert "should_run_optional_step_interval_event(" in checkpoint_block


def test_vla_trainer_pt_checkpoint_file_is_optional() -> None:
    trainer_text = (REPO_ROOT / "starVLA" / "training" / "train_starvla.py").read_text(encoding="utf-8")

    assert "self._save_pt_file_enabled" in trainer_text
    assert "safe_serialization=True" in trainer_text
    assert "self.accelerator.get_state_dict(self.model)" in trainer_text
    assert 'model_checkpoint_path = checkpoint_path + "_pytorch_model.pt"' in trainer_text
    assert "torch.save(state_dict, model_checkpoint_path)" in trainer_text
    assert "model_path=model_checkpoint_path" in trainer_text


def test_vla_trainer_can_save_bf16_safetensors_model_checkpoint() -> None:
    trainer_text = (REPO_ROOT / "starVLA" / "training" / "train_starvla.py").read_text(encoding="utf-8")

    assert "self._save_safetensors_file_enabled" in trainer_text
    assert 'model_checkpoint_path = checkpoint_path + "_model.safetensors"' in trainer_text
    assert "torch.bfloat16" in trainer_text
    assert "save_file(safetensors_state_dict, model_checkpoint_path)" in trainer_text


def test_memory_upload_scripts_drop_training_state_before_upload() -> None:
    script_dir = REPO_ROOT.parent / "scripts" / "bash_scripts" / "memory"
    script_paths = sorted(script_dir.glob("*.sh"))

    for script_path in script_paths:
        script_text = script_path.read_text(encoding="utf-8")
        if "hf upload latency-sensitive-bench/memory" not in script_text:
            continue
        assert 'compgen -G "${CHECKPOINT_DIR}/steps_*_model.safetensors"' in script_text, script_path
        assert 'find "${CHECKPOINT_DIR}" -maxdepth 1 -type d -name "steps_*_state" -exec rm -rf {} +' in script_text, script_path
        assert '--exclude "checkpoints/_initialization/**"' in script_text, script_path
        assert '--exclude "checkpoints/*_state/**"' in script_text, script_path


def test_run_train_accepts_explicit_resume_checkpoint() -> None:
    script_text = (REPO_ROOT / "examples" / "rl_games" / "scripts" / "run_train.sh").read_text(encoding="utf-8")

    assert "--checkpoint <path>" in script_text
    assert '--checkpoint) RESUME_CHECKPOINT="$2"; shift 2 ;;' in script_text
    assert '--checkpoint "${RESUME_CHECKPOINT:-}"' in script_text


def test_explicit_best_state_resume_reads_best_metadata(tmp_path: Path) -> None:
    checkpoint_dir = tmp_path / "checkpoints"
    best_state = checkpoint_dir / "best_state"
    best_state.mkdir(parents=True)
    (checkpoint_dir / "best_model_metadata.json").write_text('{"best_step": 2500}', encoding="utf-8")

    checkpoint, step, kind = _resolve_explicit_resume_checkpoint(str(best_state), checkpoint_dir)

    assert checkpoint == best_state.resolve()
    assert step == 2500
    assert kind == "state"


def test_explicit_resume_checkpoint_matches_eval_relative_checkpoint_resolution(tmp_path: Path) -> None:
    checkpoint_dir = tmp_path / "run" / "checkpoints"
    state_dir = checkpoint_dir / "steps_300_state"
    state_dir.mkdir(parents=True)
    (state_dir / "model.safetensors").write_text("weights", encoding="utf-8")

    checkpoint, step, kind = _resolve_explicit_resume_checkpoint("steps_300_state", checkpoint_dir)

    assert checkpoint == state_dir.resolve()
    assert step == 300
    assert kind == "state"
