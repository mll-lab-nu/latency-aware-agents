#!/usr/bin/env bash
set -euo pipefail
export HF_ENDPOINT=https://huggingface.co

WORKSPACE_DIR="${WORKSPACE_DIR:-/home/ubuntu/talha}"
DATA_WORKSPACE_DIR="${DATA_WORKSPACE_DIR:-/mnt/local/talha}"
export WORKSPACE_DIR DATA_WORKSPACE_DIR

bash "${WORKSPACE_DIR}/starVLA/examples/rl_games/bash_scripts/install/pre_launch.sh"

cd "${WORKSPACE_DIR}/starVLA"

bash examples/rl_games/install/install_stack.sh openvla cross_task

CONDA_BASE="$(conda info --base)"
# shellcheck source=/dev/null
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate starvla_rl_games_openvla

bash "${WORKSPACE_DIR}/starVLA/examples/rl_games/bash_scripts/install/latency_deps.sh"

mkdir -p "${DATA_WORKSPACE_DIR}"
mkdir -p "${DATA_WORKSPACE_DIR}/.cache/huggingface"
export HF_HOME="${DATA_WORKSPACE_DIR}/.cache/huggingface"
export HUGGINGFACE_HUB_CACHE="${DATA_WORKSPACE_DIR}/.cache/huggingface/hub"
export HF_HUB_CACHE="${DATA_WORKSPACE_DIR}/.cache/huggingface/hub"
export TRANSFORMERS_CACHE="${DATA_WORKSPACE_DIR}/.cache/huggingface/transformers"
export PYTHONPATH="${WORKSPACE_DIR}/latency-sensitive-bench:${PYTHONPATH:-}"

RUN_ID="${RUN_ID:-openvla_bridge_cross_mixed_all_024_5env200ep_deadly1000ep_exp1}"
MAX_EPISODES_PER_ENV="${MAX_EPISODES_PER_ENV:-200}"
MAX_EPISODES_DEADLY="${MAX_EPISODES_DEADLY:-1000}"
MIX_LATENCIES="${MIX_LATENCIES:-[0,2,4]}"

python examples/rl_games/scripts/launch_train.py \
    model=openvla \
    env=cross_task \
    init=bridge \
    mode=cross_task \
    cross_task_setup=flappy_demon_deadly_asterix_atlantis_defend_024 \
    run_id="${RUN_ID}" \
    trainer.distributed_backend=none \
    workspace_dir="${DATA_WORKSPACE_DIR}" \
    paths.run_root_dir="${DATA_WORKSPACE_DIR}/results/Checkpoints" \
    paths.dataset_local_dir="${DATA_WORKSPACE_DIR}/playground/Datasets/rl_games" \
    paths.base_model_dir="${DATA_WORKSPACE_DIR}/playground/Pretrained_models/Qwen3-VL-4B-Instruct" \
    wandb_entity="talha1503" \
    checkpoint.hf_repo_id="latency-sensitive-bench/benchmark-models" \
    checkpoint.sync.enabled=false \
    checkpoint.sync.repo_id="latency-sensitive-bench/benchmark-models" \
    checkpoint.save_best_model=false \
    checkpoint.local.keep_last_n=1 \
    trainer.max_train_steps=15000 \
    datasets.vla_data.sequential_step_sampling=true \
    datasets.vla_data.shuffle=true \
    trainer.num_warmup_steps=0 \
    trainer.eval_interval=15000 \
    trainer.save_interval=15000 \
    trainer.logging_frequency=1 \
    trainer.gradient_accumulation_steps=8 \
    trainer.per_latency_eval_num_batches=5 \
    ++wandb_tags='["final_vla_profiling_training","demon_attack","openvla","rtx3090_profile"]' \
    ++training_latency_condition=rtx3090_profile \
    ++wandb_group="final_vla_profiling_training" \
    rl_games.env_eval.mid_train.enabled=false \
    rl_games.env_eval.post_train.enabled=true \
    rl_games.env_eval.post_train.latencies="${MIX_LATENCIES}" \
    trainer.eval_action_classification=true \
    datasets.vla_data.per_device_batch_size=32 \
    datasets.vla_data.include_state=false \
    datasets.vla_data.sequential_step_sampling=true \
    datasets.vla_data.shuffle=true \
    framework.action_model.loss_type=discrete_ce \
    framework.action_model.action_dim=7 \
    framework.action_model.action_env_dim=7 \
    framework.action_model.action_layout=bridge_cross_task_7 \
    rl_games.cross_task.train_tasks.0.name=flappy \
    rl_games.cross_task.train_tasks.0.converted_name=flappy_mixed_024_200ep_cross_train \
    rl_games.cross_task.train_tasks.0.train_source_hf=latency-sensitive-bench/benchmark-datasets \
    rl_games.cross_task.train_tasks.0.train_revision=d6f0ad0995f6e4b095f315920e8de18e04297048 \
    rl_games.cross_task.train_tasks.0.prompt_source_hf=latency-sensitive-bench/benchmark-datasets \
    rl_games.cross_task.train_tasks.0.prompt_revision=d6f0ad0995f6e4b095f315920e8de18e04297048 \
    rl_games.cross_task.train_tasks.0.train_source_subdir=zero-latency/flappy/flappy_200ep/flappy_fix_latency_0_200ep \
    rl_games.cross_task.train_tasks.0.prompt_source_subdir=zero-latency/flappy/flappy_200ep/flappy_fix_latency_0_200ep \
    rl_games.cross_task.train_tasks.0.train_latency_filter="${MIX_LATENCIES}" \
    rl_games.cross_task.train_tasks.0.eval_latency_filter="${MIX_LATENCIES}" \
    rl_games.cross_task.train_tasks.0.episodes_per_latency=null \
    rl_games.cross_task.train_tasks.0.max_episodes="${MAX_EPISODES_PER_ENV}" \
    rl_games.cross_task.train_tasks.1.name=demon_attack \
    rl_games.cross_task.train_tasks.1.converted_name=demon_attack_mixed_024_200ep_cross_train \
    rl_games.cross_task.train_tasks.1.train_source_hf=latency-sensitive-bench/benchmark-datasets \
    rl_games.cross_task.train_tasks.1.train_revision=d6f0ad0995f6e4b095f315920e8de18e04297048 \
    rl_games.cross_task.train_tasks.1.prompt_source_hf=latency-sensitive-bench/benchmark-datasets \
    rl_games.cross_task.train_tasks.1.prompt_revision=d6f0ad0995f6e4b095f315920e8de18e04297048 \
    rl_games.cross_task.train_tasks.1.train_source_subdir=zero-latency/demon-attack/demon_attack_200ep/demon_attack_fix_latency_0_200ep \
    rl_games.cross_task.train_tasks.1.prompt_source_subdir=zero-latency/demon-attack/demon_attack_200ep/demon_attack_fix_latency_0_200ep \
    rl_games.cross_task.train_tasks.1.train_latency_filter="${MIX_LATENCIES}" \
    rl_games.cross_task.train_tasks.1.eval_latency_filter="${MIX_LATENCIES}" \
    rl_games.cross_task.train_tasks.1.episodes_per_latency=null \
    rl_games.cross_task.train_tasks.1.max_episodes="${MAX_EPISODES_PER_ENV}" \
    rl_games.cross_task.train_tasks.2.name=deadly_corridor \
    rl_games.cross_task.train_tasks.2.converted_name=deadly_corridor_mixed_024_1000ep_cross_train \
    rl_games.cross_task.train_tasks.2.prompt_source_hf=latency-sensitive-bench/benchmark-datasets \
    rl_games.cross_task.train_tasks.2.prompt_revision=d6f0ad0995f6e4b095f315920e8de18e04297048 \
    rl_games.cross_task.train_tasks.2.prompt_source_subdir=zero-latency/deadly-corridor/deadly_1000ep/deadly_corridor_fix_latency_0_1000ep \
    rl_games.cross_task.train_tasks.2.train_source_hf=latency-sensitive-bench/benchmark-datasets \
    rl_games.cross_task.train_tasks.2.train_revision=d6f0ad0995f6e4b095f315920e8de18e04297048 \
    rl_games.cross_task.train_tasks.2.train_source_subdir=zero-latency/deadly-corridor/deadly_1000ep/deadly_corridor_fix_latency_0_1000ep \
    rl_games.cross_task.train_tasks.2.train_latency_filter="${MIX_LATENCIES}" \
    rl_games.cross_task.train_tasks.2.eval_latency_filter="${MIX_LATENCIES}" \
    rl_games.cross_task.train_tasks.2.episodes_per_latency=null \
    rl_games.cross_task.train_tasks.2.max_episodes="${MAX_EPISODES_DEADLY}" \
    rl_games.cross_task.train_tasks.2.action_layout=multibinary_7 \
    rl_games.cross_task.train_tasks.3.name=asterix \
    rl_games.cross_task.train_tasks.3.converted_name=asterix_mixed_024_200ep_cross_train \
    rl_games.cross_task.train_tasks.3.prompt_source_hf=latency-sensitive-bench/benchmark-datasets \
    rl_games.cross_task.train_tasks.3.prompt_revision=d6f0ad0995f6e4b095f315920e8de18e04297048 \
    rl_games.cross_task.train_tasks.3.prompt_source_subdir=zero-latency/asterix/asterix_200ep/asterix_fixed_latency_0_200ep_7k2steps \
    rl_games.cross_task.train_tasks.3.train_source_hf=latency-sensitive-bench/benchmark-datasets \
    rl_games.cross_task.train_tasks.3.train_revision=d6f0ad0995f6e4b095f315920e8de18e04297048 \
    rl_games.cross_task.train_tasks.3.train_source_subdir=zero-latency/asterix/asterix_200ep/asterix_fixed_latency_0_200ep_7k2steps \
    rl_games.cross_task.train_tasks.3.train_latency_filter="${MIX_LATENCIES}" \
    rl_games.cross_task.train_tasks.3.eval_latency_filter="${MIX_LATENCIES}" \
    rl_games.cross_task.train_tasks.3.episodes_per_latency=null \
    rl_games.cross_task.train_tasks.3.max_episodes="${MAX_EPISODES_PER_ENV}" \
    rl_games.cross_task.train_tasks.3.action_layout=factorized_6 \
    rl_games.cross_task.train_tasks.4.name=atlantis \
    rl_games.cross_task.train_tasks.4.converted_name=atlantis_mixed_024_200ep_cross_train \
    rl_games.cross_task.train_tasks.4.prompt_source_hf=latency-sensitive-bench/benchmark-datasets \
    rl_games.cross_task.train_tasks.4.prompt_revision=d6f0ad0995f6e4b095f315920e8de18e04297048 \
    rl_games.cross_task.train_tasks.4.prompt_source_subdir=zero-latency/atlantis/atlantis_200ep/atlantis_fixed_latency_0_200ep_7k2steps \
    rl_games.cross_task.train_tasks.4.train_source_hf=latency-sensitive-bench/benchmark-datasets \
    rl_games.cross_task.train_tasks.4.train_revision=d6f0ad0995f6e4b095f315920e8de18e04297048 \
    rl_games.cross_task.train_tasks.4.train_source_subdir=zero-latency/atlantis/atlantis_200ep/atlantis_fixed_latency_0_200ep_7k2steps \
    rl_games.cross_task.train_tasks.4.train_latency_filter="${MIX_LATENCIES}" \
    rl_games.cross_task.train_tasks.4.eval_latency_filter="${MIX_LATENCIES}" \
    rl_games.cross_task.train_tasks.4.episodes_per_latency=null \
    rl_games.cross_task.train_tasks.4.max_episodes="${MAX_EPISODES_PER_ENV}" \
    rl_games.cross_task.train_tasks.5.name=defend_the_line \
    rl_games.cross_task.train_tasks.5.converted_name=defend_the_line_mixed_024_200ep_cross_train \
    rl_games.cross_task.train_tasks.5.prompt_source_hf=latency-sensitive-bench/benchmark-datasets \
    rl_games.cross_task.train_tasks.5.prompt_revision=d6f0ad0995f6e4b095f315920e8de18e04297048 \
    rl_games.cross_task.train_tasks.5.prompt_source_subdir=zero-latency/defend-the-line/defend_the_line_200ep/defend_the_line_fixed_latency_0_200ep_7k2steps \
    rl_games.cross_task.train_tasks.5.train_source_hf=latency-sensitive-bench/benchmark-datasets \
    rl_games.cross_task.train_tasks.5.train_revision=d6f0ad0995f6e4b095f315920e8de18e04297048 \
    rl_games.cross_task.train_tasks.5.train_source_subdir=zero-latency/defend-the-line/defend_the_line_200ep/defend_the_line_fixed_latency_0_200ep_7k2steps \
    rl_games.cross_task.train_tasks.5.train_latency_filter="${MIX_LATENCIES}" \
    rl_games.cross_task.train_tasks.5.eval_latency_filter="${MIX_LATENCIES}" \
    rl_games.cross_task.train_tasks.5.episodes_per_latency=null \
    rl_games.cross_task.train_tasks.5.max_episodes="${MAX_EPISODES_PER_ENV}" \
    rl_games.cross_task.eval_tasks.flappy.mid_train.enabled=false \
    rl_games.cross_task.eval_tasks.flappy.post_train.enabled=true \
    rl_games.cross_task.eval_tasks.flappy.post_train.latencies="${MIX_LATENCIES}" \
    rl_games.cross_task.eval_tasks.demon_attack.mid_train.enabled=false \
    rl_games.cross_task.eval_tasks.demon_attack.post_train.enabled=true \
    rl_games.cross_task.eval_tasks.demon_attack.post_train.latencies="${MIX_LATENCIES}" \
    rl_games.cross_task.eval_tasks.deadly_corridor.mid_train.enabled=false \
    rl_games.cross_task.eval_tasks.deadly_corridor.post_train.enabled=true \
    rl_games.cross_task.eval_tasks.deadly_corridor.post_train.latencies="${MIX_LATENCIES}" \
    rl_games.cross_task.eval_tasks.asterix.mid_train.enabled=false \
    rl_games.cross_task.eval_tasks.asterix.post_train.enabled=false \
    rl_games.cross_task.eval_tasks.atlantis.mid_train.enabled=false \
    rl_games.cross_task.eval_tasks.atlantis.post_train.enabled=false \
    rl_games.cross_task.eval_tasks.defend_the_line.mid_train.enabled=false \
    rl_games.cross_task.eval_tasks.defend_the_line.post_train.enabled=false \
    rl_games.env_eval.eval_backend=eval_core \
    rl_games.env_eval.deadly.action_layout=multibinary_7 \
    rl_games.env_eval.deadly.multibinary_threshold=0.0
