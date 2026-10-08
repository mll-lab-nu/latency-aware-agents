WORKSPACE_DIR="${WORKSPACE_DIR:-/workspace}"
export WORKSPACE_DIR

bash "${WORKSPACE_DIR}/starVLA/examples/rl_games/bash_scripts/install/pre_launch.sh"

cd "${WORKSPACE_DIR}/starVLA"

conda activate starvla_rl_games_gr00t

bash "${WORKSPACE_DIR}/starVLA/examples/rl_games/bash_scripts/install/latency_deps.sh"

export PYTHONPATH="${WORKSPACE_DIR}/latency-sensitive-bench:${PYTHONPATH:-}"

python examples/rl_games/scripts/launch_train.py \
    model=gr00t \
    env=demon_attack \
    init=bridge \
    mode=single \
    run_id="gr00t_bridge_demon_attack_single_latency_clean_data_exp3_action_1e-5_backbone_1e-6" \
    trainer.distributed_backend=none \
    workspace_dir="$WORKSPACE_DIR" \
    wandb_entity="talha1503" \
    checkpoint.hf_repo_id="talha15032/gr00t_bridge_demon_attack_single_latency_clean_data_exp3_action_1e-5_backbone_1e-6" \
    checkpoint.sync.enabled=true \
    checkpoint.sync.repo_id="talha15032/gr00t_bridge_demon_attack_single_latency_clean_data_exp3_action_1e-5_backbone_1e-6" \
    dataset.source_hf="latency-sensitive-bench/benchmark-datasets" \
    dataset.revision=15d17287246bd19940c837d6c8a5327564f969c5 \
    dataset.source_subdir=zero-latency/demon-attack/demon_attack_200ep/demon_attack_fix_latency_0_200ep \
    checkpoint.local.keep_last_n=1 \
    checkpoint.save_best_model=false \
    trainer.max_train_steps=5000 \
    trainer.save_interval=5000 \
    trainer.num_warmup_steps=0 \
    trainer.eval_interval=300 \
    trainer.logging_frequency=1 \
    trainer.gradient_accumulation_steps=4 \
    datasets.vla_data.per_device_batch_size=32 \
    trainer.learning_rate.base=1e-06 \
    trainer.learning_rate.qwen_vl_interface=1e-06 \
    trainer.learning_rate.action_model=1e-05 \
    rl_games.env_eval.mid_train.enabled=true \
    rl_games.env_eval.mid_train.latencies=[0] \
    rl_games.env_eval.eval_backend=eval_core \
    rl_games.env_eval.mid_train.interval_steps=300 \
    rl_games.env_eval.post_train.num_episodes=4 \
    rl_games.env_eval.post_train.max_steps_per_episode=3600 \
    rl_games.env_eval.post_train.enabled=true \
    rl_games.env_eval.post_train.latencies=[0,2,4,6,8] \
    rl_games.env_eval.post_train.num_episodes=20 \
    rl_games.env_eval.post_train.max_steps_per_episode=3600
