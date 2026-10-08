export WORKSPACE_DIR

bash "${WORKSPACE_DIR}/starVLA/examples/rl_games/bash_scripts/install/pre_launch.sh"

cd "${WORKSPACE_DIR}/starVLA"

bash examples/rl_games/install/install_stack.sh openvla demon_attack

conda activate starvla_rl_games_openvla

bash "${WORKSPACE_DIR}/starVLA/examples/rl_games/bash_scripts/install/latency_deps.sh"

export PYTHONPATH="${WORKSPACE_DIR}/latency-sensitive-bench:${PYTHONPATH:-}"

python examples/rl_games/scripts/launch_train.py \
    model=openvla \
    env=demon_attack \
    init=bridge \
    mode=mixed \
    run_id="openvla_bridge_demon_attack_mixed_latency_6_ghost_trail_memory_exp2_30" \
    trainer.distributed_backend=none \
    workspace_dir="$WORKSPACE_DIR" \
    wandb_entity="talha1503" \
    checkpoint.hf_repo_id="talha15032/openvla_bridge_demon_attack_mixed_latency_6_ghost_trail_memory_exp2_30" \
    checkpoint.sync.enabled=true \
    checkpoint.sync.repo_id="talha15032/openvla_bridge_demon_attack_mixed_latency_6_ghost_trail_memory_exp2_30" \
    dataset.source_hf="latency-sensitive-bench/benchmark-datasets" \
    dataset.source_subdir=experiments/visual-history/archive/demon-attack/demon_attack_mixed_0246_ghost30_200ep \
    dataset.revision=c65ee4c303358dc448022bed9bc25e78984449a8 \
    dataset.latency_filter=[0,2,4,6] \
    datasets.vla_data.sequential_step_sampling=true \
    datasets.vla_data.shuffle=true \
    checkpoint.local.keep_last_n=1 \
    trainer.per_latency_eval_num_batches=5 \
    dataset.episodes_per_latency=40 \
    checkpoint.save_best_model=false \
    trainer.max_train_steps=5000 \
    trainer.save_interval=5000 \
    trainer.num_warmup_steps=0 \
    trainer.eval_interval=1000 \
    trainer.logging_frequency=1 \
    trainer.gradient_accumulation_steps=4 \
    datasets.vla_data.per_device_batch_size=32 \
    rl_games.env_eval.eval_backend=eval_core \
    rl_games.env_eval.image_transform=demon_attack_ghost_trail \
    rl_games.env_eval.ghost_trail.history_frames=30 \
    rl_games.env_eval.mid_train.enabled=false \
    rl_games.env_eval.post_train.enabled=true \
    rl_games.env_eval.post_train.latencies=[0,2,4,6] \
    rl_games.env_eval.post_train.num_episodes=20 \
    rl_games.env_eval.post_train.max_steps_per_episode=3600
