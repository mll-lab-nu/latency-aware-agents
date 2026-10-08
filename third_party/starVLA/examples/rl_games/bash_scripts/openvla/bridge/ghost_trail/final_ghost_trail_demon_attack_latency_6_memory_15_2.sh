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
    mode=single \
    run_id="openvla_bridge_demon_attack_fixed_latency_6_200ep_7k2steps_ghost15_2_exp3" \
    trainer.distributed_backend=deepspeed \
    launch.use_accelerate=true \
    launch.num_processes=2 \
    paths.accelerate_config=starVLA/config/deepseeds/deepspeed_zero2.yaml \
    workspace_dir="$WORKSPACE_DIR" \
    wandb_entity="talha1503" \
    checkpoint.hf_repo_id="talha15032/openvla_bridge_demon_attack_fixed_latency_6_200ep_7k2steps_ghost15_2_exp3" \
    checkpoint.sync.enabled=true \
    checkpoint.sync.repo_id="talha15032/openvla_bridge_demon_attack_fixed_latency_6_200ep_7k2steps_ghost15_2_exp3" \
    dataset.source_hf="latency-sensitive-bench/benchmark-datasets" \
    ++dataset.revision="2457e6f0c1ba0f0874e5fe2dec557bdc4c6fe5e4" \
    dataset.source_subdir="latency-aware/demon-attack/demon_attack_fixed_latency_6_200ep_7k2steps_ghost15_2" \
    dataset.latency_filter=[6] \
    datasets.vla_data.sequential_step_sampling=true \
    datasets.vla_data.shuffle=true \
    checkpoint.local.keep_last_n=1 \
    trainer.per_latency_eval_num_batches=5 \
    dataset.episodes_per_latency=200 \
    checkpoint.save_best_model=false \
    trainer.max_train_steps=4000 \
    trainer.save_interval=4000 \
    trainer.num_warmup_steps=0 \
    trainer.eval_interval=1000 \
    trainer.logging_frequency=1 \
    trainer.gradient_accumulation_steps=4 \
    datasets.vla_data.per_device_batch_size=32 \
    rl_games.env_eval.eval_backend=latency_bench \
    rl_games.env_eval.distributed_mode=rank_sharded \
    rl_games.env_eval.image_transform=demon_attack_ghost_trail \
    rl_games.env_eval.ghost_trail.history_frames=15 \
    rl_games.env_eval.mid_train.enabled=true \
    rl_games.env_eval.mid_train.interval_steps=1000 \
    rl_games.env_eval.mid_train.latencies=[6] \
    rl_games.env_eval.mid_train.num_episodes=5 \
    rl_games.env_eval.mid_train.max_steps_per_episode=14400 \
    rl_games.env_eval.post_train.enabled=true \
    rl_games.env_eval.post_train.latencies=[6] \
    rl_games.env_eval.post_train.num_episodes=100 \
    rl_games.env_eval.post_train.max_steps_per_episode=14400