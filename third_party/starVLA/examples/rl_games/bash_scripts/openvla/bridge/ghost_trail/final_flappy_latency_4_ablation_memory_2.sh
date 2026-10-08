export WORKSPACE_DIR

bash "${WORKSPACE_DIR}/starVLA/examples/rl_games/bash_scripts/install/pre_launch.sh"

cd "${WORKSPACE_DIR}/starVLA"

bash examples/rl_games/install/install_stack.sh openvla flappy

conda activate starvla_rl_games_openvla

bash "${WORKSPACE_DIR}/starVLA/examples/rl_games/bash_scripts/install/latency_deps.sh"

export PYTHONPATH="${WORKSPACE_DIR}/latency-sensitive-bench:${PYTHONPATH:-}"

python examples/rl_games/scripts/launch_train.py \
    model=openvla \
    env=flappy \
    init=bridge \
    mode=single \
    run_id="openvla_bridge_flappy_fixed_latency_3_200ep_7k2steps_ghost7_exp3" \
    trainer.distributed_backend=deepspeed \
    launch.use_accelerate=true \
    launch.num_processes=2 \
    paths.accelerate_config=starVLA/config/deepseeds/deepspeed_zero2.yaml \
    workspace_dir="$WORKSPACE_DIR" \
    wandb_entity="talha1503" \
    checkpoint.hf_repo_id="talha15032/openvla_bridge_flappy_fixed_latency_3_200ep_7k2steps_ghost7_exp3" \
    checkpoint.sync.enabled=true \
    checkpoint.sync.repo_id="talha15032/openvla_bridge_flappy_fixed_latency_3_200ep_7k2steps_ghost7_exp3" \
    dataset.source_hf="latency-sensitive-bench/benchmark-datasets" \
    ++dataset.revision="2457e6f0c1ba0f0874e5fe2dec557bdc4c6fe5e4" \
    dataset.source_subdir="latency-aware/flappy/flappy_fixed_latency_3_200ep_7k2steps_ghost7" \
    dataset.latency_filter=[3] \
    datasets.vla_data.sequential_step_sampling=true \
    datasets.vla_data.shuffle=true \
    trainer.per_latency_eval_num_batches=5 \
    dataset.episodes_per_latency=200 \
    rl_games.env_eval.image_transform=flappy_ghost_trail \
    rl_games.env_eval.post_train.ghost_trail.history_frames=7 \
    rl_games.env_eval.eval_backend=latency_bench \
    datasets.vla_data.shuffle=true \
    checkpoint.local.keep_last_n=1 \
    trainer.max_train_steps=4000 \
    trainer.num_warmup_steps=0 \
    checkpoint.save_final_model=true \
    checkpoint.save_best_model=false \
    trainer.eval_interval=4000 \
    trainer.logging_frequency=1 \
    trainer.gradient_accumulation_steps=4 \
    rl_games.env_eval.distributed_mode=rank_sharded \
    datasets.vla_data.per_device_batch_size=32 \
    rl_games.env_eval.mid_train.enabled=false \
    rl_games.env_eval.post_train.enabled=true \
    rl_games.env_eval.post_train.latencies=[3] \
    rl_games.env_eval.post_train.num_episodes=100 \
    rl_games.env_eval.post_train.max_steps_per_episode=3600