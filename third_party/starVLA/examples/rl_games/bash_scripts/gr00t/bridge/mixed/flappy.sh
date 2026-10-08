cd starVLA

bash examples/rl_games/install/install_stack.sh gr00t flappy

conda activate starvla_rl_games_gr00t

bash examples/rl_games/install/flash_attn.sh --check >/dev/null 2>&1 || bash examples/rl_games/install/flash_attn.sh

python examples/rl_games/scripts/launch_train.py \
    model=gr00t \
    env=flappy \
    init=bridge \
    mode=mixed_latency \
    run_id="gr00t_bridge_flappy_mixed_latency_mini_exp1" \
    trainer.distributed_backend=none \
    workspace_dir="/workspace" \
    wandb_entity="talha1503" \
    checkpoint.hf_repo_id="talha15032/gr00t_bridge_flappy_mixed_latency_mini_exp1" \
    checkpoint.sync.enabled=true \
    checkpoint.sync.repo_id="talha15032/gr00t_bridge_flappy_mixed_latency_mini_exp1" \
    dataset.source_hf="talha1503/flappy_bird_mixed_latency_parquet" \
    checkpoint.save_best_model=true \
    trainer.max_train_steps=2100 \
    trainer.num_warmup_steps=0 \
    trainer.eval_interval=300 \
    trainer.logging_frequency=1 \
    trainer.gradient_accumulation_steps=32 \
    datasets.vla_data.per_device_batch_size=32 \
    dataset.latency_filter=[0,1,2] \
    dataset.episodes_per_latency=10 \
    rl_games.env_eval.mid_train.interval_steps=300 \
    rl_games.env_eval.mid_train.max_steps_per_episode=3600 \
    rl_games.env_eval.mid_train.latencies=[0,1,2] \
    rl_games.env_eval.mid_train.num_episodes=3 \
    rl_games.env_eval.post_train.enabled=true \
    rl_games.env_eval.post_train.latencies=[0,1,2] \
    rl_games.env_eval.post_train.num_episodes=20 \
    rl_games.env_eval.post_train.max_steps_per_episode=3600 \
    rl_games.env_eval.vectorized.enabled=true \
    rl_games.env_eval.vectorized.batch_size=3
