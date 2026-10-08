cd starVLA

bash examples/rl_games/install/install_stack.sh openvla flappy

conda activate starvla_rl_games_openvla

python examples/rl_games/scripts/launch_train.py \
    model=openvla \
    env=flappy \
    init=bridge \
    mode=single \
    run_id="openvla_bridge_flappy_single_latency_exp1" \
    trainer.distributed_backend=none \
    workspace_dir="/workspace" \
    wandb_entity="talha1503" \
    checkpoint.hf_repo_id="talha1503/openvla_bridge_flappy_single_latency_exp1" \
    checkpoint.sync.enabled=true \
    checkpoint.sync.repo_id="talha1503/openvla_bridge_flappy_single_latency_exp1" \
    dataset.source_hf="talha1503/flappy_bird_zero_latency_parquet" \
    checkpoint.local.keep_last_n=2 \
    trainer.max_train_steps=5000 \
    trainer.num_warmup_steps=0 \
    trainer.save_interval=10 \
    trainer.eval_interval=500 \
    trainer.logging_frequency=1 \
    trainer.gradient_accumulation_steps=16 \
    datasets.vla_data.per_device_batch_size=16 \
    rl_games.env_eval.mid_train.interval_steps=500 \
    rl_games.env_eval.mid_train.max_steps_per_episode=3600 \
    rl_games.env_eval.mid_train.num_episodes=20 \
    rl_games.env_eval.post_train.enabled=true \
    rl_games.env_eval.post_train.latencies=[0,1,2,3,4] \
    rl_games.env_eval.post_train.num_episodes=20 \
    rl_games.env_eval.post_train.max_steps_per_episode=3600
