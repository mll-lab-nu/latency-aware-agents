cd starVLA

bash examples/rl_games/install/install_stack.sh openvla flappy

conda activate starvla_rl_games_openvla

python examples/rl_games/scripts/launch_train.py \
    model=openvla \
    env=flappy \
    init=bridge \
    mode=single \
    run_id="openvla_bridge_flappy_single_latency_exp1_30ep_full_coverage_sequential" \
    trainer.distributed_backend=none \
    workspace_dir="/workspace" \
    wandb_entity="talha1503" \
    checkpoint.hf_repo_id="talha15032/openvla_bridge_flappy_single_latency_exp1_30ep_full_coverage_sequential" \
    checkpoint.sync.enabled=true \
    checkpoint.sync.repo_id="talha15032/openvla_bridge_flappy_single_latency_exp1_30ep_full_coverage_sequential" \
    dataset.source_hf="talha1503/flappy_bird_zero_latency_parquet" \
    checkpoint.save_best_model=true \
    dataset.max_episodes=30 \
    datasets.vla_data.sequential_step_sampling=true \
    datasets.vla_data.shuffle=false \
    trainer.max_train_steps=3000 \
    trainer.num_warmup_steps=0 \
    trainer.eval_interval=100 \
    trainer.logging_frequency=1 \
    trainer.gradient_accumulation_steps=16 \
    datasets.vla_data.per_device_batch_size=32 \
    rl_games.env_eval.mid_train.interval_steps=100 \
    rl_games.env_eval.mid_train.max_steps_per_episode=3600 \
    rl_games.env_eval.mid_train.num_episodes=10 \
    rl_games.env_eval.post_train.enabled=true \
    rl_games.env_eval.post_train.latencies=[0,1,2,3,4] \
    rl_games.env_eval.post_train.num_episodes=10 \
    rl_games.env_eval.post_train.max_steps_per_episode=3600
