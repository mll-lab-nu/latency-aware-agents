cd starVLA

bash examples/rl_games/install/install_stack.sh openvla deadly_corridor

conda activate starvla_rl_games_openvla

bash examples/rl_games/install/flash_attn.sh --check >/dev/null 2>&1 || bash examples/rl_games/install/flash_attn.sh

python examples/rl_games/scripts/launch_train.py \
    model=openvla \
    env=deadly_corridor \
    init=bridge \
    mode=single \
    run_id="openvla_bridge_deadly_corridor_single_latency_exp2_l1" \
    trainer.distributed_backend=none \
    workspace_dir="/workspace" \
    wandb_entity="talha1503" \
    checkpoint.hf_repo_id="talha1503/openvla_bridge_deadly_corridor_single_latency_exp2_l1" \
    checkpoint.sync.enabled=true \
    checkpoint.sync.repo_id="talha1503/openvla_bridge_deadly_corridor_single_latency_exp2_l1" \
    dataset.source_hf=talha1503/deadly_corridor_zero_latency_parquet \
    checkpoint.save_best_model=true \
    trainer.max_train_steps=800 \
    trainer.num_warmup_steps=0 \
    trainer.eval_interval=50 \
    rl_games.deadly_corridor_loss_type=l1 \
    trainer.logging_frequency=1 \
    trainer.gradient_accumulation_steps=16 \
    datasets.vla_data.per_device_batch_size=32 \
    rl_games.env_eval.mid_train.interval_steps=50 \
    rl_games.env_eval.mid_train.max_steps_per_episode=3600 \
    rl_games.env_eval.mid_train.num_episodes=5 \
    rl_games.env_eval.post_train.enabled=true \
    rl_games.env_eval.post_train.latencies=[0,1,2,3,4,5,6,7,8,9,10] \
    rl_games.env_eval.post_train.num_episodes=20 \
    rl_games.env_eval.post_train.max_steps_per_episode=3600 \
    rl_games.env_eval.vectorized.enabled=true \
    rl_games.env_eval.vectorized.batch_size=2 \