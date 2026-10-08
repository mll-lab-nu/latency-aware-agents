cd starVLA

bash examples/rl_games/install/install_stack.sh pi05 deadly_corridor

conda activate starvla_rl_games_pi05

bash examples/rl_games/install/flash_attn.sh --check >/dev/null 2>&1 || bash examples/rl_games/install/flash_attn.sh

python examples/rl_games/scripts/launch_train.py \
    model=pi05 \
    env=deadly_corridor \
    init=bridge \
    mode=single \
    run_id="pi05_bridge_deadly_corridor_single_latency_clean_data_bce_exp2" \
    trainer.distributed_backend=none \
    workspace_dir="/workspace" \
    wandb_entity="talha1503" \
    checkpoint.hf_repo_id="talha15032/pi05_bridge_deadly_corridor_single_latency_clean_data_exp2" \
    checkpoint.sync.enabled=true \
    checkpoint.sync.repo_id="talha15032/pi05_bridge_deadly_corridor_single_latency_clean_data_exp2" \
    dataset.source_hf=latency-sensitive-bench/benchmark-datasets \
    dataset.revision=d6f0ad0995f6e4b095f315920e8de18e04297048 \
    dataset.source_subdir=zero-latency/deadly-corridor/deadly_1000ep/deadly_corridor_fix_latency_0_1000ep \
    checkpoint.local.keep_last_n=1 \
    checkpoint.save_best_model=false \
    trainer.max_train_steps=500 \
    trainer.num_warmup_steps=0 \
    trainer.eval_interval=50 \
    trainer.save_interval=500 \
    rl_games.env_eval.deadly.action_layout=multibinary_7 \
    framework.action_model.action_dim=7 \
    framework.action_model.action_env_dim=7 \
    trainer.logging_frequency=1 \
    trainer.gradient_accumulation_steps=4 \
    datasets.vla_data.per_device_batch_size=32 \
    rl_games.env_eval.mid_train.enabled=false \
    rl_games.env_eval.post_train.enabled=true \
    rl_games.env_eval.post_train.latencies=[0,1,2,3,4,5,6,7,8,9,10] \
    rl_games.env_eval.post_train.num_episodes=20 \
    rl_games.env_eval.post_train.max_steps_per_episode=3600
