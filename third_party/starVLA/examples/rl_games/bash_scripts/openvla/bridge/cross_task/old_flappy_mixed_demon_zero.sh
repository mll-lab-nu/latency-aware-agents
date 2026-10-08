cd starVLA

bash examples/rl_games/install/install_stack.sh openvla cross_task

conda activate starvla_rl_games_openvla

bash examples/rl_games/install/flash_attn.sh --check >/dev/null 2>&1 || bash examples/rl_games/install/flash_attn.sh

python examples/rl_games/scripts/launch_train.py \
    model=openvla \
    env=cross_task \
    init=bridge \
    mode=cross_task \
    cross_task_setup=flappy_mixed_demon_zero \
    run_id="openvla_bridge_cross_flappy_mixed_demon_zero_exp4" \
    trainer.distributed_backend=deepspeed \
    launch.use_accelerate=true \
    launch.gpus=0,1 \
    launch.num_processes=2 \
    rl_games.env_eval.distributed_mode=rank_sharded \
    workspace_dir="/workspace" \
    wandb_entity="talha1503" \
    checkpoint.hf_repo_id="talha15032/openvla_bridge_cross_flappy_mixed_demon_zero_exp4" \
    checkpoint.sync.enabled=true \
    checkpoint.sync.repo_id="talha15032/openvla_bridge_cross_flappy_mixed_demon_zero_exp4" \
    checkpoint.save_best_model=true \
    trainer.max_train_steps=5000 \
    trainer.num_warmup_steps=0 \
    trainer.eval_interval=500 \
    trainer.logging_frequency=1 \
    trainer.gradient_accumulation_steps=16 \
    datasets.vla_data.per_device_batch_size=32 \
    datasets.vla_data.include_state=false \
    framework.action_model.loss_type=discrete_ce \
    framework.action_model.action_dim=7 \
    rl_games.cross_task.train_tasks.0.train_source_hf=talha1503/flappy_bird_mixed_latency_parquet \
    rl_games.cross_task.train_tasks.0.prompt_source_hf=talha1503/flappy_bird_mixed_latency_parquet \
    rl_games.cross_task.train_tasks.0.train_latency_filter=[0,1,2,3,4,5] \
    rl_games.cross_task.train_tasks.0.episodes_per_latency=30 \
    rl_games.cross_task.train_tasks.1.train_source_hf=latency-sensitive-bench/benchmark-datasets \
    rl_games.cross_task.train_tasks.1.train_revision=5305590625c28a6922e73f46d18d52738762c906 \
    rl_games.cross_task.train_tasks.1.train_source_subdir=training/demon_attack_fix_latency_0 \
    rl_games.cross_task.train_tasks.1.prompt_source_hf=latency-sensitive-bench/benchmark-datasets \
    rl_games.cross_task.train_tasks.1.prompt_source_subdir=experiments/task-transfer/archive/demon-attack/demon_attack_mixed_latency_min_0_max_7 \
    rl_games.cross_task.train_tasks.1.prompt_revision=c65ee4c303358dc448022bed9bc25e78984449a8 \
    rl_games.cross_task.train_tasks.1.train_latency_filter=[0] \
    rl_games.cross_task.train_tasks.1.episodes_per_latency=40 \
    rl_games.env_eval.mid_train.interval_steps=500 \
    rl_games.cross_task.eval_tasks.demon_attack.mid_train.latencies=[0,1,2,3,4,5] \
    rl_games.cross_task.eval_tasks.demon_attack.mid_train.num_episodes=3 \
    rl_games.cross_task.eval_tasks.demon_attack.mid_train.max_steps_per_episode=3600 \
    rl_games.cross_task.eval_tasks.demon_attack.post_train.latencies=[0,1,2,3,4,5] \
    rl_games.cross_task.eval_tasks.demon_attack.post_train.num_episodes=20 \
    rl_games.cross_task.eval_tasks.demon_attack.post_train.max_steps_per_episode=3600 \
    rl_games.cross_task.eval_tasks.flappy.mid_train.latencies=[0,1,2,3,4,5] \
    rl_games.cross_task.eval_tasks.flappy.mid_train.num_episodes=3 \
    rl_games.cross_task.eval_tasks.flappy.mid_train.max_steps_per_episode=3600 \
    rl_games.cross_task.eval_tasks.flappy.post_train.latencies=[0,1,2,3,4,5] \
    rl_games.cross_task.eval_tasks.flappy.post_train.num_episodes=20 \
    rl_games.cross_task.eval_tasks.flappy.post_train.max_steps_per_episode=3600 \
    rl_games.env_eval.vectorized.enabled=true \
    rl_games.env_eval.vectorized.batch_size=3 \
