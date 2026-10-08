cd starVLA

bash examples/rl_games/install/install_stack.sh openvla cross_task

conda activate starvla_rl_games_openvla

pip install flash-attn --no-build-isolation

python examples/rl_games/scripts/launch_train.py \
    model=openvla \
    env=cross_task \
    init=bridge \
    mode=cross_task \
    cross_task_setup=flappy_demon_deadly_024 \
    run_id="openvla_bridge_cross_flappy_demon_deadly_024_exp1" \
    trainer.distributed_backend=none \
    workspace_dir="/workspace" \
    wandb_entity="talha1503" \
    checkpoint.hf_repo_id="talha1503/openvla_bridge_cross_flappy_demon_deadly_024_exp1" \
    checkpoint.sync.enabled=true \
    checkpoint.sync.repo_id="talha1503/openvla_bridge_cross_flappy_demon_deadly_024_exp1" \
    checkpoint.save_best_model=false \
    trainer.max_train_steps=7000 \
    trainer.num_warmup_steps=0 \
    trainer.eval_interval=1400 \
    trainer.save_interval=7000 \
    trainer.logging_frequency=1 \
    trainer.gradient_accumulation_steps=4 \
    datasets.vla_data.per_device_batch_size=32 \
    datasets.vla_data.include_state=false \
    framework.action_model.loss_type=discrete_ce \
    framework.action_model.action_dim=7 \
    framework.action_model.action_env_dim=7 \
    trainer.per_latency_eval_num_batches=5 \
    rl_games.cross_task.eval_tasks.flappy.mid_train.enabled=false \
    rl_games.cross_task.eval_tasks.flappy.post_train.latencies=[0,2,4] \
    rl_games.cross_task.eval_tasks.flappy.post_train.num_episodes=20 \
    rl_games.cross_task.eval_tasks.flappy.post_train.max_steps_per_episode=2000 \
    rl_games.cross_task.eval_tasks.demon_attack.mid_train.enabled=false \
    rl_games.cross_task.eval_tasks.demon_attack.post_train.latencies=[0,2,4] \
    rl_games.cross_task.eval_tasks.demon_attack.post_train.num_episodes=20 \
    rl_games.cross_task.eval_tasks.demon_attack.post_train.max_steps_per_episode=3600 \
    rl_games.cross_task.eval_tasks.deadly_corridor.mid_train.enabled=false \
    rl_games.cross_task.eval_tasks.deadly_corridor.post_train.latencies=[0,2,4] \
    rl_games.cross_task.eval_tasks.deadly_corridor.post_train.num_episodes=20 \
    rl_games.cross_task.eval_tasks.deadly_corridor.post_train.max_steps_per_episode=3600
