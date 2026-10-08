WORKSPACE_DIR="${WORKSPACE_DIR:-/workspace}"
export WORKSPACE_DIR

bash "${WORKSPACE_DIR}/starVLA/examples/rl_games/bash_scripts/install/pre_launch.sh"

cd "${WORKSPACE_DIR}/starVLA"

bash examples/rl_games/install/install_stack.sh openvla demon_attack
bash examples/rl_games/install/install_stack.sh openvla deadly_corridor

conda activate starvla_rl_games_openvla

bash "${WORKSPACE_DIR}/starVLA/examples/rl_games/bash_scripts/install/latency_deps.sh"

export PYTHONPATH="${WORKSPACE_DIR}/latency-sensitive-bench:${PYTHONPATH:-}"

python examples/rl_games/scripts/launch_train.py \
    model=openvla \
    env=cross_task \
    init=bridge \
    mode=cross_task \
    cross_task_setup=demon_zero_deadly_mixed \
    run_id="openvla_bridge_cross_demon_zero_deadly_zero_exp3" \
    trainer.distributed_backend=none \
    workspace_dir="$WORKSPACE_DIR" \
    wandb_entity="talha1503" \
    checkpoint.hf_repo_id="talha15032/openvla_bridge_cross_demon_zero_deadly_zero_exp3" \
    checkpoint.sync.enabled=true \
    checkpoint.sync.repo_id="talha15032/openvla_bridge_cross_demon_zero_deadly_zero_exp3" \
    checkpoint.save_best_model=false \
    trainer.max_train_steps=2000 \
    datasets.vla_data.sequential_step_sampling=true \
    datasets.vla_data.shuffle=true \
    trainer.num_warmup_steps=0 \
    trainer.eval_interval=400 \
    trainer.save_interval=2000 \
    trainer.logging_frequency=1 \
    trainer.gradient_accumulation_steps=4 \
    trainer.eval_action_classification=true \
    datasets.vla_data.per_device_batch_size=32 \
    datasets.vla_data.include_state=false \
    framework.action_model.action_dim=7 \
    framework.action_model.action_env_dim=7 \
    trainer.per_latency_eval_num_batches=5 \
    rl_games.cross_task.train_tasks.0.name=demon_attack \
    rl_games.cross_task.train_tasks.0.converted_name=demon_attack_0_cross_train_openvla_bridge_cross_demon_zero_deadly_zero_exp3 \
    rl_games.cross_task.train_tasks.0.train_source_hf=latency-sensitive-bench/benchmark-datasets \
    rl_games.cross_task.train_tasks.0.train_revision=d6f0ad0995f6e4b095f315920e8de18e04297048 \
    rl_games.cross_task.train_tasks.0.train_source_subdir=zero-latency/demon-attack/demon_attack_200ep/demon_attack_fix_latency_0_200ep \
    rl_games.cross_task.train_tasks.0.prompt_source_hf=latency-sensitive-bench/benchmark-datasets \
    rl_games.cross_task.train_tasks.0.prompt_revision=d6f0ad0995f6e4b095f315920e8de18e04297048 \
    rl_games.cross_task.train_tasks.0.prompt_source_subdir=zero-latency/demon-attack/demon_attack_200ep/demon_attack_fix_latency_0_200ep \
    rl_games.cross_task.train_tasks.0.train_latency_filter=[0] \
    rl_games.cross_task.train_tasks.0.eval_latency_filter=[0,2,4] \
    rl_games.cross_task.train_tasks.0.episodes_per_latency=40 \
    rl_games.cross_task.train_tasks.1.name=deadly_corridor \
    rl_games.cross_task.train_tasks.1.converted_name=deadly_corridor_0_cross_train_openvla_bridge_cross_demon_zero_deadly_zero_exp3 \
    rl_games.cross_task.train_tasks.1.prompt_source_hf=latency-sensitive-bench/benchmark-datasets \
    rl_games.cross_task.train_tasks.1.prompt_revision=d6f0ad0995f6e4b095f315920e8de18e04297048 \
    rl_games.cross_task.train_tasks.1.prompt_source_subdir=zero-latency/deadly-corridor/deadly_1000ep/deadly_corridor_fix_latency_0_1000ep \
    rl_games.cross_task.train_tasks.1.train_source_hf=latency-sensitive-bench/benchmark-datasets \
    rl_games.cross_task.train_tasks.1.train_revision=d6f0ad0995f6e4b095f315920e8de18e04297048 \
    rl_games.cross_task.train_tasks.1.train_source_subdir=zero-latency/deadly-corridor/deadly_1000ep/deadly_corridor_fix_latency_0_1000ep \
    rl_games.cross_task.train_tasks.1.train_latency_filter=[0] \
    rl_games.cross_task.train_tasks.1.eval_latency_filter=[0,2,4] \
    rl_games.cross_task.train_tasks.1.episodes_per_latency=1000 \
    rl_games.cross_task.train_tasks.1.action_layout=multibinary_7 \
    rl_games.env_eval.deadly.multibinary_threshold=0.0 \
    rl_games.cross_task.train_tasks.0.max_episodes=null \
    rl_games.cross_task.train_tasks.1.max_episodes=null \
    rl_games.env_eval.eval_backend=eval_core \
    rl_games.cross_task.eval_tasks.demon_attack.mid_train.enabled=false \
    rl_games.cross_task.eval_tasks.demon_attack.post_train.latencies=[0,2,4] \
    rl_games.cross_task.eval_tasks.demon_attack.post_train.num_episodes=20 \
    rl_games.cross_task.eval_tasks.demon_attack.post_train.max_steps_per_episode=3600 \
    rl_games.cross_task.eval_tasks.deadly_corridor.mid_train.enabled=false \
    rl_games.cross_task.eval_tasks.deadly_corridor.post_train.latencies=[0,2,4] \
    rl_games.cross_task.eval_tasks.deadly_corridor.post_train.num_episodes=20 \
    rl_games.cross_task.eval_tasks.deadly_corridor.post_train.max_steps_per_episode=3600
