WORKSPACE_DIR=""
export WORKSPACE_DIR

bash "${WORKSPACE_DIR}/starVLA/examples/rl_games/bash_scripts/install/pre_launch.sh"

cd "${WORKSPACE_DIR}/starVLA"

bash examples/rl_games/install/install_stack.sh openvla flappy
bash examples/rl_games/install/install_stack.sh openvla demon_attack

conda activate starvla_rl_games_openvla

bash "${WORKSPACE_DIR}/starVLA/examples/rl_games/bash_scripts/install/latency_deps.sh"

export PYTHONPATH="${WORKSPACE_DIR}/latency-sensitive-bench:${PYTHONPATH:-}"

python examples/rl_games/scripts/launch_train.py \
    model=openvla \
    env=cross_task \
    init=bridge \
    mode=cross_task \
    cross_task_setup=flappy_mixed_demon_mixed \
    run_id="openvla_bridge_cross_flappy_024_demon_024_clean_data_exp3" \
    trainer.distributed_backend=none \
    workspace_dir="$WORKSPACE_DIR" \
    wandb_entity="talha1503" \
    checkpoint.hf_repo_id="talha15032/openvla_bridge_cross_flappy_024_demon_024_clean_data_exp3" \
    checkpoint.sync.enabled=true \
    checkpoint.sync.repo_id="talha15032/openvla_bridge_cross_flappy_024_demon_024_clean_data_exp3" \
    checkpoint.save_best_model=false \
    trainer.max_train_steps=10500 \
    trainer.num_warmup_steps=0 \
    trainer.eval_interval=1400 \
    trainer.save_interval=10500 \
    trainer.logging_frequency=1 \
    datasets.vla_data.sequential_step_sampling=true \
    datasets.vla_data.shuffle=true \
    trainer.eval_action_classification=true \
    trainer.gradient_accumulation_steps=4 \
    datasets.vla_data.per_device_batch_size=32 \
    datasets.vla_data.include_state=false \
    framework.action_model.loss_type=discrete_ce \
    framework.action_model.action_dim=7 \
    framework.action_model.action_env_dim=7 \
    trainer.per_latency_eval_num_batches=5 \
    rl_games.cross_task.train_tasks.0.name=demon_attack \
    rl_games.cross_task.train_tasks.0.converted_name=demon_attack_024_200ep_cross_train \
    rl_games.cross_task.train_tasks.0.train_source_hf=latency-sensitive-bench/benchmark-datasets \
    rl_games.cross_task.train_tasks.0.train_revision=5305590625c28a6922e73f46d18d52738762c906 \
    rl_games.cross_task.train_tasks.0.train_source_subdir=zero-latency/demon-attack/demon_attack_200ep/demon_attack_fix_latency_0_200ep \
    rl_games.cross_task.train_tasks.0.prompt_source_hf=latency-sensitive-bench/benchmark-datasets \
    rl_games.cross_task.train_tasks.0.prompt_revision=5305590625c28a6922e73f46d18d52738762c906 \
    rl_games.cross_task.train_tasks.0.prompt_source_subdir=zero-latency/demon-attack/demon_attack_200ep/demon_attack_fix_latency_0_200ep \
    rl_games.cross_task.train_tasks.0.train_latency_filter=[0,2,4] \
    rl_games.cross_task.train_tasks.0.eval_latency_filter=[0,2,4] \
    rl_games.cross_task.train_tasks.0.episodes_per_latency=40 \
    rl_games.cross_task.train_tasks.1.name=flappy \
    rl_games.cross_task.train_tasks.1.converted_name=flappy_024_200ep_cross_train \
    rl_games.cross_task.train_tasks.1.train_source_hf=latency-sensitive-bench/benchmark-datasets \
    rl_games.cross_task.train_tasks.1.train_revision=5305590625c28a6922e73f46d18d52738762c906 \
    rl_games.cross_task.train_tasks.1.train_source_subdir=zero-latency/flappy/flappy_200ep/flappy_fix_latency_0_200ep \
    rl_games.cross_task.train_tasks.1.prompt_source_hf=latency-sensitive-bench/benchmark-datasets \
    rl_games.cross_task.train_tasks.1.prompt_revision=5305590625c28a6922e73f46d18d52738762c906 \
    rl_games.cross_task.train_tasks.1.prompt_source_subdir=zero-latency/flappy/flappy_200ep/flappy_fix_latency_0_200ep \
    rl_games.cross_task.train_tasks.1.train_latency_filter=[0,2,4] \
    rl_games.cross_task.train_tasks.1.eval_latency_filter=[0,2,4] \
    rl_games.cross_task.train_tasks.1.episodes_per_latency=40 \
    rl_games.cross_task.train_tasks.0.max_episodes=null \
    rl_games.cross_task.train_tasks.1.max_episodes=null \
    rl_games.env_eval.eval_backend=eval_core \
    rl_games.cross_task.eval_tasks.demon_attack.mid_train.enabled=false \
    rl_games.cross_task.eval_tasks.demon_attack.post_train.latencies=[0,2,4] \
    rl_games.cross_task.eval_tasks.demon_attack.post_train.num_episodes=20 \
    rl_games.cross_task.eval_tasks.demon_attack.post_train.max_steps_per_episode=3600 \
    rl_games.cross_task.eval_tasks.deadly_corridor.mid_train.enabled=false \
    rl_games.cross_task.eval_tasks.deadly_corridor.post_train.enabled=false \
    rl_games.cross_task.eval_tasks.flappy.mid_train.enabled=false \
    rl_games.cross_task.eval_tasks.flappy.post_train.latencies=[0,2,4] \
    rl_games.cross_task.eval_tasks.flappy.post_train.num_episodes=20 \
    rl_games.cross_task.eval_tasks.flappy.post_train.max_steps_per_episode=3600
