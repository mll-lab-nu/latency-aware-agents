WORKSPACE_DIR="${WORKSPACE_DIR:-/workspace}"
export WORKSPACE_DIR

bash "${WORKSPACE_DIR}/starVLA/examples/rl_games/bash_scripts/install/pre_launch.sh"

cd "${WORKSPACE_DIR}/starVLA"

bash examples/rl_games/install/install_stack.sh openvla deadly_corridor

conda activate starvla_rl_games_openvla

bash "${WORKSPACE_DIR}/starVLA/examples/rl_games/bash_scripts/install/latency_deps.sh"

export PYTHONPATH="${WORKSPACE_DIR}/latency-sensitive-bench:${PYTHONPATH:-}"

python examples/rl_games/scripts/launch_train.py \
    model=openvla \
    env=deadly_corridor \
    init=bridge \
    mode=mixed \
    run_id="openvla_bridge_deadly_corridor_latency_mixed_024_exp3" \
    trainer.distributed_backend=none \
    workspace_dir="$WORKSPACE_DIR" \
    wandb_entity="talha1503" \
    checkpoint.hf_repo_id="talha15032/openvla_bridge_deadly_corridor_latency_mixed_024_exp3" \
    checkpoint.sync.enabled=true \
    checkpoint.sync.repo_id="talha15032/openvla_bridge_deadly_corridor_latency_mixed_024_exp3" \
    checkpoint.save_best_model=false \
    dataset.source_hf=latency-sensitive-bench/benchmark-datasets \
    dataset.revision=d6f0ad0995f6e4b095f315920e8de18e04297048 \
    dataset.source_subdir=zero-latency/deadly-corridor/deadly_1000ep/deadly_corridor_fix_latency_0_1000ep \
    dataset.latency_filter=[0,2,4] \
    dataset.episodes_per_latency=400 \
    dataset.converted_name=deadly_corridor_024_openvla_bridge_mixed_exp3 \
    rl_games.env_eval.deadly.action_layout=multibinary_7 \
    rl_games.env_eval.deadly.multibinary_threshold=0.0 \
    framework.action_model.loss_type=multibinary_bce \
    trainer.max_train_steps=2200 \
    datasets.vla_data.sequential_step_sampling=true \
    datasets.vla_data.shuffle=true \
    trainer.num_warmup_steps=0 \
    trainer.eval_interval=400 \
    trainer.save_interval=2200 \
    trainer.logging_frequency=1 \
    trainer.gradient_accumulation_steps=4 \
    trainer.eval_action_classification=true \
    datasets.vla_data.per_device_batch_size=32 \
    datasets.vla_data.include_state=false \
    framework.action_model.action_dim=7 \
    framework.action_model.action_env_dim=7 \
    trainer.per_latency_eval_num_batches=5 \
    rl_games.env_eval.eval_backend=eval_core \
    rl_games.env_eval.mid_train.enabled=false \
    rl_games.env_eval.post_train.enabled=true \
    rl_games.env_eval.post_train.latencies=[0,2,4] \
    rl_games.env_eval.post_train.num_episodes=20 \
    rl_games.env_eval.post_train.max_steps_per_episode=3600
