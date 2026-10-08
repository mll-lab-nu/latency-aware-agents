bash /home/ubuntu/talha/starVLA/examples/rl_games/bash_scripts/install/pre_launch.sh

cd /home/ubuntu/talha/starVLA

bash examples/rl_games/install/install_stack.sh openvla gymnasium

conda activate starvla_rl_games_openvla

bash examples/rl_games/install/flash_attn.sh --check >/dev/null 2>&1 || bash examples/rl_games/install/flash_attn.sh

bash /home/ubuntu/talha/starVLA/examples/rl_games/bash_scripts/install/latency_deps.sh

export PYTHONPATH="/home/ubuntu/talha/latency-sensitive-bench:/home/ubuntu/talha/starVLA:${PYTHONPATH:-}"

TRAIN_LATENCIES="${TRAIN_LATENCIES:-0,2,4}"
TRAIN_EPISODE_MIX="${TRAIN_EPISODE_MIX:-0:50,2:300,4:600}"
RUN_ID="${RUN_ID:-openvla_inverted_pendulum_mixed_latency_024_native_no_latency_prompt_exp1}"
DATASET_NAME="${DATASET_NAME:-inverted_pendulum_mixed_latency_train_native_no_latency_prompt}"

python examples/rl_games/scripts/launch_train.py \
    model=openvla \
    env=gymnasium \
    init=scratch \
    mode=mixed_latency \
    run_id="${RUN_ID}" \
    trainer.distributed_backend=none \
    workspace_dir="/home/ubuntu/talha" \
    wandb_entity="talha1503" \
    checkpoint.hf_repo_id="talha15032/${RUN_ID}" \
    checkpoint.sync.enabled=true \
    checkpoint.sync.repo_id="talha15032/${RUN_ID}" \
    dataset.source_hf="latency-sensitive-bench/inverted_pendulum_200ep" \
    "dataset.source_subdir='inverted_pendulum_fixed_latency_0_200ep_7k2steps,inverted_pendulum_fixed_latency_2_1000ep_7k2steps,inverted_pendulum_fixed_latency_4_1000ep_7k2steps'" \
    dataset.converted_name="${DATASET_NAME}" \
    "dataset.latency_filter=[${TRAIN_LATENCIES}]" \
    "dataset.episodes_per_latency_by_latency='${TRAIN_EPISODE_MIX}'" \
    dataset.setup_force=true \
    dataset.target_latency_unit=raw_frames \
    dataset.max_steps_per_episode=1000 \
    datasets.vla_data.sequential_step_sampling=true \
    datasets.vla_data.shuffle=true \
    datasets.vla_data.prompt_mode=latency_neutral \
    datasets.vla_data.include_state=true \
    datasets.vla_data.action_type=discrete \
    datasets.vla_data.active_action_dim=5 \
    trainer.per_latency_eval_num_batches=5 \
    checkpoint.local.keep_last_n=1 \
    checkpoint.save_final_model=true \
    checkpoint.save_best_model=false \
    trainer.max_train_steps=5000 \
    trainer.num_warmup_steps=0 \
    trainer.eval_interval=5000 \
    trainer.save_interval=5000 \
    trainer.logging_frequency=1 \
    trainer.gradient_accumulation_steps=2 \
    datasets.vla_data.per_device_batch_size=8 \
    framework.action_model.action_dim=5 \
    framework.action_model.action_env_dim=5 \
    framework.action_model.state_dim=4 \
    framework.action_model.loss_type=discrete_ce \
    framework.action_model.action_horizon=1 \
    framework.action_model.future_action_window_size=0 \
    framework.action_model.past_action_window_size=0 \
    ++rl_games.gymnasium.task_contract.task_name=inverted_pendulum \
    ++rl_games.gymnasium.task_contract.env_id=LatencyBench/InvertedPendulumDiscrete-v0 \
    ++rl_games.gymnasium.task_contract.registration_imports=[latency_bench.envs.gymnasium_mujoco] \
    ++rl_games.gymnasium.task_contract.env_fps=25.0 \
    ++rl_games.gymnasium.task_contract.obs_fps=25.0 \
    ++rl_games.gymnasium.task_contract.frame_stack=1 \
    "++rl_games.gymnasium.task_contract.base_prompt='Keep the pole upright by applying horizontal cart forces. Choose exactly one action from: push_left_hard, push_left_soft, zero_force, push_right_soft, push_right_hard.'" \
    ++rl_games.gymnasium.task_contract.action_labels=[push_left_hard,push_left_soft,zero_force,push_right_soft,push_right_hard] \
    ++rl_games.gymnasium.task_contract.action_values=[0,1,2,3,4] \
    ++rl_games.gymnasium.task_contract.noop_action_id=2 \
    ++rl_games.gymnasium.task_contract.make_kwargs.base_env_id=InvertedPendulum-v4 \
    ++rl_games.gymnasium.task_contract.make_kwargs.render_mode=rgb_array \
    ++rl_games.gymnasium.task_contract.make_kwargs.force_values=[-3.0,-1.5,0.0,1.5,3.0] \
    ++rl_games.gymnasium.task_contract.make_kwargs.base_make_kwargs={} \
    ++rl_games.gymnasium.task_contract.state_space.labels=[cart_position,pole_angle,cart_velocity,pole_angular_velocity] \
    rl_games.env_eval.enabled=false \
    rl_games.env_eval.eval_backend=latency_bench \
    rl_games.env_eval.prompt_mode=latency_neutral \
    rl_games.env_eval.latency.mode=mixed \
    rl_games.env_eval.latency.values=[0,2,4] \
    rl_games.env_eval.mid_train.enabled=false \
    rl_games.env_eval.post_train.enabled=true \
    rl_games.env_eval.post_train.latencies=[0,2,4] \
    rl_games.env_eval.post_train.num_episodes=20 \
    rl_games.env_eval.post_train.max_steps_per_episode=3600
