WORKSPACE_DIR="${WORKSPACE_DIR:-/workspace}"
export WORKSPACE_DIR

bash "${WORKSPACE_DIR}/starVLA/examples/rl_games/bash_scripts/install/pre_launch.sh"

cd "${WORKSPACE_DIR}/starVLA"

bash examples/rl_games/install/install_stack.sh gr00t demon_attack
bash examples/rl_games/install/install_stack.sh gr00t deadly_corridor
bash examples/rl_games/install/install_stack.sh gr00t defend_the_line

conda activate starvla_rl_games_gr00t

bash "${WORKSPACE_DIR}/starVLA/examples/rl_games/bash_scripts/install/latency_deps.sh"

export PYTHONPATH="${WORKSPACE_DIR}/latency-sensitive-bench:${PYTHONPATH:-}"

python examples/rl_games/scripts/launch_train.py \
    model=gr00t \
    env=defend_the_line \
    init=bridge \
    mode=single \
    run_id="gr00t_bridge_defend_the_line_single_latency_clean_data_exp1" \
    trainer.distributed_backend=none \
    workspace_dir="$WORKSPACE_DIR" \
    wandb_entity="talha1503" \
    checkpoint.hf_repo_id="talha15032/gr00t_bridge_defend_the_line_single_latency_clean_data_exp1" \
    checkpoint.sync.enabled=true \
    checkpoint.sync.repo_id="talha15032/gr00t_bridge_defend_the_line_single_latency_clean_data_exp1" \
    dataset.source_hf=latency-sensitive-bench/benchmark-datasets \
    dataset.revision=4d8be25dc6fccfe244c2fa6be4b771ee448f2c2a \
    dataset.source_subdir=zero-latency/defend-the-line/defend_the_line_fixed_latency_0_1000ep_7k2steps \
    dataset.latency_filter=[0] \
    checkpoint.local.keep_last_n=1 \
    trainer.max_train_steps=7000 \
    trainer.num_warmup_steps=0 \
    trainer.eval_interval=7000 \
    trainer.save_interval=7000 \
    checkpoint.save_best_model=false \
    trainer.logging_frequency=1 \
    trainer.gradient_accumulation_steps=4 \
    datasets.vla_data.per_device_batch_size=32 \
    rl_games.env_eval.mid_train.interval_steps=1000 \
    rl_games.env_eval.mid_train.num_episodes=5 \
    rl_games.env_eval.mid_train.max_steps_per_episode=3600 \
    rl_games.env_eval.post_train.enabled=true \
    rl_games.env_eval.post_train.latencies=[0,2,4,6] \
    rl_games.env_eval.post_train.num_episodes=20 \
    rl_games.env_eval.post_train.max_steps_per_episode=3600
