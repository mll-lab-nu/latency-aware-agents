cd starVLA

bash examples/rl_games/install/install_stack.sh openvla demon_attack

conda activate starvla_rl_games_openvla

python examples/rl_games/scripts/launch_train.py \
    model=openvla \
    env=demon_attack \
    init=bridge \
    mode=single \
    run_id="openvla_bridge_demon_attack_single_latency_clean_data_exp1" \
    trainer.distributed_backend=none \
    workspace_dir="/workspace" \
    wandb_entity="talha1503" \
    checkpoint.hf_repo_id="talha15032/openvla_bridge_demon_attack_single_latency_clean_data_exp1" \
    checkpoint.sync.enabled=true \
    checkpoint.sync.repo_id="talha15032/openvla_bridge_demon_attack_single_latency_clean_data_exp1" \
    dataset.source_hf=talha15032/demon_attack_zero_latency_parquet \
    checkpoint.local.keep_last_n=1 \
    trainer.max_train_steps=7000 \
    trainer.num_warmup_steps=0 \
    trainer.eval_interval=1000 \
    trainer.logging_frequency=1 \
    trainer.gradient_accumulation_steps=4 \
    datasets.vla_data.per_device_batch_size=32 \
    rl_games.env_eval.mid_train.interval_steps=1000 \
    rl_games.env_eval.mid_train.num_episodes=5 \
    rl_games.env_eval.mid_train.max_steps_per_episode=3600 \
    rl_games.env_eval.post_train.enabled=true \
    rl_games.env_eval.post_train.latencies=[0,1,2,3,4,5,6,7,8] \
    rl_games.env_eval.post_train.num_episodes=20 \
    rl_games.env_eval.post_train.max_steps_per_episode=3600
