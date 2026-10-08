cd starVLA

bash examples/rl_games/install/install_stack.sh pi05 flappy

conda activate starvla_rl_games_pi05

python examples/rl_games/scripts/launch_train.py \
    model=pi05 \
    env=flappy \
    init=bridge \
    mode=single \
    run_id="pi05_bridge_flappy_single_latency_clean_data_exp2" \
    trainer.distributed_backend=none \
    workspace_dir="/workspace" \
    wandb_entity="talha1503" \
    checkpoint.hf_repo_id="talha15032/pi05_bridge_flappy_single_latency_clean_data_exp2" \
    checkpoint.sync.enabled=true \
    checkpoint.sync.repo_id="talha15032/pi05_bridge_flappy_single_latency_clean_data_exp2" \
    dataset.source_hf="latency-sensitive-bench/benchmark-datasets" \
    dataset.revision=15d17287246bd19940c837d6c8a5327564f969c5 \
    dataset.source_subdir=zero-latency/flappy/flappy_200ep/flappy_fix_latency_0_200ep \
    checkpoint.local.keep_last_n=1 \
    checkpoint.save_best_model=false \
    trainer.max_train_steps=5000 \
    trainer.num_warmup_steps=0 \
    trainer.eval_interval=300 \
    trainer.save_interval=5000 \
    trainer.logging_frequency=1 \
    trainer.gradient_accumulation_steps=4 \
    datasets.vla_data.per_device_batch_size=32 \
    rl_games.env_eval.mid_train.enabled=false \
    rl_games.env_eval.post_train.enabled=true \
    rl_games.env_eval.post_train.latencies=[0,1,2,3,4,5] \
    rl_games.env_eval.post_train.num_episodes=20 \
    rl_games.env_eval.post_train.max_steps_per_episode=3600
