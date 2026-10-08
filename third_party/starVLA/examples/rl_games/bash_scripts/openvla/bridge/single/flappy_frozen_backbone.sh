cd starVLA

bash examples/rl_games/install/install_stack.sh openvla flappy

conda activate starvla_rl_games_openvla

python examples/rl_games/scripts/launch_train.py \
    model=openvla \
    env=flappy \
    init=bridge \
    mode=single \
    run_id="openvla_bridge_flappy_single_latency_frozen_backbone_exp1" \
    trainer.distributed_backend=none \
    workspace_dir="/workspace" \
    wandb_entity="talha1503" \
    checkpoint.hf_repo_id="talha1503/openvla_bridge_flappy_single_latency_frozen_backbone_exp1" \
    checkpoint.sync.enabled=true \
    checkpoint.sync.repo_id="talha1503/openvla_bridge_flappy_single_latency_frozen_backbone_exp1" \
    dataset.source_hf="latency-sensitive-bench/benchmark-datasets" \
    dataset.revision=15d17287246bd19940c837d6c8a5327564f969c5 \
    dataset.source_subdir=zero-latency/flappy/flappy_200ep/flappy_fix_latency_0_200ep \
    checkpoint.local.keep_last_n=1 \
    checkpoint.save_best_model=false \
    trainer.freeze_modules=qwen_vl_interface \
    trainer.max_train_steps=5000 \
    trainer.num_warmup_steps=0 \
    trainer.save_interval=5000 \
    trainer.eval_interval=500 \
    trainer.logging_frequency=1 \
    trainer.gradient_accumulation_steps=4 \
    datasets.vla_data.per_device_batch_size=32 \
    rl_games.env_eval.mid_train.enabled=false \
    rl_games.env_eval.post_train.enabled=true \
    rl_games.env_eval.post_train.latencies=[0,1,2,3,4] \
    rl_games.env_eval.post_train.num_episodes=20 \
    rl_games.env_eval.post_train.max_steps_per_episode=3600
