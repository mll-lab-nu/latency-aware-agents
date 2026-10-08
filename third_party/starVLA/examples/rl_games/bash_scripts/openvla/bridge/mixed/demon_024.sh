bash /workspace/starVLA/examples/rl_games/bash_scripts/install/pre_launch.sh

cd /workspace/starVLA

bash examples/rl_games/install/install_stack.sh openvla demon_attack

conda activate starvla_rl_games_openvla

bash /workspace/starVLA/examples/rl_games/bash_scripts/install/latency_deps.sh 

python examples/rl_games/scripts/launch_train.py \
    model=openvla \
    env=demon_attack \
    init=bridge \
    mode=mixed \
    run_id="openvla_bridge_demon_attack_latency_mixed_024_exp1" \
    trainer.distributed_backend=none \
    workspace_dir="/workspace" \
    wandb_entity="talha1503" \
    checkpoint.hf_repo_id="latency-sensitive-bench/benchmark-models" \
    checkpoint.sync.enabled=false \
    checkpoint.sync.repo_id="latency-sensitive-bench/benchmark-models" \
    dataset.source_hf="latency-sensitive-bench/benchmark-datasets" \
    dataset.revision=15d17287246bd19940c837d6c8a5327564f969c5 \
    dataset.source_subdir=zero-latency/demon-attack/demon_attack_200ep/demon_attack_fix_latency_0_200ep \
    dataset.latency_filter=[0,2,4] \
    datasets.vla_data.sequential_step_sampling=true \
    datasets.vla_data.shuffle=true \
    trainer.per_latency_eval_num_batches=5 \
    dataset.episodes_per_latency=40 \
    rl_games.env_eval.eval_backend=latency_bench \
    datasets.vla_data.shuffle=true \
    checkpoint.local.keep_last_n=1 \
    trainer.max_train_steps=5000 \
    trainer.num_warmup_steps=0 \
    checkpoint.save_final_model=true \
    checkpoint.save_best_model=false \
    trainer.eval_interval=5000 \
    trainer.save_interval=5000 \
    trainer.logging_frequency=1 \
    trainer.gradient_accumulation_steps=4 \
    datasets.vla_data.per_device_batch_size=32 \
    rl_games.env_eval.mid_train.enabled=false \
    rl_games.env_eval.post_train.enabled=true \
    rl_games.env_eval.post_train.latencies=[0,2,4] \
    rl_games.env_eval.post_train.num_episodes=20 \
    rl_games.env_eval.post_train.max_steps_per_episode=3600
