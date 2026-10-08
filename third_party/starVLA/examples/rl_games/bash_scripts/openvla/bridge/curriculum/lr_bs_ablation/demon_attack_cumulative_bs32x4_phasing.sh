cd starVLA

bash examples/rl_games/install/install_stack.sh openvla demon_attack

conda activate starvla_rl_games_openvla

bash examples/rl_games/install/flash_attn.sh --check >/dev/null 2>&1 || bash examples/rl_games/install/flash_attn.sh

python examples/rl_games/scripts/launch_train.py \
    model=openvla \
    env=demon_attack \
    init=bridge \
    mode=curriculum_cumulative \
    run_id="openvla_demon_attack_mixed_latency_curriculum_cumulative_clean_exp1_bs32x4_phasing" \
    trainer.distributed_backend=none \
    workspace_dir="/workspace" \
    wandb_entity="talha1503" \
    checkpoint.hf_repo_id="talha15032/openvla_demon_attack_mixed_latency_curriculum_cumulative_clean_exp1_bs32x4_phasing" \
    checkpoint.sync.enabled=true \
    checkpoint.sync.repo_id="talha15032/openvla_demon_attack_mixed_latency_curriculum_cumulative_clean_exp1_bs32x4_phasing" \
    dataset.source_hf="latency-sensitive-bench/benchmark-datasets" \
    dataset.revision=15d17287246bd19940c837d6c8a5327564f969c5 \
    dataset.source_subdir=zero-latency/demon-attack/demon_attack_200ep/demon_attack_fix_latency_0_200ep \
    dataset.latency_filter=[0,2,4,6,8] \
    datasets.vla_data.latency_curriculum.phase_steps=[300,500,800,1400,7500] \
    trainer.max_train_steps=10500 \
    trainer.save_interval=10500 \
    checkpoint.local.keep_last_n=1 \
    trainer.per_latency_eval_num_batches=40 \
    datasets.vla_data.latency_curriculum.enabled=true \
    datasets.vla_data.latency_curriculum.strategy=cumulative \
    datasets.vla_data.latency_curriculum.latencies=[0,2,4,6,8] \
    dataset.episodes_per_latency=40 \
    checkpoint.save_best_model=false \
    checkpoint.local.keep_last_n=1 \
    trainer.num_warmup_steps=0 \
    trainer.eval_interval=10500 \
    trainer.logging_frequency=1 \
    trainer.gradient_accumulation_steps=4 \
    datasets.vla_data.per_device_batch_size=32 \
    rl_games.env_eval.mid_train.enabled=false \
    rl_games.env_eval.post_train.enabled=true \
    rl_games.env_eval.post_train.latencies=[0,2,4,6,8] \
    rl_games.env_eval.post_train.num_episodes=5 \
    rl_games.env_eval.post_train.max_steps_per_episode=3600 \
