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
    run_id="openvla_bridge_demon_attack_latency_mixed_01234_40ep_exp1" \
    trainer.distributed_backend=none \
    workspace_dir="/workspace" \
    wandb_entity="talha1503" \
    checkpoint.hf_repo_id="talha15032/openvla_bridge_demon_attack_latency_mixed_01234_40ep_exp1" \
    checkpoint.sync.enabled=true \
    checkpoint.sync.repo_id="talha15032/openvla_bridge_demon_attack_latency_mixed_01234_40ep_exp1" \
    dataset.source_hf="latency-sensitive-bench/demon_attack_01234_40ep" \
    dataset.converted_name="demon_attack_01234_40ep_latency_prompt_train" \
    dataset.latency_filter=[0,1,2,3,4] \
    datasets.vla_data.sequential_step_sampling=true \
    datasets.vla_data.shuffle=true \
    trainer.per_latency_eval_num_batches=5 \
    dataset.episodes_per_latency=40 \
    rl_games.env_eval.eval_backend=latency_bench \
    datasets.vla_data.shuffle=true \
    checkpoint.local.keep_last_n=1 \
    trainer.max_train_steps=7000 \
    trainer.num_warmup_steps=0 \
    checkpoint.save_final_model=true \
    checkpoint.save_best_model=false \
    trainer.eval_interval=7000 \
    trainer.save_interval=7000 \
    trainer.logging_frequency=1 \
    trainer.gradient_accumulation_steps=4 \
    datasets.vla_data.per_device_batch_size=32 \
    rl_games.env_eval.mid_train.enabled=false \
    rl_games.env_eval.post_train.enabled=true \
    rl_games.env_eval.post_train.latencies=[0,1,2,3,4] \
    rl_games.env_eval.post_train.num_episodes=20 \
    rl_games.env_eval.post_train.max_steps_per_episode=3600
