# ## FLAPPY BIRD
# python scripts/generate_sample_factory_same_latency_sweep.py \
#   --envs flappy \
#   --seeds 0 \
#   --flappy-latencies 0 \
#   --generated-root /workspace/one_run_flappy \
#   --train-dir /workspace/sample_factory_runs \
#   --output-root /workspace/latency_sweeps \
#   --device gpu \
#   --restart-behavior resume \
#   --eval-episodes 20

# python -m training.train --config /workspace/one_run_flappy/flappy/latency_0/seed_0/train.yaml
# python -m training.train --config /workspace/one_run_flappy/flappy/latency_0/seed_0/eval.yaml

# ## DEMON ATTACK 
# python scripts/generate_sample_factory_same_latency_sweep.py \
#   --envs demon_attack \
#   --seeds 0 \
#   --demon-attack-latencies 0 \
#   --generated-root /workspace/one_run_demon \
#   --train-dir /workspace/sample_factory_runs \
#   --output-root /workspace/latency_sweeps \
#   --device gpu \
#   --restart-behavior resume \
#   --eval-episodes 20

# python -m training.train --config /workspace/one_run_demon/demon_attack/latency_0/seed_0/train.yaml
# python -m training.train --config /workspace/one_run_demon/demon_attack/latency_0/seed_0/eval.yaml


# # DEADLY CORRIDOR
# # Base config now points to the discrete-7 SF env variant:
# #   env_id: doom_deadly_corridor_discrete7
# python scripts/generate_sample_factory_same_latency_sweep.py \
#   --envs deadly_corridor \
#   --seeds 0 \
#   --deadly-corridor-latencies 0 \
#   --generated-root /workspace/one_run \
#   --train-dir /workspace/sample_factory_runs \
#   --output-root /workspace/latency_sweeps \
#   --device gpu \
#   --restart-behavior resume \
#   --eval-episodes 20

# python -m training.train --config /workspace/one_run/deadly_corridor/latency_0/seed_0/train.yaml
# python -m training.train --config /workspace/one_run/deadly_corridor/latency_0/seed_0/eval.yaml


## TEACHER DATASET EXPORTS

# # DEADLY CORRIDOR: zero-latency teacher dataset
# # --env-name selects the exporter/integration family; the actual Doom env_id is
# # loaded from each checkpoint's config.json (e.g. doom_deadly_corridor_discrete7).
# python scripts/export_sf_teacher_dataset.py \
#   --env-name deadly_corridor \
#   --checkpoint-root /workspace/latency-checkpoints/checkpoints \
#   --output-dir /workspace/teacher_datasets/deadly_corridor_zero_latency \
#   --dataset-mode zero_latency \
#   --prompt "You are playing Deadly Corridor in VizDoom. Your goal is to survive the corridor, fight enemies, and reach the green armor vest at the far end. The available actions are: MOVE_FORWARD, MOVE_BACKWARD, MOVE_LEFT, MOVE_RIGHT, TURN_LEFT, TURN_RIGHT, and ATTACK. Choose the best next action from the current visual observation." \
#   --max-steps-per-episode 2100 \
#   --max-episodes 100 \
#   --val-fraction 0.1 \
#   --num-envs 64 \
#   --image-writer-workers 16 \
#   --row-buffer-size 8192 \
#   --image-source observation \
#   --push-to-hub \
#   --hf-private \
#   --hf-token "$HF_TOKEN" \
#   --hf-max-shard-size 500MB \
#   --hf-repo-id talha1503/deadly_corridor_zero_latency_parquet

# # DEADLY CORRIDOR: mixed-latency teacher dataset
# python scripts/export_sf_teacher_dataset.py \
#   --env-name deadly_corridor \
#   --checkpoint-root /workspace/latency-checkpoints/checkpoints \
#   --output-dir /workspace/teacher_datasets/deadly_corridor_mixed_latency \
#   --dataset-mode mixed_latency \
#   --prompt "You are playing Deadly Corridor in VizDoom. Your goal is to survive the corridor, fight enemies, and reach the green armor vest at the far end. The available actions are: MOVE_FORWARD, MOVE_BACKWARD, MOVE_LEFT, MOVE_RIGHT, TURN_LEFT, TURN_RIGHT, and ATTACK." \
#   --mixed-prompt-template "{prompt} Your actions take {latency_ms} milliseconds to take effect. Choose the best next action while accounting for the delay." \
#   --max-steps-per-episode 2100 \
#   --latency-start 0 \
#   --latency-end 5 \
#   --episodes-per-latency 50 \
#   --val-episodes-per-latency 10 \
#   --num-envs 64 \
#   --image-writer-workers 16 \
#   --row-buffer-size 8192 \
#   --image-source observation \
#   --push-to-hub \
#   --hf-private \
#   --hf-token "$HF_TOKEN" \
#   --hf-max-shard-size 500MB \
#   --hf-repo-id talha1503/deadly_corridor_mixed_latency_parquet

python scripts/export_sf_teacher_dataset.py \
  --env-name deadly_corridor \
  --checkpoint-root checkpoints \
  --output-dir teacher_datasets/deadly_corridor_fix_latency_0 \
  --dataset-mode fixed_latency \
  --fixed-latency 0 \
  --prompt "You are playing Deadly Corridor in VizDoom. Your goal is to survive the corridor, fight enemies, and reach the green armor vest at the far end. The available actions are: MOVE_FORWARD, MOVE_BACKWARD, MOVE_LEFT, MOVE_RIGHT, TURN_LEFT, TURN_RIGHT, and ATTACK. Choose the best next action from the current visual observation." \
  --max-steps-per-episode 30000 \
  --max-episodes 100 \
  --val-fraction 0.1 \
  --num-envs 64 \
  --image-writer-workers 32 \
  --row-buffer-size 8192 \
  --image-source render

# Demon attack fix latency
# python scripts/export_sf_teacher_dataset.py \
#   --env-name demon_attack \
#   --checkpoint-root checkpoints \
#   --output-dir teacher_datasets/demon_attack_fix_latency_15 \
#   --dataset-mode fixed_latency \
#   --fixed-latency 15 \
#   --prompt "You are playing Demon Attack from a single game image. Choose exactly one action from: NOOP, FIRE, RIGHT, LEFT, RIGHTFIRE, LEFTFIRE. Survive, avoid enemy fire, and shoot enemies to maximize score." \
#   --max-steps-per-episode 3600 \
#   --max-episodes 100 \
#   --val-fraction 0.2 \
#   --num-envs 64 \
#   --image-writer-workers 16 \
#   --row-buffer-size 8192 \
#   --image-source render

# Demon attack jitter latency
# python scripts/export_sf_teacher_dataset.py \
#   --env-name demon_attack \
#   --checkpoint-root checkpoints \
#   --checkpoint-seed 0 \
#   --output-dir teacher_datasets/demon_attack_jitter_latency_normal_mean_2_std_2_min_0_max_15 \
#   --dataset-mode jitter_latency \
#   --jitter-latency-mean 2 \
#   --jitter-latency-std 2 \
#   --jitter-latency-min 0 \
#   --jitter-latency-max 15 \
#   --prompt "You are playing Demon Attack from a single game image. Choose exactly one action from: NOOP, FIRE, RIGHT, LEFT, RIGHTFIRE, LEFTFIRE. Survive, avoid enemy fire, and shoot enemies to maximize score." \
#   --mixed-prompt-template "{prompt} The current action latency is {latency} decision steps ({latency_ms} ms). Choose the best next action while accounting for the delay." \
#   --max-steps-per-episode 3600 \
#   --max-episodes 100 \
#   --val-fraction 0.2 \
#   --num-envs 64 \
#   --image-writer-workers 16 \
#   --row-buffer-size 8192 \
#   --image-source render

# python scripts/export_sf_teacher_dataset.py \
#   --env-name demon_attack \
#   --checkpoint-root checkpoints \
#   --checkpoint-seed 0 \
#   --output-dir teacher_datasets/demon_attack_jitter_latency_uniform_min_0_max_4 \
#   --dataset-mode jitter_latency \
#   --jitter-latency-distribution uniform \
#   --jitter-latency-min 0 \
#   --jitter-latency-max 4 \
#   --prompt "You are playing Demon Attack from a single game image. Choose exactly one action from: NOOP, FIRE, RIGHT, LEFT, RIGHTFIRE, LEFTFIRE. Survive, avoid enemy fire, and shoot enemies to maximize score." \
#   --mixed-prompt-template "{prompt} The current action latency is {latency} decision steps ({latency_ms} ms). Choose the best next action while accounting for the delay." \
#   --max-steps-per-episode 3600 \
#   --max-episodes 100 \
#   --val-fraction 0.2 \
#   --num-envs 64 \
#   --image-writer-workers 16 \
#   --row-buffer-size 8192 \
#   --image-source render

# # DEMON ATTACK: mixed-latency teacher dataset
# python scripts/export_sf_teacher_dataset.py \
#   --env-name demon_attack \
#   --checkpoint-root /workspace/latency-checkpoints/checkpoints \
#   --output-dir /workspace/teacher_datasets/demon_attack_mixed_latency \
#   --dataset-mode mixed_latency \
#   --prompt "You are playing Demon Attack from a single game image. Choose exactly one action from: NOOP, FIRE, RIGHT, LEFT, RIGHTFIRE, LEFTFIRE. Survive, avoid enemy fire, and shoot enemies to maximize score." \
#   --mixed-prompt-template "{prompt} Your actions take {latency_ms} milliseconds to take effect. Choose the best next action while accounting for the delay." \
#   --max-steps-per-episode 3600 \
#   --latency-start 0 \
#   --latency-end 7 \
#   --episodes-per-latency 50 \
#   --val-episodes-per-latency 10 \
#   --num-envs 64 \
#   --image-writer-workers 16 \
#   --row-buffer-size 8192 \
#   --image-source observation \
#   --push-to-hub \
#   --hf-token "$HF_TOKEN" \
#   --hf-max-shard-size 500MB \
#   --hf-repo-id talha1503/demon_attack_mixed_latency_parquet
