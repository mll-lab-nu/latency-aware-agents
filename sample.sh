## FLAPPY BIRD
python scripts/generate_sample_factory_same_latency_sweep.py \
  --envs flappy \
  --seeds 0 \
  --flappy-latencies 0 \
  --generated-root /workspace/one_run_flappy \
  --train-dir /workspace/sample_factory_runs \
  --output-root /workspace/latency_sweeps \
  --device gpu \
  --restart-behavior resume \
  --eval-episodes 20

python -m training.train --config /workspace/one_run_flappy/flappy/latency_0/seed_0/train.yaml
python -m training.train --config /workspace/one_run_flappy/flappy/latency_0/seed_0/eval.yaml

## DEMON ATTACK 
python scripts/generate_sample_factory_same_latency_sweep.py \
  --envs demon_attack \
  --seeds 0 \
  --demon-attack-latencies 0 \
  --generated-root /workspace/one_run_demon \
  --train-dir /workspace/sample_factory_runs \
  --output-root /workspace/latency_sweeps \
  --device gpu \
  --restart-behavior resume \
  --eval-episodes 20

python -m training.train --config /workspace/one_run_demon/demon_attack/latency_0/seed_0/train.yaml
python -m training.train --config /workspace/one_run_demon/demon_attack/latency_0/seed_0/eval.yaml


# DEADLY CORRIDOR 
python scripts/generate_sample_factory_same_latency_sweep.py \
  --envs deadly_corridor \
  --seeds 0 \
  --deadly-corridor-latencies 0 \
  --generated-root /workspace/one_run \
  --train-dir /workspace/sample_factory_runs \
  --output-root /workspace/latency_sweeps \
  --device gpu \
  --restart-behavior resume \
  --eval-episodes 20

python -m training.train --config /workspace/one_run/deadly_corridor/latency_0/seed_0/train.yaml
python -m training.train --config /workspace/one_run/deadly_corridor/latency_0/seed_0/eval.yaml


## TEACHER DATASET EXPORTS

# DEADLY CORRIDOR: zero-latency teacher dataset
python scripts/export_sf_teacher_dataset.py \
  --env-name deadly_corridor \
  --checkpoint-root /workspace/latency-checkpoints/checkpoints \
  --output-dir /workspace/teacher_datasets/deadly_corridor_zero_latency \
  --dataset-mode zero_latency \
  --prompt "You are playing Deadly Corridor in VizDoom. Your goal is to survive the corridor, fight enemies, and reach the green armor vest at the far end. The available actions are: MOVE_FORWARD, MOVE_BACKWARD, MOVE_LEFT, MOVE_RIGHT, TURN_LEFT, TURN_RIGHT, and ATTACK. Choose the best next action from the current visual observation." \
  --max-steps-per-episode 2100 \
  --max-episodes 100 \
  --val-fraction 0.1 \
  --push-to-hub \
  --hf-private \
  --hf-token "$HF_TOKEN" \
  --hf-max-shard-size 500MB \
  --hf-repo-id talha1503/deadly_corridor_zero_latency_parquet

# DEADLY CORRIDOR: mixed-latency teacher dataset
python scripts/export_sf_teacher_dataset.py \
  --env-name deadly_corridor \
  --checkpoint-root /workspace/latency-checkpoints/checkpoints \
  --output-dir /workspace/teacher_datasets/deadly_corridor_mixed_latency \
  --dataset-mode mixed_latency \
  --prompt "You are playing Deadly Corridor in VizDoom. Your goal is to survive the corridor, fight enemies, and reach the green armor vest at the far end. The available actions are: MOVE_FORWARD, MOVE_BACKWARD, MOVE_LEFT, MOVE_RIGHT, TURN_LEFT, TURN_RIGHT, and ATTACK." \
  --mixed-prompt-template "{prompt} Your actions take {latency_ms} milliseconds to take effect. Choose the best next action while accounting for the delay." \
  --max-steps-per-episode 2100 \
  --latency-start 0 \
  --latency-end 5 \
  --episodes-per-latency 50 \
  --val-episodes-per-latency 5 \
  --push-to-hub \
  --hf-private \
  --hf-token "$HF_TOKEN" \
  --hf-max-shard-size 500MB \
  --hf-repo-id talha1503/deadly_corridor_mixed_latency_parquet
