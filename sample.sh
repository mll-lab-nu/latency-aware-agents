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