# Data Collection
```bash
git clone --recurse-submodules https://github.com/ZihanWang314/latency-sensitive-bench.git
cd latency-sensitive-bench

# --- env setup ---
bash install.sh --tier use --model openvla --env all
conda activate latency

# --- small model checkpoints download ---
# flappy
hf download latency-sensitive-bench/latency-checkpoints \
  --repo-type model \
  --include "checkpoints/flappy_sf_same_latency_l0_seed0/**" \
  --local-dir ./
  
# demon attack
hf download latency-sensitive-bench/latency-checkpoints \
  --repo-type model \
  --include "checkpoints_old/demon_attack_sf_same_latency_l0_seed0/**" \
  --local-dir ./

# --- collect data ---
# flappy
# cost about 14KMiB GPU memories
MAX_EPISODES=200 \
MAX_STEPS=3600 \
VAL_FRACTION=0.1 \
FILTER_PRESET=clean_v1 \
DEVICE=cuda \
NUM_ENVS=2048 \
REPLAY_NUM_ENVS=16 \
IMAGE_WRITER_WORKERS=32 \
bash scripts/bash_scripts/run_sf_teacher_dataset_export.sh \
  checkpoints/flappy_sf_same_latency_l0_seed0 \
  outputs/teacher_datasets/flappy_fix_latency_0_200ep
  
# demon attack
MAX_EPISODES=200 \
MAX_STEPS=3600 \
VAL_FRACTION=0.1 \
FILTER_PRESET=clean_v1 \
DEVICE=cuda \
NUM_ENVS=128 \
REPLAY_NUM_ENVS=16 \
IMAGE_WRITER_WORKERS=32 \
bash scripts/bash_scripts/run_sf_teacher_dataset_export.sh \
  checkpoints/demon_attack_sf_same_latency_l0_seed0 \
  outputs/teacher_datasets/demon_attack_fix_latency_0_200ep
  
# --- push to hf ---
hf upload latency-sensitive-bench/demon_attack_200ep \
  outputs/teacher_datasets/demon_attack_fix_latency_0_200ep \
  demon_attack_fix_latency_0_200ep \
  --repo-type dataset \
  --exclude "sample_factory_init.log" \
  --commit-message "Upload demon attack fixed latency 0 teacher dataset"
```

# SFT Training
```bash
# Run from the latency-sensitive-bench repository root.
bash install.sh \
  --tier dev \
  --model openvla \
  --env all

conda activate starvla_rl_games_openvla
cd third_party/starVLA

# --- data prepare ---
# flappy
hf download latency-sensitive-bench/flappy_200ep \
  --repo-type dataset \
  --include "flappy_fix_latency_0_200ep/**" \
  --local-dir /workspace/starVLA/data/raw/flappy_fix_latency_0_200ep
  
python examples/rl_games/data_conversion/convert_flappy_to_starvla_lerobot.py \
  --dataset-name data/raw/flappy_fix_latency_0_200ep \
  --source-fps 30 \
  --obs-stride-raw-frames 1 \
  --source-latency-column latency_raw_frames \
  --target-latency-unit raw_frames \
  --output-dir data/flappy_fix_latency_0_200ep/flappy_train__bridge \
  --action-carrier bridge \
  --force
  
# demon attack
hf download latency-sensitive-bench/demon_attack_200ep \
  --repo-type dataset \
  --include "demon_attack_fix_latency_0_200ep/**" \
  --local-dir /workspace/starVLA/data/raw

python examples/rl_games/data_conversion/convert_demon_attack_to_starvla_lerobot.py \
  --dataset-name data/raw/demon_attack_fix_latency_0_200ep \
  --source-fps 15 \
  --obs-stride-raw-frames 4 \
  --source-latency-column latency_raw_frames \
  --target-latency-unit raw_frames \
  --output-dir data/demon_attack_fix_latency_0_200ep/demon_attack_train__bridge \
  --action-carrier bridge \
  --force 

# --- checkpoints prepare ---
hf download Qwen/Qwen3-VL-4B-Instruct  \
  --local-dir playground/Pretrained_models/Qwen3-VL-4B-Instruct
  
hf download StarVLA/Qwen3VL-OFT-Bridge-RT-1  \
  --local-dir playground/Pretrained_models/Qwen3VL-OFT-Bridge-RT-1
  
# --- fix latency training ---
bash commands/train_<env>_attack_openvla.sh
  
# --- checkpoints upload ---
# flappy
hf upload latency-sensitive-bench/openvla_flappy_fix_latency_200ep \
  results/Checkpoints/flappy_fix_latency_0_200ep \
  ./flappy_fix_latency_0_200ep \
  --exclude "wandb/**" \
  --repo-type model
  
# demon attack
hf upload latency-sensitive-bench/openvla_demon_attack_fix_latency_200ep \
  results/Checkpoints/demon_attack_fix_latency_0_200ep \
  ./demon_attack_fix_latency_0_200ep \
  --exclude "wandb/**" \
  --repo-type model
```
# Eval
```bash
# follow data collection pipeline to setup env

# flappy
python -m latency_bench.run \
  --config configs/experiments/flappy/starvla_openvla_flappy_fix_latency.yaml

# demon attack
python -m latency_bench.run \
  --config configs/experiments/demon_attack/starvla_demon_attack_fix_latency.yaml
```
