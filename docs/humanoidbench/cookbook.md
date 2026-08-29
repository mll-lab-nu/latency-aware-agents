### supported tasks for humanoid bench



### rollout data

Each wrapper exports 100 episodes:

The wrappers use `--force`, replacing the existing `lerobot` and
`lerobot__val` conversion outputs for that teacher.

```bash
bash scripts/bash_scripts/rollout_data/humanoid_balance_simple.sh
bash scripts/bash_scripts/rollout_data/humanoid_hurdle.sh
bash scripts/bash_scripts/rollout_data/humanoid_run.sh
```

### dataset/output mapping

Use one row with the shared training command below; the training parameters are
the same for balance, hurdle, and run.

| task | dataset path (`DATASET_PATH`) | output path (`OUTPUT_DIR`) |
| --- | --- | --- |
| balance | `outputs/official_gr00t/teachers/g1_balance_simple_raw_ramp_fasttd3_25m_43d_v2/lerobot` | `outputs/official_gr00t/g1_balance_simple_gr00t_43d_v2` |
| hurdle | `outputs/official_gr00t/teachers/g1_hurdle_sonic_residual_h025_gamma099_1m_43d_v2/lerobot` | `outputs/official_gr00t/g1_hurdle_gr00t_43d_v2` |
| run | `outputs/official_gr00t/teachers/g1_run_sonic_reference_playback_43d_v2/lerobot` | `outputs/official_gr00t/g1_run_gr00t_43d_v2` |

### generate validation statistics

```bash
VAL_DATASET_PATH=outputs/official_gr00t/teachers/g1_balance_simple_raw_ramp_fasttd3_25m_43d_v2/lerobot__val

uv run --project third_party/Isaac-GR00T python \
    third_party/Isaac-GR00T/gr00t/data/stats.py \
    --dataset-path "$VAL_DATASET_PATH" \
    --embodiment-tag NEW_EMBODIMENT \
    --modality-config-path latency_bench/policy/official_gr00t_modality.py
```

### training

```bash
export HF_HOME=/pfs/users/lxy/latency-sensitive-bench/outputs/official_gr00t/hf_cache
export HF_HUB_CACHE="$HF_HOME/hub"
export HUGGINGFACE_HUB_CACHE="$HF_HUB_CACHE"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export GROOT_PATCH_MISTRAL=1
export WANDB_MODE=online
```

```bash
DATASET_PATH=outputs/official_gr00t/teachers/g1_balance_simple_raw_ramp_fasttd3_25m_43d_v2/lerobot
OUTPUT_DIR=outputs/official_gr00t/g1_balance_simple_gr00t_43d_v2

CUDA_VISIBLE_DEVICES=0,1 uv run --project third_party/Isaac-GR00T torchrun \
    --nproc_per_node=2 \
    --master_port=29501 \
    third_party/Isaac-GR00T/gr00t/experiment/launch_finetune.py \
    --base-model-path outputs/official_gr00t/models/GR00T-N1.7-3B \
    --dataset-path "$DATASET_PATH" \
    --embodiment-tag NEW_EMBODIMENT \
    --modality-config-path latency_bench/policy/official_gr00t_modality.py \
    --num-gpus 2 \
    --output-dir "$OUTPUT_DIR" \
    --max-steps 4000 \
    --global-batch-size 64 \
    --dataloader-num-workers 8 \
    --gradient-accumulation-steps 1 \
    --state-dropout-prob 0.0 \
    --max-state-dim 135 \
    --save-steps 1000 \
    --save-total-limit 4 \
    --tune-visual \
    --use-wandb \
    --optim adamw_torch_fused \
    --load-bf16 \
    --no-backbone-trainable-params-fp32
```




可以分成两个口径看：

1. 数据集本身有多少样本：看 Parquet 行数。
2. GR00T 实际能构造多少训练样本：还要考虑 40-step action horizon。

### 1. 一行是什么？

在当前 LeRobot Parquet 中，一行对应一个时间步，也就是：

当前 observation.state
当前 observation.image
当前 action
当前 reward

不是一个标量 action，而是一个 43 维 action 向量：

action.shape = (43,)
observation.state.shape = (135,)

当前数据是 50 Hz，所以一行对应约 20 ms。

GR00T 的 40-step horizon 不代表 Parquet 每行存 40 个 action。实际是：

第 t 行的 state/image
→ 监督目标是第 t 到 t+39 行的 40 个 action

所以每个训练起点需要连续 40 行。每个 episode 最后 39 行不能作为新的训练起点。

有效训练起点数量：

有效起点 = episode 行数 - 40 + 1

### evaluation

0 latency:

```bash
MUJOCO_GL=egl uv run --project . python -m latency_bench.run --config configs/examples/gymnasium/g1_sonic_balance_simple/official_gr00t_eval_simulated.yaml
MUJOCO_GL=egl uv run --project . python -m latency_bench.run --config configs/examples/gymnasium/g1_sonic_hurdle/official_gr00t_eval_simulated.yaml
MUJOCO_GL=egl uv run --project . python -m latency_bench.run --config configs/examples/gymnasium/g1_sonic_run/official_gr00t_eval_simulated.yaml
```

realtime:

```bash
MUJOCO_GL=egl uv run --project . python -m latency_bench.run --config configs/examples/gymnasium/g1_sonic_balance_simple/official_gr00t_eval.yaml
```
