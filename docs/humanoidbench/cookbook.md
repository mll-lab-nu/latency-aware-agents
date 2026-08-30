# HumanoidBench Balance Simple cookbook

这是 Balance Simple 的正式 GR00T workflow。公开数据集位于
[latency-sensitive-bench/humanoidbench](https://huggingface.co/datasets/latency-sensitive-bench/humanoidbench)，训练入口会下载命令中指定的 revision；不要把本机绝对路径写进配置或数据集。
最终模型发布到同名的 [model repo](https://huggingface.co/latency-sensitive-bench/humanoidbench)。

## Formal setup

- Task: `g1_sonic_balance_simple`，50 Hz，prompt 为 `Keep the G1 humanoid balanced on the support board.`
- Dataset: `balance_simple/lerobot` 有 90 个 train episodes、86,048 frames；`balance_simple/lerobot__val` 有 10 个 validation episodes、9,736 frames。两者均为 LeRobot v2 image-parquet。
- Schema: `observation.state` 是 198D `float32`，`action` 是 43D `float32` native absolute joint target，`observation.image` 是 256×256 RGB head-camera video。
- GR00T recipe: H40 `repeat-current`、每个 50 Hz frame 重新规划、只执行 chunk index 0；fixed Gaussian seed 0、one-step flow、action-head dropout 0。
- Compute: 2 GPUs (`CUDA_VISIBLE_DEVICES=0,1`)、global batch 128、15,000 optimizer steps、250 warmup steps、每 500 steps 保存并评估。

每一行是一个时间步和一个当前 action；正式 runner 用 `[0] * 40` 的 delta indices，把该 action 重复为 H40 label。

## Fresh machine

```bash
git clone --recurse-submodules https://github.com/ZihanWang314/latency-sensitive-bench.git
cd latency-sensitive-bench
bash install.sh --model official_gr00t
```

完成 Hugging Face（需可访问 GR00T 相关 gated model，并能写入 `latency-sensitive-bench` organization）和 W&B 登录：

```bash
uv run --project third_party/Isaac-GR00T hf auth login
uv run --project third_party/Isaac-GR00T wandb login
```

第一次运行必须保持在线。不要设置 `HF_HUB_OFFLINE=1` 或 `TRANSFORMERS_OFFLINE=1`；训练入口需要下载数据、基础模型和场景资产。

## Train and resume

Fresh run 使用新的 W&B run id：

```bash
HF_DATASET_REVISION=<dataset-commit> WANDB_RUN_ID=<new-run-id> \
  bash scripts/official_gr00t/train_balance_simple_15k.sh
```

入口会下载 `balance_simple/lerobot` 和 `balance_simple/lerobot__val`，并使用上面的固定 recipe。断点续跑时使用同一个 W&B run id；`--start-step` 是要生成的下一个 500-step checkpoint：

```bash
HF_DATASET_REVISION=<same-dataset-commit> WANDB_RUN_ID=<same-run-id> \
  bash scripts/official_gr00t/train_balance_simple_15k.sh --start-step 5500
```

例如已有 `checkpoint-5000` 时从 `5500` 开始。恢复前应确认对应 output directory 中存在已有 checkpoint；不要 resume 已耗尽 15k cosine schedule 的 `checkpoint-15000`。

step 15000 的 validation、closed-loop evaluation 和 W&B 写入全部成功后，入口自动把最终 inference bundle 上传到 model repo 的 `balance_simple/checkpoint-15000/`。上传只包含 GR00T 加载需要的 config、processor、statistics 和 safetensors；DeepSpeed optimizer state 继续保留在本地，不作为模型发布物。

## Evaluate

先跑零延迟 simulated baseline：

```bash
MUJOCO_GL=egl uv run --project . python -m latency_bench.run \
  --config configs/examples/gymnasium/g1_sonic_balance_simple/official_gr00t_eval_simulated.yaml
```

然后再跑 realtime 配置：

```bash
MUJOCO_GL=egl uv run --project . python -m latency_bench.run \
  --config configs/examples/gymnasium/g1_sonic_balance_simple/official_gr00t_eval.yaml
```

两个配置中的 `policy.model_path` 必须指向要评估的 checkpoint；训练入口产生的正式输出目录为 `outputs/official_gr00t/g1_balance_simple_obs198_h40_repeat_fixedz0_onestep_dropout0_frompretrained_bs128_lr1e4_15k_periodic500/`。

## Regenerate or publish the dataset

如果需要重新生成 teacher data，保持 90/10 split、`--split-seed 0` 和 198D teacher observation：

```bash
MUJOCO_GL=egl uv run --project . python scripts/teachers/export_g1_sonic_gr00t.py \
  --output-root outputs/official_gr00t/teachers \
  --teacher g1_balance_simple_raw_ramp_fasttd3_25m_43d_teacher_obs198 \
  --max-episodes 100 --val-fraction 0.1 --split-seed 0 \
  --num-envs 16 --replay-num-envs 16 --device cuda --deterministic \
  --image-writer-workers 16 --row-buffer-size 8192 --force \
  --balance-teacher-observation-state
```

发布时直接用 Hugging Face 原生 CLI，不另建 uploader：

```bash
SOURCE=outputs/official_gr00t/teachers/g1_balance_simple_raw_ramp_fasttd3_25m_43d_teacher_obs198
uv run --project third_party/Isaac-GR00T hf repo create \
  latency-sensitive-bench/humanoidbench --repo-type dataset --exist-ok
uv run --project third_party/Isaac-GR00T hf upload \
  latency-sensitive-bench/humanoidbench \
  "$SOURCE/lerobot" balance_simple/lerobot --repo-type dataset \
  --exclude manifest.json
uv run --project third_party/Isaac-GR00T hf upload \
  latency-sensitive-bench/humanoidbench \
  "$SOURCE/lerobot__val" balance_simple/lerobot__val --repo-type dataset \
  --exclude manifest.json
uv run --project third_party/Isaac-GR00T hf upload \
  latency-sensitive-bench/humanoidbench \
  docs/humanoidbench/hf_dataset_card.md README.md --repo-type dataset
```

不要向 dataset repo 上传 raw rollout、model checkpoint、W&B 文件或含本机绝对路径的额外 manifest。
