# MIKASA InterceptGrabFast H1 cookbook

这份 cookbook 用于复现已经完成的 `InterceptGrabFast-VLA-v0` zero-delay H1
baseline。完整的同步、环境安装、审计和 gate 命令见
[MIKASA 执行手册](../../program/MIKASA.md)；本文只保留复现所需的 setting、入口和结果。

## Experiment setting

- Environment: `InterceptGrabFast-VLA-v0`, H1, `pd_ee_delta_pose`，7D action，20 Hz（每步
  50 ms）。本实验是 zero-delay，不包含 fixed-delay、realtime profile 或 latency-aware training。
- Teacher: 49D privileged-state PPO，`normalized_dense` reward；训练时采样，rollout 和评估时用
  deterministic mean action。trainer gate 要求 `16/16`，随后 canonical gate 要求至少 `95/100`。
- Dataset: 275 条全部成功的 deterministic teacher trajectories；前 250 条 train，后 25 条
  validation。
- Student: StarVLA `QwenOFT`（Qwen3-VL-4B-Instruct + MLP action head），只预测当前一帧 7D
  action，`include_state=false`。
- SFT: 单卡 DeepSpeed ZeRO-2，global batch 64，5,000 optimizer steps，held-out action-loss
  eval 每 500 steps。VLA gate 是 canonical zero-delay 50 episodes 中至少 `25/50` 次
  `success_once`。

vendor 基线版本：MIKASA `16634db18bef08128ed79346469c86fc12169aed`、StarVLA
`823cd2cdd4fc3934c9d89ac96c03c23b07cd8004`。本仓库固定的集成 submodule revision
分别为 MIKASA `cf4f96f319022f89c9d7cfbd639d19bc10ed44fb` 和 StarVLA
`0112c3ef366a57ec8b8279555c26271a34332c89`；RLDS nested builder 为
`7b2f4f38a7773b792f6e690f5c838aaea39754f6`。

## Published artifacts

- PPO teacher：[model repo](https://huggingface.co/latency-sensitive-bench/mikasa-robo)，revision
  `38327e16f9b33cf5f827b54a606cb4c22d4dfbfc`，包含 tuned `final_success_ckpt.pt` 和
  canonical strict-eval summary。
- 275 条成功 rollout：[dataset repo](https://huggingface.co/datasets/latency-sensitive-bench/mikasa-robo)，
  revision `28312021c634fa97f8068077dfd6eb0efe6d45ce`，同时提供原始 NPZ 和用于 StarVLA 的
  LeRobot v3 250/25 split。

已有 checkout 可以直接下载：

```bash
hf download latency-sensitive-bench/mikasa-robo \
  --repo-type dataset \
  --revision 28312021c634fa97f8068077dfd6eb0efe6d45ce \
  --local-dir outputs/mikasa/grabfast_h1_dataset_hf
hf download latency-sensitive-bench/mikasa-robo \
  oracle_checkpoints/InterceptGrabFast-VLA-v0/final_success_ckpt.pt \
  --revision 38327e16f9b33cf5f827b54a606cb4c22d4dfbfc \
  --local-dir outputs/mikasa/published_teacher
```

## Hardware and software prerequisites

实验在 `xinyuan@204.12.163.107` 上完成，远端 hostname 为
`computeinstance-u00mv1dams6a8gc51r`：

- 1× NVIDIA RTX PRO 6000 Blackwell Server Edition，97,887 MiB；driver `580.173.02`，CUDA
  13.0，compute capability `(12, 0)`。
- MIKASA Python 3.10.21，Torch `2.7.1+cu128`，同版本 Vulkan userspace。
- StarVLA Torch `2.11.0+cu128`、DeepSpeed `0.16.9`、Accelerate `1.5.2`、FlashAttention
  `2.8.3.post1`。
- 可访问 Hugging Face 和 W&B；`.env` 只在远端 shell 中加载，不能打印或上传。

需要 CUDA GPU、Conda、`uv`、Hugging Face 权限、W&B entity，以及可用的 Qwen3-VL-4B 和
Bridge/RT-1 initialization checkpoint。不要自行添加依赖；MIKASA 环境使用仓库已有
`scripts/mikasa/install.sh`，StarVLA 环境使用其已有 bootstrap。Blackwell 的 Torch/Vulkan
处理、submodule 初始化和密钥同步按 [MIKASA 执行手册 Phase 0](../../program/MIKASA.md#3-phase-0同步服务器密钥和环境) 操作。

## Quick Start

1. Clone 仓库并初始化本实验实际使用的 submodule；superproject 已固定包含 MIKASA/StarVLA
   集成代码的 revision，不需要再应用 patch：

   ```bash
   git clone https://github.com/ZihanWang314/latency-sensitive-bench.git
   cd latency-sensitive-bench
   git submodule update --init \
     third_party/MIKASA-Robo third_party/starVLA \
     third_party/flappy-bird-gymnasium third_party/sample-factory
   git -C third_party/MIKASA-Robo submodule update --init --recursive \
     utils/convert_npz_to_rlds/rlds_dataset_builder
   git -C third_party/MIKASA-Robo/utils/convert_npz_to_rlds/rlds_dataset_builder \
     checkout --detach 7b2f4f38a7773b792f6e690f5c838aaea39754f6
   ```

   如果训练机不是 clone 所在机器，再按完整 runbook 把 checkout 和 `.env` 同步到训练机。
2. 在远端加载 `.env`，执行 `bash scripts/mikasa/install.sh`，完成 MIKASA reset/step smoke 和
   StarVLA import/model smoke。
3. 下载以下固定模型目录：

   ```text
   third_party/starVLA/playground/Pretrained_models/Qwen3-VL-4B-Instruct/
   third_party/starVLA/playground/Pretrained_models/Qwen3VL-OFT-Bridge-RT-1/
   ```

4. 依次执行下面的 PPO → dataset → SFT → canonical eval；每一步的 gate 不通过就停止，不用
   未验收 checkpoint 继续下一步。

远端工作树和环境变量约定：

```bash
cd ~/latency-sensitive-bench-mikasa-h1
export REMOTE_ROOT="$PWD"
export MIKASA_PY="${REMOTE_ROOT}/third_party/MIKASA-Robo/.venv/bin/python"
export PYTHONPATH="${REMOTE_ROOT}:${REMOTE_ROOT}/third_party/MIKASA-Robo"
export NVIDIA_VULKAN_ROOT="${HOME}/.local/nvidia-580.173.02"
export LD_LIBRARY_PATH="${NVIDIA_VULKAN_ROOT}/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export VK_DRIVER_FILES="${NVIDIA_VULKAN_ROOT}/usr/share/vulkan/icd.d/nvidia_icd.json"
set -a
source .env
set +a
```

## PPO teacher

先跑 5M engineering smoke；它的权重不进入 dataset：

```bash
export PPO_ENTRY="${REMOTE_ROOT}/third_party/MIKASA-Robo/mikasa_robo_suite/vla/dataset_collectors/get_dataset_collectors_ckpt.py"
export CUDA_VISIBLE_DEVICES=0

"${MIKASA_PY}" "${PPO_ENTRY}" \
  --env-id InterceptGrabFast-VLA-v0 \
  --total-timesteps 5000000 \
  --track \
  --wandb-entity "${WANDB_ENTITY}"
```

正式 run 从头开始，默认上限为 150M。默认配置失败后只按 full runbook 规定的顺序尝试
`learning_rate=1e-4`、`6e-4`，再尝试较优 learning rate 的 `update_epochs=8`。本次最终采用
`learning_rate=1e-4`、`update_epochs=8`，命令仍使用相同入口并增加这两个参数：

```bash
"${MIKASA_PY}" "${PPO_ENTRY}" \
  --env-id InterceptGrabFast-VLA-v0 \
  --total-timesteps 150000000 \
  --learning-rate 0.0001 \
  --update-epochs 8 \
  --track \
  --wandb-entity "${WANDB_ENTITY}"
```

对 trainer 产出的唯一候选 checkpoint 做 canonical 100-episode strict eval。固定 seeds 从
`4242424242` 连续递增；必须同时通过 `16/16` trainer gate 和 `95/100` canonical gate：

```bash
export PPO_CKPT=/absolute/path/to/final_success_ckpt.pt
export PPO_EVAL_DIR="${REMOTE_ROOT}/outputs/mikasa/ppo_teacher_strict_eval"
mkdir -p "${PPO_EVAL_DIR}"

"${MIKASA_PY}" scripts/mikasa/evaluate.py \
  --policy ppo \
  --checkpoint "${PPO_CKPT}" \
  --episodes 100 \
  --output-dir "${PPO_EVAL_DIR}"

"${MIKASA_PY}" -c \
  'import json,sys; d=json.load(open(sys.argv[1])); n=sum(d["successes"]); print(f"{n}/100"); assert n >= 95' \
  "${PPO_EVAL_DIR}/summary.json"
```

本次 tuned teacher 在 12,288,000 steps 触发官方早停，trainer 为 `16/16`，canonical 为
`100/100`，mean return `36.819`。默认 candidate 的 canonical 结果为 `72/100`，不能用于
采集。

## Rollout dataset: 250 train / 25 validation

把通过 strict gate 的 checkpoint 放进独立 `STRICT_CKPT_ROOT`，避免 collector 选到其他候选；
然后用已有 collector 请求恰好 275 条成功轨迹：

```bash
export DATASET_ROOT="${REMOTE_ROOT}/outputs/mikasa/grabfast_h1_dataset"
export STRICT_CKPT_ROOT="${REMOTE_ROOT}/outputs/mikasa/strict_teacher"
mkdir -p "${STRICT_CKPT_ROOT}/oracle_checkpoints/InterceptGrabFast-VLA-v0"
cp "${PPO_CKPT}" \
  "${STRICT_CKPT_ROOT}/oracle_checkpoints/InterceptGrabFast-VLA-v0/final_success_ckpt.pt"

"${MIKASA_PY}" \
  third_party/MIKASA-Robo/mikasa_robo_suite/vla/dataset_collectors/get_mikasa_robo_datasets.py \
  --env-id InterceptGrabFast-VLA-v0 \
  --path-to-save-data "${DATASET_ROOT}" \
  --ckpt-dir "${STRICT_CKPT_ROOT}" \
  --num-train-data 275
```

collector 只计入完整成功 batch；action 与 controller 一致地 clip 到 `[-1, 1]`，episode 在
success/done 处截断。先按 [runbook Phase 2](../../program/MIKASA.md#5-phase-2采集并转换-25025-dataset) 的
NPZ audit 检查 metadata、275 个文件、275 个 unique seeds、success 标记、shape 和 action 范围，
再执行原有 NPZ → RLDS → LeRobot converter，并用固定 split：

```text
train: train[:250]  -> local/intercept_grab_fast_vla_v0_h1_train
val:   train[250:275] -> local/intercept_grab_fast_vla_v0_h1_val
fps: 20
```

最后运行已有的 `scripts/mikasa/prepare_starvla_dataset.py` 生成两个 dataset 的
`info.json`、`stats.json`、`modality.json` 和 mixture JSON。Phase 2 gate 是恰好 250/25
episodes，且 StarVLA mixture 使用上面两个准确名称。

本次 audit 结果：275 episodes、9,859 transitions，episode length
`mean/min/max = 35.851/24/57`；train 为 250 episodes/8,911 frames，validation 为
25/948 frames，20 FPS。

## StarVLA QwenOFT SFT

保持单卡、ZeRO-2 和 global batch 64。按 OOM 顺序尝试 `(microbatch, accumulation)`
`(64,1) → (32,2) → (16,4) → (8,8) → (4,16)`；OOM 后用新的 W&B run id 从 step 0 重启，
不要续 incomplete checkpoint，也不要降低 global batch。本次成功配置为 `8/8/64`，并设置
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`。

```bash
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MICRO_BATCH=8
export ACC_STEPS=8
export TRAIN_NAME=local/intercept_grab_fast_vla_v0_h1_train
export VAL_NAME=local/intercept_grab_fast_vla_v0_h1_val
export VLA_ROOT="${REMOTE_ROOT}/third_party/starVLA"
export VLA_DATA_ROOT="${DATASET_ROOT}/data_lerobot"
export INIT_CKPT="${VLA_ROOT}/playground/Pretrained_models/Qwen3VL-OFT-Bridge-RT-1/checkpoints/steps_5000_pytorch_model.pt"
export RUN_ID=mikasa_grabfast_h1_gb64_mb8_ga8_expandable

cd "${VLA_ROOT}"
accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes 1 \
  starVLA/training/train_starvla.py \
  --config_yaml examples/MIKASA/train_files/starvla_qwenoft_intercept_grab_fast_h1.yaml \
  --run_id "${RUN_ID}" \
  --wandb_project starvla_mikasa \
  --wandb_entity "${WANDB_ENTITY}" \
  --datasets.vla_data.data_root_dir "${VLA_DATA_ROOT}" \
  --datasets.vla_data.data_mix "${TRAIN_NAME}" \
  --datasets.vla_data.eval_data_mix "${VAL_NAME}" \
  --datasets.vla_data.custom_mixtures_path "${VLA_DATA_ROOT}/mikasa_intercept_grab_fast_h1.json" \
  --datasets.vla_data.per_device_batch_size "${MICRO_BATCH}" \
  --trainer.gradient_accumulation_steps "${ACC_STEPS}" \
  --trainer.max_train_steps 5000 \
  --trainer.profile_timing.enabled true \
  --trainer.pretrained_checkpoint "${INIT_CKPT}" \
  --checkpoint.save_training_state true \
  --checkpoint.save_pt_file true
```

用 run ID、`config.full.yaml` 和 stdout 共同确认 `8 × 8 = 64`；`profile_timing` 用于让 W&B
记录吞吐，held-out action-loss eval 每 500 steps，不是环境成功率。本次训练完成 `5000/5000`，
actual epochs `35.91`，W&B run 为
[`3qt4k7nc`](https://wandb.ai/saberrr-zju/starvla_mikasa/runs/3qt4k7nc)。最终文件位于：

```text
third_party/starVLA/results/Checkpoints/mikasa_grabfast_h1_gb64_mb8_ga8_expandable/config.full.yaml
third_party/starVLA/results/Checkpoints/mikasa_grabfast_h1_gb64_mb8_ga8_expandable/checkpoints/steps_5000_state/
third_party/starVLA/results/Checkpoints/mikasa_grabfast_h1_gb64_mb8_ga8_expandable/checkpoints/steps_5000_pytorch_model.pt
```

## Canonical zero-delay evaluation

启动 StarVLA policy server，使用上面的 final `.pt`，端口固定为 `10093`。另一个 shell 使用
MIKASA env 运行固定 50 seeds（`4242424242..4242424291`）：

```bash
export VLA_CKPT="${VLA_ROOT}/results/Checkpoints/${RUN_ID}/checkpoints/steps_5000_pytorch_model.pt"
cd "${VLA_ROOT}"
python deployment/model_server/server_policy.py \
  --ckpt_path "${VLA_CKPT}" \
  --port 10093 \
  --use_bf16
```

```bash
export VLA_EVAL_DIR="${REMOTE_ROOT}/outputs/mikasa/vla_h1_zero_delay_eval"
mkdir -p "${VLA_EVAL_DIR}"

"${MIKASA_PY}" scripts/mikasa/evaluate.py \
  --policy vla \
  --host 127.0.0.1 \
  --port 10093 \
  --episodes 50 \
  --output-dir "${VLA_EVAL_DIR}"

"${MIKASA_PY}" -c \
  'import json,sys; d=json.load(open(sys.argv[1])); n=sum(d["successes"]); print(f"{n}/50"); assert n >= 25' \
  "${VLA_EVAL_DIR}/summary.json"
```

本次 canonical 结果为 `27/50`（54%），mean return `22.326`，returns 全部 finite；
`outputs/mikasa/vla_h1_zero_delay_eval/summary.json` 是验收记录。

## Expected results

| Stage | Expected gate/result |
| --- | --- |
| MIKASA runtime smoke | GPU reset + one step 成功，action shape 最后一维为 7 |
| PPO smoke | 训练、eval、W&B 写入完成；smoke checkpoint 不用于 dataset |
| PPO teacher | trainer `16/16` 且 canonical `≥95/100` |
| Rollout | 275 successful episodes；train/val 为 250/25 |
| SFT | 5,000 optimizer steps，global batch 64，final model/state 存在 |
| VLA baseline | canonical zero-delay `≥25/50`；本次为 `27/50` |

## Troubleshooting

- **网络或路径错误：** 使用 `xinyuan@204.12.163.107` 和远端 worktree；不要把本地 GPU
  结果当作本实验结果。同步、`.env` 权限和 submodule 处理见 [Phase 0](../../program/MIKASA.md#3-phase-0同步服务器密钥和环境)。
- **Blackwell architecture/Vulkan error：** 先确认 capability `(12, 0)`；MIKASA 使用
  Torch `2.7.1` cu128 和同 driver 的 user-space Vulkan。不要无条件覆盖其他机器的 Torch。
- **PPO 未过 gate：** 不要采集 rollout；只按 full runbook 的有限调参顺序重新跑，并对每个
  `final_success_ckpt.pt` 做独立 100-episode eval。
- **SFT OOM：** 结束失败进程，按 OOM ladder 降 microbatch、增加 accumulation，使用新的
  W&B run id 从 step 0 重启；global batch 始终为 64。
- **找不到 final artifact：** 训练只有完成 5,000 steps 才算通过；不要把手工中断的状态当作
  可续训 checkpoint。确认 `config.full.yaml`、training state、final `.pt` 和 W&B run 同属一个
  `RUN_ID`。
- **loss 与 success 混淆：** SFT held-out action loss 不是环境 success rate；最终判断只看
  canonical 50-episode `success_once` JSON。
