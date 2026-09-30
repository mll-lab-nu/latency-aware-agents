# MIKASA InterceptGrabFast H8 teacher baseline

本文冻结 MIKASA `InterceptGrabFast-VLA-v0` 的 H8 action-chunk 数据、训练和评测合同。
H1 历史基线仍以 [MIKASA](MIKASA.md) 和 [PROGRESS](PROGRESS.md) 为准。

本页前半部分是 **7D proprio、H8×7、无 prefix 的 baseline 合同**；后半部分
“Conditioned H8”保留 **71D v2** 来源与结果。63D v3 的输入修复说明见
[Fixed-2 修复方案](MIKASA_FIXED2_PREFIX_REPAIR.md)，后续四档重采集、训练、评测和
阶段发布协议见 [prefix63 重跑协议](MIKASA_PREFIX63_RESTART.md)。各版本的 state、
normalization、checkpoint 和评测证据应成套读取，最新核验状态见 [PROGRESS](PROGRESS.md)。

## 数据合同

- 任务键：`mikasa_h8`；原 `mikasa` H1 合同不变。
- H1 数据源固定为
  [`benchmark-datasets@5421c9d`](https://huggingface.co/datasets/latency-sensitive-bench/benchmark-datasets/commit/5421c9d272d7a939bb04045b047af84519c1fba8)
  下的 `zero-latency/mikasa-intercept-grab-fast/teacher-rollouts-h1-state-v1`。
- train：250 episodes、8,911 anchors、64,288 个有效 action labels。
- validation：25 episodes、948 anchors、6,884 个有效 action labels。
- 每个 anchor 使用 `observation_t -> [action_t, ..., action_t+7]`；episode 尾部补零，
  `action_loss_mask` 排除补零项，绝不跨 episode 取 action。
- 合同：`action_horizon=8`、`action_dim=7`、
  `action_key=issued_command_chunk`、`robot_type=latency_mikasa_h8`、
  `format=latency_bench_starvla_h8_v1`。
- action normalization 只使用 mask 为真的元素，三个模型共享同一份数据与统计。

数据由训练入口在首次运行时自动从固定 revision 转换；也可单独执行：

```bash
python -m scripts.tasks.mikasa_intercept_grab_fast.prepare_h8_dataset \
  --output-train /path/to/output/lerobot
```

转换会核对上述 episode、anchor 和有效标签计数，不重新采集 teacher trajectory。

## 执行语义

环境在 reset 后立即开始，reward、episode length 和 60-transition budget 也从此刻开始；
没有首动作冻结或额外 observation phase。这个选择与任务源码的原始定义一致：球在 reset
时带初速度发射，策略必须立即反应。

1. reset 并取得 `observation_0`；
2. 第 0 个 environment transition 立即发生；若此时尚无 action，则执行 noop；
3. 每个 20Hz frame 都尝试提交当前 observation，但只有一个 inference worker；
4. worker 忙时该帧请求直接丢弃而不排队，环境继续执行当前 chunk 的下一项；
5. 新 chunk 到达时，执行 head `current_transition - obs_env_step`，不从 head 0 重放；
6. chunk 用完且尚无新结果时，保持旧 chunk 的最后一项。

评测分成两个 setting：

- `zero`：0ms simulated latency，因此每帧新 chunk 当帧到达并执行 head 0；
- `profile`：使用每个模型自己固定版本的 1×RTX 3090 temporal profile。首个 chunk
  到达前的 noop 是实际 transition，之后 worker 忙时继续消费旧 chunk。

审计覆盖单 worker、每 transition 一次请求尝试、初始 noop 数、首个 chunk 的真实到达/
应用帧、active chunk age 和实际执行的后续 head 数。仓库不再提供
`wait_for_first_action` 开关；需要 controller warm-start 时应定义为独立评测变体，不能
混入本合同。

## 训练

模型固定为 QwenOFT、QwenPI v3、QwenGR00T，均使用 Qwen3-VL-4B、fresh action
head、双相机 RGB、7D proprio、ZeRO-2 和 gradient checkpointing。seed 为 0，
global batch 为 128，训练 5,000 optimizer steps；保存 step-2,500 recovery state 和
完整 step-5,000 model/state，只使用 step-5,000。

该次 baseline 训练采用每模型单卡：物理 GPU 1/2/3 分别运行 QwenOFT/QwenPI v3/QwenGR00T。
每路从 micro batch 128、accumulation 1 开始；只在真实 CUDA OOM 时将 micro batch
减半并相应增加 accumulation，保持 global batch 128。

三个模型的完整入口：

```bash
QWENOFT_PY=/path/to/qwenoft-python \
QWENPI_PY=/path/to/qwenpi-python \
QWENGROOT_PY=/path/to/qwengr00t-python \
MIKASA_PY=/path/to/mikasa-python \
  bash scripts/tasks/mikasa_intercept_grab_fast/run_h8_baselines.sh
```

单模型两步 smoke 示例：

```bash
CUDA_VISIBLE_DEVICES=1 VLA_PY=/path/to/starvla-python \
  bash scripts/tasks/mikasa_intercept_grab_fast/train_h8.sh qwenoft \
  --smoke --num-gpus 1 --micro-batch 128 \
  --inference-device cuda:1 --simulator-device cuda:0 \
  --eval-cuda-visible-devices 0,1
```

完整入口在训练阶段只暴露对应的物理 GPU 1/2/3；评测子进程将可见卡重排为
`0,训练卡`，因此 MIKASA simulator 使用逻辑 `cuda:0`（物理 GPU 0），模型 inference
使用逻辑 `cuda:1`（原训练卡）。三路评测顺序使用物理 GPU 0，避免多个 simulator
争用。

## 正式评测与验收

- 三模型都评测 `zero` 和各自 RTX 3090 `profile`；profile revision 固定为
  `66414463f2a609316da63935941a79abf0a73ce9`，latency seed 为 0。
- 20Hz、50 episodes、共同 seeds `4242424242–4242424291`，每局 60 个实际
  environment transitions。
- 记录 `success_once`、return mean/std 和 episode length。
- H1 zero 对照：QwenOFT 56%、QwenPI v3 54%、QwenGR00T 56%。
- 每个 setting 的 `success_once >= 25/50` 为通过；未通过不挑 checkpoint、不追加训练。
- 不做 DAgger 或超参搜索。正式训练入口使用 `--no-publish`，完成协议修正与终评后，
  三套 step-5,000 inference bundle 于 2026-09-20 单独发布并由 Hub API 核验。

| 模型 | Zero success / return ± SD | RTX 3090 profile success / return ± SD |
|---|---:|---:|
| QwenOFT | 25/50；19.42654±16.93815 | 1/50；3.97404±4.89508 |
| QwenPI v3 | 34/50；26.42206±13.34111 | 14/50；9.93412±14.21984 |
| QwenGR00T | 31/50；22.92212±16.49738 | 18/50；15.01258±16.16588 |

六组 mean length 均为 60。三个 Zero setting 达到 gate，三个 Profile setting 均未达到。
正式摘要和逐 setting 审计见
[结果报告](../outputs/tasks/mikasa_intercept_grab_fast/h8_teacher_baseline_20260920/REPORT.md)。
冻结首动作的旧 Profile 结果及 H200 wall-clock measured 结果均不属于本合同。

## Conditioned H8：Profile / Fixed-2 prefix training

以下描述已发布的 **71D v2 历史合同**。新训练代码已迁移到 63D v3：
7D proprio + 按当前时刻对齐、固定动作尺度的 8×7 prefix；本轮 Fixed-2
重训入口、开发评测与选优以 [prefix 修复协议](MIKASA_FIXED2_PREFIX_REPAIR.md) 为准。

Conditioned 路线使用独立任务键 `mikasa_conditioned_h8`，不改变上面的 `mikasa_h8`
baseline。action 仍为 `8×7`，state 改为 71D：7D proprio、controller 当时真实可见的
returned chunk `8×7`，以及 active-head one-hot 8D。所有评测延迟（包括 Zero）都保留
训练合同中的 `action_prefix`；71D returned-action context 按真实 scheduler 状态因果生成。
旧评测入口在 Zero 时关闭 prefix 的行为已纠正；旧 conditioned Zero 结果不能代表本协议。
Naive H8 仍为 7D、无 prefix。数据同时保存 `action_prefix_mask` 与 `action_loss_mask`。

Profile 数据按模型分别采集，使用 RTX 3090 profile revision
`66414463f2a609316da63935941a79abf0a73ce9` 的 p95 latency 投影固定 prefix：

| 模型 | p95 latency | prefix K |
|---|---:|---:|
| QwenOFT | 100.3273270369 ms | 3 |
| QwenPI v3 | 150.3526334763 ms | 4 |
| QwenGR00T | 100.3513440609 ms | 3 |

Fixed-2 使用 `latency.method=fixed`、`fixed_latency_ms=100.0` 和
`action_prefix.mode=fixed_frames, frames=2`，三模型共享同一份数据。

Fixed-1/3/4 沿用同一协议，训练时分别使用 50/150/200ms 和 K=1/3/4。
评测时只改变实际延迟，K 从训练合同读取。请求 t 的 prefix 是当时可见计划中 t 起的
K 个动作；QwenOFT 返回后由策略适配层覆盖对应 head。实际到达 t+L 时执行 head L，
不再加 K。L<K 时可以执行复制来的 prefix；L>K 时训练假设轨迹与实际等待轨迹可能不同。
普通 checkpoint 和 resident DAgger 推理共用该适配层。
本次三路训练、5×5 矩阵和 review 边界见 [Fixed latency grid](MIKASA_FIXED_LATENCY_GRID.md)。

每个 admitted observation 先从主 scheduler 冻结 K 个 controller-visible future
actions；再把主 ManiSkill state、elapsed step、oracle info 和 reached status 复制到
private env。private env 重放 prefix 后，由 zero-latency H1 PPO teacher 逐帧推进并补齐
H8 suffix。prefix 的 loss mask 为 0，suffix 为 1，episode 终止后的 tail 为 0。完整
chunk 回到主 scheduler，主分支继续遵守 20Hz、capacity 1、busy drop、latest-ready、
hold-last 和无首动作冻结。成功和失败主分支中含有效 suffix target 的 episode 都可进入
数据集；train 和 validation 分别采到至少 64,288 和 6,884 个有效 action target，尝试
seed 从 0 顺序递增且 split 不重叠。

正式 collector 在每张 H200 上使用 128 个 ManiSkill vector slots。每个 slot 拥有独立、
由 profile seed 0 和 slot id 派生的确定性 temporal stream，并跨该 slot 的 episode 保留
machine state。RGB main env 是唯一正式主分支；PPO branch 从它复制 state 后批量生成
teacher suffix，不再同步推进第二个 state-only main env。最后一个 batch 只选取 seed 顺序上
首次达到 split target 的最短前缀，剩余尝试保留在 raw audit，但不进入 LeRobot 数据集。

完整入口分成三个可恢复 phase；每个 phase 只依赖上一阶段写出的审计或 checkpoint：

```bash
export QWENOFT_PY=/path/to/qwenoft-python
export QWENPI_PY=/path/to/qwenpi-python
export QWENGROOT_PY=/path/to/qwengr00t-python
export MIKASA_PY=/path/to/mikasa-python

bash scripts/tasks/mikasa_intercept_grab_fast/run_profile_v2_h8.sh collect
bash scripts/tasks/mikasa_intercept_grab_fast/run_profile_v2_h8.sh train-eval
bash scripts/tasks/mikasa_intercept_grab_fast/run_fixed_v2_h8.sh all
```

正式训练均从 Qwen3-VL-4B 和 fresh action head 初始化，seed 0、global batch 128、
5,000 optimizer steps，只使用 step 5,000；step 2,500 保存恢复状态。Profile 在物理
GPU 1/2/3 分别训练 OFT/PI/GR00T，Fixed OFT 在 GPU 0。GPU 6、7 不随首批启动；等
GPU 0/1/2 任意一路成功结束且对应目标卡空闲后，入口才依次提交 Fixed PI 与 Fixed
GR00T。OFT/PI 使用
micro batch 128 / accumulation 1，GR00T 使用 64 / 2，配置失败或 OOM 直接暴露。

终评对每个模型运行 `profile-trained → profile` 和
`fixed-trained → fixed-2/zero`，共 9 组；不运行 `profile-trained → zero`。
Zero-trained action-chunk baseline 保留已有的 `zero → zero/profile`，并补跑
`zero → fixed-2`，因此每个模型的完整对照矩阵有六格：zero-trained 三格、
fixed-trained 两格、profile-trained 一格。
共同协议是 50 episodes、seeds
`4242424242–4242424291`、60 transitions、20Hz、capacity 1、profile seed 0 和无首动作
冻结。已完成模型的提前评测在物理 GPU 1 上串行运行，simulator 与 inference 使用同一张
卡。主通过线只用于 `profile → profile` 与 `fixed → fixed-2`：
`success_once >= 25/50`。Zero-latency transfer 只报告，不据此挑 checkpoint、续训或改超参。
入口不会自动发布 checkpoint。

## Naive H8 DAgger 对照

QwenOFT、QwenPI v3、QwenGR00T 各自从现有 zero-latency H8 SFT step-5000
权重开始，重置优化器，在 zero latency 下用当前 student rollout。每个 admitted
observation 从当时的 main env 状态复制 private PPO env，零 prefix 地生成最多 8 个
teacher future actions；episode 终止后的 head 用 `action_loss_mask` 排除。学生输入仍为
双相机和 7D proprio，不加入 returned-action context 或 action prefix。增量数据复用
各自原 SFT 的 7D state/action normalization 与原 validation split。

训练预算与 conditioned DAgger 对齐：10 轮、每轮 500 optimizer updates、每轮至少
16,000 个有效 teacher action labels，global batch 128，bootstrap/current/history
采样权重依次为首轮 50/50，之后 40/40/20（历史轮次等权）。naive 监督完整有效
chunk，conditioned 监督 prefix 之后的有效 suffix；两者的标签位置不同，报告中必须
同时列明。每轮 zero-latency 开发评测用共同 seeds `4242424442–4242424541`，
只据该评测选 best。step-5000 后对 SFT baseline、best、latest 分别在 zero、各模型
RTX 3090 Profile、Fixed-2 下用 held-out seeds `4242424542–4242424741` 各评测
200 局；按模型和 seed 配对比较，不以延迟终评选 checkpoint。

独立队列按 OFT、PI、GR00T 顺序，在两台 H200 任意安全的双训练卡空闲时提交；
6/7 上若只有已识别的旧六路模拟器进程且显存占用低于 8 GiB，可与新 trainer
共用；新任务的 simulator 可与自身 trainer 同卡，模拟器进程按原 GPU lock 串行。
OFT 用 micro batch 32 / accumulation 2，PI 用 64 / 1，GR00T 用 32 / 2，
均保持 global batch 128。队列直接启动正式任务，入口为
`scripts/tasks/mikasa_intercept_grab_fast/dispatch_h8_naive_dagger_queue.py dispatch`；
单路入口为 `launch_h8_naive_dagger_arm.sh`。正式输出根为两机各自的
`outputs/starvla/runs/mikasa_intercept_grab_fast/h8-naive-dagger-20260923/`。
训练和核验不会自动发布模型。
