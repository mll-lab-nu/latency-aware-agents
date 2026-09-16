# Balance DAgger：问题、修复与实验上下文

更新日期：2026-09-10。本文记录本次对话中已经确认的事实、已完成的修改和实验，以及尚未解决的问题，供后续工作继续使用。更早的运行细节另见 [BALANCE_DAGGER.md](../BALANCE_DAGGER.md)。

## 1. 当前结论

- 已确认一个真实 bug：本次使用的 **bootstrap 标签是限速/限加速度后的 command target**，旧 DAgger 标签却是 **处理前的 teacher absolute request**。配置写作 50/50；真实 sequential sampler 按行混合，seed 42 的前 64,000 次采样约为 57.4% / 42.6%。
- 已修复在线采集，并对原有 64,000 行 DAgger 数据离线重标。标签修复阶段复用了旧轨迹；后续覆盖实验另采集了修正模型的 64,000 行新轨迹。环境 limiter 与 bootstrap 数据保持相同。
- 标签修复轮已从原始 QwenOFT checkpoint 完成 **500 optimizer updates + 100 局零延迟 eval**。
- 修正模型的 100 局平均回报为 **68.02 ± 3.04**，明显高于错误标签轮的 **22.86 ± 3.04**，但仍低于原始模型。
- 在相同 seed 的前 50 局上，原始模型为 **107.22**，修正 DAgger 为 **68.30**，仍下降约 **36.3%**。因此只能说标签修复缓解了崩溃，不能说当前 DAgger 配方已经有效提升策略。
- 第 10 节全参数训练对照已完成：sampler 42 的独立 50 局上，原模型 / 混合组 / bootstrap-only 分别为 **105.77 / 67.92 / 104.73**；十倍降低学习率和训练图像置换均未恢复主要损失。
- 从同一修正模型再训 500 步，刷新访问数据相对旧数据在实际 sampler **42 / 43 / 44** 上分别改善 **+11.01 [7.81, 14.02] / +17.14 [13.69, 20.60] / +11.02 [5.70, 15.39]**。每组均使用共同的 50 个未见评估 seed；相同未见 C 状态上的命令误差下降、腿部纠正增益上升也在三组成立。新组回报分别为 **84.73 / 93.93 / 83.89**，仍低于原模型 105.77。这支持**刷新访问数据带来部分恢复**，尚未独立区分 state 覆盖、图像和条件标签变化，也未解释剩余差距；完整证据与边界见 10.13–10.14。
- 原始模型、错误标签数据/模型和修正后的数据/模型均已保留。没有把本轮模型替换为默认更优模型，没有上传 Hugging Face。

下文的 `±` 均表示 episode return 的标准差，不是置信区间。

## 2. 为什么开始调查 DAgger

最初的问题是：Balance 上此前 official GR00T 的历史结果约为 353.53（10 局），换成 StarVLA 的多个模型后表现明显较差；随后尝试给 QwenOFT 加 DAgger，结果反而进一步崩溃。

StarVLA 原始模型的本地零延迟评估结果如下，均为 50 局：

| 模型 | 平均回报 | 标准差 | 平均存活步数 |
| --- | ---: | ---: | ---: |
| QwenOFT | 107.2190 | 25.8240 | 118.08 |
| QwenGR00T | 102.4996 | 23.2644 | 109.74 |
| WanOFT | 88.1339 | 12.8184 | 98.42 |
| QwenPI_v3 | 61.5132 | 9.5269 | 75.40 |

这些结果位于 `outputs/starvla/eval_backfill/results/balance/<model>/queue_eval_latency_0.json`。

历史 W&B 项目：

- [official GR00T](https://wandb.ai/saberrr-zju/finetune-gr00t-n1d7)
- [StarVLA 原始实验](https://wandb.ai/zihanwang-ai-northwestern-university/starvla_tasks)
- [本次 DAgger 实验](https://wandb.ai/saberrr-zju/starvla_tasks)

必须区分两个问题：

1. **为什么旧 DAgger 从约 107 掉到约 23？** 已找到标签含义不一致的具体 bug，修复后改善到约 68，但仍有残余退化。
2. **为什么 StarVLA 比历史 official GR00T 差？** 本次实验没有解决，也不能全部归因于缺少 DAgger。此前检查发现实际训练用的 state/action 数组相同，但预训练初始化、action head、action horizon、冻结策略、归一化方式等不同，历史运行还存在代码版本和评估局数差异。没有受控消融，不能把约三倍差距归因于其中某一项。

## 3. 已确认的标签问题

### 3.1 错在数据标签，不是关掉了限速

实际使用的 cached bootstrap 存储环境经过 velocity/acceleration conditioning 后的关节命令 `q_cmd`。旧 `scripts/starvla/balance_dagger.py` 则把
`native.g1_absolute_request_from_raw_action(teacher_action)` 的返回值直接作为标签，尚未经过同样的 conditioning。

因此混合数据中，`action` 同名、同为 43 维，却代表两种不同目标：

| 数据 | 修复前的 action 含义 |
| --- | --- |
| Cached bootstrap | 限速/限加速度后的 command target |
| 旧 DAgger | 限速/限加速度前的 absolute request |

环境执行时的 limiter 一直存在。问题不是“某轮环境不限速”，而是监督学习目标不一致。训练 loss 下降也不能证明目标含义正确。

这里的结论特指**本轮实际使用的 cached bootstrap**，不是说所有历史版本的 exporter 都输出同一种目标；以后重新导出 bootstrap 时仍需核对其动作含义。

### 3.2 证据

- 对 bootstrap train 的全部 **85,958 对相邻样本**检查：`action[t]` 与下一状态解码得到的 `q_cmd[t+1]` 最大误差约 **3.61e-7**。
- 初始 12 行真实 teacher 重放中，未处理的 request 与原标签的平均绝对误差约 **0.59544**；按 pre-action state 经过 limiter 后，最大误差约 **2.38e-6**。
- 修复完成后，另取 12 个 episode 的样本运行实际 helper：最大误差 **1.024e-6**，平均绝对误差 **1.979e-8**，没有元素超过 `1e-5`，且没有改动环境的 `_q_cmd` / `_v_cmd`。
- 扩展检查发现一个 terminal bootstrap row 的单关节标签差异；该 episode 的全部 980 个非 terminal transition 仍满足 action 与下一状态 command 对应，最大误差 **1.81e-7**。不能把这个孤立终止行差异说成所有样本都精确重建；它也不影响本次直接使用旧 DAgger 已存 request 的离线修复。

## 4. 实际改了什么

### 4.1 在线采集和离线重标共用一个 helper

`scripts/starvla/balance_dagger.py` 新增 `_conditioned_policy_target(native, state, policy_request)`：

1. 从 **198D pre-action state** 解码上一步 command：`state[112:155]` 是归一化的 `q_cmd`，`state[155:198]` 是归一化的 `v_cmd`，此处为 MuJoCo joint order。
2. 用环境 joint ranges 和 velocity limits 恢复实际命令位置、速度。
3. 将 teacher absolute request 的 body joints 从 policy order 转为 MuJoCo order。
4. 调用已有的 `native._condition_velocity`，复用相同的速度、加速度和关节限位逻辑。
5. 将处理后的 command target 转回 policy order，作为与 bootstrap 对齐的标签。

在线采集仍由 **student 执行动作**；teacher 只在相同的 pre-action state 上提供反事实标签。生成标签发生在 student `step()` 之前，不执行 teacher 动作，也不改变 student 环境的 command state。

### 4.2 修复旧数据，保持实验可比

- Collector 新增 `--relabel-from`；训练入口新增 `--dagger-relabel-from` 并记录来源。
- 离线重标直接读取旧 raw 数据的 absolute `action`，**不再次当作 FastTD3 raw action 解码**，也不加载 teacher/student 模型。
- 只替换 Arrow 的 `action` 和 `action_text` 两列，其余图片、state、reward、seed、episode/step 顺序、schema 和 metadata 保持不变。
- 64,000 行均发生标签变化，平均绝对变化约 **0.6183085**。
- 重标后仍通过现有 LeRobot conversion/preparation 流程，继续使用原 bootstrap 的 normalization 和 validation。
- 全量核对通过：最终 LeRobot 的 state/action 与修正后的 raw 数组逐元素一致；task contract 与 `meta/stats_gr00t.json` 和 bootstrap 完全一致。
- 新数据共 **64,000 training frames / 555 episodes**，没有新增 validation rows；评估 action loss 仍使用独立 bootstrap validation。

相关文件：

- [balance_dagger.py](../../scripts/starvla/balance_dagger.py)：标签 helper、在线采集、离线重标。
- [train_task.py](../../scripts/starvla/train_task.py)：重标参数传递和 provenance，沿用已有训练/eval 流程。
- [test_starvla_collect.py](../../tests/scripts/test_starvla_collect.py)：标签与 pre-action state、student 执行动作、joint order、离线数据保持性。
- [test_starvla_train_task.py](../../tests/scripts/test_starvla_train_task.py)：重标入口、原始 checkpoint 起点、固定 normalization/validation。

已有工作区中还包含先前完成的 Balance DAgger 编排、`--dagger-from`、`--no-publish`、完成状态 resume 和 exporter modality key 修复；不要把这些都误认为本次标签修复新增的改动，也不要覆盖无关的未提交修改。

## 5. 修正实验的实际参数

本轮只改变 DAgger 标签，trainer/framework/checkpoint 配置与上一轮失败实验保持一致。

| 项目 | 设置 |
| --- | --- |
| Run ID | `dagger-conditioned-r1-20260909` |
| 模型起点 | 原始约 107 分的 formal QwenOFT checkpoint |
| Optimizer 起点 | 全新 optimizer；`is_resume: false`，不是接着约 23 分的模型训练 |
| GPU 数 | 2 |
| 每卡 micro-batch | 32 |
| Gradient accumulation | 2 |
| Effective global batch | **2 × 32 × 2 = 128** |
| DAgger 轮数 / optimizer updates | 1 轮 / 500 updates |
| 配置混合权重 | 50% bootstrap + 50% 本轮重标 DAgger；实际按行采样比例见 10.1 |
| Backbone / action head LR | `3e-5` / `1e-4` |
| Warmup | 100 updates |
| Scheduler | `cosine_with_min_lr`，配置 `min_lr=1e-6`；日志中 base 终点为 `1e-6`，head 按组比例为约 `3.333e-6` |
| Optimizer | AdamW，betas=(0.9, 0.95)，eps=`1e-8`，weight decay=`1e-8` |
| Gradient clipping | 1.0 |
| 冻结策略 | 全量微调，没有冻结 backbone/LLM |
| Action / state | H1，action 43D，state 198D，L1 loss |
| 图片 | 单帧 RGB，训练尺寸 224×224 |
| 训练随机 seed | 0 |
| Eval | 100 episodes，零延迟，50 Hz，deterministic，单环境，最多 1000 steps/episode |
| 发布 | `--no-publish`；W&B logging 保留，不上传 HF |

两轮 DAgger 的 eval 配置除模型/输出路径外相同；Python executable 的路径写法不同，但已确认解析到同一个解释器。对原模型的公平对比使用相同 seed 0–49。

## 6. 实验结果

### 6.1 同 seed 前 50 局：主对比

| 模型 | 平均回报 | 标准差 | 平均存活步数 |
| --- | ---: | ---: | ---: |
| 原始 formal QwenOFT | 107.2190 | 25.8240 | 118.08 |
| 错误标签 DAgger | 22.9287 | 3.5819 | 27.82 |
| 修正标签 DAgger | 68.3031 | 3.1028 | 79.72 |

原始模型目前只有这份 50 局评估，不能把它当成已跑过同样的 100 局。

### 6.2 两轮 DAgger 各自完整 100 局

| 实验 | 平均回报 | 标准差 | 平均存活步数 |
| --- | ---: | ---: | ---: |
| 错误标签，`dagger-r1-20260908` | 22.8633 | 3.0377 | 27.44 |
| 修正标签，`dagger-conditioned-r1-20260909` | 68.0205 | 3.0410 | 79.73 |

修正轮记录的 invalid actions 为 0，训练和 eval 全流程退出码为 0。

### 6.3 Loss

| 实验 | 最终 train loss | Held-out action loss |
| --- | ---: | ---: |
| 错误标签 DAgger | 0.0745380 | 0.1185290 |
| 修正标签 DAgger | 0.0227175 | 0.0213984 |

两轮上述 held-out action loss 均使用 64 个 validation samples。原始 QwenOFT 的历史 action loss 约为 0.01706，但记录使用 128 samples，不应作为完全相同批次的精确对比。

**Loss 恢复不等于闭环控制性能恢复。** 本次能够确认修正后回报改善，但修正模型仍不如原始 checkpoint。

## 7. 仍然不理想，以及尚未证明的原因

标签修正轮结束时，已修复的是“同一个 action 字段混合了两种监督目标”，剩余问题是“即使标签对齐，DAgger 微调仍让策略从 107 掉到 68”。后续受控实验见第 10 节。

本次只进行了同配方的标签修复对照，不能据此断言剩余退化一定是以下某一项造成的：

- 500 updates / 重启 warmup 与 learning rate 是否对原有策略扰动过大。
- 全量 backbone 微调是否导致原有能力遗忘。
- 当前混合采样是否给 learner 失败状态过高权重。
- 单轮旧 learner 状态是否足以提供有效的恢复训练信号。
- 行为克隆的 L1/action loss 与长时闭环稳定性的差距。

这些都是**候选解释，不是已确认根因**。本轮复用了旧 learner 的访问状态，只重新生成标签，并未用修正后的模型重新 rollout，也没有完成多轮在线 DAgger。

以上是标签修正轮结束时的候选解释，当时未启动后续训练。新获准的全参数训练对照及其结果见第 10 节；保留原始 checkpoint 和相同 eval seed，每次只改变一个因素，不能提前将候选解释写成已验证的改进。

## 8. 产物与恢复工作入口

修正轮根目录：

`outputs/starvla/runs/humanoidbench_balance_simple/qwenoft/dagger-conditioned-r1-20260909/`

| 相对修正轮根目录的路径 | 内容 |
| --- | --- |
| `run.json` | 最终训练参数、数据引用和 provenance |
| `dagger.yaml` / `dagger_mixture.json` | 实际训练配置和采样配比 |
| `dagger.log` | 训练、validation loss、checkpoint 保存日志 |
| `dagger/round-00/raw/` | 64,000 行修正标签后的 canonical 数据 |
| `dagger/round-00/lerobot/` | 实际训练使用的数据 |
| `dagger/round-00/provenance.json` | 旧 raw 来源与 bootstrap 引用 |
| `training/dagger/checkpoints/steps_500_pytorch_model.pt` | step-500 模型 |
| `training/dagger/checkpoints/steps_500_state/` | 完整训练状态 |
| `inference/checkpoints/model.pt` | 与 step-500 checkpoint 相同文件的 inference 入口 |
| `inference/eval_l0.json` | 汇总和全部 100 局 episode metrics |
| `inference/provenance.json` | 最终 inference provenance |
| `evals/final/` | eval 配置、逐局/逐步记录、动作和 latency 日志 |

其他重要路径：

- 原始 inference bundle：`outputs/starvla/artifacts/pretrained/balance-qwenoft-formal-latest/zero-latency/humanoidbench-balance-simple/vla/starvla-qwenoft-h1/humanoidbench_balance_simple_qwenoft_formal_latest/`。
- 原始模型的 50 局 eval：`outputs/starvla/eval_backfill/results/balance/qwenoft/`。
- 固定 bootstrap：`outputs/starvla/datasets/humanoidbench_balance_simple/teacher-rollouts-h1-state-v1/`，使用 `lerobot` 和 `lerobot__val`。
- FastTD3 teacher：`outputs/starvla/artifacts/models/zero-latency/humanoidbench-balance-simple/small-policy/fasttd3-v1/checkpoint.pth`。
- 错误标签轮：`outputs/starvla/runs/humanoidbench_balance_simple/qwenoft/dagger-r1-20260908/`；离线重标来源是其 `dagger/round-00/raw/`。
- 错误标签轮下还保留 `training/dagger-microbatch1-aborted-step187/` 和 `batch-probe/probe-mb8/`，不是本次修正轮的训练起点。

[修正轮 W&B](https://wandb.ai/saberrr-zju/starvla_tasks/runs/balance-qwenoft-dagger-conditioned-r1-20260909-dagger) 已记录训练指标，并写入和回读确认了 100 局 eval summary：`eval_corrected/fixed/latency_0/{mean_return,std_return,mean_length,completed_episodes}`。

当前修正轮已经完整结束。不要对这个 run 直接加 `--resume` 并以为会开始新实验：现有完整 step-500 state 会使训练阶段跳过。下一次获准的新实验应使用独立 run ID，保留现有结果，不覆盖原始较优模型。

## 9. 验证状态与工作约束

- Collector/orchestration 两个相关测试文件最终 **17 passed**。
- 较广的数据测试选择 **28 passed、1 skipped**；另有一个已有 exporter 测试因 unknown-layout fixture 缺少 `action_labels`，先触发 KeyError 而非预期 ValueError，运行失败后被排除。该处 exporter 校验未由本次修改，不能声称整个测试集全部通过。
- 环境 limiter 相关选择 **1 passed、2 skipped**（环境依赖）。
- 修改文件的 `git diff --check` 通过。
- 用户确认继续使用双卡、micro-batch 32、accumulation 2、global batch 128；训练后必须做实际 eval，不能只看 loss。
- 不上传 HF；W&B 记录允许。保留所有旧数据和模型。2026-09-09 用户已授权执行第 10 节的短程对照计划。
- 遵守仓库偏好：不加防御式检查、隐藏 fallback 或未经需要的抽象；保留用户已有的未提交工作，不做 reset/clean/stage。

## 10. 视觉输入干预与根因定位（2026-09-09）

### 10.1 已完成的初步诊断

保持真实 198D state 每步更新，仅将视觉输入替换为该局初始图像：

| seed 0–9，平均回报 | 原始模型 | 修正标签后的 DAgger 模型 |
| --- | ---: | ---: |
| 实时图像 | 105.4654 | 69.3722 |
| 固定初始图像 | 87.3509 | 84.1922 |

固定图像后，修正模型 10/10 局改善；模型间差距从 36.0933 缩小到 3.1586。2026-09-09 13:32 UTC 已重新运行四个条件的全部 10 局，逐局回报均精确复现；pre-action 数据与逐局汇总保存于本轮实验目录的 `diagnostics/{original,corrected}/{live,initial}/`。

| seed | 原模型实时 | 原模型初始帧 | 修正模型实时 | 修正模型初始帧 |
| --- | ---: | ---: | ---: | ---: |
| 0 | 83.2491 | 71.5050 | 73.1955 | 77.6824 |
| 1 | 102.6575 | 88.6473 | 73.5193 | 79.7729 |
| 2 | 119.0221 | 88.2131 | 63.6294 | 91.2421 |
| 3 | 103.5936 | 115.2473 | 75.3851 | 97.0653 |
| 4 | 83.6828 | 72.6794 | 70.4264 | 98.2029 |
| 5 | 88.7299 | 88.1907 | 68.6932 | 78.2060 |
| 6 | 118.1036 | 91.4980 | 65.7762 | 81.8142 |
| 7 | 126.2432 | 81.1780 | 68.6349 | 76.5687 |
| 8 | 143.1442 | 105.4677 | 68.1673 | 87.0544 |
| 9 | 86.2283 | 70.8821 | 66.2942 | 74.3133 |

初始帧减实时图像的配对回报差：原模型 -18.1146，episode bootstrap 95% 区间 [-28.3217, -8.0224]；修正模型 +14.8200，区间 [9.8427, 20.1791]。原始汇总为 `diagnostics/paired_returns_seed0_n10.json`。

独立诊断 seed 1,000,000,000–1,000,000,009 的全部四个条件也已完成：

| 未见 seed，平均回报 | 原模型 | 修正模型 |
| --- | ---: | ---: |
| 实时图像 | 121.8708 | 67.9108 |
| 固定初始图像 | 84.0641 | 83.7768 |

修正模型依然 10/10 局改善，配对回报增加 15.8659，95% 区间 [11.7409, 20.4906]；原模型减少 37.8067，区间 [-54.0913, -21.7455]。两模型差距由 53.9600 缩小到 0.2873。逐步数据在 `diagnostics/{original,corrected}/{live_unseen,initial_unseen}/`，配对统计在 `diagnostics/paired_returns_seed1000000000_n10.json`；这些诊断 seed 不再用于最终确认集。

在同一批旧策略轨迹的第 30–59 步状态上，修正模型对 teacher 的 normalized action L1 从 0.04472 改善到 0.02724；在修正模型自己产生的同阶段状态上，其误差为 0.07241。原模型与修正模型之间交叉替换 VLM、动作头和 state projector，恢复均不完整。

这些结果定位了重要退化路径，但没有唯一定位训练诱因。固定首帧同时改变视觉内容和后续访问状态，不能直接把全部收益解释为局部视觉反馈方向被修复；也不能由模块交换结果断言某一个模块单独出错。

进一步只读核对：

- Balance FastTD3 teacher 的全部输入就是这 198D state：112D task observation（qpos/qvel，包含平台状态）和 86D 上一关节命令位置/速度。当前 teacher 无 RNN；给定 state，conditioned teacher label 不依赖图像或额外隐藏历史。
- “保持上一 q_cmd”的全量 normalized L1：bootstrap 86,048 行为 0.0418455，修正 DAgger 64,000 行为 0.0588566。因此数据不支持“DAgger 标签比 bootstrap 更接近保持不动”的简单解释。
- DAgger 数据的采集 seed 为 0–554，现有 seed 0–49 评估可用于同条件策略比较，但不作为独立的未见 seed 泛化测试。
- 混合配置写的是 50/50，但当前 sequential step sampler 前 64,000 个样本实际为 bootstrap 36,752 / DAgger 27,248（57.425% / 42.575%）。首轮复现保持实际采样行为，避免同时改变多个因素。
- 训练 state min-max 不截断、推理会截断的差异已经做过小规模消融；取消推理截断没有恢复修正模型回报。提高动作头/state 路径计算精度也未恢复。不能把这些问题当作已确认的主因。

### 10.2 已授权的实验计划

所有训练组沿用现有 QwenOFT 结构并进行全参数微调。先区分继续训练的更新行为、新增数据带来的视觉关联，以及新策略访问状态的监督覆盖不足。

1. 保存可复现的实时图像/固定首帧对照，以及 pre-action state、图像、网络动作、生效命令和 teacher 标签。对早期相邻状态与图像计算四种输入组合，在 state 相同时测量视觉变化造成的 teacher 误差和纠正方向；通过现有 limiter 同时计算实际生效命令。
2. 从同一个原始约 107 分 checkpoint 分别完成三组 500 optimizer updates：原配置混合数据复现、bootstrap-only、混合数据且学习率曲线缩小十倍。共同固定双卡、micro-batch 32、accumulation 2、global batch 128、warmup 100、数据预处理和实际采样方式。低学习率组 base/head/min_lr 分别为 3e-6 / 1e-5 / 1e-7。
3. 每 100 步保存权重，由独立进程评估 checkpoint；训练器内部 eval_interval 保持 500，不通过中间评估改变训练 RNG 或采样节奏。复现失败时先解释差异；bootstrap-only 也退化则新增数据不是必要条件；低学习率保住回报但没有学习新监督，不算修复。
4. 按结果追加最少对照：训练 batch 内只置换图像以验证错误视觉关联；或从修正模型采集 64,000 行新状态，并用两个同起点、同更新预算的继续训练组比较旧数据与新数据。
5. 初筛使用 seed 0–9；独立诊断使用 seed 1,000,000,000–1,000,000,009；最终确认使用 seed 1,000,000,100–1,000,000,149。候选原因与对应对照使用实际 sampler seed 42、43、44 复验，不能仅改变顶层训练 seed（loader 当前默认 sampler seed 为 42）。

根因结论需要移除/恢复因素的成对实验，同时观察固定 state 下的视觉错误、实际命令方向和闭环回报。报告配对效果与置信区间；部分恢复只记为部分解释。

### 10.3 执行记录

- 实验目录：`outputs/starvla/runs/humanoidbench_balance_simple/qwenoft/dagger-cause-20260909/`。
- 配置、临时诊断脚本、逐步/逐局数据和命令保存在该目录；复用现有 trainer、collector 和 evaluator，不新增生产 API。
- 2026-09-09 13:16 UTC：开始实施；此时尚未启动新训练。以下结果随实验完成更新。
- 13:34–14:03 UTC：完成 `reproduce` 的 500 步全参数训练。全部 100 条已记录 loss、最终 732 个权重张量与修正标签轮完全一致；比较结果保存在 `reproduce-equivalence.json`。第 100、200、300、400、500 步权重全部保留，随后运行标准闭环评估。
- 14:04 UTC：启动 `bootstrap_only`。与复现组相比，仅将训练数据改为 bootstrap；全参数微调、500 步、优化器、学习率和 global batch 128 保持相同。
- 14:15 UTC 起：在独立进程中提前评估 `bootstrap_only` 已写完的中间 checkpoint，与后续训练并行；训练器配置和训练内 validation 节奏不变。复现组在并行诊断时仍与旧轮逐张量一致，已核对该执行方式没有改变其训练结果。
- 14:35 UTC：`bootstrap_only` 完成 500 步；14:36–15:05 UTC 完成 `lr_tenth`，与复现组相同的混合数据和全参数更新，仅将 base/head/min_lr 整体缩小十倍。
- 复现组标准评估：step 500 的 seed 0–9 平均回报为 69.3722，与修正轮精确一致；step 100 则为 41.4648。step 100 的固定样本 L1：bootstrap validation 0.0349549、DAgger training 0.0339550。退化已在 warmup 结束时出现，直接改成只训 100 步不能消除问题；不能将下降简单归因于训练到 500 步才发生的过拟合。

### 10.4 固定 state 的视觉干预结果

每种策略的 seed 0–9 实时轨迹各取 t=0、10、20、30、40、50，共 120 对相邻输入；对两个模型分别评估四角组合与初始图像。生效命令使用当时环境内部的精确 q_cmd/v_cmd 重放 limiter，rollout 中逐步与 `info.action_target` 核对。`diagnose.py` 保存原始输出，`analyze_cross.py` 按 episode 重采样 20,000 次计算配对均值的 percentile 95% 区间。

以下是修正模型在**自身实时轨迹**上、固定 s_(t+1) 时的结果；负数表示换图后误差减少：

| 图像干预，前 60 步抽样 | 全 43 维 normalized 生效命令 L1 变化 | episode bootstrap 95% 区间 |
| --- | ---: | --- |
| 当前图像换为上一帧 | +0.0006908 | [+0.0001463, +0.0013402] |
| 当前图像换为初始帧 | -0.0005026 | [-0.0017122, +0.0007488] |

腿部 12 维对应变化为 +0.0010243 和 -0.0021979，后者区间仍包含 0。在旧策略轨迹上，修正模型换成初始帧反而增加全维误差 0.0045428，区间为 [0.0019938, 0.0076518]。同一旧策略状态上，实时图像下的全维生效命令 L1 是原模型 0.0326702、修正模型 0.0227877；在修正模型自身状态上，两者为 0.0477156 和 0.0457866。

**因此不保留“新模型的局部实时视觉响应普遍朝错误方向”的强解释。** 单步 teacher 误差与闭环回报没有同向变化；固定首帧提高闭环回报的现象仍成立，需要通过训练对照区分更新行为、数据关联和状态分布累积效应。原始数值见 `diagnostics/cross_analysis.json`，分阶段与腿部纠正方向也保存在其中。

### 10.5 limiter 精确状态与导出状态的差异

诊断另行比较了由 float32 198D state 重建的 conditioned teacher 标签，和使用精确 native q_cmd/v_cmd 执行同一 teacher request 的结果。独立重算与保存值一致，排除了诊断的动作重算或顺序错误。

- 原策略非 terminal 行：8/1,169 行绝对差超过 1e-5 rad，最大 0.0623136 rad；修正策略为 21/809 行，最大 0.0472317 rad。
- 原策略前 60 步没有该异常；修正策略前 60 步异常全部在手指，最早第 46 步，没有腿部异常。
- q/v 重建误差很小，但 teacher request 恰好位于关节下限时，limiter 的严格 `< lower_limit` 比较会因约 1e-17 的浮点边界误差翻转 braking 分支。全部异常都符合该机制。

这限制了“由导出 state 生成标签与实际执行在所有边界完全相同”的说法，但当前证据不支持将其当作早期主要退化根因。首轮三组训练仍保持原数据和 limiter，避免同时改变因素。诊断的模型输入保持 float32 state，额外 q/v 仅用于计算精确生效命令。

### 10.6 固定监督样本

保存 80 个 bootstrap validation 输入（每个验证 episode 均匀取 8 帧）和 512 个 DAgger training 输入（RNG seed 42，无放回抽样），由 `fit.py` 对所有模型使用相同推理预处理和 batch size 8。样本本身在 `diagnostics/fit_samples.npz`，源文件与行号在 `diagnostics/fit_samples.json`。

| 模型 | bootstrap validation，80 行 normalized 网络 L1 | DAgger training，512 行 normalized 网络 L1 |
| --- | ---: | ---: |
| 原模型 | 0.0285557 | 0.0494718 |
| 修正标签后的模型 | 0.0293237 | 0.0240075 |

DAgger 标签拟合约改善一半，bootstrap 验证误差仅小幅变化；闭环回报却明显下降。三组训练后的同样本结果用于区分“保住原策略但几乎没有学习”和“学到新监督且保住闭环表现”。该表衡量网络输出与数据集标签的误差；10.4 的表衡量 limiter 后的实际生效命令，两者不混用。

### 10.7 最终确认集的基线

原有两模型已用标准 evaluator 完成 seed 1,000,000,100–1,000,000,149 的 50 局实时图像评估；该集合与采集、诊断 seed 均不重合。

| 模型 | 平均回报 ± 跨 episode 标准差 | 平均存活步数 |
| --- | ---: | ---: |
| 原模型 | 105.7664 ± 19.6230 | 116.26 |
| 修正标签后的模型 | 67.9163 ± 3.2189 | 79.86 |

同 seed 配对的修正模型减原模型回报差为 -37.8501，episode bootstrap 95% 区间 [-43.5786, -32.4150]。配置和逐步记录位于 `evals/{original,corrected}_seed1000000100_n50/`，配对统计为 `diagnostics/confirmation_baselines.json`。后续训练组使用同一确认集合比较。

### 10.8 复现组的训练过程

所有 500 步完成后评估保存的权重，使用相同 seed 0–9；固定监督样本与 10.6 相同。step 500 权重与既有修正模型逐张量完全一致，因此复用其固定样本结果。

| updates | 闭环平均回报 | bootstrap validation 网络 L1 | DAgger training 网络 L1 |
| --- | ---: | ---: | ---: |
| 0，原模型 | 105.4654 | 0.0285557 | 0.0494718 |
| 100 | 41.4648 | 0.0349549 | 0.0339550 |
| 200 | 60.9776 | 0.0339531 | 0.0316298 |
| 300 | 70.4598 | 0.0326258 | 0.0278151 |
| 400 | 72.5329 | 0.0308391 | 0.0254503 |
| 500 | 69.3722 | 0.0293237 | 0.0240075 |

下降已在第 100 步发生，随后部分恢复；bootstrap validation 误差也先升后降。现有每 100 步保存的 checkpoint 中，没有一个恢复原模型回报。因此“只在这些点提前停止训练”不能解释或解决退化。

第 100 步模型另跑固定初始图像，平均回报 54.0273，比实时图像高 12.5626，但仍低于原模型的固定图像回报 87.3509。这说明早期更新在固定图像条件下也产生了明显退化；不能把最终 step-500 的图像干预结论直接套用到全部训练阶段。数据在 `diagnostics/reproduce_100/initial/`，训练过程汇总在 `reproduce-curve.json`。

### 10.9 bootstrap-only 对照

| updates | 同 seed 0–9 闭环回报 | bootstrap validation 网络 L1 | DAgger training 网络 L1 |
| --- | ---: | ---: | ---: |
| 100 | 89.6522 | 0.0317556 | 0.0527823 |
| 200 | 89.7200 | 0.0336794 | 0.0537304 |
| 300 | 86.1024 | 0.0308501 | 0.0508862 |
| 400 | 96.1758 | 0.0292424 | 0.0502183 |
| 500 | 104.9055 | 0.0282967 | 0.0496381 |

- 第 100 步，bootstrap-only 减原模型的配对回报差为 -15.8132，95% 区间 [-24.7332, -7.6739]。
- 同为 100 步，混合组减 bootstrap-only 为 -48.1875，区间 [-55.6052, -40.3055]。

早期去掉 DAgger 数据后仍有退化，但 **bootstrap-only 到第 500 步与原模型的回报差异未检出**：与原模型的配对差为 -0.5599，95% 区间 [-11.6702, +10.5572]；该区间尚不能证明两者等效，混合组同为 500 步仍为 69.3722。因而不能用“继续训练本身会持续退化”解释最终剩余下降。当前证据进一步指向混合数据与更新过程的相互作用。

该数据对照包含训练帧组成的变化：混合组的部分 bootstrap 帧被 DAgger 帧替代；不能进一步把差异全部归因于图像内容或某种标签错误。第 100 步配对统计在 `diagnostics/step100_data_control.json`，完整过程在 `bootstrap_only-curve.json`。

第 500 步另跑固定首帧的 10 局平均回报为 93.7687，比实时图像低 11.1368；配对 95% 区间 [-23.4102, +2.2412]，3/10 局改善。该对照没有复现混合组“固定首帧后 10/10 局改善”的现象，但 10 局样本尚不能确认固定首帧在 bootstrap-only 上的平均损害。结果在 `diagnostics/bootstrap_only_500/image_ablation.json`；同一批 O/C 轨迹的固定 state 交叉图像输出也已加入 `diagnostics/cross_analysis.json`。

14:49 UTC 完成 50 个未见 seed 的标准评估：bootstrap-only 平均回报 **104.7339 ± 16.8255**，平均存活 115.68 步；同集合原模型为 105.7664，混合组为 67.9163。bootstrap-only 减原模型的配对差为 -1.0324，95% 区间 [-6.1671, +3.9836]；减混合组为 +36.8177，区间 [+32.3227, +41.4820]，50/50 局更好。这进一步确认数据组成对最终退化的影响，尚未区分图像关联、bootstrap 监督被替代和新状态覆盖等内部原因。记录见 `diagnostics/bootstrap_only_500/confirmation.json` 与 `evals/bootstrap_only_500_seed1000000100_n50/`。

### 10.10 十分之一学习率对照

相同混合数据、相同全参数训练，只将完整学习率曲线缩小十倍；500 步训练与每 100 步评估已全部完成。

| updates | 同 seed 0–9 闭环回报 | bootstrap validation 网络 L1 | DAgger training 网络 L1 |
| --- | ---: | ---: | ---: |
| 100 | 64.5651 | 0.0295938 | 0.0373280 |
| 200 | 71.5092 | 0.0295428 | 0.0330158 |
| 300 | 65.2749 | 0.0293278 | 0.0310662 |
| 400 | 73.2137 | 0.0291548 | 0.0294506 |
| 500 | 72.2640 | 0.0289767 | 0.0290618 |

第 100/200/300/500 步 DAgger 固定样本误差较原模型下降 24.5%/33.3%/37.2%/41.3%，同时闭环回报始终低于起点 105.4654。第 500 步相对原模型的配对回报差为 -33.2015，95% 区间 [-46.5998, -20.8784]；相对原学习率混合组为 +2.8918，区间 [-0.1338, +5.8034]。该组确实学到了新监督，但十倍降学习率没有解决最终退化，不能把“原学习率过大”当作充分解释。配对结果在 `diagnostics/lr_tenth_500/paired_returns.json`。

15:15 UTC 完成 50 个未见 seed 的确认：低学习率组平均回报 **70.7327 ± 3.7248**，平均存活 84.62 步。相对原模型的配对回报差为 -35.0337，95% 区间 [-40.7810, -29.5255]，50/50 局更差；相对原学习率混合组仅改善 +2.8164，区间 [+1.7273, +3.9160]。因此低学习率有小幅收益，却留下了绝大部分退化。结果为 `diagnostics/lr_tenth_500/confirmation.json`。

首轮三组及后续图像置换组的曲线由 `plot_results.py` 从逐 checkpoint 评估和固定样本结果生成，完整数值为 `training_controls.json`，可导出图为 `training_controls.pdf`。回报误差条使用相同 10 个 seed 的配对 bootstrap 95% 区间；这些点目前来自实际 sampler seed 42 的单次训练。

低学习率最终模型的固定首帧回报为 91.0547，较实时图像高 18.7908，配对 95% 区间 [14.3513, 23.4117]，10/10 局改善。因此十倍降学习率也没有消除实时图像/固定首帧的闭环差异。该结果保存在 `diagnostics/lr_tenth_500/image_ablation.json`；它仍不等价于证明固定 state 下每一步视觉响应都朝错误方向。

低学习率模型在相同 O/C 轨迹上的固定 state 检查也已完成：在旧策略状态上，将实时图像换成初始帧使全维生效命令 L1 增加 0.0023263，95% 区间 [0.0008804, 0.0039854]；在修正策略状态上增加 0.0007480，区间 [-0.0006618, 0.0021397]。这仍不支持局部实时视觉响应普遍有害的解释。完整分阶段误差和纠正方向已加入 `diagnostics/cross_analysis.json`。

![训练对照：闭环回报、DAgger 标签拟合与 bootstrap 验证误差](../../outputs/starvla/runs/humanoidbench_balance_simple/qwenoft/dagger-cause-20260909/training_controls.png)

### 10.11 条件实验：训练图像置换

15:07 UTC 启动 `permuted`：从同一个原模型出发，使用复现组的原学习率、混合数据、500 步预算和全参数微调，仅在每个 rank 的 32 条训练样本内置换图像。state、action/teacher label、其他辅助字段和样本顺序不变；每批图像集合不变。该干预检验训练中的图像与 state/label 对应关系是否参与退化，评估时仍使用正常实时图像。

入口是本轮实验目录内的 `train_permuted.py`，随后经 `train_sampler.py` 执行原 `train_starvla.py`。实际 sampler seed 为 42；图像置换使用每 rank 独立且持续推进的 `numpy.default_rng(20260909 + RANK)`，不会重复使用 accumulation 两个 microbatch 的随机种子。训练器显式关闭 `dispatch_batches`，DataLoaderShard 保留该 collate，因此置换作用于实际本地 microbatch。仅修改 train loader，eval loader 使用原 collate。

按真实 sampler 重建第一个 optimizer update 的两个 rank、各两个 microbatch：每批 32 条来自 31–32 个不同 dataset/episode，合计 128 条中 125 条换成了其他 episode 的图像。因而该干预确实跨轨迹改变配对；逐样本来源和置换索引见 `diagnostics/permutation_batch_provenance.json`。任务背景在不同图像间仍相似，不能把置换解释为删除全部视觉信息。

`check_permutation.py` 已用真实 train/eval loader 验证：所有非图像字段不变，图像集合不变，训练图像顺序改变，验证 batch 内容完全一致，全局 NumPy RNG 不变且独立置换流逐次推进。检查结果是 `diagnostics/permutation_check.json`；配置为 `permuted.yaml`，干预说明为 `permuted-intervention.json`，实际 Accelerate 命令和 `DAGGER_SAMPLER_SEED=42` 记录在 `commands.jsonl`。此实验必须通过上述 wrapper 启动，`run_one.py train` 仍是前三组使用的原训练入口。

| updates | 同 seed 0–9 闭环回报 | bootstrap validation 网络 L1 | DAgger training 网络 L1 |
| --- | ---: | ---: | ---: |
| 100 | 49.3535 | 0.0347103 | 0.0336857 |
| 200 | 66.1362 | 0.0341981 | 0.0318086 |
| 300 | 62.4668 | 0.0331420 | 0.0291215 |
| 400 | 71.7008 | 0.0309948 | 0.0272509 |
| 500 | 67.7147 | 0.0300171 | 0.0263571 |

15:35 UTC 完成 500 步。最终 DAgger 固定样本误差较原模型下降 46.7%，但回报相对原模型下降 37.7508，配对 95% 区间 [-51.2153, -25.1761]；相对未置换混合组变化 -1.6575，区间 [-4.1342, +0.4953]。因此，在本次 sampler seed 42 的对照中，置换训练图像确实学到了新监督，却没有恢复闭环表现。配对结果为 `diagnostics/permuted_500/paired_returns.json`。

50 个未见 seed 确认：置换组回报 **68.0837 ± 3.1125**，平均存活 81.78 步。相对未置换组的配对变化为 +0.1674，95% 区间 [-0.6657, +0.9827]；相对原模型为 -37.6827，区间 [-43.4968, -32.1158]，50/50 局更差。该集合上的置换收益接近零，仍留下几乎全部下降。记录见 `diagnostics/permuted_500/confirmation.json`。

最终模型的固定首帧回报为 68.0646，较实时图像仅高 0.3499，95% 区间 [-0.0716, +0.7912]。置换后，原先显著的首帧/实时图像闭环差异大幅缩小，但控制退化仍在；不能将“有害图像配对”当作已经得到修复实验支持的充分解释，也不能由这次置换失败排除全部视觉作用。结果为 `diagnostics/permuted_500/image_ablation.json`。

固定 state 的生效命令变化也确认了干预作用，而非仅有正负误差抵消：在共同的修正策略轨迹上，将实时图像换成初始帧引起的 normalized 命令 L1 变化幅度，由未置换模型的 0.0086780 降到置换模型的 0.0005627（约减少 93.5%）；在共同旧策略轨迹上，由 0.0095698 降到 0.0004441。区间及相邻帧结果见 `diagnostics/cross_analysis.json` 的 `initial_image_command_l1`、`lag_image_command_l1`。视觉敏感性被明显削弱，但本次干预没有同时恢复闭环回报。

### 10.12 条件实验：新策略访问状态的覆盖

图像置换未恢复回报，按计划比较两个从同一修正模型开始的 500 步全参数继续训练组：`old_states` 沿用原来的修正标签 DAgger 数据；`fresh_states` 替换为该修正模型新访问的 64,000 行 state/image 及其 conditioned teacher 标签。两组使用相同 fresh optimizer、原学习率曲线、bootstrap 数据、验证集、bootstrap normalization、global batch 128 和实际 sampler seed 42。

15:38 UTC 启动 `old_states`，15:41 UTC 启动新状态采集。新数据通过 `collect_fresh.py` 调用现有 `scripts.starvla.balance_dagger` 采集，再复用 canonical converter 和 `prepare_lerobot`。沿用 8 个环境、推理 batch 8、50Hz、H1 和 seed 0 起的序列；命令中的 `--round 0` 用于保持旧采集初始 seed 序列，学生权重明确指向修正模型。新模型 episode 长度改变，因此在固定 64,000 行预算下，最终覆盖的 episode 数会变化。

两个训练配置为 `old_states.yaml`、`fresh_states.yaml`，新混合表为 `fresh_states_mixture.json`。新采集及转换产物保存在 `fresh_collection/`，旧数据和旧 checkpoint 保留。该比较检验在共同继续训练预算下刷新访问状态的效果，不能把额外 500 步本身的收益归给数据刷新。

旧数据继续训练过程（步数相对于共同的修正模型起点）：

| additional updates | 同 seed 0–9 闭环回报 | bootstrap validation 网络 L1 | 旧 DAgger training 网络 L1 |
| --- | ---: | ---: | ---: |
| 0，修正模型 | 69.3722 | 0.0293237 | 0.0240075 |
| 100 | 67.8398 | 0.0339803 | 0.0278184 |
| 200 | 61.2756 | 0.0337775 | 0.0285712 |
| 300 | 67.6495 | 0.0317908 | 0.0249441 |
| 400 | 74.6546 | 0.0303832 | 0.0231621 |
| 500 | 74.0896 | 0.0292431 | 0.0216896 |

16:08 UTC 完成旧数据组训练。第 500 步相对共同的修正模型起点，回报改善 +4.7174，配对 95% 区间 [+2.0369, +7.8484]；旧 DAgger 误差进一步下降约 9.7%。相对原始模型，回报仍下降 31.3758，区间 [-44.8299, -18.7197]。因此额外训练确有部分收益，但这 500 步继续拟合旧监督没有恢复主要损失；新数据组的收益必须与这一对照比较。配对结果为 `diagnostics/old_states_500/paired_returns.json`。

旧数据组最终模型的固定首帧回报为 80.4689，比实时图像高 6.3793，配对 95% 区间 [1.8473, 10.9008]，7/10 局改善。首帧干预仍有收益，但两种条件下都未恢复原模型表现；结果为 `diagnostics/old_states_500/image_ablation.json`。

50 个未见 seed 的旧数据组回报为 **73.7202 ± 7.7542**，平均存活 85.08 步；相对共同修正模型起点改善 +5.8039，95% 区间 [+3.7704, +8.3004]，44/50 局更好。相对原始模型仍下降 32.0462，区间 [-38.2397, -25.8566]。额外训练的部分收益在确认集成立，但主要差距仍未消除。记录为 `diagnostics/old_states_500/confirmation.json`。

16:15 UTC 新数据采集、转换和 preparation 完成：64,000 个训练帧、809 个 episode、0 个新验证帧。`check_fresh_data.py` 用真实 loader 核对了两组前 64,000 次采样，bootstrap 样本身份和出现位置逐项一致：36,752 条 bootstrap、27,248 条 DAgger；bootstrap normalization 完全一致。检查结果为 `diagnostics/fresh_data_check.json`。16:17 UTC 启动 `fresh_states` 的 500 步训练。

另从新 DAgger 数据固定抽取 512 行（无放回，RNG seed 42），追加到单独的 `diagnostics/fit_samples_with_fresh.npz`；原有 80 行验证样本和 512 行旧 DAgger 样本逐项保持一致，原文件保留。`prepare_fresh_samples.py` 和配套 JSON 保存抽样来源，`fit.py --samples` 可对共同起点和新旧数据训练结果比较同一套监督。新增 512 行用于衡量新数据拟合，属于 training samples，不能当作独立泛化验证。

| 模型 | 旧 DAgger，512 行网络 L1 | 新访问状态 DAgger，512 行网络 L1 |
| --- | ---: | ---: |
| 共同起点：修正模型 | 0.0240075 | 0.0595538 |
| 旧数据继续训练 500 步 | 0.0216896 | 0.0575078 |
| 新数据继续训练 500 步 | 0.0355994 | 0.0242486 |

继续拟合旧数据仅将新状态样本误差降低约 3.4%，其误差仍远高于旧状态。这是待检验的数据覆盖问题的直接监督证据，尚不能单独证明刷新数据就能恢复闭环。完整结果为 `diagnostics/{corrected,old_states_500}/fit_fresh/`。

新数据训练过程（同一套固定样本，新增列为新 DAgger 的 512 行训练样本）：

| additional updates | 同 seed 0–9 闭环回报 | bootstrap validation 网络 L1 | 旧 DAgger 网络 L1 | 新 DAgger 网络 L1 |
| --- | ---: | ---: | ---: | ---: |
| 0，修正模型 | 69.3722 | 0.0293237 | 0.0240075 | 0.0595538 |
| 100 | 70.4921 | 0.0343166 | 0.0360053 | 0.0339452 |
| 200 | 73.8372 | 0.0336167 | 0.0384069 | 0.0311257 |
| 300 | 95.5340 | 0.0319531 | 0.0374174 | 0.0287222 |
| 400 | 75.3501 | 0.0302270 | 0.0362495 | 0.0254784 |
| 500 | 84.5283 | 0.0290631 | 0.0355994 | 0.0242486 |

第 300 步较旧数据同一步改善 27.8845，配对 95% 区间 [17.8841, 40.3972]，但第 400 步回落至 75.35；新状态误差仍持续下降。因此不能挑选 300 步单点宣布恢复，按预定的 500 步终点及独立确认集判断。

16:45 UTC 新数据组完成 500 步，最终初筛回报 **84.5283 ± 6.4400**，平均存活 96.5 步。相对旧数据同预算组改善 **+10.4387**，95% 区间 [+6.0554, +14.7669]，9/10 局更好；相对共同修正模型起点为 +15.1561，区间 [+11.0467, +19.5653]，10/10 局更好。相对原始模型仍下降 20.9371，区间 [-32.7693, -10.5843]。这支持数据刷新带来部分收益，尚未恢复全部退化；其中旧数据额外训练本身已有约 +4.72 的收益，不能全部计入刷新效果。记录为 `diagnostics/fresh_states_500/paired_returns.json`。

新数据最终模型固定初始图像时为 69.8599，较实时图像下降 14.6684，配对 95% 区间 [-18.3309, -11.3907]，10/10 局更差。旧数据继续训练组固定首帧反而改善 6.3793。因此本次数据刷新同时改变了闭环图像干预的方向，但固定首帧也会改变后续访问状态；仍须结合固定 state 的生效命令诊断解释。结果为 `diagnostics/fresh_states_500/image_ablation.json`。

50 个未见 seed 的最终确认：新数据组 **84.7274 ± 6.9400**，平均存活 95.42 步。相对旧数据同预算组改善 **+11.0072**，95% 区间 [+7.8102, +14.0214]，45/50 局更好；相对共同修正模型起点改善 +16.8111，区间 [+14.5092, +19.2602]。相对原始模型仍下降 21.0390，区间 [-26.0999, -16.1243]。结果为 `diagnostics/fresh_states_500/confirmation.json`；部分恢复在独立评估初始 seed 上成立，训练 sampler 的复验见 10.13。

![同起点旧/新 DAgger 数据对照：预定终点与完整训练曲线](../../outputs/starvla/runs/humanoidbench_balance_simple/qwenoft/dagger-cause-20260909/coverage_controls.png)

曲线由 `plot_results.py` 生成，数值和可导出图为 `coverage_controls.json` / `.pdf`。中图始终使用共同的旧 DAgger 512 行，因此新组误差上升与上方新状态 512 行误差下降并不矛盾；它们衡量不同监督分布。

新采集 seed 为 0–808，因而额外使用已保存的未见 seed 1,000,000,000–1,000,000,009 轨迹做覆盖和固定 state 诊断，避免把采集初始 seed 的重叠当作独立证据。每个策略取每局前 60 帧，共 600 帧；最近邻支持集分别为相同 bootstrap 加旧/新 64,000 行 DAgger。使用 bootstrap min/max 归一化、不截断，以各维差的 RMS 衡量距离。

| 未见轨迹，前 60 帧平均最近邻 RMS | bootstrap + 旧 DAgger | bootstrap + 新 DAgger |
| --- | ---: | ---: |
| 修正策略 C，全部 198D | 0.1095219 | 0.0360706 |
| 修正策略 C，physical 112D | 0.0844169 | 0.0261619 |
| 修正策略 C，command 86D | 0.1231070 | 0.0414483 |
| 原策略 O，全部 198D | 0.0345205 | 0.0584290 |
| 新数据训练后策略，全部 198D | 0.0505952 | 0.0668760 |

刷新数据使 C 轨迹距离下降约 67.1%，但 O 轨迹距离增加，说明支持域发生有方向的变化。raw state 超出 bootstrap min/max 的行占比，旧 DAgger 为 2,364/64,000（3.69375%），新数据为 17,575/64,000（27.46094%）；此项用 raw 与 raw 比较，不能把归一化值与原始边界混比。它描述采集分布，不能单独证明训练因果。复现脚本为 `measure_coverage.py`，完整分阶段、分维度数据为 `diagnostics/coverage_unseen.json`。新/旧数据刷新同时改变 state、图像和对应 teacher 监督，结论应限于这一联合干预。

另保存新数据最终模型在相同 10 个未见诊断 seed 上的实时轨迹（平均回报 82.7289，`diagnostics/fresh_states_500/live_unseen/`），并加入相同前 60 帧的距离计算。更新后的策略到 bootstrap + fresh-C 的距离为 0.0668760，高于采集策略 C 自身的 0.0360706；刷新训练后访问分布再次改变。这个观察限制了“一次刷新已覆盖所有后续状态”的说法，不能证明剩余约 21 分差距全部来自这一因素，也不据此无限追加训练轮次。

在同一批 **未见 seed 的 C 轨迹**上，固定 state 与 limiter 内部状态，比较实时图像下的实际生效命令（每局 t=1、11、21、31、41、51，共 60 个状态）：

| 模型 | 全维实际命令 normalized L1 | 腿部实际命令 normalized L1 | 腿部纠正增益 |
| --- | ---: | ---: | ---: |
| 共同起点 C | 0.0457690 | 0.0821992 | 0.28510 |
| 旧数据继续训练 500 | 0.0441661 | 0.0789102 | 0.30068 |
| 新数据继续训练 500 | 0.0167861 | 0.0289989 | 0.79235 |

新减旧的全维误差为 -0.0273800，按 episode 配对的 95% 区间 [-0.0290837, -0.0257337]；腿部纠正增益增加 +0.49166，区间 [+0.43818, +0.55009]。增益是相对当前 command 的模型修正向量在 teacher 修正向量上的投影比例，1 表示沿 teacher 方向的投影量相同，不表示整体动作完全一致。

固定同一 state 后换成初始图像，旧数据组的全维误差变化为 -0.0011336，新数据组为 +0.0020120；对应区间分别为 [-0.0019172, -0.0002769]、[+0.0014123, +0.0026216]。因此刷新数据同时改善了新访问域的动作纠正，并改变该域的局部首帧/实时图像误差关系。它为监督覆盖参与退化提供了**部分解释**，但不能从联合数据干预中独立识别覆盖的因果贡献，也不能由这一个 sampler 的部分回报恢复宣布唯一根因。原始结果为 `diagnostics/cross_unseen_analysis.json`；`compare_cross.py` 使用同 episode 的结果计算配对差，保存在 `diagnostics/coverage_cross_unseen_effects.json`。

### 10.13 实际采样 seed 复验

按批准计划，候选因素和对应对照使用实际 sampler seed 43、44 复验。`run_replica.py GROUP SEED` 克隆已有配置，只更换独立 run ID / W&B 名称，由 `train_sampler.py` 进入原训练器；`DAGGER_SAMPLER_SEED` 传给实际 `get_vla_dataset`，配置中的 trainer seed 仍为 0。双卡、global batch 128、全参数训练、初始化、优化器与学习率保持对应原组设置。每次命令和实际 seed 记录在 `commands.jsonl`，成功后写独立完成文件。

`check_sampler.py` 已通过真实 loader 验证：42/43/44 的首个 128 条训练样本顺序不同，同 seed 可重现；旧/新数据组同 seed 的 bootstrap 样本身份和位置逐项相同。前三个实际采样 seed 的前 64,000 行中，混合组分别含 36,752 / 36,779 / 36,638 条 bootstrap；bootstrap-only 三者均为 64,000。完整首个 update 的样本来源和计数为 `diagnostics/sampler_replica_check.json`。

16:48 UTC 根据预定 500 步初筛终点选定 `old_states` / `fresh_states`：数据刷新有明确的部分收益，需要检验它能否跨训练采样顺序保持；不将该收益解释为完全恢复。选择发生在读取 50 局确认结果之前，依据和顺序保存于 `replica_plan.json`。串行训练顺序为 old-s43、fresh-s43、old-s44、fresh-s44，每组 500 步；每个最终模型均使用共同的 50 个未见评估 seed。此时启动第一组 old-s43。

考虑到 sampler 42 的新数据曲线在 300/400/500 步间明显起伏，复验也评估每 100 步的已保存 checkpoint，并使用相同 10 个初筛 seed 和固定监督样本记录曲线；这些独立评估进程不改变训练器的内部 validation 节奏，终点仍预先固定为 500。

四组复验训练已全部完成，每组均保存 100–500 步权重。逐 rank 日志确认实际 sampler 43 / 44，四个模型都为 4,504.133M total / trainable parameters。以下为完整的初筛回报曲线：

old-s43、fresh-s43、old-s44、fresh-s44 分别在 2026-09-09 的 17:17:00、17:44:55、18:11:31、18:38:09 UTC 完成。最终配置核对确认同一 C 起点、fresh AdamW、global batch 128、原学习率曲线与 normalization；old/fresh 的其余差异仅为数据 mixture 路径和运行身份。`config.full.yaml` 中的 `pretrained_checkpoint: null` 是实际加载后由训练器清空字段以避免二次加载，日志保留了前面的成功加载记录，不能据此误判为随机初始化。

| 实际 sampler seed | 继续训练数据 | 100 | 200 | 300 | 400 | 500 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 42 | 旧 DAgger | 67.8398 | 61.2756 | 67.6495 | 74.6546 | 74.0896 |
| 42 | fresh-C | 70.4921 | 73.8372 | 95.5340 | 75.3501 | 84.5283 |
| 43 | 旧 DAgger | 71.4730 | 90.6438 | 95.9235 | 71.6396 | 78.4605 |
| 43 | fresh-C | 114.5846 | 73.2168 | 81.6683 | 95.6255 | 95.3842 |
| 44 | 旧 DAgger | 67.6947 | 62.3905 | 72.8979 | 70.8285 | 71.6503 |
| 44 | fresh-C | 69.6807 | 79.6574 | 72.6953 | 77.1504 | 82.4817 |

采样顺序会明显改变中间闭环表现；例如 old-s43 在 300 步为 95.92、400 步为 71.64，fresh-s43 在 100 步为 114.58、200 步为 73.22。监督 loss 不能作为闭环回报的单调替代指标。这里保留全部曲线，以预定 500 步比较两种数据，不使用各 seed 的最大值进行汇总。

2026-09-10 03:02 UTC 前，六个最终模型的独立 50 局评估、固定 state 诊断和首帧消融全部完成。最终确认使用共同的 seed 1,000,000,100–1,000,000,149：

| 实际 sampler seed | 旧数据回报 ± SD | fresh-C 回报 ± SD | 平均存活步数，旧 / 新 | 新减旧回报，配对 95% 区间 | 新组更好的局数 |
| --- | ---: | ---: | ---: | --- | ---: |
| 42 | 73.7202 ± 7.7542 | 84.7274 ± 6.9400 | 85.08 / 95.42 | +11.0072 [7.8102, 14.0214] | 45/50 |
| 43 | 76.7885 ± 6.5777 | 93.9289 ± 9.9707 | 88.10 / 101.92 | +17.1404 [13.6891, 20.6035] | 44/50 |
| 44 | 72.8698 ± 14.3722 | 83.8877 ± 8.3090 | 86.04 / 92.38 | +11.0179 [5.7037, 15.3879] | 44/50 |

平均存活时间也在三组改善：新减旧分别为 +10.34 / +13.82 / +6.34 步，配对 95% 区间分别为 [6.7795, 13.7600] / [9.6395, 17.9600] / [1.0595, 10.9400]，41/50、43/50、38/50 局更长；这些步数使用相同的 50Hz 控制频率。

三个采样顺序下收益方向一致，成对均值差的平均值为 +13.0552，范围 [11.0072, 17.1404]。此处每个区间只描述**固定训练结果下，50 个成对评估 episode 的不确定性**；没有将 3×50 局当成 150 个独立训练重复。配置 trainer seed 始终是 0，复验只改变实际数据 sampler seed；也只采集了一份 fresh-C 数据，因此不是三次独立采集或全部训练随机性的复验。

与共同起点 C 相比，三个新组分别改善 +16.8111 / +26.0126 / +15.9714；相对原始模型 O 仍下降 21.0390 / 11.8375 / 21.8787，配对 95% 区间分别为 [-26.0999, -16.1243] / [-18.0297, -5.8196] / [-27.3241, -16.6008]。不能将 sampler 43 的较高回报或某个中间 checkpoint 当作已经恢复原模型的证据。

在共同的未见 C 轨迹上，仍使用 10 个 episode、每局六个固定 state 和同一 teacher / limiter：

| sampler seed | 全维生效命令 L1，旧 → 新 | 新减旧 L1，配对 95% 区间 | 腿部纠正增益，旧 → 新 | 新减旧增益，配对 95% 区间 |
| --- | ---: | --- | ---: | --- |
| 42 | 0.0441661 → 0.0167861 | -0.0273800 [-0.0290837, -0.0257337] | 0.30068 → 0.79235 | +0.49166 [0.43818, 0.55009] |
| 43 | 0.0429594 → 0.0168290 | -0.0261304 [-0.0280743, -0.0243032] | 0.32660 → 0.78633 | +0.45973 [0.40405, 0.51795] |
| 44 | 0.0437449 → 0.0165810 | -0.0271639 [-0.0290056, -0.0254355] | 0.29542 → 0.78861 | +0.49319 [0.43947, 0.55042] |

动作纠正改善跨采样顺序成立；bootstrap 重采样单位为 episode，不能把同局六个状态当作独立轨迹。这批 C 状态适合检查训练是否改善了采集策略所访问的域，不代表每个训练后模型自身的完整闭环状态分布。

首帧干预的闭环回报变化（固定初始图像减实时图像，seed 0–9）：

| sampler seed | 旧数据组，配对 95% 区间 | fresh-C 组，配对 95% 区间 |
| --- | --- | --- |
| 42 | +6.3793 [1.8473, 10.9008] | -14.6684 [-18.3309, -11.3907] |
| 43 | +2.1021 [-4.4431, 8.1716] | -25.3059 [-34.2443, -17.9336] |
| 44 | +10.1538 [1.5448, 19.2041] | -10.9909 [-14.5580, -7.5195] |

三个 fresh-C 模型均为 10/10 局固定首帧后更差；旧数据组 sampler 43 的区间包含零，不能说旧组三次都有明确收益。固定未见 C state 时，fresh-C 三组换首帧的全维 L1 分别增加 0.0020120 / 0.0023585 / 0.0024930，区间均高于零；旧组三者均值为负，但 sampler 44 的区间包含零。实时图像在刷新后的策略中有可测的正面作用，不能据此反推刷新前全部退化都由错误视觉反馈造成。

相同固定样本的最终网络 L1：

| sampler seed | 旧组拟合旧 DAgger | 新组拟合旧 DAgger | 旧组拟合 fresh-C | 新组拟合 fresh-C | bootstrap validation，旧 / 新 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 42 | 0.0216896 | 0.0355994 | 0.0575078 | 0.0242486 | 0.0292431 / 0.0290631 |
| 43 | 0.0214165 | 0.0355895 | 0.0573638 | 0.0240962 | 0.0297108 / 0.0292077 |
| 44 | 0.0215598 | 0.0359507 | 0.0574232 | 0.0239647 | 0.0293167 / 0.0291686 |

新组确实学到了新访问域的监督，并非靠几乎不更新保住原策略。旧域误差上升在三组也都存在；数据替换产生了明确的分布取舍。两类 DAgger 各 512 行都是训练样本，只有 bootstrap 的 80 行来自 validation。

![sampler 43：完整训练曲线](../../outputs/starvla/runs/humanoidbench_balance_simple/qwenoft/dagger-cause-20260909/coverage_controls_s43.png)

![sampler 44：完整训练曲线](../../outputs/starvla/runs/humanoidbench_balance_simple/qwenoft/dagger-cause-20260909/coverage_controls_s44.png)

完整三组汇总为 `replication_results.json`；各模型的 `confirmation.json` / `image_ablation.json` 保留配对结果，`diagnostics/coverage_cross_unseen_effects{,_s43,_s44}.json` 保留分阶段、分维度的固定 state 效果。曲线另有同名 `.json` 和可导出的 `.pdf`。

### 10.14 本轮解释范围

| 分类 | 证据与结论 | 边界 |
| --- | --- | --- |
| 支持 | 旧轮存在标签语义 bug；对齐 conditioned command 后，回报从约 23 改善到约 68。修正标签混合训练可逐张量复现；sampler 42 下 bootstrap-only 保持约 105，混合组降至约 68。 | 后一对照支持数据组成与更新的相互作用参与退化；替换混合数据也增加了 bootstrap 暴露量，不能把所有效应只归于 DAgger 标签内容。bootstrap-only 尚无 43/44 训练复验。 |
| 在当前条件下排除为充分解释 | 完整学习率曲线缩小十倍仍留下约 35 分差距；batch 内图像置换大幅削弱图像敏感性，却留下约 38 分差距。 | 两组仅做了 sampler 42；排除的是这两个具体干预能解决主要退化的说法，不能排除优化过程或全部视觉作用。固定 state 结果也不支持局部实时视觉响应普遍有害。 |
| 部分解释 | 从共同 C 起点、相同 500 步预算出发，fresh-C 联合数据刷新在三个 sampler 上同时改善固定 C 状态动作纠正与独立闭环回报。覆盖距离也明显缩小；配回旧数据的对应训练组保留较大误差和较低回报。 | 因果干预单位是整份访问数据。它支持刷新数据的部分恢复效应，为覆盖不足提供证据，但不能独立识别 state 覆盖作为中介的贡献，更未证明唯一根因。 |
| 仍未区分 | 刷新同步改变 state、图像、条件 teacher 标签、episode 分布和 O-policy 支持域；中间闭环回报不随拟合误差单调变化。 | 剩余约 12–22 分差距尚未解释。更新后自身访问分布再次移动是一个观察，尚未证明剩余差距都由它造成；本轮没有依据宣布单轮 DAgger 本身是 bug。 |

本轮完成了批准计划中的有限对照、条件实验和候选因素复验。结论是**已识别可重复的部分恢复因素，尚未获得全部残余退化的唯一根因或完整修复**。各原始数据和 checkpoint 均保留。

### 10.15 复现入口与产物

以下路径均相对于 `outputs/starvla/runs/humanoidbench_balance_simple/qwenoft/dagger-cause-20260909/`：

| 内容 | 文件与入口 |
| --- | --- |
| 训练、采集、评估的实际命令 | `commands.jsonl`，包含命令、工作目录及干预环境变量；`replica_plan.json` 保存候选选择时间和依据 |
| 训练控制与权重 | 各组 `.yaml`、`training/GROUP/config.full.yaml`、逐 rank 日志、`GROUP-complete.json`、`training/GROUP/checkpoints/steps_{100,200,300,400,500}_pytorch_model.pt` |
| 原三组训练及 checkpoint 评估 | `run_one.py train GROUP` / `run_one.py eval GROUP`；最终确认参数为 `--steps 500 --episodes 50 --seed-start 1000000100` |
| 实际 sampler 复验 | `run_replica.py old_states 43` 等四次调用，经 `train_sampler.py` 使用明确的数据 seed；sampler 42 的旧/新组使用原入口 |
| 图像置换干预 | `train_permuted.py` 与 `train_sampler.py`，必须保留 `commands.jsonl` 中对应的 wrapper 和环境变量 |
| fresh-C 数据 | `collect_fresh.py`、`fresh_collection/`、`fresh_states_mixture.json`；固定新监督样本由 `prepare_fresh_samples.py` 生成 |
| 逐局与逐步原始结果 | `evals/*/queue_eval_latency_0.json`；`diagnostics/*/{live,initial,live_unseen,initial_unseen,cross,cross_unseen}/`，具体条件按已执行模型保存 |
| 同输入诊断与统计 | `diagnose.py`、`fit.py`、`measure_coverage.py`、`analyze_cross.py`、`compare_cross.py`、`summarize_replicas.py` |

在仓库根目录重算已保存结果（无需 GPU 或重训）：

```bash
EXP=outputs/starvla/runs/humanoidbench_balance_simple/qwenoft/dagger-cause-20260909
PYTHONPATH=. .venv/bin/python "$EXP/analyze_cross.py" > "$EXP/cross-analysis.log"
PYTHONPATH=. .venv/bin/python "$EXP/plot_results.py"
PYTHONPATH=. .venv/bin/python "$EXP/summarize_replicas.py"
for SUFFIX in '' _s43 _s44; do
  PYTHONPATH=. .venv/bin/python "$EXP/compare_cross.py" \
    "$EXP/diagnostics/cross_unseen_analysis.json" \
    "old_states${SUFFIX}_500" "fresh_states${SUFFIX}_500" \
    "$EXP/diagnostics/coverage_cross_unseen_effects${SUFFIX}.json"
done
```

`check_permutation.py`、`check_fresh_data.py`、`check_sampler.py` 已通过真实 loader 的语义检查，结果分别保存在 `diagnostics/permutation_check.json`、`fresh_data_check.json`、`sampler_replica_check.json`。复现模型的 732 个权重张量与全部 100 条 loss 逐项一致，见 `reproduce-equivalence.json`。训练控制审计确认全部参数参与训练；本轮实验脚本和配置留在独立输出目录，没有新增生产 API。

最终核对覆盖全部 50 个训练 checkpoint 及对应初筛/固定样本结果、六个覆盖实验终点各 50 局确认结果、首帧 episode 顺序，以及由原始逐局数据重算的三组回报/存活配对区间。16 个实验脚本通过语法编译，文档通过空白检查；所有本轮 GPU 进程已结束。

## 11. 2026-09-10：持续十轮 DAgger 对照

验证单轮下降是否会在持续刷新数据后恢复：scratch 从原始 Qwen3-VL 加随机动作头和 state projector 开始，checkpoint 从原始 formal QwenOFT 开始。两组都跳过本次 SFT，各自从新的 AdamW 开始；在 `81.85.0.163`、`81.85.1.68` 的单张 RTX PRO 6000 96GB 上分别执行。

实验根目录为 `outputs/starvla/experiments/balance-dagger10-20260910/`；远端仓库根目录均为 `/home/xinyuan/balance-dagger10`。两组通过两轮真实 smoke 后，于 **08:35:44 UTC** 启动首轮采集，各自完成 64,000 行。checkpoint 组在完成 87 次更新后遇到优化器显存分配 OOM，scratch 组在第 84 步主动暂停。两组于约 **09:15 UTC** 统一改用 micro-batch 8、accumulation 16，从各自原始起点、新 optimizer 重新训练，复用首轮数据；原尝试单独归档。截至 **17:35 UTC**，两组均完成 3000/5000 updates 与对应 50 局评估，正在采集第七轮；重启后没有再发生 OOM。

**2026-09-11 02:16 UTC 更新：尚未完成。** 两组第七轮数据采集完成后，分别在前一天 **17:49:28 / 17:56:32 UTC** 的训练启动阶段 OOM，未产生第 3001 步更新。错误中还记录了两个额外 GPU 进程，分别占用约 22.04 GiB 和 686 MiB；当时训练进程自身已占用约 72.2 GiB。额外进程的归属尚未确认。自动回传仅等待后续轮次，没有及时识别训练退出，导致中断期间一直没有新更新。02:14 左右确认两张 GPU 已空闲后，两组使用既有第七轮数据和 `steps_3000_state` 重新启动；02:21 UTC 已分别推进至 3030 / 3033 步。已核对首个恢复更新为 3001，head/base LR 为 `3.788479513026289e-5 / 1.1365438539078866e-5`，与原 5000-step cosine 完全一致，未重置 warmup；继续使用 micro-batch 8、accumulation 16。失败日志与配置存于 `ARM/attempt-round07-gpu-contention/`。回传等待现已读取 launcher 退出码，非零退出会立即报错；新增进程级显存记录，以便定位再次发生的并发占用。

**2026-09-11 04:03 UTC 更新：3500/5000 updates，八轮数据均已采集。** 两组完成 3500 步的 50 局评估后，与用户在另一项目目录启动的 GPU 任务并行采完第八轮数据。新任务各占约 22.7 GiB；scratch 随后在 03:51:23 的训练启动阶段 OOM，checkpoint 尚在加载 `steps_3500_state`、没有新更新时主动停止，以保留用户的新任务。日志分别存于 `ARM/attempt-round08-gpu-sharing/`，checkpoint 的这次停止不记作 OOM。按用户的并行使用需求，`experiment.py` 现在只在启动训练前等待已有 GPU 进程退出，评估和采集不加这个等待。两组已从 3500 步重新进入流程，复用第八轮数据，当前在训练入口等待；GPU 空闲后自动继续。batch、数据和学习率协议均未改变。

**04:52 UTC：已自动续跑。** 其他 GPU 任务结束后，两组分别推进至 3603 / 3570 步；两张 GPU 上均只有本实验的训练进程。两组第七轮产物也已回传，通过 3001–3500 连续更新、64,000 行数据和 50 局评估检查；目前共有 14 轮完整产物在本地。

**06:00 UTC：4000/5000 updates 及对应评估完成。** scratch / checkpoint 的 50 局平均回报为 **72.996018 / 323.809985**，平均存活 **84.90 / 340.84** 步。两组前八轮的 16 份完整产物均已回传并通过语义检查，正在用各自 4000 步模型采集第九轮；本次恢复后的 3501–4000 更新没有再遇到 OOM。

**09:20 UTC：两组十轮训练和逐轮 50 局评估全部完成。** scratch / checkpoint 终点平均回报为 **144.532006 / 377.179249**；20 份逐轮完整产物均已回传并通过检查。scratch 的独立 100 局确认于 08:44 UTC 完成，平均回报为 **143.576013**；checkpoint 的独立确认仍在运行，随后评估原始模型。完整训练状态正在回传，尚未完成全流程验收。

### 11.1 固定协议

- 10 轮 × 500 optimizer updates，共 5000；global batch 128。轮间恢复完整 optimizer/scheduler，warmup 100 步只执行一次，cosine 覆盖 5000 步。base/head LR 为 `3e-5 / 1e-4`，base 最低为 `1e-6`。
- 每轮最新 student 执行 64,000 行、8 环境、50Hz、H1，使用已修正的 conditioned teacher command。采集 seed 为 `round_index × 1,000,000`。
- 保留全部历史 DAgger；mixture 配置为 bootstrap 0.5，其余 0.5 在已采集轮次中均分。沿用现有 sequential sampler：它按拼接后的行采样，配置权重并非实际抽样概率。逐轮记录实际消费样本，bootstrap 占比会随累计数据增加而下降。
- trainer seed 0、实际 sampler seed 42；固定 bootstrap normalization 与 validation、198D state、43D action、224×224 实时图像、L1、全参数训练。
- step 0、500……5000 各使用 50 个共同 seed `1000000200–1000000249`；终点两组及原始模型另用 100 个共同 seed `1000001000–1000001099` 确认。零延迟、单环境、每局最多 1000 步。
- 保留每轮数据与独立权重、最近两份完整训练状态；不因早期下降停止，不以最优中间点替代预定终点。

### 11.2 部署与批量实测

两机重跑同一原始模型的 50 局结果逐项相同：平均回报 **102.924974**、SD **18.104805**、平均存活 **113.22** 步。当前 seed 集不同于此前约 107 分的基线；比较以本次共同种子结果为准。原始模型文件与固定公开版本逐字节一致；来源记录见 `model-source.json`，环境版本见各主机的 `vla-environment.json` / `task-environment.json`。

| micro-batch | accumulation | .163 秒/update | .68 秒/update | 峰值 reserved GiB | 结果 |
| ---: | ---: | ---: | ---: | ---: | --- |
| 32 | 4 | — | — | — | 两机首个 forward OOM |
| 16 | 8 | 5.431 | 5.266 | 93.385 | 12 updates、validation、恢复至 13 均通过 |
| 8 | 16 | 6.829 | 6.668 | 93.035 | 12 updates、validation、恢复至 13 均通过 |

短测最初选择 micro-batch 16 × accumulation 8，长程 OOM 后改为 **micro-batch 8 × accumulation 16**。耗时统计的是完整 optimizer update，取第 4–11 次的中位数；未使用仅覆盖最后一个 micro-batch 的计时。恢复第 13 步的 head/base LR 为 `1.3e-5 / 3.9e-6`，对应一次连续 warmup。试跑权重不作为正式起点；验证后清理了这些独立试跑的训练状态，保留配置、逐步指标及日志。

上述短测未覆盖长程显存分配峰值：首轮 checkpoint 组在 DeepSpeed 将梯度分区转为 fp32 时请求 16.53 GiB，而当时可用显存为 16.47 GiB，另有 1.65 GiB reserved-but-unallocated。复查发现生产入口已设置 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`；曾短暂重试其新环境变量别名，确认不是有效改动后取消，转为降低 micro-batch。运行设置与长程峰值、耗时分别记录在 `ARM/runtime.json`、在线配置及逐步指标中。`ARM/attempt-mb16/` 保留失败或中断尝试，checkpoint 组的 `attempt-mb16-retry/` 保留取消的重试；这些更新不计入重新开始的 5000 步。

两组真实 smoke 均完成 0→2→4 updates、三点评估和跨轮完整恢复，global batch 为 128。`verify_smoke.py` 在清理试跑权重前验证了两份完整状态、逐轮实际消费各 256 行、累计数据及评估权重对应关系。期间修复了实验日志脚本在 W&B 关闭后写 summary、以及训练与评估共用内置事件 step 的冲突；模型状态已保存并成功恢复，smoke 训练指标已从本地记录补齐并通过 API 回读确认。

正式 step 0 的 50 局平均回报 / 平均存活步数为：scratch **8.469998 / 12.06**，checkpoint **102.924974 / 113.22**。原始统计及在线链接见 `step-zero.json`。

### 11.3 入口、在线记录与统计

`experiment.py` 通过生产 `scripts.starvla.train_task` 执行；`initialize.py` 生成 seed-0 scratch 起点。`profile_train.py` 记录完整 update 耗时、显存峰值、实际消费样本及在线配置。`run_arm.sh ARM 8` 连续执行正式十轮及终点确认，恢复时追加 `--resume`。

实际部署代码保存在 `deployed-code/`；两台远端的训练入口已核对一致，并与原 `code-overlay.tar.gz` 对应。工作区后来加入的 OOM 自动减小 micro-batch 逻辑没有同步到这两组运行，实验仍保持固定 micro-batch 8。`code-overlay-current.tar.gz` 保留实际部署的入口与测试快照，并补入最新的实验等待、回传和验证脚本；配合 `code.tar.gz` 和 `source-revisions.json` 复现，详见 `deployed-code/source-manifest.json`。

W&B 使用 `saberrr-zju/starvla_tasks`。最初批量试跑为 disabled，已由 `backfill_probes.py` 补传完整逐步历史，并明确标记 `backfilled: true`；后续训练、smoke 和闭环评估使用 online。每组的训练和逐轮评估接续同一个 run，评估横轴为 `global_step`：

- [部署基线](https://wandb.ai/saberrr-zju/starvla_tasks/runs/balance-dagger10-20260910-setup)
- [scratch 重启组](https://wandb.ai/saberrr-zju/starvla_tasks/runs/balance-qwenoft-balance-dagger10-20260910-scratch-dagger-mb8)
- [checkpoint 重启组](https://wandb.ai/saberrr-zju/starvla_tasks/runs/balance-qwenoft-balance-dagger10-20260910-checkpoint-dagger-mb8)

原尝试保留在上述 ID 去掉 `-mb8` 后对应的 run 中；短暂的 checkpoint 重试使用 `-expandable` 后缀。正式比较使用 `-mb8` 重启组，避免把旧尝试的前 84/87 步与重新训练后的曲线拼接。

step 0 的已有评估结果在训练达到 2500 步之后补录，W&B 内置事件 `_step=2505`，而该行的 `global_step=0`、`eval/global_steps=0`。因此使用默认 `Step` 横轴时，初始值会出现在 2.5k 附近；应选择 `global_step`。实际 step 2500 只有一份 50 局评估，scratch / checkpoint 分别为 58.265619 / 190.080707。补录仅上传已有记录，没有重新评估或更新模型。

共同 50 个 seed 的完整逐轮结果如下；独立 100 局确认尚在进行。

| Updates | scratch 平均回报 | checkpoint 平均回报 | scratch 平均存活步数 | checkpoint 平均存活步数 |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 8.470 | 102.925 | 12.06 | 113.22 |
| 500 | 23.913 | 70.696 | 43.06 | 83.74 |
| 1000 | 63.130 | 96.684 | 83.18 | 106.28 |
| 1500 | 54.467 | 158.576 | 69.30 | 172.80 |
| 2000 | 85.327 | 141.330 | 98.90 | 156.12 |
| 2500 | 58.266 | 190.081 | 72.00 | 206.40 |
| 3000 | 64.249 | 275.124 | 78.16 | 292.40 |
| 3500 | 79.934 | 263.055 | 91.14 | 281.18 |
| 4000 | 72.996 | 323.810 | 84.90 | 340.84 |
| 4500 | 101.492 | 334.003 | 115.00 | 352.86 |
| 5000 | 144.532 | 377.179 | 157.30 | 397.54 |

`micro-batch-comparison.json`、`servers/HOST/probe-mb{8,16}/` 保存批量证据；`wandb-probes.json` 保存四个试跑链接。正式结果将保存在 `ARM/evals/steps_N/`，每个评估点保留逐局 seed、回报与长度；`ARM/rounds/steps_N/` 保留对应权重，`ARM/sampling-steps-N.json` 保留实际采样。全部完成后由 `summarize.py` 生成 `rounds.csv`、`comparison.json`、PNG/SVG 曲线和 `REPORT.md`，用配对 episode bootstrap 95% 区间回答是否先降后升、是否超过原模型、scratch 是否追上 checkpoint。本次每组一个训练 seed，区间仅衡量固定模型下的 episode 评估不确定性。

回传使用 `transfer.py`：W&B 项目权限已回读确认为 `PRIVATE`，原始 SSH 下载每连接约 20–24 KiB/s，而私有 artifact 中转在首批真实数据上达到约 18–87 MiB/s。下载通过 SDK 完整性校验并成功解包后，删除临时云端 artifact 及打包文件；原始远端数据继续保留。传输记录在 `transfers/NAME/{upload,download}.json`。首轮两组数据均已返回工作区，`verify_round_data.py ARM 0` 验证各 64,000 行、198D state、43D action、嵌入图像可读及固定 bootstrap normalization；scratch / checkpoint 分别包含 5,288 / 555 个 episode。

截至 2026-09-11 09:20 UTC，两组全部十轮的独立权重、完整数据、训练日志、逐步指标和评估记录均已回传；20 轮检查均验证每轮 64,000 行、500 次连续 optimizer updates、global batch 128、正确的跨轮恢复与采集模型、50 个共同评估 seed。`return_remaining.py` 继续回传终点确认和最终两份完整训练状态；当前尚无全流程完成标记。`verify_final.py` 将在回传完成后检查 5000 次更新的学习率连续性、4500/5000 完整状态中的 AdamW/scheduler 步号，以及三组确认评估的模型路径和共同 seed。

scratch 初始权重在本地使用相同环境、构造器与 seed 0 重建，序列化文件 SHA-256 与服务器原文件完全一致，见 `scratch-initial/reconstruction.json`。因此该本地权重是字节相同的起点副本。
