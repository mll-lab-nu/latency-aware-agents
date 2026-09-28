# MIKASA conditioned H8 Profile / Fixed-2 运行记录

本记录保存 `mikasa_conditioned_h8` 的实现、smoke 和正式运行时间线。稳定的数据合同、
训练参数与复现入口见
[MIKASA_ACTION_CHUNK_H8](../MIKASA_ACTION_CHUNK_H8.md)；只有正式完成并核验的结果才写入
[PROGRESS](../PROGRESS.md)。

## 实现与数据审计

- 新任务合同为 H8×7 action、71D state，以及 `action_prefix_mask` 和
  `action_loss_mask`。QwenOFT 只使用 71D action context 与 suffix loss mask；没有新增
  OFT 专用 prefix projector。
- 三份 Profile 数据固定到 RTX 3090 revision
  `66414463f2a609316da63935941a79abf0a73ce9`，OFT/PI/GR00T 的 K 分别为 2/3/2；
  Fixed-2 严格为 100ms、K=2，三模型共享。
- collector 使用 zero-latency H1 PPO teacher 在 private ManiSkill env 中重放 prefix 并
  逐帧生成 suffix。主分支仍由真实 Profile 或 Fixed-2 scheduler 执行，失败 episode
  不写入训练集。
- state-copy smoke 比较了复制后的 state，以及相同动作推进五步后的 state、observation
  与 reward；最大偏差分别为 `1.19e-7`、`1.97e-6` 和 `1.98e-9`。
- 四个 collector smoke 均通过 raw/LeRobot 逐样本审计。Profile OFT/GR00T 和 Fixed-2
  各采到 seeds 0/1/2；Profile PI 在 35 次顺序尝试中采到 seeds 3/25/34，说明它的正式
  275-success 采集预计明显慢于另外三路。

## 训练与评测 smoke

- 六个训练组合均完成 2 optimizer-step smoke、checkpoint 和 inference bundle 写入。
  OFT/PI 的 micro batch 128 可运行；GR00T 使用 micro batch 64 / accumulation 2。
- 三类评测各完成 2 episodes：Profile PI→Profile 的首个 apply frames 为 3/3；
  Profile OFT→Zero 为 0/0；Fixed GR00T→Fixed-2 为 2/2。三者均通过 capacity、首动作、
  returned-action context、后续 chunk head 和 Fixed-2 exact-frame 审计。
- Conditioned H8 专项测试为 `16 passed`；扩大到数据、scheduler、executor factory 与
  policy 的相关选择集为 `106 passed`，另有一个既有 device-count 测试在 8 卡主机上
  使用了主机可见设备而失败，与本合同无关。

## 正式流水线

正式输出根为：

```text
/home/ubuntu/codex/mikasa-h8/outputs/starvla/datasets/mikasa_intercept_grab_fast/conditioned-h8-v1
/home/ubuntu/codex/mikasa-h8/outputs/starvla/runs/mikasa_intercept_grab_fast/h8-conditioned
```

2026-09-21 已启动四路正式采集：GPU 1/2/3 分别生成三模型 Profile 数据，GPU 0 生成
共享 Fixed-2 数据。采集结束后，后继任务先要求四份 `dataset_audit.json` 全部存在，才会
启动六路正式训练；首批只启动 GPU 0/1/2/3。GPU 6/7 等待 GPU 0/1/2 中任意一路结束，
随后依次提交。所有训练使用独立目录和 rendezvous port，OOM 不自动改 batch。

六路训练全部完成后，终评汇总 success、return、length、实际 latency、drop、underflow、
执行 head 分布与 chunk audit。
首次启动时正式数据、训练和评测尚未完成，因此没有把运行中状态或 smoke 数字写入
PROGRESS；后续状态见下节。

## 首次正式训练失败与重启

四份正式数据随后均完成 275 个成功 episode、250/25 切分和 `dataset_audit.json`
审计。首次训练没有完成任何 optimizer step：主机的 W&B 凭据已切换到另一个账号，
而启动环境仍指定旧 entity，导致 GPU 0/1/2/3 四路在 W&B 初始化时收到
`permission denied`；它们退出后又暴露了等待循环的 `kill -0` 竞态，GPU 6/7 两路被
继续提交，并与当时占用这两张卡的外部任务冲突而 OOM。六路都没有产生 checkpoint。

失败目录已完整保留为
`outputs/starvla/runs/mikasa_intercept_grab_fast/h8-conditioned-failed-20260920T2135Z`。
等待逻辑已改为：确认 GPU 0/1/2 中首个结束任务成功后才继续，并在提交 GPU 6、7 前
分别等待目标卡没有 compute process；失败不再触发后续两路。2026-09-21 使用当前凭据
对应的 W&B entity 从空正式目录重启 GPU 0/1/2/3 四路，固定 batch、数据和 seed 均未
改变；GPU 6/7 仍等待首批完成及外部占用释放。

## 评测调度修订

2026-09-21 用户取消 `profile-trained → zero`，终评矩阵保留每个模型的
`profile-trained → profile`、`fixed-trained → fixed-2` 和
`fixed-trained → zero`，共 9 组。已启动的 QwenOFT `profile → zero` 被停止，
未完成输出保留在带 `cancelled` 后缀的目录。提前评测改为在物理 GPU 1 上串行运行，
simulator 与 inference 共用该卡。

随后确认 zero-trained baseline 也需覆盖三种评测条件：已有的 `zero → zero/profile`
不重跑，新增三个模型的 `zero → fixed-2`。因此每模型最终报告六格：zero-trained
三格、fixed-trained 两格、profile-trained 一格；新增评测仍在物理 GPU 1 上串行执行。

## Fixed-2 GR00T 恢复（后续已失效）

Fixed-2 QwenPI 完成 5,000 steps 并导出 inference bundle。Fixed-2 QwenGR00T 原进程在
约 step 4,260 后退出且未写出 step-5,000；最近的完整恢复状态是 step 2,500。第一次
恢复仍指向无权限的旧 W&B entity，第二次缺少 Conda CUDA toolkit 的 `CUDA_HOME`，两次
都在继续 optimizer step 前退出。随后使用本地 `.env`、`saberrr-zju` entity 和 GR00T
Conda 环境内的 CUDA 12.8 toolkit，从 step 2,500 成功恢复；W&B 初始化、checkpoint
载入和 GPU 7 进程均已核验。恢复命令完成后串接 GPU 1 串行正式评测。

Fixed-2 QwenPI 的 `fixed-2` 与 `zero` 评测按后续调度要求改在物理 GPU 6 上串行运行，
simulator 与 inference 共用该卡；这两组不再等待 GPU 1 或 GR00T 恢复完成。
两组评测曾完成，但对应 Fixed-2 checkpoint 后续被判定有问题，因此不再保留或引用其
性能数字；本节只保留当时的调度与故障记录。

## 已核验正式结果

截至 2026-09-21，所有已完成 setting 均为 50 episodes、reset seeds
`4242424242–4242424291`、20/20Hz、60 transitions、capacity 1、无首动作冻结。
Profile 使用 revision `66414463f2a609316da63935941a79abf0a73ce9`、latency seed 0；
Fixed-2 严格为 100ms / 2 frames。下表为 `success_once；return mean±SD`。

| 模型 | zero→zero | zero→profile | profile→profile |
|---|---:|---:|---:|
| QwenOFT | 25/50；19.42654±16.93815 | 1/50；3.97404±4.89508 | 15/50；13.39470±15.06934 |
| QwenPI v3 | 34/50；26.42206±13.34111 | 14/50；9.93412±14.21984 | 14/50；12.79009±13.64533 |
| QwenGR00T | 31/50；22.92212±16.49738 | 18/50；15.01258±16.16588 | 16/50；13.35319±13.68371 |

| 模型 | zero→zero | zero→fixed-2 | fixed-2→fixed-2 |
|---|---:|---:|---:|
| QwenOFT | 25/50；19.42654±16.93815 | 5/50；6.38940±10.10137 | —（已失效并删除） |
| QwenPI v3 | 34/50；26.42206±13.34111 | 17/50；12.75486±15.02083 | —（已失效并删除） |
| QwenGR00T | 31/50；22.92212±16.49738 | 19/50；13.90434±14.82513 | —（已停止并删除） |

原始 JSON 位于正式输出根下：zero-trained 为
`h8-zero-latency/<model>/inference/eval_{l0,profile,fixed_2f}.json`，conditioned 为
`h8-conditioned/{profile-latency,fixed-2}/<model>/inference/eval_*.json`。Profile
三路的 step-5,000 checkpoint 与 inference bundle 均存在；Fixed-2 三路 checkpoint 已删除，
对应 fixed-trained 评测不再属于正式矩阵。当前有效 setting 的 mean length 均为 60，
execution audit 均通过；Profile conditioned checkpoint 尚未发布。

## Fixed-2 失效与删除

2026-09-21，用户确认三个 Fixed-2 checkpoint 的合同均有问题。正在恢复的 GR00T 训练已
停止，GPU 已释放；OFT、PI、GR00T 三个 run 下的训练 checkpoint 目录和 inference model
已永久删除，删除前训练 checkpoint 合计约 359 GB。OFT、PI 已生成的
`fixed-trained → fixed-2` JSON、acceptance 与 chunk execution audit 同时删除，相关数值
从正式结果表移除。`zero-trained → fixed-2` 使用独立的 zero H8 checkpoint，继续作为有效
baseline 保留。Fixed-trained → zero 的旧原始文件只作为失效诊断残留，不得作为正式结果引用。

## Profile-v2 时序修复与旧结果失效

旧 Profile `15/50`（QwenOFT）、`14/50`（QwenPI v3）和 `16/50`（QwenGR00T）不再作为
正式结果：旧 executor 会在同帧已到达结果尚未接收时冻结过期 prefix，旧数据又只保留
主环境成功轨迹。Profile-v2 将 single、batched executor 与 collector 统一为先接收已到达
结果、再冻结 prefix，并保留所有含有效 teacher suffix target 的成功和失败尝试；prefix 使用
RTX 3090 profile revision `66414463f2a609316da63935941a79abf0a73ce9` 的 p95，三模型
K 分别为 3、4、3。

修复后的三模型 smoke 均完成双卡 2 optimizer steps 与 2-episode `profile→profile`，执行
审计通过。正式训练保持 ZeRO-2、每卡 micro batch 64、gradient accumulation 1、global
batch 128、gradient checkpointing 开启，并禁用 OOM 自动降 batch。QwenPI 首次 smoke
卡在一次分布式数据准备 barrier，单独重启同配置后通过；未改变数据、batch 或训练语义。

用户随后要求 Fixed-2 也按修复后的 v2 数据合同完整重跑。已建立单一后台队列：Profile-v2
采集完成后依次执行三路 Profile 训练、matched `profile→profile` 评测和覆盖发布；随后重新
采集共享 Fixed-2 v2 数据，并在 GPU pairs 0–1、2–3、4–5 上并行重训三模型，最后仅运行
matched `fixed-2→fixed-2` 评测。两类训练均为 5,000 optimizer steps、每卡 micro batch
64、global batch 128、gradient accumulation 1、ZeRO-2、gradient checkpointing 开启，
且不允许 OOM 自动降 batch；不再运行 `profile→zero` 或新一轮 `fixed-2→zero`。

## 首 suffix 与 student 状态诊断（2026-09-23）

六路 seed-reset 数据和 step-5,000 bundle 的 train/validation/held-out endpoint 对照已完成。
每路 32 条原始样本与 LeRobot loader 的 state、target 和两个 mask 一致；训练内首次请求
suffix MAE 分别为 Profile/Fixed OFT `.0083/.0179`、GR00T `.0198/.0131`、PI
`.0681/.0267`，validation 分别升至 `.0505/.0470`、`.0512/.0446`、
`.0837/.0492`。Profile-PI 扩展至全部 1682 条训练首请求后 MAE `.0755`、全部
178 条 validation 为 `.0837`，区间重叠，不能断言这一路的额外泛化损失。
逐帧重放正式 rollout 前十局的执行动作和 reward 与原日志完全一致；
在相同 student 状态上重新查询 PPO 后，OFT/GR00T/PI 的第十次请求 MAE 分别达到
`.312/.189/.416`，首请求只有 `.044/.029/.028`。这确立了训练外首请求误差及后续
student 状态分布偏移，DAgger 是后续对照实验而非当前已验证的性能修复。

两台 H200 使用相同基础 commit，但第二台的 evaluator 仍把 seed 列表传给单槽 reset；
第一台已用标量 seed。第二台 Profile-GR00T、Profile-PI、Fixed-2-PI 的旧
`5/200、5/200、2/200` 虽有唯一 seed metadata 和正常非零动作，物理状态与记录 seed
不对应，不能作为共同 seed 正式结果。独立标量 reset 诊断重评三项均完成 200 episodes
和 H8 execution audit，成功数为 `11/200、6/200、5/200`，原 JSON 与 checkpoint
未改。第一台最新正式文件中的 Profile-OFT、Fixed-OFT、Fixed-GR00T 分别为
`15/200、12/200、13/200`；PROGRESS 与 RESULTS 的旧数值已按核验状态修订。
[诊断报告与原始指标](../../outputs/tasks/mikasa_intercept_grab_fast/h8_endpoint_diagnosis_20260923/REPORT.md)。
