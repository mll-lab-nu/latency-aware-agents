# H1 原生 profile-RL 交接

## 核查更新（2026-09-14）

当前四任务、四模型的进度和数值统一见 [PROGRESS](PROGRESS.md)，文档分类见
[索引](README.md)。下方 2026-09-07 的预算、初始化方式和运行消息保留为当时记录；
不代表现在仍有任务运行。英文 [Balance 交接](archive/LATENCY_RL_HANDOFF.md)
已归档，它与本文件的四任务范围不同。

| 已确认的 Balance 实验 | 评测口径 | 结果 | 后续状态 |
|---|---|---|---|
| fixed1 execution-boundary F2 | 6M；环境帧 | 平均存活 525.44 帧 | fixed1 已恢复；不能外推到 fixed2/profile |
| H8 temporal profile 修复 | 5,000,064 raw steps；25 env seeds × 4 latency seeds | 存活 72.88→172.14；return 59.45→150.37；≥100 帧 3%→93% | 历史修复对照已完成 |
| H8 第一版消融 | 4 training seeds；20 个 10M run；每模型 100 场景 | full 存活 304.73±232.32；删除 ABCHJ 为 417.24±88.05 | 只保留 D/E 的生产组合尚未直接验证；后续单独消融 D/E |

fixed1 诊断全文见 [历史诊断](archive/LATENCY_DEBUG_PLAN.md)，H8 修复见
[CHUNK](CHUNK.md)，最新消融和待验证范围见 [CHUNK_RL_ABLATION](CHUNK_RL_ABLATION.md)。
旧 [LATENCY_PLAN](archive/LATENCY_PLAN.md) 的实验 3–5 未执行且本轮产物已删除，
保持历史未执行状态；不从旧计划推断当前待运行任务。

MIKASA 已发布四类 RTX 3090 profile 对应的小模型。WanOFT 当前发布的是
10 Hz clock10、success-at-end 93.75% 的版本；此前 20 Hz 的 47.66% 记录保留为
另一条实验。QwenOFT latency-aware VLA 的本地 100-episode temporal 复核为
success 57%、return 19.4161±15.0048，baseline 同口径为 0%、0.4274±0.1224；
具体 checkpoint、配置与版本见 PROGRESS。这些新证据补充下方旧 smoke 和暂停记录。

## 2026-09-07 H1 计划与运行记录

本节替代下方旧 H40／全链路计划。四环境均只预测当前 action，用已有无延迟
teacher 初始化并重新建立优化器和训练进度；本轮止于 small RL 训练和评估。
NVIDIA GR00T 独立运行时后来已移除；该路线只保留为历史记录，当前 VLA 统一使用 StarVLA。

| 环境 | 当前动作 | 控制／观测 Hz | 正式预算 |
|---|---|---|---|
| MIKASA | 7D | 20／20 | PPO，150M steps |
| Parkour | 34D latent+yaw | 50／10 | PPO，15,000 iterations |
| HAIC | 256D latent | 20／20 | PPO，500M frames |
| Balance | 43D request | 50／50 | FastTD3，25M steps |

传输使用 `[B,D]` 单动作，训练和评估统一设置
`scheduler.hold_policy: hold`：持续保持已到达的上一动作，直到下一动作
到达；首个动作前使用原 noop，reset 清空传输状态。保留原 decoder 和动作单位。

HAIC 的真实控制步长改为 `0.05 s`，repeat=1，超时设为400步以保留20秒上限，
物理子步长不变。参考动作和离散接触标签同步重采样到20 Hz；静态物体几何不变。
原 teacher 和新策略均在20 Hz下评估。

显式选择各环境 profile，使用 temporal、latency seed=0 和单 worker capacity。
延迟毫秒值不缩放：100 ms在20 Hz为2个控制步，在50 Hz为5步。不得让旧profile
生成器将HAIC频率覆盖回50 Hz。Temporal驻留次数沿用现有按采样次数计数的语义。

四环境原生训练／评估入口见 [SCALE_MODELS](SCALE_MODELS.md) 及各任务 runbook；
英文 Balance-only 交接已归档。以下配置名为当时记录。MIKASA使用
`configs/training/latency/mikasa_h1.yaml`，Balance使用`balance_h1.yaml`；
两者均有`*_h1_eval.yaml`和`*_h1_eval_zero.yaml`，评估原teacher时显式替换checkpoint
并使用独立输出目录。先完成短训练、保存、重载及zero/profile评估，再执行正式预算。
MIKASA评估50个episode，其余各100；记录return、成功率／进度、episode length、
admission/drop、吞吐及teacher/profile/FPS/checkpoint身份。Smoke通过不等于学会任务。

Parkour使用`parkour.yaml`／`parkour_zero.yaml`，HAIC使用`haic.yaml`／
`haic_zero.yaml`。原生评估入口分别为各任务目录下的`eval_small_policy_with_latency.sh`。
Parkour原teacher评估加`--teacher-baseline`，HAIC在checkpoint参数之前加
`--teacher`；新策略评估必须显式传入本次训练保存的checkpoint。

任务入口沿用现有`small_policy`命名：无延迟训练为`train_small_policy.sh`，
带延迟训练为`train_small_policy_with_latency.sh`，对应原生评估为
`eval_small_policy_with_latency.sh`。MIKASA通过训练入口的`--evaluate`评估；
Balance继续使用现有可配置训练入口和`latency_bench.run`，不增加重复脚本。

### H1 验证与运行状态（2026-09-07）

结果目录为`outputs/latency_rl/h1_validation`。共享CPU选择通过147项，之后
重叠的transport／Sample-Factory／evaluation选择通过106项；Flappy CUDA通过
8项、CULE Demon Attack通过2项。复用现有Flappy源码和AutoROM资源，未安装依赖。

最终替换直接让`hold`保持最近到达的scalar动作，直到新动作到达或reset；共享模块为
`training.common.action_latency`。替换后的CPU选择通过332项（1项未选），HAIC通过
23项、Parkour通过3项、pipeline／evaluation通过16项，日志为
`hold_replacement_{cpu_final,haic,parkour2,pipeline2}.log`。重新运行MIKASA 50-episode
profile评估得到return 3.269509、success-once 4%、admission/drop 1409/1591，结果在
`hold_replacement_mikasa_profile`。现有online进程未重启，仍使用已加载的代码和保存的
配置，动作无限保持的行为一致。

hold重新编译后，Flappy raw/fused hold/noop定向验证通过6项，Demon CUDA core通过
13项，CULE ALE／observation对齐通过8项。

后续`latest_ready`修复使NumPy、Flappy raw/fused CUDA和CULE与scalar coordinator
一致：新结果执行后，同时丢弃所有更旧的pending请求，包括尚未到达的旧请求；不新增
状态或配置。原sampled-normal对齐失败已在两种hold策略下通过，测试使用不同的新旧
动作值。共享CPU选择通过332项，Flappy CPU子集通过6项（跳过11项GPU测试），日志为
`latest_ready_{cpu,flappy_cpu}.log`。重新编译后，Flappy GPU整文件17项全部通过，日志为
`latest_ready_flappy_gpu.log`。
重新编译的CULE后端通过13项Demon CUDA测试和8项ALE／observation对齐测试；现有
projection用例同时断言：经过旧请求到达帧后，仍执行较新的、取值不同的动作。

MIKASA四组各50-episode评估已生成`mikasa_eval_*/evaluation.json`。原teacher的
zero／profile return为36.165／18.099，success-once为100%／46%；两次更新的
smoke模型为8.181／3.270、12%／4%。每组60步，zero admission/drop为3000/0，
profile为1409/1591。这证明流程跑通，不证明训练有效。

所有正式训练必须使用W&B online。已按用户要求停止刚才的offline训练，日志和
checkpoint保留在各任务的`h1_offline_aborted_20260907_1655`目录；新运行从原teacher
重新初始化优化器和进度，不续训中止的checkpoint。

MIKASA、Balance、Parkour、HAIC已按online模式在tmux的`latency_h1_mikasa`、
`latency_h1_balance`、`latency_h1_parkour`、`latency_h1_haic`启动。原offline运行已检查的
MIKASA第31次更新／1,904,640步及Balance的1,365,632帧checkpoint均为有限值。
正式预算及最终同种子评估尚未完成，不得用smoke结果替代正式训练结论。

## 历史 H40／全链路记录

下方为旧实验事实及当时的执行约定，不再作为当前H1实验的启动要求。

# Latency command-RL——启动与 smoke 交接（中文译本）

原文：[program/archive/LATENCY_RL_HANDOFF.md](archive/LATENCY_RL_HANDOFF.md)。本文件只翻译说明文字；命令、路径、配置键和指标名称保持原样，便于复制执行。原文中的远程登录地址已用占位符替代。

四个任务入口都已实现。实际的分阶段 smoke 已验证：原生训练/重新加载、issued-label 数据转换、两次更新的 VLA 保存/重新加载，以及 baseline/new 的 zero/profile 闭环评估。这些运行复用了已有的显式 profile；“全新完整 profile-to-training”启动和正式长时间训练尚未执行。尚未建立任何 robustness（鲁棒性）提升结论。

## Profile 污染提示

Balance 的 EGL 数据集导出在 2026-09-05 UTC 约 19:15:26–19:18:01 本地运行，可能影响了 Parkour 的第一场 session。Balance 的 VLA smoke 也与同日约 19:23:07.676–19:25:00 及 19:30:31.686–19:31:00.743 仍在进行的本地 Parkour profile 重叠。第一次训练了两步但在评估时失败；第二次被停止且没有生成 checkpoint。Profile 进程本身未被触碰。这些时间段的样本不是干净的 latency 测量；在重新测量前不要接受受影响的 profile。profile 运行期间，后续 GPU 验证仅限远程服务器。

## 固定契约

MIKASA/Parkour/HAIC 保持原生 PPO，Balance 使用 FastTD3。训练 command policy，而不是下游 decoder。Parkour 使用 H40×34 latent+yaw；HAIC 使用 H40×256 latent；MIKASA 使用 H1×7；Balance 使用重复 H40 的 43D absolute request。Parkour/HAIC 的 decision/control rate 为 10/50 Hz，MIKASA 为 20/20 Hz，Balance 为 50/50 Hz。保留 privileged teacher 输入；VLA 输入仍为 RGB/proprio。

每个 admitted observation 都拥有自己的 issued label。不要从后续 policy decision 拼出 H40 label，也不要用实际已经施加的 motor action 替换它。Actor auxiliary target 必须用相同的、对齐后的 actor 输入和冻结 decoder 解码同一个 issued chunk。Balance 的 conditioning 保持在 application time。

## 全链路启动与恢复

从仓库根目录运行，并确保任务现有的 teacher、baseline VLA、预训练 base、模拟器资源和 Hugging Face 访问均已准备好。下面的每条命令都会在训练前测量 profile；不会静默复用其他 model 或 profile。发布仍是独立操作。

```bash
.venv/bin/python -m scripts.tasks.mikasa_intercept_grab_fast.run_latency_pipeline \
  --config configs/training/mikasa_latency_pipeline.yaml
.venv/bin/python -m scripts.tasks.extreme_parkour_go2.run_latency_pipeline \
  --config configs/training/parkour_latency_pipeline.yaml
.venv/bin/python -m scripts.tasks.haic_pull_cart.run_latency_pipeline \
  --config configs/training/haic_latency_pipeline.yaml
.venv/bin/python -m scripts.tasks.humanoidbench_balance_simple.run_latency_pipeline \
  --config configs/training/humanoidbench_balance_latency_pipeline.yaml
```

短训练使用对应的 `*_pipeline_smoke.yaml`。Profile 仍遵循完整协议；若要显式复用已有测量，追加 `--profile-path /absolute/path/profile.json`。相互独立的实验要使用不同的输出目录。阶段日志以及解析后的 latency/evaluation YAML 会写入各配置的 `output_dir`；遇到第一个失败阶段就停止。

在另一台机器上运行时，复制任务 YAML，并显式替换 interpreter、teacher、base-model、baseline-model 和 output 路径。保持 decoder identity、observation/action 语义和 latency 配置不变。本地 interpreter 见文档后文；远程 smoke 主机地址已从本译本省略。两张本地 GPU 已为正在进行的正式 profile 保留，不要与训练、导出渲染或评估重叠。

恢复使用原生 checkpoint，不引入新的 workflow engine：

- MIKASA：对下面列出的原生训练入口传入 `--resume <output>/teacher/training.pt`；单独的 `teacher.pt` 只是初始化，不是 optimizer resume。
- Parkour：使用相同 teacher/config 重新运行 `train_small_policy_with_latency.sh`，并加上原生 `--resume --load_run <run-name> --checkpoint <model_N.pt>`；`--max_iterations` 表示额外的原生训练预算。原生训练后继续 pipeline 时，显式设置 `native_checkpoint` 并传入 `--skip-native-train`。
- HAIC：将新的原生 `checkpoint_final.pt` 作为训练 checkpoint；其中 `last_phase=latency_command` 会选择 optimizer/progress/RNG 恢复。Pipeline 接受 `--skip-native-train --native-checkpoint <path>`。
- Balance：保持相同的 native run name，并将 `native_restart_behavior: resume`；选定的 FastTD3 checkpoint 会恢复 optimizer/progress 状态。VLA 周期恢复时，`--start-step` 应等于下一个保存边界，同时保留原 maximum step 和 interval。

恢复 VLA 训练时，保留完整的 optimizer/scheduler 文件及关联数据集。只有 model 的 checkpoint 支持 inference，不支持精确的训练恢复。失败的 smoke、`--keep-failed` 数据和两步 VLA 都不能作为 accepted demonstration 或 robustness 改善证据。

## 自动化检查（2026-09-06）

最终的 CPU 隔离 command/checkpoint、四 pipeline、Balance validation、executor、evaluation、latency、artifact 和 profile-protocol 选择共通过 **354 个测试**，跳过 1 个。另有两项被主动 deselect：Flappy executor 用例需要缺失的可选包 `flappy_bird_gymnasium`；hardware projection 用例在没有 `CUDA_VISIBLE_DEVICES=''` 的情况下单独通过。Flappy CPU/GPU renderer 模块也因该依赖缺失而排除。没有安装新依赖，也没有修改无关测试。

HAIC bootstrap/pipeline 选择最终通过 **14 个测试**。原生 HAIC admission/finite-gradient 测试在远程通过 **2 个**；TorchRL 警告其未使用的 C++ replay-buffer 扩展不可用。GR00T 的 sharded dataset 测试此前通过 **23 个**。这些选择相互重叠，不能把计数相加当作不重复的测试总数。

Command transport 对 scalar 和 batched evaluator 的两个 replica、两个 episode 做逐帧检查，覆盖 admission、capacity drop、stale H40 prefix 和独立 reset。IID stream 使用相同的全局 episode-ID 派生规则；temporal stream 保留每个 replica 的状态。

## 功能性 smoke profile

以下三个短 profile 仅可用于 pipeline smoke：

- `outputs/latency_rl/parkour/smoke_profile_session0/profile.json`：当前 Parkour 运行的第 0 个 100-episode session，共 18,898 个 admitted sample。它在 VLA 重叠之前结束，但 EGL export 仍可能影响它。仅保留作功能 smoke，不是 accepted clean measurement；正式 run 不受其影响，且该 partial profile 未发布。
- `outputs/latency_rl/balance/smoke_profile_session0/profile.json`：2026-08-31 使用显式 10k Balance model 的 100-episode session，共 4,973 个 admitted sample、2,665 个 capacity drop。这不是新的五-session 测量。
- `outputs/latency_rl/haic/co_located_measure_root/profile/profile.json`：一个完整 58-control-frame episode 中的 12 个 admitted measurement；在 public evaluator 的 100-sample warmup 后，均值为 301.340 ms。模拟器与已有 HAIC 5k GR00T checkpoint 共用远程 GPU0。这个刻意缩小且 co-located 的样本只能作为功能 smoke 证据，不是正式 latency 估计或 topology 复现。原始记录、解析后的 evaluation config 和日志均在同一父目录；命令在缓慢的 native SDK shutdown 后以 0 退出，未添加 shutdown workaround。远程 profile 路径在原文中给出。

### MIKASA accepted smoke

在 admission、checkpoint 和 seed 变更后，MIKASA 重新验证完成：两次原生更新、恢复后的第三次更新、两个完整 60-frame episode，以及 RLDS/LeRobot 转换。artifact 和日志位于 `outputs/latency_rl/mikasa/revalidation_remote`。全新的两步 VLA run 在远程 GPU 完成优化和 held-out evaluation，并保存 model 及完整 DeepSpeed optimizer state。重新加载 checkpoint 后，zero/IID 闭环评估都完成 60 帧（各自 success 为 0/1；return 分别为 0.13381242 和 0.27674512）。本地摘要位于相对 `outputs/latency_rl/mikasa` 的 `revalidation_remote/eval_zero/queue_eval_latency_0.json` 和 `revalidation_remote/eval_iid/queue_eval_latency_profile_sample.json`。全新 model 位于远程主机的原文路径。

现有 canonical baseline VLA 也使用相同 profile 重新加载并评估：zero/IID return 为 4.66181925/0.29341384，均完成 60 帧，success 各为 0/1。摘要位于 `revalidation_remote/eval_baseline_zero` 和 `revalidation_remote/eval_baseline_iid`。四个单 episode smoke 不能建立 robustness 或有统计意义的比较。

最终的 control-trace collector 也在远程重新运行：两个 60-frame episode，admitted command 分别为 28/29。每个非 noop 的 applied command 都精确匹配 `applied_source_raw_frame` 上的 issued action，并且 ready frame 不晚于 consumption；H1 chunk index 为 0，真实 noop 使用 source/index -1。RGB、proprio、issued action、admission、ready frame、done flag 和 applied command 与已经用于 VLA 训练的 revalidation dataset 按位一致。只有 execution-source logging 发生变化，VLA training 未重复执行。文件位于 `revalidation_remote/issued_data_control_trace`，日志为 `revalidation_remote/collect_control_trace.log`。

### Parkour 原生训练与 pipeline 状态（2026-09-06）

已接受的远程原生 latency-command smoke 使用 **1,024 个环境**（不是 1 个）：在标准五个 PPO epoch、四个 minibatch 下完成两次有限值更新；在指数运算前对累加的 action log-ratio 做了 clamp。日志位于远程 smoke 工作区的 `native_latency_fixed_iid_5x4.log`；update 0 的 value loss 为 `0.0130`、surrogate loss 为 `-0.0595`、entropy 为 `1931.1259`，update 1 分别为 `0.0083`、`-0.0915`、`1934.8921`。该运行在 `2026-09-05_20-35-07_latency_command_fixed_iid_5x4` 下写出了 `model_0.pt` 和 `model_1.pt`。它使用 IID latency method 和 smoke profile。仓库中的正式 launcher 与 smoke config 都使用 1,024 个环境；smoke 将原生训练限制为两次 iteration。

此前的 checkpoint-contract smoke 是另一条独立运行：初始运行没有 native resume 输入，写出 `model_0.pt`；其成功的 resume 使用固定 zero latency config，加载该精确 checkpoint 并写出新的 `model_0.pt`。之后的 `resume_next2` 同样使用固定 zero config，加载 `model_1.pt` 并写出 `model_2.pt`，仅用于诊断。这些固定 zero resume 不能与 collector 使用的 IID `model_1.pt` 混用。没有执行 temporal native resume。Latency bundle 包含 scheduler 和 RNG state，decoder freezing 已由 runtime module contract 确立。

稳定后的 `model_1.pt` 被远程 IID collector 使用，原始 teacher 作为 decoder。它提交了两个完整的 `--keep-failed` smoke episode（一个 train、一个 validation），共 1,184 个 raw frame，以及 979/205 个 control-trace frame。输出位于远程 smoke 工作区的 `collection_iid_2ep`；metadata 记录了 IID profile、H40×34 issued chunk，以及分离的 command/decoder checkpoint。shard 包含 NPZ 与 MP4 文件。

转换在本地完成，输出为 `outputs/latency_rl/parkour/collector_remote_iid_2ep`，生成 train/val LeRobot dataset，分别有 98/21 个 admitted row、`issued_command_chunk` modality metadata 和 MP4 文件。GR00T sharded loader 能提取 `(40, 34)` action，以及对齐的 `actor_observation`/`termination` objective。

干净的远程 actor-objective bootstrap 使用 official GR00T launcher 完成两次 optimizer update。它使用一张 GPU、global batch size 1、`max_steps=2`、官方 base model、转换后的 IID dataset，以及作为 fixed action actor 的 teacher。训练日志位于远程 smoke 工作区的 `actor_objective_vla_2updates_complete/train.log`；launcher 以 0 退出，并报告 `train_runtime=39.7032`、`train_loss=3.8606886863708496`。

接受的 checkpoint 是该工作区中的 `training/parkour_actor_objective_2updates_complete/checkpoint-2`，包含两个 model shard、`optimizer.pt`（14,592,708,947 bytes）、`scheduler.pt`、`rng_state.pth`、`trainer_state.json` 和 processor metadata。此前 optimizer 保存失败的输出必须视为 invalid，不得当作 accepted artifact；此前的 local-backbone wiring smoke 也不是 actor-objective 结果。

新 checkpoint 重新加载后，使用原始 teacher 作为 runtime actor 做了一个 episode 的闭环评估。两个命令都以 0 退出，配置、日志和 JSON 输出均持久化在远程 smoke 工作区：

| model | latency | steps | mean return | edge violation |
|---|---|---:|---:|---:|
| baseline | zero | 936 | 4.354336371296085 | 0.0128205 |
| baseline | IID profile sample | 1,500 | 1.2002695962437429 | 0.0026667 |
| new checkpoint-2 | zero | 1,500 | 0.7200697755761212 | 0.0 |
| new checkpoint-2 | IID profile sample | 1,500 | 0.535144853303791 | 0.0 |

较早的 baseline run 使用相同 seed 2 和 profile，摘要保存在 `outputs/latency_rl/parkour/real_chain_remote/evals/` 的 `original_zero` 与 `original_iid`。新结果为 `eval_zero/queue_eval_latency_0.json` 和 `eval_iid/queue_eval_latency_profile_sample.json`，并有本地 metadata 副本 `outputs/latency_rl/parkour/actor_objective_complete_metadata/`。新 checkpoint 依赖评估主机上的外部 `/tmp/Qwen/Qwen3-VL-cosmos` model/tokenizer metadata，因此不是可移植的发布 artifact。

上述评估使用已有 smoke profile 的 IID sample，只能证明 wiring，不是质量或 robustness 结论。

仓库中已在每个边界写明 pipeline handoff：`NATIVE_CHECKPOINT=` 标识原生 run 的最终 checkpoint；collection 传入该 command checkpoint，同时保留原始 teacher decoder；带 video 的 NPZ shard 转换为独立的 train/validation LeRobot dataset；bootstrap VLA training 消费该 dataset，并保留原始 teacher 作为固定 action decoder；生成的 zero 和 temporal evaluation config 指向新训练的 VLA model。Parkour action modality 使用 H40×34 carrier 的 `issued_command_chunk`。正式 config 仍为 1,024 个 native environment、15,000 次 iteration、250/25 collection episode、5,000 次 VLA update 和 100-episode evaluation；smoke config 将其分别限制为 native 两次 iteration、1/1 collection episode、两次 VLA update 和一个 1,500-step episode。

远程 acceptance run 刻意使用 smoke 规模和直接的单 GPU GR00T 调用；这是功能性两步 actor-objective smoke，不是正式质量运行。它使用一张 GPU、global batch 1、`--action-objective parkour` 以及显式的 base/dataset/teacher 路径。仓库中的完整 pipeline 仍可用于未来的干净 measured profile。

下面的参数差异是刻意保留的 handoff 事实，不同运行名不能互换：

| 阶段 | 正式 launcher | 签入的 smoke override | 实际运行的远程链 |
|---|---|---|---|
| native PPO | `--seed 1 --num_envs 1024 --max_iterations 15000` | `--num_envs 1024 --max_iterations 2` | `--seed 1 --num_envs 1024 --max_iterations 2`，IID debug config |
| collection | 250/25 episode，50 env | 1/1 episode，1 env | 1/1 episode，1 env；IID `model_1.pt` command checkpoint |
| VLA | 5,000 update，global batch 128，2 GPU，Parkour actor objective | 2 update，global batch 1，1 GPU，Parkour actor objective | 同左 |
| evaluation | 100 episode，1 backend env，configured worker device 上 inference batch 8 | 1 episode，1 env，inference batch 1 | 1 episode，1 env，simulator 与 worker 均使用 `cuda:0` |

可用的本地 interpreter 为 Parkour 3.11、HAIC 3.11 和 GR00T 3.12；本地 Parkour teacher、GR00T base model 和 VLA source artifact 已存在共享目录。profile 正式运行时，本地 GPU 被保留；远程 smoke 工作区包含 native 与 collector artifact。VLA 阶段仍需要可用 GPU，并需在选定主机显式准备 base/model 输入。

当这些 artifact 和干净 measured profile 准备好后，可使用以下有界 pipeline 命令：

```bash
CUDA_VISIBLE_DEVICES=0,1 .venv/bin/python \
  scripts/tasks/extreme_parkour_go2/run_latency_pipeline.py \
  --config configs/training/parkour_latency_pipeline_smoke.yaml \
  --profile-path outputs/latency_rl/parkour/smoke_profile_session0/profile.json
```

远程 native 与 collection 阶段是分别运行的，accepted VLA/evaluation 阶段则直接调用。干净的正式 profile 和 full-scale pipeline 仍是未来工作。

### HAIC 原生训练与 pipeline 状态（2026-09-06）

已接受的 HAIC 证据来自远程执行主机。复现时需设置原文中给出的远程工作区变量；本译本省略登录地址和用户名：

```bash
B=/home/<远程用户>/haic_runs/co_located_measure_root
P=/home/<远程用户>/latency-sensitive-bench-mikasa-h1
```

准备好的完整 pipeline config 为 `outputs/latency_rl/haic/haic_full_pipeline_remote.yaml`（同时同步到 `$P/outputs/latency_rl/haic/haic_full_pipeline_remote.yaml`）。它使用原 phase-1 teacher 做 native training、canonical remote HAIC latency config、现有 checkpoint-5000 baseline 做 profiling、全新的 base-model VLA run，以及 100-episode zero/profile evaluation。本次 acceptance 没有启动完整 run：

```bash
cd "$P"
.venv/bin/python scripts/tasks/haic_pull_cart/run_latency_pipeline.py \
  --config "$P/outputs/latency_rl/haic/haic_full_pipeline_remote.yaml"
```

签入的 HAIC push code 在 `push`、`push_start` 和 `push_per_motion` 中使用 IsaacLab 的 `Articulation.set_external_force_and_torque` API；没有使用删除 push 的 monkeypatch。下面的 smoke 证据是功能集成证据（小规模 native/VLA 运行及单 episode 评估），不是质量、robustness 或正式规模结果。

#### 原生 command 与 resume

原生 acceptance run 从 phase-1 teacher 开始，使用真实 latency command config、horizon 40、control repeat 5、zero action delay 和 alpha 1.0。精确命令与续跑如下：

```bash
cd "$P"
export PYTHONPATH="$P:$P/third_party/HAIC:/home/<远程用户>/IsaacLab/source/isaaclab"
export CUDA_VISIBLE_DEVICES=0 OMNI_KIT_ACCEPT_EULA=YES WANDB_MODE=offline
export HAIC_PYTHON=/home/<远程用户>/miniconda3/envs/haic/bin/python
bash scripts/tasks/haic_pull_cart/train_small_policy_with_latency.sh \
  "$P/outputs/haic/phase1_teacher/checkpoint_final.pt" \
  "$B/latency_real.yaml" \
  hydra.run.dir="$B/native_acceptance3" \
  task.num_envs=1 task.max_episode_length=100 \
  algo.train_every=8 algo.num_minibatches=1 total_frames=256 \
  save_interval=8 wandb.mode=offline

bash scripts/tasks/haic_pull_cart/train_small_policy_with_latency.sh \
  "$B/native_acceptance3/wandb/latest-run/files/checkpoint_final.pt" \
  "$B/latency_real.yaml" \
  hydra.run.dir="$B/native_acceptance3_resume" \
  task.num_envs=1 task.max_episode_length=100 \
  algo.train_every=8 algo.num_minibatches=1 total_frames=128 \
  save_interval=8 wandb.mode=offline
```

第一次运行完成 32/32 次 iteration，在 `$B/native_acceptance3/wandb/latest-run/files/` 下于 8、16、24 和 `final` 写出 checkpoint；续跑完成 16/16 次 iteration，并在 `$B/native_acceptance3_resume/wandb/latest-run/files/checkpoint_final.pt` 写出最终 checkpoint。检查到的 actor metadata 在两次运行中都是 finite：第一次结束于 `last_iter=32`、`num_updates=34`，续跑结束于 `last_iter=47`、`num_updates=50`，额外完成 14 次 finite actor update。原生 W&B 日志包含 31 条 `actor/update_applied`，记录的 actor metric 和 tensor 均为 finite。两个外层命令都自然以 return code 0 结束。

#### 完整 episode collection 与转换

使用恢复后的 native checkpoint，accepted train/validation collection 命令为：

```bash
cd "$P"
export PYTHONPATH="$P:$P/third_party/HAIC:/home/<远程用户>/IsaacLab/source/isaaclab"
export CUDA_VISIBLE_DEVICES=0 OMNI_KIT_ACCEPT_EULA=YES WANDB_MODE=offline
export HAIC_PYTHON=/home/<远程用户>/miniconda3/envs/haic/bin/python

"$HAIC_PYTHON" third_party/HAIC/scripts/haic_vla.py bootstrap-collect --headless \
  --device cuda:0 \
  --teacher-checkpoint "$B/native_acceptance3_resume/wandb/latest-run/files/checkpoint_final.pt" \
  --output-dir "$B/raw_acceptance3_train" --inference-device cuda:0 \
  --inference-batch-size 1 --num-envs 1 --seed 0 --dagger-round 0 \
  --vla-cadence 5 --split train --episode-budget 2 \
  --latency-config "$B/latency_real.yaml" --keep-failed

"$HAIC_PYTHON" third_party/HAIC/scripts/haic_vla.py bootstrap-collect --headless \
  --device cuda:0 \
  --teacher-checkpoint "$B/native_acceptance3_resume/wandb/latest-run/files/checkpoint_final.pt" \
  --output-dir "$B/raw_acceptance3_val" --inference-device cuda:0 \
  --inference-batch-size 1 --num-envs 1 --seed 1 --dagger-round 0 \
  --vla-cadence 5 --split val --episode-budget 1 \
  --latency-config "$B/latency_real.yaml" --keep-failed
```

生成的日志为 `$B/raw_acceptance3_train.log` 和 `$B/raw_acceptance3_val.log`。Train 接受 2 个 episode、217 个 raw step；validation 接受 1 个 episode、178 个 raw step。两份 split 的 shard 字段均为 `issued_command [40, 256]`、`actor_input [40, 605]`、`teacher_action [40, 23]` 和 `termination [40]`。合并后的 raw root 是 `$B/haic_vla_acceptance3/raw`；转换命令为：

```bash
cd "$P"
"$P/third_party/Isaac-GR00T/.venv/bin/python" \
  scripts/official_gr00t/prepare_haic_dataset.py \
  "$B/haic_vla_acceptance3/raw" \
  "$B/haic_vla_acceptance3/bootstrap/lerobot" \
  --force --round-index 0 \
  --modality-config latency_bench/policy/official_gr00t_haic_issued_modality.py
```

准备好的 train dataset 有 2 个 episode、45 个 frame；同级的 `$B/haic_vla_acceptance3/bootstrap/lerobot__val` 有 1 个 episode、36 个 frame。Actor adapter 为 `$B/haic_vla_acceptance3/bootstrap/actor_adapt.pt`。

#### Fresh-base VLA smoke

Fresh official GR00T base 为 `$P/outputs/official_gr00t/models/GR00T-N1.7-3B`。两步 smoke 使用准备好的 dataset，并同时把恢复后的 native checkpoint 作为 teacher 和 frozen runtime checkpoint：

```bash
cd "$P"
export HAIC_PYTHON=/home/<远程用户>/miniconda3/envs/haic/bin/python
export CUDA_VISIBLE_DEVICES=0 OMNI_KIT_ACCEPT_EULA=YES
export GROOT_HF_LOCAL_FIRST=1 GROOT_PATCH_MISTRAL=1
export LD_LIBRARY_PATH=/home/<远程用户>/.cache/latency-sensitive-bench/ffmpeg7/lib
bash scripts/tasks/haic_pull_cart/train_official_gr00t.sh \
  "$B/native_acceptance3_resume/wandb/latest-run/files/checkpoint_final.pt" \
  --base-model "$P/outputs/official_gr00t/models/GR00T-N1.7-3B" \
  --output-root "$B/haic_vla_acceptance3/vla_smoke2" \
  --bootstrap-dataset "$B/haic_vla_acceptance3/bootstrap/lerobot" \
  --actor-checkpoint "$B/haic_vla_acceptance3/bootstrap/actor_adapt.pt" \
  --policy-config "$P/configs/examples/haic/official_gr00t_eval.yaml" \
  --latency-profile-path "$B/profile/profile.json" \
  --latency-config "$B/latency_real.yaml" \
  --runtime-checkpoint "$B/native_acceptance3_resume/wandb/latest-run/files/checkpoint_final.pt" \
  --wandb-run-id haic_pull_cart_acceptance3_smoke2 \
  --start-round 0 --end-round 0 --inference-device cuda:0 \
  --inference-batch-size 1 --smoke
```

生成的 checkpoint 为 `$B/haic_vla_acceptance3/vla_smoke2/bootstrap/training/haic_pull_cart_acceptance3_smoke2_sft/checkpoint-2`。它包含两个 model shard、`optimizer.pt`（14,797,999,507 bytes）、scheduler/RNG/trainer state 以及 processor/config metadata；本地 mirror 只保留 metadata，不包含 weights 和 optimizer state。`vla_smoke2.log` 记录 2/2 update、`train_runtime=81.4045` 和 `train_loss=1.4915385246`。最初的 wrapper marker 因生成的 eval config 缺少 `executor.inference_batch_size` 而返回 1；修复配置后，直接 periodic eval 在 `$B/haic_vla_acceptance3/vla_smoke2/bootstrap/eval/` 自然以 return code 0 结束并生成预期摘要。

#### 最终 zero/profile 评估

四次最终评估使用恢复后的 native checkpoint、已有 measured profile `$B/profile/profile.json` 和一个 `eval_max_steps=1000` 的 episode。Baseline model 是原文记录的 checkpoint-5000 路径；new model 是上面的 fresh checkpoint-2。每条直接命令都使用签入的 HAIC 实现并以 code 0 自然结束。

```bash
cd "$P"
export HAIC_PYTHON=/home/<远程用户>/miniconda3/envs/haic/bin/python
export OMNI_KIT_ACCEPT_EULA=YES GROOT_HF_LOCAL_FIRST=1 GROOT_PATCH_MISTRAL=1
export PYTHONPATH="$P:$P/third_party/HAIC:$P/third_party/Isaac-GR00T"
export LD_LIBRARY_PATH=/home/<远程用户>/.cache/latency-sensitive-bench/ffmpeg7/lib
for config in eval_baseline_zero eval_baseline_profile eval_new_zero eval_new_profile; do
  "$HAIC_PYTHON" "$P/third_party/HAIC/scripts/haic_vla.py" latency-eval \
    --headless --eval-config "$B/${config}.yaml"
done
```

| model | latency | length | return | motion progress | cart progress | pullcart score | success |
|---|---|---:|---:|---:|---:|---:|---:|
| baseline checkpoint-5000 | zero | 55 | 3.066511783748865 | 0.4732142985 | 0.0897285491 | 28.1471423805 | 0 |
| baseline checkpoint-5000 | measured profile | 48 | 2.3801772370934486 | 0.4607142806 | 0.0238152742 | 24.2264777422 | 0 |
| fresh checkpoint-2 | zero | 45 | 2.468793924897909 | 0.4553571343 | 0.0725262091 | 26.3941671699 | 0 |
| fresh checkpoint-2 | measured profile | 37 | 2.047113075852394 | 0.4410714209 | 0.0119812191 | 22.6526319981 | 0 |

Profile 行复用 `$B/profile/profile.json`，不代表新的 latency 测量，也不代表质量或 robustness 结果；每个 `queue_eval_*.json` 都记录自己的 model checkpoint identity。四棵结果树已镜像到 `outputs/latency_rl/haic/acceptance3/evals/eval_{baseline,new}_{zero,profile}`；checkpoint-2 metadata（不含 model weights/optimizer state）在 `outputs/latency_rl/haic/vla_acceptance3_metadata`。

## MIKASA 单命令启动

```bash
.venv/bin/python -m scripts.tasks.mikasa_intercept_grab_fast.run_latency_pipeline \
  --config configs/training/mikasa_latency_pipeline_smoke.yaml
```

当前入口下载固定版本的已发布 QwenOFT RTX 3090 teacher 和 temporal profile，
然后执行采集、H1 数据转换、共享训练入口及 zero/temporal 对照评测。
正式配置为 `configs/training/mikasa_latency_pipeline.yaml`，开启 W&B online；
每卡 microbatch 从 64 开始，只有实际 CUDA OOM 才减半，保持 global batch 128。
具体设置见 [MIKASA 执行手册](MIKASA.md#9-latency-aware-teacher-蒸馏入口)。

## MIKASA 原生 PPO 与旧版 VLA 复现记录

以下命令保留历史运行证据；当前 H1 VLA 训练使用上面的单命令入口。

从仓库根目录运行。签入的 smoke config 显式引用已有 profile，不会重新测量：
`results/profiling/profiles/qwenoft/1x-rpbzzz6/mikasa_intercept_grab_fast/instance_e906b84a82f9e086/profile.json`。

```bash
export PYTHONPATH="$PWD:$PWD/third_party/MIKASA-Robo${PYTHONPATH:+:$PYTHONPATH}"
export VLA_ACCELERATE=/pfs/users/lxy/conda/envs/starvla_rl_games_openvla/bin/accelerate
export CUDA_VISIBLE_DEVICES=1
third_party/MIKASA-Robo/.venv/bin/python \
  third_party/MIKASA-Robo/mikasa_robo_suite/vla/dataset_collectors/get_dataset_collectors_ckpt.py \
  --env-id InterceptGrabFast-VLA-v0 \
  --latency-config configs/training/mikasa_latency_smoke.yaml \
  --checkpoint outputs/latency_rl/mikasa/source/teacher.pt \
  --latency-output-dir outputs/latency_rl/mikasa/smoke --no-track

third_party/MIKASA-Robo/.venv/bin/python \
  third_party/MIKASA-Robo/mikasa_robo_suite/vla/dataset_collectors/get_dataset_collectors_ckpt.py \
  --env-id InterceptGrabFast-VLA-v0 \
  --latency-config configs/training/mikasa_latency_resume_smoke.yaml \
  --resume outputs/latency_rl/mikasa/smoke/training.pt \
  --latency-output-dir outputs/latency_rl/mikasa/resume_smoke --no-track

bash scripts/tasks/mikasa_intercept_grab_fast/collect_latency.sh \
  --latency-config configs/training/mikasa_latency_smoke.yaml \
  --checkpoint outputs/latency_rl/mikasa/smoke/teacher.pt \
  --path-to-save-data outputs/latency_rl/mikasa/issued_data \
  --num-train-data 2 --num-envs 2 --keep-failed
```

输出为 `teacher.pt` 和 `training.pt`：初始化只加载 model weights；`--resume` 会恢复 optimizer/progress/random stream，并从新的 simulator episode 开始。Training config 会覆盖 native task preset，因此可以真正运行小规模实验，而不会静默启动 preset 的 1024 个环境。重复 collection/conversion 时使用新的 output directory。

转换使用 `scripts/tasks/mikasa_intercept_grab_fast/train_vla.sh` 中已有的命令，传入绝对的 NPZ/RLDS/LeRobot root，并使用 `train[:1]` 与 `train[1:2]` 两个 split。RLDS builder 根据 `command_admitted` 过滤，在原始 frame 上保留 RGB/state/action；原始 NPZ 保留完整 control trace。

在转换和 `scripts/mikasa/prepare_starvla_dataset.py` 之后，VLA smoke 命令为：

```bash
CUDA_VISIBLE_DEVICES=1 WANDB_MODE=disabled PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  bash scripts/tasks/mikasa_intercept_grab_fast/train_latency_vla.sh \
  outputs/latency_rl/mikasa/issued_data/data_lerobot \
  third_party/starVLA/playground/Pretrained_models/Qwen3VL-OFT-Bridge-RT-1/checkpoints/steps_5000_pytorch_model.pt \
  --run_id mikasa_latency_smoke_h1_ds --framework.qwenvl.attn_implementation sdpa \
  --datasets.vla_data.per_device_batch_size 1 --trainer.gradient_accumulation_steps 1 \
  --trainer.max_train_steps 2 --trainer.num_warmup_steps 0 \
  --trainer.eval_interval 2 --trainer.eval_num_batches 1 --trainer.save_interval 2
```

使用原生 Accelerate/DeepSpeed launcher。直接运行 `python train_starvla.py` 不会建立 distributed environment，并因寻找 MPI 而失败；没有添加 MPI 依赖。两步 loop（不含 model 初始化和 checkpoint 保存）约耗时 14 秒。保存期间观测到的 GPU 占用为 96,009 MiB，这只是单点观测，不是 peak measurement。

闭环 smoke 使用新 checkpoint，而不是原始 profile model：

```bash
CUDA_VISIBLE_DEVICES=0,1 third_party/MIKASA-Robo/.venv/bin/python \
  scripts/mikasa/evaluate.py latency-eval \
  --eval-config configs/training/mikasa_latency_vla_smoke_eval.yaml
CUDA_VISIBLE_DEVICES=0,1 third_party/MIKASA-Robo/.venv/bin/python \
  scripts/mikasa/evaluate.py latency-eval \
  --eval-config configs/training/mikasa_latency_vla_smoke_profile_eval.yaml
```

本地 interpreter：

- MIKASA：`third_party/MIKASA-Robo/.venv/bin/python`
- starVLA：`/pfs/users/lxy/conda/envs/starvla_rl_games_openvla/bin/python`
- Parkour：`/pfs/users/lxy/conda/envs/extreme-parkour/bin/python`
- HAIC：`/pfs/users/lxy/conda/envs/haic/bin/python`
- GR00T：`third_party/Isaac-GR00T/.venv/bin/python`

转换环境是 MIKASA-Robo 下现有的 `uv run --project` project；其声明依赖已在 smoke 期间安装。MIKASA native smoke（含初始化）记录约 20 aggregate steps/s；恢复后的单次 update 约 86 steps/s。这些 tiny-run 速度不能用于正式训练预算，且没有记录 peak GPU memory。

### Balance 修正后的 VLA acceptance（2026-09-06）

Balance 修正后的两步 GR00T wiring run 完成了底层的单 GPU launcher：`global_step=2`、`train_loss=1.5560086`、`train_runtime=39.008s`。本次远程运行使用的临时 wrapper 有意偏离 canonical launcher，强制 `--nproc_per_node=1`、global batch 1 和 zero dataloader worker；直接的 torchrun argv 记录在 `program/BALANCE.md`。生成的 checkpoint 包含 model shard 以及完整 optimizer/scheduler/RNG/trainer state：

```text
/home/<远程用户>/balance-latency-smoke/outputs/tasks/humanoidbench_balance_simple/real_profile_corrected/vla_final/g1_balance_corrected_vla_final_2updates/checkpoint-2
```

`optimizer.pt` 大小为 14,610,010,451 bytes。Wrapper 的可选 validation 阶段最初因为派生的 `lerobot__val` 缺少 `meta/stats.json` 而停止。Canonical runner 现在会在加载 validation data 前生成该 metadata；修复后的 validation rerun 完成，body MAE 为 `0.232301`、all-action MAE 为 `0.177278`、repeat standard deviation 为 `0.0`。这次使用全部 19 个 admitted row，没有减去 future-action horizon。下面四次显式闭环评估使用两个 episode、最大 1,000 step、一个 simulated worker，以及 slot 0 的 session-0 temporal profile：

| checkpoint | latency | mean return | std return | mean length |
|---|---|---:|---:|---:|
| original `checkpoint-10000` | zero | 24.349 | 0.009 | 33.5 |
| fresh `checkpoint-2` | zero | 7.517 | 0.014 | 20.5 |
| original `checkpoint-10000` | real profile | 34.426 | 2.309 | 48.0 |
| fresh `checkpoint-2` | real profile | 11.023 | 0.435 | 23.5 |

持久化输出位于远程 smoke 工作区的 `.../real_profile_corrected/evals/final_revalidated/`，本地副本为 `outputs/latency_rl/balance/final_revalidated/`。这些结果只接受为 Balance checkpoint/reload 和 evaluation wiring 证据，不是 robustness 或质量结论；该主机只有一张可见 GPU，不能复现 2x topology。

Native-to-VLA provenance 保存在远程 smoke 工作区。修正后的 native run 先在 4,096 个 environment frame 后保存 checkpoint，随后 resume 到 8,192-frame 目标；日志为 `corrected_native_initial.log` 和 `corrected_native_resume.log`。从恢复后的 checkpoint 导出产生 57 个 raw train row 和 57 个 raw validation row；command-admitted filtering 在每个 split 保留 19 个 LeRobot row。Dataset 位于 `outputs/tasks/humanoidbench_balance_simple/real_profile_corrected/teacher_dataset/`。

## 下一次正式实验

在执行主机上准备好配置中的 teacher/base/baseline artifact，预留所需 GPU，然后使用前面的 full-chain command。本 checkout 中 HAIC teacher 和 baseline model 仅在远程；复现时使用 HAIC section 的显式 remote config，或将其 stage 到配置的本地路径。其余三个任务配置的 teacher/model asset 已在本地找到。

运行干净的完整 profile 协议和正式 native/VLA 预算，再按配置的 episode 集合比较 baseline/new policy。微小的 failed-episode smoke dataset 只证明 wiring，不能作为 accepted training demonstration。正式长跑前要记录实际吞吐、peak memory 和磁盘增长；此前 smoke 的运行时间不能作为资源预算。

正式训练仍未完成。必须使用干净的完整 profile、显式 model path 和四个任务命令；短的、受污染的或 partial profile 只能作为功能 smoke 输入。Balance 明确使用现有 10k GR00T artifact，而不是缺失的 15k 路径，并为该选择使用单独命名的 experiment。

磁盘回收只删除了过期 smoke run 的 weights/optimizer 文件：Balance 的 `vla_new_chain_2updates` 与 `real_profile_corrected/vla_localmirror`，Parkour 的 `vla_local_backbone_2updates`，以及 HAIC 的 `vla_real2_final` 和一步的 `haic_vla_acceptance3/vla_smoke`。其日志/config/metadata 仍保留在远程；前三个 run 的 metadata 也复制到各任务本地的 `archived_remote_smokes`。被删除的 weights 不可恢复，不能使用这些历史路径做 reload。

accepted Balance `vla_final`、Parkour `actor_objective_vla_2updates_complete` 和 HAIC `haic_vla_acceptance3/vla_smoke2` 中顶层的 byte-identical weight duplicate 也已删除，但完整的 `checkpoint-2` 目录仍保留；需要时可通过复制 checkpoint shard 重建 duplicate。始终加载文档中记录的 checkpoint directory，而不是其 parent。Baseline、teacher 和 dataset 均保留。

## 正式执行记录（2026-09-06）

干净的五-session profile 和 source VLA baseline 已完成：

| 任务 | profile HF commit | zero mean return | temporal mean return | simulation commits |
|---|---|---:|---:|---|
| Parkour | `88321b5b3f6b3e343e1dd6f9a6bb46df463a9cf6` | 14.0489 | 13.4506 | `e18ca875`, `ca65a585` |
| HAIC | `cb65a3b210eb0a7dc30b4b151579d9a162e0514f` | 4.0554 | 3.0516 | `ac776be6`, `cb2cfda5` |
| Balance | `f6414b307a6d2260efb9804ed57ae5d7c4ce54b9` | 22.1493 | 32.7462 | `c4351ab7`, `97cc0edf` |
| MIKASA | `9d8bae1c6c0a17775549205a0cf61f72cbc09fc9` | 22.3260 | 0.3270 | `7bdcc4d4`, `63018122` |

MIKASA 使用 50 个 episode，其余任务使用 100 个。每次 temporal simulation 都使用 latency seed 0 和 immutable profile revision。

正式 latency-command training 按要求暂停。MIKASA 达到 iteration 1,029、63,221,760 steps；其 immutable pause bundle 为远程主机上的 `outputs/latency_rl/mikasa/formal_temporal_150m/paused_20260906_training.pt`，W&B run 为 `3xo01lnl` 和恢复后的 `dcpkb4lb`。Parkour 暂停在可恢复的 `model_600.pt`；其 run directory 为 `third_party/Isaaclab_Parkour/logs/rsl_rl/unitree_go2_parkour/2026-09-06_22-27-48_latency_command_seed1_15k`。Parkour 前 20 次 update 都是 finite，约 1,967 macro steps/s；但到 iteration 610 时 reward 和 episode length 已 collapse，因此保留 0 到 600 的 checkpoint 以通过 temporal gate。第一次修正后的 temporal evaluation 在完成前被中断，必须重启。正式配置为 `configs/training/latency/mikasa_formal_temporal.yaml` 和 `configs/training/latency/parkour_formal_temporal.yaml`。
