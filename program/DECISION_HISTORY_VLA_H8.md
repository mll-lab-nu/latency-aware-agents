# 三游戏决策历史 VLA H1 运行合同

当前范围为 Flappy、Demon Attack、Deadly Corridor × QwenOFT、QwenGR00T、QwenPI v3
× profile/zero 两种训练条件，共 **18 个独立模型**。每条从
`Qwen/Qwen3-VL-4B-Instruct@ebb281ec` 初始化，动作头随机；
不使用 Bridge 或 zero-latency VLA checkpoint。输入为当前 RGB 和本次动作执行前的
8 次决策历史，动作预测 horizon 为 1。

下表是已完成的 profile 数据采集协议，保留其原始帧上限，不重写现有训练数据。

| 游戏 | teacher 初批 | 原始帧上限 | probe / replay | state | 动作目标 |
|---|---:|---:|---:|---:|---|
| Flappy | 200 局 | 18,000 | GPU 256 / 256 | 24D | 2 类 |
| Demon | 200 局 | 14,400 | GPU 256 / 256 | 88D | 6 类 |
| Deadly | 2,000 局 | 3,600 | CPU 16 / 8 | 120D | `multibinary_7` |

三个游戏各族使用对应 [H8 teacher 发布回执](../outputs/tasks/decision_history_temporal_25m_seed42/)
中的 best checkpoint 和精确 profile 内容；模型仓库 revision 固定为
`af125bd372c22c5b00e326d9af1af64ea3751b56`。`collect_vla_hist8.py` 锁定 HF revision、
核对 checkpoint/profile SHA256，按完整 episode 作 90/10 train/val split。首批
训练标签不足 320,000 时，Flappy/Demon 每次追加 50 局、Deadly 每次追加 250 局，
直到达标。`filter_preset=none`，不以回报筛局，不遮蔽失败前最后若干决策；真正终止后
不生成标签，horizon 截断另记。每路产出 `raw/`、`lerobot/`、`quality_report.json`
和 `provenance.json`。

历史编码沿用 `DecisionActionHistory`：每次决策的接纳位、被接纳的命令、该窗口
最后实际执行的动作，按过去到现在排列；episode 起始补零。采集直接读取 teacher
动作前 observation 的 `transport`。VLA 评测由执行器按自己的动作和调度结果更新
`transport`，推理时缺少该 state 会报错。二值 history 在训练和推理中均直接使用原始
`0/1`；`state_source=transport` 只调整模型要求的形状，不做 min-max 映射。
连续状态任务仍使用原有归一化。旧评测把零 history 映射成全 `-1`，因此原结果失效；
旧权重可直接用修正后的输入重新评测。
Deadly 的 VLA 环境输出 `MultiBinary(7)`；评测历史将其语义按钮顺序重排为
teacher 的 turn / move / strafe / attack 位序，因而仍为 120D，并保留学生同时
按下相反方向按钮时的实际位值。
同模型使用固定任务提示；已实现的当前动作延迟仅留作数据审计，不进入动作前提示。

Temporal profile 的隐藏状态原本会跨局延续，使 probe 与 replay 因 slot 分配而产生
不同标签。此离线数据协议对每个**显式环境 reset seed** 开一条独立的 temporal
采样流，保留局内的相关性，并记录 `profile_seed_by_reset=env_reset_seed`。Flappy/Demon
的 GPU 模拟还受 batch 形状影响，因此两阶段都用 256 槽、按 seed 顺序分配；Deadly
每个新完整 episode 重建 CPU 环境。每局 probe/replay 长度不一致即停，不使用该数据。
Flappy/Demon 的 GPU replay 若中断，保留 probe state，清理部分 `raw/episode_rows`
和 `raw/images` 后从全部 256 槽重播；跳过已完成局会改变剩余 seed 的 slot 分配。
Deadly 的短局若使首批 2,000 局不足 320,000 个训练标签，可在保留主目录
`raw/probe_state.json` 的前提下加 `--deadly-probe-shards 3 --deadly-shard-episodes 2400`；
三个 CPU probe 子进程使用互不重叠的高位 seed 区间；合并状态按 2,000 局起、
每批 250 局核对训练标签，首次达到 320,000 即停止剩余 probe，随后以 8 槽 replay。
Deadly CPU 进程的 OpenMP/MKL 线程数固定为 8，避免多个 16 槽 probe 同时争抢全部核。

在 H200 工作目录运行一条采集：

```bash
CUDA_VISIBLE_DEVICES=0 conda run -n latency python -m scripts.tasks.decision_history.collect_vla_hist8 \
  --task flappy --model qwenoft --output-root outputs/tasks/decision_history_vla_hist8/data \
  --profile-root "$PWD"
```

新版在独立代码快照目录 `hist8-vla-v2-20260923` 运行，复用原数据和 profile。
既有 profile 采集不重做；新增 zero 条件另采集三份按游戏共享的数据。
18 路均从上述原始预训练底座初始化；在构造模型和随机动作头之前设置
共同 seed 42，构造后设置各 rank 随机流。完整断点恢复使用 checkpoint RNG。

在空闲双卡 slot 运行训练；四个位置参数依次为游戏、模型、GPU 对和分布式端口：

```bash
bash scripts/tasks/decision_history/launch_vla_hist8_train.sh flappy qwenoft 0,1 29740 \
  --run-id flappy-qwenoft-hist8-scratch-s42-v2 --vlm-lr 3e-6
```

同一命令最后加 `--resume`，从最近完整的 optimizer/scheduler state 继续，不重新
warmup。`--stop-after-steps 1000` 只设置暂停点，不缩短 5,000 步 scheduler。
正式参数为 bf16、ZeRO-2、gradient checkpointing、每卡 64、累积 1、global batch
128、5,000 optimizer steps；100 步 warmup。OFT/GR00T/PI 底座 LR 统一为 `3e-6`，
动作头 `1e-4`；投影层等其他参数组沿用原配置。cosine 显式使用 `min_lr_rate=1/30`，
逐参数组记录实际 LR；每 500 步保存并计算 held-out loss。每卡 64 是硬约束，
OOM 时停下报告。W&B key 只放在工作目录 mode-600 的 `.wandb.env`，
目标为 `saberrr-zju/starvla_tasks`。
按用户于 2026-09-24 的明确要求，全部训练和验证 dataloader 均使用每 rank
`num_workers=16`；双卡每个 loader 合计 32 个 worker。`prefetch_factor=4`，
`persistent_workers=true`、`pin_memory=true`，batch 仍为每卡 64/global 128。
LeRobot state/action 读取只索引所需决策行，不为每个样本堆叠整局最多 18,000 行。
此前启动的任务从完整中断断点切换 worker 数；切换步数记录在运行日志中。

PI 已完成 Demon 学习率选择，三游戏采用 `3e-6`，不再创建配对或诊断任务。
现有 Demon 胜者从完整断点续训；已有队列继续读取已完成的选择结果以保持 run ID
和恢复路径。新建队列直接生成固定 LR 的正式训练。历史选择证据见
[Demon PI 选优回执](../outputs/tasks/decision_history_vla_hist8_v2/selection/demon_attack-pi.reviewed.json)。
候选 1,000 步权重、配置和诊断已固定保存在服务器
`selection/demon_attack/candidates/<run-id>/`。低 LR 的 live run 已续训至 5,000 步，
其配置与 checkpoint 已更新；学习率选优审计使用候选快照，不重新执行已完成的选优。
学习率选择不代表正式环境性能已验收。

`make_hist8_v2_queue.py` 根据已核验旧资产目录生成各主机任务图；
`run_hist8_v2.py run --manifest <queue/manifest.json>` 持续调度空闲 GPU。
训练按 Demon、Flappy、Deadly 优先级入队，训练优先保留双卡 slot。常规评测也按
双卡任务占用和释放，32 个并行环境由两张卡分担推理；不会从一对空卡中拿一张
做长评测而使另一张无法训练。第一台 H200 的单独余卡 7 可做采集/32 并行单卡评测；
它不拆开当前的训练卡对。H100 承接 profile→profile 及部分旧权重评测。
H100 空闲时也可承接其他主机尚未启动的 zero 条件评测；迁移保留原 job ID 和
评测协议，源队列先移除 pending 项，目标校验 bundle SHA 后以 ready 标记接入。
数据采集与 CPU 转换/跨机同步是分开的任务，转换和传输不占 GPU slot。
训练、checkpoint 封装和正式评测是显式依赖；失败任务保留日志，需要定位后显式 retry。
训练、checkpoint、bundle、评测按 run ID 隔离；所有新版使用独立 W&B run。

本机 `codex-hist8-v2-monitor.service` 每 1,200 秒（20 分钟）查询三台队列。任务失败、
进程退出、30 分钟无日志、数据转换或正式评测完成时，使用 `codex queue`
唤醒当前任务 `01a0ce66-e6e7-72d2-be90-4c33d8055a4e`；相同事件去重，状态未变时
不发消息。监控脚本为 `monitor_hist8_v2.py`，回执在本地
`outputs/tasks/decision_history_vla_hist8_v2/monitor/`；全部队列完成后监控退出。

每条训练完成后封装本地 bundle，包含权重、训练配置、统计、质量报告、manifest、
collection provenance、实际 training step、W&B run ID、checkpoint SHA 和评测协议
`hist8-history01-par32-v3`。仅需要迁机时执行
`bash scripts/tasks/decision_history/transfer_vla_hist8_bundle.sh <task> <model> <run-id>`，
比较源和目标 SHA256。评测入口为
`launch_vla_hist8_eval.sh <task> <model> <gpu> <run-id>`。

新版每个游戏/架构各做 zero→zero、zero→profile、profile→profile 三组评测，共
**27 组 × 100 局**，共同 seeds `1,000,000–1,000,099`、`capacity=1`、parallel 32。
新版指标 namespace 为 `eval_hist8_v3/<training>_to_<environment>`。
Deadly 的渲染合同修复后使用独立 `hist8-vla-eval-sf-render-v4-20260924` 快照、
`hist8-history01-sf-render-v4` 协议及 `eval_hist8_v4/<training>_to_<environment>` 指标；
旧 Deadly 权重使用 `eval_history01_v2/sf_render_v4`。当次 Flappy/Demon 评测保留 v3。
以上版本指已归档的运行快照；当前源码新封装的 bundle 和新生成队列统一使用 v4
协议与指标 namespace，已有结果不改写。
队列为每个 job 显式记录 `runtime_root`，训练继续原快照；Deadly 封装和评测使用 v4。
跨机迁移同时映射 `runtime_root`、bundle 和 profile 路径。Deadly 的复现入口使用
对应 v4 job 保存的配置与运行目录，证据位于 `queue/jobs/<job-id>/attempt_<n>/`。

Deadly 原始图像必须匹配采集时的 Sample Factory 视图：160×120、HUD 开、准星关、
武器开、贴花关、粒子关，再按模型合同缩放。小 teacher 的 128×72 特征图分辨率
不等于其原始 RGB 渲染分辨率。此前 Gymnasium 默认 320×240、隐藏 HUD 的 Deadly
结果已失效；固定原权重的正式重评和证据见
[渲染合同修复](../outputs/tasks/decision_history_vla_hist8_v2/deadly_renderer_fix.md)。
后续正式评测与新增 zero 采集统一使用以下上限：

| 游戏 | env FPS | obs/decision Hz | max 原始帧 | 模拟时长 |
|---|---:|---:|---:|---:|
| Flappy | 10 | 10 | 3,600 | 6 分钟 |
| Demon | 60 | 15 | 7,200 | 2 分钟 |
| Deadly | 35 | 8.75 | 3,600 | 约 103 秒 |

Flappy/Demon 使用 GPU 批量环境，Deadly 使用 CPU 环境；保存逐局回报、局长、终止
信息，以及动作分布和原始分数。旧权重纠正结果写入原 W&B run 的独立
`eval_history01_v2/par32_max<frames>` namespace，不覆盖原长上限结果或已失效结果。

新增 zero teacher 使用中央仓库 `zero-latency/<game>/small-policy/sample-factory-v1`，
revision 固定为 `5b533937a72e9fa923ecf28b1c1be5a16bad488d`；checkpoint SHA 见
[zero_teacher_catalog.json](../outputs/tasks/decision_history_vla_hist8_v2/zero_teacher_catalog.json)。
保留 teacher 自身四帧视觉输入，为学生单独导出当前 RGB 与 8 次因果动作历史；
固定零延迟下，每次决策均接纳，历史在 episode/life reset 时清零。Flappy 的
published teacher 原生为 30/30 FPS；本实验采集明确覆盖为表中的 10/10 FPS，
teacher 权重不变。rollout reset seed 为 42，完整 episode 90/10 划分，不按回报过滤，
训练标签至少 320,000。每个游戏三种 VLA 共用同一份 zero 数据和 split。
`collect_zero_vla_hist8.py --stage rollout` 生成原始数据，`--stage convert` 生成
LeRobot、质量报告、provenance 和最后写入的 `collection.complete.json`；训练只在
该标记存在且依赖任务完成后启动。Demon 数据在第一台采集并同步到第二台。

旧批次六路保留 step 5,000；Flappy GR00T、Deadly GR00T、Deadly PI 分别保留
4,196、2,877、3,266 步。它们按实际步数重评，不能当作等训练量对照。
旧 optimizer state 已由用户授权的清理任务删除，九路新版不依赖这些状态。
证据见[旧批次交接](../outputs/tasks/decision_history_vla_hist8/old_history01_handoff.md)。
训练完成、评测完成和性能验收分别记录；新版执行证据见
[修复与重训记录](logs/decision-history-vla-v2-2026-09-23.md)。
已核验的纠正重评与新版正式结果集中在
[评测汇总](../outputs/tasks/decision_history_vla_hist8_v2/evaluation_summary.md)，
逐项列出实际训练步数、帧上限、并行数、动作分布和 SHA；旧协议结果不混作新版对照。
旧九路权重的正确 `0/1` history 重评均已完成 100 局，并核验全部 seeds、权重/profile
SHA、动作和逐局记录；具体帧上限与并行数以汇总逐项记录为准。此项完成不等于新版
性能自动合格。

截至 2026-09-24，18/18 个新版模型均已完成 5,000 步，27/27 组新版正式评测
（2,700 局）及 9/9 组旧权重纠正重评（900 局）全部通过记录核验。
18 路最终权重及双 rank optimizer/RNG、scheduler 状态均已保留；W&B 回读的
27 组新版均分与本地一致。82 个队列节点完成，controller 与监控正常退出。
后续无需重新入队；模型身份、SHA 与结果证据见
[收尾清单](../outputs/tasks/decision_history_vla_hist8_v2/completion_manifest.json)，
性能结论与限制见[最终报告](../outputs/tasks/decision_history_vla_hist8_v2/final_report.md)。

18 个新版模型已按用户要求发布到 `latency-sensitive-bench/benchmark-models`，
固定 revision `95aed229d8a90c0146885eb8f496518e98625d5f`。
zero 条件 9 个位于 `zero-latency/<game>/vla/starvla-<model>-h1/<run-id>`，
profile 条件 9 个位于 `latency-aware/<game>/vla/starvla-<model>-h1/<run-id>`；
路径中的 h1 是动作 horizon，hist8 是输入历史。18 个权重 SHA 与最终 revision
全部匹配，588 个元数据对象核对通过，102 个核心文件已实际下载回读。
目录、配置与成绩见[公开索引](https://huggingface.co/latency-sensitive-bench/benchmark-models/blob/af8d7bfacbb03b59b5b5eabbed331577dd5fa690/README.md)，
完整证据见[发布回执](../outputs/tasks/decision_history_vla_hist8_v2/hf_publication/README.md)。
