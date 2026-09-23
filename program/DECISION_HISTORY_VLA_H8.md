# 三游戏决策历史 VLA H1 运行合同

Flappy、Demon Attack、Deadly Corridor 各有 QwenOFT、QwenGR00T、QwenPI v3
一条独立模型。每条从 `Qwen/Qwen3-VL-4B-Instruct@ebb281ec` 初始化，动作头随机；
不使用 Bridge 或 zero-latency VLA checkpoint。输入为当前 RGB 和本次动作执行前的
8 次决策历史，动作预测 horizon 为 1。

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
采集不重做。九路均从上述原始预训练底座重训；在构造模型和随机动作头之前设置
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
按用户于 2026-09-24 的明确要求，九路训练和验证 dataloader 均使用每 rank
`num_workers=16`；双卡每个 loader 合计 32 个 worker。`prefetch_factor=4`，
`persistent_workers=true`、`pin_memory=true`，batch 仍为每卡 64/global 128。
LeRobot state/action 读取只索引所需决策行，不为每个样本堆叠整局最多 18,000 行。
此前启动的任务从完整中断断点切换 worker 数；切换步数记录在运行日志中。

PI 已完成 Demon 学习率选择，三游戏采用 `3e-6`，不再创建配对或诊断任务。
现有 Demon 胜者从完整断点续训；已有队列继续读取已完成的选择结果以保持 run ID
和恢复路径。新建队列直接生成固定 LR 的正式训练。历史选择证据见
[Demon PI 选优回执](../outputs/tasks/decision_history_vla_hist8_v2/selection/demon_attack-pi.reviewed.json)。
学习率选择不代表正式环境性能已验收。

`make_hist8_v2_queue.py` 根据已核验旧资产目录生成各主机任务图；
`run_hist8_v2.py run --manifest <queue/manifest.json>` 持续调度空闲 GPU。
训练按 Demon、Flappy、Deadly 优先级入队，训练依赖双卡 slot，评测用本机空闲单卡。
训练、checkpoint 封装和正式评测是显式依赖；失败任务保留日志，需要定位后显式 retry。
训练、checkpoint、bundle、评测按 run ID 隔离；所有新版使用独立 W&B run。

本机 `codex-hist8-v2-monitor.service` 每 1,200 秒（20 分钟）查询三台队列。任务失败、
进程退出、30 分钟无日志、数据转换或正式评测完成时，使用 `codex queue`
唤醒当前任务 `01a0ce66-e6e7-72d2-be90-4c33d8055a4e`；相同事件去重，状态未变时
不发消息。监控脚本为 `monitor_hist8_v2.py`，回执在本地
`outputs/tasks/decision_history_vla_hist8_v2/monitor/`；全部队列完成后监控退出。

每条训练完成后封装本地 bundle，包含权重、训练配置、统计、质量报告、manifest、
collection provenance、实际 training step、W&B run ID、checkpoint SHA 和评测协议
`hist8-history01-v2`。仅需要迁机时执行
`bash scripts/tasks/decision_history/transfer_vla_hist8_bundle.sh <task> <model> <run-id>`，
比较源和目标 SHA256。评测入口为
`launch_vla_hist8_eval.sh <task> <model> <gpu> <run-id>`。

新旧权重各做 100 个共同 held-out seeds（`1,000,000–1,000,099`）、各自 temporal
profile、`capacity=1` 的正式评测。Flappy 使用 10/10 env/obs FPS 和 18,000 帧上限，
Demon 60/15 FPS 和 14,400 帧，Deadly 35/8.75 FPS 和 3,600 帧。
Flappy/Demon 使用 GPU 批量环境，Deadly 使用 CPU 环境；保存逐局回报、局长、终止
信息，以及动作分布和原始分数。旧权重纠正结果写入原 W&B run 的独立
`eval_history01_v2` namespace，不覆盖已失效结果。

旧批次六路保留 step 5,000；Flappy GR00T、Deadly GR00T、Deadly PI 分别保留
4,196、2,877、3,266 步。它们按实际步数重评，不能当作等训练量对照。
旧 optimizer state 已由用户授权的清理任务删除，九路新版不依赖这些状态。
证据见[旧批次交接](../outputs/tasks/decision_history_vla_hist8/old_history01_handoff.md)。
训练完成、评测完成和性能验收分别记录；新版执行证据见
[修复与重训记录](logs/decision-history-vla-v2-2026-09-23.md)。
