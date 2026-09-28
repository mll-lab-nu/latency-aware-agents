# 九条 3090 temporal profile 决策历史 APPO 训练（2026-09-23）

## 合同与实现

每条 run 从零训练 Sample Factory 小策略 25M env steps，seed 42，单个模拟推理 worker，使用对应 OpenVLA/QwenOFT、GR00T、π0.5 RTX 3090 temporal profile。Flappy 为 10/10 FPS、`one_frame_then_noop`；Demon Attack 为 60/15 FPS、`hold`；Deadly 原计划为 60/15 FPS、`hold`，随后按下文修正为 35/8.75 FPS。均采用 `latest_ready`。九条 profile 的绝对路径和 SHA256 见 [profile_revisions.json](../../outputs/tasks/decision_history_temporal_25m_seed42/profile_revisions.json)。训练配置不自动上传 HF；用户随后指定了五条已完成 run 发布 checkpoint，记录见下文。

`decision_v1` 在当前观测中提供最近 8 个已完成决策周期的接纳状态、被接纳的命令动作和该周期最后原始帧实际执行的动作。非 noop 动作按离散分量 one-hot 编码；Flappy 为 24D、Demon 为 88D、Deadly 为 120D。worker 忙碌位和未来返回时间不进入新输入。旧 Flappy 25D checkpoint 与新 24D 模型权重不兼容；新代码只实现 24D 决策历史。Demon GPU 从写死的 IID sampler 切到按配置构造 temporal sampler。

新增 `evaluation.checkpoint_interval_steps`：每 5M 保存并留存 checkpoint，不排队里程碑评测；25M 完成后评估 final，再对实际 best 做同一组 100 局 deterministic 评测。horizon 为 Flappy 18,000、Demon 14,400、Deadly 3,600。训练与终评完成状态分别以 manifest events 和逐局文件核对。

## 验证与运行状态

共享动作编码、四原始帧决策窗口、被丢弃命令、save-only 留点测试已通过。三游戏真实短训练分别到 Flappy 98,304、Demon 90,112、Deadly 67,584 steps；三者各有 2 局 temporal 评测通过，短跑目录已清理。Demon GPU 使用的 CuLE 源码已同步到 H100；编译需 conda 环境的 `libz.so` 路径，正式启动脚本已设置。生成的九条配置逐项确认 seed 42、25M、capacity=1、temporal、历史长度 8、100 局终评，且均无上传目标。Demon/Deadly 的旧空 `eval_suites` 覆盖问题已修正，九条现在都恰有一个 temporal suite。

此前的 Flappy no-busy run 已无相关进程；最后记录约 16.25M steps，未作为九条新训练之一。H100 GPU 0 已启动 Deadly OpenVLA→GR00T→π0.5 串行队列，GPU 1 已启动 Flappy OpenVLA→GR00T→π0.5→Demon OpenVLA→GR00T→π0.5 串行队列。每项先完成训练和 final 评测，再评估 best；一个 lane 失败时停在该项并保留现场，另一 lane 继续。当前只确认两条队列的第一项开始，不能将其标记为训练或评测完成。

## 02:15 UTC：Deadly 暂停

初次远端检查时 Deadly tmux 已退出，GPU 0 无训练进程；原 Deadly 生成配置目录和本轮任务目录中的 Deadly 产物也已不在远端。该次检查未再停止进程或删除文件，暂不采信旧结果。Flappy/Demon 的 GPU 1 队列保持运行。用户随后明确要求停止并删除仍在运行的 Deadly，再修复重跑；实际清理和新队列见下文。

## Deadly 时钟修正与重跑

Deadly 的 ViZDoom 场景 `get_ticrate()` 为 35，原三条 run 却以 60/15 FPS 解释 RTX 3090 毫秒延迟。每四个游戏 tick 一次决策时，原 OFT/GR00T 训练事件末期分别约 46/34 个决策、23/15 次观测接纳；所有记录局均由环境 `terminated`，并非评测 horizon 截断。按相同 temporal profile 抽样，原 60 FPS 下 OFT/GR00T 的平均 worker 服务分别约 5.1/8.3 tick，改为真实 35 FPS 后约 3.1/5.0 tick。旧生成评测配置另将号称 best 的外部评测设为 `latest`，OFT 的 100 局 1676.327 仅是 latest 的旧协议成绩。

用户要求停止并删除旧 Deadly 训练。旧三条 W&B run、训练 checkpoint、任务输出与生成配置已删除，π0.5 训练在中途终止；这些结果不作比较基线或发表。新 [Deadly native35 配置](../../configs/experiments/deadly_corridor/decision_history_native35_temporal_25m_seed42.yaml) 使用 35/8.75 FPS、四 tick 决策、capacity=1、8 次决策历史、原 temporal profiles、seed 42 和全新 run ID。生成的评测配置将 `policy.checkpoint_kind` 与 Sample Factory 加载参数均设为 `best`；final 仍单独评估 latest。配置生成测试通过，H100 GPU 0 已启动 OFT→GR00T→π0.5 的新串行队列，训练与正式评测结果尚未完成。

## 五条完成 run 的 checkpoint 发布

用户指定 Flappy OFT/GR00T/π0.5、Demon Attack OFT、Deadly Corridor native35 OFT 的训练所选 `best_*.pth` 上传中央模型仓库。五条 W&B 源 run 均为 `finished`，各自的 25M latest checkpoint、训练所选 best checkpoint 和原始 `config.json` 均已核对。五个 best checkpoint 与配置、provenance、README 发布在 [benchmark-models@9880201](https://huggingface.co/latency-sensitive-bench/benchmark-models/tree/988020114cc8201d529992031bc0fe5d41934b2b/latency-aware) 的独立 `sample-factory-{qwenoft,gr00t,pi05}-rtx3090-temporal-hist8-capacity1[-native35]-s42-v1` 目录；完整路径、所选 checkpoint、SHA256 见 [发布回执](../../outputs/tasks/decision_history_temporal_25m_seed42/hf_publish_receipt.json)。中央逐目录文件回读及五个权重的 Hub SHA256 与源文件 SHA256 比对均通过。

本次仅发布训练所选 best checkpoint，不将 checkpoint 文件名中的训练 reward 当作正式评测成绩。Flappy 三条的独立 100 局评测指向 latest checkpoint；Deadly native35 OFT 的 100 局评测指向 best；Demon Attack OFT 的 best 独立评测尚无已核对记录。因此，本次状态为 checkpoint 已发布，不代表五条 best 的正式评测或成绩验收全部完成。用户随后取消 W&B 迁移要求，五条 run 留在 `saberrr-zju/starvla_tasks`。

## 三条追加完成 run 的 checkpoint 发布

Deadly Corridor native35 GR00T/π0.5 与 Demon Attack GR00T 的 W&B run 均为 `finished`；三条都有 25M final、训练所选 best 和原始 `config.json`。三份外部 100 局 JSON 的 `checkpoint_kind` 均为 `best`，路径逐一指向本次所选的 `best_*.pth`。best checkpoint 与配置、provenance、README 已发布到 [benchmark-models@3d24469](https://huggingface.co/latency-sensitive-bench/benchmark-models/tree/3d24469718eaddc9db876796c102ec6e8184c777/latency-aware)，详细路径和 SHA256 见 [追加发布回执](../../outputs/tasks/decision_history_temporal_25m_seed42/hf_publish_receipt_three.json)。三个中央目录的文件回读、权重 SHA256 与源文件比对均通过。W&B run 仍留在原项目。

## 03:29 UTC：Demon 终评切到 GPU 批量后端

Demon OFT 到 25M 后，原配置 `eval_parallel_envs: 1` 导致评测器选串行 CPU 后端。其 CPU final100 已完成，CPU best100 在进行中时按用户要求中断。Demon sweep 源配置及远端三份生成的训练/评测配置均改为 `eval_parallel_envs: 16` 和 `latency_bench_env_backend: demon_attack_gpu_batched`。GPU 1 原 tmux 名下改用 `demon_gpu_resume` 队列：先把 OFT final100 重评到独立的 `step_25001984_final_gpu_batched` 目录，再评 OFT best100，随后训练并终评 GR00T、Pi。CPU final 结果保留作审计，不与 GPU final/best 混作同一协议。03:29 UTC 已观察 GPU 1 上的 OFT GPU final 评测启动，resolved config 确认为 16 路 CuLE 后端；尚未标记这条 GPU 评测完成。

GPU OFT final100 在 03:35 UTC 完成：100 条逐局记录，mean return 1022.45，耗时约 6 分钟。此前 CPU 串行 final100 为 5631.6；后端改变造成明显协议差异，必须分列，不能把 CPU final 与 GPU best 相比。远端三份 `latency_bench_eval.yaml` 的 `policy.checkpoint_kind` 和 `backend.extra_args` 同时改为 `best`，以免 manifest 的 best 阶段继续加载 latest；训练内 final 保持 latest。后续如重新生成 Demon sweep，须传 `--eval-checkpoint-kind best`。

## 04:42 UTC：Deadly native35 队列完成并核对

新版 Deadly OFT、GR00T、Pi 均有 `train_succeeded`、`eval_succeeded` 事件；三条各有 25,001,984-step final 和 best 的 100 条逐局记录。最后一个 best 评测成功后，Deadly tmux 脚本因运行中被更新而在读取新增的 `demon_gpu_resume` 分支时触发 `oft_label: unbound variable`，未打印 lane complete 标记。这是队列脚本退出错误，发生在所有 Deadly 训练与评测完成之后；未重跑或删除任何正式产物。后续不在活跃 tmux 运行同一脚本时覆盖其文件。

## 九个旧 profiled-small bundle 从 main 清理

用户确认删除 Flappy、Demon Attack、Deadly Corridor 各自旧的 `sample-factory-{openvla,gr00t,pi05}-rtx3090-v1`，共九个目录。删除前 main 为 `3d24469718eaddc9db876796c102ec6e8184c777`；[清单](../../outputs/tasks/decision_history_temporal_25m_seed42/legacy_small_policy_predelete.json)记录每个目录的五个文件、大小和权重哈希。单次删除提交为 [benchmark-models@a837166](https://huggingface.co/latency-sensitive-bench/benchmark-models/commit/a837166929198f76092198ba1ef86efcad45d2ea)。父目录前后树比对确认恰好移除九个目录及其 45 个文件，没有新增或额外删除；已发布的八个 temporal-hist8 checkpoint 目录仍可见。W&B run、源 checkpoint 与旧 revision 历史未改动。

## Demon Attack π0.5 的第九个 H8 checkpoint 发布

[Demon Attack π0.5 run](https://wandb.ai/saberrr-zju/starvla_tasks/runs/demon-attack-fs4-pi05_rtx3090_temporal_hist8_capacity1-s42) 已为 `finished`，25M final checkpoint、训练所选 `best_000047936_24543232_reward_2760.000.pth` 和原始配置均齐全。独立 100 局 JSON 的 `checkpoint_kind=best` 且路径指向该权重。best checkpoint 与配置、provenance、README 已发布到 [benchmark-models@46a33b1](https://huggingface.co/latency-sensitive-bench/benchmark-models/tree/46a33b18a2db8feb5cc6b520a95c2e96662c5e5b/latency-aware/demon-attack/small-policy/sample-factory-pi05-rtx3090-temporal-hist8-capacity1-s42-v1)；[回执](../../outputs/tasks/decision_history_temporal_25m_seed42/hf_publish_receipt_demon_pi05.json)记录完整 SHA256。中央目录及 provenance 回读通过，Hub 权重 SHA256 与源文件一致。九条 H8 best checkpoint 现均已上传，W&B 保持原项目。

## 05:48 UTC：Flappy best 评测纠正

九条训练在 05:33 UTC 前均已完成。终审逐局 metadata 发现 Flappy 三份原 manifest best 阶段实际加载 `checkpoint_kind=latest`，只重复了 final checkpoint；因此原所谓 best 评测不计入验收。旧结果各移至 `seed_42_latest_duplicate` 留存，远端三份生成的 `latency_bench_eval.yaml` 同时把策略和 Sample Factory 加载参数改为 `best`。05:48 UTC 开始用两张空闲 GPU 分别执行真正的 best100：GPU 0 串行 OFT→GR00T，GPU 1 跑 Pi。三份训练不重跑，checkpoint、100 局种子和动作协议均保持原样。若未来重新生成 Flappy sweep，也须传 `--eval-checkpoint-kind best`。

## 05:55 UTC：九条训练与正式终评完成

九条独立训练均达到 25,001,984 env steps；各有约 5/10/15/20/25M 留存 checkpoint。每条 final 与 best 均完成 100 局 deterministic seeds 104771–104870。逐局核对 checkpoint 绝对路径：final 为 `checkpoint_*.pth`，best 为 `best_*.pth`。9 个固定 profile 的 SHA256 与启动时记录一致。完整逐局 JSON、里程碑路径、接纳率与 W&B 链接见 [nine_profile_results.json](../../outputs/tasks/decision_history_nine_profile_20260923/nine_profile_results.json)；阅读版见 [summary.md](../../outputs/tasks/decision_history_nine_profile_20260923/summary.md)。

| 游戏 | 模型 | final100 mean | best100 mean | final / best 接纳率 |
|---|---|---:|---:|---:|
| Flappy | OFT | 1266.32 | 1233.51 | 99.7% / 99.7% |
| Flappy | GR00T | 1995.24 | 1995.24 | 50.0% / 50.0% |
| Flappy | Pi | 1062.49 | 1194.79 | 42.8% / 42.8% |
| Demon | OFT | 1022.45 | 1525.00 | 50.0% / 50.0% |
| Demon | GR00T | 32.90 | 663.25 | 46.1% / 46.1% |
| Demon | Pi | 27.25 | 447.30 | 32.0% / 32.0% |
| Deadly native35 | OFT | 2035.13 | 1933.82 | 100.0% / 100.0% |
| Deadly native35 | GR00T | 2191.83 | 2155.71 | 50.3% / 50.2% |
| Deadly native35 | Pi | 1347.48 | 1397.40 | 50.6% / 50.7% |

Demon 三条正式终评均为 16 路 CuLE GPU 协议。OFT 同一 final checkpoint 的旧 CPU 串行100局均分 5631.60、新 GPU100局均分 1022.45；两者不可混用，CPU 结果只留作审计。Demon GR00T/Pi 的 25M final 均分仅 32.90/27.25，best 独立评测为 663.25/447.30，不能把“训练和终评完成”说成“控制能力达标”。

Flappy 三份 best 纠正重评后，OFT/Pi 的 best 步数分别为 24,543,232/23,429,120；GR00T 的 best 与 final 均在 25,001,984。Deadly 只采用真实 ViZDoom 35 tick/s、35/8.75 FPS 的新 run，旧 60/15 FPS 成绩无效。checkpoint 发布状态以本文对应发布回执为准，与训练、终评完成分别记录。
