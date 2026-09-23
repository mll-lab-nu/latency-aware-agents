# 九条 3090 temporal profile 决策历史 APPO 训练（2026-09-23）

## 合同与实现

每条 run 从零训练 Sample Factory 小策略 25M env steps，seed 42，单个模拟推理 worker，使用对应 OpenVLA/QwenOFT、GR00T、π0.5 RTX 3090 temporal profile。Flappy 为 10/10 FPS、`one_frame_then_noop`；Demon Attack 为 60/15 FPS、`hold`；Deadly 原计划为 60/15 FPS、`hold`，随后按下文修正为 35/8.75 FPS。均采用 `latest_ready`。九条 profile 的绝对路径和 SHA256 见 [profile_revisions.json](../../outputs/tasks/decision_history_temporal_25m_seed42/profile_revisions.json)。不上传 HF。

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
