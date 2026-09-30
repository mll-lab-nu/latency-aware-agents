# program 文档索引

核心进展入口为 [PROGRESS](PROGRESS.md)，只记录 Balance 与 MIKASA 的已核验结果、关键决策、最新状态和下一步。
运行时间线放在 `logs/`，原始证据放在 `outputs/`，可执行方法和实验合同放在对应任务文档。
使用仓库实际模型名 `qwenpi_v3`；历史运行消息不代表当前进程状态。

## 工程审计与重构清单

| 文档 | 内容 / 主要用途 | 状态与边界 |
|---|---|---|
| [REFACTOR](REFACTOR.md) | 全仓库 40 项重构的当前决定、实施分组和验收证据 | 39 项完成、1 项测量后保留；[934 项测试及 CPU 对照](../outputs/refactor/consolidation/report.md)，已删除 39 个一次性脚本及部署配置 |

## 任务与运行手册

| 文档 | 内容 / 主要用途 | 状态与相关文档 |
|---|---|---|
| [BALANCE](BALANCE.md) | Balance Simple 早期 NVIDIA GR00T 实验和 10k 历史结果 | 历史记录；当前 H8 StarVLA 路线见 BALANCE_LOCAL_H8 和 PROGRESS |
| [MIKASA](MIKASA.md) | InterceptGrabFast 环境、PPO teacher、275 条轨迹、LeRobot、QwenOFT 训练与评测 | 保留；早期 zero baseline，profile 详见 PROFILE |
| [MIKASA_ACTION_CHUNK_H8](MIKASA_ACTION_CHUNK_H8.md) | 7D H8 baseline，以及 71D v2 Profile/Fixed-2 prefix 历史合同与结果 | 按对应输入、checkpoint 和 seeds 阅读；63D v3 接续见 PREFIX63_RESTART，最新核验见 PROGRESS |
| [MIKASA_FIXED_LATENCY_GRID](MIKASA_FIXED_LATENCY_GRID.md) | QwenOFT H8 固定 prefix 与 5×5 成功率矩阵的原方案 | 旧 Fixed-2 五档完成；因 71D 输入缺陷先转入合同修复 |
| [MIKASA_PREFIX63_RESTART](MIKASA_PREFIX63_RESTART.md) | 63D v3 四路重采集、五档 periodic、matched 选优与阶段结束发布协议 | 接续运行协议；部署与进度按 2026-09-25 快照读取，最新核验见 PROGRESS |
| [MIKASA_FIXED2_PREFIX_REPAIR](MIKASA_FIXED2_PREFIX_REPAIR.md) | 63D v3 输入修复、Fixed-2 训练与旧数据复用方案 | 2026-09-24 修复依据；后续重新采集与发布使用 PREFIX63_RESTART |
| [SCALE_MODELS](SCALE_MODELS.md) | Balance/MIKASA×qwenoft/qwengr00t/qwenpi_v3/wanoft 的 H1 公共入口与历史 smoke | 活跃公共运行手册；正式训练不能用两步 smoke 代替 |
| [PROFILE](PROFILE.md) | MIKASA/QwenOFT 早期实机测量、五场采样、发布与回放结果 | 保留测量事实；当前 RTX 3090 覆盖与版本见 PROGRESS |
| [ATLANTIS_ASTERIX_H8](ATLANTIS_ASTERIX_H8.md) | 两游戏 H8 zero/profile VLA、CPU teacher、3090 profiling 和三组评测合同 | 已完成：12 个 VLA、6 个 profile teacher、6 份 profiles 均发布回读；18/18 组、1800 局正式评测已核验；[完整结果](../outputs/tasks/atari_hist8_v1/evaluation_progress.md) / [执行记录](logs/atlantis-asterix-h8-2026-09-24.md) |

## 实验、合同与活跃交接

Demon Attack 旧 CuLE teacher 的首次丢命终评与奖励裁剪问题、300局反事实复现、修正后的正式成绩及 Flappy 同预算对照，见 [评估合同](DEMON_ATTACK_EVAL_AUDIT.md) 和 [核验记录](logs/demon-cule-evaluation-audit-2026-09-29.md)。

本地 Balance H8 新路线见 [BALANCE_LOCAL_H8](BALANCE_LOCAL_H8.md)；运行状态以该文档的最新目录与 PROGRESS 为准。

| 文档 | 内容 / 主要用途 | 状态与重叠关系 |
|---|---|---|
| [CHUNK](CHUNK.md) | Balance FastTD3 H8 action chunk 的修复、zero/fixed/profile 对照 | 保留实验记录；最新生产简化及消融见 CHUNK_RL_ABLATION |
| [BALANCE_MC_PPO](BALANCE_MC_PPO.md) | BC 初始化的 Balance H16 MC PPO、Sample Factory 所有权、train/resume/eval 入口 | 当前算法与运行手册；实验验收见 PROGRESS |
| [CHUNK_RL_ABLATION](CHUNK_RL_ABLATION.md) | H8 五配置×四 training seeds 的最终评测、D/E 简化选择 | 第一版 20 run 完成；D/E 组合及单独消融待验证 |
| [BALANCE_DAGGER](BALANCE_DAGGER.md) | QwenOFT DAgger 标签修复短交接、十轮结果及终点验收待办 | 活跃交接；详版历史在 archive/DAGGER |
| [BALANCE_CAUSAL_EVAL](BALANCE_CAUSAL_EVAL.md) | 现有 Balance 模型补评测协议、历史结果与统一入口用法 | 因果评测协议；正式结果见 PROGRESS 与实验报告 |
| [LATENCY_EVAL_UNIFICATION](LATENCY_EVAL_UNIFICATION.md) | 保留任务 evaluator、固定均值 prefix、sampler/worker 与 seed/reset 合同 | 保留合同记录；实现验证与正式训练分别记录 |
| [ARTIFACT_HANDOFF](ARTIFACT_HANDOFF.md) | Flappy/Demon/Deadly/Atlantis/Asterix 的 5 类历史资产、统一 100 局评测与中央 HF 路径 | 资产账本；正式评测、中央发布与 revision 回读分别以对应证据确认 |
| [HANDOFF](HANDOFF.md) | 五环境原稿的接收、评测配置与发布状态快照 | 2026-09-30 收录；原稿状态日期与逐行分数来源待核实，最新任务证据见对应协议，正式结果索引见 RESULTS |
| [DECISION_HISTORY_VLA_H8](DECISION_HISTORY_VLA_H8.md) | Flappy/Demon/Deadly × 三架构 × profile/zero 条件，共 18 路 H8 VLA 与 27 组交叉评测合同 | 18/18 模型已发布 HF 并回读；27/27 新版及 9/9 旧权重重评完成；[最终报告](../outputs/tasks/decision_history_vla_hist8_v2/final_report.md) |
| [logs/decision-history-vla-v2-2026-09-23](logs/decision-history-vla-v2-2026-09-23.md) | history01 修复、PI LR 选优、18 路训练、zero teacher 采集与旧权重纠正评测 | Deadly 渲染合同已修正；全部结果核验完毕，监控正常退出 |
| [LATENCY_RL_HANDOFF.zh-CN](LATENCY_RL_HANDOFF.zh-CN.md) | 保留任务 H1 中文交接、诊断结论与旧 H40 smoke | 历史交接；当前运行状态和入口见 PROGRESS 与 SCALE_MODELS |
| [logs/starvla-resident-dagger-migration-2026-09-19](logs/starvla-resident-dagger-migration-2026-09-19.md) | StarVLA DAgger 常驻训练权重推理、smoke 验证与正式迁移 | 运行记录；未经核验的正式结果不写入 PROGRESS |
| [logs/mikasa-h8-zero-latency-2026-09-20](logs/mikasa-h8-zero-latency-2026-09-20.md) | MIKASA H8 数据审计、相关测试、三模型 smoke 与正式运行时间线 | 运行记录；完成状态和结果只以 PROGRESS 为准 |
| [logs/atari-gpu-training-gate-2026-09-22](logs/atari-gpu-training-gate-2026-09-22.md) | Atlantis/Asterix CuLE 接入、2048/4096 H100 吞吐与 100 局 GPU/CPU 兼容评测 | 2048 候选通过吞吐与兼容门；三 seed 同帧数质量 A/B 尚未完成 |
| [logs/worker-capacity-contract-2026-09-23](logs/worker-capacity-contract-2026-09-23.md) | Flappy/Demon/Deadly 训练评测容量不一致的诊断、统一实现、H100 验证与 corrected 重训协议 | 九项旧 profile-small 结果失效；corrected 产物只留挂载盘，未经批准不更新 HF |
| [logs/flappy-capacity1-profile-learning-2026-09-23](logs/flappy-capacity1-profile-learning-2026-09-23.md) | Flappy 帧堆叠评测修复、OFT/GR00T 修正基线、18k horizon 回报上限与 APPO 单变量实验 | GR00T 25D 历史在两组评测场景明显增益；忙碌位对照未完成，后续转入九条 24D/跨游戏训练；[旧 25D 配方归档](archive/flappy-capacity1-20260923/README.md) |
| [logs/decision-action-history-nine-profile-2026-09-23](logs/decision-action-history-nine-profile-2026-09-23.md) | Flappy/Demon/Deadly 共用 8 次决策历史与 3090 temporal 配方；Deadly 时钟修正与重跑 | 九条训练及 final/best 各 100 局已完成；[结果汇总](../outputs/tasks/decision_history_nine_profile_20260923/summary.md) |
| [logs/mikasa-h8-conditioned-2026-09-21](logs/mikasa-h8-conditioned-2026-09-21.md) | MIKASA conditioned H8 的实现、四路 collector、六路训练与三条终评流水线 | 运行记录；未经核验的正式结果不写入 PROGRESS |
| [logs/mikasa-prefix63-failure-2026-09-25](logs/mikasa-prefix63-failure-2026-09-25.md) | 三路DAgger失败与第一台Local SSD只读排查 | 原始trainer异常待取证；第一台阵列元数据缺失且抽样全零 |
| [logs/mikasa-prefix63-expansion-2026-09-24](logs/mikasa-prefix63-expansion-2026-09-24.md) | Fixed-1/3/4沿用63D合同的正式训练、matched选优和五档评测 | 原L2继续；三路新流水线启动 |
| [logs/mikasa-fixed2-prefix63-2026-09-24](logs/mikasa-fixed2-prefix63-2026-09-24.md) | 63D 合同验证、独立目录部署与 Fixed-2 正式重训 | 当前完成状态以 PROGRESS 为准 |
| [logs/mikasa-fixed-grid-2026-09-24](logs/mikasa-fixed-grid-2026-09-24.md) | Fixed-1/3/4 正式采集、SFT/DAgger 与 5×5 grid 的启动和运行记录 | 当前完成状态以 PROGRESS 为准 |
| [logs/mikasa-h8-dagger-2026-09-23](logs/mikasa-h8-dagger-2026-09-23.md) | 六路 conditioned H8 从 SFT step-5000 接续 DAgger 的运行、终评与发布记录 | 九路汇总结论见 PROGRESS |
| [logs/mikasa-h8-naive-dagger-2026-09-23](logs/mikasa-h8-naive-dagger-2026-09-23.md) | 三路 zero-latency naive H8 DAgger 的运行、终评与发布记录 | 九路汇总结论见 PROGRESS |

## 研究资料

| 文档 | 内容 / 主要用途 | 重叠与阅读关系 |
|---|---|---|
| [PROPOSAL](PROPOSAL.md) | 以 MIKASA 为核心的可解性→延迟退化→恢复研究论证 | 研究方针 |
| [LATENCY_RL](LATENCY_RL.md) | causal history、推理等待及信息可见性的概念说明 | 与 related_works/note、综述相关；保留解释性资料 |
| [initial_investigations](initial_investigations.md) | 初始 benchmark 筛选、机器人与飞行候选及工程缺口 | 初始调研资料；后续选定任务的事实见任务手册 |
| [related_works/README](related_works/README.md) | 延迟 RL、critic、teacher/student 信息差、chunk、DAgger 综述 | 主要研究综述入口；与概念文档和 proposals 有重叠，全文保留 |
| [related_works/note](related_works/note.md) | 请求未完成时可见的 observation、动作与等待状态 | 短研究笔记，与 LATENCY_RL/综述重叠 |

## 历史归档

归档保留原有正文、实验表格和运行记录，修正迁移后的链接。未完成事项已迁入活跃交接。

| 文档 | 历史内容 | 当前接续位置 |
|---|---|---|
| [archive/DAGGER](archive/DAGGER.md) | 标签 bug、因果对照、单轮/十轮 DAgger 详版 | BALANCE_DAGGER 保留十轮终点确认、回传与最终验收待办 |
| [archive/LATENCY_DEBUG_PLAN](archive/LATENCY_DEBUG_PLAN.md) | Balance history/critic、fixed1/F2、fixed2/profile、service/hold/admission 诊断 | 中文交接保留最新结论；fixed1 F2 成功与后续失败分开记录 |
| [archive/LATENCY_PLAN](archive/LATENCY_PLAN.md) | 早期 fixed-latency 分阶段实验计划及执行摘要 | 实验 3–5 保持历史未执行状态，本轮产物已删除 |
| [archive/LATENCY_RL_HANDOFF](archive/LATENCY_RL_HANDOFF.md) | 2026-09-09 英文 Balance PPO/FastTD3 交接及当时结论 | 中文交接与 PROGRESS 记录更新证据；旧结论按原日期阅读 |

## 进度

| 文档 | 用途 | 状态 |
|---|---|---|
| [PROGRESS](PROGRESS.md) | Balance Simple 与 MIKASA 的核心结果、关键决策、最新状态与下一步 | 只在已核验结果、决策或最新状态变化后更新 |
