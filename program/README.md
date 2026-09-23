# program 文档索引

核心进展入口为 [PROGRESS](PROGRESS.md)，只保留四任务的已核验结果、关键决策、最新状态和下一步。
运行时间线放在 `logs/`，原始证据放在 `outputs/`，可执行方法和实验合同放在对应任务文档。
使用仓库实际模型名 `qwenpi_v3`；历史运行消息不代表当前进程状态。

## 任务与运行手册

| 文档 | 内容 / 主要用途 | 状态与相关文档 |
|---|---|---|
| [BALANCE](BALANCE.md) | Balance Simple 早期 NVIDIA GR00T 实验和 10k 历史结果 | 历史记录；当前 H8 StarVLA 路线见 BALANCE_LOCAL_H8 和 PROGRESS |
| [MIKASA](MIKASA.md) | InterceptGrabFast 环境、PPO teacher、275 条轨迹、LeRobot、QwenOFT 训练与评测 | 保留；早期 zero baseline，profile 详见 PROFILE |
| [EXTREME_PARKOUR](EXTREME_PARKOUR.md) | Go2 teacher/student 与早期 NVIDIA GR00T action+yaw/DAgger 实验 | 历史记录；当前 StarVLA H1 入口见 SCALE_MODELS 和 PROGRESS |
| [HAIC](HAIC.md) | PullCart 原生 PPO 与早期 NVIDIA GR00T H40 实验 | 历史记录；当前 StarVLA H1 入口见 SCALE_MODELS 和 PROGRESS |
| [SCALE_MODELS](SCALE_MODELS.md) | 四任务×qwenoft/qwengr00t/qwenpi_v3/wanoft 的 H1 公共入口与 16 组 smoke | 活跃公共运行手册；正式训练不能用两步 smoke 代替 |
| [PROFILE](PROFILE.md) | MIKASA/QwenOFT 早期实机测量、五场采样、发布与回放结果 | 保留测量事实；当前 RTX 3090 覆盖与版本见 PROGRESS |

## 实验、合同与活跃交接

本地 Balance H8 新路线见 [BALANCE_LOCAL_H8](BALANCE_LOCAL_H8.md)；运行状态以该文档的最新目录与 PROGRESS 为准。

| 文档 | 内容 / 主要用途 | 状态与重叠关系 |
|---|---|---|
| [CHUNK](CHUNK.md) | Balance FastTD3 H8 action chunk 的修复、zero/fixed/profile 对照 | 保留实验记录；最新生产简化及消融见 CHUNK_RL_ABLATION |
| [BALANCE_MC_PPO](BALANCE_MC_PPO.md) | BC 初始化的 Balance H16 MC PPO、Sample Factory 所有权、train/resume/eval 入口 | 当前算法与运行手册；实验验收见 PROGRESS |
| [CHUNK_RL_ABLATION](CHUNK_RL_ABLATION.md) | H8 五配置×四 training seeds 的最终评测、D/E 简化选择 | 第一版 20 run 完成；D/E 组合及单独消融待验证 |
| [BALANCE_DAGGER](BALANCE_DAGGER.md) | QwenOFT DAgger 标签修复短交接、十轮结果及终点验收待办 | 活跃交接；详版历史在 archive/DAGGER |
| [BALANCE_CAUSAL_EVAL](BALANCE_CAUSAL_EVAL.md) | 现有 Balance 模型补评测协议、历史结果与统一入口用法 | 因果评测协议；正式结果见 PROGRESS 与实验报告 |
| [LATENCY_EVAL_UNIFICATION](LATENCY_EVAL_UNIFICATION.md) | 四任务 evaluator、固定均值 prefix、sampler/worker 与 seed/reset 合同 | 保留合同记录；实现验证与正式训练分别记录 |
| [ARTIFACT_HANDOFF](ARTIFACT_HANDOFF.md) | Flappy/Demon/Deadly/Atlantis/Asterix 的 5 类历史资产、统一 100 局评测与中央 HF 路径 | 活跃交接；只有正式评测与远端回读通过后才在 PROGRESS 标记发布 |
| [LATENCY_RL_HANDOFF.zh-CN](LATENCY_RL_HANDOFF.zh-CN.md) | 四任务 H1 中文交接、诊断结论与旧 H40 smoke | 历史交接；当前运行状态和入口见 PROGRESS 与 SCALE_MODELS |
| [logs/starvla-resident-dagger-migration-2026-09-19](logs/starvla-resident-dagger-migration-2026-09-19.md) | StarVLA DAgger 常驻训练权重推理、smoke 验证与正式迁移 | 运行记录；未经核验的正式结果不写入 PROGRESS |
| [logs/mikasa-h8-dagger-2026-09-23](logs/mikasa-h8-dagger-2026-09-23.md) | 六路 conditioned H8 从 SFT step-5000 接续 DAgger 的配置、smoke 与运行状态 | 运行记录；十轮与终评核验后更新 PROGRESS |
| [logs/mikasa-h8-naive-dagger-2026-09-23](logs/mikasa-h8-naive-dagger-2026-09-23.md) | 三路 zero-latency naive H8 DAgger 的队列与运行状态 | 运行记录；正式终点核验后更新 PROGRESS |
| [logs/atari-gpu-training-gate-2026-09-22](logs/atari-gpu-training-gate-2026-09-22.md) | Atlantis/Asterix CuLE 接入、2048/4096 H100 吞吐与 100 局 GPU/CPU 兼容评测 | 2048 候选通过吞吐与兼容门；三 seed 同帧数质量 A/B 尚未完成 |
| [logs/worker-capacity-contract-2026-09-23](logs/worker-capacity-contract-2026-09-23.md) | Flappy/Demon/Deadly 训练评测容量不一致的诊断、统一实现、H100 验证与 corrected 重训协议 | 九项旧 profile-small 结果失效；corrected 产物只留挂载盘，未经批准不更新 HF |
| [parkour/OPTIMIZE](parkour/OPTIMIZE.md) | CPU 调度收益、预算对齐及已回退的 PhysX/cache 实验 | 保留性能记录；训练吞吐与 VLA inference latency profile 分开解释 |

## 研究资料

| 文档 | 内容 / 主要用途 | 重叠与阅读关系 |
|---|---|---|
| [PROPOSAL](PROPOSAL.md) | 以 MIKASA 为核心的可解性→延迟退化→恢复研究论证 | 研究方针；与 HumanoidBench proposal 关注范围不同 |
| [HUMANOIDBENCH_PROPOSAL](HUMANOIDBENCH_PROPOSAL.md) | HumanoidBench teacher→VLA→profile→latency-aware 路线与 gates | 历史研究方针；当前任务为 Balance Simple |
| [LATENCY_RL](LATENCY_RL.md) | causal history、推理等待及信息可见性的概念说明 | 与 related_works/note、综述相关；保留解释性资料 |
| [initial_investigations](initial_investigations.md) | 初始 benchmark 筛选、HAIC/Parkour/飞行等候选与工程缺口 | 初始调研资料；后续选定任务的事实见任务手册 |
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
| [logs/PROGRESS-pre-cleanup-2026-09-17](logs/PROGRESS-pre-cleanup-2026-09-17.md) | PROGRESS 清理前的完整时间线与运行细节 | 仅供追溯；当前状态和结论以 PROGRESS 为准 |

## 进度

| 文档 | 用途 | 状态 |
|---|---|---|
| [PROGRESS](PROGRESS.md) | 四主任务的核心结果、关键决策、最新状态与下一步 | 只在已核验结果、决策或最新状态变化后更新 |
