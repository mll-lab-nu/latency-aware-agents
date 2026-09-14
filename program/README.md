# program 文档索引

整理日期：2026-09-14。本轮开始时有24份原文，新增本索引后共25份。进度入口为 [PROGRESS](PROGRESS.md)，包含四任务×四模型的
训练、评测、profile、发布状态及具体数值。使用仓库实际模型名 `qwenpi_v3`。
各文档中的运行消息以其记录时间为准；历史实验不会因归档而自动恢复执行。

## 任务与运行手册

| 文档 | 内容 / 主要用途 | 状态与相关文档 |
|---|---|---|
| [BALANCE](BALANCE.md) | Balance Simple 官方 GR00T teacher→数据→VLA 流程，10k 结果与 fresh 15k 计划 | 保留；官方 GR00T 与四类 StarVLA H1 是不同路线。最新产物见 PROGRESS |
| [MIKASA](MIKASA.md) | InterceptGrabFast 环境、PPO teacher、275 条轨迹、LeRobot、QwenOFT 训练与评测 | 保留；早期 zero baseline，profile 详见 PROFILE |
| [EXTREME_PARKOUR](EXTREME_PARKOUR.md) | Go2 teacher/student、VLA action+yaw、DAgger 和 latency pipeline | 保留；旧 latent-only 与 running 消息为历史，新 H1 产物见 PROGRESS |
| [HAIC](HAIC.md) | PullCart 原生 PPO、GR00T H40、数据与 latency pipeline | 保留；旧作废路线与当前 H1 结果分别记录 |
| [SCALE_MODELS](SCALE_MODELS.md) | 四任务×qwenoft/qwengr00t/qwenpi_v3/wanoft 的 H1 公共入口与 16 组 smoke | 活跃公共运行手册；正式训练不能用两步 smoke 代替 |
| [PROFILE](PROFILE.md) | MIKASA/QwenOFT 早期实机测量、五场采样、发布与回放结果 | 保留测量事实；当前 RTX 3090 覆盖与版本见 PROGRESS |

## 实验、合同与活跃交接

| 文档 | 内容 / 主要用途 | 状态与重叠关系 |
|---|---|---|
| [CHUNK](CHUNK.md) | Balance FastTD3 H8 action chunk 的修复、zero/fixed/profile 对照 | 保留实验记录；最新生产简化及消融见 CHUNK_RL_ABLATION |
| [CHUNK_RL_ABLATION](CHUNK_RL_ABLATION.md) | H8 五配置×四 training seeds 的最终评测、D/E 简化选择 | 第一版 20 run 完成；D/E 组合及单独消融待验证 |
| [BALANCE_DAGGER](BALANCE_DAGGER.md) | QwenOFT DAgger 标签修复短交接、十轮结果及终点验收待办 | 活跃交接；详版历史在 archive/DAGGER |
| [LATENCY_EVAL_UNIFICATION](LATENCY_EVAL_UNIFICATION.md) | 四任务 evaluator、sampler/worker、seed/reset 合同与撤回方案 | 保留合同记录；实现验证与正式训练分别记录 |
| [LATENCY_RL_HANDOFF.zh-CN](LATENCY_RL_HANDOFF.zh-CN.md) | 四任务 H1 中文交接、最新诊断结论与旧 H40 smoke | 活跃交接；不是英文 Balance-only 文档的逐字重复，历史段落保留日期 |
| [parkour/OPTIMIZE](parkour/OPTIMIZE.md) | CPU 调度收益、预算对齐及已回退的 PhysX/cache 实验 | 保留性能记录；训练吞吐与 VLA inference latency profile 分开解释 |

## 研究资料

| 文档 | 内容 / 主要用途 | 重叠与阅读关系 |
|---|---|---|
| [PROPOSAL](PROPOSAL.md) | 以 MIKASA 为核心的可解性→延迟退化→恢复研究论证 | 研究方针；与 HumanoidBench proposal 关注范围不同 |
| [HUMANOIDBENCH_PROPOSAL](HUMANOIDBENCH_PROPOSAL.md) | Balance/Run/Hurdle 的 teacher→VLA→profile→latency-aware 路线与 gates | 研究方针；不作为已完成实验清单 |
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

## 进度与空文件

| 文档 | 用途 | 状态 |
|---|---|---|
| [PROGRESS](PROGRESS.md) | 四主任务总览、四模型明细、评测数值、证据与下一步 | 相关工作确认结果后同步更新 |
| [balance/已排除](balance/已排除.md) | 空文件，占位用途尚未确定 | 暂时保留；没有可归纳的排除结论 |

初次梳理时的 `RUN.md`、`HURDLE.md` 在本轮开始前已不在工作区，本轮未恢复。
此前分别记录 Run history/recovery/DAgger 和 Hurdle tiny overfit；不提供失效文件链接。
