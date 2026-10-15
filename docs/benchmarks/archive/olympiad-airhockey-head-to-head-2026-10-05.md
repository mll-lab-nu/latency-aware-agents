# 空气曲棍球：Luna 与 Astra 直接对打

> **已作废（revision `airhockey-program-v1`）。** 这些试赛用的是 2026-10-05 修复前的规则（4 回合、持球超时判分、允许平局、A/B 球槌接触参数不对称、20 ms 控制步）。修复后的 `airhockey-program-v2`（USAA 规则，见 [olympiad-airhockey.md](olympiad-airhockey.md)）的结果不能与这里的结果混合统计。Opus 5.5 那场没有经过标准 `run_duel.sh`，也没有禁用 WebSearch/WebFetch，属于无效配置。

2026-10-05。这是用户要求的模型对模型试赛，双方均为隔离的模型玩家，不是模型与脚本选手比赛。

Luna 经 OpenRouter 调用 `openai/gpt-6-luna`，Astra 经 Azure 调用 `gpt-6-astra`。两席使用原生 Codex 0.158.0、medium reasoning、禁用网页搜索，分别编写、运行和修改自己的视觉控制程序。可信代理保存真实凭证；玩家不获得对手程序、模拟器或隐藏状态。

每项两场，第二场交换模型席位。两场均为 DUEL_SEED=1、RCB_SLOWDOWN=10、FIRST_PLAYER=A；duel=True，PLAYER_MODE_A/B=model，因此不执行单人模式的脚本对手。比赛开始前等待双方 start，赛间为冻结 review；计算时比赛持续推进。源码在四场测试期间冻结。

赛车每场两轮并交换内外起跑位置；空气曲棍球每场四回合、各两次发球。交换模型席位进一步平衡 A/B 视角和初始几何。采用同一 seed 保持场地/发球配置一致，但模型生成与实际动作时序并非相同。

## 完整比赛结果

| 场次 | A 席 | B 席 | Luna 成绩 | Astra 成绩 | 胜者 | 有效 / 独立重放 |
|---|---|---|---|---|---|---|
| 1 | Luna | Astra | 0 分；0 进球 | 0 分；0 进球 | 平局 | 1 / 1 |
| 2 | Astra | Luna | 1 分；0 进球 | 2 分；0 进球 | Astra | 1 / 1 |

空气曲棍球只比较四回合总分。真实进球与持球超时得分分别报告；0:0 是有效平局，不是技术失败。单回合八秒无分结束不额外给任何一方分。

## 分轮／回合结果

| 场次 | 回合 | 发球模型 | 得分模型 | 原因 |
|---|---|---|---|---|
| 1 | 1 | Luna | 无 | 回合超时，无分 |
| 1 | 2 | Astra | 无 | 回合超时，无分 |
| 1 | 3 | Luna | 无 | 回合超时，无分 |
| 1 | 4 | Astra | 无 | 回合超时，无分 |
| 2 | 1 | Astra | Luna | 持球超时 |
| 2 | 2 | Luna | Astra | 持球超时 |
| 2 | 3 | Astra | 无 | 回合超时，无分 |
| 2 | 4 | Luna | Astra | 持球超时 |

## 技术有效性与用量

| 场次 | 席位／模型 | 完成请求 | 输入 token（含缓存） | 缓存输入 | 输出 token | 失败请求／限流重试 |
|---|---|---:|---:|---:|---:|---|
| 1 | A / Luna | 69 | 3,176,404 | 3,097,412 | 29,575 | 0 / 0 |
| 1 | B / Astra | 48 | 1,317,700 | 1,275,086 | 8,826 | 0 / 0 |
| 2 | A / Astra | 49 | 1,246,133 | 1,209,538 | 6,862 | 0 / 0 |
| 2 | B / Luna | 64 | 2,162,438 | 2,108,959 | 24,533 | 0 / 0 |

每场 final、独立 verifier 与 provider metadata 均已检查。独立裁判在固定 MuJoCo 3.3.7 / NumPy 2.2.6 CPU 镜像中启动两次新进程重放，要求完整状态、裁判和得分一致；不只依赖 final.json 的标志。玩家 A/B 的源码和日志分别收集到 players/A、players/B。

- 场 1：结束原因 `match_over`，valid_trial=1，deterministic=1，proxy_error=0，meter_error=0，technical_failure=0。接触 tick=48，实际 slowdown=10.000009。
- 场 2：结束原因 `match_over`，valid_trial=1，deterministic=1，proxy_error=0，meter_error=0，technical_failure=0。接触 tick=19，实际 slowdown=10.000011。

## 证据与解释边界

持久证据：`/data/guichi/olympiad_review/sports_head_to_head_2026-10-05/`，包含四场的 replay.mp4、可信动作记录、独立 reward、双方程序/日志、计量和去正文的路由元数据。`summary.json` 明确模型与席位映射，`tested_source/` 保留精确测试源码。未归档 .env、真实凭证或 Harbor 配置，复制前按已配置凭证值检查内容。

每项只有两场、一个种子，是直接对打的兼容性和行为试验，不构成稳定能力排名。比分不能与上一轮脚本对手试赛混算。比赛中的程序修改、起跑位置和随机生成差异也不能单独解释为模型改进的因果证据。空气曲棍球仍为 preview，其进球率、时限和对手策略覆盖需要进一步校准。没有改变比赛规则来追求非平局。

网站、共享 manifest 和榜单登记仍待集成；此报告不代表线上已经展示这些比赛。
