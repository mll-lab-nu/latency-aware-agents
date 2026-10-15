# 空气曲棍球：实现与模型试赛验收

> **已作废（revision `airhockey-program-v1`）。** 这些试赛用的是 2026-10-05 修复前的规则（4 回合、持球超时判分、允许平局、A/B 球槌接触参数不对称、20 ms 控制步）。修复后的 `airhockey-program-v2`（USAA 规则，见 [olympiad-airhockey.md](olympiad-airhockey.md)）的结果不能与这里的结果混合统计。Opus 5.5 那场没有经过标准 `run_duel.sh`，也没有禁用 WebSearch/WebFetch，属于无效配置。

日期：2026-10-05。分支：`olympiad-airhockey`。实现范围、许可证、规则偏离和信息边界见 [任务说明](olympiad-airhockey.md)。当前适合作为 preview；这份报告不是正式排行榜或完整难度校准。

用户要求先测三维赛车再开发下一项。赛车已先完成 OpenRouter `openai/gpt-6-luna` 与 Azure `gpt-6-astra` 两条实测，再开始空气曲棍球。本项也用相同模型/提供方做了实测，采用原生 Codex 0.158.0、medium reasoning，禁用网页搜索。两场均为 L1、seed 101、slowdown x10；真实凭证只进入可信 sidecar，玩家使用占位凭证经过计量代理。

## 结果

| 试验 | 配置 | 比分 A:B | A 进球 | 有效 | 两次独立重放 |
|---|---|---:|---:|---:|---:|
| 相机参考选手 | L1 / 101 / x1 | 4:0 | 1 | 1 | 一致 |
| 相机参考选手 | L2 / 212 / x2 | 0:2 | 0 | 1 | 一致 |
| GPT-6-Luna / OpenRouter | L1 / 101 / x10 | 3:1 | 0 | 1 | 一致 |
| GPT-6-Astra / Azure | L1 / 101 / x10 | 4:0 | 0 | 1 | 一致 |
| 双人相机参考选手 | seed 2 / x10 / B 先发 | 0:0 | 0 | 1 | 一致 |
| 最终双人产物检查 | seed 1 / x1 / A 先发（诊断） | 0:1 | 0 | 1 | 一致 |
| nop 未开始 | L1 / 101 / x20 | 弃权给 B | 0 | 1 | 一致 |

单人 `success=1` 要求有效获胜，因此参考选手 L2 的有效失败和 nop 的有效弃权均为 success=0。双人 success 只表示有效竞技试验；具体胜负看 A_won / B_won / draw。

还有一次双人测试启动配置错误：误用 `RCB_PLAYER_MODE_A/B`，正确变量为 `PLAYER_MODE_A/B`。两席没有模型计量而被拒绝 start，停止该试验后以正确参数重跑。原试验 `hockey-duel-oracle-x10-B` 留档为 technical_failure=1、meter_error=1，不纳入竞技表现；正确重跑名称为 `hockey-duel-oracle-x10-B-fixed`。未隐去技术失败。

## 模型行为与限制

Luna 第一回合持球超时，第二回合对手超时，第三回合已能把球送进对方半场并迫使对手超时，第四回合为对手发球超时。模型编写了 RGB 红色检测、公开标定投影、速度估计和击球程序，并在回合间修改程序。第一与第三回合均为 A 发球但种子产生的初始 y 不同，不能把得分改善直接归因于修改。

Astra 四回合均通过对手超时得分，也编写并运行视觉控制程序。两场模型没有进球；4:0 不代表 Astra 已经掌握精准射门，也不能从一场 L1 试赛推导模型排名。参考选手的真实进球证明目标可达，但尚不是高难度任务解法证明。

L1 对手守在球门中心但不主动发球，存在两次容易获得的对手超时分；玩家仍需处理自己的两次发球才能赢。L2–L4 是主动脚本，当前参考策略离线多次超时平局，L2 实际 Docker 试验则 0:2 失利。因此正式首发榜单前需要改善对手、扩展种子、校准时限，并验证足够多的进球与攻守转换。当前代码实现和模型通道已经可用；竞技难度和得分构成仍需实验。

## 提供方与用量

| 实测通道 | 已完成模型请求 | 输入 token（含缓存） | 缓存输入 token | 输出 token | 失败模型请求 / 限流重试 |
|---|---:|---:|---:|---:|---:|
| OpenRouter / openai/gpt-6-luna | 33 | 873,503 | 828,967 | 15,428 | 0 / 0 |
| Azure / gpt-6-astra | 35 | 833,002 | 799,913 | 4,588 | 0 / 0 |

代理日志确认实际提供方和模型。Azure 日志记录了 `gpt-6-astra` 与 `gpt-6-astra-2` 部署；沿用已有故障转移机制。两场独立 verifier 均判定 proxy_error=0、meter_error=0；不将 token 数换算成未核实的费用。用量是本次任务运行量，不是训练或 leaderboard 总成本。

## 验证

五项性质测试通过：动作的非有限数/布尔/错半场拒绝、完整球进门与撞柱区分、真实边轨反弹、力矩受限的机械臂跟踪、四次交替发球重放与篡改拒绝。完整记录在两个新进程重放相同，改分数不再匹配，改 source hash 被拒绝。CPU-only OSMesa 在固定镜像中验证两相机不同，物理移动改变 RGB。

生成一致性检查、compileall、协议/隔离合同检查通过；两个任务分别通过已有 program-play、Azure mock failover 与 API 错误分类检查（ALL OK）。CI 仅报告两项未加入共享 `tasks/dataset.toml`，按添加运动工作流留给集成；没有更改其他 agent 的任务或共享注册表。

L1 x1 有效 slowdown 1.000014；L2 x2 为 2.000009；Luna x10 为 10.000014；Astra x10 为 10.000012。比赛未使用 GPU，正式验证使用固定 MuJoCo 3.3.7 / NumPy 2.2.6 Docker 镜像。完整 MuJoCo 状态、控制器、RNG、接触计数和裁判结果一致，而不只是比分相同。

模型测试期间单人 task 和 physics 源码保持不变，归档源码与轨迹 source hash 一致。双人首场暴露了继承的 artifact 配置问题：`/app/output/` 错从空闲 main 收集，已改为分别从 seat_a / seat_b 收集到 players/A / players/B。保留合同要求的 main `/app/output/` 条目并在空闲 main 创建空目录，实际玩家数据单独收集。原场核心轨迹和 verifier 仍完整；最终配置 x1 双人诊断检查通过，0:1，有效重放，players/A 与 players/B 的日志均已实际收集（不属于官方 slowdown）。Harbor 的来源重叠校验仍对同名源路径发出警告，但 host destination 不同，manifest 中两席 status=ok；这些玩家日志不参与物理裁判。本项没有赛后换物理再冒充原模型试赛。

## 证据与复现

持久证据目录：`/data/guichi/olympiad_review/airhockey_validation_2026-10-05/`。包含各试验的独立 reward、动作轨迹、重放视频、玩家程序和去掉正文的 API 路由元数据，另含原测试源码/机器人 XML 与 summary。复制前按当前真实凭证值扫描内容；不归档 `.env`、真实凭证或 Harbor 配置。

在相同体育镜像中执行 `python tested_source/replay.py <trial>/verifier/trajectory.json /tmp/replay.json` 可验证记录。实际任务可用 Harbor 显式路径运行；参考选手设 `RCB_PLAYER_MODE=oracle`，双人则设 `PLAYER_MODE_A=oracle PLAYER_MODE_B=oracle`，分别指定 `--seat A=oracle --seat B=oracle`。

待集成项是 manifest、排名聚合、广播与网站登记。本报告未声称网站已部署或新项目已纳入正式榜单。


后续用户要求的 Luna/OpenRouter 与 Astra/Azure 直接对打，另见 [模型对打报告](olympiad-airhockey-head-to-head-2026-10-05.md)。两类试赛分别统计。
