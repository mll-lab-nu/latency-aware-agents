# 空气曲棍球：Opus 5.5 与 Astra 对打

> **已作废（revision `airhockey-program-v1`）。** 这些试赛用的是 2026-10-05 修复前的规则（4 回合、持球超时判分、允许平局、A/B 球槌接触参数不对称、20 ms 控制步）。修复后的 `airhockey-program-v2`（USAA 规则，见 [olympiad-airhockey.md](olympiad-airhockey.md)）的结果不能与这里的结果混合统计。Opus 5.5 那场没有经过标准 `run_duel.sh`，也没有禁用 WebSearch/WebFetch，属于无效配置。

2026-10-05。用户要求的实际环境试赛，每项一场；Opus 固定 A 席，Astra 固定 B 席。

Opus 使用 Kangrui 的 Claude 订阅 token，通过 Claude Code 2.1.289 调用 `claude-opus-5-5`；Astra 使用 Azure，通过 Codex 0.158.0 调用 `gpt-6-astra`，medium reasoning，禁用网页搜索。双方都经过可信计量代理，玩家只获得占位凭据。

配置沿用 seed 1 / x10 / FIRST_PLAYER=A，PLAYER_MODE_A/B=model，duel=True。赛车每场两轮、交换内外起跑位；空气曲棍球四回合、双方各发球两次。两席自己编写、执行、修改视觉控制程序，比赛中世界持续推进，赛间 review 冻结。测试期间没有修改规则、物理或裁判。

## 结果

胜者：**draw**。结束原因 `match_over`。

计分记录：

```json
{
  "A": {
    "points": 0,
    "goals": 0,
    "possession_faults": 0
  },
  "B": {
    "points": 0,
    "goals": 0,
    "possession_faults": 0
  }
}
```

逐轮记录：

```json
[
  {
    "rally": 1,
    "winner": null,
    "reason": "rally_timeout",
    "steps": 400,
    "serve": "A"
  },
  {
    "rally": 2,
    "winner": null,
    "reason": "rally_timeout",
    "steps": 400,
    "serve": "B"
  },
  {
    "rally": 3,
    "winner": null,
    "reason": "rally_timeout",
    "steps": 400,
    "serve": "A"
  },
  {
    "rally": 4,
    "winner": null,
    "reason": "rally_timeout",
    "steps": 400,
    "serve": "B"
  }
]
```

## 验证与用量

```json
{
  "success": 1,
  "valid_trial": 1,
  "technical_failure": 0,
  "deterministic": 1,
  "replay_error": 0,
  "meter_error": 0,
  "proxy_error": 0,
  "grader_error": 0,
  "missing_log": 0,
  "official_config": 1,
  "level": 1,
  "seed": 1,
  "slowdown": 10.0,
  "n_actions": 2396,
  "sim_steps": 32000,
  "steps": 1600,
  "A_won": 0,
  "B_won": 0,
  "draw": 1,
  "won": 0,
  "match_over": 1,
  "ready_forfeit": 0,
  "start_forfeit": 0,
  "video_rendered": 1,
  "mallet_contact_steps": 73,
  "rail_contact_steps": 7,
  "transient_api_errors": 0,
  "unrecovered_api_errors": 0,
  "A_points": 0,
  "A_goals": 0,
  "A_possession_faults": 0,
  "A_refused_busy": 0,
  "B_points": 0,
  "B_goals": 0,
  "B_possession_faults": 0,
  "B_refused_busy": 0,
  "sim_late_max_s": 0.33979392307810485,
  "sim_late_mean_s": 0.0003882655927736778,
  "sim_effective_slowdown": 10.000013727221813
}
```

| 席位 | 模型 | 完成请求 | 输入 token | 缓存输入 | 输出 token | 失败请求 |
|---|---|---:|---:|---:|---:|---:|
| A | opus5.5 | 33 | 1658614 | 1610226 | 19720 | 0 |
| B | astra | 46 | 1288761 | 1217734 | 8751 | 0 |

独立裁判在固定 CPU 镜像中进行两次新进程重放；有效比赛与胜负分别记录。代理元数据已核对实际模型和路由。

[重放视频](/data/guichi/olympiad_review/sports_opus55_head_to_head_2026-10-05/airhockey-opus55-A-astra-B/verifier/replay.mp4)。[完整结果与用量](/data/guichi/olympiad_review/sports_opus55_head_to_head_2026-10-05/summary.json)。双方程序与日志在同场 `players/A`、`players/B`，精确源码在归档的 `tested_source/`。复制前扫描真实凭据值，不归档 .env 或 Harbor 启动配置。

一场、一个 seed 且未交换模型席位，只用于模型兼容性和行为观察，不能作为稳定能力排名。Claude 与 Codex 使用不同 harness，本次也不是严格的同 harness 模型比较。

本任务仍为 preview。真实进球、持球超时分及无分回合超时必须分开解释；不能把有效重放等同于足够丰富的竞技行为。

本场记录到 73 个球槌接触步、7 个边墙接触步；四回合均为无分超时，没有进球，也没有持球超时罚分。
