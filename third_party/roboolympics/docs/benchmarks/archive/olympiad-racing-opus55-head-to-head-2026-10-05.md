# 三维赛车：Opus 5.5 与 Astra 对打

2026-10-05。用户要求的实际环境试赛，每项一场；Opus 固定 A 席，Astra 固定 B 席。

Opus 使用 Kangrui 的 Claude 订阅 token，通过 Claude Code 2.1.289 调用 `claude-opus-5-5`；Astra 使用 Azure，通过 Codex 0.158.0 调用 `gpt-6-astra`，medium reasoning，禁用网页搜索。双方都经过可信计量代理，玩家只获得占位凭据。

配置沿用 seed 1 / x10 / FIRST_PLAYER=A，PLAYER_MODE_A/B=model，duel=True。赛车每场两轮、交换内外起跑位；空气曲棍球四回合、双方各发球两次。两席自己编写、执行、修改视觉控制程序，比赛中世界持续推进，赛间 review 冻结。测试期间没有修改规则、物理或裁判。

## 结果

胜者：**astra**。结束原因 `race_over`。

计分记录：

```json
{
  "A": {
    "completed_laps": 2,
    "distance": 74.04112606227554,
    "finish_time": 20.561564352015083
  },
  "B": {
    "completed_laps": 2,
    "distance": 74.04112606227554,
    "finish_time": 19.23117887911635
  }
}
```

逐轮记录：

```json
[
  {
    "heat": 1,
    "progress": {
      "A": 37.02056303113777,
      "B": 37.02056303113777
    },
    "finished": {
      "A": 11.164428637375332,
      "B": 9.373919889507434
    },
    "steps": 559,
    "outer_grid": "A"
  },
  {
    "heat": 2,
    "progress": {
      "A": 37.02056303113777,
      "B": 37.02056303113777
    },
    "finished": {
      "A": 9.397135714639749,
      "B": 9.857258989608916
    },
    "steps": 493,
    "outer_grid": "B"
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
  "n_actions": 1532,
  "sim_steps": 10520,
  "steps": 1052,
  "A_won": 0,
  "B_won": 1,
  "draw": 0,
  "won": 0,
  "race_over": 1,
  "ready_forfeit": 0,
  "start_forfeit": 0,
  "video_rendered": 1,
  "car_contact_steps": 0,
  "transient_api_errors": 0,
  "unrecovered_api_errors": 0,
  "A_completed_laps": 2,
  "A_distance": 74.04112606227554,
  "A_finish_time": 20.561564352015083,
  "A_wall_contact_steps": 0,
  "A_refused_busy": 0,
  "B_completed_laps": 2,
  "B_distance": 74.04112606227554,
  "B_finish_time": 19.23117887911635,
  "B_wall_contact_steps": 0,
  "B_refused_busy": 0,
  "sim_late_max_s": 0.41382891801185906,
  "sim_late_mean_s": 0.0007062347359271868,
  "sim_effective_slowdown": 10.000009744633042
}
```

| 席位 | 模型 | 完成请求 | 输入 token | 缓存输入 | 输出 token | 失败请求 |
|---|---|---:|---:|---:|---:|---:|
| A | opus5.5 | 26 | 1242405 | 1175693 | 19924 | 0 |
| B | astra | 40 | 809903 | 778733 | 5463 | 0 |

独立裁判在固定 CPU 镜像中进行两次新进程重放；有效比赛与胜负分别记录。代理元数据已核对实际模型和路由。

[重放视频](/data/guichi/olympiad_review/sports_opus55_head_to_head_2026-10-05/racing-opus55-A-astra-B/verifier/replay.mp4)。[完整结果与用量](/data/guichi/olympiad_review/sports_opus55_head_to_head_2026-10-05/summary.json)。双方程序与日志在同场 `players/A`、`players/B`，精确源码在归档的 `tested_source/`。复制前扫描真实凭据值，不归档 .env 或 Harbor 启动配置。

一场、一个 seed 且未交换模型席位，只用于模型兼容性和行为观察，不能作为稳定能力排名。Claude 与 Codex 使用不同 harness，本次也不是严格的同 harness 模型比较。

两轮均无车车接触或碰墙计数。首轮 Opus 11.164s / Astra 9.374s；次轮 Opus 9.397s / Astra 9.857s。总用时 Astra 19.231s / Opus 20.562s。
