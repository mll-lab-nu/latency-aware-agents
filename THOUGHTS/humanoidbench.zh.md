# HumanoidBench G1 延迟研究

> 状态：Gate 1 仍在验证
>
> 最后更新：2026-08-28

## 结论

本仓库只维护面向延迟研究的 G1 adaptation，不维护其他 morphology 的原生
运行入口。G1 使用官方 SONIC 对齐的 29D body controller，并把 14D
hand command 作为独立流保留：

```text
HumanoidBench 官方 task / scene / reward
        ↓
G1 privileged teacher
        ↓
64D SONIC body token + 14D hand target = 78D command
        ↓
50 Hz frozen SONIC decoder → 29D body position target
        ↓
RGB + language + G1 proprioception → latency benchmark
        ↓
VLA chunk prediction and latency evaluation
```

当前主线先验证 reference playback 能否完成 Hurdle，再考虑学习 correction
或导出 VLA 数据。旧的 flat-action 和 selector 实验不再维护；29D residual
FastTD3 仅保留用于重现已有 gamma099 1M 配置和 checkpoint，不代表 Gate 1
会继续启动 RL。

## 当前 G1 合同

- 上游共 32 个 task；本 adaptation 固定支持 30 个，两个 High Bar task
  没有验证过的 G1 grasp contract，因此明确排除。
- task 使用官方 HumanoidBench 的 scene、task class、reset、reward 和
  termination；只替换 G1 robot、keyframe、controller 和必要的 observation
  / camera wiring。
- raw body contract 是 29D position target，带 hand 后为 43D；SONIC
  contract 是连续 `Box(78)`，前 64D 是 checkpoint-specific body token，后
  14D 是 hand target。
- `sonic_token_composed` 是当前 reference playback 和 gamma099 1M
  checkpoint 的必要 control mode，必须保留。它组合 reference token 与
  checkpoint 需要的 action contract，不引入额外兼容别名。
- decoder、G1 asset bundle 和 task contract 都固定 revision；缺失或不匹配
  的 metadata 直接失败，不静默改用其他 scene、robot 或 action 字段。

## 当前运行状态

### Gate 0：G1 contract 与 asset alignment

Gate 0 已覆盖 deterministic reset、action/observation shape、SONIC decoder
history、joint mapping、RGB render、单步执行和短程 rollout。对齐脚本只用于
检查官方 task 语义与 G1 candidate 的关系，不生成新的训练抽象。

标准入口：

```bash
python -m scripts.experiments.humanoidbench.alignment build-and-compare \
  --output-root results/humanoidbench_g1_alignment
```

### Gate 1：zero-residual reference teacher

Hurdle teacher 播放 `[run, obstacle_jump]` reference，在下一根杆前触发 jump，
落地后恢复到最近的 run phase。reference qpos 经固定 revision 的官方
SONIC encoder/FSQ 变为 64D token，hands 使用 neutral target；实际执行的
command 仍是完整 78D。

成功合同不是 reward proxy：必须越过十根杆、零杆碰撞、没有摔倒或非脚部
触地，并记录 body joint、root 姿态和 foot-contact phase 的 tracking error。
固定 seed 对 trigger distance 和 `jump_start_phase` 做 sweep。

当前结果：phase 30、约 1.5 m trigger 已证明 decoder 会执行起跳，pelvis
可升至约 1.07 m；右膝/右踝仍在第一根杆接触，尚无通过全部合同的 setting。
因此暂不启动 RL，也不导出 VLA 数据。

标准入口：

```bash
python -m scripts.experiments.humanoidbench.hurdle_teacher \
  --cache-dir results/humanoidbench_g1_sonic_assets \
  --jump-start-phase 30
```

## VLA 数据与 GR00T

reference teacher 通过 hard contract 后，才导出 head-camera RGB、language、
93D G1 proprioception 和实际执行的 29D absolute body target。quality metrics、
raw reward、command age 和 latency metadata 只用于过滤与分析。

官方 NVIDIA GR00T N1.7 使用独立的 `official_gr00t` 路径和 pinned
`third_party/Isaac-GR00T` source；现有 StarVLA `gr00t` 名称保持不变。第一版
`NEW_EMBODIMENT` 使用 93D state、40×29D body-target chunk 和官方 processor，
不复制模型代码或转换 checkpoint。

## Latency 语义

延迟在 VLA observation/inference boundary 注入，而不是 SONIC 50 Hz controller
内部：

```text
t0  capture RGB + proprioception; start VLA inference
t1  new chunk arrives; discard stale prefix
t2  execute remaining chunk at 50 Hz
```

报告 observation-to-chunk latency、command age、stale-prefix length、chunk
underflow、controller compute 和 task success。fallback action 必须在实验
配置中显式指定。

## 决策日志

| 日期 | 决策 | 原因 |
|---|---|---|
| 2026-08-28 | 只维护 G1 adaptation，支持 30 个 task | 研究目标是受控 latency benchmark，未验证 morphology contract 不进入主线 |
| 2026-08-28 | 保留 `sonic_token_composed` | reference playback 与 gamma099 1M checkpoint 仍依赖该 mode |
| 2026-08-27 | 先验证不可学习的 `[run, obstacle_jump]` teacher | 先把 motion/controller 可行性与 RL optimization 分开 |
| 2026-08-26 | 用 task-space 定义 G1 asset parity | 不同 morphology 不应按 joint index 强行复制 qpos |
| 2026-08-26 | task 使用官方 scene、class、reward、termination | 避免 simplified local scene 让结果无法解释 |
| 2026-08-28 | GR00T N1.7 使用独立 `official_gr00t` 集成 | 官方 processor、DiT 和 checkpoint contract 不能由旧 StarVLA 路径近似 |
| 2026-08-25 | 在 VLA chunk boundary 定义 latency | 保持 controller 高频执行并隔离 high-level delay |

## 暂不实现

- 不恢复已移除的其他 morphology 运行入口、配置或专用训练脚本。
- 不在 reference teacher 通过 hard contract 前增加 residual、selector、
  planner 或 PPO 特例。
- 不把 decoder 输出的 29D target 冒充 actor/replay 的 78D action。
- 不增加兼容 alias、候选路径探测或隐式 fallback。

## 下一步

1. 完成 Hurdle 的 phase/trigger sweep，并确认至少一个 setting 通过 hard
   success 与 tracking 合同。
2. 在通过后导出最小 teacher dataset，运行官方 GR00T open-loop 和 closed-loop
   smoke。
3. 仅根据 measured failure 决定是否加入受限 correction，再开始 latency
   sweep。
