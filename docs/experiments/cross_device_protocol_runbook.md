# Cross-Device 协议 sim2real gap 诊断报告

诊断时间：2026-07-04。数据来源：`results/cross_device/_hf_raw/raw/<device>/instances/<instance_id>/cross_device/`
（exp2 同机 sim-vs-realtime，openvla-flappy，n=100 episodes/侧）。

## 结论 1：大部分"gap"其实是评测噪声，不是模拟误差

对 exp2 的每台机器计算 return 差的显著性（episode return 的 sd≈160，n=100，
两组均值差的标准误 ≈23 return ≈ **8-9%**）：

| 机器 | gap | z 值 | 判定 |
|---|---|---|---|
| 4090 | +0.27% | +0.03 | 噪声 |
| 5090 | +0.99% | +0.11 | 噪声 |
| 3090 | +2.6% | +0.32 | 噪声 |
| **A100** | **-8.0%** | **-1.08** | **不显著，很可能就是噪声** |
| **L40S** | **+26.6%** | **+1.87** | **唯一接近显著的真 gap** |

n=100 下该协议的分辨率下限约 ±10%——低于这个量级的"gap 差异"不需要机制解释。
**最优先要修的是统计功效：加 episode 数，或在表里带置信区间。**

## 结论 2：L40S 是真 gap，主因是"推理延迟的热漂移 × 零余量工作点"，被 iid sim 抹平

因果链（每一步都有数据支撑，见下）：

1. **推理延迟随运行时间漂移（热/时钟行为）**：episode 均值 inference 延迟从冷机
   ~39.0ms（ep 0-19）爬升到 43.5ms（ep 40-49），之后在 41.3~43.5ms 震荡——
   典型 GPU 时钟/热稳态形态。全程漂移范围 38.6 → 44.9ms（±8%）。
2. **工作点没有余量**：标定选出的 env_fps=38 正好是 worker 利用率 cap=0.8 的上限
   （所有设备都是 `selected_env_fps == max_env_fps`，被 cap 而非 ideal 决定）。
   延迟从 38.6 漂到 44.9ms 意味着利用率 0.73 → 0.85。
3. **利用率逼近 1 时掉帧爆炸**：per-episode 相关性 corr(延迟, 掉帧率)=**0.68**；
   低延迟 episode 掉帧≈0%，高延迟 episode 掉帧 2~7%（最坏 17.2%）。
4. **掉帧毁掉 return**：corr(延迟, return)=**-0.53**，corr(掉帧, return)=**-0.50**。
   高延迟 episode 的 return 是 4~102，低延迟 episode 可达 217~445。
5. **iid sim 抹掉了这一切**：sim 每个 episode 都从同一个 pooled 分布采样 →
   利用率恒定 0.78 → 掉帧均匀地低（0.69% vs real 1.9%）→ 没有灾难 episode →
   sim 乐观 26%。sim 的 episode 均值延迟 sd 只有 0.20ms（real 是 1.86ms，9 倍差）。

对比数字：real 49/100 个 episode 的 p99 > 2 帧（sim 11 个）；real 0.42% 的动作
延迟 3 帧而 profile CDF 的 P(>2帧) 只有 0.0006——超额部分是高利用率下的排队聚簇。

### bench_overhead 500ms 尖峰的定位：parent 进程 GC 停顿，罕见且非主因（已修复，待重测验证）

> 根因已确认为 CPython gen-2 GC：日志记录全程攒在 parent 内存（run 结束才落盘），
> 堆越大 gen-2 停顿越长。指纹：停顿间隔几何倍增（×1.8~2.5）、时长随记录数线性增长
> （132→978ms）、五台机器全部一致、每次成对卡住 worker→parent 与 env→parent 两条管道。
> 修复（2026-07-04）：TrajectoryLogger 每 episode 增量落盘并清空内存；realtime 三进程
> episode 内禁用自动 GC、episode 边界显式 collect+freeze。验证方式：重跑 measure，
> `realtime_pipeline_profile.jsonl` 里 `worker_to_parent_ipc_ms`/`env_to_parent_ipc_ms`
> 的 max 不应再出现"间隔倍增、幅度递增"的尖峰序列。

`bench_overhead_ms` 本身**非常平稳**：均值 1.85ms，118k 条记录里只有 **7 条**
超过 20ms。max=497.7ms 来自整场 52 分钟运行中约 4 次"parent 进程停顿"事件：

- 尖峰**成对出现在相邻 obs step**：`worker_to_parent_ipc` 飙 289ms 的下一两步，
  `env_to_parent_ipc` 飙 276ms（三组事件都是同一模式：484/276、289/276、182/167、
  132/119）。这两个计时器都是"发送方打点 → parent drain 管道的时刻"，天然包含
  **接收方（parent）被调度前的排队等待**。一次 parent 停顿（OS 调度/GC/共享宿主
  机争抢，vast.ai 环境）会同时被两条管道各记一次——同一事件，看到两次。
- 这不是测量 bug：动作确实晚了这么久才到 env（对应实测里零星 7~21 帧的 apply
  延迟离群点），计入 e2e profile 是忠实的。只是 stage 名字叫 "IPC" 有误导性，
  实为 parent 调度延迟。
- 但它**不是 gap 主因**：7 个事件解释不了 ~500 个 3 帧动作和 26% 的 return 差，
  主因是上面的热漂移链条。

## 结论 3：五台机器的 sim 输入不同质（协议执行层面的混杂）

从 manifest 和 sim 侧记录的延迟统计对出来的：

- 五次 sim 用了三个不同的 git 版本（l40s 最老 6/17 `4e79a451`，3090 `e308dfdb`，
  其余 `227e5808`）。
- **3090 的 sim 回放的是 inference 分布而不是 e2e**（sim 延迟均值 73.6 == infer
  均值，≠ e2e 的 76.2）。
- **A100 的 profile 是旧 schema**（`semi_markov_latency_frames`），burst 样本的
  `action_ready_ms` 全是 90.9090...ms == 恰好 2×frame_ms——是从帧量化后的
  ready_time 构建的，sim 回放常数 2 帧延迟。这里行为上近似中性（A100 的延迟本来
  就全落在 (1,2] 帧），但是个实打实的 artifact bug。
- 4090/5090 用了正确的连续 e2e 分布——恰好也是 gap<1% 的两台。
- 注意：4090/5090/l40s 现存 profile 没有 burst model 且 `latency_method=None`，
  现在的 master 代码 `profile_default` 会直接 raise——这些机器需要重跑 measure
  （profile 重建脚本已随协议精简一并删除）。

## 已排除的假设

- **wall-clock 健康问题（H3）**：所有机器 actual_fps≈target、deadline miss≈0、
  lag p95 < 8ms → 排除。
- **分布层面的 gross mismatch**：各机 sim 的延迟均值/方差与其 profile 基本一致
  → 排除（问题在时序结构，不在边际分布）。

## 建议的行动顺序

1. 先修统计功效：exp2/exp3 表加 SE/置信区间（`cross_device_protocol_tables.py`），
   或把 n_episodes 提到 300+。
2. 统一重建全部 profile（同一代码、`latency_method=temporal`、连续 action_ready），
   然后同一 git 重跑五台机器的 sim-same——预期 3090/A100 的残余 gap 消失。
3. L40S 类机器：确保用 temporal burst 模型；如果重建后 gap 仍在，需要在 burst
   模型里加跨 episode 的非平稳状态（现在 Markov 链每 episode 重置，坏时段会被洗掉）。
4. 复核 exp1 里 L40S inference 39ms 比 5090 的 49ms 还快这个反常识数据点——可能
   那台 vast 实例本身不典型。
