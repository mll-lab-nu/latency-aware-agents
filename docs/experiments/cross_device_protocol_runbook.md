# Cross-Device Sim2Real Runbook

稳定入口统一位于 `scripts/experiments/sim2real/`。脚本用途、论文固定矩阵、11 个 hardware
ID 以及底层 protocol 的逐命令说明见
[`scripts/experiments/sim2real/README.md`](../../scripts/experiments/sim2real/README.md)。

## 实验模型

目标 GPU 上的 real measurement、由其拟合出的 latency profile，以及引用该 profile 的
simulation 是三个独立 artifact：

```text
raw session -> profile revision -> simulation seed
```

Real session 数量和 simulation 数量不属于存储身份。同一台 physical target、同一 workload
下可以继续追加 real sessions；任一 profile revision 可以选择任意 temporal、IID、fixed 或
normal latency seeds。Simulation host 只影响运行成本，不进入路径或结果身份。

两个论文 workload 配置为：

```text
configs/experiments/cross_device.yaml
  workload_id: openvla-flappy
  lifecycle: consecutive

configs/experiments/cross_device_isolated_30s.yaml
  workload_id: openvla-flappy-isolated-30s
  lifecycle: isolated
  cooldown_s: 30
```

Workload ID 表示不可变的 policy、checkpoint、environment 和基础运行语义。上述语义改变时
使用新的 workload ID，不覆盖旧 artifact。

## 数据布局

本地 `results/cross_device/` 与 Hugging Face dataset main 使用相同相对布局。下面列出后续
消费者依赖的关键文件；运行时还会保留 generated config、hardware、summary 和诊断 artifact：

```text
raw/<device>/<instance>/<workload>/
  calibration.json
  hardware.json
  calibration.DONE
  sessions/<session_id>/
    calibration.json
    hardware.json
    generated.yaml
    episode_metrics.jsonl
    e2e_latencies.jsonl
    infer_latencies.jsonl
    actions.jsonl
    steps.jsonl
    warmup/episode_<episode_id>/  # isolated lifecycle only
    DONE

profiles/<device>/<instance>/<workload>/<profile_revision>/
  profile.json
  e2e_latencies.jsonl
  infer_latencies.jsonl
  latency_distribution.json
  latency_burst_model.json
  hardware.json
  DONE

sim/<device>/<instance>/<workload>/<profile_revision>/<temporal|iid|fixed|normal>/<latency_seed>/
  generated.yaml
  episode_metrics.jsonl
  DONE
```

`instance` 默认由 GPU UUID 生成，也可通过 `CROSS_DEVICE_INSTANCE_ID` 显式指定。同型号的
physical instances 不会自动合并。

本地测量目录还会保存 `calibration.generated.yaml` 和 calibration probe；远端 calibration
leaf 只发布后续消费者需要的 calibration、hardware 与 completion marker。

## 论文完整流程

所有命令可从任意工作目录运行。以下示例假定当前位于仓库根目录。

### 1. 在目标硬件上测量

对论文的五个目标 hardware ID（`rtx3090`、`rtx4090`、`rtx5090`、`l40s`、
`a100-sxm4-80gb`），分别在对应物理主机运行：

```bash
bash scripts/experiments/sim2real/measure_consecutive.sh <hardware>
bash scripts/experiments/sim2real/measure_isolated_30s.sh <hardware>
```

Consecutive 脚本采集并发布 sessions 0--4，再拟合并发布 `fit-0-4`。Isolated 脚本采集并
发布 session 0，再拟合并发布 `fit-0`。两个脚本都依次执行 measure、publish raw、fit 和
publish profile。

默认 consecutive lifecycle 只创建并 warm up 一次 realtime runtime，然后连续运行 session
内全部 episodes。Isolated lifecycle 对每个 episode（包括 episode 0）执行：

```text
关闭 inference workers、模型 CUDA context 和 realtime runtime
-> 等待 30 秒
-> 重建 runtime
-> stress warmup
-> 运行一个正式 episode
```

最后一个 episode 结束后只清理资源，不再等待。固定等待不轮询 GPU 温度、功耗或利用率，也
不控制外部 GPU 作业。两种 lifecycle 必须使用不同 workload ID，不能把 traces 拟合到同一
profile 中。

这两个测量入口还接受 config 中列出的其他六个明确 hardware ID；完整列表见脚本目录
README。Realtime executor 会使用所有可见 CUDA devices，测量前应只暴露属于目标机器的
GPU。

### 2. 运行论文 simulation 矩阵

```bash
bash scripts/experiments/sim2real/simulate.sh
```

该脚本固定使用论文的五个已测 instance refs：

- consecutive：`fit-0-4`，四种 methods，各 latency seeds `0 1 2 3 4`；
- isolated：`fit-0`，四种 methods，各 latency seed `0`。

总计 120 个任务。Simulation 主机必须提供 CUDA devices `0..7`；脚本将其固定分为
`0,1`、`2,3`、`4,5`、`6,7` 四个双卡 slot。每个任务成功后立即发布，单任务失败不停止其余
任务；全部任务结束后统一报告失败并返回非零。运行前，对应 profile revisions 必须已经发布到
配置指定的 Hugging Face repository。

### 3. 构建论文数据和图片

```bash
bash scripts/experiments/sim2real/build_figures.sh
```

该脚本按顺序运行 `paper/latency-sensitive-vla/exps/sim2real/build_data.py` 和 `plot.py`。
`build_data.py` 读取 `sources.json` 中的精确 refs；新的 physical instance 会得到新的 ID，不会
自动替换论文 refs。

## 底层 protocol 分步运行

`scripts/experiments/sim2real/cross_device.sh` 会激活 OpenVLA Conda environment，并把参数
传给 `python -m latency_bench.cross_device.protocol`。普通论文复现应使用上面的高层脚本；
以下命令用于手动运行、诊断和恢复单个阶段。

```bash
PROTOCOL=scripts/experiments/sim2real/cross_device.sh
CONFIG=configs/experiments/cross_device.yaml

TARGET_REF=$(bash "$PROTOCOL" target-ref \
  --config "$CONFIG" --hardware rtx4090)

bash "$PROTOCOL" measure \
  --config "$CONFIG" --hardware rtx4090 \
  --sessions 0 1 2 3 4 --resume

bash "$PROTOCOL" publish raw \
  --config "$CONFIG" --hardware rtx4090 \
  --sessions 0 1 2 3 4 --resume

bash "$PROTOCOL" fit \
  --config "$CONFIG" --target-ref "$TARGET_REF" \
  --profile-revision fit-0-4 --sessions 0 1 2 3 4 --resume

bash "$PROTOCOL" publish profile \
  --config "$CONFIG" --profile-ref "$TARGET_REF/fit-0-4" --resume

bash "$PROTOCOL" sim \
  --config "$CONFIG" --profile-ref "$TARGET_REF/fit-0-4" \
  --method temporal --latency-seeds 0 1 2 3 4 --resume

bash "$PROTOCOL" publish sim \
  --config "$CONFIG" --profile-ref "$TARGET_REF/fit-0-4" \
  --method temporal --latency-seeds 0 1 2 3 4 --resume
```

`fit` 读取本地 raw sessions。在另一台主机拟合之前，先显式下载：

```bash
bash "$PROTOCOL" download raw \
  --config "$CONFIG" --target-ref "$TARGET_REF" \
  --sessions 0 1 2 3 4
```

`sim` 默认从 Hugging Face 下载 profile。只有要使用已经完成但尚未发布的本地 profile 时，才
给 `sim` 加 `--local-profile`。

四种 simulation method 为：

- `temporal`：profile 的 burst model；
- `iid`：profile inverse-CDF distribution 的独立采样；
- `fixed`：profile summary 的 `mean_ms`；
- `normal`：profile summary 的 `mean_ms` 和总体标准差 `std_ms`，并沿用非负截零行为。

## 通用 aggregate 分析

`aggregate` 根据命令中显式指定的 profile refs、real sessions、methods 和 latency seeds，从
Hugging Face 下载相关 profiles、raw sessions 和 simulation leaves，再生成通用分析 CSV：

```bash
bash scripts/experiments/sim2real/cross_device.sh aggregate \
  --config configs/experiments/cross_device.yaml \
  --profile-ref a100-sxm4-80gb/instance_1d6030f8/openvla-flappy/fit-0-4 \
  --real-sessions 0 1 2 3 4 \
  --methods temporal iid fixed normal \
  --latency-seeds 0 1 2 3 4
```

多个 profile 通过重复 `--profile-ref` 指定。默认输出：

```text
results/cross_device/analysis/
  latency_samples.csv
  latency_profile_summary.csv
  real_vs_sim.csv
```

Real 以 session mean 为统计单位，simulation 以 latency-seed mean 为统计单位。Evaluation
sessions 全部属于 fit sessions 时标记 `in_sample`，完全不相交时为 `heldout`，部分重叠时为
`mixed`。

`aggregate` 不是论文绘图的前置阶段。论文的 `build_figures.sh` 直接使用 `sources.json ->
build_data.py -> plot.py`，不读取上述三个 CSV。

## Resume、Overwrite 与 Publish

- Calibration、raw session、profile revision 和 simulation seed 各自只有 leaf completion
  marker；不存在 measurement、method 或 study-level `DONE`，因为 sessions 和 seeds 可以继续
  追加。
- `--resume` 只跳过已完成的请求 leaf；无 `DONE` 的 partial 本地 leaf 会清空后重跑，避免
  JSONL 混合。
- Publish 先上传 payload，最后上传 `DONE`。已完成的远端 leaf 默认禁止覆盖；`--resume` 会
  跳过它。
- `download raw --overwrite` 明确替换本地选择。`publish profile --overwrite` 和
  `publish sim --overwrite` 明确替换远端选择，并与 publish 的 `--resume` 互斥。
- `--overwrite` 只作用于命令显式指定的 leaf；没有 latest profile、自动 seed 分配或
  simulation-host key。

HF tag `v3` 保存迁移前的完整旧布局；当前 main 只用于 target-first artifact 布局。
