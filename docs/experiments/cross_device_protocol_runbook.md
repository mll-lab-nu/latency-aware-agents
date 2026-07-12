# Cross-Device Sim2Real Runbook

## 实验模型

目标 GPU 上的 real measurement、由其拟合出的 latency profile，以及引用该 profile 的 simulation 是三个独立 artifact：

```text
raw session -> profile revision -> simulation seed
```

Real session 数量和 simulation 数量不属于存储身份。同一台 physical target、同一 workload 下可以持续追加 real sessions；任一 profile revision 可以选择任意 temporal/IID latency seeds。Simulation host 只影响运行成本，不进入路径或结果身份。

当前 workload config 是：

```text
configs/experiments/cross_device.yaml
```

`workload_id=openvla-flappy` 表示一套不可变的 policy、checkpoint、environment 和基础运行语义。上述语义改变时使用新的 workload ID，不覆盖旧 artifact。

## 数据布局

本地 `results/cross_device/` 与 HF dataset main 使用相同相对布局。下面列出消费者依赖的关键文件，运行时还会保留 generated config、hardware、summary 和诊断 artifact：

```text
raw/<device>/<instance>/<workload>/
  calibration.json
  calibration.generated.yaml
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
    ...
    DONE

profiles/<device>/<instance>/<workload>/<profile_revision>/
  profile.json
  e2e_latencies.jsonl
  infer_latencies.jsonl
  latency_distribution.json
  latency_burst_model.json
  hardware.json
  ...
  DONE

sim/<device>/<instance>/<workload>/<profile_revision>/<temporal|iid>/<latency_seed>/
  generated.yaml
  episode_metrics.jsonl
  ...
  DONE
```

`instance` 默认由 GPU UUID 生成，也可通过 `CROSS_DEVICE_INSTANCE_ID` 显式覆盖。同型号的 physical instances 不会自动合并。

## 一键快速流程

```bash
bash start.sh
```

默认选择 session 0、profile revision `fit-0`、temporal seeds 0/1/2，并依次执行 measure、publish raw、fit、publish profile、sim、publish sim。

这些值只控制本次调用，不会进入共同目录名：

```bash
SESSIONS="0 1 2" \
FIT_SESSIONS="0 1" \
PROFILE_REVISION=fit-0-1 \
METHOD=iid \
LATENCY_SEEDS="4 7 9" \
bash start.sh
```

## 分步运行

所有命令都可通过 `cross_device.sh` 运行；脚本负责激活 OpenVLA conda environment。

### 1. 测量并发布任意 real sessions

```bash
bash cross_device.sh measure --sessions 0 1 2 --resume
bash cross_device.sh publish raw --sessions 0 1 2 --resume
```

第一次测量会创建 target/workload 的 calibration。以后追加 sessions 会复用同一 calibration，并在每个 session 中保存其快照。

### 2. 从显式 sessions 拟合 profile revision

```bash
TARGET_REF=$(bash cross_device.sh target-ref)

bash cross_device.sh fit \
  --target-ref "$TARGET_REF" \
  --profile-revision fit-0-1 \
  --sessions 0 1 \
  --resume

bash cross_device.sh publish profile \
  --profile-ref "$TARGET_REF/fit-0-1" \
  --resume
```

Profile revision 是不可变 leaf。增加 fit sessions 时创建新 revision，例如保留 `fit-0`，再创建 `fit-0-4`。

### 3. 在任意主机追加 simulation seeds

```bash
PROFILE_REF=a100/instance_01b95985/openvla-flappy/fit-0-1

bash cross_device.sh sim \
  --profile-ref "$PROFILE_REF" \
  --method temporal \
  --latency-seeds 0 2 7 \
  --resume

bash cross_device.sh publish sim \
  --profile-ref "$PROFILE_REF" \
  --method temporal \
  --latency-seeds 0 2 7 \
  --resume
```

IID 使用完全相同的命令，只将 `--method` 改为 `iid`。

### 4. 显式选择需要聚合的数据

```bash
bash aggregate.sh \
  --profile-ref a100/instance_01b95985/openvla-flappy/fit-0-4 \
  --profile-ref l40s/instance_ce58e2a5/openvla-flappy/fit-0-4 \
  --real-sessions 5 6 7 8 9 \
  --methods temporal iid \
  --latency-seeds 0 1 2 3 4
```

默认输出：

```text
results/cross_device/analysis/
  latency_samples.csv
  latency_profile_summary.csv
  real_vs_sim.csv
```

Real 以 session mean 为统计单位，simulation 以 latency-seed mean 为统计单位。Evaluation sessions 全部属于 fit sessions 时标记 `in_sample`，完全不相交时为 `heldout`，部分重叠时为 `mixed`。

## Resume 与 Publish

- Calibration、raw session、profile revision 和 simulation seed 各自只有 leaf completion marker。
- 不存在 measurement、method 或 study-level `DONE`，因为 sessions 和 seeds 永远可以继续追加。
- `--resume` 只跳过已完成的请求 leaf；无 `DONE` 的 partial leaf 会清空后重跑，避免 JSONL 混合。
- Publish 先上传 payload，最后上传 `DONE`。
- 已完成的远端 leaf 默认禁止覆盖；`--resume` 会跳过它。
- 不存在 overwrite、latest profile、自动 seed 分配、manifest/hash identity 或 simulation-host key。

HF tag `v3` 保存迁移前的完整旧布局；当前 main 只用于 target-first artifact 布局。
