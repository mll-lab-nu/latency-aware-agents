# Cross-Device Sim2Real Study Runbook

## 实验问题

目标设备只负责测量真实延迟与闭环表现。测得的 latency profile 随后可以在任意计算主机上重放，验证 simulation 是否能够预测目标设备的真实表现。

Simulation host 只影响运行成本和墙钟时间，不属于实验变量。结果按 target device、target instance 和 latency method 标识，不再区分 same-device/cross-device replay，也不再维护 replay-pair panel。

当前保留两个 study：

| 配置 | Real sessions | Simulation seeds | 用途 |
|---|---:|---:|---|
| `cross_device_1real_3sim.yaml` | fit/evaluation 都是 session 0 | 3 | 快速 in-sample calibration |
| `cross_device_10real_20sim.yaml` | fit 0-4，evaluation 5-9 | 20 | 正式 held-out sim2real study |

两个 study 都使用 OpenVLA + Flappy、100 episodes/run，并比较 temporal 与 IID latency sampling。

## 数据模型

每个物理 GPU 由 `device_id/instance_id` 唯一标识。`instance_id` 默认由 GPU UUID 生成，也可通过 `CROSS_DEVICE_INSTANCE_ID` 显式指定。同型号的多个 instance 始终作为独立结果，不自动池化或选择最新项。

```text
<output_root>/
  measure/<device>/<instance>/
    calibration.json
    calibration.DONE
    sessions/<session_id>/
      generated.yaml
      episode_metrics.jsonl
      e2e_latencies.jsonl
      infer_latencies.jsonl
      actions.jsonl
      steps.jsonl
      DONE
    profile/
      profile.json
      e2e_latencies.jsonl
      infer_latencies.jsonl
      latency_burst_model.json
      DONE
    DONE
  sim/<device>/<instance>/<temporal|iid>/
    <latency_seed>/
      generated.yaml
      episode_metrics.jsonl
      DONE
    DONE
  analysis/
    latency_samples.csv
    latency_profile_summary.csv
    real_vs_sim.csv
```

HF dataset 中使用相同布局，前缀为 `cross_device/<study_id>/`。

## Real1 一键流程

```bash
bash start.sh
```

默认使用 `cross_device_1real_3sim.yaml` 和 temporal sampler，依次执行 measure、publish measurement、simulation、publish simulation。

可显式覆盖：

```bash
SPEC=configs/experiments/cross_device_1real_3sim.yaml METHOD=iid bash start.sh
```

real1 的同一个 session 同时用于 profile fitting 和 real evaluation，因此聚合结果标记为 `evaluation_scope=in_sample`，不能解释为 held-out prediction。

## 10real/20sim 正式流程

### 1. 每台目标 GPU 测量并发布

```bash
bash measure_10.sh
bash upload_10.sh
```

需要在 A100、L40S、RTX 4090、RTX 5090 上分别完成。`measure_10.sh` 只计算本地产物，`upload_10.sh` 显式发布。

### 2. 任意主机运行并发布 simulation

```bash
bash sim_20.sh
SPEC=configs/experiments/cross_device_10real_20sim.yaml \
  METHOD=temporal bash publish.sh sim

bash sim_iid_20.sh
SPEC=configs/experiments/cross_device_10real_20sim.yaml \
  METHOD=iid bash publish.sh sim
```

Simulation 会下载当前 study 已完成的 measurement profiles。可以在 RTX 5090 或其他主机执行；主机身份不会写入结果键。

### 3. 聚合

```bash
bash aggregate.sh
```

聚合命令下载整个远端 study，并要求每个配置设备至少有一个完成的 measurement instance。每个 instance 必须同时具有 temporal 和 IID simulation 结果。

## Resume 语义

`measure.sh` 和 `sim.sh` 默认传入 `--resume`。Resume 只信任细粒度 `DONE` marker：

- 已完成 calibration 直接读取 `calibration.json`；
- 已完成 real session、profile 或 simulation seed 直接跳过；
- 缺少 leaf `DONE` 的部分目录会删除后重跑；
- 所有 leaf 完成后才写 group-level `DONE`。

Resume 不读取 manifest、不计算 hash、不做 completion identity reconstruction，也不支持 overwrite。若实验配置、代码语义或数据应整体改变，必须使用新的 `study_id` 和 `output_root`。

## Publish 语义

Measure/sim 与 publish 是独立命令。发布时先上传数据目录，最后上传空的 `DONE`。消费者只读取带 `DONE` 的远端单元。

不要手工创建、复制或修改 `DONE`。Marker 与其目录内容不一致时，读取应直接失败，而不是尝试猜测或修复。

## 统计口径

- profile 只使用 fit sessions；temporal fitting 不跨 session boundary；
- real 指标只使用 evaluation sessions；
- 每个 real session 先对 episodes 求均值，再以 session means 计算统计量；
- simulation 每个 latency seed 是一个独立 replicate；
- real10 使用 session-level t intervals 和 real-vs-sim Welch interval；
- real1 只有一个 real session，real std/CI 和 difference CI 留空；
- `real_vs_sim.csv` 同时写 temporal 与 IID，temporal 是主结果，IID 是 sampler ablation；
- simulation host 和不同 physical instances 都不会被隐式合并。
