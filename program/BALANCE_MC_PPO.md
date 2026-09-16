# Balance Simple：Sample Factory MC PPO

当前实验结论、checkpoint 和验收状态见 [PROGRESS](PROGRESS.md)。
配方为 [ppo_train.yaml](../configs/examples/gymnasium/g1_sonic_balance_simple/ppo_train.yaml)。
这条路径从已有 BC 初始化；from-scratch 和其他任务迁移尚未验证。

## 训练与评测

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m training.train \
  --config configs/examples/gymnasium/g1_sonic_balance_simple/ppo_train.yaml \
  --run-name balance_mcppo_sf_s1

CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m training.train \
  --config configs/examples/gymnasium/g1_sonic_balance_simple/ppo_train.yaml \
  --run-name balance_mcppo_sf_s1 --restart-behavior resume

CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m training.train \
  --config configs/examples/gymnasium/g1_sonic_balance_simple/ppo_train.yaml \
  --run-name balance_mcppo_sf_s1 --mode eval --checkpoint-kind best
```

最后一条将 `best` 换成 `latest` 可评测训练终点。评测写入新的
`eval_<kind>_<timestamp>/result.json`。正式训练完成后自动评测 BC、best、latest，
写入实验目录的 `result.json`；启用 W&B 时同步 heldout 汇总。
checkpoint 在 `checkpoint_p0/`，round JSON 保留每集种子、存活帧数、回报、KL、
更新次数和耗时。继续训练可提高 YAML 的 `training.ppo.rounds` 后 resume；
已完成预算的 resume 不再采样。

## 算法与所有权

- [根适配器](../training/gymnasium_sf/integration.py)负责配置和原有 latency/task wrapper。
  原独立训练脚本已删除。
- [SF mc_ppo](../third_party/sample-factory/sample_factory/algo/mc_ppo/)负责模型、完整集采样、
  MC 更新和固定种子评测；复用 SF runner、inference worker、共享 buffer、batcher、
  parameter server 和 checkpoint 生命周期。
- 每轮动态分配128个 episode seed，每个 episode 占一个最多200 decision 的 buffer；
  提前终止后的 padding 由 valid mask 排除。整轮只有一个行为策略，收齐后才更新。
  推理请求可以动态组成 batch，模型内部固定 GEMM64，避免改变 BC 的数值协议。
- 固定100ms、capacity1、50Hz；每5个真实帧决策一次。H16 中实际生效的 heads5:10
  使用 std0.02 的 pre-tanh Gaussian。训练奖励为实际存活帧数/1000，完整集
  gamma=1 MC return；物理观察 normalization 冻结，value 与 actor 参数独立。
- round0 仅训练 value；round1–200 更新 PPO。整轮优势归一化；每次 actor step 后
  计算整轮的精确联合 Gaussian KL，超过0.01时恢复 actor 和 Adam，并停止本轮
  actor 更新，value 仍完成预算。actor LR1e-7、value LR3e-4，2 epochs、minibatch1024。
  `kl_control: early_stop` 和 `normalize_advantages: false` 用于留存的对照实验。
- 每10轮用固定32集 development 选 best；BC 初始化也参与 best 比较。
  新 heldout64集只用于最终比较。KL 上限约束局部策略变化，不保证任务回报单调。

checkpoint 保存 round、actor/value 更新数、训练帧数/集数、两套 Adam、模型、
Torch/CUDA RNG 和 best 选择状态。种子按 round 分配，batch 按 seed 排序，
恢复从完整轮边界继续；checkpoint 文件按更新数和帧数的数值排序。

探索噪声绑定 episode seed，避免 worker 完成顺序改变采样。旧脚本使用共享
Torch 噪声流，因此旧训练轨迹不作逐步复现承诺；算法比较使用同一 SF 采样器下
的 control 与 candidate。冻结 BC 的逐集存活长度已与旧采样器核对一致。

## 验证证据

本轮脚本、配置、源代码快照、profile、三次重复的吞吐和训练对照汇总放在
[mcppo_sf_audit](../outputs/tasks/humanoidbench_balance_simple/mcppo_sf_audit/)。
`throughput_comparison.json` 比较固定 BC 的相同128集；吞吐不包含启动、评测和学习，
也不代表 VLA inference latency。`resume_comparison.json` 对比真实环境的中断续训与
连续训练。长期回归测试见 [test_gymnasium_ppo.py](../tests/training/test_gymnasium_ppo.py)。
