# Balance：固定均值 prefix 的因果评测

本页记录 2026-09-20 已完成的补评测。公共实现与新训练合同见
[LATENCY_EVAL_UNIFICATION](LATENCY_EVAL_UNIFICATION.md#固定-profile-均值-prefix)。

本协议复用已训练的 `zero_full` / `profile_conditioned` checkpoint，只改变
evaluation。正式结果以 [PROGRESS](PROGRESS.md) 和
[实验报告](../outputs/tasks/humanoidbench_balance_simple/causal_profile_eval_20260920/REPORT.md) 为准。

## 模型输入与执行合同

- `none`：不向模型传入 `action_prefix` 或 `action_prefix_mask`，用于 zero-trained baseline。
- `profile_mean`：从冻结 profile 的 `summary.mean_ms` 计算
  `K = ceil(mean_ms * env_fps / 1000)`，一次评测内不变。
- 50Hz 下，QwenOFT / QwenPI v3 / QwenGR00T 的原始均值分别为
  37.88078026420736 / 99.17626147712788 / 60.013397574146865ms，K 为 2 / 5 / 4。
- prefix 仅由请求开始时已经返回的 scheduler 动作生成；不采样未来延迟，
  不推进未返回结果的 timeline。启动沿用现有 noop，chunk 耗尽沿用 hold-last。
- 实际 latency 仍由原 temporal profile 采样。新结果到达即按 observation timestamp
  选择 head；不人为等待 K，不按实际延迟回写 conditioning mask。
- 保留原有物理 state、旧动作 chunk 和当前位置。QwenOFT 不消费显式 prefix；
  其两臂对照反映原有 action-context 模型的训练差异，不能称为 flow prefix 机制对照。
- `d<K` 时 flow 模型新 chunk 的剩余 prefix 仍为旧动作；`d>K` 时跳过过期 heads，
  因低估延迟带来的衔接误差是本协议要测量的行为，不做 oracle 修正。

## 独立入口

在配好依赖的仓库根目录运行；`--environment` 必须指向对应模型原先的
profile-conditioned 环境配置，baseline 也使用同一份 profile 环境。

```bash
CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
  python -m scripts.starvla.evaluate \
  --bundle /path/to/zero_full/bundles/dagger_10_5000 \
  --environment /path/to/profile_conditioned/environment.json \
  --output /path/to/new-eval/baseline \
  --task balance --prefix-mode none \
  --vla-python /path/to/vla/bin/python \
  --backbone /path/to/Qwen3-VL-4B-Instruct
```

conditioned 臂换成对应 bundle，使用 `--prefix-mode profile_mean --prefix-profile /path/to/profile.json` 和新输出目录。
统一入口从 bundle 合同读取 horizon 和动作维度；不再按模型名称推断。
开发集验证加 `--seed-start 810001 --episodes 2`；正式默认 seeds 为
`820001–820064`，64 episodes。每组输出目录必须是新目录，避免混合旧结果。

## 固定协议与证据

50/50Hz、capacity 1、1000-frame cap；profile seed 271828、inference seed
941731、batch 8、16 rollout workers、每次一个 GPU inference worker。
使用 RTX 3090 observation-to-action profile 作为模拟部署延迟；执行评测的 H200
墙钟耗时不替代该 profile。不同模型的 profile 和 horizon 不同，不构成同延迟的模型排名。

统一入口的 `provenance.json` 保存 bundle 和已解析 `action_prefix`（mode、profile、原始均值、FPS、K）；`environment.json` 保存实际配置；
`episodes.json` 保存每个 seed 的结果、应用 head 计数、underflow 和逐请求
`request_audit`。其中实际延迟只在请求提交后的诊断记录中出现，不参与构造模型输入。
`summary.json` 保存指标标准差和 `d<K / d=K / d>K` 的请求数。

比较按相同环境 seed 配对，使用 10,000 次 bootstrap、随机种子 42，报告 mean
length / return gain 的 95% 区间。区间只反映这些 checkpoint 的环境 seed 变动，
不反映跨 training seeds 的不确定性；不根据 held-out 结果挑 K 或 checkpoint。

历史 oracle-delay 结果仅作对照，新协议需要重跑两臂。训练、模型发布、新协议评测和
性能结论分别报告；本轮不训练、不重新发布模型。
