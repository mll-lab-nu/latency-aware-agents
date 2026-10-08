# 训练中周期 latency_bench eval

`configs/base/flappy.yaml`、`configs/base/demon_attack.yaml`、`configs/base/deadly_corridor.yaml` 已经包含周期 latency_bench eval 默认字段。

完整可运行示例配置在：

```bash
configs/examples/flappy_periodic_latency_bench_eval.yaml
```

启动方式不变：

```bash
python -m training.train --config configs/examples/flappy_periodic_latency_bench_eval.yaml
```

`configs/train/` 存放本地生成的配置，不属于发布内容。长期保存的配方应使用上面的 `configs/examples` 示例，或继承 `configs/base/<env>.yaml`。

关键字段：

```yaml
evaluation:
  eval_interval_steps: 1000000
  eval_episodes: 20
  eval_parallel_envs: 100
  eval_suites:
    fixed:
    - [0, 15]
    normal:
    - [2.0, 1.0, 0, 15]
    - [4.0, 2.0, 0, 15]
    uniform:
    - [0, 5]
```

语义：

- `eval_interval_steps`: 训练中每隔多少 env steps 跑一次 latency_bench eval。删掉这个字段就回到原来的纯训练路径。
- `fixed: [[0, 15]]`: inclusive sweep，也就是 eval latency raw frame `0..15`。
- `fixed: [[0, 1, 2, 3, 4]]`: explicit sweep，只跑列表中写出的 raw frame latency。
- `normal: [[mean, std, min, max], ...]`: 可以写多组 normal distribution，单位是 raw frames，运行时会按当前 env 的 `env_fps` 转成 ms。
- `uniform: [[min, max], ...]`: 可以写多组 uniform distribution，单位同样是 raw frames。
- final eval 不需要配置，训练结束后默认会再跑一次。
- periodic eval 在后台进程执行；训练 loop 负责保存 checkpoint、提交 eval job、轮询结果。若上一轮 eval 还没结束，新的周期 eval 会进入无界 FIFO 队列，按提交顺序依次执行；final eval 会先等待正在运行和队列中的 periodic eval 全部结束，再跑最终 checkpoint。

输出位置：

```text
<logging.output_dir>/latency_bench_train_eval/step_<env_steps>/
<logging.output_dir>/latency_bench_train_eval/step_<env_steps>_final/
```

每个 suite 会有自己的子目录，例如：

```text
fixed_0_to_15/
normal_mean_2_std_1_min_0_max_15/
uniform_min_0_max_5/
```

wandb 记录使用普通 scalar，不再使用 custom chart。metric key 形如：

```text
eval/fixed/latency_0/mean_return
eval/fixed/latency_0/std_return
eval/fixed/latency_0/mean_length
eval/fixed/latency_0/completed_episodes
eval/normal/normal_mean_2_std_1_min_0_max_15/mean_return
eval/uniform/uniform_min_0_max_5/mean_return
```

每次 eval payload 都会写 `global_step` 和兼容字段 `eval/global_steps`，`wandb.define_metric("eval/*", step_metric="global_step")` 会把这些普通 scalar 绑定到 Sample Factory env steps。W&B 网页端手动切 X axis 时选择 `global_step`。

周期 eval 使用 `configs/base/<env>.yaml` 补齐 latency_bench 需要的 `executor`、`policy`、env action map 等字段，再用当前 train config 覆盖训练 run 的 env/backend/evaluation/logging 设置。
