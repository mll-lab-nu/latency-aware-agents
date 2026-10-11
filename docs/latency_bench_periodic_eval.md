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
executor:
  parallel_envs: 100
policy:
  deterministic: true
evaluation:
  eval_interval_steps: 1000000
  episodes: 20
  suites:
    - name: zero
      latency: {method: zero}
    - name: fixed_100ms
      latency: {method: fixed, fixed_latency_ms: 100}
    - name: normal_100ms
      latency:
        method: normal
        mean_latency_ms: 100
        std_latency_ms: 20
        min_latency_ms: 0
        max_latency_ms: 500
        seed: 42
```

每个 suite 声明名称和完整的模拟延迟。延迟字段与单次运行相同，单位为 ms。
`episodes` 表示每个条件的局数。`executor.parallel_envs` 表示同时运行的环境数量。
`policy.deterministic` 控制动作选择。

`eval_interval_steps` 指定周期评估间隔。删除该字段即可停用周期评估。
训练结束后默认再评估一次。周期评估使用后台进程，任务按 FIFO 顺序执行。
最终评估等待已提交的周期评估完成后运行。

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
