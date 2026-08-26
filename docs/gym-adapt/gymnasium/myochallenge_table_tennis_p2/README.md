# MyoChallenge 2025 Table Tennis P2 adaptation

Open [`index.html`](index.html) for the evidence-backed adaptation report.

This integration preserves the official P2 control interval: MuJoCo advances
five 2 ms simulation steps per environment step, so both `env_fps` and
`obs_fps` are 100 Hz. The benchmark delays complete native `Box(275)` action
vectors rather than reducing the task to a discrete action catalog.

The implementation has been smoke-tested through Sample Factory training,
closed-loop latency sweeps, deterministic action-trace replay, canonical
teacher export, and native 275-dimensional LeRobot conversion. The two smoke
policies were trained for only 4,608 collected frames; their returns are
pipeline evidence, not evidence that latency-aware training improves task
performance. Continuous StarVLA registration and VLA fine-tuning remain
`NOT RUN`; this release deliberately emits no StarVLA robot type or mixture for
the continuous dataset.
