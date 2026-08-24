# HumanoidBench H1Hand Balance Hard adaptation

Open [`index.html`](index.html) for the evidence-backed adaptation report.

This integration keeps HumanoidBench's native privileged 177-value state and
normalized 61-value position-control action space at 50 Hz. A task-owned
wrapper applies the narrow MuJoCo 3.3 named-index compatibility patch and
converts Sample Factory's Python action list to the `float32` array required by
HumanoidBench.

The APPO run is deliberately limited to 20,480 collected environment steps. It
verifies training, checkpoint creation, deterministic loading, and zero-latency
evaluation; it does not establish convergence. VLA training and online StarVLA
evaluation are `NOT RUN`.
