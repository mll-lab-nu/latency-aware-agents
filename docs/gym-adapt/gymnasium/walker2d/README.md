# Walker2D continuous-action adaptation

Open [`index.html`](index.html) for the Walker2D adaptation record. The task
uses the generic `env.name: gymnasium` path and preserves the native
six-dimensional continuous torque space.

The zero-latency APPO run using the vendored Sample Factory MuJoCo recipe
completed 10,006,528 frames on H200. Its best checkpoint reached a training
statistic of 4,212.219 and was evaluated for ten episodes at each of latency 0
through 4. L0 achieved 3,743.507 mean return, while one or more frames of
latency caused the zero-latency policy to fail quickly. The six native torque
values remain unchanged through OpenVLA training and evaluation. With online
observations aligned to the model's saved image size, the 5k continuous-projector
QwenVLA checkpoint reaches 983.915 mean return and 469.5 mean steps at L0; 4 of
20 episodes reach the full 1,000-step horizon. The report uses this post-fix L0
result directly and does not retain the superseded evaluation comparison.
