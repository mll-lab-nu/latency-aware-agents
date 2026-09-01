# Walker2D continuous-action adaptation

Open [`index.html`](index.html) for the Walker2D adaptation record. The task
uses the generic `env.name: gymnasium` path and preserves the native
six-dimensional continuous torque space.

The zero-latency APPO run using the vendored Sample Factory MuJoCo recipe
completed 10,006,528 frames on H200. Its best checkpoint reached a training
statistic of 4,212.219 and was evaluated for ten episodes at each of latency 0
through 4. L0 achieved 3,743.507 mean return, while one or more frames of
latency caused the zero-latency policy to fail quickly. The six native torque
values remain unchanged through OpenVLA training and evaluation. The RGB+state
OpenVLA path is verified, but its return remains far below the small model. A
controlled continuous-state projector rerun produces a mixed result: L0 mean
return falls from 164.741 to 95.046, while L2 rises from 251.058 to 615.544.
The report preserves both 5k-step runs, their 40-episode evaluations, and the
latency-dependent reversal rather than presenting the projector as a general win.
