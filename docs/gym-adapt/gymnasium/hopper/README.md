# Hopper RGB+state adaptation

Open [`index.html`](index.html) for the canonical Hopper-v4 adaptation record.
The current path uses an independent task-owned wrapper, the native three-dimensional
continuous torque space, an ordered 11-dimensional state contract, L0 Sample Factory
training, L0-L4 latency-bench evaluation, 100 accepted L0 teacher episodes,
native RGB+11D-state LeRobot conversion, and a 5k-step QwenOFT run. The final
VLA checkpoint failed closed-loop evaluation (L0 mean return 173.136; L2 mean
return 9.513); the report attributes the failure to behavior-cloning covariate
shift using same-seed controls and teacher-prefix interventions.
