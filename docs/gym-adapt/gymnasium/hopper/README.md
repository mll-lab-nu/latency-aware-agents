# Hopper RGB+state adaptation

Open [`index.html`](index.html) for the canonical Hopper-v4 adaptation record.
The current path uses an independent task-owned wrapper, the native three-dimensional
continuous torque space, an ordered 11-dimensional state contract, L0 Sample Factory
training, L0-L4 latency-bench evaluation, 100 accepted L0 teacher episodes,
native RGB+11D-state LeRobot conversion, and a 5k-step QwenOFT run. The final
text-bin VLA checkpoint failed closed-loop evaluation (L0 mean return 173.136;
L2 mean return 9.513). A controlled 5k-step continuous-state projector rerun
raises those means to 1,419.935 and 315.890, respectively, while remaining well
below the same-seed L0 teacher. The report preserves the original failure
diagnosis and separates representation gains from the remaining distribution gap.
