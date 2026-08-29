# HalfCheetah adaptation

Open [`index.html`](index.html) for the verified HalfCheetah-v4 native continuous-action record.
The adaptation uses a task-owned wrapper around the native six-dimensional torque space and
trains its zero-latency small model with the vendored Sample Factory MuJoCo recipe. The completed
RGB+state path includes 100 teacher episodes, native LeRobot conversion, 5k-step QwenOFT training,
and L0/L2 latency-bench evaluation of the final checkpoint.
