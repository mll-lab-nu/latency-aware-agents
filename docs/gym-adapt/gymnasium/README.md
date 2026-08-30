# Gymnasium adaptation examples

These reports are evidence-backed records for tasks adapted through the
separate `env.name: gymnasium` path.

| Task | Native action space | Verified stages | Report |
| --- | --- | --- | --- |
| Inverted Pendulum | Continuous `Box(1)` force | Probe, wrapper, L1 APPO training, small-model evaluation | [Open report](inverted_pendulum/index.html) |
| Pendulum | Continuous `Box(1)` torque | Probe, wrapper, L1 APPO training, small-model evaluation | [Open report](pendulum/index.html) |
| Lunar Lander | Native `Discrete(4)` engines | Probe, direct L1 APPO training, small-model evaluation | [Open report](lunar_lander/index.html) |
| Hopper | Native continuous `Box(3)` torques + 11D state | L0 APPO/eval, 100 gated teacher episodes, RGB+state conversion, 5k QwenOFT and closed-loop failure diagnosis | [Open report](hopper/index.html) |
| Hopper Continuous | Native continuous `Box(3)` torques | L0 continuous APPO training and best-checkpoint evaluation | [Open report](hopper_continuous/index.html) |
| Walker2D | Native continuous `Box(6)` torques | Continuous APPO, L0–L4 evaluation, 7D OpenVLA carrier | [Open report](walker2d/index.html) |
| HalfCheetah | Native continuous `Box(6)` torques | L0 APPO, RGB+17D-state rollout/conversion, 5k QwenOFT, L0/L2 VLA evaluation | [Open report](half_cheetah/index.html) |
| Ant | Native continuous `Box(8)` torques + 27D state | EGL probe, L0 APPO 10M training, best-checkpoint L0-L4 evaluation | [Open report](ant/index.html) |
| Humanoid | Native continuous `Box(17)` torques + 376D state | Wrapper, configs, probe scaffold, and semantic tests; runtime stages not run | [Open report](humanoid/index.html) |
| AirRaid | Native `Discrete(6)` Atari actions | Probe, CHW wrapper, PAL L1 APPO training, original + random-noop evaluation | [Open report](air_raid/index.html) |
| Asterix | Native `Discrete(9)` Atari actions | Probe, random-noop CHW wrapper, NTSC L1 APPO training, small-model evaluation | [Open report](asterix/index.html) |
| MyoChallenge 2025 Table Tennis P2 | Native continuous `Box(275)` muscle/actuator controls | 100 Hz wrapper, two APPO smoke checkpoints, latency sweeps, trace replay, teacher export, native LeRobot conversion | [Open report](myochallenge_table_tennis_p2/index.html) |

Each report separates runtime API facts from task-author decisions, records the
frozen environment contract, links generated files and commands, and labels
unexecuted rollout/VLA stages as `NOT RUN`.

The normative implementation instructions and agent prompt are in the
[Gymnasium adaptation guide](../../gymnasium_adapter.md).
