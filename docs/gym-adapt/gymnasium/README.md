# Gymnasium adaptation examples

These reports are evidence-backed records for tasks adapted through the
separate `env.name: gymnasium` path.

| Task | Native action space | Verified stages | Report |
| --- | --- | --- | --- |
| Inverted Pendulum | Continuous `Box(1)` force | Probe, wrapper, L1 APPO training, small-model evaluation | [Open report](inverted_pendulum/index.html) |
| Pendulum | Continuous `Box(1)` torque | Probe, wrapper, L1 APPO training, small-model evaluation | [Open report](pendulum/index.html) |
| Lunar Lander | Native `Discrete(4)` engines | Probe, direct L1 APPO training, small-model evaluation | [Open report](lunar_lander/index.html) |
| Hopper | Continuous `Box(3)` torques | Probe, 27-action wrapper, L1 APPO training, small-model evaluation | [Open report](hopper/index.html) |
| AirRaid | Native `Discrete(6)` Atari actions | Probe, CHW wrapper, PAL L1 APPO training, original + random-noop evaluation | [Open report](air_raid/index.html) |
| Asterix | Native `Discrete(9)` Atari actions | Probe, random-noop CHW wrapper, NTSC L1 APPO training, small-model evaluation | [Open report](asterix/index.html) |

Each report separates runtime API facts from task-author decisions, records the
frozen environment contract, links generated files and commands, and labels
unexecuted rollout/VLA stages as `NOT RUN`.

The normative implementation instructions and agent prompt are in the
[Gymnasium adaptation guide](../../gymnasium_adapter.md).
