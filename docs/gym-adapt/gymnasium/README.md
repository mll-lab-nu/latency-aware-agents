# Gymnasium adaptation examples

These reports are evidence-backed records for tasks adapted through the
separate `env.name: gymnasium` path.

Completed GR00T reports: [AirRaid](air_raid/index.html#gr00t),
[InvertedPendulum H1 / 5 FPS](inverted_pendulum/index.html#gr00t),
[Hopper H1 / 5 FPS](hopper/index.html#gr00t-results),
[Walker2D H1 / 5 FPS](walker2d/index.html#gr00t-results), and
[Ant H1 / 5 FPS](ant/index.html#gr00t-results).
Each environment report is updated after verified completion or a scientific gate
stops its profile branch, with pinned model/data references.
[HalfCheetah GR00T 5 FPS](half_cheetah/index.html#gr00t) stopped at its teacher gate;
B20 is complete, including the documented recovery after a container restart.
[Humanoid GR00T H1 / 5 FPS](humanoid/index.html#gr00t-results) stopped at the
collection limit: 16 qualified episodes in 120 attempts, below the target of 100.
Its L0/P/B, teacher selection, E10 and initial probe completed; replay, profile VLA
and F were not run. The earlier native H8 / 15 FPS experiments are terminated;
their preserved results are labeled as historical within the task reports.

| Task | Native action space | Verified stages | Report |
| --- | --- | --- | --- |
| Inverted Pendulum | Native `Box(1)` force; five-force policy interface | Completed OpenVLA 10 FPS and GR00T H1 / 5 FPS pipelines; measured B/F and paired-seed audit | [Open report](inverted_pendulum/index.html) |
| Pendulum | Continuous `Box(1)` torque | Probe, wrapper, L1 APPO training, small-model evaluation | [Open report](pendulum/index.html) |
| Lunar Lander | Native `Discrete(4)` engines | Probe, direct L1 APPO training, small-model evaluation | [Open report](lunar_lander/index.html) |
| Hopper | Native continuous `Box(3)` torques + 11D state | Completed GR00T 5 FPS B/F: 8.55 ± 0.09 → 242.55 ± 127.57, 20 matched-seed gains; historical OpenVLA record preserved | [Open report](hopper/index.html) |
| Walker2D | Native continuous `Box(6)` torques + 17D state | Completed GR00T 5 FPS B/F: 63.24 ± 6.58 → 852.87 ± 357.10, 20 matched-seed gains; historical OpenVLA record preserved | [Open report](walker2d/index.html) |
| HalfCheetah | Native continuous `Box(6)` torques + 17D state | Historical OpenVLA B/F complete; GR00T 5 FPS L0/P complete, profile teacher gate failed (0/20 eligible); B20 −262.47 ± 74.94, recovery audited | [Open report](half_cheetah/index.html) |
| Ant | Native continuous `Box(8)` torques + 27D state | Completed GR00T 5 FPS B/F: 165.37 ± 224.74 → 1955.97 ± 780.51, 19 matched-seed gains and 1 loss; historical OpenVLA record preserved | [Open report](ant/index.html) |
| Swimmer | Native continuous `Box(2)` torques + 8D state | MuJoCo 3 compatibility wrapper, 4-frame L0 APPO training, best-checkpoint L0-L4 evaluation | [Open report](swimmer/index.html) |
| Humanoid | Native continuous `Box(17)` torques + 376D state | GR00T 5 FPS L0/P/B complete; B20 1158.63 ± 646.19; profile collection stopped at 16/120 qualified attempts, target 100; historical OpenVLA evidence preserved | [Open report](humanoid/index.html) |
| AirRaid | Native `Discrete(6)` Atari actions | Completed OpenVLA and GR00T pipelines; measured B/F, APPO teacher and data verification | [Open report](air_raid/index.html) |
| Asterix | Native `Discrete(9)` Atari actions | Probe, random-noop CHW wrapper, NTSC L1 APPO training, small-model evaluation | [Open report](asterix/index.html) |
| MyoChallenge 2025 Table Tennis P2 | Native continuous `Box(275)` muscle/actuator controls | 100 Hz wrapper, two APPO smoke checkpoints, latency sweeps, trace replay, teacher export, native LeRobot conversion | [Open report](myochallenge_table_tennis_p2/index.html) |

## GR00T H1 / 5 FPS — six-environment conclusion

RTX 3090 measured-only B/F; 20 episodes per completed evaluation, matched seeds
42–61, cap 1,000. Values are return mean ± population SD. The teacher eligibility
threshold is not a VLA acceptance threshold; realized latency is reported separately
in each task report. These rows describe each environment, not a cross-task ranking.

| Task | B: zero-trained VLA | F: profile-trained VLA | Conclusion |
| --- | ---: | ---: | --- |
| [InvertedPendulum](inverted_pendulum/index.html#gr00t) | 15.55 ± 4.79 | 1000.00 ± 0.00 | Complete; 20 wins, all F episodes reach cap |
| [Hopper](hopper/index.html#gr00t-results) | 8.55 ± 0.09 | 242.55 ± 127.57 | Complete; 20 wins |
| [Walker2D](walker2d/index.html#gr00t-results) | 63.24 ± 6.58 | 852.87 ± 357.10 | Complete; 20 wins |
| [Ant](ant/index.html#gr00t-results) | 165.37 ± 224.74 | 1955.97 ± 780.51 | Complete; 19 wins, 1 loss |
| [HalfCheetah](half_cheetah/index.html#gr00t) | −262.47 ± 74.94 | Not run | Teacher gate: Final and best each 0/20 above 3000 |
| [Humanoid](humanoid/index.html#gr00t-results) | 1158.63 ± 646.19 | Not run | Collection ceiling: 16 qualified / 120 attempts, target 100 |

HalfCheetah B preserves 16 complete pre-restart episodes plus four remaining
seeds; some metadata for the first 16 episodes is unavailable, as documented in
its report. No completed episodes were repeated. Four profile VLA evaluations
completed; two profile branches concluded at their stated scientific gates.
All six task reports retain historical OpenVLA evidence under its original conditions.

Each report separates runtime API facts from task-author decisions, records the
frozen environment contract, links generated files and commands, and labels
unexecuted rollout/VLA stages as `NOT RUN`.

The normative implementation instructions and agent prompt are in the
[Gymnasium adaptation guide](../../gymnasium_adapter.md).
