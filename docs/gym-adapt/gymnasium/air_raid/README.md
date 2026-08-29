# AirRaid agent-assisted adaptation

Open [`index.html`](index.html) for the complete AirRaid release record. It
documents the PAL raw-frame contract, native six-action catalog, four-frame CHW
policy observation, zero-latency 10M-frame Atari APPO training, seeded
random-noop + FireReset evaluation, 100 gated teacher episodes, native six-action
LeRobot conversion, 5k-step QwenOFT training, and final L0/L2 latency-bench results.

The current rerun contract is encoded in
[`scripts/bash_scripts/rollout_data/air_raid.sh`](../../../../scripts/bash_scripts/rollout_data/air_raid.sh):
teacher rollout uses random-noop 30 plus FireReset, 3,600 raw frames, and
`context_window=0`. The small-model and VLA evaluation configs also use a
3,600-raw-frame episode budget. The HTML report and synchronized evaluation
assets now use this contract.

The final VLA checkpoint reaches mean return 2,000.0 at L0 and 681.25 at L2
over 20 episodes per latency. The corresponding teacher dataset and W&B run are
linked from the HTML report.
