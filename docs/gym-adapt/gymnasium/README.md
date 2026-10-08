# Gymnasium results

[HTML overview](index.html).

These reports preserve each experiment's settings and results. Commands inside
reports may contain original machine paths. Prepare the corresponding published
artifacts and replace local paths before running them. Use the
[artifact guide](../../artifacts.md#prepare-a-backbone-for-evaluation) for portable inputs.
The task tables include experiments beyond the paper's selected benchmark.

| Environment | Model | Profile mean / P95 (ms) | L0 teacher → L0 | L0 VLA → L0 | L0 VLA → real-time (B) | Profile teacher → profile (E10) | Profile VLA → real-time (F) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| [InvertedPendulum](inverted_pendulum/index.html) | GR00T · Completed | 108.09 / 114.50 ms | — | 1,000.00 ± 0.00 | 15.55 ± 4.79 | 1,000.00 ± 0.00 | 1,000.00 ± 0.00 |
| [InvertedPendulum](inverted_pendulum/index.html) | OpenVLA · Completed | 67.09 / 71.26 ms | — | 707.15 ± 357.20 | 19.80 ± 6.21 | 791.90 ± 280.04 | 586.85 ± 457.53 |
| [InvertedPendulum](inverted_pendulum/index.html) | Pi0.5 · Completed | 141.77 / 148.17 ms | — | 1,000.00 ± 0.00 | 20.15 ± 6.48 | 1,000.00 ± 0.00 | 830.40 ± 339.20 |
| [AirRaid](air_raid/index.html) | GR00T · Completed | 81.49 / 97.82 ms | 2,557.50 ± 491.81 | 2,101.25 ± 667.01 | 733.75 ± 268.71 | 2,572.50 ± 259.43 | 1,586.25 ± 564.83 |
| [AirRaid](air_raid/index.html) | OpenVLA · Completed | 58.41 / 59.03 ms | 2,557.50 ± 491.81 | 2,000.00 ± 706.13 | 878.75 ± 301.69 | 3,242.50 ± 300.01 | 2,353.75 ± 425.39 |
| [AirRaid](air_raid/index.html) | Pi0.5 · Completed | 134.55 / 139.89 ms | 2,557.50 ± 491.81 | 1,943.75 ± 776.85 | 1,386.25 ± 434.99 | 3,785.00 ± 384.09 | 3,227.50 ± 1,095.58 |
| [Ant](ant/index.html) | GR00T · Completed | 111.64 / 118.45 ms | 5,011.32 ± 1,880.32 | 6,222.60 ± 133.92 | 165.37 ± 224.74 | 3,199.56 ± 317.95 | 1,955.97 ± 780.51 |
| [Ant](ant/index.html) | OpenVLA · Completed | 90.56 / 92.13 ms | 5,011.32 ± 1,880.32 | 5,565.29 ± 1,351.05 | 194.33 ± 259.05 | 2,301.03 ± 37.60 | 1,853.33 ± 508.69 |
| [Ant](ant/index.html) | Pi0.5 · Completed | 145.42 / 151.63 ms | 5,011.32 ± 1,880.32 | 5,782.50 ± 963.07 | 131.60 ± 93.72 | 4,497.52 ± 71.51 | 1,774.71 ± 824.61 |
| [HalfCheetah](half_cheetah/index.html) | GR00T · Scientific stop | 110.88 / 117.14 ms | 5,414.26 ± 740.00 | 5,012.47 ± 742.29 | -262.47 ± 74.94 | — | — |
| [HalfCheetah](half_cheetah/index.html) | OpenVLA · Completed | 71.82 / 73.09 ms | 5,414.26 ± 740.00 | 3,675.25 ± 1,032.71 | 357.85 ± 285.91 | 5,217.47 ± 108.45 | 288.53 ± 248.11 |
| [HalfCheetah](half_cheetah/index.html) | Pi0.5 · Scientific stop | 145.71 / 151.53 ms | 5,414.26 ± 740.00 | 5,357.98 ± 736.99 | -347.44 ± 82.66 | — | — |
| [Hopper](hopper/index.html) | GR00T · Completed | 114.62 / 122.61 ms | 2,682.09 ± 900.29 | 872.83 ± 479.65 | 8.55 ± 0.09 | 3,814.84 ± 8.80 | 242.55 ± 127.57 |
| [Hopper](hopper/index.html) | OpenVLA · Completed | 71.63 / 74.11 ms | 2,682.09 ± 900.29 | 1,419.93 ± 552.54 | 67.88 ± 150.09 | 3,657.61 ± 13.88 | 424.58 ± 348.01 |
| [Hopper](hopper/index.html) | Pi0.5 · Completed | 199.10 / 264.06 ms | 2,682.09 ± 900.29 | 3,482.49 ± 290.80 | 8.40 ± 0.66 | 2,616.17 ± 629.87 | 235.17 ± 28.38 |
| [Walker2D](walker2d/index.html) | GR00T · Completed | 111.29 / 118.32 ms | 3,743.51 ± 996.07 | 3,527.99 ± 945.27 | 63.24 ± 6.58 | 4,228.54 ± 14.59 | 852.87 ± 357.10 |
| [Walker2D](walker2d/index.html) | OpenVLA · Completed | 70.50 / 72.07 ms | 3,743.51 ± 996.07 | 983.91 ± 373.09 | 103.65 ± 4.90 | 4,102.87 ± 593.64 | 857.44 ± 971.04 |
| [Walker2D](walker2d/index.html) | Pi0.5 · Completed | 188.57 / 209.44 ms | 3,743.51 ± 996.07 | 3,631.42 ± 775.19 | 80.95 ± 10.98 | 3,305.17 ± 1,032.90 | 371.09 ± 123.54 |
| [Humanoid](humanoid/index.html) | GR00T · Scientific stop | 118.38 / 125.04 ms | 6,331.47 ± 407.20 | 6,225.77 ± 1,077.26 | 1,158.63 ± 646.19 | 2,883.10 ± 1,804.11 | — |
| [Humanoid](humanoid/index.html) | OpenVLA · Scientific stop | 286.64 / 292.69 ms | 6,331.47 ± 407.20 | 272.06 ± 36.17 | 226.92 ± 56.90 | 1,036.93 ± 280.76 | — |
| [Humanoid](humanoid/index.html) | Pi0.5 · Scientific stop | 182.02 / 198.61 ms | 6,331.47 ± 407.20 | 6,667.56 ± 46.18 | 891.67 ± 623.80 | — | — |

Returns are mean ± population standard deviation across episodes. The profile column reports latency mean / P95 in milliseconds. L0 is simulated zero latency; B/F use measured real-time inference latency. E10 is the ten-episode simulated profile-teacher evaluation, not selection20; repeated selection seeds are disclosed per model. Source L0 teachers retain their stated clocks. A dash means unavailable or not run; the model status explains which. One training run and separately realized delays do not establish controlled architecture rankings or equal-latency causal effects. Teacher data gates are not VLA acceptance thresholds.

## Report maintenance

Each environment’s results.json records the comparison metrics. Detailed narratives, protocols and figures are curated in its report. `python scripts/render_gymnasium_reports.py` refreshes the overview and result tables while preserving page layouts and media.
