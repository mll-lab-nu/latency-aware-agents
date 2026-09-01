# Frozen Sim2Real paper artifacts

The old cross-device measurement and simulation launchers were replaced by
[`scripts/profile.sh`](../../profile.sh). New profiles use the model/GPU
deployment/workload layout documented in
[`docs/profiling/cookbook.md`](../../../docs/profiling/cookbook.md).

This directory retains only `build_figures.sh` for rebuilding the committed
historical Sim2Real figures from the paper experiment's commit-pinned source
manifest. It does not collect or simulate new profiles.
