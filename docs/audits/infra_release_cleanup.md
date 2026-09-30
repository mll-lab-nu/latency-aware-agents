# Source-release infrastructure cleanup — 2026-09-20

Code and documentation cleanup is implemented; release acceptance remains open.
Existing uncommitted task/training changes were retained. Baseline comparisons
below use tracked revision `1f127ecf`, not a claim that the whole initial dirty
worktree was reproduced.

## Implemented boundaries

- Executors accept their actual dependencies; batched cleanup owns its backend
  directly, and realtime metadata no longer needs a dummy sampler.
- Evaluation defaults and filesystem inputs have one resolution owner. Missing
  experiment seeds retain their policy/suite meaning. Absolute path conversion
  preserves HF snapshot symlinks and checkpoint sibling files.
- Each run writes `resolved_config.yaml` after executor preparation, inside the
  resource-cleanup boundary. Bundled resources share a source-checkout anchor.
- Checkpoint and resident StarVLA runners share input configuration, retaining
  their existing prompt, multiview, output and model-ownership differences.
- Backbone preparation uses explicit model coordinates or a local path outside
  inference workers; downloads use the HF cache. Historical recipes remain
  evidence; portable overrides are documented in the artifact guide.
- Stale implementation plans are archived; current ownership is in
  [core infrastructure](../infra.md).

## Verification

An isolated Python 3.10 environment used CPU Torch 2.11.0, NumPy 1.26.4,
Gymnasium 0.29.1 and the pinned Flappy/Sample Factory submodules. No simulator
or GPU throughput claim follows from these checks.

Two disjoint test selections produced **581 passed, 10 failed, 1 skipped,
1 GPU test deselected**:

```bash
python -m pytest tests/eval tests/core tests/logging tests/scheduler \
  tests/executors tests/integration/test_run.py tests/profiling tests/latency \
  tests/envs/test_env_factory.py tests/policies/test_starvla_task_policy.py \
  tests/scripts/test_starvla_train_task.py tests/scripts/test_starvla_artifacts.py -q
python -m pytest tests/policies/test_starvla_policy.py \
  tests/policies/test_generic_gymnasium_starvla_policy.py -q
```

All ten failures reproduce on the tracked baseline in the same environment:

| Cases | Remaining issue |
| --- | --- |
| 3 pi0.5 overlay cases | Pinned source digests differ for Hopper, Ant and Humanoid; original files and expected digests were not rewritten. |
| 1 realtime device case | Test mocks Torch's visible device count, while production first finds this host's GPU through `nvidia-smi`. |
| 3 generic Gymnasium cases | Discrete task decoder routing and explicit action dimension disagree with the existing tests. |
| 2 Deadly Corridor cases | Existing factorized/joint action-layout names disagree with the pinned decoder. |
| 1 stitch case | The isolated environment lacks `transformers`, imported by the existing stitching dependency. |

Temporary constructor/configuration probes compared 16 model-construction and
36 evaluation-configuration combinations. The existing HF boundary test now
also covers evaluation path resolution; tests remain in their owning suites.

An editable source install ran the CLI from outside the checkout: two simulated
DummyEnv episodes, two slots, 30 Hz, 50 ms fixed latency, seed 7, ten steps each.
Baseline and refactor produced identical action, step and latency JSONL records
(20 each), returns of 10 and mean latency of 50 ms. This is a portability and
semantic check, not the formal performance benchmark.

The temporary metrics probe confirmed differing score/alias priorities between
streaming and trace paths; their implementations were deliberately unchanged.
Compilation and whitespace checks passed. Raw local test output is under
`outputs/infra_release_review/`.

## Open release gates

Resolve the baseline failures under their owning semantic contracts, complete
the model/simulator environments, and run the existing acceleration benchmark
with fixed hardware and artifacts. GPU batching, real model inference, all
simulators and formal training remain unverified. Complete the full fresh-clone
installation and reproduction matrix before tagging a release. Historical
experiment results and publication state were not changed by this cleanup.
