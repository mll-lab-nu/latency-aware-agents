# Test suite conventions

Tests protect observable behavior, required invariants, and plausible regressions.
Run the suite in the reference `latency` environment.

```bash
conda run -n latency python -m pytest
conda run -n latency python -m pytest tests/executors -q
conda run -n latency python -m pytest -m gpu
```

The default run excludes GPU tests.
Optional integrations skip when their dependencies are absent.
Report skipped checks separately from successful validation.

## Test ownership

Place a test in the suite that owns the behavior:

| Suite | Owner |
| --- | --- |
| `core/`, `scheduler/`, `latency/` | Configuration, clocks, actions, scheduling, latency profiles |
| `executors/` | Paused, simulated, batched, and realtime execution |
| `envs/` | Environment factories, adapters, task registration, episode contracts |
| `policies/` | Policy construction and inference adapters |
| `data/` | Collection, alignment, splits, storage, and metadata |
| `eval/`, `logging/`, `analysis/` | Evaluation, traces, recording, and statistics |
| `profiling/` | Deployment measurement, fitting, recovery, and input resolution |
| `integrations/starvla/` | Task contracts, conversion, loaders, training evaluation, resident DAgger |
| `training/` | Reference training and framework compatibility |
| `training/gpu/` | CUDA environment and training behavior |
| `scripts/` | Task recipes, experiment runners, and Bash entrypoints |
| `integration/` | Cross-module and CLI behavior |
| `integration/sample_factory/` | Checkpoint loading, execution adapters, and teacher export |

Keep shared support in the test root.
Use globally unique test basenames because pytest uses its default import mode.
Do not add `__init__.py` files to test directories.
Import `REPO_ROOT` from `helpers` for checkout paths.

## Test contracts

- Assert outputs, files, exceptions, public state, and meaningful collaborator behavior.
- Use one representative test for each distinct behavioral contract.
- Mock network, GPU, subprocess, and other external boundaries when necessary.
- Use `pytest.approx` for computed floating-point values.
- Preserve useful fixtures for failures that need real processes or files.
- Keep tests valid when internal code changes without changing public behavior.
- Remove obsolete tests with their feature.
- Place retained tests in the responsible suite and document their contract.

Internal call counts, private attributes, and snapshots usually do not define public behavior.
A focused internal test can protect an independent semantic contract when a public test cannot exercise it.
Do not duplicate production logic to calculate the expected result.

## GPU and simulator checks

Mark optional CUDA tests with `gpu` and ViZDoom tests with `vizdoom`.
The GPU subtree also applies its marker automatically.
Keep dependency guards so optional suites remain runnable in their supported environments.

Demon Attack CPU/ALE and CUDA/CuLE comparisons use identical seeds, reset settings, and action traces.
RAM, raw RGB, and 84×84 observations must match element by element.
Tests must not synchronize state or compensate images to force agreement.

## Existing integration coverage

`tests/data/test_rollout.py` owns canonical collection behavior.
`tests/integration/test_external_factories.py` exercises external projects, worker processes, collection, conversion, and profile replay.
Executor suites own scheduling and buffer lifetime.
These tests protect long-lived interfaces and regressions.
