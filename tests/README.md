# Test suite conventions

This is the contract for writing tests in this repo. Read it before adding or
changing tests. It exists because the suite previously drifted into 100+ flat,
disorganized files where many tests locked *internal implementation* instead of
*observable behavior*. Don't reintroduce that.

---

## 1. Where a test goes

Tests mirror the source tree. A test lives in the subdirectory that matches the
**source package it primarily exercises**:

| Test directory | Source it mirrors |
|---|---|
| `tests/core/` | `latency_bench/core/` (config, clock, timing, actions, types) |
| `tests/scheduler/` | `latency_bench/scheduler/` (action queue, decision, persistence) |
| `tests/latency/` | `latency_bench/latency/` (samplers, profiles, bursts, calibration, hardware) |
| `tests/executors/` | `latency_bench/executors/` (realtime, simulated, batched, paused, factory, warmup) |
| `tests/envs/` | `latency_bench/envs/` (flappy, demon_attack, deadly_corridor adapters, factory) |
| `tests/policies/` | `latency_bench/policy/` (starvla, sample_factory, factory) |
| `tests/eval/` | `latency_bench/eval/` (suites, driver, periodic eval, collect/report) |
| `tests/logging/` | `latency_bench/logging/` (records, trajectory logger, video, pipeline) |
| `tests/analysis/` | `latency_bench/analysis/` (tables, stats, figures, diagnostics, fits) |
| `tests/cross_device/` | `latency_bench/cross_device/` (formal cross-device protocol, runner, profiles, contracts, and entrypoints) |
| `tests/data/` | `latency_bench/data/` (rollout schema/filters/parquet, exporters, sf teacher) |
| `tests/training/` | `training/` (sample_factory backend, sf-compat shims, episode metrics) |
| `tests/training/gpu/` | `training/{flappy_sf,demon_attack_sf}` and anything needing a CUDA GPU |
| `tests/scripts/` | `scripts/` (sweeps, manifest jobs, bash scripts, bench harnesses) |
| `tests/integration/` | end-to-end runs (`latency_bench/run.py`) |

When you add a new source module, put its tests in the matching directory (create
it if needed). Don't add new files to the `tests/` root — the root holds only
shared support (`conftest.py`, `helpers.py`, `executor_mode_helpers.py`).

Test file basenames must be **globally unique** across the tree (pytest's default
import mode requires it). No `__init__.py` in test directories.

---

## 2. Test behavior, not implementation

A test locks the **observable contract** of a unit: given input X, assert output Y,
plus the corner cases (empty, boundary, error, concurrency). It must survive any
internal rewrite that keeps that contract.

**The one rule that decides everything:**

> Would this assertion break if I rewrote the internals while keeping the
> public behavior identical? If yes, it's locking implementation — fix it.

### ✅ Do assert
- Return values, raised exceptions, and their messages/types.
- Files/artifacts written, their schema, and their contents.
- Public state and metrics exposed by the object's API.
- Calls made to a **collaborator you injected** through the public API (a fake
  passed into the constructor / function) — verifying calls at that *seam* is
  legitimate behavior verification (e.g. "each slot consulted its own latency
  source").

### ❌ Don't assert
- Private state on a real object: `obj._completed_buffers`, `env._raw_episode_steps`,
  `obj._uses_cuda`. If the only way to check something is a private attribute, it
  usually has no observable behavior — reconsider whether it's worth a test.
- Internal call counts/order of *production-internal* helpers (vs. injected fakes).
- Exact snapshots of intermediate/internal data structures.
- Directly testing `_private` functions when a public entry point covers them.
  (A focused unit test of a small private helper is acceptable when it's a real
  seam; reaching past the public API just to inspect internals is not.)

### Gray areas that are OK
- **Equivalence / metamorphic properties** — e.g. "batched path == map over the
  single-item path", where the reference *reuses the real production primitive*
  and only re-derives the composition. This locks a property, not a tautology.
- **Golden-reference guards for a perf optimization** — an optimized path asserted
  equal to a readable generic reference *spec*. Keep the reference; it's the
  contract the fast path must satisfy.
- **Tuning-constant monkeypatch for tractability** — patching a module-level tuning
  constant (e.g. warmup sample thresholds) so a test needs 3 samples instead of
  100, *while still asserting the public output*. Tolerable. Patching/asserting
  internal *state or call sequences* is not.

### Anti-patterns to never add
- Re-implementing production logic in the test and asserting `new == my_copy`
  (a true tautology — both can share the same bug).
- `assert mock._internal_helper.call_count == 3` on a production-internal.
- Exact float equality on computed values — use `pytest.approx`.

---

## 3. Conventions

- One test, one behavior. Name it for the behavior it locks
  (`test_<unit>_<condition>_<expected>`), not for the function it calls.
- Mock only at **boundaries** (network, GPU, subprocess, filesystem, external
  servers). Never mock the internals of the unit under test.
- Floats: `pytest.approx`. Strings/dicts: assert the meaningful contract, not an
  over-specified blob that breaks on benign change.
- Repo paths: never compute the repo root with `Path(__file__).parents[N]` — that
  breaks when a test moves. Import the anchored root:
  `from helpers import REPO_ROOT`.
- GPU/CUDA/cule/vizdoom tests go under `tests/training/gpu/` (auto-marked `gpu`) or
  carry `pytestmark = pytest.mark.gpu` / `vizdoom`. Keep their own
  `importorskip`/skip guards so they degrade gracefully when deps are absent.

---

## 4. Running

Use the `latency` conda env.

```bash
conda run -n latency python -m pytest            # default: excludes GPU (-m "not gpu")
conda run -n latency python -m pytest -m gpu     # GPU subtree, on a CUDA box
conda run -n latency python -m pytest tests/executors -q   # one area
```

Config lives in `pytest.ini` (testpaths, registered markers, default `-m "not gpu"`).

> The `latency` env is missing some optional deps (e.g. `omegaconf` for starVLA),
> so a clean checkout already has pre-existing env-dependent failures/skips. When
> judging whether *your* change broke something, **diff against a baseline run of
> the unchanged tree** — don't assume every red test is yours.

---

## 5. Dead-test discipline

- Delete a feature ⇒ delete its tests in the same change. Don't leave orphans.
- Don't commit a test that is unconditionally skipped as a placeholder. Use a
  marker (`gpu`/`vizdoom`/`slow`) so it's deselected-but-runnable, not dead.
- If a test only passes because it's skipped in every environment, it isn't
  testing anything — either give it a runnable path or remove it.
