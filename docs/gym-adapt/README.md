# Gym Adapt worked examples and agent deliverables

This directory contains transparent adaptation records for the separate
Gymnasium path.

## Who this directory is for

- **Users** can inspect what an adaptation changed, which decisions were made,
  which stages were actually verified, what artifacts were produced, and how
  to continue teacher rollout, data conversion, OpenVLA SFT, and final
  latency-bench evaluation.
- **Adaptation agents** use these records as process examples and as the
  required delivery format for a new environment. The normative implementation
  instructions remain in
  [`docs/gymnasium_adapter.md`](../gymnasium_adapter.md).

Open [`index.html`](index.html) in a browser for the complete InvertedPendulum
example. It records a difficult continuous-action migration: API probing,
explicit user decisions, a task-owned finite-action wrapper, generated configs
and tests, contract propagation, server commands, synchronized evidence, and
the boundary between verified small-model experiments and unexecuted VLA
stages.

## Required output from a future adaptation agent

A new task report should follow the same information architecture and visual
language, use a distinct path such as `docs/gym-adapt/<task_name>.html`, and be
linked from this README. Do not overwrite the InvertedPendulum `index.html`.
It must include:

1. scope, pinned runtime, and truthful stage-status summary;
2. API-derived facts versus user-approved scientific decisions;
3. the complete frozen environment/task contract;
4. every generated wrapper, config, test, and registration entry;
5. a field-by-field explanation of how the baseline pipeline consumes the
   contract;
6. an exact command ledger and linked evidence ledger;
7. the next commands for rollout, conversion, OpenVLA SFT, and final eval;
8. legacy-path isolation, unsupported modes, blockers, and limitations.

Use `VERIFIED`, `NOT RUN`, and `BLOCKED` literally. A planned command or expected
artifact is not evidence.

Curated, reasonably sized report assets belong under:

```text
docs/gym-adapt/assets/<task_name>/
```

Large checkpoints, raw experiment directories, caches, and intermediate
rollouts belong in ignored `artifacts/`, `outputs/`, or external storage. Link
them only when they are intended to remain available.

The complete reporting contract and copyable agent prompt are in
[`docs/gymnasium_adapter.md`](../gymnasium_adapter.md#adaptation-report-contract).
