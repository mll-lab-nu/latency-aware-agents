# Engineering Preference: No Defensive Checks

This rule applies to the entire repository.

The user strongly dislikes defensive checks added by the assistant.

Do not add defensive validation such as:
- `if x <= 0: raise ValueError(...)`
- constructor argument guards added "just in case"
- shape/type/range assertions that are not required by an existing public contract
- duplicate validation already handled by config/schema/tests/upstream code
- extra fallback/default normalization that hides invalid state

When editing code:
- Do not preserve, move, or expand nearby defensive checks unless they are required by an existing test, public API contract, or explicit user request.
- Prefer letting existing code fail naturally over adding proactive guards.
- If a check seems necessary, stop and explain why before adding it.
- If unsure whether a check is defensive or contract-required, ask first.

Allowed only when clearly required:
- security boundary validation
- external/user-facing API input validation already established by the codebase
- checks explicitly requested by the user
- checks required to satisfy an existing test or documented contract

# Engineering Preference: No Hidden Fallbacks or Over-Engineering

Bad code patterns to avoid and remove in this repository:

- Do not use `mapping.get(key, default)` when `key` is required by the local contract. Use `mapping[key]` so missing required state fails immediately.
- Do not silently substitute another field when one exact field is expected. If semantics require `observation`, do not fall back to `obs`, `image`, first dict value, empty list, `unknown`, or `-1`.
- Do not coerce already-contracted internal values with `int()`, `float()`, `bool()`, `str()`, `list()`, or `dict()` just to be safe.
- Do not swallow broad exceptions to keep execution going. Let the original failure surface unless the boundary explicitly requires aggregation or cleanup.
- Do not normalize invalid state into a runnable value. Missing config, model metadata, action ids, latency metadata, or table spec fields should fail at the point of use.

Testing guidance:

- Do not add tests that assert implementation style, such as "the code does not call `.get()`".
- Tests should assert semantic behavior: required fields fail when missing, wrong actions are rejected, malformed specs fail, and no hidden substitute changes meaning.
- Prefer updating existing tests for the touched behavior instead of adding new tests.

# program Progress Maintenance

For training, evaluation, profiling, publication, pause, or failure work related
to HumanoidBench Balance Simple, MIKASA InterceptGrabFast, Extreme Parkour Go2,
or HAIC PullCart, update `program/PROGRESS.md` once the outcome is verified.
Record the audit date, exact checkpoint/training step and configuration, evidence
links and revisions, numeric evaluation metrics with episode counts, and the next
step. Keep training completion, evaluation completion, acceptance, and publication
separate. Distinguish latency/FPS/seed protocols and retain historical results;
old running messages do not establish current process state. Use
`program/README.md` to locate active handoffs and archived experiment records.
