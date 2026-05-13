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
