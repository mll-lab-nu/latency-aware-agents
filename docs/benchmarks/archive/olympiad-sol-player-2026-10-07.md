# Player sol (GPT-6.1 SOL, second Azure resource): wiring, API speed, round robin of the Codex models (2026-10-07)

Status: **final (2026-10-07 15:30 UTC).** 22 of the 24 planned matches are valid. The two SOL-Luna billiards matches
are **blocked: single-seat TPM in long matches** (see Observations). Every match is seed 1, with both first orders.
Each model runs on Codex 0.158.0 with reasoning effort medium, program play. SOL matches run strictly one at a time
host-wide (a shared `flock` on `.sol.lock` from 14:15 UTC), with at most 2 Astra seats on the host.

## Resource and API

- The SOL endpoint is an Azure AI Services (`*.cognitiveservices.azure.com`) resource in Norway East.
- The deployment list has `gpt-6.1-sol` (model `gpt-6.1-sol`, version `gpt-6.1-sol-2026-09-29`, standard scale,
  succeeded) and `gpt-6-luna`.
- Both Responses (`/openai/v1/responses`, the API Codex needs) and Chat Completions (`/openai/v1/chat/completions`)
  answer 200.
- The context and output limits are not exposed by the deployment or model listing.
- SOL limits, from the `x-ratelimit-*` headers: 333 requests and 333k tokens per 60 s, on deployment
  `gpt-6.1-sol`. There is no `AZURE_SOL_DEPLOYMENTS` pool, so no failover.
- Astra: 250 requests and 250k tokens per 60 s on each of `gpt-6-astra`, `-2` and `-3` (East US 2), with failover.
- Luna goes through OpenRouter, which sends no rate-limit headers.
- An earlier endpoint (`genai-test-aoai-cua-1.openai.azure.com`) did not exist (NXDOMAIN). The user replaced it.

## Wiring (branch sol-player, rebased on origin/main)

- **Proxy route.** `players/metering/think_proxy.py` has a route `azure_sol` with its own `AZURE_SOL_ENDPOINT`,
  `AZURE_SOL_API_KEY` and `AZURE_SOL_DEPLOYMENTS` (same wire and failover as `azure`).
  - Calls are logged with route `azure_sol`.
  - A missing SOL key gives a 500 and never falls back to the Astra key.
  - The `azure` route is unchanged: main's proxy and this one send the same upstream requests, return the same
    bytes and write the same call records.
- **Runner.** `run_duel.sh --a/--b sol` reads `AZURE_SOL_*` from `SOL_ENV_FILE` (never `.env`) and exports them
  only as `AZURE_SOL_<VAR>_<seat>`, so only that seat's metering proxy has them.
- **Player list.** `players/players.toml` registers player `sol`: model `openrouter/openai/gpt-6-sol`,
  `deployment = "gpt-6.1-sol"`, config_version 2.
- **Tasks.** The tasks are regenerated.
- **Tests.**
  - `scripts/olympiad/test_sol_route.py` passes (5/5), and `ci_checks/run_all.sh` passes.
  - A live proxy call to SOL through route `azure_sol` completed.
  - In the matches, the call logs show route `azure_sol` (seat SOL), `azure` (seat Astra) and `openrouter`
    (seat Luna).
- **Harness metadata, fixed at 09:25 UTC (commit d0127ef0).** Codex 0.158.0 has no metadata for
  `gpt-6.1-sol`. It logged "Model metadata ... not found. Defaulting to fallback metadata" and ran SOL with
  generic instructions, tools and limits. Astra and Luna are known slugs.
  - Codex's bundled model list has the family's slug `gpt-6-sol`. Its context window (272k, maximum 872k), tool
    mode (`code_mode_only`), freeform `apply_patch`, truncation policy (10k tokens) and reasoning levels match
    `gpt-6-astra` and `gpt-6-luna`. Its instructions are the SOL and Luna family template.
  - The SOL seat now sends `openai/gpt-6-sol`, and the pinned route `azure_sol` serves it with deployment
    `gpt-6.1-sol`. The call log keeps the name sent as `model_alias`.
  - Checks: a Codex 0.158.0 `exec` through the pinned proxy shows no warning with `gpt-6-sol` and shows it again
    with `gpt-6.1-sol`. The matches' call logs show `gpt-6-sol` served by `gpt-6.1-sol`.
  - The SOL matches played before the fix are listed separately as superseded and rerun.

## API speed

Requests are streamed Responses calls, sequential unless marked xN (N in parallel):
- **small:** a 25-token prompt, effort low.
- **real:** about 5.4k input tokens (the table tennis and darts instructions plus a 37 KB camera JPEG), effort
  medium.
- **long:** about 850 output tokens of plain text, effort low.

TTFT is the time to the first streamed delta. Decode is visible output tokens/s after the first delta.

| model | test | n | ok | 429 / 5xx | TTFT mean / p50 / p95 (s) | total mean / p50 / p95 (s) | in tok | out tok (reasoning) | decode tok/s |
|---|---|---|---|---|---|---|---|---|---|
| SOL | small x1 | 30 | 30 | 0 / 0 | 1.54 / 1.49 / 1.87 | 1.71 / 1.62 / 2.26 | 25 | 5 (0) | - |
| SOL | small x2 | 10 | 10 | 0 / 0 | 1.44 / 1.44 / 1.65 | 1.57 / 1.57 / 1.80 | 25 | 5 (0) | - |
| SOL | small x4 | 8 | 8 | 0 / 0 | 1.50 / 1.48 / 1.70 | 1.63 / 1.61 / 1.82 | 25 | 5 (0) | - |
| SOL | real x1 | 10 | 10 | 0 / 0 | 2.26 / 2.06 / 2.96 | 3.37 / 3.28 / 4.11 | 5392 | 88 (13) | - |
| SOL | real x2 | 4 | 4 | 0 / 0 | 2.29 / 2.37 / 2.68 | 3.51 / 3.48 / 3.93 | 5392 | 87 (12) | - |
| SOL | real x4 | 8 | 8 | 0 / 0 | 2.13 / 1.94 / 2.75 | 3.43 / 3.50 / 4.13 | 5392 | 89 (12) | - |
| SOL | long x1 | 5 | 5 | 0 / 0 | 1.95 / 1.81 / 2.61 | 17.28 / 16.89 / 18.81 | 43 | 862 (17) | 55.26 |
| Astra | small x1 | 30 | 30 | 0 / 0 | 1.22 / 1.03 / 2.40 | 1.41 / 1.23 / 2.63 | 25 | 5 (0) | - |
| Astra | small x2 | 10 | 10 | 0 / 0 | 1.28 / 1.19 / 1.84 | 1.63 / 1.47 / 2.81 | 25 | 5 (0) | - |
| Astra | real x1 | 10 | 10 | 0 / 0 | 2.72 / 2.25 / 4.42 | 4.23 / 3.82 / 6.08 | 5392 | 82 (0) | - |
| Astra | real x2 | 4 | 4 | 0 / 0 | 2.01 / 1.87 / 2.90 | 3.59 / 3.25 / 5.18 | 5392 | 76 (0) | - |
| Astra | long x1 | 5 | 5 | 0 / 0 | 0.96 / 0.93 / 1.12 | 18.09 / 18.38 / 18.79 | 43 | 821 (0) | 47.99 |
| Luna | small x1 | 10 | 10 | 0 / 0 | 1.84 / 1.18 / 5.17 | 1.91 / 1.24 / 5.29 | 25 | 13 (7) | - |
| Luna | real x1 | 5 | 5 | 0 / 0 | 2.67 / 2.51 / 3.19 | 3.26 / 3.05 / 3.88 | 5392 | 214 (165) | - |
| Luna | long x1 | 3 | 3 | 0 / 0 | 1.21 / 1.18 / 1.50 | 11.56 / 12.14 / 12.24 | 43 | 1081 (52) | 100.77 |

## Matches

Columns:
- **match time:** the match service's elapsed time.
- **think s:** the sum of the durations of completed model calls.
- **call latency:** per model call, the time from request to end of stream.
- **429 / API errors:** the validity rule's counts. A match is invalid with any unrecovered error or more than 5 transient ones.
- **score A-B:** table tennis games, darts legs, curling points; in billiards the balls left on the table
  (fewer wins).
- **winner:** the match service's winner. It counts in the W-L only when the match is valid.
- Not listed: billiards SOL-Astra B of 13:24 UTC (job `...20261007T132431Z`). It was cut off at 13:38 when the
  previous operator's session ended (the harbor process died, no `final.json`). Rerun as #24.

| # | sport | seat A | seat B | first | score A-B | winner | valid | match time | calls A / B | input tok A / B (M, cached %) | output tok A / B (reasoning) | think s A / B | call latency mean/p95 s A / B | 429 / API errors | note |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | tabletennis | Astra | Luna | A | 5-2 | Astra | yes | 772 s | 60 / 77 | 2.2 (98%) / 4.4 (98%) | 8719 (2094) / 34291 (24244) | 370 / 534 | 6.16/12.41 / 6.94/19.93 | 0 / transient 0+0, unrecovered 0+0 |  |
| 2 | tabletennis | Astra | Luna | B | 5-0 | Astra | yes | 694 s | 41 / 47 | 1.4 (96%) / 2.5 (97%) | 6437 (1097) / 30955 (21795) | 285 / 402 | 6.95/14.55 / 8.56/29.20 | 0 / transient 0+0, unrecovered 0+0 |  |
| 3 | tabletennis | SOL | Astra | A | 5-2 | SOL | yes | 863 s | 60 / 58 | 3.8 (97%) / 2.4 (97%) | 7895 (1419) / 7737 (1583) | 279 / 323 | 4.65/7.50 / 5.57/10.33 | 0 / transient 0+0, unrecovered 0+0 |  |
| 4 | tabletennis | SOL | Astra | B | 5-3 | SOL | yes | 865 s | 77 / 52 | 3.5 (98%) / 2.0 (97%) | 10460 (2842) / 10695 (3419) | 386 / 493 | 5.01/7.54 / 9.48/31.52 | 0 / transient 0+0, unrecovered 0+0 |  |
| 5 | tabletennis | SOL | Luna | A | 5-0 | SOL | yes | 561 s | 47 / 57 | 1.8 (97%) / 2.2 (97%) | 6164 (873) / 26711 (17933) | 210 / 375 | 4.48/5.37 / 6.70/23.34 | 0 / transient 0+1, unrecovered 0+0 |  |
| 6 | tabletennis | SOL | Luna | B | 5-0 | SOL | yes | 604 s | 44 / 47 | 1.5 (97%) / 1.8 (96%) | 7042 (1391) / 32898 (24511) | 219 / 427 | 4.98/7.33 / 9.09/36.01 | 0 / transient 0+0, unrecovered 0+0 |  |
| 7 | darts | SOL | Astra | A | 0-0 | Astra | NO | 824 s | 54 / 56 | 1.1 (96%) / 2.0 (97%) | 9284 (4423) / 13335 (4784) | 255 / 467 | 6.23/14.67 / 8.33/19.02 | 0 / transient 7+0, unrecovered 6+0 | infra: SOL TPM exceeded (2 SOL seats at once); stopped by me; rerun |
| 8 | darts | SOL | Astra | B | 0-2 | Astra | NO | 633 s | 73 / 67 | 2.1 (98%) / 2.7 (98%) | 12377 (6157) / 13214 (4377) | 384 / 434 | 6.09/10.80 / 6.47/15.74 | 0 / transient 10+0, unrecovered 0+0 | infra: SOL TPM exceeded (2 SOL seats at once); rerun |
| 9 | darts | Astra | Luna | A | 2-0 | Astra | yes | 1452 s | 119 / 143 | 5.5 (99%) / 10.5 (99%) | 11842 (1601) / 44154 (34418) | 614 / 1033 | 5.16/10.80 / 7.33/17.30 | 0 / transient 0+2, unrecovered 0+0 |  |
| 10 | darts | Astra | Luna | B | 2-0 | Astra | yes | 1399 s | 112 / 104 | 4.8 (99%) / 8.7 (98%) | 13970 (3876) / 41562 (31475) | 593 / 1065 | 5.29/11.55 / 10.24/20.98 | 0 / transient 0+0, unrecovered 0+0 |  |
| 11 | darts | SOL | Astra | A | 1-2 | Astra | yes | 891 s | 93 / 87 | 4.7 (98%) / 3.9 (98%) | 12041 (3027) / 18865 (7418) | 503 / 606 | 5.41/8.10 / 6.96/17.17 | 0 / transient 0+0, unrecovered 0+0 |  |
| 12 | darts | SOL | Astra | B | 0-2 | Astra | yes | 870 s | 86 / 73 | 3.6 (98%) / 2.7 (98%) | 15138 (6273) / 20362 (9816) | 514 / 650 | 5.98/13.31 / 8.91/30.28 | 0 / transient 0+0, unrecovered 0+0 |  |
| 13 | darts | SOL | Luna | A | 2-1 | SOL | yes | 1493 s | 132 / 114 | 6.3 (99%) / 8.9 (98%) | 14633 (4795) / 47132 (36480) | 702 / 1004 | 5.32/8.64 / 8.81/16.18 | 0 / transient 0+0, unrecovered 0+0 |  |
| 14 | darts | SOL | Luna | B | 2-1 | SOL | yes | 1890 s | 164 / 183 | 8.2 (99%) / 16.0 (99%) | 14000 (3316) / 40619 (28452) | 748 / 1347 | 4.56/6.92 / 7.36/13.37 | 0 / transient 0+0, unrecovered 0+0 |  |
| 15 | curling | Astra | Luna | A | 6-0 | Astra | yes | 1143 s | 98 / 89 | 4.5 (98%) / 6.6 (98%) | 14369 (6221) / 41857 (32282) | 564 / 727 | 5.76/14.97 / 8.17/20.28 | 0 / transient 0+0, unrecovered 0+0 |  |
| 16 | curling | Astra | Luna | B | 5-2 | Astra | yes | 980 s | 93 / 92 | 5.1 (98%) / 6.0 (98%) | 14085 (5949) / 22877 (15853) | 640 / 502 | 6.88/16.43 / 5.46/13.88 | 0 / transient 0+0, unrecovered 0+0 |  |
| 17 | curling | SOL | Astra | A | 2-4 | Astra | yes | 891 s | 101 / 93 | 4.3 (97%) / 4.8 (98%) | 14545 (5441) / 16006 (7566) | 530 / 678 | 5.25/9.26 / 7.29/17.39 | 0 / transient 0+0, unrecovered 0+0 |  |
| 18 | curling | SOL | Astra | B | 1-2 | Astra | yes | 1154 s | 115 / 125 | 6.2 (98%) / 7.9 (99%) | 17278 (4542) / 23466 (11236) | 571 / 918 | 4.96/10.11 / 7.35/16.75 | 0 / transient 0+0, unrecovered 0+0 |  |
| 19 | curling | SOL | Luna | A | 5-2 | SOL | yes | 1460 s | 95 / 135 | 4.6 (98%) / 11.9 (99%) | 16896 (4688) / 47421 (35893) | 534 / 1106 | 5.62/9.48 / 8.19/18.75 | 0 / transient 0+0, unrecovered 0+0 |  |
| 20 | curling | SOL | Luna | B | 9-1 | SOL | yes | 794 s | 61 / 87 | 2.1 (98%) / 5.5 (98%) | 9610 (2000) / 21910 (15783) | 289 / 540 | 4.73/8.36 / 6.28/13.27 | 0 / transient 0+1, unrecovered 0+0 |  |
| 21 | billiards | Astra | Luna | A | 0-6 | Astra | yes | 455 s | 31 / 24 | 1.2 (95%) / 0.7 (93%) | 8438 (3162) / 13662 (11570) | 258 / 191 | 8.34/20.48 / 7.95/27.60 | 0 / transient 0+0, unrecovered 0+0 |  |
| 22 | billiards | Astra | Luna | B | 0-6 | Astra | yes | 1028 s | 71 / 44 | 4.8 (98%) / 2.0 (96%) | 17989 (9148) / 42021 (39212) | 677 / 530 | 9.53/22.41 / 12.04/57.12 | 0 / transient 0+0, unrecovered 0+0 |  |
| 23 | billiards | SOL | Astra | A | 4-0 | Astra | yes | 507 s | 34 / 37 | 1.2 (96%) / 1.5 (96%) | 10250 (3656) / 12102 (5789) | 240 / 396 | 7.05/18.76 / 10.71/25.69 | 0 / transient 0+0, unrecovered 0+0 |  |
| 24 | billiards | SOL | Astra | B | 1-0 | Astra | yes | 649 s | 62 / 52 | 2.6 (98%) / 2.8 (97%) | 10679 (2670) / 12341 (5880) | 366 / 439 | 5.90/15.98 / 8.44/19.86 | 0 / transient 0+0, unrecovered 0+0 |  |
| 25 | billiards | SOL | Luna | A | 0-3 | SOL | NO | 1699 s | 196 / 102 | 11.4 (99%) / 10.1 (98%) | 20455 (4511) / 59385 (50655) | 777 / 994 | 5.71/10.62 / 9.75/30.26 | 0 / transient 60+0, unrecovered 0+0 | infra: SOL TPM exceeded by its one seat (long match, growing context); SOL cleared the table first; blocked: single-seat TPM in long matches |
| 26 | billiards | SOL | Luna | B | 0-1 | Luna | NO | 1692 s | 373 / 128 | 11.6 (99%) / 10.4 (99%) | 20538 (4404) / 43429 (34920) | 772 / 839 | 5.08/10.64 / 6.55/19.73 | 0 / transient 218+0, unrecovered 1+0 | infra: SOL TPM exceeded by its one seat (218 transient + 1 unrecovered); SOL cleared the table first; blocked: single-seat TPM in long matches |

| model | W | L | D |
|---|---|---|---|
| SOL | 8 | 6 | 0 |
| Astra | 14 | 2 | 0 |
| Luna | 0 | 14 | 0 |

| sport | SOL W-L | Astra W-L | Luna W-L |
|---|---|---|---|
| tabletennis | 4-0 | 2-2 | 0-4 |
| darts | 2-2 | 4-0 | 0-4 |
| curling | 2-2 | 4-0 | 0-4 |
| billiards | 0-2 | 4-0 | 0-2 |

### Fallback metadata (superseded)

Played before the fix: Codex ran SOL with fallback model metadata. Not counted in the W-L table; rerun above.

| # | sport | seat A | seat B | first | score A-B | winner | valid | match time | calls A / B | input tok A / B (M, cached %) | output tok A / B (reasoning) | think s A / B | call latency mean/p95 s A / B | 429 / API errors | note |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | tabletennis | SOL | Astra | A | 1-5 | Astra | yes | 1145 s | 58 / 86 | 2.7 (97%) / 4.6 (98%) | 12625 (2104) / 8620 (1785) | 326 / 443 | 5.61/9.87 / 5.15/9.46 | 0 / transient 0+0, unrecovered 0+0 |  |
| 2 | tabletennis | SOL | Luna | A | 7-5 | SOL | yes | 1540 s | 84 / 134 | 3.9 (98%) / 12.5 (99%) | 14959 (2152) / 76014 (58013) | 451 / 1016 | 5.37/11.36 / 7.58/26.20 | 0 / transient 0+0, unrecovered 0+0 |  |
| 3 | tabletennis | SOL | Astra | B | 0-5 | Astra | yes | 535 s | 62 / 42 | 2.2 (97%) / 1.2 (97%) | 11514 (2837) / 4541 (472) | 353 / 167 | 5.78/11.54 / 3.97/4.83 | 0 / transient 1+0, unrecovered 0+0 |  |
| 4 | tabletennis | SOL | Luna | B | 5-1 | SOL | yes | 758 s | 75 / 54 | 2.5 (98%) / 3.5 (97%) | 13589 (2271) / 45781 (35488) | 393 / 578 | 5.24/8.63 / 10.71/32.01 | 0 / transient 0+0, unrecovered 0+0 |  |
| 5 | darts | SOL | Astra | A | 0-2 | Astra | yes | 683 s | 71 / 67 | 1.7 (98%) / 2.6 (98%) | 13970 (6068) / 12327 (4503) | 441 / 471 | 6.22/10.75 / 7.04/15.58 | 0 / transient 0+0, unrecovered 0+0 |  |

## Observations

- **Final W-L (22 valid matches):** Astra 14-2, SOL 8-6, Luna 0-14.
  - SOL against Astra: 2-6. SOL won table tennis twice (5-2, 5-3). Astra won darts twice (2-1, 2-0), curling
    twice (4-2, 2-1) and billiards twice (SOL left 4 and 1 balls).
  - SOL against Luna: 6-0 in table tennis, darts and curling. Billiards is blocked (below).
  - Astra against Luna: 8-0.
- **SOL with fallback metadata** (superseded, before 09:25 UTC): it lost both table tennis matches to Astra (1-5,
  0-5) and the darts match (0-2). With Codex's proper SOL instructions and tools it won both table tennis
  matches against Astra. One seed per order is still a small sample.
- **Speed.** SOL has the lowest call latency of the three in every sport: about 4.5 to 7 s mean (p95 5 to 19 s),
  against 5 to 11 s for Astra and 5.5 to 12 s for Luna. It also spends the least reasoning per match.
- **Astra** is the only model whose calls see Azure 429s. Failover over its 3 deployments absorbed them with no
  errors.
- **Luna** thinks far more per call (about 75% of its output is reasoning) with slow calls (p95 13 to 57 s). In
  table tennis its programs applied only 1 to 9 acts, against 25 to 99 for SOL.
- **SOL quota: blocked in long matches.** The SOL resource allows 333k tokens per minute on its one deployment
  (no failover pool). Cached input tokens still count.
  - Two SOL seats at once broke the limit: the two darts SOL-Astra matches of 09:17 were invalid (transient
    errors 10 and 13, above 5). From then on SOL ran strictly one at a time.
  - **A single SOL seat breaks it too in long billiards matches.** Both SOL-Luna billiards matches ran alone on
    SOL. Each match was invalid: A had 60 transient errors, B had 218 transient and 1 unrecovered. In both, SOL
    cleared the table first (0 balls left against 3 and 1). From about minute 10, Codex's context
    grew. Each call re-sends it, so the seat went past 333k tokens per minute. Azure then cut the HTTP 200 streams
    about 0.1 s after the first byte ("stream closed before response.completed").
  - The endpoint itself is healthy: it gives full quota when idle. SOL-Luna billiards is recorded as **blocked:
    single-seat TPM in long matches** and not rerun. A higher TPM, or a second SOL deployment for failover, would
    unblock it.
