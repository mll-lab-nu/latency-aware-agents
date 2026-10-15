# Broadcast picture quality

Only the spectator renderer changes. It reads copied scenes and state logs; match simulation and verification
remain in their original containers. GPU rendering still shares host/GPU resources, so its cost should be monitored.

## Run

Build the two viewer-only images (NVIDIA Container Toolkit must expose graphics driver libraries):

```sh
docker build -f scripts/olympiad/broadcast/Dockerfile.egl -t eai-broadcast-egl:0.1.0 .
docker build -f scripts/olympiad/broadcast/Dockerfile.egl \
  --build-arg BASE_IMAGE=ghcr.io/mll-lab-nu/eai-olympiad-soccer:0.1.0 \
  -t eai-broadcast-soccer-egl:0.1.0 .
BROADCAST_RENDERER=egl BROADCAST_GPU=0 BROADCAST_QUALITY=1080p BROADCAST_FPS=30 \
  python3 scripts/olympiad/broadcast.py --port 8048
```

`BROADCAST_QUALITY`: `540p`, `720p`, `1080p`. Default: 540p for CPU, 720p for EGL.
`BROADCAST_FPS`: ceiling, 3–60 (default 30); actual frame rate follows each helper's measured capacity.
`BROADCAST_RENDERER`: `osmesa` (default) or `egl`. `BROADCAST_GPU` selects one device (default 0).
Image overrides: `BROADCAST_EGL_IMAGE`, `BROADCAST_SOCCER_EGL_IMAGE`.

Before using EGL, a small real render verifies the NVIDIA renderer string. Missing images, failed contexts,
or Mesa software EGL trigger a CPU fallback, cached until server restart. `/api/matches.video` reports configured
size and each started helper's actual backend. CPU fallback preserves configured resolution but can lower FPS.
For a CPU-only machine or a slower connection, run with `BROADCAST_RENDERER=osmesa BROADCAST_QUALITY=540p`.

Live JPEGs use quality 90 and full chroma resolution to keep colored text sharp. New point MP4s are H.264 CRF 20,
30 FPS with faststart. Existing cached replays keep their original encoding. Billiards uses its own CPU rasterizer:
1080p is drawn natively, avoiding a 4K supersampling intermediate; smaller sizes keep the task's 2x sampling.
Text measurement/wrapping is bounded and cached; scoreboard/caption contents still update normally.

The server sends the latest frame, caps socket buffering, and disconnects a socket after a 2-second blocked write.
The page reconnects after an error, and disconnects its live feed when hidden. Several viewers of the same feed
share a renderer. Different feeds sharing a helper divide its budget; soccer's plain feed counts alongside its program.

## Per-viewer quality selection

The same player and delivery path cover all currently broadcastable sports (13 sport/mode combinations):

| Sport | Modes | Shared quality menu, default 720p, fullscreen auto-hide |
| --- | --- | --- |
| Table tennis | Single / duel | Yes |
| Tennis | Single / duel | Yes |
| Badminton | Single / duel | Yes |
| Baseball | Single / duel | Yes |
| Billiards | Single / duel | Yes |
| Soccer | Single / duel / four-agent team | Yes |

`Program`, `BilliardsProgram` and `SoccerProgram` all inherit the same `Stream.delivery` cache. No per-sport player
copy is needed. Soccer's page legends describe the scene's actual A blue/cyan and B red/purple jerseys, rather
than the orange seat color used for racket duels. Balls and billiards groups retain their sport-specific colors;
baseball's shared field/bat retains its original visuals and uses batting/turn graphics for identity.

Cross-sport checks: `/data/guichi/broadcast-quality-evidence/all-sports-player/`. The real HTTP/player code was
exercised against rendered snapshot fixtures for six sports: four delivered sizes, default 720p, quality switching,
fullscreen idle hide and pointer restore. This is a UI/transport regression, not six new live model matches.
Soccer director regressions and a fresh native 1080p billiards break replay render also pass.


The player has a Quality menu, including in presentation/fullscreen view. New viewers default to 720p;
the selection is stored locally and changes only that viewer's live connection. In fullscreen, the quality and
fullscreen controls fade out after three idle seconds. Pointer movement, touch or keyboard activity restores
them; an open quality menu and keyboard focus within the controls keep them visible. Leaving fullscreen restores
the normal controls. Reduced-motion settings disable the fade animation. A `?quality=720p` page URL can
set an explicit initial choice. The menu applies to the program feed, including automatic replays and transitions.
Archived stills disable this menu; separately requested point MP4s keep their existing encoding.

| Menu tier | Maximum size | FPS ceiling | JPEG quality |
| --- | --- | --- | --- |
| Original | Configured render size (1080p on 8048) | Configured render FPS | Original bytes |
| HD (default) | 1280 × 720 | 24 | 78 |
| SD | 960 × 540 | 20 | 70 |
| Smooth | 640 × 360 | 12 | 60 |

`/program/<id>?quality=original|720p|540p|360p` and `/stream/<id>` accept the same parameter.
Omitting it retains original bytes for existing embeds/recorders; unsupported values return HTTP 400.
Images are never enlarged. The browser disables unavailable size choices when the renderer is configured lower.
Actual FPS can be lower during source pauses, shot holds, or renderer/network limits.

The source scene is still rendered once. Each feed owns a bounded latest-JPEG cache for each smaller tier;
viewers of one tier share conversion and a frame-rate budget. Conversion runs in a request thread outside the
source renderer and its condition lock. Slow sockets keep the existing two-second timeout. No match endpoint,
physics, agent observation camera or simulation setting changes. Smaller tiers reduce outbound bandwidth but
add bounded JPEG conversion CPU work, so this is not a reduction in source GPU resolution.

Checks: `test_broadcast_delivery.py` (cache sharing, no upscale, rate cadence and original passthrough), existing
broadcast quality/smooth regressions; actual HTTP four-tier captures and Chromium switching/persistence/mobile/
visibility checks under `/data/guichi/broadcast-quality-evidence/quality-tiers/` and `quality-tiers-final/`.
In a simultaneous four-tier HTTP sample of the recorded duel (including source holds), original/720p/540p/360p
measured 28.34/22.84/19.05/11.42 FPS and 43.15/10.50/5.47/1.44 Mbps respectively. These are sample-dependent,
not guaranteed bitrates; captions and scene complexity affect JPEG payloads.

## Validation on mll-4090-1, 2026-10-05

A recorded Opus/Astra table-tennis duel was fed through the real director and HTTP handler using `program_harness.py`.
The host was shared; these are local measurements, not a WAN guarantee.

| Measurement | Previous CPU 540p | EGL 1080p + cached text |
| --- | ---: | ---: |
| HTTP FPS, 450-frame sample including holds/transitions | 18.49 | 26.90 |
| Median / p95 inter-frame gap | 53.1 / 61.2 ms | 33.5 / 40.2 ms |
| Mean JPEG payload | 56 KB | 211 KB |
| Helper CPU over the harness run (one core = 100%) | 102.5% | 57.5% |

Active motion reaches 30 FPS. Unchanged frames, source pauses and camera transitions can create longer gaps
(~0.54 s in this sample). A representative GPU frame including two long captions and JPEG encoding measured
16–18 ms for racket sports/baseball and 20 ms for soccer. First billiards cue-line frame: 31 ms at 1080p.
NVIDIA rendering added roughly 210 MiB of GPU memory in the single-feed test.

1080p MJPEG uses about **50 Mbit/s per viewer** at 30 FPS in this caption-heavy sample (soccer can use more).
For limited-bandwidth public distribution, 720p/540p or a subsequent H.264/WebRTC delivery layer is appropriate;
raising rendering FPS alone cannot remove network bottlenecks.

Checks: broadcast smooth/program/billiards/soccer/quality regression scripts; real EGL draws across five MuJoCo sports;
1080p/30 H.264 point replay; Chromium desktop, presentation, mobile, hidden-tab disconnect/reconnect without JS errors;
old/new cached caption output pixel-identical at 540p and 1080p for mixed CJK/Latin text.
Evidence: `/data/guichi/broadcast-quality-evidence/` (recordings, screenshots, HTTP measurements and harness timelines).

To undo only the rendering profile, restart with `BROADCAST_RENDERER=osmesa BROADCAST_QUALITY=540p BROADCAST_FPS=24`.
Original production source was commit `3f669d61d`; retain the existing broadcast cache across restarts.

## Review an entire match, including the director

Use `program_harness.py` with no `--seconds` limit, `--record-fps 30`, and a review-pause cap long enough for a
complete replay package (`--max-pause 25`). Set `--tail 32` to include the final point's replay and post-match picture.
Use an unused `--serve` port and `--http-capture` to record both the program output and the actual MJPEG endpoint.
Both MP4s now use CRF 20 / faststart and the requested recording FPS. A recording's constant frame rate can include
held frames: `rendered_frames.json` measures actual new renders, with mode, camera, transition state and render time.
`timeline.json` includes all point results, frame-aligned video chapters, and the first-frame time offset.

To share exported videos with chapter seeking, copy only public artifacts to a dedicated directory and run:

```sh
python3 scripts/olympiad/broadcast_review.py --directory /path/to/public/review --port 8058
```

This small static review server supports byte-range requests, including open-ended/suffix ranges and HEAD, so a
viewer can seek to a late replay without downloading the entire file. It does not discover or run any matches.

Full-match review evidence (2026-10-05): `/data/guichi/broadcast-quality-evidence/full-match/`, six points,
final score 1–5, 291.63 s program recording and 291.67 s HTTP recording, both native 1920×1080 / 30 FPS.
All files decode without errors; review-page chapter seeking and switching sources passed in Chromium.
Actual replay frames: median interval 33.4 ms, p95 35.7 ms; transitions: median 36.5 ms, p95 45.0 ms.
One 2447.6 ms render-tick outlier occurred at harness time 57.11 s (video hold starts around 00:53).
Two targeted 65-second virtual-time transition checks, with and without the recorded captions, did not reproduce
it. The cause remains unresolved; preserve the original recording and the review page's outlier navigation.
Do not report this recording as uninterrupted 30 FPS merely because the container video frame rate is 30.

## Model identity and audience captions

Duel results use the model names from match metadata, including the model responsible for a referee-reported
fault. Replay titles say which model the camera is behind and show the replayed point number. Same-model duels
append Blue/Orange to distinguish the two entrants. A/B remain internal identifiers.

Player colors belong to sporting equipment, not the whole robot. The spectator's copied models use
blue for the first seat and orange for the second, matching the score and caption bars. Floating robot-name tags
are removed. Original robot materials, grips, shafts and string translucency are preserved.

| Sport | Colored spectator geometry | Preserved |
| --- | --- | --- |
| Table tennis | Paddle blade | White arms, wood handles |
| Tennis | Racket frame and throat | White arms, dark grip, white strings |
| Badminton | Racket frame | White arms, dark handle, metal shaft, white strings |

Exact equipment geometry names plus robot ancestry determine ownership, including reversed cameras. Only
`geom_rgba` RGB and the selected equipment's material binding change. All other geometry materials and alpha,
body mass/inertia, contacts, joint limits, damping, controls and original qpos are checked against the unmodified
models. Evidence: `/data/guichi/broadcast-quality-evidence/live-equipment/check_equipment.py`.

Other sports still use their existing visuals. For subsequent adaptation, prefer team bibs/armbands for soccer,
mallets for air hockey, and small vehicle livery panels for racing/drone events. Shared objects (balls, pucks) must
not imply ownership; turn-based sports need an active-player indicator because equipment can change hands.
These are design directions, not implemented color changes. Check actual wide and reverse shots before claiming
that small equipment colors alone are legible at a sport's broadcast distance.

Recognized command activities become plain-language descriptions. Waits and unrecognized tool calls are omitted;
an observed script invocation does not imply success or a specific tactical improvement. Submitted model reviews
have separate Assessment / Next attempt lines and remain explicitly attributed; referee results are separate.

Optional plain-language paraphrases run in one background worker, never the frame loop:
`BROADCAST_SUMMARIZE_REVIEWS=1`, with `OPENROUTER_API_KEY` provided via the server environment.
The default `BROADCAST_SUMMARY_MODEL` is `openai/gpt-6-luna` on OpenRouter. Only redacted public review text and the
sport are sent, never hidden reasoning. The prompt treats that text as quoted data and forbids invented causes or
success claims. Output shape/length are checked; viewers see “Model review · summary”. This label is important:
a model paraphrase is not independently verified referee commentary.

`BROADCAST_REVIEW_CACHE` defaults to `<broadcast cache>/review-summaries`. Cached paraphrases work offline; an
unavailable key, failed call or pending job falls back to the attributed source review. Requests are deduplicated,
time out after 20 seconds, and do not retry repeatedly in the same server session. Model/wording-version/sport/source
are part of the cache key. The API credential is not passed into render containers.

Evidence: `/data/guichi/broadcast-quality-evidence/audience-summary/` and the review page
`http://mll-4090-1.cs.northwestern.edu:8058/audience-preview.html` show the first point, transitions and both reviews.
