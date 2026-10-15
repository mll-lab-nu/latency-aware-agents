# RoboOlympics rendering: standard (competition) picture and user-selectable re-render engines

Team decision (meeting 2026-10-06): the competition keeps its rendering (cheap, identical for every player); a
render-quality switch exists for spectator material only; demo and blog videos are matches **re-rendered after the
fact from their deterministic replays**. A showcase video may improve lighting, materials, resolution, anti-aliasing
and post-processing. It must not change the scene, its geometry, its colours or the camera ("no demo much prettier
than the actual task").

Follow-up direction (user, 2026-10-06): large-scale matches always run with the standard MuJoCo rendering; the
re-render tool offers the renderer as a **public, documented, user-selectable option**
(`--engine mujoco | mujoco-showcase | cycles | isaac`). Promotion may use the better renderers; because the
option, its guarantees and a tag on every re-rendered frame are public, this is not deceptive. An optional,
off-by-default promo layer (`--decor`) may add surroundings outside the field of play.

Code: `scripts/olympiad/showcase/` (viewer-only, imported by no sim and no task image).

## 1. What was measured

One real Season-1 match per sport, re-run from its replay record (all three reproduce the verifier's final
`state_sha` before anything is drawn):

| Sport | Match (read-only copy) | Window |
| --- | --- | --- |
| Table tennis duel | `rcb-oly-run4/jobs/show-tabletennis-duel-x10-s1-opusA-astraB-…/…__SCqKDUv` (B wins 5:1) | whole match, steps 1–1350 (13.5 s, 1x) |
| Soccer 2v2 team | `rcb-oly-soccer/jobs/socceracc-commson-soccer-team-x10-s1-oracleA-oracleB-…` (oracle vs oracle) | steps 2870–3616, the 15 s before A1's goal (2:0) |
| Tennis duel | `rcb-oly-run4/jobs/r1-tennis-duel-x10-s1-opusA-lunaB-…` | stills only |

Host: mll-4090-1 (one RTX 4090, 32 cores, load ~35 from Season 1 while measuring). Times are per 1920x1080 frame.

### 1a. MuJoCo rasterizer (MuJoCo 3.3.7, the version in our sim images)

| Configuration | Backend | Table tennis | Soccer |
| --- | --- | --- | --- |
| standard scene, 960x540 | EGL (GPU) | 0.6 ms | 0.8 ms |
| standard scene, 1920x1080 | EGL (GPU) | 2 ms | 1.6 ms |
| showcase, 1080p, MSAA 8x, shadows 8192, reflection, no SSAA, + grade | EGL | 57 ms (2 GL + 54 post) | 55 ms |
| showcase, 1080p, rendered at 4K (2x SSAA) + box downsample + grade | EGL | 105–125 ms (13 GL + 34 downsample + 50 post) | 104 ms |
| full clip incl. H.264 encode of 3 outputs (standard, showcase, side-by-side) | EGL | 187 ms/frame (675 frames: 2.1 min) | 135 ms/frame (745 frames: 1.7 min) |
| standard scene, 1080p | OSMesa (CPU, 8 cores) | 55 ms | |
| showcase, 1080p, no SSAA | OSMesa | 253 ms | |
| showcase, 1080p, 2x SSAA | OSMesa | 576 ms | |
| broadcast helper, live feed 960x540 / 1280x720 | OSMesa, 4 cores | standard 24 / 37 ms, showcase 191 / 233 ms | |

The GL work of the showcase look is cheap (2–15 ms on the GPU); the cost is CPU-side (4K readback, downsample,
grade in PIL). OSMesa does not implement MSAA, so on CPU the shadows and edges stay aliased unless SSAA is used.

What the MuJoCo knobs give (all visual-only; tried on the three sports):

| Knob | Effect | Kept |
| --- | --- | --- |
| `visual/global offwidth/offheight` 1920x1080 (3840x2160 for SSAA) | resolution | yes |
| `visual/quality offsamples=8` (MSAA) | clean edges on EGL; ignored by OSMesa | yes |
| 2x supersampling (render 4K, box filter) | removes remaining shimmer of thin lines (net, pitch lines, robot links) | yes (offline) |
| `light castshadow` + `shadowsize=8192` + `bulbradius` (soft PCF) | contact shadows: ball on table, robots on pitch, net on court | yes, on a **directional** light: a spot light's shadow frustum leaves a visible square edge on the floor |
| `material reflectance` on the floor | glossy arena floor with robot reflections (TT) | yes, small (0.08) |
| `material specular/shininess` | painted-metal robots, semi-gloss table | yes |
| procedural detail textures (sport floor, mown grass, acrylic court) | surface detail; mean colour forced equal to the standard colour (error ≤ 0.002) | yes |
| headlight down, key/fill/rim lights | modelling instead of flat camera light | yes |
| `visual/map haze` + `rgba haze` | horizon haze (outdoor sports) | yes (soccer, tennis) |
| fog (`mjRND_FOG`) | flattens a small arena; looks like a different scene | no |
| `material metallic/roughness`, `light intensity` | PBR fields exist in 3.3.7's MjSpec but are only used by the experimental Filament renderer, not shipped in the pip wheel | n/a |
| camera FOV | unchanged by design (the camera is part of the guaranteed-identical view) | no |

Grade (`render.post`): soft bloom of highlights, mild S-curve, saturation x1.06, vignette 0.22.

### 1b. Offline path tracing from the replay

| Path | Setup | Frame time (1080p) | Result |
| --- | --- | --- | --- |
| **Direct export -> Blender 4.2 LTS Cycles** (`showcase/blender/`): geom meshes from the compiled model (`mesh_vert/face`, tessellated primitives), per-frame `geom_xpos/xmat`, the same camera, showcase materials/lights mapped to Principled BSDF / sun / spot | Docker image `rcb-showcase-blender:4.2.23` (Ubuntu 22.04 + official tarball, sha256-pinned, 1.8 GB, builds in ~1 min) | TT 1.03 s, soccer 1.31 s mean over the full clips; 0.90 s / 1.19 s in 60-frame runs alone on the GPU (64 spp, adaptive, OIDN denoise on GPU, OptiX); first frame +3–12 s (kernel load / BVH) | best quality: real soft shadows from every light, bounce light, contact darkening, glossy reflections; geometry/camera verified identical |
| MuJoCo `mujoco.usd` exporter -> Blender USD import | needs `pip install usd-core` in the sim image (not installed) | export 15 ms/frame; Blender import 0.1 s; Cycles 13.8 s for one frame (unoptimised) | **not faithful out of the box**: colours washed out (no colour-space handling), boxes exported as rounded meshes (table legs became cylinders), ball missing, lights at 10000 W blow out the scene (`render_showcase/stills/usd_exporter_blender_tt.png`). Would need patching; the direct exporter is ~250 lines and exact. |
| Blender EEVEE (Next) | same image | not measured: EEVEE needs an OpenGL/EGL context in a headless container; Cycles at ~1 s/frame already fits the use | — |
| **Isaac Sim 4.5 RTX** (the robosuite/RoboCasa Omniverse route; `showcase/isaac/isaac_render.py`): a USD stage built with pxr from the same export (UsdPreviewSurface with linear colours, UsdLux lights, camera with the model's fovy), Replicator, PathTracing 64 spp + OptiX denoiser | `nvcr.io/nvidia/isaac-sim:4.5.0` (15 GB; pulls without an NGC login; driver 580.178 works); first start ~11 min of shader compilation (648 s), later starts 10–20 s with the caches kept on disk | alone on the GPU: TT 1.65 s median (0.07 s when nothing moves), soccer 3.1 s (60-frame runs; Cycles on the same frames: 0.90 s and 1.19 s); with decor TT 2.15 s, soccer 4.0 s (full clips) | quality on par with Cycles (soft shadows, bounce light, denoised); slightly flatter on the table top (spot hot-spot), geometry/camera identical by construction |
| robosuite's USD exporter (`robosuite/utils/usd`, used by `render_dataset_with_omniverse.py`) | robosuite source, usd-core | — | **does not run on our MuJoCo 3.3.7**: it exits at import for MuJoCo >= 3.2 ("use the exporter in the mujoco repository"), and with that gate removed it fails on `model.tex_rgb` (removed from MjModel). It is the predecessor of `mujoco.usd` (same mesh/colour code: sRGB colours written as linear `diffuseColor`, same primitive generation), so it would show the same colour wash-out. |
| Genesis, Filament | own importers / not in MuJoCo's wheel | not tried | Cycles and Isaac cover the need |

Comparison:

| | `mujoco` (competition) | `mujoco-showcase` | `cycles` | `isaac` |
| --- | --- | --- | --- | --- |
| Quality | flat headlight, no shadows, aliasing | shadows, gloss, AA, haze, grade: clearly better, still "MuJoCo-looking" | photographic light (soft shadows, bounce, contact) | photographic, on par with Cycles |
| Frame time 1080p | 2 ms GPU / 55 ms CPU | 0.1 s GPU / 0.6 s CPU | 1–1.3 s (4090) | 1.7–3.1 s (4090) |
| 15 s clip at 50 fps | seconds | ~2 min | 12–16 min | 20–40 min |
| Setup | none | EGL viewer image or the CPU sim image | 1.8 GB image + 1 GPU | 15 GB image + 1 RTX GPU, 11 min first start |
| Reproducibility | bit-exact | pinned image; GPU raster can differ in last bits between drivers | pinned Blender, seeded sampler; denoiser can vary across GPUs | pinned image; RTX/denoiser can vary across GPUs/drivers |
| Fidelity guarantee | — | `looks.same_scene()` before every render | same check on export; model's own meshes; same Director camera | same export as Cycles |

## 2. Chosen design

**Matches: always the standard MuJoCo picture. Re-renders: the user picks the engine** (`rerender.py --engine`).
Suggested use: `mujoco-showcase` for everyday demos (fast, any machine), `cycles` or `isaac` for the few
blog/landing clips (`cycles` needs a 1.8 GB image, `isaac` a 15 GB image and an RTX GPU). All engines read the same
capture and pass the same checks; every re-rendered frame is tagged with its engine.

- `render_quality: standard | showcase`. `standard` is the competition scene unchanged. `showcase` =
  `looks.build(sport, simdir, "showcase")`: the match's own scene files (MjSpec), plus visual-only edits.
- Where the switch exists:
  - `scripts/olympiad/showcase/render.py --quality standard|showcase|both` (offline re-render).
  - `scripts/olympiad/showcase/rerender.py --engine mujoco,mujoco-showcase,cycles,isaac [--decor …]` (one
    command, below; a missing GPU or image stops it before any work, with the engine's requirements).
  - Broadcast: `BROADCAST_RENDER_QUALITY=standard|showcase python3 scripts/olympiad/broadcast.py` passes
    `RENDER_QUALITY` to the render helper, which swaps in the showcase model for duels of supported sports
    (table tennis, tennis). Default `standard`. On the CPU helper it costs ~8x per frame (191 vs 24 ms at
    540p), so it is for demo screenings, not the Season feed; with the EGL helper of the broadcast-quality branch it
    would be cheap.
- **Never** used by a sim: `showcase/` is not copied into any task, no sim imports it, the agents' head/table
  cameras and their scene files are untouched (this branch changes no file under `tasks/`).

### What is guaranteed identical to the competition view

`looks.same_scene(standard_model, showcase_model)` must return no difference, or nothing is drawn. It compares:

- kinematics and geometry: `nq, nv, nbody, njnt, ngeom, nmesh, nmocap`, body tree and poses, masses, joints,
  every geom's type, size, pose, body, group and mesh, all mesh vertices and faces;
- cameras: all model cameras (`cam_*`) and the free-camera field of view (`vis.global.fovy`);
- colours: every untextured geom is drawn with exactly the same RGBA (`effective_rgba`: geom rgba, or material
  rgba when the geom has the default); every re-textured surface keeps its mean colour within 0.02
  (measured: TT floor 0.0006, soccer grass 0.0014, tennis court 0.002, apron 0.0018).

Additionally, by construction: the states are the replay's (capture.py re-runs the match with its own sim code and
refuses unless the final `state_sha` equals the verifier's `replay_check.json`), the broadcast camera is the same
`Director` (TT/tennis: `broadcast/adapters.py` "main"; soccer: the program feed's ball-following side-high camera)
for every look, and the frame times are shared, so a showcase frame shows the exact moment and viewpoint of the
standard frame next to it.

What differs (and only this): lights (positions, colours, shadows), headlight, materials (specular, shininess,
reflectance, detail textures), haze, resolution/MSAA/SSAA, the colour grade, and for Cycles/Isaac the light
transport. Every frame of a non-standard output carries a tag (bottom right) "re-rendered: <engine>".

### Decor (promo only, `--decor`, off by default)

**Promotion material uses no decor (user decision 2026-10-06):** promo stills and clips show the match itself
(default promo renderer: `cycles`); the decor code stays available but off.

`--decor arena` (table tennis) / `--decor stadium` (soccer) adds surroundings for promotional clips: court barriers,
an end wall with LED boards, tiered stands with simple spectator silhouettes, lamp rigs / floodlight masts.
Rules, all checked by `looks.same_scene(standard, model, decor=sport)` before drawing:

- the decor is one extra static body (`decor`, the last body, no joint, no mass) whose geoms are named `decor_*`,
  have no contact, and lie entirely outside the sport's play volume (table tennis |x|<2.6 m, |y|<3.2 m, z<3 m;
  soccer |x|<6.2 m, |y|<4.9 m incl. the penalty benches at |y|=4.3 m, z<4 m);
- every scene body, geom and camera keeps its id and values (the checks above, on the prefix of the arrays);
- only showcase engines get decor; the competition picture (`--engine mujoco`) never does;
- every decor frame is tagged "re-rendered for presentation" in addition to the engine tag.

What decor can still change inside the play area: lighting. In the rasterizer it adds no light (LED/lamp
materials only glow); in the path tracers the stands and walls bounce and block a little light (the table and
pitch colours stay; their shading changes slightly). Cost: crowd geometry (~600–700 geoms) makes the MuJoCo
showcase ~1.3x and the path tracers ~2–3x slower per frame.

## 3. Re-render a match in one command

```sh
PY=/data/guichi/harbor-src/.venv/bin/python
# competition picture + MuJoCo showcase, side by side (GPU 0 via EGL; --gpu none = OSMesa on CPU)
$PY scripts/olympiad/showcase/rerender.py <job>/<trial> --sport tabletennis --out /tmp/tt --dur 13.5 --stills 1.2,5.6
# pick any engines; all four of them:
$PY scripts/olympiad/showcase/rerender.py <job>/<trial> --sport soccer --k0 2870 --k1 3617 \
    --engine mujoco,mujoco-showcase,cycles,isaac --out /tmp/soccer --stills 3,9
# promotional version with the stadium (showcase engines only, tagged "re-rendered for presentation")
$PY scripts/olympiad/showcase/rerender.py <job>/<trial> --sport soccer --k0 2870 --k1 3617 \
    --engine cycles --decor stadium --out /tmp/soccer_promo
```

Outputs: `<sport>[_<decor>]_standard.mp4` / `_showcase.mp4` / `_side_by_side.mp4` (MuJoCo),
`_cycles.mp4`, `_isaac.mp4`, a 2x2 three-way clip when mujoco, mujoco-showcase and cycles are all requested, stills
and per-engine timing reports.

Images: the sport's sim image (capture), `eai-broadcast-soccer-egl:0.1.0` (EGL render; built from
`scripts/olympiad/broadcast/Dockerfile.egl` of the olympiad-broadcast-quality branch, already on mll-4090-1),
`rcb-showcase-blender:4.2.23` (`showcase/blender/Dockerfile`), `nvcr.io/nvidia/isaac-sim:4.5.0` (Isaac caches in
`~/.cache/rcb-showcase-isaac`, the host's `/usr/share/nvidia/nvoptix.bin` is mounted for the OptiX denoiser).
The sim directory must have the match's fingerprint
(default: this checkout's `tasks/<sport>-duel-…/environment/sim`; pass `--sim` for older matches, and for soccer,
whose tasks live on the olympiad-soccer branch: `--sim /data/guichi/rcb-oly-soccer/tasks/soccer-team-…/environment/sim`).

Outputs of this study: `/data/guichi/olympiad_review/render_showcase/` (`tabletennis/`, `soccer/`: standard,
showcase, side-by-side, Cycles, Isaac, three-way and decor (`*_arena_*`, `*_stadium_*`) 1080p50 clips, compare
stills and triptychs, render reports;
`tennis/`: compare stills; `stills/`: the USD-exporter attempt and the broadcast helper in both qualities).

## 4. Findings on the way

- The broadcast adapters for tennis and badminton set state with `mj_kinematics` only, which leaves
  `light_xpos/xdir` at zero: their scene lights do not light the broadcast picture (only the headlight does). The
  showcase path calls `mj_camlight`; the standard broadcast was left as is (out of scope, changing it changes the
  current spectator picture).
- MuJoCo spot-light shadows end at the shadow frustum with a visible hard rectangle on large floors; use a
  directional light for shadows.
- OptiX denoiser: the NVIDIA container runtime does not inject `/usr/share/nvidia/nvoptix.bin`; Blender uses OIDN
  on the GPU instead, Isaac Sim gets the host file mounted read-only.
- Isaac Sim headless on this host: no NGC login needed for the 4.5.0 image, driver 580.178 is fine, ~11 min shader
  compilation on the very first start (kept in the cache volumes).
- robosuite's Omniverse route (`render_dataset_with_omniverse.py`) depends on robosuite's USD exporter, which does
  not run on MuJoCo >= 3.2; the working route for us is our own exporter -> USD stage -> Isaac Sim RTX.

## 5. Open decisions

1. Which engine the public promo clips use (Cycles and Isaac look alike; Cycles is the lighter install).
2. How much brighter the showcase may be: the TT table is ~1.5x the standard luminance on the broadcast camera
   (table green 158 vs 104, colour unchanged). Should we cap per-surface luminance against the standard frame?
3. Whether to add showcase looks for the other sports (badminton, baseball, basketball, billiards...) and a soccer
   adapter on the olympiad branch (soccer broadcast lives on the broadcast-quality branch).
4. Whether to fix the `mj_kinematics`-only lighting in the standard broadcast adapters (changes the live look).
5. Decor content: the current arena/stadium are simple (box stands, capsule spectators); worth a richer set or
   textures for the landing page?
