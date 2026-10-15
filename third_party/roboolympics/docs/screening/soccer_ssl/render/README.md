# SSL overhead cameras: render cost and "is the picture enough" (C line)

Camera code: `sports/soccer_ssl/sim/camera.py`. Report (Chinese, for the lead):
`/data/guichi/olympiad_review/screening/soccer_ssl_camera_zh.md`.

| file | what |
|---|---|
| `stub_scene.py` | stand-in division B scene (field, lines, goals, walls, 10 robots with the standard vision pattern, ball) |
| `real_scene.py` | the same interface on B's `field.build_scene` + A's `bodies/ssl_robot` (`SSL_CAMERA_FIELD=b`) |
| `world_live.py` | the cameras in B's running `World(cameras=True)`, detection against `World.snapshot()` |
| `bench.py` | render + PNG time and memory per observation, one fresh process per configuration |
| `run_bench.sh` | the series of the report (`--cpus` 1 / 2 / 4, MSAA 0 / 4), one container at a time |
| `detect.py` | colour-threshold detector of ball, team, id, position, heading; errors against the true state |
| `pngcost.py` | PNG encoding at zlib level 1 / 3 / 6 |
| `analyze.py` | tables under the match's time model (slowdown, observation rate) |
| `results/` | raw JSON lines and the detection overlays |

All runs: image `ghcr.io/mll-lab-nu/eai-olympiad:0.1.0` (`images/olympiad/VERSION`), `MUJOCO_GL=osmesa`
(the image has no EGL), `docker run --cpus N -e LP_NUM_THREADS=N`, host load 38-60 on 32 cores.

## Recommended configuration

`ssl4` (four cameras at 4.0 m over the quadrants, like the real division B camera bar), 800 x 600 per camera
(6.25 mm per pixel at the robots' tops; ball about 7 px, id dots about 6 px wide), MSAA off, no shadows, venue
lighting (`add_cameras(venue_lighting=True)`: the scene's lights directional, the carpet matte, planes drawn as one
cell). All of these are `camera.py`'s defaults.

Final numbers: B's field and A's final robot (ssl-world 274cd650, merged into this branch), `--cpus 2`, medians of
30 observations, host load 38 (`bench_final2_*`): render 43 ms + PNG 25 ms = 68 ms per observation (4 tiles, 30 KB
PNG), render worker peak 293 MB. At slowdown 10 (a 10 ms control step every 100 ms wall) one render worker can deliver
up to 14.7 observations per wall second = 147 Hz of game time; observing at the real SSL vision rate (75 Hz) keeps it
busy 51 % of the time, at 60 Hz 41 %, at 30 Hz 20 % (slowdown 5: 102 / 82 / 41 %; slowdown 20: 26 / 20 / 10 %).
MSAA on: 150 ms; shadows: 336 ms. `--cpus` 1 / 2 / 4: 69 / 68 / 69 ms.

Detection, B's field + A's robot (40 random scenes, 400 robots, `detect_final.json`): 400 / 400 robots with the right
team and id, position error median 0.6 mm (p95 1.1 mm, 0.18 px), heading median 0.19 deg (p95 0.59 deg); free balls
29 / 29 (median 3.2 mm, p95 5.9 mm); balls resting against a robot's front 9 / 11 (the other 2 are hidden by the
robot's body from the oblique camera, a real effect of overhead cameras).

Live world (`world_live.py`, `results/world_live.json`): B's `World(cameras=True)` for 12 s of game time, robots
driven to random targets, the ball shot every 0.5 s at 2-6.5 m/s, observed every 2 control steps (50 Hz): 600
observations, 6000 / 6000 robots right, 600 / 600 balls found (up to 6.4 m/s; median 4.1 mm, p95 8.0 mm). World step
4.2 ms per control step against 44 ms render (+ about 25 ms PNG) per observation: rendering is about 90 % of the sim's
work at 50 Hz.

B's own lighting (a spot light, the default carpet specular; `SSL_CAMERA_RAW_LIGHT=1`, `detect_final_rawlight.json`):
dim corners and highlights, 56.5 % of the ids right, PNG 512 KB, 64 ms to encode. Plane render grids (MuJoCo draws a
plane as a grid of size[2] cells): 12 ms of a 4-camera observation on B's field (`bench_final_*`: before the
one-cell planes, 59-83 ms render).

Earlier stages (kept for the report): `bench_stub_*` / `detect_stub.json` (stand-in field and robots), `bench_robotA_*`
/ `detect_robotA.json` (A's interim robot 31bd8868 on the stand-in field), `*fieldBwip*` (B's work in progress).

Reproduce (in the image, from the repository root mounted at /srv): `run_bench.sh stub`;
`SSL_CAMERA_FIELD=b bench.py --scene real`, `detect.py --scene real`, `world_live.py`; tables:
`python3 analyze.py --glob 'bench_final2_*.jsonl'`.
