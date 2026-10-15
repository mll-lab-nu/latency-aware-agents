# Soccer numbered jerseys (soccer-2v2-2)

Team A wears blue and team B red. Every trunk carries a bold white number on both faces, unique within its team.
The public `game spec.player_identity` roster maps seats to teams and numbers, without positions or detections.
The rules tell players to read visible numbers, track players, or coordinate over the team channel when numbers
are occluded. The same scene is used by head cameras and spectator rendering. Numbers support 1–99; the match,
referee, controls, seats and message routing remain 2v2. A future 6v6 task needs those systems extended separately.

The task revision and verifier revision are now soccer-2v2-2. Source fingerprints also change. Older episodes must
be replayed and graded with their original revision; their published scores are not new-version evaluations.
The three generated task directories and their manifest digests have been refreshed.

Validation:

- `scripts/olympiad/soccer/test_identity.py` in the soccer Docker image: team palette and roster, front/back numbers
  1/2/6/12/99, labels have no collision contribution. Mass and inertia arrays are unchanged; 200 control steps with
  walking commands produce bit-identical qpos, qvel and final state hashes with and without the visual numbers.
- New-version service smoke: each of four seats receives its own state and the public identity spec. A 350-step,
  28-command episode replays in a separate process with the identical state hash. Changing its revision to the
  previous revision is rejected. This is a replay check, not a complete model match or win-rate evaluation.
- `ci_checks/run_all.sh` on the three soccer tasks passes, including generator consistency, protocol, manifest and
  verifier isolation. The existing repository-wide legacy client-image warning remains unrelated to this change.
- The camera-only reference player detects the orange ball and exchanges positions via team messages; it does not
  classify the previous cyan/purple jerseys. No reference policy was changed.
- Rendered real recorded actions using the new scene: both head cameras and spectator frames show numbered team
  jerseys. Preview: http://mll-4090-1.cs.northwestern.edu:8058/player-identity-v2/ . These are reconstructed historical
  actions, not decisions newly made from the changed observation images.

Local evidence: `/data/guichi/broadcast-quality-evidence/soccer-identity-ci.log`, `soccer-identity-replay/`,
`sport-stills/check_soccer_revision.py`, and `demo/player-identity-v2/`.
