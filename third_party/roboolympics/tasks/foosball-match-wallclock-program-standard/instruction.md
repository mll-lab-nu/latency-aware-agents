# Foosball: program play (match)

You play foosball against a scripted opponent (you are seat `A`). Table football on a physical table: you drive the four motorised rods of one team (goalkeeper rod with 3 figures, 2-rod 2, 5-rod 5, 3-rod 3) by slide and angle targets that the motors follow along a speed- and acceleration-limited path, and see the table only through an overhead camera; one game to 5 goals under the ITSF rules (opening sequence, possession time limits per rod, no spinning); in the `match` format the other team is a scripted computer team, in the `duel` format another player.
This task is generated from `sports/foosball/sport.toml` (task `foosball-match-wallclock-program-standard`).

## Time
The world runs in real time at 1/S of real speed (S = `timing.slowdown` in `game spec`); your thinking costs game time. At the end of every unit the world freezes for the reviews.

Whenever the world freezes for a review (after every unit: a point, a shot, a heat; see `game spec`), send `game ready "WHY" "PLAN"` (a short review) before the next unit starts.
The match is graded when your session ends: stay until it is over. **If your session ends before the match is over, you lose the match** (a valid result, success 0), also if you never sent `game start`.

## Write a program
You may write a program that observes, measures in the images, computes your next action and sends it, without a
new model response for each action. You stay in the loop: watch its log, and stop, fix and restart it whenever
needed. Only you send `game start` and `game ready`.

- Use only the camera images, the service's replies and this document; no simulator internals or hidden state.
- One action at a time: an action sent before the previous one has returned is refused (HTTP 429).
- Your container: Python 3.12 with numpy, SciPy and Pillow; no internet during the match. Keep your scripts, logs
  and notes in `/app/output/`.

## Interface
Use the `game` command, or the same calls as HTTP with JSON at `http://sim:8000/<SEAT>/` (your seat is `$SEAT`):

| `game` command | HTTP | |
| --- | --- | --- |
| `game spec` | `GET /spec` | the actions (types, fields, ranges), the cameras with their intrinsics, the rule constants |
| `game status` | `GET /status` | the current state, no image |
| `game start` | `POST /start {}` | you are ready; the match begins when every seat has started |
| `game wait DT` | `POST /wait {"dt": DT}` | let DT game seconds pass (0 < DT <= 1) |
| `game observe` | `POST /observe {}` | the state and the camera images (one PNG of tiles: `observation.tiles`) |
| `game act JSON` | `POST /act JSON` | one action |
| `game ready "WHY" "PLAN"` | `POST /ready {"why":"...","plan":"..."}` | your review after each of your actions |

Errors: 400 = an invalid action (nothing happens), 409 = not now (not your turn, or a review is due), 429 = one
action in flight, or more than 20 requests per second or 4 open requests from your seat (empty body: retry more slowly).

## Rules
- table football on a physical table: you drive the four motorised rods of one team (goalkeeper rod with 3 figures, 2-rod 2, 5-rod 5, 3-rod 3) by slide and angle targets that the motors follow along a speed- and acceleration-limited path, and see the table only through an overhead camera; one game to 5 goals under the ITSF rules (opening sequence, possession time limits per rod, no spinning); in the `match` format the other team is a scripted computer team, in the `duel` format another player.
- `real_time`: this sport is real time (where the generic text above says otherwise, this line holds): `game wait DT` (POST /wait {"dt": DT}, 0 < DT <= 1) lets the world run DT game seconds; `game ready "WHY" "PLAN"` is due only after each rally, not after each action; acts are never out of turn, one in flight per seat (it may address several rods)
- `action`: game act '{"type": "rods", "rods": {"midfield": {"slide": 0.03, "angle": 1.1, "omega": 60}, "goalie": {"slide": -0.05}}}' (POST /act with that body; "type" may be left out): new targets for any of your rods goalie, defence (2-rod), midfield (5-rod) and attack (3-rod); rods you do not name keep their targets. slide (m) moves the rod along your local v (0 = centre; within +-0.11 goalie, +-0.19 defence, +-0.06 midfield, +-0.11 attack); angle (rad) turns it (0 = figures hanging straight down, + = feet swing towards the opponent's goal, up to +-2.6; 1.5 is about level, so a ball passes underneath); optional omega (rad/s, at most 80) and speed (m/s, at most 2.5) cap the rod's angular and sliding speed for this target. Each rod follows a smooth path to the target (sliding at most 2.5 m/s and 40 m/s^2, turning at most 80 rad/s and 4000 rad/s^2) with limited force (60 N) and torque (5 N m); a command reaches the motors 2 control steps (20 ms of game time) after the control step that takes it (taken_step, effect_step in the reply) and holds until replaced; the rods cannot make a full turn (stops at +-2.8 rad). Out of range: HTTP 400
- `format`: first to 5 goals wins; every goal, possession-time foul, incomplete opening sequence, illegal spin, dead ball or ball off table ends a rally: the world freezes for the reviews, every rod returns to slide 0, angle 0, and the next rally starts with the restart; the game also ends after 240 s of play (frozen time does not count) or 40 rallies
- `kickoff`: a rally starts with the ball at rest beside the central figure of the restart rod. The first kick-off is match.first_kickoff's (game spec); after a goal the team that conceded kicks off. Before the ball may leave the restart rod (the 5-rod, or the goalkeeper rod and 2-rod together), the restarting team must complete the opening sequence: a move by one figure, a contact by another figure, then a contact by any figure other than that second one (restart.opening_complete in the state). If the ball leaves first, the opponent gets the ball on its 5-rod (incomplete_opening). The world is not frozen during the opening
- `goal`: the ball wholly over the goal line scores 1 point for the other team, whoever touched it last
- `possession`: while the ball is within reach of a rod (a figure of it can touch the ball: within 0.077 m of the rod along u), that rod's possession clock runs; it restarts when the ball leaves the rod's reach. The goalkeeper rod and 2-rod count as one (the goalie rod). Limits: goalie rod 15 s, 5-rod 10 s, 3-rod 15 s. Over the limit is a foul: after the goalie rod or the 5-rod the opponent gets the ball on its 5-rod, after the 3-rod on its goalie rod. A foul gives no point
- `spinning`: a transition (the ball leaving a rod's reach) after the rod turned more than 360 degrees before or after its contact is a foul (the opponent gets the ball on its 5-rod; a goal scored that way does not count); the rods' stops make a full turn impossible
- `dead_ball`: a ball at rest for 1 s that no rod can reach is put back into play: between the two 5-rods by a kick-off (by the team that conceded the last goal, else the first kick-off's team), behind a 5-rod by the goalie rod of that half
- `ball_off_table`: it goes to the goalie rod of the team that did not touch it last (a transparent cover 4 cm above the playfield normally keeps the ball on the table)
- `match_format`: match: you are seat A (blue) against a scripted computer team (seat B, red) of level RCB_LEVEL 1-4 (match.opponent_level in game spec): it plays the same player policy as the reference player on the true ball position a reaction delay ago and never reads your commands; level 1 reacts after 140 ms, slides at 0.6 m/s, kicks at 20 rad/s and does not shoot from its 5-rod and 3-rod; level 2 after 90 ms, 1 m/s, 35 rad/s; level 3 after 60 ms, 1.6 m/s, 55 rad/s, aims past your figures, passes, lifts its rods out of its shots and splits the goal between goalkeeper and 2-rod; level 4 the same after 30 ms at full speed. Success = winning the game
- `duel_format`: duel: seat A (blue, defending the goal at world x = -0.60) against seat B (red, defending x = +0.60), identical rods and motors; FIRST_PLAYER has the first kick-off
- `observation`: the state: your own rods (slide, angle, their velocities and the command at the motors), the score, fouls, the time of play and of the rally, the play time left, the referee's restart (which team and rod puts the ball into play, whether its opening sequence is complete) and the rallies that ended (last_rally, rallies: how, who scored or fouled, the restart); no ball or opponent coordinates and no seed. The image: one 384 x 240 RGB tile from the fixed overhead camera straight above the table centre at 1.45 m (32 degree vertical field of view; game spec camera / cameras), turned so that your own goal is on the left and the image's right and up are your local +u and +v. The ball is orange, seat A's figures blue, seat B's red, the rods grey; figures and rods can hide the ball
- `world`: metres, seconds. World frame: x along the table (seat A's goal at x = -0.60, seat B's at x = +0.60), y across, z up, playfield z = 0 with its centre at (0, 0). Your local frame (the image, your rods, game spec geometry): u towards the opponent's goal, v to the left when you look along u (seat A: u = x, v = y; seat B: u = -x, v = -y)
- `table`: playfield 1.20 x 0.68 m between side walls with rounded fillets (radius 0.02 m), goal mouths 0.20 m wide; ball radius 0.0175 m, 24 g, it rolls with little resistance (about 0.1 m/s^2), bounces off the walls and is stopped by a figure it runs into; a kick sends it off at about the foot's speed, at an angle when the foot hits it off centre. Rods 0.09 m above the playfield at u = -0.525 (your goalkeeper rod), -0.375 (2-rod), -0.075 (5-rod), 0.225 (3-rod), the opponent's at the negated u; figures at slide 0 (v): goalkeeper and 3-rod -0.20, 0, 0.20; 2-rod -0.12, 0.12; 5-rod -0.24, -0.12, 0, 0.12, 0.24; a foot is a capsule along u (radius 0.012 m, half length 0.014 m) 0.075 m below the rod. Physics 1000 Hz, control 100 Hz (10 ms steps)
- Winner: the first player to score 5 goals wins; at the cap (240 s of play or 40 rallies) the player with more goals wins; at equal goals the player who scored the last goal; if nobody has scored, the player with the ball less time in its own half; last resort (not from the rulebook): the player who did not have the first kick-off. There are no draws.
- Deviation D1 (rulebook 3): one game to 5 points instead of a match of several games (the points and games targets are set by the event)
- Deviation D2 (rulebook 3, 7.1): a safety cap of 240 s of play or 40 rallies (the rulebook has no time limit); then the player with more goals wins, at equal goals the scorer of the last goal, at 0:0 the player with the ball less time in its own half, and if that is equal the player who did not have the first kick-off (the last resort; no draws)
- Deviation D3 (rulebook 4, 5.1): the initial flip is the run-time parameter FIRST_PLAYER (who has the first kick-off); the referee places the ball at rest beside the central figure of the restart rod (its side and a few millimetres along the rod from the seed) and play starts when the reviews are in, instead of the ready protocol
- Deviation D4 (rulebook 5.2, 5.3): no delay-of-game rule; the restart rod's possession clock already runs during the opening sequence (it starts with the possession clock figure)
- Deviation D5 (rulebook 6, 7.1.1.1, 7.1.2, 17, 18): fouls are always given as the possession award (5-rod or goalie rod), without the non-offending team's option to continue, warnings, penalty shots or time-outs; the pause between points is the review
- Deviation D9 (rulebook 7.2, 11, 12): the passing and wall rules and playing the ball dead are not judged (they need a referee's judgement of a stopped, pinned or intentional ball); the dead-ball restart for a ball at rest that no rod can reach is judged

## Geometry and physics
Everything you need is in `game spec`: the world frame, the release point, the cameras (pose, field of view, size,
`focal_px`, `principal_point_px`; pinhole: u = cx + f * x_right / depth, v = cy - f * y_up / depth) and the
execution scatter. The physics engine is mujoco (CPU, 1 kHz rigid bodies: a rolling, bouncing ball and 2 x 4 rods driven by force- and torque-limited motors).
