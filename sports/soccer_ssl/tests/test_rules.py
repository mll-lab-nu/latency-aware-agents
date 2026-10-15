"""One test per rule of the automatic referee (sim/referee.py), each in the physical world with the robots of
field.default_body(). Rulebook: RoboCup SSL rules 2026-09-19 (ssl-rules commit 2ff4be0c); sections in the names'
docstrings. Run: python3 sports/soccer_ssl/tests/test_rules.py [name-filter ...]."""

from __future__ import annotations

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _util import (
    Carrot,
    F,
    ball,
    events,
    place,
    run,
    run_all,
    running,
    sideline,
    target,  # noqa: E402
    world,
)

B1, B2, B0 = ("blue", 1), ("blue", 2), ("blue", 0)
Y1, Y2, Y0 = ("yellow", 1), ("yellow", 2), ("yellow", 0)
FRONT = 0.09 + F.BALL_R
KICK_DX = None  # the robot's centre -> the ball's resting point (set from the first world)


def _kick_dx():
    global KICK_DX
    if KICK_DX is None:
        w = world()
        place(w, B2, 0.0, 0.0, 0.0)
        KICK_DX = w.kick_point(B2)[0]
    return KICK_DX


def kick_from(w, rid, x, y, yaw, speed, mode="flat", run_s=1.5, armed=0.3):
    """robot rid at (x, y) facing yaw with the ball touching its front; kick and run."""
    place(w, rid, x, y, yaw)
    ball(w, *w.kick_point(rid))
    running(w)
    w.apply(rid[0], rid[1], "kick", dict(speed=speed, mode=mode, armed_for=armed))
    run(w, run_s)


def only(w, *keep):
    sideline(w, keep)


# ------------------------------------------------------------------------------------------------ goals (§7)
def test_goal_scored_and_kickoff_for_the_other_team():
    """§7 a goal: the ball fully enters the goal between the posts; §5.3.2 the other team kicks off."""
    w = world()
    only(w, B2)
    kick_from(w, B2, 3.0, 0.0, 0.0, 3.0, run_s=1.5)
    g = events(w, "GOAL")
    assert len(g) == 1 and g[0]["team"] == "blue", g
    assert w.referee.score == {"blue": 1, "yellow": 0}
    assert w.referee.next_cmd == ("KICKOFF", "yellow")
    run(w, 3.5)
    assert w.referee.state == "PREPARE_KICKOFF" and w.referee.for_team == "yellow"
    assert math.hypot(*w.ball.pos[:2]) < 1e-6  # the referee placed the ball on the centre mark


def test_goal_invalid_when_ball_was_higher_than_015():
    """§7: no goal if the ball was higher than 0.15 m after the scoring team's last touch (a chip)."""
    w = world()
    only(w, B2)
    kick_from(w, B2, 3.72, 0.0, 0.0, 2.8, mode="chip", run_s=1.5)
    inv = events(w, "INVALID_GOAL")
    assert inv and inv[0]["reason"] == "BALL_TOO_HIGH", events(w, "INVALID_GOAL", "GOAL", "BALL_LEFT_FIELD_GOAL_LINE")
    assert w.referee.score["blue"] == 0
    assert w.referee.next_cmd == ("FREE_KICK", "yellow")  # continued as a goal-line crossing: goal kick


def test_early_termination_at_10_goals():
    """§4.4.3: the match ends when a team has 10 goals and leads by more than one."""
    w = world()
    w.referee.score["blue"] = 9
    only(w, B2)
    kick_from(w, B2, 3.0, 0.0, 0.0, 3.0)
    end = events(w, "MATCH_END")
    assert end and end[0]["how"] == "EARLY_TERMINATION" and end[0]["winner"] == "blue", end
    assert w.referee.stage == "POST_GAME"


# ------------------------------------------------------------------------------------------------ ball out (§6)
def test_throw_in():
    """§6.1.1: over the touch line -> free kick for the other team 0.2 m inside the touch line where it crossed."""
    w = world()
    only(w, B2)
    kick_from(w, B2, 1.0, 2.5, math.pi / 2, 2.0, run_s=0.8)
    e = events(w, "BALL_LEFT_FIELD_TOUCH_LINE")
    assert e and e[0]["team"] == "blue", e
    R = w.referee
    assert R.next_cmd == ("FREE_KICK", "yellow")
    assert abs(R.placement[1] - 2.8) < 1e-9 and abs(R.placement[0] - 1.0) < 0.1, R.placement
    place(w, B2, 1.0, 1.5, 0.0)  # the defender keeps 0.5 m away: the free kick is taken
    run(w, 4.0)
    assert events(w, "FREE_KICK") and events(w, "FREE_KICK")[-1]["team"] == "yellow"
    assert math.hypot(w.ball.pos[0] - R.placement[0], w.ball.pos[1] - R.placement[1]) < 0.01


def test_goal_kick():
    """§6.2.1: over the goal line of the team that did not touch last -> free kick 1 m from the goal line, 0.2 m from
    the touch line."""
    w = world()
    only(w, B2)
    kick_from(w, B2, 3.0, 1.5, 0.0, 3.0, run_s=1.0)
    assert events(w, "BALL_LEFT_FIELD_GOAL_LINE")
    assert w.referee.next_cmd == ("FREE_KICK", "yellow")
    assert w.referee.placement == (3.5, 2.8), w.referee.placement


def test_corner_kick():
    """§6.2.2: over its own goal line by the team that touched last -> free kick 0.2 m from both lines."""
    w = world()
    only(w, Y1)
    kick_from(w, Y1, 3.0, 1.5, 0.0, 3.0, run_s=1.0)  # yellow defends the goal at +x in the first half
    assert events(w, "BALL_LEFT_FIELD_GOAL_LINE")
    assert w.referee.next_cmd == ("FREE_KICK", "blue")
    assert w.referee.placement == (4.3, 2.8), w.referee.placement


def test_aimless_kick():
    """§6.2.3 (Division B): touched in the own half, over the halfway line and out over the opponent goal line without
    another touch -> free kick for the other team where the ball was kicked."""
    w = world()
    only(w, B2)
    kick_from(w, B2, -1.0, 1.5, 0.0, 6.0, run_s=2.0)
    e = events(w, "AIMLESS_KICK")
    assert e and e[0]["team"] == "blue", events(w, "AIMLESS_KICK", "BALL_LEFT_FIELD_GOAL_LINE")
    assert w.referee.next_cmd == ("FREE_KICK", "yellow")
    px, py = w.referee.placement
    assert abs(px - (-1.0 + _kick_dx())) < 0.05 and abs(py - 1.5) < 0.05, w.referee.placement


def test_boundary_crossing():
    """§8.4.1.6: a ball kicked over the field boundary (the walls) is a foul; free kick from the kick location."""
    w = world()
    only(w, B2)
    kick_from(w, B2, 1.0, 2.6, math.pi / 2, 4.0, mode="chip", run_s=1.5)
    assert events(w, "BALL_LEFT_FIELD_TOUCH_LINE")
    f = events(w, "BOUNDARY_CROSSING")
    assert f and f[0]["team"] == "blue" and w.referee.fouls["blue"] == 1, f
    assert w.referee.next_cmd == ("FREE_KICK", "yellow")
    px, py = w.referee.placement  # the kick location (inside the valid area)
    assert abs(px - 1.0) < 0.05 and abs(py - (2.6 + _kick_dx())) < 0.03, w.referee.placement


# ------------------------------------------------------------------------------------------------ fouls (§8)
def test_ball_speed_over_65_is_a_foul_and_cancels_the_goal():
    """§8.4.2.2: a robot must not accelerate the ball above 6.5 m/s (non-stopping foul); §7: no goal within 2 s of a
    non-stopping foul of the scoring team."""
    w = world()
    only(w, B2)
    kick_from(w, B2, 1.0, 0.0, 0.0, 7.5, run_s=1.5)
    f = events(w, "BOT_KICKED_BALL_TOO_FAST")
    assert f and f[0]["team"] == "blue" and f[0]["speed"] > 6.5, f
    inv = events(w, "INVALID_GOAL")
    assert inv and inv[0]["reason"] == "NON_STOPPING_FOUL", events(w, "GOAL", "INVALID_GOAL")
    w2 = world()
    only(w2, B2)
    kick_from(w2, B2, 1.0, 0.0, 0.0, 5.5, run_s=1.5)
    assert not events(w2, "BOT_KICKED_BALL_TOO_FAST") and events(w2, "GOAL")


def test_excessive_dribbling():
    """§8.4.1.8: dribbling more than 1 m from where the contact began -> stopping foul, free kick for the other team
    from the start of the dribbling."""
    w = world()
    only(w, B2)
    place(w, B2, 0.0, 0.5, 0.0)
    x0, y0 = w.kick_point(B2)
    ball(w, x0, y0)
    running(w)
    w.apply("blue", 2, "dribble", dict(rpm=5000))
    run(w, 2.5, Carrot(w, B2, 1, 0, 0.8))
    f = events(w, "BOT_DRIBBLED_BALL_TOO_FAR")
    assert f and f[0]["team"] == "blue", f
    assert abs(f[0]["start"][0] - x0) < 0.03, (f, x0)
    assert w.referee.next_cmd == ("FREE_KICK", "yellow")


def test_dribbling_under_1m_is_legal():
    """§8.4.1.8: 0.7 m of dribbling is legal."""
    w = world()
    only(w, B2)
    place(w, B2, 0.0, 0.5, 0.0)
    ball(w, *w.kick_point(B2))
    running(w)
    w.apply("blue", 2, "dribble", dict(rpm=5000))
    c = Carrot(w, B2, 1, 0, 0.7)
    run(w, 1.0, c)
    run(w, 1.0)
    assert not events(w, "BOT_DRIBBLED_BALL_TOO_FAR")
    assert w.ball.pos[0] > 0.6


def test_double_touch():
    """§8.2: the kicker of a free kick touches the ball again (here: stays in contact over more than 0.05 m) before
    anyone else -> stop, free kick for the other team."""
    w = world()
    only(w, B2)
    place(w, B2, 0.0, 0.5, 0.0)
    kx, ky = w.kick_point(B2)
    ball(w, kx + 0.002, ky)
    w.referee.force("FREE_KICK", "blue", ball=w.ball.pos)
    run(w, 1.2, Carrot(w, B2, 1, 0, 0.5))
    assert events(w, "BALL_IN_PLAY")[0]["robot"] == "blue2"
    e = events(w, "ATTACKER_DOUBLE_TOUCHED_BALL")
    assert e and e[0]["team"] == "blue", e
    assert w.referee.next_cmd == ("FREE_KICK", "yellow")


def test_crash_faster_robot_fouls():
    """§8.4.2.3: projected speed difference > 1.5 m/s at the collision: the faster robot fouls (non-stopping)."""
    w = world()
    only(w, B2, Y2)
    place(w, B2, -1.5, 0.5, 0.0)
    place(w, Y2, 0.5, 0.5, math.pi)
    ball(w, 0.0, -2.0)
    running(w)
    c = Carrot(w, B2, 1, 0, 2.5)

    def until_contact(w):  # drive at 2.5 m/s until the first contact, then stop pushing
        if events(w, "BOT_CRASH_UNIQUE", "BOT_CRASH_DRAWN") or w.robot_pose(B2)[0][0] > 0.2:
            p, _ = w.robot_pose(B2)
            target(w, B2, p[0] - 0.05, p[1], 0.0)
        else:
            c(w)

    run(w, 2.0, until_contact)
    f = events(w, "BOT_CRASH_UNIQUE")
    assert len(f) == 1 and f[0]["team"] == "blue" and f[0]["crash_speed"] > 1.5, events(
        w, "BOT_CRASH_UNIQUE", "BOT_CRASH_DRAWN"
    )
    assert w.referee.state == "RUNNING" and w.referee.fouls == {"blue": 1, "yellow": 0}


def test_crash_slow_contact_is_legal():
    """§8.4.2.3: a contact at 0.8 m/s is not a crash."""
    w = world()
    only(w, B2, Y2)
    place(w, B2, -1.0, 0.5, 0.0)
    place(w, Y2, 0.5, 0.5, math.pi)
    ball(w, 0.0, -2.0)
    running(w)
    run(w, 2.0, Carrot(w, B2, 1, 0, 0.8))
    assert not events(w, "BOT_CRASH_UNIQUE", "BOT_CRASH_DRAWN")


def test_crash_both_foul_when_speeds_similar():
    """§8.4.2.3: projected difference > 1.5 m/s and speeds within 0.3 m/s of each other: both robots foul."""
    w = world()
    only(w, B2, Y2)
    place(w, B2, -1.2, 0.5, 0.0)
    place(w, Y2, 1.2, 0.5, math.pi)
    ball(w, 0.0, -2.0)
    running(w)
    run(w, 1.5, Carrot(w, B2, 1, 0, 1.3), Carrot(w, Y2, -1, 0, 1.3))
    f = events(w, "BOT_CRASH_DRAWN")
    assert {e["team"] for e in f} == {"blue", "yellow"}, events(w, "BOT_CRASH_UNIQUE", "BOT_CRASH_DRAWN")


def test_pushing():
    """§8.4.1.2 (S1 measurable): robot pushes an opponent that gives way more than 0.3 m -> stopping foul."""
    w = world()
    only(w, B2, Y2)
    place(w, B2, 0.0, 0.5, 0.0)
    place(w, Y2, 0.181, 0.5, math.pi)
    ball(w, 0.0, -2.0)
    running(w)

    def gives_way(w):
        p, _ = w.robot_pose(Y2)
        target(w, Y2, p[0], p[1], p[2])

    run(w, 3.5, Carrot(w, B2, 1, 0, 0.5), gives_way)
    f = events(w, "BOT_PUSHED_BOT")
    assert f and f[0]["team"] == "blue" and f[0]["victim"] == "yellow2", f
    assert not events(w, "BOT_CRASH_UNIQUE", "BOT_CRASH_DRAWN")
    assert w.referee.next_cmd == ("FREE_KICK", "yellow")


def test_ball_holding():
    """§8.4.1.3 (S1 measurable): two robots of a team close around the ball for 1 s -> stopping foul."""
    w = world()
    only(w, B1, B2)
    place(w, B1, 0.0, 0.13, -math.pi / 2)
    place(w, B2, 0.0, -0.13, math.pi / 2)
    running(w, 0.0, 0.0)
    run(w, 1.5)
    f = events(w, "BOT_HELD_BALL_DELIBERATELY")
    assert f and f[0]["team"] == "blue", f


def test_multiple_defenders_gives_a_penalty_kick():
    """§8.4.1.5: a robot other than the keeper touches the ball while entirely inside its own defense area -> penalty
    kick for the other team (the foul counter does not increase)."""
    w = world()
    only(w, B1)
    place(w, B1, -4.0, 0.0, 0.0)
    ball(w, -3.6, 0.0, vel=(-1.0, 0.0, 0.0))
    running(w)
    run(w, 1.0)
    e = events(w, "DEFENDER_IN_DEFENSE_AREA")
    assert e and e[0]["team"] == "blue" and e[0]["foul"] is False, e
    assert w.referee.fouls["blue"] == 0
    assert w.referee.next_cmd == ("PENALTY", "yellow")
    assert w.referee.placement == (1.5, 0.0)  # the mark 6 m from the goal at -4.5


def test_keeper_may_touch_in_own_defense_area():
    """§8.4.1.5: the keeper may."""
    w = world()
    only(w, B0)
    place(w, B0, -4.0, 0.0, 0.0)
    ball(w, -3.6, 0.0, vel=(-1.0, 0.0, 0.0))
    running(w)
    run(w, 1.0)
    assert not events(w, "DEFENDER_IN_DEFENSE_AREA") and w.referee.state == "RUNNING"


def test_attacker_touched_ball_in_opponent_defense_area():
    """§8.4.2.1: a robot partially inside the opponent defense area touches the ball -> non-stopping foul."""
    w = world()
    only(w, B2)
    place(w, B2, 3.5, 0.5, math.pi)
    ball(w, 3.0, 0.5, vel=(1.5, 0.0, 0.0))
    running(w)
    run(w, 1.0)
    f = events(w, "ATTACKER_TOUCHED_BALL_IN_DEFENSE_AREA")
    assert f and f[0]["team"] == "blue", f
    assert w.referee.state == "RUNNING"


def test_keeper_held_ball():
    """§8.4.1.7: the ball in a defense area for more than 10 s (Division B) -> foul of that team, free kick."""
    w = world()
    only(w, B0)
    place(w, B0, -4.3, 0.8, 0.0)
    running(w, -4.0, 0.3)
    run(w, 10.5)
    f = events(w, "KEEPER_HELD_BALL")
    assert f and f[0]["team"] == "blue", events(w, "KEEPER_HELD_BALL", "NO_PROGRESS_IN_GAME")
    assert w.referee.next_cmd[0] == "FREE_KICK" and w.referee.next_cmd[1] == "yellow"
    px, py = w.referee.placement
    assert not F.point_in_rect(px, py, F.defense_area(-1)) and abs(px) <= 2.5 or abs(py) >= 2.0


def test_no_progress_force_start():
    """§8.1: no progress for 10 s while both teams may play -> stop, then a force start."""
    w = world()
    only(w)
    running(w, 0.0, 1.0)
    run(w, 10.5)
    assert events(w, "NO_PROGRESS_IN_GAME") and w.referee.next_cmd == ("FORCE_START", None)
    run(w, 4.0)
    assert events(w, "FORCE_START") and w.referee.state == "RUNNING"


def test_stop_speed():
    """§8.4.3.2: faster than 1.5 m/s in stop after a 2 s grace -> foul (once per robot and stoppage)."""
    w = world()
    only(w, B2, Y2)
    place(w, B2, -3.0, 2.0, 0.0)
    place(w, Y2, 3.0, -1.0, math.pi)
    ball(w, 0.0, -2.0)
    w.referee.stop(("FREE_KICK", "yellow"), (0.0, -2.0))
    run(w, 2.8, Carrot(w, B2, 1, 0, 2.0), Carrot(w, Y2, -1, 0, 1.2))
    f = events(w, "BOT_TOO_FAST_IN_STOP")
    assert len(f) == 1 and f[0]["robot"] == "blue2" and f[0]["t"] > 2.0, f


def test_too_close_to_opponent_defense_area_in_stop():
    """§8.4.1.1: in stop, closer than 0.2 m to the opponent defense area after 2 s -> foul, free kick other team."""
    w = world()
    only(w, B2)
    place(w, B2, 3.3, 0.0, 0.0)
    ball(w, 0.0, -1.0)
    w.referee.stop(("FREE_KICK", "blue"), (0.0, -1.0))
    run(w, 2.5)
    f = events(w, "ATTACKER_TOO_CLOSE_TO_DEFENSE_AREA")
    assert f and f[0]["team"] == "blue", f
    assert w.referee.next_cmd == ("FREE_KICK", "yellow")


def test_defender_too_close_to_free_kick():
    """§8.4.3.1: closer than 0.5 m to the ball during an opponent free kick -> foul; §5.4 the free kick is in play
    10 s later if nobody kicks."""
    w = world()
    only(w, B2)
    place(w, B2, 0.4, 0.0, math.pi)
    ball(w, 0.0, 0.0)
    w.referee.force("FREE_KICK", "yellow", ball=w.ball.pos)
    run(w, 0.5)
    f = events(w, "DEFENDER_TOO_CLOSE_TO_KICK_POINT")
    assert len(f) == 1 and f[0]["team"] == "blue", f
    place(w, B2, 1.5, 0.0, math.pi)
    run(w, 10.0)
    e = events(w, "BALL_IN_PLAY")
    assert e and e[0]["by"] == "timeout"


# ------------------------------------------------------------------------------------------------ restarts and stages
def test_kickoff_and_ball_in_play():
    """§5.3.2: robots in their own halves -> 2 s later the kick-off; §5.4 the ball is in play after moving 0.05 m; the
    clock (§4.4.1) runs only from the kick-off on."""
    w = world()
    R = w.referee
    assert R.state == "PREPARE_KICKOFF" and R.for_team == "blue"
    run(w, 2.1)
    assert R.state == "KICKOFF" and R.play_time < 0.15
    w.apply("blue", 1, "kick", dict(speed=2.0, armed_for=1.0))
    run(w, 0.8, Carrot(w, B1, 1, 0, 0.5))
    e = events(w, "BALL_IN_PLAY")
    assert e and e[0]["by"] == "moved" and e[0]["robot"] == "blue1", e
    assert R.state == "RUNNING"


def test_kickoff_robots_out_of_position_are_moved():
    """§5.3.2 + S1: a robot in the opponent half 10 s into the kick-off preparation is moved, one foul."""
    w = world()
    place(w, Y2, -1.0, 1.0, math.pi)  # yellow's own half is x > 0 in the first half
    run(w, 10.2)
    f = events(w, "ROBOTS_OUT_OF_POSITION")
    assert f and f[0]["team"] == "yellow" and f[0]["robots"] == ["yellow2"], f
    assert w.robot_pose(Y2)[0][0] > 0.0
    assert events(w, "KICKOFF")


def test_half_time_switches_ends():
    """§4.4.1 two halves; the teams change ends at half time; the second half starts with the other team's kick-off."""
    w = world(half_s=4.0)
    run(w, 7.0)
    assert events(w, "HALF_TIME")
    R = w.referee
    assert R.half == 2 and R.sign("blue") == -1 and R.for_team == "yellow"
    assert all(w.robot_pose(("blue", k))[0][0] > 0 for k in range(5))
    run(w, 2.5)
    assert events(w, "KICKOFF")[-1]["team"] == "yellow"


def test_level_after_full_time_goes_to_shootout_and_penalty_goal():
    """S5 (§10): level after two halves -> shoot-out, penalty kicks (§5.3.5); a shot into the goal counts."""
    w = world(half_s=3.0)
    run(w, 14.0)
    R = w.referee
    assert events(w, "FULL_TIME") and R.stage == "SHOOTOUT", events(w, "FULL_TIME", "SHOOTOUT_START")
    assert R.state in ("PREPARE_PENALTY", "PENALTY") and R.penalty["team"] == "blue"
    assert sum(w.on_field.values()) == 2
    run(w, 2.2)
    assert R.state == "PENALTY"
    # blue1 shoots past the keeper at the -x? no: blue attacks -x in the second half
    p, _ = w.robot_pose(B1)
    bx, by = w.ball.pos[:2]
    gx = -4.5
    yaw = math.atan2(0.3 - by, gx - bx)
    target(w, B1, p[0], p[1], yaw)
    run(w, 0.6)
    w.apply("blue", 1, "kick", dict(speed=5.0, armed_for=3.0))
    run(w, 3.0, Carrot(w, B1, math.cos(yaw), math.sin(yaw), 0.4, yaw=yaw))
    k = events(w, "SHOOTOUT_KICK")
    assert k and k[0]["team"] == "blue" and k[0]["goal"] is True, k
    assert R.shootout["goals"]["blue"] == 1


def test_shootout_decision_and_coin_toss():
    """§10: 5 kicks each, stops when a team cannot catch up; sudden death; S5: after 10 rounds a seeded coin toss."""
    w = world()
    R = w.referee
    R.stage, R.half = "SECOND_HALF", 2
    R._start_shootout()
    for g_blue, g_yellow in [(1, 0), (1, 0), (1, 0), (0, 0)]:
        R._shootout_record("blue", g_blue)
        R._shootout_record("yellow", g_yellow)
    assert R.stage == "POST_GAME" and R.winner == "blue"  # 3:0 after 4 rounds: yellow cannot catch up
    w2 = world(seed=7)
    R = w2.referee
    R.stage, R.half = "SECOND_HALF", 2
    R._start_shootout()
    for _ in range(10):
        R._shootout_record("blue", False)
        if R.stage == "POST_GAME":
            break
        R._shootout_record("yellow", False)
    assert events(w2, "COIN_TOSS") and R.winner in ("blue", "yellow") and R.stage == "POST_GAME"


def test_penalty_kick_procedure():
    """§5.3.5: after a multiple-defenders foul the penalty is prepared (keeper on the goal line, others 1 m behind the
    ball), started 2 s after the positions are legal, and a goal is given; then the other team kicks off."""
    from sim.world import penalty_formation

    w = world()
    R = w.referee
    w.referee.stop(("PENALTY", "yellow"), F.penalty_mark(-1))
    run(w, 3.1)
    assert R.state == "PREPARE_PENALTY"
    for rid, pose in penalty_formation(5, 1, "yellow", (1.5, 0.0)).items():
        place(w, rid, *pose)
    run(w, 2.2)
    assert R.state == "PENALTY", (R.state, R._penalty_check(1.5, 0.0) if R.penalty else None)
    yaw = math.atan2(0.3 - 0.0, -4.5 - 1.5)
    p, _ = w.robot_pose(Y1)
    target(w, Y1, p[0], p[1], yaw)
    run(w, 0.5)
    w.apply("yellow", 1, "kick", dict(speed=5.0, armed_for=3.0))
    run(w, 3.0, Carrot(w, Y1, math.cos(yaw), math.sin(yaw), 0.4, yaw=yaw))
    g = events(w, "GOAL")
    assert g and g[0]["team"] == "yellow" and g[0].get("penalty"), events(w, "GOAL", "PENALTY_KICK_FAILED")
    assert R.next_cmd == ("KICKOFF", "blue")


def test_cards():
    """§8.4: every third foul a yellow card; §5.5.1 one robot fewer for 120 s of playing time, removed within 10 s
    (else by the referee, then a force start); a third card while two are active is red (§5.5.2)."""
    w = world()
    R = w.referee
    running(w, 0.0, 1.0)
    for _ in range(3):
        R.foul("TEST_FOUL", "blue", ("blue", 3))
    assert events(w, "YELLOW_CARD") and R.allowed("blue") == 4
    run(w, 10.2)
    rem = events(w, "ROBOT_REMOVED")
    assert rem and rem[0]["robot"] == "blue3" and not w.on_field[("blue", 3)], rem
    assert w.robot_pose(("blue", 3))[0][1] < -F.WALL_Y  # off the field, outside the walls
    assert R.next_cmd == ("FORCE_START", None)
    assert w.apply("blue", 3, "move", dict(x=0, y=0, theta=0))["status"] == "refused"
    for _ in range(3):
        R.foul("TEST_FOUL", "blue", ("blue", 2))
    assert len(R.yellow["blue"]) == 2 and R.allowed("blue") == 3
    for _ in range(3):
        R.foul("TEST_FOUL", "blue", ("blue", 4))
    assert R.red["blue"] == 1 and events(w, "RED_CARD") and R.allowed("blue") == 2
    R.play_time += 121.0
    R._expire_cards()
    assert R.allowed("blue") == 4 and len(events(w, "YELLOW_CARD_EXPIRED")) == 2


def test_goal_invalid_with_too_many_robots():
    """§7: no goal while the scoring team has more robots on the field than allowed (yellow card not served yet)."""
    w = world()
    only(w, B0, B1, B2, ("blue", 3), ("blue", 4))  # all five blue robots stay on; yellow's are off
    R = w.referee
    for _ in range(3):
        R.foul("TEST_FOUL", "blue", ("blue", 3))
    kick_from(w, B2, 3.0, 0.0, 0.0, 3.0)
    inv = events(w, "INVALID_GOAL")
    assert inv and inv[0]["reason"] == "TOO_MANY_ROBOTS", events(w, "GOAL", "INVALID_GOAL")


def test_substitution_out_only_from_substitution_area():
    """§9.1.2, §9.2: a robot can be taken out only from its substitution area, with the ball 0.5 m away."""
    w = world()
    running(w, 0.0, 1.0)
    assert w.substitute_out("blue", 2) == "not_in_substitution_area"
    place(w, B2, -0.5, -3.1, 0.0)
    w.step()
    assert w.substitute_out("blue", 2) == "ok" and not w.on_field[B2]
    R = w.referee
    assert w.substitute_in("blue", 2) == "ok" and w.on_field[B2]
    for _ in range(3):
        R.foul("TEST_FOUL", "blue", None)
    place(w, B2, -0.5, -3.1, 0.0)
    w.step()
    assert w.substitute_out("blue", 2) == "ok"
    assert w.substitute_in("blue", 2) == "too_many_robots"


def test_determinism():
    """the same seed and commands give the same match, bit for bit."""

    def play(seed):
        w = world(seed=seed)
        out = []
        for k in range(400):
            if k == 210:
                w.apply("blue", 1, "kick", dict(speed=4.0, armed_for=0.5))
            if k == 260:
                w.apply("yellow", 2, "move", dict(x=-0.5, y=0.3, theta=0.0))
            w.step()
        out.append(w.snapshot())
        out.append([e for e in w.referee.events])
        return out

    assert play(3) == play(3)


if __name__ == "__main__":
    sys.exit(1 if run_all(globals()) else 0)
