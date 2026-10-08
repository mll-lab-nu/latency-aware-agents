# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License.

"""Component-based Control-Critical F1 (CC-F1) for discrete-action RL-games val.

Motivation
----------
Raw accuracy / per-action F1 on Atari-style discrete control is misleading: the
majority class (NOOP / coasting) dominates, and a dashboard full of per-action
metrics drifting up and down is impossible to act on. Instead we score, per task,
a single primary number ``CC-F1``:

1. Decode each action into a few binary *component predicates* (e.g. Demon: does
   the action contain FIRE? is it LEFT? is it RIGHT?).
2. For each component compute an event-level F1 with ``±K`` temporal tolerance
   (firing one frame late should not count as both a miss and a spurious fire).
3. Aggregate components into groups (macro-F1), then groups into CC-F1 via a
   *weighted harmonic mean* so that one collapsed critical skill tanks the score.

Because positive-class F1 ignores true negatives, the NOOP majority never dilutes
it. ``K=0`` reduces event matching to plain per-frame confusion, so Flappy
(single component, K=0) is exactly the per-frame flap-F1 we shipped first.

This module is pure numpy (no torch) so it is unit-testable standalone. The tiny
decoders mirror ``eval_core.decode_*`` (trivial argmax / threshold).
"""

from __future__ import annotations

from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np

DEADLY_FACTORIZED_11 = "factorized_11"
DEADLY_MULTIBINARY_7 = "multibinary_7"


# --------------------------------------------------------------------------- #
# Decoders (mirror starVLA/training/rl_games/eval_core.py decode_*).
# --------------------------------------------------------------------------- #
def _argmax_id(vec: Sequence[float], n_actions: int) -> int:
    return int(np.argmax(np.asarray(vec, dtype=np.float32)[:n_actions]))


def _factorized_11(vec: Sequence[float]) -> List[int]:
    arr = np.asarray(vec, dtype=np.float32)
    if arr.shape[0] < 11:
        raise ValueError(f"Expected >=11 dims for factorized_11 decode, got {arr.shape[0]}")
    return [
        int(np.argmax(arr[0:3])),   # turn:   NOOP / LEFT / RIGHT
        int(np.argmax(arr[3:6])),   # move:   NOOP / FORWARD / BACKWARD
        int(np.argmax(arr[6:9])),   # strafe: NOOP / LEFT / RIGHT
        int(np.argmax(arr[9:11])),  # attack: NOOP / ATTACK
    ]


def _multibinary_7(vec: Sequence[float], threshold: float = 0.5) -> List[int]:
    arr = np.asarray(vec, dtype=np.float32)
    return [int(x >= threshold) for x in arr[:7]]


# --------------------------------------------------------------------------- #
# Component extractors: raw action vector -> {component_name: 0/1}.
# --------------------------------------------------------------------------- #
def _comp_flappy(vec: Sequence[float]) -> Dict[str, int]:
    return {"flap": int(_argmax_id(vec, 2) == 1)}


def _comp_demon(vec: Sequence[float]) -> Dict[str, int]:
    # labels: NOOP, FIRE, RIGHT, LEFT, RIGHTFIRE, LEFTFIRE
    aid = _argmax_id(vec, 6)
    return {
        "fire": int(aid in (1, 4, 5)),
        "left": int(aid in (3, 5)),
        "right": int(aid in (2, 4)),
    }


def _comp_defend_the_line(vec: Sequence[float]) -> Dict[str, int]:
    # labels: NOOP, ATTACK, TURN_LEFT, TURN_LEFT + ATTACK, TURN_RIGHT, TURN_RIGHT + ATTACK
    aid = _argmax_id(vec, 6)
    return {
        "attack": int(aid in (1, 3, 5)),
        "turn_left": int(aid in (2, 3)),
        "turn_right": int(aid in (4, 5)),
    }


def _comp_deadly_factorized(vec: Sequence[float]) -> Dict[str, int]:
    turn, move, strafe, attack = _factorized_11(vec)
    return {
        "attack": int(attack == 1),
        "turn_left": int(turn == 1),
        "turn_right": int(turn == 2),
        "move_forward": int(move == 1),
        "move_backward": int(move == 2),
        "strafe_left": int(strafe == 1),
        "strafe_right": int(strafe == 2),
    }


def _comp_deadly_multibinary(vec: Sequence[float]) -> Dict[str, int]:
    # semantic order: MOVE_FORWARD, MOVE_BACKWARD, MOVE_LEFT, MOVE_RIGHT,
    #                 TURN_LEFT, TURN_RIGHT, ATTACK  (MOVE_LEFT/RIGHT == strafe)
    b = _multibinary_7(vec)
    return {
        "move_forward": b[0],
        "move_backward": b[1],
        "strafe_left": b[2],
        "strafe_right": b[3],
        "turn_left": b[4],
        "turn_right": b[5],
        "attack": b[6],
    }


# --------------------------------------------------------------------------- #
# Task specs.
# --------------------------------------------------------------------------- #
class CCF1Spec:
    """Static description of how to score one task's CC-F1.

    Attributes
    ----------
    task: env name.
    comp_fn: raw action vector -> {component_name: 0/1}.
    component_order: stable component ordering (for tensor pack/unpack).
    labels: component_name -> metric label (the part after ``eval/``).
    groups: group_name -> [component_names] aggregated by macro-F1.
    group_labels: group_name -> metric label.
    group_weights: group_name -> weight in the CC-F1 harmonic mean.
    cc_key: metric label for the CC-F1 scalar.
    default_k: temporal tolerance in frames (decision steps).
    """

    def __init__(
        self,
        task: str,
        comp_fn: Callable[[Sequence[float]], Dict[str, int]],
        labels: Dict[str, str],
        groups: Dict[str, List[str]],
        group_labels: Dict[str, str],
        group_weights: Dict[str, float],
        cc_key: str,
        default_k: int,
    ) -> None:
        self.task = task
        self.comp_fn = comp_fn
        self.labels = labels
        self.component_order = list(labels.keys())
        self.groups = groups
        self.group_labels = group_labels
        self.group_weights = group_weights
        self.cc_key = cc_key
        self.default_k = default_k


def get_spec(task: str, deadly_layout: str = DEADLY_FACTORIZED_11) -> CCF1Spec:
    """Resolve the CC-F1 spec for a task (deadly_corridor depends on layout)."""
    if task == "flappy":
        return CCF1Spec(
            task="flappy",
            comp_fn=_comp_flappy,
            labels={"flap": "flap"},
            groups={"flap": ["flap"]},
            group_labels={"flap": "flap"},
            group_weights={"flap": 1.0},
            cc_key="flap_cc_f1",
            default_k=0,
        )
    if task == "demon_attack":
        return CCF1Spec(
            task="demon_attack",
            comp_fn=_comp_demon,
            labels={"fire": "demon_fire", "left": "demon_left", "right": "demon_right"},
            groups={"fire": ["fire"], "move": ["left", "right"]},
            group_labels={"fire": "demon_fire", "move": "demon_move"},
            group_weights={"fire": 0.7, "move": 0.3},
            cc_key="demon_cc_f1",
            default_k=1,
        )
    if task == "air_raid":
        return CCF1Spec(
            task="air_raid",
            comp_fn=_comp_demon,
            labels={"fire": "air_raid_fire", "left": "air_raid_left", "right": "air_raid_right"},
            groups={"fire": ["fire"], "move": ["left", "right"]},
            group_labels={"fire": "air_raid_fire", "move": "air_raid_move"},
            group_weights={"fire": 0.7, "move": 0.3},
            cc_key="air_raid_cc_f1",
            default_k=1,
        )
    if task == "deadly_corridor":
        if deadly_layout == DEADLY_MULTIBINARY_7:
            comp_fn = _comp_deadly_multibinary
        elif deadly_layout == DEADLY_FACTORIZED_11:
            comp_fn = _comp_deadly_factorized
        else:
            raise ValueError(f"Unsupported deadly layout: {deadly_layout!r}")
        labels = {
            "attack": "deadly_attack",
            "turn_left": "deadly_turn_left",
            "turn_right": "deadly_turn_right",
            "move_forward": "deadly_move_forward",
            "move_backward": "deadly_move_backward",
            "strafe_left": "deadly_strafe_left",   # diagnostic only
            "strafe_right": "deadly_strafe_right",  # diagnostic only
        }
        return CCF1Spec(
            task="deadly_corridor",
            comp_fn=comp_fn,
            labels=labels,
            groups={
                "attack": ["attack"],
                "turn": ["turn_left", "turn_right"],
                "move": ["move_forward", "move_backward"],
            },
            group_labels={"attack": "deadly_attack", "turn": "deadly_turn", "move": "deadly_move"},
            group_weights={"attack": 0.4, "turn": 0.35, "move": 0.25},
            cc_key="deadly_cc_f1",
            default_k=1,
        )
    if task == "defend_the_line":
        return CCF1Spec(
            task="defend_the_line",
            comp_fn=_comp_defend_the_line,
            labels={
                "attack": "defend_attack",
                "turn_left": "defend_turn_left",
                "turn_right": "defend_turn_right",
            },
            groups={"attack": ["attack"], "turn": ["turn_left", "turn_right"]},
            group_labels={"attack": "defend_attack", "turn": "defend_turn"},
            group_weights={"attack": 0.6, "turn": 0.4},
            cc_key="defend_cc_f1",
            default_k=1,
        )
    raise ValueError(f"No CC-F1 spec for task {task!r}")


SUPPORTED_TASKS = ("flappy", "demon_attack", "air_raid", "defend_the_line", "deadly_corridor")


# --------------------------------------------------------------------------- #
# Counts vector: [tp, fp, fn, teacher_events, model_events] per component,
# followed by a single total_frames slot. Stable order = spec.component_order.
# This flat layout is what the trainer all_reduces across ranks.
# --------------------------------------------------------------------------- #
_PER_COMPONENT = 5


def vector_size(spec: CCF1Spec) -> int:
    return _PER_COMPONENT * len(spec.component_order) + 1


def new_counts(spec: CCF1Spec) -> np.ndarray:
    return np.zeros(vector_size(spec), dtype=np.float64)


def _match_events(t_frames: List[int], m_frames: List[int], k: int) -> Tuple[int, int, int]:
    """Greedy ±K one-to-one event matching -> (tp, fp, fn)."""
    t_sorted = sorted(t_frames)
    m_sorted = sorted(m_frames)
    used = [False] * len(m_sorted)
    tp = 0
    for t in t_sorted:
        best_j = -1
        best_d = None
        for j, m in enumerate(m_sorted):
            if used[j]:
                continue
            d = abs(m - t)
            if d <= k and (best_d is None or d < best_d):
                best_d = d
                best_j = j
        if best_j >= 0:
            used[best_j] = True
            tp += 1
    fn = len(t_sorted) - tp
    fp = len(m_sorted) - sum(used)
    return tp, fp, fn


def accumulate_episode(
    spec: CCF1Spec,
    frames: Sequence[int],
    teacher_comps: Sequence[Dict[str, int]],
    model_comps: Sequence[Dict[str, int]],
    k: int,
    counts: np.ndarray,
) -> None:
    """Match one episode's frames per component and add into ``counts`` in place.

    ``frames`` are the per-frame indices (e.g. base_index); ``teacher_comps`` and
    ``model_comps`` are the decoded component dicts aligned to ``frames``.
    """
    frames = list(frames)
    for idx, name in enumerate(spec.component_order):
        t_frames = [frames[i] for i, c in enumerate(teacher_comps) if c.get(name)]
        m_frames = [frames[i] for i, c in enumerate(model_comps) if c.get(name)]
        tp, fp, fn = _match_events(t_frames, m_frames, k)
        base = idx * _PER_COMPONENT
        counts[base + 0] += tp
        counts[base + 1] += fp
        counts[base + 2] += fn
        counts[base + 3] += len(t_frames)  # teacher events
        counts[base + 4] += len(m_frames)  # model events
    counts[-1] += len(frames)


def _f1(tp: float, fp: float, fn: float) -> Tuple[float, float, float]:
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def _weighted_harmonic_mean(values: Dict[str, float], weights: Dict[str, float]) -> float:
    num = 0.0
    den = 0.0
    for name, f in values.items():
        w = weights[name]
        if f <= 0.0:
            return 0.0  # a collapsed critical skill tanks CC-F1
        num += w
        den += w / f
    return num / den if den > 0 else 0.0


def reduce_metrics(spec: CCF1Spec, counts: np.ndarray) -> Dict[str, float]:
    """Turn an accumulated (and possibly all-reduced) counts vector into metrics."""
    total_frames = float(counts[-1])
    comp_f1: Dict[str, float] = {}
    comp_active: Dict[str, bool] = {}
    metrics: Dict[str, float] = {}

    for idx, name in enumerate(spec.component_order):
        base = idx * _PER_COMPONENT
        tp, fp, fn, tev, mev = (float(counts[base + j]) for j in range(_PER_COMPONENT))
        precision, recall, f1 = _f1(tp, fp, fn)
        comp_f1[name] = f1
        comp_active[name] = (tp + fp + fn) > 0
        label = spec.labels[name]
        metrics[f"eval/{label}_precision"] = precision
        metrics[f"eval/{label}_recall"] = recall
        metrics[f"eval/{label}_f1"] = f1
        metrics[f"eval/{label}_rate_model"] = (mev / total_frames) if total_frames > 0 else 0.0
        metrics[f"eval/{label}_rate_teacher"] = (tev / total_frames) if total_frames > 0 else 0.0

    group_f1: Dict[str, float] = {}
    for gname, members in spec.groups.items():
        active_members = [m for m in members if comp_active[m]]
        macro = float(np.mean([comp_f1[m] for m in active_members])) if active_members else 0.0
        group_f1[gname] = macro
        metrics[f"eval/{spec.group_labels[gname]}_f1"] = macro

    cc = _weighted_harmonic_mean(group_f1, spec.group_weights)
    metrics[f"eval/{spec.cc_key}"] = cc
    return metrics


# --------------------------------------------------------------------------- #
# Convenience entry point for tests / non-distributed use.
# --------------------------------------------------------------------------- #
def compute_from_episodes(
    task: str,
    episodes: Sequence[Dict[str, Sequence]],
    deadly_layout: str = DEADLY_FACTORIZED_11,
    k: int = None,
) -> Tuple[Dict[str, float], np.ndarray]:
    """Compute CC-F1 metrics from raw per-episode action vectors.

    Each episode is ``{"frames": [idx...], "teacher": [vec...], "model": [vec...]}``
    where ``vec`` are raw normalized action vectors (one-hot / logits). Returns
    ``(metrics, counts_vector)``.
    """
    spec = get_spec(task, deadly_layout)
    if k is None:
        k = spec.default_k
    counts = new_counts(spec)
    for ep in episodes:
        frames = list(ep["frames"])
        teacher_comps = [spec.comp_fn(v) for v in ep["teacher"]]
        model_comps = [spec.comp_fn(v) for v in ep["model"]]
        accumulate_episode(spec, frames, teacher_comps, model_comps, k, counts)
    return reduce_metrics(spec, counts), counts
