from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from scripts.experiments.humanoidbench import hurdle_teacher


def test_cli_passes_cache_phase_trigger_sweep_and_seed(monkeypatch, capsys):
    calls = {}

    def fake_evaluate(**kwargs):
        calls.update(kwargs)
        return {"best": {"trigger_distance": 1.5}}

    monkeypatch.setattr(hurdle_teacher, "evaluate", fake_evaluate)

    hurdle_teacher.main(
        [
            "--cache-dir",
            "/tmp/sonic-cache",
            "--jump-start-phase",
            "30",
            "--trigger-distance",
            "1.5",
            "--trigger-distance",
            "2.0",
            "--seed",
            "7",
        ]
    )

    assert calls == {
        "cache_dir": Path("/tmp/sonic-cache"),
        "trigger_distances": (1.5, 2.0),
        "jump_start_phase": 30,
        "seed": 7,
    }
    assert json.loads(capsys.readouterr().out) == {
        "best": {"trigger_distance": 1.5}
    }


def test_evaluate_encodes_pinned_skill_references_without_writing_tokens(
    tmp_path, monkeypatch
):
    cache_dir = tmp_path / "cache"
    asset_root = cache_dir / "assets" / "sonic_g1"
    decoder_path = cache_dir / "decoder" / "model_decoder_dynamic.onnx"
    encoder_path = cache_dir / "encoder" / "model_encoder.onnx"
    skill_paths = {
        "fast_run": cache_dir / "fast_run.csv",
        "obstacle_jump": cache_dir / "obstacle_jump.csv",
    }
    monkeypatch.setattr(
        hurdle_teacher,
        "prepare_g1_sonic_assets",
        lambda path: (asset_root, decoder_path),
    )
    monkeypatch.setattr(
        hurdle_teacher, "prepare_g1_sonic_encoder", lambda path: encoder_path
    )
    monkeypatch.setattr(
        hurdle_teacher,
        "prepare_g1_sonic_skill_references",
        lambda path: skill_paths,
    )
    monkeypatch.setattr(
        hurdle_teacher.ort,
        "InferenceSession",
        lambda path, providers: SimpleNamespace(path=path, providers=providers),
    )
    qpos = np.zeros((40, 36), dtype=np.float32)
    qpos[:, 3] = 1.0
    monkeypatch.setattr(
        hurdle_teacher,
        "load_bones_seed_reference_qpos",
        lambda path: (qpos, 4 if path == skill_paths["fast_run"] else 5),
    )
    monkeypatch.setattr(hurdle_teacher, "canonicalize_reference_heading", lambda value: value)
    encoded = []
    monkeypatch.setattr(
        hurdle_teacher,
        "encode_qpos_tokens",
        lambda session, value, horizon: encoded.append((session.path, value, horizon))
        or np.zeros((horizon + 1, 64), dtype=np.float32),
    )
    monkeypatch.setattr(
        hurdle_teacher,
        "_reference_foot_contacts",
        lambda scene, value: np.zeros((len(value), 2), dtype=bool),
    )
    monkeypatch.setattr(
        hurdle_teacher,
        "g1_raw_reset_action",
        lambda scene, learn_hands: np.zeros(78, dtype=np.float32),
    )
    sweep = {"teacher_accepted": False, "results": []}
    monkeypatch.setattr(
        hurdle_teacher,
        "evaluate_trigger_distances",
        lambda env_factory, run_tokens, jump_tokens, run_reference, jump_reference,
        run_contacts, jump_contacts, neutral_hands, trigger_distances, *, seed,
        hurdle_positions: sweep,
    )

    result = hurdle_teacher.evaluate(
        cache_dir=cache_dir,
        jump_start_phase=2,
        trigger_distances=(1.0,),
        seed=7,
    )

    assert result == {"jump_start_phase": 2, **sweep}
    assert [entry[0] for entry in encoded] == [str(encoder_path), str(encoder_path)]
    assert [entry[2] for entry in encoded] == [3, 4]
