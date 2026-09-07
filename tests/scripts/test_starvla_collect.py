import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from scripts.starvla import collect


def test_collect_bootstrap_uses_smoke_budget_and_returns_canonical_paths(tmp_path, monkeypatch):
    teacher = tmp_path / "teacher.pt"
    teacher.write_bytes(b"teacher")
    captured = {}

    def fake_collect(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(collect, "_collect_parkour", fake_collect)

    train, val = collect.collect_bootstrap(
        "extreme_parkour_go2",
        teacher,
        tmp_path / "bootstrap",
        "task-python",
        smoke=True,
        device="cuda:3",
    )

    assert train == (tmp_path / "bootstrap/lerobot").resolve()
    assert val == (tmp_path / "bootstrap/lerobot__val").resolve()
    assert captured["train_episodes"] == 2
    assert captured["val_episodes"] == 1
    assert captured["keep_failed"] is True
    assert captured["device"] == "cuda:3"


def test_balance_collection_is_zero_latency_and_uses_state198(tmp_path, monkeypatch):
    teacher = tmp_path / "teacher.pt"
    teacher.write_bytes(b"teacher")
    commands = []
    monkeypatch.setattr(collect, "_run", lambda command, **kwargs: commands.append(command))
    monkeypatch.setattr(collect, "_prepare_converted", lambda task, raw, output: commands.append([task, str(raw), str(output)]))

    collect._collect_balance(
        teacher_checkpoint=teacher,
        output_root=tmp_path / "balance",
        task_python="task-python",
        train_episodes=90,
        val_episodes=10,
        keep_failed=False,
        device="cuda:0",
    )

    export = commands[0]
    assert export[export.index("--max-episodes") + 1] == "100"
    assert export[export.index("--val-fraction") + 1] == "0.1"
    assert "--balance-teacher-observation-state" in export
    assert "--latency-config" not in export
    prepare = commands[1]
    assert prepare[0] == "humanoidbench_balance_simple"
    assert prepare[-1].endswith("/balance/lerobot")


def test_haic_collection_runs_train_and_val_at_teacher_50hz(tmp_path, monkeypatch):
    teacher = tmp_path / "teacher.pt"
    teacher.write_bytes(b"teacher")
    commands = []
    monkeypatch.setattr(collect, "_run", lambda command, **kwargs: commands.append(command))
    monkeypatch.setattr(collect, "_prepare_converted", lambda task, raw, output: commands.append([task, str(raw), str(output)]))

    collect._collect_haic(
        teacher_checkpoint=teacher,
        output_root=tmp_path / "haic",
        task_python="task-python",
        train_episodes=250,
        val_episodes=25,
        keep_failed=False,
        device="cuda:1",
    )

    assert commands[0][2:4] == ["teacher-collect", "--teacher-checkpoint"]
    assert commands[0][commands[0].index("--split") + 1] == "train"
    assert commands[0][commands[0].index("--episode-budget") + 1] == "250"
    assert commands[0][commands[0].index("--seed") + 1] == "0"
    assert commands[1][commands[1].index("--split") + 1] == "val"
    assert commands[1][commands[1].index("--episode-budget") + 1] == "25"
    assert commands[1][commands[1].index("--seed") + 1] == "1"
    assert commands[-1][-2:] == [str(teacher.resolve()), str((tmp_path / "haic/actor_adapt.pt").resolve())]


def test_mikasa_collection_uses_converter_task_directory_name(tmp_path, monkeypatch):
    teacher = tmp_path / "teacher.pt"
    teacher.write_bytes(b"teacher")
    commands = []
    moves = []
    monkeypatch.setattr(collect, "_run", lambda command, **kwargs: commands.append(command))
    monkeypatch.setattr(collect.shutil, "move", lambda source, destination: moves.append((source, destination)))
    monkeypatch.setattr(collect, "_write_mikasa_contract", lambda train, val: None)

    collect._collect_mikasa(
        teacher_checkpoint=teacher,
        output_root=tmp_path / "mikasa",
        task_python="task-python",
        train_episodes=2,
        val_episodes=1,
        keep_failed=True,
        device="cuda:0",
    )

    assert commands[0][-1] == "--keep-failed"
    assert Path(moves[0][0]).name == "intercept_grab_fast_vla_v0_h1_train"
    assert Path(moves[1][0]).name == "intercept_grab_fast_vla_v0_h1_val"


def test_mikasa_contract_writes_loader_modality_metadata(tmp_path):
    dirs = []
    for split in ("train", "val"):
        dataset = tmp_path / split
        (dataset / "data/chunk-000").mkdir(parents=True)
        (dataset / "meta").mkdir()
        pq.write_table(
            pa.table({"observation.state": [[0.0] * 7], "action": [[0.0] * 7]}),
            dataset / "data/chunk-000/file-000.parquet",
        )
        dirs.append(dataset)

    collect._write_mikasa_contract(*dirs)

    modality = json.loads(
        (dirs[0] / "meta/modality.json").read_text(encoding="utf-8")
    )
    assert modality["state"]["proprio"]["original_key"] == "observation.state"
    assert modality["action"]["target"]["original_key"] == "action"
    assert list(modality["video"]) == ["top", "wrist"]
