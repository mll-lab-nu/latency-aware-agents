from pathlib import Path


def test_deadly_corridor_envpool_cfg_is_not_published_by_in_place_write(monkeypatch, tmp_path):
    from training.deadly_corridor_sf import envpool_env

    source_path = tmp_path / "deadly_corridor.cfg"
    source_path.write_text(
        "\n".join(
            [
                "doom_scenario_path = deadly_corridor.wad",
                "screen_resolution = RES_320X240",
                "available_buttons = ",
                "\t{ ",
                "\t\tTURN_LEFT",
                "\t\tTURN_RIGHT",
                "\t\tMOVE_FORWARD",
                "\t\tMOVE_BACKWARD",
                "\t\tMOVE_LEFT",
                "\t\tMOVE_RIGHT",
                "\t\tATTACK",
                "\t}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    out_dir = tmp_path / "latency_bench_envpool"
    out_dir.mkdir()
    target_path = out_dir / "deadly_corridor_res_160x120.cfg"
    target_path.write_text("old complete cfg\navailable_buttons = { ATTACK }\n", encoding="utf-8")

    monkeypatch.setattr(envpool_env.tempfile, "gettempdir", lambda: str(tmp_path))
    original_open = Path.open

    def open_without_direct_target_write(self, mode="r", buffering=-1, encoding=None, errors=None, newline=None):
        if self == target_path and any(flag in mode for flag in ("w", "a", "x", "+")):
            raise AssertionError("published cfg target must not be opened for direct writes")
        return original_open(self, mode, buffering=buffering, encoding=encoding, errors=errors, newline=newline)

    monkeypatch.setattr(Path, "open", open_without_direct_target_write)

    env = object.__new__(envpool_env.DeadlyCorridorEnvPoolEnv)
    env.raw_width = 160
    env.raw_height = 120

    assert env._patched_envpool_cfg_path(source_path) == target_path
    published = target_path.read_text(encoding="utf-8")
    assert "screen_resolution = RES_160X120" in published
    assert "TURN_LEFT" in published
    assert "ATTACK" in published
