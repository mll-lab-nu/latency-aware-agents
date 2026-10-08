import os
import signal
from types import SimpleNamespace


def test_launcher_first_interrupt_writes_stop_file_without_killing(monkeypatch, tmp_path):
    from examples.rl_games.scripts import launch_train

    calls = {}

    class FakeProcess:
        pid = 12345

        def wait(self):
            os.kill(os.getpid(), signal.SIGINT)
            return 0

    def fake_popen(cmd, *, cwd, env, start_new_session):
        calls["cmd"] = cmd
        calls["cwd"] = cwd
        calls["env"] = env
        calls["start_new_session"] = start_new_session
        return FakeProcess()

    monkeypatch.setattr(launch_train.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(launch_train.os, "killpg", lambda *args: calls.setdefault("killpg", []).append(args))

    stop_file = tmp_path / "run" / "STOP"
    returncode = launch_train.graceful_run_launch_command(
        ["python", "train.py"],
        cwd=tmp_path,
        stop_file=stop_file,
        env={},
    )

    assert returncode == 0
    assert calls["cmd"] == ["python", "train.py"]
    assert calls["cwd"] == str(tmp_path)
    assert calls["env"]["STARVLA_STOP_FILE"] == str(stop_file)
    assert calls["start_new_session"] is True
    assert stop_file.exists()
    assert "signal=SIGINT" in stop_file.read_text(encoding="utf-8")
    assert "killpg" not in calls


def test_launcher_second_interrupt_sends_sigterm(monkeypatch, tmp_path):
    from examples.rl_games.scripts import launch_train

    calls = {"killpg": []}

    class FakeProcess:
        pid = 54321

        def wait(self):
            os.kill(os.getpid(), signal.SIGINT)
            os.kill(os.getpid(), signal.SIGINT)
            return -signal.SIGTERM

    monkeypatch.setattr(
        launch_train.subprocess,
        "Popen",
        lambda *args, **kwargs: FakeProcess(),
    )
    monkeypatch.setattr(launch_train.os, "killpg", lambda *args: calls["killpg"].append(args))

    launch_train.graceful_run_launch_command(
        ["python", "train.py"],
        cwd=tmp_path,
        stop_file=tmp_path / "STOP",
        env={},
    )

    assert calls["killpg"] == [(54321, signal.SIGTERM)]


def test_trainer_refresh_graceful_stop_writes_stop_file(monkeypatch, tmp_path):
    from starVLA.training import train_starvla
    from starVLA.training.train_starvla import VLATrainer

    trainer = VLATrainer.__new__(VLATrainer)
    trainer.accelerator = SimpleNamespace(is_main_process=True)
    trainer.graceful_stop_requested = False
    trainer._graceful_stop_logged = False
    trainer._graceful_stop_file = tmp_path / "STOP"

    monkeypatch.setattr(train_starvla, "_GRACEFUL_STOP_REQUESTED", True)
    monkeypatch.setattr(train_starvla, "_GRACEFUL_STOP_SIGNAL", signal.SIGTERM)

    assert trainer._refresh_graceful_stop_requested() is True
    assert trainer.graceful_stop_requested is True
    assert trainer._graceful_stop_file.exists()
    assert "reason=SIGTERM" in trainer._graceful_stop_file.read_text(encoding="utf-8")
