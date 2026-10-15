"""Offline tests of harbor_guard (scripts/olympiad/v02_run.sh): the run scripts start only the pinned multi-seat Harbor
fork CHIGUI0/harbor@RO_HARBOR_PIN. Upstream PyPI harbor 0.23.0 has no --seat, [[seats]] or setup_networks: with it the
agent would run in the `main` service on the default network, with internet.
  * the guard reads the `harbor` distribution of $HARBOR's interpreter (its #! line): a git install at the pinned
    commit passes; a PyPI install (no direct_url.json), another commit, an editable checkout at another commit or with
    changed tracked files, or a non-editable install from a directory is refused; so is a harbor whose `run --help`
    lists no --seat, or a missing one;
  * run_single.sh (every player, oracle and nop too), run_duel.sh, run_team.sh and run_race.sh stop with exit 2 before
    Harbor runs a trial when the guard refuses.
Fake Harbors in throwaway venvs (python -m venv, no pip): nothing is installed, no docker, no model call.
Standard library only (python >= 3.11):
    python3 scripts/olympiad/test_harbor_guard.py"""
import json
import os
import re
import shutil
import subprocess
import sys
import sysconfig
import tempfile
import venv
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
V02 = REPO / "scripts" / "olympiad" / "v02_run.sh"
PIN = re.search(r"^RO_HARBOR_PIN=([0-9a-f]{40})\b", V02.read_text(), re.M).group(1)
UPSTREAM = None                                   # a PyPI install records no direct_url.json
FORK = {"url": "https://github.com/CHIGUI0/harbor", "vcs_info": {"vcs": "git", "commit_id": PIN, "requested_revision": PIN}}

HELP = "Usage: harbor run [OPTIONS]\\n --seat SEAT=AGENT[@MODEL]\\n --seat-kwarg SEAT:KEY=VALUE\\n"
FAKE_HARBOR = """#!{python}
import json, os, sys
if sys.argv[1:] == ["run", "--help"]:
    print({help!r}, end="")
    sys.exit(0)
keep = {{k: v for k, v in os.environ.items() if k.startswith(("THINK_MODEL", "PLAYER_MODE_", "RCB_PLAYER_MODE", "CODEX_", "THINK_CHATGPT", "THINK_OPENAI_ROUTE"))}}
with open(os.environ["FAKE_HARBOR_OUT"], "a") as f:
    f.write(json.dumps({{"argv": sys.argv[1:], "env": keep}}) + "\\n")
"""


def fake_harbor(dirpath: Path, direct_url: dict | None = FORK, help_text: str = HELP) -> Path:
    """A venv at dirpath whose `harbor` distribution records direct_url (default: the pinned fork's git install) and
    whose bin/harbor answers `run --help` with help_text and otherwise appends its argv and seat env to
    $FAKE_HARBOR_OUT (the trial it would start). direct_url None (UPSTREAM): a PyPI install."""
    venv.create(dirpath, with_pip=False, symlinks=True)
    site = Path(sysconfig.get_path("purelib", vars={"base": str(dirpath), "platbase": str(dirpath)}))
    info = site / "harbor-0.23.0.dist-info"
    info.mkdir(parents=True)
    (info / "METADATA").write_text("Metadata-Version: 2.1\nName: harbor\nVersion: 0.23.0\n")
    if direct_url is not None:
        (info / "direct_url.json").write_text(json.dumps(direct_url))
    exe = dirpath / "bin" / "harbor"
    exe.write_text(FAKE_HARBOR.format(python=dirpath / "bin" / "python", help=help_text))
    exe.chmod(0o755)
    return exe


def guard(harbor: str) -> subprocess.CompletedProcess:
    return subprocess.run(["bash", "-c", f'source "{V02}"; harbor_guard && echo PASSED'], capture_output=True,
                          text=True, env={**os.environ, "HARBOR": harbor}, timeout=60)


def git_checkout(path: Path) -> str:
    """A local git repository (a harbor checkout); returns its HEAD."""
    path.mkdir()
    g = lambda *a: subprocess.run(["git", "-C", str(path), *a], check=True, capture_output=True, text=True)
    g("init", "-q")
    (path / "pyproject.toml").write_text("[project]\nname = 'harbor'\n")
    g("add", ".")
    g("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "x")
    return g("rev-parse", "HEAD").stdout.strip()


def git_shim(bindir: Path, real_head: str) -> dict:
    """PATH with a `git` that reports the checkout's HEAD as PIN (a fresh repository cannot have the pinned commit);
    everything else goes to the real git."""
    bindir.mkdir()
    real = shutil.which("git")
    (bindir / "git").write_text(f'#!/bin/bash\nout=$("{real}" "$@") || exit $?\n'
                                f'[ "$out" = "{real_head}" ] && out={PIN}\n[ -z "$out" ] || echo "$out"\n')
    (bindir / "git").chmod(0o755)
    return {"PATH": f"{bindir}:{os.environ['PATH']}"}


def test_pinned_fork_git_install_passes():
    with tempfile.TemporaryDirectory() as tmp:
        r = guard(str(fake_harbor(Path(tmp) / "fork")))
        assert r.returncode == 0 and "PASSED" in r.stdout, (r.stdout, r.stderr)


def test_upstream_pypi_harbor_is_refused():
    with tempfile.TemporaryDirectory() as tmp:
        r = guard(str(fake_harbor(Path(tmp) / "pypi", UPSTREAM)))      # its help may even list --seat: still refused
        assert r.returncode == 2 and "PASSED" not in r.stdout, (r.stdout, r.stderr)
        assert "not the pinned multi-seat fork" in r.stderr and "PyPI" in r.stderr and "internet" in r.stderr, r.stderr
        assert f"uv tool install git+https://github.com/CHIGUI0/harbor@{PIN}" in r.stderr, r.stderr


def test_another_commit_of_the_fork_is_refused():
    with tempfile.TemporaryDirectory() as tmp:
        other = {"url": "https://github.com/CHIGUI0/harbor", "vcs_info": {"vcs": "git", "commit_id": "0" * 40}}
        r = guard(str(fake_harbor(Path(tmp) / "old", other)))
        assert r.returncode == 2 and "0" * 40 in r.stderr, r.stderr


def test_no_seat_option_is_refused():
    with tempfile.TemporaryDirectory() as tmp:
        r = guard(str(fake_harbor(Path(tmp) / "noseat", help_text="Usage: harbor run [OPTIONS]\\n --agent AGENT\\n")))
        assert r.returncode == 2 and "no --seat" in r.stderr, r.stderr


def test_missing_or_non_python_harbor_is_refused():
    with tempfile.TemporaryDirectory() as tmp:
        r = guard(str(Path(tmp) / "nowhere" / "harbor"))
        assert r.returncode == 2 and "not found" in r.stderr, r.stderr
        sh = Path(tmp) / "harbor.sh"
        sh.write_text("echo --seat\n")
        sh.chmod(0o755)
        r = guard(str(sh))
        assert r.returncode == 2 and "no #! line" in r.stderr, r.stderr


def test_local_checkout_at_the_pin_passes_only_when_clean():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        head = git_checkout(tmp / "src")
        local = {"url": (tmp / "src").as_uri(), "dir_info": {"editable": True}}
        exe = fake_harbor(tmp / "dev", local)
        r = guard(str(exe))                                            # its HEAD is a fresh commit: refused
        assert r.returncode == 2 and head in r.stderr, r.stderr
        env = git_shim(tmp / "shim", head)                             # HEAD reported as the pin
        run = lambda: subprocess.run(["bash", "-c", f'source "{V02}"; harbor_guard && echo PASSED'],
                                     capture_output=True, text=True, env={**os.environ, **env, "HARBOR": str(exe)})
        r = run()
        assert r.returncode == 0 and "PASSED" in r.stdout, (r.stdout, r.stderr)
        (tmp / "src" / "pyproject.toml").write_text("[project]\nname = 'harbor-changed'\n")
        r = run()
        assert r.returncode == 2 and "tracked files are changed" in r.stderr, r.stderr


def test_non_editable_directory_install_is_refused():
    """pip install <dir> (not -e) copies the code at install time: the directory's HEAD (even the pin, clean) does
    not describe what runs, so only a git-URL or an editable install passes."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        head = git_checkout(tmp / "src")
        env = git_shim(tmp / "shim", head)                             # HEAD reported as the pin, no changes
        for dir_info in ({}, {"editable": False}):
            exe = fake_harbor(tmp / f"copy{len(dir_info)}", {"url": (tmp / "src").as_uri(), "dir_info": dir_info})
            r = subprocess.run(["bash", "-c", f'source "{V02}"; harbor_guard && echo PASSED'], capture_output=True,
                               text=True, env={**os.environ, **env, "HARBOR": str(exe)}, timeout=60)
            assert r.returncode == 2 and "PASSED" not in r.stdout and "not editable" in r.stderr, (dir_info, r.stderr)


def test_run_scripts_stop_before_upstream_harbor_starts_a_trial():
    """Every run script and player path (oracle and nop also run as --seat) stops with exit 2 and starts no trial."""
    tm = __import__("test_model_lock")                                 # its scratch copy of the run scripts
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        repo = tm.scratch_repo(tmp)                                    # its fake_harbor is the pinned fork
        cases = [("run_single.sh", ["--agent", a, "--level", "1", "--seed", "1"]) for a in ("opus", "astra-sub", "oracle", "nop")]
        cases += [("run_duel.sh", ["--a", "oracle", "--b", "oracle"]), ("run_team.sh", ["--a", "oracle", "--b", "opus"]),
                  ("run_race.sh", ["--players", "oracle,oracle,oracle,oracle"])]
        for script, args in cases:                                     # the pinned fork: the trial starts
            rc, calls = tm.run(tmp, repo, script, *args)
            assert rc == 0 and len(calls) == 1, (script, args, rc, calls)
        upstream = fake_harbor(tmp / "pypi", UPSTREAM)
        for script, args in cases:
            rc, calls = tm.run(tmp, repo, script, *args, extra_env={"HARBOR": str(upstream)})
            assert rc == 2 and not calls, (script, args, rc, calls)


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    names = [n for n in sorted(globals()) if n.startswith("test_")]
    bad = 0
    for n in names:
        try:
            globals()[n](); print("ok  ", n)
        except Exception:  # noqa: BLE001
            bad += 1; import traceback; traceback.print_exc(); print("FAIL", n)
    print(f"{len(names) - bad}/{len(names)} passed"); sys.exit(1 if bad else 0)
