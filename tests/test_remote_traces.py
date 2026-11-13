"""scripts/remote_traces.sh against a stand-in machine: a fake ssh that runs the remote side in a
local directory, so rsync, the start script and the fetch really run. Nothing leaves this host."""

import os
import shutil
import stat
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "remote_traces.sh"
BASH = shutil.which("bash")

pytestmark = pytest.mark.skipif(
    shutil.which("rsync") is None or BASH is None or sys.platform == "win32", reason="needs rsync and bash"
)

# Options first, then the host, then the command, as ssh takes them.
FAKE_SSH = """#!/usr/bin/env bash
while [ $# -gt 0 ]; do
    case "$1" in
        # Linux rsync splits user@host into -l user host.
        -o|-i|-p|-l) shift 2 ;;
        -*) shift ;;
        *) echo "$1" >> "$REMOTE_HOME/hosts.log"; shift; break ;;
    esac
done
[ ! -f "$REMOTE_HOME/down" ] || exit 255
cd "$REMOTE_HOME" && HOME="$REMOTE_HOME" exec bash -c "$*"
"""
LOGGER = """#!/usr/bin/env bash
echo "$*" >> "$REMOTE_HOME/{name}.log"
"""
# sg GROUP -c COMMAND runs COMMAND with the group; here it just runs it.
FAKE_SG = """#!/usr/bin/env bash
exec bash -c "$3"
"""


def executable(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


@pytest.fixture
def world(tmp_path):
    repo, remote, bin_dir = tmp_path / "repo", tmp_path / "remote", tmp_path / "bin"
    remote.mkdir()
    executable(repo / "scripts" / "remote_traces.sh", SCRIPT.read_text())
    files = {
        "Makefile": "traces:\n", "layover/replay/reexecute.py": "", "traces/prepared/manifest.json": "{}",
        ".git/info/exclude": "private-notes/\n", "private-notes/plan.md": "mine",
        ".venv/bin/python": "", ".notes/n.md": "", "results/old.txt": "", ".env": "SECRET=1",
        "id_ed25519": "key", "native/build/lib.so": "", "layover/__pycache__/x.pyc": "",
    }
    for name, text in files.items():
        (repo / name).parent.mkdir(parents=True, exist_ok=True)
        (repo / name).write_text(text)
    executable(bin_dir / "ssh", FAKE_SSH)
    executable(bin_dir / "sg", FAKE_SG)
    for name in ("sudo", "make", "docker"):
        executable(bin_dir / name, LOGGER.format(name=name))
    env = {**os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}", "REMOTE_HOME": str(remote)}
    env.pop("DEADMAN_MINUTES", None)

    def run(*args, **extra):
        return subprocess.run([BASH, str(repo / "scripts" / "remote_traces.sh"), *args], env={**env, **extra}, capture_output=True, text=True, timeout=60)

    return repo, remote, run


def wait_for(path: Path, timeout: float = 10) -> str:
    import time

    deadline = time.time() + timeout
    while not path.exists():
        assert time.time() < deadline, f"{path} never appeared"
        time.sleep(0.05)
    return path.read_text()


def test_start_syncs_the_tree_but_nothing_private_and_starts_the_job_with_a_dead_man(world):
    repo, remote, run = world
    done = run("user@box", "-i", "key", "--start")
    assert done.returncode == 0, done.stdout + done.stderr
    synced = remote / "layover-traces"
    assert (synced / "Makefile").exists() and (synced / "traces/prepared/manifest.json").exists()
    assert (synced / "layover/replay/reexecute.py").exists()
    for private in ("private-notes", ".git", ".venv", ".notes", "results", ".env", "id_ed25519", "native/build", "layover/__pycache__"):
        assert not (synced / private).exists(), private
    assert "started" in done.stdout
    assert "shutdown -h +240" in (remote / "sudo.log").read_text()
    assert wait_for(remote / "make.log").strip() == "traces PYTHON=python3"
    assert set((remote / "hosts.log").read_text().split()) == {"user@box"}


def test_the_dead_man_follows_its_setting(world):
    _, remote, run = world
    assert run("user@box", DEADMAN_MINUTES="30").returncode == 0
    assert "shutdown -h +30" in (remote / "sudo.log").read_text()


def test_status_shows_the_log_and_fetch_copies_the_trace_home(world):
    repo, remote, run = world
    (remote / "layover-traces/traces").mkdir(parents=True)
    (remote / "layover-traces/traces/swebench_30.jsonl").write_text('{"source": "recorded:x"}\n')
    (remote / "layover-traces/traces.log").write_text("task 30 of 30 done\n")
    status = run("user@box", "--status")
    assert status.returncode == 0 and "task 30 of 30 done" in status.stdout and "swebench_30.jsonl" in status.stdout
    fetched = run("user@box", "--fetch")
    assert fetched.returncode == 0, fetched.stdout + fetched.stderr
    dest = repo / "results/traces-box"
    assert (dest / "traces/swebench_30.jsonl").read_text() == '{"source": "recorded:x"}\n'
    assert (dest / "traces.log").exists() and f"copied to {dest}" in fetched.stdout


def test_a_dry_run_prints_the_commands_and_touches_nothing(world):
    repo, remote, run = world
    done = run("user@box", "--fetch", "--dry-run")
    assert done.returncode == 0
    assert "rsync" in done.stdout and "results/traces-box" in done.stdout
    assert not (repo / "results/traces-box").exists() and not (remote / "hosts.log").exists()


@pytest.mark.parametrize(
    "args, remove, message",
    [
        ((), None, "no host given"),
        (("user@box", "--bogus"), None, "unknown option: --bogus"),
        (("user@box",), "traces/prepared/manifest.json", "run make traces-prepare first"),
        (("user@box",), ".git/info/exclude", "lists the private files"),
    ],
    ids=["no-host", "unknown-option", "no-prepared-traces", "no-private-list"],
)
def test_it_refuses_to_start_without_what_it_needs(world, args, remove, message):
    repo, remote, run = world
    if remove:
        (repo / remove).unlink()
    done = run(*args)
    assert done.returncode == 2 and message in done.stderr
    assert not (remote / "layover-traces").exists()


def test_a_machine_that_cannot_be_reached_fails(world):
    _, remote, run = world
    (remote / "down").write_text("")
    assert run("user@box", "--status").returncode != 0
