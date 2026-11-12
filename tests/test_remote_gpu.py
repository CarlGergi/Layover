"""The remote GPU runner and machine setup, without a GPU.

Shim ssh and rsync run the "remote" side in a local directory, so these tests cover the flow,
detaching, resuming and failure handling on any machine. The real thing, with real ssh, real rsync
and a dropped connection against a container running sshd, runs with LAYOVER_DOCKER_E2E=1.
"""

import json
import os
import shlex
import shutil
import signal
import stat
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
REAL_RUNNER = SCRIPTS / "remote_gpu.sh"
REAL_SETUP = SCRIPTS / "setup_gpu.sh"
# On macOS this is bash 3.2, the old version the local side has to work on.
BASH = "/bin/bash"
HEAD_TIMEOUT = 60
ORDER = ["setup", "smoke", "baselines", "gpu-tests", "sanitize", "traces", "reproduce"]

pytestmark = pytest.mark.skipif(
    shutil.which("rsync") is None or sys.platform == "win32", reason="needs rsync and a POSIX shell"
)

# Assembled here so that this file cannot trip the scan for credentials it is testing.
FAKE_API_KEY = "sk-" + "proj-" + "a1B2c3D4" * 6

STAGE_STUB = """#!/bin/bash
# What a real stage does: record itself, write some results, then maybe stall or fail.
printf '%s|%s|%s|%s\n' "$1" "${MATRIX_ARGS-unset}" "${TRACES_ARGS-unset}" "${SMOKE_PLAN-unset}" >> "$HOME/stage_env.log"
echo "$1" >> "$HOME/calls.log"
mkdir -p "results/$1"
echo partial > "results/$1/partial.txt"
[ ! -f "$HOME/hours-$1" ] || cp "$HOME/hours-$1" "results/$1/refined_hours.txt"
while [ -f "$HOME/hold-$1" ]; do sleep 0.05; done
if [ -f "$HOME/fail-$1" ]; then
    echo "stub failure in $1"
    exit 3
fi
echo final > "results/$1/final.txt"
"""

# Fixture targets, because the flow is the point here and not what the real Makefile holds.
MAKEFILE_STUB = """.PHONY: test-gpu gpu-smoke sanitize traces reproduce
gpu-smoke:
\t@bash stage_stub.sh smoke
test-gpu:
\t@bash stage_stub.sh gpu-tests
sanitize:
\t@bash stage_stub.sh sanitize
traces:
\t@bash stage_stub.sh traces
\t@echo written-on-the-machine > traces/durations.json
reproduce:
\t@bash stage_stub.sh reproduce
"""

NVIDIA_SMI_STUB = """#!/bin/bash
case "$*" in
    *--query-gpu=*) cat "$HOME/gpu-line" 2>/dev/null || echo "NVIDIA L4, 23034, 595.71.05" ;;
    *) echo "stub nvidia-smi" ;;
esac
"""

DOCKER_STUB = """#!/bin/bash
case "$1" in
    run)
        if [ -f "$HOME/docker-broken" ] && [ -z "${ROOT_NOW:-}" ]; then
            echo "docker: Cannot connect to the Docker daemon at unix:///var/run/docker.sock."
            exit 125
        fi
        echo "Hello from Docker!"
        ;;
    info) echo "$HOME" ;;
    *) echo "stub docker" ;;
esac
"""

UNAME_STUB = """#!/bin/bash
machine=$(cat "$HOME/machine" 2>/dev/null || echo x86_64)
case "$1" in
    -s) echo Linux ;;
    -m) echo "$machine" ;;
    *) echo "Linux $machine" ;;
esac
"""


# Setup leaves an environment, and the baselines stage's preflight runs through its python.
SETUP_STUB = """#!/bin/bash
printf '%s|%s|%s\\n' "${LAYOVER_NO_DOCKER-unset}" "${XDG_CACHE_HOME-unset}" "${HF_HOME-unset}" >> "$HOME/setup_env.log"
mkdir -p .venv/bin
cat > .venv/bin/python <<'PY'
#!/bin/bash
echo "$*" >> "$HOME/preflight.log"
[ ! -f "$HOME/fail-preflight" ]
PY
chmod +x .venv/bin/python
exec bash stage_stub.sh setup
"""

# Only sudo-docker-ok lets sudo work, so that the real sudo is never tried from a test.
SUDO_STUB = """#!/bin/bash
[ "$1" = -n ] && shift
# shutdown resolves to the logging stub below, never the real one.
case "$1" in shutdown|true) [ -f "$HOME/no-sudo" ] && exit 1; exec "$@" ;; esac
[ -f "$HOME/sudo-docker-ok" ] || exit 1
ROOT_NOW=1 exec "$@"
"""

SHUTDOWN_STUB = """#!/bin/bash
echo "$*" >> "$HOME/shutdown.log"
"""

# What a machine's libc says about its RAM. A page is 4 KiB, so a GiB is 262144 pages.
GETCONF_STUB = """#!/bin/bash
case "$1" in
    PAGESIZE) echo 4096 ;;
    _PHYS_PAGES) echo $(( $(cat "$HOME/ram-gib" 2>/dev/null || echo 64) * 262144 )) ;;
esac
"""

# RunPod's CLI, as in a pod: it works with the pod's own key. runpodctl-v1 makes it the older one,
# whose commands put the verb first.
RUNPODCTL_STUB = """#!/bin/bash
echo "$*" >> "$HOME/runpodctl.log"
[ ! -f "$HOME/runpodctl-broken" ] && [ -n "${RUNPOD_API_KEY:-}" ] || exit 1
if [ -f "$HOME/runpodctl-v1" ]; then
    case "$1 $2" in "get pod"|"stop pod") exit 0 ;; esac
else
    case "$1 $2" in "pod get"|"pod stop") exit 0 ;; esac
fi
echo "unknown command" >&2
exit 1
"""

# The metadata service, or with no imds file a machine off AWS, where nothing answers.
# The real curl would wait out its timeout on every preflight.
CURL_STUB = """#!/bin/bash
[ -f "$HOME/imds" ] || exit 28
case "$*" in
    *"-X PUT"*) echo token-1 ;;
    *spot/instance-action*) [ -f "$HOME/spot-notice" ] || exit 22; cat "$HOME/spot-notice" ;;
    *"X-aws-ec2-metadata-token: token-1"*) cat "$HOME/imds" ;;
    *) printf 'no token\\n401' ;;
esac
"""


def write_executable(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def build_repo(repo):
    """A small repository with every kind of file the sync has to keep or leave behind."""
    if shutil.which("git") is None:
        pytest.skip("needs git: the runner records the commit it syncs")
    write_executable(repo / "scripts" / "remote_gpu.sh", REAL_RUNNER.read_text())
    write_executable(repo / "scripts" / "setup_gpu.sh", SETUP_STUB)
    write_executable(repo / "scripts" / "setup_baselines.sh", "#!/bin/bash\nexec bash stage_stub.sh baselines\n")
    write_executable(repo / "stage_stub.sh", STAGE_STUB)
    (repo / "Makefile").write_text(MAKEFILE_STUB)
    (repo / "README.md").write_text("readme\n")
    git_env = dict(
        os.environ,
        GIT_AUTHOR_NAME="t",
        GIT_AUTHOR_EMAIL="t@example.com",
        GIT_COMMITTER_NAME="t",
        GIT_COMMITTER_EMAIL="t@example.com",
    )
    (repo / ".gitignore").write_text("traces/\nresults/\n.venv/\n.notes/\n.hypothesis/\n__pycache__/\nbuild/\n*.so\n*.pem\n.env\n")
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    (repo / ".git" / "info" / "exclude").write_text("# Local only.\nPRIVATE_NOTES.md\n.private/\n")
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-qm", "base"], cwd=repo, check=True, env=git_env)
    # None of these are committed, and that is the point.
    for name, text in {
        "PRIVATE_NOTES.md": "private",
        ".private/notes.txt": "private",
        ".notes/NOTES.md": "notes",
        ".venv/bin/python": "mac python",
        "results/old_local_result.txt": "keep me",
        "traces/job1.jsonl": "{}\n",
        "traces/prepared/manifest.json": "{}\n",
        ".env": "API_KEY=x",
        "deploy.pem": "x",
        "native/build/host/lib.so": "mac binary",
        "native/source.cc": "// source",
        "layover/__pycache__/a.pyc": "x",
        ".hypothesis/examples": "x",
    }.items():
        (repo / name).parent.mkdir(parents=True, exist_ok=True)
        (repo / name).write_text(text)


class World:
    """A repository, a fake remote home, and shims that connect them."""

    def __init__(self, base):
        # Spaces on purpose: every quoted path gets exercised on every run.
        self.repo = base / "my repo"
        self.home = base / "remote home"
        self.shims = base / "shims"
        self.remote_bin = base / "remote_bin"
        self.ssh_log = base / "ssh.log"
        self.rsync_log = base / "rsync.log"
        self.ssh_down = base / "ssh_down"
        self.home.mkdir()
        self.repo.mkdir()
        build_repo(self.repo)
        write_executable(self.remote_bin / "nvidia-smi", NVIDIA_SMI_STUB)
        write_executable(self.remote_bin / "docker", DOCKER_STUB)
        write_executable(self.remote_bin / "uname", UNAME_STUB)
        write_executable(self.remote_bin / "sudo", SUDO_STUB)
        write_executable(self.remote_bin / "shutdown", SHUTDOWN_STUB)
        write_executable(self.remote_bin / "getconf", GETCONF_STUB)
        write_executable(self.remote_bin / "curl", CURL_STUB)
        write_executable(self.remote_bin / "runpodctl", RUNPODCTL_STUB)
        self.extra_env = {}
        self.write_shims()

    def write_shims(self):
        real_rsync = shutil.which("rsync")
        write_executable(
            self.shims / "ssh",
            f"""#!/bin/bash
printf '%q ' "$@" >> {shlex.quote(str(self.ssh_log))}; echo >> {shlex.quote(str(self.ssh_log))}
if [ -f {shlex.quote(str(self.ssh_down))} ]; then echo "ssh: connect to host: Connection refused" >&2; exit 255; fi
while [ $# -gt 0 ]; do
    case "$1" in
        -T) shift ;;
        -o|-p|-i) shift 2 ;;
        *) shift; break ;;
    esac
done
cd {shlex.quote(str(self.home))} || exit 1
remote_path={shlex.quote(str(self.remote_bin))}:/usr/bin:/bin:/usr/sbin:/sbin
HOME={shlex.quote(str(self.home))} PATH="${{TEST_REMOTE_PATH:-$remote_path}}" exec bash -c "$*"
""",
        )
        write_executable(
            self.shims / "rsync",
            f"""#!/bin/bash
# The listing for the secret scan stays on this machine, so it does not count as a transfer.
case " $* " in
    *" --list-only "*) ;;
    *) printf '%q ' "$@" >> {shlex.quote(str(self.rsync_log))}; echo >> {shlex.quote(str(self.rsync_log))} ;;
esac
args=()
remote=0
while [ $# -gt 0 ]; do
    case "$1" in
        -e) shift 2; continue ;;
        user@box:*) remote=1; args+=({shlex.quote(str(self.home))}/"${{1#user@box:}}") ;;
        *) args+=("$1") ;;
    esac
    shift
done
if [ "$remote" = 1 ] && [ -f {shlex.quote(str(self.ssh_down))} ]; then echo "rsync: connection refused" >&2; exit 255; fi
exec {shlex.quote(real_rsync)} "${{args[@]}}"
""",
        )

    @property
    def env(self):
        return dict(os.environ, PATH=f"{self.shims}:{os.environ['PATH']}", **self.extra_env)

    @property
    def run_dir(self):
        return self.home / "layover-run"

    @property
    def state(self):
        return self.run_dir / ".layover-run"

    @property
    def calls(self):
        path = self.home / "calls.log"
        return path.read_text().split() if path.exists() else []

    def command(self, *args):
        return [BASH, str(self.repo / "scripts" / "remote_gpu.sh"), "user@box", "--poll", "0.1", "--min-disk-gb", "0", *args]

    def run(self, *args, timeout=HEAD_TIMEOUT):
        return subprocess.run(self.command(*args), env=self.env, capture_output=True, text=True, timeout=timeout)

    def start(self, *args):
        return subprocess.Popen(self.command(*args), env=self.env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

    def stage_status(self, stage):
        path = self.state / "stages" / stage
        return path.read_text().split()[0] if path.exists() else None

    def wait_for_stage(self, stage, status="running", timeout=30):
        deadline = time.time() + timeout
        while time.time() < deadline:
            # The marker goes up just before the stage starts, so wait for the stage itself:
            # killing the runner in that gap would leave nothing to run twice.
            if self.stage_status(stage) == status and (status != "running" or stage in self.calls):
                return
            time.sleep(0.02)
        raise AssertionError(f"{stage} never reached {status}")

    def session_dir(self):
        found = sorted((self.repo / "results").glob("gpu-*"))
        assert len(found) == 1, found
        return found[0]

    def times(self):
        return json.loads((self.session_dir() / "stage_times.json").read_text())

    def stage_rows(self):
        return {row["name"]: row for row in self.times()["stages"]}

    def control(self, name, text=""):
        (self.home / name).write_text(text)

    def release(self, stage):
        (self.home / f"hold-{stage}").unlink()

    def clean_up(self):
        for hold in self.home.glob("hold-*"):
            hold.unlink()
        for name in ("runner.pid", "stop_timer.pid"):
            pid_file = self.state / name
            if pid_file.exists():
                try:
                    os.kill(int(pid_file.read_text()), signal.SIGKILL)
                except (ProcessLookupError, ValueError):
                    pass


@pytest.fixture
def world(tmp_path):
    w = World(tmp_path)
    yield w
    w.clean_up()


def test_scripts_parse_under_system_bash():
    for script in (REAL_RUNNER, REAL_SETUP):
        subprocess.run([BASH, "-n", str(script)], check=True)


def local_part():
    return REAL_RUNNER.read_text().split("The local side.")[1]


def test_local_part_avoids_what_macos_lacks():
    local = local_part()
    for banned in ("declare -A", "mapfile", "readarray", "date -d", "sed -i", "&>>", "${!", "|&", "sed -r", "--iso-8601"):
        assert banned not in local, banned


def test_local_part_never_deletes_or_stops_anything():
    # The preflight script is sent to the machine and runs there, so it is not this side.
    local = local_part()
    start = local.index("preflight_script() {")
    words = (local[:start] + local[local.index("\nREMOTE\n}", start):]).split()
    for banned in ("rm", "rmdir", "shutdown", "poweroff", "halt", "reboot", "terminate-instances"):
        assert banned not in words, banned


def test_dry_run_contacts_nothing(world):
    result = world.run("--dry-run")
    assert result.returncode == 0, result.stderr
    assert not world.ssh_log.exists() and not world.rsync_log.exists()
    positions = [result.stdout.index(f"\n{stage} ") for stage in ORDER]
    assert positions == sorted(positions)


@pytest.mark.parametrize(
    "args",
    [
        ["--unknown"],
        ["-p"],
        ["-p", "notaport"],
        ["-i", "/no/such/key"],
        ["--stage", "nonsense"],
        ["--from", "nonsense"],
        ["--from", "sync"],
        ["--stage", "sanitize", "--from", "traces"],
        ["--status", "--fetch-only"],
        ["--status", "--stage", "traces"],
        ["--status", "--stop-after-smoke"],
        ["--fetch-only", "--stop-after-smoke"],
        ["--stage", "traces", "--stop-after-smoke"],
        ["--from", "baselines", "--stop-after-smoke"],
        ["--restore-from", "/no/such/dir"],
        ["--restore-from"],
        ["--fetch-every-runs", "soon"],
        ["--hourly-usd", "abc"],
        ["--poll", "x"],
        ["other@box"],
        ["--remote-dir", "/"],
        ["--remote-dir", ".."],
        ["--remote-dir", "a b"],
        ["--headline-only"],
        ["--plan"],
        ["--plan", "a b"],
        ["--plan", "x;touch pwned"],
        ["--plan", "$(touch pwned)"],
        ["--plan", "-x"],
        ["--plan", "it's"],
        ["--max-hours"],
        ["--max-hours", "soon"],
        ["--max-hours", "1;touch pwned"],
    ],
)
def test_bad_usage_is_exit_2_and_contacts_nothing(world, args):
    result = world.run(*args)
    assert result.returncode == 2, result.stdout + result.stderr
    assert not world.ssh_log.exists()


@pytest.mark.parametrize("host", ["-oProxyCommand=x", "user@host;ls", "user@ho st"])
def test_unusable_host_is_rejected(world, host):
    result = subprocess.run(
        [BASH, str(world.repo / "scripts" / "remote_gpu.sh"), host], env=world.env, capture_output=True, text=True
    )
    assert result.returncode == 2
    assert not world.ssh_log.exists()


def test_full_run_in_order_with_results_and_traces_back(world):
    result = world.run()
    assert result.returncode == 0, result.stdout + result.stderr
    assert world.calls == ORDER
    session = world.session_dir()
    assert (session / "results" / "reproduce" / "final.txt").exists()
    assert (session / "traces" / "job1.jsonl").exists()
    assert (session / "traces" / "durations.json").read_text().strip() == "written-on-the-machine"
    assert (session / "logs" / "traces.log").exists()
    rows = world.stage_rows()
    assert [row["name"] for row in world.times()["stages"]] == ["preflight", "sync", *ORDER, "fetch"]
    assert all(row["status"] == "done" and row["seconds"] is not None for row in rows.values())
    assert world.times()["gpu"].startswith("NVIDIA L4")
    assert "shuts itself down" in result.stdout
    assert (world.repo / "results" / "old_local_result.txt").read_text() == "keep me"
    assert "Cost:" not in result.stdout and world.times()["cost_usd"] is None
    # Everything but the one sync towards the machine only adds. openrsync re-runs itself through
    # the shim as --server, so only the client lines count.
    deleting = [line for line in world.rsync_log.read_text().splitlines() if "--delete" in line and "--server" not in line]
    assert len(deleting) == 1 and deleting[0].rstrip().endswith("user@box:layover-run/")
    recorded = (world.state / "synced_commit").read_text().strip()
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=world.repo, check=True, capture_output=True, text=True).stdout.strip()
    assert recorded.startswith(head)
    assert (session / "synced_commit").read_text().strip() == recorded == world.times()["commit"]
    log = world.ssh_log.read_text()
    assert "ForwardAgent=no" in log and "SendEnv=-" in log


def test_sync_leaves_private_things_home_and_brings_traces(world):
    assert world.run("--stage", "sync").returncode == 0
    listing = {str(p.relative_to(world.run_dir)) for p in world.run_dir.rglob("*")}
    for gone in (".git", ".venv", ".notes", ".private", "PRIVATE_NOTES.md", ".env", "deploy.pem",
                 "native/build", "layover/__pycache__", ".hypothesis", "results/old_local_result.txt"):
        assert gone not in listing, gone
    for kept in ("traces/job1.jsonl", "native/source.cc", "scripts/remote_gpu.sh", "Makefile", ".gitignore"):
        assert kept in listing, kept
    assert world.calls == []


def test_uncommitted_state_warns_and_marks_the_commit(world):
    (world.repo / "README.md").write_text("changed\n")
    result = world.run()
    assert "uncommitted changes" in result.stderr
    assert "-dirty-" in (world.state / "synced_commit").read_text()


def test_resync_mirrors_code_but_spares_what_the_machine_made(world):
    assert world.run("--stage", "sync").returncode == 0
    (world.run_dir / ".venv").mkdir()
    (world.run_dir / ".venv" / "marker").write_text("x")
    (world.run_dir / "results").mkdir(exist_ok=True)
    (world.run_dir / "results" / "machine.txt").write_text("x")
    (world.run_dir / "traces" / "durations.json").write_text("from the machine")
    # A trace the machine rewrote is newer than the one here, and stays.
    rewritten = world.run_dir / "traces" / "job1.jsonl"
    rewritten.write_text("rewritten")
    os.utime(rewritten, (time.time() + 60, time.time() + 60))
    # A trace prepared here after the first sync still goes over.
    (world.repo / "traces" / "job2.jsonl").write_text("new")
    (world.repo / "README.md").unlink()
    assert world.run("--stage", "sync").returncode == 0
    assert (world.run_dir / ".venv" / "marker").exists()
    assert (world.run_dir / "results" / "machine.txt").exists()
    assert (world.run_dir / "traces" / "durations.json").read_text() == "from the machine"
    assert rewritten.read_text() == "rewritten"
    assert (world.run_dir / "traces" / "job2.jsonl").read_text() == "new"
    assert not (world.run_dir / "README.md").exists()


def test_refuses_to_mirror_into_a_directory_it_did_not_make(world):
    world.run_dir.mkdir()
    (world.run_dir / "someone_elses_file").write_text("precious")
    result = world.run()
    assert result.returncode == 10
    assert "Refusing to mirror" in result.stdout
    assert (world.run_dir / "someone_elses_file").read_text() == "precious"
    assert not world.rsync_log.exists()


# Stages behind make report make's own exit 2; the others report the stub's 3.
@pytest.mark.parametrize(
    "stage,code,stage_rc",
    [("setup", 12, 3), ("smoke", 19, 2), ("sanitize", 14, 2), ("traces", 16, 2), ("reproduce", 17, 2)],
)
def test_failed_stage_stops_later_ones_fetches_and_sets_the_exit_code(world, stage, code, stage_rc):
    world.control(f"fail-{stage}")
    result = world.run()
    assert result.returncode == code, result.stdout + result.stderr
    assert world.calls == ORDER[: ORDER.index(stage) + 1]
    session = world.session_dir()
    # What it wrote before failing comes back, with the log.
    assert (session / "results" / stage / "partial.txt").exists()
    assert not (session / "results" / stage / "final.txt").exists()
    assert "stub failure" in (session / "logs" / f"{stage}.log").read_text()
    row = world.stage_rows()[stage]
    assert row["status"] == "failed" and row["exit_code"] == stage_rc
    assert f"FAILED: {stage}" in result.stdout
    assert "shuts itself down" in result.stdout


def test_resume_runs_from_the_failed_stage(world):
    world.control("fail-sanitize")
    assert world.run().returncode == 14
    session = world.session_dir()
    (world.home / "fail-sanitize").unlink()
    result = world.run()
    assert result.returncode == 0, result.stdout + result.stderr
    assert world.calls == ["setup", "smoke", "baselines", "gpu-tests", "sanitize", "sanitize", "traces", "reproduce"]
    assert world.session_dir() == session


def test_rerun_after_success_does_no_work_again(world):
    assert world.run().returncode == 0
    assert world.run().returncode == 0
    assert world.calls == ORDER


def test_continue_on_failure_runs_the_rest_and_reports_the_first(world):
    world.control("fail-sanitize")
    world.control("fail-traces")
    result = world.run("--continue-on-failure")
    assert result.returncode == 14
    assert world.calls == ORDER
    assert "FAILED: sanitize" in result.stdout and "FAILED: traces" in result.stdout


def test_stage_runs_only_that_stage_even_if_done(world):
    assert world.run().returncode == 0
    assert world.run("--stage", "traces").returncode == 0
    assert world.calls == [*ORDER, "traces"]


def test_failed_single_stage_sets_the_exit_code(world):
    assert world.run().returncode == 0
    world.control("fail-baselines")
    assert world.run("--stage", "baselines").returncode == 15


def test_from_reruns_that_stage_and_the_later_ones(world):
    assert world.run().returncode == 0
    assert world.run("--from", "traces").returncode == 0
    assert world.calls == [*ORDER, "traces", "reproduce"]


def test_stage_resyncs_so_it_runs_the_code_here(world):
    assert world.run().returncode == 0
    (world.repo / "README.md").write_text("fixed\n")
    assert world.run("--stage", "sanitize").returncode == 0
    assert (world.run_dir / "README.md").read_text() == "fixed\n"


def test_stage_preflight_alone_changes_nothing(world):
    assert world.run("--stage", "preflight").returncode == 0
    assert not world.rsync_log.exists() and world.calls == []


def test_status_with_no_run_changes_nothing(world):
    result = world.run("--status")
    assert result.returncode == 0
    assert "no run on this machine" in result.stdout
    assert not world.run_dir.exists() and not world.rsync_log.exists()


def test_status_when_unreachable_is_an_error(world):
    world.ssh_down.write_text("")
    assert world.run("--status").returncode == 1


def test_status_reports_stages_and_runner(world):
    world.control("hold-gpu-tests")
    process = world.start()
    try:
        world.wait_for_stage("gpu-tests")
        status = world.run("--status")
        assert status.returncode == 0
        assert "runner:   running" in status.stdout
        lines = [line.split()[:2] for line in status.stdout.splitlines()]
        assert ["gpu-tests", "running"] in lines and ["setup", "done"] in lines and ["traces", "pending"] in lines
    finally:
        world.release("gpu-tests")
        process.communicate(timeout=HEAD_TIMEOUT)


def test_connection_drop_mid_stage_reconnects_and_the_work_goes_on(world):
    world.control("hold-gpu-tests")
    process = world.start()
    try:
        world.wait_for_stage("gpu-tests")
        pid = (world.state / "runner.pid").read_text()
        world.ssh_down.write_text("")
        time.sleep(0.6)
        # The run keeps going with nobody connected.
        assert world.stage_status("gpu-tests") == "running"
        os.kill(int(pid), 0)
        world.ssh_down.unlink()
        world.release("gpu-tests")
        output, _ = process.communicate(timeout=HEAD_TIMEOUT)
    finally:
        process.kill()
    assert process.returncode == 0, output
    assert "lost the connection" in output and "reconnected" in output
    assert world.calls == ORDER
    assert (world.state / "runner.pid").read_text() == pid
    assert (world.session_dir() / "results" / "reproduce" / "final.txt").exists()


def test_connection_that_stays_down_gives_up_with_exit_1(world):
    world.control("hold-gpu-tests")
    process = world.start("--reconnect-timeout", "1")
    world.wait_for_stage("gpu-tests")
    world.ssh_down.write_text("")
    output, _ = process.communicate(timeout=HEAD_TIMEOUT)
    assert process.returncode == 1, output
    assert "no connection" in output and "still billing" in output
    assert world.stage_status("gpu-tests") == "running"


def test_losing_this_side_leaves_the_run_going_and_a_rerun_attaches(world):
    world.control("hold-gpu-tests")
    process = world.start()
    world.wait_for_stage("gpu-tests")
    process.kill()
    process.wait()
    assert world.stage_status("gpu-tests") == "running"
    syncs = world.rsync_log.read_text().count("--delete")
    again = world.start()
    follower = Follower(again)
    follower.wait_for("attaching")
    world.release("gpu-tests")
    output = follower.finish()
    assert again.returncode == 0, output
    assert "attaching" in output
    assert world.calls == ORDER
    # Code never changes under a running stage.
    assert world.rsync_log.read_text().count("--delete") == syncs


def test_other_stages_are_refused_while_a_runner_is_going(world):
    world.control("hold-gpu-tests")
    process = world.start()
    try:
        world.wait_for_stage("gpu-tests")
        for args in (["--stage", "sanitize"], ["--from", "traces"], ["--stage", "sync"]):
            result = world.run(*args)
            assert result.returncode == 1, args
            assert "a runner is going" in result.stderr
    finally:
        world.release("gpu-tests")
        process.communicate(timeout=HEAD_TIMEOUT)


def test_dead_runner_is_resumed_from_the_interrupted_stage(world):
    # Reboots look like this: the marker says running, and the pid is gone.
    world.control("hold-gpu-tests")
    process = world.start()
    world.wait_for_stage("gpu-tests")
    process.kill()
    process.wait()
    os.kill(int((world.state / "runner.pid").read_text()), signal.SIGKILL)
    # The killed process can take a moment to disappear on a loaded machine.
    deadline = time.time() + 10
    while True:
        status = world.run("--status")
        if ["gpu-tests", "interrupted"] in [line.split()[:2] for line in status.stdout.splitlines()]:
            break
        assert time.time() < deadline, status.stdout
        time.sleep(0.1)
    world.release("gpu-tests")
    result = world.run()
    assert result.returncode == 0, result.stdout + result.stderr
    assert world.calls == ["setup", "smoke", "baselines", "gpu-tests", "gpu-tests", "sanitize", "traces", "reproduce"]


def test_partial_results_arrive_during_a_long_stage(world):
    world.control("hold-gpu-tests")
    process = world.start("--fetch-every", "0")
    try:
        world.wait_for_stage("gpu-tests")
        deadline = time.time() + 20
        while time.time() < deadline:
            if list((world.repo / "results").glob("gpu-*/results/gpu-tests/partial.txt")):
                break
            time.sleep(0.05)
        else:
            raise AssertionError("no partial results arrived while the stage was running")
        assert world.stage_status("gpu-tests") == "running"
        world.release("gpu-tests")
        output, _ = process.communicate(timeout=HEAD_TIMEOUT)
    finally:
        process.kill()
    assert process.returncode == 0, output
    assert "partial results copied" in output


def test_fetch_only_starts_nothing(world):
    assert world.run().returncode == 0
    calls = world.calls
    syncs = world.rsync_log.read_text().count("--delete")
    (world.run_dir / "results" / "late.txt").write_text("late")
    result = world.run("--fetch-only")
    assert result.returncode == 0
    assert (world.session_dir() / "results" / "late.txt").exists()
    assert world.calls == calls and world.rsync_log.read_text().count("--delete") == syncs


def test_fetch_only_without_a_session_explains_itself(world):
    result = world.run("--fetch-only")
    assert result.returncode == 18
    assert "no session" in result.stderr
    assert not list((world.repo / "results").glob("gpu-*"))


def test_fetch_never_removes_results_here(world):
    assert world.run().returncode == 0
    mine = world.session_dir() / "results" / "my_analysis.txt"
    mine.write_text("mine")
    (world.run_dir / "results" / "reproduce" / "final.txt").unlink()
    assert world.run("--fetch-only").returncode == 0
    assert mine.read_text() == "mine"
    assert (world.session_dir() / "results" / "reproduce" / "final.txt").exists()


def test_fetch_failure_has_its_own_exit_code(world):
    # rsync is what breaks, on a results/ that has gone, while ssh still answers.
    assert world.run().returncode == 0
    shutil.rmtree(world.run_dir / "results")
    result = world.run("--fetch-only")
    assert result.returncode == 18, result.stdout + result.stderr
    assert "--fetch-only" in result.stderr
    assert world.stage_rows()["fetch"]["status"] == "failed"


def test_cost_is_printed_and_recorded(world):
    result = world.run("--hourly-usd", "1.20")
    assert "at $1.20/hour" in result.stdout
    times = world.times()
    assert times["hourly_usd"] == 1.2
    assert times["cost_usd"] == pytest.approx(times["wall_clock_s"] / 3600 * 1.2, abs=0.01)


def test_key_with_a_space_and_port_reach_ssh_and_rsync(world):
    key = world.home.parent / "a dir" / "id_test"
    key.parent.mkdir()
    key.write_text("x")
    assert world.run("-p", "2222", "-i", str(key)).returncode == 0
    assert "-p 2222" in world.ssh_log.read_text()
    recorded = world.rsync_log.read_text()
    assert "2222" in recorded and "id_test" in recorded


# ---- smoke --------------------------------------------------------------------------------------

REFINED = "12.5 19\n  estimated total: 12.5 hours, up to 19.0 hours\n  Budget: 20 hours\n"


def test_a_broken_setup_never_reaches_the_baselines(world):
    world.control("fail-smoke")
    result = world.run()
    assert result.returncode == 19, result.stdout + result.stderr
    assert world.calls == ["setup", "smoke"]
    assert "FAILED: smoke" in result.stdout


def test_the_baselines_stage_also_runs_their_preflight(world):
    assert world.run().returncode == 0
    assert (world.home / "preflight.log").read_text().split() == [
        "bench/run_matrix.py", "--preflight-only", "--preflight-kinds", "vllm", "sglang", "--out", "results/matrix_preflight",
    ]


def test_a_baseline_that_will_not_start_fails_its_own_stage(world):
    world.control("fail-preflight")
    result = world.run()
    assert result.returncode == 15, result.stdout + result.stderr
    assert world.calls == ["setup", "smoke", "baselines"]


def test_a_baseline_that_will_not_install_skips_its_preflight(world):
    world.control("fail-baselines")
    assert world.run().returncode == 15
    assert not (world.home / "preflight.log").exists()


def test_smokes_hours_are_back_and_printed_while_the_baselines_run(world):
    world.control("hours-smoke", REFINED)
    world.control("hold-baselines")
    # The default fetch interval is ten minutes, so only the end of smoke can bring this back.
    process = world.start("--hourly-usd", "2")
    try:
        world.wait_for_stage("baselines")
        deadline = time.time() + 20
        while not list((world.repo / "results").glob("gpu-*/results/smoke/refined_hours.txt")):
            assert time.time() < deadline, "smoke's results did not come back"
            time.sleep(0.05)
        assert world.stage_status("baselines") == "running"
    finally:
        world.release("baselines")
        output, _ = process.communicate(timeout=HEAD_TIMEOUT)
    assert process.returncode == 0, output
    assert "12.5 to 19 hours" in output and "Budget: 20 hours" in output
    assert "$25.00 to $38.00" in output
    assert output.count("refined plan") == 1
    # The header line is for this script, not the reader.
    assert "12.5 19\n" not in output


def test_a_failed_smoke_with_no_hours_says_where_to_look(world):
    world.control("fail-smoke")
    result = world.run()
    assert "no refined hours" in result.stdout and "smoke.json" in result.stdout


def test_a_reattach_does_not_print_smokes_hours_again(world):
    world.control("hours-smoke", REFINED)
    world.control("hold-gpu-tests")
    process = world.start()
    world.wait_for_stage("gpu-tests")
    deadline = time.time() + 20
    while not list((world.repo / "results").glob("gpu-*/.smoke_reported")):
        assert time.time() < deadline
        time.sleep(0.05)
    process.kill()
    process.wait()
    again = world.start()
    follower = Follower(again)
    follower.wait_for("attaching")
    world.release("gpu-tests")
    output = follower.finish()
    assert again.returncode == 0, output
    assert "refined plan" not in output


def test_stop_after_smoke_stops_there_and_the_same_command_goes_on(world):
    world.control("hours-smoke", REFINED)
    result = world.run("--stop-after-smoke")
    assert result.returncode == 0, result.stdout + result.stderr
    assert world.calls == ["setup", "smoke"]
    assert "12.5 to 19 hours" in result.stdout
    assert "without --stop-after-smoke" in result.stdout and "shuts itself down" in result.stdout
    assert (world.session_dir() / "results" / "smoke" / "refined_hours.txt").exists()
    # It is the same session, going on from where it stopped.
    session = world.session_dir()
    result = world.run()
    assert result.returncode == 0, result.stdout + result.stderr
    assert world.calls == ORDER
    assert world.session_dir() == session


def test_stop_after_smoke_with_a_broken_setup_is_still_a_failure(world):
    world.control("fail-smoke")
    result = world.run("--stop-after-smoke")
    assert result.returncode == 19
    assert "without --stop-after-smoke" not in result.stdout


def test_stop_after_smoke_does_not_change_a_run_it_attaches_to(world):
    world.control("hold-gpu-tests")
    process = world.start()
    try:
        world.wait_for_stage("gpu-tests")
        attached = world.start("--stop-after-smoke")
        follower = Follower(attached)
        follower.wait_for("attaching")
        world.release("gpu-tests")
        output = follower.finish()
    finally:
        process.kill()
    assert "applies only when a run starts" in output
    assert world.calls == ORDER


def test_status_lists_smoke(world):
    assert world.run("--stop-after-smoke").returncode == 0
    lines = [line.split()[:2] for line in world.run("--status").stdout.splitlines()]
    assert ["smoke", "done"] in lines and ["baselines", "pending"] in lines


# ---- spot instances -----------------------------------------------------------------------------


class Follower:
    """Reads a watcher's output as it comes, to wait until it has attached however loaded the machine."""

    def __init__(self, process):
        self.process = process
        self.lines = []
        self.thread = threading.Thread(target=self.pump, daemon=True)
        self.thread.start()

    def pump(self):
        for line in self.process.stdout:
            self.lines.append(line)

    def wait_for(self, text, timeout=30):
        deadline = time.time() + timeout
        while not any(text in line for line in list(self.lines)):
            assert time.time() < deadline, "".join(self.lines)
            time.sleep(0.02)

    def finish(self):
        self.process.wait(timeout=HEAD_TIMEOUT)
        self.thread.join(HEAD_TIMEOUT)
        return "".join(self.lines)


def wait_for_session_file(world, pattern, timeout=20):
    deadline = time.time() + timeout
    while time.time() < deadline:
        found = list((world.repo / "results").glob(f"gpu-*/{pattern}"))
        if found:
            return found[0]
        time.sleep(0.05)
    raise AssertionError(f"{pattern} never arrived")


def test_results_come_back_quickly_during_the_long_stages_only(world):
    world.control("hold-gpu-tests")
    world.control("hold-reproduce")
    process = world.start("--fetch-every", "3600", "--fetch-every-runs", "0")
    try:
        world.wait_for_stage("gpu-tests")
        # Smoke's end brings everything back once, so this looks at what is written after that.
        wait_for_session_file(world, ".smoke_reported")
        time.sleep(0.3)
        (world.run_dir / "results" / "gpu-tests" / "late.txt").write_text("x")
        time.sleep(1)
        assert not list((world.repo / "results").glob("gpu-*/results/gpu-tests/late.txt"))
        world.release("gpu-tests")
        world.wait_for_stage("reproduce")
        wait_for_session_file(world, "results/reproduce/partial.txt")
        assert world.stage_status("reproduce") == "running"
    finally:
        world.clean_up()
        process.communicate(timeout=HEAD_TIMEOUT)


def test_a_finished_run_reaches_this_machine_within_the_interval(world):
    world.control("hold-reproduce")
    process = world.start("--fetch-every-runs", "0")
    try:
        world.wait_for_stage("reproduce")
        run = world.run_dir / "results" / "matrix" / "runs" / "1.json"
        run.parent.mkdir(parents=True)
        run.write_text('{"status": "ok"}')
        arrived = wait_for_session_file(world, "results/matrix/runs/1.json")
        assert json.loads(arrived.read_text()) == {"status": "ok"}
    finally:
        world.clean_up()
        process.communicate(timeout=HEAD_TIMEOUT)


def test_the_default_for_the_long_stages_is_a_minute_at_most():
    assert "FETCH_EVERY_RUNS=60" in local_part()


def lose_the_instance(world, process):
    """It goes away with the work in flight, and a replacement starts from nothing."""
    pid_file = world.state / "runner.pid"
    try:
        os.kill(int(pid_file.read_text()), signal.SIGKILL)
    except (FileNotFoundError, ProcessLookupError, ValueError):
        pass
    world.clean_up()
    process.communicate(timeout=HEAD_TIMEOUT)
    # What was running there may take a moment to notice.
    deadline = time.time() + 10
    while True:
        try:
            shutil.rmtree(world.run_dir)
            break
        except OSError:
            assert time.time() < deadline
            time.sleep(0.05)
    (world.home / "calls.log").unlink()


def test_an_instance_that_vanishes_is_reported_with_the_command_to_resume(world):
    key = world.home.parent / "id_test"
    key.write_text("x")
    world.control("hold-reproduce")
    process = world.start("--reconnect-timeout", "1", "--fetch-every-runs", "0", "--hourly-usd", "1.5", "-i", str(key), "--from", "setup")
    world.wait_for_stage("reproduce")
    # A fetch brings the stage markers last, so this one means the copy is whole.
    wait_for_session_file(world, "stages/reproduce")
    world.ssh_down.write_text("")
    output, _ = process.communicate(timeout=HEAD_TIMEOUT)
    world.ssh_down.unlink()
    assert process.returncode == 1, output
    session = world.session_dir()
    assert "no connection" in output and "looks gone" in output and "safe in" in output
    command = next(line.strip() for line in output.splitlines() if "user@NEW-HOST" in line and line.startswith("    "))
    assert f"--restore-from results/{session.name}" in command
    assert "--hourly-usd 1.5" in command and str(key) in command and "--reconnect-timeout 1" in command
    assert "user@box" not in command and "--from" not in command
    # Nothing fetched was lost, and the time table says what happened.
    assert (session / "results" / "reproduce" / "partial.txt").exists()
    assert world.stage_rows()["reproduce"]["status"] == "interrupted"
    assert world.times()["cost_usd"] is not None


def test_a_rerun_against_a_vanished_instance_says_how_to_resume(world):
    world.control("hold-reproduce")
    process = world.start("--fetch-every-runs", "0")
    world.wait_for_stage("reproduce")
    wait_for_session_file(world, "stages/reproduce")
    process.kill()
    process.wait()
    world.ssh_down.write_text("")
    result = world.run()
    world.ssh_down.unlink()
    world.clean_up()
    assert result.returncode == 10
    assert f"--restore-from results/{world.session_dir().name}" in result.stderr


def run_until_lost_in_reproduce(world, *args):
    world.control("hold-reproduce")
    process = world.start("--fetch-every-runs", "0", *args)
    world.wait_for_stage("reproduce")
    wait_for_session_file(world, "results/reproduce/partial.txt")
    wait_for_session_file(world, "stages/reproduce")
    # Everything before reproduce reached this machine too, as it does within a minute.
    (world.run_dir / "results" / "matrix").mkdir(exist_ok=True)
    (world.run_dir / "results" / "matrix" / "run_1.json").write_text("finished before the loss")
    wait_for_session_file(world, "results/matrix/run_1.json")
    wait_for_session_file(world, "traces/durations.json")
    return process


def test_restore_reuses_results_and_reruns_only_what_lives_on_the_machine(world):
    process = run_until_lost_in_reproduce(world, "--reconnect-timeout", "1")
    world.ssh_down.write_text("")
    process.communicate(timeout=HEAD_TIMEOUT)
    world.ssh_down.unlink()
    lose_the_instance(world, process)
    session = world.session_dir()
    mine = session / "results" / "my_analysis.txt"
    mine.write_text("mine")
    old_log = (session / "logs" / "setup.log").read_text()

    result = world.run("--restore-from", f"results/{session.name}")
    assert result.returncode == 0, result.stdout + result.stderr
    # Setup and baselines live on the machine, and what was cut short goes on through its guards.
    assert world.calls == ["setup", "baselines", "reproduce"]
    # Everything the lost machine made was there before anything ran on this one.
    assert (world.run_dir / "results" / "matrix" / "run_1.json").read_text() == "finished before the loss"
    assert (world.run_dir / "results" / "gpu-tests" / "final.txt").exists()
    assert (world.run_dir / "traces" / "durations.json").read_text().strip() == "written-on-the-machine"
    # One directory, holding all of it, so a second restore would find it all.
    assert world.session_dir() == session
    assert mine.read_text() == "mine"
    assert (session / "results" / "reproduce" / "final.txt").exists()
    assert (session / "results" / "matrix" / "run_1.json").exists()
    assert all(row["status"] == "done" for row in world.stage_rows().values())
    # Only the one mirror towards the machine deletes, and the old machine's logs step aside.
    deleting = [line for line in world.rsync_log.read_text().splitlines() if "--delete" in line and "--server" not in line]
    assert all(line.rstrip().endswith("user@box:layover-run/") for line in deleting)
    before = list(session.glob("logs-before-*"))
    assert len(before) == 1 and (before[0] / "setup.log").read_text() == old_log


def test_a_second_interruption_loses_nothing_either(world):
    process = run_until_lost_in_reproduce(world, "--reconnect-timeout", "1")
    world.ssh_down.write_text("")
    process.communicate(timeout=HEAD_TIMEOUT)
    world.ssh_down.unlink()
    lose_the_instance(world, process)
    session = world.session_dir()

    world.control("hold-reproduce")
    process = world.start("--fetch-every-runs", "0", "--reconnect-timeout", "1", "--restore-from", f"results/{session.name}")
    world.wait_for_stage("reproduce")
    (world.run_dir / "results" / "matrix" / "run_2.json").write_text("second machine")
    wait_for_session_file(world, "results/matrix/run_2.json")
    world.ssh_down.write_text("")
    process.communicate(timeout=HEAD_TIMEOUT)
    world.ssh_down.unlink()
    lose_the_instance(world, process)

    result = world.run("--restore-from", f"results/{session.name}")
    assert result.returncode == 0, result.stdout + result.stderr
    for name in ("run_1.json", "run_2.json"):
        assert (world.run_dir / "results" / "matrix" / name).exists(), name
        assert (session / "results" / "matrix" / name).exists(), name
    assert world.session_dir() == session


def test_restore_runs_again_what_was_done_at_another_commit(world):
    process = run_until_lost_in_reproduce(world)
    lose_the_instance(world, process)
    session = world.session_dir()
    (world.repo / "README.md").write_text("fixed\n")
    result = world.run("--restore-from", f"results/{session.name}")
    assert result.returncode == 0, result.stdout + result.stderr
    assert world.calls == ORDER


@pytest.mark.parametrize("line", ["NVIDIA A10G, 23034, 595.71.05", "NVIDIA L4, 23034, 600.10.01"], ids=["another_card", "another_driver"])
def test_restore_refuses_another_gpu_or_driver_before_pushing_anything(world, line):
    # run_matrix.py would refuse this too, but only after setup, the baselines and the rest.
    process = run_until_lost_in_reproduce(world)
    lose_the_instance(world, process)
    world.control("gpu-line", line + "\n")
    session = world.session_dir()
    log_before = world.rsync_log.read_text()
    result = world.run("--restore-from", f"results/{session.name}")
    assert result.returncode == 10, result.stdout + result.stderr
    assert line in result.stderr and "NVIDIA L4, 23034, 595.71.05" in result.stderr
    assert world.rsync_log.read_text() == log_before
    assert world.calls == []


def test_restore_takes_a_session_directory_given_by_any_path(world):
    process = run_until_lost_in_reproduce(world)
    lose_the_instance(world, process)
    session = world.session_dir()
    result = world.run("--restore-from", f"{session}/")
    assert result.returncode == 0, result.stdout + result.stderr
    assert world.session_dir() == session


@pytest.mark.parametrize("where", ["outside", "no-stages"])
def test_restore_refuses_what_is_not_a_session_directory_here(world, tmp_path, where):
    target = tmp_path / "gpu-elsewhere" if where == "outside" else world.repo / "results" / "gpu-empty"
    target.mkdir(parents=True)
    if where == "outside":
        (target / "stages").mkdir()
    result = world.run("--restore-from", str(target))
    assert result.returncode == 2, result.stdout + result.stderr
    assert not world.ssh_log.exists()


def test_restore_cannot_take_stage_or_from(world):
    process = run_until_lost_in_reproduce(world)
    lose_the_instance(world, process)
    name = world.session_dir().name
    for extra in (["--stage", "traces"], ["--from", "traces"]):
        assert world.run("--restore-from", f"results/{name}", *extra).returncode == 2


def test_restore_refuses_a_machine_holding_another_session(world):
    process = run_until_lost_in_reproduce(world)
    world.clean_up()
    process.communicate(timeout=HEAD_TIMEOUT)
    other = world.repo / "results" / "gpu-20000101-000000"
    (other / "stages").mkdir(parents=True)
    result = world.run("--restore-from", "results/gpu-20000101-000000")
    assert result.returncode == 1, result.stdout + result.stderr
    assert "would mix them" in result.stderr
    assert not (world.run_dir / "results" / "matrix" / "other").exists()


def test_a_new_session_warns_of_one_that_never_finished(world):
    process = run_until_lost_in_reproduce(world)
    lose_the_instance(world, process)
    old = world.session_dir().name
    time.sleep(1.1)
    result = world.run("--stage", "sync")
    assert result.returncode == 0
    assert f"results/{old} never finished" in result.stderr and "--restore-from" in result.stderr


def test_secret_scan_covers_what_a_restore_would_push(world):
    process = run_until_lost_in_reproduce(world)
    lose_the_instance(world, process)
    session = world.session_dir()
    (session / "results" / "leak.txt").write_text(FAKE_API_KEY)
    log_before = world.rsync_log.read_text()
    result = world.run("--restore-from", f"results/{session.name}")
    assert result.returncode == 10, result.stdout + result.stderr
    assert f"results/{session.name}/results/leak.txt" in result.stderr
    assert FAKE_API_KEY not in result.stdout + result.stderr
    assert world.rsync_log.read_text() == log_before


def test_dry_run_of_a_restore_shows_the_push_and_contacts_nothing(world):
    session = world.repo / "results" / "gpu-20000101-000000"
    (session / "stages").mkdir(parents=True)
    result = world.run("--dry-run", "--restore-from", "results/gpu-20000101-000000")
    assert result.returncode == 0, result.stderr
    assert "setup and baselines rerun" in result.stdout
    restore_lines = [line for line in result.stdout.splitlines() if line.startswith("restore ") and "rsync" in line]
    assert len(restore_lines) == 2 and not any("--delete" in line for line in restore_lines)
    assert not world.ssh_log.exists()


def test_a_spot_notice_is_logged_and_results_come_back_at_once(world):
    world.control("imds", "not found\n404")
    world.control("hold-gpu-tests")
    process = world.start("--spot-notice", "--fetch-every", "3600")
    try:
        world.wait_for_stage("gpu-tests")
        # Smoke's end brings everything back once, so this looks at what is written after that.
        wait_for_session_file(world, ".smoke_reported")
        time.sleep(0.3)
        (world.run_dir / "results" / "gpu-tests" / "late.txt").write_text("x")
        time.sleep(0.5)
        assert not list((world.repo / "results").glob("gpu-*/results/gpu-tests/late.txt"))
        world.control("spot-notice", '{"action": "terminate", "time": "2026-11-05T00:00:00Z"}')
        wait_for_session_file(world, "results/gpu-tests/late.txt")
        notice = wait_for_session_file(world, "logs/spot_notice.json")
        assert json.loads(notice.read_text())["action"] == "terminate"
    finally:
        world.clean_up()
        output, _ = process.communicate(timeout=HEAD_TIMEOUT)
    assert output.count("SPOT INTERRUPTION NOTICE") == 1
    assert "--restore-from" in output


def test_the_watcher_is_harmless_off_aws_and_nothing_asks_without_the_flag(world):
    assert world.run("--spot-notice").returncode == 0
    assert world.calls == ORDER
    world.control("imds", "not found\n404")
    world.control("spot-notice", '{"action": "terminate"}')
    result = world.run("--from", "reproduce")
    assert result.returncode == 0
    assert "SPOT" not in result.stdout


# ---- what the stages are given ------------------------------------------------------------------


def stage_env(world, stage):
    rows = [line.split("|") for line in (world.home / "stage_env.log").read_text().splitlines()]
    last = [row for row in rows if row[0] == stage][-1]
    return dict(zip(("MATRIX_ARGS", "TRACES_ARGS", "SMOKE_PLAN"), last[1:]))


def test_plan_and_max_hours_reach_every_stage(world):
    result = world.run("--plan", "under-20h", "--max-hours", "16.5")
    assert result.returncode == 0, result.stdout + result.stderr
    for stage in ORDER:
        assert stage_env(world, stage) == {
            "MATRIX_ARGS": "--plan under-20h --max-hours 16.5", "TRACES_ARGS": "unset", "SMOKE_PLAN": "under-20h"}


def test_max_hours_alone_leaves_the_plan_to_the_config(world):
    assert world.run("--max-hours", "3").returncode == 0
    assert stage_env(world, "reproduce") == {"MATRIX_ARGS": "--max-hours 3", "TRACES_ARGS": "unset", "SMOKE_PLAN": "unset"}


def test_without_options_the_variables_stay_unset(world):
    # An empty one would count as set for make's ?=, and a leaked one must not reach a stage.
    env = dict(world.env, MATRIX_ARGS="--leaked", TRACES_ARGS="--leaked", SMOKE_PLAN="leaked")
    result = subprocess.run(world.command(), env=env, capture_output=True, text=True, timeout=HEAD_TIMEOUT)
    assert result.returncode == 0, result.stdout + result.stderr
    assert stage_env(world, "traces") == {"MATRIX_ARGS": "unset", "TRACES_ARGS": "unset", "SMOKE_PLAN": "unset"}


def test_allow_iam_role_reaches_the_traces_stage(world):
    assert world.run("--allow-iam-role").returncode == 0
    assert stage_env(world, "traces")["TRACES_ARGS"] == "--allow-iam-role"


def test_dry_run_shows_what_the_stages_are_given(world):
    result = world.run("--dry-run", "--plan", "p1", "--max-hours", "16", "--allow-iam-role")
    assert "MATRIX_ARGS=[--plan p1 --max-hours 16] SMOKE_PLAN=[p1] TRACES_ARGS=[--allow-iam-role]" in result.stdout
    assert "over 16 hours" in result.stdout
    assert "over 16 hours" not in world.run("--dry-run", "--max-hours", "16", "--continue-over-budget").stdout


def fake_session(world, settings):
    session = world.repo / "results" / "gpu-20000101-000000"
    (session / "stages").mkdir(parents=True)
    (session / "settings").write_text(settings)
    return f"results/{session.name}"


def test_dry_run_of_a_restore_shows_the_saved_settings(world):
    session = fake_session(world, "plan p1\nmax_hours 16\nallow_iam_role 1\ncontinue_over_budget 1\n")
    assert "MATRIX_ARGS=[--plan p1 --max-hours 16] SMOKE_PLAN=[p1] TRACES_ARGS=[]" in world.run("--dry-run", "--restore-from", session).stdout
    assert "MATRIX_ARGS=[--plan p2 --max-hours 16]" in world.run("--dry-run", "--restore-from", session, "--plan", "p2").stdout


@pytest.mark.parametrize("line", ["plan $(touch pwned)", "plan a b", "max_hours 1;touch pwned"])
def test_saved_settings_are_checked_like_the_command_line(world, line):
    session = fake_session(world, line + "\n")
    result = world.run("--restore-from", session)
    assert result.returncode == 1, result.stdout + result.stderr
    assert "saved" in result.stderr
    assert not world.ssh_log.exists() and not list(world.repo.rglob("pwned"))


@pytest.mark.parametrize(
    "extra,matrix_args,plan,calls",
    [
        ([], "--plan p1 --max-hours 16", "p1", ["setup", "baselines", "reproduce"]),
        (["--max-hours", "20"], "--plan p1 --max-hours 20", "p1", ["setup", "baselines", "reproduce"]),
        # Smoke priced the old plan, so it runs again; the rest goes by the commit as ever.
        (["--plan", "p2"], "--plan p2 --max-hours 16", "p2", ["setup", "smoke", "baselines", "reproduce"]),
    ],
    ids=["kept", "hours_overridden", "plan_overridden"],
)
def test_restore_keeps_plan_and_hours_unless_given_again(world, extra, matrix_args, plan, calls):
    process = run_until_lost_in_reproduce(world, "--plan", "p1", "--max-hours", "16", "--allow-iam-role")
    lose_the_instance(world, process)
    result = world.run("--restore-from", f"results/{world.session_dir().name}", *extra)
    assert result.returncode == 0, result.stdout + result.stderr
    assert world.calls == calls
    # The IAM role is this machine's question, asked again by preflight, so the answer does not carry.
    assert stage_env(world, "reproduce") == {"MATRIX_ARGS": matrix_args, "TRACES_ARGS": "unset", "SMOKE_PLAN": plan}


def test_settings_survive_an_instance_lost_before_any_fetch(world):
    world.control("hold-setup")
    process = world.start("--plan", "p1", "--fetch-every", "3600")
    world.wait_for_stage("setup")
    assert "plan p1" in wait_for_session_file(world, "settings").read_text()
    lose_the_instance(world, process)
    result = world.run("--restore-from", f"results/{world.session_dir().name}")
    assert result.returncode == 0, result.stdout + result.stderr
    assert stage_env(world, "reproduce")["SMOKE_PLAN"] == "p1"


def test_a_resume_here_keeps_plan_and_hours(world):
    assert world.run("--stop-after-smoke", "--plan", "p1", "--max-hours", "99").returncode == 0
    assert world.run().returncode == 0
    assert world.calls == ORDER
    assert stage_env(world, "reproduce")["MATRIX_ARGS"] == "--plan p1 --max-hours 99"


def test_a_resume_from_another_checkout_takes_the_machines_settings(world):
    assert world.run("--stop-after-smoke", "--plan", "p1").returncode == 0
    (world.session_dir() / "settings").unlink()
    assert world.run().returncode == 0
    assert stage_env(world, "reproduce")["SMOKE_PLAN"] == "p1"


def test_a_new_plan_is_priced_again_and_the_old_price_does_not_count(world):
    world.control("hours-smoke", REFINED)
    assert world.run("--stop-after-smoke", "--plan", "p1").returncode == 0
    (world.home / "hours-smoke").unlink()
    result = world.run("--plan", "p2", "--max-hours", "10")
    assert result.returncode == 0, result.stdout + result.stderr
    assert world.calls == ["setup", "smoke", "smoke", *ORDER[2:]]
    assert stage_env(world, "smoke")["SMOKE_PLAN"] == "p2"


def test_attaching_does_not_change_a_runs_settings(world):
    world.control("hold-gpu-tests")
    process = world.start("--plan", "p1")
    try:
        world.wait_for_stage("gpu-tests")
        attached = world.start("--plan", "p2")
        follower = Follower(attached)
        follower.wait_for("attaching")
        world.release("gpu-tests")
        output = follower.finish()
    finally:
        process.kill()
    assert "apply only when a run starts" in output
    assert stage_env(world, "reproduce")["MATRIX_ARGS"] == "--plan p1"
    assert (world.state / "settings").read_text().split() == ["plan", "p1", "auto_shutdown", "1", "deadman_hours", "48"]


# ---- the budget ---------------------------------------------------------------------------------


def test_over_budget_stops_after_smoke_and_says_so(world):
    world.control("hours-smoke", REFINED)
    result = world.run("--max-hours", "10")
    assert result.returncode == 20, result.stdout + result.stderr
    assert world.calls == ["setup", "smoke"]
    assert "OVER BUDGET" in result.stdout and "at least 12.5 hours" in result.stdout and "--max-hours is 10" in result.stdout
    assert "--continue-over-budget" in result.stdout and "shuts itself down" in result.stdout
    assert "Not finished" not in result.stdout
    assert world.stage_rows()["baselines"]["status"] == "pending"


def test_a_low_estimate_at_the_budget_goes_on(world):
    world.control("hours-smoke", REFINED)
    result = world.run("--max-hours", "12.5")
    assert result.returncode == 0, result.stdout + result.stderr
    assert world.calls == ORDER and "OVER BUDGET" not in result.stdout


def test_continue_over_budget_goes_on_and_says_so(world):
    world.control("hours-smoke", REFINED)
    result = world.run("--max-hours", "10", "--continue-over-budget")
    assert result.returncode == 0, result.stdout + result.stderr
    assert world.calls == ORDER
    assert result.stdout.count("going on, as --continue-over-budget asks") == 1 and "OVER BUDGET" not in result.stdout


@pytest.mark.parametrize("answer", [["--continue-over-budget"], ["--max-hours", "13"]], ids=["continue", "more_hours"])
def test_a_stopped_session_stays_stopped_until_answered(world, answer):
    world.control("hours-smoke", REFINED)
    assert world.run("--max-hours", "10").returncode == 20
    assert world.run().returncode == 20
    assert world.calls == ["setup", "smoke"]
    result = world.run(*answer)
    assert result.returncode == 0, result.stdout + result.stderr
    assert world.calls == ORDER and "OVER BUDGET" not in result.stdout


def test_continue_over_budget_is_not_remembered(world):
    world.control("hours-smoke", REFINED)
    assert world.run("--stop-after-smoke", "--max-hours", "10", "--continue-over-budget").returncode == 0
    assert world.run().returncode == 20


def test_the_machine_stops_itself_with_nobody_watching(world):
    world.control("hours-smoke", REFINED)
    world.control("hold-smoke")
    process = world.start("--max-hours", "10")
    world.wait_for_stage("smoke")
    process.kill()
    process.wait()
    world.release("smoke")
    deadline = time.time() + 20
    log = world.state / "logs" / "runner.log"
    while "runner finished" not in log.read_text():
        assert time.time() < deadline
        time.sleep(0.05)
    assert world.calls == ["setup", "smoke"]
    assert (world.state / "budget_stop").read_text().split() == ["12.5", "10"]


def test_attaching_judges_smoke_by_the_runs_own_budget(world):
    world.control("hours-smoke", REFINED)
    world.control("hold-smoke")
    process = world.start("--max-hours", "10")
    world.wait_for_stage("smoke")
    process.kill()
    process.wait()
    attached = world.start("--max-hours", "99")
    follower = Follower(attached)
    follower.wait_for("attaching")
    world.release("smoke")
    output = follower.finish()
    assert attached.returncode == 20, output
    assert "That low estimate is over --max-hours 10." in output


def test_a_restore_holds_the_same_stop(world):
    world.control("hours-smoke", REFINED)
    assert world.run("--max-hours", "10").returncode == 20
    shutil.rmtree(world.run_dir)
    (world.home / "calls.log").unlink()
    result = world.run("--restore-from", f"results/{world.session_dir().name}")
    assert result.returncode == 20, result.stdout + result.stderr
    assert world.calls == ["setup"]


def test_stop_after_smoke_reports_the_overrun_as_what_was_asked(world):
    world.control("hours-smoke", REFINED)
    result = world.run("--stop-after-smoke", "--max-hours", "10")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "over --max-hours 10" in result.stdout and "OVER BUDGET" not in result.stdout


def test_no_estimate_goes_on_and_says_so(world):
    result = world.run("--max-hours", "10")
    assert result.returncode == 0, result.stdout + result.stderr
    assert world.calls == ORDER and "no estimate" in result.stdout


def test_a_lone_stage_is_held_only_if_it_is_the_matrix(world):
    world.control("hours-smoke", REFINED)
    assert world.run("--max-hours", "10").returncode == 20
    assert world.run("--stage", "sanitize").returncode == 0
    assert world.run("--stage", "reproduce").returncode == 20
    assert world.calls == ["setup", "smoke", "sanitize"]


def test_an_earlier_failure_keeps_its_own_exit_code_over_the_budget(world):
    world.control("hours-smoke", REFINED)
    world.control("fail-setup")
    assert world.run("--continue-on-failure", "--max-hours", "10").returncode == 12


# ---- what a tree is called ------------------------------------------------------------------------


def synced_id(world):
    result = world.run("--stage", "sync")
    assert result.returncode == 0, result.stdout + result.stderr
    return (world.state / "synced_commit").read_text().strip()


def test_the_dirty_id_follows_untracked_contents(world):
    untracked = world.repo / "new_module.py"
    untracked.write_text("one\n")
    first = synced_id(world)
    assert "-dirty-" in first and synced_id(world) == first
    untracked.write_text("two\n")
    assert synced_id(world) != first
    untracked.write_text("one\n")
    assert synced_id(world) == first


@pytest.mark.parametrize(
    "path", ["results/calibration/calibration.json", "traces/job2.jsonl", "native/source.cc"], ids=["calibration", "trace", "source"])
def test_the_dirty_id_covers_everything_the_sync_sends(world, path):
    (world.repo / "new_module.py").write_text("one\n")
    (world.repo / path).parent.mkdir(parents=True, exist_ok=True)
    (world.repo / path).write_text("one\n")
    first = synced_id(world)
    (world.repo / path).write_text("two\n")
    assert synced_id(world) != first


def test_the_dirty_id_follows_a_symbolic_link(world):
    (world.repo / "new_module.py").write_text("one\n")
    (world.repo / "link").symlink_to("README.md")
    first = synced_id(world)
    (world.repo / "link").unlink()
    (world.repo / "link").symlink_to("Makefile")
    assert synced_id(world) != first


def test_the_dirty_id_ignores_what_stays_here_and_the_index(world):
    (world.repo / "new_module.py").write_text("one\n")
    first = synced_id(world)
    for name in (".notes/NOTES.md", "PRIVATE_NOTES.md", ".private/notes.txt", "results/old_local_result.txt", "native/build/host/lib.so"):
        (world.repo / name).write_text("changed\n")
    subprocess.run(["git", "add", "new_module.py"], cwd=world.repo, check=True)
    assert synced_id(world) == first


def test_a_tree_with_no_commits_still_tells_trees_apart(world):
    subprocess.run(["git", "update-ref", "-d", "HEAD"], cwd=world.repo, check=True)
    first = synced_id(world)
    assert first.startswith("no-commit-")
    (world.repo / "README.md").write_text("changed\n")
    assert synced_id(world) != first


def test_restore_of_a_dirty_tree_skips_what_was_done_at_this_id(world):
    # What the restore pushes grows between the machines, and the code does not.
    (world.repo / "new_module.py").write_text("one\n")
    process = run_until_lost_in_reproduce(world)
    lose_the_instance(world, process)
    result = world.run("--restore-from", f"results/{world.session_dir().name}")
    assert result.returncode == 0, result.stdout + result.stderr
    assert world.calls == ["setup", "baselines", "reproduce"]
    tree = (world.state / "synced_commit").read_text().strip()
    assert "-dirty-" in tree
    for stage in ORDER:
        assert (world.state / "stages" / stage).read_text().split()[4] == tree, stage


def test_restore_runs_everything_again_after_an_untracked_file_changed(world):
    (world.repo / "new_module.py").write_text("one\n")
    process = run_until_lost_in_reproduce(world)
    lose_the_instance(world, process)
    (world.repo / "new_module.py").write_text("two\n")
    result = world.run("--restore-from", f"results/{world.session_dir().name}")
    assert result.returncode == 0, result.stdout + result.stderr
    assert world.calls == ORDER


def test_restore_onto_the_same_machine_does_not_trust_its_old_markers(world):
    (world.repo / "new_module.py").write_text("one\n")
    process = run_until_lost_in_reproduce(world)
    world.clean_up()
    process.communicate(timeout=HEAD_TIMEOUT)
    (world.home / "calls.log").unlink()
    (world.repo / "new_module.py").write_text("two\n")
    result = world.run("--restore-from", f"results/{world.session_dir().name}")
    assert result.returncode == 0, result.stdout + result.stderr
    assert world.calls == ORDER


# ---- preflight --------------------------------------------------------------------------------


def preflight_refused(world, *needles):
    result = world.run()
    output = result.stdout + result.stderr
    assert result.returncode == 10, output
    for needle in needles:
        assert needle in output, output
    # Nothing that costs anything has started.
    assert not world.rsync_log.exists() and world.calls == []
    assert not list((world.repo / "results").glob("gpu-*"))
    assert "still billing" in output


def test_preflight_without_a_driver_says_so(world):
    (world.remote_bin / "nvidia-smi").unlink()
    if Path("/usr/bin/nvidia-smi").exists():
        pytest.skip("this machine has a real nvidia-smi")
    preflight_refused(world, "no NVIDIA driver")


def test_preflight_with_a_broken_driver_says_so(world):
    write_executable(world.remote_bin / "nvidia-smi", "#!/bin/bash\necho 'couldn't communicate with the driver'\nexit 9\n")
    preflight_refused(world, "the driver is not working")


def test_preflight_reports_too_little_gpu_memory(world):
    world.control("gpu-line", "NVIDIA T4, 15360, 595.71.05\n")
    preflight_refused(world, "15360 MiB")
    assert world.run("--min-gpu-mib", "15000").returncode == 0


def test_preflight_reports_a_driver_too_old_for_cuda_13(world):
    world.control("gpu-line", "NVIDIA L4, 23034, 570.195.03\n")
    preflight_refused(world, "570.195.03", "580")


def test_preflight_refuses_a_machine_that_is_not_x86(world):
    world.control("machine", "aarch64")
    preflight_refused(world, "aarch64", "x86_64")


def test_preflight_refuses_docker_that_cannot_run_containers(world):
    world.control("docker-broken")
    preflight_refused(world, "docker run --rm hello-world failed", "Cannot connect", "themselves containers")


def only_system_docker(world):
    (world.remote_bin / "docker").unlink()
    if any(Path(p, "docker").exists() for p in ("/usr/bin", "/bin", "/usr/sbin", "/sbin")) or Path("/.dockerenv").exists():
        pytest.skip("this machine has a docker of its own, or is itself a container")


def test_preflight_refuses_missing_docker_when_it_cannot_be_installed(world):
    only_system_docker(world)
    preflight_refused(world, "docker is missing", "themselves containers")


def test_preflight_lets_setup_install_docker_on_a_plain_vm(world):
    only_system_docker(world)
    write_executable(world.remote_bin / "apt-get", "#!/bin/bash\n")
    write_executable(world.remote_bin / "sudo", "#!/bin/bash\n")
    result = world.run("--stage", "preflight")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "setup installs it first" in result.stdout


@pytest.mark.skipif(os.geteuid() == 0, reason="root has no docker group to be missing from")
def test_preflight_passes_docker_that_only_root_can_use_for_setup_to_fix(world):
    world.control("docker-broken")
    world.control("sudo-docker-ok")
    result = world.run("--stage", "preflight")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "runs containers through sudo" in result.stdout
    assert "PROBLEM" not in result.stdout
    # The disk check looks at what sudo's docker reports, not at a guess.
    assert f"at {world.home}" in result.stdout


def test_preflight_refuses_a_machine_with_too_little_ram(world):
    world.control("ram-gib", "32")
    preflight_refused(world, "32 GiB of RAM", "64 GiB", "--min-ram-gb")
    assert world.run("--min-ram-gb", "32", "--stage", "preflight").returncode == 0
    assert world.run("--min-ram-gb", "0", "--stage", "preflight").returncode == 0


def test_preflight_allows_for_what_the_kernel_keeps_of_ram(world):
    world.control("ram-gib", "62")
    result = world.run("--stage", "preflight")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "ram: 62 GiB" in result.stdout


def test_preflight_refuses_what_was_enough_before_the_pinned_memory_was_counted(world):
    world.control("ram-gib", "48")
    preflight_refused(world, "48 GiB of RAM", "64 GiB")


def test_preflight_says_when_it_cannot_read_the_ram(world):
    write_executable(world.remote_bin / "getconf", "#!/bin/bash\nexit 1\n")
    preflight_refused(world, "cannot read this machine's RAM", "--min-ram-gb 0")


def test_preflight_refuses_an_instance_with_an_iam_role(world):
    world.control("imds", "my-role\n200")
    preflight_refused(world, "IAM role my-role", "--allow-iam-role")
    result = world.run("--allow-iam-role", "--stage", "preflight")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "not checked" in result.stdout


def test_preflight_passes_an_aws_instance_without_a_role(world):
    world.control("imds", "not found\n404")
    result = world.run("--stage", "preflight")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "iam role: none" in result.stdout


def test_preflight_passes_a_machine_off_aws(world):
    result = world.run("--stage", "preflight")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "no AWS metadata service answered" in result.stdout


def test_preflight_refuses_when_the_metadata_service_will_not_say(world):
    world.control("imds", "oops\n500")
    preflight_refused(world, "would not say", "500", "--allow-iam-role")


def test_preflight_refuses_when_it_has_no_curl_to_ask_with(world):
    (world.remote_bin / "curl").unlink()
    if any(Path(p, "curl").exists() for p in ("/usr/bin", "/bin", "/usr/sbin", "/sbin")):
        pytest.skip("this machine has a curl of its own")
    preflight_refused(world, "curl is missing", "--allow-iam-role")


def test_preflight_requires_the_disk(world):
    result = subprocess.run(
        [BASH, str(world.repo / "scripts" / "remote_gpu.sh"), "user@box", "--min-disk-gb", "999999"],
        env=world.env, capture_output=True, text=True, timeout=HEAD_TIMEOUT,
    )
    assert result.returncode == 10
    assert "--min-disk-gb" in result.stdout
    assert not world.rsync_log.exists()


def test_low_disk_does_not_block_a_resume(world):
    world.control("fail-sanitize")
    assert world.run().returncode == 14
    (world.home / "fail-sanitize").unlink()
    result = subprocess.run(
        [BASH, str(world.repo / "scripts" / "remote_gpu.sh"), "user@box", "--poll", "0.1", "--min-disk-gb", "999999"],
        env=world.env, capture_output=True, text=True, timeout=HEAD_TIMEOUT,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_stages_get_a_docker_group_added_after_the_runner_started(world):
    # Setup can add the user to the group, which the running process cannot see for itself.
    write_executable(world.remote_bin / "id", '#!/bin/bash\ncase "$*" in -u) echo 1000 ;; -un) echo gpu ;; "-nG gpu") echo gpu docker ;; -nG) echo gpu ;; esac\n')
    write_executable(world.remote_bin / "sg", '#!/bin/bash\necho sg >> "$HOME/sg.log"\nexec bash -c "$3"\n')
    assert world.run().returncode == 0
    assert world.calls == ORDER
    assert (world.home / "sg.log").read_text().split() == ["sg"] * len(ORDER)


def test_failed_preflight_on_a_resume_still_fetches(world):
    world.control("fail-sanitize")
    assert world.run().returncode == 14
    (world.run_dir / "results" / "before_docker_broke.txt").write_text("x")
    world.control("docker-broken")
    assert world.run().returncode == 10
    assert (world.session_dir() / "results" / "before_docker_broke.txt").exists()


def test_unreachable_machine_is_a_preflight_failure(world):
    world.ssh_down.write_text("")
    result = world.run()
    assert result.returncode == 10
    assert "cannot reach" in result.stderr
    assert not world.rsync_log.exists()


@pytest.mark.parametrize("target", ["gpu-smoke", "test-gpu", "traces"])
def test_missing_make_target_fails_before_any_sync(world, target):
    text = (world.repo / "Makefile").read_text().split("\n")
    keep, skipping = [], False
    for line in text:
        if not line.startswith("\t"):
            skipping = line.startswith(target + ":")
        if not skipping:
            keep.append(line)
    (world.repo / "Makefile").write_text("\n".join(keep))
    result = world.run()
    assert result.returncode == 10
    assert f"no {target} target" in result.stderr
    assert not world.rsync_log.exists() and world.calls == []


def test_a_target_named_like_a_directory_must_be_phony(world):
    # make would call traces/ up to date, and the stage would pass having done nothing.
    text = (world.repo / "Makefile").read_text().split("\n", 1)[1]
    (world.repo / "Makefile").write_text(text)
    result = world.run()
    assert result.returncode == 10
    assert "not .PHONY" in result.stderr
    assert not world.rsync_log.exists() and world.calls == []


def test_a_single_stage_does_not_need_the_other_targets(world):
    (world.repo / "Makefile").write_text("sanitize:\n\t@bash stage_stub.sh sanitize\n")
    assert world.run("--stage", "sanitize").returncode == 0


def test_missing_baselines_script_fails_before_any_sync(world):
    (world.repo / "scripts" / "setup_baselines.sh").unlink()
    result = world.run()
    assert result.returncode == 10
    assert "setup_baselines.sh" in result.stderr
    assert not world.rsync_log.exists()


def test_the_real_makefile_has_a_target_for_every_make_stage(world):
    shutil.copy(SCRIPTS.parent / "Makefile", world.repo / "Makefile")
    # The local checks all come before the first contact, so a machine that is down ends this there.
    world.ssh_down.write_text("")
    result = world.run()
    assert result.returncode == 10, result.stdout + result.stderr
    assert "cannot reach" in result.stderr and "Makefile" not in result.stderr


def test_calibration_syncs_and_nothing_else_of_results(world):
    calibration = world.repo / "results" / "calibration" / "calibration.json"
    calibration.parent.mkdir(parents=True)
    calibration.write_text('{"made": "here"}')
    (world.repo / "results" / "gpu-old").mkdir()
    (world.repo / "results" / "gpu-old" / "big.bin").write_text("x")
    assert world.run("--stage", "sync").returncode == 0
    arrived = {str(p.relative_to(world.run_dir / "results")) for p in (world.run_dir / "results").rglob("*") if p.is_file()}
    assert arrived == {"calibration/calibration.json"}
    assert json.loads((world.run_dir / "results" / "calibration" / "calibration.json").read_text()) == {"made": "here"}
    # A re-sync leaves what the machine made elsewhere in results/ alone.
    (world.run_dir / "results" / "made.txt").write_text("there")
    assert world.run("--stage", "sync").returncode == 0
    assert (world.run_dir / "results" / "made.txt").exists()


def test_calibration_the_machine_made_survives_a_resync_from_a_tree_without_one(world):
    assert world.run("--stage", "sync").returncode == 0
    made = world.run_dir / "results" / "calibration" / "calibration.json"
    made.parent.mkdir(parents=True)
    made.write_text("made there")
    assert world.run("--stage", "sync").returncode == 0
    assert made.read_text() == "made there"


def test_secret_scan_covers_the_calibration_too(world):
    calibration = world.repo / "results" / "calibration" / "calibration.json"
    calibration.parent.mkdir(parents=True)
    calibration.write_text(FAKE_API_KEY)
    result = world.run("--stage", "sync")
    assert result.returncode == 10, result.stdout + result.stderr
    assert "results/calibration/calibration.json" in result.stderr
    assert not world.rsync_log.exists()


def test_no_prepared_trajectories_fail_before_any_sync(world):
    (world.repo / "traces" / "prepared" / "manifest.json").unlink()
    result = world.run()
    assert result.returncode == 10
    assert "traces/prepared/manifest.json is missing" in result.stderr
    assert not world.rsync_log.exists()
    assert world.run("--stage", "sanitize").returncode == 0


def test_key_inside_a_trace_stops_the_sync_without_printing_it(world):
    trace = world.repo / "traces" / "job2.jsonl"
    trace.write_text(json.dumps({"env": {"SOME_API_KEY": FAKE_API_KEY}}) + "\n")
    result = world.run()
    assert result.returncode == 10
    assert "traces/job2.jsonl" in result.stderr
    assert FAKE_API_KEY not in result.stdout + result.stderr
    assert not world.rsync_log.exists()
    assert not world.run_dir.exists()


def test_private_key_in_an_oddly_named_file_is_caught(world):
    (world.repo / "notes.txt").write_text("-----BEGIN OPENSSH " + "PRIVATE KEY-----\nabc\n")
    result = world.run()
    assert result.returncode == 10
    assert "notes.txt" in result.stderr


def test_excluded_files_do_not_trip_the_secret_scan(world):
    (world.repo / ".env").write_text(FAKE_API_KEY)
    assert world.run().returncode == 0


def test_secret_scan_can_be_turned_off_for_false_alarms(world):
    (world.repo / "traces" / "job2.jsonl").write_text(f'"{FAKE_API_KEY}"\n')
    result = world.run("--no-secret-scan")
    assert result.returncode == 0
    assert "--no-secret-scan" in result.stderr


def test_secret_scan_does_not_trip_on_these_scripts():
    # The runner's own patterns must not match the runner's own source.
    hits = subprocess.run(
        ["grep", "-IlE", "-e", "(^|[^A-Za-z0-9])sk-[A-Za-z0-9_-]{32,}", "-e", "AKIA[0-9A-Z]{16}",
         "-e", "hf_[A-Za-z0-9]{30,}", "-e", "gh[pousr]_[A-Za-z0-9]{36}", "-e", "BEGIN [A-Z ]*PRIVATE KEY",
         str(REAL_RUNNER), str(REAL_SETUP), __file__],
        capture_output=True, text=True,
    )
    assert hits.stdout == ""


def test_missing_exclude_file_stops_the_sync(world):
    (world.repo / ".git" / "info" / "exclude").unlink()
    result = world.run()
    assert result.returncode == 10
    assert ".git/info/exclude" in result.stderr


def test_the_documented_commands_are_executable():
    # The README says scripts/remote_gpu.sh user@host, so bash cannot be assumed in front of it.
    for script in ("remote_gpu.sh", "setup_gpu.sh", "setup_baselines.sh"):
        assert os.access(SCRIPTS / script, os.X_OK), script


def test_real_tree_keeps_its_private_files_home():
    """The real tree, listed with the runner's real excludes, as its dry run prints them."""
    root = REAL_RUNNER.parent.parent
    exclude = root / ".git" / "info" / "exclude"
    if not exclude.exists():
        pytest.skip("not a git checkout")
    dry = subprocess.run([BASH, str(REAL_RUNNER), "user@box", "--dry-run"], capture_output=True, text=True, check=True).stdout
    sync_line = next(line for line in dry.splitlines() if line.startswith("sync"))
    excludes = [word for word in shlex.split(sync_line.replace("\\ ", " ")) if word.startswith("--exclude")]
    excludes = [word for word in excludes if word != "--exclude=/traces/"]
    listing = subprocess.run(["rsync", "--list-only", "-rl", *excludes, f"{root}/"], capture_output=True, text=True, check=True).stdout
    paths = [line.split(maxsplit=4)[-1] for line in listing.splitlines() if line.strip()]
    names = {path.split("/")[0] for path in paths}
    patterns = {line.strip().rstrip("/") for line in exclude.read_text().splitlines() if line.strip() and not line.startswith("#")}
    assert patterns
    assert not names & (patterns | {".git", ".venv", ".notes", ".hypothesis", "results"})
    assert not any("__pycache__" in path or path.startswith("native/build") for path in paths)
    assert "scripts" in names and "Makefile" in names and "layover" in names


# ---- setup_baselines.sh, with stubs in place of what a GPU machine has ------------------------

# Ubuntu 24.04's Python, as the environment uv built on it runs it: no ensurepip.
BASELINE_PYTHON = """#!/bin/bash
echo "The virtual environment was not created successfully because ensurepip is not available."
exit 1
"""

BASELINE_UV = """#!/bin/bash
echo "uv $*" >> "$HOME/calls.log"
for venv; do :; done
mkdir -p "$venv/bin"
printf '#!/bin/bash\\necho "pip $*" >> "$HOME/calls.log"\\n' > "$venv/bin/pip"
printf '#!/bin/bash\\nexit 0\\n' > "$venv/bin/python"
chmod +x "$venv/bin/pip" "$venv/bin/python"
"""


class BaselineWorld:
    def __init__(self, base):
        self.repo = base / "repo"
        self.home = base / "home"
        self.bin = base / "bin"
        self.home.mkdir()
        write_executable(self.repo / "scripts" / "setup_baselines.sh", (SCRIPTS / "setup_baselines.sh").read_text())
        write_executable(self.bin / "python3.12", BASELINE_PYTHON)

    def run(self, *args):
        env = {k: v for k, v in os.environ.items() if k != "XDG_CACHE_HOME"}
        env.update(HOME=str(self.home), PATH=f"{self.bin}:/usr/bin:/bin")
        return subprocess.run(
            [BASH, str(self.repo / "scripts" / "setup_baselines.sh"), *args],
            env=env, capture_output=True, text=True, timeout=HEAD_TIMEOUT,
        )

    @property
    def calls(self):
        path = self.home / "calls.log"
        return path.read_text().splitlines() if path.exists() else []

    def cached_uv(self):
        write_executable(self.home / ".cache" / "layover" / "uv-0.13.1" / "uv", BASELINE_UV)


@pytest.fixture
def baseline_world(tmp_path):
    if shutil.which("uv", path="/usr/bin:/bin"):
        pytest.skip("this machine has a uv of its own")
    return BaselineWorld(tmp_path)


@pytest.mark.parametrize("where", ["cache", "path"])
def test_baselines_get_an_environment_on_a_python_without_ensurepip(baseline_world, where):
    if where == "cache":
        baseline_world.cached_uv()
    else:
        write_executable(baseline_world.bin / "uv", BASELINE_UV)
    result = baseline_world.run("vllm")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "uv venv --clear --seed --python python3.12 .venvs/vllm" in baseline_world.calls
    assert any(call.startswith("pip install --quiet vllm==") for call in baseline_world.calls)


def test_baselines_rerun_keeps_a_finished_environment(baseline_world):
    baseline_world.cached_uv()
    assert baseline_world.run("vllm").returncode == 0
    (baseline_world.home / "calls.log").unlink()
    result = baseline_world.run("vllm")
    assert result.returncode == 0, result.stdout + result.stderr
    assert not [call for call in baseline_world.calls if call.startswith("uv ")]


def test_baselines_rebuild_an_environment_left_without_pip(baseline_world):
    baseline_world.cached_uv()
    (baseline_world.repo / ".venvs" / "vllm" / "bin").mkdir(parents=True)
    result = baseline_world.run("vllm")
    assert result.returncode == 0, result.stdout + result.stderr
    assert any("--clear" in call for call in baseline_world.calls)


def test_baselines_without_uv_or_ensurepip_say_what_to_install(baseline_world):
    result = baseline_world.run("vllm")
    assert result.returncode == 1
    assert "python3-venv" in result.stderr and "ERROR" in result.stderr


# ---- setup_gpu.sh, with stubs in place of what a GPU machine has ------------------------------

SETUP_NVIDIA_SMI = """#!/bin/bash
echo "NVIDIA L4, 23034, ${DRIVER_VERSION:-595.71.05}"
"""

SETUP_PYTHON = """#!/bin/bash
# Stands in for the virtual environment's python. Reads a script from stdin or -c and acts on it.
case "$*" in
    *'sys.version_info'*) exit 0 ;;
    *'torch.version.cuda or'*) echo "${TORCH_CUDA-13.2}"; exit 0 ;;
    *'torch.__version__'*) echo "2.15.0+cpu"; exit 0 ;;
    *--version*) echo "Python 3.12.0"; exit 0 ;;
    *'import torch, triton'*) echo "torch: 2.15.0"; exit 0 ;;
esac
cat > /dev/null
[ -z "${TORCH_BROKEN:-}" ] || { echo "torch.cuda.is_available() is False"; exit 1; }
exit 0
"""

APT_GET_STUB = """#!/bin/bash
echo "apt-get $*" >> "$SETUP_LOG"
case " $* " in
    *" -s "*)
        [ -n "${APT_NO_PACKAGE:-}" ] && { echo "E: Unable to locate package"; exit 100; }
        [ -n "${APT_PULLS_DRIVER:-}" ] && echo "Inst nvidia-driver-595 (595.71.05 NVIDIA CUDA)"
        echo "Inst cuda-nvcc-13-2 (13.2.2-1 NVIDIA CUDA)"
        exit 0
        ;;
    *" install "*cuda-toolkit-*)
        [ -n "${APT_INSTALLS_NOTHING:-}" ] && exit 0
        mkdir -p "$TOOLKIT_DIR/bin"
        printf '#!/bin/bash\\necho "Build cuda_13.2.r13.2/compiler"\\n' > "$TOOLKIT_DIR/bin/nvcc"
        printf '#!/bin/bash\\necho "Version 2026.1.1.0"\\n' > "$TOOLKIT_DIR/bin/compute-sanitizer"
        chmod +x "$TOOLKIT_DIR/bin/nvcc" "$TOOLKIT_DIR/bin/compute-sanitizer"
        ;;
esac
"""


class SetupWorld:
    def __init__(self, base):
        self.repo = base / "repo with space"
        self.stubs = base / "stubs"
        self.log = base / "setup.log"
        self.toolkit = base / "toolkit"
        self.os_release = base / "os-release"
        self.os_release.write_text('ID=ubuntu\nVERSION_ID="24.04"\nPRETTY_NAME="Ubuntu 24.04"\n')
        self.apt = base / "apt"
        (self.apt / "sources.list.d").mkdir(parents=True)
        write_executable(self.repo / "scripts" / "setup_gpu.sh", REAL_SETUP.read_text())
        (self.repo / "native").mkdir()
        (self.repo / "native" / "Makefile").write_text("cuda:\n\t@echo built\n")
        self.cache = base / "cache"
        write_executable(self.cache / "layover" / "uv-0.13.1" / "uv", f'#!/bin/bash\necho "uv $*" >> "{self.log}"\n')
        write_executable(self.repo / ".venv" / "bin" / "python", SETUP_PYTHON)
        for name, text in {
            "uname": '#!/bin/bash\ncase "$1" in -s) echo Linux ;; -m) echo x86_64 ;; *) echo 6.8.0 ;; esac\n',
            "nvidia-smi": SETUP_NVIDIA_SMI,
            "apt-get": APT_GET_STUB,
            "dpkg": '#!/bin/bash\necho "dpkg $*" >> "$SETUP_LOG"\n[ "$1" != -s ]\n',
            "sudo": '#!/bin/bash\n[ -z "${NO_SUDO:-}" ] || exit 1\n[ "$1" = -n ] && shift\n'
                    'echo "sudo $*" >> "$SETUP_LOG"\n"$@"\n',
            "curl": '#!/bin/bash\necho "curl $*" >> "$SETUP_LOG"\n'
                    '[ -z "${CURL_FAILS:-}" ] || exit 22\n'
                    'while [ $# -gt 0 ]; do [ "$1" = -o ] && : > "$2"; shift; done\n',
            "make": '#!/bin/bash\necho "make $* CUDA_HOME=${CUDA_HOME:-}" >> "$SETUP_LOG"\necho "GNU Make 4"\n',
            "g++": "#!/bin/bash\necho g++ 13\n",
            "sha256sum": "#!/bin/bash\ncat > /dev/null\n",
            "id": '#!/bin/bash\n[ "$1" = -u ] && echo 1000 || echo gpu\n',
            "docker": '#!/bin/bash\necho "docker $*" >> "$SETUP_LOG"\n[ -z "${DOCKER_BROKEN:-}" ] || exit 125\n',
            "usermod": '#!/bin/bash\necho "usermod $*" >> "$SETUP_LOG"\n',
        }.items():
            write_executable(self.stubs / name, text)

    def run(self, *args, **env):
        full_env = dict(
            os.environ,
            PATH=f"{self.toolkit}/bin:{self.stubs}:{os.environ['PATH']}",
            SETUP_LOG=str(self.log),
            TOOLKIT_DIR=str(self.toolkit),
            OS_RELEASE=str(self.os_release),
            XDG_CACHE_HOME=str(self.cache),
            APT_DIR=str(self.apt),
            **env,
        )
        return subprocess.run(
            [BASH, str(self.repo / "scripts" / "setup_gpu.sh"), *args],
            env=full_env, capture_output=True, text=True, timeout=HEAD_TIMEOUT,
        )

    @property
    def logged(self):
        return self.log.read_text() if self.log.exists() else ""


@pytest.fixture
def setup_world(tmp_path):
    return SetupWorld(tmp_path)


def test_setup_without_a_driver_explains_and_installs_nothing(setup_world):
    (setup_world.stubs / "nvidia-smi").unlink()
    if shutil.which("nvidia-smi"):
        pytest.skip("this machine has a real nvidia-smi")
    result = setup_world.run()
    assert result.returncode == 1
    assert "no NVIDIA driver" in result.stderr
    assert setup_world.logged == ""


def test_setup_rejects_a_driver_below_the_cuda13_floor(setup_world):
    result = setup_world.run(DRIVER_VERSION="570.195.03")
    assert result.returncode == 1
    assert "570.195.03" in result.stderr and "580" in result.stderr and "595.71.05" in result.stderr
    assert setup_world.logged == ""


def test_setup_warns_for_a_driver_under_minor_version_compatibility(setup_world):
    result = setup_world.run(DRIVER_VERSION="580.65.06")
    assert result.returncode == 0, result.stderr
    assert "minor version compatibility" in result.stderr


def test_setup_is_quiet_about_a_driver_at_the_toolkit_version(setup_world):
    result = setup_world.run(DRIVER_VERSION="595.71.05")
    assert result.returncode == 0, result.stderr
    assert "minor version compatibility" not in result.stderr


def test_setup_compares_driver_versions_numerically(setup_world):
    # As strings, 1000 sorts before 595.
    result = setup_world.run(DRIVER_VERSION="1000.10.1")
    assert result.returncode == 0, result.stderr
    assert "minor version compatibility" not in result.stderr


def test_setup_check_only_touches_nothing(setup_world):
    result = setup_world.run("--check-only")
    assert result.returncode == 0, result.stderr
    assert "not found, will be installed" in result.stdout
    assert "apt-get" not in setup_world.logged and "dpkg" not in setup_world.logged


def test_setup_installs_the_toolkit_from_nvidias_repository_at_torchs_version(setup_world):
    result = setup_world.run()
    assert result.returncode == 0, result.stderr
    log = setup_world.logged
    # The URL scheme and key NVIDIA documents, verified against the live repository.
    assert (
        "https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb" in log
    )
    assert "sudo dpkg -i" in log
    # Versioned: plain cuda-toolkit tracks the newest release, not the one torch is built for.
    assert "install -y --no-install-recommends cuda-toolkit-13-2" in log
    assert "cudatools.pub" not in log


def test_setup_uses_the_ubuntu_2204_repository(setup_world):
    setup_world.os_release.write_text('ID=ubuntu\nVERSION_ID="22.04"\nPRETTY_NAME="Ubuntu 22.04"\n')
    assert setup_world.run().returncode == 0
    assert "/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb" in setup_world.logged


def test_setup_points_the_native_build_at_the_right_toolkit(setup_world):
    result = setup_world.run()
    assert result.returncode == 0, result.stderr
    assert f"make -C native cuda CUDA_HOME={setup_world.toolkit}" in setup_world.logged
    env_file = (setup_world.repo / ".venv" / "cuda_env.sh").read_text()
    assert str(setup_world.toolkit / "bin") in env_file and f'CUDA_HOME="{setup_world.toolkit}"' in env_file


def test_setup_is_idempotent_about_the_toolkit(setup_world):
    assert setup_world.run().returncode == 0
    first = setup_world.logged
    assert "cuda-toolkit-13-2" in first
    setup_world.log.unlink()
    assert setup_world.run().returncode == 0
    assert "apt-get" not in setup_world.logged
    # The native build runs every time and is make's to skip.
    assert "make -C native cuda" in setup_world.logged


def test_setup_replaces_a_toolkit_of_another_version(setup_world):
    other = setup_world.base_other = setup_world.toolkit.parent / "other"
    write_executable(other / "nvcc", '#!/bin/bash\necho "Build cuda_12.8.r12.8/compiler"\n')
    write_executable(other / "compute-sanitizer", "#!/bin/bash\n")
    write_executable(setup_world.stubs / "nvcc", (other / "nvcc").read_text())
    write_executable(setup_world.stubs / "compute-sanitizer", "#!/bin/bash\n")
    assert setup_world.run().returncode == 0
    assert "cuda-toolkit-13-2" in setup_world.logged


def test_setup_requires_compute_sanitizer_with_the_toolkit(setup_world):
    write_executable(setup_world.stubs / "nvcc", '#!/bin/bash\necho "Build cuda_13.2.r13.2/compiler"\n')
    result = setup_world.run(APT_INSTALLS_NOTHING="1")
    assert result.returncode == 1
    assert "compute-sanitizer" in result.stderr


def test_setup_will_not_pull_in_a_driver(setup_world):
    result = setup_world.run(APT_PULLS_DRIVER="1")
    assert result.returncode == 1
    assert "would change the NVIDIA driver" in result.stderr
    assert "install -y --no-install-recommends cuda-toolkit" not in setup_world.logged


def test_setup_without_sudo_fails_clearly(setup_world):
    result = setup_world.run(NO_SUDO="1")
    assert result.returncode == 1
    assert "passwordless sudo" in result.stderr


def test_setup_reports_a_missing_repository(setup_world):
    result = setup_world.run(CURL_FAILS="1")
    assert result.returncode == 1
    assert "no NVIDIA repository" in result.stderr


def test_setup_reports_a_package_apt_cannot_find(setup_world):
    result = setup_world.run(APT_NO_PACKAGE="1")
    assert result.returncode == 1
    assert "apt cannot install cuda-toolkit-13-2" in result.stderr


def test_setup_needs_a_cuda_build_of_torch(setup_world):
    result = setup_world.run(TORCH_CUDA="")
    assert result.returncode == 1
    assert "CPU-only torch" in result.stderr


def test_setup_warns_when_torch_moves_to_another_cuda(setup_world):
    # This covers a torch release later than the one the driver table was read for.
    write_executable(setup_world.toolkit / "bin" / "nvcc", '#!/bin/bash\necho "Build cuda_13.4.r13.4/compiler"\n')
    write_executable(setup_world.toolkit / "bin" / "compute-sanitizer", "#!/bin/bash\n")
    result = setup_world.run(TORCH_CUDA="13.4")
    assert result.returncode == 0, result.stderr
    assert "torch moved to CUDA 13.4" in result.stderr


def test_setup_reports_when_torch_cannot_use_the_gpu(setup_world):
    result = setup_world.run(TORCH_BROKEN="1")
    assert result.returncode == 1
    assert "PyTorch cannot use the GPU" in result.stderr


def test_setup_needs_the_native_makefile(setup_world):
    (setup_world.repo / "native" / "Makefile").unlink()
    result = setup_world.run()
    assert result.returncode == 1
    assert "native/Makefile" in result.stderr


def test_setup_records_versions_with_the_results(setup_world):
    assert setup_world.run().returncode == 0
    recorded = (setup_world.repo / "results" / "setup_versions.txt").read_text()
    assert "NVIDIA L4" in recorded and "nvcc:" in recorded and "compute-sanitizer:" in recorded


def test_setup_checks_docker_before_any_large_download(setup_world):
    result = setup_world.run(DOCKER_BROKEN="1", NO_SUDO="1")
    assert result.returncode == 1
    assert "hello-world" in result.stderr and "themselves containers" in result.stderr
    assert "uv " not in setup_world.logged and "cuda-toolkit" not in setup_world.logged
    assert "make -C native" not in setup_world.logged


def test_setup_adds_the_user_to_the_docker_group_when_only_root_can_use_it(setup_world):
    write_executable(setup_world.stubs / "docker", '#!/bin/bash\n[ -n "${AS_ROOT:-}" ] || exit 1\n')
    write_executable(setup_world.stubs / "sudo", '#!/bin/bash\n[ "$1" = -n ] && shift\necho "sudo $*" >> "$SETUP_LOG"\nAS_ROOT=1 "$@"\n')
    result = setup_world.run()
    assert result.returncode == 0, result.stderr
    assert "usermod -aG docker gpu" in setup_world.logged


def test_setup_leaves_docker_alone_for_a_run_without_traces(setup_world):
    # A rented container: Docker cannot run, and need not.
    result = setup_world.run(DOCKER_BROKEN="1", LAYOVER_NO_DOCKER="1")
    assert result.returncode == 0, result.stderr
    assert "Docker: left alone" in result.stdout
    assert "docker run" not in setup_world.logged and "usermod" not in setup_world.logged
    assert "cuda-toolkit-13-2" in setup_world.logged


def test_setup_uses_nvidias_repository_where_the_image_already_has_it(setup_world):
    (setup_world.apt / "sources.list.d" / "cuda.list").write_text(
        "deb [signed-by=/usr/share/keyrings/cuda-archive-keyring.gpg] https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/ /\n"
    )
    result = setup_world.run()
    assert result.returncode == 0, result.stderr
    assert "already configured" in result.stdout
    assert "cuda-keyring" not in setup_world.logged and "dpkg -i" not in setup_world.logged
    assert "install -y --no-install-recommends cuda-toolkit-13-2" in setup_world.logged


def test_setup_unpacks_uv_without_setting_owners(setup_world):
    # Root on a pod's network volume may not chown, and tar as root tries to.
    shutil.rmtree(setup_world.cache / "layover")
    write_executable(setup_world.stubs / "tar", '#!/bin/bash\necho "tar $*" >> "$SETUP_LOG"\nwhile [ $# -gt 0 ]; do [ "$1" = -C ] && dir=$2; shift; done\nprintf "#!/bin/bash\\necho \\"uv \\$*\\" >> \\"$SETUP_LOG\\"\\n" > "$dir/uv"; chmod +x "$dir/uv"\n')
    result = setup_world.run()
    assert result.returncode == 0, result.stderr
    assert "tar --no-same-owner -xzf" in setup_world.logged


def test_setup_needs_no_key_or_model_account():
    script = REAL_SETUP.read_text()
    for banned in ("HF_TOKEN", "huggingface-cli login", "API_KEY", "ssh-keygen"):
        assert banned not in script, banned


def test_setup_refuses_a_machine_that_is_not_x86(setup_world):
    write_executable(setup_world.stubs / "uname", '#!/bin/bash\ncase "$1" in -s) echo Linux ;; -m) echo aarch64 ;; esac\n')
    result = setup_world.run("--check-only")
    assert result.returncode == 1
    assert "x86_64" in result.stderr


def test_setup_refuses_to_run_off_linux(setup_world):
    write_executable(setup_world.stubs / "uname", "#!/bin/bash\necho Darwin\n")
    result = setup_world.run("--check-only")
    assert result.returncode == 1
    assert "needs Linux" in result.stderr


# ---- the real thing: real ssh and rsync against a container running sshd -------------------------

DOCKERFILE = """FROM ubuntu:24.04
RUN apt-get update && apt-get install -y --no-install-recommends openssh-server rsync make procps \\
    && rm -rf /var/lib/apt/lists/*
RUN useradd -m -s /bin/bash gpu && mkdir -p /run/sshd /home/gpu/.ssh && chown -R gpu /home/gpu
COPY nvidia-smi docker /usr/local/bin/
RUN chmod +x /usr/local/bin/nvidia-smi /usr/local/bin/docker
CMD ["bash", "-c", "/usr/sbin/sshd && exec sleep infinity"]
"""

docker_e2e = pytest.mark.skipif(
    os.environ.get("LAYOVER_DOCKER_E2E") != "1" or shutil.which("docker") is None or shutil.which("ssh-keygen") is None,
    reason="set LAYOVER_DOCKER_E2E=1, with docker and ssh on this machine",
)


class Container:
    def __init__(self, base):
        self.name = f"layover-e2e-{os.getpid()}"
        context = base / "context"
        context.mkdir()
        (context / "Dockerfile").write_text(DOCKERFILE)
        (context / "nvidia-smi").write_text(NVIDIA_SMI_STUB)
        (context / "docker").write_text(DOCKER_STUB)
        # An x86_64 image even on an arm host, so preflight checks a real answer from uname.
        subprocess.run(
            ["docker", "build", "-q", "--platform", "linux/amd64", "-t", "layover-sshd-e2e", str(context)],
            check=True, capture_output=True,
        )
        # The key and known-hosts file are throwaways, and the host's ~/.ssh stays untouched.
        self.key = base / "ssh dir" / "id_test"
        self.key.parent.mkdir()
        subprocess.run(["ssh-keygen", "-q", "-t", "ed25519", "-N", "", "-f", str(self.key)], check=True)
        self.known_hosts = base / "known_hosts"
        subprocess.run(
            ["docker", "run", "-d", "--rm", "--platform", "linux/amd64", "--name", self.name, "-p", "127.0.0.1::22", "layover-sshd-e2e"],
            check=True, capture_output=True,
        )
        self.port = subprocess.run(
            ["docker", "port", self.name, "22/tcp"], check=True, capture_output=True, text=True
        ).stdout.strip().rsplit(":", 1)[1]
        public = Path(str(self.key) + ".pub").read_text()
        self.exec_root(f"echo {shlex.quote(public)} > /home/gpu/.ssh/authorized_keys && chown gpu /home/gpu/.ssh/authorized_keys && chmod 600 /home/gpu/.ssh/authorized_keys")
        time.sleep(1)

    def exec_root(self, script, check=True):
        return subprocess.run(["docker", "exec", self.name, "bash", "-c", script], check=check, capture_output=True, text=True)

    def exec_gpu(self, script, check=True):
        return subprocess.run(
            ["docker", "exec", "-u", "gpu", "-w", "/home/gpu", "-e", "HOME=/home/gpu", self.name, "bash", "-c", script],
            check=check, capture_output=True, text=True,
        )

    def stop(self):
        subprocess.run(["docker", "rm", "-f", self.name], capture_output=True)


@pytest.fixture(scope="module")
def container(tmp_path_factory):
    c = Container(tmp_path_factory.mktemp("e2e"))
    yield c
    c.stop()


class RealRun:
    """A repository driving the container for real. Each gets its own run directory."""

    def __init__(self, container, base, run_dir):
        self.container = container
        self.run_dir = run_dir
        self.repo = base / "my repo"
        self.repo.mkdir(parents=True)
        build_repo(self.repo)
        self.container.exec_gpu("rm -f ~/calls.log ~/fail-* ~/hold-*")

    def command(self, *args):
        return [
            BASH, str(self.repo / "scripts" / "remote_gpu.sh"), "gpu@127.0.0.1",
            "-p", self.container.port, "-i", str(self.container.key),
            "--ssh-option", f"UserKnownHostsFile={self.container.known_hosts}",
            "--remote-dir", self.run_dir, "--poll", "1", "--min-disk-gb", "0", "--min-ram-gb", "0", "--allow-iam-role",
            "--no-auto-shutdown", *args,
        ]

    def run(self, *args, timeout=180):
        return subprocess.run(self.command(*args), capture_output=True, text=True, timeout=timeout)

    def start(self, *args):
        return subprocess.Popen(self.command(*args), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

    def calls(self):
        return self.container.exec_gpu("cat ~/calls.log 2>/dev/null || true").stdout.split()

    def stage_status(self, stage):
        return self.container.exec_gpu(f"cut -d' ' -f1 {self.run_dir}/.layover-run/stages/{stage} 2>/dev/null || true").stdout.strip()

    def wait_for_stage(self, stage, status="running", timeout=90):
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.stage_status(stage) == status:
                return
            time.sleep(0.5)
        raise AssertionError(f"{stage} never reached {status}")

    def session_dir(self):
        found = sorted((self.repo / "results").glob("gpu-*"))
        assert len(found) == 1, found
        return found[0]


@docker_e2e
def test_e2e_dropped_connections_do_not_stop_the_work(container, tmp_path):
    run = RealRun(container, tmp_path, "run-drop")
    container.exec_gpu("touch ~/hold-gpu-tests")
    process = run.start("--reconnect-timeout", "120")
    try:
        run.wait_for_stage("gpu-tests")
        pid = container.exec_gpu("cat run-drop/.layover-run/runner.pid").stdout.strip()
        # Kill the listener and every live session, as a network drop would.
        container.exec_root("ps -eo pid,args | awk '/[s]shd/ {print $1}' | xargs -r kill -9")
        time.sleep(4)
        assert container.exec_gpu(f"kill -0 {pid} && echo alive").stdout.strip() == "alive"
        assert run.stage_status("gpu-tests") == "running"
        container.exec_root("/usr/sbin/sshd")
        container.exec_gpu("rm ~/hold-gpu-tests")
        output, _ = process.communicate(timeout=180)
    finally:
        process.kill()
    assert process.returncode == 0, output
    assert "lost the connection" in output and "reconnected" in output
    assert run.calls() == ORDER
    assert (run.session_dir() / "results" / "reproduce" / "final.txt").exists()
    assert (run.session_dir() / "traces" / "durations.json").exists()
    times = json.loads((run.session_dir() / "stage_times.json").read_text())
    assert times["gpu"].startswith("NVIDIA L4")
    assert {row["name"]: row["status"] for row in times["stages"]}["gpu-tests"] == "done"


@docker_e2e
def test_e2e_killing_this_side_then_status_then_resume(container, tmp_path):
    run = RealRun(container, tmp_path, "run-resume")
    container.exec_gpu("touch ~/hold-gpu-tests")
    process = run.start()
    run.wait_for_stage("gpu-tests")
    process.kill()
    process.wait()
    status = run.run("--status")
    assert status.returncode == 0
    assert "runner:   running" in status.stdout
    assert ["gpu-tests", "running"] in [line.split()[:2] for line in status.stdout.splitlines()]
    container.exec_gpu("rm ~/hold-gpu-tests")
    result = run.run()
    assert result.returncode == 0, result.stdout + result.stderr
    assert "attaching" in result.stdout
    assert run.calls() == ORDER


@docker_e2e
def test_e2e_killing_the_runner_resumes_from_that_stage(container, tmp_path):
    run = RealRun(container, tmp_path, "run-killed")
    container.exec_gpu("touch ~/hold-sanitize")
    process = run.start()
    run.wait_for_stage("sanitize")
    process.kill()
    process.wait()
    container.exec_gpu("kill -9 -- -$(cat run-killed/.layover-run/runner.pid); rm ~/hold-sanitize")
    result = run.run()
    assert result.returncode == 0, result.stdout + result.stderr
    assert run.calls() == ["setup", "smoke", "baselines", "gpu-tests", "sanitize", "sanitize", "traces", "reproduce"]


@docker_e2e
def test_e2e_failure_fetches_partial_results_then_resumes(container, tmp_path):
    run = RealRun(container, tmp_path, "run-fail")
    container.exec_gpu("touch ~/fail-traces")
    result = run.run()
    assert result.returncode == 16, result.stdout + result.stderr
    session = run.session_dir()
    assert (session / "results" / "traces" / "partial.txt").exists()
    assert not (session / "results" / "traces" / "final.txt").exists()
    assert "stub failure" in (session / "logs" / "traces.log").read_text()
    assert run.calls() == ["setup", "smoke", "baselines", "gpu-tests", "sanitize", "traces"]
    container.exec_gpu("rm ~/fail-traces")
    second = run.run()
    assert second.returncode == 0, second.stdout + second.stderr
    assert run.calls() == ["setup", "smoke", "baselines", "gpu-tests", "sanitize", "traces", "traces", "reproduce"]
    assert run.session_dir() == session


@docker_e2e
def test_e2e_restore_onto_a_new_directory_with_real_rsync(container, tmp_path):
    run = RealRun(container, tmp_path, "run-spot-1")
    container.exec_gpu("touch ~/fail-reproduce")
    result = run.run("--plan", "p1", "--max-hours", "16")
    assert result.returncode == 17, result.stdout + result.stderr
    session = run.session_dir()
    # A new machine: the run directory is gone, as a new instance's would be, and so is its record.
    container.exec_gpu("rm -rf run-spot-1 ~/calls.log ~/fail-reproduce")
    result = run.run("--restore-from", f"results/{session.name}")
    assert result.returncode == 0, result.stdout + result.stderr
    assert run.calls() == ["setup", "baselines", "reproduce"]
    # Through real ssh and the settings that came back: a multi-line argument, and no flags this time.
    assert container.exec_gpu("tail -n 1 ~/stage_env.log").stdout.strip() == "reproduce|--plan p1 --max-hours 16|--allow-iam-role|p1"
    assert run.session_dir() == session
    assert (session / "results" / "reproduce" / "final.txt").exists()
    assert (session / "results" / "gpu-tests" / "final.txt").exists()
    assert (session / "traces" / "durations.json").exists()


@docker_e2e
def test_e2e_an_instance_that_vanishes_is_reported_not_waited_for(container, tmp_path):
    run = RealRun(container, tmp_path, "run-vanish")
    container.exec_gpu("touch ~/hold-reproduce")
    process = run.start("--reconnect-timeout", "5", "--fetch-every-runs", "0")
    try:
        run.wait_for_stage("reproduce")
        deadline = time.time() + 60
        while not list((run.repo / "results").glob("gpu-*/results/reproduce/partial.txt")):
            assert time.time() < deadline
            time.sleep(0.5)
        # Nothing answers any more, so every later attempt fails as it would for a lost instance.
        container.exec_root("ps -eo pid,args | awk '/[s]shd/ {print $1}' | xargs -r kill -9")
        output, _ = process.communicate(timeout=120)
    finally:
        process.kill()
        container.exec_root("/usr/sbin/sshd", check=False)
        container.exec_gpu("rm -f ~/hold-reproduce")
        # The next test connects at once, and a listener takes a moment to come up.
        time.sleep(1)
    assert process.returncode == 1, output
    assert "--restore-from results/" in output and "user@NEW-HOST" in output
    assert (run.session_dir() / "results" / "reproduce" / "partial.txt").exists()


@docker_e2e
def test_e2e_excludes_hold_and_nothing_secret_arrives(container, tmp_path):
    run = RealRun(container, tmp_path, "run-excludes")
    (run.repo / "results" / "calibration").mkdir(parents=True)
    (run.repo / "results" / "calibration" / "calibration.json").write_text("{}")
    result = run.run("--stage", "sync")
    assert result.returncode == 0, result.stdout + result.stderr
    listing = set(container.exec_gpu("cd run-excludes && find . -mindepth 1 | sed 's#^./##'").stdout.split())
    for gone in (".git", ".venv", ".notes", ".private", "PRIVATE_NOTES.md", ".env", "deploy.pem",
                 "native/build", "layover/__pycache__", ".hypothesis", "results/old_local_result.txt"):
        assert gone not in listing, gone
    for kept in ("traces/job1.jsonl", "native/source.cc", "scripts/remote_gpu.sh", "Makefile", ".layover-run/synced_commit",
                 "results/calibration/calibration.json"):
        assert kept in listing, kept
    # A real environment and written traces survive re-syncing, and ~/ outside the directory is untouched.
    container.exec_gpu("mkdir -p run-excludes/.venv && echo x > run-excludes/.venv/marker && echo keep > ~/outside "
                       "&& echo made > run-excludes/traces/durations.json")
    (run.repo / "README.md").unlink()
    assert run.run("--stage", "sync").returncode == 0
    listing = set(container.exec_gpu("cd run-excludes && find . -mindepth 1 | sed 's#^./##'").stdout.split())
    assert ".venv/marker" in listing and "traces/durations.json" in listing and "README.md" not in listing
    assert container.exec_gpu("cat ~/outside").stdout.strip() == "keep"


@docker_e2e
def test_e2e_real_tree_keeps_its_private_files_home(container):
    root = REAL_RUNNER.parent.parent
    exclude = root / ".git" / "info" / "exclude"
    if not exclude.exists():
        pytest.skip("not a git checkout")
    result = subprocess.run(
        [BASH, str(REAL_RUNNER), "gpu@127.0.0.1", "-p", container.port, "-i", str(container.key),
         "--ssh-option", f"UserKnownHostsFile={container.known_hosts}", "--remote-dir", "real-tree",
         "--min-disk-gb", "0", "--min-ram-gb", "0", "--allow-iam-role", "--stage", "sync"],
        capture_output=True, text=True, timeout=180,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    listing = set(container.exec_gpu("cd real-tree && find . -mindepth 1 | sed 's#^./##'").stdout.split())
    patterns = [
        line.strip().rstrip("/") for line in exclude.read_text().splitlines() if line.strip() and not line.startswith("#")
    ]
    assert patterns
    for gone in [*patterns, ".git", ".venv", "results/gpu"]:
        assert gone not in listing, gone
    assert "scripts/remote_gpu.sh" in listing and "layover/engine.py" in listing
    assert ".layover-run/synced_commit" in listing


@docker_e2e
def test_e2e_a_container_without_docker_is_refused_by_name(container, tmp_path):
    # This machine is what many rentals are: a container, with no way to run more of them.
    run = RealRun(container, tmp_path, "run-nodocker")
    container.exec_root("mv /usr/local/bin/docker /usr/local/bin/docker.off")
    try:
        result = run.run()
    finally:
        container.exec_root("mv /usr/local/bin/docker.off /usr/local/bin/docker")
    assert result.returncode == 10, result.stdout + result.stderr
    assert "itself a container" in result.stdout
    assert run.calls() == []


# ---- the machine never bills idle --------------------------------------------------------------


def shutdowns(world):
    path = world.home / "shutdown.log"
    return path.read_text().splitlines() if path.exists() else []


def test_the_machine_shuts_down_a_minute_after_the_run_ends(world):
    result = world.run()
    assert result.returncode == 0, result.stdout + result.stderr
    calls = shutdowns(world)
    # The dead-man first, then the end; each replaces the one pending.
    assert calls[0].startswith("-h +2880 ") and calls[-1].startswith("-h +1 ")
    assert "shuts itself down when its run ends" in result.stdout + result.stderr


def test_a_failed_stage_also_shuts_the_machine_down(world):
    world.control("fail-gpu-tests")
    result = world.run()
    assert result.returncode == 13
    assert shutdowns(world)[-1].startswith("-h +1 ")


def test_the_dead_man_limit_follows_the_plan(world):
    world.run("--stop-after-smoke")
    assert shutdowns(world)[0].startswith("-h +120 ")
    (world.home / "shutdown.log").unlink()
    world.run("--max-hours", "16.5")
    assert shutdowns(world)[0].startswith("-h +1350 ")
    (world.home / "shutdown.log").unlink()
    world.run("--max-session-hours", "0.5", "--from", "setup")
    assert shutdowns(world)[0].startswith("-h +30 ")


def test_no_auto_shutdown_leaves_the_machine_alone(world):
    result = world.run("--no-auto-shutdown")
    assert result.returncode == 0
    assert shutdowns(world) == []
    assert "still billing (--no-auto-shutdown)" in result.stdout + result.stderr


def test_without_passwordless_sudo_preflight_refuses_before_anything_runs(world):
    world.control("no-sudo")
    result = world.run()
    assert result.returncode == 10
    assert "auto-shutdown needs shutdown and passwordless sudo" in result.stdout + result.stderr
    assert world.calls == []
    assert "no run started on it" in result.stdout + result.stderr
    assert world.run("--no-auto-shutdown").returncode == 0


def test_detach_returns_while_the_run_goes_on(world):
    world.control("hold-gpu-tests")
    result = world.run("--detach")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Detached" in result.stdout and "--fetch-only" in result.stdout
    world.wait_for_stage("gpu-tests")
    assert world.stage_status("reproduce") is None
    world.release("gpu-tests")
    world.wait_for_stage("reproduce", "done", timeout=60)
    deadline = time.time() + 20
    while not shutdowns(world)[-1].startswith("-h +1 "):
        assert time.time() < deadline, shutdowns(world)
        time.sleep(0.05)


def test_fetch_only_stops_a_machine_whose_run_ended(world):
    assert world.run().returncode == 0
    (world.home / "shutdown.log").unlink()
    result = world.run("--fetch-only")
    assert result.returncode == 0, result.stdout + result.stderr
    assert shutdowns(world) == ["-h +1 Layover results fetched"]
    (world.home / "shutdown.log").unlink()
    assert world.run("--fetch-only", "--leave-running").returncode == 0
    assert shutdowns(world) == []


def test_fetch_only_leaves_a_running_machine_alone(world):
    world.control("hold-gpu-tests")
    process = world.start()
    try:
        world.wait_for_stage("gpu-tests")
        log = world.home / "shutdown.log"
        before = log.read_text() if log.exists() else ""
        assert world.run("--fetch-only").returncode == 0
        assert (log.read_text() if log.exists() else "") == before
    finally:
        world.release("gpu-tests")
        process.wait(timeout=120)


def test_stages_runs_only_the_listed_stages_in_the_usual_order(world):
    result = world.run("--stages", "traces setup smoke")
    assert result.returncode == 0, result.stdout + result.stderr
    assert world.calls == ["setup", "smoke", "traces"]


@pytest.mark.parametrize("args", [["--stages", "setup nonsense"], ["--stages", "setup", "--stage", "smoke"], ["--stages", "setup", "--from", "smoke"]])
def test_stages_refuses_unknown_names_and_mixing(world, args):
    result = world.run(*args)
    assert result.returncode == 2 and not world.ssh_log.exists()


# ---- a RunPod pod: a container that stops itself through runpodctl ------------------------------


def runpod(world, **env):
    world.extra_env.update({"RUNPOD_POD_ID": "pod123", "RUNPOD_API_KEY": "pod-scoped", **env})
    # A pod has no passwordless sudo to fall back on, and needs none.
    world.control("no-sudo")
    return ("--runpod", "--remote-dir", "layover-run")


def runpodctl_calls(world):
    path = world.home / "runpodctl.log"
    return path.read_text().splitlines() if path.exists() else []


def stop_timer(world):
    path = world.state / "stop_timer"
    return path.read_text().strip() if path.exists() else None


def wait_for(predicate, timeout=20):
    deadline = time.time() + timeout
    while not predicate():
        assert time.time() < deadline
        time.sleep(0.05)


def test_a_pod_stops_itself_through_runpodctl_and_never_through_shutdown(world):
    args = runpod(world)
    result = world.run(*args)
    output = result.stdout + result.stderr
    assert result.returncode == 0, output
    assert "auto-stop: runpodctl" in output and "for pod pod123 (the stop is not tried here)" in output
    # Preflight never stops the pod to prove that it can.
    assert not any("stop" in call for call in runpodctl_calls(world))
    # The dead-man went first, and the end of the run replaced it.
    wait_for(lambda: stop_timer(world) == "1 Layover run ended")
    assert shutdowns(world) == []
    assert "Delete the pod once the results are here" in output


def test_the_pods_timer_stops_it_and_a_newer_one_replaces_it(world):
    args = runpod(world)
    world.control("hold-gpu-tests")
    process = world.start(*args)
    try:
        world.wait_for_stage("gpu-tests")
        assert stop_timer(world) == "2880 Layover dead-man limit reached"
        dead_man = int((world.state / "stop_timer.pid").read_text())
        os.kill(dead_man, 0)
    finally:
        world.release("gpu-tests")
        process.wait(timeout=120)
    wait_for(lambda: stop_timer(world) == "1 Layover run ended")
    with pytest.raises(ProcessLookupError):
        for _ in range(100):
            os.kill(dead_man, 0)
            time.sleep(0.05)
    # What a timer does when it fires, with the commands of either version of runpodctl.
    for marker, expected in ((None, "pod stop pod123"), ("runpodctl-v1", "stop pod pod123")):
        if marker:
            world.control(marker)
        fired = subprocess.run(
            [BASH, "scripts/remote_gpu.sh", "__stop_pod_after", "0", "test"], cwd=world.run_dir,
            env=dict(world.env, HOME=str(world.home), PATH=f"{world.remote_bin}:/usr/bin:/bin"), capture_output=True, text=True, timeout=30,
        )
        assert fired.returncode == 0, fired.stdout + fired.stderr
        assert "stopping pod pod123: test" in fired.stdout
        assert runpodctl_calls(world)[-1] == expected


def test_a_pod_that_cannot_stop_itself_is_refused_before_anything_runs(world):
    args = runpod(world)
    del world.extra_env["RUNPOD_POD_ID"]
    result = world.run(*args)
    assert result.returncode == 10 and "RUNPOD_POD_ID is not set" in result.stdout
    world.extra_env["RUNPOD_POD_ID"] = "pod123"
    del world.extra_env["RUNPOD_API_KEY"]
    result = world.run(*args)
    assert result.returncode == 10 and "RUNPOD_API_KEY is not set" in result.stdout
    world.extra_env["RUNPOD_API_KEY"] = "pod-scoped"
    os.remove(world.remote_bin / "runpodctl")
    result = world.run(*args)
    assert result.returncode == 10 and "runpodctl is missing" in result.stdout
    assert world.calls == [] and not world.rsync_log.exists()
    result = world.run(*args, "--no-auto-shutdown")
    assert result.returncode == 0, result.stdout + result.stderr


def test_a_pod_runs_on_its_volume_disk_unless_told_otherwise(world):
    result = world.run("--runpod", "--dry-run")
    assert "run directory: /workspace/layover-run" in result.stdout
    assert "the pod stops itself with runpodctl" in result.stdout
    assert "run directory: elsewhere" in world.run("--runpod", "--remote-dir", "elsewhere", "--dry-run").stdout
    assert "run directory: layover-run" in world.run("--dry-run").stdout


def test_the_caches_go_to_the_volume_and_a_new_container_runs_setup_again(world):
    args = runpod(world)
    result = world.run(*args, "--stages", "setup smoke")
    assert result.returncode == 0, result.stdout + result.stderr
    cache = world.run_dir / ".cache"
    assert (world.home / "setup_env.log").read_text().splitlines() == [f"1|{cache}|{cache}/huggingface"]
    # A restarted pod is a new container: the volume kept the markers, the container disk lost the toolkit.
    (world.home / ".layover-setup-here").unlink()
    result = world.run(*args, "--stage", "gpu-tests")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "this container is new" in result.stdout
    assert world.calls == ["setup", "smoke", "setup", "gpu-tests"]
    assert (world.home / ".layover-setup-here").exists()
    # The same container keeps what setup did.
    assert world.run(*args, "--stage", "gpu-tests").returncode == 0
    assert world.calls[-1] == "gpu-tests" and world.calls.count("setup") == 2


def test_a_machine_is_never_marked_as_a_pod_without_the_flag(world):
    assert world.run("--stages", "setup").returncode == 0
    assert not (world.home / ".layover-setup-here").exists()
    assert "runpod" not in (world.state / "settings").read_text()


def test_without_the_traces_stage_docker_is_neither_checked_nor_set_up(world):
    world.control("docker-broken")
    result = world.run("--stages", "setup smoke")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "docker: not needed, since the traces stage is not in this run" in result.stdout
    assert (world.home / "setup_env.log").read_text().split("|")[0] == "1"
    # With the traces stage, or for a lone preflight, Docker must run a container.
    assert world.run("--stages", "setup traces").returncode == 10
    assert world.run("--stage", "preflight").returncode == 10


def test_a_containers_memory_limit_is_its_ram(world, tmp_path):
    cgroup = tmp_path / "cgroup"
    cgroup.mkdir()
    world.extra_env["LAYOVER_CGROUP_ROOT"] = str(cgroup)
    world.control("ram-gib", "512")
    (cgroup / "memory.max").write_text(f"{50 * 2**30}\n")
    preflight_refused(world, "ram: 50 GiB, this container's limit on a 512 GiB host", "the machine has 50 GiB of RAM")
    assert world.run("--min-ram-gb", "48", "--stage", "preflight").returncode == 0
    # No limit, in either version of cgroups, leaves the host's RAM.
    (cgroup / "memory.max").write_text("max\n")
    result = world.run("--stage", "preflight")
    assert result.returncode == 0 and "ram: 512 GiB" in result.stdout
    (cgroup / "memory.max").unlink()
    (cgroup / "memory").mkdir()
    (cgroup / "memory" / "memory.limit_in_bytes").write_text("9223372036854771712\n")
    assert "ram: 512 GiB" in world.run("--stage", "preflight").stdout


def test_a_root_login_without_rsync_gets_it_installed(world, tmp_path):
    # A bin with what a bare container has: no rsync until apt-get puts it there.
    bare = tmp_path / "bare_bin"
    for tool in ("bash", "cat", "df", "awk", "dirname", "mkdir", "ls", "cut", "head", "tr", "sed", "grep", "hostname", "env", "sleep", "ln"):
        found = shutil.which(tool)
        if found:
            (bare / tool).parent.mkdir(parents=True, exist_ok=True)
            (bare / tool).symlink_to(found)
    for stub in ("nvidia-smi", "uname", "getconf", "curl", "runpodctl"):
        (bare / stub).symlink_to(world.remote_bin / stub)
    write_executable(bare / "id", '#!/bin/bash\ncase "$1" in -u) echo 0 ;; *) echo root ;; esac\n')
    write_executable(bare / "apt-get", f'#!/bin/bash\necho "apt-get $*" >> "$HOME/apt.log"\ncase " $* " in *" install "*) ln -sf {shlex.quote(shutil.which("rsync"))} {shlex.quote(str(bare))}/rsync ;; esac\n')
    args = runpod(world, TEST_REMOTE_PATH=str(bare))
    result = world.run(*args, "--stage", "preflight", "--no-auto-shutdown")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "rsync: missing; installing it" in result.stdout
    assert "install -y --no-install-recommends rsync" in (world.home / "apt.log").read_text()
