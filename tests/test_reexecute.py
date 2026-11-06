"""Re-execution against a fake docker on PATH: fidelity, replacement, resume, cleanup, labels."""

import hashlib
import io
import json
import os
import platform
import stat
import subprocess
import sys
import urllib.error
from pathlib import Path

import pytest

from layover.replay import reexecute as rx
from layover.replay.reexecute import Options, convert, docker_exec_argv, docker_run_argv, execute, machine_label, similarity
from layover.replay.trace import load_jsonl
from layover.replay.trajectories import MANIFEST_SCHEMA, PREPARED_SCHEMA, SUBMIT_MARK

SUBMISSION = "sub_mini"
ENVIRONMENT = {"cwd": "/testbed", "env": {"PAGER": "cat"}, "timeout": 5, "interpreter": ["bash", "-c"]}

SHIM = r'''#!{python} -S
import fcntl, json, os, signal, sys, time
home = os.environ["SHIM_DIR"]
argv = sys.argv[1:]
with open(os.path.join(home, "lock"), "w") as lock:
    fcntl.flock(lock, fcntl.LOCK_EX)
    with open(os.path.join(home, "calls.jsonl"), "a") as f:
        f.write(json.dumps(argv) + "\n")
    with open(os.path.join(home, "config.json")) as f:
        config = json.load(f)
    with open(os.path.join(home, "state.json")) as f:
        state = json.load(f)
    code = 0
    out = ""
    err = ""
    behaviour = None
    if argv[0] == "version":
        if config.get("daemon_down"):
            code = 1
        elif "Arch" in argv[-1]:
            out = config.get("server_arch", "amd64") + "\n"
        else:
            out = "29.0.0\n"
    elif argv[:2] == ["image", "inspect"]:
        image = argv[-1]
        if image not in state["images"]:
            code = 1
        elif "Architecture" in argv[3]:
            if config.get("inspect_fail"):
                code = 1
            out = config.get("image_arch", "amd64") + "\n"
        elif "RepoDigests" in argv[3]:
            out = json.dumps([image + "@sha256:abc"]) + "\n"
        else:
            out = "sha256:id\n"
    elif argv[0] == "pull":
        image = argv[-1]
        message = config.get("pull_fail", {}).get(image)
        if message:
            code, err = 1, message
        else:
            state["images"].append(image)
    elif argv[0] == "rmi":
        state["images"].remove(argv[-1])
    elif argv[0] == "run":
        name = argv[argv.index("--name") + 1]
        image = argv[argv.index("sleep") - 1]
        if image in config.get("run_fail", []):
            code, err = 125, config.get("run_fail_text", "cannot start")
        else:
            state["containers"][name] = image
            out = "cid\n"
    elif argv[0] == "rm":
        state["containers"].pop(argv[-1], None)
    elif argv[0] == "ps":
        out = "".join(n + "\n" for n in state["containers"])
    elif argv[0] == "exec":
        state["execs"] = state.get("execs", 0) + 1
        name = argv[argv.index("bash") - 1]
        image = state["containers"][name]
        behaviour = json.loads(argv[-1])
        behaviour.update(config.get("exec_override", {}).get(image, {}))
        if config.get("kill_parent_at_exec") == state["execs"]:
            with open(os.path.join(home, "state.json"), "w") as f:
                json.dump(state, f)
            os.kill(os.getppid(), signal.SIGKILL)
    with open(os.path.join(home, "state.json"), "w") as f:
        json.dump(state, f)
if behaviour is not None:
    time.sleep(behaviour.get("sleep", 0))
    out, code = behaviour.get("out", ""), behaviour.get("rc", 0)
sys.stdout.write(out)
sys.stderr.write(err)
sys.exit(code)
'''


class Shim:
    def __init__(self, home: Path, config=None, images=()) -> None:
        self.home = home
        self.write_config(config or {})
        (home / "state.json").write_text(json.dumps({"images": list(images), "containers": {}}))
        script = home / "bin" / "docker"
        script.parent.mkdir()
        script.write_text(SHIM.replace("{python}", sys.executable))
        script.chmod(script.stat().st_mode | stat.S_IEXEC)
        self.bin = script.parent

    def write_config(self, config) -> None:
        (self.home / "config.json").write_text(json.dumps(config))

    def calls(self, first=None) -> list[list[str]]:
        path = self.home / "calls.jsonl"
        if not path.exists():
            return []
        calls = [json.loads(line) for line in path.read_text().splitlines()]
        return [c for c in calls if first is None or c[0] == first]

    @property
    def state(self) -> dict:
        return json.loads((self.home / "state.json").read_text())


@pytest.fixture
def shim(tmp_path, monkeypatch):
    home = tmp_path / "shim"
    home.mkdir()
    fake = Shim(home)
    monkeypatch.setenv("SHIM_DIR", str(home))
    monkeypatch.setenv("PATH", f"{fake.bin}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setattr(rx, "IMDS", "http://127.0.0.1:9/latest")
    monkeypatch.setattr(platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(os, "cpu_count", lambda: 16)
    return fake


def behaves(rc=0, out="", sleep=0):
    return json.dumps({"rc": rc, "out": out, "sleep": sleep})


def image_of(instance):
    return f"swebench/sweb.eval.x86_64.{instance}:latest"


def document(instance, commands=None):
    """Turns of (behaviour, recorded rc, recorded output); the last is the submit."""
    commands = commands or [[(behaves(0, "ok\n"), 0, "ok\n")], [(behaves(1, "no\n"), 1, "no\n"), (behaves(0, "yes\n"), 0, "yes\n")]]
    turns = []
    for number, group in enumerate(commands):
        turns.append({
            "turn": number, "new_tokens": 300 if number == 0 else 20, "output_tokens": 10, "context_before": 0,
            "provider_prompt_tokens": 100 * (number + 1), "provider_completion_tokens": 8, "provider_reasoning_tokens": 2,
            "recorded_gap_s": 2.5,
            "commands": [
                {"command": c, "recorded_returncode": rc, "recorded_exception": False,
                 "recorded_output_sha256": None, "recorded_output": out}
                for c, rc, out in group
            ],
        })
    return {
        "schema": PREPARED_SCHEMA, "instance_id": instance, "submission": SUBMISSION, "s3_key": f"k/{instance}", "raw_sha256": "sha" + instance,
        "mini_version": "2.1.0", "model": "open/model", "trajectory_format": "mini-swe-agent-1.1", "image": image_of(instance),
        "environment": ENVIRONMENT, "initial_prompt_tokens": 300, "peak_context_tokens": 500, "render_checked": True, "drift_tokens": 0,
        "submission_sha256": None, "turns": turns,
    }


def write_prepared(directory: Path, tasks, reserves=(), commands=None, selection="seeded"):
    directory.mkdir(parents=True, exist_ok=True)
    commands = commands or {}
    for instance in [*tasks, *reserves]:
        (directory / f"{instance}.json").write_text(json.dumps(document(instance, commands.get(instance))))
    manifest = {
        "schema": MANIFEST_SCHEMA, "submission": SUBMISSION, "selection": selection, "tasks": list(tasks), "reserves": list(reserves),
        "tokenizer_repo": "Qwen/Qwen3-4B", "tokenizer_revision": "rev", "shared_prefix_tokens": 100,
        "walked": 50, "exclusion_rate": 0.2, "exclusion_counts": {"context_over_limit": 10}, "max_context": 32768,
        "bias_note": "The longest sessions are gone.",
        "raw": {i: {"sha256": "sha" + i} for i in [*tasks, *reserves]},
    }
    (directory / "manifest.json").write_text(json.dumps(manifest))
    return directory


def run_cli(tmp_path, *extra):
    with pytest.raises(SystemExit) as raised:
        rx.main(["run", "--prepared", str(tmp_path / "prepared"), "--out", str(tmp_path / "results"), "--workers", "1", *extra])
    return raised.value.code


def log_of(tmp_path, instance):
    return rx.read_log(tmp_path / "results" / SUBMISSION / f"{instance}.jsonl")


def run_state(tmp_path):
    return json.loads((tmp_path / "results" / SUBMISSION / "run.json").read_text())


@pytest.mark.parametrize("no_network", [False, True])
def test_docker_argv_has_no_mounts_or_privileges(no_network):
    run = docker_run_argv("swebench/sweb.eval.x86_64.x:latest", "layover-reexec-1", Options(memory="8g", pids_limit=99, no_network=no_network))
    forbidden = ("-v", "--volume", "--mount", "--privileged", "--pid", "--ipc", "--cap-add", "--device", "--gpus", "--env-file", "--volumes-from")
    assert not any(arg in forbidden or arg.startswith(tuple(f"{f}=" for f in forbidden)) for arg in run)
    assert not any("docker.sock" in arg or ":/" in arg for arg in run)
    assert run[:4] == ["docker", "run", "-d", "--rm"]
    assert run[run.index("--memory") + 1] == "8g" and run[run.index("--pids-limit") + 1] == "99"
    assert ("--network" in run) == no_network and "host" not in run
    if no_network:
        assert run[run.index("--network") + 1] == "none"
    command = "rm -rf / --no-preserve-root"
    exec_argv = docker_exec_argv("c", ENVIRONMENT, command)
    assert exec_argv == ["docker", "exec", "-w", "/testbed", "-e", "PAGER=cat", "c", "bash", "-c", command]


def test_a_real_run_uses_those_arguments_and_forwards_no_host_environment(tmp_path, shim, monkeypatch):
    monkeypatch.setenv("SECRET_KEY", "hunter2")
    write_prepared(tmp_path / "prepared", ["a__a-1"])
    assert run_cli(tmp_path, "--memory", "2g", "--pids-limit", "50") == 0
    runs = shim.calls("run")
    assert len(runs) == 1
    assert runs[0][runs[0].index("--memory") + 1] == "2g" and runs[0][runs[0].index("--pids-limit") + 1] == "50"
    assert "--rm" in runs[0] and not {"-v", "--volume", "--mount", "--privileged"} & set(runs[0])
    assert not any("hunter2" in arg or "SECRET_KEY" in arg for call in shim.calls() for arg in call)
    execs = shim.calls("exec")
    assert len(execs) == 3 and all(c[1:5] == ["-w", "/testbed", "-e", "PAGER=cat"] for c in execs)
    assert [c[-1] for c in execs] == [behaves(0, "ok\n"), behaves(1, "no\n"), behaves(0, "yes\n")]  # In order.


def test_fidelity_counts_return_codes_bytes_and_similarity(tmp_path, shim):
    commands = {"a__a-1": [
        [(behaves(0, "same\n"), 0, "same\n")],
        [(behaves(0, "different\n"), 0, "another\n")],
        [(behaves(2, "x\n"), 2, "x at 0xdeadbeef\n")],
        [(behaves(0, "at 0x1 took 1.5s\n"), 0, "at 0xff took 2.0s\n")],
    ]}
    write_prepared(tmp_path / "prepared", ["a__a-1"], commands=commands)
    assert run_cli(tmp_path) == 0
    fidelity = log_of(tmp_path, "a__a-1")[-1]["fidelity"]
    assert fidelity["commands_compared"] == 4 and fidelity["rc_match_rate"] == 1.0
    assert fidelity["byte_exact_rate"] == 0.25
    assert fidelity["mean_similarity"] == pytest.approx((1 + 0 + 0 + 1) / 4)


def test_a_recorded_timeout_matches_a_replayed_timeout(tmp_path, shim):
    write_prepared(tmp_path / "prepared", ["a__a-1"], commands={"a__a-1": [[(behaves(0, "", sleep=30), -1, "")], [(behaves(), 0, "")]]})
    prepared = tmp_path / "prepared" / "a__a-1.json"
    raw = json.loads(prepared.read_text())
    raw["environment"]["timeout"] = 0.3
    prepared.write_text(json.dumps(raw))
    assert run_cli(tmp_path) == 0
    records = log_of(tmp_path, "a__a-1")
    assert records[1]["timed_out"] and records[1]["returncode"] == -1 and records[1]["rc_match"]
    assert 0.3 <= records[1]["duration_s"] < 5
    assert records[-1]["fidelity"]["timed_out_commands"] == 1


def test_the_final_patch_is_compared_with_the_submission(tmp_path, shim):
    patch = "diff --git a b\n+1\n"
    for name, out, expected in (("a__a-1", f"{SUBMIT_MARK}\n{patch}", True), ("b__b-2", f"{SUBMIT_MARK}\nother\n", False)):
        directory = tmp_path / name
        commands = {name: [[(behaves(0, "ok\n"), 0, "ok\n")], [(json.dumps({"rc": 0, "out": out, "mark": SUBMIT_MARK}), None, None)]]}
        write_prepared(directory / "prepared", [name], commands=commands)
        path = directory / "prepared" / f"{name}.json"
        raw = json.loads(path.read_text())
        raw["submission_sha256"] = hashlib.sha256(patch.encode()).hexdigest()
        path.write_text(json.dumps(raw))
        assert run_cli(directory) == (0 if expected else 1)
        assert log_of(directory, name)[-1]["fidelity"]["final_submission_match"] is expected
        assert log_of(directory, name)[-1]["fidelity"]["commands_compared"] == 1


def test_low_fidelity_is_replaced_from_the_reserve_in_order(tmp_path, shim):
    shim.write_config({"exec_override": {image_of("b__b-2"): {"rc": 7}}})
    write_prepared(tmp_path / "prepared", ["a__a-1", "b__b-2", "c__c-3"], ["r__r-1", "r__r-2"])
    assert run_cli(tmp_path) == 0
    state = run_state(tmp_path)
    assert state["valid"] == ["a__a-1", "c__c-3", "r__r-1"] and state["failed"] == ["b__b-2"]
    assert not (tmp_path / "results" / SUBMISSION / "r__r-2.jsonl").exists()
    logged = json.loads((tmp_path / "results" / SUBMISSION / "replacements.json").read_text())
    assert [(r["failed"], r["replaced_by"], r["status"]) for r in logged] == [("b__b-2", "r__r-1", "low_fidelity")]
    assert logged[0]["rc_match_rate"] < rx.RC_MATCH_THRESHOLD


def test_threshold_is_inclusive_at_ninety_percent(tmp_path, shim):
    good = [[(behaves(), 0, "")] for _ in range(9)]
    write_prepared(tmp_path / "prepared", ["a__a-1"], commands={"a__a-1": [*good, [(behaves(1), 0, "")]]})
    assert run_cli(tmp_path) == 0
    write_prepared(tmp_path / "p2" / "prepared", ["a__a-1"], commands={"a__a-1": [*good[:8], [(behaves(1), 0, "")]]})
    assert run_cli(tmp_path / "p2") == 1


def test_a_failed_image_is_retried_a_bounded_number_of_times_then_replaced(tmp_path, shim):
    shim.write_config({"pull_fail": {image_of("a__a-1"): "manifest unknown"}})
    write_prepared(tmp_path / "prepared", ["a__a-1"], ["r__r-1"])
    assert run_cli(tmp_path, "--pull-attempts", "2") == 0
    assert sum(c[-1] == image_of("a__a-1") for c in shim.calls("pull")) == 2
    assert log_of(tmp_path, "a__a-1")[-1]["status"] == "image_failed"
    assert run_state(tmp_path)["valid"] == ["r__r-1"]


def test_a_container_that_will_not_start_is_replaced(tmp_path, shim):
    shim.write_config({"run_fail": [image_of("a__a-1")], "run_fail_text": "exec /bin/sleep: exec format error"})
    write_prepared(tmp_path / "prepared", ["a__a-1"], ["r__r-1"])
    assert run_cli(tmp_path) == 0
    assert log_of(tmp_path, "a__a-1")[-1]["status"] == "container_failed"


def test_an_image_of_the_wrong_architecture_fails(tmp_path, shim):
    shim.write_config({"image_arch": "arm64"})
    write_prepared(tmp_path / "prepared", ["a__a-1"])
    assert run_cli(tmp_path) == 1
    assert "not amd64" in log_of(tmp_path, "a__a-1")[-1]["reason"]


def test_too_few_valid_tasks_is_loud_and_nonzero(tmp_path, shim, capsys):
    shim.write_config({"exec_override": {image_of("a__a-1"): {"rc": 9}, image_of("r__r-1"): {"rc": 9}}})
    write_prepared(tmp_path / "prepared", ["a__a-1"], ["r__r-1"])
    assert run_cli(tmp_path) == 1
    assert "only 0 valid tasks of 1" in capsys.readouterr().err
    with pytest.raises(SystemExit, match="only 0 valid"):
        convert(tmp_path / "prepared", tmp_path / "results", tmp_path / "t.jsonl")
    assert not (tmp_path / "t.jsonl").exists()


def test_a_rate_limit_stops_the_run_without_using_a_reserve_and_resumes(tmp_path, shim, capsys):
    shim.write_config({"pull_fail": {image_of("a__a-1"): "toomanyrequests: rate limit exceeded"}})
    write_prepared(tmp_path / "prepared", ["a__a-1"], ["r__r-1"])
    assert run_cli(tmp_path) == 3
    assert "same command resumes" in capsys.readouterr().err
    assert not (tmp_path / "results" / SUBMISSION / "r__r-1.jsonl").exists()
    assert len(shim.calls("pull")) == 1
    shim.write_config({})
    assert run_cli(tmp_path) == 0
    assert run_state(tmp_path)["valid"] == ["a__a-1"]


def test_resume_after_a_crash_repeats_only_the_unfinished_task(tmp_path, shim):
    write_prepared(tmp_path / "prepared", ["a__a-1", "b__b-2"])
    shim.write_config({"kill_parent_at_exec": 5})  # Second command of the second task.
    command = [sys.executable, "-m", "layover.replay.reexecute", "run", "--prepared", str(tmp_path / "prepared"),
               "--out", str(tmp_path / "results"), "--workers", "1", "--allow-emulated"]
    root = Path(__file__).parents[1]
    crashed = subprocess.run(command, cwd=root, capture_output=True, text=True)
    assert crashed.returncode == -9, crashed.stderr
    assert len(shim.state["containers"]) == 1  # The kill left it behind.
    results = tmp_path / "results" / SUBMISSION
    assert rx.ended_tasks(results).keys() == {"a__a-1"}

    shim.write_config({})
    executed = len(shim.calls("exec"))
    resumed = subprocess.run(command, cwd=root, capture_output=True, text=True)
    assert resumed.returncode == 0, resumed.stderr
    assert shim.state["containers"] == {}
    assert len(shim.calls("exec")) - executed == 3  # Only the second task, from its start.
    assert rx.ended_tasks(results).keys() == {"a__a-1", "b__b-2"}
    assert len(list(results.glob("b__b-2.crashed-*"))) == 1
    assert sum(r["type"] == "command" for r in log_of(tmp_path, "b__b-2")) == 3
    assert log_of(tmp_path, "b__b-2")[-1]["valid"]


def test_every_command_result_is_durable_when_it_finishes(tmp_path, shim, monkeypatch):
    synced = []
    real = os.fsync
    monkeypatch.setattr(rx.os, "fsync", lambda fd: (synced.append(fd), real(fd))[1])
    write_prepared(tmp_path / "prepared", ["a__a-1"])
    assert run_cli(tmp_path) == 0
    assert len(synced) == sum(1 for _ in log_of(tmp_path, "a__a-1"))


def test_the_container_is_removed_when_the_run_fails_partway(tmp_path, shim, monkeypatch):
    write_prepared(tmp_path / "prepared", ["a__a-1"])
    real = rx.TaskLog.append

    def fail_on_command(self, record):
        if record["type"] == "command":
            raise OSError("disk full")
        real(self, record)

    monkeypatch.setattr(rx.TaskLog, "append", fail_on_command)
    with pytest.raises(OSError, match="disk full"):
        run_cli(tmp_path)
    assert shim.state["containers"] == {} and len(shim.calls("rm")) == 1
    assert shim.state["images"] == []  # And the image this run pulled.


def test_images_go_after_each_task_unless_kept_and_others_are_left_alone(tmp_path, shim):
    write_prepared(tmp_path / "prepared", ["a__a-1", "b__b-2"])
    assert run_cli(tmp_path) == 0
    assert [c[-1] for c in shim.calls("rmi")] == [image_of("a__a-1"), image_of("b__b-2")]
    assert shim.state["images"] == []

    (tmp_path / "keep").mkdir()
    write_prepared(tmp_path / "keep" / "prepared", ["c__c-3"])
    assert run_cli(tmp_path / "keep", "--keep-images") == 0
    assert shim.state["images"] == [image_of("c__c-3")]

    # An image that was here before the run is not this run's to remove.
    (tmp_path / "again").mkdir()
    write_prepared(tmp_path / "again" / "prepared", ["c__c-3"])
    pulls = len(shim.calls("pull"))
    assert run_cli(tmp_path / "again") == 0
    assert len(shim.calls("pull")) == pulls and shim.state["images"] == [image_of("c__c-3")]


def test_pull_time_is_kept_apart_from_tool_time(tmp_path, shim):
    write_prepared(tmp_path / "prepared", ["a__a-1"])
    assert run_cli(tmp_path) == 0
    records = log_of(tmp_path, "a__a-1")
    commands = [r for r in records if r["type"] == "command"]
    assert records[-1]["pull_s"] > 0 and records[-1]["pulled"]
    assert records[-1]["tool_s_total"] == pytest.approx(sum(c["duration_s"] for c in commands))


@pytest.mark.parametrize("machine, arch", [("arm64", "arm64"), ("x86_64", "arm64"), ("arm64", "amd64")])
def test_refuses_anything_but_native_x86(tmp_path, shim, monkeypatch, machine, arch):
    monkeypatch.setattr(platform, "machine", lambda: machine)
    shim.write_config({"server_arch": arch})
    write_prepared(tmp_path / "prepared", ["a__a-1"])
    with pytest.raises(SystemExit, match="refusing to measure"):
        rx.main(["run", "--prepared", str(tmp_path / "prepared"), "--out", str(tmp_path / "results")])
    assert shim.calls("pull") == [] and shim.calls("run") == []


def test_no_docker_daemon_names_the_rental_problem(tmp_path, shim):
    shim.write_config({"daemon_down": True})
    write_prepared(tmp_path / "prepared", ["a__a-1"])
    with pytest.raises(SystemExit, match="themselves containers"):
        rx.main(["run", "--prepared", str(tmp_path / "prepared"), "--out", str(tmp_path / "results")])


def required_tasks():
    return [f"t__t-{i}" for i in range(rx.REQUIRED_TASKS)]


@pytest.fixture
def short_requirement(monkeypatch):
    """Three tasks stand in for the required thirty, to keep the shim's process count down."""
    assert rx.REQUIRED_TASKS == 30
    monkeypatch.setattr(rx, "REQUIRED_TASKS", 3)


def test_emulated_results_can_only_be_labelled_a_pipeline_check(tmp_path, shim, monkeypatch, short_requirement):
    monkeypatch.setattr(platform, "machine", lambda: "arm64")
    shim.write_config({"server_arch": "arm64"})
    write_prepared(tmp_path / "prepared", required_tasks())
    assert run_cli(tmp_path, "--allow-emulated") == 0
    jobs = convert(tmp_path / "prepared", tmp_path / "results", tmp_path / "t.jsonl")
    assert len(jobs) == rx.REQUIRED_TASKS
    assert all(j.source.startswith("pipeline-check:swebench-experiments/sub_mini:reexecuted:") for j in jobs)
    assert all(j.source.endswith(":emulated") and j.meta["emulated"] for j in jobs)


def test_a_native_run_converts_to_the_recorded_source_with_measured_tool_time(tmp_path, shim, short_requirement):
    write_prepared(tmp_path / "prepared", required_tasks())
    assert run_cli(tmp_path) == 0
    out = tmp_path / "out" / "swebench_30.jsonl"
    convert(tmp_path / "prepared", tmp_path / "results", out)
    jobs = load_jsonl(out)  # Which also validates each job.
    assert len(jobs) == rx.REQUIRED_TASKS
    machine = run_state(tmp_path)["machine"]
    assert {j.source for j in jobs} == {f"recorded:swebench-experiments/{SUBMISSION}:reexecuted:{machine_label(machine)}"}
    job = jobs[0]
    assert len(job.sessions) == 1 and "no forks" in job.meta["sessions"]
    durations = [r["duration_s"] for r in log_of(tmp_path, job.job_id) if r["type"] == "command"]
    first, second = job.main.turns
    assert first.tool_s == pytest.approx(durations[0]) and second.tool_s == pytest.approx(durations[1] + durations[2])
    assert first.tool_s != 2.5  # The recorded gap stays in meta.
    assert job.meta["recorded_gap_s"] == [2.5, 2.5]
    assert job.shared_prefix_tokens == 100 and first.new_tokens == 200 and second.new_tokens == 20 and first.output_tokens == 10
    assert first.provider_new_tokens == 100 and second.provider_new_tokens == 200 - 108 and second.provider_output_tokens == 8
    assert job.meta["submission"] == SUBMISSION and job.meta["machine"]["machine"] == "x86_64"
    assert job.meta["workers"] == 1 and job.meta["no_network"] is False and job.meta["fidelity"]["rc_match_rate"] == 1.0
    assert job.meta["image_digests"] == [f"{image_of(job.job_id)}@sha256:abc"]
    # Counts only: no command text or output reaches the trace.
    assert "ok\\n" not in out.read_text() and "bash" not in out.read_text()


def test_a_subset_or_named_manifest_is_never_labelled_recorded(tmp_path, shim, short_requirement):
    write_prepared(tmp_path / "prepared", ["a__a-1"])
    assert run_cli(tmp_path) == 0
    jobs = convert(tmp_path / "prepared", tmp_path / "results", tmp_path / "t.jsonl")
    assert jobs[0].source.startswith("pipeline-check:") and ":subset-of-1" in jobs[0].source

    named = tmp_path / "named"
    write_prepared(named / "prepared", required_tasks(), selection="named")
    assert run_cli(named) == 0
    assert convert(named / "prepared", named / "results", named / "t.jsonl")[0].source.startswith("pipeline-check:")


def test_no_network_is_applied_and_recorded(tmp_path, shim):
    write_prepared(tmp_path / "prepared", ["a__a-1"])
    assert run_cli(tmp_path, "--no-network") == 0
    run = shim.calls("run")[0]
    assert run[run.index("--network") + 1] == "none"
    assert log_of(tmp_path, "a__a-1")[0]["no_network"] is True
    assert run_state(tmp_path)["no_network"] is True


def test_resume_refuses_different_settings(tmp_path, shim):
    write_prepared(tmp_path / "prepared", ["a__a-1"])
    assert run_cli(tmp_path) == 0
    assert "different settings" in str(run_cli(tmp_path, "--no-network"))


def test_convert_refuses_tasks_measured_on_different_machines(tmp_path, shim):
    write_prepared(tmp_path / "prepared", ["a__a-1", "b__b-2"])
    assert run_cli(tmp_path) == 0
    path = tmp_path / "results" / SUBMISSION / "b__b-2.jsonl"
    lines = path.read_text().splitlines()
    start = json.loads(lines[0])
    start["machine_label"] = "elsewhere"
    path.write_text("\n".join([json.dumps(start), *lines[1:]]) + "\n")
    with pytest.raises(SystemExit, match="different settings"):
        convert(tmp_path / "prepared", tmp_path / "results", tmp_path / "t.jsonl")


def test_workers_default_and_concurrent_runs_finish_every_task(tmp_path, shim, capsys, monkeypatch):
    for count in (2, 8, 16, 64):
        monkeypatch.setattr(os, "cpu_count", lambda count=count: count)
        assert rx.default_workers() == 1
    with pytest.raises(SystemExit):
        rx.main(["run", "--help"])
    usage = " ".join(capsys.readouterr().out.split())
    assert "benchmark" in usage and "pulls and removals" in usage and "overlap" in usage
    shim.write_config({"exec_override": {image_of("b__b-2"): {"rc": 7}}})
    write_prepared(tmp_path / "prepared", ["a__a-1", "b__b-2", "c__c-3", "d__d-4"], ["r__r-1"])
    with pytest.raises(SystemExit) as raised:
        rx.main(["run", "--prepared", str(tmp_path / "prepared"), "--out", str(tmp_path / "results"), "--workers", "3"])
    assert raised.value.code == 0
    state = run_state(tmp_path)
    assert state["valid"] == ["a__a-1", "c__c-3", "d__d-4", "r__r-1"] and state["workers"] == 3
    assert log_of(tmp_path, "a__a-1")[0]["workers"] == 3 and shim.state["containers"] == {}


def test_machine_record_and_label(monkeypatch):
    files = {
        "/proc/cpuinfo": "processor : 0\nmodel name\t: AMD EPYC 7R13\n",
        "/proc/meminfo": "MemTotal:       1024 kB\n",
        "/sys/class/dmi/id/sys_vendor": "Amazon EC2",
        "/sys/class/dmi/id/product_name": "g6.4xlarge",
    }
    monkeypatch.setattr(rx, "_read", files.get)
    monkeypatch.setattr(rx, "_imds", lambda: ("g6.4xlarge", False))
    answers = {"nvidia-smi": "NVIDIA L4", "docker": "29.0.0"}
    monkeypatch.setattr(rx, "_text", lambda argv, timeout=30.0: answers.get(argv[0]))
    record = rx.machine_record()
    assert record["cpu_model"] == "AMD EPYC 7R13" and record["memory_bytes"] == 1024 * 1024
    assert record["dmi_sys_vendor"] == "Amazon EC2" and record["gpu"] == "NVIDIA L4" and record["docker_server_version"] == "29.0.0"
    assert record["cpu_count"] == os.cpu_count() and record["kernel"] == platform.release()
    assert machine_label(record) == "g6.4xlarge"
    assert machine_label({**record, "aws_instance_type": "unknown", "dmi_sys_vendor": "ACME Inc", "dmi_product_name": "Box 1"}) == "ACME-Inc-Box-1"
    assert machine_label({**record, "aws_instance_type": "unknown", "dmi_product_name": "unknown", "machine": "arm64", "cpu_model": "Apple M3"}) == "arm64-Apple-M3"


def test_imds_failure_is_labelled_unknown(monkeypatch):
    monkeypatch.setattr(rx, "IMDS", "http://127.0.0.1:9/latest")
    assert rx._imds() == ("unknown", None)


def test_execute_bounds_output_but_hashes_all_of_it(monkeypatch):
    monkeypatch.setattr(rx, "KEEP_BYTES", 10)
    result = execute([sys.executable, "-S", "-c", "print('x' * 99, end='')"], 30)
    assert result.returncode == 0 and result.truncated and len(result.text) == 10 and result.output_bytes == 99
    assert result.output_sha256 == hashlib.sha256(b"x" * 99).hexdigest()


def test_execute_merges_stderr_and_reports_the_return_code():
    code = "import sys; print('out', flush=True); print('err', file=sys.stderr, flush=True); sys.exit(4)"
    result = execute([sys.executable, "-S", "-c", code], 30)
    assert result.returncode == 4 and result.text == "out\nerr\n" and not result.timed_out


def test_execute_kills_the_client_at_the_timeout():
    result = execute([sys.executable, "-S", "-c", "import time; print('partial', flush=True); time.sleep(30)"], 0.5)
    assert result.timed_out and result.returncode == -1 and result.text == "partial\n" and 0.4 < result.duration_s < 10


def test_similarity_masks_volatile_text():
    assert similarity("at 0xabc in 1.25s\n", "at 0xdef in 9.00s\n") == 1.0
    assert similarity("a\nb\n", "a\nc\n") == 0.5
    assert similarity("", "") == 1.0


def test_stale_cleanup_touches_only_this_tools_containers(tmp_path, shim):
    state = {"images": [], "containers": {"layover-reexec-old": "x"}}
    (shim.home / "state.json").write_text(json.dumps(state))
    assert rx.remove_stale_containers() == ["layover-reexec-old"]
    assert all(c[-1].startswith(rx.CONTAINER_PREFIX) for c in shim.calls("rm"))
    filters = shim.calls("ps")[0]
    assert filters[filters.index("--filter") + 1] == f"name=^{rx.CONTAINER_PREFIX}"


def test_read_log_tolerates_a_cut_last_line(tmp_path):
    path = tmp_path / "a.jsonl"
    path.write_text('{"type": "start"}\n{"type": "comm')
    assert rx.read_log(path) == [{"type": "start"}]


UNRELATED_OUTPUT = [[(behaves(0, ""), 0, "recorded line\n")] for _ in range(4)]


def test_matching_return_codes_with_unrelated_output_do_not_count(tmp_path, shim):
    write_prepared(tmp_path / "prepared", ["a__a-1"], ["r__r-1"], commands={"a__a-1": UNRELATED_OUTPUT})
    assert run_cli(tmp_path) == 0
    end = log_of(tmp_path, "a__a-1")[-1]
    assert end["fidelity"]["rc_match_rate"] == 1.0 and end["fidelity"]["mean_similarity"] == 0.0
    assert not end["valid"] and end["status"] == "low_fidelity" and "similarity" in end["reason"]
    assert run_state(tmp_path)["valid"] == ["r__r-1"]


def test_a_final_patch_that_differs_is_replaced_and_says_why(tmp_path, shim):
    commands = {"a__a-1": [[(behaves(0, "ok\n"), 0, "ok\n")], [(json.dumps({"out": f"{SUBMIT_MARK}\nother\n", "mark": SUBMIT_MARK}), None, None)]]}
    write_prepared(tmp_path / "prepared", ["a__a-1"], ["r__r-1"], commands=commands)
    path = tmp_path / "prepared" / "a__a-1.json"
    path.write_text(json.dumps({**json.loads(path.read_text()), "submission_sha256": "not this"}))
    assert run_cli(tmp_path) == 0
    assert "final patch" in log_of(tmp_path, "a__a-1")[-1]["reason"]
    assert run_state(tmp_path)["valid"] == ["r__r-1"]


def test_a_floor_corrected_after_the_run_applies_without_executing_again(tmp_path, shim, monkeypatch):
    write_prepared(tmp_path / "prepared", ["a__a-1"], commands={"a__a-1": UNRELATED_OUTPUT})
    assert run_cli(tmp_path) == 1
    with pytest.raises(SystemExit, match="only 0 valid"):
        convert(tmp_path / "prepared", tmp_path / "results", tmp_path / "t.jsonl")
    monkeypatch.setattr(rx, "SIMILARITY_FLOOR", 0.0)
    assert [j.job_id for j in convert(tmp_path / "prepared", tmp_path / "results", tmp_path / "t.jsonl")] == ["a__a-1"]


def test_a_recorded_timeout_that_now_finishes_is_counted_apart_not_as_a_mismatch(tmp_path, shim):
    # Two of five would be 60% under the old rule, well below the threshold.
    commands = [*[[(behaves(0, "partial\n"), -1, "")] for _ in range(2)], *[[(behaves(0, "ok\n"), 0, "ok\n")] for _ in range(3)]]
    write_prepared(tmp_path / "prepared", ["a__a-1"], commands={"a__a-1": commands})
    assert run_cli(tmp_path) == 0
    fidelity = log_of(tmp_path, "a__a-1")[-1]["fidelity"]
    assert fidelity["commands_compared"] == 3 and fidelity["rc_match_rate"] == 1.0 and fidelity["mean_similarity"] == 1.0
    assert fidelity["recorded_timeouts"] == 2 and fidelity["recorded_timeouts_completed"] == 2


def test_a_recorded_timeout_that_times_out_again_is_not_counted_as_completed(tmp_path, shim):
    write_prepared(tmp_path / "prepared", ["a__a-1"], commands={"a__a-1": [[(behaves(0, "", sleep=30), -1, "")], [(behaves(), 0, "")]]})
    prepared = tmp_path / "prepared" / "a__a-1.json"
    prepared.write_text(json.dumps({**json.loads(prepared.read_text()), "environment": {**ENVIRONMENT, "timeout": 0.3}}))
    assert run_cli(tmp_path) == 0
    fidelity = log_of(tmp_path, "a__a-1")[-1]["fidelity"]
    assert fidelity["recorded_timeouts"] == 1 and fidelity["recorded_timeouts_completed"] == 0 and fidelity["commands_compared"] == 1


def test_only_recorded_timeouts_leaves_nothing_compared_and_fails(tmp_path, shim):
    write_prepared(tmp_path / "prepared", ["a__a-1"], commands={"a__a-1": [[(behaves(0, ""), -1, "")]]})
    assert run_cli(tmp_path) == 1
    assert "no commands to compare" in log_of(tmp_path, "a__a-1")[-1]["reason"]


def test_piped_commands_are_counted_because_their_return_code_shows_only_the_last_stage(tmp_path, shim):
    piped = json.dumps({"rc": 0, "cmd": "pip install x 2>&1 | tail -3"})
    either = json.dumps({"rc": 0, "cmd": "a || b"})
    write_prepared(tmp_path / "prepared", ["a__a-1"], commands={"a__a-1": [[(piped, 0, "")], [(either, 0, "")]]})
    assert run_cli(tmp_path) == 0
    assert log_of(tmp_path, "a__a-1")[-1]["fidelity"]["piped_rc_blind_share"] == 0.5


def converted(tmp_path, shim, **prepared_args):
    write_prepared(tmp_path / "prepared", **prepared_args)
    assert run_cli(tmp_path) == 0
    return convert(tmp_path / "prepared", tmp_path / "results", tmp_path / "t.jsonl")


def test_the_replacement_note_gives_recorded_tool_time_before_and_after(tmp_path, shim):
    shim.write_config({"exec_override": {image_of("a__a-1"): {"rc": 7}}})
    write_prepared(tmp_path / "prepared", ["a__a-1", "b__b-2"], ["r__r-1"])
    path = tmp_path / "prepared" / "a__a-1.json"
    raw = json.loads(path.read_text())
    raw["turns"][0]["recorded_gap_s"] = 491.0
    path.write_text(json.dumps(raw))
    assert run_cli(tmp_path) == 0
    jobs = convert(tmp_path / "prepared", tmp_path / "results", tmp_path / "t.jsonl")
    assert "Recorded tool time 498 s for the 2 tasks selected, 10 s for the set measured" in jobs[0].meta["replacements"]


def test_no_replacement_says_so(tmp_path, shim):
    jobs = converted(tmp_path, shim, tasks=["a__a-1"])
    assert jobs[0].meta["replacements"] == "none; all 1 selected tasks passed"


@pytest.mark.parametrize("command, expected", [
    ("pip install -e .", True), ("cd /testbed && pip3 install numpy 2>&1 | tail -5", True), ("python -m pip install --quiet x", True),
    ("pip -q install x", True), ("conda install -y x", True), ("apt-get update", True), ("curl -sL https://x", True), ("git clone u", True),
    ("python setup.py develop", True),
    ("grep -n pip setup.py", False), ("cat requirements.txt", False), ("pip list", False), ("python -m pytest tests", False), ("git diff", False),
])
def test_install_commands_are_recognised(command, expected):
    assert bool(rx.INSTALL_COMMAND.search(command)) is expected


def test_a_no_network_run_is_never_labelled_recorded(tmp_path, shim, short_requirement):
    write_prepared(tmp_path / "prepared", required_tasks())
    assert run_cli(tmp_path, "--no-network") == 0
    jobs = convert(tmp_path / "prepared", tmp_path / "results", tmp_path / "t.jsonl")
    assert all(j.source.startswith("pipeline-check:") and j.source.endswith(":no-network") for j in jobs)
    assert "network off" in jobs[0].meta["tool_s_basis"] and "network off" in jobs[0].meta["run_settings"]


def test_resume_refuses_a_different_worker_count(tmp_path, shim):
    write_prepared(tmp_path / "prepared", ["a__a-1"])
    assert run_cli(tmp_path) == 0
    with pytest.raises(SystemExit, match="different settings"):
        rx.main(["run", "--prepared", str(tmp_path / "prepared"), "--out", str(tmp_path / "results"), "--workers", "2"])


def test_convert_refuses_tasks_run_with_different_worker_counts(tmp_path, shim):
    write_prepared(tmp_path / "prepared", ["a__a-1", "b__b-2"])
    assert run_cli(tmp_path) == 0
    path = tmp_path / "results" / SUBMISSION / "b__b-2.jsonl"
    lines = path.read_text().splitlines()
    path.write_text("\n".join([json.dumps({**json.loads(lines[0]), "workers": 4}), *lines[1:]]) + "\n")
    with pytest.raises(SystemExit, match="different settings"):
        convert(tmp_path / "prepared", tmp_path / "results", tmp_path / "t.jsonl")


# As daemons word them, from real failed pulls: Docker 29 with the containerd image store.
IMAGE_ANSWERS = [
    "Error response from daemon: pull access denied for swebench/x, repository does not exist or may require 'docker login'",
    'Error response from daemon: failed to resolve reference "docker.io/library/x:tag": docker.io/library/x:tag: not found',
    "Error response from daemon: no matching manifest for linux/sparc64 in the manifest list entries: no match for platform in manifest: not found",
    "Error response from daemon: manifest for swebench/x:latest not found: manifest unknown: manifest unknown",
]
OUTAGES = [
    "Error response from daemon: Get https://registry-1.docker.io/v2/: dial tcp: lookup registry-1.docker.io: i/o timeout",
    "Error response from daemon: Get https://registry-1.docker.io/v2/: net/http: TLS handshake timeout",
    "read tcp 10.0.0.1:443: read: connection reset by peer",
    'error getting credentials - err: exec: "docker-credential-gone": executable file not found in $PATH',
    "Error response from daemon: something nobody has seen before",
]


@pytest.fixture
def no_backoff(monkeypatch):
    monkeypatch.setattr(rx, "TRANSIENT_BACKOFF_S", 0.0)
    monkeypatch.setattr(rx, "IMAGE_RETRY_BACKOFF_S", 0.0)


@pytest.mark.parametrize("answer", IMAGE_ANSWERS)
def test_a_registry_answer_about_the_image_uses_a_reserve(tmp_path, shim, no_backoff, answer):
    shim.write_config({"pull_fail": {image_of("a__a-1"): answer}})
    write_prepared(tmp_path / "prepared", ["a__a-1"], ["r__r-1"])
    assert run_cli(tmp_path) == 0
    assert log_of(tmp_path, "a__a-1")[-1]["status"] == "image_failed"
    assert run_state(tmp_path)["valid"] == ["r__r-1"]


@pytest.mark.parametrize("outage", OUTAGES)
def test_an_outage_aborts_without_a_final_record_and_the_same_command_resumes(tmp_path, shim, no_backoff, capsys, outage):
    shim.write_config({"pull_fail": {image_of("a__a-1"): outage}})
    write_prepared(tmp_path / "prepared", ["a__a-1"], ["r__r-1"])
    assert run_cli(tmp_path, "--pull-attempts", "3") == 3
    assert "same command resumes" in capsys.readouterr().err
    results = tmp_path / "results" / SUBMISSION
    assert rx.ended_tasks(results) == {}
    assert not (results / "r__r-1.jsonl").exists()
    assert sum(c[-1] == image_of("a__a-1") for c in shim.calls("pull")) == 3
    shim.write_config({})
    assert run_cli(tmp_path, "--pull-attempts", "3") == 0
    assert run_state(tmp_path)["valid"] == ["a__a-1"]


class Waits:
    """Stands in for the stop event, and remembers how long each backoff was."""

    def __init__(self, stopped=False):
        self.stopped = stopped
        self.seconds = []

    def wait(self, seconds):
        self.seconds.append(seconds)
        return self.stopped


class NoLog:
    def append(self, record):
        pass


def failing_pull(monkeypatch, message):
    monkeypatch.setattr(rx, "_text", lambda argv, timeout=30.0: None)
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(a, 1, "", message))


def test_an_outage_is_waited_out_far_longer_than_a_missing_image(monkeypatch):
    failing_pull(monkeypatch, OUTAGES[0])
    waits = Waits()
    with pytest.raises(rx.RunAborted, match="none because of the image"):
        rx._pull("i", Options(), NoLog(), waits)
    assert waits.seconds == [30.0, 60.0, 120.0]

    failing_pull(monkeypatch, IMAGE_ANSWERS[0])
    waits = Waits()
    ok, _, _, error = rx._pull("i", Options(), NoLog(), waits)
    assert not ok and "pull access denied" in error and waits.seconds == [1.0, 2.0, 4.0]


def test_stopping_cuts_a_backoff_short(monkeypatch):
    failing_pull(monkeypatch, OUTAGES[0])
    waits = Waits(stopped=True)
    with pytest.raises(rx.RunAborted, match="stopping"):
        rx._pull("i", Options(), NoLog(), waits)
    assert waits.seconds == [30.0]


def test_a_pull_that_keeps_timing_out_aborts(monkeypatch):
    monkeypatch.setattr(rx, "_text", lambda argv, timeout=30.0: None)

    def hang(argv, **kwargs):
        raise subprocess.TimeoutExpired(argv, kwargs["timeout"])

    monkeypatch.setattr(subprocess, "run", hang)
    with pytest.raises(rx.RunAborted, match="timed out"):
        rx._pull("i", Options(pull_attempts=2), NoLog(), Waits())


@pytest.mark.parametrize("message", [
    "docker: Error response from daemon: failed to create task: no space left on device",
    "docker: Error response from daemon: cannot start",
    "docker: Cannot connect to the Docker daemon at unix:///var/run/docker.sock",
])
def test_a_container_start_failing_for_the_daemons_reasons_aborts_and_resumes(tmp_path, shim, message):
    shim.write_config({"run_fail": [image_of("a__a-1")], "run_fail_text": message})
    write_prepared(tmp_path / "prepared", ["a__a-1"], ["r__r-1"])
    assert run_cli(tmp_path) == 3
    results = tmp_path / "results" / SUBMISSION
    assert rx.ended_tasks(results) == {} and not (results / "r__r-1.jsonl").exists()
    shim.write_config({})
    assert run_cli(tmp_path) == 0
    assert run_state(tmp_path)["valid"] == ["a__a-1"]


def test_a_container_start_that_hangs_aborts(tmp_path, monkeypatch):
    monkeypatch.setattr(rx, "_remove_container", lambda name: None)

    def hang(argv, **kwargs):
        raise subprocess.TimeoutExpired(argv, kwargs["timeout"])

    monkeypatch.setattr(subprocess, "run", hang)
    with pytest.raises(rx.RunAborted, match="no answer in 300"):
        rx._commands(document("a__a-1"), Options(), NoLog(), {}, Waits())


def test_failing_to_inspect_an_image_is_not_the_images_fault(tmp_path, shim):
    shim.write_config({"inspect_fail": True})
    write_prepared(tmp_path / "prepared", ["a__a-1"], ["r__r-1"])
    assert run_cli(tmp_path) == 3
    assert rx.ended_tasks(tmp_path / "results" / SUBMISSION) == {}
    shim.write_config({})
    assert run_cli(tmp_path) == 0


@pytest.mark.parametrize("instance_type, attached", [("g6.xlarge", True), ("g6.xlarge", None)])
def test_an_iam_role_is_refused_unless_explicitly_allowed(tmp_path, shim, monkeypatch, capsys, instance_type, attached):
    monkeypatch.setattr(rx, "_imds", lambda: (instance_type, attached))
    write_prepared(tmp_path / "prepared", ["a__a-1"])
    refusal = str(run_cli(tmp_path))
    assert refusal.startswith("refusing to run") and "--allow-iam-role" in refusal
    assert "HttpTokens=required" in refusal and "a hop limit alone leaves IMDSv1 open" in refusal
    assert shim.calls("pull") == [] and shim.calls("run") == [] and shim.calls("exec") == []
    assert run_cli(tmp_path, "--allow-iam-role") == 0
    assert "WARNING" in capsys.readouterr().err
    assert log_of(tmp_path, "a__a-1")[0]["machine"]["aws_iam_role_attached"] is attached


@pytest.mark.parametrize("instance_type, attached", [("g6.xlarge", False), ("unknown", None)])
def test_no_role_or_no_cloud_is_not_refused(tmp_path, shim, monkeypatch, instance_type, attached):
    monkeypatch.setattr(rx, "_imds", lambda: (instance_type, attached))
    write_prepared(tmp_path / "prepared", ["a__a-1"])
    assert run_cli(tmp_path) == 0


class FakeOpener:
    def __init__(self, role):
        self.role = role

    def open(self, request, timeout):
        url = request.full_url
        if url.endswith("security-credentials/"):
            if isinstance(self.role, Exception):
                raise self.role
            return io.BytesIO(self.role)
        return io.BytesIO(b"g6.xlarge")


@pytest.mark.parametrize("role, expected", [
    (b"my-role\n", True), (b"", False),
    (urllib.error.HTTPError("u", 404, "Not Found", {}, None), False),
    (urllib.error.HTTPError("u", 500, "Oops", {}, None), None),
    (urllib.error.URLError("timed out"), None),
])
def test_only_a_404_says_there_is_no_iam_role(monkeypatch, role, expected):
    monkeypatch.setattr(rx.urllib.request, "build_opener", lambda *a: FakeOpener(role))
    assert rx._imds() == ("g6.xlarge", expected)
