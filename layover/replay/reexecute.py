"""Measure real tool durations by re-running trajectories' commands in their containers.

The commands are third-party text from a public leaderboard, so the container is the
safety boundary: no mounts, no docker socket, not privileged, bounded memory and pids, and
only the recording's cwd, environment and interpreter pass through. The recording's
run_args and forward_env are never used. That boundary has limits: unless --no-network is
given, containers sit on Docker's default network, so they run as root with default
capabilities and can reach the host's bridge gateway and the cloud metadata service. Hence
the refusal to run on an instance with an IAM role.

`run` executes on the x86 machine and writes one JSONL log per task, fsynced per command.
A task with an end record is done; one without restarts from scratch, because its
container's state is gone. A task whose image is unusable, by the registry's own answer, or
whose replay fails fidelity_failure, is replaced by the next reserve in manifest order. Any
other failure of Docker or the network stops the run with no end record, so the same command
resumes and no reserve is spent. Which task replaces which is derived from the logs, so a
resumed run reaches the same list.

`convert` writes the canonical trace and puts what a reader must know into every job's meta
as strings: replacements, fidelity, exclusions, install share, workers and network. These
are single-session jobs: mini-swe-agent does not fork sub-agents, so no job here has a child
session.

Execution mirrors mini-swe-agent's DockerEnvironment.execute: one `docker exec` per
command, stdout and stderr merged, and on timeout only the client is killed, so the command
may keep running inside the container exactly as it did when recorded.
"""

from __future__ import annotations

import argparse
import concurrent.futures as futures
import difflib
import hashlib
import json
import os
import platform
import re
import signal
import statistics
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass
from pathlib import Path

from layover.replay.trace import TraceJob, TraceSession, TraceTurn, save_jsonl, tool_category
from layover.replay.trajectories import SUBMIT_MARK, load_manifest, load_prepared

# Stated before any run: below these a replay has diverged from what was recorded.
RC_MATCH_THRESHOLD = 0.9
# Loose on purpose. Output legitimately varies between runs, but a replay sharing under half
# its lines did not do what was recorded, whatever its return codes say.
SIMILARITY_FLOOR = 0.5
# How the recording writes a command that hit its timeout.
RECORDED_TIMEOUT_RC = -1
REQUIRED_TASKS = 30
CONTAINER_PREFIX = "layover-reexec-"
LOG_SCHEMA = "layover-reexec-1"
# Long enough for any task's commands at their recorded timeout; the container is removed anyway.
CONTAINER_LIFETIME = "6h"
PLATFORM = "linux/amd64"
# Kept per command for comparison. The rest of a larger output is hashed and discarded.
KEEP_BYTES = 8 * 1024 * 1024
SIMILARITY_LINES = 5000
PULL_TIMEOUT_S = 1800
IMDS = "http://169.254.169.254/latest"
# Waiting will not fix these within a session, so stop at once.
FATAL_PULL_TEXT = ("toomanyrequests", "rate limit", "no space left", "cannot connect to the docker daemon")
# The registry's answer about the image: the only pull failure that may use up a reserve. The
# wording differs by image store, hence ": not found" beside "manifest unknown". A bare "not
# found" would also catch a missing credential helper's "executable file not found".
IMAGE_ANSWER_TEXT = ("manifest unknown", ": not found", "no matching manifest", "pull access denied", "repository does not exist")
# Anything else (DNS, TLS, resets, a timeout) says nothing about the image: wait out a short
# outage, then abort so that a resume can retry.
TRANSIENT_BACKOFF_S = 30.0
IMAGE_RETRY_BACKOFF_S = 1.0
# The image cannot run at all. Any other `docker run` failure is the daemon's or the disk's.
IMAGE_START_TEXT = ("exec format error", "executable file not found")
# A pipeline's return code is its last stage's, so a failed `pip install | tail` still matches.
PIPE = re.compile(r"(?<!\|)\|(?!\|)")
# Commands whose duration depends on the rental's network and the state of a package index.
INSTALL_COMMAND = re.compile(
    r"\b(?:pip3?|conda|mamba|apt-get|apt|yum|npm)\s+(?:-\S+\s+)*(?:install|create|update|upgrade)\b"
    r"|\bsetup\.py\s+(?:install|develop)\b|\bgit\s+(?:clone|fetch|pull)\b|\b(?:curl|wget)\s"
)


class RunAborted(Exception):
    """Stop now; the logs are intact and the same command resumes."""


@dataclass(frozen=True)
class Options:
    memory: str = "16g"
    pids_limit: int = 4096
    no_network: bool = False
    keep_images: bool = False
    pull_attempts: int = 4
    emulated: bool = False
    workers: int = 1
    machine_label: str = "unknown"


def docker_run_argv(image: str, name: str, options: Options) -> list[str]:
    argv = [
        "docker", "run", "-d", "--rm", "--name", name, "--platform", PLATFORM, "--memory", options.memory,
        "--pids-limit", str(options.pids_limit), "--security-opt", "no-new-privileges",
    ]
    if options.no_network:
        argv += ["--network", "none"]
    return [*argv, image, "sleep", CONTAINER_LIFETIME]


def docker_exec_argv(container: str, environment: dict, command: str) -> list[str]:
    argv = ["docker", "exec", "-w", environment["cwd"]]
    for key, value in (environment.get("env") or {}).items():
        argv += ["-e", f"{key}={value}"]
    return [*argv, container, *environment["interpreter"], command]


def _text(argv: list[str], timeout: float = 30.0) -> str | None:
    try:
        result = subprocess.run(argv, capture_output=True, text=True, errors="replace", timeout=timeout)
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def _read(path: str) -> str | None:
    try:
        return Path(path).read_text(errors="replace").strip() or None
    except OSError:
        return None


def _cpu_model() -> str:
    for line in (_read("/proc/cpuinfo") or "").splitlines():
        if line.startswith("model name"):
            return line.split(":", 1)[1].strip()
    return _text(["sysctl", "-n", "machdep.cpu.brand_string"]) or "unknown"


def _memory_bytes() -> int | None:
    for line in (_read("/proc/meminfo") or "").splitlines():
        if line.startswith("MemTotal:"):
            return int(line.split()[1]) * 1024
    value = _text(["sysctl", "-n", "hw.memsize"])
    return int(value) if value and value.isdigit() else None


def _imds() -> tuple[str, bool | None]:
    """The AWS instance type through IMDSv2, and whether an IAM role is attached."""
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    token_request = urllib.request.Request(
        f"{IMDS}/api/token", method="PUT", headers={"X-aws-ec2-metadata-token-ttl-seconds": "60"}
    )
    try:
        token = opener.open(token_request, timeout=1.0).read().decode()
        headers = {"X-aws-ec2-metadata-token": token}
        request = urllib.request.Request(f"{IMDS}/meta-data/instance-type", headers=headers)
        instance_type = opener.open(request, timeout=1.0).read().decode().strip()
    except (OSError, ValueError):
        return "unknown", None
    try:
        request = urllib.request.Request(f"{IMDS}/meta-data/iam/security-credentials/", headers=headers)
        return instance_type, bool(opener.open(request, timeout=1.0).read().strip())
    except urllib.error.HTTPError as error:
        return instance_type, False if error.code == 404 else None
    except (OSError, ValueError):
        # Only a 404 says there is no role; a lost answer says nothing.
        return instance_type, None


def machine_record() -> dict:
    """Facts about this machine that travel with every measurement."""
    instance_type, iam_role = _imds()
    return {
        "machine": platform.machine(),
        "system": platform.system(),
        "kernel": platform.release(),
        "cpu_model": _cpu_model(),
        "cpu_count": os.cpu_count(),
        "memory_bytes": _memory_bytes(),
        "dmi_sys_vendor": _read("/sys/class/dmi/id/sys_vendor") or "unknown",
        "dmi_product_name": _read("/sys/class/dmi/id/product_name") or "unknown",
        "aws_instance_type": instance_type,
        "aws_iam_role_attached": iam_role,
        "gpu": _text(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"]) or "none",
        "docker_server_version": _text(["docker", "version", "--format", "{{.Server.Version}}"]) or "unknown",
        "docker_server_arch": _text(["docker", "version", "--format", "{{.Server.Arch}}"]) or "unknown",
    }


def machine_label(record: dict) -> str:
    """The machine type for a source label: the most specific name available."""
    if record["aws_instance_type"] != "unknown":
        name = record["aws_instance_type"]
    elif record["dmi_product_name"] != "unknown":
        name = f"{record['dmi_sys_vendor']}-{record['dmi_product_name']}"
    else:
        name = f"{record['machine']}-{record['cpu_model']}"
    return re.sub(r"[^A-Za-z0-9._-]+", "-", name).strip("-")


def iam_role_problem(machine: dict) -> str | None:
    """Why this instance's IAM role is a risk to these commands; None when it is not."""
    attached = machine["aws_iam_role_attached"]
    if not attached and (attached is False or machine["aws_instance_type"] == "unknown"):
        return None
    state = "has an IAM role" if attached else "may have an IAM role (the metadata service did not say)"
    return (
        f"this instance {state}, and these third-party commands can reach its metadata service from Docker's default "
        "network. Detach the role, or require IMDSv2 with a hop limit of 1 (HttpTokens=required and "
        "HttpPutResponseHopLimit=1): a hop limit alone leaves IMDSv1 open."
    )


def check_host(allow_emulated: bool) -> None:
    server_arch = _text(["docker", "version", "--format", "{{.Server.Arch}}"])
    if server_arch is None:
        raise SystemExit(
            "cannot reach a Docker daemon. Re-execution needs a real one, which rules out rental offerings that are "
            "themselves containers."
        )
    if allow_emulated:
        return
    if platform.machine() != "x86_64" or server_arch != "amd64":
        raise SystemExit(
            f"refusing to measure tool durations on {platform.machine()} with a {server_arch} Docker daemon: emulated "
            "commands run at an unrelated speed. Use the x86 machine, or --allow-emulated for a pipeline check, whose "
            "source label can never count as a result."
        )


@dataclass
class CommandResult:
    returncode: int
    duration_s: float
    timed_out: bool
    output_bytes: int
    output_sha256: str
    text: str
    truncated: bool


def execute(argv: list[str], timeout: float) -> CommandResult:
    """Run one command, time it, and keep a bounded part of its merged output."""
    digest = hashlib.sha256()
    kept: list[bytes] = []
    sizes = [0]
    start = time.perf_counter()
    process = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)

    def drain() -> None:
        while chunk := process.stdout.read(65536):
            digest.update(chunk)
            if sizes[0] < KEEP_BYTES:
                kept.append(chunk[: KEEP_BYTES - sizes[0]])
            sizes[0] += len(chunk)

    reader = threading.Thread(target=drain, daemon=True)
    reader.start()
    # Like subprocess.run: the command is done when it exits and its pipe closes.
    reader.join(timeout)
    timed_out = reader.is_alive()
    returncode = -1
    if not timed_out:
        try:
            returncode = process.wait(max(timeout - (time.perf_counter() - start), 0.001))
        except subprocess.TimeoutExpired:
            timed_out = True
    if timed_out:
        process.kill()
        process.wait()
    duration = time.perf_counter() - start
    reader.join(5.0)
    process.stdout.close()
    return CommandResult(
        returncode, duration, timed_out, sizes[0], digest.hexdigest(), b"".join(kept).decode("utf-8", "replace"), sizes[0] > KEEP_BYTES
    )


_VOLATILE = [
    (re.compile(r"0x[0-9a-fA-F]+"), "0x"),
    (re.compile(r"\b\d+\.\d+\s*(?:s|ms|sec|seconds)\b"), "<time>"),
    (re.compile(r"\b\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}[.,\d]*"), "<date>"),
    (re.compile(r"\s+"), " "),
]


def normalized_lines(text: str) -> list[str]:
    lines = []
    for line in text.splitlines()[:SIMILARITY_LINES]:
        for pattern, replacement in _VOLATILE:
            line = pattern.sub(replacement, line)
        lines.append(line.strip())
    return lines


def similarity(a: str, b: str) -> float:
    """Line-level match after masking addresses, durations and dates; 1.0 means the same."""
    left, right = normalized_lines(a), normalized_lines(b)
    if left == right:
        return 1.0
    return difflib.SequenceMatcher(None, left, right, autojunk=False).ratio()


def submission_of(text: str) -> str:
    """What mini-swe-agent keeps as the submission: everything after the mark's line."""
    return "".join(text.lstrip().splitlines(keepends=True)[1:])


class TaskLog:
    """Append-only and fsynced per record, so a crash loses at most the command in flight."""

    def __init__(self, directory: Path, instance: str) -> None:
        self.path = directory / f"{instance}.jsonl"
        if self.path.exists():
            # An unfinished attempt: keep it as evidence, out of the way of the new one.
            number = sum(1 for _ in directory.glob(f"{instance}.crashed-*"))
            self.path.rename(directory / f"{instance}.crashed-{number}")
        self._file = self.path.open("a")

    def append(self, record: dict) -> None:
        self._file.write(json.dumps(record) + "\n")
        self._file.flush()
        os.fsync(self._file.fileno())

    def close(self) -> None:
        self._file.close()


def read_log(path: Path) -> list[dict]:
    """A crash can cut the last line short; earlier lines are always whole."""
    records = []
    for line in path.read_text().splitlines():
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            break
    return records


def _pull(image: str, options: Options, log: TaskLog, stop: threading.Event) -> tuple[bool, float, bool, str | None]:
    """Returns (ok, pull seconds, whether this run pulled it, error). Timed apart from tool time."""
    if _text(["docker", "image", "inspect", "--format", "{{.Id}}", image]):
        return True, 0.0, False, None
    start = time.perf_counter()
    error = None
    about_image = False
    for attempt in range(options.pull_attempts):
        about_image = False
        try:
            result = subprocess.run(
                ["docker", "pull", "--platform", PLATFORM, image], capture_output=True, text=True, errors="replace", timeout=PULL_TIMEOUT_S
            )
        except subprocess.TimeoutExpired:
            error = f"timed out after {PULL_TIMEOUT_S} s"
        else:
            if result.returncode == 0:
                return True, time.perf_counter() - start, True, None
            error = (result.stderr or result.stdout).strip()[-400:]
            if any(text in error.lower() for text in FATAL_PULL_TEXT):
                raise RunAborted(f"pulling {image}: {error}")
            about_image = any(text in error.lower() for text in IMAGE_ANSWER_TEXT)
        log.append({"type": "pull_retry", "attempt": attempt, "error": error, "about_image": about_image})
        if attempt + 1 < options.pull_attempts:
            backoff = IMAGE_RETRY_BACKOFF_S if about_image else TRANSIENT_BACKOFF_S
            if stop.wait(backoff * 2**attempt):
                raise RunAborted("stopping")
    if not about_image:
        raise RunAborted(f"pulling {image}: {options.pull_attempts} attempts failed, none because of the image: {error}")
    return False, time.perf_counter() - start, False, error


def _remove_container(name: str) -> None:
    subprocess.run(["docker", "rm", "-f", name], capture_output=True, timeout=120)


def remove_stale_containers() -> list[str]:
    """Containers a killed run left behind. Only this tool's own name prefix is touched."""
    listing = _text(["docker", "ps", "-a", "--filter", f"name=^{CONTAINER_PREFIX}", "--format", "{{.Names}}"]) or ""
    names = [n for n in listing.split() if n.startswith(CONTAINER_PREFIX)]
    for name in names:
        _remove_container(name)
    return names


def run_task(document: dict, directory: Path, options: Options, machine: dict, stop: threading.Event) -> dict:
    """Execute one task's commands in order. Returns its end record."""
    instance = document["instance_id"]
    image = document["image"]
    log = TaskLog(directory, instance)
    log.append({
        "type": "start", "schema": LOG_SCHEMA, "instance_id": instance, "image": image, "submission": document["submission"],
        "emulated": options.emulated, "no_network": options.no_network, "workers": options.workers,
        "machine_label": options.machine_label, "machine": machine, "memory": options.memory, "pids_limit": options.pids_limit,
    })
    end = {"type": "end", "instance_id": instance, "valid": False}
    try:
        ok, pull_s, pulled, error = _pull(image, options, log, stop)
        end.update(pull_s=pull_s, pulled=pulled)
        if not ok:
            end.update(status="image_failed", reason=error)
            log.append(end)
            return end
        try:
            architecture = _text(["docker", "image", "inspect", "--format", "{{.Architecture}}", image])
            if architecture is None:
                raise RunAborted(f"could not inspect {image}, which was just there")
            digests = _text(["docker", "image", "inspect", "--format", "{{json .RepoDigests}}", image])
            end["image_digests"] = json.loads(digests) if digests else []
            if architecture != "amd64":
                end.update(status="image_failed", reason=f"image architecture is {architecture}, not amd64")
                log.append(end)
                return end
            _commands(document, options, log, end, stop)
        finally:
            if pulled and not options.keep_images:
                subprocess.run(["docker", "rmi", image], capture_output=True, timeout=300)
        log.append(end)
        return end
    finally:
        log.close()


def _commands(document: dict, options: Options, log: TaskLog, end: dict, stop: threading.Event) -> None:
    environment = document["environment"]
    name = f"{CONTAINER_PREFIX}{uuid.uuid4().hex[:12]}"
    try:
        try:
            started = subprocess.run(docker_run_argv(document["image"], name, options), capture_output=True, text=True, errors="replace", timeout=300)
        except subprocess.TimeoutExpired as error:
            raise RunAborted(f"starting a container from {document['image']}: no answer in {error.timeout} s") from error
        if started.returncode != 0:
            message = started.stderr.strip()[-400:]
            if not any(text in message.lower() for text in IMAGE_START_TEXT):
                raise RunAborted(f"starting a container from {document['image']}: {message}")
            end.update(status="container_failed", reason=message)
            return
        compared = matched = exact = piped = 0
        similarities: list[float] = []
        total = 0.0
        timeouts = 0
        recorded_timeouts = recorded_timeouts_completed = 0
        final_match = None
        for turn in document["turns"]:
            for index, command in enumerate(turn["commands"]):
                if stop.is_set():
                    raise RunAborted("stopping")
                result = execute(docker_exec_argv(name, environment, command["command"]), environment["timeout"])
                total += result.duration_s
                timeouts += result.timed_out
                record = {
                    "type": "command", "turn": turn["turn"], "index": index, "returncode": result.returncode,
                    "duration_s": result.duration_s, "timed_out": result.timed_out, "output_bytes": result.output_bytes,
                    "output_sha256": result.output_sha256, "truncated": result.truncated,
                }
                recorded_rc = command["recorded_returncode"]
                if recorded_rc is not None:
                    record["recorded_returncode"] = recorded_rc
                    record["rc_match"] = result.returncode == recorded_rc
                if recorded_rc == RECORDED_TIMEOUT_RC:
                    # Whether a command crosses 60 s says which machine ran it, not whether the
                    # replay diverged, and these are the slow installs. Counted, never compared.
                    recorded_timeouts += 1
                    recorded_timeouts_completed += not result.timed_out
                elif recorded_rc is not None:
                    compared += 1
                    piped += bool(PIPE.search(command["command"]))
                    record["byte_exact"] = result.text == (command["recorded_output"] or "") and not result.truncated
                    record["similarity"] = 1.0 if record["byte_exact"] else similarity(result.text, command["recorded_output"] or "")
                    matched += record["rc_match"]
                    exact += record["byte_exact"]
                    similarities.append(record["similarity"])
                elif SUBMIT_MARK in command["command"] and document.get("submission_sha256"):
                    # The agent ended here, so the recording's check is the patch it submitted.
                    own = hashlib.sha256(submission_of(result.text).encode("utf-8", "surrogatepass")).hexdigest()
                    final_match = own == document["submission_sha256"]
                    record["final_submission_match"] = final_match
                log.append(record)
        rate = matched / compared if compared else 0.0
        end["fidelity"] = {
            "commands_compared": compared,
            "rc_match_rate": rate,
            "byte_exact_rate": exact / compared if compared else 0.0,
            "mean_similarity": statistics.mean(similarities) if similarities else 0.0,
            "final_submission_match": final_match,
            "timed_out_commands": timeouts,
            "recorded_timeouts": recorded_timeouts,
            "recorded_timeouts_completed": recorded_timeouts_completed,
            "piped_rc_blind_share": piped / compared if compared else 0.0,
            "threshold": RC_MATCH_THRESHOLD,
            "similarity_floor": SIMILARITY_FLOOR,
        }
        end["tool_s_total"] = total
        failure = fidelity_failure(end["fidelity"])
        end["valid"] = failure is None
        end["status"] = "ok" if end["valid"] else "low_fidelity"
        if failure:
            end["reason"] = failure
    finally:
        _remove_container(name)


def fidelity_failure(fidelity: dict) -> str | None:
    """Why a replay does not count as the recorded run; None when it does."""
    problems = []
    if not fidelity["commands_compared"]:
        problems.append("no commands to compare")
    if fidelity["rc_match_rate"] < RC_MATCH_THRESHOLD:
        problems.append(f"return-code match {fidelity['rc_match_rate']:.3f} is under {RC_MATCH_THRESHOLD}")
    if fidelity["mean_similarity"] < SIMILARITY_FLOOR:
        problems.append(f"mean output similarity {fidelity['mean_similarity']:.3f} is under {SIMILARITY_FLOOR}")
    if fidelity["final_submission_match"] is False:
        problems.append("the final patch differs from the recorded submission")
    return "; ".join(problems) or None


def end_failure(end: dict) -> str | None:
    """Judged from the logged values, not a stored verdict, so a floor can be corrected after
    the run without executing anything again."""
    if "fidelity" in end:
        return fidelity_failure(end["fidelity"])
    return end.get("reason") or end.get("status") or "no fidelity recorded"


def ended_tasks(directory: Path) -> dict[str, dict]:
    ended = {}
    for path in directory.glob("*.jsonl"):
        records = read_log(path)
        if records and records[-1].get("type") == "end":
            ended[path.stem] = records[-1]
    return ended


def plan(order: list[str], ended: dict[str, dict], target: int) -> tuple[list[str], list[str], list[str]]:
    """Walk manifest order until the target is met or could be: (valid, failed, still to run)."""
    valid, failed, pending = [], [], []
    for instance in order:
        if len(valid) + len(pending) == target:
            break
        result = ended.get(instance)
        if result is None:
            pending.append(instance)
        elif end_failure(result) is None:
            valid.append(instance)
        else:
            failed.append(instance)
    return valid, failed, pending


def replacements(manifest: dict, ended: dict[str, dict]) -> list[dict]:
    """The k-th failure, in manifest order, is covered by the k-th reserve."""
    order = [*manifest["tasks"], *manifest["reserves"]]
    _, failed, _ = plan(order, ended, len(manifest["tasks"]))
    reserves = manifest["reserves"]
    return [
        {
            "failed": instance,
            "status": "low_fidelity" if "fidelity" in ended[instance] else ended[instance].get("status"),
            "reason": end_failure(ended[instance]),
            "rc_match_rate": (ended[instance].get("fidelity") or {}).get("rc_match_rate"),
            "mean_similarity": (ended[instance].get("fidelity") or {}).get("mean_similarity"),
            "final_submission_match": (ended[instance].get("fidelity") or {}).get("final_submission_match"),
            "replaced_by": reserves[index] if index < len(reserves) else None,
        }
        for index, instance in enumerate(failed)
    ]


def _write_json(path: Path, value: dict | list) -> None:
    partial = path.with_name(path.name + ".partial")
    partial.write_text(json.dumps(value, indent=1) + "\n")
    partial.replace(path)


def run(prepared: Path, out: Path, options: Options, machine: dict) -> int:
    manifest = load_manifest(prepared)
    directory = out / manifest["submission"]
    directory.mkdir(parents=True, exist_ok=True)
    run_path = directory / "run.json"
    settings = {"machine_label": options.machine_label, "emulated": options.emulated, "no_network": options.no_network, "workers": options.workers}
    if run_path.exists():
        before = json.loads(run_path.read_text())
        changed = {k: (before.get(k), v) for k, v in settings.items() if before.get(k) != v}
        if changed:
            raise SystemExit(f"{directory} holds results from different settings {changed}; use another --out")
    stale = remove_stale_containers()
    if stale:
        print(f"removed {len(stale)} containers left by an earlier run: {stale}", flush=True)
    order = [*manifest["tasks"], *manifest["reserves"]]
    target = len(manifest["tasks"])
    stop = threading.Event()
    aborted: str | None = None

    def write_state() -> None:
        ended = ended_tasks(directory)
        valid, failed, pending = plan(order, ended, target)
        _write_json(directory / "replacements.json", replacements(manifest, ended))
        _write_json(run_path, {
            **settings, "submission": manifest["submission"], "machine": machine,
            "memory": options.memory, "pids_limit": options.pids_limit, "rc_match_threshold": RC_MATCH_THRESHOLD,
            "valid": valid, "failed": failed, "pending": pending, "target": target,
        })

    write_state()
    inflight: dict[futures.Future, str] = {}
    with futures.ThreadPoolExecutor(options.workers) as pool:
        try:
            while True:
                ended = ended_tasks(directory)
                _, _, pending = plan(order, ended, target)
                running = set(inflight.values())
                if not aborted:
                    for instance in [p for p in pending if p not in running][: options.workers - len(inflight)]:
                        document = load_prepared(prepared, instance)
                        print(f"start {instance}", flush=True)
                        inflight[pool.submit(run_task, document, directory, options, machine, stop)] = instance
                if not inflight:
                    break
                done, _ = futures.wait(inflight, return_when=futures.FIRST_COMPLETED)
                for future in done:
                    instance = inflight.pop(future)
                    try:
                        end = future.result()
                    except RunAborted as error:
                        aborted = aborted or str(error)
                        stop.set()
                        continue
                    fidelity = end.get("fidelity") or {}
                    print(
                        f"end {instance}: {end.get('status')} rc_match={fidelity.get('rc_match_rate')} "
                        f"similarity={fidelity.get('mean_similarity')} final_patch={fidelity.get('final_submission_match')} "
                        f"exact={fidelity.get('byte_exact_rate')} tool_s={end.get('tool_s_total')} pull_s={end.get('pull_s')}",
                        flush=True,
                    )
                    write_state()
        except BaseException:
            stop.set()
            raise
    write_state()
    if aborted:
        print(f"ABORTED: {aborted}\nThe logs are intact; the same command resumes.", file=sys.stderr)
        return 3
    valid, failed, _ = plan(order, ended_tasks(directory), target)
    for row in replacements(manifest, ended_tasks(directory)):
        print(f"replaced {row['failed']} ({row['status']}: {row['reason']}) with {row['replaced_by']}", flush=True)
    if len(valid) < target:
        print(f"FAILED: only {len(valid)} valid tasks of {target}; {len(failed)} failed and the reserves are used up.", file=sys.stderr)
        return 1
    print(f"{len(valid)} valid tasks, {len(failed)} replaced. Results in {directory}", flush=True)
    return 0


def _source(manifest: dict, label: str, emulated: bool, target: int, no_network: bool = False) -> str:
    base = f"swebench-experiments/{manifest['submission']}:reexecuted:{label}"
    if emulated:
        return f"pipeline-check:{base}:emulated"
    if no_network:
        # Installs fail fast without a network, so these durations are not the recorded work.
        return f"pipeline-check:{base}:no-network"
    if manifest.get("selection") != "seeded" or target != REQUIRED_TASKS:
        return f"pipeline-check:{base}:subset-of-{target}"
    return f"recorded:{base}"


def _network_note(no_network: bool) -> str:
    return "network off" if no_network else "network on, unpinned installs resolved at run time"


def _recorded_tool_s(document: dict) -> float:
    return sum(turn["recorded_gap_s"] or 0.0 for turn in document["turns"])


def _install_seconds(document: dict, records: list[dict]) -> float:
    """Measured seconds in commands whose duration depends on the network."""
    commands = {(turn["turn"], index): command["command"] for turn in document["turns"] for index, command in enumerate(turn["commands"])}
    return sum(
        record["duration_s"] for record in records
        if record.get("type") == "command" and INSTALL_COMMAND.search(commands[record["turn"], record["index"]])
    )


def _min_and_median(values: list[float]) -> str:
    return f"min {min(values):.3f}, median {statistics.median(values):.3f}"


def _trace_notes(manifest: dict, ended: dict[str, dict], documents: dict[str, dict], logs: dict[str, list[dict]], selected: dict[str, dict]) -> dict[str, str]:
    """One string per fact, identical on every job: the report keeps a meta key only when few
    distinct values exist across a trace, so a per-job number would vanish from it."""
    valid = list(documents)
    fidelities = [ended[i]["fidelity"] for i in valid]
    checked = [f["final_submission_match"] for f in fidelities if f["final_submission_match"] is not None]
    failed = replacements(manifest, ended)
    if failed:
        brought_in = [i for i in valid if i not in manifest["tasks"]]
        before = sum(_recorded_tool_s(d) for d in selected.values())
        after = sum(_recorded_tool_s(d) for d in documents.values())
        details = "; ".join(f"{r['failed']} ({r['status']}: {' '.join(str(r['reason']).split())[:200]})" for r in failed)
        replaced = (
            f"{len(failed)} failed and left out: {details}. Reserves brought in: {', '.join(brought_in)}. "
            f"Recorded tool time {before:.0f} s for the {len(selected)} tasks selected, {after:.0f} s for the set measured"
        )
    else:
        replaced = f"none; all {len(manifest['tasks'])} selected tasks passed"
    total = sum(ended[i]["tool_s_total"] for i in valid)
    install = sum(_install_seconds(documents[i], logs[i]) for i in valid)
    start = logs[valid[0]][0]
    return {
        "replacements": replaced,
        "fidelity_summary": (
            f"return-code match {_min_and_median([f['rc_match_rate'] for f in fidelities])} (floor {RC_MATCH_THRESHOLD}); "
            f"output similarity {_min_and_median([f['mean_similarity'] for f in fidelities])} (floor {SIMILARITY_FLOOR}); "
            f"final patch matched in {sum(checked)} of {len(checked)} checked; "
            f"{sum(f['recorded_timeouts'] for f in fidelities)} recorded timeouts left out of the comparison, "
            f"{sum(f['recorded_timeouts_completed'] for f in fidelities)} of them finished in the replay"
        ),
        "exclusions": (
            f"{manifest['exclusion_rate']:.0%} of {manifest['walked']} walked tasks excluded before selection "
            f"({json.dumps(manifest['exclusion_counts'], sort_keys=True)}) at max_context {manifest['max_context']}. {manifest['bias_note'].rstrip('.')}"
        ),
        "install_share": (
            f"{install / total if total else 0.0:.1%} of measured tool time ({install:.0f} of {total:.0f} s) is install or download commands"
        ),
        "run_settings": f"{start['workers']} worker(s) at once; {_network_note(start['no_network'])}",
    }


def build_job(document: dict, records: list[dict], source: str, shared_prefix: int, arrival_s: float, manifest: dict, notes: dict[str, str]) -> TraceJob:
    instance = document["instance_id"]
    start = records[0]
    end = records[-1]
    durations: dict[int, list[float]] = {}
    for record in records:
        if record.get("type") == "command":
            durations.setdefault(record["turn"], []).append(record["duration_s"])
    turns = []
    gaps = []
    previous_context = None
    for turn in document["turns"]:
        expected = len(turn["commands"])
        got = durations.get(turn["turn"], [])
        if len(got) != expected:
            raise ValueError(f"{instance}: turn {turn['turn']} has {len(got)} measured commands, expected {expected}")
        prompt, completion = turn["provider_prompt_tokens"], turn["provider_completion_tokens"]
        first_command = turn["commands"][0]["command"] if turn["commands"] else ""
        turns.append(TraceTurn(
            turn["new_tokens"], turn["output_tokens"], tool_category(first_command), sum(got),
            prompt if previous_context is None else prompt - previous_context, completion,
        ))
        previous_context = prompt + completion
        gaps.append(turn["recorded_gap_s"])
    if turns[0].new_tokens <= shared_prefix:
        raise ValueError(f"{instance}: initial prompt is not longer than the shared prefix")
    turns[0].new_tokens -= shared_prefix
    meta = {
        "submission": document["submission"],
        "mini_version": document["mini_version"],
        "model": document["model"],
        "s3_key": document["s3_key"],
        "raw_sha256": document["raw_sha256"],
        "tokenizer_repo": manifest["tokenizer_repo"],
        "tokenizer_revision": manifest["tokenizer_revision"],
        "image": document["image"],
        "image_digests": end.get("image_digests"),
        "machine": start["machine"],
        "workers": start["workers"],
        "no_network": start["no_network"],
        "emulated": start["emulated"],
        "pull_s": end.get("pull_s"),
        "fidelity": end["fidelity"],
        **notes,
        "tool_s_basis": f"sum of re-executed command durations; docker exec, recording's timeout, cold container; {_network_note(start['no_network'])}",
        "recorded_gap_s": gaps,
        "recorded_gap_note": "the submitter's tool time on an unknown machine; for cross-checking only",
        "sessions": "single-session job: no forks",
    }
    return TraceJob(instance, arrival_s, [TraceSession(f"{instance}/main", turns)], source, shared_prefix, meta)


def convert(prepared: Path, results: Path, out: Path, arrival_rate: float = 0.2) -> list[TraceJob]:
    """Write the canonical trace. Refuses a short or mixed set of results."""
    manifest = load_manifest(prepared)
    directory = results / manifest["submission"]
    order = [*manifest["tasks"], *manifest["reserves"]]
    target = len(manifest["tasks"])
    ended = ended_tasks(directory)
    valid, _, pending = plan(order, ended, target)
    if len(valid) < target:
        raise SystemExit(f"only {len(valid)} valid tasks of {target} in {directory} ({len(pending)} never ran); not writing a trace")
    logs = {i: read_log(directory / f"{i}.jsonl") for i in valid}
    settings = {(logs[i][0]["machine_label"], logs[i][0]["emulated"], logs[i][0]["no_network"], logs[i][0]["workers"]) for i in valid}
    if len(settings) != 1:
        raise SystemExit(f"tasks in {directory} ran under different settings: {sorted(settings)}")
    label, emulated, no_network, _ = next(iter(settings))
    source = _source(manifest, label, emulated, target, no_network)
    documents = {i: load_prepared(prepared, i) for i in valid}
    for instance, document in documents.items():
        if document["raw_sha256"] != manifest["raw"][instance]["sha256"]:
            raise SystemExit(f"{instance}: prepared file does not match the manifest")
    selected = {i: documents.get(i) or load_prepared(prepared, i) for i in manifest["tasks"]}
    notes = _trace_notes(manifest, ended, documents, logs, selected)
    jobs = [
        build_job(document, logs[instance], source, manifest["shared_prefix_tokens"], index / arrival_rate, manifest, notes)
        for index, (instance, document) in enumerate(documents.items())
    ]
    out.parent.mkdir(parents=True, exist_ok=True)
    save_jsonl(jobs, out)
    return jobs


def default_workers() -> int:
    # One, whatever the machine: a second worker's pulls and removals would overlap timed commands.
    return 1


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(prog="python -m layover.replay.reexecute", description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    runner = commands.add_parser(
        "run",
        help="re-execute the manifest's commands; x86 only",
        description=(
            "Do not run the benchmark at the same time: CPU and disk contention in either direction would change "
            "the durations measured here."
        ),
    )
    runner.add_argument("--prepared", type=Path, default=Path("traces/prepared"))
    runner.add_argument("--out", type=Path, default=Path("traces/reexec"))
    runner.add_argument(
        "--workers", type=int, default=default_workers(),
        help=(
            "tasks at once; default 1. More is an opt-in: pulls and removals by one worker overlap the timed commands of "
            "another, with no CPU limit between containers. The benchmark must not run at the same time. Recorded with "
            "results, and a resume must use the same count"
        ),
    )
    runner.add_argument("--memory", default=Options.memory, help="container memory limit")
    runner.add_argument("--pids-limit", type=int, default=Options.pids_limit)
    runner.add_argument(
        "--pull-attempts", type=int, default=Options.pull_attempts,
        help="per image, with growing backoff; then a failure that is not about the image aborts the run, which can resume",
    )
    runner.add_argument("--keep-images", action="store_true", help="keep images this run pulled; default removes them after each task")
    runner.add_argument(
        "--no-network", action="store_true",
        help="no network in containers; recorded. Off by default because trajectories run pip install and would diverge",
    )
    runner.add_argument(
        "--allow-emulated", action="store_true",
        help="run anywhere, as a pipeline check: the source label begins pipeline-check:, which the benchmark refuses for headlines",
    )
    runner.add_argument(
        "--allow-iam-role", action="store_true",
        help="run although the instance has an IAM role, once the metadata service is protected from containers",
    )
    converter = commands.add_parser("convert", help="write the canonical trace from a finished run")
    converter.add_argument("--prepared", type=Path, default=Path("traces/prepared"))
    converter.add_argument("--results", type=Path, default=Path("traces/reexec"))
    converter.add_argument("--arrival-rate", type=float, default=0.2, help="jobs per second; the benchmark retimes arrivals anyway")
    converter.add_argument("--out", type=Path, default=Path("traces/swebench_30.jsonl"))
    args = parser.parse_args(argv)

    if args.command == "convert":
        jobs = convert(args.prepared, args.results, args.out, args.arrival_rate)
        total = sum(t.tool_s for j in jobs for t in j.main.turns)
        print(f"{len(jobs)} jobs, {total:.1f} s of tool time, source {jobs[0].source}\nwrote {args.out}")
        return

    check_host(args.allow_emulated)
    machine = machine_record()
    problem = iam_role_problem(machine)
    if problem and not args.allow_iam_role:
        raise SystemExit(f"refusing to run: {problem} --allow-iam-role runs anyway once that is done.")
    if problem:
        print(f"WARNING: {problem}", file=sys.stderr)
    options = Options(
        args.memory, args.pids_limit, args.no_network, args.keep_images, args.pull_attempts, args.allow_emulated, args.workers,
        machine_label(machine),
    )

    def interrupted(*_) -> None:
        raise KeyboardInterrupt

    # Turn termination into an exception, so `finally` removes the containers.
    signal.signal(signal.SIGTERM, interrupted)
    raise SystemExit(run(args.prepared, args.out, options, machine))


if __name__ == "__main__":
    main()
