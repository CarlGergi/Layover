"""The systems under test: how each is launched, checked, measured, and stopped.

Every system gets the same model, precision, context limit, KV budget, CPU tier budget,
and concurrency limit, translated to its own flags by translate(). The translation is
then checked against the running server, because a flag can be accepted and still not do
what its name says. bench/configs/baseline_flags.md gives a reason for every flag, how it
was checked, and what could not be checked without a GPU.
"""

from __future__ import annotations

import os
import re
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import httpx

GIB = 2**30
REPO = Path(__file__).resolve().parents[1]

# Flags and metric names were checked against these versions; baseline_flags.md says how.
VLLM_VERSION = "0.33.0"
SGLANG_VERSION = "0.5.23"

_DTYPE_NAMES = {"vllm": {"bf16": "bfloat16"}, "sglang": {"bf16": "bfloat16"}}
_KV_DTYPE_NAMES = {"vllm": {"bf16": "bfloat16"}, "sglang": {"bf16": "bf16"}}
BLOCK_TOKENS = 16


class ServerFailed(RuntimeError):
    """A server died, never became ready, or misbehaved. The run may be retried."""


class BudgetMismatch(ServerFailed):
    """A system is not holding the budget it was given, so its result would not be comparable."""


class GpuNotReleased(RuntimeError):
    """GPU memory stayed occupied after a stop. Later runs would be contaminated, so the matrix ends."""


@dataclass(frozen=True)
class MatchedSettings:
    """What every system must share. Nothing here is specific to one system."""

    model: str
    device: str
    dtype: str
    max_context_tokens: int
    kv_gib: float
    cpu_offload_gib: float
    max_running: int
    prefill_chunk: int
    seed: int = 0

    @classmethod
    def from_config(cls, raw: dict) -> "MatchedSettings":
        return cls(**{name: raw[name] for name in cls.__dataclass_fields__ if name in raw})


@dataclass(frozen=True)
class Translation:
    """One system's knobs for the matched settings, and the capacity they should produce."""

    kv_flag: str
    kv_value: str
    kv_tokens_expected: int
    cpu_flag: str
    cpu_value: str
    cpu_gib_effective: float
    concurrency_flag: str
    prefill_flag: str
    prefill_value: int


def translate(kind: str, settings: MatchedSettings, kv_bytes_per_token: int) -> Translation:
    """Turn the shared budgets into one system's flags.

    vLLM and Layover allocate whole 16-token blocks, so their capacity rounds down to a block.
    SGLang counts tokens. Its --hicache-size takes only whole decimal gigabytes (an int field,
    host_size * 1e9 in sglang/srt/mem_cache/pool_host/base.py at 0.5.23), and no whole number
    is 32 GiB: 34 GB is 31.66 GiB. --hicache-ratio is a float over the device pool's tokens, at
    the same bytes per token, so it reaches the budget to within a page (same file).
    vLLM's --max-num-batched-tokens covers decode tokens too, so it needs room for a full
    decode batch on top of the prefill chunk the others are given.
    """
    kv_bytes = int(settings.kv_gib * GIB)
    if kv_bytes_per_token <= 0:
        raise ValueError(f"kv_bytes_per_token must be positive, got {kv_bytes_per_token}")
    block_bytes = kv_bytes_per_token * BLOCK_TOKENS
    block_tokens = kv_bytes // block_bytes * BLOCK_TOKENS
    if kind == "vllm":
        return Translation(
            "--kv-cache-memory-bytes", str(kv_bytes), block_tokens,
            "--kv-offloading-size", _number(settings.cpu_offload_gib), settings.cpu_offload_gib,
            "--max-num-seqs", "--max-num-batched-tokens", settings.prefill_chunk + settings.max_running,
        )
    if kind == "sglang":
        return Translation(
            "--max-total-tokens", str(kv_bytes // kv_bytes_per_token), kv_bytes // kv_bytes_per_token,
            "--hicache-ratio", repr(settings.cpu_offload_gib / settings.kv_gib), settings.cpu_offload_gib,
            "--max-running-requests", "--chunked-prefill-size", settings.prefill_chunk,
        )
    if kind == "layover":
        return Translation(
            "--pool-gib", _number(settings.kv_gib), block_tokens,
            "--cpu-tier-gib", _number(settings.cpu_offload_gib), settings.cpu_offload_gib,
            "--max-batch", "--prefill-chunk", settings.prefill_chunk,
        )
    raise ValueError(f"no translation for system kind {kind!r}")


def _number(value: float) -> str:
    return repr(value) if value != int(value) else str(int(value))


def offload_record_bytes(cfg, offload_dtype: str | None) -> int:
    """What one block occupies in Layover's CPU tier, as layover/server.py sizes it.

    The budget is bytes, so a tier of FP8 records holds about twice the blocks of a BF16 one.
    """
    if offload_dtype == "fp8":
        from layover.kernels.fp8 import Fp8Layout

        return Fp8Layout(BLOCK_TOKENS * cfg.head_dim, 2 * cfg.num_hidden_layers * cfg.num_key_value_heads).record_bytes
    return BLOCK_TOKENS * cfg.kv_bytes_per_token("bf16")


def parse_prometheus(text: str) -> dict[str, float]:
    """Metric name to value, summed over label sets."""
    values: dict[str, float] = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        match = re.match(r"([a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{.*\})?\s+(\S+)", line)
        if match is None:
            continue
        try:
            values[match[1]] = values.get(match[1], 0.0) + float(match[2])
        except ValueError:
            continue
    return values


def _find_key(data, key: str):
    if isinstance(data, dict):
        if key in data:
            return data[key]
        data = list(data.values())
    if isinstance(data, list):
        for item in data:
            found = _find_key(item, key)
            if found is not None:
                return found
    return None


class System:
    """A server the matrix can run. Subclasses say how; the replay client is shared."""

    kind = "base"
    # Session and tool hints mean something only to Layover.
    hints = False
    # A system that loses its KV at every tool call. Time estimates plan on it recomputing.
    recomputes_context = False
    # Where reported_cpu_tier_gib gets its number, for the budget table.
    cpu_tier_source = "not exposed"

    def __init__(self, name: str, settings: MatchedSettings, kv_bytes_per_token: int):
        self.name = name
        self.settings = settings
        self.kv_bytes_per_token = kv_bytes_per_token

    @property
    def translation(self) -> Translation | None:
        return translate(self.kind, self.settings, self.kv_bytes_per_token)

    def command(self, host: str, port: int) -> list[str]:
        raise NotImplementedError

    def environment(self) -> dict[str, str]:
        return {}

    def installed_version(self) -> str | None:
        return None

    def expected_version(self) -> str | None:
        return None

    def reported_kv_tokens(self, client: httpx.Client, log_text: str) -> int | None:
        """The KV capacity the running server says it has."""
        return None

    def reported_cpu_tier_gib(self, client: httpx.Client, log_text: str) -> float | None:
        """The CPU tier the running server says it has, in GiB, or None if it does not say."""
        return None

    def kv_used_fraction(self, client: httpx.Client) -> float | None:
        """The KV pool's share that requests hold, with cached blocks nothing holds counted as free.

        Every system reports it that way, or the column would compare definitions. NVML cannot
        tell: every system preallocates its pool.
        """
        return None

    def kv_fractions(self, client: httpx.Client) -> tuple[float | None, float | None]:
        """kv_used_fraction, and the share held only by a radix tree where that is known apart."""
        return self.kv_used_fraction(client), None

    def server_metrics(self, client: httpx.Client) -> dict:
        """The system's own counters, kept apart from the client's measurements."""
        return {}

    def _prometheus(self, client: httpx.Client) -> dict[str, float]:
        response = client.get("/metrics")
        response.raise_for_status()
        return parse_prometheus(response.text)


def _venv_path(venv: Path) -> dict[str, str]:
    """The environment's bin first on PATH, as activating it would. vLLM builds kernels at start-up
    with ninja from there; launched by full path without this, it failed on the L4."""
    return {"PATH": f"{venv / 'bin'}{os.pathsep}{os.environ.get('PATH', '')}"}


class VllmSystem(System):
    kind = "vllm"
    # vLLM 0.33.0 exposes no tier capacity counter; this only proves the flag reached it.
    cpu_tier_source = "startup log echo of the flag; no capacity counter"

    def __init__(self, name, settings, kv_bytes_per_token, venv: str = ".venvs/vllm"):
        super().__init__(name, settings, kv_bytes_per_token)
        self.venv = REPO / venv

    def command(self, host: str, port: int) -> list[str]:
        s, t = self.settings, self.translation
        return [
            str(self.venv / "bin" / "vllm"), "serve", s.model,
            "--served-model-name", s.model,
            "--host", host, "--port", str(port),
            "--dtype", _DTYPE_NAMES["vllm"][s.dtype],
            "--kv-cache-dtype", _KV_DTYPE_NAMES["vllm"][s.dtype],
            "--max-model-len", str(s.max_context_tokens),
            t.kv_flag, t.kv_value,
            "--enable-prefix-caching",
            "--kv-offloading-backend", "native",
            t.cpu_flag, t.cpu_value,
            "--scheduling-policy", "fcfs",
            t.concurrency_flag, str(s.max_running),
            "--enable-chunked-prefill",
            t.prefill_flag, str(t.prefill_value),
            "--stream-interval", "1",
            "--generation-config", "vllm",
            "--enable-prompt-tokens-details",
            "--disable-uvicorn-access-log",
            "--seed", str(s.seed),
        ]

    def environment(self) -> dict[str, str]:
        return _venv_path(self.venv)

    def installed_version(self) -> str | None:
        return _package_version(self.venv / "bin" / "python", "vllm")

    def expected_version(self) -> str:
        return VLLM_VERSION

    def reported_kv_tokens(self, client, log_text):
        match = re.search(r"KV cache size: ([\d,]+) tokens", log_text)
        return int(match[1].replace(",", "")) if match else None

    def reported_cpu_tier_gib(self, client, log_text):
        match = re.search(r"""['"]kv_offloading_size['"]:\s*([\d.]+)""", log_text)
        return float(match[1]) if match else None

    def kv_used_fraction(self, client):
        return self._prometheus(client).get("vllm:kv_cache_usage_perc")

    def server_metrics(self, client):
        metrics = self._prometheus(client)
        names = (
            "vllm:request_queue_time_seconds_sum", "vllm:request_queue_time_seconds_count",
            "vllm:time_to_first_token_seconds_sum", "vllm:time_to_first_token_seconds_count",
            "vllm:prompt_tokens_total", "vllm:prefix_cache_hits_total", "vllm:prefix_cache_queries_total",
            "vllm:external_prefix_cache_hits_total", "vllm:external_prefix_cache_queries_total",
            "vllm:num_preemptions_total",
        )
        return {name: metrics[name] for name in names if name in metrics}


class SglangSystem(System):
    kind = "sglang"
    cpu_tier_source = "host pool tokens x KV bytes per token, from its startup log or sglang:hicache_host_total_tokens"

    def __init__(self, name, settings, kv_bytes_per_token, venv: str = ".venvs/sglang"):
        super().__init__(name, settings, kv_bytes_per_token)
        self.venv = REPO / venv

    def command(self, host: str, port: int) -> list[str]:
        s, t = self.settings, self.translation
        return [
            str(self.venv / "bin" / "sglang"), "serve",
            "--model-path", s.model,
            "--served-model-name", s.model,
            "--host", host, "--port", str(port),
            "--dtype", _DTYPE_NAMES["sglang"][s.dtype],
            "--kv-cache-dtype", _KV_DTYPE_NAMES["sglang"][s.dtype],
            "--context-length", str(s.max_context_tokens),
            t.kv_flag, t.kv_value,
            # High enough that --max-total-tokens is what binds; the check after startup proves it.
            "--mem-fraction-static", "0.9",
            "--enable-hierarchical-cache",
            t.cpu_flag, t.cpu_value,
            "--schedule-policy", "fcfs",
            t.concurrency_flag, str(s.max_running),
            t.prefill_flag, str(t.prefill_value),
            "--stream-interval", "1",
            "--sampling-defaults", "openai",
            "--enable-metrics",
            "--enable-cache-report",
            "--random-seed", str(s.seed),
        ]

    def environment(self) -> dict[str, str]:
        return _venv_path(self.venv)

    def installed_version(self) -> str | None:
        return _package_version(self.venv / "bin" / "python", "sglang")

    def expected_version(self) -> str:
        return SGLANG_VERSION

    def reported_kv_tokens(self, client, log_text):
        try:
            response = client.get("/server_info")
            response.raise_for_status()
            tokens = _find_key(response.json(), "max_total_num_tokens")
            if tokens is not None:
                return int(tokens)
        except (httpx.HTTPError, ValueError):
            pass
        match = re.search(r"max_total_num_tokens=(\d+)", log_text)
        return int(match[1]) if match else None

    def reported_cpu_tier_gib(self, client, log_text):
        # The gauge is pushed only with scheduler stats, so at startup only the log has it. The
        # log's own bytes come first: they do not assume its per-token size equals ours.
        match = re.search(r"hierarchical KV host pool: (\d+) tokens, ([\d.]+) GB host memory", log_text)
        if match:
            return float(match[2]) * 1e9 / GIB
        match = re.search(r"hierarchical KV host pool: (\d+) tokens", log_text)
        tokens = int(match[1]) if match else None
        if tokens is None and client is not None:
            tokens = self._prometheus(client).get("sglang:hicache_host_total_tokens") or None
        return tokens * self.kv_bytes_per_token / GIB if tokens else None

    def kv_used_fraction(self, client):
        return self._prometheus(client).get("sglang:token_usage")

    def server_metrics(self, client):
        metrics = self._prometheus(client)
        names = (
            "sglang:queue_time_seconds_sum", "sglang:queue_time_seconds_count",
            "sglang:time_to_first_token_seconds_sum", "sglang:time_to_first_token_seconds_count",
            "sglang:cache_hit_rate", "sglang:kv_cache_memory_usage_gb", "sglang:hicache_host_used_tokens",
            "sglang:hicache_host_total_tokens", "sglang:load_back_tokens_total", "sglang:hicache_backup_tokens_total",
            "sglang:evicted_tokens_total",
        )
        return {name: metrics[name] for name in names if name in metrics}


class LayoverSystem(System):
    kind = "layover"
    hints = True
    cpu_tier_source = "cpu_tier_capacity_blocks x bytes per record, from /metrics"

    def __init__(
        self, name, settings, kv_bytes_per_token, policy: str, cost_model: str, ttl_cap_s: float = 30.0,
        decode_backend: str | None = None, transfer: str | None = None, offload_dtype: str | None = None,
        prefix_cache: bool = True, record_bytes: int | None = None,
    ):
        super().__init__(name, settings, kv_bytes_per_token)
        self.policy = policy
        self.prefix_cache = prefix_cache
        self.cost_model = cost_model
        self.ttl_cap_s = ttl_cap_s
        self.decode_backend = decode_backend
        self.transfer = transfer
        self.offload_dtype = offload_dtype
        self.record_bytes = record_bytes

    @property
    def recomputes_context(self) -> bool:
        # With the prefix cache on, a freed turn stays reusable until pressure evicts it.
        return self.policy == "discard" and not self.prefix_cache

    def command(self, host: str, port: int) -> list[str]:
        s, t = self.settings, self.translation
        command = [
            sys.executable, "-m", "layover.server",
            "--model", s.model,
            "--device", s.device,
            "--dtype", s.dtype,
            t.kv_flag, t.kv_value,
            t.cpu_flag, t.cpu_value,
            "--policy", self.policy,
            "--cost-model", str(self.cost_model),
            "--ttl-cap-s", _number(self.ttl_cap_s),
            t.concurrency_flag, str(s.max_running),
            t.prefill_flag, str(t.prefill_value),
            "--host", host, "--port", str(port),
        ]
        for flag, value in (
            ("--decode-backend", self.decode_backend),
            ("--transfer", self.transfer),
            ("--offload-dtype", self.offload_dtype),
        ):
            if value is not None:
                command += [flag, value]
        if not self.prefix_cache:
            command.append("--no-prefix-cache")
        return command

    def _metrics(self, client) -> dict:
        response = client.get("/metrics")
        response.raise_for_status()
        return response.json()

    def reported_kv_tokens(self, client, log_text):
        return self._metrics(client)["blocks_total"] * BLOCK_TOKENS

    def reported_cpu_tier_gib(self, client, log_text):
        if self.record_bytes is None or client is None:
            return None
        return self._metrics(client)["cpu_tier_capacity_blocks"] * self.record_bytes / GIB

    def kv_fractions(self, client):
        metrics = self._metrics(client)
        tree_only = metrics.get("radix_only_blocks")
        # Without the count there is no figure that compares, and blocks_free alone is a different one.
        if tree_only is None:
            return None, None
        total = metrics["blocks_total"]
        # A block only the tree holds is evictable: vLLM's free queue holds exactly those.
        return 1 - (metrics["blocks_free"] + tree_only) / total, tree_only / total

    def kv_used_fraction(self, client):
        return self.kv_fractions(client)[0]

    def server_metrics(self, client):
        return self._metrics(client)


class CommandSystem(System):
    """Any command that serves the API. The matrix accepts it only for a pipeline check."""

    kind = "command"

    def __init__(self, name, settings, kv_bytes_per_token, command: list[str]):
        super().__init__(name, settings, kv_bytes_per_token)
        self.template = command

    @property
    def translation(self) -> None:
        return None

    def command(self, host: str, port: int) -> list[str]:
        return [part.format(host=host, port=port, python=sys.executable, model=self.settings.model) for part in self.template]

    def server_metrics(self, client):
        try:
            return self._prometheus(client)
        except httpx.HTTPError:
            return {}


def _package_version(python: Path, package: str) -> str | None:
    if not python.exists():
        return None
    result = subprocess.run(
        [str(python), "-c", f"import importlib.metadata as m; print(m.version({package!r}))"],
        capture_output=True, text=True, timeout=120,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def build_system(spec: dict, settings: MatchedSettings, kv_bytes_per_token: int, cfg=None) -> System:
    """Build one system from its config entry. With cfg, Layover's tier size can be checked in bytes."""
    kind = spec["kind"]
    if kind == "vllm":
        return VllmSystem(spec["name"], settings, kv_bytes_per_token, spec.get("venv", ".venvs/vllm"))
    if kind == "sglang":
        return SglangSystem(spec["name"], settings, kv_bytes_per_token, spec.get("venv", ".venvs/sglang"))
    if kind == "layover":
        return LayoverSystem(
            spec["name"], settings, kv_bytes_per_token, spec["policy"], spec["cost_model"], spec.get("ttl_cap_s", 30.0),
            spec.get("decode_backend"), spec.get("transfer"), spec.get("offload_dtype"), spec.get("prefix_cache", True),
            offload_record_bytes(cfg, spec.get("offload_dtype")) if cfg is not None else None,
        )
    if kind == "command":
        return CommandSystem(spec["name"], settings, kv_bytes_per_token, spec["command"])
    raise ValueError(f"unknown system kind {kind!r}")


def _process_table() -> list[tuple[int, int, str, str]]:
    """(pid, parent, state, command) for every process, on Linux and macOS.

    Linux reads /proc, because slim images have no ps. Anywhere else, a missing ps is an
    error: an empty table would let a stray worker keep its GPU memory without a word.
    """
    proc = Path("/proc")
    if not (proc / "self" / "stat").exists():
        listing = subprocess.run(
            ["ps", "-A", "-o", "pid=,ppid=,stat=,command="], capture_output=True, text=True, timeout=10, check=True
        ).stdout
        table = []
        for line in listing.splitlines():
            parts = line.split(None, 3)
            if len(parts) >= 3:
                table.append((int(parts[0]), int(parts[1]), parts[2][0], parts[3] if len(parts) > 3 else ""))
        return table
    table = []
    for entry in proc.iterdir():
        if not entry.name.isdigit():
            continue
        try:
            stat = (entry / "stat").read_text(errors="replace")
            command = (entry / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace").strip()
        except OSError:  # It exited while being read.
            continue
        # The command name in field 2 may hold spaces and parentheses; fields after it may not.
        state, parent = stat[stat.rindex(")") + 2 :].split()[:2]
        table.append((int(entry.name), int(parent), state, command))
    return table


def process_running(pid: int) -> bool:
    """Whether pid exists and is not a zombie, which holds no memory."""
    return any(found == pid and state != "Z" for found, _, state, _ in _process_table())


def running_commands() -> list[str]:
    return [command for _, _, state, command in _process_table() if state != "Z"]


def _descendants(pid: int) -> list[int]:
    """Every process under pid."""
    children: dict[int, list[int]] = {}
    for child, parent, _, _ in _process_table():
        children.setdefault(parent, []).append(child)
    found, frontier = [], [pid]
    while frontier:
        for child in children.get(frontier.pop(), []):
            found.append(child)
            frontier.append(child)
    return found


def _kill_quietly(pid: int, sig: int) -> None:
    try:
        os.kill(pid, sig)
    except (ProcessLookupError, PermissionError):
        pass


class ManagedServer:
    """One server process with its own process group and log file."""

    def __init__(self, system: System, log_path: Path, port: int, host: str = "127.0.0.1", stop_timeout_s: float = 30.0):
        self.system = system
        self.log_path = Path(log_path)
        self.port = port
        self.host = host
        self.stop_timeout_s = stop_timeout_s
        self.process: subprocess.Popen | None = None
        self.log_start = 0
        self.url = f"http://{host}:{port}"
        self.client = httpx.Client(base_url=self.url, timeout=30.0)

    def start(self) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("ab") as log:
            # A killed attempt's log is appended to; its capacity lines must not answer for this server.
            self.log_start = self.log_path.stat().st_size
            log.write(f"$ {' '.join(self.system.command(self.host, self.port))}\n".encode())
            log.flush()
            # A session of its own, so one signal reaches every worker it spawns.
            self.process = subprocess.Popen(
                self.system.command(self.host, self.port), stdout=log, stderr=subprocess.STDOUT,
                cwd=REPO, env={**os.environ, **self.system.environment()}, start_new_session=True,
            )

    def log_text(self) -> str:
        if not self.log_path.exists():
            return ""
        with self.log_path.open("rb") as log:
            log.seek(self.log_start)
            return log.read().decode(errors="replace")

    def log_tail(self, lines: int = 25) -> str:
        return "\n".join(self.log_text().splitlines()[-lines:])

    def wait_ready(self, timeout_s: float, poll_s: float = 1.0) -> None:
        """Wait for /health, then confirm the model is the one asked for.

        Only at startup: SGLang's /health generates a token when the server is idle, so it
        must never be polled during a measured run.
        """
        deadline = time.monotonic() + timeout_s
        while True:
            if self.process.poll() is not None:
                raise ServerFailed(
                    f"{self.system.name} exited with code {self.process.returncode} during startup; "
                    f"log {self.log_path}:\n{self.log_tail()}"
                )
            try:
                if self.client.get("/health").status_code == 200:
                    break
            except httpx.HTTPError:
                pass
            if time.monotonic() > deadline:
                raise ServerFailed(f"{self.system.name} not healthy after {timeout_s}s; log {self.log_path}:\n{self.log_tail()}")
            time.sleep(poll_s)
        response = self.client.get("/v1/models")
        response.raise_for_status()
        served = [model["id"] for model in response.json()["data"]]
        if self.system.settings.model not in served:
            raise ServerFailed(f"{self.system.name} serves {served}, not {self.system.settings.model}")

    def stop(self) -> None:
        """Stop the whole group, and any descendant that left it. Safe to call twice."""
        self.client.close()
        process = self.process
        if process is None:
            return
        self.process = None
        strays = _descendants(process.pid)
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=self.stop_timeout_s)
        except subprocess.TimeoutExpired:
            pass
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        for pid in strays:
            _kill_quietly(pid, signal.SIGKILL)
        process.wait(timeout=10)


def check_kv_capacity(name: str, reported: int | None, expected: int, tolerance_frac: float) -> dict:
    """Fail when a server's own KV capacity is off budget, or cannot be read.

    An unreadable capacity fails too: a log line that changed wording would otherwise leave
    the budget unchecked for the whole matrix.
    """
    result = {"reported_kv_tokens": reported, "expected_kv_tokens": expected, "tolerance_frac": tolerance_frac}
    if reported is None:
        raise BudgetMismatch(f"{name} did not report its KV capacity, so its budget of {expected:,} tokens cannot be verified")
    if abs(reported - expected) > tolerance_frac * expected:
        raise BudgetMismatch(
            f"{name} reports {reported:,} KV tokens against a budget of {expected:,} "
            f"(tolerance {tolerance_frac:.0%}); results would not be comparable"
        )
    return result


def check_cpu_tier(name: str, reported_gib: float | None, expected_gib: float, tolerance_frac: float) -> dict:
    """Fail when a server's own CPU tier is off budget.

    Unlike the KV pool this may be unreadable: not every system exposes it, and the budget
    table says which do. What a system does report has to agree.
    """
    result = {"reported_cpu_tier_gib": reported_gib, "expected_cpu_tier_gib": expected_gib}
    if reported_gib is not None and abs(reported_gib - expected_gib) > tolerance_frac * expected_gib:
        raise BudgetMismatch(
            f"{name} reports a CPU tier of {reported_gib:.2f} GiB against a budget of {expected_gib:g} GiB "
            f"(tolerance {tolerance_frac:.0%}); results would not be comparable"
        )
    return result


def check_gpu_memory(name: str, used_bytes: int, idle_bytes: int, weights_bytes: int, kv_bytes: int, tolerance_gib: float) -> dict:
    """Fail when a loaded server's GPU memory is far from weights plus the KV budget.

    The tolerance covers what no budget names: CUDA context, activations, CUDA graphs. The
    KV capacity check above is the tight one; this one catches a pool that was promised
    and not allocated, or a system holding much more than its share.
    """
    held = used_bytes - idle_bytes
    expected = weights_bytes + kv_bytes
    deviation = (held - expected) / GIB
    result = {
        "gpu_held_gib": held / GIB,
        "gpu_expected_gib": expected / GIB,
        "gpu_deviation_gib": deviation,
        "gpu_tolerance_gib": tolerance_gib,
    }
    if abs(deviation) > tolerance_gib:
        raise BudgetMismatch(
            f"{name} holds {held / GIB:.2f} GiB of GPU memory against {expected / GIB:.2f} GiB of weights "
            f"plus KV budget, off by {deviation:+.2f} GiB (tolerance {tolerance_gib} GiB)"
        )
    return result


def wait_gpu_released(gpu, limit_bytes: int, timeout_s: float = 60.0, poll_s: float = 1.0) -> int:
    """Wait for GPU memory to come back. Returns what is still in use."""
    deadline = time.monotonic() + timeout_s
    while True:
        used = gpu.read().memory_used_bytes
        if used <= limit_bytes:
            return used
        if time.monotonic() > deadline:
            raise GpuNotReleased(
                f"GPU still holds {used / GIB:.2f} GiB, limit {limit_bytes / GIB:.2f} GiB, "
                f"{timeout_s}s after stopping. Processes: {gpu.compute_processes()}"
            )
        time.sleep(poll_s)
