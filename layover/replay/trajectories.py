"""Choose public SWE-bench trajectories and count their tokens with Qwen3's tokenizer.

`prepare` runs anywhere and costs nothing: it reads the SWE-bench submissions bucket over
plain HTTPS, fetches tokenizer files only, and executes nothing from a trajectory. Tool
durations come later, from layover/replay/reexecute.py on the x86 machine: the gaps in a
trajectory are the submitter's tool time on an unknown machine, so they are kept only as
recorded_gap_s for cross-checking.

The replayer's context is append-only (previous context + generated ids + new tokens), so
counts are split the same way. Each piece comes out of the real chat template:

    initial prompt   system, tool schema and first user message, with the generation prompt
    output_tokens    what the model emits after that prompt, up to and including <|im_end|>
    new_tokens       the newline, tool responses and the next generation prompt

Reasoning stays in the context. Provider usage shows it was re-sent (manifest field
reasoning_evidence), and Qwen3's template also keeps it after the last real user message.

The licence of the trajectories is not stated, so raw files and prepared files, which hold
commands and their recorded output, stay in the gitignored traces/ folder.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import random
import re
import ssl
import statistics
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

BUCKET = "https://swe-bench-submissions.s3.amazonaws.com"
PREPARED_SCHEMA = "layover-prepared-1"
MANIFEST_SCHEMA = "layover-trajectory-manifest-1"
SUBMIT_MARK = "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"

# mini-swe-agent v2.1.0, minisweagent/models/utils/actions_toolcall.py. A trajectory does
# not record the schema it was sent, so it is restated here.
BASH_TOOL = {
    "type": "function",
    "function": {
        "name": "bash",
        "description": "Execute a bash command",
        "parameters": {
            "type": "object",
            "properties": {"command": {"type": "string", "description": "The bash command to execute"}},
            "required": ["command"],
        },
    },
}

MAX_CONTEXT_RATIONALE = (
    "32768 is the largest context length the KV crossover benchmark measures (DEFAULT_TOKENS in "
    "bench/kv_crossover.py). Qwen3-4B's max_position_embeddings is 40960, so the model would accept more, but a "
    "decision above 32768 would rest on an unmeasured cost."
)
BIAS_NOTE = (
    "Tasks whose re-tokenized context reaches the limit are excluded, which removes the longest sessions and biases "
    "the workload towards shorter contexts. Read exclusion_rate and exclusion_counts alongside any result."
)


@dataclass(frozen=True)
class PrepareConfig:
    """Defaults for `prepare`. The minimax submission is open-weight and mostly fits the limit."""

    submission: str = "20260217_mini-v2.0.0_minimax-2-5-high"
    tasks: int = 30
    reserves: int = 10
    seed: int = 0
    max_context: int = 32768
    tokenizer_repo: str = "Qwen/Qwen3-4B"
    tokenizer_revision: str = "1cfa9a7208912126459214e8b04321603b3df60c"


CONFIG = PrepareConfig()
IMAGE_PATTERN = re.compile(r"(?:docker\.io/)?swebench/sweb\.eval\.x86_64\.[A-Za-z0-9_.-]+(?::[A-Za-z0-9_.-]+)?")


def _tls_context() -> ssl.SSLContext:
    # python.org builds on macOS ship without root certificates; certifi comes with httpx.
    try:
        import certifi
    except ImportError:
        return ssl.create_default_context()
    return ssl.create_default_context(cafile=certifi.where())


def _get(url: str, attempts: int = 4, timeout: float = 120.0) -> bytes:
    """Fetch with bounded retries. A network failure must stop prepare, not exclude a task."""
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(url, timeout=timeout, context=_tls_context()) as response:
                return response.read()
        except (urllib.error.URLError, TimeoutError, ConnectionError) as error:
            if attempt == attempts - 1:
                raise RuntimeError(f"cannot fetch {url}: {error}") from error
            time.sleep(2**attempt)
    raise AssertionError("unreachable")


def list_keys(submission: str, get: Callable[[str], bytes] = _get) -> dict[str, str]:
    """Instance id to S3 key for every trajectory in a submission."""
    prefix = f"bash-only/{submission}/trajs/"
    keys: dict[str, str] = {}
    token = None
    while True:
        query = {"list-type": "2", "prefix": prefix, "max-keys": "1000"}
        if token:
            query["continuation-token"] = token
        body = get(f"{BUCKET}/?{urllib.parse.urlencode(query)}").decode()
        for raw in re.findall(r"<Key>([^<]+)</Key>", body):
            key = html.unescape(raw)
            match = re.fullmatch(re.escape(prefix) + r"([A-Za-z0-9_.-]+)/\1\.traj\.json", key)
            if match:
                keys[match.group(1)] = key
        found = re.search(r"<NextContinuationToken>([^<]+)</NextContinuationToken>", body)
        if not found:
            return keys
        token = html.unescape(found.group(1))


def fetch_raw(key: str, raw_dir: Path, get: Callable[[str], bytes] = _get) -> bytes:
    """A trajectory's bytes, cached in the gitignored traces/ folder."""
    path = raw_dir / Path(key).name
    if path.exists():
        return path.read_bytes()
    data = get(f"{BUCKET}/{urllib.parse.quote(key)}")
    raw_dir.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(".partial")
    partial.write_bytes(data)
    partial.replace(path)
    return data


class Ineligible(Exception):
    """A trajectory fails a selection rule. The message is the manifest's reason."""


@dataclass
class Action:
    command: str
    tool_call_id: str | None
    returncode: int | None = None
    output: str | None = None
    exception_info: str = ""


@dataclass
class Turn:
    message: dict
    actions: list[Action]
    following: list[dict]
    prompt_tokens: int | None
    completion_tokens: int | None
    reasoning_tokens: int
    recorded_gap_s: float | None


@dataclass
class Trajectory:
    instance_id: str
    trajectory_format: str
    mini_version: str | None
    exit_status: str | None
    submission_text: str | None
    model: str | None
    environment: dict | None
    # System and user messages before the first assistant turn. Usually two; a format error
    # on the first response adds a user message, since the failed response is not kept.
    opening: list[dict]
    turns: list[Turn] = field(default_factory=list)


def _usage(extra: dict) -> tuple[int | None, int | None, int]:
    usage = (extra.get("response") or {}).get("usage") or {}
    details = usage.get("completion_tokens_details") or {}
    return usage.get("prompt_tokens"), usage.get("completion_tokens"), details.get("reasoning_tokens") or 0


def parse(raw: dict) -> Trajectory:
    """Pair each turn's actions with their tool messages, by id and in order."""
    trajectory_format = raw.get("trajectory_format")
    if trajectory_format != "mini-swe-agent-1.1":
        raise Ineligible(f"unsupported_format:{trajectory_format}")
    messages = [m for m in raw.get("messages") or [] if m.get("role") != "exit"]
    if len(messages) < 3 or messages[0].get("role") != "system" or messages[1].get("role") != "user":
        raise Ineligible("unexpected_message_order")
    info = raw.get("info") or {}
    config = info.get("config") or {}
    trajectory = Trajectory(
        raw.get("instance_id", ""), trajectory_format, info.get("mini_version"), info.get("exit_status"), info.get("submission"),
        (config.get("model") or {}).get("model_name"), config.get("environment"), messages[:2],
    )
    for message in messages[2:]:
        if message.get("role") != "assistant":
            if trajectory.turns:
                trajectory.turns[-1].following.append(message)
            elif message.get("role") == "user":
                trajectory.opening.append(message)
            else:
                raise Ineligible("unexpected_message_order")
            continue
        extra = message.get("extra") or {}
        actions = [Action(a.get("command", ""), a.get("tool_call_id")) for a in extra.get("actions") or []]
        trajectory.turns.append(Turn(message, actions, [], *_usage(extra), None))
    if not trajectory.turns:
        raise Ineligible("no_assistant_turns")
    for turn in trajectory.turns:
        _attach_results(turn)
    return trajectory


def _attach_results(turn: Turn) -> None:
    tools = [m for m in turn.following if m.get("role") == "tool"]
    waiting = list(turn.actions)
    for tool in tools:
        extra = tool.get("extra") or {}
        # Ids repeat in some providers' output, so the first unfilled match wins.
        action = next((a for a in waiting if a.tool_call_id == tool.get("tool_call_id")), None)
        if action is None:
            raise Ineligible("tool_message_without_action")
        waiting.remove(action)
        action.returncode = extra.get("returncode")
        action.output = extra.get("raw_output")
        action.exception_info = extra.get("exception_info") or ""
    stamps = [(m.get("extra") or {}).get("timestamp") for m in tools]
    started = (turn.message.get("extra") or {}).get("timestamp")
    if tools and started is not None and all(s is not None for s in stamps):
        turn.recorded_gap_s = max(max(stamps) - started, 0.0)


def executed_actions(trajectory: Trajectory) -> list[list[Action]]:
    """Per turn, the actions the agent ran: it stops at the first submit in the last turn."""
    per_turn = [list(t.actions) for t in trajectory.turns]
    last = per_turn[-1]
    for index, action in enumerate(last):
        if SUBMIT_MARK in action.command and action.returncode is None:
            del last[index + 1 :]
            break
    return per_turn


def _template_message(message: dict) -> dict:
    """Parsed arguments render as Qwen3 emits them; reasoning goes where its template reads it."""
    calls = []
    for call in message.get("tool_calls") or []:
        function = dict(call.get("function") or {})
        try:
            function["arguments"] = json.loads(function.get("arguments") or "{}")
        except (TypeError, ValueError):
            pass  # Malformed output stays as the string the model wrote.
        calls.append({"type": "function", "function": function})
    rendered = {"role": "assistant", "content": message.get("content") or "", "tool_calls": calls}
    rendered["reasoning_content"] = message.get("reasoning") or ""
    return rendered


def _following_message(message: dict) -> dict:
    return {"role": message["role"], "content": message.get("content") or ""}


_ANCHOR = {"role": "user", "content": "q"}
_SENTINEL = "\x00layover\x00"


class Accounting:
    """Split a trajectory's tokens into the pieces the replayer appends."""

    def __init__(self, tokenizer) -> None:
        self.tokenizer = tokenizer
        self._head = self._render([_ANCHOR], True)
        probe = {"role": "tool", "content": _SENTINEL}
        filler = {"role": "assistant", "content": "a"}
        rendered = self._render([_ANCHOR, filler, probe], False)
        self._tail = rendered[rendered.index("<|im_end|>", len(self._head)) + len("<|im_end|>") :]

    def _render(self, messages: Sequence[dict], generation_prompt: bool, tools: list[dict] | None = None) -> str:
        return self.tokenizer.apply_chat_template(
            list(messages), tools=tools, tokenize=False, add_generation_prompt=generation_prompt
        )

    def count(self, text: str) -> int:
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def initial_text(self, trajectory: Trajectory) -> str:
        return self._render([_following_message(m) for m in trajectory.opening], True, [BASH_TOOL])

    def system_tokens(self, trajectory: Trajectory) -> int:
        return self.count(self._render([_following_message(trajectory.opening[0])], False, [BASH_TOOL]))

    def output_text(self, turn: Turn) -> str:
        """Rendered between a user anchor and a tool response, so the template keeps reasoning."""
        probe = {"role": "tool", "content": _SENTINEL}
        rendered = self._render([_ANCHOR, _template_message(turn.message), probe], False)
        if not rendered.startswith(self._head) or not rendered.endswith(self._tail):
            raise ValueError("the chat template no longer frames messages as this accounting assumes")
        return rendered[len(self._head) : -len(self._tail)]

    def following_text(self, turn: Turn) -> str:
        """What follows the turn's <|im_end|>. Rendered apart from the assistant message,
        because a later user message would make the template drop that message's reasoning."""
        if not turn.following:
            return self._head[self._head.index("<|im_end|>") + len("<|im_end|>") :]
        marker = {"role": "assistant", "content": _SENTINEL}
        following = [_following_message(m) for m in turn.following]
        rendered = self._render([_ANCHOR, marker, *following], True)
        prefix = self._head + _SENTINEL + "<|im_end|>"
        if not rendered.startswith(prefix):
            raise ValueError("the chat template no longer frames messages as this accounting assumes")
        return rendered[len(prefix) :]

    def full_text(self, trajectory: Trajectory) -> str:
        """The conversation in one render, closed with a tool message so no turn is `last`."""
        messages = [_following_message(m) for m in trajectory.opening]
        for turn in trajectory.turns:
            messages.append(_template_message(turn.message))
            messages += [_following_message(m) for m in turn.following]
        messages.append({"role": "tool", "content": _SENTINEL})
        return self._render(messages, True, [BASH_TOOL])


@dataclass
class Counts:
    initial_tokens: int
    output_tokens: list[int]
    new_tokens: list[int]  # new_tokens[i] is appended before turn i; [0] is the initial prompt.
    peak_context: int
    render_checked: bool
    drift_tokens: int


def count_tokens(trajectory: Trajectory, accounting: Accounting) -> Counts:
    """Monotonic counts, checked against one render of the whole conversation.

    A mid-conversation user message (a format error) moves the template's last-query mark
    and strips earlier reasoning from a single render, which an append-only context does
    not do, so only conversations without one can be compared.
    """
    initial = accounting.initial_text(trajectory)
    outputs = [accounting.output_text(t) for t in trajectory.turns]
    following = [accounting.following_text(t) for t in trajectory.turns]
    output_tokens = [accounting.count(t) for t in outputs]
    new_tokens = [accounting.count(initial)] + [accounting.count(t) for t in following[:-1]]
    peak = sum(new_tokens) + sum(output_tokens)
    interrupted = any(m.get("role") == "user" for t in trajectory.turns for m in t.following)
    checked = not interrupted
    drift = 0
    if checked:
        probe = accounting.following_text(_probe_turn(trajectory.turns[-1]))
        pieces = initial + "".join(o + f for o, f in zip(outputs[:-1], following[:-1])) + outputs[-1] + probe
        full = accounting.full_text(trajectory)
        if pieces != full:
            raise ValueError(f"{trajectory.instance_id}: split render differs from the whole render")
        drift = peak + accounting.count(probe) - accounting.count(full)
    return Counts(new_tokens[0], output_tokens, new_tokens, peak, checked, drift)


def _probe_turn(turn: Turn) -> Turn:
    return Turn(turn.message, [], [{"role": "tool", "content": _SENTINEL}], None, None, 0, None)


def check_eligibility(trajectory: Trajectory) -> None:
    """Rules that need no tokenizer."""
    if trajectory.exit_status != "Submitted":
        raise Ineligible(f"exit_status:{trajectory.exit_status}")
    if any(t.prompt_tokens is None or t.completion_tokens is None for t in trajectory.turns):
        raise Ineligible("missing_usage")
    environment = trajectory.environment
    if not isinstance(environment, dict) or not environment.get("image"):
        raise Ineligible("no_environment_config")
    if not IMAGE_PATTERN.fullmatch(str(environment["image"])):
        raise Ineligible("image_not_x86_64_sweb_eval")
    # The executor takes only these from a recording; anything unusual is left out, not run.
    if environment.get("interpreter") != ["bash", "-c"]:
        raise Ineligible("unexpected_interpreter")
    cwd = environment.get("cwd")
    if not isinstance(cwd, str) or not cwd.startswith("/"):
        raise Ineligible("no_environment_config")
    if not isinstance(environment.get("timeout"), (int, float)) or environment["timeout"] <= 0:
        raise Ineligible("no_environment_config")
    if not isinstance(environment.get("env") or {}, dict):
        raise Ineligible("no_environment_config")
    if not any(a.command for t in trajectory.turns for a in t.actions):
        raise Ineligible("no_commands")


def common_prefix_length(sequences: Sequence[Sequence]) -> int:
    length = 0
    for items in zip(*sequences):
        if any(item != items[0] for item in items[1:]):
            break
        length += 1
    return length


def _sha256_text(text: str | None) -> str | None:
    return None if text is None else hashlib.sha256(text.encode("utf-8", "surrogatepass")).hexdigest()


def prepared_document(trajectory: Trajectory, counts: Counts, submission: str, key: str, raw_sha256: str) -> dict:
    environment = trajectory.environment or {}
    turns = []
    context = 0
    for index, (turn, actions) in enumerate(zip(trajectory.turns, executed_actions(trajectory))):
        context += counts.new_tokens[index]
        turns.append({
            "turn": index,
            "new_tokens": counts.new_tokens[index],
            "output_tokens": counts.output_tokens[index],
            "context_before": context,
            "provider_prompt_tokens": turn.prompt_tokens,
            "provider_completion_tokens": turn.completion_tokens,
            "provider_reasoning_tokens": turn.reasoning_tokens,
            "recorded_gap_s": turn.recorded_gap_s,
            "commands": [
                {
                    "command": a.command,
                    "recorded_returncode": a.returncode,
                    "recorded_exception": bool(a.exception_info),
                    "recorded_output_sha256": _sha256_text(a.output),
                    "recorded_output": a.output,
                }
                for a in actions
            ],
        })
        context += counts.output_tokens[index]
    return {
        "schema": PREPARED_SCHEMA,
        "instance_id": trajectory.instance_id,
        "submission": submission,
        "s3_key": key,
        "raw_sha256": raw_sha256,
        "mini_version": trajectory.mini_version,
        "model": trajectory.model,
        "trajectory_format": trajectory.trajectory_format,
        "image": str(environment["image"]).removeprefix("docker.io/"),
        "environment": {k: environment.get(k) for k in ("cwd", "env", "timeout", "interpreter")},
        "initial_prompt_tokens": counts.initial_tokens,
        "peak_context_tokens": counts.peak_context,
        "render_checked": counts.render_checked,
        "drift_tokens": counts.drift_tokens,
        "submission_sha256": _sha256_text(trajectory.submission_text),
        "turns": turns,
    }


def reasoning_evidence(pairs: Sequence[dict]) -> dict:
    """Which hypothesis explains provider prompt growth between consecutive turns.

    Each pair has growth (next prompt minus this prompt), completion, reasoning (both the
    provider's) and following (Qwen3 count of what the client appended).
    """
    def residual(pair: dict, resent: bool) -> float:
        kept = pair["completion"] if resent else pair["completion"] - pair["reasoning"]
        return abs(pair["growth"] - (kept + pair["following"]))

    large = [p for p in pairs if p["reasoning"] > 100]
    if not large:
        return {"pairs": len(pairs), "pairs_with_reasoning_over_100": 0, "conclusion": "no_evidence"}
    closer = sum(residual(p, True) < residual(p, False) for p in large) / len(large)
    resent = statistics.median(residual(p, True) for p in large)
    dropped = statistics.median(residual(p, False) for p in large)
    return {
        "pairs": len(pairs),
        "pairs_with_reasoning_over_100": len(large),
        "median_abs_residual_if_resent": resent,
        "median_abs_residual_if_not_resent": dropped,
        "fraction_closer_to_resent": closer,
        "mean_provider_reasoning_tokens": statistics.mean(p["reasoning"] for p in large),
        "conclusion": "resent" if resent < dropped else "not_resent",
    }


def _summary(values: Sequence[float]) -> dict:
    ordered = sorted(values)
    if not ordered:
        return {"n": 0}
    pick = lambda q: ordered[min(int(q * len(ordered)), len(ordered) - 1)]  # noqa: E731
    return {
        "n": len(ordered), "min": ordered[0], "p5": pick(0.05), "median": statistics.median(ordered), "mean": statistics.mean(ordered),
        "p95": pick(0.95), "max": ordered[-1],
    }


def _describe(documents: Sequence[dict]) -> dict:
    prompt_ratio, output_ratio = [], []
    for document in documents:
        for turn in document["turns"]:
            prompt_ratio.append(turn["context_before"] / max(turn["provider_prompt_tokens"], 1))
            output_ratio.append(turn["output_tokens"] / max(turn["provider_completion_tokens"], 1))
    tool_time = [sum(t["recorded_gap_s"] or 0.0 for t in d["turns"]) for d in documents]
    return {
        "turns": _summary([len(d["turns"]) for d in documents]),
        "commands": _summary([sum(len(t["commands"]) for t in d["turns"]) for d in documents]),
        "peak_context_tokens": _summary([d["peak_context_tokens"] for d in documents]),
        "recorded_tool_s_per_task": _summary(tool_time),
        "qwen3_over_provider_prompt_per_turn": _summary(prompt_ratio),
        "qwen3_over_provider_output_per_turn": _summary(output_ratio),
    }


def load_tokenizer(repo: str, revision: str):
    """Tokenizer files only, at a pinned revision. Weights are never fetched."""
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer

    path = snapshot_download(repo, revision=revision, allow_patterns=["tokenizer*", "vocab.json", "merges.txt"])
    return AutoTokenizer.from_pretrained(path)


def prepare(
    submission: str,
    out_dir: Path,
    raw_dir: Path,
    tokenizer,
    tasks: int = CONFIG.tasks,
    reserves: int = CONFIG.reserves,
    seed: int = CONFIG.seed,
    max_context: int = CONFIG.max_context,
    tokenizer_repo: str = CONFIG.tokenizer_repo,
    tokenizer_revision: str = CONFIG.tokenizer_revision,
    instances: Sequence[str] | None = None,
    get: Callable[[str], bytes] = _get,
) -> dict:
    """Walk a seeded shuffle of the submission until enough tasks qualify; write the manifest."""
    keys = list_keys(submission, get)
    if not keys:
        raise RuntimeError(f"submission {submission!r} lists no trajectories")
    if instances:
        missing = [i for i in instances if i not in keys]
        if missing:
            raise RuntimeError(f"not in submission {submission}: {missing}")
        order, tasks, reserves = list(instances), len(instances), 0
    else:
        # Sorted first: the shuffle must not depend on the listing's order.
        order = sorted(keys)
        random.Random(seed).shuffle(order)
    accounting = Accounting(tokenizer)
    wanted = tasks + reserves
    chosen: list[dict] = []
    exclusions: list[dict] = []
    raw_files: dict[str, dict] = {}
    pairs: list[dict] = []
    prompts: list[list[int]] = []
    walked = 0
    for instance in order:
        if len(chosen) == wanted:
            break
        walked += 1
        key = keys[instance]
        data = fetch_raw(key, raw_dir, get)
        digest = hashlib.sha256(data).hexdigest()
        try:
            try:
                trajectory = parse(json.loads(data))
            except (ValueError, TypeError, KeyError, AttributeError) as error:
                raise Ineligible(f"unparseable:{type(error).__name__}") from error
            if trajectory.instance_id != instance:
                raise Ineligible("instance_id_mismatch")
            check_eligibility(trajectory)
            counts = count_tokens(trajectory, accounting)
            if counts.peak_context >= max_context:
                raise Ineligible("context_over_limit")
        except Ineligible as reason:
            exclusions.append({"instance_id": instance, "s3_key": key, "sha256": digest, "reason": str(reason)})
            continue
        chosen.append(prepared_document(trajectory, counts, submission, key, digest))
        prompts.append(tokenizer.encode(accounting.initial_text(trajectory), add_special_tokens=False))
        raw_files[instance] = {"s3_key": key, "sha256": digest, "bytes": len(data)}
        for index, (turn, later) in enumerate(zip(trajectory.turns, trajectory.turns[1:])):
            pairs.append({
                "growth": later.prompt_tokens - turn.prompt_tokens,
                "completion": turn.completion_tokens,
                "reasoning": turn.reasoning_tokens,
                "following": counts.new_tokens[index + 1],
            })
    if len(chosen) < wanted:
        print(f"WARNING: only {len(chosen)} of {wanted} tasks qualify in {submission}", file=sys.stderr)

    evidence = reasoning_evidence(pairs)
    if evidence["conclusion"] == "not_resent":
        raise RuntimeError(f"this submission did not re-send reasoning, which the accounting assumes: {evidence}")

    # Reserves count too, so no replacement can start from a different shared prefix. With
    # one prompt there is nothing to compare, so only the system section counts as shared.
    if len(prompts) > 1:
        shared_prefix = common_prefix_length(prompts)
    else:
        shared_prefix = accounting.system_tokens(trajectory) if prompts else 0

    out_dir.mkdir(parents=True, exist_ok=True)
    for document in chosen:
        (out_dir / f"{document['instance_id']}.json").write_text(json.dumps(document) + "\n")
    ids = [d["instance_id"] for d in chosen]
    reasons = Counter(e["reason"].split(":")[0] for e in exclusions)
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "submission": submission,
        "bucket": BUCKET,
        "selection": "named" if instances else "seeded",
        "seed": seed,
        "listed": len(keys),
        "walked": walked,
        "mini_versions": sorted({str(d["mini_version"]) for d in chosen}),
        "models": sorted({str(d["model"]) for d in chosen}),
        "trajectory_format": "mini-swe-agent-1.1",
        "tokenizer_repo": tokenizer_repo,
        "tokenizer_revision": tokenizer_revision,
        "max_context": max_context,
        "max_context_rationale": MAX_CONTEXT_RATIONALE,
        "tasks": ids[:tasks],
        "reserves": ids[tasks:],
        "raw": raw_files,
        "exclusions": exclusions,
        "exclusion_rate": len(exclusions) / max(walked, 1),
        "exclusion_counts": dict(reasons),
        "bias_note": BIAS_NOTE,
        "shared_prefix_tokens": shared_prefix,
        "reasoning_evidence": evidence,
        "max_abs_drift_tokens": max((abs(d["drift_tokens"]) for d in chosen), default=0),
        "tasks_with_render_check": sum(d["render_checked"] for d in chosen),
        "describe_tasks": _describe(chosen[:tasks]),
        "describe_all_chosen": _describe(chosen),
        "licence_note": "Terms of the source trajectories are unknown. Keep raw and prepared files out of the repository.",
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    return manifest


def load_manifest(directory: Path) -> dict:
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest.get("schema") != MANIFEST_SCHEMA:
        raise ValueError(f"{directory}: not a {MANIFEST_SCHEMA} manifest")
    return manifest


def load_prepared(directory: Path, instance: str) -> dict:
    document = json.loads((directory / f"{instance}.json").read_text())
    if document.get("schema") != PREPARED_SCHEMA:
        raise ValueError(f"{instance}: not a {PREPARED_SCHEMA} file")
    return document


def _hub_size(image: str, get: Callable[[str], bytes]) -> int:
    """Compressed amd64 size from Docker Hub's public API. Nothing is pulled."""
    name, _, tag = image.partition(":")
    data = json.loads(get(f"https://hub.docker.com/v2/repositories/{name}/tags/{tag or 'latest'}"))
    sizes = [i["size"] for i in data.get("images", []) if i.get("architecture") == "amd64"]
    if not sizes:
        raise RuntimeError(f"{image}: Docker Hub lists no amd64 image")
    return sizes[0]


def estimate(directory: Path, get: Callable[[str], bytes] = _get) -> dict:
    """Disk and time for `reexecute run`, from recorded gaps and registry sizes.

    Recorded gaps were measured on the submitter's machine, so the time is an estimate and
    nothing about it goes into a trace. Pull time is left to the reader: it depends on a
    bandwidth nothing here measures.
    """
    manifest = load_manifest(directory)
    rows = []
    for instance in [*manifest["tasks"], *manifest["reserves"]]:
        document = load_prepared(directory, instance)
        rows.append({
            "instance_id": instance,
            "image": document["image"],
            "compressed_bytes": _hub_size(document["image"], get),
            "recorded_tool_s": sum(t["recorded_gap_s"] or 0.0 for t in document["turns"]),
            "commands": sum(len(t["commands"]) for t in document["turns"]),
            "timeout_s": document["environment"]["timeout"],
        })
    selected = rows[: len(manifest["tasks"])]
    return {
        "tasks": selected,
        "reserves": rows[len(manifest["tasks"]) :],
        "compressed_gb_tasks": sum(r["compressed_bytes"] for r in selected) / 1e9,
        "compressed_gb_all": sum(r["compressed_bytes"] for r in rows) / 1e9,
        "largest_compressed_gb": max((r["compressed_bytes"] for r in rows), default=0) / 1e9,
        "recorded_tool_hours_tasks": sum(r["recorded_tool_s"] for r in selected) / 3600,
        "commands_tasks": sum(r["commands"] for r in selected),
    }


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(prog="python -m layover.replay.trajectories", description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare", help="choose tasks and count tokens; free, and runs nothing from a trajectory")
    prep.add_argument("--submission", default=CONFIG.submission)
    prep.add_argument("--tasks", type=int, default=CONFIG.tasks)
    prep.add_argument("--reserves", type=int, default=CONFIG.reserves)
    prep.add_argument("--seed", type=int, default=CONFIG.seed)
    prep.add_argument("--max-context", type=int, default=CONFIG.max_context, help=MAX_CONTEXT_RATIONALE)
    prep.add_argument("--tokenizer-repo", default=CONFIG.tokenizer_repo)
    prep.add_argument("--tokenizer-revision", default=CONFIG.tokenizer_revision)
    prep.add_argument("--raw-dir", type=Path, help="default traces/raw/<submission>")
    prep.add_argument("--out", type=Path, default=Path("traces/prepared"))
    prep.add_argument("--instance", action="append", help="use these tasks only; a manifest made this way is never a result")
    size = commands.add_parser("estimate", help="disk and time for the re-execution, from Docker Hub sizes; pulls nothing")
    size.add_argument("--prepared", type=Path, default=Path("traces/prepared"))
    size.add_argument("--out", type=Path, help="default <prepared>/estimate.json")
    args = parser.parse_args(argv)

    if args.command == "estimate":
        result = estimate(args.prepared)
        (args.out or args.prepared / "estimate.json").write_text(json.dumps(result, indent=1) + "\n")
        print(json.dumps({k: v for k, v in result.items() if k not in ("tasks", "reserves")}, indent=1))
        return
    tokenizer = load_tokenizer(args.tokenizer_repo, args.tokenizer_revision)
    raw_dir = args.raw_dir or Path("traces/raw") / args.submission
    manifest = prepare(
        args.submission, args.out, raw_dir, tokenizer, args.tasks, args.reserves, args.seed, args.max_context,
        args.tokenizer_repo, args.tokenizer_revision, args.instance,
    )
    shown = ("listed", "walked", "exclusion_rate", "exclusion_counts", "shared_prefix_tokens", "reasoning_evidence",
             "max_abs_drift_tokens", "tasks_with_render_check", "describe_tasks")
    print(json.dumps({k: manifest[k] for k in shown}, indent=1))
    print(f"{len(manifest['tasks'])} tasks and {len(manifest['reserves'])} reserves written to {args.out}")
    if len(manifest["tasks"]) + len(manifest["reserves"]) < args.tasks + args.reserves and not args.instance:
        raise SystemExit(f"fewer than {args.tasks + args.reserves} tasks qualify")


if __name__ == "__main__":
    main()
