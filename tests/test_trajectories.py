"""Choosing public SWE-bench trajectories and counting their tokens with Qwen3's tokenizer.

The fixtures are hand-made and shaped like mini-swe-agent-1.1 files, because the licence of
the real ones is not stated. Qwen3-0.6B stands in for Qwen3-4B: their tokenizers are the same.
"""

import hashlib
import json
import urllib.parse

import pytest
from transformers import AutoTokenizer

from layover.replay import trajectories as tj
from layover.replay.trajectories import (
    Accounting, BASH_TOOL, Ineligible, check_eligibility, common_prefix_length, count_tokens, executed_actions,
    list_keys, parse, prepare, reasoning_evidence,
)

SUBMISSION = "sub_mini"
TASK = "Fix the crash when the list is empty."


@pytest.fixture(scope="module")
def tokenizer():
    return AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")


@pytest.fixture(scope="module")
def accounting(tokenizer):
    return Accounting(tokenizer)


def assistant(commands, reasoning="Think about it.", content="", stamp=100.0, prompt=500, completion=40, reasoning_tokens=5, ids=None):
    ids = ids or [f"call_{i}_{stamp}" for i in range(len(commands))]
    return {
        "role": "assistant",
        "content": content,
        "reasoning": reasoning,
        "tool_calls": [
            {"type": "function", "index": i, "id": call, "function": {"name": "bash", "arguments": json.dumps({"command": c})}}
            for i, (c, call) in enumerate(zip(commands, ids))
        ],
        "extra": {
            "actions": [{"command": c, "tool_call_id": call} for c, call in zip(commands, ids)],
            "response": {"provider": "P", "usage": {
                "prompt_tokens": prompt, "completion_tokens": completion, "completion_tokens_details": {"reasoning_tokens": reasoning_tokens},
            }},
            "timestamp": stamp,
        },
    }


def tool(call, output, returncode=0, stamp=101.0, exception=""):
    observation = f"<returncode>{returncode}</returncode>\n<output>\n{output}</output>"
    return {
        "role": "tool", "content": observation, "tool_call_id": call,
        "extra": {"raw_output": output, "returncode": returncode, "timestamp": stamp, "exception_info": exception},
    }


def trajectory(instance="r__r-1", task=TASK, exit_status="Submitted", image=None, **environment):
    """Three turns: two actions in one, a timeout in the second, and the submit in the last."""
    image = image or f"docker.io/swebench/sweb.eval.x86_64.{instance.replace('__', '_1776_')}:latest"
    first = assistant(["ls /testbed", "cat a.py"], stamp=100.0, ids=["a0", "a1"], prompt=500, completion=40)
    second = assistant(["python slow.py"], content="Now run it.", stamp=110.0, ids=["b0"], prompt=700, completion=30)
    third = assistant([f"echo {tj.SUBMIT_MARK} && git diff"], stamp=180.0, ids=["c0"], prompt=800, completion=20)
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": f"<pr_description>\n{task}\n</pr_description>\n<instructions>Work in /testbed.</instructions>"},
        first, tool("a0", "a.py\n", stamp=101.0), tool("a1", "print(1)\n", stamp=103.5),
        second, tool("b0", "", returncode=-1, stamp=170.0, exception="timed out"),
        third,
        {"role": "exit", "content": "the patch\n", "extra": {"exit_status": exit_status, "submission": "the patch\n"}},
    ]
    return {
        "trajectory_format": "mini-swe-agent-1.1",
        "instance_id": instance,
        "messages": messages,
        "info": {
            "exit_status": exit_status, "submission": "the patch\n", "mini_version": "2.1.0",
            "config": {
                "model": {"model_name": "open/model"},
                "environment": {
                    "image": image, "cwd": "/testbed", "env": {"PAGER": "cat"}, "timeout": 60, "interpreter": ["bash", "-c"],
                    "run_args": ["--rm", "-v", "/:/host"], "forward_env": ["SECRET"], **environment,
                },
            },
        },
    }


def test_several_actions_in_one_turn_pair_with_their_results():
    parsed = parse(trajectory())
    first, second, third = parsed.turns
    assert [a.command for a in first.actions] == ["ls /testbed", "cat a.py"]
    assert [(a.returncode, a.output) for a in first.actions] == [(0, "a.py\n"), (0, "print(1)\n")]
    assert first.recorded_gap_s == pytest.approx(3.5)  # The last result ends the pause.
    assert second.actions[0].returncode == -1 and second.actions[0].exception_info
    assert third.actions[0].returncode is None and third.recorded_gap_s is None
    assert parsed.opening[1]["role"] == "user" and parsed.exit_status == "Submitted" and parsed.model == "open/model"


def test_a_format_error_before_the_first_turn_joins_the_opening():
    raw = trajectory()
    raw["messages"].insert(2, {"role": "user", "content": "Tool call error", "extra": {"interrupt_type": "FormatError"}})
    parsed = parse(raw)
    assert len(parsed.opening) == 3 and len(parsed.turns) == 3


def test_actions_after_the_submit_never_ran():
    raw = trajectory()
    raw["messages"][7] = assistant([f"echo {tj.SUBMIT_MARK}", "rm -rf /"], stamp=180.0, ids=["c0", "c1"])
    assert [a.command for a in executed_actions(parse(raw))[-1]] == [f"echo {tj.SUBMIT_MARK}"]


def test_an_unknown_tool_call_id_is_an_error():
    raw = trajectory()
    raw["messages"][3]["tool_call_id"] = "nope"
    with pytest.raises(Ineligible, match="tool_message_without_action"):
        parse(raw)


def template_ids(tokenizer, raw):
    """One independent render of the conversation through the real template."""
    messages = []
    for m in raw["messages"]:
        if m["role"] == "assistant":
            calls = [{"type": "function", "function": {"name": "bash", "arguments": json.loads(c["function"]["arguments"])}} for c in m["tool_calls"]]
            messages.append({"role": "assistant", "content": m["content"], "reasoning_content": m["reasoning"], "tool_calls": calls})
        elif m["role"] != "exit":
            messages.append({"role": m["role"], "content": m["content"]})
    # A closing tool message keeps the last assistant from being `last`, which changes its framing.
    messages.append({"role": "tool", "content": "x"})
    rendered = tokenizer.apply_chat_template(messages, tools=[BASH_TOOL], tokenize=False, add_generation_prompt=True)
    closing = "\n<|im_start|>user\n<tool_response>\nx\n</tool_response><|im_end|>\n<|im_start|>assistant\n"
    assert rendered.endswith(closing)
    return tokenizer.encode(rendered[: -len(closing)], add_special_tokens=False)


@pytest.mark.weights
def test_counts_add_up_to_one_render_of_the_real_template(tokenizer, accounting):
    raw = trajectory()
    counts = count_tokens(parse(raw), accounting)
    assert counts.peak_context == len(template_ids(tokenizer, raw))
    assert sum(counts.new_tokens) + sum(counts.output_tokens) == counts.peak_context
    assert counts.render_checked and counts.drift_tokens == 0
    assert len(counts.new_tokens) == len(counts.output_tokens) == 3 and min(counts.output_tokens) > 0


@pytest.mark.weights
def test_each_piece_is_framed_as_the_replayer_appends_it(accounting):
    parsed = parse(trajectory())
    first, second, third = parsed.turns
    initial = accounting.initial_text(parsed)
    assert initial.startswith("<|im_start|>system\n") and initial.endswith("<|im_start|>assistant\n")
    assert '"name": "bash"' in initial and TASK in initial
    output = accounting.output_text(first)
    assert output.startswith("<think>\nThink about it.\n</think>\n\n") and output.endswith("</tool_call><|im_end|>")
    assert output.count("<tool_call>") == 2 and '"arguments": {"command": "ls /testbed"}' in output
    following = accounting.following_text(first)
    assert following.startswith("\n<|im_start|>user\n<tool_response>\n") and following.endswith("<|im_start|>assistant\n")
    assert following.count("<tool_response>") == 2 and following.count("<|im_start|>user") == 1
    assert accounting.following_text(third) == "\n<|im_start|>assistant\n"


@pytest.mark.weights
def test_reasoning_stays_in_the_context_when_a_later_user_message_appears(accounting):
    """A single render would drop it; the append-only context does not."""
    raw = trajectory()
    plain = count_tokens(parse(raw), accounting)
    raw["messages"].insert(5, {"role": "user", "content": "Tool call error", "extra": {"interrupt_type": "FormatError"}})
    counted = count_tokens(parse(raw), accounting)
    assert not counted.render_checked
    assert counted.output_tokens == plain.output_tokens
    assert counted.new_tokens[1] > plain.new_tokens[1]


@pytest.mark.weights
def test_context_is_monotonic_and_prepared_files_keep_only_whitelisted_environment(accounting):
    parsed = parse(trajectory())
    counts = count_tokens(parsed, accounting)
    document = tj.prepared_document(parsed, counts, SUBMISSION, "key", "abc")
    contexts = [t["context_before"] for t in document["turns"]]
    assert contexts == sorted(contexts) and contexts[0] == counts.initial_tokens
    assert contexts[-1] + document["turns"][-1]["output_tokens"] == document["peak_context_tokens"]
    assert set(document["environment"]) == {"cwd", "env", "timeout", "interpreter"}
    assert "run_args" not in json.dumps(document) and "/:/host" not in json.dumps(document)
    assert document["image"].startswith("swebench/")
    recorded = document["turns"][0]["commands"][0]
    assert recorded["recorded_output_sha256"] == hashlib.sha256(b"a.py\n").hexdigest()
    assert [c["recorded_returncode"] for c in document["turns"][1]["commands"]] == [-1]
    assert document["turns"][2]["commands"][0]["recorded_returncode"] is None
    assert document["submission_sha256"] == hashlib.sha256(b"the patch\n").hexdigest()


@pytest.mark.parametrize(
    "change, reason",
    [
        (lambda r: r["info"].update(exit_status="LimitsExceeded"), "exit_status:LimitsExceeded"),
        (lambda r: r["messages"][2]["extra"]["response"].update(usage=None), "missing_usage"),
        (lambda r: r["messages"][2]["extra"].pop("response"), "missing_usage"),
        (lambda r: r["info"]["config"].pop("environment"), "no_environment_config"),
        (lambda r: r["info"]["config"]["environment"].pop("cwd"), "no_environment_config"),
        (lambda r: r["info"]["config"]["environment"].update(image="evil/sweb.eval.x86_64.x:latest"), "image_not_x86_64_sweb_eval"),
        (lambda r: r["info"]["config"]["environment"].update(image="swebench/sweb.eval.arm64.x:latest"), "image_not_x86_64_sweb_eval"),
        (lambda r: r["info"]["config"]["environment"].update(interpreter=["sh", "-c"]), "unexpected_interpreter"),
    ],
)
def test_eligibility_rules(change, reason):
    raw = trajectory()
    check_eligibility(parse(raw))
    change(raw)
    with pytest.raises(Ineligible, match=reason):
        check_eligibility(parse(raw))


def test_unsupported_format_is_ineligible():
    raw = trajectory()
    raw["trajectory_format"] = "mini-swe-agent-1"
    with pytest.raises(Ineligible, match="unsupported_format"):
        parse(raw)


class Bucket:
    """The public bucket's two endpoints, from fixtures."""

    def __init__(self, files, reverse_listing=False):
        self.files = {f"bash-only/{SUBMISSION}/trajs/{i}/{i}.traj.json": json.dumps(r).encode() for i, r in files.items()}
        self.reverse_listing = reverse_listing
        self.fetched = []

    def __call__(self, url):
        parsed = urllib.parse.urlparse(url)
        if parsed.query:
            keys = sorted(self.files, reverse=self.reverse_listing)
            return "".join(f"<Key>{k}</Key>" for k in keys).encode()
        key = urllib.parse.unquote(parsed.path.lstrip("/"))
        self.fetched.append(key)
        return self.files[key]


def fleet(count=8):
    files = {f"r__r-{i}": trajectory(f"r__r-{i}", task=f"Task number {i} is different.") for i in range(count)}
    files["bad__exit-1"] = trajectory("bad__exit-1", exit_status="LimitsExceeded")
    files["bad__usage-1"] = trajectory("bad__usage-1")
    files["bad__usage-1"]["messages"][5]["extra"]["response"]["usage"] = None
    files["bad__long-1"] = trajectory("bad__long-1", task="word " * 6000)
    return files


def run_prepare(tmp_path, tokenizer, name="a", bucket=None, **kwargs):
    arguments = {"tasks": 4, "reserves": 2, "seed": 3, "max_context": 4096, **kwargs}
    return prepare(SUBMISSION, tmp_path / name / "prepared", tmp_path / name / "raw", tokenizer, get=bucket or Bucket(fleet()), **arguments)


@pytest.mark.weights
def test_selection_is_deterministic_and_ignores_listing_order(tmp_path, tokenizer):
    one = run_prepare(tmp_path, tokenizer, "a")
    two = run_prepare(tmp_path, tokenizer, "b", Bucket(fleet(), reverse_listing=True))
    other = run_prepare(tmp_path, tokenizer, "c", seed=4)
    assert (one["tasks"], one["reserves"]) == (two["tasks"], two["reserves"])
    assert (one["tasks"], one["reserves"]) != (other["tasks"], other["reserves"])
    assert len(one["tasks"]) == 4 and len(one["reserves"]) == 2


@pytest.mark.weights
def test_manifest_records_exclusions_and_provenance(tmp_path, tokenizer):
    files = fleet()
    manifest = run_prepare(tmp_path, tokenizer, bucket=Bucket(files), tasks=8, reserves=0)
    assert manifest["exclusion_counts"] == {"exit_status": 1, "missing_usage": 1, "context_over_limit": 1}
    assert manifest["walked"] == 11 and manifest["exclusion_rate"] == pytest.approx(3 / 11)
    assert {e["instance_id"]: e["reason"] for e in manifest["exclusions"]}["bad__long-1"] == "context_over_limit"
    assert manifest["submission"] == SUBMISSION and manifest["mini_versions"] == ["2.1.0"] and manifest["models"] == ["open/model"]
    assert manifest["tokenizer_repo"] and manifest["tokenizer_revision"] and manifest["seed"] == 3
    assert "biases" in manifest["bias_note"] and "32768" in manifest["max_context_rationale"]
    entry = manifest["raw"][manifest["tasks"][0]]
    assert entry["sha256"] == hashlib.sha256(json.dumps(files[manifest["tasks"][0]]).encode()).hexdigest()
    saved = json.loads((tmp_path / "a" / "prepared" / "manifest.json").read_text())
    assert saved["tasks"] == manifest["tasks"]


@pytest.mark.weights
def test_context_limit_is_exclusive(tmp_path, tokenizer):
    files = {"r__r-1": trajectory("r__r-1"), "r__r-2": trajectory("r__r-2")}
    peak = count_tokens(parse(files["r__r-1"]), Accounting(tokenizer)).peak_context
    fits = run_prepare(tmp_path, tokenizer, "a", Bucket(files), tasks=1, reserves=0, max_context=peak + 1)
    assert fits["exclusions"] == [] and len(fits["tasks"]) == 1
    over = run_prepare(tmp_path, tokenizer, "b", Bucket(files), tasks=1, reserves=0, max_context=peak)
    assert over["tasks"] == [] and over["exclusion_counts"] == {"context_over_limit": 2}


@pytest.mark.weights
def test_shared_prefix_is_a_true_common_token_prefix(tmp_path, tokenizer):
    manifest = run_prepare(tmp_path, tokenizer)
    accounting = Accounting(tokenizer)
    files = fleet()
    prompts = [
        tokenizer.encode(accounting.initial_text(parse(files[i])), add_special_tokens=False)
        for i in [*manifest["tasks"], *manifest["reserves"]]
    ]
    prefix = manifest["shared_prefix_tokens"]
    assert prefix == common_prefix_length(prompts)
    assert all(p[:prefix] == prompts[0][:prefix] for p in prompts)
    assert len({p[prefix] for p in prompts}) > 1  # Task text differs right after it.


@pytest.mark.weights
def test_named_instances_are_marked_and_not_seeded(tmp_path, tokenizer):
    manifest = run_prepare(tmp_path, tokenizer, instances=["r__r-2"])
    assert manifest["tasks"] == ["r__r-2"] and manifest["reserves"] == [] and manifest["selection"] == "named"
    with pytest.raises(RuntimeError, match="not in submission"):
        run_prepare(tmp_path, tokenizer, "b", instances=["missing__task-1"])


@pytest.mark.weights
def test_a_network_failure_stops_prepare_rather_than_excluding(tmp_path, tokenizer):
    bucket = Bucket(fleet())

    def flaky(url):
        if urllib.parse.urlparse(url).query:
            return bucket(url)
        raise RuntimeError("cannot fetch")

    with pytest.raises(RuntimeError, match="cannot fetch"):
        run_prepare(tmp_path, tokenizer, bucket=flaky)


@pytest.mark.weights
def test_prepare_refuses_a_submission_that_did_not_resend_reasoning(tmp_path, tokenizer, monkeypatch):
    monkeypatch.setattr(tj, "reasoning_evidence", lambda pairs: {"conclusion": "not_resent"})
    with pytest.raises(RuntimeError, match="did not re-send"):
        run_prepare(tmp_path, tokenizer)


def test_reasoning_evidence_tells_the_hypotheses_apart():
    resent = [{"growth": 500 + 300, "completion": 500, "reasoning": 400, "following": 300}] * 5
    dropped = [{"growth": 100 + 300, "completion": 500, "reasoning": 400, "following": 300}] * 5
    assert reasoning_evidence(resent)["conclusion"] == "resent"
    assert reasoning_evidence(dropped)["conclusion"] == "not_resent"
    assert reasoning_evidence(dropped)["fraction_closer_to_resent"] == 0
    assert reasoning_evidence([])["conclusion"] == "no_evidence"


def test_listing_follows_continuation_tokens_and_keeps_only_trajectories():
    prefix = f"bash-only/{SUBMISSION}/trajs/"
    pages = [
        f"<Key>{prefix}a__a-1/a__a-1.traj.json</Key><Key>{prefix}a__a-1/other.json</Key>"
        "<NextContinuationToken>t&amp;1</NextContinuationToken>",
        f"<Key>{prefix}b__b-2/b__b-2.traj.json</Key><Key>{prefix}../x/../x.traj.json</Key>",
    ]
    urls = []

    def get(url):
        urls.append(url)
        return pages[len(urls) - 1].encode()

    assert sorted(list_keys(SUBMISSION, get)) == ["a__a-1", "b__b-2"]
    assert "continuation-token=t%261" in urls[1]


def test_tokenizer_fetch_excludes_weights(monkeypatch):
    import huggingface_hub
    import transformers

    seen = {}
    monkeypatch.setattr(huggingface_hub, "snapshot_download", lambda repo, revision, allow_patterns: seen.update(
        repo=repo, revision=revision, patterns=allow_patterns) or "path")
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda path: "tokenizer")
    assert tj.load_tokenizer("Qwen/Qwen3-4B", "abc") == "tokenizer"
    assert seen["revision"] == "abc" and not any("safetensors" in p or p == "*" for p in seen["patterns"])


@pytest.mark.weights
def test_estimate_uses_registry_sizes_and_recorded_gaps(tmp_path, tokenizer):
    bucket = Bucket(fleet())
    run_prepare(tmp_path, tokenizer, bucket=bucket, tasks=2, reserves=1)

    def get(url):
        assert url.startswith("https://hub.docker.com/v2/repositories/swebench/")
        return json.dumps({"images": [{"architecture": "unknown", "size": 1}, {"architecture": "amd64", "size": 2_000_000_000}]}).encode()

    result = tj.estimate(tmp_path / "a" / "prepared", get)
    assert result["compressed_gb_tasks"] == pytest.approx(4.0) and result["compressed_gb_all"] == pytest.approx(6.0)
    # 3.5 s and 60 s from the fixture, twice over.
    assert result["recorded_tool_hours_tasks"] == pytest.approx(2 * 63.5 / 3600)
    assert result["commands_tasks"] == 8
