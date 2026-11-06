# Layover

An inference engine for AI agent workloads. When an agent pauses for a tool call, Layover
decides per call whether to keep its KV cache in GPU memory with a time-to-live, swap it to
CPU memory, or drop it and recompute later, using a cost model measured on the GPU. Parallel
sub-agents share KV blocks through a copy-on-write prefix tree.

## KV size

KV bytes per token = 2 × layers × KV heads × head dimension × bytes per element, with values
read from each model's `config.json`:

| Model | Layers | KV heads | Head dim | BF16 bytes/token | FP8 bytes/token | 32,768 tokens (BF16) |
| --- | --- | --- | --- | --- | --- | --- |
| Qwen3-4B | 36 | 8 | 128 | 147,456 | 73,728 | 4.5 GiB |
| Qwen3-0.6B | 28 | 8 | 128 | 114,688 | 57,344 | 3.5 GiB |

`head_dim` must come from the config: for Qwen3-4B, `hidden_size / num_attention_heads` is 80,
not 128.

## Status

**No benchmark results exist yet.** Nothing below has run on a GPU; GPU code has only been compiled
for sm_89 and exercised through CPU stand-ins.

| Part | State |
| --- | --- |
| Engine core: paged KV, continuous batching, chunked prefill, greedy decoding | Matches `transformers` per layer and in greedy output on Qwen3-0.6B, on CPU |
| Radix prefix tree, copy-on-write fork, scheduler, keep / swap / drop policies | Tested and fuzzed on CPU, on a virtual clock |
| OpenAI-compatible API, trace replayer | Tested on CPU over real sockets |
| Triton paged decode attention (`--decode-backend triton`) | Matches the PyTorch reference in Triton's interpreter on CPU; compiles for sm_89; never run on a GPU |
| CUDA gather/scatter, copy stream, FP8 offload (`--transfer native --offload-dtype fp8`) | Host build and FP8 tested on CPU; CUDA build compiles; never run on a GPU |
| KV crossover, decode bandwidth, transfer benchmarks | Written; pipeline-checked on CPU; no GPU numbers |
| Benchmark matrix against vLLM and SGLang | Written; pipeline-checked on CPU against a stub; neither baseline has answered a real request |

## Running

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e ".[dev]"
make test
```

Tests run on CPU with a very small random-weight model and with real Qwen3-0.6B weights, which
download on first use. GPU-only tests are marked and skip unless `--gpu` is passed. Triton has no
macOS wheels, so `make docker-image test-docker` runs the same suite on Linux, where the kernels
run in Triton's interpreter.

To serve a model on CPU, produce a cost model and start the server:

```bash
.venv/bin/python bench/kv_crossover.py --model Qwen/Qwen3-0.6B --tokens 64 128 256 512 --out results/cpu
```

```bash
.venv/bin/python -m layover.server --model Qwen/Qwen3-0.6B --cost-model results/cpu/cost_model.json
```

To run everything on a rented x86 Linux VM with a 24 GB class NVIDIA GPU and at least 64 GiB of RAM,
from this checkout:

```bash
make traces-prepare
```

```bash
scripts/remote_gpu.sh user@host -i key --plan under-20h --stop-after-smoke
```

That syncs the tree over SSH and runs setup, then a short GPU smoke stage, so a broken setup costs
minutes. With `--stop-after-smoke` it stops there and prints the plan's hours priced from measured
rates. Run the same command without that flag, and with `--max-hours`, to continue through the
baselines, GPU tests, compute-sanitizer, trace re-execution, and `make reproduce`. `results/` and
`traces/` come back during and after the run. It survives dropped connections and resumes, needs a
working Docker daemon (rentals that are themselves containers usually cannot run it), refuses an
instance with an IAM role unless `--allow-iam-role`, and never terminates the instance.

On spot instances add `--spot-notice --reconnect-timeout 600`. Each finished run is written
atomically and fetched within about a minute while this machine stays online and awake. On a
replacement, `--restore-from results/gpu-<timestamp>` pushes what came back and skips what is
finished; the budget carries over, and another GPU type or driver is refused.

`make reproduce` regenerates every number for the chosen plan and refuses to run without a CUDA GPU
and the re-executed trace. A re-run keeps complete measurements (`FORCE=1` re-measures), because a
new cost model would invalidate saved runs.

## Benchmark plans

`bench/run_matrix.py --plan NAME` selects one; `--dry-run` prices it. Loads are fractions of vLLM's
saturation rate, measured once and applied as the same absolute rates to every system.

| Plan | Headline: `layover-combined`, `vllm`, `sglang` | Ablations at 100% load |
| --- | --- | --- |
| `first-pass` (default) | 50%, 100%, 125% load, 5 runs each | 3 runs each |
| `under-20h` | 100% load 5 runs, 125% load 3 runs | 3 runs each |
| `three-loads-lean` | all three loads, 3 runs each | 2 runs each |
| `full` | as `first-pass`, plus synthetic sweeps | 3 runs each |

The ablations are `layover-discard`, `layover-discard-nocache` (a strict lower bound: no prefix cache
at all), `layover-combined-bf16` (isolates the FP8 gain), `layover-keep_ttl`, and `layover-swap`. Each
ratio to `layover-combined` compares the same seeds. `layover-discard` leaves freed blocks reusable
until memory pressure, as vLLM does. Every offloading system gets 32 GiB of CPU memory. Settings,
flags, and remaining differences between systems are in `bench/configs/baseline_flags.md`.

## API

`POST /v1/chat/completions` and `POST /v1/completions` follow the OpenAI API. Prompts may be token
ids. Decoding is greedy, so `temperature` and `top_p` are accepted and have no effect. Extra body
fields, all optional:

```json
{
  "session_id": "agent-1",
  "layover": {
    "program_id": "job-7",
    "tool_name": "run_tests",
    "expected_tool_s": 12.0,
    "end_session": false
  }
}
```

The tool name is also read from the model's own `<tool_call>` output when the client gives none.
`POST /v1/sessions/{id}/fork` starts a sub-agent sharing its parent's KV copy-on-write,
`DELETE /v1/sessions/{id}` frees a session, and `GET /metrics` reports placements and memory.

## Traces

The headline benchmark uses real agent trajectories; synthetic traces are only for sweeps. Every
result carries its trace's `source`, and the headline refuses anything that does not start with
`recorded:`.

**Recorded.** 30 tasks from a public
[SWE-bench](https://github.com/SWE-bench/experiments) leaderboard submission, a
[mini-swe-agent](https://github.com/SWE-agent/mini-swe-agent) run on SWE-bench Verified:

1. `make traces-prepare` picks the tasks with a fixed seed and counts each turn's prompt and output
   text with the Qwen3-4B tokenizer. Tasks whose context exceeds 32,768 tokens are excluded, which
   skews the workload shorter; the manifest states the exclusion rate.
2. `make traces`, on the x86 GPU machine, re-executes each trajectory's commands in order inside
   the task's container, because a submitter's recorded gaps come from an unknown machine. It
   checks each command against the recording and replaces tasks that diverge, logging every
   replacement. A task is valid only if at least 90% of comparable return codes match, its final
   patch matches, and output similarity reaches 0.5. **These are third-party commands with network
   access**, so the container gets no mounts or privileges, the instance should have no IAM role,
   and the host's bridge gateway and metadata service remain reachable from the container. Installs
   resolve unpinned packages at run time, so their durations depend on the network; the share of
   tool time they take is measured and reported with each trace.

The source submission and the machine type travel with every trace. Prompts replayed from a trace
are random token ids matching the real lengths and shared prefix, since a trace holds counts, not
text. Raw trajectories stay in the gitignored `traces/` folder.

**Synthetic.** `python -m layover.replay generate` makes sweeps over tool latency and branching
factor, with a shape following the statistics in the Continuum paper. `make calibration` fits it to
the per-job durations published in Continuum's
[repository](https://github.com/Hanchenli/vllm-continuum). That fit matches one distribution with
three parameters; it does not validate the simulator or measure tool latency. Those durations are
all that is used from Continuum: this project does not have its per-turn traces.

## Prior work

InferCept (Abhyankar et al., ICML 2024) introduced the preserve, swap, or discard choice. Continuum
(Li et al., ICLR 2026) pins KV with a time-to-live. Autellix (Luo et al., 2025) schedules at the
program level. The paged KV layout follows Kwon et al. (SOSP 2023), the prefix tree SGLang's
RadixAttention (Zheng et al., NeurIPS 2024), continuous batching Orca (Yu et al., OSDI 2022), and
chunked prefill Sarathi-Serve (Agrawal et al., OSDI 2024). No code is taken from these systems.
