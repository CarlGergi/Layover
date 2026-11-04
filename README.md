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

| Part | State |
| --- | --- |
| KV crossover benchmark (`bench/kv_crossover.py`) | Written and tested on CPU. **No GPU numbers yet.** |
| Engine core: paged KV, continuous batching, chunked prefill, greedy decoding | Done; matches `transformers` per layer and in greedy output on Qwen3-0.6B |
| Radix prefix tree with copy-on-write fork | Done |
| Scheduler: program priority, aging, service cap, keep / swap / drop | Done, fuzzed on a virtual clock |
| OpenAI-compatible API with session and tool hints | Done |
| Trace replayer | Done; see the note on traces below |
| Triton paged decode attention | Not started (needs the GPU) |
| CUDA transfer kernel, pinned staging, copy stream, FP8 | Not started (needs the GPU); the CPU tier currently uses plain copies |
| Benchmarks against vLLM and SGLang | Not started (needs the GPU) |

There are no benchmark results in this repository yet. The cost model's coefficients must come
from a GPU run of the crossover benchmark; the server requires a GPU-measured file when run with
`--device cuda`.

## Running

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e ".[dev]"
make test
```

Tests run on CPU with a very small random-weight model and with real Qwen3-0.6B weights, which
download on first use. GPU-only tests are marked and skip unless `--gpu` is passed.

To serve a model on CPU, produce a cost model and start the server:

```bash
.venv/bin/python bench/kv_crossover.py --model Qwen/Qwen3-0.6B --tokens 64 128 256 512 --out results/cpu
```

```bash
.venv/bin/python -m layover.server --model Qwen/Qwen3-0.6B --cost-model results/cpu/cost_model.json
```

On a fresh Linux machine with an NVIDIA GPU:

```bash
scripts/setup_gpu.sh --reproduce
```

`make reproduce` regenerates every benchmark number and refuses to run without a CUDA GPU.

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

`python -m layover.replay` replays a trace against any OpenAI-compatible server. The format is
JSONL, documented in `layover/replay/trace.py`, and it can be produced in two ways:

- `convert` reads real [mini-swe-agent](https://github.com/SWE-agent/mini-swe-agent) trajectories,
  the agent Continuum evaluated with. Tool gaps come from message timestamps; a trajectory without
  them is labelled as having assumed tool times.
- `generate` makes synthetic traces whose shape follows the workload statistics published in the
  Continuum paper, with sweeps over tool latency and branching factor.

The Continuum authors state that they will release their traces, but at the time of writing their
[repository](https://github.com/Hanchenli/vllm-continuum) holds per-job durations, not per-turn
traces. Every result carries its trace's `source`, so synthetic results cannot pass for recorded ones.

## Prior work

InferCept (Abhyankar et al., ICML 2024) introduced the preserve, swap, or discard choice. Continuum
(Li et al., ICLR 2026) pins KV with a time-to-live. Autellix (Luo et al., 2025) schedules at the
program level. The paged KV layout follows Kwon et al. (SOSP 2023), the prefix tree SGLang's
RadixAttention (Zheng et al., NeurIPS 2024), continuous batching Orca (Yu et al., OSDI 2022), and
chunked prefill Sarathi-Serve (Agrawal et al., OSDI 2024). No code is taken from these systems.
