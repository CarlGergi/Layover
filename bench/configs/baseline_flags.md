# Flags for each system

Versions: **vLLM 0.33.0**, **SGLang 0.5.23**, both from PyPI, Python 3.12. They pin different
versions of torch (2.14.0 and 2.14.1), so `scripts/setup_baselines.sh` gives each its own environment.

How the flags were checked, in a throwaway Linux container on a machine with no NVIDIA GPU:

- Every flag `bench/systems.py` uses appears in the real output of `vllm serve --help=all` (with
  `VLLM_TARGET_DEVICE=cpu`, since the CLI will not start otherwise) and `sglang serve --help`, from
  the linux/arm64 wheels.
- The linux/x86_64 wheels, which the GPU machine installs, were unpacked and read for what the
  tables below cite. The argument definitions compared byte for byte with the arm64 ones.
- The exact flag set was passed to each real command with a bad model path. vLLM logged all of them
  as `non-default args`, and SGLang got past its parser to model resolution, so all were accepted.
- The request below was parsed by each system's real `CompletionRequest`.

**Not verified, because nothing here can serve a request:** that either baseline answers it,
returns token ids, honours `ignore_eos`, reports `cached_tokens`, reaches the expected KV capacity,
or uses its CPU tier. The response format is read from source. `bench/run_matrix.py --preflight-only
--preflight-kinds vllm sglang` checks all but the CPU tier with real completions on the GPU, needs
no Layover cost model or trace, and saves each baseline's `/metrics` to `logs/preflight/` so the
names below can be checked against the build. The report's CPU tier evidence table gives each
baseline's counter of what came back from its tier, for each workload and load, and a zero or
missing counter flags that baseline's table cells rather than letting them read as offload working.
The KV capacity log line, the `/server_info` field, and the Prometheus names are also read from
source, and the same preflight exercises the capacity checks.

## Matched settings

From `bench/configs/matrix.json`, the same for all systems.

| Setting | vLLM | SGLang | Layover | Checked at runtime |
| --- | --- | --- | --- | --- |
| Model | positional | `--model-path` | `--model` | `/v1/models` |
| Precision, BF16 | `--dtype bfloat16` | `--dtype bfloat16` | `--dtype bf16` | |
| KV precision, BF16 | `--kv-cache-dtype bfloat16` | `--kv-cache-dtype bf16` | the pool's dtype | |
| Max context | `--max-model-len` | `--context-length` | no flag; the runner refuses longer traces | |
| KV budget (GiB) | `--kv-cache-memory-bytes` = GiB x 2^30 | `--max-total-tokens` = GiB x 2^30 / KV bytes per token | `--pool-gib` | see below |
| CPU tier (GiB) | `--kv-offloading-size`, in GiB | `--hicache-ratio` = CPU GiB / KV GiB, over its device tokens | `--cpu-tier-gib` | see Fairness |
| Running requests | `--max-num-seqs` | `--max-running-requests` | `--max-batch` | |
| Prefill chunk | `--max-num-batched-tokens` = chunk + running | `--chunked-prefill-size` | `--prefill-chunk` | |

The KV budget is checked twice after each startup, before anything is measured. The server's own
capacity must be within `tolerances.kv_capacity_frac` of the budget: vLLM's log line
`KV cache size: N tokens`, SGLang's `max_total_num_tokens` from `/server_info`, Layover's
`blocks_total` from `/metrics`. Then the GPU memory NVML shows as held must be within
`tolerances.gpu_memory_gib` of weights plus KV budget. Either failure fails the run, loudly, and so
does a server whose capacity cannot be read at all.

## vLLM 0.33.0

| Flag | Why |
| --- | --- |
| `serve <model>` | Positional, the form `--help` shows. |
| `--served-model-name` | The same name every system answers to, so the client sends the same request. |
| `--host`, `--port` | Harness picks a free port per run. |
| `--dtype bfloat16` | Same precision everywhere. |
| `--kv-cache-dtype bfloat16` | KV precision stated, not left to `auto`. |
| `--max-model-len` | Same context limit everywhere. |
| `--kv-cache-memory-bytes` | Sets the KV size directly; `--help` says it overrides `--gpu-memory-utilization`. |
| `--enable-prefix-caching` | Default is None, resolved at startup, so it is stated. The design calls for vLLM with prefix caching. |
| `--kv-offloading-backend native` | vLLM's own CPU offload, the design's baseline. The default, stated. It selects `OffloadingConnector`. |
| `--kv-offloading-size` | In GiB, per `cpu_bytes_to_use = size * (1 << 30)` in `vllm/config/vllm.py`. `--help` says offload is off without it. |
| `--scheduling-policy fcfs` | The default, stated. |
| `--max-num-seqs` | Same running-request limit. |
| `--enable-chunked-prefill` | So a long prefill does not stall decoding, as on the other two. |
| `--max-num-batched-tokens` | Per-step token budget, decode included, so chunk + running gives the same prefill chunk. |
| `--stream-interval 1` | The default, stated: each token is sent at once, so first-token time is comparable. |
| `--generation-config vllm` | Keeps the checkpoint's generation_config from changing sampling or stop tokens. |
| `--enable-prompt-tokens-details` | So usage reports cached tokens. Off by default. |
| `--disable-uvicorn-access-log` | A log line per request is overhead nobody needs here. |
| `--seed` | Same seed everywhere. |

Left at their defaults on purpose: `--attention-backend` (None, automatic), `--async-scheduling`
(None, automatic), `--block-size` (16 per `CacheConfig.DEFAULT_BLOCK_SIZE`). Each system runs its
best out-of-box kernel, and the resolved choice is in the run's log. There is no cuBLAS flag in
`--help=all`.

## SGLang 0.5.23

| Flag | Why |
| --- | --- |
| `serve` | The entry point `python -m sglang.launch_server` warns against. |
| `--model-path`, `--served-model-name` | As above. |
| `--host`, `--port` | As above. |
| `--dtype bfloat16`, `--kv-cache-dtype bf16` | Same precision everywhere; the two flags spell it differently. |
| `--context-length` | Same context limit everywhere. |
| `--max-total-tokens` | Sets the KV pool in tokens. SGLang takes the smaller of this and what memory allows (`utils/common.py`). Its help calls this option "typically used for development and debugging", so the startup check matters here. |
| `--mem-fraction-static 0.9` | High enough that `--max-total-tokens` is what binds on 24 GB; the startup check proves it did. |
| `--enable-hierarchical-cache` | SGLang's CPU tier, the counterpart of vLLM's offload. |
| `--hicache-ratio` | The host pool in device tokens: `device_capacity_tokens * host_to_device_ratio`, rounded up a page, in `mem_cache/pool_host/base.py`. A float, so it reaches the budget. `--hicache-size` takes only whole decimal gigabytes and would override it, so it is not passed. Passing a ratio also turns off SGLang's host memory auto-sizing (`mem_cache/hicache_auto_size.py`). |
| `--schedule-policy fcfs` | Same first-come order as vLLM. |
| `--max-running-requests` | Same running-request limit. |
| `--chunked-prefill-size` | Same prefill chunk. |
| `--stream-interval 1` | Each token sent at once, as on the others. |
| `--sampling-defaults openai` | Default is `model`, which would apply the checkpoint's generation_config. |
| `--enable-metrics` | Prometheus `/metrics`, for `sglang:queue_time_seconds`, `sglang:token_usage`, and `sglang:load_back_tokens_total`, the CPU tier's evidence. `sglang:hicache_host_used_tokens` is kept but proves nothing: under write-through it rises with no reload. Off by default. |
| `--enable-cache-report` | So usage reports cached tokens. Off by default. |
| `--random-seed` | Same seed everywhere. |

Left at their defaults on purpose: `--attention-backend`, `--page-size`, `--hicache-write-policy`
(`write_through`), and the overlap scheduler (on). `--help` mentions cuBLAS only inside a backend
option for other models.

SGLang's `/health` generates one token when the server is idle, so the harness calls it only
during startup.

## Layover

`--decode-backend triton`, `--transfer native`, and `--offload-dtype fp8` come from the `layover`
block of `matrix.json`. FP8 applies to Layover's CPU tier only: every system gets the same CPU tier
budget in bytes, the baselines offload BF16, and `layover/server.py` sizes the tier by the bytes an
FP8 record occupies, so Layover's tier holds about twice the tokens per GiB. That is one of the
things it is built to do. `--policy` selects the ablation.

The ablations, all at 100% load:

| System | What it isolates |
| --- | --- |
| `layover-combined-bf16` | The FP8 gain alone: `--offload-dtype bf16`, nothing else changed. Same 32 GiB, so its tier holds about half the blocks. It reads the same cost model file: `bench/kv_crossover.py --offload-dtype fp8` times BF16 copies too and records them as the base `swap_*_s_coef`, and `CostModel.for_offload(False)` uses those. |
| `layover-keep_ttl`, `layover-swap` | One placement policy alone, in place of the cost model's choice. |
| `layover-discard` | No keep or swap, with the prefix cache on. See Fairness: the same evictable pool as vLLM. |
| `layover-discard-nocache` | **A strict lower bound, not a fair comparison, and labelled so everywhere.** `--no-prefix-cache` (`"prefix_cache": false`): every resumed turn recomputes its whole context, and even the shared system prompt is not reused. No real server is configured this way. |

## The request every system receives

`/v1/completions` with `prompt` as a token id list, `max_tokens`, `temperature` 0, `top_p` 1,
`stream` true, `stream_options.include_usage` true, `ignore_eos` true, `return_token_ids` true.

| Field | vLLM 0.33.0 | SGLang 0.5.23 |
| --- | --- | --- |
| The request | Parsed by its real `CompletionRequest`: accepted | Parsed by its real `CompletionRequest`: accepted |
| `return_token_ids` | Read from source: `token_ids` delta on each chunk, `prompt_token_ids` on the first | Read from source: the same two fields |
| Final chunk | Read from source: with include_usage, one more chunk with empty `choices` | Read from source: same behaviour handled |
| `session_id` | A declared field, "stable session identity" | A declared field, used by its session machinery |

`session_id` and `layover` are sent only to Layover, because `session_id` is not an ignored
extra field on either baseline.

Queue metrics, which are reported separately because their definitions differ:
`vllm:request_queue_time_seconds` (time in the WAITING phase) and `sglang:queue_time_seconds`.
Layover's `/metrics` has no queue-time metric at this commit. The comparable measure across all
three is the client's `first_token_s`.

## Fairness

### What `layover-discard` does with freed KV, and why

The ablation has to isolate Layover's decisions, not hold it to a stricter cache than the baseline
has. So `layover-discard` frees a turn's KV the way vLLM does. Read from the published sdist of
vLLM 0.33.0, commit g4ba787ec9:

- `vllm/v1/core/block_pool.py`, `BlockPool.free_blocks`: a block whose reference count reaches zero
  goes on the free queue, and a block with a hash keeps it, so it can still be matched. Blocks with
  no hash are reused first; hashed ones are appended at the back.
- `vllm/v1/core/block_pool.py`, `BlockPool.get_new_blocks` and `_maybe_evict_cached_block`: a block
  loses its hash only when it is popped from the front of the queue for a new allocation. That is
  the only eviction, so it happens only under memory pressure.
- `vllm/v1/core/kv_cache_utils.py`, `FreeKVCacheBlockQueue`: least recently used at the front, and
  within one sequence the tail of the chain goes first.
- `vllm/v1/core/single_type_kv_cache_manager.py`: a request frees its blocks in reverse order to
  give that tail-first order.

So a vLLM agent that returns before its blocks are displaced finds its KV. `layover-discard` matches
that: a finished turn's blocks stay in the radix tree with only the tree's reference, reusable if
the agent returns, and the tree gives them up least recently used first when a new turn needs room.
It is Layover's scheduler without the keep and swap decisions, not a harsher cache.
`layover-discard-nocache` is the strict lower bound beside it.

The KV pool used column follows from this. vLLM's `vllm:kv_cache_usage_perc` counts what is not on
that free queue, so a cached block nothing holds reads as free. Layover's `blocks_free` would count
such a block as used, so its column subtracts the blocks only the radix tree holds and publishes them
in a column of their own. A figure that leaves cached blocks out can stay low while eviction runs, so
no tier flag is derived from it. SGLang's `sglang:token_usage` is taken as it comes; its definition
was not re-read for this column.

One difference remains: vLLM can also offload those blocks to its CPU tier, and `layover-discard`
has no CPU tier in use, since moving KV to CPU is one of the decisions being removed.

### 32 GiB of CPU memory for every system that offloads

All of them take the one budget, `matched.cpu_offload_gib`:

| System | Flags at 12 GiB KV, 32 GiB CPU | What they give | Checked after each startup through |
| --- | --- | --- | --- |
| vLLM | `--kv-offloading-size 32` | 32 x 2^30 bytes: `cpu_bytes_to_use` in `vllm/config/vllm.py` | The startup log's echo of the flag only; 0.33.0 has no capacity counter |
| SGLang | `--max-total-tokens 87381 --hicache-ratio 2.6666666666666665` | 233,024 tokens x 147,456 bytes = 32.00 GiB, once rounded to its 64-token pages | The bytes its startup log states for the host pool, or `sglang:hicache_host_total_tokens` |
| Layover FP8 | `--cpu-tier-gib 32 --offload-dtype fp8` | 29,098 records of 1,180,800 bytes, 32.00 GiB | `cpu_tier_capacity_blocks` x bytes per record, from `/metrics` |
| Layover BF16 | `--cpu-tier-gib 32 --offload-dtype bf16` | 14,563 blocks of 2,359,296 bytes, 32.00 GiB | The same |

The working notes once had SGLang at 31.66 GiB. That was `--hicache-size 34`: an int field in
decimal gigabytes, so `34e9 / 2^30`, and no whole number is 32 GiB. `--hicache-ratio` is a float, and
replaced it, so the gap is gone. A reported tier more than `tolerances.cpu_tier_frac` (default 1%)
from the budget fails the run. The SGLang figure above comes from reading `mem_cache/pool_host/base.py`
at 0.5.23, not from running it: the first check on the GPU is that startup comparison.

### Saturation and load

The definition is `calibration.definition` in `matrix.json`, and each report repeats it: the rate at
which vLLM's mean JCT reaches `target_slowdown` times unloaded, on the full headline workload with
warm-up excluded, log-interpolated between the two rungs that bracket it. Runs happen at 50%, 100%
and 125% of that rate, the same absolute rates for every system. A ladder that never crosses, or
starts past the crossing, stops the matrix before any other run, unless `--accept-saturation-edge`
is passed, and then the top of the report says so.

### Spot instances

A run's record is fsynced and renamed into place when the run ends, so a reclaim costs the run in
progress and whatever has not been copied off the machine yet. A resume on a new machine accepts a
new instance id, and refuses a different GPU or driver unless `--accept-mixed-hardware` is passed,
when the report flags each table it touches. Each record names its instance and each table says how
many it spans. The cost model is compared by content, so a restored file passes and a re-measured
one is refused.

### Enforced by the code

Each with a test in `tests/test_matrix.py`:

- One client, one seed per run: the same arrivals and the same token ids before anything is
  generated, on every system. Layover's hints sit beside the prompt, and never include a tool's
  duration. Contexts diverge after the first generated turn only if numerics differ; the report
  counts turns identical to the reference.
- A run fails if any turn returned fewer tokens than asked or had to be re-encoded from text.
- The first `warmup_jobs` jobs by planned arrival are left out of the job-time and first-token
  columns. The GPU memory, KV pool, and PCIe columns cover a time window from the first measured
  arrival to the last measured finish, which includes warm-up jobs still running in it. Servers' own
  counters cover the whole run, warm-up included, and the report says so where it shows them.
- Each run gets a new server. GPU memory must be under `tolerances.gpu_idle_gib` before it starts
  and after it stops, or the matrix ends.
- A preflight sends the same prompt twice to each baseline and refuses it if the second
  `cached_tokens` is missing or under half the prompt, so a baseline cannot lose because its prefix
  cache was off.
- The arrival rate is calibrated once per group on vLLM and held for every system. Calibrating on
  Layover would set the load at its own knee, past the knee of anything slower. Each rung of the
  ladder replays the group's whole reference workload with warm-up jobs left out, as measured runs
  do. The unloaded time comes from `unloaded_jobs` jobs run alone, and the slowdown compares those
  same jobs in the rung. The rungs were set to bracket the knee in simulation at assumed rates, with
  vLLM's offload working or inert; each report gives every rung measured and its slowdown.

### Differences that remain

By design, or because they cannot be removed:

- Layover gets session and tool-name hints and forks; the baselines see plain prompts. That is what
  it is built to use. The baselines' `session_id` has other meanings, so it is not sent.
- Layover's CPU tier holds FP8, so it fits more tokens per GiB than the baselines' BF16 tiers.
- `return_token_ids` makes both baselines send the whole prompt's ids on each turn's first chunk,
  which Layover does not. It costs them serialization per turn, unmeasured here. The client stamps
  arrival before parsing, so it does not count against their first-token time on the client's side.
- Each system runs its own kernels at their defaults, including attention backends and batching.
- A trace holds token counts, not text. The ids sent are random and repeatable, so only lengths and
  prefix sharing match the recording.
