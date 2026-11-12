PYTHON ?= .venv/bin/python
GPU_MODEL ?= Qwen/Qwen3-4B
MATRIX_ARGS ?=
TRACES_ARGS ?=
# FORCE=1 re-measures outputs that already exist. A matrix with saved runs refuses a new cost model.
FORCE ?=

DOCKER_IMAGE ?= layover-dev
HF_CACHE ?= $(HOME)/.cache/huggingface
NATIVE_DEBUG_LIB := native/build/cuda-debug-$(shell uname -s)-$(shell uname -m)/liblayover_transfer.so
TRACE := traces/swebench_30.jsonl

# Measuring again would invalidate saved runs and repay finished work on a resume, so each
# measurement is skipped when its complete output is there. $(1) is the file, $(2) a Python
# expression over its parsed JSON, $(3) the command.
define unless_measured
@if [ -z "$(FORCE)" ] && [ -f "$(1)" ] && $(PYTHON) -c "import json, sys; d = json.load(open('$(1)')); sys.exit(0 if ($(2)) else 1)"; then \
	echo "keeping $(1); FORCE=1 re-measures it"; \
else \
	$(3); \
fi
endef

.PHONY: traces
.PHONY: test test-docker docker-image test-gpu gpu-smoke sanitize smoke \
	traces-prepare baselines kv-crossover decode-bandwidth transfer-gather calibration \
	matrix-dry-run matrix-preflight matrix matrix-report matrix-pipeline-check reproduce profile

# Everything that runs without a GPU, on the micro model and real Qwen3-0.6B weights.
test:
	$(PYTHON) -m pytest -q

docker-image:
	docker build -t $(DOCKER_IMAGE) -f docker/dev.Dockerfile docker

# The same suite on Linux, where Triton installs, so kernels run in its interpreter.
test-docker:
	docker run --rm -v $(CURDIR):/src -v $(HF_CACHE):/root/.cache/huggingface \
		-e PYTHONDONTWRITEBYTECODE=1 $(DOCKER_IMAGE) python -m pytest -q -p no:cacheprovider

# The GPU-only tests, on the target model.
test-gpu:
	$(PYTHON) -m pytest -q --gpu -m gpu --checkpoint $(GPU_MODEL)

# Prefill and decode rates with an Nsight Systems capture and torch.profiler traces, then vLLM at
# the same shapes for the target. About 40 minutes after setup on an L4.
profile:
	mkdir -p results/profile
	LAYOVER_NVTX=1 nsys profile --trace=cuda,nvtx,osrt --force-overwrite=true -o results/profile/layover \
		$(PYTHON) bench/profile_hot_path.py --quick --out results/profile/layover_nsys
	$(PYTHON) bench/profile_hot_path.py --torch-trace --model $(GPU_MODEL) --out results/profile/layover
	bash scripts/setup_baselines.sh vllm
	.venvs/vllm/bin/python bench/profile_hot_path.py --system vllm --model $(GPU_MODEL) --out results/profile/vllm

# Right after setup and before anything expensive: a broken machine costs minutes.
gpu-smoke:
	$(PYTHON) bench/gpu_smoke.py --model $(GPU_MODEL)

# Every custom kernel under compute-sanitizer. Either tool finding anything fails. Without
# torch's allocator cache each tensor is its own CUDA allocation, so overruns are visible.
sanitize:
	$(MAKE) -C native cuda-debug
	PYTORCH_NO_CUDA_MEMORY_CACHING=1 LAYOVER_NATIVE_LIB=$(NATIVE_DEBUG_LIB) compute-sanitizer --tool memcheck --error-exitcode 1 $(PYTHON) tests/sanitize_transfer.py
	PYTORCH_NO_CUDA_MEMORY_CACHING=1 LAYOVER_NATIVE_LIB=$(NATIVE_DEBUG_LIB) compute-sanitizer --tool racecheck --error-exitcode 1 $(PYTHON) tests/sanitize_transfer.py

# Free, runs anywhere: choose the public trajectories and count their tokens.
traces-prepare:
	$(PYTHON) -m layover.replay.trajectories prepare

# On the x86 GPU machine, never during a benchmark: runs third-party commands in task containers.
# Re-running after a replacement would change the trace hash, so a finished trace is kept.
traces:
	@if [ -z "$(FORCE)" ] && [ -f "$(TRACE)" ] && $(PYTHON) -c "import json, sys; sys.exit(0 if json.loads(open('$(TRACE)').readline())['source'].startswith('recorded:') else 1)"; then \
		echo "keeping $(TRACE); FORCE=1 re-executes it"; \
	else \
		$(PYTHON) -m layover.replay.reexecute run $(TRACES_ARGS) && $(PYTHON) -m layover.replay.reexecute convert; \
	fi

baselines:
	bash scripts/setup_baselines.sh

# The native transfer path with BF16 and with FP8 offload. The last is what the matrix's cost model
# reads. The plain path (bench/kv_crossover.py without --transfer) is left out to fit the budget.
kv-crossover:
	$(call unless_measured,results/kv_crossover_native_bf16/cost_model.json,d.get('measured_on_gpu'),$(PYTHON) bench/kv_crossover.py --device cuda --model $(GPU_MODEL) --transfer native --out results/kv_crossover_native_bf16)
	$(call unless_measured,results/kv_crossover/cost_model.json,d.get('measured_on_gpu'),$(PYTHON) bench/kv_crossover.py --device cuda --model $(GPU_MODEL) --transfer native --offload-dtype fp8 --out results/kv_crossover)

decode-bandwidth:
	$(call unless_measured,results/decode_bandwidth/decode_bandwidth.json,d.get('complete') and d.get('measured_on_gpu'),$(PYTHON) bench/decode_bandwidth.py --device cuda --out results/decode_bandwidth)

transfer-gather:
	$(call unless_measured,results/transfer_gather/transfer_gather.json,d.get('complete') and d.get('measured_on_gpu'),$(PYTHON) bench/transfer_gather.py --device cuda --model $(GPU_MODEL) --out results/transfer_gather)

# For the synthetic sweeps only, so the first-pass plan does not need it. Runs here for free.
calibration:
	$(call unless_measured,results/calibration/calibration.json,True,$(PYTHON) bench/calibrate_synthetic.py)

matrix-dry-run:
	$(PYTHON) bench/run_matrix.py --dry-run $(MATRIX_ARGS)

matrix-preflight:
	$(PYTHON) bench/run_matrix.py --preflight-only --out results/matrix_preflight

matrix:
	$(PYTHON) bench/run_matrix.py --out results/matrix $(MATRIX_ARGS)

matrix-report:
	$(PYTHON) bench/report.py results/matrix

# A pipeline check on CPU. Its numbers describe nothing and are labelled as such.
matrix-pipeline-check: smoke
	$(PYTHON) bench/run_matrix.py --config bench/configs/pipeline_check.json --out results/matrix_pipeline_check --pipeline-check --no-gpu

# Every benchmark number for the chosen plan (--plan in MATRIX_ARGS, default final).
# Refuses without a CUDA GPU and the re-executed trace; a re-run skips what is finished.
reproduce: kv-crossover decode-bandwidth transfer-gather matrix-preflight matrix matrix-report

smoke:
	$(PYTHON) bench/kv_crossover.py --out results/smoke_kv_crossover
