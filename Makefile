PYTHON ?= .venv/bin/python
GPU_MODEL ?= Qwen/Qwen3-4B

.PHONY: test test-gpu reproduce smoke

# Everything that runs without a GPU, on the micro model and real Qwen3-0.6B weights.
test:
	$(PYTHON) -m pytest -q

# The GPU-only tests, on the target model.
test-gpu:
	$(PYTHON) -m pytest -q --gpu -m gpu --checkpoint $(GPU_MODEL)

# Every benchmark number. Refuses to run without a CUDA GPU.
reproduce:
	$(PYTHON) bench/kv_crossover.py --device cuda --model $(GPU_MODEL) --out results/kv_crossover

# A pipeline check on CPU. Its numbers describe nothing and are labelled as such.
smoke:
	$(PYTHON) bench/kv_crossover.py --out results/smoke_kv_crossover
