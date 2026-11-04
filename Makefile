PYTHON ?= .venv/bin/python

.PHONY: test test-gpu kv-crossover

test:
	$(PYTHON) -m pytest -q

test-gpu:
	$(PYTHON) -m pytest -q --gpu

kv-crossover:
	$(PYTHON) bench/kv_crossover.py --device cuda --model Qwen/Qwen3-4B --out results/kv_crossover
