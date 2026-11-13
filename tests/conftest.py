import os
import sys

# The interpreter runs Triton kernels as Python on CPU tensors. It must be set before
# triton is imported, and never for a GPU run.
if "--gpu" not in sys.argv:
    os.environ.setdefault("TRITON_INTERPRET", "1")

import pytest


def pytest_addoption(parser):
    parser.addoption("--gpu", action="store_true", help="run tests that need a CUDA GPU")
    parser.addoption("--checkpoint", default="Qwen/Qwen3-0.6B", help="checkpoint for real-weight parity tests")


def pytest_collection_modifyitems(config, items):
    # CI has no network for tests: LAYOVER_OFFLINE=1 skips those that download weights or a tokenizer.
    if os.environ.get("LAYOVER_OFFLINE") == "1":
        skip_weights = pytest.mark.skip(reason="needs downloaded Hugging Face files; LAYOVER_OFFLINE=1 is set")
        for item in items:
            if "weights" in item.keywords:
                item.add_marker(skip_weights)
    if config.getoption("--gpu"):
        return
    skip_gpu = pytest.mark.skip(reason="needs a CUDA GPU; run with --gpu")
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip_gpu)


@pytest.fixture(scope="session")
def checkpoint(request):
    return request.config.getoption("--checkpoint")
