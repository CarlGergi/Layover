"""NVTX ranges for Nsight Systems, off unless LAYOVER_NVTX=1.

    LAYOVER_NVTX=1 nsys profile --trace=cuda,nvtx python -m layover.server ...

Off, a range is one shared do-nothing context manager, so the hot path pays a call and nothing
else. On, each range is a push and a pop on the current thread's NVTX stack.
"""

from __future__ import annotations

import os
from contextlib import nullcontext

ENABLED = os.environ.get("LAYOVER_NVTX") == "1"
_OFF = nullcontext()


class _Range:
    __slots__ = ("name",)

    def __init__(self, name: str):
        self.name = name

    def __enter__(self):
        import torch

        torch.cuda.nvtx.range_push(self.name)
        return self

    def __exit__(self, *exc):
        import torch

        torch.cuda.nvtx.range_pop()
        return False


def nvtx_range(name: str):
    """A context manager that marks `name` in a profile when LAYOVER_NVTX=1, else does nothing."""
    return _Range(name) if ENABLED else _OFF
