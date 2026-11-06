"""Sample GPU memory, utilization, and PCIe throughput through NVML during a run.

NVML answers a PCIe throughput query from a window of about 20 ms, so the gigabytes here
come from integrating those samples over time. They are an estimate, and every summary
says so. Exact byte counts exist only inside a system, such as Layover's
cpu_tier_bytes_moved, and those are reported separately.
"""

from __future__ import annotations

import csv
import statistics
import threading
import time
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass
from pathlib import Path

PCIE_NOTE = (
    "NVML PCIe figures integrate sampled throughput (about 20 ms per query) over the run, "
    "reading NVML's KB/s as KiB/s. They are estimates, not byte counts. Like the GPU memory and "
    "KV usage figures, they cover a time window, so warm-up jobs still running in it are counted."
)
GIB = 2**30


@dataclass(frozen=True)
class GpuReading:
    memory_used_bytes: int
    memory_total_bytes: int
    gpu_util_pct: float | None = None
    # NVML's memory utilization: the share of time memory was being read or written.
    memory_busy_pct: float | None = None
    pcie_tx_kib_s: float | None = None
    pcie_rx_kib_s: float | None = None


class NvmlBackend:
    """One GPU through pynvml, which the nvidia-ml-py package provides."""

    def __init__(self, index: int = 0):
        import pynvml

        self.nvml = pynvml
        pynvml.nvmlInit()
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(index)

    def _optional(self, query: Callable, *args):
        # Virtualized GPUs refuse some counters; a missing one must not end the run.
        try:
            return query(self.handle, *args)
        except self.nvml.NVMLError:
            return None

    def read(self) -> GpuReading:
        memory = self.nvml.nvmlDeviceGetMemoryInfo(self.handle)
        rates = self._optional(self.nvml.nvmlDeviceGetUtilizationRates)
        return GpuReading(
            memory.used,
            memory.total,
            rates.gpu if rates else None,
            rates.memory if rates else None,
            self._optional(self.nvml.nvmlDeviceGetPcieThroughput, self.nvml.NVML_PCIE_UTIL_TX_BYTES),
            self._optional(self.nvml.nvmlDeviceGetPcieThroughput, self.nvml.NVML_PCIE_UTIL_RX_BYTES),
        )

    def device_info(self) -> dict:
        def text(value) -> str:
            return value.decode() if isinstance(value, bytes) else str(value)

        return {
            "name": text(self.nvml.nvmlDeviceGetName(self.handle)),
            "driver": text(self.nvml.nvmlSystemGetDriverVersion()),
            "memory_total_bytes": self.nvml.nvmlDeviceGetMemoryInfo(self.handle).total,
        }

    def compute_processes(self) -> list[dict]:
        processes = self._optional(self.nvml.nvmlDeviceGetComputeRunningProcesses) or []
        return [{"pid": p.pid, "used_bytes": p.usedGpuMemory} for p in processes]

    def close(self) -> None:
        self.nvml.nvmlShutdown()


class StubBackend:
    """Scripted readings for tests. The last one repeats."""

    def __init__(self, readings: Iterable[GpuReading]):
        self.readings = list(readings)
        self.next_index = 0

    def read(self) -> GpuReading:
        reading = self.readings[min(self.next_index, len(self.readings) - 1)]
        self.next_index += 1
        return reading

    def device_info(self) -> dict:
        return {"name": "stub", "driver": "stub", "memory_total_bytes": self.readings[0].memory_total_bytes}

    def compute_processes(self) -> list[dict]:
        return []

    def close(self) -> None:
        pass


def _integrate(samples: list[tuple[float, float | None]]) -> float | None:
    """Trapezoid integral over the samples that have a value."""
    known = [(t, v) for t, v in samples if v is not None]
    if not known:
        return None
    return sum((a + b) / 2 * (t1 - t0) for (t0, a), (t1, b) in zip(known, known[1:]))


def _mean(values: list[float | None]) -> float | None:
    known = [v for v in values if v is not None]
    return statistics.fmean(known) if known else None


class GpuSampler:
    def __init__(self, backend, period_s: float = 0.25, clock: Callable[[], float] = time.perf_counter):
        self.backend = backend
        self.period_s = period_s
        self.clock = clock
        self.samples: list[tuple[float, GpuReading]] = []
        self.read_errors = 0
        self._stopping = threading.Event()
        self._thread: threading.Thread | None = None

    def sample_once(self) -> None:
        try:
            self.samples.append((self.clock(), self.backend.read()))
        except Exception:  # A lost sample shows in read_errors; the run itself goes on.
            self.read_errors += 1

    def _run(self) -> None:
        while not self._stopping.is_set():
            self.sample_once()
            self._stopping.wait(self.period_s)

    def start(self) -> None:
        self._stopping.clear()
        self._thread = threading.Thread(target=self._run, daemon=True, name="gpu-sampler")
        self._thread.start()

    def stop(self) -> None:
        self._stopping.set()
        if self._thread is not None:
            self._thread.join(timeout=10)
        self.sample_once()

    def summary(self, start_s: float | None = None, end_s: float | None = None) -> dict:
        """Averages and PCIe totals, over the samples between two clock readings when given."""
        samples = [
            (t, r) for t, r in self.samples
            if (start_s is None or t >= start_s) and (end_s is None or t <= end_s)
        ]
        if not samples:
            return {"samples": 0, "read_errors": self.read_errors, "pcie_note": PCIE_NOTE}
        readings = [r for _, r in samples]
        total = readings[0].memory_total_bytes
        used = [r.memory_used_bytes for r in readings]
        tx = _integrate([(t, r.pcie_tx_kib_s) for t, r in samples])
        rx = _integrate([(t, r.pcie_rx_kib_s) for t, r in samples])
        tx_gb = None if tx is None else tx * 1024 / 1e9
        rx_gb = None if rx is None else rx * 1024 / 1e9
        return {
            "samples": len(readings),
            "read_errors": self.read_errors,
            "duration_s": samples[-1][0] - samples[0][0],
            "gpu_mem_used_gib_mean": statistics.fmean(used) / GIB,
            "gpu_mem_used_gib_peak": max(used) / GIB,
            "gpu_mem_frac_mean": statistics.fmean(used) / total,
            "gpu_mem_frac_peak": max(used) / total,
            "gpu_util_pct_mean": _mean([r.gpu_util_pct for r in readings]),
            "gpu_mem_busy_pct_mean": _mean([r.memory_busy_pct for r in readings]),
            "pcie_tx_gb": tx_gb,
            "pcie_rx_gb": rx_gb,
            "pcie_gb": None if tx_gb is None or rx_gb is None else tx_gb + rx_gb,
            "pcie_note": PCIE_NOTE,
        }

    def write_csv(self, path: str | Path) -> None:
        with Path(path).open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["t_s", *GpuReading.__dataclass_fields__])
            writer.writeheader()
            for t, reading in self.samples:
                writer.writerow({"t_s": f"{t - self.samples[0][0]:.3f}", **asdict(reading)})
