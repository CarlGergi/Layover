import csv
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

spec = importlib.util.spec_from_file_location(
    "decode_bandwidth", Path(__file__).resolve().parents[1] / "bench" / "decode_bandwidth.py"
)
db = importlib.util.module_from_spec(spec)
sys.modules["decode_bandwidth"] = db
spec.loader.exec_module(db)

QWEN = db.SHAPES["qwen3-4b"]
MICRO = db.SHAPES["micro"]
SMOKE = ["--shape", "micro", "--pool-gib", "0.001", "--tokens", "32", "48", "--batch-sizes", "1", "2", "--repeats", "3"]


def test_qwen3_4b_block_matches_the_kv_arithmetic():
    assert QWEN.block_bytes(2) == 2_359_296
    assert QWEN.query_heads == 32


def test_bytes_loaded_counts_query_heads_kinds_and_whole_blocks():
    # 40 tokens is 3 blocks, 1 token is 1 block. A block is 2 kinds * 16 slots * 128 dims * 2 B
    # = 8 KiB for one query head, and all 32 query heads load it.
    assert db.bytes_loaded_per_layer([40, 1], QWEN, 2) == 4 * 32 * 8192
    # Exactly aligned does not round up.
    assert db.bytes_loaded_per_layer([32], QWEN, 2) == 2 * 32 * 8192
    assert db.bytes_loaded_per_layer([33], QWEN, 2) == 3 * 32 * 8192


def test_unique_bytes_remove_only_gqa_re_reads():
    loaded = db.bytes_loaded_per_layer([40, 1], QWEN, 2)
    assert db.unique_bytes_per_layer([40, 1], QWEN, 2) == loaded // QWEN.kv_groups == 4 * 8 * 8192


def test_accounting_equals_the_loads_the_kernel_issues(monkeypatch):
    """Count the tile loads the interpreter actually executes."""
    pytest.importorskip("triton")
    from layover.kernels import triton_decode as td

    counted = {"elements": 0, "calls": 0}
    real = td._load_tile

    def counting(pointer, in_range, PADDED):
        if len(pointer.shape) == 2:  # q is one-dimensional; K and V tiles are two.
            counted["calls"] += 1
            counted["elements"] += int(in_range.handle.data.sum()) if PADDED else int(in_range.handle.data.size)
        return real(pointer, in_range, PADDED)

    monkeypatch.setattr(td, "_load_tile", counting)
    shape = db.KVShape(layers=2, kv_heads=2, kv_groups=2, head_dim=24, slots=16)
    lengths = [1, 16, 17, 40]
    blocks = 12
    pool = torch.randn(shape.pool_shape(blocks))
    table = torch.arange(12).reshape(4, 3).to(torch.int32)
    q = torch.randn(4, shape.query_heads, shape.head_dim)
    td.paged_decode_attention_triton(q, pool, 1, table, torch.tensor(lengths, dtype=torch.int32), shape.kv_groups)
    assert counted["calls"] == 2 * db.sequence_blocks(lengths, shape.slots) * shape.query_heads
    assert counted["elements"] * 4 == db.bytes_loaded_per_layer(lengths, shape, 4)


@pytest.mark.parametrize(
    "name,expected",
    [
        ("NVIDIA L4", 300.0),
        ("L4", 300.0),
        ("NVIDIA GeForce RTX 4090", 1008.0),
        ("NVIDIA GeForce RTX 4090 Laptop GPU", None),
        ("NVIDIA GeForce RTX 4090 Ti", None),
        ("NVIDIA A100-SXM4-40GB", None),
    ],
)
def test_peak_table_is_exact_about_names(name, expected):
    assert db.resolve_peak(name, None)[0] == expected


def test_peak_override_wins():
    assert db.resolve_peak("NVIDIA L4", 123.0) == (123.0, "--peak-gbps")
    assert db.resolve_peak("Anything", 50.0)[0] == 50.0


def test_rtx_4090_peak_is_its_bus_arithmetic():
    assert db.PEAK_GBPS["RTX 4090"] == 384 / 8 * 21


def test_percent_needs_a_peak():
    assert db.percent(150.0, 300.0) == 50.0
    assert db.percent(150.0, None) is None


def test_fragmented_table_is_scattered_and_in_order_table_is_not():
    gen = torch.Generator().manual_seed(0)
    scattered = db.block_table("fragmented", 2, 8, 64, "cpu", gen)
    assert scattered.dtype == torch.int32 and scattered.shape == (2, 8)
    assert len(set(scattered.flatten().tolist())) == 16
    assert ((scattered[:, 1:] - scattered[:, :-1]) != 1).any()
    ordered = db.block_table("contiguous", 2, 8, 64, "cpu", gen)
    assert ((ordered[:, 1:] - ordered[:, :-1]) == 1).all()


def test_summary_mean_and_spread():
    def row(layout, gbps, pct):
        return db.Row(
            "m", "k", layout, 1, 32, 1, 1, 1, 1.0, 1.0, 1.0, gbps, gbps, gbps, gbps, 0.0, gbps / 4,
            pct, pct, pct, pct, pct / 4, 1.0, 1.0, 0.0,
        )

    summary = db.summarize([row("fragmented", 10.0, 20.0), row("fragmented", 30.0, 60.0), row("contiguous", 8.0, 16.0)])
    assert summary["fragmented"]["achieved_pct_mean"] == 40.0
    assert summary["fragmented"]["achieved_pct_min"] == 20.0
    assert summary["fragmented"]["achieved_pct_max"] == 60.0
    assert summary["fragmented"]["unique_pct_mean"] == 10.0
    assert summary["contiguous"]["cases"] == 1


def check_outputs(meta, out_dir, capsys):
    assert meta["measured_on_gpu"] is False and db.NOT_GPU_LABEL in meta["label"]
    saved = json.loads((out_dir / "decode_bandwidth.json").read_text())
    assert saved["label"] == db.NOT_GPU_LABEL
    assert "GQA" in saved["accounting"]
    with (out_dir / "decode_bandwidth.csv").open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == len(saved["rows"]) == 2 * 2 * 2
    assert {r["measurement"] for r in rows} == {db.NOT_GPU_LABEL}
    assert {r["layout"] for r in rows} == {"fragmented", "contiguous"}
    for row in saved["rows"]:
        assert len(row["samples_s"]) == 3
        assert row["kernel_min_s"] <= row["kernel_s"] <= row["kernel_max_s"]
        lengths = [row["tokens"]] * row["batch"]
        element = 2
        assert row["bytes_per_sample"] == db.bytes_loaded_per_layer(lengths, MICRO, element) * row["layers_per_sample"]
        # The rate is those bytes over the median sample, in decimal GB.
        expected = sorted(row["bytes_per_sample"] / s / 1e9 for s in row["samples_s"])[1]
        assert row["achieved_gbps"] == pytest.approx(expected)
        assert row["reference_s"] > 0
    assert (out_dir / "decode_bandwidth.png").stat().st_size > 0
    printed = capsys.readouterr().out
    assert db.NOT_GPU_LABEL in printed
    assert "WARNING" in printed


def test_end_to_end_with_reference_kernel(tmp_path, capsys):
    meta = db.main([*SMOKE, "--kernel", "reference", "--out", str(tmp_path)])
    assert meta["kernel"] == "reference"
    check_outputs(meta, tmp_path, capsys)


def test_end_to_end_with_interpreted_kernel(tmp_path, capsys):
    pytest.importorskip("triton")
    meta = db.main([*SMOKE, "--kernel", "triton", "--out", str(tmp_path)])
    assert meta["kernel"] == "triton"
    # Run on the same inputs, the kernel and the reference must agree. These outputs are
    # BF16 with magnitude below 8, whose ulp is at most 2^-5; a couple of those covers it.
    assert all(row["max_abs_diff_from_reference"] < 2**-4 for row in meta["rows"])
    check_outputs(meta, tmp_path, capsys)


def test_no_percentage_without_a_peak(tmp_path, capsys):
    meta = db.main([*SMOKE, "--kernel", "reference", "--out", str(tmp_path)])
    assert meta["peak_gbps"] is None
    assert all(row["achieved_pct"] is None for row in meta["rows"])
    printed = capsys.readouterr().out
    assert "refusing to print a percentage" in printed
    assert "% of" not in printed
    with (tmp_path / "decode_bandwidth.csv").open() as f:
        assert all(row["achieved_pct"] == "" for row in csv.DictReader(f))


def test_percentage_with_an_override_stays_labelled(tmp_path, capsys):
    meta = db.main([*SMOKE, "--kernel", "reference", "--peak-gbps", "100", "--out", str(tmp_path)])
    assert meta["peak_source"] == "--peak-gbps"
    for row in meta["rows"]:
        assert row["achieved_pct"] == pytest.approx(row["achieved_gbps"])
    assert meta["summary"]["fragmented"]["achieved_pct_mean"] > 0
    printed = capsys.readouterr().out
    assert "% of 100 GB/s peak" in printed
    assert db.NOT_GPU_LABEL in printed


def plotted_lines(monkeypatch):
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib.axes import Axes

    drawn = []
    real = Axes.plot

    def spy(self, x, y, *args, **kwargs):
        drawn.append((kwargs.get("label"), list(y)))
        return real(self, x, y, *args, **kwargs)

    monkeypatch.setattr(Axes, "plot", spy)
    return drawn


def test_the_plot_draws_the_whole_range_as_a_percentage(tmp_path, monkeypatch, capsys):
    drawn = plotted_lines(monkeypatch)
    meta = db.main([*SMOKE, "--layouts", "contiguous", "--batch-sizes", "1", "--kernel", "reference", "--peak-gbps", "100", "--out", str(tmp_path)])
    rows = sorted(meta["rows"], key=lambda r: r["tokens"])
    assert ("batch 1, with GQA re-reads", [r["achieved_pct"] for r in rows]) in drawn
    assert ("batch 1, each KV head once", [r["unique_pct"] for r in rows]) in drawn
    printed = capsys.readouterr().out
    stats = meta["summary"]["contiguous"]
    # The summary leads with the range, and every case prints both ends.
    assert f"contiguous: {stats['unique_pct_mean']:.1f}% to {stats['achieved_pct_mean']:.1f}% of 100 GB/s peak" in printed
    for row in rows:
        assert f"{row['unique_pct']:6.1f}% to {row['achieved_pct']:6.1f}% of peak" in printed


def test_the_plot_draws_the_range_in_gigabytes_without_a_peak(tmp_path, monkeypatch):
    drawn = plotted_lines(monkeypatch)
    meta = db.main([*SMOKE, "--layouts", "contiguous", "--batch-sizes", "1", "--kernel", "reference", "--out", str(tmp_path)])
    rows = sorted(meta["rows"], key=lambda r: r["tokens"])
    assert ("batch 1, with GQA re-reads", [r["achieved_gbps"] for r in rows]) in drawn
    assert ("batch 1, each KV head once", [r["unique_gbps"] for r in rows]) in drawn


def test_cases_larger_than_the_pool_are_skipped_and_recorded(tmp_path):
    meta = db.main(
        ["--shape", "micro", "--pool-gib", "0.0002", "--tokens", "16", "16000", "--batch-sizes", "1", "--repeats", "1",
         "--kernel", "reference", "--layouts", "fragmented", "--out", str(tmp_path)]
    )
    assert [r["tokens"] for r in meta["rows"]] == [16]
    assert meta["skipped"][0]["tokens"] == 16000


def test_refuses_when_nothing_fits(tmp_path):
    with pytest.raises(SystemExit):
        db.main(["--shape", "micro", "--pool-gib", "0.0001", "--tokens", "160000", "--kernel", "reference",
                 "--out", str(tmp_path)])


def test_rejects_cuda_without_gpu(tmp_path):
    if torch.cuda.is_available():
        pytest.skip("a GPU is present")
    with pytest.raises(SystemExit):
        db.main(["--device", "cuda", "--out", str(tmp_path)])


def test_refuses_to_time_the_interpreter_on_a_gpu(monkeypatch):
    triton = pytest.importorskip("triton")
    monkeypatch.setattr(triton.knobs.runtime, "interpret", True)
    with pytest.raises(SystemExit):
        db.load_triton_backend(torch.device("cuda"))


def test_a_wrong_kernel_does_not_get_to_report_a_bandwidth(tmp_path, monkeypatch):
    def wrong(name, device):
        def launch(q, pool, layer, table, lengths, groups, out):
            out.fill_(1e4)

        return "reference", launch, None

    monkeypatch.setattr(db, "select_kernel", wrong)
    with pytest.raises(RuntimeError):
        db.main([*SMOKE, "--out", str(tmp_path)])


def test_a_kernel_that_disagrees_with_the_reference_does_not_report(tmp_path, monkeypatch):
    def wrong(name, device):
        def launch(q, pool, layer, table, lengths, groups, out):
            out.copy_(pool[0, 1, layer, 0, 0].expand_as(out))

        return "reference", launch, None

    monkeypatch.setattr(db, "select_kernel", wrong)
    with pytest.raises(RuntimeError, match="differs from the reference"):
        db.main([*SMOKE, "--out", str(tmp_path)])


def test_reference_running_out_of_memory_keeps_the_measurement(tmp_path, monkeypatch, capsys):
    calls = {"launched": 0}

    def launch(q, pool, layer, table, lengths, groups, out):
        calls["launched"] += 1
        out.zero_()

    def out_of_memory(*args):
        raise torch.cuda.OutOfMemoryError("no room")

    monkeypatch.setattr(db, "select_kernel", lambda name, device: ("triton", launch, None))
    monkeypatch.setattr(db, "paged_decode_attention_torch", out_of_memory)
    meta = db.main([*SMOKE, "--out", str(tmp_path)])
    assert calls["launched"] > 0 and meta["complete"] is True
    assert all(row["reference_s"] is None and row["achieved_gbps"] > 0 for row in meta["rows"])
    assert "does not fit in memory" in capsys.readouterr().out


def test_a_crash_keeps_what_was_measured(tmp_path, monkeypatch):
    real = db.measure
    seen = []

    def fail_on_second(*args):
        if seen:
            raise RuntimeError("boom")
        seen.append(1)
        return real(*args)

    monkeypatch.setattr(db, "measure", fail_on_second)
    with pytest.raises(RuntimeError, match="boom"):
        db.main([*SMOKE, "--kernel", "reference", "--out", str(tmp_path)])
    saved = json.loads((tmp_path / "decode_bandwidth.json").read_text())
    assert saved["complete"] is False and len(saved["rows"]) == 1
    assert saved["label"] == db.NOT_GPU_LABEL


@pytest.mark.parametrize("flags", [["--warmup", "0"], ["--repeats", "0"], ["--peak-gbps", "0"], ["--layers-per-sample", "0"]])
def test_refuses_arguments_that_would_misreport(tmp_path, flags):
    with pytest.raises(SystemExit):
        db.main([*SMOKE, "--kernel", "reference", *flags, "--out", str(tmp_path)])


def clean_environment():
    """As a user's shell has it: the test process's interpreter setting must not leak in."""
    root = Path(__file__).resolve().parents[1]
    env = {k: v for k, v in os.environ.items() if k != "TRITON_INTERPRET"}
    return {**env, "PYTHONPATH": str(root), "PYTHONDONTWRITEBYTECODE": "1"}


def test_command_line_default_kernel_runs_without_any_environment(tmp_path):
    """--kernel auto probes for triton, and importing it there used to fix GPU mode too early."""
    done = subprocess.run(
        [sys.executable, str(Path(db.__file__)), *SMOKE, "--layouts", "fragmented", "--out", str(tmp_path)],
        env=clean_environment(), capture_output=True, text=True, timeout=300,
    )
    assert done.returncode == 0, done.stderr[-2000:]
    assert db.NOT_GPU_LABEL in done.stdout


def test_kernel_module_imports_without_triton_and_refuses_when_called():
    """What a Mac sees: blocking the import here gives the same result where triton exists."""
    script = """
import sys
sys.modules["triton"] = None
sys.modules["triton.language"] = None
import torch
import layover, layover.model, layover.engine
from layover.kernels import triton_decode
try:
    triton_decode.paged_decode_attention_triton(*[None] * 6)
except RuntimeError as error:
    print("refused:", error)
"""
    done = subprocess.run(
        [sys.executable, "-c", script], env=clean_environment(), capture_output=True, text=True, timeout=120
    )
    assert done.returncode == 0, done.stderr[-2000:]
    assert "refused: the Triton decode kernel needs triton" in done.stdout


@pytest.mark.gpu
def test_gpu_end_to_end(tmp_path):
    meta = db.main(
        ["--device", "cuda", "--kernel", "triton", "--pool-gib", "1", "--tokens", "256", "1024", "--batch-sizes", "1", "4",
         "--repeats", "3", "--peak-gbps", "300", "--out", str(tmp_path)]
    )
    assert meta["measured_on_gpu"] is True and meta["label"] == db.GPU_LABEL
    assert meta["layers_per_sample"] == 36
    assert all(row["achieved_gbps"] > 0 and row["achieved_pct"] > 0 for row in meta["rows"])
    assert (tmp_path / "decode_bandwidth.png").stat().st_size > 0
