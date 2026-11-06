"""Every transfer backend, through its ticket interface, against CRC and the FP8 bound."""

import importlib.util
import json
import sys
import threading
import zlib
from pathlib import Path

import numpy as np
import pytest
import torch

from layover.config import ModelConfig
from layover.kernels.fp8 import Fp8Layout, group_shifts, round_trip_bound
from layover.kv import BlockTable, PagedKVCache
from layover.kv.tier import CpuTier
from layover.kv.transfer import DelayedBackend, HostMemoryError, ModelledBackend, NativeBackend, PlainCopyBackend, ReadyTicket, make_backend

CFG = ModelConfig.micro()


def bits_of(pool):
    return pool.view(torch.uint16).reshape(-1).cpu().numpy().tobytes()


def filled_cache(blocks=32, device="cpu", dtype=torch.bfloat16, seed=0):
    cache = PagedKVCache(CFG, blocks, device, dtype)
    generator = torch.Generator().manual_seed(seed)
    values = torch.randn(cache.pool.shape, generator=generator) * torch.exp(
        2 * torch.randn(cache.pool.shape, generator=generator)
    )
    cache.pool.copy_(values.to(dtype).to(device))
    return cache


def scattered(count, blocks, seed=1):
    return [int(i) for i in np.random.default_rng(seed).permutation(blocks)[:count]]


def native(staging_blocks=3, **options):
    cache = PagedKVCache(CFG, 1, dtype=torch.bfloat16)
    return NativeBackend("cpu", staging_bytes=staging_blocks * cache.block_bytes, ids_capacity=256, **options)


@pytest.fixture(params=["plain", "delayed", "native"])
def backend(request):
    if request.param == "plain":
        created = PlainCopyBackend()
    elif request.param == "delayed":
        created = DelayedBackend(polls=3)
    else:
        created = native()
    yield created
    created.close()


def poll_done(ticket, limit=2_000_000):
    for _ in range(limit):
        if ticket.done():
            return
    raise AssertionError("ticket never finished")


def crc_blocks(cache, ids):
    return zlib.crc32(cache.pool[torch.tensor(ids)].view(torch.uint16).numpy().tobytes())


def assert_fp8_close(before, after, ids, layout):
    before_bits = before[torch.tensor(ids)].view(torch.uint16).reshape(-1, layout.group_elems)
    bound = round_trip_bound(before_bits)
    values = before_bits.view(torch.bfloat16).to(torch.float64)
    restored = after[torch.tensor(ids)].to(torch.float64).reshape(-1, layout.group_elems)
    finite = values.isfinite()
    assert ((restored - values).abs()[finite] <= bound[finite]).all()


def test_bf16_round_trip_is_byte_exact_with_scattered_blocks(backend):
    cache = filled_cache()
    ids = scattered(10, 32)
    before = crc_blocks(cache, ids)
    out = backend.start_out(cache, ids, False)
    poll_done(out)
    assert out.payload.nbytes == 10 * cache.block_bytes
    assert zlib.crc32(out.payload.memory.numpy().tobytes()[: out.payload.nbytes]) == before
    target = scattered(10, 32, seed=2)
    into = backend.start_in(cache, target, out.payload)
    poll_done(into)
    assert crc_blocks(cache, target) == before
    backend.free(out.payload)


def test_fp8_round_trip_stays_within_the_bound(backend):
    cache = filled_cache()
    layout = Fp8Layout.from_cache(cache)
    original = cache.pool.clone()
    ids = scattered(10, 32)
    out = backend.start_out(cache, ids, True)
    poll_done(out)
    assert out.payload.fp8 and out.payload.nbytes == 10 * layout.record_bytes
    assert out.payload.nbytes < 0.51 * 10 * cache.block_bytes
    target = scattered(10, 32, seed=2)
    into = backend.start_in(cache, target, out.payload)
    poll_done(into)
    bound = original[torch.tensor(ids)]
    after = cache.pool[torch.tensor(target)]
    bits = bound.view(torch.uint16).reshape(-1, layout.group_elems)
    limit = round_trip_bound(bits)
    gap = (after.to(torch.float64) - bound.to(torch.float64)).abs().reshape(-1, layout.group_elems)
    assert (gap <= limit).all()
    assert (group_shifts(bits) > -200).all()
    backend.free(out.payload)


def test_prefix_restore_takes_only_the_first_records(backend):
    cache = filled_cache()
    ids = scattered(8, 32)
    original = cache.pool.clone()
    expected = crc_blocks(cache, ids[:3])
    out = backend.start_out(cache, ids, False)
    poll_done(out)
    target = [30, 2, 17]
    into = backend.start_in(cache, target, out.payload)
    poll_done(into)
    assert crc_blocks(cache, target) == expected
    others = [i for i in range(32) if i not in target]
    assert bits_of(cache.pool[torch.tensor(others)]) == bits_of(original[torch.tensor(others)])


def test_any_dtype_moves_byte_exact_when_not_quantizing(backend):
    cache = filled_cache(dtype=torch.float32)
    ids = [5, 1, 9]
    before = cache.pool[torch.tensor(ids)].numpy().tobytes()
    out = backend.start_out(cache, ids, False)
    poll_done(out)
    cache.pool[torch.tensor(ids)] = 0
    into = backend.start_in(cache, ids, out.payload)
    poll_done(into)
    assert cache.pool[torch.tensor(ids)].numpy().tobytes() == before


def test_plain_finishes_before_returning():
    cache = filled_cache()
    ticket = PlainCopyBackend().start_out(cache, [1, 2], False)
    assert ticket.done()
    assert isinstance(ticket, ReadyTicket)


def test_delayed_counts_polls_then_finishes_and_works_as_late_as_it_can():
    cache = filled_cache()
    backend = DelayedBackend(polls=2)
    before = crc_blocks(cache, [3])
    ticket = backend.start_out(cache, [3], False)
    assert [ticket.done(), ticket.done(), ticket.done(), ticket.done()] == [False, False, True, True]
    assert zlib.crc32(ticket.payload.memory.numpy().tobytes()) == before
    into = backend.start_in(cache, [4], ticket.payload)
    # The restore's blocks are unusable until the ticket says otherwise.
    assert (cache.pool[4].float() > 1e10).all()
    into.wait()
    assert crc_blocks(cache, [4]) == before


def test_delayed_accepts_a_poll_count_per_ticket():
    counts = iter([0, 5])
    backend = DelayedBackend(polls=lambda: next(counts))
    cache = filled_cache()
    assert backend.start_out(cache, [1], False).done()
    assert not backend.start_out(cache, [2], False).done()


def test_native_runs_out_of_order_with_a_paused_stream():
    cache = filled_cache()
    backend = native()
    try:
        before = crc_blocks(cache, [1, 7])
        backend.engine.set_paused(True)
        ticket = backend.start_out(cache, [1, 7], False)
        assert not ticket.done()
        backend.engine.set_paused(False)
        ticket.wait()
        assert ticket.done()
        assert zlib.crc32(ticket.payload.memory.numpy().tobytes()[: ticket.payload.nbytes]) == before
    finally:
        backend.close()


def test_native_fp8_slow_copies_do_not_let_a_later_kernel_overwrite_its_stage():
    # Segments share two stages. Only the event chain keeps a quantize kernel from overwriting
    # a stage the copy stream has not read yet.
    cache = filled_cache()
    reference = PlainCopyBackend().start_out(cache, list(range(16)), True).payload.memory
    backend = native(staging_blocks=1)
    try:
        backend.engine.set_copy_delay_us(1000)
        ticket = backend.start_out(cache, list(range(16)), True)
        ticket.wait()
        assert ticket.payload.memory[: ticket.payload.nbytes].equal(reference)
        target = PagedKVCache(CFG, 32, dtype=torch.bfloat16)
        plain = PagedKVCache(CFG, 32, dtype=torch.bfloat16)
        PlainCopyBackend().start_in(plain, list(range(16)), ticket.payload)
        restore = backend.start_in(target, list(range(16)), ticket.payload)
        restore.wait()
        assert bits_of(target.pool) == bits_of(plain.pool)
    finally:
        backend.close()


def test_native_two_overlapping_offloads_in_both_formats():
    cache = filled_cache()
    backend = native()
    try:
        backend.engine.set_paused(True)
        first = backend.start_out(cache, [1, 2, 3, 4, 5, 6, 7], False)
        second = backend.start_out(cache, [8, 9, 10, 11], True)
        third = backend.start_out(cache, [12, 13], False)
        assert not first.done() and not second.done() and not third.done()
        backend.engine.set_paused(False)
        for ticket in (third, second, first):
            ticket.wait()
        plain = PlainCopyBackend()
        assert first.payload.memory.equal(plain.start_out(cache, [1, 2, 3, 4, 5, 6, 7], False).payload.memory[: first.payload.nbytes]) or (
            first.payload.memory[: first.payload.nbytes].equal(plain.start_out(cache, [1, 2, 3, 4, 5, 6, 7], False).payload.memory)
        )
        assert second.payload.memory[: second.payload.nbytes].equal(plain.start_out(cache, [8, 9, 10, 11], True).payload.memory)
        assert third.payload.memory[: third.payload.nbytes].equal(plain.start_out(cache, [12, 13], False).payload.memory)
    finally:
        backend.close()


def test_native_empty_block_list_is_ready_at_once():
    cache = filled_cache()
    backend = native()
    try:
        ticket = backend.start_out(cache, [], False)
        assert ticket.done() and ticket.payload.num_blocks == 0
        assert backend.start_in(cache, [], ticket.payload).done()
    finally:
        backend.close()


def test_native_close_mid_flight_waits_and_invalidates_payloads():
    cache = filled_cache()
    backend = native()
    backend.engine.set_copy_delay_us(2000)
    ticket = backend.start_out(cache, list(range(16)), False)
    assert not ticket.done()
    backend.close()
    assert ticket.done()
    assert ticket.payload.memory is None
    backend.close()


def test_native_refuses_a_pool_on_another_device():
    backend = native()
    try:
        with pytest.raises(ValueError):
            NativeBackend("mps")
        cache = filled_cache()
        cache.device = torch.device("cuda", 0)
        with pytest.raises(ValueError, match="this backend drives"):
            backend.start_out(cache, [0], False)
    finally:
        backend.close()


def test_native_prepare_does_first_use_work_and_keeps_a_zero_pool_zero():
    cache = PagedKVCache(CFG, 4, dtype=torch.bfloat16)
    backend = native()
    try:
        backend.prepare(cache, fp8=False)
        backend.prepare(cache, fp8=True)
        assert (cache.pool == 0).all()
        assert backend._stages
    finally:
        backend.close()


def test_native_fp8_staging_must_hold_one_record():
    cache = filled_cache()
    backend = NativeBackend("cpu", staging_bytes=64, ids_capacity=16)
    try:
        with pytest.raises(ValueError, match="cannot hold one"):
            backend.start_out(cache, [0], True)
    finally:
        backend.close()


def test_make_backend_builds_by_name():
    assert isinstance(make_backend("plain"), PlainCopyBackend)
    created = make_backend("native", "cpu", staging_bytes=1 << 20)
    assert isinstance(created, NativeBackend) and created.engine.staging_bytes == 1 << 20
    created.close()
    with pytest.raises(ValueError):
        make_backend("rocket")


def test_modelled_copies_finish_on_the_clock_one_after_another():
    now = [10.0]
    backend = ModelledBackend(lambda: now[0], lambda tokens: tokens * 0.001, lambda tokens: tokens * 0.002)
    cache = filled_cache(8)
    before = crc_blocks(cache, [1, 2])
    out = backend.start_out(cache, [1, 2], False)
    second = backend.start_out(cache, [3], False)
    assert out.ready_at == pytest.approx(10.032) and second.ready_at == pytest.approx(10.048)
    assert not out.done()
    now[0] = 10.04
    assert out.done() and not second.done()
    into = backend.start_in(cache, [4, 5], out.payload)
    assert into.ready_at == pytest.approx(10.048 + 32 * 0.002)
    assert crc_blocks(cache, [4, 5]) == before
    second.wait()
    assert second.done()


def test_native_refuses_a_block_id_outside_the_pool():
    cache = filled_cache(8)
    backend = native()
    try:
        for bad in ([8], [0, -1]):
            with pytest.raises(ValueError, match="outside a pool"):
                backend.start_out(cache, bad, False)
            with pytest.raises(ValueError, match="outside a pool"):
                backend.start_in_per_block(cache, bad, None)
    finally:
        backend.close()


@pytest.mark.parametrize("fp8", [False, True], ids=["bf16", "fp8"])
def test_per_block_baseline_round_trips_and_copies_once_per_record(fp8):
    cache = filled_cache(16)
    original = cache.pool.clone()
    backend = native()
    try:
        ids, target = scattered(5, 8), [8 + i for i in scattered(5, 8, seed=2)]
        out = backend.start_out_per_block(cache, ids, fp8)
        poll_done(out)
        into = backend.start_in_per_block(cache, target, out.payload)
        poll_done(into)
        if fp8:
            layout = Fp8Layout.from_cache(cache)
            assert_fp8_close(original[torch.tensor(ids)], cache.pool[torch.tensor(target)], list(range(5)), layout)
        else:
            assert crc_blocks(cache, target) == zlib.crc32(original[torch.tensor(ids)].view(torch.uint16).numpy().tobytes())
        assert backend.engine.stats()["memcpys"] == 10
    finally:
        backend.close()


def test_per_block_fp8_kernels_stay_off_the_calling_threads_stream(monkeypatch):
    """On torch's stream they would queue the interference benchmark's matmul behind them."""
    from layover.kv import transfer

    threads = []
    for name in ("quantize_blocks", "dequantize_blocks"):
        real = getattr(transfer, name)

        def spy(*args, real=real, **kwargs):
            threads.append(threading.get_ident())
            return real(*args, **kwargs)

        monkeypatch.setattr(transfer, name, spy)
    cache = filled_cache(8)
    backend = NativeBackend("cpu", staging_bytes=4 * cache.block_bytes, ids_capacity=64)
    try:
        before = cache.pool[[0, 1, 2]].clone()
        out = backend.start_out_per_block(cache, [0, 1, 2], True)
        out.wait()
        assert backend.start_in_per_block(cache, [3, 4, 5], out.payload).done()
        after = cache.pool[[3, 4, 5]]
        # The kernels had finished by the time each call returned, on the stream, not here.
        assert len(threads) == 2 and threading.get_ident() not in threads
        bits = before.view(torch.uint16).reshape(-1, Fp8Layout.from_cache(cache).group_elems)
        gap = (after.to(torch.float64) - before.to(torch.float64)).abs().reshape(bits.shape)
        assert (gap <= round_trip_bound(bits)).all()
        backend.free(out.payload)
    finally:
        backend.close()


def test_a_failed_pin_raises_its_own_error_before_any_work_is_queued():
    cache = filled_cache(4)
    backend = NativeBackend("cpu", staging_bytes=2 * cache.block_bytes, ids_capacity=64)
    try:
        backend.engine.set_pin_limit(0)
        for fp8 in (False, True):
            with pytest.raises(HostMemoryError):
                backend.start_out(cache, [0, 1], fp8)
        assert backend.pin_stats()["host_pin_failures"] == 2
        backend.engine.set_pin_limit(1 << 40)
        out = backend.start_out(cache, [0, 1], False)
        out.wait()
        backend.free(out.payload)
    finally:
        backend.close()


def test_a_backend_that_pins_nothing_reports_no_pins():
    assert PlainCopyBackend().pin_stats() == {}


def test_gather_bench_verifies_fp8_a_slice_at_a_time(monkeypatch):
    from layover.kernels import fp8

    bench = load_bench()
    cache = filled_cache(48)
    seen, real = [], fp8.round_trip_bound
    monkeypatch.setattr(fp8, "round_trip_bound", lambda bits: (seen.append(bits.numel()), real(bits))[1])
    bench.check_restored(cache, list(range(40)), list(range(40)), True)
    assert sum(seen) == cache.pool[:40].numel() and max(seen) * 2 <= sum(seen)


def test_gather_bench_keeps_finished_rows_when_a_late_phase_crashes(tmp_path, monkeypatch):
    bench = load_bench()

    def crash(*args):
        raise MemoryError("the host ran out in the last phase")

    monkeypatch.setattr(bench, "measure_interference", crash)
    with pytest.raises(MemoryError):
        bench.main(["--blocks", "2", "--repeats", "1", "--out", str(tmp_path)])
    saved = json.loads((tmp_path / "transfer_gather.json").read_text())
    assert saved["complete"] is False and saved["sweep"] and saved["interference"] == []
    assert (tmp_path / "transfer_gather.csv").stat().st_size > 0


def load_bench():
    path = Path(__file__).resolve().parents[1] / "bench" / "transfer_gather.py"
    spec = importlib.util.spec_from_file_location("transfer_gather", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["transfer_gather"] = module
    spec.loader.exec_module(module)
    return module


def test_gather_bench_on_cpu_is_labelled_a_pipeline_check_and_compares_both_paths(tmp_path, capsys):
    bench = load_bench()
    result = bench.main(
        ["--blocks", "2", "4", "--repeats", "2", "--interference-blocks", "2", "--interference-s", "0.1", "--matmul-size", "32", "--out", str(tmp_path)]
    )
    assert result["measured_on_gpu"] is False and "pipeline check" in result["note"]
    assert "pipeline check" in capsys.readouterr().out
    rows = result["sweep"]
    assert len(rows) == 2 * 3 * 2 * 2
    assert {(r["path"], r["format"]) for r in rows} == {(p, f) for p in bench.PATHS for f in bench.FORMATS}
    for row in rows:
        assert row["out_min_s"] <= row["out_s"] <= row["out_max_s"]
        assert row["in_min_s"] <= row["in_s"] <= row["in_max_s"]
        if row["path"] == "per_block":
            assert row["copies_per_round_trip"] == 2 * row["blocks"]
        else:
            assert row["native_kernels_per_round_trip"] == (0 if row["format"] == "fp8" else 2)
            assert row["copies_per_round_trip"] == 2
    assert {(e["path"], e["format"]) for e in result["interference"]} == {(p, f) for p in bench.PATHS for f in bench.FORMATS}
    assert all(e["busy_iterations"] > 0 and e["idle_iterations"] > 0 for e in result["interference"])
    saved = json.loads((tmp_path / "transfer_gather.json").read_text())
    assert saved["note"] == result["note"] and (tmp_path / "transfer_gather.csv").stat().st_size > 0


def test_gather_bench_stops_on_a_corrupted_round_trip():
    bench = load_bench()
    cache = filled_cache(8)
    with pytest.raises(RuntimeError, match="bytes changed"):
        bench.check_restored(cache, [0, 1], [2, 3], False)
    with pytest.raises(RuntimeError, match="error bound"):
        bench.check_restored(cache, [0, 1], [2, 3], True)


def test_gather_bench_refuses_a_gpu_it_does_not_have(tmp_path):
    if torch.cuda.is_available():
        pytest.skip("a GPU is present")
    with pytest.raises(SystemExit):
        load_bench().main(["--device", "cuda", "--out", str(tmp_path)])


# CUDA. Nothing below has run: this machine has no GPU.


def gpu_backend(staging_blocks=4):
    cache = PagedKVCache(CFG, 1, dtype=torch.bfloat16)
    return NativeBackend("cuda", staging_bytes=staging_blocks * cache.block_bytes, ids_capacity=512)


def gpu_cache(blocks=64):
    return filled_cache(blocks, device="cuda")


def gpu_crc(cache, ids):
    return zlib.crc32(cache.pool[torch.tensor(ids, device="cuda")].cpu().view(torch.uint16).numpy().tobytes())


@pytest.mark.gpu
def test_gpu_bf16_round_trip_scattered_blocks_is_byte_exact():
    cache = gpu_cache()
    backend = gpu_backend(staging_blocks=64)
    try:
        ids = scattered(40, 64)
        before = gpu_crc(cache, ids)
        out = backend.start_out(cache, ids, False)
        poll_done(out)
        assert zlib.crc32(out.payload.memory[: out.payload.nbytes].numpy().tobytes()) == before
        target = scattered(40, 64, seed=2)
        into = backend.start_in(cache, target, out.payload)
        poll_done(into)
        assert gpu_crc(cache, target) == before
        stats = backend.engine.stats()
        assert stats["memcpys"] == 2 and stats["kernels"] == 2
    finally:
        backend.close()


@pytest.mark.gpu
def test_gpu_multi_segment_transfer_bigger_than_staging():
    cache = gpu_cache()
    backend = gpu_backend(staging_blocks=3)
    try:
        ids = scattered(50, 64)
        before = gpu_crc(cache, ids)
        out = backend.start_out(cache, ids, False)
        poll_done(out)
        target = scattered(50, 64, seed=2)
        into = backend.start_in(cache, target, out.payload)
        poll_done(into)
        assert gpu_crc(cache, target) == before
        assert backend.engine.stats()["memcpys"] == 2 * 17
    finally:
        backend.close()


@pytest.mark.gpu
def test_gpu_two_overlapping_transfers():
    cache = gpu_cache()
    backend = gpu_backend(staging_blocks=3)
    try:
        first_ids, second_ids = scattered(30, 64), scattered(20, 64, seed=5)
        expected = gpu_crc(cache, first_ids), gpu_crc(cache, second_ids)
        first = backend.start_out(cache, first_ids, False)
        second = backend.start_out(cache, second_ids, False)
        poll_done(second)
        poll_done(first)
        assert zlib.crc32(first.payload.memory[: first.payload.nbytes].numpy().tobytes()) == expected[0]
        assert zlib.crc32(second.payload.memory[: second.payload.nbytes].numpy().tobytes()) == expected[1]
    finally:
        backend.close()


@pytest.mark.gpu
def test_gpu_fp8_round_trip_within_the_bound_and_matches_the_reference():
    cache = gpu_cache()
    layout = Fp8Layout.from_cache(cache)
    backend = gpu_backend(staging_blocks=3)
    try:
        backend.prepare(PagedKVCache(CFG, 2, "cuda", torch.bfloat16), fp8=True)
        ids = scattered(30, 64)
        host_pool = cache.pool.cpu()
        reference = torch.zeros(30 * layout.record_bytes, dtype=torch.uint8)
        from layover.kernels.fp8 import quantize_blocks

        quantize_blocks(host_pool, torch.tensor(ids, dtype=torch.int32), reference, layout, impl="torch")
        out = backend.start_out(cache, ids, True)
        poll_done(out)
        assert out.payload.memory[: out.payload.nbytes].equal(reference)
        cache.pool.zero_()
        # A restore's blocks must be idle, which a queued zero_ on torch's stream is not.
        torch.cuda.current_stream().synchronize()
        into = backend.start_in(cache, ids, out.payload)
        poll_done(into)
        assert_fp8_close(host_pool, cache.pool.cpu(), ids, layout)
    finally:
        backend.close()


@pytest.mark.gpu
def test_gpu_tier_round_trip_through_native_backend():
    cache = gpu_cache()
    backend = gpu_backend(staging_blocks=3)
    tier = CpuTier(64, backend)
    try:
        table = BlockTable(cache.allocator.alloc(20), 20 * 16)
        before = gpu_crc(cache, table.blocks)
        handle = tier.offload(cache, table)
        restored = tier.restore(cache, handle)
        assert gpu_crc(cache, restored.blocks) == before
        assert tier.used_blocks == 0
    finally:
        backend.close()


@pytest.mark.gpu
@pytest.mark.parametrize("fp8", [False, True], ids=["bf16", "fp8"])
def test_gpu_close_mid_flight_waits(fp8):
    cache = gpu_cache()
    backend = gpu_backend(staging_blocks=3)
    ticket = backend.start_out(cache, scattered(50, 64), fp8)
    backend.close()
    assert ticket.done()


@pytest.mark.gpu
def test_gpu_decode_writes_queued_on_torchs_stream_reach_the_gather():
    """Without the wait in start_out, a gather can overtake a queued write on another stream."""
    cache = gpu_cache()
    backend = gpu_backend(staging_blocks=8)
    try:
        for round_number in range(20):
            block = round_number % 16
            noise = torch.randn(cache.pool[block].shape, device="cuda", dtype=torch.float32)
            # Big enough to still be running when the gather starts.
            cache.pool[block] = (noise @ noise.transpose(-1, -2) @ noise).to(torch.bfloat16)
            out = backend.start_out(cache, [block], False)
            poll_done(out)
            assert zlib.crc32(out.payload.memory[: out.payload.nbytes].numpy().tobytes()) == gpu_crc(cache, [block])
            backend.free(out.payload)
    finally:
        backend.close()


@pytest.mark.gpu
def test_gpu_real_qwen3_4b_block_past_4_gib_round_trips():
    """The real block size at ids whose byte offsets do not fit in 32 bits."""
    cfg = ModelConfig.micro(num_hidden_layers=36, num_key_value_heads=8, num_attention_heads=8, head_dim=128)
    cache = PagedKVCache(cfg, 2048, "cuda", torch.bfloat16)
    assert cache.block_bytes == 2_359_296
    backend = NativeBackend("cuda", staging_bytes=8 * cache.block_bytes)
    try:
        ids, target = [2047, 2000, 1900, 1], [1950, 1820, 2, 2040]
        assert ids[0] * cache.block_bytes > 1 << 32
        for block in ids:
            cache.pool[block] = torch.randn(cache.pool[block].shape, device="cuda", dtype=torch.float32).to(torch.bfloat16)
        before = gpu_crc(cache, ids)
        out = backend.start_out(cache, ids, False)
        poll_done(out)
        into = backend.start_in(cache, target, out.payload)
        poll_done(into)
        assert gpu_crc(cache, target) == before
        fp8_out = backend.start_out(cache, ids, True)
        poll_done(fp8_out)
        layout = Fp8Layout.from_cache(cache)
        fp8_into = backend.start_in(cache, target, fp8_out.payload)
        poll_done(fp8_into)
        assert_fp8_close(cache.pool[torch.tensor(ids, device="cuda")].cpu(), cache.pool[torch.tensor(target, device="cuda")].cpu(), [0, 1, 2, 3], layout)
    finally:
        backend.close()


@pytest.mark.gpu
def test_gpu_gather_bench_runs_and_is_labelled_as_measured(tmp_path):
    result = load_bench().main(
        ["--device", "cuda", "--blocks", "16", "--repeats", "2", "--interference-blocks", "16", "--interference-s", "1", "--out", str(tmp_path)]
    )
    assert result["measured_on_gpu"] is True and result["note"] is None
