"""The native transfer library: the host build on any machine, the CUDA build with --gpu."""

import ctypes
import importlib.util
import mmap
import random
import subprocess
import sys
import time
import zlib
from pathlib import Path

import numpy as np
import pytest
import torch

from layover.kv.native import NATIVE_DIR, Engine, HostMemoryError, NativeError, NativeUnavailable, build_host, library_path, load_library

BLOCK = 8192
ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module", autouse=True)
def library():
    build_host()


@pytest.fixture
def engine():
    created = Engine("host", staging_bytes=2 * BLOCK, ids_capacity=64, host_cache_bytes=1 << 24)
    yield created
    created.destroy()


def filled_pool(blocks, block_bytes=BLOCK, seed=0):
    generator = torch.Generator().manual_seed(seed)
    return torch.randint(0, 256, (blocks * block_bytes,), dtype=torch.uint8, generator=generator)


def host_bytes(pointer, nbytes):
    return bytes((ctypes.c_uint8 * nbytes).from_address(pointer))


def blocks_of(pool, ids, block_bytes=BLOCK):
    return b"".join(pool[i * block_bytes : (i + 1) * block_bytes].numpy().tobytes() for i in ids)


def round_trip(engine, pool, ids, out=None, into=None, block_bytes=BLOCK):
    out = out or engine.out_gather
    into = into or engine.in_scatter
    before = zlib.crc32(blocks_of(pool, ids, block_bytes))
    host = engine.host_alloc(len(ids) * block_bytes)
    transfer = out(pool.data_ptr(), block_bytes, ids, host)
    engine.wait(transfer)
    engine.release(transfer)
    assert zlib.crc32(host_bytes(host, len(ids) * block_bytes)) == before
    for i in ids:
        pool[i * block_bytes : (i + 1) * block_bytes] = 0
    transfer = into(pool.data_ptr(), block_bytes, ids, host)
    engine.wait(transfer)
    engine.release(transfer)
    assert zlib.crc32(blocks_of(pool, ids, block_bytes)) == before
    engine.host_free(host)


def test_flavor_and_abi_are_checked():
    lib = load_library("host")
    assert lib.lt_flavor() == b"host"
    with pytest.raises(NativeUnavailable):
        load_library("no-such-flavor")


def test_build_is_keyed_by_platform_and_flavor():
    path = library_path("host")
    assert path.exists() and "host-" in path.parent.name
    assert library_path("cuda").parent.name.startswith("cuda-")


@pytest.mark.parametrize("ids", [[0], [7, 2, 5], [3, 0, 7, 1, 6, 2, 5, 4]], ids=["one", "few", "all_shuffled"])
def test_gather_and_scatter_round_trip_scattered_blocks(engine, ids):
    round_trip(engine, filled_pool(8), ids)


def test_host_order_follows_the_id_list_and_other_blocks_stay_put(engine):
    pool = filled_pool(8)
    original = pool.clone()
    ids = [6, 1, 4]
    host = engine.host_alloc(3 * BLOCK)
    transfer = engine.out_gather(pool.data_ptr(), BLOCK, ids, host)
    engine.wait(transfer)
    engine.release(transfer)
    assert host_bytes(host, 3 * BLOCK) == blocks_of(original, ids)
    pool[:] = 0
    transfer = engine.in_scatter(pool.data_ptr(), BLOCK, [2, 5, 0], host)
    engine.wait(transfer)
    engine.release(transfer)
    assert blocks_of(pool, [2, 5, 0]) == blocks_of(original, ids)
    assert blocks_of(pool, [1, 3, 4, 6, 7]) == b"\0" * (5 * BLOCK)


@pytest.mark.parametrize(
    "block_bytes",
    [64, 8, 2048, 65536, 65536 + 8, 262144, 2_359_296 // 16],
    ids=lambda b: f"{b}B",
)
def test_block_sizes_around_chunk_and_lane_boundaries(block_bytes):
    engine = Engine("host", staging_bytes=4 * block_bytes, ids_capacity=8)
    try:
        round_trip(engine, filled_pool(5, block_bytes), [3, 0, 4], block_bytes=block_bytes)
    finally:
        engine.destroy()


def test_transfer_larger_than_staging_splits_into_segments(engine):
    # Two blocks fit a staging buffer, so nine need five segments in each direction.
    pool = filled_pool(16)
    before = engine.stats()
    round_trip(engine, pool, [15, 3, 8, 0, 12, 1, 9, 6, 2])
    after = engine.stats()
    assert after["kernels"] - before["kernels"] == 10
    assert after["memcpys"] - before["memcpys"] == 10
    assert after["ids_uploads"] - before["ids_uploads"] == 2
    assert after["bytes_to_host"] - before["bytes_to_host"] == 9 * BLOCK
    assert after["bytes_to_device"] - before["bytes_to_device"] == 9 * BLOCK


def test_one_dma_per_segment_and_not_per_block(engine):
    wide = Engine("host", staging_bytes=64 * BLOCK, ids_capacity=64)
    try:
        round_trip(wide, filled_pool(64), list(range(64)))
        stats = wide.stats()
        assert stats["memcpys"] == 2 and stats["kernels"] == 2
    finally:
        wide.destroy()


def test_rejected_baseline_copies_once_per_block_and_is_also_correct(engine):
    pool = filled_pool(8)
    round_trip(engine, pool, [7, 2, 5, 0], engine.out_per_block, engine.in_per_block)
    stats = engine.stats()
    assert stats["memcpys"] == 8 and stats["kernels"] == 0 and stats["ids_uploads"] == 0


def test_slow_copies_do_not_let_a_later_gather_overwrite_staging(engine):
    # With a slow copy stream the gather for segment 3 gets to slot 0 long before segment 1's
    # DMA has read it. Only the wait on slot.released keeps those bytes intact.
    engine.set_copy_delay_us(2000)
    round_trip(engine, filled_pool(16), [15, 3, 8, 0, 12, 1, 9, 6, 2, 10, 4, 14])


def test_two_overlapping_transfers_finish_correct(engine):
    pool = filled_pool(16)
    original = pool.clone()
    engine.set_paused(True)
    first_ids, second_ids = [1, 9, 3, 11, 5], [14, 0, 7]
    first_host = engine.host_alloc(5 * BLOCK)
    second_host = engine.host_alloc(3 * BLOCK)
    first = engine.out_gather(pool.data_ptr(), BLOCK, first_ids, first_host)
    second = engine.out_gather(pool.data_ptr(), BLOCK, second_ids, second_host)
    assert not engine.poll(first) and not engine.poll(second)
    engine.set_paused(False)
    engine.wait(second)
    engine.wait(first)
    assert host_bytes(first_host, 5 * BLOCK) == blocks_of(original, first_ids)
    assert host_bytes(second_host, 3 * BLOCK) == blocks_of(original, second_ids)


def test_poll_does_not_block_and_turns_true_on_completion(engine):
    pool = filled_pool(4)
    engine.set_paused(True)
    host = engine.host_alloc(4 * BLOCK)
    transfer = engine.out_gather(pool.data_ptr(), BLOCK, [0, 1, 2, 3], host)
    start = time.perf_counter()
    assert engine.poll(transfer) is False
    assert time.perf_counter() - start < 0.5
    with pytest.raises(NativeError, match="in flight"):
        engine.release(transfer)
    engine.set_paused(False)
    deadline = time.monotonic() + 30
    while not engine.poll(transfer):
        assert time.monotonic() < deadline
    engine.release(transfer)
    with pytest.raises(NativeError, match="not a live transfer"):
        engine.poll(transfer)


def test_destroy_mid_flight_waits_and_frees_everything():
    engine = Engine("host", staging_bytes=2 * BLOCK, ids_capacity=64)
    pool = filled_pool(8)
    engine.set_copy_delay_us(5000)
    host = engine.host_alloc(8 * BLOCK)
    data = (ctypes.c_uint8 * (8 * BLOCK)).from_address(host)
    transfer = engine.out_gather(pool.data_ptr(), BLOCK, list(range(8)), host)
    assert not engine.poll(transfer)
    engine.destroy()
    assert engine.closed
    engine.destroy()
    with pytest.raises(NativeError, match="closed"):
        engine.stats()
    del data


def test_destroy_while_paused_does_not_hang():
    engine = Engine("host", staging_bytes=2 * BLOCK, ids_capacity=64)
    pool = filled_pool(4)
    engine.set_paused(True)
    host = engine.host_alloc(4 * BLOCK)
    engine.out_gather(pool.data_ptr(), BLOCK, [0, 1, 2, 3], host)
    engine.destroy()


def test_id_ring_wraps_and_reclaims_across_many_transfers():
    # Far more ids in flight, over the run, than the ring holds.
    engine = Engine("host", staging_bytes=8 * BLOCK, ids_capacity=8)
    try:
        pool = filled_pool(8)
        for round_index in range(40):
            ids = list(np.random.default_rng(round_index).permutation(8)[: 1 + round_index % 8])
            round_trip(engine, pool, [int(i) for i in ids])
    finally:
        engine.destroy()


def test_full_id_ring_applies_backpressure_instead_of_failing():
    engine = Engine("host", staging_bytes=8 * BLOCK, ids_capacity=8)
    try:
        pool = filled_pool(8)
        original = pool.clone()
        engine.set_copy_delay_us(1000)
        hosts = [engine.host_alloc(6 * BLOCK) for _ in range(5)]
        ids = [[(i + j) % 8 for j in range(6)] for i in range(5)]
        transfers = [engine.out_gather(pool.data_ptr(), BLOCK, row, host) for row, host in zip(ids, hosts)]
        for transfer in transfers:
            engine.wait(transfer)
        for row, host in zip(ids, hosts):
            assert host_bytes(host, 6 * BLOCK) == blocks_of(original, row)
    finally:
        engine.destroy()


def test_transfers_address_the_pool_with_64_bits():
    """A Qwen3-4B block at an id past 4 GiB, in a mapping only its pages of which get touched."""
    block = 2_359_296
    pool_blocks, source, target = 2400, 2399, 2100
    assert source * block > 1 << 32
    mapped = mmap.mmap(-1, pool_blocks * block, flags=mmap.MAP_PRIVATE | getattr(mmap, "MAP_NORESERVE", 0))
    pointer = ctypes.addressof(ctypes.c_char.from_buffer(mapped))
    pattern = bytes(random.Random(0).getrandbits(8) for _ in range(4096)) * (block // 4096)
    mapped[source * block : (source + 1) * block] = pattern
    engine = Engine("host", staging_bytes=2 * block, ids_capacity=8)
    try:
        host = engine.host_alloc(block)
        transfer = engine.out_gather(pointer, block, [source], host)
        engine.wait(transfer)
        engine.release(transfer)
        assert host_bytes(host, block) == pattern
        transfer = engine.in_scatter(pointer, block, [target], host)
        engine.wait(transfer)
        engine.release(transfer)
        assert mapped[target * block : (target + 1) * block] == pattern
        # A 32-bit product would have landed here instead.
        wrapped = (target * block) % (1 << 32)
        assert mapped[wrapped : wrapped + 4096] == bytes(4096)
    finally:
        engine.destroy()


def test_segments_stay_under_the_cuda_grid_limit():
    # 131,072 eight-byte blocks would fit in staging, but a launch's y dimension stops at 65,535.
    engine = Engine("host", staging_bytes=1 << 20, ids_capacity=1 << 18)
    try:
        count = 70_000
        pool = torch.arange(count, dtype=torch.int64)
        host = engine.host_alloc(8 * count)
        transfer = engine.out_gather(pool.data_ptr(), 8, list(range(count)), host)
        engine.wait(transfer)
        engine.release(transfer)
        assert host_bytes(host, 8 * count) == pool.numpy().tobytes()
        stats = engine.stats()
        assert stats["kernels"] == 2 and stats["memcpys"] == 2
    finally:
        engine.destroy()


@pytest.mark.parametrize(
    "request_bytes,pinned",
    [(1, 1 << 20), (1 << 20, 1 << 20), ((1 << 20) + 1, 5 << 18), (40 << 20, 40 << 20), ((40 << 20) + 1, 48 << 20)],
    ids=["tiny", "smallest_class", "just_over", "on_a_class", "just_over_a_class"],
)
def test_pinned_allocations_waste_at_most_a_quarter(request_bytes, pinned):
    engine = Engine("host", staging_bytes=1 << 20, ids_capacity=8)
    try:
        pointer = engine.host_alloc(request_bytes)
        assert engine.stats()["host_bytes_live"] == pinned
        assert pinned <= max(1 << 20, request_bytes * 5 // 4)
        engine.host_free(pointer)
    finally:
        engine.destroy()


def test_host_buffers_are_reused_and_accounted(engine):
    first = engine.host_alloc(1000)
    held = engine.stats()["host_bytes_held"]
    engine.host_free(first)
    assert engine.stats()["host_bytes_live"] == 0
    assert engine.host_alloc(2000) == first
    assert engine.stats()["host_bytes_held"] == held
    with pytest.raises(NativeError, match="not a live host buffer"):
        engine.host_free(12345)


def test_cache_limit_returns_memory_to_the_system():
    engine = Engine("host", staging_bytes=BLOCK, ids_capacity=8, host_cache_bytes=0)
    try:
        pointer = engine.host_alloc(1000)
        engine.host_free(pointer)
        assert engine.stats()["host_bytes_held"] == 0
    finally:
        engine.destroy()


@pytest.mark.parametrize(
    "call,match",
    [
        (lambda e, p, h: e.out_gather(p, 16 * BLOCK, [0], h), "larger than a staging"),
        (lambda e, p, h: e.out_gather(p + 1, BLOCK, [0], h), "8-byte aligned"),
        (lambda e, p, h: e.out_gather(p, BLOCK - 4, [0], h), "multiple of 8"),
        (lambda e, p, h: e.out_gather(p, BLOCK, list(range(65)), h), "ids_capacity"),
        (lambda e, p, h: e.in_scatter(None, BLOCK, [0], h), "null pool"),
        (lambda e, p, h: e.ext_copy_out(p, h, BLOCK, 99), "bad slot"),
    ],
    ids=["block_too_big", "misaligned_pool", "odd_block", "too_many_ids", "null_pool", "bad_slot"],
)
def test_misuse_raises_with_the_reason_and_engine_stays_usable(engine, call, match):
    pool = filled_pool(2)
    host = engine.host_alloc(2 * BLOCK)
    with pytest.raises(NativeError, match=match):
        call(engine, pool.data_ptr(), host)
    round_trip(engine, filled_pool(4), [1, 3])


def test_bad_construction_is_refused():
    with pytest.raises(NativeError):
        Engine("host", staging_bytes=0)


def test_failed_kernel_is_surfaced_and_stops_further_work():
    engine = Engine("host", staging_bytes=BLOCK, ids_capacity=8)
    stream = engine.host_stream()

    def kernel():
        raise ValueError("boom")

    keep = engine.host_call(stream, kernel)
    transfer = engine.ext_finish(stream)
    with pytest.raises(NativeError, match="boom"):
        for _ in range(100_000):
            engine.poll(transfer)
    with pytest.raises(NativeError):
        engine.host_alloc(16)
    with pytest.raises(NativeError, match="host kernel failed"):
        engine.destroy()
    assert engine.closed
    del keep


def test_a_null_stream_is_refused_instead_of_meaning_the_legacy_default(engine):
    with pytest.raises(NativeError, match="null stream"):
        engine.ext_wait(None, 0, 0)
    with pytest.raises(NativeError, match="null stream"):
        engine.ext_record(None, 0, 0)


def test_c_checks_pass_under_address_and_undefined_sanitizers():
    done = subprocess.run(["make", "-C", str(NATIVE_DIR), "check"], capture_output=True, text=True, timeout=300)
    assert done.returncode == 0, done.stderr[-3000:] + done.stdout[-3000:]
    assert "host_check: ok" in done.stdout


def test_sanitize_driver_passes_on_cpu():
    done = subprocess.run(
        [sys.executable, str(ROOT / "tests" / "sanitize_transfer.py"), "--device", "cpu"],
        capture_output=True,
        text=True,
        timeout=300,
        cwd=ROOT,
    )
    assert done.returncode == 0, done.stderr[-3000:]
    assert "ok" in done.stdout


def test_sanitize_driver_exits_non_zero_on_mismatch():
    done = subprocess.run(
        [sys.executable, str(ROOT / "tests" / "sanitize_transfer.py"), "--device", "cpu", "--corrupt"],
        capture_output=True,
        text=True,
        timeout=300,
        cwd=ROOT,
    )
    assert done.returncode != 0
    assert "mismatch" in done.stderr


# The CUDA build. Compile-checked without a GPU; nothing below can run without one.


@pytest.mark.gpu
def test_gpu_a_host_pointer_passed_as_the_pool_is_refused(gpu_engine):
    pool = filled_pool(4)
    host = gpu_engine.host_alloc(BLOCK)
    with pytest.raises(NativeError, match="not memory of CUDA device"):
        gpu_engine.out_gather(pool.data_ptr(), BLOCK, [1], host)
    gpu_engine.host_free(host)


@pytest.fixture
def gpu_engine():
    subprocess.run(["make", "-C", str(NATIVE_DIR), "cuda"], check=True)
    torch.cuda.synchronize()
    created = Engine("cuda", staging_bytes=4 * BLOCK, ids_capacity=64)
    yield created
    created.destroy()


def gpu_pool(blocks, block_bytes=BLOCK):
    return filled_pool(blocks, block_bytes).to("cuda")


@pytest.mark.gpu
def test_gpu_flavor_is_the_cuda_build(gpu_engine):
    assert load_library("cuda").lt_flavor() == b"cuda"


@pytest.mark.gpu
def test_gpu_gather_scatter_round_trip_is_byte_exact(gpu_engine):
    pool = gpu_pool(32)
    ids = [int(i) for i in np.random.default_rng(0).permutation(32)[:24]]
    before = zlib.crc32(blocks_of(pool.cpu(), ids))
    host = gpu_engine.host_alloc(24 * BLOCK)
    transfer = gpu_engine.out_gather(pool.data_ptr(), BLOCK, ids, host)
    gpu_engine.wait(transfer)
    gpu_engine.release(transfer)
    assert zlib.crc32(host_bytes(host, 24 * BLOCK)) == before
    pool.zero_()
    # In-scatter writes on the library's stream, so nothing else may still be writing.
    torch.cuda.current_stream().synchronize()
    transfer = gpu_engine.in_scatter(pool.data_ptr(), BLOCK, ids, host)
    gpu_engine.wait(transfer)
    gpu_engine.release(transfer)
    assert zlib.crc32(blocks_of(pool.cpu(), ids)) == before
    assert gpu_engine.stats()["kernels"] == 12


@pytest.mark.gpu
def test_gpu_per_block_baseline_round_trip(gpu_engine):
    pool = gpu_pool(8)
    ids = [5, 0, 3]
    before = zlib.crc32(blocks_of(pool.cpu(), ids))
    host = gpu_engine.host_alloc(3 * BLOCK)
    transfer = gpu_engine.out_per_block(pool.data_ptr(), BLOCK, ids, host)
    gpu_engine.wait(transfer)
    gpu_engine.release(transfer)
    assert zlib.crc32(host_bytes(host, 3 * BLOCK)) == before


@pytest.mark.gpu
def test_gpu_destroy_mid_flight_waits():
    subprocess.run(["make", "-C", str(NATIVE_DIR), "cuda"], check=True)
    engine = Engine("cuda", staging_bytes=1 << 22, ids_capacity=4096)
    block = 1 << 20
    pool = gpu_pool(512, block)
    host = engine.host_alloc(512 * block)
    engine.out_gather(pool.data_ptr(), block, list(range(512)), host)
    engine.destroy()
    assert engine.closed
    del pool
    torch.cuda.synchronize()


# Pinned host memory: which buffers stay, and what a failed pin leaves behind.


MIB = 1 << 20


def test_a_full_host_cache_gives_up_what_it_has_held_longest(engine):
    """Contexts grow, so stale smaller sizes must not make every larger offload pin again."""
    old, kept = engine.host_alloc(4 * MIB), engine.host_alloc(5 * MIB)
    engine.host_free(old)
    engine.host_free(kept)
    # 9 MiB cached of 16: this one fits only if the oldest goes, and then the other can stay.
    large = engine.host_alloc(10 * MIB)
    engine.host_free(large)
    pins = engine.stats()["host_pins"]
    assert engine.host_alloc(10 * MIB) == large
    assert engine.host_alloc(5 * MIB) == kept
    assert engine.stats()["host_pins"] == pins
    engine.host_alloc(4 * MIB)
    assert engine.stats()["host_pins"] == pins + 1


def test_a_buffer_larger_than_the_cache_is_not_kept(engine):
    pointer = engine.host_alloc(20 * MIB)
    engine.host_free(pointer)
    assert engine.stats()["host_bytes_held"] == 0


def test_pins_are_counted_and_timed_and_a_cached_buffer_is_not_a_pin(engine):
    pointer = engine.host_alloc(MIB)
    stats = engine.stats()
    assert stats["host_pins"] == 1 and stats["host_pin_bytes"] == MIB
    assert stats["host_pin_ns"] > 0 and stats["host_pin_max_ns"] == stats["host_pin_ns"]
    engine.host_free(pointer)
    engine.host_alloc(MIB)
    assert engine.stats()["host_pins"] == 1


def test_a_failed_pin_is_its_own_error_and_leaves_the_engine_working(engine):
    engine.set_pin_limit(MIB)
    with pytest.raises(HostMemoryError):
        engine.host_alloc(2 * MIB)
    assert engine.stats()["host_pin_failures"] == 1
    assert engine.stats()["host_bytes_held"] == 0
    engine.set_pin_limit(1 << 40)
    original = filled_pool(4)
    host = engine.host_alloc(2 * BLOCK)
    transfer = engine.out_gather(original.data_ptr(), BLOCK, [2, 0], host)
    engine.wait(transfer)
    engine.release(transfer)
    assert host_bytes(host, 2 * BLOCK) == blocks_of(original, [2, 0])


def test_a_failed_pin_gives_back_cached_memory_before_giving_up(engine):
    engine.set_pin_limit(5 * MIB)
    engine.host_free(engine.host_alloc(4 * MIB))
    engine.host_alloc(3 * MIB)
    stats = engine.stats()
    assert stats["host_pin_failures"] == 0
    assert stats["host_bytes_held"] == 3 * MIB


def test_a_bad_request_is_not_mistaken_for_a_failed_pin(engine):
    with pytest.raises(NativeError) as raised:
        engine.host_alloc(0)
    assert not isinstance(raised.value, HostMemoryError)


def test_sanitize_driver_refuses_the_caching_allocator_on_cuda(capsys):
    path = ROOT / "tests" / "sanitize_transfer.py"
    spec = importlib.util.spec_from_file_location("sanitize_transfer_check", path)
    driver = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(driver)
    with pytest.raises(SystemExit) as refused:
        driver.require_uncached_allocator("cuda", {})
    assert refused.value.code != 0 and "PYTORCH_NO_CUDA_MEMORY_CACHING=1" in capsys.readouterr().err
    driver.require_uncached_allocator("cuda", {"PYTORCH_NO_CUDA_MEMORY_CACHING": "1"})
    driver.require_uncached_allocator("cpu", {})
