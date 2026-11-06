"""FP8 quantize and dequantize: the torch reference, and the Triton kernels against it."""

import json
import mmap
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from layover.config import ModelConfig
from layover.kernels import fp8
from layover.kernels.fp8 import Fp8Layout, decode_bits, encode_bits, group_shifts, round_trip_bound
from layover.kv import PagedKVCache

ALL_BF16 = torch.arange(65536, dtype=torch.int32).to(torch.uint16)
ALL_FP8 = torch.arange(256, dtype=torch.int32).to(torch.uint8)
CFG = ModelConfig.micro()
INF_BITS = {0x7F80, 0xFF80}


def torch_fp8(bits):
    return bits.view(torch.bfloat16).to(torch.float8_e4m3fn).view(torch.uint8)


def as_float(bits):
    return bits.view(torch.bfloat16).to(torch.float64)


def classify(bits):
    values = as_float(bits)
    return torch.isnan(values), torch.isinf(values), values.abs() > 464.0


def assert_encode_matches_torch(encoded):
    """Exact on everything FP8 can represent, and our documented choices elsewhere."""
    nan, inf, too_big = classify(ALL_BF16)
    reference = torch_fp8(ALL_BF16)
    in_range = ~nan & ~inf & ~too_big
    assert in_range.sum() > 30000
    assert encoded[in_range].equal(reference[in_range])
    assert encoded[nan].equal(reference[nan])
    # Torch saturates these to a finite 448. Ours stay visibly non-finite.
    sign = (ALL_BF16.to(torch.int32) >> 15).to(torch.uint8) << 7
    assert encoded[inf | too_big].equal(sign[inf | too_big] | fp8.FP8_NAN)


def test_encode_matches_torch_on_every_bf16_pattern():
    assert_encode_matches_torch(encode_bits(ALL_BF16, 0))


def test_decode_matches_torch_on_every_fp8_pattern():
    reference = ALL_FP8.view(torch.float8_e4m3fn).to(torch.bfloat16).view(torch.uint16)
    assert decode_bits(ALL_FP8, 0).equal(reference)


def test_ties_round_to_even():
    # 1.0625 is a tie between 1.0 (even) and 1.125 (odd); 1.1875 between 1.125 and 1.25 (even).
    values = torch.tensor([1.0625, 1.1875, 3 * 2.0**-10, 2.0**-10], dtype=torch.bfloat16)
    expected = torch.tensor([1.0, 1.25, 2 * 2.0**-9, 0.0], dtype=torch.float64)
    decoded = decode_bits(encode_bits(values.view(torch.uint16), 0), 0).view(torch.bfloat16).to(torch.float64)
    assert decoded.equal(expected)


def group_of(values):
    return torch.tensor(values, dtype=torch.int32).to(torch.uint16)


def round_trip(bits):
    shift = group_shifts(bits[None])[0]
    return shift, decode_bits(encode_bits(bits, shift), shift)


def assert_within_error(bits, restored, shift):
    """The module's stated bound, restated here so a change to its code cannot move the goalposts."""
    before, after = as_float(bits), as_float(restored)
    nan, inf, _ = classify(bits)
    assert torch.isnan(after[nan | inf]).all()
    raw = (bits.to(torch.int32) >> 7) & 0xFF
    subnormal_or_zero = raw == 0
    assert (after[subnormal_or_zero] == 0).all()
    assert ((restored[subnormal_or_zero].to(torch.int32) >> 15) == (bits[subnormal_or_zero].to(torch.int32) >> 15)).all()
    normal = ~subnormal_or_zero & ~nan & ~inf
    error = (after[normal] - before[normal]).abs()
    scaled_normal = raw[normal] - 127 - int(shift) >= -6
    magnitude = before[normal].abs()
    assert (error[scaled_normal] <= magnitude[scaled_normal] * 2.0**-4).all()
    floor = max(2.0 ** (int(shift) - 10), 2.0**-126)
    assert (error[~scaled_normal] <= floor).all()
    assert not torch.isnan(after[normal]).any() and not torch.isinf(after[normal]).any()


@settings(max_examples=300, deadline=None)
@given(st.lists(st.integers(0, 65535), min_size=1, max_size=128))
def test_round_trip_error_on_arbitrary_bits(patterns):
    bits = group_of(patterns)
    shift, restored = round_trip(bits)
    assert_within_error(bits, restored, shift)


@settings(max_examples=200, deadline=None)
@given(st.integers(-126, 127), st.lists(st.floats(1.0, 2.0, width=32), min_size=1, max_size=64), st.integers(0, 30))
def test_round_trip_error_across_magnitudes(exponent, significands, spread):
    # Values around one exponent, the case the per-group scale is for.
    values = [s * 2.0 ** max(exponent - (i % (spread + 1)), -126) for i, s in enumerate(significands)]
    bits = torch.tensor(values, dtype=torch.float32).to(torch.bfloat16).view(torch.uint16)
    finite = ~torch.isinf(as_float(bits))
    bits = bits[finite]
    if bits.numel():
        shift, restored = round_trip(bits)
        assert_within_error(bits, restored, shift)


def test_the_scale_extends_range_far_beyond_plain_fp8():
    bits = torch.tensor([6500.0, -3000.0, 1.0], dtype=torch.bfloat16).view(torch.uint16)
    shift, restored = round_trip(bits)
    assert shift == 12 - 7
    assert as_float(restored)[:2].tolist() == pytest.approx([6500.0, -3000.0], rel=2.0**-4)
    # Tiny groups gain precision the same way.
    bits = torch.tensor([3e-30, 1e-30], dtype=torch.bfloat16).view(torch.uint16)
    shift, restored = round_trip(bits)
    assert shift < -100
    assert as_float(restored).tolist() == pytest.approx(as_float(bits).tolist(), rel=2.0**-4)


def test_scale_leaves_room_for_rounding_at_the_top_of_bf16():
    largest = group_of([fp8.BF16_MAX_FINITE])
    shift, restored = round_trip(largest)
    # The carry would make 2**128: it saturates instead of becoming infinity.
    assert restored.tolist() == [fp8.BF16_MAX_FINITE]
    assert shift == 127 - 7


def test_non_finite_values_stay_non_finite_and_do_not_set_the_scale():
    bits = torch.tensor([float("inf"), float("-inf"), float("nan"), 1.5, -0.25], dtype=torch.bfloat16).view(torch.uint16)
    shift, restored = round_trip(bits)
    after = as_float(restored)
    assert after[:3].isnan().all()
    assert shift == 0 - 7
    assert after[3:].tolist() == [1.5, -0.25]
    assert encode_bits(group_of([0x7F80]), 0).tolist() == [fp8.FP8_NAN]


def test_all_zero_and_subnormal_groups_round_trip_to_zero_keeping_sign():
    bits = group_of([0x0000, 0x8000, 0x0001, 0x807F])
    shift, restored = round_trip(bits)
    assert shift == 0
    assert restored.tolist() == [0x0000, 0x8000, 0x0000, 0x8000]


def random_pool_cache(cfg=CFG, blocks=8, seed=0):
    cache = PagedKVCache(cfg, blocks, dtype=torch.bfloat16)
    generator = torch.Generator().manual_seed(seed)
    scale = torch.exp(torch.randn(cache.pool.shape, generator=generator, dtype=torch.float32) * 2)
    cache.pool.copy_((torch.randn(cache.pool.shape, generator=generator) * scale).to(torch.bfloat16))
    return cache


def test_record_layout_is_half_of_bf16_plus_a_small_header():
    cfg = ModelConfig.from_hf_dict(
        {
            "vocab_size": 1, "hidden_size": 2560, "intermediate_size": 1, "num_hidden_layers": 36,
            "num_attention_heads": 32, "num_key_value_heads": 8, "head_dim": 128, "max_position_embeddings": 1,
        }
    )
    layout = Fp8Layout(16 * cfg.head_dim, 2 * cfg.num_hidden_layers * cfg.num_key_value_heads)
    bf16_bytes = 16 * cfg.kv_bytes_per_token("bf16")
    assert bf16_bytes == 2_359_296
    assert layout.block_elems == bf16_bytes // 2
    assert layout.header_bytes == 1152
    assert layout.record_bytes < 0.501 * bf16_bytes


def test_torch_blocks_round_trip_in_scattered_order():
    cache = random_pool_cache()
    layout = Fp8Layout.from_cache(cache)
    original = cache.pool.clone()
    ids = torch.tensor([5, 0, 3], dtype=torch.int32)
    stage = torch.zeros(3 * layout.record_bytes + 16, dtype=torch.uint8)
    fp8.quantize_blocks(cache.pool, ids, stage, layout)
    assert stage[-16:].equal(torch.zeros(16, dtype=torch.uint8))
    cache.pool.zero_()
    fp8.dequantize_blocks(stage, ids, cache.pool, layout)
    untouched = torch.tensor([1, 2, 4, 6, 7])
    assert (cache.pool[untouched] == 0).all()
    bits = original.view(torch.uint16)
    restored = cache.pool.view(torch.uint16)
    for block, record in zip(ids.tolist(), range(3)):
        start = record * layout.record_bytes + layout.block_elems
        shifts = stage[start : start + layout.header_bytes].clone().view(torch.int16)
        groups = bits[block].reshape(-1, layout.group_elems)
        assert shifts.to(torch.int32).equal(group_shifts(groups))
        for group, out, shift in zip(groups, restored[block].reshape(-1, layout.group_elems), shifts):
            assert_within_error(group, out, shift)


def test_prefix_of_records_restores_only_those_blocks():
    cache = random_pool_cache()
    layout = Fp8Layout.from_cache(cache)
    stage = torch.zeros(4 * layout.record_bytes, dtype=torch.uint8)
    fp8.quantize_blocks(cache.pool, torch.tensor([1, 2, 3, 4], dtype=torch.int32), stage, layout)
    other = random_pool_cache(seed=9)
    untouched = other.pool[:6].clone()
    fp8.dequantize_blocks(stage, torch.tensor([6, 7], dtype=torch.int32), other.pool, layout)
    assert other.pool[:6].view(torch.uint16).equal(untouched.view(torch.uint16))
    bits = cache.pool[1].view(torch.uint16).reshape(-1, layout.group_elems)
    restored = other.pool[6].view(torch.uint16).reshape(-1, layout.group_elems)
    for group, out in zip(bits, restored):
        assert_within_error(group, out, group_shifts(group[None])[0])


def test_a_non_bf16_pool_is_refused():
    cache = PagedKVCache(CFG, 2, dtype=torch.float32)
    layout = Fp8Layout.from_cache(cache)
    stage = torch.zeros(layout.record_bytes, dtype=torch.uint8)
    with pytest.raises(ValueError):
        fp8.quantize_blocks(cache.pool, torch.tensor([0], dtype=torch.int32), stage, layout)


def test_round_trip_bound_rejects_a_real_violation():
    bits = torch.tensor([1.5, 100.0], dtype=torch.bfloat16).view(torch.uint16)
    shift, restored = round_trip(bits)
    bound = round_trip_bound(bits[None])[0]
    assert ((as_float(restored) - as_float(bits)).abs() <= bound).all()
    assert (bound < torch.tensor([1.0, 7.0], dtype=torch.float64)).all()


# The Triton kernels, in the interpreter. Skipped where Triton does not install.


@pytest.fixture
def triton_module():
    return pytest.importorskip("triton")


def test_kernel_encode_matches_on_every_bf16_pattern(triton_module):
    encoded = fp8.encode_bits_triton(ALL_BF16, 0)
    assert encoded.equal(encode_bits(ALL_BF16, 0))
    assert_encode_matches_torch(encoded)


def test_kernel_decode_matches_on_every_fp8_pattern(triton_module):
    flat = ALL_FP8.clone()
    reference = flat.view(torch.float8_e4m3fn).to(torch.bfloat16).view(torch.uint16)
    assert fp8.decode_bits_triton(flat, 0).equal(reference)


@pytest.mark.parametrize("shift", [-133, -40, -7, 5, 40, 120])
def test_kernel_arithmetic_matches_reference_at_any_shift(triton_module, shift):
    assert fp8.encode_bits_triton(ALL_BF16, shift).equal(encode_bits(ALL_BF16, shift))
    assert fp8.decode_bits_triton(ALL_FP8, shift).equal(decode_bits(ALL_FP8, shift))


@pytest.mark.parametrize(
    "cfg,tokens",
    [(CFG, 16), (ModelConfig.micro(num_hidden_layers=1, num_key_value_heads=1, head_dim=128, num_attention_heads=1), 16)],
    ids=["padded_tile", "power_of_two_group"],
)
def test_block_kernels_write_the_same_bytes_as_the_reference(triton_module, cfg, tokens):
    cache = random_pool_cache(cfg, blocks=6)
    layout = Fp8Layout.from_cache(cache)
    ids = torch.tensor([4, 1, 2], dtype=torch.int32)
    ours = torch.zeros(3 * layout.record_bytes, dtype=torch.uint8)
    theirs = torch.zeros_like(ours)
    fp8.quantize_blocks(cache.pool, ids, ours, layout, impl="triton")
    fp8.quantize_blocks(cache.pool, ids, theirs, layout, impl="torch")
    assert ours.equal(theirs)

    target = random_pool_cache(cfg, blocks=6, seed=3)
    reference = target.pool.clone()
    fp8.dequantize_blocks(ours, ids, target.pool, layout, impl="triton")
    fp8.dequantize_blocks(ours, ids, reference, layout, impl="torch")
    assert target.pool.view(torch.uint16).equal(reference.view(torch.uint16))


def test_block_kernels_address_a_pool_past_4_gib(triton_module):
    """A 10 GiB pool holds more uint16s than int32 can count, so offsets must be 64-bit."""
    layout = Fp8Layout(2048, 8)
    block_bytes = 2 * layout.block_elems
    pool_blocks = 330_000
    source, target = pool_blocks - 1, pool_blocks - 2
    assert source * layout.block_elems > 1 << 31
    mapped = mmap.mmap(-1, pool_blocks * block_bytes, flags=mmap.MAP_PRIVATE | getattr(mmap, "MAP_NORESERVE", 0))
    pool = torch.frombuffer(mapped, dtype=torch.bfloat16).reshape(pool_blocks, layout.block_elems)
    values = torch.randn(1, layout.block_elems, generator=torch.Generator().manual_seed(0)).to(torch.bfloat16)
    pool[source] = values[0]

    ids = torch.tensor([source], dtype=torch.int32)
    stage = torch.zeros(layout.record_bytes, dtype=torch.uint8)
    reference = torch.zeros_like(stage)
    fp8.quantize_blocks(pool, ids, stage, layout, impl="triton")
    fp8.quantize_blocks(values, torch.zeros(1, dtype=torch.int32), reference, layout, impl="torch")
    assert stage.equal(reference)

    restored = torch.zeros_like(values)
    fp8.dequantize_blocks(reference, torch.zeros(1, dtype=torch.int32), restored, layout, impl="torch")
    fp8.dequantize_blocks(stage, torch.tensor([target], dtype=torch.int32), pool, layout, impl="triton")
    assert pool[target].view(torch.uint16).equal(restored[0].view(torch.uint16))


def launch_path_compile(name, group_elems, groups):
    """Compile through Triton's launch path, stopping where it would need a device."""
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource, make_backend
    from triton.runtime.jit import create_function_from_signature
    import triton

    layout = Fp8Layout(group_elems, groups)
    pool_bits = torch.empty(2, layout.block_elems, dtype=torch.uint16)
    ids = torch.zeros(2, dtype=torch.int32)
    stage = torch.empty(2 * layout.record_bytes, dtype=torch.uint8)
    # Triton's driver reports this target on an L4 and on an RTX 4090.
    target = GPUTarget("cuda", 89, 32)
    backend = make_backend(target)
    report = {}
    for fn, launch, first, last in (
        (fp8._quantize_kernel, fp8.quantize_launch, pool_bits, stage),
        (fp8._dequantize_kernel, fp8.dequantize_launch, stage, pool_bits),
    ):
        _, args, options = launch(first, ids, last, layout)
        binder = create_function_from_signature(fn.signature, fn.params, backend)
        bound, specialization, parsed = binder(*args, **options)
        parsed, signature, constexprs, attrs = fn._pack_args(backend, options, bound, specialization, parsed)
        compiled = triton.compile(ASTSource(fn, signature, constexprs, attrs), target=target, options=parsed.__dict__)
        report[fn.__name__] = {"signature": signature, "ptx": compiled.asm["ptx"], "cubin_bytes": len(compiled.asm["cubin"])}
    return report


@pytest.fixture(scope="module")
def compiled(tmp_path_factory):
    """In a child process, because @triton.jit decides at import whether to interpret."""
    pytest.importorskip("triton")
    env = {
        **os.environ,
        "TRITON_INTERPRET": "0",
        "TRITON_CACHE_DIR": str(tmp_path_factory.mktemp("triton_cache")),
        "PYTHONPATH": str(Path(__file__).resolve().parents[1]),
    }
    done = subprocess.run([sys.executable, __file__, "compile"], env=env, capture_output=True, text=True, timeout=600)
    assert done.returncode == 0, done.stderr[-4000:]
    return json.loads(done.stdout.splitlines()[-1])


def test_block_kernels_compile_for_sm_89_without_a_gpu(compiled):
    for shape, kernels in compiled.items():
        for name, kernel in kernels.items():
            assert kernel["cubin_bytes"] > 0, (shape, name)
            assert ".target sm_89" in kernel["ptx"], (shape, name)
    signature = compiled["qwen3_4b"]["_quantize_kernel"]["signature"]
    assert signature["pool_ptr"] == "*u16" and signature["ids_ptr"] == "*i32"
    assert signature["data_ptr"] == "*u8" and signature["shift_ptr"] == "*i16"
    signature = compiled["qwen3_4b"]["_dequantize_kernel"]["signature"]
    assert signature["pool_ptr"] == "*u16" and signature["shift_ptr"] == "*i16"


@pytest.mark.gpu
def test_block_kernels_on_cuda_match_the_reference():
    cache = random_pool_cache(blocks=64)
    layout = Fp8Layout.from_cache(cache)
    ids = torch.tensor(np.random.default_rng(0).permutation(64)[:32].tolist(), dtype=torch.int32)
    reference = torch.zeros(32 * layout.record_bytes, dtype=torch.uint8)
    fp8.quantize_blocks(cache.pool, ids, reference, layout, impl="torch")

    pool = cache.pool.to("cuda")
    stage = torch.zeros_like(reference, device="cuda")
    fp8.quantize_blocks(pool, ids.to("cuda"), stage, layout)
    assert stage.cpu().equal(reference)
    restored = torch.zeros_like(pool)
    fp8.dequantize_blocks(stage, ids.to("cuda"), restored, layout)
    expected = torch.zeros_like(cache.pool)
    fp8.dequantize_blocks(reference, ids, expected, layout, impl="torch")
    assert restored.cpu().view(torch.uint16).equal(expected.view(torch.uint16))


if __name__ == "__main__":
    # The compile step of the fixture above, in its child process.
    assert sys.argv[1:] == ["compile"]
    print(json.dumps({"qwen3_4b": launch_path_compile("qwen3_4b", 2048, 576), "padded": launch_path_compile("padded", 384, 8)}))


def _pool_pair(blocks=40):
    cache = PagedKVCache(CFG, blocks, dtype=torch.bfloat16)
    cache.pool.copy_(torch.randn(cache.pool.shape, generator=torch.Generator().manual_seed(3)).to(torch.bfloat16))
    layout = Fp8Layout.from_cache(cache)
    return cache.pool.clone(), cache.pool.clone(), layout.group_elems


def test_round_trip_check_widens_only_a_slice_at_a_time(monkeypatch):
    """Its widened temporaries take 36 times what they check, so a whole pool would be fatal."""
    before, after, group = _pool_pair()
    seen = []
    real = fp8.round_trip_bound
    monkeypatch.setattr(fp8, "round_trip_bound", lambda bits: (seen.append(bits.numel()), real(bits))[1])
    assert fp8.round_trip_within_bound(before, after, group, slice_blocks=16)
    per_block = before[0].numel()
    assert max(seen) == 16 * per_block
    assert sum(seen) == before.numel()


@pytest.mark.parametrize("block", [0, 17, 39], ids=["first_slice", "middle_slice", "last_slice"])
def test_round_trip_check_finds_a_bad_value_in_any_slice(block):
    before, after, group = _pool_pair()
    after[block, 1, 1, 1, 7, 3] = after[block, 1, 1, 1, 7, 3] + 1.0
    assert not fp8.round_trip_within_bound(before, after, group)


def test_round_trip_check_slices_give_the_whole_pool_answer():
    before, after, group = _pool_pair()
    bits = before.view(torch.uint16).reshape(-1, group)
    gap = (after.to(torch.float64).reshape(-1, group) - before.to(torch.float64).reshape(-1, group)).abs()
    assert (gap <= round_trip_bound(bits)).all() == fp8.round_trip_within_bound(before, after, group)
    # A group's shift depends on the whole group: any slicing must see it entire.
    groups = fp8.group_shifts(bits)
    sliced = torch.cat([fp8.group_shifts(before[i:i + 16].view(torch.uint16).reshape(-1, group)) for i in range(0, 40, 16)])
    assert torch.equal(groups, sliced)


def test_round_trip_check_can_ignore_values_that_were_never_finite():
    before, after, group = _pool_pair()
    before[3, 0, 0, 0, 0, 0] = float("inf")
    assert not fp8.round_trip_within_bound(before, after, group)
    assert fp8.round_trip_within_bound(before, after, group, ignore_nonfinite=True)
