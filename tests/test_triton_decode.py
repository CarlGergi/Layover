"""The Triton kernel against the PyTorch reference, and its GPU compile path.

Tolerances come from error_bound below, a forward error bound for FP32 online softmax,
and not from what happened to pass. The sharpest checks compare with the exact answer,
computed in float64 from the same inputs.
"""

import importlib.util
import json
import math
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest
import torch

triton = pytest.importorskip("triton")

from hypothesis import given, settings  # noqa: E402
from hypothesis import strategies as st  # noqa: E402

from layover.config import ModelConfig  # noqa: E402
from layover.engine import Engine, EngineConfig, Request  # noqa: E402
from layover.kernels import triton_decode as td  # noqa: E402
from layover.kernels.decode_attention import paged_decode_attention_torch as reference  # noqa: E402
from layover.kernels.triton_decode import paged_decode_attention_triton as kernel  # noqa: E402
from layover.kv import PagedKVCache  # noqa: E402
from layover.model import Qwen3  # noqa: E402

interpreted = pytest.mark.skipif(not triton.knobs.runtime.interpret, reason="needs TRITON_INTERPRET=1")

FP32_ROUNDOFF = 2.0**-24
BF16_ROUNDOFF = 2.0**-8
# torch.exp is within a few ulp and the GPU's ex2.approx within 2^-22.5.
EXP_RELATIVE_ERROR = 2.0**-22
QWEN3_4B = dict(layers=36, kv_heads=8, kv_groups=4, head_dim=128, slots=16)


@pytest.fixture(scope="module", autouse=True)
def round_like_cuda():
    """Make the interpreter round FP32 to BF16 to nearest, as compiled code does.

    Triton's interpreter truncates this cast, while on CUDA it is cvt.rn, which
    test_compiled_store_rounds_to_nearest pins down. Unpatched, every BF16 result here
    would be off by up to a whole ulp for a reason that does not exist on the GPU.
    """
    if not triton.knobs.runtime.interpret:
        yield
        return
    from triton.language import bfloat16, float32
    from triton.runtime.interpreter import InterpreterBuilder, TensorHandle

    original = InterpreterBuilder.create_fp_trunc

    def nearest(self, src, dst_type):
        if src.dtype.scalar == float32 and dst_type.scalar == bfloat16:
            rounded = torch.from_numpy(src.data.copy()).to(torch.bfloat16)
            return TensorHandle(rounded.view(torch.uint16).numpy(), dst_type.scalar)
        return original(self, src, dst_type)

    InterpreterBuilder.create_fp_trunc = nearest
    yield
    InterpreterBuilder.create_fp_trunc = original


def make_case(lengths, kv_heads=2, kv_groups=2, head_dim=24, slots=16, layers=3, dtype=torch.float32, seed=0, gain=1.0):
    """Random KV behind a shuffled block table, so no sequence is contiguous in the pool."""
    gen = torch.Generator().manual_seed(seed)
    batch = len(lengths)
    width = max(-(-n // slots) for n in lengths)
    blocks = batch * width + 3
    pool = torch.randn(blocks, 2, layers, kv_heads, slots, head_dim, generator=gen).to(dtype)
    table = torch.randperm(blocks, generator=gen)[: batch * width].reshape(batch, width).to(torch.int32)
    q = (gain * torch.randn(batch, kv_heads * kv_groups, head_dim, generator=gen)).to(dtype)
    return q, pool, table, torch.tensor(lengths, dtype=torch.int32), kv_groups


def random_on_cuda(shape, generator):
    """BF16 normals through FP32, filling a large pool in pieces to avoid holding it twice."""
    tensor = torch.empty(shape, dtype=torch.bfloat16, device="cuda")
    for start in range(0, shape[0], 256):
        piece = tensor[start : start + 256]
        piece.copy_(torch.randn(piece.shape, generator=generator, device="cuda"))
    return tensor


def exact(q, pool, layer, table, lengths, kv_groups):
    # One layer only: the whole of a GPU-sized pool in float64 would not fit on the GPU.
    return reference(q.double(), pool[:, :, layer : layer + 1].double(), 0, table, lengths, kv_groups)


def error_bound(q, pool, layer, lengths, kv_groups):
    """Worst-case |kernel - exact| for FP32 arithmetic, as a scalar.

    A score s = q.k / sqrt(d) has error at most (d + 2) u ||q|| ||k|| / sqrt(d) (Higham,
    Accuracy and Stability of Numerical Algorithms, ch. 3). A weight exp(s - m) inherits
    twice that, plus one exp error per rescale. The result is a weighted mean of V, so a
    relative weight error e moves it by at most 2 e max|V|, and summing n terms adds
    n u max|V|. The bound grows with the score scale because softmax amplifies rounding
    in the exponent, which is real and not slack.
    """
    d = q.shape[-1]
    slots = pool.shape[4]
    u = FP32_ROUNDOFF
    k_norm = pool[:, 0, layer].double().norm(dim=-1).max()
    v_max = pool[:, 1, layer].double().abs().max()
    n = int(lengths.max())
    blocks = -(-n // slots)
    score_error = (d + 2) * u * q.double().norm(dim=-1).max() * k_norm / math.sqrt(d)
    weight_error = 2 * score_error + (blocks + 2) * EXP_RELATIVE_ERROR
    return float((2 * weight_error + (n + d) * u) * v_max)


def assert_matches_exact(out, case):
    """FP32 output within the bound; BF16 output also within rounding to nearest."""
    q, pool, table, lengths, groups = case[:5]
    layer = case[5] if len(case) > 5 else 1
    truth = exact(q, pool, layer, table, lengths, groups)
    allowed = error_bound(q, pool, layer, lengths, groups) + 1e-37
    if out.dtype == torch.bfloat16:
        allowed = allowed * (1 + BF16_ROUNDOFF) + truth.abs() * BF16_ROUNDOFF * 1.001
    assert ((out.double() - truth).abs() <= allowed).all(), (out.double() - truth).abs().max().item()


def assert_scattered(table):
    assert table.shape[1] == 1 or ((table[:, 1:] - table[:, :-1]) != 1).any()


LENGTHS = {
    "one_token": [1],
    "one_token_each": [1, 1, 1],
    "block_aligned": [16, 32, 48],
    "one_past_a_block": [17, 33],
    "one_short_of_a_block": [15, 31],
    "mixed": [1, 16, 17, 47, 48, 80, 3, 64],
}


@interpreted
@pytest.mark.parametrize("name", LENGTHS)
@pytest.mark.parametrize("groups", [1, 2, 4])
def test_fp32_matches_reference(name, groups):
    q, pool, table, lengths, groups = make_case(LENGTHS[name], kv_groups=groups, seed=groups)
    assert_scattered(table)
    out = kernel(q, pool, 1, table, lengths, groups)
    assert out.shape == q.shape and out.dtype == q.dtype
    expected = reference(q, pool, 1, table, lengths, groups)
    # Both are FP32 computations of the same quantity, each within the bound of the truth.
    tolerance = 2 * error_bound(q, pool, 1, lengths, groups)
    assert (out - expected).abs().max() <= tolerance
    # The bound is the worst case; real rounding errors partly cancel. Measured differences
    # stay under 1e-6 on these inputs, so this leaves a factor of 10 and would notice the
    # arithmetic quietly dropping to lower precision, which the worst case would allow.
    assert (out - expected).abs().max() <= 1e-5


@interpreted
@pytest.mark.parametrize("name", LENGTHS)
@pytest.mark.parametrize("groups", [1, 2, 4])
def test_bf16_is_the_rounded_exact_answer(name, groups):
    case = make_case(LENGTHS[name], kv_groups=groups, dtype=torch.bfloat16, seed=10 + groups)
    out = kernel(*case[:2], 1, *case[2:])
    assert out.dtype == torch.bfloat16
    assert_matches_exact(out, case)


@interpreted
@pytest.mark.parametrize("name", LENGTHS)
def test_bf16_matches_bf16_reference(name):
    q, pool, table, lengths, groups = make_case(LENGTHS[name], dtype=torch.bfloat16, seed=20)
    out = kernel(q, pool, 1, table, lengths, groups).double()
    expected = reference(q, pool, 1, table, lengths, groups).double()
    # The reference accumulates in BF16 when lengths differ, so its rounding is relative
    # to partial sums, which are bounded by max|V|, and not to the result. Each rounding is
    # at most u max|V|; the worst case over n additions is never approached because a few
    # terms carry the weight (measured: 2 u max|V|). Four leaves a factor of two.
    allowed = 4 * BF16_ROUNDOFF * pool[:, 1, 1].double().abs().max() + 2 * error_bound(q, pool, 1, lengths, groups)
    assert (out - expected).abs().max() <= allowed


@interpreted
@pytest.mark.parametrize("layer", [0, 2])
def test_reads_the_requested_layer(layer):
    q, pool, table, lengths, groups = make_case([20, 40], dtype=torch.float32, seed=30)
    out = kernel(q, pool, layer, table, lengths, groups)
    assert (out - reference(q, pool, layer, table, lengths, groups)).abs().max() <= 2 * error_bound(
        q, pool, layer, lengths, groups
    )
    other = reference(q, pool, (layer + 1) % 3, table, lengths, groups)
    assert (out - other).abs().max() > 0.1


@interpreted
def test_query_heads_map_to_their_kv_head():
    """Every query head in a group sees its own KV head, whichever that is."""
    q, pool, table, lengths, groups = make_case([16, 16], kv_heads=4, kv_groups=2, seed=31)
    q[:] = q[:, :1]  # Identical queries, so any head difference comes from K and V.
    out = kernel(q, pool, 1, table, lengths, groups)
    for head in range(0, 8, 2):
        assert torch.equal(out[:, head], out[:, head + 1])
    assert (out[:, 0] - out[:, 2]).abs().max() > 0.1


@interpreted
@pytest.mark.parametrize("peak_block", [0, 3], ids=["first_block", "last_block"])
def test_online_softmax_when_the_maximum_moves(peak_block):
    """A large score late must rescale what was accumulated, and an early one must hold."""
    q, pool, table, lengths, groups = make_case([64], seed=32, gain=3.0)
    block = table[0, peak_block]
    pool[block, 0, 1] = pool[block, 0, 1] * 4
    out = kernel(q, pool, 1, table, lengths, groups)
    assert_matches_exact(out, (q, pool, table, lengths, groups))


@interpreted
def test_scores_too_large_for_an_unscaled_softmax():
    """exp(score) alone would overflow FP32 here; the running maximum keeps it finite."""
    q, pool, table, lengths, groups = make_case([48], seed=33, gain=40.0)
    pool[:, 0] *= 40
    out = kernel(q, pool, 1, table, lengths, groups)
    assert out.isfinite().all()
    assert_matches_exact(out, (q, pool, table, lengths, groups))


@interpreted
def test_masked_slots_cannot_win_even_with_the_largest_scores():
    q, pool, table, lengths, groups = make_case([20], seed=34)
    # Slots past token 20 hold huge K and huge V: reading either would dominate the result.
    tail = table[0, 1]
    pool[tail, :, :, :, 4:] = 1e4
    out = kernel(q, pool, 1, table, lengths, groups)
    assert out.abs().max() < 10
    assert_matches_exact(out, (q, pool, table, lengths, groups))


@interpreted
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("poison", [float("nan"), float("inf")], ids=["nan", "inf"])
def test_unwritten_slots_cannot_poison_the_result(dtype, poison):
    """Slots past the end can hold anything, and a zero weight times NaN is still NaN."""
    q, pool, table, lengths, groups = make_case([20, 3], dtype=dtype, seed=37)
    clean = kernel(q, pool, 1, table, lengths, groups)
    pool[table[0, 1], :, :, :, 4:] = poison
    pool[table[1, 0], :, :, :, 3:] = poison
    out = kernel(q, pool, 1, table, lengths, groups)
    assert out.isfinite().all()
    assert torch.equal(out, clean)


@interpreted
def test_int64_block_table_and_non_contiguous_q():
    q, pool, table, lengths, groups = make_case([20, 40], seed=35)
    expected = kernel(q, pool, 1, table, lengths, groups)
    view = torch.zeros(2, 4, 3, 24)
    view[:, :, 1] = q
    got = kernel(view[:, :, 1], pool, 1, table.long(), lengths.long(), groups)
    assert torch.equal(got, expected)
    strided = torch.zeros(q.shape[0], q.shape[1] * 2, q.shape[2])
    strided[:, ::2] = q
    assert not strided[:, ::2].is_contiguous()
    assert torch.equal(kernel(strided[:, ::2], pool, 1, table, lengths, groups), expected)


@interpreted
def test_result_on_a_q_of_a_different_dtype_stays_in_q_dtype():
    q, pool, table, lengths, groups = make_case([20], dtype=torch.bfloat16, seed=36)
    out = kernel(q.float(), pool, 1, table, lengths, groups)
    assert out.dtype == torch.float32
    assert_matches_exact(out, (q.float(), pool, table, lengths, groups))


@interpreted
def test_block_offsets_past_32_bits():
    """A 24 GB card passes 2^31 BF16 elements; a 32-bit offset would wrap and read elsewhere."""
    head_dim, layers, kv_heads, slots = 128, 36, 2, 16
    stride = 2 * layers * kv_heads * slots * head_dim
    blocks = 2**31 // stride + 8
    # torch.empty pages in only what is touched, so this costs a few blocks of real memory.
    pool = torch.empty(blocks, 2, layers, kv_heads, slots, head_dim, dtype=torch.bfloat16)
    assert pool.numel() > 2**31
    gen = torch.Generator().manual_seed(40)
    ids = torch.tensor([blocks - 1, blocks - 5, 3])
    small = torch.randn(3, 2, layers, kv_heads, slots, head_dim, generator=gen).to(torch.bfloat16)
    pool[ids] = small
    q = torch.randn(1, kv_heads, head_dim, generator=gen).to(torch.bfloat16)
    lengths = torch.tensor([40], dtype=torch.int32)
    out = kernel(q, pool, 7, ids[None].to(torch.int32), lengths, 1)
    # The answer computed on just those blocks, renumbered, so offsets stay small.
    assert_matches_exact(out, (q, small, torch.arange(3)[None].to(torch.int32), lengths, 1, 7))


@pytest.mark.parametrize(
    "mutate",
    [
        lambda c: {**c, "lengths": torch.tensor([0, 5], dtype=torch.int32)},
        lambda c: {**c, "lengths": torch.tensor([33, 5], dtype=torch.int32)},
        lambda c: {**c, "table": torch.tensor([[0, 99], [1, 2]], dtype=torch.int32)},
        lambda c: {**c, "table": torch.tensor([[0, -1], [1, 2]], dtype=torch.int32)},
        lambda c: {**c, "layer": 3},
        lambda c: {**c, "groups": 3},
        lambda c: {**c, "q": c["q"][:, :, :-1]},
        lambda c: {**c, "q": c["q"][:1]},
        lambda c: {**c, "pool": c["pool"].transpose(0, 1)},
        lambda c: {**c, "pool": c["pool"][::2]},
        lambda c: {**c, "q": c["q"].double()},
        lambda c: {**c, "pool": c["pool"].double()},
    ],
    ids=[
        "empty_sequence",
        "longer_than_table",
        "block_past_pool",
        "negative_block",
        "layer_past_pool",
        "groups_disagree",
        "head_dim_disagrees",
        "batch_disagrees",
        "kinds_not_two",
        "non_contiguous_pool",
        "q_float64",
        "pool_float64",
    ],
)
def test_invalid_inputs_are_refused_before_launch(mutate):
    q, pool, table, lengths, groups = make_case([20, 5], seed=41)
    good = dict(q=q, pool=pool, table=table, lengths=lengths, layer=1, groups=groups)
    bad = mutate(good)
    with pytest.raises(ValueError):
        kernel(bad["q"], bad["pool"], bad["layer"], bad["table"], bad["lengths"], bad["groups"])


@interpreted
@settings(max_examples=60, deadline=None)
@given(
    seed=st.integers(0, 2**16),
    kv_heads=st.sampled_from([1, 2, 3]),
    groups=st.sampled_from([1, 2, 4]),
    head_dim=st.sampled_from([8, 16, 24, 31, 32]),
    slots=st.sampled_from([4, 7, 16]),
    layer=st.integers(0, 2),
    gain=st.sampled_from([0.1, 1.0, 8.0]),
    dtype=st.sampled_from([torch.float32, torch.bfloat16]),
    lengths=st.lists(st.integers(1, 70), min_size=1, max_size=4),
)
def test_property_matches_exact(seed, kv_heads, groups, head_dim, slots, layer, gain, dtype, lengths):
    case = make_case(lengths, kv_heads, groups, head_dim, slots, 3, dtype, seed, gain)
    q, pool, table, lens, groups = case
    out = kernel(q, pool, layer, table, lens, groups)
    assert out.isfinite().all()
    assert_matches_exact(out, (q, pool, table, lens, groups, layer))


def greedy(backend, dtype, prompts, new_tokens=12):
    torch.manual_seed(0)
    cfg = ModelConfig.micro()
    model = Qwen3(cfg).to(dtype=dtype).eval()
    model.decode_attention = backend
    engine = Engine(
        model, PagedKVCache(cfg, 96, dtype=dtype), EngineConfig(max_batch=4, prefill_chunk=9, prefill_budget=9)
    )
    requests = [Request(str(i), p, max_new_tokens=new_tokens) for i, p in enumerate(prompts)]
    for request in requests:
        engine.add_request(request)
    engine.run_until_idle()
    assert engine.cache.allocator.num_used == 0
    return [r.generated for r in requests]


def greedy_prompts():
    gen = torch.Generator().manual_seed(50)
    # Lengths that make decode tokens land on a block boundary, in its middle, and on token 1.
    return [torch.randint(0, 512, (n,), generator=gen).tolist() for n in (1, 15, 16, 31, 40, 64)]


@interpreted
def test_engine_greedy_tokens_match_torch_backend_fp32():
    prompts = greedy_prompts()
    assert greedy(kernel, torch.float32, prompts) == greedy(reference, torch.float32, prompts)


@interpreted
def test_engine_greedy_tokens_match_torch_backend_bf16():
    prompts = greedy_prompts()
    assert greedy(kernel, torch.bfloat16, prompts) == greedy(reference, torch.bfloat16, prompts)


def _launch_path_compile(spec: dict) -> dict:
    """Compile through Triton's own launch path up to the point of needing a device.

    The binder and _pack_args are what JITFunction.run calls: they turn real tensors into
    the signature and constexprs, including Triton's specialization on argument values,
    which the interpreter skips. Only the driver call at the end is left out.
    """
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource, make_backend
    from triton.runtime.jit import create_function_from_signature

    dtype = getattr(torch, spec["dtype"])
    shape = spec["shape"]
    query_heads = shape["kv_heads"] * shape["kv_groups"]
    pool = torch.empty(
        4, 2, shape["layers"], shape["kv_heads"], shape["slots"], shape["head_dim"], dtype=dtype
    )
    q = torch.empty(2, query_heads, shape["head_dim"], dtype=dtype)
    out = torch.empty_like(q)
    table = torch.zeros(2, spec.get("width", 3), dtype=getattr(torch, spec.get("table", "int32")))
    lengths = torch.ones(2, dtype=torch.int32)
    grid, args, options = td.kernel_launch(q, pool, spec.get("layer", 5), table, lengths, shape["kv_groups"], out)
    options = {**options, **spec.get("options", {})}
    # Triton's driver reports GPUTarget("cuda", 89, 32) on an L4 and on an RTX 4090.
    target = GPUTarget("cuda", 89, 32)
    backend = make_backend(target)
    fn = td._paged_decode_kernel
    binder = create_function_from_signature(fn.signature, fn.params, backend)
    bound, specialization, parsed = binder(*args, **options)
    parsed, signature, constexprs, attrs = fn._pack_args(backend, options, bound, specialization, parsed)
    compiled = triton.compile(ASTSource(fn, signature, constexprs, attrs), target=target, options=parsed.__dict__)
    assembler = _ptxas_report(compiled.asm["ptx"])
    return {
        "assembler": assembler,
        "name": spec["name"],
        "grid": grid,
        "signature": signature,
        "num_warps": compiled.metadata.num_warps,
        "shared": compiled.metadata.shared,
        "hash": compiled.metadata.hash,
        "ttir": compiled.asm["ttir"],
        "ptx": compiled.asm["ptx"],
        "cubin_bytes": len(compiled.asm["cubin"]),
    }


def _ptxas_report(ptx: str) -> str:
    """What ptxas says about registers and spills, which compiled.metadata leaves out."""
    ptxas = Path(triton.__file__).parent / "backends" / "nvidia" / "bin" / "ptxas"
    with tempfile.TemporaryDirectory() as folder:
        source = Path(folder) / "kernel.ptx"
        source.write_text(ptx)
        command = [str(ptxas), "-v", "--gpu-name", "sm_89", "-O3", str(source), "-o", str(source.with_suffix(".o"))]
        return subprocess.run(command, capture_output=True, text=True, check=True).stderr


def _refuses_cpu_tensors_without_interpreter() -> bool:
    q, pool, table, lengths, groups = make_case([5])
    try:
        kernel(q, pool, 1, table, lengths, groups)
    except ValueError:
        return True
    return False


COMPILE_SPECS = [
    {"name": "qwen3_4b_bf16", "dtype": "bfloat16", "shape": QWEN3_4B},
    {"name": "qwen3_4b_bf16_layer0_int64", "dtype": "bfloat16", "shape": QWEN3_4B, "layer": 0, "table": "int64"},
    {"name": "qwen3_4b_bf16_width_multiple_of_16", "dtype": "bfloat16", "shape": QWEN3_4B, "width": 32},
    {"name": "qwen3_4b_fp32", "dtype": "float32", "shape": QWEN3_4B},
    {"name": "qwen3_4b_bf16_debug", "dtype": "bfloat16", "shape": QWEN3_4B, "options": {"debug": True}},
    # The block sizes the design measures, 8 and 32, the largest with twice the tile.
    {"name": "qwen3_4b_8_slots", "dtype": "bfloat16", "shape": {**QWEN3_4B, "slots": 8}},
    {"name": "qwen3_4b_32_slots", "dtype": "bfloat16", "shape": {**QWEN3_4B, "slots": 32}},
    {"name": "qwen3_4b_32_slots_fp32", "dtype": "float32", "shape": {**QWEN3_4B, "slots": 32}},
    {
        "name": "micro_bf16_padded",
        "dtype": "bfloat16",
        "shape": dict(layers=2, kv_heads=2, kv_groups=2, head_dim=24, slots=16),
    },
    {
        "name": "non_power_of_two_slots",
        "dtype": "bfloat16",
        "shape": dict(layers=2, kv_heads=2, kv_groups=2, head_dim=32, slots=12),
    },
]


@pytest.fixture(scope="module")
def compiled(tmp_path_factory):
    """Compile in a child process: @triton.jit decides at import whether to interpret, so
    a test process running kernels under TRITON_INTERPRET=1 cannot compile them."""
    env = {
        **os.environ,
        "TRITON_INTERPRET": "0",
        "TRITON_CACHE_DIR": str(tmp_path_factory.mktemp("triton_cache")),
        "PYTHONPATH": str(Path(__file__).resolve().parents[1]),
    }
    done = subprocess.run(
        [sys.executable, __file__, "compile"], env=env, capture_output=True, text=True, timeout=600
    )
    assert done.returncode == 0, done.stderr[-4000:]
    report = json.loads(done.stdout.splitlines()[-1])
    report["specs"] = {r["name"]: r for r in report["specs"]}
    return report


def test_compile_uses_gpu_mode_and_assembles_with_ptxas(compiled):
    assert compiled["interpret"] is False
    for spec in compiled["specs"].values():
        assert spec["cubin_bytes"] > 0, spec["name"]
        assert ".target sm_89" in spec["ptx"]


def test_compile_signature_matches_the_launch(compiled):
    signature = compiled["specs"]["qwen3_4b_bf16"]["signature"]
    assert signature["q_ptr"] == signature["pool_ptr"] == signature["out_ptr"] == "*bf16"
    assert signature["table_ptr"] == signature["lengths_ptr"] == "*i32"
    assert signature["HEAD_DIM"] == signature["SLOTS"] == "constexpr"
    assert compiled["specs"]["qwen3_4b_bf16_layer0_int64"]["signature"]["table_ptr"] == "*i64"
    assert compiled["specs"]["qwen3_4b_fp32"]["signature"]["pool_ptr"] == "*fp32"


def test_compiled_block_offset_is_64_bit(compiled):
    """The interpreter's arithmetic need not wrap as a GPU's would, so read the IR."""
    ttir = compiled["specs"]["qwen3_4b_bf16"]["ttir"]
    multiply = [line for line in ttir.splitlines() if "arith.muli" in line and ": i64" in line]
    assert multiply, "no 64-bit multiply in the kernel"
    assert any("tt.addptr %pool_ptr" in line and "i64" in line for line in ttir.splitlines())
    # And that it survives to what the GPU runs.
    assert "mul.lo.s64" in compiled["specs"]["qwen3_4b_bf16"]["ptx"]


def test_compiled_kernels_do_not_spill(compiled):
    """A spill here would be a stack write per tile in the hottest loop of the engine."""
    for spec in compiled["specs"].values():
        if spec["name"].endswith("_debug"):
            continue
        assert "0 bytes spill stores, 0 bytes spill loads" in spec["assembler"], spec["name"]
        registers = int(re.search(r"Used (\d+) registers", spec["assembler"]).group(1))
        # Half of what a thread can have at all, so nothing is close to the limit.
        assert registers < 128, (spec["name"], registers)


def tile_loads(ttir, tile):
    return [line.split(":")[0] for line in ttir.splitlines() if "tt.load" in line and tile in line]


def test_compiled_qwen3_kernel_has_no_masked_tile_loads(compiled):
    """Masks would cost a branch per tile on a shape that needs none."""
    loads = tile_loads(compiled["specs"]["qwen3_4b_bf16"]["ttir"], "16x128")
    assert len(loads) == 2
    assert all("," not in load for load in loads)
    padded = tile_loads(compiled["specs"]["micro_bf16_padded"]["ttir"], "16x32")
    assert len(padded) == 2
    assert all("," in load for load in padded)


def test_table_width_does_not_cause_a_recompile(compiled):
    specs = compiled["specs"]
    assert specs["qwen3_4b_bf16"]["hash"] == specs["qwen3_4b_bf16_width_multiple_of_16"]["hash"]


def test_compiled_store_rounds_to_nearest(compiled):
    assert "cvt.rn.bf16.f32" in compiled["specs"]["qwen3_4b_bf16"]["ptx"]


def test_compiled_kernel_stays_single_precision(compiled):
    ptx = compiled["specs"]["qwen3_4b_bf16"]["ptx"]
    assert ".f64" not in ptx


def test_compiled_kernel_shape_is_sane(compiled):
    spec = compiled["specs"]["qwen3_4b_bf16"]
    assert spec["grid"] == [2, 32]
    assert spec["num_warps"] == 4
    # An L4's compute shared memory is far larger; this catches a layout gone wrong.
    assert spec["shared"] < 32 * 1024
    assert "trap" not in spec["ptx"]
    assert "trap" in compiled["specs"]["qwen3_4b_bf16_debug"]["ptx"]


def test_gpu_mode_refuses_cpu_tensors(compiled):
    assert compiled["refused_cpu_tensors"] is True


def _sanitize_driver():
    path = Path(__file__).parent / "sanitize_transfer.py"
    spec = importlib.util.spec_from_file_location("sanitize_transfer", path)
    driver = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(driver)
    return driver


@interpreted
def test_sanitize_driver_launches_the_kernel_where_an_overrun_would_show(monkeypatch):
    """Every path memcheck needs to see: both tile modes, padded rows, the pool's last byte."""
    driver = _sanitize_driver()
    launches = []
    real = driver.paged_decode_attention_triton

    def spy(q, pool, layer, table, lengths, groups):
        launches.append((pool.shape, layer, table.clone(), lengths.clone()))
        return real(q, pool, layer, table, lengths, groups)

    monkeypatch.setattr(driver, "paged_decode_attention_triton", spy)
    assert driver.decode_launches("cpu") == 2 == len(launches)
    padded_tiles = []
    for shape, layer, table, lengths in launches:
        blocks, _, layers, _, slots, head_dim = shape
        padded_tiles.append(td.next_power_of_two(slots) != slots or td.next_power_of_two(head_dim) != head_dim)
        assert layer == layers - 1, "the last layer holds the pool's highest addresses"
        full, short = lengths.tolist()
        # The longest row reads the last slot of the pool's last block.
        assert full == table.shape[1] * slots and table[0, -1] == blocks - 1
        # The short row stops partway through a block, ahead of padded table entries.
        assert short % slots and short <= (table.shape[1] - 1) * slots
        assert len(set(table[0].tolist())) == table.shape[1], "a scattered table has no repeats"
        assert table[0].tolist() != sorted(table[0].tolist()) or table[0, 0] != 0
    assert padded_tiles == [False, True]


@interpreted
def test_sanitize_driver_main_reaches_the_decode_launches(capsys):
    assert _sanitize_driver().main(["--device", "cpu"]) == 0
    assert "decode launches: 2" in capsys.readouterr().out


@interpreted
def test_sanitize_driver_catches_a_wrong_decode(monkeypatch, capsys):
    driver = _sanitize_driver()
    real = driver.paged_decode_attention_triton
    monkeypatch.setattr(driver, "paged_decode_attention_triton", lambda *a: real(*a) + 1.0)
    with pytest.raises(SystemExit):
        driver.decode_launches("cpu")
    assert "mismatch" in capsys.readouterr().err


@pytest.mark.gpu
def test_kernel_matches_reference_on_cuda_at_qwen3_4b_shape():
    assert not triton.knobs.runtime.interpret
    device = "cuda"
    shape = QWEN3_4B
    gen = torch.Generator(device).manual_seed(60)
    blocks = 2048  # 4.8 GB, which puts block offsets past 32 bits.
    pool = random_on_cuda((blocks, 2, shape["layers"], shape["kv_heads"], shape["slots"], shape["head_dim"]), gen)
    assert pool.numel() > 2**31
    lengths_list = [1, 16, 17, 255, 256, 1024, 4000, 8192]
    needed = [-(-n // shape["slots"]) for n in lengths_list]
    # Each sequence gets its own blocks, the last of the pool among them, and short rows
    # are padded with zeros as block_table_tensor pads them.
    ids = torch.randperm(blocks - 1, generator=gen, device=device)[: sum(needed)]
    ids[0] = blocks - 1
    table = torch.zeros(len(lengths_list), max(needed), dtype=torch.int32, device=device)
    for row, chunk in enumerate(ids.split(needed)):
        table[row, : len(chunk)] = chunk.to(torch.int32)
    lengths = torch.tensor(lengths_list, dtype=torch.int32, device=device)
    q = random_on_cuda((len(lengths_list), 32, 128), gen)

    for layer in (0, 17, 35):
        out = kernel(q, pool, layer, table, lengths, shape["kv_groups"])
        assert out.device == q.device and out.dtype == q.dtype and out.isfinite().all()
        assert_matches_exact(out, (q, pool, table, lengths, shape["kv_groups"], layer))
        got_bf16 = reference(q, pool, layer, table, lengths, shape["kv_groups"]).float()
        # cuBLAS and softmax may round in BF16 on the way, unlike torch on CPU. Rounding
        # partial sums gives a score error with standard deviation near u sqrt(d) / 2.45,
        # and weights inherit twice that, on V of unit variance. Six deviations of it, plus
        # the final rounding of both results.
        values = pool[:, 1, layer].float()
        spread = 2 * 6 * BF16_ROUNDOFF * math.sqrt(128) / 2.45 * values.pow(2).mean().sqrt()
        tolerance = spread + 2 * BF16_ROUNDOFF * values.abs().max()
        assert (out.float() - got_bf16).abs().max() <= tolerance


@pytest.mark.gpu
def test_kernel_overflow_checks_stay_quiet_on_cuda():
    """debug=True compiles Triton's integer overflow traps in, which crash loudly if hit."""
    shape = QWEN3_4B
    gen = torch.Generator("cuda").manual_seed(61)
    blocks = 2048
    pool = random_on_cuda((blocks, 2, shape["layers"], shape["kv_heads"], shape["slots"], shape["head_dim"]), gen)
    table = torch.tensor([[blocks - 1, blocks - 2, blocks - 3]], dtype=torch.int32, device="cuda")
    lengths = torch.tensor([40], dtype=torch.int32, device="cuda")
    q = random_on_cuda((1, 32, 128), gen)
    out = torch.empty_like(q)
    grid, args, options = td.kernel_launch(q, pool, 35, table, lengths, shape["kv_groups"], out)
    td._paged_decode_kernel[grid](*args, **options, debug=True)
    assert_matches_exact(out, (q, pool, table, lengths, shape["kv_groups"], 35))


@pytest.mark.gpu
def test_unwritten_slots_cannot_poison_on_cuda():
    """The first block's rescale is exp(-inf) on ex2.approx, and masked V holds NaN."""
    shape = QWEN3_4B
    gen = torch.Generator("cuda").manual_seed(63)
    pool = random_on_cuda((8, 2, shape["layers"], shape["kv_heads"], shape["slots"], shape["head_dim"]), gen)
    table = torch.tensor([[5, 2], [7, 0]], dtype=torch.int32, device="cuda")
    lengths = torch.tensor([20, 1], dtype=torch.int32, device="cuda")
    q = random_on_cuda((2, 32, 128), gen)
    clean = kernel(q, pool, 3, table, lengths, shape["kv_groups"])
    pool[2, :, :, :, 4:] = float("nan")
    pool[7, :, :, :, 1:] = float("inf")
    out = kernel(q, pool, 3, table, lengths, shape["kv_groups"])
    assert out.isfinite().all()
    assert torch.equal(out, clean)


@pytest.mark.gpu
def test_engine_greedy_tokens_match_torch_backend_on_cuda():
    gen = torch.Generator().manual_seed(62)
    prompts = [torch.randint(0, 512, (n,), generator=gen).tolist() for n in (1, 16, 40, 64)]

    def run(backend):
        torch.manual_seed(0)
        cfg = ModelConfig.micro()
        model = Qwen3(cfg).to(dtype=torch.float32, device="cuda").eval()
        model.decode_attention = backend
        engine = Engine(model, PagedKVCache(cfg, 96, "cuda", torch.float32), EngineConfig(prefill_chunk=9))
        requests = [Request(str(i), p, max_new_tokens=12) for i, p in enumerate(prompts)]
        for request in requests:
            engine.add_request(request)
        engine.run_until_idle()
        return [r.generated for r in requests]

    assert run(kernel) == run(reference)


if __name__ == "__main__" and sys.argv[1:] == ["compile"]:
    print(
        json.dumps(
            {
                "interpret": bool(triton.knobs.runtime.interpret),
                "refused_cpu_tensors": _refuses_cpu_tensors_without_interpreter(),
                "specs": [_launch_path_compile(spec) for spec in COMPILE_SPECS],
            }
        )
    )
