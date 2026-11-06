import pytest
import torch

from triton.experimental import gluon
import triton.experimental.gluon.language as ttgl
from triton._internal_testing import is_hip_cdna4
from triton.tools.mxfp import MXFP4Tensor


@gluon.jit
def scaled_downcast_fp4_kernel(
    input_ptr,
    scale_ptr,
    output_ptr,
    BLOCK_M: ttgl.constexpr,
    BLOCK_K: ttgl.constexpr,
):
    # BLOCK_K is the packed output width along the scaled axis; the unpacked
    # input is twice as wide.
    out_layout: ttgl.constexpr = ttgl.BlockedLayout([1, 4], [8, 8], [1, 1], [1, 0])
    in_layout: ttgl.constexpr = ttgl.BlockedLayout([1, 8], [8, 8], [1, 1], [1, 0])
    scale_layout: ttgl.constexpr = ttgl.DistributedLinearLayout(
        reg_bases=[[8, 0]],
        lane_bases=[
            [0, 0],
            [0, 0],
            [0, 1],
            [1, 0],
            [2, 0],
            [4, 0],
        ],
        warp_bases=[],
        block_bases=[],
        shape=[BLOCK_M, BLOCK_K // 16],
    )
    in_offsets_m = ttgl.arange(0, BLOCK_M, layout=ttgl.SliceLayout(1, in_layout))
    in_offsets_k = ttgl.arange(0, 2 * BLOCK_K, layout=ttgl.SliceLayout(0, in_layout))
    in_offsets = in_offsets_m[:, None] * (2 * BLOCK_K) + in_offsets_k[None, :]
    input = ttgl.load(input_ptr + in_offsets)

    scale_offsets_m = ttgl.arange(0, BLOCK_M, layout=ttgl.SliceLayout(1, scale_layout))
    scale_offsets_k = ttgl.arange(0, BLOCK_K // 16, layout=ttgl.SliceLayout(0, scale_layout))
    scale_offsets = (scale_offsets_m[:, None] * (BLOCK_K // 16) + scale_offsets_k[None, :])
    scale = ttgl.load(scale_ptr + scale_offsets)

    output = ttgl.amd.cdna4.scaled_downcast(input, scale, "e2m1", axis=1)

    out_offsets_m = ttgl.arange(0, BLOCK_M, layout=ttgl.SliceLayout(1, out_layout))
    out_offsets_k = ttgl.arange(0, BLOCK_K, layout=ttgl.SliceLayout(0, out_layout))
    out_offsets = out_offsets_m[:, None] * BLOCK_K + out_offsets_k[None, :]
    ttgl.store(output_ptr + out_offsets, output)


@pytest.mark.skipif(not is_hip_cdna4(), reason="Requires CDNA4")
@pytest.mark.parametrize("scale_byte", [0, 1, 127])
@pytest.mark.parametrize(
    "dtype,instruction_suffix",
    [
        (torch.float16, "f16"),
        (torch.bfloat16, "bf16"),
        (torch.float32, "f32"),
    ],
)
def test_runtime_scaled_downcast_fp4(dtype, instruction_suffix, scale_byte):
    block_m, block_k = 16, 32
    # Interleave scaled 1.0 and 2.0 so dividing by the scale packs 0x42.
    input = torch.empty((block_m, 2 * block_k), dtype=dtype)
    input[:, 0::2] = 2.0**(scale_byte - 127)
    input[:, 1::2] = 2.0**(scale_byte - 126)
    input = input.cuda()
    scale = torch.full((block_m, block_k // 16), scale_byte, dtype=torch.uint8).cuda()
    output = torch.empty((block_m, block_k), dtype=torch.uint8, device="cuda")

    program = scaled_downcast_fp4_kernel[(1, )](
        input,
        scale,
        output,
        block_m,
        block_k,
        num_warps=1,
        allow_flush_denorm=False,
    )

    # The smallest scaled inputs already underflow in FP16.
    expected_byte = 0 if dtype == torch.float16 and scale_byte < 127 else 0x42
    expected = torch.full((block_m, block_k), expected_byte, dtype=torch.uint8)
    torch.testing.assert_close(output.cpu(), expected)
    assert (f"v_cvt_scalef32_pk_fp4_{instruction_suffix}" in program.asm["amdgcn"])


@gluon.jit
def scaled_downcast_per_lane_kernel(x_ptr, scale_ptr, y_ptr, BLOCK_M: ttgl.constexpr, BLOCK_K: ttgl.constexpr,
                                    OUT_K: ttgl.constexpr, SCALE_FACTOR: ttgl.constexpr,
                                    VALUES_PER_LANE: ttgl.constexpr, FORMAT: ttgl.constexpr):
    layout: ttgl.constexpr = ttgl.BlockedLayout([1, VALUES_PER_LANE], [8, 8], [4, 1], [1, 0])
    offs_m = ttgl.arange(0, BLOCK_M, layout=ttgl.SliceLayout(1, layout))
    offs_k = ttgl.arange(0, BLOCK_K, layout=ttgl.SliceLayout(0, layout))
    offs = offs_m[:, None] * BLOCK_K + offs_k[None, :]
    x = ttgl.load(x_ptr + offs)
    # The scale is stored expanded (one byte per value). The max over each block
    # gives the compact scale in the layout that scaled_downcast derives from `layout`.
    expanded = ttgl.load(scale_ptr + offs)
    scale = ttgl.max(expanded.reshape(BLOCK_M, BLOCK_K // SCALE_FACTOR, SCALE_FACTOR), axis=2).to(ttgl.uint8)

    y = ttgl.amd.cdna4.scaled_downcast(x, scale, FORMAT, axis=1)
    out_layout: ttgl.constexpr = y.type.layout
    offs_out_m = ttgl.arange(0, BLOCK_M, layout=ttgl.SliceLayout(1, out_layout))
    offs_out_k = ttgl.arange(0, OUT_K, layout=ttgl.SliceLayout(0, out_layout))
    ttgl.store(y_ptr + offs_out_m[:, None] * OUT_K + offs_out_k[None, :], y)


@pytest.mark.skipif(not is_hip_cdna4(), reason="Requires CDNA4")
@pytest.mark.parametrize("values_per_lane", [2, 4, 8])
@pytest.mark.parametrize("format,torch_fp8,out_mnemonic", [
    ("e2m1", None, "fp4"),
    ("e4m3", torch.float8_e4m3fn, "fp8"),
    ("e5m2", torch.float8_e5m2, "bf8"),
])
@pytest.mark.parametrize("dtype,in_suffix", [
    (torch.float16, "f16"),
    (torch.bfloat16, "bf16"),
    (torch.float32, "f32"),
])
def test_runtime_scaled_downcast_per_lane(values_per_lane, format, torch_fp8, out_mnemonic, dtype, in_suffix):
    # A lane holds fewer than 8 values of a scale block, which the CDNA5 pk8
    # rule rejects. The CDNA4 instructions convert one pair at a time, so these
    # layouts are valid on CDNA4.
    BLOCK_M, BLOCK_K, SCALE_FACTOR = 32, 128, 32
    torch.manual_seed(0)
    if format == "e2m1":
        values = MXFP4Tensor(size=(BLOCK_M, BLOCK_K)).random()
        expected = values.to_packed_tensor(1)
        values_ref = values.to(torch.float32)
    else:
        expected = torch.randint(20, 40, (BLOCK_M, BLOCK_K), dtype=torch.uint8)
        values_ref = expected.view(torch_fp8).to(torch.float32)
    # Each scale block of a row gets a different scale, so a scale that is applied
    # to the wrong block changes the output.
    num_blocks = BLOCK_K // SCALE_FACTOR
    scale = (124 + torch.arange(BLOCK_M * num_blocks) % 7).to(torch.uint8).reshape(BLOCK_M, num_blocks)
    scale = scale.repeat_interleave(SCALE_FACTOR, dim=1).contiguous()
    x = (values_ref * torch.exp2(scale.to(torch.float32) - 127)).to(dtype)

    out_k = BLOCK_K // 2 if format == "e2m1" else BLOCK_K
    y = torch.empty((BLOCK_M, out_k), dtype=torch.uint8 if torch_fp8 is None else torch_fp8, device="cuda")
    program = scaled_downcast_per_lane_kernel[(1, )](x.cuda(), scale.cuda(), y, BLOCK_M, BLOCK_K, out_k, SCALE_FACTOR,
                                                     values_per_lane, format, num_warps=4)

    assert f"v_cvt_scalef32_pk_{out_mnemonic}_{in_suffix}" in program.asm["amdgcn"]
    torch.testing.assert_close(y.view(torch.uint8).cpu(), expected, atol=0, rtol=0)


@gluon.jit
def scaled_downcast_broadcast_register_kernel(x_ptr, scale_ptr, y_ptr, BLOCK_M: ttgl.constexpr, BLOCK_K: ttgl.constexpr,
                                              OUT_K: ttgl.constexpr, SCALE_K: ttgl.constexpr, FORMAT: ttgl.constexpr):
    # Register basis 1 is zero, so each thread holds 16 registers but only 8
    # distinct values. Register bases 2 and 3 step 128 and 256 columns, into
    # other scale blocks. The bases assume BLOCK_M=4 and BLOCK_K=512.
    layout: ttgl.constexpr = ttgl.DistributedLinearLayout(
        reg_bases=[[0, 1], [0, 0], [0, 128], [0, 256]],
        lane_bases=[[0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [0, 64]],
        warp_bases=[[1, 0], [2, 0]],
        block_bases=[],
        shape=[BLOCK_M, BLOCK_K],
    )
    # The compact scale layout that scaled_downcast derives from `layout` for
    # 32-element scale blocks.
    scale_layout: ttgl.constexpr = ttgl.DistributedLinearLayout(
        reg_bases=[[0, 4], [0, 8]],
        lane_bases=[[0, 0], [0, 0], [0, 0], [0, 0], [0, 1], [0, 2]],
        warp_bases=[[1, 0], [2, 0]],
        block_bases=[],
        shape=[BLOCK_M, SCALE_K],
    )
    offs_m = ttgl.arange(0, BLOCK_M, layout=ttgl.SliceLayout(1, layout))
    offs_k = ttgl.arange(0, BLOCK_K, layout=ttgl.SliceLayout(0, layout))
    x = ttgl.load(x_ptr + offs_m[:, None] * BLOCK_K + offs_k[None, :])
    scale_m = ttgl.arange(0, BLOCK_M, layout=ttgl.SliceLayout(1, scale_layout))
    scale_k = ttgl.arange(0, SCALE_K, layout=ttgl.SliceLayout(0, scale_layout))
    scale = ttgl.load(scale_ptr + scale_m[:, None] * SCALE_K + scale_k[None, :])

    y = ttgl.amd.cdna4.scaled_downcast(x, scale, FORMAT, axis=1)
    out_layout: ttgl.constexpr = y.type.layout
    offs_out_m = ttgl.arange(0, BLOCK_M, layout=ttgl.SliceLayout(1, out_layout))
    offs_out_k = ttgl.arange(0, OUT_K, layout=ttgl.SliceLayout(0, out_layout))
    ttgl.store(y_ptr + offs_out_m[:, None] * OUT_K + offs_out_k[None, :], y)


@pytest.mark.skipif(not is_hip_cdna4(), reason="Requires CDNA4")
@pytest.mark.parametrize("format,torch_fp8", [
    ("e2m1", None),
    ("e4m3", torch.float8_e4m3fn),
    ("e5m2", torch.float8_e5m2),
])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_runtime_scaled_downcast_broadcast_register(format, torch_fp8, dtype):
    # The lowering converts only the distinct registers, so it must look up
    # each one's scale in the same register space. A lookup over all 16
    # registers gives the values 128 and 256 columns away the scale of an
    # earlier block.
    BLOCK_M, BLOCK_K, SCALE_FACTOR = 4, 512, 32
    torch.manual_seed(0)
    if format == "e2m1":
        values = MXFP4Tensor(size=(BLOCK_M, BLOCK_K)).random()
        expected = values.to_packed_tensor(1)
        values_ref = values.to(torch.float32)
    else:
        expected = torch.randint(20, 40, (BLOCK_M, BLOCK_K), dtype=torch.uint8)
        values_ref = expected.view(torch_fp8).to(torch.float32)
    # Blocks 4 apart get different scales, so a scale taken from the wrong
    # block changes the output.
    scale_k = BLOCK_K // SCALE_FACTOR
    scale = (124 + torch.arange(BLOCK_M * scale_k) % 7).to(torch.uint8).reshape(BLOCK_M, scale_k)
    expanded = scale.repeat_interleave(SCALE_FACTOR, dim=1)
    x = (values_ref * torch.exp2(expanded.to(torch.float32) - 127)).to(dtype)

    out_k = BLOCK_K // 2 if format == "e2m1" else BLOCK_K
    y = torch.empty((BLOCK_M, out_k), dtype=torch.uint8 if torch_fp8 is None else torch_fp8, device="cuda")
    scaled_downcast_broadcast_register_kernel[(1, )](x.cuda(), scale.cuda(), y, BLOCK_M, BLOCK_K, out_k, scale_k,
                                                     format, num_warps=4)

    torch.testing.assert_close(y.view(torch.uint8).cpu(), expected, atol=0, rtol=0)
