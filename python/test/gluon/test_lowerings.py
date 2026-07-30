import math

import torch
import pytest

import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as ttgl
from triton._internal_testing import (is_blackwell, is_cuda, is_hip, is_hip_gfx1250, is_hopper_or_newer,
                                       get_hip_lds_size)
from triton._C.libtriton import ir, gluon_ir
from triton._C.libtriton.gluon_ir import make_cga_layout
from triton.experimental.gluon.language.amd.gfx1250 import PartitionedSharedLayout
from triton.experimental.gluon.language.nvidia.blackwell import TensorMemoryLayout, allocate_tensor_memory

THREADS_PER_WARP = triton.runtime.driver.active.get_current_target().warp_size


def _is_layout_applicable(layout) -> bool:
    if isinstance(layout, (ttgl.BlockedLayout, ttgl.SwizzledSharedLayout, ttgl.DistributedLinearLayout)):
        return True
    elif isinstance(layout, ttgl.SliceLayout):
        return _is_layout_applicable(layout.parent)
    elif is_cuda():
        if isinstance(layout, ttgl.NVMMASharedLayout):
            return True
        mma_layout = layout.parent if isinstance(layout, ttgl.DotOperandLayout) else layout
        if not isinstance(mma_layout, ttgl.NVMMADistributedLayout):
            return False
        if mma_layout.version[0] >= 3 and not is_hopper_or_newer():
            return False
        return True
    elif is_hip():
        if layout in ["padded_shared_layout_single_interval", "padded_shared_layout_multi_interval"]:
            return True
        if THREADS_PER_WARP == 32:
            return isinstance(layout, ttgl.amd.AMDWMMALayout)
        return isinstance(layout, ttgl.amd.AMDMFMALayout)
    else:
        return True


def _filter_layouts(layouts):
    return [l for l in layouts if _is_layout_applicable(l)]


@gluon.constexpr_function
def _make_cga_broadcast(rank: ttgl.constexpr, num_ctas: ttgl.constexpr):
    if num_ctas == 1:
        return []
    n = num_ctas.bit_length() - 1
    return [[0] * rank for _ in range(n)]


@gluon.jit
def _combine(a, b):
    return a + b


@gluon.jit
def convert_1d_to_2d_slice_cga_kernel(out, HEAD: ttgl.constexpr, NUM_CTAS: ttgl.constexpr):
    layout_d: ttgl.constexpr = ttgl.BlockedLayout(
        [1],
        [32],
        [ttgl.num_warps()],
        [0],
        _make_cga_broadcast(1, NUM_CTAS),
    )
    layout_nd: ttgl.constexpr = ttgl.BlockedLayout(
        [1, 1],
        [1, 32],
        [ttgl.num_warps(), 1],
        [1, 0],
        _make_cga_broadcast(2, NUM_CTAS),
    )

    d = ttgl.arange(0, HEAD, layout=layout_d)
    x = d.to(ttgl.float32)
    dd = ttgl.arange(0, HEAD, layout=ttgl.SliceLayout(0, layout_nd))
    y = ttgl.convert_layout(x, ttgl.SliceLayout(0, layout_nd))
    ttgl.store(out + dd, y)


@gluon.jit
def scan_kernel(x_ptr, z_ptr, M: ttgl.constexpr, N: ttgl.constexpr, layout: ttgl.constexpr, axis: ttgl.constexpr):
    x_offs_m = ttgl.arange(0, M, layout=ttgl.SliceLayout(1, layout))[:, None]
    x_offs_n = ttgl.arange(0, N, layout=ttgl.SliceLayout(0, layout))[None, :]
    x = ttgl.load(x_ptr + x_offs_m * N + x_offs_n)
    y = ttgl.associative_scan(x, axis=axis, combine_fn=_combine)
    ttgl.store(z_ptr + x_offs_m * N + x_offs_n, y)


@pytest.mark.parametrize("M, N", [(32, 16), (32, 32), (32, 64), (64, 32)])
@pytest.mark.parametrize(
    "src_layout",
    _filter_layouts([
        ttgl.BlockedLayout([1, 4], [4, THREADS_PER_WARP // 4], [4, 1], [0, 1]),
        ttgl.BlockedLayout([1, 4], [8, THREADS_PER_WARP // 8], [4, 1], [0, 1]),
        ttgl.BlockedLayout([4, 1], [4, THREADS_PER_WARP // 4], [1, 4], [0, 1]),
        ttgl.BlockedLayout([2, 2], [4, THREADS_PER_WARP // 4], [2, 2], [0, 1]),
        ttgl.BlockedLayout([2, 2], [8, THREADS_PER_WARP // 8], [2, 2], [0, 1]),
        ttgl.BlockedLayout([1, 4], [4, THREADS_PER_WARP // 4], [4, 1], [1, 0]),
        ttgl.BlockedLayout([1, 4], [8, THREADS_PER_WARP // 8], [4, 1], [1, 0]),
        ttgl.BlockedLayout([4, 1], [4, THREADS_PER_WARP // 4], [1, 4], [1, 0]),
        ttgl.BlockedLayout([2, 2], [4, THREADS_PER_WARP // 4], [2, 2], [1, 0]),
        ttgl.BlockedLayout([2, 2], [8, THREADS_PER_WARP // 8], [2, 2], [1, 0]),
        ttgl.BlockedLayout([1, 2], [1, THREADS_PER_WARP], [1, 4], [1, 0]),
    ]))
@pytest.mark.parametrize("axis", [0, 1])
@pytest.mark.parametrize("sanitize_overflow", [False, True])
def test_scan_layouts(M, N, src_layout, axis, sanitize_overflow, device):

    torch.manual_seed(0)

    x = torch.randint(-100, 100, (M, N), dtype=torch.int32, device=device)
    z = torch.zeros((M, N), dtype=torch.int32, device=device)
    z_tri = torch.empty_like(z)

    scan_kernel[(1, 1, 1)](x, z_tri, M, N, src_layout, axis, num_warps=4, sanitize_overflow=sanitize_overflow,
                           debug=sanitize_overflow)

    z_ref = torch.cumsum(x, dim=axis, dtype=torch.int32)
    torch.testing.assert_close(z_tri, z_ref)


def test_scan_blocked_broadcast_layout(device):
    if not is_cuda():
        pytest.skip("requires CUDA")
    if THREADS_PER_WARP != 32:
        pytest.skip("requires 32-thread warps")

    M = 32
    # Broadcasting in register, lane and warp
    # - register=1 -> (1, 0)
    # - lane=1 -> (0, 0)
    #   lane=2 -> (2, 0)
    #   lane=4 -> (4, 0)
    #   lane=8 -> (8, 0)
    #   lane=16 -> (16, 0)
    # - warp=1 -> (0, 0)
    #   warp=2 -> (0, 0)
    # - block is a size 1 dimension
    src_layout = ttgl.BlockedLayout([2, 4], [16, 2], [2, 2], [1, 0])

    torch.manual_seed(0)
    x = torch.randn((M, 1), dtype=torch.float32, device=device)
    y = torch.empty_like(x)
    scan_kernel[(1, )](x, y, M, 1, src_layout, 0, num_warps=4)

    torch.testing.assert_close(y, torch.cumsum(x, dim=0))


def test_scan_blocked_broadcast_layout_multiblock(device):
    if not is_cuda():
        pytest.skip("requires CUDA")
    if THREADS_PER_WARP != 32:
        pytest.skip("requires 32-thread warps")

    M = 64
    # Broadcasting in lane for dim1 and multiple scan blocks along axis 0.
    src_layout = ttgl.BlockedLayout([2, 4], [16, 2], [1, 2], [1, 0])

    torch.manual_seed(0)
    x = torch.randn((M, 1), dtype=torch.float32, device=device)
    y = torch.empty_like(x)
    scan_kernel[(1, )](x, y, M, 1, src_layout, 0, num_warps=2)

    torch.testing.assert_close(y, torch.cumsum(x, dim=0))


@pytest.mark.skipif(not is_hopper_or_newer(), reason="Requires Hopper or newer")
@pytest.mark.parametrize("num_ctas", [2, 4, 8])
def test_convert_1d_to_2d_slice_cga(num_ctas, device):
    head = 64
    out = torch.empty((head, ), device=device, dtype=torch.float32)

    convert_1d_to_2d_slice_cga_kernel[(1, )](out, head, num_ctas, num_warps=2, num_ctas=num_ctas)

    torch.testing.assert_close(out, torch.arange(head, device=device, dtype=torch.float32))


@pytest.mark.skipif(not is_hopper_or_newer(), reason="Requires NVIDIA Hopper or newer")
@pytest.mark.parametrize("warp_specialize", [False, True])
def test_atomic_poll_two_ctas(warp_specialize, device):

    @gluon.jit
    def poll_partition(payload, flag, out):
        pid = ttgl.program_id(0)
        if pid == 0:
            ttgl.store(payload, 42)
            ttgl.atomic_xchg(flag, 1, sem="release", scope="gpu")
        else:
            matched = ttgl.atomic_poll(flag, 1, sem="acquire", scope="gpu", timeout_ns=1_000_000_000)
            if matched:
                ttgl.store(out, ttgl.load(payload))

    @gluon.jit
    def empty_partition():
        pass

    @gluon.jit
    def kernel(payload, flag, out, WARP_SPECIALIZE: ttgl.constexpr):
        if WARP_SPECIALIZE:
            ttgl.warp_specialize([
                (poll_partition, (payload, flag, out)),
                (empty_partition, ()),
            ], [4])
        else:
            poll_partition(payload, flag, out)

    payload = torch.zeros((1, ), device=device, dtype=torch.int32)
    flag = torch.zeros((1, ), device=device, dtype=torch.int32)
    out = torch.full((1, ), -1, device=device, dtype=torch.int32)

    kernel[(2, )](payload, flag, out, warp_specialize, num_warps=4)

    assert out.item() == 42


@pytest.mark.skipif(not is_hopper_or_newer(), reason="Requires NVIDIA Hopper or newer")
@pytest.mark.parametrize("num_ctas", [2, 4])
def test_cluster_barrier_in_warp_specialize(device, num_ctas):
    BLOCK = ttgl.constexpr(128)

    @gluon.jit
    def partition(out, offset: ttgl.constexpr):
        layout: ttgl.constexpr = ttgl.BlockedLayout([1], [32], [4], [0],
                                                    cga_layout=_make_cga_broadcast(1, ttgl.num_ctas()))
        offs = offset + ttgl.arange(0, BLOCK, layout=layout)
        ttgl.barrier(cluster=True)
        ttgl.store(out + offs, offs)

    @gluon.jit
    def kernel(out):
        ttgl.warp_specialize([
            (partition, (out, 0)),
            (partition, (out, BLOCK)),
        ], [4])

    out = torch.empty((2 * BLOCK.value, ), device=device, dtype=torch.int32)
    compiled = kernel[(1, )](out, num_warps=4, num_ctas=num_ctas)

    ptx = compiled.asm["ptx"]
    assert ptx.count("mbarrier.arrive.release.cluster.shared::cluster") == 2
    assert "mapa" not in ptx
    torch.testing.assert_close(out, torch.arange(2 * BLOCK.value, device=device, dtype=torch.int32))


@pytest.mark.skipif(not is_hopper_or_newer(), reason="Requires NVIDIA Hopper or newer")
@pytest.mark.parametrize("use_worker_partition", [False, True])
def test_convert_layout_cross_cta_in_warp_specialize(use_worker_partition, device):
    M = ttgl.constexpr(64)
    N = ttgl.constexpr(128)
    src_cga_layout = [[0, 1], [0, 2]]
    dst_cga_layout = [[0, 1], [1, 0]]
    src_layout = _with_cga_layout(_2d_layouts[0], src_cga_layout)
    dst_layout = _with_cga_layout(_2d_layouts[1], dst_cga_layout)

    @gluon.jit
    def convert_partition(x_ptr, y_ptr, src_layout: ttgl.constexpr, dst_layout: ttgl.constexpr):
        offs_m_src = ttgl.arange(0, M, layout=ttgl.SliceLayout(1, src_layout))[:, None]
        offs_n_src = ttgl.arange(0, N, layout=ttgl.SliceLayout(0, src_layout))[None, :]
        x = ttgl.load(x_ptr + offs_m_src * N + offs_n_src)
        y = ttgl.convert_layout(x, layout=dst_layout)
        offs_m_dst = ttgl.arange(0, M, layout=ttgl.SliceLayout(1, dst_layout))[:, None]
        offs_n_dst = ttgl.arange(0, N, layout=ttgl.SliceLayout(0, dst_layout))[None, :]
        ttgl.store(y_ptr + offs_m_dst * N + offs_n_dst, y)

    @gluon.jit
    def empty_partition():
        pass

    @gluon.jit
    def kernel(x_ptr, y_ptr, src_layout: ttgl.constexpr, dst_layout: ttgl.constexpr,
               use_worker_partition: ttgl.constexpr):
        if use_worker_partition:
            ttgl.warp_specialize([
                (empty_partition, ()),
                (convert_partition, (x_ptr, y_ptr, src_layout, dst_layout)),
            ], [4])
        else:
            ttgl.warp_specialize([
                (convert_partition, (x_ptr, y_ptr, src_layout, dst_layout)),
                (empty_partition, ()),
            ], [4])

    torch.manual_seed(0)
    x = torch.randn((M.value, N.value), dtype=torch.float16, device=device)
    y = torch.zeros_like(x)
    compiled = kernel[(1, )](x, y, src_layout, dst_layout, use_worker_partition, num_warps=4, num_ctas=4)

    ptx = compiled.asm["ptx"]
    assert "ld.shared::cluster" in ptx
    torch.testing.assert_close(y, x, rtol=0, atol=0)


def _swizzled_warp_layouts_1d():
    """1D DistributedLinearLayout test layouts (non-injective, lowered as GenericLinearEncoding)."""

    def ilog2(x):
        return x.bit_length() - 1

    return [
        # Non-injective 1D: warp bases overlap with lane coverage
        ttgl.DistributedLinearLayout(
            reg_bases=[[1], [2]],
            lane_bases=[[4], [8], [16], [32], [64]] + ([[0]] * (ilog2(THREADS_PER_WARP) - 5)),
            warp_bases=[[64], [128]],
            block_bases=[],
            shape=[256],
        ),
        # Non-injective 1D: warp bases overlap with register coverage
        ttgl.DistributedLinearLayout(
            reg_bases=[[1], [2], [4]],
            lane_bases=[[8], [16], [32], [64], [128]] + ([[0]] * (ilog2(THREADS_PER_WARP) - 5)),
            warp_bases=[[4], [256]],
            block_bases=[],
            shape=[512],
        ),
        # Non-power-of-two basis (96 = 64 + 32) in the warp bases
        ttgl.DistributedLinearLayout(
            reg_bases=[[1], [2]],
            lane_bases=[[4], [8], [16], [32], [64]] + ([[0]] * (ilog2(THREADS_PER_WARP) - 5)),
            warp_bases=[[96], [128]],
            block_bases=[],
            shape=[256],
        ),
    ]


def _swizzled_warp_layouts_2d():
    """2D DistributedLinearLayout test layouts (swizzled warp bases and/or non-injective)."""

    def ilog2(x):
        return x.bit_length() - 1

    return [
        # Mildly swizzled: one warp base touches both dims
        ttgl.DistributedLinearLayout(
            reg_bases=[[1, 0], [0, 1]],
            lane_bases=[[2, 0], [4, 0], [8, 0], [0, 2], [0, 4]] + ([[0, 0]] * (ilog2(THREADS_PER_WARP) - 5)),
            warp_bases=[[16, 8], [0, 8]],
            block_bases=[],
            shape=[32, 16],
        ),
        # Aggressively swizzled: both warp bases touch both dims
        ttgl.DistributedLinearLayout(
            reg_bases=[[1, 0], [0, 1]],
            lane_bases=[[2, 0], [4, 0], [8, 0], [0, 2], [0, 4]] + ([[0, 0]] * (ilog2(THREADS_PER_WARP) - 5)),
            warp_bases=[[4, 2], [8, 4]],
            block_bases=[],
            shape=[16, 8],
        ),
        # Swizzled warp + broadcasting in registers
        ttgl.DistributedLinearLayout(
            reg_bases=[[1, 0], [0, 0], [0, 1]],
            lane_bases=[[2, 0], [4, 0], [8, 0], [0, 2], [0, 4]] + ([[0, 0]] * (ilog2(THREADS_PER_WARP) - 5)),
            warp_bases=[[16, 8], [0, 8]],
            block_bases=[],
            shape=[32, 16],
        ),
        # non-injective
        ttgl.DistributedLinearLayout(
            reg_bases=[[0, 1], [0, 2], [0, 4], [0, 16], [32, 0]],
            lane_bases=[[1, 0], [2, 0], [4, 0], [8, 0], [0, 8]] + ([[0, 0]] * (ilog2(THREADS_PER_WARP) - 5)),
            warp_bases=[[32, 0], [16, 0]],
            block_bases=[],
            shape=[64, 32],
        ),
        # non-power-of-two basis (24 = 16 + 8) in the warp bases
        ttgl.DistributedLinearLayout(
            reg_bases=[[1, 0], [0, 1]],
            lane_bases=[[2, 0], [4, 0], [8, 0], [0, 2], [0, 4]] + ([[0, 0]] * (ilog2(THREADS_PER_WARP) - 5)),
            warp_bases=[[24, 0], [0, 8]],
            block_bases=[],
            shape=[32, 16],
        ),
    ]


def _swizzled_warp_layouts():
    """All swizzled/non-injective DistributedLinearLayout test layouts (1D and 2D)."""
    return _swizzled_warp_layouts_1d() + _swizzled_warp_layouts_2d()


# ===--- Tests with swizzled/non-injective DistributedLinearLayout ---===


@pytest.mark.parametrize("src_layout", _filter_layouts(_swizzled_warp_layouts()))
def test_elementwise_generic_linear(src_layout, device):
    shape = src_layout.shape
    num_warps = 2**len(src_layout.warp_bases)

    if len(shape) == 1:
        N, = shape

        @gluon.jit
        def kernel(x_ptr, y_ptr, N: ttgl.constexpr, layout: ttgl.constexpr):
            offs = ttgl.arange(0, N, layout=layout)
            x = ttgl.load(x_ptr + offs)
            y = x * x + x
            ttgl.store(y_ptr + offs, y)

        x = torch.randn(N, dtype=torch.float32, device=device)
        y = torch.empty_like(x)
        kernel[(1, )](x, y, N, src_layout, num_warps=num_warps)
    else:
        M, N = shape

        @gluon.jit
        def kernel(x_ptr, y_ptr, M: ttgl.constexpr, N: ttgl.constexpr, layout: ttgl.constexpr):
            offs_m = ttgl.arange(0, M, layout=ttgl.SliceLayout(1, layout))[:, None]
            offs_n = ttgl.arange(0, N, layout=ttgl.SliceLayout(0, layout))[None, :]
            x = ttgl.load(x_ptr + offs_m * N + offs_n)
            y = x * x + x
            ttgl.store(y_ptr + offs_m * N + offs_n, y)

        x = torch.randn((M, N), dtype=torch.float32, device=device)
        y = torch.empty_like(x)
        kernel[(1, )](x, y, M, N, src_layout, num_warps=num_warps)

    torch.testing.assert_close(y, x * x + x)


@pytest.mark.parametrize("src_layout", _filter_layouts(_swizzled_warp_layouts_2d()))
def test_expand_dims_generic_linear(src_layout, device):
    M, N = src_layout.shape
    num_warps = 2**len(src_layout.warp_bases)

    @gluon.jit
    def kernel(x_ptr, y_ptr, M: ttgl.constexpr, layout: ttgl.constexpr):
        offs = ttgl.arange(0, M, layout=ttgl.SliceLayout(1, layout))
        x = ttgl.load(x_ptr + offs)
        x_2d = ttgl.expand_dims(x, axis=1)
        offs_2d = ttgl.expand_dims(offs, axis=1)
        ttgl.store(y_ptr + offs_2d, x_2d)

    torch.manual_seed(17)
    x = torch.randint(0, 4, (M, 1), dtype=torch.float32, device=device)
    y = torch.zeros((M, 1), dtype=torch.float32, device=device)
    kernel[(1, )](x, y, M, src_layout, num_warps=num_warps)
    torch.testing.assert_close(y, x)


@pytest.mark.parametrize("src_layout", _filter_layouts(_swizzled_warp_layouts()))
def test_reshape_generic_linear(src_layout, device):
    shape = src_layout.shape
    num_warps = 2**len(src_layout.warp_bases)
    total = 1
    for s in shape:
        total *= s

    if len(shape) == 1:
        N, = shape

        @gluon.jit
        def kernel(x_ptr, y_ptr, N: ttgl.constexpr, layout: ttgl.constexpr):
            offs = ttgl.arange(0, N, layout=layout)
            x = ttgl.load(x_ptr + offs)
            reshaped = x.reshape([N // 2, 2])
            flat = reshaped.reshape([N])
            ttgl.store(y_ptr + offs, flat)

        torch.manual_seed(0)
        x = torch.randn(N, dtype=torch.float32, device=device)
        y = torch.zeros_like(x)
        kernel[(1, )](x, y, N, src_layout, num_warps=num_warps)
    else:
        M, N = shape

        @gluon.jit
        def kernel(x_ptr, y_ptr, M: ttgl.constexpr, N: ttgl.constexpr, layout: ttgl.constexpr):
            offs_m = ttgl.arange(0, M, layout=ttgl.SliceLayout(1, layout))[:, None]
            offs_n = ttgl.arange(0, N, layout=ttgl.SliceLayout(0, layout))[None, :]
            x = ttgl.load(x_ptr + offs_m * N + offs_n)
            flat = x.reshape([M * N])
            y = flat.reshape([M, N])
            ttgl.store(y_ptr + offs_m * N + offs_n, y)

        torch.manual_seed(0)
        x = torch.randn((M, N), dtype=torch.float32, device=device)
        y = torch.zeros_like(x)
        kernel[(1, )](x, y, M, N, src_layout, num_warps=num_warps)

    torch.testing.assert_close(y, x)


@pytest.mark.parametrize("src_layout", _filter_layouts(_swizzled_warp_layouts_2d()))
def test_permute_generic_linear(src_layout, device):
    M, N = src_layout.shape
    num_warps = 2**len(src_layout.warp_bases)

    @gluon.jit
    def kernel(x_ptr, y_ptr, M: ttgl.constexpr, N: ttgl.constexpr, layout: ttgl.constexpr):
        offs_m = ttgl.arange(0, M, layout=ttgl.SliceLayout(1, layout))[:, None]
        offs_n = ttgl.arange(0, N, layout=ttgl.SliceLayout(0, layout))[None, :]
        x = ttgl.load(x_ptr + offs_m * N + offs_n)
        xt = ttgl.permute(x, [1, 0])
        y = ttgl.permute(xt, [1, 0])
        ttgl.store(y_ptr + offs_m * N + offs_n, y)

    torch.manual_seed(0)
    x = torch.randn((M, N), dtype=torch.float32, device=device)
    y = torch.zeros_like(x)
    kernel[(1, )](x, y, M, N, src_layout, num_warps=num_warps)
    torch.testing.assert_close(y, x)


@pytest.mark.parametrize("src_layout", _filter_layouts(_swizzled_warp_layouts()))
def test_split_join_generic_linear(src_layout, device):
    if any(all(v == 0 for v in b) for b in src_layout.reg_bases):
        pytest.skip(
            "Broadcast register bases cause join/split types mismatch. This is not related to GenericLinearEncodingAttr."
        )
    shape = src_layout.shape
    num_warps = 2**len(src_layout.warp_bases)

    if len(shape) == 1:
        N, = shape

        @gluon.jit
        def kernel(x_ptr, y_ptr, N: ttgl.constexpr, layout: ttgl.constexpr):
            offs = ttgl.arange(0, N, layout=layout)
            x = ttgl.load(x_ptr + offs)
            joined = ttgl.join(x, x * 2)
            a, b = ttgl.split(joined)
            result = a + b
            result_layout: ttgl.constexpr = result.type.layout
            ttgl.store(y_ptr + ttgl.arange(0, N, layout=result_layout), result)

        torch.manual_seed(0)
        x = torch.randn(N, dtype=torch.float32, device=device)
        y = torch.zeros_like(x)
        kernel[(1, )](x, y, N, src_layout, num_warps=num_warps)
    else:
        M, N = shape

        @gluon.jit
        def kernel(x_ptr, y_ptr, M: ttgl.constexpr, N: ttgl.constexpr, layout: ttgl.constexpr):
            offs_m = ttgl.arange(0, M, layout=ttgl.SliceLayout(1, layout))[:, None]
            offs_n = ttgl.arange(0, N, layout=ttgl.SliceLayout(0, layout))[None, :]
            x = ttgl.load(x_ptr + offs_m * N + offs_n)
            joined = ttgl.join(x, x * 2)
            a, b = ttgl.split(joined)
            result = a + b
            result_layout: ttgl.constexpr = result.type.layout
            out_offs_m = ttgl.arange(0, M, layout=ttgl.SliceLayout(1, result_layout))[:, None]
            out_offs_n = ttgl.arange(0, N, layout=ttgl.SliceLayout(0, result_layout))[None, :]
            ttgl.store(y_ptr + out_offs_m * N + out_offs_n, result)

        torch.manual_seed(0)
        x = torch.randn((M, N), dtype=torch.float32, device=device)
        y = torch.zeros_like(x)
        kernel[(1, )](x, y, M, N, src_layout, num_warps=num_warps)

    torch.testing.assert_close(y, x + x * 2)


@pytest.mark.parametrize("src_layout", _filter_layouts(_swizzled_warp_layouts_2d()))
def test_broadcast_generic_linear(src_layout, device):
    M, N = src_layout.shape
    num_warps = 2**len(src_layout.warp_bases)

    @gluon.jit
    def kernel(x_ptr, y_ptr, z_ptr, M: ttgl.constexpr, N: ttgl.constexpr, layout: ttgl.constexpr):
        offs_m = ttgl.arange(0, M, layout=ttgl.SliceLayout(1, layout))[:, None]
        offs_n = ttgl.arange(0, N, layout=ttgl.SliceLayout(0, layout))[None, :]
        col = ttgl.load(x_ptr + offs_m)
        row = ttgl.load(y_ptr + offs_n)
        result = col + row
        ttgl.store(z_ptr + offs_m * N + offs_n, result)

    torch.manual_seed(0)
    x = torch.randn(M, dtype=torch.float32, device=device)
    y = torch.randn(N, dtype=torch.float32, device=device)
    z = torch.zeros((M, N), dtype=torch.float32, device=device)
    kernel[(1, )](x, y, z, M, N, src_layout, num_warps=num_warps)
    torch.testing.assert_close(z, x[:, None] + y[None, :])


def _shared_layout_kinds():
    kinds = ["swizzled_trivial", "swizzled", "padded"]
    if is_hip():
        kinds += ["partitioned_swizzled", "partitioned_padded"]
    return kinds


def _make_shared_layout(kind, shape):
    order = list(reversed(range(len(shape))))
    if kind == "swizzled_trivial":
        return ttgl.SwizzledSharedLayout(vec=1, per_phase=1, max_phase=1, order=order)
    if kind == "swizzled":
        return ttgl.SwizzledSharedLayout(vec=4, per_phase=2, max_phase=4, order=order)
    if kind == "padded":
        return ttgl.PaddedSharedLayout.with_identity_for(interval_padding_pairs=[[16, 4]], shape=list(shape),
                                                         order=order)
    if kind == "partitioned_swizzled":
        inner = ttgl.SwizzledSharedLayout(vec=4, per_phase=2, max_phase=4, order=order)
        return PartitionedSharedLayout(num_partitions=2, num_groups=1, partition_dim=0, partition_layout=inner)
    if kind == "partitioned_padded":
        inner = ttgl.PaddedSharedLayout.with_identity_for(interval_padding_pairs=[[16, 4]], shape=list(shape),
                                                          order=order)
        return PartitionedSharedLayout(num_partitions=2, num_groups=1, partition_dim=0, partition_layout=inner)
    raise ValueError(f"Unknown shared layout kind: {kind}")


@pytest.mark.parametrize("src_layout", _filter_layouts(_swizzled_warp_layouts()))
@pytest.mark.parametrize("shared_kind", _shared_layout_kinds())
def test_local_load_store_generic_linear(src_layout, shared_kind, device):
    """Round-trip through shared memory using a swizzled/non-injective DistributedLinearLayout.

    Exercises local_store (smem.store) and local_load (smem.load) lowerings for
    GenericLinearEncoding sources across various shared-memory layouts.
    """
    shape = tuple(src_layout.shape)
    shared_layout = _make_shared_layout(shared_kind, shape)
    num_warps = 2**len(src_layout.warp_bases)

    @gluon.jit
    def kernel(x_ptr, y_ptr, shape: ttgl.constexpr, layout: ttgl.constexpr, shared_layout: ttgl.constexpr):
        if len(shape) == 1:
            offs = ttgl.arange(0, shape[0], layout=layout)
        else:
            offs_m = ttgl.arange(0, shape[0], layout=ttgl.SliceLayout(1, layout))[:, None]
            offs_n = ttgl.arange(0, shape[1], layout=ttgl.SliceLayout(0, layout))[None, :]
            offs = offs_m * shape[1] + offs_n
        x = ttgl.load(x_ptr + offs)
        smem = ttgl.allocate_shared_memory(x.dtype, shape, shared_layout)
        smem.store(x)
        y = smem.load(layout)
        ttgl.store(y_ptr + offs, y)

    torch.manual_seed(0)
    x = torch.randn(shape, dtype=torch.float32, device=device)
    y = torch.empty_like(x)
    kernel[(1, )](x, y, shape, src_layout, shared_layout, num_warps=num_warps)

    torch.testing.assert_close(y, x)


@pytest.mark.skipif(not is_hopper_or_newer(), reason="Requires Hopper or newer")
def test_local_store_tmem_32x32b_2cta_splitm_to_splitk(device):
    shape = (256, 128)
    src_layout = ttgl.DistributedLinearLayout(
        reg_bases=[[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [128, 0]],
        lane_bases=[[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]],
        warp_bases=[[32, 0], [64, 0]],
        block_bases=[[0, 64]],
        shape=list(shape),
    )
    dst_layout = ttgl.BlockedLayout(
        size_per_thread=[1, shape[1]],
        threads_per_warp=[THREADS_PER_WARP, 1],
        warps_per_cta=[4, 1],
        order=[0, 1],
        cga_layout=[[1, 0]],
    )
    shared_layout = ttgl.NVMMASharedLayout(
        swizzle_byte_width=128,
        transposed=False,
        element_bitwidth=16,
        rank=2,
        cga_layout=[[1, 0]],
    )

    @gluon.jit
    def kernel(x_ptr, y_ptr, shape: ttgl.constexpr, src_layout: ttgl.constexpr, dst_layout: ttgl.constexpr,
               shared_layout: ttgl.constexpr):
        K: ttgl.constexpr = shape[0]
        M: ttgl.constexpr = shape[1]
        src_offs_k = ttgl.arange(0, K, layout=ttgl.SliceLayout(1, src_layout))[:, None]
        src_offs_m = ttgl.arange(0, M, layout=ttgl.SliceLayout(0, src_layout))[None, :]
        src_offs = src_offs_k * M + src_offs_m

        x = ttgl.load(x_ptr + src_offs)
        smem = ttgl.allocate_shared_memory(x.dtype, shape, shared_layout)
        smem.store(x)
        ttgl.barrier(cluster=True)
        y = smem.load(dst_layout)

        dst_offs_k = ttgl.arange(0, K, layout=ttgl.SliceLayout(1, dst_layout))[:, None]
        dst_offs_m = ttgl.arange(0, M, layout=ttgl.SliceLayout(0, dst_layout))[None, :]
        dst_offs = dst_offs_k * M + dst_offs_m
        ttgl.store(y_ptr + dst_offs, y)

    torch.manual_seed(0)
    x = torch.randn(shape, dtype=torch.float16, device=device)
    y = torch.empty_like(x)

    kernel[(1, )](x, y, shape, src_layout, dst_layout, shared_layout, num_warps=4, num_ctas=2)

    torch.testing.assert_close(y, x, rtol=0, atol=0)


@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
@pytest.mark.parametrize("instr_variant", ["32x32b", "16x64b", "16x128b", "16x256b"])
def test_tmem_load_store_instruction_sizes(instr_variant, device):
    shape = (128, 128)

    @gluon.jit
    def kernel(x_ptr, y_ptr, shape: ttgl.constexpr, instr_variant: ttgl.constexpr):
        M: ttgl.constexpr = shape[0]
        N: ttgl.constexpr = shape[1]
        tmem_layout: ttgl.constexpr = TensorMemoryLayout(block=shape, col_stride=1)
        tmem = allocate_tensor_memory(ttgl.float32, shape, tmem_layout)
        tmem_reg_layout: ttgl.constexpr = tmem.get_reg_layout(instr_variant=instr_variant)
        offs_m = ttgl.arange(0, M, layout=ttgl.SliceLayout(1, tmem_reg_layout))[:, None]
        offs_n = ttgl.arange(0, N, layout=ttgl.SliceLayout(0, tmem_reg_layout))[None, :]
        offsets = offs_m * N + offs_n
        x = ttgl.load(x_ptr + offsets)
        tmem.store(x)
        y = tmem.load(tmem_reg_layout)
        ttgl.store(y_ptr + offsets, y)

    torch.manual_seed(0)
    x = torch.randn(shape, dtype=torch.float32, device=device)
    y = torch.empty_like(x)

    compiled = kernel[(1, )](x, y, shape, instr_variant, num_warps=4)

    torch.testing.assert_close(y, x, rtol=0, atol=0)
    ptx = compiled.asm["ptx"]
    assert f"tcgen05.st.sync.aligned.{instr_variant}" in ptx
    assert f"tcgen05.ld.sync.aligned.{instr_variant}" in ptx


def _funky_reduce_layouts():

    def ilog2(x):
        return x.bit_length() - 1

    # Broadcasting here and there and bases in a weird order
    layouts = [
        # Funky layout where the warp bases fit in the lane bases
        ttgl.DistributedLinearLayout(
            reg_bases=[[0, 8], [1, 0], [0, 0], [2, 0], [4, 0], [8, 0], [16, 0]],
            lane_bases=[[0, 1], [0, 0], [64, 0], [0, 2], [0, 4]] + ([[0, 0]] * (ilog2(THREADS_PER_WARP) - 5)),
            warp_bases=[[32, 0], [0, 16]],
            block_bases=[],
            shape=[128, 32],
        ),
        # Another funky layout for good measure
        ttgl.DistributedLinearLayout(
            reg_bases=[[1, 0], [2, 0]],
            lane_bases=[[0, 1], [4, 0], [0, 2], [8, 0], [0, 4]] + ([[0, 0]] * (ilog2(THREADS_PER_WARP) - 5)),
            warp_bases=[[16, 0], [32, 0]],
            block_bases=[],
            shape=[64, 8],
        ),
        # Funky layout where warp bases do *not* fit in the lane bases
        ttgl.DistributedLinearLayout(
            reg_bases=[[1, 0], [2, 0]],
            lane_bases=[[0, 1], [4, 0], [0, 2], [8, 0], [0, 4]] + ([[0, 0]] * (ilog2(THREADS_PER_WARP) - 5)),
            warp_bases=[[16, 0], [32, 0], [64, 0]],
            block_bases=[],
            shape=[128, 8],
        ),
        # Basic funky layout with block bases. They fit in the lane bases
        ttgl.DistributedLinearLayout(
            reg_bases=[[1, 0], [2, 0]],
            lane_bases=[[0, 1], [4, 0], [0, 2], [8, 0], [0, 0]] + ([[0, 0]] * (ilog2(THREADS_PER_WARP) - 5)),
            warp_bases=[[16, 0], [32, 0]],
            block_bases=[[64, 0]],
            shape=[128, 4],
        ),
        # Funky layout with two convert_layouts with block_bases
        ttgl.DistributedLinearLayout(
            reg_bases=[],
            lane_bases=[[0, 1], [0, 4], [0, 2], [1, 0], [0, 0]] + ([[0, 0]] * (ilog2(THREADS_PER_WARP) - 5)),
            warp_bases=[[4, 0], [8, 0]],
            block_bases=[[2, 0]],
            shape=[16, 8],
        ),
        # Three convert_layouts
        ttgl.DistributedLinearLayout(
            reg_bases=[],
            lane_bases=[[0, 1], [0, 4], [0, 2], [1, 0], [0, 0]] + ([[0, 0]] * (ilog2(THREADS_PER_WARP) - 5)),
            warp_bases=[[4, 0], [8, 0], [16, 0], [128, 0], [512, 0]],
            block_bases=[[2, 0], [32, 0], [64, 0], [256, 0]],
            shape=[1024, 8],
        ),
    ]
    for axis in [0, 1]:
        for layout in layouts:
            yield (layout, axis)


@pytest.mark.parametrize("src_layout, axis", list(_funky_reduce_layouts()))
def test_reduce_funky_layout(src_layout, axis, device):

    shape = tuple(src_layout.shape)
    num_warps = 2**len(src_layout.warp_bases)
    num_ctas = 2**len(src_layout.block_bases)
    # TODO: Remove this once AMD supports num_ctas > 1
    if num_ctas > 1 and not is_hopper_or_newer():
        pytest.skip("num_ctas > 1 requires NVIDIA SM90+ (Hopper)")

    torch.manual_seed(0)
    x = torch.randn(shape, dtype=torch.float32, device=device)
    y = torch.empty(shape[1 - axis], dtype=torch.float32, device=device)

    @gluon.jit
    def kernel(x_ptr, y_ptr, shape: ttgl.constexpr, axis: ttgl.constexpr, layout: ttgl.constexpr):
        x_offs_m = ttgl.arange(0, shape[0], layout=ttgl.SliceLayout(1, layout))[:, None]
        x_offs_n = ttgl.arange(0, shape[1], layout=ttgl.SliceLayout(0, layout))[None, :]
        x = ttgl.load(x_ptr + x_offs_m * shape[1] + x_offs_n)
        y = ttgl.sum(x, axis=axis)
        y_offs = ttgl.arange(0, shape[1 - axis])
        ttgl.store(y_ptr + y_offs, y)

    pm = kernel[(1, )](x, y, shape, axis, src_layout, num_warps=num_warps, num_ctas=num_ctas)

    torch.testing.assert_close(y, torch.sum(x, dim=axis))

    def bases_along_axis(bases, axis):
        return sum(basis[axis] != 0 for basis in bases)

    axis_warps = bases_along_axis(src_layout.warp_bases, axis)
    axis_blocks = bases_along_axis(src_layout.block_bases, axis)

    # warp-sync
    if is_cuda() and axis_warps + axis_blocks == 0:
        assert pm.asm["ptx"].count("bar.sync") == 0


def _reduce_linear_layouts():
    if THREADS_PER_WARP == 32:
        return [
            ttgl.DistributedLinearLayout(
                reg_bases=[[0, 16], [1, 0], [2, 0], [4, 0], [8, 0], [16, 0]],
                lane_bases=[[0, 0], [0, 1], [0, 2], [0, 4], [0, 8]],
                warp_bases=[[32, 0], [0, 32]],
                block_bases=[],
                shape=[64, 64],
            )
        ]
    elif THREADS_PER_WARP == 64:
        return [
            ttgl.DistributedLinearLayout(
                reg_bases=[[0, 16], [1, 0], [2, 0], [4, 0], [8, 0], [16, 0]],
                lane_bases=[[0, 0], [0, 1], [0, 2], [0, 4], [0, 8], [0, 64]],
                warp_bases=[[32, 0], [0, 32]],
                block_bases=[],
                shape=[64, 128],
            )
        ]
    else:
        raise RuntimeError(f"Unsupported THREADS_PER_WARP: {THREADS_PER_WARP}")


def _reduce_layouts():
    shapes = [(128, 16), (32, 128), (32, 32), (16, 16)]
    layouts = _filter_layouts([
        # FIXME: Do not enable these tests until the SLPVectorizor problem with nvptx target has been resolved
        # SliceLayout(dim=1, parent=BlockedLayout([1, 4, 1], [1, 8, THREADS_PER_WARP // 8], [1, 1, 4], [2, 0, 1], [1, 1, 1], [1, 1, 1], [0, 1, 2])),
        # SliceLayout(dim=0, parent=BlockedLayout([1, 4, 1], [1, 8, THREADS_PER_WARP // 8], [1, 4, 1], [2, 1, 0], [1, 1, 1], [1, 1, 1], [0, 1, 2])),
        ttgl.BlockedLayout([1, 4], [8, THREADS_PER_WARP // 8], [4, 1], [1, 0]),
        ttgl.BlockedLayout([1, 4], [8, THREADS_PER_WARP // 8], [4, 1], [0, 1]),
        ttgl.NVMMADistributedLayout(version=[2, 0], warps_per_cta=[2, 4], instr_shape=[16, 8]),
        ttgl.NVMMADistributedLayout(version=[3, 0], warps_per_cta=[4, 1], instr_shape=[16, 16, 16]),
        ttgl.amd.AMDMFMALayout(version=1, instr_shape=[32, 32, 8], transposed=True, warps_per_cta=[1, 4]),
        ttgl.amd.AMDMFMALayout(version=2, instr_shape=[32, 32, 8], transposed=True, warps_per_cta=[1, 4]),
        ttgl.amd.AMDMFMALayout(version=3, instr_shape=[32, 32, 8], transposed=True, warps_per_cta=[1, 4]),
        ttgl.amd.AMDMFMALayout(version=4, instr_shape=[32, 32, 16], transposed=True, warps_per_cta=[1, 4]),
        ttgl.amd.AMDWMMALayout(version=1, transposed=True, warp_bases=[[0, 1], [0, 2]]),
        ttgl.amd.AMDWMMALayout(version=2, transposed=True, warp_bases=[[0, 1], [0, 2]]),
        ttgl.DotOperandLayout(
            parent=ttgl.NVMMADistributedLayout(version=[2, 0], warps_per_cta=[2, 4], instr_shape=[16, 8]),
            operand_index=1, k_width=8),
        ttgl.DotOperandLayout(
            parent=ttgl.NVMMADistributedLayout(version=[3, 0], warps_per_cta=[8, 1], instr_shape=[16, 32, 16]),
            operand_index=0, k_width=2),
        ttgl.SliceLayout(
            dim=0, parent=ttgl.NVMMADistributedLayout(version=[2, 0], warps_per_cta=[4, 1, 1], instr_shape=[1, 16, 8])),
        ttgl.SliceLayout(
            dim=1, parent=ttgl.DotOperandLayout(
                parent=ttgl.NVMMADistributedLayout(version=[2, 0], warps_per_cta=[4, 1, 1], instr_shape=[1, 16, 8]),
                operand_index=1, k_width=2)),
    ])

    rets = []
    for (M, N) in shapes:
        for layout in layouts:
            if isinstance(layout, (ttgl.amd.AMDMFMALayout, ttgl.amd.AMDWMMALayout, ttgl.NVMMADistributedLayout)):
                instr_shape = layout.instr_shape
                if M < instr_shape[0] or N < instr_shape[1]:
                    continue
            rets.append((M, N, layout))
    return rets


def _reduce_cases():
    for layout in _reduce_linear_layouts():
        yield (layout.shape[0], layout.shape[1], layout)
    for M, N, layout in _reduce_layouts():
        yield (M, N, layout)


@pytest.mark.parametrize("M, N, src_layout", _reduce_cases())
@pytest.mark.parametrize("axis", [0, 1])
@pytest.mark.parametrize("epilogue_kind", ['reduce1d', 'reduce2d', 'expand_reduce2d'])
@pytest.mark.parametrize("dtype_str, sanitize_overflow", [("int32", False), ("int32", True), ("float32", False),
                                                          ("float16", False)])
@pytest.mark.parametrize("reduce_op", ["sum", "max"])
def test_reduce_layouts(M, N, src_layout, axis, epilogue_kind, dtype_str, sanitize_overflow, reduce_op, device):

    @gluon.jit
    def _add(a, b):
        return a + b

    @gluon.jit
    def _max(a, b):
        return ttgl.maximum(a, b)

    combine_fn = _add if reduce_op == "sum" else _max

    @gluon.jit
    def kernel(x_ptr, z_ptr, M: ttgl.constexpr, N: ttgl.constexpr, layout: ttgl.constexpr, axis: ttgl.constexpr,
               epilogue_kind: ttgl.constexpr):
        x_offs_m = ttgl.arange(0, M, layout=ttgl.SliceLayout(1, layout))[:, None]
        x_offs_n = ttgl.arange(0, N, layout=ttgl.SliceLayout(0, layout))[None, :]
        x = ttgl.load(x_ptr + x_offs_m * N + x_offs_n)
        y = ttgl.reduce(x, axis=axis, combine_fn=combine_fn)
        if epilogue_kind == "reduce1d":
            if axis == 0:
                z_offs = ttgl.arange(0, N, layout=ttgl.SliceLayout(0, layout))
            else:
                z_offs = ttgl.arange(0, M, layout=ttgl.SliceLayout(1, layout))
            ttgl.store(z_ptr + z_offs, y)
        elif epilogue_kind == "reduce2d":
            y = ttgl.reduce(y, axis=0, combine_fn=combine_fn)
            ttgl.store(z_ptr, y)
        elif epilogue_kind == "expand_reduce2d":
            y = ttgl.expand_dims(y, axis=axis)
            y = ttgl.reduce(y, axis=1 - axis, combine_fn=combine_fn)
            z_offs = ttgl.arange(0, 1, layout=ttgl.SliceLayout(1 - axis, layout))
            ttgl.store(z_ptr + z_offs, y)

    torch.manual_seed(0)

    torch_dtype = getattr(torch, dtype_str)
    x = torch.randint(-10, 10, (M, N), dtype=torch.int32, device=device).to(torch_dtype)
    out_shape = (1, 1) if "reduce2d" in epilogue_kind else (1, N) if axis == 0 else (M, 1)
    z = torch.empty(out_shape, dtype=torch_dtype, device=device)

    num_warps = int(torch.prod(torch.tensor(ttgl._layouts.warps_per_cta(src_layout, (M, N)))))
    kernel[(1, 1, 1)](x, z, M, N, src_layout, axis, num_warps=num_warps, epilogue_kind=epilogue_kind,
                      sanitize_overflow=sanitize_overflow, debug=sanitize_overflow)

    reduce_fn = torch.sum if reduce_op == "sum" else torch.amax
    z_ref = reduce_fn(x, dim=axis, keepdim=True)
    if epilogue_kind in ("expand_reduce2d", "reduce2d"):
        z_ref = reduce_fn(z_ref, dim=1 - axis, keepdim=True)
    torch.testing.assert_close(z, z_ref.to(torch_dtype))


@pytest.mark.parametrize("M", [32, 64, 128, 256])
@pytest.mark.parametrize(
    "src_layout",
    _filter_layouts([
        ttgl.BlockedLayout([1, 4], [1, THREADS_PER_WARP], [4, 1], [1, 0]),
        ttgl.BlockedLayout([1, 4], [1, THREADS_PER_WARP], [2, 2], [1, 0]),
        ttgl.NVMMADistributedLayout(version=[2, 0], warps_per_cta=[4, 1], instr_shape=[16, 8]),
    ]))
def test_store_layouts(M, src_layout, device):

    @gluon.jit
    def kernel(x_ptr, y_ptr, M: ttgl.constexpr, layout: ttgl.constexpr):
        offs = ttgl.arange(0, M, layout=ttgl.SliceLayout(1, layout))
        x = ttgl.load(x_ptr + offs)
        x_2d = ttgl.expand_dims(x, axis=1)
        offs_2d = ttgl.expand_dims(offs, axis=1)
        ttgl.store(y_ptr + offs_2d, x_2d)

    torch.manual_seed(17)
    x = torch.randint(0, 4, (M, 1), dtype=torch.float32, device=device)
    y = torch.zeros((M, 1), dtype=torch.float32, device=device)
    kernel[(1, )](x, y, M, src_layout, num_warps=4)
    torch.testing.assert_close(y, x)


_1d_layouts = _filter_layouts([
    ttgl.BlockedLayout([1, 4], [1, THREADS_PER_WARP], [4, 1], [1, 0]),
    ttgl.BlockedLayout([1, 4], [1, THREADS_PER_WARP], [2, 2], [1, 0]),
    ttgl.NVMMADistributedLayout(version=[3, 0], warps_per_cta=[4, 1], instr_shape=[16, 32, 16]),
    ttgl.NVMMADistributedLayout(version=[2, 0], warps_per_cta=[4, 1], instr_shape=[16, 8]),
    ttgl.DotOperandLayout(
        parent=ttgl.NVMMADistributedLayout(version=[3, 0], warps_per_cta=[4, 1], instr_shape=[16, 32, 16]),
        operand_index=0, k_width=2),
    ttgl.DotOperandLayout(parent=ttgl.NVMMADistributedLayout(version=[2, 0], warps_per_cta=[2, 2], instr_shape=[16, 8]),
                          operand_index=0, k_width=2),
])


def _histogram_cases():
    if THREADS_PER_WARP not in (32, 64):
        raise RuntimeError(f"Unsupported THREADS_PER_WARP: {THREADS_PER_WARP}")

    m_bins = [(2048, 2), (8, 512), (32, 32)]
    layouts = [(ttgl.BlockedLayout([1], [THREADS_PER_WARP], [4],
                                   [0]), ttgl.BlockedLayout([1], [THREADS_PER_WARP], [4], [0]))]
    for m, bins in m_bins:
        for src_layout, dst_layout in layouts:
            yield (m, bins, src_layout, dst_layout)
    import math

    linear_layouts = [(
        ttgl.DistributedLinearLayout(
            reg_bases=[[1 << (5 + i)] for i in range(int(math.log2(m)) - 5)],
            lane_bases=[[0], [16], [4], [2], [1]] + ([[0]] if THREADS_PER_WARP == 64 else []),
            warp_bases=[[0], [8]],
            block_bases=[],
            shape=(m, ),
        ),
        bins,
    ) for (m, bins) in m_bins if m >= 32]
    for linear_layout, bins in linear_layouts:
        yield (linear_layout.shape[0], bins, linear_layout, ttgl.BlockedLayout([1], [THREADS_PER_WARP], [4], [0]))


@pytest.mark.parametrize("M, bins, src_layout, dst_layout", _histogram_cases())
def test_histogram(M, bins, src_layout, dst_layout, device):

    @gluon.jit
    def kernel(x_ptr, z_ptr, M: ttgl.constexpr, B: ttgl.constexpr, src_layout: ttgl.constexpr,
               dst_layout: ttgl.constexpr):
        offs = ttgl.arange(0, M, layout=src_layout)
        x = ttgl.load(x_ptr + offs)
        h = ttgl.histogram(x, B, layout=dst_layout)
        z_offs = ttgl.arange(0, B, layout=dst_layout)
        ttgl.store(z_ptr + z_offs, h)

    torch.manual_seed(0)
    x = torch.randint(0, bins, (M, ), dtype=torch.int32, device=device)
    z = torch.zeros((bins, ), dtype=torch.int32, device=device)
    z_torch = torch.histc(x.float(), bins=bins, min=0, max=bins - 1).to(torch.int32)
    kernel[(1, )](x, z, M, bins, src_layout, dst_layout, num_warps=4)
    torch.testing.assert_close(z, z_torch, atol=0, rtol=0)


@pytest.mark.parametrize("M", [64, 128, 256])
@pytest.mark.parametrize("src_layout", _1d_layouts)
@pytest.mark.parametrize("dst_layout", _1d_layouts)
@pytest.mark.parametrize("src_dim", [0, 1])
@pytest.mark.parametrize("dst_dim", [0, 1])
@pytest.mark.parametrize("is_bool", [True, False])
def test_convert1d_layouts(M, src_layout, dst_layout, src_dim, dst_dim, is_bool, device):

    @gluon.jit
    def kernel(x_ptr, y_ptr, M: ttgl.constexpr, src_layout: ttgl.constexpr, dst_layout: ttgl.constexpr,
               src_dim: ttgl.constexpr, dst_dim: ttgl.constexpr):
        offs_src = ttgl.arange(0, M, layout=ttgl.SliceLayout(src_dim, src_layout))
        x = ttgl.load(x_ptr + offs_src)
        y = ttgl.convert_layout(x, layout=ttgl.SliceLayout(dst_dim, dst_layout))
        offs_dst = ttgl.arange(0, M, layout=ttgl.SliceLayout(dst_dim, dst_layout))
        ttgl.store(y_ptr + offs_dst, y)

    torch.manual_seed(17)
    x = torch.randint(0, 4, (M, ), dtype=torch.int32, device=device)
    x = x.to(torch.bool) if is_bool else x
    y = torch.zeros((M, ), dtype=torch.int32, device=device)
    kernel[(1, )](x, y, M, src_layout, dst_layout, src_dim, dst_dim, num_warps=4)
    torch.testing.assert_close(y, x.to(torch.int32))


_2d_layouts = _filter_layouts([
    ttgl.BlockedLayout([1, 1], [THREADS_PER_WARP, 1], [2, 2], [0, 1]),
    ttgl.BlockedLayout([1, 16], [8, THREADS_PER_WARP // 8], [4, 1], [1, 0]),
    ttgl.NVMMADistributedLayout(version=[3, 0], warps_per_cta=[4, 1], instr_shape=[16, 32, 16]),
    ttgl.DotOperandLayout(
        parent=ttgl.NVMMADistributedLayout(version=[3, 0], warps_per_cta=[4, 1], instr_shape=[16, 32, 16]),
        operand_index=0, k_width=2),
    ttgl.DotOperandLayout(
        parent=ttgl.NVMMADistributedLayout(version=[3, 0], warps_per_cta=[4, 1], instr_shape=[16, 32, 16]),
        operand_index=0, k_width=1),
    ttgl.NVMMADistributedLayout(version=[2, 0], warps_per_cta=[4, 1], instr_shape=[16, 8]),
    ttgl.DotOperandLayout(parent=ttgl.NVMMADistributedLayout(version=[2, 0], warps_per_cta=[4, 1], instr_shape=[16, 8]),
                          operand_index=1, k_width=2),
    ttgl.DotOperandLayout(parent=ttgl.NVMMADistributedLayout(version=[2, 0], warps_per_cta=[2, 2], instr_shape=[16, 8]),
                          operand_index=0, k_width=2),
    ttgl.DotOperandLayout(parent=ttgl.NVMMADistributedLayout(version=[2, 0], warps_per_cta=[4, 1], instr_shape=[16, 8]),
                          operand_index=0, k_width=8),
    ttgl.SliceLayout(
        dim=1, parent=ttgl.DotOperandLayout(
            parent=ttgl.NVMMADistributedLayout(version=[3, 0], warps_per_cta=[4, 1, 1], instr_shape=[16, 32, 16]),
            operand_index=0, k_width=2)),
    ttgl.SliceLayout(
        dim=1, parent=ttgl.DotOperandLayout(
            parent=ttgl.NVMMADistributedLayout(version=[2, 0], warps_per_cta=[4, 1, 1], instr_shape=[1, 16, 8]),
            operand_index=1, k_width=2)),
])

_intermediate_layouts = _filter_layouts([
    None,
    ttgl.SwizzledSharedLayout(vec=1, per_phase=1, max_phase=1, order=[0, 1]),
    ttgl.SwizzledSharedLayout(vec=1, per_phase=1, max_phase=1, order=[1, 0]),
    ttgl.SwizzledSharedLayout(vec=4, per_phase=2, max_phase=4, order=[1, 0]),
    ttgl.SwizzledSharedLayout(vec=2, per_phase=2, max_phase=4, order=[1, 0]),
    "padded_shared_layout_single_interval",
    "padded_shared_layout_multi_interval",
])


def _with_cga_layout(layout, cga_layout):
    if isinstance(layout, ttgl.BlockedLayout):
        return ttgl.BlockedLayout(layout.size_per_thread, layout.threads_per_warp, layout.warps_per_cta, layout.order,
                                  cga_layout=cga_layout)
    if isinstance(layout, ttgl.NVMMADistributedLayout):
        return ttgl.NVMMADistributedLayout(layout.version, layout.warps_per_cta, layout.instr_shape,
                                           cga_layout=cga_layout)
    if isinstance(layout, ttgl.DotOperandLayout):
        return ttgl.DotOperandLayout(parent=_with_cga_layout(layout.parent, cga_layout),
                                     operand_index=layout.operand_index, k_width=layout.k_width)
    if isinstance(layout, ttgl.SliceLayout):
        parent_cga_layout = [basis[:layout.dim] + [0] + basis[layout.dim:] for basis in cga_layout]
        return ttgl.SliceLayout(dim=layout.dim, parent=_with_cga_layout(layout.parent, parent_cga_layout))
    if isinstance(layout, ttgl.SwizzledSharedLayout):
        return ttgl.SwizzledSharedLayout(layout.vec, layout.per_phase, layout.max_phase, layout.order,
                                         cga_layout=cga_layout)
    raise AssertionError(f"Unsupported multi-CTA layout {type(layout)}")


_single_cta_convert2d_layout_cases = [(None, None, interm_layout, src_layout, dst_layout)
                                      for interm_layout in _intermediate_layouts
                                      for src_layout in _2d_layouts
                                      for dst_layout in _2d_layouts]
# Pair each layout with the next one so multi-CTA coverage stays small while every layout appears as source and dest.
_multi_cta_2d_layout_pairs = list(zip(_2d_layouts, _2d_layouts[1:] + _2d_layouts[:1]))
# Use different source/destination CGA shapes to cover CTA repartitioning during convert_layout.
_multi_cta_cga_layout_pairs = [([1, 4], [2, 2]), ([4, 1], [2, 2])]
_multi_cta_convert2d_layout_cases = [(src_ctas_per_cga, dst_ctas_per_cga, None, src_layout, dst_layout)
                                     for src_ctas_per_cga, dst_ctas_per_cga in _multi_cta_cga_layout_pairs
                                     for src_layout, dst_layout in _multi_cta_2d_layout_pairs]
_convert2d_layout_cases = _single_cta_convert2d_layout_cases + _multi_cta_convert2d_layout_cases


@pytest.mark.parametrize("M, N", [[64, 1], [64, 64], [64, 128], [1, 64]])
@pytest.mark.parametrize("dtype", ["float16"])
@pytest.mark.parametrize("src_ctas_per_cga, dst_ctas_per_cga, interm_layout, src_layout, dst_layout",
                         _convert2d_layout_cases)
def test_convert2d_layouts(M, N, src_ctas_per_cga, dst_ctas_per_cga, interm_layout, src_layout, dst_layout, dtype,
                           device):
    num_ctas = 1
    if src_ctas_per_cga is not None:
        if not is_cuda() or not is_hopper_or_newer():
            pytest.skip("num_ctas > 1 requires NVIDIA Hopper or newer")
        if M % src_ctas_per_cga[0] != 0 or N % src_ctas_per_cga[1] != 0:
            pytest.skip("Shape must be divisible by the source CGA shape")
        if M % dst_ctas_per_cga[0] != 0 or N % dst_ctas_per_cga[1] != 0:
            pytest.skip("Shape must be divisible by the destination CGA shape")
        if src_ctas_per_cga[0] * src_ctas_per_cga[1] != dst_ctas_per_cga[0] * dst_ctas_per_cga[1]:
            pytest.skip("Source and destination CGA shapes must have the same number of CTAs")
        src_cga_layout = make_cga_layout(src_ctas_per_cga, src_ctas_per_cga, [1, 0])
        dst_cga_layout = make_cga_layout(dst_ctas_per_cga, dst_ctas_per_cga, [1, 0])
        num_ctas = src_ctas_per_cga[0] * src_ctas_per_cga[1]
        src_layout = _with_cga_layout(src_layout, src_cga_layout)
        dst_layout = _with_cga_layout(dst_layout, dst_cga_layout)
    else:
        if dst_ctas_per_cga is not None:
            pytest.skip("Destination CGA shape requires a source CGA shape")

    if str(src_layout) == str(dst_layout):
        pytest.skip("Source and destination layouts are the same")

    if interm_layout in ["padded_shared_layout_single_interval", "padded_shared_layout_multi_interval"]:
        int_pad_pairs = [[32, 8]] if "single" in interm_layout else [[64, 4], [128, 8]]
        interm_layout = ttgl.PaddedSharedLayout.with_identity_for(int_pad_pairs, [M, N], [1, 0])

    def compute_scratch_buffer_shape(src_layout, dst_layout, shape):

        def compute_rep_shape(layout):
            if type(layout) is ttgl.BlockedLayout:
                warp_shape = torch.tensor(layout.size_per_thread) * torch.tensor(layout.threads_per_warp)
                rep_shape = warp_shape * torch.tensor(layout.warps_per_cta)
                return rep_shape
            else:
                assert False, "TODO: support compute_rep_shape for layout " + str(type(layout))

        src_rep_shape = compute_rep_shape(src_layout)
        dst_rep_shape = compute_rep_shape(dst_layout)
        full_scratch_shape = torch.maximum(src_rep_shape, dst_rep_shape)
        return torch.minimum(full_scratch_shape, torch.tensor(shape))

    if is_hip():
        try:
            scratch_shape = compute_scratch_buffer_shape(src_layout, dst_layout, (M, N))
        except AssertionError:
            pytest.skip("Can't compute scratch buffer size")
        lds_size = get_hip_lds_size()
        # consider int32 dtype in scratch buffer size,
        # because it is the largest dtype used in convert_layout in this test
        int32_size = 4
        # skip even if scratch buffer equal to lds_size, because real scratch buffer is typically larger due to padding
        if scratch_shape[0] * scratch_shape[1] * int32_size >= lds_size:
            pytest.skip("Scratch buffer is too large")

    @gluon.jit
    def kernel(x_ptr, y_ptr, M: ttgl.constexpr, N: ttgl.constexpr, src_layout: ttgl.constexpr,
               dst_layout: ttgl.constexpr, interm_layout: ttgl.constexpr):
        # Create offsets for src layout
        offs_m_src = ttgl.arange(0, M, layout=ttgl.SliceLayout(1, src_layout))[:, None]
        offs_n_src = ttgl.arange(0, N, layout=ttgl.SliceLayout(0, src_layout))[None, :]

        # Load data
        x = ttgl.load(x_ptr + offs_m_src * N + offs_n_src)

        # Convert layout (with or without intermediate shared memory)
        if interm_layout is None:
            y = ttgl.convert_layout(x, layout=dst_layout)
        else:
            # Store to shared memory and load back before converting
            shared_desc = ttgl.allocate_shared_memory(x.dtype, (M, N), interm_layout, value=x)
            x_shared = shared_desc.load(src_layout)
            y = ttgl.convert_layout(x_shared, layout=dst_layout)

        # Create offsets for dst layout and store
        offs_m_dst = ttgl.arange(0, M, layout=ttgl.SliceLayout(1, dst_layout))[:, None]
        offs_n_dst = ttgl.arange(0, N, layout=ttgl.SliceLayout(0, dst_layout))[None, :]
        ttgl.store(y_ptr + offs_m_dst * N + offs_n_dst, y)

    torch.manual_seed(0)
    torch_dtype = getattr(torch, dtype)
    x = torch.randn((M, N), dtype=torch_dtype, device=device)
    y = torch.zeros_like(x)
    compiled = kernel[(1, )](x, y, M, N, src_layout, dst_layout, interm_layout, num_ctas=num_ctas)

    torch.testing.assert_close(y, x, rtol=0, atol=0)
    if src_ctas_per_cga != dst_ctas_per_cga:
        # Replicated values may be loaded from the local CTA.
        assert "st.shared::cluster" not in compiled.asm["ptx"]


# MMA layout pairs for MMA-to-MMA conversion tests
_mma_pairs = [
    # MMA v2.0 layouts
    [
        ttgl.NVMMADistributedLayout(version=[2, 0], warps_per_cta=[1, 4], instr_shape=[16, 8]),
        ttgl.NVMMADistributedLayout(version=[2, 0], warps_per_cta=[4, 1], instr_shape=[16, 8]),
    ],
    [
        ttgl.NVMMADistributedLayout(version=[2, 0], warps_per_cta=[2, 8], instr_shape=[16, 8]),
        ttgl.NVMMADistributedLayout(version=[2, 0], warps_per_cta=[8, 2], instr_shape=[16, 8]),
    ],
    # MMA v2.1 layouts
    [
        ttgl.NVMMADistributedLayout(version=[2, 1], warps_per_cta=[1, 4], instr_shape=[16, 8]),
        ttgl.NVMMADistributedLayout(version=[2, 1], warps_per_cta=[4, 1], instr_shape=[16, 8]),
    ],
    [
        ttgl.NVMMADistributedLayout(version=[2, 1], warps_per_cta=[2, 8], instr_shape=[16, 8]),
        ttgl.NVMMADistributedLayout(version=[2, 1], warps_per_cta=[8, 2], instr_shape=[16, 8]),
    ],
    # MMA v3.0 layouts
    [
        ttgl.NVMMADistributedLayout(version=[3, 0], warps_per_cta=[4, 1], instr_shape=[16, 32, 32]),
        ttgl.NVMMADistributedLayout(version=[3, 0], warps_per_cta=[4, 1], instr_shape=[16, 64, 32]),
    ],
    [
        ttgl.NVMMADistributedLayout(version=[3, 0], warps_per_cta=[1, 4], instr_shape=[16, 32, 32]),
        ttgl.NVMMADistributedLayout(version=[3, 0], warps_per_cta=[4, 1], instr_shape=[16, 64, 32]),
    ],
    [
        ttgl.NVMMADistributedLayout(version=[3, 0], warps_per_cta=[2, 8], instr_shape=[16, 64, 32]),
        ttgl.NVMMADistributedLayout(version=[3, 0], warps_per_cta=[8, 2], instr_shape=[16, 32, 32]),
    ],
    [
        ttgl.NVMMADistributedLayout(version=[3, 0], warps_per_cta=[4, 1], instr_shape=[16, 128, 16]),
        ttgl.NVMMADistributedLayout(version=[3, 0], warps_per_cta=[4, 1], instr_shape=[16, 64, 16]),
    ],
    # AMD MFMA v1 layouts
    [
        ttgl.amd.AMDMFMALayout(version=1, instr_shape=[32, 32, 8], transposed=True, warps_per_cta=[2, 2]),
        ttgl.amd.AMDMFMALayout(version=1, instr_shape=[32, 32, 8], transposed=True, warps_per_cta=[4, 1]),
    ],
    [
        ttgl.amd.AMDMFMALayout(version=1, instr_shape=[16, 16, 8], transposed=True, warps_per_cta=[4, 4]),
        ttgl.amd.AMDMFMALayout(version=1, instr_shape=[16, 16, 8], transposed=True, warps_per_cta=[16, 1]),
    ],
    # AMD MFMA v2 layouts
    [
        ttgl.amd.AMDMFMALayout(version=2, instr_shape=[32, 32, 8], transposed=True, warps_per_cta=[2, 2]),
        ttgl.amd.AMDMFMALayout(version=2, instr_shape=[32, 32, 8], transposed=True, warps_per_cta=[4, 1]),
    ],
    [
        ttgl.amd.AMDMFMALayout(version=2, instr_shape=[16, 16, 16], transposed=True, warps_per_cta=[4, 4]),
        ttgl.amd.AMDMFMALayout(version=2, instr_shape=[16, 16, 16], transposed=True, warps_per_cta=[16, 1]),
    ],
    # AMD MFMA v3 layouts
    [
        ttgl.amd.AMDMFMALayout(version=3, instr_shape=[32, 32, 8], transposed=True, warps_per_cta=[2, 2]),
        ttgl.amd.AMDMFMALayout(version=3, instr_shape=[32, 32, 8], transposed=True, warps_per_cta=[4, 1]),
    ],
    [
        ttgl.amd.AMDMFMALayout(version=3, instr_shape=[16, 16, 16], transposed=True, warps_per_cta=[4, 4]),
        ttgl.amd.AMDMFMALayout(version=3, instr_shape=[16, 16, 16], transposed=True, warps_per_cta=[16, 1]),
    ],
    # AMD MFMA v4 layouts
    [
        ttgl.amd.AMDMFMALayout(version=4, instr_shape=[32, 32, 16], transposed=True, warps_per_cta=[2, 2]),
        ttgl.amd.AMDMFMALayout(version=4, instr_shape=[32, 32, 16], transposed=True, warps_per_cta=[4, 1]),
    ],
    [
        ttgl.amd.AMDMFMALayout(version=4, instr_shape=[16, 16, 32], transposed=True, warps_per_cta=[4, 4]),
        ttgl.amd.AMDMFMALayout(version=4, instr_shape=[16, 16, 32], transposed=True, warps_per_cta=[16, 1]),
    ],
    # AMD WMMA v1 layouts
    [
        ttgl.amd.AMDWMMALayout(version=1, transposed=True, warp_bases=[[0, 1], [0, 2], [1, 0], [2, 0]]),
        ttgl.amd.AMDWMMALayout(version=1, transposed=True, warp_bases=[[1, 0], [2, 0], [4, 0], [8, 0]]),
    ],
    # AMD WMMA v2 layouts
    [
        ttgl.amd.AMDWMMALayout(version=2, transposed=True, warp_bases=[[0, 1], [0, 2], [1, 0], [2, 0]]),
        ttgl.amd.AMDWMMALayout(version=2, transposed=True, warp_bases=[[1, 0], [2, 0], [4, 0], [8, 0]]),
    ],
]


@pytest.mark.parametrize("M, N", [[16, 16], [64, 1], [1, 64], [64, 64], [128, 128], [256, 256]])
@pytest.mark.parametrize("dtype", ["float16"])
@pytest.mark.parametrize("mma_pair",
                         [pair for pair in _mma_pairs if all(_is_layout_applicable(layout) for layout in pair)])
def test_convert_mma2mma_layouts(M, N, mma_pair, dtype, device):
    src_layout, dst_layout = mma_pair

    @gluon.jit
    def kernel(x_ptr, y_ptr, M: ttgl.constexpr, N: ttgl.constexpr, src_layout: ttgl.constexpr,
               dst_layout: ttgl.constexpr):
        # Create offsets for src layout
        offs_m_src = ttgl.arange(0, M, layout=ttgl.SliceLayout(1, src_layout))[:, None]
        offs_n_src = ttgl.arange(0, N, layout=ttgl.SliceLayout(0, src_layout))[None, :]

        # Load data and convert layout
        x = ttgl.load(x_ptr + offs_m_src * N + offs_n_src)
        y = ttgl.convert_layout(x, layout=dst_layout)

        # Create offsets for dst layout and store
        offs_m_dst = ttgl.arange(0, M, layout=ttgl.SliceLayout(1, dst_layout))[:, None]
        offs_n_dst = ttgl.arange(0, N, layout=ttgl.SliceLayout(0, dst_layout))[None, :]
        ttgl.store(y_ptr + offs_m_dst * N + offs_n_dst, y)

    torch.manual_seed(0)
    torch_dtype = getattr(torch, dtype)
    x = torch.randn((M, N), dtype=torch_dtype, device=device)

    # Calculate num_warps based on layout
    num_warps = int(torch.prod(torch.tensor(ttgl._layouts.warps_per_cta(src_layout, (M, N)))))
    y = torch.zeros_like(x)
    kernel[(1, )](x, y, M, N, src_layout, dst_layout, num_warps=num_warps)
    torch.testing.assert_close(y, x, rtol=0, atol=0)

    y = torch.zeros_like(x)
    kernel[(1, )](x, y, M, N, dst_layout, src_layout, num_warps=num_warps)
    torch.testing.assert_close(y, x, rtol=0, atol=0)


_warp_local_layouts = _filter_layouts([
    ttgl.BlockedLayout([1, 1], [THREADS_PER_WARP, 1], [1, 1], [1, 0]),
    ttgl.BlockedLayout([1, 1], [THREADS_PER_WARP // 2, 2], [1, 1], [1, 0]),
    ttgl.BlockedLayout([1, 1], [THREADS_PER_WARP // 4, 4], [1, 1], [1, 0]),
    ttgl.BlockedLayout([1, 1], [THREADS_PER_WARP // 8, 8], [1, 1], [1, 0]),
    ttgl.BlockedLayout([1, 1], [THREADS_PER_WARP // 16, 16], [1, 1], [1, 0]),
    ttgl.BlockedLayout([1, 1], [THREADS_PER_WARP // 32, 32], [1, 1], [1, 0]),
    ttgl.BlockedLayout([32, 1], [1, THREADS_PER_WARP], [1, 1], [1, 0]),
    ttgl.BlockedLayout([16, 1], [2, THREADS_PER_WARP // 2], [1, 1], [1, 0]),
    ttgl.BlockedLayout([1, 4], [THREADS_PER_WARP, 1], [1, 1], [1, 0]),
    ttgl.BlockedLayout([1, 4], [THREADS_PER_WARP // 2, 2], [1, 1], [1, 0]),
    ttgl.BlockedLayout([1, 4], [THREADS_PER_WARP // 4, 4], [1, 1], [1, 0]),
    ttgl.BlockedLayout([1, 4], [THREADS_PER_WARP // 8, 8], [1, 1], [1, 0]),
    ttgl.BlockedLayout([1, 4], [THREADS_PER_WARP // 16, 16], [1, 1], [1, 0]),
    ttgl.BlockedLayout([1, 4], [THREADS_PER_WARP // 32, 32], [1, 1], [1, 0]),
])


@pytest.mark.parametrize("M, N", [[32, 32], [64, 64]])
@pytest.mark.parametrize("dtype", ["float16"])
@pytest.mark.parametrize("src_layout", _warp_local_layouts)
@pytest.mark.parametrize("dst_layout", _warp_local_layouts)
def test_convert_warp_local_layouts(M, N, src_layout, dst_layout, dtype, device):
    if str(src_layout) == str(dst_layout):
        pytest.skip("Source and destination layouts are the same")

    # Test layout pairs that are likely to codegen warp shuffles.
    a, b = list(torch.tensor(src_layout.threads_per_warp) // torch.tensor(dst_layout.threads_per_warp))
    c = a if a != 0 else b
    if c > 2:
        pytest.skip("Layout pair too complex for warp-local conversion")

    @gluon.jit
    def kernel(x_ptr, y_ptr, M: ttgl.constexpr, N: ttgl.constexpr, src_layout: ttgl.constexpr,
               dst_layout: ttgl.constexpr):
        # Create offsets for src layout
        offs_m_src = ttgl.arange(0, M, layout=ttgl.SliceLayout(1, src_layout))[:, None]
        offs_n_src = ttgl.arange(0, N, layout=ttgl.SliceLayout(0, src_layout))[None, :]

        # Load data and convert layout
        x = ttgl.load(x_ptr + offs_m_src * N + offs_n_src)
        y = ttgl.convert_layout(x, layout=dst_layout)

        # Create offsets for dst layout and store
        offs_m_dst = ttgl.arange(0, M, layout=ttgl.SliceLayout(1, dst_layout))[:, None]
        offs_n_dst = ttgl.arange(0, N, layout=ttgl.SliceLayout(0, dst_layout))[None, :]
        ttgl.store(y_ptr + offs_m_dst * N + offs_n_dst, y)

    torch.manual_seed(0)
    torch_dtype = getattr(torch, dtype)
    x = torch.randn((M, N), dtype=torch_dtype, device=device)
    y = torch.zeros_like(x)

    num_warps = int(torch.prod(torch.tensor(ttgl._layouts.warps_per_cta(src_layout, (M, N)))))
    kernel[(1, )](x, y, M, N, src_layout, dst_layout, num_warps=num_warps)

    torch.testing.assert_close(y, x, rtol=0, atol=0)


@pytest.mark.skipif(is_hip(), reason="Assumes 32 threads per warp")
def test_regress_warp_shuffle_convert_layout(tmp_path):
    rows = 2
    cols = 8
    # We have previously incorrectly lowered a layout conversion between these
    # two layouts when that conversion was forced to use warp shuffles. Test
    # that it works.
    src_layout = ttgl.DistributedLinearLayout(
        reg_bases=[[0, 1], [0, 2], [0, 4]],
        lane_bases=[[1, 0], [0, 0], [0, 0], [0, 0], [0, 0]],
        warp_bases=[],
        block_bases=[],
        shape=(rows, cols),
    )
    dst_layout = ttgl.DistributedLinearLayout(
        reg_bases=[[1, 0], [0, 4]],
        lane_bases=[[0, 0], [0, 0], [0, 1], [0, 2], [0, 0]],
        warp_bases=[],
        block_bases=[],
        shape=(rows, cols),
    )
    axis0_layout = ttgl.SliceLayout(dim=1, parent=src_layout)
    axis1_layout = ttgl.SliceLayout(dim=0, parent=src_layout)
    out_axis0_layout = ttgl.SliceLayout(dim=1, parent=dst_layout)
    out_axis1_layout = ttgl.SliceLayout(dim=0, parent=dst_layout)

    @gluon.jit
    def load_cvt_store(out_ptr, in_ptr):
        offs0 = ttgl.arange(0, 2, layout=axis0_layout)[:, None]
        offs1 = ttgl.arange(0, 8, layout=axis1_layout)[None, :]
        offsets = offs0 * 8 + offs1
        x = ttgl.load(in_ptr + offsets)
        y = ttgl.convert_layout(x, dst_layout)

        out_offs0 = ttgl.arange(0, 2, layout=out_axis0_layout)[:, None]
        out_offs1 = ttgl.arange(0, 8, layout=out_axis1_layout)[None, :]
        out_offsets = out_offs0 * 8 + out_offs1
        ttgl.store(out_ptr + out_offsets, y)

    torch.manual_seed(0)
    x = torch.randint(-128, 128, (rows, cols), dtype=torch.int16, device="cuda")
    ref = torch.zeros_like(x)
    out = torch.zeros_like(x)

    # Extract the TTGIR and force using warp shuffles for lowering the
    # convert_layout.
    compiled_load_cvt_store = load_cvt_store.warmup(ref, x, grid=(1, 1, 1), num_warps=1)
    ttgir = compiled_load_cvt_store.asm["ttgir"]
    ttgir = ttgir.replace(
        "attributes {noinline = false}",
        "attributes {always_use_warp_shuffle, noinline = false}",
        1,
    )

    temp_file = tmp_path / "test_override_ttgir_always_use_warp_shuffle.ttgir"
    temp_file.write_text(ttgir)

    load_cvt_store_warp_shuffle = triton.compile(str(temp_file))

    load_cvt_store[(1, 1, 1)](ref, x, num_warps=1)
    load_cvt_store_warp_shuffle[(1, 1, 1)](out, x)

    assert torch.equal(ref, x)
    assert torch.equal(out, x)


_ld_st_dot_layouts = _filter_layouts([
    ttgl.DotOperandLayout(parent=ttgl.NVMMADistributedLayout(version=[2, 0], warps_per_cta=[4, 1], instr_shape=[16, 8]),
                          operand_index=0, k_width=4),
    ttgl.DotOperandLayout(parent=ttgl.NVMMADistributedLayout(version=[2, 0], warps_per_cta=[4, 1], instr_shape=[16, 8]),
                          operand_index=1, k_width=4),
    ttgl.DotOperandLayout(parent=ttgl.NVMMADistributedLayout(version=[2, 0], warps_per_cta=[4, 1], instr_shape=[16, 8]),
                          operand_index=0, k_width=2),
    ttgl.DotOperandLayout(parent=ttgl.NVMMADistributedLayout(version=[2, 0], warps_per_cta=[4, 1], instr_shape=[16, 8]),
                          operand_index=1, k_width=2),
])

_ld_st_mma_layouts = _filter_layouts([
    ttgl.NVMMADistributedLayout(version=[2, 0], warps_per_cta=[1, 4], instr_shape=[16, 8]),
    ttgl.NVMMADistributedLayout(version=[3, 0], warps_per_cta=[4, 1], instr_shape=[16, 128, 16]),
    ttgl.NVMMADistributedLayout(version=[3, 0], warps_per_cta=[4, 2], instr_shape=[16, 128, 16]),
    ttgl.NVMMADistributedLayout(version=[3, 0], warps_per_cta=[4, 2], instr_shape=[16, 64, 16]),
    ttgl.NVMMADistributedLayout(version=[3, 0], warps_per_cta=[8, 1], instr_shape=[16, 128, 16]),
    ttgl.NVMMADistributedLayout(version=[3, 0], warps_per_cta=[8, 4], instr_shape=[16, 64, 16]),
])

_ld_st_shared_layouts = _filter_layouts([
    ttgl.NVMMASharedLayout(swizzle_byte_width=0, transposed=False, element_bitwidth=16, rank=2),
    ttgl.NVMMASharedLayout(swizzle_byte_width=64, transposed=False, element_bitwidth=16, rank=2),
    ttgl.NVMMASharedLayout(swizzle_byte_width=64, transposed=True, element_bitwidth=16, rank=2),
    ttgl.NVMMASharedLayout(swizzle_byte_width=128, transposed=False, element_bitwidth=16, rank=2),
    ttgl.NVMMASharedLayout(swizzle_byte_width=32, transposed=False, element_bitwidth=8, rank=2),
    ttgl.SwizzledSharedLayout(vec=8, per_phase=1, max_phase=1, order=[1, 0]),
    ttgl.SwizzledSharedLayout(vec=4, per_phase=2, max_phase=4, order=[0, 1]),
    ttgl.SwizzledSharedLayout(vec=8, per_phase=1, max_phase=8, order=[1, 0]),
    ttgl.SwizzledSharedLayout(vec=16, per_phase=1, max_phase=16, order=[1, 0]),
    "shared_linear_layout",
])


@pytest.mark.skipif(not is_cuda(), reason="Requires CUDA")
def test_local_load_transposed_nvmma_zero_swizzle(device):
    rows = 128
    cols = 16
    src_layout = ttgl.BlockedLayout([1, 1], [32, 1], [4, 1], [1, 0])
    dst_layout = ttgl.BlockedLayout([1, 16], [1, 32], [1, 4], [1, 0])
    shared_layout = ttgl.NVMMASharedLayout(swizzle_byte_width=0, transposed=False, element_bitwidth=8, rank=2)

    @gluon.jit
    def kernel(x_ptr, y_ptr, rows: ttgl.constexpr, cols: ttgl.constexpr, src_layout: ttgl.constexpr,
               dst_layout: ttgl.constexpr, shared_layout: ttgl.constexpr):
        offs_row = ttgl.arange(0, rows, layout=ttgl.SliceLayout(1, src_layout))[:, None]
        offs_col = ttgl.arange(0, cols, layout=ttgl.SliceLayout(0, src_layout))[None, :]
        x = ttgl.load(x_ptr + offs_row * cols + offs_col)
        smem = ttgl.allocate_shared_memory(ttgl.uint8, [rows, cols], shared_layout, x)

        y = smem.permute((1, 0)).load(dst_layout)
        offs_col = ttgl.arange(0, cols, layout=ttgl.SliceLayout(1, dst_layout))[:, None]
        offs_row = ttgl.arange(0, rows, layout=ttgl.SliceLayout(0, dst_layout))[None, :]
        ttgl.store(y_ptr + offs_col * rows + offs_row, y)

    x = torch.arange(rows * cols, device=device, dtype=torch.int64).to(torch.uint8).reshape(rows, cols)
    y = torch.empty((cols, rows), device=device, dtype=torch.uint8)
    kernel[(1, )](x, y, rows, cols, src_layout, dst_layout, shared_layout, num_warps=4)
    torch.testing.assert_close(y, x.T)


@pytest.mark.parametrize("shape, dtype", [
    ((16, 32), "float8_e5m2"),
    ((16, 32), "float16"),
    ((16, 32), "float32"),
    ((128, 128), "float16"),
])
@pytest.mark.parametrize("dist_layout", _ld_st_dot_layouts + _ld_st_mma_layouts)
@pytest.mark.parametrize("shared_layout", _ld_st_shared_layouts)
def test_local_load_store_2d_layouts(shape, dtype, dist_layout, shared_layout, device):
    if shared_layout == "shared_linear_layout":
        rank = len(shape)
        assert rank == 2
        offset_bases = []
        for dim, size in enumerate(shape):
            assert size > 0 and (size & (size - 1)) == 0
            stride = 1
            while stride < size:
                basis = [0] * rank
                basis[dim] = stride
                offset_bases.append(basis)
                stride <<= 1
        shared_layout = ttgl.SharedLinearLayout(offset_bases=offset_bases)

    if isinstance(shared_layout, ttgl.NVMMASharedLayout):
        contig_dim = 0 if shared_layout.transposed else 1
        if shape[contig_dim] < (8 * shared_layout.swizzle_byte_width) / shared_layout.element_bitwidth:
            pytest.skip("contig_dim too small for swizzle_byte_width in NVMMASharedLayout")

    # A simple blocked layout
    num_warps = int(torch.prod(torch.tensor(ttgl._layouts.warps_per_cta(dist_layout, shape))))
    blocked_layout = ttgl.BlockedLayout(size_per_thread=[1, 1], threads_per_warp=[4, THREADS_PER_WARP // 4],
                                        warps_per_cta=[1, num_warps], order=[0, 1])

    @gluon.jit
    def kernel(x_ptr, y_ptr, shape_tuple: ttgl.constexpr, src_layout: ttgl.constexpr, dst_layout: ttgl.constexpr,
               shared_layout: ttgl.constexpr):
        M: ttgl.constexpr = shape_tuple[0]
        N: ttgl.constexpr = shape_tuple[1]
        offs_m_src = ttgl.arange(0, M, layout=ttgl.SliceLayout(1, src_layout))[:, None]
        offs_n_src = ttgl.arange(0, N, layout=ttgl.SliceLayout(0, src_layout))[None, :]

        x = ttgl.load(x_ptr + offs_m_src * N + offs_n_src)

        shared_desc = ttgl.allocate_shared_memory(x.dtype, shape_tuple, shared_layout, value=x)
        y = shared_desc.load(dst_layout)

        offs_m_dst = ttgl.arange(0, M, layout=ttgl.SliceLayout(1, dst_layout))[:, None]
        offs_n_dst = ttgl.arange(0, N, layout=ttgl.SliceLayout(0, dst_layout))[None, :]
        ttgl.store(y_ptr + offs_m_dst * N + offs_n_dst, y)

    torch.manual_seed(0)
    torch_dtype = getattr(torch, dtype)

    if "float8" in dtype:
        x = torch.randn(shape, device=device, dtype=torch.float16).to(torch_dtype)
    else:
        x = torch.randn(shape, device=device, dtype=torch_dtype)

    float8_dtypes = {torch.float8_e5m2}
    if hasattr(torch, "float8_e4m3fn"):
        float8_dtypes.add(torch.float8_e4m3fn)

    def _assert_close(actual, expected):
        if actual.dtype in float8_dtypes:
            torch.testing.assert_close(actual.to(torch.float16), expected.to(torch.float16), rtol=0, atol=0)
        else:
            torch.testing.assert_close(actual, expected)

    y = torch.zeros_like(x)
    kernel[(1, )](x, y, shape, blocked_layout, dist_layout, shared_layout, num_warps=num_warps)
    _assert_close(y, x)

    y = torch.zeros_like(x)
    obj = kernel[(1, )](x, y, shape, dist_layout, blocked_layout, shared_layout, num_warps=num_warps)
    _assert_close(y, x)
    if (isinstance(shared_layout, ttgl.NVMMASharedLayout) and dist_layout in _ld_st_mma_layouts
            and dist_layout.version[0] >= 3 and dtype == "float16"):
        assert "stmatrix" in obj.asm["ptx"]


_ld_st_3d_layouts = _filter_layouts([
    ttgl.BlockedLayout([4, 4, 1], [1, 8, THREADS_PER_WARP // 8], [2, 2, 1], [2, 1, 0]),
    ttgl.BlockedLayout([1, 1, 4], [8, THREADS_PER_WARP // 8, 1], [2, 1, 2], [1, 2, 0]),
    ttgl.DotOperandLayout(
        parent=ttgl.NVMMADistributedLayout(version=[2, 0], warps_per_cta=[4, 1, 1], instr_shape=[1, 16, 8]),
        operand_index=0, k_width=1),
])

_ld_st_3d_shared_layouts = _filter_layouts([
    ttgl.SwizzledSharedLayout(vec=1, per_phase=1, max_phase=1, order=[2, 1, 0]),
    ttgl.SwizzledSharedLayout(vec=4, per_phase=2, max_phase=4, order=[1, 2, 0]),
    ttgl.SwizzledSharedLayout(vec=8, per_phase=2, max_phase=4, order=[0, 2, 1]),
    ttgl.SwizzledSharedLayout(vec=4, per_phase=2, max_phase=1, order=[2, 0, 1]),
])


@pytest.mark.parametrize("shape, dtype", [
    ((8, 16, 32), "float32"),
])
@pytest.mark.parametrize("dist_layout", _ld_st_3d_layouts)
@pytest.mark.parametrize("shared_layout", _ld_st_3d_shared_layouts)
def test_local_load_store_3d_layouts(shape, dtype, dist_layout, shared_layout, device):
    # A simple blocked layout
    num_warps = int(torch.prod(torch.tensor(ttgl._layouts.warps_per_cta(dist_layout, shape))))
    blocked_layout = ttgl.BlockedLayout(
        size_per_thread=[1, 1, 1],
        threads_per_warp=[1, 4, THREADS_PER_WARP // 4],
        warps_per_cta=[1, 1, num_warps],
        order=[2, 1, 0],
    )

    @gluon.jit
    def kernel(x_ptr, y_ptr, shape_tuple: ttgl.constexpr, src_layout: ttgl.constexpr, dst_layout: ttgl.constexpr,
               shared_layout: ttgl.constexpr):
        M: ttgl.constexpr = shape_tuple[0]
        N: ttgl.constexpr = shape_tuple[1]
        K: ttgl.constexpr = shape_tuple[2]
        offs_m_src = ttgl.arange(0, M, layout=ttgl.SliceLayout(1, parent=ttgl.SliceLayout(2, src_layout)))[:, None,
                                                                                                           None]
        offs_n_src = ttgl.arange(0, N, layout=ttgl.SliceLayout(0, parent=ttgl.SliceLayout(2, src_layout)))[None, :,
                                                                                                           None]
        offs_k_src = ttgl.arange(0, K, layout=ttgl.SliceLayout(0, parent=ttgl.SliceLayout(1, src_layout)))[None,
                                                                                                           None, :]

        x = ttgl.load(x_ptr + offs_m_src * N * K + offs_n_src * K + offs_k_src)

        shared_desc = ttgl.allocate_shared_memory(x.dtype, shape_tuple, shared_layout, value=x)
        y = shared_desc.load(dst_layout)

        offs_m_dst = ttgl.arange(0, M, layout=ttgl.SliceLayout(1, parent=ttgl.SliceLayout(2, dst_layout)))[:, None,
                                                                                                           None]
        offs_n_dst = ttgl.arange(0, N, layout=ttgl.SliceLayout(0, parent=ttgl.SliceLayout(2, dst_layout)))[None, :,
                                                                                                           None]
        offs_k_dst = ttgl.arange(0, K, layout=ttgl.SliceLayout(0, parent=ttgl.SliceLayout(1, dst_layout)))[None,
                                                                                                           None, :]
        ttgl.store(y_ptr + offs_m_dst * N * K + offs_n_dst * K + offs_k_dst, y)

    torch.manual_seed(0)
    torch_dtype = getattr(torch, dtype)
    x = torch.randn(shape, device=device, dtype=torch_dtype)

    y = torch.zeros_like(x)
    kernel[(1, )](x, y, shape, blocked_layout, dist_layout, shared_layout, num_warps=num_warps)
    torch.testing.assert_close(y, x)

    y = torch.zeros_like(x)
    kernel[(1, )](x, y, shape, dist_layout, blocked_layout, shared_layout, num_warps=num_warps)
    torch.testing.assert_close(y, x)


@gluon.jit
def _gather_kernel_1d(
    src_ptr,
    idx_ptr,
    out_ptr,
    axis: ttgl.constexpr,
    src_dim: ttgl.constexpr,
    idx_dim: ttgl.constexpr,
    src_layout: ttgl.constexpr,
    idx_layout: ttgl.constexpr,
):
    src_offs = ttgl.arange(0, src_dim, layout=src_layout)
    src = ttgl.load(src_ptr + src_offs)

    idx_offs = ttgl.arange(0, idx_dim, layout=idx_layout)
    idx = ttgl.load(idx_ptr + idx_offs)

    out = ttgl.gather(src, idx, axis)

    ttgl.store(out_ptr + idx_offs, out)


@gluon.jit
def _gather_kernel_2d(
    src_ptr,
    idx_ptr,
    out_ptr,
    axis: ttgl.constexpr,
    src_dim0: ttgl.constexpr,
    src_dim1: ttgl.constexpr,
    idx_dim0: ttgl.constexpr,
    idx_dim1: ttgl.constexpr,
    src_layout: ttgl.constexpr,
    idx_layout: ttgl.constexpr,
):
    offs_src_dim0 = ttgl.arange(0, src_dim0, layout=ttgl.SliceLayout(1, src_layout))[:, None]
    offs_src_dim1 = ttgl.arange(0, src_dim1, layout=ttgl.SliceLayout(0, src_layout))[None, :]
    src_offs = offs_src_dim0 * src_dim1 + offs_src_dim1
    src = ttgl.load(src_ptr + src_offs)

    offs_idx_dim0 = ttgl.arange(0, idx_dim0, layout=ttgl.SliceLayout(1, idx_layout))[:, None]
    offs_idx_dim1 = ttgl.arange(0, idx_dim1, layout=ttgl.SliceLayout(0, idx_layout))[None, :]
    idx_offs = offs_idx_dim0 * idx_dim1 + offs_idx_dim1
    idx = ttgl.load(idx_ptr + idx_offs)

    out = ttgl.gather(src, idx, axis)

    ttgl.store(out_ptr + idx_offs, out)


def _gather_linear_layouts():
    if THREADS_PER_WARP == 32:
        return [(0,
                 ttgl.DistributedLinearLayout(
                     reg_bases=[[0, 2], [2, 0]],
                     lane_bases=[[0, 8], [8, 0], [1, 0], [4, 0], [16, 0]],
                     warp_bases=[[0, 1], [0, 4]],
                     block_bases=[],
                     shape=[32, 16],
                 ),
                 ttgl.DistributedLinearLayout(
                     reg_bases=[[2, 0], [0, 2]],
                     lane_bases=[[0, 8], [16, 0], [1, 0], [8, 0], [4, 0]],
                     warp_bases=[[0, 1], [0, 4]],
                     block_bases=[],
                     shape=[32, 16],
                 )),
                (0,
                 ttgl.DistributedLinearLayout(
                     reg_bases=[[0, 2], [32, 0], [2, 0], [0, 16], [0, 32], [64, 0]],
                     lane_bases=[[0, 8], [8, 0], [1, 0], [4, 0], [16, 0]],
                     warp_bases=[[0, 1], [0, 4]],
                     block_bases=[],
                     shape=[128, 64],
                 ),
                 ttgl.DistributedLinearLayout(
                     reg_bases=[[0, 2], [32, 0], [0, 32], [2, 0], [0, 16], [64, 0], [128, 0]],
                     lane_bases=[[0, 8], [8, 0], [1, 0], [4, 0], [16, 0]],
                     warp_bases=[[0, 1], [0, 4]],
                     block_bases=[],
                     shape=[256, 64],
                 )),
                (0,
                 ttgl.DistributedLinearLayout(
                     reg_bases=[],
                     lane_bases=[[1], [2], [4], [8], [16]],
                     warp_bases=[],
                     block_bases=[],
                     shape=[32],
                 ),
                 ttgl.DistributedLinearLayout(
                     reg_bases=[],
                     lane_bases=[[1], [2], [4], [8], [16]],
                     warp_bases=[],
                     block_bases=[],
                     shape=[32],
                 )),
                (0,
                 ttgl.DistributedLinearLayout(
                     reg_bases=[],
                     lane_bases=[[1], [2], [4], [8], [16]],
                     warp_bases=[],
                     block_bases=[],
                     shape=[32],
                 ),
                 ttgl.DistributedLinearLayout(
                     reg_bases=[[32]],
                     lane_bases=[[1], [2], [4], [8], [16]],
                     warp_bases=[],
                     block_bases=[],
                     shape=[64],
                 )),
                (0,
                 ttgl.DistributedLinearLayout(
                     reg_bases=[[1]],
                     lane_bases=[[2], [4], [8], [16], [32]],
                     warp_bases=[],
                     block_bases=[],
                     shape=[64],
                 ),
                 ttgl.DistributedLinearLayout(
                     reg_bases=[],
                     lane_bases=[[1], [2], [4], [8], [16]],
                     warp_bases=[],
                     block_bases=[],
                     shape=[32],
                 )),
                (0,
                 ttgl.DistributedLinearLayout(
                     reg_bases=[[0, 1]],
                     lane_bases=[[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]],
                     warp_bases=[],
                     block_bases=[],
                     shape=[32, 2],
                 ),
                 ttgl.DistributedLinearLayout(
                     reg_bases=[[0, 1]],
                     lane_bases=[[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]],
                     warp_bases=[],
                     block_bases=[],
                     shape=[32, 2],
                 )),
                (0,
                 ttgl.DistributedLinearLayout(
                     reg_bases=[[1, 0]],
                     lane_bases=[[2, 0], [4, 0], [8, 0], [16, 0], [0, 1]],
                     warp_bases=[],
                     block_bases=[],
                     shape=[32, 2],
                 ),
                 ttgl.DistributedLinearLayout(
                     reg_bases=[[1, 0]],
                     lane_bases=[[2, 0], [4, 0], [8, 0], [16, 0], [0, 1]],
                     warp_bases=[],
                     block_bases=[],
                     shape=[32, 2],
                 ))]
    elif THREADS_PER_WARP == 64:
        return [(0,
                 ttgl.DistributedLinearLayout(
                     reg_bases=[[0, 2], [2, 0]],
                     lane_bases=[[0, 8], [8, 0], [1, 0], [4, 0], [16, 0], [32, 0]],
                     warp_bases=[[0, 1], [0, 4]],
                     block_bases=[],
                     shape=[64, 16],
                 ),
                 ttgl.DistributedLinearLayout(
                     reg_bases=[[2, 0], [0, 2]],
                     lane_bases=[[0, 8], [16, 0], [1, 0], [8, 0], [4, 0], [32, 0]],
                     warp_bases=[[0, 1], [0, 4]],
                     block_bases=[],
                     shape=[64, 16],
                 )),
                (0,
                 ttgl.DistributedLinearLayout(
                     reg_bases=[[0, 2], [2, 0], [0, 16], [0, 32]],
                     lane_bases=[[0, 8], [8, 0], [1, 0], [4, 0], [16, 0], [32, 0]],
                     warp_bases=[[0, 1], [0, 4]],
                     block_bases=[],
                     shape=[64, 64],
                 ),
                 ttgl.DistributedLinearLayout(
                     reg_bases=[[0, 2], [0, 32], [2, 0], [0, 16], [64, 0]],
                     lane_bases=[[0, 8], [8, 0], [1, 0], [4, 0], [16, 0], [32, 0]],
                     warp_bases=[[0, 1], [0, 4]],
                     block_bases=[],
                     shape=[128, 64],
                 )),
                (0,
                 ttgl.DistributedLinearLayout(
                     reg_bases=[],
                     lane_bases=[[1], [2], [4], [8], [16], [32]],
                     warp_bases=[],
                     block_bases=[],
                     shape=[64],
                 ),
                 ttgl.DistributedLinearLayout(
                     reg_bases=[],
                     lane_bases=[[1], [2], [4], [8], [16], [32]],
                     warp_bases=[],
                     block_bases=[],
                     shape=[64],
                 )),
                (0,
                 ttgl.DistributedLinearLayout(
                     reg_bases=[],
                     lane_bases=[[1], [2], [4], [8], [16], [32]],
                     warp_bases=[],
                     block_bases=[],
                     shape=[64],
                 ),
                 ttgl.DistributedLinearLayout(
                     reg_bases=[[64]],
                     lane_bases=[[1], [2], [4], [8], [16], [32]],
                     warp_bases=[],
                     block_bases=[],
                     shape=[128],
                 )),
                (0,
                 ttgl.DistributedLinearLayout(
                     reg_bases=[[1]],
                     lane_bases=[[2], [4], [8], [16], [32], [64]],
                     warp_bases=[],
                     block_bases=[],
                     shape=[128],
                 ),
                 ttgl.DistributedLinearLayout(
                     reg_bases=[],
                     lane_bases=[[1], [2], [4], [8], [16], [32]],
                     warp_bases=[],
                     block_bases=[],
                     shape=[64],
                 )),
                (0,
                 ttgl.DistributedLinearLayout(
                     reg_bases=[[0, 1]],
                     lane_bases=[[1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [32, 0]],
                     warp_bases=[],
                     block_bases=[],
                     shape=[64, 2],
                 ),
                 ttgl.DistributedLinearLayout(
                     reg_bases=[[0, 1]],
                     lane_bases=[[1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [32, 0]],
                     warp_bases=[],
                     block_bases=[],
                     shape=[64, 2],
                 )),
                (0,
                 ttgl.DistributedLinearLayout(
                     reg_bases=[[1, 0]],
                     lane_bases=[[2, 0], [4, 0], [8, 0], [16, 0], [0, 1], [32, 0]],
                     warp_bases=[],
                     block_bases=[],
                     shape=[64, 2],
                 ),
                 ttgl.DistributedLinearLayout(
                     reg_bases=[[1, 0]],
                     lane_bases=[[2, 0], [4, 0], [8, 0], [16, 0], [0, 1], [32, 0]],
                     warp_bases=[],
                     block_bases=[],
                     shape=[64, 2],
                 ))]
    else:
        raise RuntimeError(f"Unsupported THREADS_PER_WARP: {THREADS_PER_WARP}")


def _gather_layouts():
    return [
        (
            0,
            ttgl.BlockedLayout(
                size_per_thread=[1],
                threads_per_warp=[THREADS_PER_WARP],
                warps_per_cta=[4],
                order=[0],
            ),
            ttgl.BlockedLayout(
                size_per_thread=[1],
                threads_per_warp=[THREADS_PER_WARP],
                warps_per_cta=[4],
                order=[0],
            ),
            [16],
        ),
        (
            0,
            ttgl.BlockedLayout(
                size_per_thread=[2, 1],
                threads_per_warp=[THREADS_PER_WARP, 1],
                warps_per_cta=[1, 4],
                order=[1, 0],
            ),
            ttgl.BlockedLayout(
                size_per_thread=[2, 1],
                threads_per_warp=[THREADS_PER_WARP, 1],
                warps_per_cta=[1, 4],
                order=[1, 0],
            ),
            [64, 1],
        ),
    ]


def _gather_cases():
    # Normalize linear-layout cases to include explicit src/idx shapes
    for axis, s_layout, i_layout in _gather_linear_layouts():
        yield (axis, s_layout, i_layout, tuple(s_layout.shape), tuple(i_layout.shape))
    # Normalize non-linear cases to (src_shape, idx_shape) form
    for axis, s_layout, i_layout, shape in _gather_layouts():
        shape_t = tuple(shape)
        yield (axis, s_layout, i_layout, shape_t, shape_t)


@pytest.mark.parametrize("axis, src_layout, index_layout, src_shape, idx_shape", _gather_cases())
def test_gather_layouts(axis, src_layout, index_layout, src_shape, idx_shape, device):
    src = torch.randn(src_shape, device=device)
    indices = torch.randint(0, src.shape[axis], idx_shape, device=device)
    out = torch.zeros_like(indices, device=device, dtype=src.dtype)
    ref = torch.gather(src, axis, indices)

    # Compute num_warps uniformly from layout/shape for both linear and non-linear cases
    num_warps = int(torch.prod(torch.tensor(ttgl._layouts.warps_per_cta(src_layout, src_shape))))

    if len(src_shape) == 1:
        obj = _gather_kernel_1d[(1, )](
            src,
            indices,
            out,
            axis,
            src_shape[0],
            idx_shape[0],
            src_layout,
            index_layout,
            num_warps=num_warps,
        )
    elif len(src_shape) == 2:
        obj = _gather_kernel_2d[(1, )](
            src,
            indices,
            out,
            axis,
            src_shape[0],
            src_shape[1],
            idx_shape[0],
            idx_shape[1],
            src_layout,
            index_layout,
            num_warps=num_warps,
        )
    else:
        raise RuntimeError(f"Unsupported shape: {src_shape}")

    torch.testing.assert_close(out, ref, rtol=0, atol=0)
    assert ("nvvm.shfl.sync.idx" in obj.asm["llir"]) or ("llvm.amdgcn.ds.bpermute" in obj.asm["llir"])


@pytest.mark.parametrize("M, N, M_tile_size, N_tile_size",
                         [[128, 128, 64, 64], [128, 128, 64, 32], [128, 64, 64, 32], [256, 128, 64, 64]])
@pytest.mark.parametrize("shared_layout_cfg", [
    pytest.param(("swizzled", None, None, None), id="swizzled"),
    pytest.param(("partitioned-swizzled", 0, 2, 1), id="partitioned-swizzled-dim0-p2-g1"),
    pytest.param(("partitioned-swizzled", 0, 2, 2), id="partitioned-swizzled-dim0-p2-g2"),
    pytest.param(("partitioned-swizzled", 0, 4, 1), id="partitioned-swizzled-dim0-p4-g1"),
    pytest.param(("partitioned-swizzled", 1, 2, 1), id="partitioned-swizzled-dim1-p2-g1"),
    pytest.param(("partitioned-swizzled", 1, 2, 2), id="partitioned-swizzled-dim1-p2-g2"),
    pytest.param(("partitioned-swizzled", 1, 4, 1), id="partitioned-swizzled-dim1-p4-g1"),
    pytest.param(("partitioned-padded", 0, 2, 1), id="partitioned-padded-dim0-p2-g1"),
    pytest.param(("partitioned-padded", 0, 2, 2), id="partitioned-padded-dim0-p2-g2"),
    pytest.param(("partitioned-padded", 0, 4, 1), id="partitioned-padded-dim0-p4-g1"),
    pytest.param(("partitioned-padded", 1, 2, 1), id="partitioned-padded-dim1-p2-g1"),
    pytest.param(("partitioned-padded", 1, 2, 2), id="partitioned-padded-dim1-p2-g2"),
    pytest.param(("partitioned-padded", 1, 4, 1), id="partitioned-padded-dim1-p4-g1"),
])
def test_memdesc_subslice(M, N, M_tile_size, N_tile_size, shared_layout_cfg, device):
    if M % M_tile_size != 0 or N % N_tile_size != 0:
        pytest.skip(f"Shape size ({M}, {N}) must be divisible by tile size ({M_tile_size}, {N_tile_size})")

    layout_type, partition_dim, num_partitions, num_groups = shared_layout_cfg
    if layout_type == "swizzled":
        shared_layout = ttgl.SwizzledSharedLayout(vec=8, per_phase=1, max_phase=8, order=[1, 0])
    else:
        assert layout_type in ("partitioned-swizzled", "partitioned-padded")
        if not is_hip():
            pytest.skip("PartitionedSharedLayout is supported only on AMD backend")
        if layout_type == "partitioned-swizzled":
            inner_layout = ttgl.SwizzledSharedLayout(vec=4, per_phase=2, max_phase=8, order=[1, 0])
        else:
            pad_interval, pad_amount = 16, 4
            # Skip cases that would not fit in LDS on the current architecture.
            elem_size = 2  # float16
            padded_bytes = ((M * N * (pad_interval + pad_amount)) // pad_interval) * elem_size
            if padded_bytes >= get_hip_lds_size():
                pytest.skip(f"Partitioned-padded allocation ({padded_bytes} B) exceeds LDS ({get_hip_lds_size()} B)")
            inner_layout = ttgl.PaddedSharedLayout.with_identity_for(
                interval_padding_pairs=[[pad_interval, pad_amount]],
                shape=[M, N],
                order=[1, 0],
            )
        shared_layout = PartitionedSharedLayout(
            num_partitions=num_partitions,
            num_groups=num_groups,
            partition_dim=partition_dim,
            partition_layout=inner_layout,
        )

    num_rows_per_warp = THREADS_PER_WARP // 4
    blocked_layout = ttgl.BlockedLayout(size_per_thread=[1, 8], threads_per_warp=[num_rows_per_warp, 4],
                                        warps_per_cta=[4, 1], order=[1, 0])

    @gluon.jit
    def kernel(
        out,
        M: ttgl.constexpr,
        N: ttgl.constexpr,
        BLOCK_SIZE_M: ttgl.constexpr,
        BLOCK_SIZE_N: ttgl.constexpr,
        blocked_layout: ttgl.constexpr,
        shared_layout: ttgl.constexpr,
    ):
        offs_m = ttgl.arange(0, M, layout=ttgl.SliceLayout(1, blocked_layout))[:, None]
        offs_n = ttgl.arange(0, N, layout=ttgl.SliceLayout(0, blocked_layout))[None, :]
        vals = ttgl.load(out + offs_m * N + offs_n)

        smem: ttgl.shared_memory_descriptor = ttgl.allocate_shared_memory(vals.dtype, (M, N), shared_layout, value=vals)
        for i in ttgl.static_range(M // BLOCK_SIZE_M):
            for j in ttgl.static_range(N // BLOCK_SIZE_N):
                tile = smem.slice(i * BLOCK_SIZE_M, BLOCK_SIZE_M, dim=0).slice(j * BLOCK_SIZE_N, BLOCK_SIZE_N, dim=1)
                tile_vals = tile.load(blocked_layout)
                tile_offs_m = ttgl.arange(0, BLOCK_SIZE_M, layout=ttgl.SliceLayout(1, blocked_layout))[:, None]
                tile_offs_n = ttgl.arange(0, BLOCK_SIZE_N, layout=ttgl.SliceLayout(0, blocked_layout))[None, :]
                linear_idx = tile_offs_m * N + tile_offs_n + i * BLOCK_SIZE_M * N + j * BLOCK_SIZE_N
                tile.store(linear_idx + tile_vals)

        vals = smem.load(blocked_layout)
        ttgl.store(out + offs_m * N + offs_n, vals)

    out = torch.zeros((M, N), device=device, dtype=torch.float16)
    kernel[(1, )](out, M, N, M_tile_size, N_tile_size, blocked_layout, shared_layout)

    out_ref = torch.arange(0, M * N, device=device).reshape((M, N)).to(torch.float16)
    torch.testing.assert_close(out, out_ref, rtol=0, atol=0)


@pytest.mark.skipif(not is_hopper_or_newer(), reason="num_ctas > 1 requires NVIDIA SM90+ (Hopper)")
def test_memdesc_subslice_two_cta_broadcasted_cga(device):
    ALLOC = 512
    SLICE = 256
    NUM_CTAS = 2
    alloc_layout = ttgl.BlockedLayout([4], [THREADS_PER_WARP], [4], [0], cga_layout=[[0]])
    load_layout = ttgl.BlockedLayout([2], [THREADS_PER_WARP], [4], [0], cga_layout=[[0]])
    store_layout = ttgl.BlockedLayout([1], [THREADS_PER_WARP], [4], [0], cga_layout=[[1]])
    shared_layout = ttgl.SwizzledSharedLayout(
        vec=1,
        per_phase=1,
        max_phase=1,
        order=[0],
        cga_layout=[[0]],
    )

    @gluon.jit
    def kernel(
        in_ptr,
        out_ptr,
        ALLOC: ttgl.constexpr,
        SLICE: ttgl.constexpr,
        alloc_layout: ttgl.constexpr,
        load_layout: ttgl.constexpr,
        store_layout: ttgl.constexpr,
        shared_layout: ttgl.constexpr,
    ):
        alloc_offs = ttgl.arange(0, ALLOC, layout=alloc_layout)
        vals = ttgl.load(in_ptr + alloc_offs)

        smem = ttgl.allocate_shared_memory(vals.dtype, (ALLOC, ), shared_layout, value=vals)
        ttgl.barrier(cluster=True)

        tile = smem.slice(0, SLICE)
        tile_vals = tile.load(load_layout)

        store_offs = ttgl.arange(0, SLICE, layout=store_layout)
        store_vals = ttgl.convert_layout(tile_vals, store_layout)
        ttgl.store(out_ptr + store_offs, store_vals + store_offs + 1)

    inp = torch.arange(0, ALLOC, device=device, dtype=torch.int32)
    out = torch.zeros((SLICE, ), device=device, dtype=torch.int32)
    kernel[(1, )](inp, out, ALLOC, SLICE, alloc_layout, load_layout, store_layout, shared_layout, num_warps=4,
                  num_ctas=NUM_CTAS)

    out_ref = inp[:SLICE] + torch.arange(1, SLICE + 1, device=device, dtype=torch.int32)
    torch.testing.assert_close(out, out_ref, rtol=0, atol=0)


@pytest.mark.skipif(is_cuda(), reason="PartitionedSharedLayout is not supported in NV backend")
@pytest.mark.parametrize("M, K", [(64, 32), (128, 64)])
@pytest.mark.parametrize("num_partitions", [2, 4])
@pytest.mark.parametrize("num_groups", [1, 2])
@pytest.mark.parametrize("partition_dim", [0, 1])
@pytest.mark.parametrize("partition_layout_type", ["swizzled", "padded"])
def test_partitioned_shared_layout(M, K, num_partitions, num_groups, partition_dim, partition_layout_type):
    """
    Test that PartitionedSharedLayout works correctly with various configurations.

    This test allocates shared memory with partitioned layout, performs a
    round-trip copy (global -> shared -> global), and verifies data integrity.

    Parameters:
    - M, K: Tensor dimensions
    - num_partitions: Number of physical memory partitions (2 or 4)
    - num_groups: Number of groups (1 or 2)
    - partition_dim: Dimension along which to partition (0=rows, 1=cols)
    - partition_layout_type: Layout within each piece ("swizzled" or "padded")
    """

    blocked_layout = ttgl.BlockedLayout(
        size_per_thread=[1, 8],
        threads_per_warp=[THREADS_PER_WARP // 4, 4],
        warps_per_cta=[4, 1],
        order=[1, 0],
    )

    @gluon.jit
    def partitioned_copy_kernel(
        input_ptr,
        output_ptr,
        M: ttgl.constexpr,
        K: ttgl.constexpr,
        blocked: ttgl.constexpr,
        partitioned_layout: ttgl.constexpr,
    ):
        # Create 2D indices
        row_idx = ttgl.arange(0, M, layout=ttgl.SliceLayout(1, blocked))[:, None]
        col_idx = ttgl.arange(0, K, layout=ttgl.SliceLayout(0, blocked))[None, :]
        offsets = row_idx * K + col_idx

        # Load data from global memory
        data = ttgl.load(input_ptr + offsets)

        # Allocate partitioned shared memory and store data
        smem = ttgl.allocate_shared_memory(ttgl.float16, [M, K], partitioned_layout, data)

        # Load from shared memory
        loaded = smem.load(blocked)

        # Store back to global memory
        ttgl.store(output_ptr + offsets, loaded)

    # Create partition layout
    if partition_layout_type == "swizzled":
        inner_layout = ttgl.SwizzledSharedLayout(
            vec=4,
            per_phase=2,
            max_phase=8,
            order=[1, 0],
        )
    elif partition_layout_type == "padded":
        inner_layout = ttgl.PaddedSharedLayout.with_identity_for(
            interval_padding_pairs=[[16, 4]],
            shape=[M, K],
            order=[1, 0],
        )
    else:
        raise ValueError(f"Unknown partition_layout_type: {partition_layout_type}")

    # Create partitioned layout
    partitioned_layout = PartitionedSharedLayout(
        num_partitions=num_partitions,
        num_groups=num_groups,
        partition_dim=partition_dim,
        partition_layout=inner_layout,
    )

    # Create input/output tensors
    input_tensor = torch.randn((M, K), device="cuda", dtype=torch.float16)
    output_tensor = torch.empty_like(input_tensor)

    # Run the kernel
    partitioned_copy_kernel[(1, )](
        input_tensor,
        output_tensor,
        M,
        K,
        blocked_layout,
        partitioned_layout,
        num_warps=4,
    )

    # Verify output matches input
    torch.testing.assert_close(output_tensor, input_tensor, atol=0, rtol=0)



# =============================================================================
# gfx1250 `amdgpu.cvt_scale_pk` (gluon `cvt_scale_pk`) on-device numeric tests
# =============================================================================

# fp4 (e2m1) 4-bit code -> value LUT (sign-magnitude).
_FP4_LUT = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]

_TTGL_DT = {"f16": ttgl.float16, "bf16": ttgl.bfloat16, "f32": ttgl.float32}
_TORCH_DT = {"f16": torch.float16, "bf16": torch.bfloat16, "f32": torch.float32}


def _decode_fp4(packed, pack_axis=1):
    """Unpack each integer into adjacent low-to-high fp4 nibbles."""
    lut = torch.tensor(_FP4_LUT, device=packed.device, dtype=torch.float32)
    packed_i64 = packed.to(torch.int64)
    packing_factor = torch.iinfo(packed.dtype).bits // 4
    values = [
        lut[((packed_i64 >> (4 * nibble)) & 0xF).long()] for nibble in range(packing_factor)
    ]
    return torch.stack(values, dim=pack_axis + 1).flatten(pack_axis, pack_axis + 1)


def _cvt_scale_pk_to_linear_layout(layout, shape):
    if isinstance(layout, ttgl.DistributedLinearLayout):
        assert list(layout.shape) == list(shape)
        return layout
    # cvt_scale_pk is gfx1250-only; use its target when expanding higher-level
    # blocked, slice, WMMA, and dot-operand layouts on the host.
    context = ir.context()
    ir.load_dialects(context)
    builder = gluon_ir.GluonOpBuilder(context, "gfx1250")
    return builder.to_linear_layout(layout._to_ir(builder), list(shape))


def _cvt_scale_pk_fp4_element_layout(packed_linear, out_shape, pack_axis, packing_factor=2):
    """Expand a packed integer layout to its logical low-to-high-nibble layout."""

    def expand(bases):
        return [basis[:pack_axis] + [packing_factor * basis[pack_axis]] + basis[pack_axis + 1:]
                for basis in bases]

    minor_element_bases = []
    for bit in range(packing_factor.bit_length() - 1):
        basis = [0] * len(out_shape)
        basis[pack_axis] = 1 << bit
        minor_element_bases.append(basis)
    return ttgl.DistributedLinearLayout(
        minor_element_bases + expand(packed_linear.reg_bases),
        expand(packed_linear.lane_bases),
        expand(packed_linear.warp_bases),
        expand(packed_linear.block_bases),
        out_shape,
    )


@gluon.constexpr_function
def _cvt_scale_pk_compact_scale_layout(out_linear, out_shape, scale_shape, axis, scale_factor):
    """Map element coordinates to k // scale_factor scale coordinates."""

    def compact(bases):
        return [basis[:axis] + [basis[axis] // scale_factor] + basis[axis + 1:] for basis in bases]

    # Broadcast register bases do not carry unique tensor elements and are not
    # materialized by the lowering. Lane/warp/block broadcasts remain because
    # those domains still participate in execution.
    reg_bases = [basis for basis in compact(out_linear.reg_bases) if any(basis)]

    return ttgl.DistributedLinearLayout(
        reg_bases,
        compact(out_linear.lane_bases),
        compact(out_linear.warp_bases),
        compact(out_linear.block_bases),
        scale_shape,
    )


def _cvt_scale_pk_make_scale_layout(val_layout, out_shape, axis, scale_factor, pack_axis=None,
                                    packing_factor=2):
    if pack_axis is None:
        out_linear = _cvt_scale_pk_to_linear_layout(val_layout, out_shape)
    else:
        packed_shape = list(out_shape)
        packed_shape[pack_axis] //= packing_factor
        packed_linear = _cvt_scale_pk_to_linear_layout(val_layout, packed_shape)
        out_linear = _cvt_scale_pk_fp4_element_layout(packed_linear, out_shape, pack_axis, packing_factor)
    scale_shape = list(out_shape)
    scale_shape[axis] //= scale_factor
    return _cvt_scale_pk_compact_scale_layout(out_linear, out_shape, scale_shape, axis, scale_factor)


@gluon.jit
def _cvt_scale_pk_layout_kernel(val_ptr, scale_ptr, out_ptr, V0: ttgl.constexpr, V1: ttgl.constexpr, S0: ttgl.constexpr,
                                S1: ttgl.constexpr, O0: ttgl.constexpr, O1: ttgl.constexpr, VAL_LAYOUT: ttgl.constexpr,
                                SCALE_LAYOUT: ttgl.constexpr, AXIS: ttgl.constexpr, SCALE_SEL: ttgl.constexpr,
                                ELEM_TYPE: ttgl.constexpr, K_WIDTH: ttgl.constexpr, PACK_AXIS: ttgl.constexpr):
    v0 = ttgl.arange(0, V0, layout=ttgl.SliceLayout(1, VAL_LAYOUT))[:, None]
    v1 = ttgl.arange(0, V1, layout=ttgl.SliceLayout(0, VAL_LAYOUT))[None, :]
    val = ttgl.load(val_ptr + v0 * V1 + v1)

    s0 = ttgl.arange(0, S0, layout=ttgl.SliceLayout(1, SCALE_LAYOUT))[:, None]
    s1 = ttgl.arange(0, S1, layout=ttgl.SliceLayout(0, SCALE_LAYOUT))[None, :]
    scale = ttgl.load(scale_ptr + s0 * S1 + s1)

    if PACK_AXIS < 0 and K_WIDTH < 0:
        res = ttgl.amd.gfx1250.cvt_scale_pk(val, scale, axis=AXIS, scale_sel=SCALE_SEL, elem_type=ELEM_TYPE)
    elif PACK_AXIS < 0:
        res = ttgl.amd.gfx1250.cvt_scale_pk(val, scale, axis=AXIS, scale_sel=SCALE_SEL, elem_type=ELEM_TYPE,
                                            k_width=K_WIDTH)
    elif K_WIDTH < 0:
        res = ttgl.amd.gfx1250.cvt_scale_pk(val, scale, axis=AXIS, scale_sel=SCALE_SEL, elem_type=ELEM_TYPE,
                                            pack_axis=PACK_AXIS)
    else:
        res = ttgl.amd.gfx1250.cvt_scale_pk(val, scale, axis=AXIS, scale_sel=SCALE_SEL, elem_type=ELEM_TYPE,
                                            k_width=K_WIDTH, pack_axis=PACK_AXIS)

    out_layout: ttgl.constexpr = res.type.layout
    o0 = ttgl.arange(0, O0, layout=ttgl.SliceLayout(1, out_layout))[:, None]
    o1 = ttgl.arange(0, O1, layout=ttgl.SliceLayout(0, out_layout))[None, :]
    ttgl.store(out_ptr + o0 * O1 + o1, res)


def _cvt_scale_pk_blocked_layout(axis):
    if axis == 0:
        return ttgl.BlockedLayout([8, 1], [32, 1], [1, 4], [0, 1])
    return ttgl.BlockedLayout([1, 8], [1, 32], [4, 1], [1, 0])


def _cvt_scale_pk_blocked_slice_layout(axis):
    layout = _cvt_scale_pk_blocked_layout(axis)
    parent = ttgl.BlockedLayout(
        [1] + layout.size_per_thread,
        [1] + layout.threads_per_warp,
        [1] + layout.warps_per_cta,
        [dim + 1 for dim in layout.order] + [0],
    )
    return ttgl.SliceLayout(0, parent)


def _cvt_scale_pk_wmma_layout(axis, sliced=False):
    # WMMA v3 has its accumulator element dimension on opposite axes for the
    # two transpose modes. Pick the mode whose fast element dimension is axis.
    transposed = axis == 1
    if not sliced:
        return ttgl.amd.AMDWMMALayout(3, transposed, [[0, 1], [1, 0]], instr_shape=[16, 16, 128])
    parent = ttgl.amd.AMDWMMALayout(3, transposed, [[0, 0, 1], [0, 1, 0]], instr_shape=[16, 16, 128], rank=3)
    return ttgl.SliceLayout(0, parent)


def _cvt_scale_pk_dot_layout(axis, sliced=False, packed=False):
    operand_index = 1 if axis == 0 else 0
    k_width = 8 if packed else 16
    if not sliced:
        parent = ttgl.amd.AMDWMMALayout(3, True, [[0, 1], [1, 0]], instr_shape=[16, 16, 64 if packed else 128])
        return ttgl.DotOperandLayout(operand_index, parent, k_width)
    parent = ttgl.amd.AMDWMMALayout(3, True, [[0, 0, 1], [0, 1, 0]],
                                             instr_shape=[16, 16, 64 if packed else 128], rank=3)
    return ttgl.SliceLayout(0, ttgl.DotOperandLayout(operand_index, parent, k_width))


def _cvt_scale_pk_fp8_layout_cases():
    cases = []
    for axis in (0, 1):
        blocked_shape = (256, 4) if axis == 0 else (4, 256)
        dot_shape = (128, 32) if axis == 0 else (32, 128)
        for sliced in (False, True):
            suffix = "-slice" if sliced else ""
            blocked = _cvt_scale_pk_blocked_slice_layout(axis) if sliced else _cvt_scale_pk_blocked_layout(axis)
            cases.append((blocked, blocked_shape, axis, 256, f"blocked{suffix}-axis{axis}"))
            cases.append((_cvt_scale_pk_wmma_layout(axis,
                                                    sliced), (32, 32), axis, 16, f"wmma-v3{suffix}-axis{axis}"))
            cases.append(
                (_cvt_scale_pk_dot_layout(axis, sliced), dot_shape, axis, 32, f"dot-op{suffix}-axis{axis}"))
    return cases


def _cvt_scale_pk_fp4_layout_cases():
    cases = []
    for axis in (0, 1):
        pack_axis = axis
        blocked_shape = (512, 4) if axis == 0 else (4, 512)
        dot_shape = (128, 32) if axis == 0 else (32, 128)
        cases.append(
            (_cvt_scale_pk_blocked_layout(axis), blocked_shape, axis, pack_axis, 512,
             f"blocked-axis{axis}-pack{pack_axis}"))
        cases.append(
            (_cvt_scale_pk_dot_layout(axis, packed=True), dot_shape, axis, pack_axis, 32,
             f"dot-op-axis{axis}-pack{pack_axis}"))
    return cases


def _cvt_scale_pk_scale_sel_cases(is_fp4):
    # Enumerate every scale width that contains the byte(s) named by scale_sel.
    dtypes = {
        "b0": (torch.int8, torch.int16, torch.int32),
        "b1": (torch.int16, torch.int32),
        "b2": (torch.int32, ),
        "b3": (torch.int32, ),
        "b0b1": (torch.int16, torch.int32),
        "b2b3": (torch.int32, ),
        "b0b2": (torch.int32, ),
        "b1b3": (torch.int32, ),
    }
    scale_bytes = ("b0b1", "b2b3", "b0b2", "b1b3") if is_fp4 else ("b0", "b1", "b2", "b3", "b0b1", "b2b3")
    return [(((lane, bytes_), ), dtype, f"{lane}-{bytes_}-{str(dtype).removeprefix('torch.')}")
            for lane in ("h0", "h1")
            for bytes_ in scale_bytes
            for dtype in dtypes[bytes_]]


def _make_cvt_scale_pk_scale(shape, dtype, device):
    """Create compact scales with a distinct E8M0 payload in every byte."""
    info = torch.iinfo(dtype)
    bytes_per_element = info.bits // 8
    element_indices = torch.arange(math.prod(shape), dtype=torch.int64)
    words = torch.zeros_like(element_indices)
    for byte in range(bytes_per_element):
        # The prime period keeps lane^16 partners distinct for the power-of-two
        # layouts under test while limiting factors to 2^-8 through 2^8.
        exponents = 119 + (bytes_per_element * element_indices + byte) % 17
        words |= exponents << (8 * byte)
    if info.min < 0:
        sign_bit = 1 << (info.bits - 1)
        words = torch.where(words >= sign_bit, words - (1 << info.bits), words)
    return words.reshape(shape).to(device=device, dtype=dtype)


def _make_cvt_scale_pk_packed_fp4(shape, dtype, device):
    """Create packed fp4 storage with independently randomized nibbles."""
    packing_factor = torch.iinfo(dtype).bits // 4
    words = torch.zeros(shape, device=device, dtype=torch.int64)
    for nibble in range(packing_factor):
        codes = torch.randint(0, 16, shape, device=device, dtype=torch.int64)
        words |= codes << (4 * nibble)
    return words.to(dtype)


def _cvt_scale_pk_reference_factors(scale, scale_layout, out_layout, out_shape, k_axis, scale_factor, k_width,
                                    *, scale_sel=(("h0", "b0"), ), is_fp4=False):
    """Route compact E8M0 payloads to logical output coordinates."""
    if not 0 <= k_axis < scale.ndim:
        raise ValueError(f"invalid k_axis {k_axis} for a rank-{scale.ndim} scale tensor")
    if scale_factor <= 0:
        raise ValueError(f"scale_factor must be positive, got {scale_factor}")
    if k_width <= 0:
        raise ValueError(f"k_width must be positive, got {k_width}")
    if not scale_sel:
        raise ValueError("scale_sel must not be empty")
    if any(lane not in ("h0", "h1") for lane, _ in scale_sel):
        raise ValueError(f"invalid scale lane in {scale_sel}")

    scale_linear = _cvt_scale_pk_to_linear_layout(scale_layout, scale.shape)
    out_linear = _cvt_scale_pk_to_linear_layout(out_layout, out_shape)
    out_linear = ttgl.DistributedLinearLayout(
        [basis for basis in out_linear.reg_bases if any(basis)],
        out_linear.lane_bases,
        out_linear.warp_bases,
        out_linear.block_bases,
        out_shape,
    )
    rank = scale.ndim
    device = scale.device
    bases_by_domain = [out_linear.reg_bases, out_linear.lane_bases, out_linear.warp_bases, out_linear.block_bases]
    domain_indices = torch.meshgrid(
        *[torch.arange(1 << len(bases), device=device, dtype=torch.int64) for bases in bases_by_domain], indexing="ij")
    domain_indices = [indices.reshape(-1) for indices in domain_indices]

    def apply_bases(indices, bases):
        coords = torch.zeros((indices.numel(), rank), device=device, dtype=torch.int64)
        for bit, basis in enumerate(bases):
            basis = torch.tensor(basis, device=device, dtype=torch.int64)
            coords ^= ((indices >> bit) & 1)[:, None] * basis
        return coords

    out_domain_coords = [apply_bases(indices, bases) for indices, bases in zip(domain_indices, bases_by_domain)]
    reg_coords, lane_coords, warp_coords, block_coords = out_domain_coords
    dst_coords = reg_coords ^ lane_coords ^ warp_coords ^ block_coords
    required_scale_coords = dst_coords.clone()
    required_scale_coords[:, k_axis] //= scale_factor

    # One scale_sel entry is used by each register-local k_width group. The
    # selected source preserves lane bits 0..3 and chooses lane-half bit 4.
    selector_indices = (reg_coords[:, k_axis] // k_width) % len(scale_sel)
    selector_source_halves = torch.tensor([lane == "h1" for lane, _ in scale_sel], device=device, dtype=torch.int64)
    source_lanes = (domain_indices[1] & 15) | (selector_source_halves[selector_indices] << 4)

    # The emitted scale register is fixed across the warp. Find it from lane 0
    # for every output register, then verify that the selected lane x/x^16 owns
    # k // scale_factor for every physical output instance.
    out_reg_indices = torch.arange(1 << len(out_linear.reg_bases), device=device, dtype=torch.int64)
    base_out_coords = apply_bases(out_reg_indices, out_linear.reg_bases)
    base_scale_coords = base_out_coords.clone()
    base_scale_coords[:, k_axis] //= scale_factor
    base_selector_indices = (base_out_coords[:, k_axis] // k_width) % len(scale_sel)
    base_source_lanes = selector_source_halves[base_selector_indices] << 4
    scale_reg_candidates = torch.arange(1 << len(scale_linear.reg_bases), device=device, dtype=torch.int64)
    scale_reg_coords = apply_bases(scale_reg_candidates, scale_linear.reg_bases)
    scale_reg_by_out_reg = []
    for out_reg in range(out_reg_indices.numel()):
        source_lane_coords = apply_bases(base_source_lanes[out_reg:out_reg + 1], scale_linear.lane_bases)
        candidate_coords = scale_reg_coords ^ source_lane_coords
        matches = torch.all(candidate_coords == base_scale_coords[out_reg], dim=1).nonzero().flatten()
        if matches.numel() == 0:
            raise AssertionError("required scale is not owned by the selected lane")
        scale_reg_by_out_reg.append(matches[0])
    scale_reg_by_out_reg = torch.stack(scale_reg_by_out_reg)

    source_reg_indices = scale_reg_by_out_reg[domain_indices[0]]
    source_coords = apply_bases(source_reg_indices, scale_linear.reg_bases)
    source_coords ^= apply_bases(source_lanes, scale_linear.lane_bases)
    source_coords ^= apply_bases(domain_indices[2], scale_linear.warp_bases)
    source_coords ^= apply_bases(domain_indices[3], scale_linear.block_bases)
    if torch.any(source_coords != required_scale_coords):
        raise AssertionError("required scale is not owned by lane x or its half-warp peer")

    shape = torch.tensor(scale.shape, device=device, dtype=torch.int64)
    if torch.any(source_coords >= shape):
        raise AssertionError("scale layout produced an out-of-bounds coordinate")
    scale_words = scale.to(torch.int64)[tuple(source_coords[:, dim] for dim in range(rank))]

    # gfx1250 stepping behavior aliases the block16 byte encodings as captured
    # by the exhaustive OPSEL tests below.
    if is_fp4:
        byte_routes = {"b0b1": (0, 1), "b2b3": (2, 3), "b0b2": (0, 1), "b1b3": (2, 3)}
    else:
        byte_routes = {
            "b0": (0, 0),
            "b1": (1, 1),
            "b2": (2, 2),
            "b3": (3, 3),
            "b0b1": (0, 0),
            "b2b3": (2, 2),
        }
    if any(scale_bytes not in byte_routes for _, scale_bytes in scale_sel):
        raise ValueError(f"invalid scale byte selection in {scale_sel}")
    selector_byte_routes = torch.tensor([byte_routes[scale_bytes] for _, scale_bytes in scale_sel], device=device,
                                        dtype=torch.int64)
    destination_halves = ((domain_indices[1] >> 4) & 1)[:, None]
    byte_indices = selector_byte_routes[selector_indices].gather(1, destination_halves).squeeze(1)
    exponents = (scale_words >> (8 * byte_indices)) & 0xFF

    # Coalesce hardware aliases and verify that they agree on each output.
    logical_strides = []
    stride = 1
    for size in reversed(out_shape):
        logical_strides.append(stride)
        stride *= size
    logical_strides.reverse()
    flat_dst = (dst_coords * torch.tensor(logical_strides, device=device, dtype=torch.int64)).sum(dim=1)
    order = torch.argsort(flat_dst, stable=True)
    flat_dst = flat_dst[order]
    exponents = exponents[order]
    is_first = torch.ones_like(flat_dst, dtype=torch.bool)
    is_first[1:] = flat_dst[1:] != flat_dst[:-1]
    previous_exponents = torch.roll(exponents, 1)
    if torch.any(~is_first & (exponents != previous_exponents)):
        raise AssertionError("scale layout aliases route inconsistent scale values")
    if int(is_first.sum()) != math.prod(out_shape):
        raise AssertionError("output layout does not cover every tensor coordinate")

    routed_scale = torch.empty(math.prod(out_shape), device=device, dtype=torch.float32)
    routed_scale.index_copy_(0, flat_dst[is_first], torch.exp2(exponents[is_first].float() - 127))
    return routed_scale.reshape(out_shape)


def _cvt_scale_pk_contiguous_width(layout, shape, axis):
    linear = _cvt_scale_pk_to_linear_layout(layout, shape)
    width = 1
    while True:
        basis = [0] * len(shape)
        basis[axis] = width
        if basis not in linear.reg_bases:
            return width
        width *= 2


def _check_cvt_scale_pk(val, val_layout, scale, scale_layout, axis, scale_sel, *, elem_type=ttgl.float32,
                        out_dtype=torch.float32, k_width=None, pack_axis=None, num_warps=4):
    """Run cvt_scale_pk in Gluon and compare it with the Torch reference."""
    val_shape = tuple(val.shape)
    scale_shape = tuple(scale.shape)
    out_shape = list(val_shape)
    is_fp4 = pack_axis is not None
    packing_factor = torch.iinfo(val.dtype).bits // 4 if is_fp4 else 1
    if is_fp4:
        out_shape[pack_axis] *= packing_factor
    out_shape = tuple(out_shape)

    assert all(scale_shape[dim] == out_shape[dim] for dim in range(2) if dim != axis)
    assert out_shape[axis] % scale_shape[axis] == 0
    scale_factor = out_shape[axis] // scale_shape[axis]
    if pack_axis is None:
        out_layout = val_layout
    else:
        packed_linear = _cvt_scale_pk_to_linear_layout(val_layout, val_shape)
        out_layout = _cvt_scale_pk_fp4_element_layout(packed_linear, out_shape, pack_axis, packing_factor)
    effective_k_width = k_width
    if effective_k_width is None:
        effective_k_width = _cvt_scale_pk_contiguous_width(out_layout, out_shape, axis)

    out = torch.empty(out_shape, device=val.device, dtype=out_dtype)
    _cvt_scale_pk_layout_kernel[(1, )](val, scale, out, *val_shape, *scale_shape, *out_shape, val_layout, scale_layout,
                                       axis, scale_sel, elem_type, k_width if k_width is not None else -1,
                                       pack_axis if is_fp4 else -1, num_warps=num_warps)

    ref_val = _decode_fp4(val, pack_axis) if is_fp4 else val.float()
    factor = _cvt_scale_pk_reference_factors(scale, scale_layout, out_layout, out_shape, axis, scale_factor,
                                             effective_k_width, scale_sel=scale_sel, is_fp4=is_fp4)
    torch.testing.assert_close(out, (ref_val * factor).to(out_dtype), atol=0, rtol=0)


def test_cvt_scale_pk_reference_factors():
    out_shape = (1, 256)
    out_layout = ttgl.BlockedLayout([1, 8], [1, 32], [1, 1], [1, 0])
    scale_layout = _cvt_scale_pk_make_scale_layout(out_layout, out_shape, 1, 256)
    scale = _make_cvt_scale_pk_scale((1, 1), torch.int8, "cpu")

    factor = _cvt_scale_pk_reference_factors(scale, scale_layout, out_layout, out_shape, 1, 256, 8)

    exponent = scale.to(torch.int64) & 0xFF
    expected = torch.exp2(exponent.float() - 127).expand(out_shape)
    torch.testing.assert_close(factor, expected, atol=0, rtol=0)


def _cvt_scale_pk_fp8_cases():
    cases = []
    for val_layout, shape, axis, scale_factor, layout_id in _cvt_scale_pk_fp8_layout_cases():
        scale_shape = list(shape)
        scale_shape[axis] //= scale_factor
        scale_shape = tuple(scale_shape)
        scale_layout = _cvt_scale_pk_make_scale_layout(val_layout, shape, axis, scale_factor)
        for scale_sel, scale_dtype, scale_sel_id in _cvt_scale_pk_scale_sel_cases(is_fp4=False):
            cases.append(
                pytest.param(shape, val_layout, scale_shape, scale_layout, axis, None, scale_sel, scale_dtype,
                             torch.float8_e4m3fn, ttgl.float32, torch.float32, 4, id=f"{layout_id}-{scale_sel_id}"))

    shape = (1, 256)
    val_layout = ttgl.BlockedLayout([1, 8], [1, 32], [1, 1], [1, 0])
    scale_shape = (1, 1)
    scale_layout = _cvt_scale_pk_make_scale_layout(val_layout, shape, 1, 256)
    for val_kind, val_dtype in (("e4m3", torch.float8_e4m3fn), ("e5m2", torch.float8_e5m2)):
        for out_dt in ("f16", "bf16", "f32"):
            cases.append(
                pytest.param(shape, val_layout, scale_shape, scale_layout, 1, None, (("h0", "b0"), ), torch.int8,
                             val_dtype, _TTGL_DT[out_dt], _TORCH_DT[out_dt], 1, id=f"dtype-{val_kind}-{out_dt}"))

    wide_shape = (1, 1024)
    wide_layout = ttgl.BlockedLayout([1, 32], [1, 32], [1, 1], [1, 0])
    wide_scale_layout = _cvt_scale_pk_make_scale_layout(wide_layout, wide_shape, 1, 1024)
    cases.extend([
        pytest.param(wide_shape, wide_layout, (1, 1), wide_scale_layout, 1, 8,
                     (("h0", "b0"), ("h0", "b2")), torch.int32, torch.float8_e4m3fn, ttgl.float32,
                     torch.float32, 1, id="explicit-k-width-scale-sel-round-robin"),
        pytest.param(shape, val_layout, scale_shape, scale_layout, 1, None, (("h0", "b0b1"), ), torch.int16,
                     torch.float8_e4m3fn, ttgl.bfloat16, torch.bfloat16, 1, id="scale-i16-block16"),
    ])
    return cases


def _cvt_scale_pk_fp4_cases():
    cases = []
    for val_layout, i8_out_shape, axis, pack_axis, scale_factor, layout_id in _cvt_scale_pk_fp4_layout_cases():
        val_shape = list(i8_out_shape)
        val_shape[pack_axis] //= 2
        val_shape = tuple(val_shape)
        for val_dtype in (torch.uint8, torch.uint16, torch.uint32):
            packing_factor = torch.iinfo(val_dtype).bits // 4
            out_shape = list(val_shape)
            out_shape[pack_axis] *= packing_factor
            packed_scale_factor = scale_factor * packing_factor // 2
            scale_shape = list(out_shape)
            scale_shape[axis] //= packed_scale_factor
            scale_shape = tuple(scale_shape)
            scale_layout = _cvt_scale_pk_make_scale_layout(
                val_layout, out_shape, axis, packed_scale_factor, pack_axis, packing_factor)
            val_dtype_id = str(val_dtype).removeprefix("torch.")
            for scale_sel, scale_dtype, scale_sel_id in _cvt_scale_pk_scale_sel_cases(is_fp4=True):
                cases.append(
                    pytest.param(val_shape, val_layout, scale_shape, scale_layout, axis, pack_axis, None, scale_sel,
                                 val_dtype, scale_dtype, ttgl.float32, torch.float32, 4,
                                 id=f"{layout_id}-{val_dtype_id}-{scale_sel_id}"))

    val_shape = (1, 128)
    out_shape = (1, 256)
    val_layout = ttgl.BlockedLayout([1, 4], [1, 32], [1, 1], [1, 0])
    scale_shape = (1, 1)
    scale_layout = _cvt_scale_pk_make_scale_layout(val_layout, out_shape, 1, 256, 1)
    cases.extend([
        pytest.param(val_shape, val_layout, scale_shape, scale_layout, 1, 1, None, (("h0", "b0b1"), ), torch.uint8,
                     torch.int16, _TTGL_DT[out_dt], _TORCH_DT[out_dt], 1, id=f"dtype-{out_dt}")
        for out_dt in ("f16", "bf16", "f32")
    ])

    cross_val_shape = (2, 128)
    cross_val_layout = ttgl.BlockedLayout([1, 4], [1, 32], [1, 1], [1, 0])
    for val_dtype in (torch.uint8, torch.uint16, torch.uint32):
        packing_factor = torch.iinfo(val_dtype).bits // 4
        cross_out_shape = (cross_val_shape[0] * packing_factor, cross_val_shape[1])
        cross_scale_shape = (cross_out_shape[0], 1)
        cross_scale_layout = _cvt_scale_pk_make_scale_layout(
            cross_val_layout, cross_out_shape, 1, 128, 0, packing_factor)
        val_dtype_id = str(val_dtype).removeprefix("torch.")
        cases.append(
            pytest.param(cross_val_shape, cross_val_layout, cross_scale_shape, cross_scale_layout, 1, 0, 4,
                         (("h0", "b0b2"), ), val_dtype, torch.int32, ttgl.float32, torch.float32, 1,
                         id=f"pack-axis-ne-scale-axis-{val_dtype_id}-explicit-k-width"))
    return cases


@pytest.mark.skipif(not is_hip_gfx1250(), reason="Requires AMD gfx1250 (cvt.scale.pk8)")
@pytest.mark.parametrize(
    [
        "shape",
        "val_layout",
        "scale_shape",
        "scale_layout",
        "axis",
        "k_width",
        "scale_sel",
        "scale_dtype",
        "val_dtype",
        "elem_type",
        "out_dtype",
        "num_warps",
    ],
    _cvt_scale_pk_fp8_cases(),
)
def test_amd_cvt_scale_pk_fp8(device, shape, val_layout, scale_shape, scale_layout, axis, k_width, scale_sel,
                              scale_dtype, val_dtype, elem_type, out_dtype, num_warps):
    torch.manual_seed(0)
    val = (torch.randint(-4, 5, shape, device=device).float() * 0.5).to(val_dtype)
    scale = _make_cvt_scale_pk_scale(scale_shape, scale_dtype, device)
    _check_cvt_scale_pk(val, val_layout, scale, scale_layout, axis, scale_sel, elem_type=elem_type, out_dtype=out_dtype,
                        k_width=k_width, num_warps=num_warps)


@pytest.mark.skipif(not is_hip_gfx1250(), reason="Requires AMD gfx1250 (cvt.scale.pk8)")
@pytest.mark.parametrize(
    [
        "shape",
        "val_layout",
        "scale_shape",
        "scale_layout",
        "axis",
        "pack_axis",
        "k_width",
        "scale_sel",
        "val_dtype",
        "scale_dtype",
        "elem_type",
        "out_dtype",
        "num_warps",
    ],
    _cvt_scale_pk_fp4_cases(),
)
def test_amd_cvt_scale_pk_fp4(device, shape, val_layout, scale_shape, scale_layout, axis, pack_axis, k_width,
                              scale_sel, val_dtype, scale_dtype, elem_type, out_dtype, num_warps):
    torch.manual_seed(0)
    packed = _make_cvt_scale_pk_packed_fp4(shape, val_dtype, device)
    scale = _make_cvt_scale_pk_scale(scale_shape, scale_dtype, device)
    _check_cvt_scale_pk(packed, val_layout, scale, scale_layout, axis, scale_sel, elem_type=elem_type,
                        out_dtype=out_dtype, k_width=k_width, pack_axis=pack_axis, num_warps=num_warps)
