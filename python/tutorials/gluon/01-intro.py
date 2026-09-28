"""
Introduction to Gluon
=====================

Gluon is a GPU programming language based on the same compiler stack as Triton.
But unlike Triton, Gluon is a lower-level language that gives the user more
control and responsibility when implementing kernels.

This tutorial series covers GPU kernel development in Gluon, from the basics to
advanced optimization techniques and modern GPU hardware features, culminating
in building an efficient GEMM kernel. Basic familiarity with Triton is assumed.

At a high level, Gluon and Triton share many similarities. Both implement a
tile-based SPMD programming model, where tiles represent N-dimensional arrays
distributed over a "program". Both are Python DSLs sharing the same frontend
and JIT infrastructure.

Triton, however, abstracts many details of implementing kernels and GPU hardware
from the user. It defers to the compiler to manage tile layouts, memory
allocation, data movement, and asynchronity.

Getting these details right is important to kernel performance. While the Triton
compiler does a good job of generating efficient code for a wide range of
kernels, it can be beaten by hand-tuned low-level code. When this happens,
there is little the user can do to significantly improve performance since all
the details are hidden.

In Gluon, these details are exposed to the user. This means writing Gluon
kernels requires a deeper understanding of GPU hardware and the many aspects of
GPU programming, but it also enables writing more performant kernels by finely
controlling these low-level details.
"""

# %%
# Let's define a Gluon kernel and write its launcher. Use the `@gluon.jit`
# decorator to declare a Gluon kernel, and it can be invoked from Python with
# the same interface as a Triton kernel.

import pytest
import torch
import triton
from intj import make_launcher
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

# %%
# We illustrate this with a trivial kernel that copies a scalar.


@make_launcher(grid_arg=1, options={"num_warps": 1})
@gluon.jit
def copy_scalar_kernel(in_ptr, out_ptr):
    value = gl.load(in_ptr)
    gl.store(out_ptr, value)


# %%
# The launcher is host-side code that invokes the kernel. Here intj's
# `make_launcher` above wraps it: calls take the device, the stream, one int per
# grid dimension, then the kernel arguments, all positional. PyTorch tensors are
# converted to global memory pointers when passed to Gluon kernels, just like in
# Triton.


def copy_scalar(input, output):
    # Launch a single program.
    copy_scalar_kernel(torch.cuda.current_device(), torch.cuda.current_stream().cuda_stream, 1, input, output)


# %%
# Let's test the kernel. You can run the test with `pytest 01-intro.py`.


def test_copy_scalar():
    input = torch.tensor([42.0], device="cuda")
    output = torch.empty_like(input)
    copy_scalar(input, output)
    torch.testing.assert_close(input, output, atol=0, rtol=0)


# %%
# We can write a kernel with hyperparameters passed as constexpr arguments in
# much the same way as Triton. This is a trivial memcpy kernel implemented by
# subtiling the tensors into 1D blocks, where each program processes one block.


@gluon.jit
def memcpy_kernel(in_ptr, out_ptr, xnumel, XBLOCK: gl.constexpr):
    # Each program processes the addresses [pid, pid + BLOCK_X), clamped into
    # the range [0, xnumel).
    pid = gl.program_id(0)
    start = pid * XBLOCK
    end = min(start + XBLOCK, xnumel)
    for i in range(start, end):
        value = gl.load(in_ptr + i)
        gl.store(out_ptr + i, value)


# memcpy_kernel stays a JIT function because memcpy_kernel_autotune below calls it.
_memcpy_launcher = make_launcher(memcpy_kernel, grid_arg=1, options={"num_warps": 1})


def memcpy(input, output, XBLOCK):
    xnumel = input.numel()
    _memcpy_launcher(torch.cuda.current_device(), torch.cuda.current_stream().cuda_stream, triton.cdiv(xnumel, XBLOCK),
                     input, output, xnumel, XBLOCK)


@pytest.mark.parametrize("XBLOCK", [64])
@pytest.mark.parametrize("xnumel", [40, 500])
def test_memcpy(XBLOCK, xnumel):
    torch.manual_seed(0)
    input = torch.randn(xnumel, device="cuda")
    output = torch.empty_like(input)
    memcpy(input, output, XBLOCK)
    torch.testing.assert_close(input, output, atol=0, rtol=0)


# %%
# Gluon hyperparameters can be autotuned like Triton as well. Let's autotune
# XBLOCK as an example.


# The grid reads the tuned XBLOCK; intj compiles it into the launcher.
def _memcpy_autotune_grid(xnumel: int, XBLOCK: int):
    return (triton.cdiv(xnumel, XBLOCK), )


@make_launcher(grid_cpp=_memcpy_autotune_grid)
@triton.autotune(
    configs=[triton.Config({"XBLOCK": 2**i}, num_warps=1) for i in range(8, 14)],
    key=["xnumel"],
)
@gluon.jit
def memcpy_kernel_autotune(in_ptr, out_ptr, xnumel, XBLOCK: gl.constexpr):
    memcpy_kernel(in_ptr, out_ptr, xnumel, XBLOCK)


def memcpy_autotune(input, output):
    # XBLOCK is tuned, so it is not passed.
    memcpy_kernel_autotune(torch.cuda.current_device(), torch.cuda.current_stream().cuda_stream, input, output,
                           input.numel())


# %%
# Run this with `TRITON_PRINT_AUTOTUNING=1 python 01-intro.py` to see which
# XBLOCK gets selected. On GB200, the best XBLOCK ends up being 2048 to copy
# 8 GB of data at about 666 GB/s, far from the 8 TB/s peak bandwidth of the GPU.
#
# ```
# Time:        24.00 ms
# Throughput: 666.24 GB/s
# ```

if __name__ == "__main__":
    torch.manual_seed(0)
    xnumel = 2 << 30
    input = torch.randn(xnumel, device="cuda")
    output = torch.empty_like(input)

    fn = lambda: memcpy_autotune(input, output)
    ms = triton.testing.do_bench(fn)
    gbytes = 2 * xnumel * input.element_size() >> 30
    print("Benchmarking memcpy")
    print("===================")
    print(f"Time:        {ms:.2f} ms")
    print(f"Throughput: {gbytes / (ms * 1e-3):.2f} GB/s")

# %%
# Since performance is the main motiviation for writing kernels in Gluon, let's
# spend time exploring that. First, we are not fully utilizing the parallelism
# of the GPU. Each Gluon "program" corresponds to a thread block (CTA) on the
# GPU, and while the GPU can execute many CTAs at once, in our kernel each CTA
# copies 1 element at a time.
#
# In order to copy many elements at once, we need to load and store tiles, but
# that will require picking a layout and understanding which layouts perform
# better than others. In the next tutorial, we will cover the basics of layouts
# in Gluon and how they can affect performance.
#
# The main things you should take away from this tutorial are:
#
# - The high-level aspects of writing Gluon kernels are the same as writing
#   Triton kernels.
# - Gluon implements a tile-based SPMD programming model that should be familiar
#   to those experienced with Triton.
# - Gluon changes how device code is written, and only changes host-side code
#   insofar as Gluon kernels may have more hyperparameters.
