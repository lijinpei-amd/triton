import torch

import triton
import triton.language as tl
from triton.runtime.jit import JITFunction
from intj import make_launcher


def _intj_launcher(kernel, grid):
    """One intj launcher, which owns its tuner cache; tuned values are not passed."""
    launcher = make_launcher(kernel, **({"grid_py": grid} if callable(grid) else {"grid_arg": len(grid)}))
    dims = () if callable(grid) else tuple(grid)
    return lambda *args: launcher(torch.cuda.current_device(), torch.cuda.current_stream().cuda_stream, *dims, *args)


def do_bench(kernel_call, quantiles, use_cuda_graph=False):
    return triton.testing.do_bench(kernel_call, quantiles=quantiles, warmup=1, rep=1)


def test_autotune_listener_fresh(device: str, fresh_knobs) -> None:
    """Test that the listener fires on fresh benchmark with all fields populated."""
    captured = []

    def listener(*, fn, key, best_config, configs_timings, duration, cache_hit):
        captured.append({
            "fn": fn,
            "key": key,
            "best_config": best_config,
            "configs_timings": configs_timings,
            "duration": duration,
            "cache_hit": cache_hit,
        })

    fresh_knobs.autotuning.listener = listener

    configs = [triton.Config({"BLOCK_SIZE": 32}), triton.Config({"BLOCK_SIZE": 128})]

    @triton.autotune(configs=configs, key=["N"], do_bench=do_bench)
    @triton.jit
    def _kernel(dst, src, N, BLOCK_SIZE: tl.constexpr):
        offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        x = tl.load(src + offsets, mask=offsets < N)
        tl.store(dst + offsets, x, mask=offsets < N)

    N = 1024
    src = torch.randn(N, device=device)
    dst = torch.empty(N, device=device)
    _intj_launcher(_kernel, (triton.cdiv(N, 32), ))(dst, src, N)

    # Listener fired exactly once
    assert len(captured) == 1
    result = captured[0]

    # Fresh benchmark, not cache hit
    assert not result["cache_hit"]

    # fn is unwrapped to JITFunction
    assert isinstance(result["fn"], JITFunction)

    # best_config is one of the configs tested
    assert result["best_config"] in result["configs_timings"]

    # All configs have timings
    assert len(result["configs_timings"]) == 2

    # Duration is set for fresh benchmark
    assert result["duration"] is not None and result["duration"] > 0


def test_autotune_listener_in_memory_cache_hit(device: str, fresh_knobs) -> None:
    """Test that the listener does NOT fire on in-memory cache hit."""
    captured = []

    def listener(*, fn, key, best_config, configs_timings, duration, cache_hit):
        captured.append(True)

    fresh_knobs.autotuning.listener = listener

    configs = [triton.Config({"BLOCK_SIZE": 32}), triton.Config({"BLOCK_SIZE": 128})]

    @triton.autotune(configs=configs, key=["N"], do_bench=do_bench)
    @triton.jit
    def _kernel(dst, src, N, BLOCK_SIZE: tl.constexpr):
        offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        x = tl.load(src + offsets, mask=offsets < N)
        tl.store(dst + offsets, x, mask=offsets < N)

    N = 1024
    src = torch.randn(N, device=device)
    dst = torch.empty(N, device=device)

    launch = _intj_launcher(_kernel, (triton.cdiv(N, 32), ))
    # First call: fresh benchmark
    launch(dst, src, N)
    assert len(captured) == 1

    # Second call with same key: in-memory cache hit, listener should NOT fire again
    launch(dst, src, N)
    assert len(captured) == 1


def test_autotune_listener_disk_cache_hit(device: str, fresh_knobs, fresh_triton_cache) -> None:
    """Test that the listener fires with cache_hit=True and duration=None on disk cache hit."""
    captured = []

    def listener(*, fn, key, best_config, configs_timings, duration, cache_hit):
        captured.append({
            "cache_hit": cache_hit,
            "duration": duration,
        })

    fresh_knobs.autotuning.listener = listener

    configs = [triton.Config({"BLOCK_SIZE": 32}), triton.Config({"BLOCK_SIZE": 128})]

    @triton.autotune(configs=configs, key=["N"], do_bench=do_bench, cache_results=True)
    @triton.jit
    def _kernel(dst, src, N, BLOCK_SIZE: tl.constexpr):
        offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        x = tl.load(src + offsets, mask=offsets < N)
        tl.store(dst + offsets, x, mask=offsets < N)

    N = 1024
    src = torch.randn(N, device=device)
    dst = torch.empty(N, device=device)

    # First call: fresh benchmark, populates disk cache
    _intj_launcher(_kernel, (triton.cdiv(N, 32), ))(dst, src, N)
    assert len(captured) == 1
    assert not captured[0]["cache_hit"]
    assert captured[0]["duration"] is not None and captured[0]["duration"] > 0

    # Clear in-memory cache so next call hits disk cache
    _kernel.cache.clear()

    # Second call: disk cache hit (a new launcher starts with an empty in-memory cache)
    _intj_launcher(_kernel, (triton.cdiv(N, 32), ))(dst, src, N)
    assert len(captured) == 2
    assert captured[1]["cache_hit"]
    assert captured[1]["duration"] is None


def test_autotune_listener_single_config(device: str, fresh_knobs) -> None:
    """Test that the listener does NOT fire for single-config autotune (no benchmarking)."""
    captured = []

    def listener(*, fn, key, best_config, configs_timings, duration, cache_hit):
        captured.append(True)

    fresh_knobs.autotuning.listener = listener

    @triton.autotune(configs=[triton.Config({"BLOCK_SIZE": 32})], key=["N"], do_bench=do_bench)
    @triton.jit
    def _kernel(dst, src, N, BLOCK_SIZE: tl.constexpr):
        offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        x = tl.load(src + offsets, mask=offsets < N)
        tl.store(dst + offsets, x, mask=offsets < N)

    N = 1024
    src = torch.randn(N, device=device)
    dst = torch.empty(N, device=device)
    _intj_launcher(_kernel, (triton.cdiv(N, 32), ))(dst, src, N)

    # Single config: no autotune benchmarking, listener should not fire
    assert len(captured) == 0
