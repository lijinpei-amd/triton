# intj launch-site migration

This branch routes supported Triton and Gluon launches through the sibling
`intj` launcher (its `develop` branch, at or after `2a876ff`: `make_launcher`
used as a decorator builds lazily on the first call, takes per-call compile
options and knobs through `dynamic_options`, and keys `str`, `tl.dtype` and
JIT-function constexprs by value). Install this Triton
checkout as described in [README.md](README.md), then make the intj checkout
importable in the same Python environment:

```sh
python3 -m pip install -e ../intj
python3 -c 'import triton, intj; print(triton.__version__, intj.__file__)'
```

The relative path assumes both checkouts are under the same `development/`
directory. An installed intj package or a `PYTHONPATH` entry pointing at its
checkout works as well. Converted source files import intj directly; a missing
or unsupported intj installation raises an error rather than silently using
Triton's launcher. Triton and intj both declare Python 3.10 as their minimum;
intj declares `triton>=3.7`. This experimental branch does not add intj to
Triton's package dependencies.

## Launch choice

Use `grid_arg` when the caller already has one to three grid dimensions. Use an
annotated, capture-free `def` with `grid_cpp` for a grid computed solely from
launch-time scalar arguments. Use `grid_py` when the grid needs Python metadata
or a selected autotuning configuration. The direct examples are
[vector add](python/tutorials/01-vector-add.py),
[dropout](python/tutorials/04-low-memory-dropout.py), and
[softmax](python/tutorials/02-fused-softmax.py).

Decorate the kernel with `@make_launcher(grid_arg=..., options=...)` and call
it positionally as `(device, stream, *grid, *dynamic_options, *args)`. Tests
take the pair from `triton._internal_testing.device_stream()`. The launcher
builds on its first call, so decorating a module-level kernel needs no GPU at
import, and an option only one backend accepts (such as AMD's
`allow_flush_denorm`) is checked then too. Keep the JIT function and build the
handle beside it when something else needs the kernel object: another JIT
function calls it, the triton-to-gluon translator reads it, or a test inspects
the `Autotuner`. `make_launcher` still refuses `TRITON_INTERPRET=1` when it is
called, so a module-level decorator makes its module unimportable under the
interpreter; library modules that must import there build the handle on
first use (`functools.cache`), as `compaction()` in triton_kernels does.

What varies per call is an argument, not another launcher:
- compile options: name them in `dynamic_options` and pass their values right
  after the grid, as the layer-norm tutorial does with `num_warps`;
- Triton knobs a test flips between calls: name the knob path in
  `dynamic_options` and pass its live value. `test_fpsan.py` declares
  `knobs.compilation.instrumentation_mode` on every launcher; intj raises
  `ValueError` if the passed value is not the live one;
- `str`, `tl.dtype` and JIT-function constexprs: pass them positionally, like
  any other argument. The matrix-multiplication tutorial's `ACTIVATION` string
  is one.

`make_launcher` rejects `TensorWrapper` (`triton.reinterpret`) arguments. Where
the reinterpreted dtype is a torch dtype, pass `tensor.view(dtype)` instead, as
`test_fpsan.py` does.

`intj.compat.launch` refuses `@triton.autotune` and `@triton.heuristics`
wrappers, so decorated kernels always use `make_launcher`: each launcher owns
its tuner cache, the call omits tuned and heuristic values, and a `grid_cpp`
function reads tuned values, such as `_block_grid` in the
[autotuner tests](python/test/unit/runtime/test_autotuner.py) or `_matmul_grid`
in the [matrix multiplication tutorial](python/tutorials/03-matrix-multiplication.py).

A launch stays on `intj.compat.launch` (or, for interpreter-marked tests,
`intj.compat.launch_or_interpret`) when a direct launcher cannot serve it: it
passes a `TensorWrapper` from `to_triton` (unsigned integers) or
`triton.reinterpret` (fp8 types, some with no torch dtype, such as
`float8e4b15`); a compile-warmup test reaches it; or its test
relies on the bridge's CPU-pointer `ValueError`. Launches whose tests cannot
run on the gfx942 validation machine (CUDA-, gfx950- or gfx1250-only, proton,
plugin) also stay on the bridge until they can be verified.

The bridges use `grid_arg` for concrete grids and `grid_py` for callable
grids. In compiled execution they raise for unsupported kernels. Direct
`make_launcher` is the lower-overhead path: on gfx942 a one-program launch
takes about 19.5 us through the bridge and 6.3 us through a handle that reads
`device_stream()` per call (3.7 us with the device and stream hoisted).
Compile warmup uses Triton's indexed launcher so its test harness can
intercept launches, and `launch_or_interpret` also uses it when
`TRITON_INTERPRET=1`.

Tests that inspect assembly or metadata pass `return_compiled=True` to receive
the cached Triton `CompiledKernel` after launch. This opt-in also works with
direct `make_launcher`; in interpreter mode, `launch_or_interpret` preserves the
original indexed launch's return value. The per-call exceptions below retain
Triton's launcher where a different unsupported constraint applies.

## Coverage and exceptions

The migration inventory contains **1,253 launch expressions**: 1,250 ordinary
Python calls and three inside embedded Python scripts. In compiled execution,
**465** ordinary sites use intj: 150 through direct `make_launcher` handles
(23 of them on autotuned kernels), 140 through `intj.compat.launch`, and 175
through `intj.compat.launch_or_interpret`. Of the 140 bridge sites, 107 are
in tests that cannot run on the gfx942 validation machine, 20 are reached by
compile warmup (13 in tests, 7 in triton_kernels library code), 11 pass
`TensorWrapper` arguments and 2 expect the bridge's CPU-pointer `ValueError`.
The remaining **789** are listed
individually in [intj_launch_exceptions.json](intj_launch_exceptions.json):
786 ordinary calls and three embedded-script calls. (The inventory had 1,257
expressions until the autotuner tests' three keyword/positional launch pairs
each became one positional intj call, and test_triton_heuristic's became
another.) Entries identify a source path, current line, target, concrete reason, and detail. An exception applies to that launch, not
to every launch in its file.

The largest reasons are sanitizer profile scratch, GSan runtime behavior,
multi-CTA launches, Gluon layout constexpr arguments, and host descriptors.
Interpreter-marked tests now use `launch_or_interpret` unless another concrete
constraint applies. The full reason-count table follows.

<!-- BEGIN INTJ REASON COUNTS -->
<details>
<summary>All 58 reason counts (789 retained sites)</summary>

| Reason | Sites |
| --- | ---: |
| `unsupported_profile_scratch` | 99 |
| `gsan_runtime` | 87 |
| `unsupported_num_ctas` | 82 |
| `unsupported_layout_constexpr` | 75 |
| `unsupported_host_descriptor` | 56 |
| `unsupported_captured_global` | 53 |
| `proton_launch_hook` | 45 |
| `proton_instrumentation` | 34 |
| `unsupported_scratch_allocator` | 30 |
| `unsupported_tuple_constexpr` | 20 |
| `captured_global_xblock` | 18 |
| `triton_cache_test` | 17 |
| `unsupported_descriptor_argument` | 15 |
| `triton_global_cache_test` | 14 |
| `compile_warmup_interception` | 10 |
| `unsupported_tuple_argument` | 10 |
| `triton_runtime_launch_test` | 9 |
| `triton_cache_mutation_test` | 8 |
| `consan_instrumentation` | 7 |
| `triton_async_cache_test` | 7 |
| `mir_dump_swap_pipeline` | 6 |
| `triton_stage_hook_test` | 6 |
| `unsupported_tuple_and_layout_constexpr` | 6 |
| `unsupported_tuple_pointer_argument` | 6 |
| `exempt_compiled_kernel` | 5 |
| `tuple_constexpr` | 5 |
| `exempt_triton_annotation_contract` | 4 |
| `triton_compiler_diagnostic_test` | 4 |
| `triton_debug_runtime_test` | 4 |
| `triton_ir_dump_test` | 4 |
| `captured_global_specialization` | 3 |
| `llvm_pass_plugin_output` | 3 |
| `precompiled_kernel_not_jit` | 3 |
| `triton_autotuner_cache_state` | 3 |
| `triton_runtime_diagnostic_test` | 3 |
| `unsupported_function_constexpr` | 3 |
| `custom_cpu_backend` | 2 |
| `triton_compilation_listener_test` | 2 |
| `triton_jit_cache_hook_test` | 2 |
| `captured_global_amdwmmalayout` | 1 |
| `custom_kernel_interface_return` | 1 |
| `descriptor_multicta_compilation_listener` | 1 |
| `exempt_triton_reproducer` | 1 |
| `jit_global_and_compiler_visitor` | 1 |
| `plugin_stage_hook` | 1 |
| `tensor_descriptor_and_no_torch` | 1 |
| `triton_autotuner_keyword_prune_args` | 1 |
| `triton_launch_overhead_baseline` | 1 |
| `triton_link_hook_test` | 1 |
| `uncached_compiler_knob` | 1 |
| `unsupported_annotation` | 1 |
| `unsupported_cooperative_grid` | 1 |
| `unsupported_descriptor_and_jit_class` | 1 |
| `unsupported_enum_constexpr` | 1 |
| `unsupported_keyword_only_parameters` | 1 |
| `unsupported_launch_pdl` | 1 |
| `unsupported_non_scalar_constexpr` | 1 |
| `unsupported_tuple_layout_and_compile_error_test` | 1 |

</details>
<!-- END INTJ REASON COUNTS -->

## Validation

The current source audit parsed 398 tracked Python files and matched all 786
ordinary indexed launches and three embedded-script launches to the 789
manifest entries. Run it after changing launch sites or the manifest:

```sh
python3 tools/audit_intj_launches.py
```

The audit checks path, line, and target coverage. It cannot prove that intj
supports a particular kernel; verify compiled and interpreter paths on the
applicable GPU. For example:

```sh
PYTHONPATH="$PWD/python:../intj" \
  python3 -m pytest python/test/unit/language/test_core.py -s --tb=short
```

The autotune conversions ran on gfx942 with Triton 3.8.0 and intj `bb25722`:
`test_autotuner.py`, `test_autotune_listener.py` and `test_decorator.py` pass
(29 passed, 4 skipped), as do the matrix-multiplication, grouped-GEMM and Gluon
intro tutorials' correctness sections. The four skipped autotuner tests
(`test_override_ttir`, `test_override_ttgir`, `test_override_ptx`,
`test_exceed_tmem`) are CUDA-only and untested through intj. intj `bb25722`
added `.data_ptr()` support to lowered heuristics, so `test_triton_heuristic`
now also launches through intj.

The direct `make_launcher` conversions ran on the same setup. Every touched
test file passes in full, including `test_core.py` (15,027 passed, 2,226
skipped), Gluon `test_core.py`, `test_pipeliner.py`, `test_triton_to_gluon.py`,
the touched triton_kernels and AMD tests, and the Gluon intro tutorial.
Interpreter mode still collects and runs `test_tuple.py`, `test_random.py` and
`test_autotuner.py::test_kwargs`. Launches whose tests skip on gfx942 stay on
`intj.compat.launch` and are untested through a direct handle.

The dynamic-launcher migration ran on the same setup with intj `429d8e1`:
39 fpsan sites, 13 per-call-variant sites, 56 `functools.partial` stacks and
16 handles built beside their kernels moved to `@make_launcher(...)`. Every touched
test file passes in full: `test_core.py` (15,027 passed, 2,226 skipped),
`test_fpsan.py` (119 passed), Gluon `test_core.py`, `test_libdevice.py`,
`test_random.py`, `test_tuple.py`, `test_pipeliner.py`, the autotuner,
listener, driver and triton-to-gluon tests, the touched triton_kernels and AMD
tests, the fused-attention tests, and the vector-add, matmul, dropout,
layer-norm, grouped-GEMM and Gluon intro tutorials' correctness sections. The
two CUDA-only fpsan launches (`test_bf16_dot_sharding`,
`test_f32_loop_preserves_snan_payload`) are untested. Median host time per
launch, device and stream read per call as the tests do:

| site | before | after |
| --- | ---: | ---: |
| fpsan `_binop_kernel` (str `OP`, fpsan knob, reinterpreted pointers) | 28.6 us (bridge) | 7.9 us |
| `test_sum_dtype` (two `tl.dtype` constexprs) | 20.1 us (bridge) | 5.7 us |
| layer-norm forward (`num_warps` per call) | 9.4 us (`functools.cache` factory) | 8.3 us |

With the device and stream hoisted the three take 3.2, 3.0 and 3.7 us.

Interpreter mode collects and runs `test_tuple.py`, `test_random.py` and
`test_autotuner.py::test_kwargs` (118 passed). `cuda/test_libdevice_cuda.py`
and the fused-attention tutorial still fail to import there, as before this
migration: their module-level launchers call `make_launcher`, which refuses
`TRITON_INTERPRET=1`.

A JIT launch that only needs the returned `CompiledKernel` can now use
`return_compiled=True`. The remaining exceptions retain Triton's launcher for
their listed constraints, including unsupported scratch, descriptor or
aggregate arguments, multi-CTA execution, runtime hooks, and compiler
diagnostics.
