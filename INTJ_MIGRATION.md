# intj launch-site migration

This branch routes supported Triton and Gluon launches through the sibling
`intj/callable_grid` launcher. Install this Triton checkout
as described in [README.md](README.md), then make the intj checkout importable
in the same Python environment:

```sh
python3 -m pip install -e ../workspace/intj/callable_grid
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

Most converted call sites use `intj.compat.launch` so their existing argument
binding and options remain intact. That bridge uses `grid_arg` for concrete
grids and `grid_py` for callable grids. It is a migration bridge, not the
low-overhead direct `make_launcher` call path. It never silently falls back to
Triton. The per-call exceptions below keep Triton's launcher where a test or
kernel requires behavior intj does not implement.

## Coverage and exceptions

The migration inventory contains **1,257 launch expressions**: 1,254 ordinary
Python AST calls and three inside embedded Python scripts. Of these, **231**
ordinary calls use intj (216 through `intj.compat.launch`, 15 through direct
`make_launcher` handles). The remaining **1,026** are listed individually in
[intj_launch_exceptions.json](intj_launch_exceptions.json): 1,023 ordinary calls
and three embedded-script calls. Entries identify a source path, current line,
target, concrete reason, and detail. An exception applies to that launch, not
to every launch in its file.

The largest concrete reasons are interpreter execution, sanitizer profile
scratch, GSan runtime behavior, multi-CTA launches, and Gluon layout constexpr
arguments. The full reason-count table follows so changes to the exception set
are reviewable.

<!-- BEGIN INTJ REASON COUNTS -->
<details>
<summary>All 64 reason counts (1,026 retained sites)</summary>

| Reason | Sites |
| --- | ---: |
| `exempt_interpreter` | 214 |
| `unsupported_profile_scratch` | 99 |
| `gsan_runtime` | 87 |
| `unsupported_num_ctas` | 78 |
| `unsupported_layout_constexpr` | 68 |
| `exempt_compiled_result` | 53 |
| `proton_launch_hook` | 45 |
| `unsupported_host_descriptor` | 43 |
| `unsupported_captured_global` | 38 |
| `proton_instrumentation` | 34 |
| `triton_autotuner_test` | 19 |
| `needs_compiled_kernel_asm` | 19 |
| `captured_global_xblock` | 18 |
| `triton_cache_test` | 17 |
| `exempt_interpreter_helper` | 16 |
| `unsupported_descriptor_argument` | 15 |
| `triton_global_cache_test` | 14 |
| `needs_compiled_kernel_result` | 11 |
| `triton_runtime_launch_test` | 9 |
| `unsupported_tuple_constexpr` | 8 |
| `unsupported_scratch_allocator` | 7 |
| `triton_async_cache_test` | 7 |
| `triton_cache_mutation_test` | 7 |
| `consan_instrumentation` | 7 |
| `mir_dump_swap_pipeline` | 6 |
| `unsupported_tuple_argument` | 6 |
| `triton_autotune_listener_test` | 6 |
| `unsupported_tuple_pointer_argument` | 6 |
| `exempt_compiled_kernel` | 5 |
| `tuple_constexpr` | 5 |
| `compiled_kernel_asm_and_stage_hook` | 4 |
| `triton_debug_runtime_test` | 4 |
| `triton_ir_dump_test` | 4 |
| `llvm_pass_plugin_output` | 3 |
| `triton_compiler_diagnostic_test` | 3 |
| `captured_global_specialization` | 3 |
| `precompiled_kernel_not_jit` | 3 |
| `custom_cpu_backend` | 2 |
| `exempt_triton_annotation_contract` | 2 |
| `exempt_interpreter_shared_helper` | 2 |
| `compiled_kernel_asm` | 2 |
| `triton_compilation_listener_test` | 2 |
| `triton_stage_hook_test` | 2 |
| `triton_runtime_hook_and_compiled_result` | 2 |
| `unsupported_tuple_and_layout_constexpr` | 2 |
| `unsupported_enum_constexpr` | 1 |
| `descriptor_multicta_compilation_listener` | 1 |
| `triton_launch_overhead_baseline` | 1 |
| `unsupported_annotation` | 1 |
| `unsupported_function_constexpr` | 1 |
| `exempt_autotune_heuristics` | 1 |
| `exempt_triton_reproducer` | 1 |
| `plugin_stage_hook` | 1 |
| `tensor_descriptor_and_no_torch` | 1 |
| `jit_global_and_compiler_visitor` | 1 |
| `custom_kernel_interface_return` | 1 |
| `compile_warmup_interception` | 1 |
| `triton_link_hook_test` | 1 |
| `unsupported_launch_pdl` | 1 |
| `needs_compiled_kernel_asm_and_layout` | 1 |
| `unsupported_descriptor_and_jit_class` | 1 |
| `uncached_compiler_knob` | 1 |
| `unsupported_tuple_layout_and_compile_error_test` | 1 |
| `captured_global_amdwmmalayout` | 1 |

</details>
<!-- END INTJ REASON COUNTS -->

## Validation

The launch audit and Python syntax checks were rerun for this correction.
Earlier GPU checks ran on a gfx942 host with Python 3.12.3 and Triton 3.8.0:

| Check | Result |
| --- | --- |
| Launch audit | 1,023 ordinary and three embedded launches match 1,026 exceptions |
| Python syntax | All 53 changed Python files compiled |
| AMD int8 small-block dot | Four parameter cases passed |
| Direct `grid_cpp` examples | Vector add matched Torch for 98,432 elements; dropout repeated the same seeded output |
| CDNA5 ConSan | 54 tests collected; execution requires CDNA5 |

The restored Hopper, Blackwell, CDNA5, CUDA profiler, and FpSan dot paths have not been GPU tested in this review.

Run the audit from this checkout after changing launch sites or the exception
manifest:

```sh
python3 tools/audit_intj_launches.py
```

The audit checks that each retained launch is documented with its current
path, line, and target and that no manifest entry is stale. It cannot prove
that intj supports a particular kernel; keep the reason detail tied to the
specific launch and verify converted kernels on the applicable GPU.

For Python-only verification, compile edited files and collect their test
modules. For GPU verification, use the branch's regular pytest commands with
both the Triton source and intj checkout importable, for example:

```sh
PYTHONPATH="$PWD/python:../workspace/intj/callable_grid" \
  python3 -m pytest python/test/unit/language/test_core.py -s --tb=short
```

CDNA5 ConSan subprocesses and other tests that assert Triton runtime behavior
remain on Triton; `intj` rejects their profile-scratch requirements and does
not provide the same diagnostics or `CompiledKernel` return value. CDNA5 and
NVIDIA execution need those GPUs and were not exercised on the gfx942 host.
