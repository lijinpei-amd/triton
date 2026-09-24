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

Most converted call sites use `intj.compat.launch` or, for interpreter-marked
tests, `intj.compat.launch_or_interpret` so their existing argument binding and
options remain intact. The bridges use `grid_arg` for concrete grids and
`grid_py` for callable grids; direct `make_launcher` remains the lower-overhead
path. In compiled execution they raise for unsupported kernels. Compile warmup
uses Triton's indexed launcher so its test harness can intercept launches, and
`launch_or_interpret` also uses it when `TRITON_INTERPRET=1`.

Tests that inspect assembly or metadata pass `return_compiled=True` to receive
the cached Triton `CompiledKernel` after launch. This opt-in also works with
direct `make_launcher`; in interpreter mode, `launch_or_interpret` preserves the
original indexed launch's return value. The per-call exceptions below retain
Triton's launcher where a different unsupported constraint applies.

## Coverage and exceptions

The migration inventory contains **1,257 launch expressions**: 1,254 ordinary
Python calls and three inside embedded Python scripts. In compiled execution,
**446** ordinary sites use intj: 256 through `intj.compat.launch`, 175 through
`intj.compat.launch_or_interpret`, and 15 through direct `make_launcher`
handles. The remaining **811** are listed individually in
[intj_launch_exceptions.json](intj_launch_exceptions.json): 808 ordinary calls
and three embedded-script calls. Entries identify a source path, current line,
target, concrete reason, and detail. An exception applies to that launch, not
to every launch in its file.

The largest reasons are sanitizer profile scratch, GSan runtime behavior,
multi-CTA launches, Gluon layout constexpr arguments, and host descriptors.
Interpreter-marked tests now use `launch_or_interpret` unless another concrete
constraint applies. The full reason-count table follows.

<!-- BEGIN INTJ REASON COUNTS -->
<details>
<summary>All 59 reason counts (811 retained sites)</summary>

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
| `triton_autotuner_test` | 19 |
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
| `triton_autotune_listener_test` | 6 |
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
| `triton_runtime_diagnostic_test` | 3 |
| `unsupported_function_constexpr` | 3 |
| `custom_cpu_backend` | 2 |
| `triton_compilation_listener_test` | 2 |
| `triton_jit_cache_hook_test` | 2 |
| `captured_global_amdwmmalayout` | 1 |
| `custom_kernel_interface_return` | 1 |
| `descriptor_multicta_compilation_listener` | 1 |
| `exempt_autotune_heuristics` | 1 |
| `exempt_triton_reproducer` | 1 |
| `jit_global_and_compiler_visitor` | 1 |
| `plugin_stage_hook` | 1 |
| `tensor_descriptor_and_no_torch` | 1 |
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

The current source audit parsed 398 tracked Python files and matched all 808
ordinary indexed launches and three embedded-script launches to the 811
manifest entries. Run it after changing launch sites or the manifest:

```sh
python3 tools/audit_intj_launches.py
```

The audit checks path, line, and target coverage. It cannot prove that intj
supports a particular kernel; verify compiled and interpreter paths on the
applicable GPU. For example:

```sh
PYTHONPATH="$PWD/python:../workspace/intj/callable_grid" \
  python3 -m pytest python/test/unit/language/test_core.py -s --tb=short
```

A JIT launch that only needs the returned `CompiledKernel` can now use
`return_compiled=True`. The remaining exceptions retain Triton's launcher for
their listed constraints, including unsupported scratch, descriptor or
aggregate arguments, multi-CTA execution, runtime hooks, and compiler
diagnostics.
