from __future__ import annotations
from typing import TYPE_CHECKING

from triton import knobs
from triton.experimental.gluon.language import _core as ttgl
from triton._C.libtriton import ir
from ..._core import builtin, int8, uint8, uint16, bfloat16, _unwrap_if_constexpr
from .._layouts import AMDMFMALayout
from .._ops import _scaled_upcast, get_scaled_upcast_fp4_scale_layout
from ..._semantic import _check
from ._cache_policy import CachePolicy

if TYPE_CHECKING:
    from ..._semantic import GluonSemantic

# `buffer` stays out of __all__: cdna4 star-imports this module and defines
# its own `buffer` submodule.
from . import buffer

__all__ = [
    "CachePolicy", "buffer_atomic_add", "buffer_atomic_and", "buffer_atomic_min", "buffer_atomic_max",
    "buffer_atomic_or", "buffer_atomic_xor", "buffer_atomic_xor", "buffer_load", "buffer_store", "mfma",
    "scaled_upcast", "get_scaled_upcast_fp4_scale_layout", "sched_barrier", "sched_group_barrier", "iglp_opt", "wave_id", "bare_barrier",
    "cond_barrier", "setprio"
]

_atomic_op_str_to_op = {
    "smax": ir.ATOMIC_OP.MAX, "smin": ir.ATOMIC_OP.MIN, "umax": ir.ATOMIC_OP.UMAX, "umin": ir.ATOMIC_OP.UMIN, "fadd":
    ir.ATOMIC_OP.FADD, "iadd": ir.ATOMIC_OP.ADD, "and": ir.ATOMIC_OP.AND, "or": ir.ATOMIC_OP.OR, "xor":
    ir.ATOMIC_OP.XOR, "xchg": ir.ATOMIC_OP.XCHG
}


def _verify_buffer_ops(ptr, offsets, mask=None, other=None):
    assert ptr.type.is_ptr(), "ptr must be a scalar pointer type"

    assert isinstance(offsets.type, ttgl.distributed_type), "expected offsets type to be a distributed_type"
    assert offsets.dtype.is_int32() or offsets.dtype.is_uint32(), "offsets element type must be int32 or uint32"

    mask = _unwrap_if_constexpr(mask)
    other = _unwrap_if_constexpr(other)
    if other is not None:
        assert mask is not None, "when other is not None, mask should not be None"


def _verify_element_type_and_dispatch_op(op, elem_type, arch):
    supported_types = [
        ttgl.float16, ttgl.float32, ttgl.bfloat16, ttgl.float64, ttgl.int32, ttgl.int64, ttgl.uint32, ttgl.uint64
    ]
    assert elem_type in supported_types, f"{elem_type} is not supported in buffer atomic on {arch}."

    if op in ['and', 'or', 'xor', 'xchg']:
        assert elem_type in [ttgl.int32, ttgl.int64], f"{op} with {elem_type} is not supported on CDNA3 or CDNA4"
        return _atomic_op_str_to_op[_unwrap_if_constexpr(op)]

    if op in ['max', 'min']:
        if elem_type in [ttgl.int32, ttgl.int64, ttgl.float64]:
            op = 's' + op
            return _atomic_op_str_to_op[_unwrap_if_constexpr(op)]
        elif elem_type in [ttgl.uint32, ttgl.uint64]:
            op = 'u' + op
            return _atomic_op_str_to_op[_unwrap_if_constexpr(op)]
        else:
            raise ValueError(f"{op} with {elem_type} is not supported on CDNA3 and CDNA4")

    if op == 'add':
        if elem_type in [ttgl.uint32, ttgl.uint64]:
            op = 'i' + op
            return _atomic_op_str_to_op[_unwrap_if_constexpr(op)]
        elif elem_type in [ttgl.float16, ttgl.float32, ttgl.float64]:
            op = 'f' + op
            return _atomic_op_str_to_op[_unwrap_if_constexpr(op)]
        elif elem_type is ttgl.bfloat16:
            assert arch == "cdna4", "Buffer atomic fadd with bf16 is only supported on CDNA4 for now."
            op = 'f' + op
            return _atomic_op_str_to_op[_unwrap_if_constexpr(op)]
        else:
            raise ValueError(f"{op} with {elem_type} is not supported on CDNA3 and CDNA4")

    raise ValueError(f"Unknown {op} on CDNA3 or CDNA4")


def _buffer_atomic_rmw_impl(op, ptr, offsets, value, arch, mask, sem, scope, _semantic):
    _verify_buffer_ops(ptr, offsets, mask)

    op = _verify_element_type_and_dispatch_op(op, ptr.type.scalar.element_ty, arch)

    mask = _unwrap_if_constexpr(mask)
    if mask is not None:
        mask = _semantic.to_tensor(mask)
        mask = _semantic.cast(mask, ttgl.int1)
        _, mask = _semantic.broadcast_impl_value(offsets, mask)
    mask = mask.handle if mask is not None else ir.value()

    value = _unwrap_if_constexpr(value)
    value = _semantic.to_tensor(value)
    _, value = _semantic.broadcast_impl_value(offsets, value)

    sem = _semantic._str_to_sem(sem)
    scope = _semantic._str_to_scope(scope)
    return _semantic.tensor(
        _semantic.builder.create_buffer_atomic_rmw(op, ptr.handle, offsets.handle, value.handle, sem, scope, mask),
        value.type)


@builtin
def buffer_load(ptr, offsets, mask=None, other=None, cache=None, contiguity=1, soffset=0,
                _semantic=None):
    """
    Deprecated. Use :func:`buffer.load <triton.experimental.gluon.language.amd.cdna3.buffer.load>` with a descriptor from
    :func:`buffer.make_buffer_descriptor <triton.experimental.gluon.language.amd.cdna3.buffer.make_buffer_descriptor>` instead.

    AMD buffer load from global memory via a scalar base pointer and a tensor of
    offsets instead of a tensor of pointers. This operation will load data
    directly into registers.

    Args:
        ptr (pointer to scalar): Global memory scalar base pointer to load from.
        offsets (tensor): Offsets tensor for the load operation.
        mask (tensor, optional): Mask tensor for predicated loads. Defaults to None.
        other (tensor or scalar, optional): Tensor or scalar providing default values for masked elements. Defaults to None.
        cache (str, optional): Cache modifier specifier. Defaults to ``None``.
        contiguity (int, optional): Number of elements each lane loads from consecutive
            addresses. Only a lower bound is inferred from the offsets tensor, and that
            inference is per-axis -- so when a lane's elements are contiguous in memory
            but span two axes of the offsets tensor (an MFMA scale fragment, say), it
            sees 1 and emits one scalar load per element. Declaring the real value lets
            the AMD lowering widen the access. It is not checked: an overstated value
            produces wrong addresses. Defaults to 1.
        soffset (int or scalar, optional): Uniform 32-bit byte offset added by the
            hardware to every lane's address, lowered to the buffer load's ``soffset``
            (SGPR) field. Unlike ``offsets`` it costs no VGPRs and no per-lane
            arithmetic, so it suits a base that is uniform across the block. Defaults
            to 0, which leaves the emitted instruction unchanged.
    """
    _verify_buffer_ops(ptr, offsets, mask, other)

    mask = _unwrap_if_constexpr(mask)
    if mask is not None:
        offsets, mask = _semantic.broadcast_impl_value(offsets, mask)

    other = _unwrap_if_constexpr(other)
    if other is not None:
        other = _semantic.to_tensor(other)
        other = _semantic.cast(other, ptr.dtype.element_ty)
        offsets, other = _semantic.broadcast_impl_value(offsets, other)

    other = other.handle if other is not None else ir.value()
    mask = mask.handle if mask is not None else ir.value()
    cache = _unwrap_if_constexpr(cache)
    cache_modifier = _semantic._str_to_load_cache_modifier(cache) if cache is not None else ir.CACHE_MODIFIER.NONE

    # soffset is optional in the op: a literal 0 stays absent, so existing callers
    # produce byte-identical IR and the same instruction as before. It is given in
    # bytes here; the op counts elements of the loaded type.
    soffset = _unwrap_if_constexpr(soffset)
    elem_bytes = ptr.type.scalar.element_ty.primitive_bitwidth // 8
    if isinstance(soffset, int) and soffset == 0:
        soffset = ir.value()
    elif isinstance(soffset, int):
        _check(soffset % elem_bytes == 0,
               lambda: f"soffset {soffset} is not a multiple of the {elem_bytes}-byte element")
        soffset = _semantic.cast(_semantic.to_tensor(soffset // elem_bytes), ttgl.int32).handle
    else:
        soffset = _semantic.to_tensor(soffset)
        _check(not soffset.type.is_block(), lambda: "expected soffset to be a scalar")
        soffset = _semantic.cast(soffset, ttgl.int32)
        if elem_bytes > 1:
            soffset = _semantic.floordiv(soffset, _semantic.to_tensor(elem_bytes))
        soffset = soffset.handle

    ret_ty = offsets.type.with_element_ty(ptr.type.scalar.element_ty)
    builder = _semantic.builder
    handle = builder.create_buffer_load(ret_ty.to_ir(builder), ptr.handle, offsets.handle, mask, other,
                                        cache_modifier, _unwrap_if_constexpr(contiguity), soffset)
    return ttgl.tensor(handle, ret_ty)


@builtin
def buffer_store(stored_value, ptr, offsets, mask=None, cache=None, _semantic: GluonSemantic = None):
    """
    Deprecated. Use :func:`buffer.store <triton.experimental.gluon.language.amd.cdna3.buffer.store>` with a descriptor from
    :func:`buffer.make_buffer_descriptor <triton.experimental.gluon.language.amd.cdna3.buffer.make_buffer_descriptor>` instead.

    AMD buffer store a tensor directly to global memory via a scalar base pointer and a tensor of
    offsets instead of a tensor of pointers.

    Args:
        stored_value (tensor to be stored): The tensor to be stored to global memory.
        ptr (pointer to scalar): Global memory scalar base pointer to store to.
        offsets (tensor): Offsets tensor for the store operation.
        mask (tensor, optional): Mask tensor for predicated store. Defaults to None.
        cache (str, optional): Cache modifier specifier. Defaults to ``None``.
    """
    _verify_buffer_ops(ptr, offsets, mask)

    offsets_shape = offsets.shape
    mask = _unwrap_if_constexpr(mask)
    if mask is None:
        offsets, stored_value = _semantic.broadcast_tensors(offsets, stored_value)
    else:
        offsets, stored_value, mask = _semantic.broadcast_tensors(offsets, stored_value, mask)
    if offsets_shape != offsets.shape:
        raise ValueError(f"Expected `offsets` argument to have shape {offsets.shape} but got {offsets_shape}")

    mask = mask.handle if mask is not None else ir.value()
    cache = _unwrap_if_constexpr(cache)
    cache_modifier = _semantic._str_to_store_cache_modifier(cache) if cache is not None else ir.CACHE_MODIFIER.NONE

    _semantic.builder.create_buffer_store(stored_value.handle, ptr.handle, offsets.handle, mask, cache_modifier)


# Attribute that carries `cd_regclass` on the dot op; read by the MFMA lowering.
_CD_REGCLASS_ATTR = "amdg.cd_regclass"


def _check_cd_regclass(cd_regclass, acc):
    """Validate `cd_regclass` ("a" = AGPRs, "v" = VGPRs, None) for an MFMA on `acc`, and return it unwrapped."""
    cd_regclass = _unwrap_if_constexpr(cd_regclass)
    if cd_regclass is not None:
        _check(cd_regclass in ("a", "v"), lambda: f"cd_regclass must be None, 'a' or 'v', got {cd_regclass!r}")
        _check(isinstance(acc.type.layout, AMDMFMALayout),
               lambda: f"cd_regclass requires an accumulator with AMDMFMALayout, got {acc.type.layout}")
    return cd_regclass


def _set_cd_regclass(handle, cd_regclass, semantic):
    """Record a checked `cd_regclass` on the dot op that defines `handle`."""
    if cd_regclass is not None:
        handle.set_attr(_CD_REGCLASS_ATTR, semantic.builder.get_string_attr(cd_regclass))


@builtin
def mfma(a, b, acc, cd_regclass=None, _semantic: GluonSemantic = None):
    """
    Computes matrix multiplication ``a * b + acc`` using AMD native matrix core units.

    Args:
        a (tensor): The first operand of mfma.
        b (tensor): The second operand of mfma.
        acc (tensor): The accumulator tensor.
        cd_regclass (str, optional): Experimental. Register class for the accumulator input (C)
            and result (D) of the MFMA instructions: ``"a"`` for AGPRs or ``"v"`` for VGPRs.
            Requires an ``AMDMFMALayout`` accumulator. ``None`` (default) leaves the choice to
            the compiler.
    """
    assert acc is not None, "acc is required"
    ret_type = acc.type
    acc = ttgl._unwrap_if_constexpr(acc)

    cd_regclass = _check_cd_regclass(cd_regclass, acc)
    handle = _semantic.dot(a, b, acc, input_precision=knobs.language.fp32_default, max_num_imprecise_acc=None,
                           out_dtype=acc.dtype).handle
    _set_cd_regclass(handle, cd_regclass, _semantic)
    return ttgl.tensor(handle, ret_type)


@builtin
def scaled_upcast(src, scale, elem_type, axis=None, _semantic=None):
    """
    Upcast an fp4 or fp8 tensor and fold raw E8M0 scale payload into the
    CDNA3 scaled-upcast op.

    CDNA3 lowers this through the software-emulated scaled-upcast path; it
    does not use native hardware scaled-upcast instructions.

    The ``scale`` tensor must use raw E8M0 payload in ``int8`` or ``uint8``, and must
    already have the expanded output shape and scaled-upcast result layout.
    For fp4 inputs, that is the canonical unpacked layout implied by ``src``
    and ``axis``. ``elem_type`` must be ``fp16`` or ``bf16``.
    """
    axis = _unwrap_if_constexpr(axis)
    elem_type = _unwrap_if_constexpr(elem_type)
    assert scale.dtype in (int8, uint8), \
        f"Expected scale to use raw E8M0 payload in int8/uint8 but got {scale.dtype}"
    return _scaled_upcast(src, scale, elem_type, axis, _semantic)


"""
AMD Buffer Atomic RMW operations.
The supported operations are max, min, add, and, or, xor, xchg.
Similar to normal atomic ops: it loads data at ptr plus offsets, do `op` with `value`, and store result to `ptr` plus `offsets` with
the specified memory semantics and scope.

Buffer atomics access global memory via a scalar base pointer and a tensor of offsets instead of a tensor of pointers.
Similar to other buffer ops, the `mask` is a boolean vector that determines if a given element should be processed with
the atomic RMW op. Elements with `mask[i] == 0` are dropped (i.e., the atomic is not executed).

Buffer Atomic RMW ops return the pre-op value in the global memory.

Args:
    ptr (pointer to scalar): Global memory scalar base pointer to load from.
    offsets (tensor): Offsets tensor for the load operation.
    value (tensor): Another operand of `op`.
    mask (tensor, optional): Mask tensor for predicated loads. Defaults to None.
    sem (str, optional): Memory Semantic Descriptor. Default is None which means acq_rel memory semantic.
    scope (str, optional): Memory Sync Scope for atomic accesses. Default is None and it will be mapped to `gpu`, which is called `agent` for AMDGPU. Please ref https://llvm.org/docs/AMDGPUUsage.html#memory-model-gfx942 for details.
"""


@builtin
def buffer_atomic_max(ptr, offsets, value, mask=None, sem=None, scope=None, _semantic=None):
    return _buffer_atomic_rmw_impl('max', ptr, offsets, value, "cdna3", mask=mask, sem=sem, scope=scope,
                                   _semantic=_semantic)


@builtin
def buffer_atomic_min(ptr, offsets, value, mask=None, sem=None, scope=None, _semantic=None):

    return _buffer_atomic_rmw_impl('min', ptr, offsets, value, "cdna3", mask=mask, sem=sem, scope=scope,
                                   _semantic=_semantic)


@builtin
def buffer_atomic_add(ptr, offsets, value, mask=None, sem=None, scope=None, _semantic=None):

    return _buffer_atomic_rmw_impl('add', ptr, offsets, value, "cdna3", mask=mask, sem=sem, scope=scope,
                                   _semantic=_semantic)


@builtin
def buffer_atomic_and(ptr, offsets, value, mask=None, sem=None, scope=None, _semantic=None):

    return _buffer_atomic_rmw_impl('and', ptr, offsets, value, "cdna3", mask=mask, sem=sem, scope=scope,
                                   _semantic=_semantic)


@builtin
def buffer_atomic_or(ptr, offsets, value, mask=None, sem=None, scope=None, _semantic=None):

    return _buffer_atomic_rmw_impl('or', ptr, offsets, value, "cdna3", mask=mask, sem=sem, scope=scope,
                                   _semantic=_semantic)


@builtin
def buffer_atomic_xor(ptr, offsets, value, mask=None, sem=None, scope=None, _semantic=None):

    return _buffer_atomic_rmw_impl('xor', ptr, offsets, value, "cdna3", mask=mask, sem=sem, scope=scope,
                                   _semantic=_semantic)


@builtin
def buffer_atomic_xchg(ptr, offsets, value, mask=None, sem=None, scope=None, _semantic=None):

    return _buffer_atomic_rmw_impl('xchg', ptr, offsets, value, "cdna3", mask=mask, sem=sem, scope=scope,
                                   _semantic=_semantic)


@builtin
def sched_barrier(mask=0, _semantic=None):
    """
    Fence the backend instruction scheduler.

    Lowers to ``llvm.amdgcn.sched.barrier``. Instructions may not be moved across
    it, so a hand-written interleave -- say a group of ``mfma`` followed by the
    loads that feed the next group -- survives into the final schedule instead of
    being re-clustered. Without one the backend is free to sink every memory
    operation past the whole MFMA run, which is usually what it does.

    .. warning::
        Known to miscompile when placed among ``buffer_load_to_shared`` copies: it
        appears to disturb the commit/wait-group accounting, and the damage shows up
        only at some pipeline depths. Prefer expressing the order in the source.

    Args:
        mask (int): ROCDL ``SchedGroupMask``. 0, the default, lets nothing cross.
            Bits allow individual classes through: 1 ALU, 2 VALU, 4 SALU, 8 MFMA,
            16 VMEM, 32 VMEM read, 64 VMEM write, 128 DS, 256 DS read,
            512 DS write, 1024 transcendental.
    """
    mask = _unwrap_if_constexpr(mask)
    assert isinstance(mask, int) and 0 <= mask < (1 << 11), f"invalid sched mask: {mask}"
    _semantic.builder.create_amd_sched_barrier(mask)


@builtin
def sched_group_barrier(mask, size, sync_id=0, _semantic=None):
    """
    Ask the backend scheduler to form one group of instructions.

    Lowers to ``llvm.amdgcn.sched.group.barrier``. Unlike :func:`sched_barrier`,
    which only forbids motion, this one *builds* a schedule: a sequence of these
    is a recipe the scheduler tries to satisfy in order, so alternating an MFMA
    group with a memory group interleaves them even though the source emits one
    large dot followed by all the loads.

    Args:
        mask (int): instruction class, as in :func:`sched_barrier` (8 MFMA,
            32 VMEM read, 256 DS read, ...).
        size (int): how many instructions of that class belong to this group.
        sync_id (int): groups sharing an id are ordered against each other.
    """
    mask = _unwrap_if_constexpr(mask)
    size = _unwrap_if_constexpr(size)
    sync_id = _unwrap_if_constexpr(sync_id)
    assert isinstance(mask, int) and 0 <= mask < (1 << 11), f"invalid sched mask: {mask}"
    _semantic.builder.create_amd_sched_group_barrier(mask, size, sync_id)


@builtin
def iglp_opt(variant=0, _semantic=None):
    """
    Enable one of the backend's canned instruction-interleaving strategies.

    Lowers to ``llvm.amdgcn.iglp.opt``. Variant 0 is the generic small-GEMM
    interleave, variant 1 the one written for a single-wave-per-SIMD GEMM whose
    MFMAs and global loads should ping-pong. Cheaper to try than a hand-written
    :func:`sched_group_barrier` recipe, and mutually exclusive with one.
    """
    variant = _unwrap_if_constexpr(variant)
    assert isinstance(variant, int), f"invalid iglp variant: {variant}"
    _semantic.builder.create_amd_iglp_opt(variant)

@builtin
def wave_id(_semantic=None):
    """
    Index of the current wavefront within the workgroup.

    Computed as ``tid.x / warp_size`` -- gfx9 has no wave-id register, and this is
    the same expression the warp pipeliner uses to split its ping-pong groups.

    Prefer expressing per-wave work in a *layout* (``warps_per_cta`` /
    ``warp_bases``) where you can: a layout is resolved at compile time and keeps
    the value analysable, whereas this returns an opaque runtime scalar that
    nothing layout-aware can consume.

    Returns:
        tensor: an ``int32`` scalar, uniform across the wave.
    """
    handle = _semantic.builder.create_amd_wave_id()
    return ttgl.tensor(handle, ttgl.int32)


@builtin
def bare_barrier(_semantic=None):
    """
    Execution-only workgroup barrier: ``s_barrier`` with no memory fence.

    ``gl.barrier()`` lowers to ``ttg.barrier{AddrSpace::Local}``, which the AMD
    backend expands to ``fence release / s_barrier / fence acquire``; the release
    fence over LDS makes ``SIMemoryLegalizer`` emit ``s_waitcnt lgkmcnt(0)``. This
    emits the rendezvous alone.

    .. warning::
        No memory ordering whatsoever. Prior LDS writes are **not** guaranteed
        visible to other waves after it. Only correct where the LDS ordering is
        already established some other way -- e.g. an ``async_copy.wait_group``
        that covers the writes in question. Use ``gl.barrier()`` unless you can
        point at what provides the ordering.
    """
    _semantic.builder.create_amd_bare_barrier()


@builtin
def cond_barrier(pred, _semantic=None):
    """
    Workgroup barrier executed only by the waves for which ``pred`` is true.

    The waves with ``pred`` false skip it entirely. Issued once before a loop it
    phase-shifts one wave group by half an iteration, which is what makes two
    groups ping-pong: thereafter a plain barrier inside the loop alternates them.

    .. warning::
        Deadlocks if the true-set is not consistent across the workgroup for the
        whole barrier's lifetime -- ``pred`` must be wave-uniform and loop-invariant.

    Args:
        pred (tensor): wave-uniform ``int1`` predicate.
    """
    _semantic.builder.create_amd_cond_barrier(pred.handle)


@builtin
def setprio(prio, _semantic=None):
    """
    Set the wave's scheduling priority (``s_setprio``).

    Raising priority around an MFMA run keeps the other wave group from stealing
    issue slots mid-cluster, which is what the warp pipeliner does at its stage
    boundaries.

    Args:
        prio (int): 0..3.
    """
    _semantic.builder.create_amd_setprio(_unwrap_if_constexpr(prio))
