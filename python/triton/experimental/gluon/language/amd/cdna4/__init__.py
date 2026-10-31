from triton.runtime.jit import constexpr_function
from triton._C.libtriton.gluon_ir import (
    get_amd_mfma_scale_layout as _get_mfma_scale_layout,
    compute_amd_efficient_padded_shared_layout as _compute_efficient_padded_shared_layout,
)

from ... import _core as ttgl
from ..._core import builtin, int8, int32, uint8, _unwrap_if_constexpr
from ..._layouts import DotOperandLayout
from .._layouts import AMDMFMALayout
from .._ops import _load_shared_fp4_repacked, _mma_scaled, _scaled_upcast, scaled_downcast
from ..cdna3 import _buffer_atomic_rmw_impl, _check_cd_regclass, _set_cd_regclass
from ..cdna3 import *  # NOQA: F403
from ..cdna3 import __all__ as __cdna3_all
from . import async_copy
from . import buffer

__all__ = [
    *__cdna3_all,
    "async_copy",
    "buffer",
    "mfma_scaled",
    "mfma_scaled_packed",
    "scaled_upcast",
    "scaled_downcast",
    "load_shared_fp4_repacked",
    "get_mfma_scale_layout",
    "compute_efficient_padded_shared_layout",
]


@builtin
def mfma_scaled(a, a_scale, a_format, b, b_scale, b_format, acc, cd_regclass=None, _semantic=None):
    """
    AMD Scaled MFMA operation.

    .. code-block:: text

        c = a * a_scale @ b * b_scale + acc

    ``a`` and ``b`` use microscaling formats described in
    "OCP Microscaling Formats (MX) Specification":
    https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf.
    Currently supported only on CDNA4 hardware.

    Args:
        a (tensor): The operand A to be multiplied.
        a_scale (Optional[tensor]): Scale factor for operand A.
        a_format (str): Format of operand A. Available formats: ``e2m1``,
            ``e4m3``, ``e5m2``.
        b (tensor): The operand B to be multiplied.
        b_scale (Optional[tensor]): Scale factor for operand B.
        b_format (str): Format of operand B. Available formats: ``e2m1``,
            ``e4m3``, ``e5m2``.
        acc (tensor): Accumulator tensor.
        cd_regclass (str, optional): Experimental. Register class for the accumulator input (C)
            and result (D) of the scaled MFMA instructions, as for ``mfma``: ``"a"`` for AGPRs or
            ``"v"`` for VGPRs. ``None`` (default) leaves the choice to the compiler.
    """
    layout = acc.type.layout
    assert isinstance(layout, AMDMFMALayout), "Expected layout to be an instance of AMDMFMALayout"
    assert (isinstance(a.type.layout, DotOperandLayout) and a.type.layout.parent== layout), \
            "Expected lhs layout to be a DotOperandLayout with parent matching MFMA layout"
    assert (isinstance(b.type.layout, DotOperandLayout) and b.type.layout.parent == layout), \
            "Expected rhs layout to be a DotOperandLayout with parent matching MFMA layout"

    a_format = _unwrap_if_constexpr(a_format)
    b_format = _unwrap_if_constexpr(b_format)
    a_scale = _unwrap_if_constexpr(a_scale)
    b_scale = _unwrap_if_constexpr(b_scale)
    assert a_format in {"e2m1", "e4m3", "e5m2"}, f"Unsupported lhs_format: {a_format}"
    assert b_format in {"e2m1", "e4m3", "e5m2"}, f"Unsupported rhs_format: {b_format}"

    cd_regclass = _check_cd_regclass(cd_regclass, acc)
    ret = _mma_scaled(a, a_scale, a_format, b, b_scale, b_format, acc, get_mfma_scale_layout, _semantic)
    _set_cd_regclass(ret.handle, cd_regclass, _semantic)
    return ret


@builtin
def mfma_scaled_packed(a, a_scale, a_scale_sel, a_format, b, b_scale, b_scale_sel, b_format, acc, _semantic=None):
    """
    AMD scaled MFMA taking pre-packed scale operands.

    Same arithmetic as :func:`mfma_scaled`, but the scales arrive as ``int32``
    tensors in which the kernel has already assembled four E8M0 bytes per word,
    and the caller states which byte each matrix instruction must select. This
    hands the caller two things ``mfma_scaled`` decides on its own:

    * **No packing code.** ``mfma_scaled`` receives ``uint8`` scales and has to
      build the 32-bit operands itself, which costs a ``v_perm_b32`` per word.
      Here the loaded word goes straight to ``v_mfma_scale_*`` untouched, so a
      scale layout designed around ``ds_read_b32``/``buffer_load_dword`` keeps
      that property all the way down.
    * **The byte order.** ``mfma_scaled`` packs along K first and then along the
      non-K dimension. A pre-shuffled scale tensor generally does not follow
      that order, and ``a_scale_sel``/``b_scale_sel`` are how you say so.

    The selector lists are walked against the lowering's matrix-instruction
    order, which is row-major over ``(non_k, k)`` repetitions: instruction
    ``i = non_k * num_rep_k + k`` uses selector ``sel[i % len(sel)]`` of scale
    register ``i // len(sel)``. Registers are taken from the scale tensor in the
    per-thread order its layout produces, so ``len(sel)`` must divide
    ``num_rep_m * num_rep_k`` (respectively ``num_rep_n * num_rep_k``), and the
    tensor must hold at least ``num_rep_m * num_rep_k // len(sel)`` words per
    thread. None of that is checked at compile time -- a wrong selector list
    silently scales by the wrong exponent.

    Args:
        a (tensor): The operand A to be multiplied.
        a_scale (tensor): ``int32`` packed scale factors for operand A, or the
            ordinary ``uint8`` scales if ``a_scale_sel`` is None.
        a_scale_sel (Optional[List[int]]): Byte selectors, each in ``[0, 3]``,
            for A. None leaves this operand on the ``mfma_scaled`` path, which
            is what a scale layout that cannot be packed per lane needs.
        a_format (str): Format of operand A: ``e2m1``, ``e4m3`` or ``e5m2``.
        b (tensor): The operand B to be multiplied.
        b_scale (tensor): ``int32`` packed scale factors for operand B, or the
            ordinary ``uint8`` scales if ``b_scale_sel`` is None.
        b_scale_sel (Optional[List[int]]): Byte selectors, each in ``[0, 3]``, for B.
        b_format (str): Format of operand B: ``e2m1``, ``e4m3`` or ``e5m2``.
        acc (tensor): Accumulator tensor.
    """
    layout = acc.type.layout
    assert isinstance(layout, AMDMFMALayout), "Expected layout to be an instance of AMDMFMALayout"
    assert (isinstance(a.type.layout, DotOperandLayout) and a.type.layout.parent == layout), \
            "Expected lhs layout to be a DotOperandLayout with parent matching MFMA layout"
    assert (isinstance(b.type.layout, DotOperandLayout) and b.type.layout.parent == layout), \
            "Expected rhs layout to be a DotOperandLayout with parent matching MFMA layout"

    a_format = _unwrap_if_constexpr(a_format)
    b_format = _unwrap_if_constexpr(b_format)
    assert a_format in {"e2m1", "e4m3", "e5m2"}, f"Unsupported lhs_format: {a_format}"
    assert b_format in {"e2m1", "e4m3", "e5m2"}, f"Unsupported rhs_format: {b_format}"

    def _check_operand(name, sel, scale):
        sel = _unwrap_if_constexpr(sel)
        if sel is None:
            # Not pre-packed: this side falls back to mfma_scaled's own packing.
            assert scale is not None, f"{name} must not be None"
            return []
        sel = [_unwrap_if_constexpr(s) for s in sel]
        assert len(sel) > 0, f"{name}_sel must not be empty"
        assert all(isinstance(s, int) and 0 <= s <= 3 for s in sel), \
            f"{name}_sel entries must be ints in [0, 3], got {sel}"
        assert scale is not None and scale.dtype == int32, \
            f"{name} must be an int32 tensor of pre-packed scales, got {None if scale is None else scale.dtype}"
        return sel

    a_scale_sel = _check_operand("a_scale", a_scale_sel, a_scale)
    b_scale_sel = _check_operand("b_scale", b_scale_sel, b_scale)
    assert a_scale_sel or b_scale_sel, "at least one of a_scale_sel / b_scale_sel must be given"

    handle = _semantic.builder.create_dot_scaled_packed(a.handle, a_scale.handle,
                                                        _semantic._str_to_fp_type(a_format), b.handle, b_scale.handle,
                                                        _semantic._str_to_fp_type(b_format), acc.handle, a_scale_sel,
                                                        b_scale_sel)
    return ttgl.tensor(handle, acc.type)


@builtin
def scaled_upcast(src, scale, elem_type, axis=None, _semantic=None):
    """
    Upcast an fp4 or fp8 tensor and fold raw E8M0 scale payload into the
    CDNA4 scaled-upcast op.

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


@builtin
def load_shared_fp4_repacked(mem_desc, layout, _semantic=None):
    """
    Load M/N-packed fp4 bytes from shared memory into a K-packed MFMA dot operand layout.

    The source shared memory descriptor must contain ``int8`` or ``uint8`` packed fp4
    values. The destination shape is inferred from the source shape and dot
    operand index in ``layout``.
    """
    layout = _unwrap_if_constexpr(layout)
    return _load_shared_fp4_repacked(mem_desc, layout, _semantic, parent_type=AMDMFMALayout)


def _get_mfma_scale_layout_impl(*args, **kwargs):
    return _get_mfma_scale_layout(*args, **kwargs)


_get_mfma_scale_layout_impl.__triton_builtin__ = True


@constexpr_function
def get_mfma_scale_layout(dot_operand_layout, shape, scale_factor=32):
    """Get the scale layout for MFMA scaled operands.

    Args:
        dot_operand_layout (DotOperandLayout): The dot operand layout.
        shape (List[int]): The shape of the scale tensor.
        scale_factor (int): The scale factor.
    Returns:
        DistributedLinearLayout: The scale layout.
    """
    assert scale_factor == 32, "Only scale factor 32 is supported for CDNA4 Scaled MFMA"
    op_idx = dot_operand_layout.operand_index
    parent = dot_operand_layout.parent
    assert isinstance(parent, AMDMFMALayout), "Expected parent to be an instance of AMDMFMALayout"
    mdim = parent.instr_shape[0]
    tiles_per_warp = parent.tiles_per_warp
    warps_per_cta = parent.warps_per_cta
    return _get_mfma_scale_layout_impl(op_idx, shape, mdim, tiles_per_warp, warps_per_cta)


def _compute_efficient_padded_shared_layout_impl(*args, **kwargs):
    return _compute_efficient_padded_shared_layout(*args, **kwargs)


_compute_efficient_padded_shared_layout_impl.__triton_builtin__ = True


@constexpr_function
def compute_efficient_padded_shared_layout(dot_operand_layout, shape, dtype, is_k_contig=True):
    """Compute an efficient padded shared layout that avoids bank conflicts.

    Args:
        dot_operand_layout (DotOperandLayout): The layout for the dot operand
            that will be copied to shared memory with padding. Must have an
            AMDMFMALayout v4 (CDNA4) parent.
        shape (List[int]): Shared memory tile shape for the dot operand —
            ``[BM, BK]`` for operand A or ``[BK, BN]`` for operand B.
        dtype (dtype): Element type of the tensor that will live in this
            shared memory allocation (e.g. ``ttgl.float16``, ``ttgl.float8e4nv``).
            Only types with bitwidth in {4, 8, 16} are supported. For packed
            fp4 (two values per byte), pass ``ttgl.uint8`` — at the LDS level
            4-bit shares the 8-bit padding pattern.
        is_k_contig (bool): K is the contiguous dim in shared memory.
    Returns:
        The layout, or ``None`` if the input falls outside the supported set.
        Common reasons for ``None``: ``k_width`` not in {4, 8, 16}; element
        bitwidth not in {4, 8, 16}; or an MFMA instruction shape or kWidth
        combination not handled by the underlying algorithm.
    """
    parent = dot_operand_layout.parent
    assert isinstance(parent, AMDMFMALayout), \
        "Expected dot operand's parent to be an AMDMFMALayout"
    assert parent.version == 4, \
        "compute_efficient_padded_shared_layout only supports MFMA v4 (CDNA4)"
    return _compute_efficient_padded_shared_layout_impl(
        dot_operand_layout.operand_index,
        dot_operand_layout.k_width,
        parent.version,
        list(parent.warps_per_cta),
        list(parent.instr_shape),
        parent.transposed,
        list(parent.tiles_per_warp),
        parent.element_bitwidth,
        list(parent.cga_layout),
        list(shape),
        dtype.primitive_bitwidth,
        is_k_contig,
    )


"""
buffer_atomic_rmw of cnda4 shares the same signature and functionalities as cdna3.buffer_atomic_rmw.
The cdna4 version additionally supports `fadd` with `bf16`.
"""


@builtin
def buffer_atomic_max(ptr, offsets, value, mask=None, sem=None, scope=None, _semantic=None):
    return _buffer_atomic_rmw_impl('max', ptr, offsets, value, "cdna4", mask=mask, sem=sem, scope=scope,
                                   _semantic=_semantic)


@builtin
def buffer_atomic_min(ptr, offsets, value, mask=None, sem=None, scope=None, _semantic=None):

    return _buffer_atomic_rmw_impl('min', ptr, offsets, value, "cdna4", mask=mask, sem=sem, scope=scope,
                                   _semantic=_semantic)


@builtin
def buffer_atomic_add(ptr, offsets, value, mask=None, sem=None, scope=None, _semantic=None):

    return _buffer_atomic_rmw_impl('add', ptr, offsets, value, "cdna4", mask=mask, sem=sem, scope=scope,
                                   _semantic=_semantic)


@builtin
def buffer_atomic_and(ptr, offsets, value, mask=None, sem=None, scope=None, _semantic=None):

    return _buffer_atomic_rmw_impl('and', ptr, offsets, value, "cdna4", mask=mask, sem=sem, scope=scope,
                                   _semantic=_semantic)


@builtin
def buffer_atomic_or(ptr, offsets, value, mask=None, sem=None, scope=None, _semantic=None):

    return _buffer_atomic_rmw_impl('or', ptr, offsets, value, "cdna4", mask=mask, sem=sem, scope=scope,
                                   _semantic=_semantic)


@builtin
def buffer_atomic_xor(ptr, offsets, value, mask=None, sem=None, scope=None, _semantic=None):

    return _buffer_atomic_rmw_impl('xor', ptr, offsets, value, "cdna4", mask=mask, sem=sem, scope=scope,
                                   _semantic=_semantic)


@builtin
def buffer_atomic_xchg(ptr, offsets, value, mask=None, sem=None, scope=None, _semantic=None):

    return _buffer_atomic_rmw_impl('xchg', ptr, offsets, value, "cdna4", mask=mask, sem=sem, scope=scope,
                                   _semantic=_semantic)
