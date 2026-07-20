import math

from triton import knobs
from triton.experimental.gluon.language import _core as ttgl
from triton.experimental.gluon.language._semantic import _check

from .._core import _unwrap_if_constexpr
from .._layouts import DotOperandLayout
from ._layouts import AMDWMMALayout


def _wrap_scaled_upcast_result(handle, elem_type, semantic):
    shape = semantic.builder.get_shape_from_tensor(handle)
    layout = semantic.builder.get_gluon_layout_from_tensor(handle)
    ret_ty = ttgl.distributed_type(elem_type, shape, layout)
    return ttgl.tensor(handle, ret_ty)


def _verify_wmma(version, a, b, acc):
    _check(acc is not None, lambda: "acc is required")

    layout = acc.type.layout
    _check(
        isinstance(layout, AMDWMMALayout) and layout.version == version,
        lambda: f"Expected layout to be an instance of AMDWMMALayout with version {version}")

    a_layout = a.type.layout
    _check(
        isinstance(a_layout, DotOperandLayout) and isinstance(a_layout.parent, AMDWMMALayout)
        and a_layout.parent.version == version,
        lambda: "Expected a's layout to be a DotOperandLayout with parent matching AMDWMMALayout")

    b_layout = b.type.layout
    _check(
        isinstance(b_layout, DotOperandLayout) and isinstance(b_layout.parent, AMDWMMALayout)
        and b_layout.parent.version == version,
        lambda: "Expected b's layout to be a DotOperandLayout with parent matching AMDWMMALayout")


def _wmma(version, a, b, acc, semantic):
    """ Shared implementation for AMD WMMA operations for Gluon builtins """
    _verify_wmma(version, a, b, acc)

    handle = semantic.dot(a, b, acc, input_precision=knobs.language.fp32_default, max_num_imprecise_acc=None,
                          out_dtype=acc.dtype).handle
    return ttgl.tensor(handle, acc.type)


def _mma_scaled(a, a_scale, a_format, b, b_scale, b_format, acc, scale_fn, semantic):
    """ Shared implementation for AMD WMMA scaled and MFMA scaled operation. """

    def _get_scale_shape(op_idx, operand, format, scale_factor):
        operand_shape = [s for s in operand.type.shape]
        scale_shape = operand_shape
        unpack_factor = 2 if format == "e2m1" else 1
        if op_idx == 0:
            k = scale_shape[-1] * unpack_factor
            scale_shape[-1] = k // scale_factor
        else:
            k = scale_shape[-2] * unpack_factor
            scale_shape[-2] = k // scale_factor
            scale_shape[-2], scale_shape[-1] = scale_shape[-1], scale_shape[-2]
        return scale_shape

    def _get_default_scale_dtype_and_unit_value(op_idx):
        default_value_by_dtype = {ttgl.uint8: 0x7F, ttgl.float8e4nv: 1.0}

        if a_scale is None and b_scale is None:
            return ttgl.uint8, 0x7F

        if a_format == b_format == "e2m1":
            # Fp4 x Fp4 requries to use the same scale dtype for both operands.
            other_scale = b_scale if op_idx == 0 else a_scale
            return other_scale.dtype, default_value_by_dtype[other_scale.dtype]

        return ttgl.uint8, 0x7F

    def _create_and_broadcast_default_scale(op_idx, scale, format, scale_factor):
        operand = a if op_idx == 0 else b

        scale_shape = _get_scale_shape(op_idx, operand, format, scale_factor)
        if isinstance(scale, ttgl.tensor) and scale.numel.value != 1:
            # In the case of scale pre-shuffling, the input shape is different from the default shape. We only check
            # the number of elements here.
            assert math.prod(scale_shape) == scale.numel.value, "Incompatible scale shape"
            return scale

        scale_layout = scale_fn(operand.type.layout, scale_shape, scale_factor)
        scale_value = _unwrap_if_constexpr(scale)
        if scale_value is None:
            scale_dtype, scale_value = _get_default_scale_dtype_and_unit_value(op_idx)
        elif isinstance(scale_value, int):
            scale_dtype = ttgl.uint8
        elif isinstance(scale_value, float):
            scale_dtype = ttgl.float8e4nv
        else:
            scale_dtype = scale.dtype

        return semantic.full(scale_shape, scale_value, scale_dtype, scale_layout)

    scale_factor = semantic.deduce_scale_factor(a, a_scale, a_format, True, b, b_scale, b_format, True)

    a_scale = _create_and_broadcast_default_scale(0, a_scale, a_format, scale_factor)
    b_scale = _create_and_broadcast_default_scale(1, b_scale, b_format, scale_factor)
    output = semantic.dot_scaled(a, a_scale, a_format, b, b_scale, b_format, acc, fast_math=False, lhs_k_pack=True,
                                 rhs_k_pack=True, out_dtype=ttgl.float32)
    return ttgl.tensor(output.handle, acc.type)


def _scaled_upcast(src, scale, elem_type, axis, semantic):
    _check(isinstance(src.type, ttgl.distributed_type),
           lambda: f"Expected src to have a distributed_type but got {src.type}")
    _check(isinstance(scale.type, ttgl.distributed_type),
           lambda: f"Expected scale to have a distributed_type but got {scale.type}")
    _check(elem_type in {ttgl.float16, ttgl.bfloat16},
           lambda: f"Expected elem_type to be fp16 or bf16 but got {elem_type}")

    if src.dtype in {ttgl.float8e4nv, ttgl.float8e5}:
        _check(axis is None, lambda: "axis must be None for fp8 scaled_upcast")
        _check(scale.type.shape == src.type.shape,
               lambda: f"Expected scale shape for fp8 scaled_upcast to be {src.type.shape} but got {scale.type.shape}")
        _check(
            scale.type.layout == src.type.layout,
            lambda: f"Expected scale layout for fp8 scaled_upcast to be {src.type.layout} but got {scale.type.layout}")
        # Note: bf16 is allowed due to CDNA3/CDNA4 conversion before passing to scaled_upcast
        _check(scale.dtype in {ttgl.int8, ttgl.uint8, ttgl.bfloat16},
               lambda: f"Unsupported scale dtype for fp8 scaled_upcast: {scale.dtype}")
        ret_ty = scale.type.with_element_ty(elem_type)
        handle = semantic.builder.create_scaled_upcast_fp8(ret_ty.to_ir(semantic.builder), src.handle, scale.handle)
        return _wrap_scaled_upcast_result(handle, elem_type, semantic)

    _check(src.dtype in {ttgl.int8, ttgl.uint8},
           lambda: f"Expected packed fp4 input in int8/uint8 or fp8 input, but got {src.dtype}")
    _check(axis is not None, lambda: "axis is required for packed fp4 scaled_upcast")

    rank = len(src.type.shape)
    _check(-rank <= axis < rank, lambda: f"axis {axis} out of range for rank {rank}")
    if axis < 0:
        axis += rank

    expected_shape = list(src.type.shape)
    expected_shape[axis] *= 2
    _check(scale.type.shape == expected_shape,
           lambda: f"Expected scale shape for fp4 scaled_upcast to be {expected_shape} but got {scale.type.shape}")
    _check(scale.dtype in {ttgl.int8, ttgl.uint8, ttgl.bfloat16},
           lambda: f"Unsupported scale dtype for fp4 scaled_upcast: {scale.dtype}")

    handle = semantic.builder.create_scaled_upcast_fp4(src.handle, scale.handle, elem_type.to_ir(semantic.builder),
                                                       axis)
    return _wrap_scaled_upcast_result(handle, elem_type, semantic)


def _scale_sel_bytes(is_fp4, sel):
    """Decode which Vscale byte(s) `sel` (OPSEL) selects per Tables 52/53.

    Returns (bytes, reserved). Bytes 2 and 3 are the "upper half".
    """
    if is_fp4:
        block16 = (sel >> 2) & 1
        hi = (sel >> 1) & 1
        if not block16:
            return ([2, 3] if hi else [0, 1]), False
        return ([1, 3] if hi else [0, 2]), False
    block16 = (sel >> 3) & 1
    o1 = (sel >> 1) & 1
    o2 = (sel >> 2) & 1
    if not block16:
        return [2 * o1 + o2], False
    if o2:
        return [], True
    return ([2, 3] if o1 else [0, 1]), False


def _scale_upcast(val, scale, axis, scale_sel, elem_type, pack_axis, semantic):
    _check(isinstance(val.type, ttgl.distributed_type),
           lambda: f"Expected val to have a distributed_type but got {val.type}")
    _check(isinstance(scale.type, ttgl.distributed_type),
           lambda: f"Expected scale to have a distributed_type but got {scale.type}")
    _check(elem_type in {ttgl.float16, ttgl.bfloat16, ttgl.float32},
           lambda: f"Expected elem_type to be fp16, bf16 or fp32 but got {elem_type}")

    axis = _unwrap_if_constexpr(axis)
    scale_sel = _unwrap_if_constexpr(scale_sel)
    pack_axis = _unwrap_if_constexpr(pack_axis)

    rank = len(val.type.shape)
    _check(-rank <= axis < rank, lambda: f"axis {axis} out of range for rank {rank}")
    if axis < 0:
        axis += rank

    packed_fp4 = val.dtype in {ttgl.int8, ttgl.uint8}
    _check(packed_fp4 or val.dtype in {ttgl.float8e4nv, ttgl.float8e5},
           lambda: f"Expected val to be fp8 (e4m3/e5m2) or packed fp4 (int8/uint8) but got {val.dtype}")

    # `pack_axis` (fp4 only) is the storage->element x2 expansion dim; defaults
    # to `axis`. Shape/layout constraints are on the element (output) layout.
    if pack_axis is None:
        pack_axis_eff = axis
    else:
        _check(packed_fp4, lambda: "pack_axis is only valid for packed fp4 (int8/uint8) val")
        pack_axis_eff = pack_axis
        _check(-rank <= pack_axis_eff < rank, lambda: f"pack_axis {pack_axis} out of range for rank {rank}")
        if pack_axis_eff < 0:
            pack_axis_eff += rank
        # The lowering only supports the x2 fp4 expansion along `axis`.
        _check(pack_axis_eff == axis,
               lambda: f"pack_axis ({pack_axis_eff}) != axis ({axis}) is not yet supported by cvt_scale_pk")

    out_shape = list(val.type.shape)
    if packed_fp4:
        out_shape[pack_axis_eff] *= 2

    _check(scale.dtype in {ttgl.int16, ttgl.uint16, ttgl.int32, ttgl.uint32},
           lambda: f"Expected scale to be int16/uint16/int32/uint32 but got {scale.dtype}")
    scale_bits = 16 if scale.dtype in {ttgl.int16, ttgl.uint16} else 32

    _check(len(scale.type.shape) == rank, lambda: "scale must have the same rank as val")
    for d in range(rank):
        if d == axis:
            continue
        _check(scale.type.shape[d] == out_shape[d],
               lambda: f"scale and output must have equal size on non-axis dim {d}")
    out_k = out_shape[axis]
    scale_k = scale.type.shape[axis]
    _check(scale_k > 0 and out_k % scale_k == 0,
           lambda: f"output axis size {out_k} must be a positive multiple of scale axis size {scale_k}")
    k_scale = out_k // scale_k
    # k_scale must be a power of two (the layout expands the scale by
    # identity1D(k_scale, ...)); k_scale > 8 reuses a scale across k_scale/8 pk8
    # groups, k_scale < 8 pads the pk8 inputs with undef and drops the extra
    # outputs.
    _check(k_scale > 0 and (k_scale & (k_scale - 1)) == 0,
           lambda: f"k_scale (output/scale along axis) must be a power of two, got {k_scale}")
    _check(not packed_fp4 or k_scale >= 2,
           lambda: f"k_scale must be even (>= 2) for packed fp4, got {k_scale}")

    sel_bits = 3 if packed_fp4 else 4
    _check(0 <= scale_sel < (1 << sel_bits),
           lambda: f"scale_sel {scale_sel} out of range [0, {1 << sel_bits}) for {'fp4' if packed_fp4 else 'fp8'}")

    # The block-select bit only means block16/block32 when the scale spans a
    # hardware block of 16 or 32; cross-check it only then.
    block16 = (scale_sel >> (2 if packed_fp4 else 3)) & 1
    if k_scale in (16, 32):
        _check((16 if block16 else 32) == k_scale,
               lambda: f"scale_sel encodes block{16 if block16 else 32} but k_scale is {k_scale}")

    used_bytes, reserved = _scale_sel_bytes(packed_fp4, scale_sel)
    _check(not reserved, lambda: f"scale_sel {scale_sel} is a reserved combination")
    if scale_bits == 16:
        _check(all(b < 2 for b in used_bytes),
               lambda: f"scale_sel {scale_sel} selects the upper half of a 16-bit scale")

    pack_axis_arg = -1 if pack_axis is None else pack_axis_eff
    handle = semantic.builder.create_scale_upcast(val.handle, scale.handle, elem_type.to_ir(semantic.builder), axis,
                                                  scale_sel, pack_axis_arg)
    return _wrap_scaled_upcast_result(handle, elem_type, semantic)
