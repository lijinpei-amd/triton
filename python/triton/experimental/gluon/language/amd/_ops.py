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
    _check(
        scale.type.shape[:axis] + scale.type.shape[axis + 1:] == expected_shape[:axis] + expected_shape[axis + 1:],
        lambda: f"Expected scale shape for scaled_upcast to match output shape on non-axis dims: "
        f"{expected_shape}, but got {scale.type.shape}")
    _check(
        scale.type.shape[axis] > 0 and expected_shape[axis] % scale.type.shape[axis] == 0,
        lambda: f"Expected output axis extent {expected_shape[axis]} to be divisible by scale axis extent "
        f"{scale.type.shape[axis]}")
    _check(scale.dtype in {ttgl.int8, ttgl.uint8, ttgl.bfloat16},
           lambda: f"Unsupported scale dtype for fp4 scaled_upcast: {scale.dtype}")

    handle = semantic.builder.create_scaled_upcast_fp4(src.handle, scale.handle, elem_type.to_ir(semantic.builder),
                                                       axis)
    return _wrap_scaled_upcast_result(handle, elem_type, semantic)


_CVT_SCALE_PK_LANES = ("h0", "h1")
_CVT_SCALE_PK_BYTES = ("b0", "b1", "b2", "b3", "b0b1", "b2b3", "b0b2", "b1b3")
_CVT_SCALE_PK_BYTE_INDICES = {
    "b0": (0, ),
    "b1": (1, ),
    "b2": (2, ),
    "b3": (3, ),
    "b0b1": (0, 1),
    "b2b3": (2, 3),
    "b0b2": (0, 2),
    "b1b3": (1, 3),
}


def _cvt_scale_pk(val, scale, axis, scale_sel, elem_type, k_width, pack_axis, semantic):
    _check(isinstance(val.type, ttgl.distributed_type),
           lambda: f"Expected val to have a distributed_type but got {val.type}")
    _check(isinstance(scale.type, ttgl.distributed_type),
           lambda: f"Expected scale to have a distributed_type but got {scale.type}")
    _check(elem_type in {ttgl.float16, ttgl.bfloat16, ttgl.float32},
           lambda: f"Expected elem_type to be fp16, bf16 or fp32 but got {elem_type}")

    axis = _unwrap_if_constexpr(axis)
    scale_sel = _unwrap_if_constexpr(scale_sel)
    k_width = _unwrap_if_constexpr(k_width)
    pack_axis = _unwrap_if_constexpr(pack_axis)

    tuple_types = (tuple, list, ttgl.tuple)
    _check(isinstance(scale_sel, tuple_types) and len(scale_sel) > 0,
           lambda: "scale_sel must be a non-empty sequence of (scale_lane, scale_bytes) tuples")
    normalized_scale_sel = []
    for scale_sel_idx, selection in enumerate(scale_sel):
        selection = _unwrap_if_constexpr(selection)
        _check(isinstance(selection, tuple_types) and len(selection) == 2,
               lambda: f"scale_sel[{scale_sel_idx}] must be a tuple of two strings: (scale_lane, scale_bytes)")
        scale_lane, scale_bytes = selection
        _check(scale_lane in _CVT_SCALE_PK_LANES,
               lambda: f"invalid scale lane {scale_lane!r} in scale_sel[{scale_sel_idx}]; expected 'h0' or 'h1'")
        _check(scale_bytes in _CVT_SCALE_PK_BYTES,
               lambda: f"invalid scale bytes {scale_bytes!r} in scale_sel[{scale_sel_idx}]; "
               f"expected one of {', '.join(_CVT_SCALE_PK_BYTES)}")
        normalized_scale_sel.append((scale_lane, scale_bytes))
    scale_sel = normalized_scale_sel

    rank = len(val.type.shape)
    _check(-rank <= axis < rank, lambda: f"axis {axis} out of range for rank {rank}")
    if axis < 0:
        axis += rank

    packed_fp4_dtypes = {ttgl.int8, ttgl.uint8, ttgl.int16, ttgl.uint16, ttgl.int32, ttgl.uint32}
    packed_fp4 = val.dtype in packed_fp4_dtypes
    _check(packed_fp4 or val.dtype in {ttgl.float8e4nv, ttgl.float8e5},
           lambda: f"Expected val to be fp8 (e4m3/e5m2) or packed fp4 "
           f"(8/16/32-bit integer) but got {val.dtype}")

    # `pack_axis` (fp4 only) is expanded by the number of fp4 values in each
    # storage element; defaults to `axis`. Shape/layout constraints are on the
    # element (output) layout.
    if pack_axis is None:
        pack_axis_eff = axis
    else:
        _check(packed_fp4, lambda: "pack_axis is only valid for packed fp4 integer val")
        pack_axis_eff = pack_axis
        _check(-rank <= pack_axis_eff < rank, lambda: f"pack_axis {pack_axis} out of range for rank {rank}")
        if pack_axis_eff < 0:
            pack_axis_eff += rank

    out_shape = list(val.type.shape)
    if packed_fp4:
        out_shape[pack_axis_eff] *= val.dtype.primitive_bitwidth // 4

    scale_dtypes = {ttgl.int8, ttgl.uint8, ttgl.int16, ttgl.uint16, ttgl.int32, ttgl.uint32}
    _check(scale.dtype in scale_dtypes,
           lambda: f"Expected scale to be an 8/16/32-bit integer but got {scale.dtype}")
    scale_bits = scale.dtype.primitive_bitwidth
    _check(not packed_fp4 or scale_bits >= 16,
           lambda: "Expected packed fp4 scale to be a 16/32-bit integer")

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
    scale_factor = out_k // scale_k
    _check(scale_factor > 0 and (scale_factor & (scale_factor - 1)) == 0,
           lambda: f"scale_factor (output/scale along axis) must be a power of two, got {scale_factor}")
    _check(not packed_fp4 or scale_factor >= 2,
           lambda: f"scale_factor must be even (>= 2) for packed fp4, got {scale_factor}")

    if k_width is not None:
        _check(isinstance(k_width, int) and not isinstance(k_width, bool) and k_width > 0,
               lambda: f"k_width must be a positive integer or None, got {k_width!r}")
        _check((k_width & (k_width - 1)) == 0,
               lambda: f"k_width must be a power of two, got {k_width}")

    fp4_bytes = ("b0b1", "b2b3", "b0b2", "b1b3")
    fp8_bytes = ("b0", "b1", "b2", "b3", "b0b1", "b2b3")
    supported_bytes = fp4_bytes if packed_fp4 else fp8_bytes
    for scale_sel_idx, (_, scale_bytes) in enumerate(scale_sel):
        _check(scale_bytes in supported_bytes,
               lambda: f"scale bytes {scale_bytes!r} in scale_sel[{scale_sel_idx}] are not supported for "
               f"{'packed fp4' if packed_fp4 else 'fp8'}; expected one of {', '.join(supported_bytes)}")

        if scale_bits < 32:
            used_bytes = _CVT_SCALE_PK_BYTE_INDICES[scale_bytes]
            if scale_bits == 16:
                _check(all(b < 2 for b in used_bytes),
                       lambda: f"scale bytes {scale_bytes!r} in scale_sel[{scale_sel_idx}] select the upper half "
                       "of a 16-bit scale")
            else:
                unavailable_byte = next((b for b in used_bytes if b >= 1), None)
                _check(unavailable_byte is None,
                       lambda: f"scale bytes {scale_bytes!r} in scale_sel[{scale_sel_idx}] select Vscale byte "
                       f"{unavailable_byte}, which is not available for an 8-bit scale")

    pack_axis_arg = -1 if pack_axis is None else pack_axis_eff
    k_width_arg = -1 if k_width is None else k_width
    handle = semantic.builder.create_cvt_scale_pk(val.handle, scale.handle, elem_type.to_ir(semantic.builder), axis,
                                                  scale_sel, k_width_arg, pack_axis_arg)
    return _wrap_scaled_upcast_result(handle, elem_type, semantic)
