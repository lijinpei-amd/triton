from .._core import builtin, _unwrap_if_constexpr

__all__ = ["sched_barrier", "sched_order_barrier"]


@builtin
def sched_barrier(allow=None, _semantic=None):
    """Insert a barrier that constrains LLVM instruction scheduling.

    The barrier prevents the LLVM compiler's instruction scheduler from moving
    instructions across this point. It is a LLVM compiler hint only.

    Args:
        allow (str or tuple[str, ...], optional): Instruction classes allowed to
            cross the barrier. None means that no instruction classes may cross.
            Currently only None is allowed.
    """
    allow = _unwrap_if_constexpr(allow)
    if allow is None or isinstance(allow, str):
        allow = (allow, )

    # map instruction classes to their corresponding masks
    instruction_masks = {None: 0}
    mask = 0
    for instruction in allow:
        instruction = _unwrap_if_constexpr(instruction)
        if instruction is not None and not isinstance(instruction, str):
            raise TypeError("instruction classes must be strings")
        if instruction not in instruction_masks:
            raise ValueError(f"Unsupported instruction class: {instruction!r}")
        mask |= instruction_masks[instruction]

    _semantic.builder.create_amd_sched_barrier(mask)


_SCHED_ORDER_CLASSES = {
    "valu": 0x2,
    "salu": 0x4,
    "mfma": 0x8,
    "vmem": 0x10,
    "vmem_read": 0x20,
    "vmem_write": 0x40,
    "ds": 0x80,
    "ds_read": 0x100,
    "ds_write": 0x200,
    "trans": 0x400,
    "async_load": 0x800,
}


def _sched_order_mask(classes):
    classes = _unwrap_if_constexpr(classes)
    if classes is None:
        return 0
    if isinstance(classes, str):
        classes = (classes, )
    mask = 0
    for c in classes:
        c = _unwrap_if_constexpr(c)
        if c not in _SCHED_ORDER_CLASSES:
            raise ValueError(f"Unsupported instruction class: {c!r}")
        mask |= _SCHED_ORDER_CLASSES[c]
    return mask


@builtin
def sched_order_barrier(before=None, after=None, _semantic=None):
    """Order instruction classes around this point for the LLVM scheduler.

    Instructions before this point whose class is in ``before`` are scheduled
    before it, and instructions after it whose class is in ``after`` are
    scheduled after it. Other instructions may move freely. Classes:
    "valu", "salu", "mfma", "vmem", "vmem_read" (sync loads), "vmem_write",
    "ds", "ds_read", "ds_write", "trans", "async_load".
    """
    _semantic.builder.create_amd_sched_order_barrier(_sched_order_mask(before), _sched_order_mask(after))
