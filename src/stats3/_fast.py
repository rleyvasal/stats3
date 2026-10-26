"""The optional compiled kernels (``stats3[fast]``): a loader and a switch.

Some of R's algorithms are row-by-row loops whose every step depends on the
last (k-means transfers, the nearest-neighbour chain of hclust), or sums
that must be rounded as R's compiled code rounds them (fused multiply-adds
on ARM). stats3 runs them in NumPy, and, when Numba is installed, through
``stats3._compiled``, whose kernels do the same operations in the same
order and give the same bits. ``COMPILED = False`` keeps everything in
NumPy, for checking the two paths against each other.
"""

from __future__ import annotations

from typing import Any

COMPILED = True

_KERNELS: dict[str, Any] = {}


def compiled(name: str) -> Any:
    """The kernel ``name`` from ``stats3._compiled`` when Numba is installed
    and ``COMPILED`` is on, else None."""
    if not COMPILED:
        return None
    if name not in _KERNELS:
        try:
            from stats3 import _compiled as module
        except ImportError:
            _KERNELS[name] = None
        else:
            _KERNELS[name] = getattr(module, name)
    return _KERNELS[name]
