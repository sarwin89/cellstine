"""Bounded integer boxes in the same order as an ij-indexed meshgrid."""

from __future__ import annotations

import math
import operator
from collections.abc import Iterator, Sequence

import numpy as np


def _integer_box_blocks(bounds: Sequence[int], block_size: int = 65536) -> Iterator[np.ndarray]:
    """Yield fresh int64 blocks of ``product(range(-b, b+1) for b in bounds)``.

    Later axes vary fastest. Whole suffix axes and a slice of the preceding
    axis form each block, avoiding both a full meshgrid and per-point integer
    division. Coordinate bounds must fit int64; the box's total size need not.
    """
    bounds = tuple(operator.index(value) for value in bounds)
    block_size = operator.index(block_size)
    if not bounds or min(bounds) < 0 or block_size < 1:
        raise ValueError("nonnegative bounds and a positive block size are required")
    if max(bounds) > np.iinfo(np.int64).max:
        raise ValueError("integer search coordinate exceeds int64")
    widths = tuple(2 * value + 1 for value in bounds)
    split = len(bounds) - 1
    suffix_size = 1
    while split > 0 and suffix_size * widths[split] <= block_size:
        suffix_size *= widths[split]
        split -= 1
    step = max(1, block_size // suffix_size)
    for prefix_flat in range(math.prod(widths[:split])):
        prefix = [0] * split
        remaining = prefix_flat
        for axis in range(split - 1, -1, -1):
            remaining, remainder = divmod(remaining, widths[axis])
            prefix[axis] = remainder - bounds[axis]
        for start in range(0, widths[split], step):
            count = min(step, widths[split] - start)
            shape = (count, *widths[split + 1:])
            values = np.empty((*shape, len(bounds)), dtype=np.int64)
            for axis, value in enumerate(prefix):
                values[..., axis] = value
            for axis in range(split, len(bounds)):
                length = count if axis == split else widths[axis]
                lower = start - bounds[axis] if axis == split else -bounds[axis]
                numbers = np.arange(length, dtype=np.int64)
                numbers += np.int64(lower)
                broadcast = [1] * len(shape)
                broadcast[axis - split] = length
                values[..., axis] = numbers.reshape(broadcast)
            yield values.reshape(-1, len(bounds))
