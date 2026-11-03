"""Indented JSON writing with bounded whole-document encoding eligibility."""

from __future__ import annotations

import json
import math
from typing import Any, TextIO


def _small_json_budget(value: Any, remaining: int = 1 << 20, depth: int = 0) -> int:
    """Remaining conservative text budget for plain JSON, or a negative value.

    This bounds eligibility for the faster whole-string encoder, not peak RSS.
    Large containers short-circuit; custom types and deep/cyclic objects retain
    the stream encoder's traversal and diagnostics. No user conversions run here.
    """
    if remaining < 0 or depth > 32:
        return -1
    kind = type(value)
    if kind is str:
        # ensure_ascii=True uses twelve characters for an astral code point.
        return remaining - 12 * len(value) - 2
    if kind is int:
        bits = value.bit_length()
        # At most309 decimal digits, below Python's minimum configurable digit
        # limit. Larger integers retain streaming conversion/error behavior.
        return -1 if bits > 1024 else remaining - bits // 3 - 3
    if kind is float:
        return remaining - 32 if math.isfinite(value) else -1
    if kind is bool or value is None:
        return remaining - 5
    if kind not in (dict, list, tuple) or len(value) > 1024:
        return -1
    if not value:
        return remaining - 2
    # Include closing indentation as well as per-entry indentation/separators.
    remaining -= 2 + 2 * depth + len(value) * (2 * (depth + 1) + 4)
    if remaining < 0:
        return -1
    if kind is dict:
        for key, item in value.items():
            if type(key) not in (str, int, float, bool, type(None)):
                return -1
            remaining = _small_json_budget(key, remaining, depth + 1)
            if type(key) is not str:
                remaining -= 2  # Primitive non-string keys acquire quotes.
            if remaining < 0:
                return -1
            remaining = _small_json_budget(item, remaining, depth + 1)
            if remaining < 0:
                return -1
    else:
        for item in value:
            remaining = _small_json_budget(item, remaining, depth + 1)
            if remaining < 0:
                return -1
    return remaining


def write_indented_json(handle: TextIO, value: Any, *, allow_nan: bool = True) -> None:
    """Write the existing indent2 format without owning the handle or newline.

    Whole encoding is limited to bounded plain documents. Large/unusual values
    retain streaming allocation and serialization failures. A single write can
    naturally have different I/O failure boundaries from many small writes.
    """
    try:
        small = _small_json_budget(value) >= 0
    except RecursionError:
        small = False
    if small:
        handle.write(json.dumps(value, indent=2, allow_nan=allow_nan))
    else:
        json.dump(value, handle, indent=2, allow_nan=allow_nan)
