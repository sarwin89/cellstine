"""Small helper functions for the optional spglib-backed workflow path."""

from __future__ import annotations

from typing import Any

import numpy as np


def dataset_value(dataset: Any, key: str, default: Any = None) -> Any:
    """Read a value from either a spglib dataset object or mapping."""

    if dataset is None:
        return default
    if hasattr(dataset, key):
        return getattr(dataset, key)
    try:
        return dataset[key]
    except Exception:
        return default


def crystal_system(number: int | None) -> str | None:
    """Return the crystal-system name for an international space-group number."""

    if number is None:
        return None
    value = int(number)
    if 1 <= value <= 2:
        return "triclinic"
    if 3 <= value <= 15:
        return "monoclinic"
    if 16 <= value <= 74:
        return "orthorhombic"
    if 75 <= value <= 142:
        return "tetragonal"
    if 143 <= value <= 167:
        return "trigonal"
    if 168 <= value <= 194:
        return "hexagonal"
    if 195 <= value <= 230:
        return "cubic"
    return None


def has_inversion(rotations: np.ndarray | None) -> bool | None:
    """Return whether a rotation set contains inversion."""

    if rotations is None:
        return None
    inversion = -np.eye(3, dtype=int)
    return any(np.array_equal(np.asarray(rotation, dtype=int), inversion) for rotation in rotations)
