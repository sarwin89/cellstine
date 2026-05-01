"""Shared lattice helpers used by higher-level workflows."""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np

def in_plane_lengths_and_angle(lattice: np.ndarray) -> tuple[float, float, float]:
    basis = np.asarray(lattice, dtype=float)[:2, :2]
    vector_a = basis[0]
    vector_b = basis[1]
    length_a = float(np.linalg.norm(vector_a))
    length_b = float(np.linalg.norm(vector_b))
    denominator = max(length_a * length_b, 1e-12)
    cosine = np.clip(float(np.dot(vector_a, vector_b) / denominator), -1.0, 1.0)
    gamma_deg = float(np.degrees(np.arccos(cosine)))
    return length_a, length_b, gamma_deg


def build_target_lattice(a_length: float, b_length: float, angle_deg: float, c_length: float = 30.0) -> np.ndarray:
    angle_rad = math.radians(float(angle_deg))
    return np.array(
        [
            [float(a_length), 0.0, 0.0],
            [float(b_length) * math.cos(angle_rad), float(b_length) * math.sin(angle_rad), 0.0],
            [0.0, 0.0, float(c_length)],
        ],
        dtype=float,
    )


def apply_inplane_prestrain(
    lattice: np.ndarray,
    *,
    mode: str = "none",
    magnitude: float = 0.0,
    axis: str | None = None,
) -> np.ndarray:
    strained = np.array(lattice, dtype=float, copy=True)
    resolved_mode = str(mode).lower()
    if resolved_mode == "none" or abs(float(magnitude)) <= 1e-15:
        return strained
    if resolved_mode == "biaxial":
        strained[0] *= 1.0 + float(magnitude)
        strained[1] *= 1.0 + float(magnitude)
        return strained
    if resolved_mode == "uniaxial":
        axis_name = (axis or "a").lower()
        axis_index = 0 if axis_name in {"a", "x", "0"} else 1
        strained[axis_index] *= 1.0 + float(magnitude)
        return strained
    raise ValueError("prestrain mode must be one of: none, biaxial, uniaxial")


def lattice_mismatch_fraction(bottom_lattice: np.ndarray, top_lattice: np.ndarray) -> float:
    bottom_inplane = np.asarray(bottom_lattice, dtype=float)[:2, :]
    top_inplane = np.asarray(top_lattice, dtype=float)[:2, :]
    denominator = max(float(np.linalg.norm(bottom_inplane)), 1e-12)
    return float(np.linalg.norm(bottom_inplane - top_inplane) / denominator)


def parse_float_list(raw_values: Sequence[str] | None, expected: int | None = None) -> list[float] | None:
    if raw_values is None:
        return None
    values = [float(value) for value in raw_values]
    if expected is not None and len(values) != expected:
        raise ValueError(f"expected {expected} numeric values, received {len(values)}")
    return values
