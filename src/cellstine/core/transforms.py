"""Coordinate and structure transforms shared across workflows."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Sequence

import numpy as np

if TYPE_CHECKING:
    from ..io.models import StructureRecord


def rotation_matrix_about_axis(axis: Sequence[float], angle_deg: float) -> np.ndarray:
    vector = np.asarray(axis, dtype=float)
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-15:
        raise ValueError("rotation axis cannot be zero")
    unit = vector / norm
    x_val, y_val, z_val = unit
    angle = np.deg2rad(float(angle_deg))
    cosine = float(np.cos(angle))
    sine = float(np.sin(angle))
    one_minus = 1.0 - cosine
    return np.array(
        [
            [cosine + x_val * x_val * one_minus, x_val * y_val * one_minus - z_val * sine, x_val * z_val * one_minus + y_val * sine],
            [y_val * x_val * one_minus + z_val * sine, cosine + y_val * y_val * one_minus, y_val * z_val * one_minus - x_val * sine],
            [z_val * x_val * one_minus - y_val * sine, z_val * y_val * one_minus + x_val * sine, cosine + z_val * z_val * one_minus],
        ],
        dtype=float,
    )


def right_handed_lattice(lattice: np.ndarray, positions_direct: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lattice_array = np.array(lattice, dtype=float, copy=True)
    positions = np.array(positions_direct, dtype=float, copy=True)
    if float(np.linalg.det(lattice_array)) < 0.0:
        lattice_array[[0, 1]] = lattice_array[[1, 0]]
        positions[:, [0, 1]] = positions[:, [1, 0]]
    return lattice_array, positions


def strained_copy(structure: "StructureRecord", target_inplane_lattice: np.ndarray) -> "StructureRecord":
    lattice = np.array(structure.lattice, dtype=float, copy=True)
    lattice[:2, :] = np.asarray(target_inplane_lattice, dtype=float)[:2, :]
    cartesian = np.asarray(structure.positions_direct, dtype=float) @ lattice
    return replace(
        structure,
        lattice=lattice,
        positions_cartesian=cartesian,
        positions_direct=np.asarray(structure.positions_direct, dtype=float, copy=True),
        metadata={**structure.metadata, "strained_to_interface": True},
    )
