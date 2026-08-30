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


def rotation_matrix_z(angle_deg: float) -> np.ndarray:
    theta = np.deg2rad(float(angle_deg))
    cos_theta = float(np.cos(theta))
    sin_theta = float(np.sin(theta))
    return np.array(
        [
            [cos_theta, -sin_theta, 0.0],
            [sin_theta, cos_theta, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )


def rotation_matrix_x(angle_deg: float) -> np.ndarray:
    theta = np.deg2rad(float(angle_deg))
    cos_theta = float(np.cos(theta))
    sin_theta = float(np.sin(theta))
    return np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, cos_theta, -sin_theta],
            [0.0, sin_theta, cos_theta],
        ],
        dtype=float,
    )


def rotation_matrix_y(angle_deg: float) -> np.ndarray:
    theta = np.deg2rad(float(angle_deg))
    cos_theta = float(np.cos(theta))
    sin_theta = float(np.sin(theta))
    return np.array(
        [
            [cos_theta, 0.0, sin_theta],
            [0.0, 1.0, 0.0],
            [-sin_theta, 0.0, cos_theta],
        ],
        dtype=float,
    )


def yaw_pitch_roll_matrix(yaw_deg: float, pitch_deg: float, roll_deg: float) -> np.ndarray:
    """Return a yaw-z, pitch-y, roll-x rotation matrix."""

    return rotation_matrix_x(roll_deg) @ rotation_matrix_y(pitch_deg) @ rotation_matrix_z(yaw_deg)


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


def integer_supercell_matrix(matrix: Sequence[Sequence[int]] | Sequence[int]) -> np.ndarray:
    """Return ``matrix`` as an integer ``3x3`` supercell matrix.

    The rows are the new lattice vectors written in the basis of the old ones,
    so the new lattice is ``matrix @ lattice`` and the new cell holds
    ``|det matrix|`` copies of the old one.  Nine numbers, given either as a
    flat sequence or as three rows, are accepted; they must be integers, since
    a non-integer matrix would not map the old lattice into the new one.
    """

    array = np.asarray(matrix, dtype=float).reshape(-1)
    if array.size != 9:
        raise ValueError("a supercell matrix needs nine integer entries")
    rounded = np.rint(array)
    if not np.allclose(array, rounded, atol=1e-9):
        raise ValueError("a supercell matrix must have integer entries")
    integers = rounded.astype(np.int64).reshape(3, 3)
    determinant = int(np.rint(np.linalg.det(integers.astype(float))))
    if determinant == 0:
        raise ValueError("a supercell matrix must be invertible")
    return integers


def supercell_cosets(matrix: Sequence[Sequence[int]] | Sequence[int]) -> np.ndarray:
    """Return the ``|det matrix|`` lattice translations inside the supercell.

    The translations are the coset representatives of the old lattice modulo
    the new one: every atom of the old cell is copied once per representative.
    They are found by exact integer arithmetic -- a translation ``t`` lies in
    the supercell exactly when every entry of ``t @ adj(matrix)`` lies between
    ``0`` and ``det matrix`` -- so no rounding tolerance decides membership and
    the count of representatives is always exactly ``|det matrix|``.
    """

    integers = integer_supercell_matrix(matrix)
    determinant = int(np.rint(np.linalg.det(integers.astype(float))))
    adjugate = np.rint(np.linalg.inv(integers.astype(float)) * float(determinant)).astype(np.int64)
    if not np.array_equal(integers @ adjugate, determinant * np.eye(3, dtype=np.int64)):
        raise ArithmeticError("could not invert the supercell matrix exactly")
    lower = np.minimum(integers, 0).sum(axis=0)
    upper = np.maximum(integers, 0).sum(axis=0)
    ranges = [np.arange(int(lower[axis]), int(upper[axis]) + 1, dtype=np.int64) for axis in range(3)]
    grid = np.stack(np.meshgrid(*ranges, indexing="ij"), axis=-1).reshape(-1, 3)
    scaled = grid @ adjugate
    if determinant > 0:
        inside = np.all((scaled >= 0) & (scaled < determinant), axis=1)
    else:
        inside = np.all((scaled <= 0) & (scaled > determinant), axis=1)
    cosets = grid[inside]
    if int(cosets.shape[0]) != abs(determinant):
        raise ArithmeticError(  # pragma: no cover - defensive
            "the supercell enumeration found "
            f"{int(cosets.shape[0])} translations rather than {abs(determinant)}"
        )
    order = np.lexsort((cosets[:, 2], cosets[:, 1], cosets[:, 0]))
    return cosets[order]


def supercell_structure(
    structure: "StructureRecord", matrix: Sequence[Sequence[int]] | Sequence[int]
) -> "StructureRecord":
    """Return ``structure`` on the supercell ``matrix @ lattice``.

    This is the general form of :func:`repeat_structure`: the new cell is any
    integer combination of the old lattice vectors, not only a diagonal repeat,
    which is what lets a defect be put in the roundest available cell of a
    given size.  The images of one atom stay together in the atom list, so the
    species blocks a POSCAR needs survive, and the Cartesian geometry of the
    old cell is reproduced exactly inside the new one.
    """

    integers = integer_supercell_matrix(matrix)
    determinant = int(np.rint(np.linalg.det(integers.astype(float))))
    cells = abs(determinant)
    if np.array_equal(integers, np.eye(3, dtype=np.int64)):
        return structure.copy()
    cosets = supercell_cosets(integers)
    lattice = integers.astype(float) @ np.asarray(structure.lattice, dtype=float)
    inverse = np.linalg.inv(integers.astype(float))
    base = np.asarray(structure.positions_direct, dtype=float).reshape(-1, 3)
    images = (base[:, None, :] + cosets[None, :, :].astype(float)) @ inverse
    positions_direct = np.mod(images.reshape(-1, 3), 1.0)
    # A coordinate that lands a rounding error below 1 must not be wrapped to 0
    # and then reported as a different site, so pull it back to zero explicitly.
    positions_direct[np.abs(positions_direct - 1.0) < 1e-12] = 0.0
    flags = None
    if structure.selective_flags is not None:
        flags = [tuple(entry) for entry in structure.selective_flags for _ in range(cells)]
    return replace(
        structure,
        lattice=lattice,
        counts=[int(count) * cells for count in structure.counts],
        positions_direct=positions_direct,
        positions_cartesian=positions_direct @ lattice,
        selective_flags=flags,
    )


def repeat_structure(structure: "StructureRecord", repeats: Sequence[int]) -> "StructureRecord":
    """Return ``structure`` repeated ``repeats`` times along its three axes.

    The images of one atom stay together in the atom list, so the species blocks
    a POSCAR needs survive and each count is simply multiplied by the number of
    cells.  Fractional coordinates are rescaled rather than recomputed, so an
    atom that sat exactly on a cell face still does, and the Cartesian geometry
    of the original cell is reproduced exactly inside the repeated one.
    """

    counts_out = [int(value) for value in repeats]
    if len(counts_out) != 3 or any(value < 1 for value in counts_out):
        raise ValueError("repeats must be three integers of at least 1")
    if counts_out == [1, 1, 1]:
        return structure.copy()

    scale = np.asarray(counts_out, dtype=float)
    lattice = np.asarray(structure.lattice, dtype=float) * scale[:, None]
    shifts = np.stack(
        np.meshgrid(*[np.arange(value, dtype=float) for value in counts_out], indexing="ij"),
        axis=-1,
    ).reshape(-1, 3)
    base = np.asarray(structure.positions_direct, dtype=float)
    images = (base[:, None, :] + shifts[None, :, :]) / scale[None, None, :]
    positions_direct = images.reshape(-1, 3)
    cells = int(shifts.shape[0])
    flags = None
    if structure.selective_flags is not None:
        flags = [tuple(entry) for entry in structure.selective_flags for _ in range(cells)]
    return replace(
        structure,
        lattice=lattice,
        counts=[int(count) * cells for count in structure.counts],
        positions_direct=positions_direct,
        positions_cartesian=positions_direct @ lattice,
        selective_flags=flags,
    )
