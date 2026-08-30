"""Shared lattice helpers used by higher-level workflows."""

from __future__ import annotations

import math

import numpy as np


def build_target_lattice(
    a_length: float, b_length: float, angle_deg: float, c_length: float = 30.0
) -> np.ndarray:
    """Return a cell with the requested in-plane lengths, angle, and height.

    The first vector is placed along ``+x`` and the second at ``angle_deg`` from
    it, so the cell is right handed and the third vector is the surface normal.
    """

    angle_rad = math.radians(float(angle_deg))
    return np.array(
        [
            [float(a_length), 0.0, 0.0],
            [float(b_length) * math.cos(angle_rad), float(b_length) * math.sin(angle_rad), 0.0],
            [0.0, 0.0, float(c_length)],
        ],
        dtype=float,
    )


def vector_angle_deg(first: np.ndarray, second: np.ndarray) -> float:
    """Return the angle between two nonzero vectors, in degrees.

    Reading the angle as ``arccos(u . v / (|u| |v|))`` is accurate only away from
    ``0`` and ``180`` degrees: ``arccos`` has an infinite derivative at its end
    points, so a cosine carrying a relative error ``eps`` gives an angle carrying
    an error of order ``sqrt(eps)``, and a cosine rounded just outside ``[-1, 1]``
    has to be clipped before it can be used at all.  Cell angles of nearly
    degenerate supercells and the interlayer twist of an almost aligned bilayer
    both live exactly there.

    This uses Kahan's formula instead: with ``a`` and ``b`` the two unit vectors,

    ``angle = 2 arctan2(|a - b|, |a + b|)``,

    which is the half-angle written through the chord and is uniformly accurate
    over the whole range, needs no clipping, and returns exactly ``90`` degrees
    for exactly orthogonal vectors.  It works in any dimension.
    """

    left = np.asarray(first, dtype=float).ravel()
    right = np.asarray(second, dtype=float).ravel()
    left_norm = float(np.linalg.norm(left))
    right_norm = float(np.linalg.norm(right))
    if left_norm <= 0.0 or right_norm <= 0.0:
        raise ValueError("the angle between vectors needs two nonzero vectors")
    unit_left = left / left_norm
    unit_right = right / right_norm
    chord = float(np.linalg.norm(unit_left - unit_right))
    span = float(np.linalg.norm(unit_left + unit_right))
    return float(np.degrees(2.0 * math.atan2(chord, span)))


def inplane_principal_log_strains(
    bottom_lattice: np.ndarray, top_lattice: np.ndarray
) -> tuple[float, float]:
    """Return the two principal logarithmic strains that map the top cell onto the bottom one.

    Forcing the top slab to adopt the in-plane cell of the bottom slab applies the
    linear map ``F = T^-1 B`` to the top layer, where ``B`` and ``T`` hold the two
    in-plane lattice vectors as rows.  Its principal stretches are the singular
    values of ``F``, so the principal logarithmic (Hencky) strains are their
    logarithms.  Unlike a norm of ``B - T`` this does not depend on how either
    cell happens to be oriented or on which pair of lattice vectors was chosen to
    describe it, and it is the strain a plane-wave calculation actually sees.
    """

    bottom = np.asarray(bottom_lattice, dtype=float)[:2, :2]
    top = np.asarray(top_lattice, dtype=float)[:2, :2]
    if abs(float(np.linalg.det(top))) <= 1e-12:
        raise ValueError("top in-plane lattice vectors must be linearly independent")
    deformation = np.linalg.solve(top, bottom)
    stretches = np.linalg.svd(deformation, compute_uv=False)
    strains = np.sort(np.log(np.maximum(stretches, 1e-300)))
    return float(strains[0]), float(strains[1])
