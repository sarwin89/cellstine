"""Independent brute-force reference for the bilayer commensuration search.

This module deliberately shares no code with :mod:`cellstine.moire.search.gram`.
It enumerates *every* right-handed pair of short lattice vectors of each layer,
accepts a pair of supercells when the relative deformation has principal
logarithmic strains inside the combined budget, and reports the resulting
physical observables.  It is slow and obvious on purpose: the fast engine is
checked against it in ``benchmarks/benchmark_gram_search.py`` and in the test
suite.

Observables reported per candidate class

``twist_deg``
    the twist angle folded into the fundamental range implied by the two layer
    point groups,
``top_atoms`` / ``bottom_atoms``
    atoms per moire cell contributed by each layer,
``strains``
    the two relative principal logarithmic strains, sorted ascending,
``top_area``
    the area of the top supercell in square Angstrom.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

__all__ = ["ReferenceConfig", "ReferenceCandidate", "reference_search"]


@dataclass(frozen=True)
class ReferenceConfig:
    """Inputs of the brute-force search, mirroring the production limits."""

    top_basis: np.ndarray
    bottom_basis: np.ndarray
    max_length: float
    top_strain: float
    bottom_strain: float
    top_atoms: int = 1
    bottom_atoms: int = 1
    min_length: float | None = None
    max_atoms: int | None = None
    max_aspect_ratio: float = 12.0
    min_cell_angle_deg: float = 25.0
    max_cell_angle_deg: float = 155.0
    primitive_only: bool = True
    top_group: np.ndarray | None = None
    bottom_group: np.ndarray | None = None


@dataclass(frozen=True)
class ReferenceCandidate:
    """One physically distinct commensurate bilayer class."""

    twist_deg: float
    top_atoms: int
    bottom_atoms: int
    strains: tuple[float, float]
    top_area: float

    def signature(self) -> tuple[float, int, int, float, float, float]:
        return (
            self.twist_deg,
            self.top_atoms,
            self.bottom_atoms,
            self.strains[0],
            self.strains[1],
            self.top_area,
        )


def _short_vectors(basis: np.ndarray, radius: float) -> np.ndarray:
    """Return every nonzero integer coefficient pair with ``|B x| <= radius``."""

    inverse_norm = float(np.linalg.norm(np.linalg.inv(basis), ord=2))
    bound = int(math.ceil(radius * inverse_norm)) + 1
    grid = range(-bound, bound + 1)
    keep = []
    for i in grid:
        for j in grid:
            if i == 0 and j == 0:
                continue
            vector = basis @ np.array([i, j], dtype=float)
            if float(vector @ vector) <= radius * radius * (1.0 + 1e-12):
                keep.append((i, j))
    return np.asarray(keep, dtype=np.int64)


def _right_handed_pairs(coefficients: np.ndarray) -> np.ndarray:
    """Return every ordered pair of coefficients spanning a right-handed basis."""

    count = len(coefficients)
    left = np.repeat(np.arange(count), count)
    right = np.tile(np.arange(count), count)
    first, second = coefficients[left], coefficients[right]
    determinant = first[:, 0] * second[:, 1] - first[:, 1] * second[:, 0]
    keep = determinant > 0
    return np.stack([first[keep], second[keep]], axis=2)


def _grams(matrices: np.ndarray, metric: np.ndarray) -> np.ndarray:
    """Return the ``(n, 3)`` Gram triples ``(g11, g12, g22)`` of ``B @ M``."""

    transformed = np.einsum("nji,jk,nkl->nil", matrices, metric, matrices)
    return np.stack(
        [transformed[:, 0, 0], transformed[:, 0, 1], transformed[:, 1, 1]], axis=1
    )


def _coincidence_index(top: np.ndarray, bottom: np.ndarray) -> int:
    """Count the primitive coincidence cells inside one reported supercell.

    Brute force over the finite group ``M^-1 Z^2 / Z^2``: a coset representative
    ``x`` is a coincidence point when both ``M x`` and ``N x`` are integral.
    """

    denominator = int(round(abs(np.linalg.det(top))))
    if denominator == 0:
        return 0
    grid = np.arange(denominator)
    points = (
        np.stack(np.meshgrid(grid, grid, indexing="ij"), axis=-1).reshape(-1, 2)
        / float(denominator)
    )
    mapped_top = points @ np.asarray(top, dtype=float).T
    mapped_bottom = points @ np.asarray(bottom, dtype=float).T
    integral = np.all(np.abs(mapped_top - np.round(mapped_top)) <= 1e-9, axis=1)
    integral &= np.all(np.abs(mapped_bottom - np.round(mapped_bottom)) <= 1e-9, axis=1)
    return int(np.count_nonzero(integral))


def _cartesian_angle(basis: np.ndarray, element: np.ndarray) -> float:
    cartesian = basis @ np.asarray(element, dtype=float) @ np.linalg.inv(basis)
    return math.atan2(float(cartesian[1, 0]), float(cartesian[0, 0]))


def _fold_offsets(
    top_basis: np.ndarray,
    bottom_basis: np.ndarray,
    top_group: Iterable[np.ndarray],
    bottom_group: Iterable[np.ndarray],
) -> list[tuple[float, float]]:
    """Return ``(sign, offset)`` pairs generating the twist-angle equivalences."""

    def _split(group: Iterable[np.ndarray]) -> tuple[list, list]:
        proper, improper = [], []
        for element in group:
            array = np.asarray(element, dtype=np.int64)
            determinant = int(array[0, 0] * array[1, 1] - array[0, 1] * array[1, 0])
            (proper if determinant > 0 else improper).append(array)
        return proper, improper

    top_proper, top_improper = _split(top_group)
    bottom_proper, bottom_improper = _split(bottom_group)
    offsets = []
    for left in top_proper:
        for right in bottom_proper:
            offsets.append(
                (
                    1.0,
                    _cartesian_angle(bottom_basis, right)
                    - _cartesian_angle(top_basis, left),
                )
            )
    for left in top_improper:
        for right in bottom_improper:
            offsets.append(
                (
                    -1.0,
                    _cartesian_angle(bottom_basis, right)
                    - _cartesian_angle(top_basis, left),
                )
            )
    return offsets


def _fold_angle(angle: float, offsets: Sequence[tuple[float, float]]) -> float:
    """Return the smallest equivalent twist angle, preferring the positive sign.

    When a layer pair admits a mirror the angles ``+theta`` and ``-theta``
    describe the same bilayer, so the representative has to be pinned down by a
    convention; the positive one is reported.
    """

    best = None
    best_score = None
    for sign, offset in offsets:
        candidate = sign * angle + offset
        candidate -= 2.0 * math.pi * round(candidate / (2.0 * math.pi))
        score = abs(candidate) - 1e-12 * (1.0 if candidate > 0.0 else -1.0 if candidate < 0.0 else 0.0)
        if best_score is None or score < best_score:
            best, best_score = candidate, score
    return float(best if best is not None else angle)


def _identity_group() -> list[np.ndarray]:
    return [np.eye(2, dtype=np.int64)]


def reference_search(config: ReferenceConfig) -> list[ReferenceCandidate]:
    """Return every physically distinct commensurate class, by brute force."""

    top_basis = np.asarray(config.top_basis, dtype=float)
    bottom_basis = np.asarray(config.bottom_basis, dtype=float)
    budget = float(config.top_strain) + float(config.bottom_strain)
    lower, upper = math.exp(-2.0 * budget), math.exp(2.0 * budget)

    top_metric = top_basis.T @ top_basis
    bottom_metric = bottom_basis.T @ bottom_basis

    top_pairs = _right_handed_pairs(_short_vectors(top_basis, config.max_length))
    bottom_pairs = _right_handed_pairs(
        _short_vectors(bottom_basis, math.exp(budget) * config.max_length)
    )
    if len(top_pairs) == 0 or len(bottom_pairs) == 0:
        return []

    top_gram = _grams(top_pairs, top_metric)
    bottom_gram = _grams(bottom_pairs, bottom_metric)

    # Report each top sublattice through its Lagrange-reduced right-handed basis.
    reduced = (top_gram[:, 0] <= top_gram[:, 2] * (1.0 + 1e-12)) & (
        2.0 * np.abs(top_gram[:, 1]) <= top_gram[:, 0] * (1.0 + 1e-12)
    )
    aspect = top_gram[:, 2] <= (
        config.max_aspect_ratio**2 * top_gram[:, 0] * (1.0 + 1e-12)
    )
    cosine = top_gram[:, 1] / np.sqrt(top_gram[:, 0] * top_gram[:, 2])
    shape = (cosine >= math.cos(math.radians(config.max_cell_angle_deg)) - 1e-12) & (
        cosine <= math.cos(math.radians(config.min_cell_angle_deg)) + 1e-12
    )
    keep = reduced & aspect & shape
    if config.min_length is not None:
        keep &= top_gram[:, 0] >= config.min_length**2 * (1.0 - 1e-12)
    top_pairs, top_gram = top_pairs[keep], top_gram[keep]
    if len(top_pairs) == 0:
        return []

    top_index = np.abs(
        top_pairs[:, 0, 0] * top_pairs[:, 1, 1] - top_pairs[:, 0, 1] * top_pairs[:, 1, 0]
    )
    bottom_index = np.abs(
        bottom_pairs[:, 0, 0] * bottom_pairs[:, 1, 1]
        - bottom_pairs[:, 0, 1] * bottom_pairs[:, 1, 0]
    )

    offsets = _fold_offsets(
        top_basis,
        bottom_basis,
        _identity_group() if config.top_group is None else config.top_group,
        _identity_group() if config.bottom_group is None else config.bottom_group,
    )

    seen: dict[tuple, ReferenceCandidate] = {}
    for row in range(len(top_pairs)):
        p11, p12, p22 = top_gram[row]
        q11, q12, q22 = bottom_gram[:, 0], bottom_gram[:, 1], bottom_gram[:, 2]
        a11, a12, a22 = q11 - lower * p11, q12 - lower * p12, q22 - lower * p22
        b11, b12, b22 = upper * p11 - q11, upper * p12 - q12, upper * p22 - q22
        accepted = (
            (a11 + a22 >= 0.0)
            & (a11 * a22 - a12 * a12 >= 0.0)
            & (b11 + b22 >= 0.0)
            & (b11 * b22 - b12 * b12 >= 0.0)
        )
        if config.max_atoms is not None:
            accepted &= (
                top_index[row] * config.top_atoms + bottom_index * config.bottom_atoms
                <= config.max_atoms
            )
        for column in np.nonzero(accepted)[0]:
            top_matrix = top_pairs[row]
            bottom_matrix = bottom_pairs[column]
            if config.primitive_only and _coincidence_index(top_matrix, bottom_matrix) != 1:
                continue
            top_cell = top_basis @ top_matrix
            bottom_cell = bottom_basis @ bottom_matrix
            deformation = bottom_cell @ np.linalg.inv(top_cell)
            left, stretches, right = np.linalg.svd(deformation)
            rotation = left @ right
            angle = math.atan2(float(rotation[1, 0]), float(rotation[0, 0]))
            strains = np.sort(np.log(stretches))
            folded = _fold_angle(angle, offsets)
            candidate = ReferenceCandidate(
                twist_deg=math.degrees(folded),
                top_atoms=int(top_index[row]) * int(config.top_atoms),
                bottom_atoms=int(bottom_index[column]) * int(config.bottom_atoms),
                strains=(float(strains[0]), float(strains[1])),
                top_area=float(abs(np.linalg.det(top_cell))),
            )
            key = (
                round(candidate.twist_deg, 6),
                candidate.top_atoms,
                candidate.bottom_atoms,
                round(candidate.strains[0], 9),
                round(candidate.strains[1], 9),
                round(candidate.top_area, 6),
            )
            seen.setdefault(key, candidate)
    return sorted(
        seen.values(),
        key=lambda item: (
            item.top_atoms + item.bottom_atoms,
            abs(item.twist_deg),
            item.strains,
        ),
    )
