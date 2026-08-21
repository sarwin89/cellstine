"""Native vectorised Gram-form search for commensurate bilayer supercells.

The implementation is a direct adaptation of the optimized Aristotle progression:
Lagrange--Gauss gauge reduction, metric vector shells, reduced-basis arc generation,
proper point-group folding, a four-probe bucketed Gram join, and a closed-form finish.

``top_strain`` and ``bottom_strain`` are absolute bounds on principal logarithmic
(Hencky) strain.  A budget ``e`` therefore permits principal stretches in
``[exp(-e), exp(e)]``; it is not an engineering-strain percentage.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

__all__ = [
    "SearchConfig",
    "SearchResult",
    "SymmetricBranchUnavailable",
    "search",
    "symmetric_branch_applies",
]

_TWO_PI = 2.0 * np.pi
_REL = 1e-9
_SLACK = 1e-9
_SHELL_RATIO = 1.6
_JOIN_CHUNK = 2048
_CERTIFICATION_MARGIN = 1e-10


class SymmetricBranchUnavailable(ValueError):
    """Raised when the restricted square/hexagonal search cannot be used."""


def _validated_basis(value: np.ndarray, name: str) -> np.ndarray:
    basis = np.asarray(value, dtype=float)
    if basis.shape != (2, 2):
        raise ValueError(f"{name} must be a 2x2 Cartesian column basis")
    if not np.all(np.isfinite(basis)):
        raise ValueError(f"{name} must contain only finite values")
    determinant = float(np.linalg.det(basis))
    scale = max(float(np.linalg.norm(basis, ord=np.inf)) ** 2, 1.0)
    if abs(determinant) <= 64.0 * np.finfo(float).eps * scale:
        raise ValueError(f"{name} must be nonsingular")
    validated = np.array(basis, copy=True)
    validated.setflags(write=False)
    return validated


def _positive_integer(value: int, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    if int(value) <= 0:
        raise ValueError(f"{name} must be positive")
    return int(value)


@dataclass(frozen=True)
class SearchConfig:
    """Physical limits for a two-layer Gram-form search.

    Bases are finite nonsingular 2x2 Cartesian column bases.  Strain values are
    nonnegative principal logarithmic-strain budgets, and at least one must be positive.
    """

    top_basis: np.ndarray
    bottom_basis: np.ndarray
    max_length: float
    top_strain: float
    bottom_strain: float
    min_length: float | None = None
    max_atoms: int | None = None
    top_atoms: int = 1
    bottom_atoms: int = 1
    max_aspect_ratio: float = 12.0
    min_cell_angle_deg: float = 25.0
    max_cell_angle_deg: float = 155.0
    fold_symmetry: bool = True
    symmetric: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "top_basis", _validated_basis(self.top_basis, "top_basis"))
        object.__setattr__(
            self, "bottom_basis", _validated_basis(self.bottom_basis, "bottom_basis")
        )
        numeric = {
            "max_length": self.max_length,
            "top_strain": self.top_strain,
            "bottom_strain": self.bottom_strain,
            "max_aspect_ratio": self.max_aspect_ratio,
            "min_cell_angle_deg": self.min_cell_angle_deg,
            "max_cell_angle_deg": self.max_cell_angle_deg,
        }
        if any(not np.isfinite(float(value)) for value in numeric.values()):
            raise ValueError("search limits and strain budgets must be finite")
        if self.max_length <= 0.0:
            raise ValueError("max_length must be positive")
        if self.top_strain < 0.0 or self.bottom_strain < 0.0:
            raise ValueError("strain budgets must be nonnegative")
        if self.top_strain + self.bottom_strain <= 0.0:
            raise ValueError("at least one strain budget must be positive")
        if self.min_length is not None:
            if not np.isfinite(float(self.min_length)) or self.min_length <= 0.0:
                raise ValueError("min_length must be finite and positive")
            if self.min_length > self.max_length:
                raise ValueError("min_length cannot exceed max_length")
        if self.max_atoms is not None:
            object.__setattr__(self, "max_atoms", _positive_integer(self.max_atoms, "max_atoms"))
        object.__setattr__(self, "top_atoms", _positive_integer(self.top_atoms, "top_atoms"))
        object.__setattr__(
            self, "bottom_atoms", _positive_integer(self.bottom_atoms, "bottom_atoms")
        )
        if self.max_aspect_ratio < 1.0:
            raise ValueError("max_aspect_ratio must be at least one")
        if not 0.0 < self.min_cell_angle_deg < self.max_cell_angle_deg < 180.0:
            raise ValueError("cell-angle limits must satisfy 0 < min < max < 180")

    @property
    def _budget(self) -> float:
        return float(self.top_strain + self.bottom_strain)

    @property
    def _band(self) -> tuple[float, float]:
        return math.exp(-2.0 * self._budget), math.exp(2.0 * self._budget)


@dataclass(frozen=True)
class SearchResult:
    """Parallel arrays describing deterministic canonical candidate classes.

    ``principal_strains`` stores the two principal relative logarithmic strains.  The
    layer strains are obtained by multiplying by ``sharing_fraction`` for the top and by
    ``sharing_fraction - 1`` for the bottom.
    """

    top_matrices: np.ndarray
    bottom_matrices: np.ndarray
    top_gram: np.ndarray
    bottom_gram: np.ndarray
    twist_radians: np.ndarray
    twist_degrees: np.ndarray
    principal_strains: np.ndarray
    sharing_fraction: np.ndarray
    top_atom_counts: np.ndarray
    bottom_atom_counts: np.ndarray
    atom_counts: np.ndarray
    loewner_certified: np.ndarray
    loewner_borderline: np.ndarray
    top_affine: np.ndarray
    bottom_affine: np.ndarray
    shared_lattice: np.ndarray
    canonical_keys: np.ndarray
    pareto_optimal: np.ndarray
    rank: np.ndarray
    stats: dict[str, Any]

    def __len__(self) -> int:
        return int(self.top_matrices.shape[0])


def _gram_of_basis(basis: np.ndarray) -> np.ndarray:
    return basis.T @ basis


def _reduce_basis(basis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Lagrange--Gauss reduce columns, returning ``(basis @ gauge, gauge)``."""
    reduced = np.array(basis, dtype=float, copy=True)
    gauge = np.eye(2, dtype=np.int64)
    for _ in range(64):
        first_norm = float(reduced[:, 0] @ reduced[:, 0])
        second_norm = float(reduced[:, 1] @ reduced[:, 1])
        if first_norm > second_norm:
            reduced = reduced[:, ::-1].copy()
            gauge = gauge[:, ::-1].copy()
            first_norm = second_norm
        multiplier = int(np.round(float(reduced[:, 0] @ reduced[:, 1]) / first_norm))
        if multiplier == 0:
            break
        reduced[:, 1] -= multiplier * reduced[:, 0]
        gauge[:, 1] -= multiplier * gauge[:, 0]
    else:
        raise ArithmeticError("Lagrange--Gauss basis reduction did not converge")
    if np.linalg.det(reduced) < 0.0:
        reduced[:, 1] *= -1.0
        gauge[:, 1] *= -1
    return reduced, gauge


def _lattice_vectors(metric: np.ndarray, radius_squared: float):
    g11, g12, g22 = float(metric[0, 0]), float(metric[0, 1]), float(metric[1, 1])
    determinant = g11 * g22 - g12 * g12
    m_max = int(np.floor(np.sqrt(radius_squared * g22 / determinant))) + 1
    n_max = int(np.floor(np.sqrt(radius_squared * g11 / determinant))) + 1
    m, n = np.meshgrid(
        np.arange(-m_max, m_max + 1, dtype=np.int64),
        np.arange(-n_max, n_max + 1, dtype=np.int64),
        indexing="ij",
    )
    m, n = m.ravel(), n.ravel()
    squared = g11 * m * m + 2.0 * g12 * m * n + g22 * n * n
    keep = (squared <= radius_squared) & ~((m == 0) & (n == 0))
    order = np.argsort(squared[keep], kind="stable")
    return np.stack([m[keep][order], n[keep][order]], axis=1), squared[keep][order]


def _gram_triples(metric: np.ndarray, first: np.ndarray, second: np.ndarray):
    g11, g12, g22 = float(metric[0, 0]), float(metric[0, 1]), float(metric[1, 1])

    def bilinear(left, right):
        return (
            g11 * left[:, 0] * right[:, 0]
            + g12 * (left[:, 0] * right[:, 1] + left[:, 1] * right[:, 0])
            + g22 * left[:, 1] * right[:, 1]
        )

    return bilinear(first, first), bilinear(first, second), bilinear(second, second)


def _point_group(metric: np.ndarray, tolerance: float = 1e-9) -> np.ndarray:
    g11, g12, g22 = float(metric[0, 0]), float(metric[0, 1]), float(metric[1, 1])
    scale = max(g11, g22)
    vectors, squared = _lattice_vectors(
        metric, scale * (1.0 + 10.0 * tolerance + 1e-12)
    )
    first = np.nonzero(np.abs(squared - g11) <= tolerance * scale)[0]
    second = np.nonzero(np.abs(squared - g22) <= tolerance * scale)[0]
    if first.size == 0 or second.size == 0:
        return np.eye(2, dtype=np.int64)[None, :, :]
    left_index, right_index = np.meshgrid(first, second, indexing="ij")
    left, right = vectors[left_index.ravel()], vectors[right_index.ravel()]
    cross = (
        g11 * left[:, 0] * right[:, 0]
        + g12 * (left[:, 0] * right[:, 1] + left[:, 1] * right[:, 0])
        + g22 * left[:, 1] * right[:, 1]
    )
    determinant = left[:, 0] * right[:, 1] - left[:, 1] * right[:, 0]
    keep = (np.abs(cross - g12) <= tolerance * scale) & (np.abs(determinant) == 1)
    return np.stack([left[keep], right[keep]], axis=2).astype(np.int64)


def _proper_subgroup(group: np.ndarray) -> np.ndarray:
    determinant = group[:, 0, 0] * group[:, 1, 1] - group[:, 0, 1] * group[:, 1, 0]
    return group[determinant > 0]


def _bezout(left: np.ndarray, right: np.ndarray):
    old_r, r = left.astype(np.int64).copy(), right.astype(np.int64).copy()
    old_s, s = np.ones_like(old_r), np.zeros_like(old_r)
    old_t, t = np.zeros_like(old_r), np.ones_like(old_r)
    while np.any(r != 0):
        active = r != 0
        quotient = np.zeros_like(r)
        quotient[active] = old_r[active] // r[active]
        old_r, r = np.where(active, r, old_r), np.where(
            active, old_r - quotient * r, r
        )
        old_s, s = np.where(active, s, old_s), np.where(
            active, old_s - quotient * s, s
        )
        old_t, t = np.where(active, t, old_t), np.where(
            active, old_t - quotient * t, t
        )
    sign = np.where(old_r < 0, -1, 1)
    return old_s * sign, old_t * sign


def _hermite_normal_form(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    a, c = first[:, 0].astype(np.int64), first[:, 1].astype(np.int64)
    b, d = second[:, 0].astype(np.int64), second[:, 1].astype(np.int64)
    determinant = np.abs(a * d - b * c)
    h22 = np.gcd(c, d)
    h11 = determinant // np.where(h22 == 0, 1, h22)
    x, y = _bezout(c, d)
    h12 = (x * a + y * b) % np.where(h11 == 0, 1, h11)
    return np.stack([h11, h12, h22], axis=1)


def _lexicographic_minimum(stack: np.ndarray) -> np.ndarray:
    best = stack[0].copy()
    for candidate in stack[1:]:
        less = np.zeros(best.shape[0], dtype=bool)
        equal = np.ones(best.shape[0], dtype=bool)
        for column in range(best.shape[1]):
            less |= equal & (candidate[:, column] < best[:, column])
            equal &= candidate[:, column] == best[:, column]
        best = np.where(less[:, None], candidate, best)
    return best


def _first_per_key(keys: np.ndarray) -> np.ndarray:
    if len(keys) == 0:
        return np.zeros(0, dtype=np.int64)
    contiguous = np.ascontiguousarray(keys)
    view = contiguous.view(np.dtype((np.void, keys.dtype.itemsize * keys.shape[1]))).ravel()
    _, first = np.unique(view, return_index=True)
    return np.sort(first)


def _vector_orbit_representatives(vectors: np.ndarray, group: np.ndarray) -> np.ndarray:
    if len(vectors) == 0 or len(group) <= 1:
        return np.ones(len(vectors), dtype=bool)
    best = _lexicographic_minimum(np.stack([vectors @ symmetry.T for symmetry in group]))
    return np.all(best == vectors, axis=1)


def _fold_sublattices(first: np.ndarray, second: np.ndarray, group: np.ndarray):
    if len(first) == 0 or len(group) <= 1:
        return np.arange(len(first))
    keys = np.stack(
        [_hermite_normal_form(first @ symmetry.T, second @ symmetry.T) for symmetry in group]
    )
    return _first_per_key(_lexicographic_minimum(keys))


def _fold_bases(first: np.ndarray, second: np.ndarray, group: np.ndarray):
    if len(first) == 0 or len(group) <= 1:
        return np.arange(len(first))
    keys = np.stack(
        [np.concatenate([first @ symmetry.T, second @ symmetry.T], axis=1) for symmetry in group]
    )
    return _first_per_key(_lexicographic_minimum(keys))


@dataclass
class _VectorTable:
    vectors: np.ndarray
    squared: np.ndarray
    angles: np.ndarray
    shell_indices: list[np.ndarray]
    shell_keys: list[np.ndarray]
    shell_low: np.ndarray
    shell_high: np.ndarray

    @property
    def shell_count(self) -> int:
        return len(self.shell_indices)


def _vector_table(metric: np.ndarray, basis: np.ndarray, radius_squared: float) -> _VectorTable:
    vectors, squared = _lattice_vectors(metric, radius_squared)
    if len(vectors) == 0:
        return _VectorTable(
            vectors,
            squared,
            np.zeros(0),
            [],
            [],
            np.zeros(0),
            np.zeros(0),
        )
    cartesian = basis @ vectors.T.astype(float)
    angles = np.mod(np.arctan2(cartesian[1], cartesian[0]), _TWO_PI)
    low, high = float(squared[0]), float(squared[-1])
    shell_count = max(1, int(np.ceil(np.log(high / low) / np.log(_SHELL_RATIO))))
    edges = low * _SHELL_RATIO ** np.arange(shell_count + 1)
    edges[0] = 0.0
    edges[-1] = max(edges[-1], high) * (1.0 + 1e-12)
    shell_id = np.clip(
        np.searchsorted(edges, squared, side="right") - 1, 0, shell_count - 1
    )
    starts = np.searchsorted(shell_id, np.arange(shell_count + 1))
    order = np.lexsort((angles, shell_id))
    shell_indices: list[np.ndarray] = []
    shell_keys: list[np.ndarray] = []
    shell_low = np.zeros(shell_count)
    shell_high = np.zeros(shell_count)
    for shell in range(shell_count):
        indices = order[starts[shell] : starts[shell + 1]]
        shell_indices.append(indices)
        shell_angles = angles[indices]
        shell_keys.append(np.concatenate([shell_angles, shell_angles + _TWO_PI]))
        if len(indices):
            shell_low[shell] = float(squared[indices].min())
            shell_high[shell] = float(squared[indices].max())
        else:
            shell_low[shell], shell_high[shell] = np.inf, -np.inf
    return _VectorTable(
        vectors, squared, angles, shell_indices, shell_keys, shell_low, shell_high
    )


@dataclass
class _Table:
    first: np.ndarray
    second: np.ndarray
    g11: np.ndarray
    g12: np.ndarray
    g22: np.ndarray
    index: np.ndarray

    def __len__(self) -> int:
        return int(self.first.shape[0])

    def take(self, selection) -> _Table:
        return _Table(
            self.first[selection],
            self.second[selection],
            self.g11[selection],
            self.g12[selection],
            self.g22[selection],
            self.index[selection],
        )


def _empty_table() -> _Table:
    integers = np.zeros((0, 2), dtype=np.int64)
    floats = np.zeros(0)
    return _Table(integers, integers, floats, floats, floats, np.zeros(0, dtype=np.int64))


def _expand(low: np.ndarray, high: np.ndarray, query_indices: np.ndarray):
    counts = high - low
    total = int(counts.sum())
    if total == 0:
        empty = np.zeros(0, dtype=np.int64)
        return empty, empty
    starts = np.repeat(low, counts)
    base = np.repeat(np.cumsum(counts) - counts, counts)
    return np.repeat(query_indices, counts), starts + (np.arange(total) - base)


def _basis_table(
    table: _VectorTable,
    metric: np.ndarray,
    *,
    partner: bool,
    lower: float = 1.0,
    upper: float = 1.0,
    first_indices: np.ndarray | None = None,
    area_squared_max: float | None = None,
) -> _Table:
    """Generate reduced/partner bases through shell-local angular arc lookup."""
    g11m, g12m, g22m = float(metric[0, 0]), float(metric[0, 1]), float(metric[1, 1])
    vectors, squared, angles = table.vectors, table.squared, table.angles
    first = (
        np.arange(len(vectors), dtype=np.int64)
        if first_indices is None
        else np.asarray(first_indices, dtype=np.int64)
    )
    if first.size == 0:
        return _empty_table()
    first_squared, first_angles = squared[first], angles[first]
    ratio_low = lower / upper if partner else 1.0
    outputs = []
    for shell in range(table.shell_count):
        indices = table.shell_indices[shell]
        if indices.size == 0:
            continue
        shell_low, shell_high = table.shell_low[shell], table.shell_high[shell]
        first_max = shell_high / ratio_low
        if area_squared_max is not None and not partner:
            first_max = min(first_max, 4.0 * area_squared_max / (3.0 * shell_low))
        count = int(
            np.searchsorted(first_squared, first_max * (1.0 + _REL), side="right")
        )
        if count == 0:
            continue
        p = first_squared[:count]
        if partner:
            half_width = (
                (upper - lower) * (p + shell_high) + upper * p
            ) / (2.0 * lower * np.sqrt(p * shell_low))
        else:
            half_width = 0.5 * np.sqrt(p / shell_low)
        delta = np.arcsin(np.clip(half_width * (1.0 + _REL), 0.0, 1.0)) + 1e-12
        centre = first_angles[:count] + 0.5 * np.pi
        centre = np.where(centre - delta < 0.0, centre + _TWO_PI, centre)
        key = table.shell_keys[shell]
        low = np.searchsorted(key, centre - delta, side="left")
        high = np.searchsorted(key, centre + delta, side="right")
        top_index, position = _expand(low, high, np.arange(count))
        if top_index.size == 0:
            continue
        second_index = indices[position % indices.size]
        first_vectors, second_vectors = vectors[first[top_index]], vectors[second_index]
        determinant = (
            first_vectors[:, 0] * second_vectors[:, 1]
            - first_vectors[:, 1] * second_vectors[:, 0]
        )
        keep = determinant > 0
        if not np.any(keep):
            continue
        first_vectors, second_vectors = first_vectors[keep], second_vectors[keep]
        determinant = determinant[keep]
        top_index, second_index = top_index[keep], second_index[keep]
        g11, g22 = first_squared[top_index], squared[second_index]
        g12 = (
            g11m * first_vectors[:, 0] * second_vectors[:, 0]
            + g12m
            * (
                first_vectors[:, 0] * second_vectors[:, 1]
                + first_vectors[:, 1] * second_vectors[:, 0]
            )
            + g22m * first_vectors[:, 1] * second_vectors[:, 1]
        )
        if partner:
            selected = (lower * g11 <= upper * g22 * (1.0 + _REL)) & (
                2.0 * lower * np.abs(g12)
                <= ((upper - lower) * (g11 + g22) + upper * g11) * (1.0 + _REL)
            )
        else:
            selected = (g11 <= g22 * (1.0 + _REL)) & (
                2.0 * np.abs(g12) <= g11 * (1.0 + _REL)
            )
        if np.any(selected):
            outputs.append(
                (
                    first_vectors[selected],
                    second_vectors[selected],
                    g11[selected],
                    g12[selected],
                    g22[selected],
                    determinant[selected],
                )
            )
    if not outputs:
        return _empty_table()
    return _Table(*[np.concatenate([output[column] for output in outputs]) for column in range(6)])


class _BottomIndex:
    """Partner rows bucketed by both diagonal Gram entries and sorted by off-diagonal."""

    def __init__(self, bottom: _Table, lower: float, upper: float):
        self.width = np.log(upper / lower)
        bucket11 = np.floor(np.log(bottom.g11) / self.width).astype(np.int64)
        bucket22 = np.floor(np.log(bottom.g22) / self.width).astype(np.int64)
        self.min11 = int(bucket11.min()) - 1
        self.min22 = int(bucket22.min()) - 1
        self.count22 = int(bucket22.max()) - self.min22 + 2
        composite = (bucket11 - self.min11) * self.count22 + (bucket22 - self.min22)
        self.order = np.lexsort((bottom.g12, composite))
        self.g11 = bottom.g11[self.order]
        self.g12 = bottom.g12[self.order]
        self.g22 = bottom.g22[self.order]
        self.scale = 4.0 * (float(np.abs(bottom.g12).max()) + 1.0)
        self.key = composite[self.order].astype(float) * self.scale + self.g12
        self.max11 = int(bucket11.max())
        self.max22 = int(bucket22.max())

    def composite(self, bucket11, bucket22):
        return ((bucket11 - self.min11) * self.count22 + (bucket22 - self.min22)).astype(
            float
        )


def _loewner_mask(p11, p12, p22, q11, q12, q22, lower, upper, tolerance=0.0):
    """Four-inequality Löwner acceptance for scalar or array Gram triples."""
    a11, a12, a22 = q11 - lower * p11, q12 - lower * p12, q22 - lower * p22
    b11, b12, b22 = upper * p11 - q11, upper * p12 - q12, upper * p22 - q22
    return (
        (a11 + a22 >= -tolerance)
        & (a11 * a22 - a12 * a12 >= -tolerance)
        & (b11 + b22 >= -tolerance)
        & (b11 * b22 - b12 * b12 >= -tolerance)
    )


def _join_candidates(top: _Table, index: _BottomIndex, lower: float, upper: float):
    """Four-probe bucket join with bucket-local off-diagonal windows."""
    output_top, output_bottom = [], []
    width = index.width
    bucket_top11 = np.floor((np.log(top.g11) + np.log(lower)) / width).astype(np.int64)
    bucket_top22 = np.floor((np.log(top.g22) + np.log(lower)) / width).astype(np.int64)
    for start in range(0, len(top), _JOIN_CHUNK):
        stop = min(start + _JOIN_CHUNK, len(top))
        selection = slice(start, stop)
        p11, p12, p22 = top.g11[selection], top.g12[selection], top.g22[selection]
        top_indices = np.arange(start, stop)
        for offset11 in (0, 1):
            for offset22 in (0, 1):
                bucket11 = bucket_top11[selection] + offset11
                bucket22 = bucket_top22[selection] + offset22
                valid = (
                    (bucket11 >= index.min11)
                    & (bucket11 <= index.max11)
                    & (bucket22 >= index.min22)
                    & (bucket22 <= index.max22)
                )
                q11_low = np.maximum(lower * p11, np.exp(width * bucket11))
                q11_high = np.minimum(upper * p11, np.exp(width * (bucket11 + 1)))
                q22_low = np.maximum(lower * p22, np.exp(width * bucket22))
                q22_high = np.minimum(upper * p22, np.exp(width * (bucket22 + 1)))
                low_radius = np.sqrt(
                    np.maximum(q11_high - lower * p11, 0.0)
                    * np.maximum(q22_high - lower * p22, 0.0)
                )
                high_radius = np.sqrt(
                    np.maximum(upper * p11 - q11_low, 0.0)
                    * np.maximum(upper * p22 - q22_low, 0.0)
                )
                padding = _SLACK * (1.0 + np.abs(p12))
                low12 = np.maximum(
                    lower * p12 - low_radius, upper * p12 - high_radius
                ) - padding
                high12 = np.minimum(
                    lower * p12 + low_radius, upper * p12 + high_radius
                ) + padding
                valid &= (
                    (q11_low <= q11_high)
                    & (q22_low <= q22_high)
                    & (low12 <= high12)
                )
                base = index.composite(bucket11, bucket22) * index.scale
                low = np.searchsorted(index.key, base + low12, side="left")
                high = np.searchsorted(index.key, base + high12, side="right")
                high = np.where(valid, high, low)
                top_rows, bottom_rows = _expand(low, high, top_indices)
                if top_rows.size == 0:
                    continue
                top11, bottom11 = top.g11[top_rows], index.g11[bottom_rows]
                keep = (bottom11 >= lower * top11 * (1.0 - _SLACK)) & (
                    bottom11 <= upper * top11 * (1.0 + _SLACK)
                )
                if not np.any(keep):
                    continue
                top_rows, bottom_rows = top_rows[keep], bottom_rows[keep]
                top11, bottom11 = top11[keep], bottom11[keep]
                top22, bottom22 = top.g22[top_rows], index.g22[bottom_rows]
                keep = (bottom22 >= lower * top22 * (1.0 - _SLACK)) & (
                    bottom22 <= upper * top22 * (1.0 + _SLACK)
                )
                if not np.any(keep):
                    continue
                top_rows, bottom_rows = top_rows[keep], bottom_rows[keep]
                top11, bottom11 = top11[keep], bottom11[keep]
                top22, bottom22 = top22[keep], bottom22[keep]
                top12, bottom12 = top.g12[top_rows], index.g12[bottom_rows]
                tolerance = 1e-9 * top11 * top22
                keep = _loewner_mask(
                    top11,
                    top12,
                    top22,
                    bottom11,
                    bottom12,
                    bottom22,
                    lower,
                    upper,
                    tolerance,
                )
                if np.any(keep):
                    output_top.append(top_rows[keep])
                    output_bottom.append(bottom_rows[keep])
    if not output_top:
        empty = np.zeros(0, dtype=np.int64)
        return empty, empty
    return np.concatenate(output_top), np.concatenate(output_bottom)


def _stretches_from_gram(p11, p12, p22, q11, q12, q22):
    """Principal stretches and relative logarithmic strains from two Gram forms."""
    determinant_p = p11 * p22 - p12 * p12
    determinant_q = q11 * q22 - q12 * q12
    trace = (p22 * q11 - 2.0 * p12 * q12 + p11 * q22) / determinant_p
    product = np.sqrt(np.maximum(determinant_q / determinant_p, 0.0))
    stretch_sum = np.sqrt(np.maximum(trace + 2.0 * product, 0.0))
    root = np.sqrt(np.maximum(stretch_sum * stretch_sum - 4.0 * product, 0.0))
    first = 0.5 * (stretch_sum + root)
    second = 0.5 * (stretch_sum - root)
    return first, second, np.log(first), np.log(second)


def _twist_angles(
    top_basis: np.ndarray,
    bottom_basis: np.ndarray,
    top_first: np.ndarray,
    top_second: np.ndarray,
    bottom_first: np.ndarray,
    bottom_second: np.ndarray,
) -> np.ndarray:
    """Polar-rotation angles from the integer adjugate expression, without forming F."""
    adjugate_top = np.array(
        [
            [top_basis[1, 1], -top_basis[0, 1]],
            [-top_basis[1, 0], top_basis[0, 0]],
        ]
    )
    trace_coefficients = adjugate_top @ bottom_basis
    skew_coefficients = (
        adjugate_top @ np.array([[0.0, 1.0], [-1.0, 0.0]]) @ bottom_basis
    )
    k11 = bottom_first[:, 0] * top_second[:, 1] - bottom_second[:, 0] * top_first[:, 1]
    k12 = bottom_second[:, 0] * top_first[:, 0] - bottom_first[:, 0] * top_second[:, 0]
    k21 = bottom_first[:, 1] * top_second[:, 1] - bottom_second[:, 1] * top_first[:, 1]
    k22 = bottom_second[:, 1] * top_first[:, 0] - bottom_first[:, 1] * top_second[:, 0]
    trace = (
        trace_coefficients[0, 0] * k11
        + trace_coefficients[0, 1] * k21
        + trace_coefficients[1, 0] * k12
        + trace_coefficients[1, 1] * k22
    )
    skew = (
        skew_coefficients[0, 0] * k11
        + skew_coefficients[0, 1] * k21
        + skew_coefficients[1, 0] * k12
        + skew_coefficients[1, 1] * k22
    )
    return np.arctan2(skew, trace)


def _pareto_front(first_cost: np.ndarray, second_cost: np.ndarray) -> np.ndarray:
    """Deterministic indices of strict Pareto records in increasing first-cost order."""
    order = np.lexsort((np.arange(len(first_cost)), second_cost, first_cost))
    sorted_second = second_cost[order]
    previous_best = np.minimum.accumulate(np.concatenate([[np.inf], sorted_second[:-1]]))
    return order[sorted_second < previous_best]


def _canonical_pair_keys(top_matrices: np.ndarray, bottom_matrices: np.ndarray) -> np.ndarray:
    """Exact class keys modulo the common right action by unimodular matrices.

    The top matrix is sent to its column HNF and the identical right transform is applied
    to the bottom matrix.  This is an integer-only canonicalization, unlike rounded
    floating-point geometry keys.
    """
    count = len(top_matrices)
    if count == 0:
        return np.zeros((0, 8), dtype=np.int64)
    top_first, top_second = top_matrices[:, :, 0], top_matrices[:, :, 1]
    triples = _hermite_normal_form(top_first, top_second)
    hnf = np.zeros((count, 2, 2), dtype=np.int64)
    hnf[:, 0, 0], hnf[:, 0, 1], hnf[:, 1, 1] = (
        triples[:, 0],
        triples[:, 1],
        triples[:, 2],
    )
    determinant = (
        top_matrices[:, 0, 0] * top_matrices[:, 1, 1]
        - top_matrices[:, 0, 1] * top_matrices[:, 1, 0]
    )
    adjugate = np.empty_like(top_matrices)
    adjugate[:, 0, 0] = top_matrices[:, 1, 1]
    adjugate[:, 0, 1] = -top_matrices[:, 0, 1]
    adjugate[:, 1, 0] = -top_matrices[:, 1, 0]
    adjugate[:, 1, 1] = top_matrices[:, 0, 0]
    numerator = adjugate @ hnf
    if np.any(numerator % determinant[:, None, None]):
        raise ArithmeticError("column-HNF transform was not unimodular integral")
    transform = numerator // determinant[:, None, None]
    canonical_bottom = bottom_matrices @ transform
    return np.concatenate([hnf.reshape(count, 4), canonical_bottom.reshape(count, 4)], axis=1)


def _shape_mask(table: _Table, config: SearchConfig) -> np.ndarray:
    keep = table.g22 <= (
        config.max_aspect_ratio * config.max_aspect_ratio * table.g11 * (1.0 + _REL)
    )
    cosine = table.g12 / np.sqrt(table.g11 * table.g22)
    lower_cosine = math.cos(math.radians(config.max_cell_angle_deg))
    upper_cosine = math.cos(math.radians(config.min_cell_angle_deg))
    keep &= (cosine >= lower_cosine - _REL) & (cosine <= upper_cosine + _REL)
    return keep


def _matrix_power_spd(
    stretch: np.ndarray, first: np.ndarray, second: np.ndarray, power: np.ndarray
) -> np.ndarray:
    """Vectorised analytic power of a symmetric positive-definite 2x2 matrix."""
    first_power = np.power(first, power)
    second_power = np.power(second, power)
    difference = first - second
    near = np.abs(difference) <= 1e-10 * np.maximum(first, 1.0)
    alpha = np.empty_like(first)
    beta = np.empty_like(first)
    alpha[~near] = (first_power[~near] - second_power[~near]) / difference[~near]
    beta[~near] = (
        first[~near] * second_power[~near] - second[~near] * first_power[~near]
    ) / difference[~near]
    common = 0.5 * (first + second)
    alpha[near] = power[near] * np.power(common[near], power[near] - 1.0)
    beta[near] = (1.0 - power[near]) * np.power(common[near], power[near])
    result = alpha[:, None, None] * stretch
    result[:, 0, 0] += beta
    result[:, 1, 1] += beta
    return result


def _affine_geometry(
    config: SearchConfig,
    top_matrices: np.ndarray,
    bottom_matrices: np.ndarray,
    twist: np.ndarray,
    first_stretch: np.ndarray,
    second_stretch: np.ndarray,
    sharing: np.ndarray,
):
    count = len(top_matrices)
    if count == 0:
        empty = np.zeros((0, 2, 2))
        return empty, empty, empty
    top_cells = np.einsum("ij,njk->nik", config.top_basis, top_matrices)
    bottom_cells = np.einsum("ij,njk->nik", config.bottom_basis, bottom_matrices)
    deformation = bottom_cells @ np.linalg.inv(top_cells)
    cosine, sine = np.cos(twist), np.sin(twist)
    rotation = np.empty((count, 2, 2))
    rotation[:, 0, 0], rotation[:, 0, 1] = cosine, -sine
    rotation[:, 1, 0], rotation[:, 1, 1] = sine, cosine
    stretch = np.swapaxes(rotation, 1, 2) @ deformation
    top_power = _matrix_power_spd(stretch, first_stretch, second_stretch, sharing)
    bottom_power = _matrix_power_spd(
        stretch, first_stretch, second_stretch, sharing - 1.0
    )
    top_affine = rotation @ top_power
    bottom_affine = rotation @ bottom_power @ np.swapaxes(rotation, 1, 2)
    shared = top_affine @ top_cells
    return top_affine, bottom_affine, shared


def _finalize(
    config: SearchConfig,
    top_matrices: np.ndarray,
    bottom_matrices: np.ndarray,
    top_gram: np.ndarray,
    bottom_gram: np.ndarray,
    top_multiplicity: np.ndarray,
    bottom_multiplicity: np.ndarray,
    twist: np.ndarray,
    stats: dict[str, Any],
) -> SearchResult:
    finalize_started = time.perf_counter()
    first_stretch, second_stretch, first_strain, second_strain = _stretches_from_gram(
        top_gram[:, 0],
        top_gram[:, 1],
        top_gram[:, 2],
        bottom_gram[:, 0],
        bottom_gram[:, 1],
        bottom_gram[:, 2],
    )
    principal_strains = np.stack([first_strain, second_strain], axis=1)
    sharing = np.full(len(top_matrices), config.top_strain / config._budget)
    lower, upper = config._band
    certified = _loewner_mask(
        top_gram[:, 0],
        top_gram[:, 1],
        top_gram[:, 2],
        bottom_gram[:, 0],
        bottom_gram[:, 1],
        bottom_gram[:, 2],
        lower * (1.0 + _CERTIFICATION_MARGIN),
        upper * (1.0 - _CERTIFICATION_MARGIN),
    )
    top_atom_counts = top_multiplicity.astype(np.int64) * config.top_atoms
    bottom_atom_counts = bottom_multiplicity.astype(np.int64) * config.bottom_atoms
    atom_counts = top_atom_counts + bottom_atom_counts
    keys = _canonical_pair_keys(top_matrices, bottom_matrices)
    strain_cost = (
        np.max(np.abs(principal_strains), axis=1)
        if len(principal_strains)
        else np.zeros(0)
    )
    pareto = np.zeros(len(top_matrices), dtype=bool)
    pareto[_pareto_front(atom_counts.astype(float), strain_cost)] = True
    if len(top_matrices):
        ordering_keys = [keys[:, column] for column in range(keys.shape[1] - 1, -1, -1)]
        order = np.lexsort(
            tuple(
                ordering_keys
                + [np.abs(twist), strain_cost, atom_counts, (~pareto).astype(np.int8)]
            )
        )
    else:
        order = np.zeros(0, dtype=np.int64)
    top_matrices = top_matrices[order]
    bottom_matrices = bottom_matrices[order]
    top_gram = top_gram[order]
    bottom_gram = bottom_gram[order]
    twist = twist[order]
    principal_strains = principal_strains[order]
    first_stretch, second_stretch = first_stretch[order], second_stretch[order]
    sharing = sharing[order]
    top_atom_counts, bottom_atom_counts = top_atom_counts[order], bottom_atom_counts[order]
    atom_counts = atom_counts[order]
    certified, pareto, keys = certified[order], pareto[order], keys[order]
    top_affine, bottom_affine, shared = _affine_geometry(
        config,
        top_matrices,
        bottom_matrices,
        twist,
        first_stretch,
        second_stretch,
        sharing,
    )
    stats = dict(stats)
    finalize_elapsed = time.perf_counter() - finalize_started
    stats["t_finalize"] = finalize_elapsed
    stats["t_finish"] = stats.get("t_finish", 0.0) + finalize_elapsed
    stats["t_total"] = stats.get("t_total", 0.0) + finalize_elapsed
    stats["n_accepted"] = int(len(top_matrices))
    stats["n_pareto"] = int(pareto.sum())
    stats["n_borderline"] = int((~certified).sum())
    return SearchResult(
        top_matrices=top_matrices,
        bottom_matrices=bottom_matrices,
        top_gram=top_gram,
        bottom_gram=bottom_gram,
        twist_radians=twist,
        twist_degrees=np.degrees(twist),
        principal_strains=principal_strains,
        sharing_fraction=sharing,
        top_atom_counts=top_atom_counts,
        bottom_atom_counts=bottom_atom_counts,
        atom_counts=atom_counts,
        loewner_certified=certified.astype(bool),
        loewner_borderline=(~certified).astype(bool),
        top_affine=top_affine,
        bottom_affine=bottom_affine,
        shared_lattice=shared,
        canonical_keys=keys,
        pareto_optimal=pareto,
        rank=np.arange(1, len(top_matrices) + 1, dtype=np.int64),
        stats=stats,
    )


def _general_search(config: SearchConfig) -> SearchResult:
    clock = time.perf_counter
    started = clock()
    lower, upper = config._band
    top_basis, top_gauge = _reduce_basis(config.top_basis)
    bottom_basis, bottom_gauge = _reduce_basis(config.bottom_basis)
    top_metric, bottom_metric = _gram_of_basis(top_basis), _gram_of_basis(bottom_basis)
    max_squared = config.max_length * config.max_length
    min_squared = 0.0 if config.min_length is None else config.min_length**2
    top_area = float(np.sqrt(np.linalg.det(top_metric)))
    bottom_area = float(np.sqrt(np.linalg.det(bottom_metric)))
    if config.fold_symmetry:
        top_group = _proper_subgroup(_point_group(top_metric, tolerance=1e-10))
        bottom_group = _proper_subgroup(_point_group(bottom_metric, tolerance=1e-10))
    else:
        top_group = bottom_group = np.eye(2, dtype=np.int64)[None, :, :]

    top_vectors = _vector_table(top_metric, top_basis, max_squared)
    bottom_vectors = _vector_table(bottom_metric, bottom_basis, upper * max_squared)
    after_vectors = clock()

    area_squared_max = None
    if config.max_atoms is not None:
        multiplicity_cap = config.max_atoms / (
            config.top_atoms + config.bottom_atoms * lower * top_area / bottom_area
        )
        area_squared_max = (multiplicity_cap * top_area) ** 2
    first_top = (
        np.nonzero(_vector_orbit_representatives(top_vectors.vectors, top_group))[0]
        if config.fold_symmetry
        else None
    )
    top = _basis_table(
        top_vectors,
        top_metric,
        partner=False,
        first_indices=first_top,
        area_squared_max=area_squared_max,
    )
    top_unfolded = len(top)
    top = top.take(_first_per_key(_hermite_normal_form(top.first, top.second)))
    if config.fold_symmetry and len(top_group) > 1:
        top = top.take(_fold_sublattices(top.first, top.second, top_group))
    top_after_fold = len(top)
    top = top.take(_shape_mask(top, config))
    if min_squared > 0.0:
        top = top.take(top.g11 >= min_squared * (1.0 - _REL))
    if config.max_atoms is not None:
        minimum_weight = (
            config.top_atoms + config.bottom_atoms * lower * top_area / bottom_area
        )
        top = top.take(top.index * minimum_weight <= config.max_atoms)
    after_top = clock()

    first_bottom = (
        np.nonzero(_vector_orbit_representatives(bottom_vectors.vectors, bottom_group))[0]
        if config.fold_symmetry
        else None
    )
    bottom = _basis_table(
        bottom_vectors,
        bottom_metric,
        partner=True,
        lower=lower,
        upper=upper,
        first_indices=first_bottom,
    )
    bottom_unfolded = len(bottom)
    if config.fold_symmetry and len(bottom_group) > 1:
        bottom = bottom.take(_fold_bases(bottom.first, bottom.second, bottom_group))
    bottom_after_fold = len(bottom)
    # Proven necessary transfers of the top shape/length/atom filters.
    keep = bottom.g22 <= (
        (upper / lower)
        * config.max_aspect_ratio**2
        * bottom.g11
        * (1.0 + _REL)
    )
    cosine_limit = max(
        abs(math.cos(math.radians(config.min_cell_angle_deg))),
        abs(math.cos(math.radians(config.max_cell_angle_deg))),
    )
    coefficient = (lower * cosine_limit + upper - lower) ** 2
    keep &= lower**2 * bottom.g12**2 <= (
        coefficient * bottom.g11 * bottom.g22 * (1.0 + _REL)
    )
    bottom = bottom.take(keep)
    if min_squared > 0.0:
        threshold = lower * min_squared * (1.0 - _REL)
        bottom = bottom.take((bottom.g11 >= threshold) & (bottom.g22 >= threshold))
    if config.max_atoms is not None:
        minimum_weight = (
            config.bottom_atoms + config.top_atoms * bottom_area / (upper * top_area)
        )
        bottom = bottom.take(bottom.index * minimum_weight <= config.max_atoms)
    after_bottom = clock()

    if len(top) == 0 or len(bottom) == 0:
        top_rows = bottom_rows = np.zeros(0, dtype=np.int64)
    else:
        index = _BottomIndex(bottom, lower, upper)
        top_rows, sorted_bottom_rows = _join_candidates(top, index, lower, upper)
        bottom_rows = index.order[sorted_bottom_rows]
    after_join = clock()
    if config.max_atoms is not None:
        exact_atoms = (
            top.index[top_rows] * config.top_atoms
            + bottom.index[bottom_rows] * config.bottom_atoms
            <= config.max_atoms
        )
        top_rows, bottom_rows = top_rows[exact_atoms], bottom_rows[exact_atoms]

    top_first, top_second = top.first[top_rows], top.second[top_rows]
    bottom_first, bottom_second = bottom.first[bottom_rows], bottom.second[bottom_rows]
    p11, p12, p22 = top.g11[top_rows], top.g12[top_rows], top.g22[top_rows]
    q11, q12, q22 = (
        bottom.g11[bottom_rows],
        bottom.g12[bottom_rows],
        bottom.g22[bottom_rows],
    )
    top_multiplicity, bottom_multiplicity = top.index[top_rows], bottom.index[bottom_rows]
    twist = _twist_angles(
        top_basis,
        bottom_basis,
        top_first,
        top_second,
        bottom_first,
        bottom_second,
    )
    top_matrices_reduced = np.stack([top_first, top_second], axis=2)
    bottom_matrices_reduced = np.stack([bottom_first, bottom_second], axis=2)
    pair_keys = _canonical_pair_keys(top_matrices_reduced, bottom_matrices_reduced)
    unique = _first_per_key(pair_keys)
    top_matrices_reduced = top_matrices_reduced[unique]
    bottom_matrices_reduced = bottom_matrices_reduced[unique]
    p11, p12, p22 = p11[unique], p12[unique], p22[unique]
    q11, q12, q22 = q11[unique], q12[unique], q22[unique]
    top_multiplicity = top_multiplicity[unique]
    bottom_multiplicity = bottom_multiplicity[unique]
    twist = twist[unique]
    top_matrices = np.einsum("ij,njk->nik", top_gauge, top_matrices_reduced)
    bottom_matrices = np.einsum("ij,njk->nik", bottom_gauge, bottom_matrices_reduced)
    stats = {
        "branch": "general",
        "n_top_rows": len(top),
        "n_bottom_rows": len(bottom),
        "n_top_rows_unfolded": top_unfolded,
        "n_bottom_rows_unfolded": bottom_unfolded,
        "n_top_rows_after_fold": top_after_fold,
        "n_bottom_rows_after_fold": bottom_after_fold,
        "group_order_top": int(len(top_group)),
        "group_order_bottom": int(len(bottom_group)),
        "n_shells_top": top_vectors.shell_count,
        "n_shells_bottom": bottom_vectors.shell_count,
        "t_vectors": after_vectors - started,
        "t_top_table": after_top - after_vectors,
        "t_bottom_table": after_bottom - after_top,
        "t_join": after_join - after_bottom,
        "t_finish": clock() - after_join,
        "t_total": clock() - started,
    }
    return _finalize(
        config,
        top_matrices,
        bottom_matrices,
        np.stack([p11, p12, p22], axis=1),
        np.stack([q11, q12, q22], axis=1),
        top_multiplicity,
        bottom_multiplicity,
        twist,
        stats,
    )


def _right_handed_rotation(rotation: np.ndarray) -> np.ndarray:
    if rotation[1, 0] >= 0:
        return rotation.astype(np.int64)
    trace = rotation[0, 0] + rotation[1, 1]
    return np.array(
        [
            [trace - rotation[0, 0], -rotation[0, 1]],
            [-rotation[1, 0], trace - rotation[1, 1]],
        ],
        dtype=np.int64,
    )


def _rotation_generator(metric: np.ndarray, tolerance: float = 1e-10):
    group = _proper_subgroup(_point_group(metric, tolerance=tolerance))
    traces = group[:, 0, 0] + group[:, 1, 1]
    square = np.nonzero(traces == 0)[0]
    if square.size:
        return _right_handed_rotation(group[square[0]]), 0
    if np.any((traces == 1) | (traces == -1)):
        hexagonal = np.nonzero(traces == 1)[0]
        if hexagonal.size:
            return _right_handed_rotation(group[hexagonal[0]]), -1
    return None, None


def symmetric_branch_applies(config: SearchConfig) -> bool:
    """Return whether both layers have the same square or hexagonal rotation family."""
    top_basis, _ = _reduce_basis(config.top_basis)
    bottom_basis, _ = _reduce_basis(config.bottom_basis)
    top_rotation, top_kind = _rotation_generator(_gram_of_basis(top_basis))
    bottom_rotation, bottom_kind = _rotation_generator(_gram_of_basis(bottom_basis))
    return (
        top_rotation is not None
        and bottom_rotation is not None
        and top_kind == bottom_kind
    )


def _invariant_table(
    metric: np.ndarray,
    rotation: np.ndarray,
    radius_squared: float,
    fold: bool,
):
    vectors, squared = _lattice_vectors(metric, radius_squared)
    if fold and len(vectors):
        keep = _vector_orbit_representatives(
            vectors, _proper_subgroup(_point_group(metric, tolerance=1e-10))
        )
        vectors, squared = vectors[keep], squared[keep]
    rotated = vectors @ rotation.T
    index = np.abs(
        vectors[:, 0] * rotated[:, 1] - vectors[:, 1] * rotated[:, 0]
    )
    return vectors, rotated, squared, index


def _symmetric_search(config: SearchConfig) -> SearchResult:
    clock = time.perf_counter
    started = clock()
    lower, upper = config._band
    top_basis, top_gauge = _reduce_basis(config.top_basis)
    bottom_basis, bottom_gauge = _reduce_basis(config.bottom_basis)
    top_metric, bottom_metric = _gram_of_basis(top_basis), _gram_of_basis(bottom_basis)
    top_rotation, top_kind = _rotation_generator(top_metric)
    bottom_rotation, bottom_kind = _rotation_generator(bottom_metric)
    if top_rotation is None or bottom_rotation is None or top_kind != bottom_kind:
        raise SymmetricBranchUnavailable(
            "the symmetric branch requires both layers to be square or hexagonal "
            "with the same rotation order"
        )
    max_squared = config.max_length**2
    top_first, top_second, top_squared, top_index = _invariant_table(
        top_metric, top_rotation, max_squared, config.fold_symmetry
    )
    if config.min_length is not None:
        keep = top_squared >= config.min_length**2 * (1.0 - _REL)
        top_first, top_second = top_first[keep], top_second[keep]
        top_squared, top_index = top_squared[keep], top_index[keep]
    bottom_first, bottom_second, bottom_squared, bottom_index = _invariant_table(
        bottom_metric, bottom_rotation, upper * max_squared, config.fold_symmetry
    )
    after_tables = clock()
    low = np.searchsorted(
        bottom_squared, lower * top_squared * (1.0 - _REL), side="left"
    )
    high = np.searchsorted(
        bottom_squared, upper * top_squared * (1.0 + _REL), side="right"
    )
    top_rows, bottom_rows = _expand(low, high, np.arange(len(top_squared)))
    p, q = top_squared[top_rows], bottom_squared[bottom_rows]
    exact = (q >= lower * p) & (q <= upper * p)
    top_rows, bottom_rows = top_rows[exact], bottom_rows[exact]
    if config.max_atoms is not None:
        atom_keep = (
            top_index[top_rows] * config.top_atoms
            + bottom_index[bottom_rows] * config.bottom_atoms
            <= config.max_atoms
        )
        top_rows, bottom_rows = top_rows[atom_keep], bottom_rows[atom_keep]
    after_join = clock()
    top_first_result, top_second_result = top_first[top_rows], top_second[top_rows]
    bottom_first_result, bottom_second_result = (
        bottom_first[bottom_rows],
        bottom_second[bottom_rows],
    )
    p11, p12, p22 = _gram_triples(
        top_metric, top_first_result, top_second_result
    )
    q11, q12, q22 = _gram_triples(
        bottom_metric, bottom_first_result, bottom_second_result
    )
    shape_table = _Table(
        top_first_result,
        top_second_result,
        p11,
        p12,
        p22,
        top_index[top_rows],
    )
    shape_keep = _shape_mask(shape_table, config)
    top_first_result, top_second_result = (
        top_first_result[shape_keep],
        top_second_result[shape_keep],
    )
    bottom_first_result, bottom_second_result = (
        bottom_first_result[shape_keep],
        bottom_second_result[shape_keep],
    )
    p11, p12, p22 = p11[shape_keep], p12[shape_keep], p22[shape_keep]
    q11, q12, q22 = q11[shape_keep], q12[shape_keep], q22[shape_keep]
    top_multiplicity = top_index[top_rows][shape_keep]
    bottom_multiplicity = bottom_index[bottom_rows][shape_keep]
    top_cartesian = top_first_result @ top_basis.T
    bottom_cartesian = bottom_first_result @ bottom_basis.T
    twist = np.arctan2(
        top_cartesian[:, 0] * bottom_cartesian[:, 1]
        - top_cartesian[:, 1] * bottom_cartesian[:, 0],
        top_cartesian[:, 0] * bottom_cartesian[:, 0]
        + top_cartesian[:, 1] * bottom_cartesian[:, 1],
    )
    top_reduced = np.stack([top_first_result, top_second_result], axis=2)
    bottom_reduced = np.stack([bottom_first_result, bottom_second_result], axis=2)
    unique = _first_per_key(_canonical_pair_keys(top_reduced, bottom_reduced))
    top_reduced, bottom_reduced = top_reduced[unique], bottom_reduced[unique]
    p11, p12, p22 = p11[unique], p12[unique], p22[unique]
    q11, q12, q22 = q11[unique], q12[unique], q22[unique]
    top_multiplicity, bottom_multiplicity = (
        top_multiplicity[unique],
        bottom_multiplicity[unique],
    )
    twist = twist[unique]
    top_matrices = np.einsum("ij,njk->nik", top_gauge, top_reduced)
    bottom_matrices = np.einsum("ij,njk->nik", bottom_gauge, bottom_reduced)
    stats = {
        "branch": "symmetric",
        "symmetry_kind": "hexagonal" if top_kind == -1 else "square",
        "n_top_rows": int(len(top_squared)),
        "n_bottom_rows": int(len(bottom_squared)),
        "t_tables": after_tables - started,
        "t_join": after_join - after_tables,
        "t_finish": clock() - after_join,
        "t_total": clock() - started,
    }
    return _finalize(
        config,
        top_matrices,
        bottom_matrices,
        np.stack([p11, p12, p22], axis=1),
        np.stack([q11, q12, q22], axis=1),
        top_multiplicity,
        bottom_multiplicity,
        twist,
        stats,
    )


def search(config: SearchConfig) -> SearchResult:
    """Search canonical bilayer candidates using the general or restricted engine."""
    if not isinstance(config, SearchConfig):
        raise TypeError("search expects a SearchConfig")
    return _symmetric_search(config) if config.symmetric else _general_search(config)
