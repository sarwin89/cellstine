"""Lattice primitives of the Gram-form search.

Gauge reduction of a basis, the vector shells of a metric, the point group of a
Gram form, Hermite normal forms and the folded basis tables that the join stage
consumes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ...core.reduction import gauss_reduction_multiplier
from .gram_config import SearchConfig, _REL, _SHELL_RATIO, _TWO_PI

def _gram_of_basis(basis: np.ndarray) -> np.ndarray:
    return basis.T @ basis


def _internal_length_scale(config: SearchConfig) -> float:
    """Common power-of-two scale keeping metric products away from under/overflow."""
    reference = max(
        float(np.max(np.abs(config.top_basis))),
        float(np.max(np.abs(config.bottom_basis))),
        float(config.max_length),
    )
    _, exponent = math.frexp(reference)
    return math.ldexp(1.0, exponent - 1)


_gauss_reduction_multiplier = gauss_reduction_multiplier
"""Nearest shear, treating a roundoff-width ``|dot| = norm/2`` boundary as reduced."""


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
        multiplier = _gauss_reduction_multiplier(
            float(reduced[:, 0] @ reduced[:, 1]), first_norm
        )
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


def _gauge_group(group: np.ndarray, gauge: np.ndarray) -> np.ndarray:
    """Express an integer point group in the Lagrange-reduced gauge.

    With ``A_reduced = A @ P`` an operation ``G`` of ``A`` acts as ``P^-1 G P``.
    """

    determinant = int(round(float(np.linalg.det(gauge))))
    if determinant not in (1, -1):  # pragma: no cover - gauges are always unimodular
        raise ValueError("gauge transform must be unimodular")
    inverse = determinant * np.array(
        [[gauge[1, 1], -gauge[0, 1]], [-gauge[1, 0], gauge[0, 0]]], dtype=np.int64
    )
    return inverse @ np.asarray(group, dtype=np.int64) @ gauge


def _bezout(left: np.ndarray, right: np.ndarray):
    """Vectorised extended Euclid: ``x * left + y * right = gcd(left, right) >= 0``.

    The recursion is run on a shrinking index set rather than on the whole array,
    so the cost is the *total* number of Euclid steps instead of the array length
    times the worst-case number of steps.  The arithmetic is exact integer
    arithmetic and is unchanged by the compaction.
    """

    old_r = np.ascontiguousarray(left, dtype=np.int64).copy()
    r = np.ascontiguousarray(right, dtype=np.int64).copy()
    old_s, s = np.ones_like(old_r), np.zeros_like(old_r)
    old_t, t = np.zeros_like(old_r), np.ones_like(old_r)
    active = np.flatnonzero(r)
    while active.size:
        remainder, previous = r[active], old_r[active]
        quotient = previous // remainder
        old_r[active], r[active] = remainder, previous - quotient * remainder
        coefficient, previous_coefficient = s[active], old_s[active]
        old_s[active], s[active] = (
            coefficient,
            previous_coefficient - quotient * coefficient,
        )
        coefficient, previous_coefficient = t[active], old_t[active]
        old_t[active], t[active] = (
            coefficient,
            previous_coefficient - quotient * coefficient,
        )
        active = active[r[active] != 0]
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


def _lexicographic_min_pair(best: np.ndarray, candidate: np.ndarray) -> np.ndarray:
    """Row-wise lexicographic minimum of two equally shaped key tables."""

    less = np.zeros(best.shape[0], dtype=bool)
    equal = np.ones(best.shape[0], dtype=bool)
    for column in range(best.shape[1]):
        less |= equal & (candidate[:, column] < best[:, column])
        equal &= candidate[:, column] == best[:, column]
        if not equal.any():
            break
    return np.where(less[:, None], candidate, best)


def _lexicographic_minimum(stack: np.ndarray) -> np.ndarray:
    best = stack[0].copy()
    for candidate in stack[1:]:
        best = _lexicographic_min_pair(best, candidate)
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
