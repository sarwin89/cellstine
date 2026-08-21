"""Reproducible, non-gating legacy-reference versus native Gram benchmark.

Run from the repository root with::

    python benchmarks/benchmark_gram_search.py

The script exits nonzero if the exact canonical candidate-class sets differ.  Timings are
measurements, not assertions, because wall-clock performance varies by host.
"""

from __future__ import annotations

import itertools
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cellstine.moire.search.gram import SearchConfig, search  # noqa: E402

MAX_LENGTHS = (6.0, 8.0, 10.0)
TOP_STRAIN = 0.01
BOTTOM_STRAIN = 0.01


def hex_basis(length: float) -> np.ndarray:
    return np.array(
        [[length, -0.5 * length], [0.0, 0.5 * math.sqrt(3.0) * length]]
    )


def lattice_points(metric: np.ndarray, radius_squared: float):
    g11, g12, g22 = metric[0, 0], metric[0, 1], metric[1, 1]
    determinant = g11 * g22 - g12 * g12
    m_max = int(math.floor(math.sqrt(radius_squared * g22 / determinant))) + 1
    n_max = int(math.floor(math.sqrt(radius_squared * g11 / determinant))) + 1
    points = []
    for m in range(-m_max, m_max + 1):
        for n in range(-n_max, n_max + 1):
            if m == 0 and n == 0:
                continue
            squared = g11 * m * m + 2.0 * g12 * m * n + g22 * n * n
            if squared <= radius_squared:
                points.append(((m, n), squared))
    return sorted(points, key=lambda item: item[1])


def gram(metric: np.ndarray, first, second):
    def bilinear(left, right):
        return float(np.asarray(left) @ metric @ np.asarray(right))

    return bilinear(first, first), bilinear(first, second), bilinear(second, second)


def extended_gcd(left: int, right: int):
    old_r, r, old_s, s, old_t, t = left, right, 1, 0, 0, 1
    while r:
        quotient = old_r // r
        old_r, r = r, old_r - quotient * r
        old_s, s = s, old_s - quotient * s
        old_t, t = t, old_t - quotient * t
    sign = -1 if old_r < 0 else 1
    return old_s * sign, old_t * sign


def hnf(first, second):
    a, c = first
    b, d = second
    determinant = abs(a * d - b * c)
    h22 = math.gcd(c, d)
    h11 = determinant // h22
    x, y = extended_gcd(c, d)
    return h11, (x * a + y * b) % h11, h22


UNIMODULAR = np.stack(
    [
        np.array(values, dtype=np.int64).reshape(2, 2)
        for values in itertools.product(range(-2, 3), repeat=4)
        if abs(values[0] * values[3] - values[1] * values[2]) == 1
    ]
)


def canonical_class_set(top, bottom, top_metric, bottom_metric):
    """Independent bounded-unimodular Gram canonicalization from Aristotle's oracle."""
    if len(top) == 0:
        return set()
    best = np.zeros((len(top), 6), dtype=np.int64)
    found = np.zeros(len(top), dtype=bool)
    for transform in UNIMODULAR:
        transformed_top = top @ transform
        transformed_bottom = bottom @ transform
        top_gram = np.swapaxes(transformed_top, 1, 2) @ top_metric @ transformed_top
        bottom_gram = (
            np.swapaxes(transformed_bottom, 1, 2) @ bottom_metric @ transformed_bottom
        )
        top_det = (
            transformed_top[:, 0, 0] * transformed_top[:, 1, 1]
            - transformed_top[:, 0, 1] * transformed_top[:, 1, 0]
        )
        bottom_det = (
            transformed_bottom[:, 0, 0] * transformed_bottom[:, 1, 1]
            - transformed_bottom[:, 0, 1] * transformed_bottom[:, 1, 0]
        )
        valid = (
            (top_det > 0)
            & (bottom_det > 0)
            & (top_gram[:, 0, 0] <= top_gram[:, 1, 1] * (1.0 + 1e-9))
            & (2.0 * np.abs(top_gram[:, 0, 1]) <= top_gram[:, 0, 0] * (1.0 + 1e-9))
        )
        key = np.rint(
            np.stack(
                [
                    top_gram[:, 0, 0],
                    top_gram[:, 0, 1],
                    top_gram[:, 1, 1],
                    bottom_gram[:, 0, 0],
                    bottom_gram[:, 0, 1],
                    bottom_gram[:, 1, 1],
                ],
                axis=1,
            )
            * 1_000_000
        ).astype(np.int64)
        less = np.zeros(len(top), dtype=bool)
        equal = np.ones(len(top), dtype=bool)
        for column in range(6):
            less |= equal & (key[:, column] < best[:, column])
            equal &= key[:, column] == best[:, column]
        take = valid & (~found | less)
        best[take] = key[take]
        found |= valid
    if not np.all(found):
        raise ArithmeticError("benchmark canonicalizer found no reduced representative")
    return {tuple(int(value) for value in row) for row in best}


def legacy_reference(
    top_basis: np.ndarray, bottom_basis: np.ndarray, max_length: float
) -> tuple[np.ndarray, np.ndarray]:
    """Original-style exhaustive nested loop with direct SVD acceptance."""
    top_metric, bottom_metric = top_basis.T @ top_basis, bottom_basis.T @ bottom_basis
    radius_squared = max_length * max_length
    upper = math.exp(2.0 * (TOP_STRAIN + BOTTOM_STRAIN))
    seen_top, top_cells = set(), []
    top_points = lattice_points(top_metric, radius_squared)
    for (first, _) in top_points:
        for (second_raw, _) in top_points:
            determinant = first[0] * second_raw[1] - first[1] * second_raw[0]
            if determinant == 0:
                continue
            second = second_raw if determinant > 0 else (-second_raw[0], -second_raw[1])
            top_gram = gram(top_metric, first, second)
            if top_gram[0] > top_gram[2] or 2.0 * abs(top_gram[1]) > top_gram[0] * (1 + 1e-12):
                continue
            name = hnf(first, second)
            if name not in seen_top:
                seen_top.add(name)
                top_cells.append((first, second))

    accepted_top, accepted_bottom = [], []
    bottom_points = lattice_points(bottom_metric, upper * radius_squared)
    for first, second in top_cells:
        top = np.array([[first[0], second[0]], [first[1], second[1]]], dtype=np.int64)
        for bottom_first, _ in bottom_points:
            for bottom_second, _ in bottom_points:
                if bottom_first[0] * bottom_second[1] - bottom_first[1] * bottom_second[0] <= 0:
                    continue
                bottom = np.array(
                    [
                        [bottom_first[0], bottom_second[0]],
                        [bottom_first[1], bottom_second[1]],
                    ],
                    dtype=np.int64,
                )
                deformation = (bottom_basis @ bottom) @ np.linalg.inv(top_basis @ top)
                singular_values = np.linalg.svd(deformation, compute_uv=False)
                if np.max(np.abs(np.log(singular_values))) <= (
                    TOP_STRAIN + BOTTOM_STRAIN + 64.0 * np.finfo(float).eps
                ):
                    accepted_top.append(top)
                    accepted_bottom.append(bottom)
    return np.stack(accepted_top), np.stack(accepted_bottom)


def main() -> int:
    top_basis, bottom_basis = hex_basis(2.46), hex_basis(2.504)
    measurements = []
    print("max_length_A  classes  legacy_s  gram_s  legacy_scale  gram_scale")
    for max_length in MAX_LENGTHS:
        started = time.perf_counter()
        legacy_top, legacy_bottom = legacy_reference(top_basis, bottom_basis, max_length)
        legacy_seconds = time.perf_counter() - started
        expected = canonical_class_set(
            legacy_top,
            legacy_bottom,
            top_basis.T @ top_basis,
            bottom_basis.T @ bottom_basis,
        )

        started = time.perf_counter()
        result = search(
            SearchConfig(
                top_basis,
                bottom_basis,
                max_length,
                TOP_STRAIN,
                BOTTOM_STRAIN,
                max_aspect_ratio=100.0,
                fold_symmetry=False,
            )
        )
        gram_seconds = time.perf_counter() - started
        actual = canonical_class_set(
            result.top_matrices,
            result.bottom_matrices,
            top_basis.T @ top_basis,
            bottom_basis.T @ bottom_basis,
        )
        missing, extra = expected - actual, actual - expected
        if missing or extra:
            raise AssertionError(
                f"candidate-class mismatch at {max_length:g} A: "
                f"missing={len(missing)}, extra={len(extra)}"
            )
        measurements.append((max_length, len(actual), legacy_seconds, gram_seconds))
        base_legacy, base_gram = measurements[0][2], measurements[0][3]
        print(
            f"{max_length:12.1f}  {len(actual):7d}  {legacy_seconds:8.6f}  "
            f"{gram_seconds:7.6f}  {legacy_seconds / base_legacy:12.3f}  "
            f"{gram_seconds / base_gram:10.3f}"
        )
    print("candidate classes equal at every length; timings are measured wall-clock values")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
