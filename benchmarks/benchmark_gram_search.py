"""Reproducible, non-gating legacy-reference versus native Gram benchmark.

Run from the repository root with::

    python benchmarks/benchmark_gram_search.py

The script exits nonzero if the exact canonical candidate-class sets differ.  Timings are
measurements, not assertions, because wall-clock performance varies by host.
"""

from __future__ import annotations

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


def class_key(top: np.ndarray, bottom: np.ndarray):
    h11, h12, h22 = hnf(tuple(top[:, 0]), tuple(top[:, 1]))
    normal = np.array([[h11, h12], [0, h22]], dtype=np.int64)
    determinant = int(round(np.linalg.det(top)))
    adjugate = np.array(
        [[top[1, 1], -top[0, 1]], [-top[1, 0], top[0, 0]]], dtype=np.int64
    )
    numerator = adjugate @ normal
    if np.any(numerator % determinant):
        raise ArithmeticError("nonintegral canonical transform in benchmark reference")
    canonical_bottom = bottom @ (numerator // determinant)
    return tuple(int(value) for value in np.concatenate([normal.ravel(), canonical_bottom.ravel()]))


def legacy_reference(
    top_basis: np.ndarray, bottom_basis: np.ndarray, max_length: float
) -> set[tuple[int, ...]]:
    """Original-style exhaustive nested loop with scalar Löwner acceptance."""
    top_metric, bottom_metric = top_basis.T @ top_basis, bottom_basis.T @ bottom_basis
    radius_squared = max_length * max_length
    lower = math.exp(-2.0 * (TOP_STRAIN + BOTTOM_STRAIN))
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
                top_cells.append((first, second, top_gram))

    classes = set()
    bottom_points = lattice_points(bottom_metric, upper * radius_squared)
    for first, second, top_gram in top_cells:
        top = np.array([[first[0], second[0]], [first[1], second[1]]], dtype=np.int64)
        for bottom_first, _ in bottom_points:
            for bottom_second, _ in bottom_points:
                if bottom_first[0] * bottom_second[1] - bottom_first[1] * bottom_second[0] <= 0:
                    continue
                bottom_gram = gram(bottom_metric, bottom_first, bottom_second)
                a11 = bottom_gram[0] - lower * top_gram[0]
                a12 = bottom_gram[1] - lower * top_gram[1]
                a22 = bottom_gram[2] - lower * top_gram[2]
                b11 = upper * top_gram[0] - bottom_gram[0]
                b12 = upper * top_gram[1] - bottom_gram[1]
                b22 = upper * top_gram[2] - bottom_gram[2]
                accepted = (
                    a11 + a22 >= 0.0
                    and a11 * a22 - a12 * a12 >= 0.0
                    and b11 + b22 >= 0.0
                    and b11 * b22 - b12 * b12 >= 0.0
                )
                if accepted:
                    bottom = np.array(
                        [
                            [bottom_first[0], bottom_second[0]],
                            [bottom_first[1], bottom_second[1]],
                        ],
                        dtype=np.int64,
                    )
                    classes.add(class_key(top, bottom))
    return classes


def main() -> int:
    top_basis, bottom_basis = hex_basis(2.46), hex_basis(2.504)
    measurements = []
    print("max_length_A  classes  legacy_s  gram_s  legacy_scale  gram_scale")
    for max_length in MAX_LENGTHS:
        started = time.perf_counter()
        expected = legacy_reference(top_basis, bottom_basis, max_length)
        legacy_seconds = time.perf_counter() - started

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
        actual = {
            class_key(top, bottom)
            for top, bottom in zip(result.top_matrices, result.bottom_matrices, strict=True)
        }
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
