"""Independent contract and exhaustiveness tests for the native Gram search."""

from __future__ import annotations

import itertools
import math
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cellstine.moire.search.gram import (
    SearchConfig,
    SymmetricBranchUnavailable,
    _canonical_pair_keys,
    _loewner_mask,
    _pareto_front,
    _stretches_from_gram,
    _twist_angles,
    search,
    symmetric_branch_applies,
)


def _hex_basis(a: float) -> np.ndarray:
    return np.array([[a, -0.5 * a], [0.0, math.sqrt(3.0) * 0.5 * a]])


def _rect_basis(a: float, b: float) -> np.ndarray:
    return np.array([[a, 0.0], [0.0, b]])


CASES = [
    ("graphene / hBN", _hex_basis(2.46), _hex_basis(2.504), 8.0, 0.01, 0.01),
    ("hex wide budget", _hex_basis(2.46), _hex_basis(2.50), 12.0, 0.02, 0.02),
    ("square / square", _rect_basis(3.0, 3.0), _rect_basis(3.1, 3.1), 12.0, 0.02, 0.02),
    ("rectangular", _rect_basis(3.0, 4.1), _rect_basis(3.2, 4.0), 12.0, 0.03, 0.01),
    ("hex / rectangular", _hex_basis(2.46), _rect_basis(2.6, 4.3), 11.0, 0.025, 0.025),
    ("identical hex", _hex_basis(2.46), _hex_basis(2.46), 9.0, 0.01, 0.01),
]


def _gram(metric: np.ndarray, first: tuple[int, int], second: tuple[int, int]):
    def bilinear(left, right):
        return float(np.asarray(left) @ metric @ np.asarray(right))

    return (
        bilinear(first, first),
        bilinear(first, second),
        bilinear(second, second),
    )


def _lattice_points(metric: np.ndarray, radius_squared: float):
    g11, g12, g22 = metric[0, 0], metric[0, 1], metric[1, 1]
    determinant = g11 * g22 - g12 * g12
    m_max = int(math.floor(math.sqrt(radius_squared * g22 / determinant))) + 1
    n_max = int(math.floor(math.sqrt(radius_squared * g11 / determinant))) + 1
    points = []
    for m in range(-m_max, m_max + 1):
        for n in range(-n_max, n_max + 1):
            if m == 0 and n == 0:
                continue
            squared_length = g11 * m * m + 2.0 * g12 * m * n + g22 * n * n
            if squared_length <= radius_squared:
                points.append((m, n, squared_length))
    return sorted(points, key=lambda item: item[2])


def _extended_gcd(left: int, right: int):
    old_r, r = left, right
    old_s, s = 1, 0
    old_t, t = 0, 1
    while r:
        quotient = old_r // r
        old_r, r = r, old_r - quotient * r
        old_s, s = s, old_s - quotient * s
        old_t, t = t, old_t - quotient * t
    sign = -1 if old_r < 0 else 1
    return old_s * sign, old_t * sign


def _hnf(first: tuple[int, int], second: tuple[int, int]):
    a, c = first
    b, d = second
    determinant = abs(a * d - b * c)
    h22 = math.gcd(c, d)
    h11 = determinant // h22
    x, y = _extended_gcd(c, d)
    return h11, (x * a + y * b) % h11, h22


def _reference_search(
    top_basis: np.ndarray,
    bottom_basis: np.ndarray,
    max_length: float,
    top_strain: float,
    bottom_strain: float,
):
    """Bounded nested-loop oracle, deliberately independent of the staged engine."""
    top_metric = top_basis.T @ top_basis
    bottom_metric = bottom_basis.T @ bottom_basis
    radius_squared = max_length * max_length
    budget = top_strain + bottom_strain
    lower, upper = math.exp(-2.0 * budget), math.exp(2.0 * budget)

    seen_top = set()
    top_cells = []
    points = _lattice_points(top_metric, radius_squared)
    for m1, n1, _ in points:
        for m2, n2, _ in points:
            determinant = m1 * n2 - n1 * m2
            if determinant == 0:
                continue
            first, second = (m1, n1), (m2, n2)
            if determinant < 0:
                second = (-m2, -n2)
            gram = _gram(top_metric, first, second)
            if gram[0] > gram[2] or 2.0 * abs(gram[1]) > gram[0] * (1.0 + 1e-12):
                continue
            key = _hnf(first, second)
            if key not in seen_top:
                seen_top.add(key)
                top_cells.append((first, second, gram))

    bottom_points = _lattice_points(bottom_metric, upper * radius_squared)
    accepted = []
    for first, second, top_gram in top_cells:
        top_matrix = np.array([[first[0], second[0]], [first[1], second[1]]], dtype=np.int64)
        for m1, n1, _ in bottom_points:
            for m2, n2, _ in bottom_points:
                if m1 * n2 - n1 * m2 <= 0:
                    continue
                bottom_first, bottom_second = (m1, n1), (m2, n2)
                bottom_gram = _gram(bottom_metric, bottom_first, bottom_second)
                a11 = bottom_gram[0] - lower * top_gram[0]
                a12 = bottom_gram[1] - lower * top_gram[1]
                a22 = bottom_gram[2] - lower * top_gram[2]
                b11 = upper * top_gram[0] - bottom_gram[0]
                b12 = upper * top_gram[1] - bottom_gram[1]
                b22 = upper * top_gram[2] - bottom_gram[2]
                if (
                    a11 + a22 >= 0.0
                    and a11 * a22 - a12 * a12 >= 0.0
                    and b11 + b22 >= 0.0
                    and b11 * b22 - b12 * b12 >= 0.0
                ):
                    bottom_matrix = np.array(
                        [[bottom_first[0], bottom_second[0]], [bottom_first[1], bottom_second[1]]],
                        dtype=np.int64,
                    )
                    accepted.append((top_matrix, bottom_matrix))
    return accepted


def _pair_class_key(top_matrix: np.ndarray, bottom_matrix: np.ndarray):
    """Exact common-right-unimodular class name, independently derived via column HNF."""
    first = tuple(int(value) for value in top_matrix[:, 0])
    second = tuple(int(value) for value in top_matrix[:, 1])
    h11, h12, h22 = _hnf(first, second)
    hnf = np.array([[h11, h12], [0, h22]], dtype=np.int64)
    determinant = int(round(np.linalg.det(top_matrix)))
    adjugate = np.array(
        [[top_matrix[1, 1], -top_matrix[0, 1]], [-top_matrix[1, 0], top_matrix[0, 0]]],
        dtype=np.int64,
    )
    transform_numerator = adjugate @ hnf
    assert np.all(transform_numerator % determinant == 0)
    transform = transform_numerator // determinant
    canonical_bottom = bottom_matrix @ transform
    return tuple(int(value) for value in np.concatenate([hnf.ravel(), canonical_bottom.ravel()]))


def _result_class_set(result):
    return {
        _pair_class_key(top, bottom)
        for top, bottom in zip(result.top_matrices, result.bottom_matrices, strict=True)
    }


_UNIMODULAR = np.stack(
    [
        np.array(values, dtype=np.int64).reshape(2, 2)
        for values in itertools.product(range(-2, 3), repeat=4)
        if abs(values[0] * values[3] - values[1] * values[2]) == 1
    ]
)


def _physical_class_set(result, top_metric: np.ndarray, bottom_metric: np.ndarray):
    """Gauge-independent Gram class names, adapted from the Aristotle test oracle."""
    classes = set()
    for top_matrix, bottom_matrix in zip(
        result.top_matrices, result.bottom_matrices, strict=True
    ):
        candidates = []
        for transform in _UNIMODULAR:
            top = top_matrix @ transform
            bottom = bottom_matrix @ transform
            top_gram = top.T @ top_metric @ top
            bottom_gram = bottom.T @ bottom_metric @ bottom
            if (
                np.linalg.det(top) > 0
                and np.linalg.det(bottom) > 0
                and top_gram[0, 0] <= top_gram[1, 1] * (1.0 + 1e-9)
                and 2.0 * abs(top_gram[0, 1]) <= top_gram[0, 0] * (1.0 + 1e-9)
            ):
                candidates.append(
                    tuple(
                        int(value)
                        for value in np.rint(
                            np.array(
                                [
                                    top_gram[0, 0],
                                    top_gram[0, 1],
                                    top_gram[1, 1],
                                    bottom_gram[0, 0],
                                    bottom_gram[0, 1],
                                    bottom_gram[1, 1],
                                ]
                            )
                            * 1_000_000
                        )
                    )
                )
        classes.add(min(candidates))
    return classes


@pytest.mark.parametrize(
    "changes",
    [
        {"top_basis": np.eye(3)},
        {"top_basis": np.array([[1.0, 2.0], [2.0, 4.0]])},
        {"bottom_basis": np.array([[1.0, 0.0], [0.0, np.nan]])},
        {"max_length": 0.0},
        {"top_strain": -0.01},
        {"top_strain": 0.0, "bottom_strain": 0.0},
        {"min_length": 11.0},
        {"max_atoms": 0},
        {"top_atoms": 0},
        {"max_aspect_ratio": 0.9},
        {"min_cell_angle_deg": 160.0, "max_cell_angle_deg": 155.0},
    ],
)
def test_invalid_configuration_is_rejected(changes):
    values = {
        "top_basis": np.eye(2),
        "bottom_basis": np.eye(2),
        "max_length": 10.0,
        "top_strain": 0.01,
        "bottom_strain": 0.01,
    }
    values.update(changes)
    with pytest.raises((TypeError, ValueError)):
        SearchConfig(**values)


def test_scalar_loewner_acceptance_agrees_with_svd_reference():
    rng = np.random.default_rng(4821)
    budget = 0.08
    lower, upper = math.exp(-2.0 * budget), math.exp(2.0 * budget)
    for _ in range(300):
        top_cell = rng.normal(size=(2, 2))
        bottom_cell = rng.normal(size=(2, 2))
        if np.linalg.det(top_cell) == 0.0 or np.linalg.det(bottom_cell) == 0.0:
            continue
        top_gram = top_cell.T @ top_cell
        bottom_gram = bottom_cell.T @ bottom_cell
        got = bool(
            _loewner_mask(
                top_gram[0, 0],
                top_gram[0, 1],
                top_gram[1, 1],
                bottom_gram[0, 0],
                bottom_gram[0, 1],
                bottom_gram[1, 1],
                lower,
                upper,
            )
        )
        singular_values = np.linalg.svd(bottom_cell @ np.linalg.inv(top_cell), compute_uv=False)
        expected = bool(np.all(np.abs(np.log(singular_values)) <= budget))
        assert got == expected


def test_gauge_equivalent_inputs_produce_the_same_physical_classes():
    top = _hex_basis(2.46)
    bottom = _rect_basis(2.55, 3.8)
    top_gauge = np.array([[1, 2], [0, 1]], dtype=np.int64)
    bottom_gauge = np.array([[1, -1], [1, 0]], dtype=np.int64)
    base = SearchConfig(top, bottom, 8.0, 0.025, 0.025, fold_symmetry=False)
    gauged = SearchConfig(
        top @ top_gauge,
        bottom @ bottom_gauge,
        8.0,
        0.025,
        0.025,
        fold_symmetry=False,
    )
    first = search(base)
    second = search(gauged)
    assert _physical_class_set(first, top.T @ top, bottom.T @ bottom) == _physical_class_set(
        second,
        gauged.top_basis.T @ gauged.top_basis,
        gauged.bottom_basis.T @ gauged.bottom_basis,
    )


def test_closed_form_stretch_and_twist_match_direct_deformation_gradient():
    top = np.array([[2.7, 0.35], [0.1, 3.2]])
    bottom = np.array([[2.55, -0.2], [0.25, 3.05]])
    first_top = np.array([[1, 0], [1, 1], [2, -1]], dtype=np.int64)
    second_top = np.array([[0, 1], [-1, 2], [1, 2]], dtype=np.int64)
    first_bottom = np.array([[1, 0], [1, 1], [2, -1]], dtype=np.int64)
    second_bottom = np.array([[0, 1], [-1, 2], [1, 2]], dtype=np.int64)
    top_metric, bottom_metric = top.T @ top, bottom.T @ bottom

    def triples(metric, first, second):
        return (
            np.einsum("ij,jk,ik->i", first, metric, first),
            np.einsum("ij,jk,ik->i", first, metric, second),
            np.einsum("ij,jk,ik->i", second, metric, second),
        )

    p11, p12, p22 = triples(top_metric, first_top, second_top)
    q11, q12, q22 = triples(bottom_metric, first_bottom, second_bottom)
    _, _, first_strain, second_strain = _stretches_from_gram(
        p11, p12, p22, q11, q12, q22
    )
    twist = _twist_angles(top, bottom, first_top, second_top, first_bottom, second_bottom)
    for row in range(len(first_top)):
        top_cell = top @ np.column_stack([first_top[row], second_top[row]])
        bottom_cell = bottom @ np.column_stack([first_bottom[row], second_bottom[row]])
        deformation = bottom_cell @ np.linalg.inv(top_cell)
        left, singular_values, right = np.linalg.svd(deformation)
        rotation = left @ right
        expected_twist = math.atan2(rotation[1, 0], rotation[0, 0])
        assert np.sort([first_strain[row], second_strain[row]]) == pytest.approx(
            np.sort(np.log(singular_values)), abs=1e-11
        )
        assert math.atan2(math.sin(twist[row] - expected_twist), math.cos(twist[row] - expected_twist)) == pytest.approx(0.0, abs=1e-12)


def test_canonicalization_and_pareto_are_exact_and_deterministic():
    top = np.array(
        [
            [[2, 1], [0, 1]],
            [[3, -1], [1, 0]],
            [[1, 0], [0, 1]],
        ],
        dtype=np.int64,
    )
    bottom = np.array(
        [
            [[1, 0], [1, 2]],
            [[2, 1], [-1, 1]],
            [[2, 0], [0, 1]],
        ],
        dtype=np.int64,
    )
    transform = np.array([[1, 1], [0, 1]], dtype=np.int64)
    doubled_top = np.concatenate([top, top @ transform])
    doubled_bottom = np.concatenate([bottom, bottom @ transform])
    keys = _canonical_pair_keys(doubled_top, doubled_bottom)
    assert np.array_equal(keys[:3], keys[3:])

    first = _pareto_front(np.array([3, 2, 2, 4, 1]), np.array([1.0, 2.0, 2.0, 0.5, 3.0]))
    second = _pareto_front(np.array([3, 2, 2, 4, 1]), np.array([1.0, 2.0, 2.0, 0.5, 3.0]))
    assert np.array_equal(first, np.array([4, 1, 0, 3]))
    assert np.array_equal(first, second)

    config = SearchConfig(_hex_basis(2.46), _hex_basis(2.504), 8.0, 0.01, 0.01)
    folded = search(config)
    repeated = search(config)
    unfolded = search(replace(config, fold_symmetry=False))
    assert np.array_equal(folded.canonical_keys, repeated.canonical_keys)
    assert np.array_equal(folded.rank, np.arange(1, len(folded) + 1))
    assert len({tuple(row) for row in folded.canonical_keys}) == len(folded)
    assert _physical_class_set(folded, config.top_basis.T @ config.top_basis, config.bottom_basis.T @ config.bottom_basis) == _physical_class_set(
        unfolded, config.top_basis.T @ config.top_basis, config.bottom_basis.T @ config.bottom_basis
    )


@pytest.mark.parametrize(
    "name,top,bottom,max_length,top_strain,bottom_strain",
    CASES,
    ids=[case[0] for case in CASES],
)
def test_all_six_small_cases_match_the_independent_bounded_oracle(
    name, top, bottom, max_length, top_strain, bottom_strain
):
    del name
    reference = _reference_search(top, bottom, max_length, top_strain, bottom_strain)
    expected = {_pair_class_key(top_matrix, bottom_matrix) for top_matrix, bottom_matrix in reference}
    result = search(
        SearchConfig(
            top,
            bottom,
            max_length,
            top_strain,
            bottom_strain,
            max_aspect_ratio=100.0,
            fold_symmetry=False,
        )
    )
    assert _result_class_set(result) == expected


def test_search_result_contains_buildable_certified_geometry():
    config = SearchConfig(
        _hex_basis(2.46),
        _hex_basis(2.504),
        8.0,
        0.01,
        0.01,
        top_atoms=2,
        bottom_atoms=3,
        fold_symmetry=False,
    )
    result = search(config)
    assert len(result) > 0
    assert result.top_gram.shape == result.bottom_gram.shape == (len(result), 3)
    assert result.principal_strains.shape == (len(result), 2)
    assert result.twist_degrees == pytest.approx(np.degrees(result.twist_radians))
    assert result.sharing_fraction == pytest.approx(np.full(len(result), 0.5))
    assert np.array_equal(
        result.atom_counts, result.top_atom_counts + result.bottom_atom_counts
    )
    assert result.loewner_certified.dtype == np.bool_
    assert result.loewner_borderline.dtype == np.bool_
    assert np.array_equal(result.loewner_borderline, ~result.loewner_certified)
    for row in range(len(result)):
        top_cell = config.top_basis @ result.top_matrices[row]
        bottom_cell = config.bottom_basis @ result.bottom_matrices[row]
        assert result.top_affine[row] @ top_cell == pytest.approx(result.shared_lattice[row])
        assert result.bottom_affine[row] @ bottom_cell == pytest.approx(result.shared_lattice[row])
    assert result.stats["branch"] == "general"


def _similar_subfamily(result, primitive_c: int):
    top, bottom = result.top_gram, result.bottom_gram
    scale = bottom[:, 0] / top[:, 0]
    return (
        np.isclose(bottom[:, 1], scale * top[:, 1], rtol=1e-7, atol=1e-8)
        & np.isclose(bottom[:, 2], scale * top[:, 2], rtol=1e-7, atol=1e-8)
        & np.isclose(top[:, 0], top[:, 2], rtol=1e-7, atol=1e-8)
        & np.isclose(2.0 * np.abs(top[:, 1]), abs(primitive_c) * top[:, 0], rtol=1e-7, atol=1e-8)
    )


@pytest.mark.parametrize(
    "top,bottom,applies",
    [
        (_hex_basis(2.46), _hex_basis(2.504), True),
        (_rect_basis(3.0, 3.0), _rect_basis(3.1, 3.1), True),
        (_hex_basis(2.46), _rect_basis(3.0, 3.0), False),
        (_hex_basis(2.46), _rect_basis(3.0, 4.0), False),
    ],
)
def test_symmetric_branch_applicability_and_general_subfamily(top, bottom, applies):
    general_config = SearchConfig(top, bottom, 12.0, 0.02, 0.02, fold_symmetry=False)
    assert symmetric_branch_applies(general_config) is applies
    symmetric_config = replace(general_config, symmetric=True)
    if not applies:
        with pytest.raises(SymmetricBranchUnavailable, match="square or hexagonal"):
            search(symmetric_config)
        return

    general = search(general_config)
    symmetric = search(symmetric_config)
    primitive_c = -1 if np.isclose(top[0, 1] / top[0, 0], -0.5) else 0
    subfamily = _similar_subfamily(general, primitive_c)
    expected = {
        _pair_class_key(top_matrix, bottom_matrix)
        for top_matrix, bottom_matrix in zip(
            general.top_matrices[subfamily], general.bottom_matrices[subfamily], strict=True
        )
    }
    assert _result_class_set(symmetric) == expected
    assert symmetric.stats["branch"] == "symmetric"
