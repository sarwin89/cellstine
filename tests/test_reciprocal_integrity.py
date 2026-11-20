"""Exact reciprocal meshes must not merge points through integer overflow."""

from __future__ import annotations

import tracemalloc

import numpy as np
import pytest

from cellstine.core import reciprocal as rc


def test_large_denominator_keys_keep_distinct_points_distinct():
    denominator = 2**21
    points = np.array([[-denominator // 2, 0, 0], [0, 0, 0]], dtype=np.int64)
    assert len(np.unique(rc._encode(points, denominator))) == 2


@pytest.mark.parametrize("time_reversal", [False, True])
def test_identity_orbits_do_not_merge_overflowed_keys(time_reversal):
    denominator = 2**21
    points = np.array([[-denominator // 2, 0, 0], [0, 0, 0]], dtype=np.int64)
    labels = rc._orbit_labels(points, denominator, [np.eye(3, dtype=np.int64)],
                              time_reversal=time_reversal)
    assert np.array_equal(labels, [0, 1])


@pytest.mark.parametrize("spacing", [1e-20, 1e-300])
def test_unrepresentable_spacing_request_is_refused_instead_of_undersampled(spacing):
    with pytest.raises(ValueError, match="range|represent"):
        rc.mesh_divisions_for_spacing(np.eye(3), spacing)


def test_modular_images_use_exact_products_for_large_integer_rotations():
    denominator = 2000006
    points = np.array([[1000000, 7, -3], [-999999, -4, 9]], dtype=np.int64)
    rotation = np.array([[1, 2**50, 0], [0, -1, 0], [0, 0, -1]], dtype=np.int64)
    expected = np.asarray((points.astype(object) @ rotation.astype(object)) % denominator,
                          dtype=np.int64)
    expected = np.where(expected >= (denominator + 1) // 2, expected - denominator, expected)
    assert np.array_equal(rc._mesh_images(points, denominator, rotation), expected)


def test_large_integral_finite_group_keeps_exact_inverse_and_orbits():
    n = 2**28
    rotation = np.array([[n, -n*n-1, 0], [1, -n, 0], [0, 0, 1]], dtype=np.int64)
    inverse = rc._integer_inverse(rotation)
    assert np.array_equal(rotation.astype(object) @ inverse.astype(object), np.eye(3, dtype=object))
    mesh = rc.build_mesh(np.eye(3), divisions=(3, 3, 1), rotations=[rotation], time_reversal=False)
    assert mesh.operations_given == mesh.operations_used == 4
    numerators, denominator = rc._mesh_numerators((3, 3, 1), (0, 0, 0))
    lookup = {tuple(int(v) % denominator for v in point): index
              for index, point in enumerate(numerators)}
    labels = []
    for point in numerators:
        image = point.astype(object)
        members = []
        for _ in range(4):
            members.append(lookup[tuple(int(v) % denominator for v in image)])
            image = image @ rotation.astype(object)
        labels.append(min(members))
    representatives, weights = np.unique(labels, return_counts=True)
    assert np.array_equal(mesh.points, numerators[representatives].astype(float) / denominator)
    assert np.array_equal(mesh.weights, weights)


def test_numerator_grid_avoids_three_retained_meshgrid_arrays():
    tracemalloc.start()
    try:
        points, denominator = rc._mesh_numerators((128, 128, 16), (0, 1, 0))
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert points.shape == (128 * 128 * 16, 3)
    assert denominator == 256
    assert peak < 9 * 1024**2


def test_small_denominator_key_allocation_is_one_column_not_a_full_shifted_grid():
    points = np.zeros((100000, 3), dtype=np.int64)
    tracemalloc.start()
    try:
        keys = rc._encode(points, 64)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert keys.shape == (len(points),)
    assert peak < 2 * 1024**2


def test_wrapping_near_integer_limit_does_not_overflow_the_halfway_comparison():
    denominator = 2**63 - 2
    values = np.array([0, denominator // 2, denominator - 1], dtype=np.int64)
    assert np.array_equal(rc._wrap_numerators(values, denominator), [0, -denominator // 2, -1])


@pytest.mark.parametrize("shift", [float("nan"), float("inf"), 1e308])
def test_nonfinite_or_unrepresentable_shifts_raise_a_clear_value_error(shift):
    with pytest.raises(ValueError, match="finite|range"):
        rc.mesh_points((2, 2, 2), shift=(shift, 0, 0))


@pytest.mark.parametrize("divisions", [(2**62, 1, 1), (2**30, 2**30, 2)])
def test_unrepresentable_mesh_is_refused_before_attempting_allocation(divisions):
    with pytest.raises(ValueError, match="range"):
        rc.mesh_points(divisions)
