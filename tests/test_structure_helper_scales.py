"""Finite rescaling must preserve rotations, frames and physical vacuum gaps."""

import numpy as np
import pytest

from cellstine.core import directions, transforms, vacuum
from cellstine.io.models import StructureRecord


@pytest.mark.parametrize("axis", [(1., 0., 0.), (1., 1., 1.), (1., -2., 3.)])
def test_finite_long_axis_preserves_the_rotation(axis):
    small = np.array(axis)
    large = small * 1e200
    with np.errstate(over="ignore", invalid="ignore"):
        actual = transforms.rotation_matrix_about_axis(large, 37.)
    assert np.allclose(actual, transforms.rotation_matrix_about_axis(small, 37.), atol=1e-14, rtol=0.)
    assert np.allclose(actual @ actual.T, np.eye(3), atol=1e-14, rtol=0.)
    assert np.linalg.det(actual) == pytest.approx(1., abs=1e-14)


@pytest.mark.parametrize("axis", [(1., 0., 0.), (1., 1., 1.), (1., -2., 3.)])
def test_finite_long_observation_axis_preserves_unit_and_family(axis):
    with np.errstate(over="ignore", invalid="ignore"):
        actual = directions.resolve_direction(np.eye(3), np.array(axis) * 1e200)
    expected = directions.resolve_direction(np.eye(3), axis)
    assert np.allclose(actual.unit, expected.unit, atol=1e-14, rtol=0.)
    assert actual.miller == expected.miller
    assert actual.spacing == pytest.approx(expected.spacing, abs=1e-14)


def test_finite_long_view_direction_keeps_a_right_handed_frame():
    with np.errstate(over="ignore", invalid="ignore"):
        actual = directions.orthonormal_frame(np.array([1e200, 0., 0.]))
    expected = directions.orthonormal_frame(np.array([1., 0., 0.]))
    assert np.allclose(actual, expected, atol=1e-14, rtol=0.)
    assert np.linalg.det(actual) == pytest.approx(1., abs=1e-14)


def test_large_finite_lattice_retains_its_in_plane_orientation():
    lattice = np.array([[1e200, 1e200, 0.], [-1., 1., 0.], [0., 0., 10.]])
    with np.errstate(over="ignore", invalid="ignore"):
        frame = directions.orthonormal_frame(np.array([0., 0., 1.]), lattice)
    assert np.allclose(frame[0], [2**-0.5, 2**-0.5, 0.], atol=1e-14, rtol=0.)
    assert np.allclose(frame @ frame.T, np.eye(3), atol=1e-14, rtol=0.)


@pytest.mark.parametrize("scale", [1e100, 1e200, 1e-100, 1e-200])
@pytest.mark.parametrize("sign", [1., -1.])
def test_scaled_surface_plane_keeps_normal_height_and_fitted_gap(scale, sign):
    lattice = np.array([[scale, 2*scale, 0.], [-2*scale, scale, 0.], [0., 0., sign*10.]])
    points = np.array([[0., 0., 1.], [0., 0., 2.]])
    with np.errstate(over="ignore", under="ignore", invalid="ignore", divide="ignore"):
        normal = vacuum.surface_normal(lattice)
        assert np.array_equal(normal, [0., 0., sign])
        assert vacuum.cell_height(lattice) == pytest.approx(10.)
        fitted, shifted = vacuum.fit_cell_to_vacuum(lattice, points, 20.)
        assert vacuum.vacuum_gap(fitted, shifted) == pytest.approx(20., abs=1e-12)
    assert np.array_equal(lattice[2], [0., 0., sign*10.])
    assert np.allclose(shifted[1] - shifted[0], points[1] - points[0], atol=1e-14)


def test_absolute_zero_thresholds_are_preserved():
    with pytest.raises(ValueError):
        transforms.rotation_matrix_about_axis([1e-16, 0., 0.], 30.)
    with pytest.raises(ValueError):
        directions.resolve_direction(np.eye(3), [1e-10, 0., 0.])
    with pytest.raises(ValueError):
        vacuum.surface_normal(np.array([[1., 0., 0.], [2., 0., 0.], [0., 0., 1.]]))


def test_strained_copy_uses_numpy126_compatible_array_ownership(monkeypatch):
    points = np.array([[.1, .2, .3]])
    points.flags.writeable = False
    original = StructureRecord("source", np.eye(3), ["C"], [1], points, points.copy(), metadata={"source": True})
    target = np.diag([2., 3., 1.])
    asarray = np.asarray

    def numpy126_asarray(a, dtype=None, order=None, *, like=None):
        # Dependency boundary from the declared NumPy1.26 API: copy is not an
        # asarray keyword. This test also runs against the real minimum job.
        return asarray(a, dtype=dtype, order=order, like=like)

    monkeypatch.setattr(np, "asarray", numpy126_asarray)
    copied = transforms.strained_copy(original, target)
    assert np.array_equal(copied.lattice, target)
    assert np.allclose(copied.positions_cartesian, points @ target, atol=1e-14)
    assert copied.metadata == {"source": True, "strained_to_interface": True}
    assert original.metadata == {"source": True}
    copied.positions_direct[0, 0] = .8
    assert original.positions_direct[0, 0] == .1


@pytest.mark.parametrize("scale", [1e-79, 1e-80, 1e-158, 1e-160, 1e-162])
def test_tilted_tiny_planes_do_not_reuse_quantized_cross_products(scale):
    base = np.array([[1., 2., 3.], [-2., 1., .7], [0., 0., 10.]])
    scaled = base.copy()
    scaled[:2] *= scale
    expected = np.cross(base[0], base[1])
    expected /= np.linalg.norm(expected)
    normal = vacuum.surface_normal(scaled)
    assert np.allclose(normal, expected, atol=1e-14, rtol=0.)
    assert vacuum.cell_height(scaled) == pytest.approx(abs(base[2] @ expected), abs=1e-13)
    points = np.array([[0., 0., 1.], [0., 0., 2.]])
    fitted, shifted = vacuum.fit_cell_to_vacuum(scaled, points, 20.)
    assert vacuum.vacuum_gap(fitted, shifted) == pytest.approx(20., abs=1e-12)


def test_extreme_aspect_ratio_does_not_erase_a_nonparallel_plane():
    lattice = np.array([[1e100, 1e-300, 0.], [1e100, 0., 1e-300], [0., 0., 10.]])
    normal = vacuum.surface_normal(lattice)
    assert np.allclose(normal, [0., 2**-0.5, 2**-0.5], atol=1e-14, rtol=0.)
    assert np.linalg.norm(normal) == pytest.approx(1., abs=1e-14)


@pytest.mark.parametrize("minor", [1e-273, 2e-273])
def test_subnormal_scaled_cross_does_not_quantize_unequal_components(minor):
    lattice = np.array([[1e50, minor, 0.], [1e50, 0., .7*minor], [0., 0., 10.]])
    expected = np.array([0., .7, 1.])
    expected /= np.linalg.norm(expected)
    assert np.allclose(vacuum.surface_normal(lattice), expected, atol=1e-14, rtol=0.)
