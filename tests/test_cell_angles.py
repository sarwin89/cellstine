"""The reported cell angles must be accurate over the whole range.

Every cell angle CELLSTINE reports -- the moire cell gamma, the in-plane angle
of a primitive surface cell, the alpha/beta/gamma of a symmetry report -- comes
from one shared helper.  Reading an angle as ``arccos`` of a normalised dot
product is only accurate away from ``0`` and ``180`` degrees: the derivative of
``arccos`` blows up there, so half of the digits are lost, and a nearly
degenerate cell is exactly where the number matters.  These tests compare the
helper against exact angles across the range and check that the workflows use
it.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from cellstine.core.lattice import vector_angle_deg


def _rotated_pair(angle_deg: float) -> tuple[np.ndarray, np.ndarray]:
    radians = math.radians(angle_deg)
    return (
        np.array([1.0, 0.0]),
        np.array([math.cos(radians), math.sin(radians)]),
    )


@pytest.mark.parametrize(
    "angle_deg",
    [1e-7, 1e-5, 1e-3, 0.1, 1.0, 17.3, 30.0, 60.0, 90.0, 120.0, 175.0, 179.999],
)
def test_the_angle_is_accurate_across_the_whole_range(angle_deg: float) -> None:
    """The relative error stays at the level of the arithmetic itself."""

    first, second = _rotated_pair(angle_deg)
    reported = vector_angle_deg(first, second)
    assert reported == pytest.approx(angle_deg, rel=1e-13, abs=1e-13)


def test_an_almost_aligned_pair_is_not_reported_as_aligned() -> None:
    """``arccos`` returns exactly zero here; the chord formula does not."""

    first, second = _rotated_pair(1e-7)
    cosine = float(np.dot(first, second))
    assert math.degrees(math.acos(min(1.0, cosine))) == 0.0
    assert vector_angle_deg(first, second) == pytest.approx(1e-7, rel=1e-12)


def test_orthogonal_vectors_give_exactly_ninety_degrees() -> None:
    assert vector_angle_deg([1.0, 0.0, 0.0], [0.0, 1.0, 0.0]) == 90.0
    assert vector_angle_deg([0.0, 2.5, 0.0], [0.0, 0.0, -0.5]) == 90.0


def test_the_angle_is_symmetric_and_scale_free() -> None:
    first = np.array([1.7, -0.3, 0.9])
    second = np.array([-0.4, 2.2, 0.1])
    forward = vector_angle_deg(first, second)
    assert forward == vector_angle_deg(second, first)
    assert forward == pytest.approx(vector_angle_deg(13.0 * first, 0.017 * second), abs=1e-13)


def test_antiparallel_vectors_give_one_hundred_and_eighty_degrees() -> None:
    assert vector_angle_deg([1.0, 2.0, -3.0], [-2.0, -4.0, 6.0]) == pytest.approx(180.0)


def test_a_zero_vector_is_rejected() -> None:
    with pytest.raises(ValueError):
        vector_angle_deg([0.0, 0.0], [1.0, 0.0])


def test_the_moire_and_surface_reports_use_the_shared_helper() -> None:
    """The reported cell angles are the helper's, not a private copy."""

    from cellstine.interface.surface import surface_cell, surface_supercell
    from cellstine.moire.search import nlayer, results
    from cellstine.symmetry import symmetry as symmetry_module

    for module in (
        results,
        nlayer,
        surface_cell,
        surface_supercell,
        symmetry_module,
    ):
        assert module.vector_angle_deg is vector_angle_deg


def test_a_hexagonal_moire_cell_angle_is_exactly_one_hundred_and_twenty() -> None:
    """A cell built from an exact hexagonal basis must report 120 degrees."""

    from cellstine.moire.search.results import _cell_angle_deg

    lattice = np.array([[2.46, -1.23], [0.0, 1.23 * math.sqrt(3.0)]])
    assert _cell_angle_deg(lattice) == pytest.approx(120.0, abs=1e-12)
