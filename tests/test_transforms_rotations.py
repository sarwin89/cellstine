"""A rotation has to be a rotation, and the shortcuts have to agree with it.

``rotation_matrix_about_axis`` is the general Rodrigues rotation; the ``x``,
``y`` and ``z`` routines are the three special cases that the rest of the tree
uses, and ``yaw_pitch_roll_matrix`` composes them.  If any of them picked up a
sign error the structures they turn would come out mirrored -- a chiral
molecule placed on a slab would be the wrong enantiomer, and nothing else in
the pipeline would notice.  So they are checked against each other and against
the two properties that define a rotation: it preserves every length, and it
preserves handedness.
"""

from __future__ import annotations

import numpy as np
import pytest

from cellstine.core.transforms import (
    rotation_matrix_about_axis,
    rotation_matrix_x,
    rotation_matrix_y,
    rotation_matrix_z,
    yaw_pitch_roll_matrix,
)

AXES = [
    (1.0, 0.0, 0.0),
    (0.0, 1.0, 0.0),
    (0.0, 0.0, 1.0),
    (1.0, 1.0, 1.0),
    (0.3, -0.7, 0.2),
    (-2.0, 0.5, 4.0),
]

ANGLES = [0.0, 17.0, 90.0, 123.5, 180.0, -47.0, 360.0]


@pytest.mark.parametrize("axis", AXES)
@pytest.mark.parametrize("angle", ANGLES)
def test_a_rotation_keeps_lengths_and_handedness(axis, angle):
    matrix = rotation_matrix_about_axis(axis, angle)
    assert matrix @ matrix.T == pytest.approx(np.eye(3), abs=1e-12)
    assert float(np.linalg.det(matrix)) == pytest.approx(1.0, abs=1e-12)


@pytest.mark.parametrize("axis", AXES)
@pytest.mark.parametrize("angle", ANGLES)
def test_the_axis_is_the_one_direction_that_does_not_move(axis, angle):
    unit = np.asarray(axis, dtype=float)
    unit = unit / np.linalg.norm(unit)
    assert rotation_matrix_about_axis(axis, angle) @ unit == pytest.approx(unit, abs=1e-12)


@pytest.mark.parametrize("axis", AXES)
@pytest.mark.parametrize("angle", [17.0, 90.0, 123.5, -47.0])
def test_the_turn_is_by_the_angle_asked_for_and_about_the_right_way(axis, angle):
    unit = np.asarray(axis, dtype=float)
    unit = unit / np.linalg.norm(unit)
    seed = np.array([1.0, -0.4, 0.25])
    perpendicular = seed - np.dot(seed, unit) * unit
    perpendicular /= np.linalg.norm(perpendicular)
    turned = rotation_matrix_about_axis(axis, angle) @ perpendicular
    cosine = float(np.dot(perpendicular, turned))
    sine = float(np.dot(np.cross(unit, perpendicular), turned))
    assert np.degrees(np.arctan2(sine, cosine)) == pytest.approx(
        np.degrees(np.arctan2(np.sin(np.radians(angle)), np.cos(np.radians(angle)))), abs=1e-9
    )


@pytest.mark.parametrize("angle", ANGLES)
def test_the_three_named_axes_are_the_general_rotation(angle):
    assert rotation_matrix_x(angle) == pytest.approx(
        rotation_matrix_about_axis((1.0, 0.0, 0.0), angle), abs=1e-12
    )
    assert rotation_matrix_y(angle) == pytest.approx(
        rotation_matrix_about_axis((0.0, 1.0, 0.0), angle), abs=1e-12
    )
    assert rotation_matrix_z(angle) == pytest.approx(
        rotation_matrix_about_axis((0.0, 0.0, 1.0), angle), abs=1e-12
    )


@pytest.mark.parametrize("axis", AXES)
@pytest.mark.parametrize("angle", ANGLES)
def test_turning_back_undoes_the_turn(axis, angle):
    forward = rotation_matrix_about_axis(axis, angle)
    assert forward @ rotation_matrix_about_axis(axis, -angle) == pytest.approx(np.eye(3), abs=1e-12)
    thrice = np.linalg.matrix_power(rotation_matrix_about_axis(axis, angle / 3.0), 3)
    assert thrice == pytest.approx(forward, abs=1e-12)


@pytest.mark.parametrize("axis", AXES)
def test_a_longer_axis_vector_means_the_same_rotation(axis):
    unit = np.asarray(axis, dtype=float)
    assert rotation_matrix_about_axis(unit * 7.5, 33.0) == pytest.approx(
        rotation_matrix_about_axis(unit, 33.0), abs=1e-12
    )


def test_a_zero_axis_is_not_a_rotation():
    with pytest.raises(ValueError):
        rotation_matrix_about_axis((0.0, 0.0, 0.0), 30.0)


@pytest.mark.parametrize(
    "yaw,pitch,roll", [(0.0, 0.0, 0.0), (30.0, 0.0, 0.0), (0.0, 45.0, 0.0), (10.0, -20.0, 75.0)]
)
def test_yaw_pitch_roll_applies_yaw_first(yaw, pitch, roll):
    composed = yaw_pitch_roll_matrix(yaw, pitch, roll)
    expected = rotation_matrix_x(roll) @ rotation_matrix_y(pitch) @ rotation_matrix_z(yaw)
    assert composed == pytest.approx(expected, abs=1e-12)
    assert composed @ composed.T == pytest.approx(np.eye(3), abs=1e-12)
    assert float(np.linalg.det(composed)) == pytest.approx(1.0, abs=1e-12)
