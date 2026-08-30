"""What a panel of the structure view shows, and what it hides.

Each 2D panel drops one coordinate.  `RequestProject/ViewProjection.lean` says
what that costs: the separation seen in the panel is never larger than the true
one and is equal to it exactly for two atoms at the same depth
(`Cellstine.planarProj_dist_le`, `Cellstine.planarProj_dist_eq_iff`), and two
atoms land on the same point exactly when the vector between them points along
the direction of observation (`Cellstine.planarProj_eq_iff_smul`) -- which is
when one hides the other.

The renderer therefore draws the species furthest from the observer first
(`Cellstine.drawn_later_of_separated`).  The markers of one species are
identical, so the order inside a species cannot change the picture; ordering the
species is enough, and it resolves every overlap whenever the species do not
interleave in depth, which is the case for an adsorbate on a substrate or the
two sides of an interface.
"""

from __future__ import annotations

import numpy as np
import pytest

from cellstine.core.directions import resolve_direction
from cellstine.visualize.backends.matplotlib import species_depth_order


CUBIC = np.diag([4.05, 4.05, 4.05])


def _frame(spec: str) -> np.ndarray:
    return resolve_direction(CUBIC, spec).frame(CUBIC)


@pytest.mark.parametrize("spec", ["c", "111", "[112]", "cart:1,2,3"])
def test_a_panel_never_exaggerates_a_separation(spec):
    frame = _frame(spec)
    rng = np.random.default_rng(20240817)
    points = rng.normal(size=(40, 3)) * 3.0
    turned = points @ frame.T

    for first, second in zip(turned[:-1], turned[1:]):
        seen = float(np.linalg.norm(first[:2] - second[:2]))
        true = float(np.linalg.norm(first - second))
        depth = abs(float(first[2] - second[2]))
        assert seen <= true + 1e-12
        assert seen**2 + depth**2 == pytest.approx(true**2, rel=1e-12, abs=1e-12)


def test_a_panel_shows_the_true_separation_of_two_atoms_at_one_depth():
    frame = _frame("111")
    direction = resolve_direction(CUBIC, "111").unit
    base = np.array([0.4, -1.2, 2.0])
    # a displacement perpendicular to the direction keeps the depth
    perpendicular = np.cross(direction, np.array([0.3, 0.7, -0.2]))
    other = base + perpendicular

    turned = np.vstack((base, other)) @ frame.T
    seen = float(np.linalg.norm(turned[0, :2] - turned[1, :2]))
    true = float(np.linalg.norm(base - other))
    assert turned[0, 2] == pytest.approx(turned[1, 2])
    assert seen == pytest.approx(true)


def test_two_atoms_overlap_exactly_along_the_direction_of_observation():
    frame = _frame("110")
    direction = resolve_direction(CUBIC, "110").unit
    base = np.array([1.0, 0.5, -0.75])

    hidden = base + 3.2 * direction
    turned = np.vstack((base, hidden)) @ frame.T
    assert np.allclose(turned[0, :2], turned[1, :2], atol=1e-12)
    assert turned[1, 2] - turned[0, 2] == pytest.approx(3.2)

    visible = base + np.array([0.0, 0.0, 1.0])
    turned = np.vstack((base, visible)) @ frame.T
    assert not np.allclose(turned[0, :2], turned[1, :2], atol=1e-9)


# --------------------------------------------------------------------------
# The drawing order of the species
# --------------------------------------------------------------------------


def _species(labels):
    return np.array(labels, dtype=object)


def test_a_molecule_above_a_substrate_is_drawn_last():
    species = _species(["Al", "Al", "C", "O"])
    positions = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 0.5],
            [0.5, 0.5, 5.0],
            [0.5, 0.5, 6.1],
        ]
    )
    # the plan view drops z, so the molecule species come last
    assert species_depth_order(species, positions, 2, ["Al", "C", "O"]) == ["Al", "C", "O"]


def test_the_order_follows_the_axis_the_panel_drops():
    species = _species(["A", "B"])
    positions = np.array([[9.0, 0.0, 0.0], [0.0, 0.0, 9.0]])
    unique = ["A", "B"]
    assert species_depth_order(species, positions, 0, unique) == ["B", "A"]  # x-panel drops x
    assert species_depth_order(species, positions, 2, unique) == ["A", "B"]
    assert species_depth_order(species, positions, 1, unique) == ["A", "B"]  # tied: file order


def test_the_order_is_a_permutation_of_the_species_and_is_deterministic():
    rng = np.random.default_rng(7)
    labels = ["Si", "O", "H", "Al"]
    species = _species([labels[index % len(labels)] for index in range(40)])
    positions = rng.normal(size=(40, 3))
    for axis in (0, 1, 2):
        order = species_depth_order(species, positions, axis, labels)
        assert sorted(order) == sorted(labels)
        assert species_depth_order(species, positions, axis, labels) == order


def test_a_species_with_no_atoms_does_not_break_the_order():
    species = _species(["Al", "Al"])
    positions = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 2.0]])
    assert species_depth_order(species, positions, 2, ["Ghost", "Al"]) == ["Ghost", "Al"]


def test_separated_species_are_drawn_from_the_far_side_forward():
    """The guarantee of `Cellstine.drawn_later_of_separated`, on real numbers."""

    rng = np.random.default_rng(11)
    substrate = np.column_stack(
        (rng.normal(size=12), rng.normal(size=12), rng.uniform(-3.0, 0.0, size=12))
    )
    molecule = np.column_stack(
        (rng.normal(size=4), rng.normal(size=4), rng.uniform(4.0, 6.0, size=4))
    )
    positions = np.vstack((substrate, molecule))
    species = _species(["Al"] * 12 + ["C"] * 4)

    order = species_depth_order(species, positions, 2, ["Al", "C"])
    assert order == ["Al", "C"]
    # every atom of the species drawn first is behind every atom drawn later
    first, last = order
    assert positions[species == first, 2].max() < positions[species == last, 2].min()
