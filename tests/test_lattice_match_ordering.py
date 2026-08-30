"""The ranking of interface lattice matches, and the resolution it uses.

``interface/workflow/lattice_match.py`` orders its matches by strain, then by
cell size, then by area, then by twist, and it compares the strain only after
snapping it to a grid of spacing ``STRAIN_ORDER_RESOLUTION``.  The guarantees
that snapping has to provide are proved in ``RequestProject/MatchOrdering.lean``
(``Cellstine.lt_of_quantise_lt``, ``Cellstine.abs_sub_le_of_quantise_eq``,
``Cellstine.quantise_lt_of_add_lt``, ``Cellstine.strain_lt_or_tie``); this file
pins them onto the Python that is actually run.
"""

from __future__ import annotations

import itertools
import math
import random

import numpy as np
import pytest

from cellstine.interface.workflow.lattice_match import (
    STRAIN_ORDER_RESOLUTION,
    MatchRequest,
    match_order_key,
    sort_matches,
)


def _entry(strain: float, *, atoms: int = 10, area: float = 1.0, angle: float = 0.0) -> dict:
    return {
        "strain": float(strain),
        "total_atoms": int(atoms),
        "surface_area": float(area),
        "angle_deg": float(angle),
    }


def _bucket(strain: float) -> float:
    return match_order_key(_entry(strain))[0]


# --------------------------------------------------------------------------
# The snapped strain: Cellstine.quantise
# --------------------------------------------------------------------------


def test_snapping_moves_a_strain_by_at_most_half_the_resolution():
    """``Cellstine.abs_quantise_sub_self_le``."""

    rng = random.Random(20240828)
    for _ in range(2000):
        strain = rng.uniform(0.0, 0.2)
        assert abs(_bucket(strain) - strain) <= 0.5 * STRAIN_ORDER_RESOLUTION * (1 + 1e-9)


def test_snapping_is_order_preserving():
    """``Cellstine.quantise_mono``."""

    rng = random.Random(11)
    values = sorted(rng.uniform(0.0, 0.05) for _ in range(500))
    buckets = [_bucket(value) for value in values]
    assert all(left <= right for left, right in zip(buckets, buckets[1:]))


def test_a_lower_bucket_means_a_genuinely_lower_strain():
    """``Cellstine.lt_of_quantise_lt``: the ranking never inverts a real difference."""

    rng = random.Random(12)
    values = [rng.uniform(0.0, 0.05) for _ in range(300)]
    for left, right in itertools.combinations(values, 2):
        if _bucket(left) < _bucket(right):
            assert left < right


def test_an_equal_bucket_means_the_strains_are_within_the_resolution():
    """``Cellstine.abs_sub_le_of_quantise_eq``: a tie is a genuine tie."""

    rng = random.Random(13)
    base = 0.0134567890123
    for _ in range(2000):
        other = base + rng.uniform(-2.0, 2.0) * STRAIN_ORDER_RESOLUTION
        if _bucket(base) == _bucket(other):
            assert abs(base - other) <= STRAIN_ORDER_RESOLUTION * (1 + 1e-9)


def test_strains_further_apart_than_the_resolution_never_merge():
    """``Cellstine.quantise_lt_of_add_lt``: nothing coarser than the resolution is lost."""

    rng = random.Random(14)
    for _ in range(2000):
        left = rng.uniform(0.0, 0.05)
        right = left + STRAIN_ORDER_RESOLUTION * rng.uniform(1.0001, 50.0)
        assert _bucket(left) < _bucket(right)


def test_last_bit_noise_in_a_strain_does_not_decide_the_ranking():
    """The motivating case: the same deformation reached two ways.

    A strain of about 1e-2 has a floating-point spacing near 1.7e-18, so a few
    ulps of disagreement is far below the resolution and must not let the larger
    of two equally strained cells be offered first.
    """

    strain = 0.0134567890123
    noisy = strain
    for _ in range(64):
        noisy = math.nextafter(noisy, 1.0)
    assert noisy != strain
    assert abs(noisy - strain) < STRAIN_ORDER_RESOLUTION

    small = _entry(noisy, atoms=24, area=30.0)
    large = _entry(strain, atoms=96, area=120.0)
    ordered = sort_matches([large, small])
    assert [entry["total_atoms"] for entry in ordered] == [24, 96]
    assert [entry["index"] for entry in ordered] == [1, 2]


# --------------------------------------------------------------------------
# The sort itself
# --------------------------------------------------------------------------


def test_sort_matches_reindexes_and_never_inverts_a_real_strain_difference():
    rng = random.Random(15)
    entries = [
        _entry(
            rng.choice([0.001, 0.001 + 1e-16, 0.004, 0.02]),
            atoms=rng.randrange(4, 200),
            area=rng.uniform(5.0, 500.0),
            angle=rng.uniform(-30.0, 30.0),
        )
        for _ in range(120)
    ]
    ordered = sort_matches(entries)

    assert [entry["index"] for entry in ordered] == list(range(1, len(entries) + 1))
    # The input records are not mutated by the sort.
    assert all("index" not in entry for entry in entries)

    for earlier, later in zip(ordered, ordered[1:]):
        # ``Cellstine.strain_lt_or_tie``.
        assert (
            earlier["strain"] < later["strain"]
            or abs(earlier["strain"] - later["strain"]) <= STRAIN_ORDER_RESOLUTION
        )
    # Within one strain bucket the smaller cell comes first.
    for earlier, later in zip(ordered, ordered[1:]):
        if match_order_key(earlier)[0] == match_order_key(later)[0]:
            assert earlier["total_atoms"] <= later["total_atoms"]


def test_the_tie_break_runs_size_then_area_then_twist():
    strain = 0.003
    entries = [
        _entry(strain, atoms=40, area=90.0, angle=21.0),
        _entry(strain, atoms=40, area=90.0, angle=-3.0),
        _entry(strain, atoms=40, area=12.0, angle=17.0),
        _entry(strain, atoms=12, area=400.0, angle=0.0),
    ]
    ordered = sort_matches(entries)
    assert [(e["total_atoms"], e["surface_area"], e["angle_deg"]) for e in ordered] == [
        (12, 400.0, 0.0),
        (40, 12.0, 17.0),
        (40, 90.0, -3.0),
        (40, 90.0, 21.0),
    ]


def test_sorting_is_a_total_order_so_the_result_does_not_depend_on_the_input_order():
    rng = random.Random(16)
    entries = [
        _entry(
            rng.choice([0.001, 0.004]),
            atoms=rng.randrange(4, 40),
            area=rng.uniform(5.0, 50.0),
            angle=rng.uniform(-30.0, 30.0),
        )
        for _ in range(40)
    ]
    reference = [match_order_key(entry) for entry in sort_matches(entries)]
    for _ in range(20):
        shuffled = list(entries)
        rng.shuffle(shuffled)
        assert [match_order_key(entry) for entry in sort_matches(shuffled)] == reference


# --------------------------------------------------------------------------
# The strain budgets the search is run under
# --------------------------------------------------------------------------


def test_strain_budgets_match_the_documented_modes():
    """``shared`` splits the relative strain; ``film`` holds the substrate rigid."""

    shared = MatchRequest(max_length=30.0, max_strain=0.03, strain_mode="shared")
    assert shared.strain_budgets == (0.03, 0.03)

    film = MatchRequest(max_length=30.0, max_strain=0.03, strain_mode="film")
    assert film.strain_budgets == (0.0, 0.03)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_length": 0.0},
        {"max_length": -1.0},
        {"max_length": float("nan")},
        {"max_length": 10.0, "max_strain": 0.0},
        {"max_length": 10.0, "max_strain": float("inf")},
        {"max_length": 10.0, "strain_mode": "rigid"},
    ],
)
def test_a_meaningless_request_is_refused(kwargs):
    with pytest.raises(ValueError):
        MatchRequest(**kwargs)


def test_the_reported_surface_area_is_the_determinant_of_the_shared_lattice():
    from cellstine.interface.workflow.lattice_match import _cell_area

    rng = np.random.default_rng(17)
    for _ in range(200):
        lattice = rng.normal(size=(2, 2))
        assert _cell_area(lattice.tolist()) == pytest.approx(abs(np.linalg.det(lattice)), rel=1e-12)
