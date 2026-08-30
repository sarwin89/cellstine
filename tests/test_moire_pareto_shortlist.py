"""The Pareto front of the moire search, and the shortlist built from it.

A wide search returns far more admissible supercells than anyone wants written
out, and :func:`cellstine.moire.search.results.shortlist_offsets` decides which
survive: the whole Pareto front of *(atom count, relative strain)*, then the
smallest remaining cells in rank order.  The front itself is the running-minimum
sweep :func:`cellstine.moire.search.gram_pairs._pareto_front`.

The statements checked here are the ones proved in
``RequestProject/ParetoFront.lean``:

* the sweep keeps a candidate exactly when nothing scanned before it is at least
  as good in both costs (``Cellstine.Pareto.isRecord_iff_not_dominated``),
* a kept candidate is undominated outright
  (``Cellstine.Pareto.eq_costs_of_isRecord_of_le``),
* every candidate is matched or beaten in both costs by a kept one, so the
  shortlist loses nothing (``Cellstine.Pareto.exists_isRecord_le``,
  ``Cellstine.Pareto.exists_mem_le_of_isRecord_subset``),
* the front is a strict staircase: distinct kept candidates have distinct atom
  counts, and more atoms buys strictly less strain
  (``Cellstine.Pareto.eq_of_isRecord_of_first_eq``,
  ``Cellstine.Pareto.second_lt_of_isRecord_of_first_lt``).

The last three are invariant under relabelling the candidates, so they are also
checked on a real graphene-on-graphene search, whose arrays are re-sorted after
the sweep has run.
"""

from __future__ import annotations

import math
import random

import numpy as np
import pytest

from cellstine.moire.search import gram
from cellstine.moire.search.gram_pairs import _pareto_front
from cellstine.moire.search.results import shortlist_offsets

from conftest import hexagonal_basis


def _reference_front(first: np.ndarray, second: np.ndarray) -> set[int]:
    """The proved characterisation, evaluated by brute force over all pairs.

    ``i`` is kept exactly when no ``j`` that sorts before it under
    ``(first, second, index)`` is at least as good in both costs.
    """

    def before(j: int, i: int) -> bool:
        return (float(first[j]), float(second[j]), j) < (
            float(first[i]),
            float(second[i]),
            i,
        )

    return {
        i
        for i in range(len(first))
        if not any(
            before(j, i) and first[j] <= first[i] and second[j] <= second[i]
            for j in range(len(first))
        )
    }


def _cases() -> list[tuple[np.ndarray, np.ndarray]]:
    rng = random.Random(20240607)
    cases: list[tuple[np.ndarray, np.ndarray]] = [
        (np.zeros(0), np.zeros(0)),
        (np.array([3.0]), np.array([0.5])),
        (np.array([1.0, 1.0, 1.0]), np.array([2.0, 2.0, 2.0])),
        (np.array([1.0, 2.0, 3.0]), np.array([3.0, 2.0, 1.0])),
        (np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 3.0])),
    ]
    for _ in range(40):
        size = rng.randint(1, 30)
        # Small integer costs, so ties in one or both coordinates are common.
        first = np.array([float(rng.randint(0, 5)) for _ in range(size)])
        second = np.array([float(rng.randint(0, 5)) for _ in range(size)])
        cases.append((first, second))
    for _ in range(10):
        size = rng.randint(1, 30)
        first = np.array([float(rng.randint(1, 200)) for _ in range(size)])
        second = np.array([rng.random() for _ in range(size)])
        cases.append((first, second))
    return cases


CASES = _cases()


@pytest.mark.parametrize("first,second", CASES)
def test_the_sweep_returns_exactly_the_undominated_candidates(first, second):
    assert set(_pareto_front(first, second).tolist()) == _reference_front(first, second)


@pytest.mark.parametrize("first,second", CASES)
def test_the_sweep_reports_increasing_cells_of_strictly_decreasing_strain(first, second):
    front = _pareto_front(first, second)
    assert list(front) == sorted(front, key=lambda i: (first[i], second[i], i))
    for previous, current in zip(front, front[1:]):
        assert first[previous] < first[current]
        assert second[current] < second[previous]


@pytest.mark.parametrize("first,second", CASES)
def test_no_kept_candidate_is_dominated_and_none_is_lost(first, second):
    front = _pareto_front(first, second)
    for i in front:
        for j in range(len(first)):
            if first[j] <= first[i] and second[j] <= second[i]:
                # Anything at least as good in both costs has the same costs.
                assert first[j] == first[i] and second[j] == second[i]
    for i in range(len(first)):
        assert any(first[k] <= first[i] and second[k] <= second[i] for k in front)


def test_the_front_is_stable_under_relabelling_the_candidates():
    rng = random.Random(11)
    first = np.array([float(rng.randint(0, 6)) for _ in range(25)])
    second = np.array([float(rng.randint(0, 6)) for _ in range(25)])
    permutation = np.array(rng.sample(range(25), 25))
    front = _pareto_front(first, second)
    permuted = _pareto_front(first[permutation], second[permutation])
    # The kept *cost pairs* cannot depend on the order the candidates arrive in;
    # only which of several identical candidates is picked can.
    assert sorted(zip(first[front], second[front])) == sorted(
        zip(first[permutation][permuted], second[permutation][permuted])
    )


def _fake_result(pareto: list[bool]):
    """A `SearchResult` carrying nothing but the flags the shortlist reads."""

    count = len(pareto)
    zeros = np.zeros(count)
    return gram.SearchResult(
        top_matrices=np.zeros((count, 2, 2), dtype=np.int64),
        bottom_matrices=np.zeros((count, 2, 2), dtype=np.int64),
        top_gram=np.zeros((count, 3)),
        bottom_gram=np.zeros((count, 3)),
        twist_radians=zeros,
        twist_degrees=zeros,
        principal_strains=np.zeros((count, 2)),
        sharing_fraction=zeros,
        top_atom_counts=np.zeros(count, dtype=np.int64),
        bottom_atom_counts=np.zeros(count, dtype=np.int64),
        atom_counts=np.zeros(count, dtype=np.int64),
        loewner_certified=np.zeros(count, dtype=bool),
        loewner_borderline=np.zeros(count, dtype=bool),
        top_affine=np.zeros((count, 2, 2)),
        bottom_affine=np.zeros((count, 2, 2)),
        shared_lattice=np.zeros((count, 2, 2)),
        canonical_keys=np.zeros((count, 8), dtype=np.int64),
        pareto_optimal=np.array(pareto, dtype=bool),
        rank=np.arange(count, dtype=np.int64),
        stats={},
        raw_twist_radians=zeros,
        coincidence_indices=np.ones(count, dtype=np.int64),
    )


def test_the_shortlist_keeps_the_whole_front_and_fills_with_the_smallest_cells():
    flags = [False, True, False, False, True, False, False, True]
    result = _fake_result(flags)
    assert shortlist_offsets(result, None) == list(range(8))
    assert shortlist_offsets(result, 0) == list(range(8))
    assert shortlist_offsets(result, 20) == list(range(8))
    # The front is never dropped, and the rest of the budget goes to the
    # smallest cells, which lead the rank order.
    assert shortlist_offsets(result, 5) == [0, 1, 2, 4, 7]
    assert shortlist_offsets(result, 4) == [0, 1, 4, 7]
    # A budget below the size of the front is exceeded rather than truncating it.
    assert shortlist_offsets(result, 2) == [1, 4, 7]


def test_the_shortlist_still_matches_every_candidate_in_both_costs():
    rng = random.Random(5)
    first = np.array([float(rng.randint(0, 8)) for _ in range(40)])
    second = np.array([float(rng.randint(0, 8)) for _ in range(40)])
    order = np.lexsort((second, first))
    first, second = first[order], second[order]
    flags = [False] * 40
    for index in _pareto_front(first, second):
        flags[int(index)] = True
    kept = shortlist_offsets(_fake_result(flags), 12)
    assert len(kept) >= sum(flags)
    for i in range(40):
        assert any(first[k] <= first[i] and second[k] <= second[i] for k in kept)


@pytest.fixture(scope="module")
def mismatched_result():
    """A mismatched hexagonal bilayer, which has a genuine trade-off front."""

    return gram.search(
        gram.SearchConfig(
            top_basis=hexagonal_basis(2.46),
            bottom_basis=hexagonal_basis(2.504),
            top_atoms=2,
            bottom_atoms=2,
            max_length=30.0,
            top_strain=0.02,
            bottom_strain=0.02,
        )
    )


def test_a_real_search_reports_an_undominated_front(mismatched_result):
    result = mismatched_result
    assert len(result) > 0
    atoms = result.atom_counts.astype(float)
    strain = np.max(np.abs(result.principal_strains), axis=1)
    front = np.flatnonzero(result.pareto_optimal)
    # A genuine trade-off: several cells, none of which dominates another.
    assert front.size >= 3 and len(result) > 1000
    for i in front:
        for j in range(len(result)):
            if atoms[j] <= atoms[i] and strain[j] <= strain[i]:
                assert atoms[j] == atoms[i] and strain[j] == strain[i]


def test_a_real_search_loses_no_trade_off(mismatched_result):
    result = mismatched_result
    atoms = result.atom_counts.astype(float)
    strain = np.max(np.abs(result.principal_strains), axis=1)
    front = np.flatnonzero(result.pareto_optimal)
    for i in range(len(result)):
        assert any(atoms[k] <= atoms[i] and strain[k] <= strain[i] for k in front)


def test_a_real_front_is_a_strict_staircase(mismatched_result):
    result = mismatched_result
    atoms = result.atom_counts.astype(float)
    strain = np.max(np.abs(result.principal_strains), axis=1)
    front = sorted(np.flatnonzero(result.pareto_optimal), key=lambda i: atoms[i])
    for previous, current in zip(front, front[1:]):
        assert atoms[previous] < atoms[current]
        assert strain[current] < strain[previous]
    # The smallest cell on the front is the smallest cell in the whole search,
    # and the least strained one is the least strained in the whole search.
    assert atoms[front[0]] == atoms.min()
    assert math.isclose(strain[front[-1]], strain.min(), rel_tol=0.0, abs_tol=0.0)


def test_a_real_shortlist_keeps_the_front_and_respects_the_budget(mismatched_result):
    result = mismatched_result
    atoms = result.atom_counts.astype(float)
    strain = np.max(np.abs(result.principal_strains), axis=1)
    kept = shortlist_offsets(result, 20)
    assert len(kept) == 20
    assert set(np.flatnonzero(result.pareto_optimal).tolist()) <= set(kept)
    for i in range(len(result)):
        assert any(atoms[k] <= atoms[i] and strain[k] <= strain[i] for k in kept)
