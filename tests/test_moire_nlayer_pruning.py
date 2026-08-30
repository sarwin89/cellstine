"""The multi-layer combination stage must prune, and prune nothing real.

``moire findn`` runs the bilayer engine once per upper layer and then has to
pick one match per layer and intersect the base supercells.  Doing that with a
flat product over the per-layer lists costs ``limit ** layers`` intersections:
with the default ``--per-layer-limit 40`` a three-layer stack already needs
64000 of them, and a four-layer stack 2.56 million, so the stage dominated the
run time long before the search itself did.

Almost all of that work is wasted, because the two size limits are monotone:
the shared cell of a longer prefix is a sublattice of the shared cell of a
shorter one, so neither the atom count nor the cell length can ever come back
down as more layers are fixed.  :func:`viable_combinations` therefore walks the
layers depth first and abandons a prefix as soon as its own lower bounds break
a limit.

These tests pin both halves of that claim.  The enumeration must be *exactly* a
filter on the flat product -- same combinations, same order, and every dropped
combination one that :func:`combine_layer_matches` would have rejected anyway --
so the reported candidates are unchanged; and it must actually prune, both on
synthetic matches and end to end through :func:`run_findn`.
"""

from __future__ import annotations

from itertools import product
from pathlib import Path

import numpy as np
import pytest

from cellstine.core.species import group_species
from cellstine.io import native as io_mod
from cellstine.moire.search.nlayer import (
    LayerMatch,
    combine_layer_matches,
    read_nlayer_results,
    run_findn,
    viable_combinations,
)

from conftest import hexagonal_basis


def _hexagonal_lattice(constant: float = 2.46) -> np.ndarray:
    lattice = np.zeros((3, 3))
    lattice[:2, :2] = hexagonal_basis(constant).T
    lattice[2, 2] = 20.0
    return lattice


def _match(layer_index: int, base_matrix, layer_matrix, atom_count: int = 2) -> LayerMatch:
    return LayerMatch(
        layer_index=layer_index,
        poscar=f"layer_{layer_index}.vasp",
        atom_count=int(atom_count),
        base_matrix=np.asarray(base_matrix, dtype=np.int64),
        layer_matrix=np.asarray(layer_matrix, dtype=np.int64),
        affine=np.eye(2),
        angle_deg=0.0,
        strain=(0.0, 0.0),
    )


def _synthetic_layers(layers: int, per_layer: int, seed: int) -> list[list[LayerMatch]]:
    """Return per-layer match lists built from small nonsingular integer cells."""

    rng = np.random.default_rng(seed)
    matches_by_layer: list[list[LayerMatch]] = []
    for index in range(1, layers + 1):
        chosen: list[LayerMatch] = []
        while len(chosen) < per_layer:
            base = rng.integers(-3, 4, size=(2, 2))
            top = rng.integers(-3, 4, size=(2, 2))
            if round(float(np.linalg.det(base))) == 0 or round(float(np.linalg.det(top))) == 0:
                continue
            chosen.append(_match(index, base, top))
        matches_by_layer.append(chosen)
    return matches_by_layer


def _brute_force(matches_by_layer, **limits):
    return [
        combination
        for combination in product(*matches_by_layer)
        if combine_layer_matches(combination, **limits) is not None
    ]


@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize(
    "layers, per_layer, max_atoms, max_length",
    [
        (2, 6, 400, 30.0),
        (3, 4, 600, 40.0),
        (3, 4, None, 25.0),
        (3, 4, 200, None),
        (2, 8, 120, 18.0),
    ],
)
def test_pruning_keeps_every_admissible_combination(seed, layers, per_layer, max_atoms, max_length):
    """The walk drops only combinations the combiner would have rejected."""

    lattice = _hexagonal_lattice()
    matches_by_layer = _synthetic_layers(layers, per_layer, seed)
    limits = dict(
        base_lattice=lattice,
        base_atom_count=2,
        max_atoms=max_atoms,
        max_length=max_length,
    )
    walked = list(viable_combinations(matches_by_layer, **limits))
    kept = [
        combination
        for combination in walked
        if combine_layer_matches(combination, **limits) is not None
    ]
    assert kept == _brute_force(matches_by_layer, **limits)


@pytest.mark.parametrize("seed", range(4))
def test_pruning_preserves_the_product_order(seed):
    """The survivors appear in the order a flat product would have visited them."""

    lattice = _hexagonal_lattice()
    matches_by_layer = _synthetic_layers(3, 4, seed)
    limits = dict(base_lattice=lattice, base_atom_count=2, max_atoms=500, max_length=35.0)
    walked = list(viable_combinations(matches_by_layer, **limits))
    positions = {
        tuple(id(match) for match in combination): index
        for index, combination in enumerate(product(*matches_by_layer))
    }
    indices = [positions[tuple(id(match) for match in combo)] for combo in walked]
    assert indices == sorted(indices)


@pytest.mark.parametrize("seed", range(4))
def test_pruning_removes_most_of_the_product(seed):
    """A tight budget must leave far fewer than ``per_layer ** layers`` nodes."""

    lattice = _hexagonal_lattice()
    matches_by_layer = _synthetic_layers(3, 6, seed)
    limits = dict(base_lattice=lattice, base_atom_count=2, max_atoms=200, max_length=20.0)
    walked = list(viable_combinations(matches_by_layer, **limits))
    assert len(walked) < 6**3 // 2


def test_pruning_is_a_noop_without_limits():
    """With no size limits every combination survives, in product order."""

    lattice = _hexagonal_lattice()
    matches_by_layer = _synthetic_layers(2, 5, seed=11)
    walked = list(viable_combinations(matches_by_layer, base_lattice=lattice, base_atom_count=2))
    assert walked == list(product(*matches_by_layer))


def test_empty_layer_lists_yield_nothing():
    lattice = _hexagonal_lattice()
    assert list(viable_combinations([], base_lattice=lattice, base_atom_count=2)) == []
    matches_by_layer = _synthetic_layers(1, 3, seed=3) + [[]]
    assert list(viable_combinations(matches_by_layer, base_lattice=lattice, base_atom_count=2)) == []


def _honeycomb(constant: float, species: list[str]):
    lattice = _hexagonal_lattice(constant)
    positions = np.array([[1.0 / 3.0, 2.0 / 3.0, 0.5], [2.0 / 3.0, 1.0 / 3.0, 0.5]])
    return lattice, positions, list(species)


def _write(path: Path, lattice, positions, species) -> str:
    ordered, counts, order = group_species(list(species))
    io_mod.write_poscar(
        str(path),
        np.asarray(lattice, dtype=float),
        np.asarray(positions, dtype=float)[order],
        [int(value) for value in counts],
        list(ordered),
        "layer",
        positions_are_cartesian=False,
    )
    return str(path)


def test_three_upper_layers_stay_fast_and_report_the_same_stacks(tmp_path):
    """End to end: the pruned walk reproduces the recorded candidates."""

    graphene = _write(tmp_path / "graphene.vasp", *_honeycomb(2.46, ["C", "C"]))
    hbn = _write(tmp_path / "hbn.vasp", *_honeycomb(2.504, ["B", "N"]))
    mos2 = _write(tmp_path / "mos2.vasp", *_honeycomb(3.16, ["Mo", "S"]))
    run = run_findn(
        base_poscar=graphene,
        upper_poscars=[hbn, graphene, mos2],
        max_length=25.0,
        layer_strains=0.03,
        max_atoms=2000,
        per_layer_limit=40,
        max_candidates=200,
        output_root=str(tmp_path / "out"),
    )
    document = read_nlayer_results(run.result_path)
    assert document["candidates"], "a graphene-based three-layer stack must have candidates"
    for candidate in document["candidates"]:
        assert candidate["total_atoms"] <= 2000
        assert max(candidate["cell_lengths"]) <= 25.0 + 1e-9
        assert len(candidate["layers"]) == 3
    # The flat product needs 40**3 = 64000 intersections here and took more than
    # ten seconds; the pruned walk takes well under one.  The bound is loose so
    # that a slow machine does not fail the test, but tight enough to catch a
    # return to the product.
    assert run.timings["combine_s"] < 5.0
