"""The twisted-bilayer-graphene series, checked against its closed form.

Every other moire test in this tree compares the engine either with itself
(brute force over the same definitions) or with a structure it built.  This one
compares it with the classical closed form for commensurate rotations of a
honeycomb bilayer, which is independent of anything in this repository: a
rotation of a hexagonal lattice onto itself is commensurate exactly for the
coprime pairs ``(m, r)`` with

    cos theta(m, r) = (3 m^2 + 3 m r + r^2 / 2) / (3 m^2 + 3 m r + r^2),

and the moire cell then holds

    Sigma(m, r) = (3 m^2 + 3 m r + r^2) / gcd(r, 3)

primitive cells of each layer, so it is ``a sqrt(Sigma)`` long and carries
``4 Sigma`` atoms in a bilayer.  Rotating by 60 degrees is a symmetry of the
lattice and of the honeycomb basis, so ``theta`` and ``60 - theta`` describe the
same bilayer and the angles fold into ``[0, 30]``.

The first few members are the familiar ones -- 21.787 degrees with 28 atoms in a
6.51 A cell, 13.174 with 76, 9.430 with 148, 7.341 with 244 -- and the test below
asserts that the unstrained search reports *exactly* the closed-form set, with
no member missing and nothing invented, and that the structure the builder then
writes really has those atoms in that cell.

The closed form itself is proved in ``RequestProject/HexagonalTwist.lean``:
``Cellstine.HexTwist.dotProduct_latVec`` (the Gram form, hence ``a sqrt(Q)``),
``Cellstine.HexTwist.twistCos_eq`` (the cosine),
``Cellstine.HexTwist.twistMatrix_gram``, ``twistMatrix_det`` and
``twistMatrix_mulVec`` (it is a rotation, and commensurate because it is
``1 / Q`` times an integer matrix in the lattice basis), and
``Cellstine.HexTwist.coincidence_mulVec``, ``coincidence_det`` and
``coincidence_length`` (the coincidence cell holds ``Q`` primitive cells).
"""

from __future__ import annotations

import math
from math import gcd
from pathlib import Path

import numpy as np
import pytest

from cellstine.core.species import group_species
from cellstine.io import native as io_mod
from cellstine.moire.builder.make import generate_from_results
from cellstine.moire.search.find import run_find

from conftest import hexagonal_basis

GRAPHENE_CONSTANT = 2.46
MAX_LENGTH = 20.0


def _write_graphene(path: Path) -> str:
    lattice = np.zeros((3, 3))
    lattice[:2, :2] = hexagonal_basis(GRAPHENE_CONSTANT).T
    lattice[2, 2] = 20.0
    positions = np.array([[1.0 / 3.0, 2.0 / 3.0, 0.5], [2.0 / 3.0, 1.0 / 3.0, 0.5]])
    ordered, counts, order = group_species(["C", "C"])
    io_mod.write_poscar(
        str(path),
        lattice,
        positions[order],
        [int(value) for value in counts],
        list(ordered),
        "graphene",
        positions_are_cartesian=False,
    )
    return str(path)


def commensurate_series(max_length: float, constant: float = GRAPHENE_CONSTANT):
    """Return the closed-form ``(angle, Sigma)`` pairs with a cell that fits.

    The enumeration is over coprime ``(m, r)`` and uses nothing from the
    package: it is the reference the search is measured against.
    """

    sigma_max = (max_length / constant) ** 2
    bound = int(math.isqrt(int(sigma_max)) * 3) + 4
    series: dict[tuple[float, int], tuple[int, int]] = {}
    for m in range(0, bound):
        for r in range(0, bound):
            if (m, r) == (0, 0) or gcd(m, r) != 1:
                continue
            quadratic = 3 * m * m + 3 * m * r + r * r
            divisor = 3 if r % 3 == 0 else 1
            sigma, remainder = divmod(quadratic, divisor)
            if remainder or sigma > sigma_max + 1e-9:
                continue
            cosine = (3 * m * m + 3 * m * r + 0.5 * r * r) / quadratic
            angle = math.degrees(math.acos(min(1.0, max(-1.0, cosine))))
            angle = min(angle, 60.0 - angle)
            series[(round(angle, 6), sigma)] = (m, r)
    return series


@pytest.fixture(scope="module")
def search(tmp_path_factory):
    workspace = tmp_path_factory.mktemp("graphene-twist-series")
    graphene = _write_graphene(workspace / "graphene.vasp")
    run = run_find(
        top_poscar=graphene,
        bottom_poscar=graphene,
        max_length=MAX_LENGTH,
        top_strain=0.0,
        bottom_strain=0.0,
        max_atoms=100000,
        max_candidates=500,
        output_root=str(workspace / "search"),
    )
    return graphene, run


def test_the_search_reports_exactly_the_closed_form_series(search):
    """No commensurate rotation is missed and none is invented."""

    _, run = search
    reported = {
        (round(float(candidate["angle_deg"]), 6), int(candidate["atom_count"]) // 4)
        for candidate in run.candidates
    }
    assert reported == set(commensurate_series(MAX_LENGTH))


def test_the_series_is_not_vacuous():
    """The reference really does list the members it is supposed to."""

    series = commensurate_series(MAX_LENGTH)
    assert len(series) == 9
    assert (0.0, 1) in series or (-0.0, 1) in series


@pytest.mark.parametrize("max_length", [12.0, 30.0, 40.0])
def test_a_longer_cell_budget_still_reports_exactly_the_series(tmp_path, max_length):
    """The agreement is not an accident of one cell budget."""

    graphene = _write_graphene(tmp_path / "graphene.vasp")
    run = run_find(
        top_poscar=graphene,
        bottom_poscar=graphene,
        max_length=max_length,
        top_strain=0.0,
        bottom_strain=0.0,
        max_atoms=1000000,
        max_candidates=5000,
        output_root=str(tmp_path / "search"),
    )
    reported = {
        (round(float(candidate["angle_deg"]), 6), int(candidate["atom_count"]) // 4)
        for candidate in run.candidates
    }
    assert reported == set(commensurate_series(max_length))


def test_every_reported_cell_has_the_closed_form_size(search):
    """``Sigma`` fixes the cell length, the cell angle and the atom count."""

    _, run = search
    series = commensurate_series(MAX_LENGTH)
    for candidate in run.candidates:
        angle = round(float(candidate["angle_deg"]), 6)
        sigma = int(candidate["atom_count"]) // 4
        assert (angle, sigma) in series
        expected = GRAPHENE_CONSTANT * math.sqrt(sigma)
        assert float(candidate["moire_a"]) == pytest.approx(expected, abs=1e-6)
        assert float(candidate["moire_b"]) == pytest.approx(expected, abs=1e-6)
        assert float(candidate["moire_gamma_deg"]) == pytest.approx(120.0, abs=1e-6)
        assert int(candidate["atom_count"]) == 4 * sigma
        assert max(abs(float(value)) for value in candidate["strain"]) < 1e-12


@pytest.mark.parametrize(
    "angle_deg, atoms",
    [
        (21.786789, 28),
        (13.173551, 76),
        (9.430008, 148),
        (7.340993, 244),
    ],
)
def test_the_named_members_of_the_series_are_reported(search, angle_deg, atoms):
    """The angles quoted for twisted bilayer graphene come out to six figures."""

    _, run = search
    matches = [
        candidate
        for candidate in run.candidates
        if abs(float(candidate["angle_deg"]) - angle_deg) < 1e-4
    ]
    assert len(matches) == 1, f"{angle_deg} deg should be reported exactly once"
    candidate = matches[0]
    assert int(candidate["atom_count"]) == atoms
    length = GRAPHENE_CONSTANT * math.sqrt(atoms // 4)
    assert float(candidate["moire_a"]) == pytest.approx(length, abs=1e-6)


def test_the_built_cell_of_the_first_member_matches_the_closed_form(search, tmp_path):
    """21.787 degrees really builds a 28-atom, 6.51 A bilayer with no clashes."""

    _, run = search
    index = next(
        int(candidate["index"])
        for candidate in run.candidates
        if abs(float(candidate["angle_deg"]) - 21.786789) < 1e-4
    )
    built = generate_from_results(
        str(run.result_path),
        index=index,
        interlayer_distance=3.35,
        output_dir=str(tmp_path / "build"),
    )
    structure = io_mod.read_poscar(str(built.output_path))
    assert structure.natoms == 28
    expected = GRAPHENE_CONSTANT * math.sqrt(7.0)
    lattice = np.asarray(structure.lattice, dtype=float)
    assert float(np.linalg.norm(lattice[0])) == pytest.approx(expected, abs=1e-6)
    assert float(np.linalg.norm(lattice[1])) == pytest.approx(expected, abs=1e-6)
    cartesian = np.asarray(structure.positions_cartesian, dtype=float)
    closest = math.inf
    for first in (-1, 0, 1):
        for second in (-1, 0, 1):
            shift = first * lattice[0] + second * lattice[1]
            deltas = cartesian[:, None, :] - (cartesian + shift)[None, :, :]
            distances = np.linalg.norm(deltas, axis=2)
            if first == 0 and second == 0:
                np.fill_diagonal(distances, math.inf)
            closest = min(closest, float(distances.min()))
    # The two layers sit 3.35 A apart and the in-plane bond is 1.42 A, so no
    # pair may come closer than a carbon bond.
    assert closest == pytest.approx(GRAPHENE_CONSTANT / math.sqrt(3.0), abs=1e-3)
