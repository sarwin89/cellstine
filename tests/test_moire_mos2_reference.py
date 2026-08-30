"""Published twisted-bilayer MoS2 numbers, reproduced end to end.

The reference is a set of AA'-stacked (2H) MoS2 bilayers twisted to the
commensurate angles below.  For each one the literature records the size of the
moire cell and how many genuinely different sulfur sites it contains:

===========  =====  =========  ==================  ==========================
twist angle  atoms  S atoms    S atoms per plane   inequivalent S sites/plane
===========  =====  =========  ==================  ==========================
21.787         42       28              7                      3
27.796         78       52             13                      5
13.173        114       76             19                      7
9.430         222      148             37                     13
16.426        294      196             49                     17
7.341         366      244             61                     21
===========  =====  =========  ==================  ==========================

Each row is one commensurate index ``N = m^2 + m n + n^2`` of the triangular
lattice: the moire cell holds ``N`` primitive cells per layer, hence ``6 N``
atoms, ``4 N`` of them sulfur, ``N`` in each of the four sulfur planes.  The
last column is the number of orbits of the three-fold rotation on one such
plane, ``(N + 2) / 3``; the derivation is machine-checked in
``RequestProject/TwistedBilayer.lean``.

The whole pipeline is exercised here: the rigid Gram-form search must produce
the angle and the atom counts, the builder must write a structure with those
counts, and the defect analysis must find exactly that many inequivalent sites
in each sulfur plane.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from cellstine.cli.main import execute_namespace
from cellstine.cli.parsers import build_parser
from cellstine.defect.workflow import Defect
from cellstine.io import native as io

from conftest import hexagonal_basis, write_poscar

# angle in degrees, (m, n), total atoms, S atoms, S per plane, sites per plane
REFERENCE_ROWS = [
    (21.787, (1, 2), 42, 28, 7, 3),
    (27.796, (1, 3), 78, 52, 13, 5),
    (13.173, (2, 3), 114, 76, 19, 7),
    (9.430, (3, 4), 222, 148, 37, 13),
    (16.426, (3, 5), 294, 196, 49, 17),
    (7.341, (4, 5), 366, 244, 61, 21),
]

LATTICE_CONSTANT = 3.16
VACUUM = 24.0
SULFUR_HEIGHT = 1.5636


def _monolayer(path: Path, *, mirrored: bool) -> Path:
    """Write a 1H-MoS2 monolayer; ``mirrored`` is the A' partner of A.

    A and A' differ by the in-plane 180 degree rotation that takes the sulfur
    column from one hollow of the molybdenum lattice to the other, which is
    exactly what AA' (2H) stacking puts on top of A.
    """

    lattice = np.zeros((3, 3))
    lattice[:2, :2] = hexagonal_basis(LATTICE_CONSTANT).T
    lattice[2, 2] = VACUUM
    height = SULFUR_HEIGHT / VACUUM
    sulfur = (2.0 / 3.0, 1.0 / 3.0) if mirrored else (1.0 / 3.0, 2.0 / 3.0)
    positions = np.array(
        [
            [0.0, 0.0, 0.5],
            [sulfur[0], sulfur[1], 0.5 + height],
            [sulfur[0], sulfur[1], 0.5 - height],
        ]
    )
    return write_poscar(
        path,
        lattice,
        ["Mo", "S"],
        [1, 2],
        positions,
        comment="MoS2 monolayer" + (" (A')" if mirrored else " (A)"),
    )


def _reference_angles(m: int, n: int) -> tuple[float, float]:
    """The two commensurate twist angles of one index pair, in degrees.

    A hexagonal index pair gives a moire cell at ``arccos((m^2 + 4mn + n^2) /
    (2 N))`` and a second, equally large one at its supplement to 60 degrees:
    the layers have three-fold, not six-fold, symmetry, so the two are
    different structures with the same cell.
    """

    index = m * m + m * n + n * n
    angle = math.degrees(math.acos((m * m + 4 * m * n + n * n) / (2.0 * index)))
    return angle, 60.0 - angle


def _angle_error(m: int, n: int, angle: float) -> float:
    return min(abs(value - angle) for value in _reference_angles(m, n))


@pytest.fixture(scope="module")
def workspace(tmp_path_factory):
    return tmp_path_factory.mktemp("mos2_aa_prime")


@pytest.fixture(autouse=True)
def _run_inside_the_workspace(workspace, monkeypatch):
    """Every stage writes its run directory beside the inputs, not in the repo."""

    monkeypatch.chdir(workspace)


@pytest.fixture(scope="module")
def bilayer_search(workspace):
    """Run one rigid search over the AA' pair and build the six candidates."""

    bottom = _monolayer(workspace / "mos2_A.vasp", mirrored=False)
    top = _monolayer(workspace / "mos2_Aprime.vasp", mirrored=True)

    patch = pytest.MonkeyPatch()
    patch.chdir(workspace)
    try:
        found = execute_namespace(
            build_parser().parse_args(
                [
                    "moire",
                    "search",
                    str(top),
                    str(bottom),
                    "--length",
                    "26.0",
                    "--top-strain",
                    "0",
                    "--bottom-strain",
                    "0",
                    "--atoms",
                    "400",
                    "--preview-limit",
                    "0",
                ]
            )
        )
    finally:
        patch.undo()
    results_file = found.artifacts["results_json"]
    document = json.loads(Path(results_file).read_text())
    return workspace, results_file, document


def _candidate_for_angle(document, angle: float):
    for candidate in document["candidates"]:
        if abs(float(candidate["angle_deg"]) - angle) < 2e-3:
            return candidate
    raise AssertionError(f"the search missed the commensurate angle {angle} deg")


@pytest.mark.parametrize(
    "angle,indices,atoms,sulfur,per_plane,sites", REFERENCE_ROWS
)
def test_the_search_finds_the_reference_cell(
    bilayer_search, angle, indices, atoms, sulfur, per_plane, sites
):
    """The rigid search reproduces the angle, the cell index and the count."""

    _, _, document = bilayer_search
    assert _angle_error(*indices, angle) < 1e-3
    candidate = _candidate_for_angle(document, angle)
    assert int(candidate["atom_count"]) == atoms
    assert int(candidate["top_atom_count"]) == atoms // 2
    assert int(candidate["bottom_atom_count"]) == atoms // 2
    assert int(candidate["coincidence_index"]) == 1
    assert bool(candidate["loewner_certified"])
    # A rigid search strains neither layer.
    assert max(abs(float(value)) for value in candidate["strain"]) < 1e-9


@pytest.mark.parametrize(
    "angle,indices,atoms,sulfur,per_plane,sites", REFERENCE_ROWS
)
def test_the_built_bilayer_has_the_reference_defect_census(
    bilayer_search, angle, indices, atoms, sulfur, per_plane, sites
):
    """Building and analysing the cell reproduces the whole reference row."""

    workspace, results_file, document = bilayer_search
    candidate = _candidate_for_angle(document, angle)
    output = workspace / f"build_{candidate['index']}"
    built = execute_namespace(
        build_parser().parse_args(
            [
                "moire",
                "build",
                str(results_file),
                "--indexes",
                str(int(candidate["index"])),
                "--interlayer-distance",
                "3.1",
                "--output-dir",
                str(output),
            ]
        )
    )
    structure_path = built.artifacts["structures"][0]
    record = io.read_poscar(structure_path)
    assert int(record.natoms) == atoms
    counts = dict(zip(record.species, (int(value) for value in record.counts)))
    assert counts["S"] == sulfur
    assert counts["Mo"] == sulfur // 2

    analysis = Defect().analyse(structure_path, structure_kind="bulk").payload["analysis"]
    # A twist leaves the three-fold axis and the three in-plane two-fold axes
    # of the AA' stack, and destroys the inversion centre: point group 32 (D3).
    assert analysis["point_group"] == "32"
    assert analysis["operation_count"] == 6
    sulfur_planes = [
        layer
        for layer in analysis["layers"]
        if set(layer["species_counts"]) == {"S"}
    ]
    assert len(sulfur_planes) == 4
    for plane in sulfur_planes:
        assert plane["species_counts"]["S"] == per_plane
        assert plane["inequivalent_sites"]["S"] == sites
    # Every sulfur atom of the cell is covered, and the two layers are related
    # by the in-plane two-fold axes, so the cell holds twice the per-plane count
    # of inequivalent sulfur sites.
    distinct = {
        site["site_id"]
        for site in analysis["sites"]
        if site["site_kind"] == "atom" and site["species"] == "S"
    }
    assert len(distinct) == 2 * sites
    assert sum(
        int(site["multiplicity"])
        for site in analysis["sites"]
        if site["site_kind"] == "atom" and site["species"] == "S"
    ) == sulfur


def _minimum_image_distances(record) -> np.ndarray:
    """Return the matrix of nearest-image distances of a periodic structure."""

    lattice = np.asarray(record.lattice, dtype=float)
    direct = np.asarray(record.positions_direct, dtype=float) % 1.0
    difference = direct[:, None, :] - direct[None, :, :]
    shifts = np.array(
        [[i, j, 0] for i in (-1, 0, 1) for j in (-1, 0, 1)], dtype=float
    )
    candidates = np.linalg.norm(
        (difference[None, :, :, :] + shifts[:, None, None, :]) @ lattice, axis=-1
    )
    return candidates.min(axis=0)


@pytest.mark.parametrize(
    "angle,indices,atoms,sulfur,per_plane,sites", REFERENCE_ROWS
)
def test_the_built_bilayer_is_geometrically_sound(
    bilayer_search, angle, indices, atoms, sulfur, per_plane, sites
):
    """Twisting must move whole layers: bonds, cell area and gap are exact."""

    workspace, results_file, document = bilayer_search
    candidate = _candidate_for_angle(document, angle)
    output = workspace / f"geometry_{candidate['index']}"
    built = execute_namespace(
        build_parser().parse_args(
            [
                "moire",
                "build",
                str(results_file),
                "--indexes",
                str(int(candidate["index"])),
                "--interlayer-distance",
                "3.1",
                "--output-dir",
                str(output),
            ]
        )
    )
    record = io.read_poscar(built.artifacts["structures"][0])
    species = np.array(
        [name for name, count in zip(record.species, record.counts) for _ in range(count)]
    )
    distances = _minimum_image_distances(record)
    np.fill_diagonal(distances, np.inf)

    index = indices[0] ** 2 + indices[0] * indices[1] + indices[1] ** 2
    lattice = np.asarray(record.lattice, dtype=float)
    area = abs(
        float(lattice[0, 0] * lattice[1, 1] - lattice[0, 1] * lattice[1, 0])
    )
    primitive = LATTICE_CONSTANT ** 2 * math.sqrt(3.0) / 2.0
    assert area == pytest.approx(index * primitive, rel=1e-9)

    molybdenum = species == "Mo"
    sulfur_mask = species == "S"
    bond = math.hypot(LATTICE_CONSTANT / math.sqrt(3.0), SULFUR_HEIGHT)
    assert distances[np.ix_(molybdenum, sulfur_mask)].min() == pytest.approx(bond, rel=1e-9)
    assert distances[np.ix_(molybdenum, molybdenum)].min() == pytest.approx(
        LATTICE_CONSTANT, rel=1e-9
    )
    # The two layers are rigid and 3.1 A apart, so nothing is closer than that
    # across the gap and nothing at all is closer than the Mo-S bond.
    assert distances.min() == pytest.approx(bond, rel=1e-9)
    heights = np.asarray(record.positions_cartesian, dtype=float)[:, 2]
    sulfur_heights = np.sort(np.unique(np.round(heights[sulfur_mask], 6)))
    assert len(sulfur_heights) == 4
    assert sulfur_heights[2] - sulfur_heights[1] == pytest.approx(3.1, rel=1e-9)


@pytest.mark.parametrize(
    "angle,indices,atoms,sulfur,per_plane,sites", REFERENCE_ROWS[:2]
)
def test_one_vacancy_is_generated_per_inequivalent_sulfur_site(
    bilayer_search, angle, indices, atoms, sulfur, per_plane, sites
):
    """The generator writes exactly the inequivalent sulfur vacancies."""

    workspace, results_file, document = bilayer_search
    candidate = _candidate_for_angle(document, angle)
    output = workspace / f"vacancy_{candidate['index']}"
    built = execute_namespace(
        build_parser().parse_args(
            [
                "moire",
                "build",
                str(results_file),
                "--indexes",
                str(int(candidate["index"])),
                "--interlayer-distance",
                "3.1",
                "--output-dir",
                str(output / "pristine"),
            ]
        )
    )
    generated = Defect().generate(
        built.artifacts["structures"][0],
        "vacancy",
        original_species="S",
        structure_kind="bulk",
        output_dir=output / "defects",
    )
    written = generated.artifacts["structures"]
    # Four sulfur planes, related in pairs by the in-plane two-fold axes.
    assert len(written) == 2 * sites
    for path in written:
        defective = io.read_poscar(path)
        assert int(defective.natoms) == atoms - 1
        counts = dict(zip(defective.species, (int(value) for value in defective.counts)))
        assert counts["S"] == sulfur - 1
        assert counts["Mo"] == sulfur // 2


def test_the_commensurate_index_predicts_every_reference_row():
    """The arithmetic behind the table, checked against the table itself."""

    for angle, (m, n), atoms, sulfur, per_plane, sites in REFERENCE_ROWS:
        index = m * m + m * n + n * n
        assert index == per_plane
        assert atoms == 6 * index
        assert sulfur == 4 * index
        assert index % 3 == 1
        assert sites == (index + 2) // 3
        assert _angle_error(m, n, angle) < 1e-3
