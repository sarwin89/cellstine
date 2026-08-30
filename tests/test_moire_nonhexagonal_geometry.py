"""The moire loop closes for layers that are not hexagonal either.

``test_moire_built_geometry`` re-measures a built bilayer of graphene and hBN
against what ``results.json`` reported.  Hexagonal layers are the easy case: the
two lattices are similar, the strains are tiny and the twist is small.  This
module runs the same measurement -- the layer's own translation lattice,
recovered from the coordinates of the written POSCAR and compared with the
pristine monolayer through the least-strain integer correspondence -- for a
square, a rectangular, an oblique and a mismatched square-on-rectangle pair,
where the cells are far from similar and the moire cell is not hexagonal.

A twist is only defined up to the point group of the two layers.  A square layer
brings its own four-fold axis and its mirrors, so the twist ``theta`` cannot be
told from ``-theta`` or from ``90 - theta``; a rectangular or oblique layer only
has the inversion, so it folds by ``180`` degrees.  The comparison below folds
by whichever period the pair allows, and never by more.
"""

from __future__ import annotations

import numpy as np
import pytest

from cellstine.core.species import expand_species
from cellstine.io import native as io
from cellstine.moire.builder import generator
from cellstine.moire.moire import Moire

from conftest import layer_translation_lattice, rotation_and_log_strain, write_poscar

SQUARE = np.array([[3.0, 0.0], [0.0, 3.0]])
SQUARE_WIDE = np.array([[3.15, 0.0], [0.0, 3.15]])
RECTANGLE = np.array([[3.0, 0.0], [0.0, 4.2]])
RECTANGLE_WIDE = np.array([[3.1, 0.0], [0.0, 4.3]])
OBLIQUE = np.array([[3.0, 0.0], [0.7, 3.6]])
TALL_RECTANGLE = np.array([[3.0, 0.0], [0.0, 4.5]])

# name -> (top rows, bottom rows, twist period in degrees)
PAIRS = {
    "square_on_square": (SQUARE, SQUARE_WIDE, 90.0),
    "rectangle_on_rectangle": (RECTANGLE, RECTANGLE_WIDE, 180.0),
    "oblique_on_oblique": (OBLIQUE, OBLIQUE, 180.0),
    "square_on_rectangle": (SQUARE, TALL_RECTANGLE, 90.0),
}

CANDIDATES_PER_PAIR = 4


def _write_layer(path, rows, symbol):
    lattice = np.zeros((3, 3))
    lattice[:2, :2] = np.asarray(rows, dtype=float)
    lattice[2, 2] = 20.0
    return write_poscar(
        path, lattice, [symbol], [1], np.array([[0.0, 0.0, 0.5]]), comment="monolayer"
    )


def _fold(angle: float, period: float) -> float:
    return (angle + period / 2.0) % period - period / 2.0


def _measure(structure, top_rows, bottom_rows):
    cartesian = np.asarray(structure.positions_direct, dtype=float) @ np.asarray(
        structure.lattice, dtype=float
    )
    labels = np.asarray(expand_species(structure.species, structure.counts))
    lower = cartesian[:, 2] < cartesian[:, 2].mean()
    measurement: dict[str, object] = {}
    for name, mask, pristine in (
        ("bottom", lower, bottom_rows),
        ("top", ~lower, top_rows),
    ):
        angle, strain = rotation_and_log_strain(
            layer_translation_lattice(
                np.asarray(structure.lattice, dtype=float),
                np.asarray(structure.positions_direct, dtype=float)[mask],
                labels[mask],
            ),
            np.asarray(pristine, dtype=float),
            bound=3,
        )
        measurement[f"{name}_angle"] = angle
        measurement[f"{name}_strain"] = strain
        measurement[f"{name}_atoms"] = int(np.count_nonzero(mask))
    return measurement


def _built(workspace, name):
    top_rows, bottom_rows, _period = PAIRS[name]
    top = _write_layer(workspace / "top.vasp", top_rows, "C")
    bottom = _write_layer(workspace / "bottom.vasp", bottom_rows, "N")
    workflow = Moire(runs_root=str(workspace / "runs"), output_root=str(workspace / "output"))
    found = workflow.find(
        top_poscar=str(top),
        bottom_poscar=str(bottom),
        max_length=18.0,
        top_strain=0.02,
        bottom_strain=0.02,
        preview_limit=0,
    )
    results = found.artifacts["results_json"]
    _, _, candidates, _ = generator.parse_results(results)
    assert len(candidates) >= CANDIDATES_PER_PAIR
    measured = []
    for record in candidates[:CANDIDATES_PER_PAIR]:
        build = workflow.make(
            results_file=results, indexes=[int(record["index"])], interlayer_distance=3.3
        )
        structure = io.read_poscar(build.artifacts["structures"][0])
        measured.append((record, _measure(structure, top_rows, bottom_rows)))
    return measured


@pytest.fixture(scope="module")
def measured(tmp_path_factory):
    return {
        name: _built(tmp_path_factory.mktemp(f"moire-{name}"), name) for name in PAIRS
    }


@pytest.mark.parametrize("name", sorted(PAIRS))
def test_the_built_layers_carry_the_reported_strain(measured, name):
    for record, measurement in measured[name]:
        for side in ("top", "bottom"):
            reported = np.sort(np.asarray(record[f"{side}_layer_strain"], dtype=float))
            assert measurement[f"{side}_strain"] == pytest.approx(reported, abs=1e-7), (
                f"{name} candidate {record['index']}: {side} layer strain"
            )


@pytest.mark.parametrize("name", sorted(PAIRS))
def test_the_built_layers_carry_the_reported_twist(measured, name):
    period = PAIRS[name][2]
    for record, measurement in measured[name]:
        reported = _fold(float(record["angle_deg"]), period)
        twist = _fold(
            float(measurement["top_angle"]) - float(measurement["bottom_angle"]), period
        )
        # The sense of the rotation is a convention, and a mirror of the layer
        # sends ``theta`` to ``-theta``; the magnitude is not a convention.
        assert abs(twist) == pytest.approx(abs(reported), abs=1e-6), (
            f"{name} candidate {record['index']}: twist angle"
        )


@pytest.mark.parametrize("name", sorted(PAIRS))
def test_the_built_layers_have_the_reported_atom_counts(measured, name):
    for record, measurement in measured[name]:
        assert measurement["top_atoms"] == int(record["top_atom_count"])
        assert measurement["bottom_atoms"] == int(record["bottom_atom_count"])
        assert (
            measurement["top_atoms"] + measurement["bottom_atoms"]
            == int(record["atom_count"])
        )


def test_the_measurement_would_notice_a_wrong_answer(measured):
    """A twisted candidate really is twisted, and a strained one really strained."""

    twists = []
    strains = []
    for name, entries in measured.items():
        period = PAIRS[name][2]
        for _record, measurement in entries:
            twists.append(
                abs(
                    _fold(
                        float(measurement["top_angle"]) - float(measurement["bottom_angle"]),
                        period,
                    )
                )
            )
            strains.append(float(np.max(np.abs(measurement["top_strain"]))))
    assert max(twists) > 1.0
    assert max(strains) > 1e-4
