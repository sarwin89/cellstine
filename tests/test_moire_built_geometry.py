"""What the search reports is what the built structure actually is.

`results.json` reports a twist angle and a per-layer strain for every candidate.
Those numbers come out of the integer search; nothing else in the pipeline
checks them against the structure the builder finally writes.  This module
closes that loop *from the atomic coordinates alone*:

1. the built bilayer is split into its two layers by height;
2. each layer's own in-plane translation lattice is recovered by testing which
   interatomic displacements map the layer onto itself under the periodic
   boundary of the moiré cell -- no information from the search is used;
3. that lattice is compared with the pristine monolayer through the best
   integer correspondence, which gives the rotation and the two principal
   logarithmic strains the layer actually carries;
4. the measured strains, the measured twist and the measured per-layer atom
   counts are compared with the reported ones.

The measured twist is the rotation of the top layer relative to the bottom one.
It is defined modulo the 60 degree rotation of a hexagonal layer and up to the
overall sense of rotation, so it is compared as a magnitude folded into
`(-30, 30]`.
"""

from __future__ import annotations

import numpy as np
import pytest

from cellstine.core.species import expand_species
from cellstine.io import native as io
from cellstine.moire.builder import generator
from cellstine.moire.moire import Moire

from conftest import hexagonal_basis, layer_translation_lattice, rotation_and_log_strain

GRAPHENE = hexagonal_basis(2.46).T  # rows are the two lattice vectors
HBN = hexagonal_basis(2.504).T


def _fold(angle: float) -> float:
    """Fold an angle into the fundamental range of a hexagonal layer."""

    return (angle + 30.0) % 60.0 - 30.0


def _measure(structure) -> dict[str, object]:
    cartesian = structure.positions_direct @ structure.lattice
    lower = cartesian[:, 2] < cartesian[:, 2].mean()
    labels = np.asarray(expand_species(structure.species, structure.counts))
    measurement: dict[str, object] = {}
    for name, mask in (("bottom", lower), ("top", ~lower)):
        pristine = GRAPHENE if set(labels[mask]) == {"C"} else HBN
        angle, strain = rotation_and_log_strain(
            layer_translation_lattice(
                structure.lattice, structure.positions_direct[mask], labels[mask]
            ),
            pristine,
        )
        measurement[f"{name}_angle"] = angle
        measurement[f"{name}_strain"] = strain
        measurement[f"{name}_atoms"] = int(np.count_nonzero(mask))
    measurement["twist"] = _fold(
        float(measurement["top_angle"]) - float(measurement["bottom_angle"])
    )
    return measurement


def _run(workspace, top_poscar, bottom_poscar, max_length):
    workflow = Moire(
        runs_root=str(workspace / "runs"), output_root=str(workspace / "output")
    )
    found = workflow.find(
        top_poscar=str(top_poscar),
        bottom_poscar=str(bottom_poscar),
        max_length=max_length,
        top_strain=0.01,
        bottom_strain=0.01,
        preview_limit=0,
    )
    results = found.artifacts["results_json"]
    _, _, candidates, _ = generator.parse_results(results)
    measured = []
    for record in candidates:
        built = workflow.make(
            results_file=results, indexes=[int(record["index"])], interlayer_distance=3.35
        )
        structure = io.read_poscar(built.artifacts["structures"][0])
        measured.append((record, _measure(structure)))
    return measured


@pytest.fixture(scope="module")
def measured_bilayers(tmp_path_factory, graphene_poscar, hbn_poscar):
    homo = _run(
        tmp_path_factory.mktemp("moire-homo"), graphene_poscar, graphene_poscar, 14.0
    )
    hetero = _run(tmp_path_factory.mktemp("moire-hetero"), graphene_poscar, hbn_poscar, 12.0)
    assert len(homo) >= 4 and len(hetero) >= 2
    return homo + hetero


def test_the_built_layers_carry_the_reported_strain(measured_bilayers):
    for record, measurement in measured_bilayers:
        for name in ("top", "bottom"):
            reported = np.sort(np.asarray(record[f"{name}_layer_strain"], dtype=float))
            assert measurement[f"{name}_strain"] == pytest.approx(reported, abs=1e-7), (
                f"candidate {record['index']}: {name} layer strain"
            )


def test_the_built_layers_carry_the_reported_twist(measured_bilayers):
    for record, measurement in measured_bilayers:
        reported = _fold(float(record["angle_deg"]))
        measured = float(measurement["twist"])
        # The sense of the rotation is a convention; the magnitude is not.
        assert abs(measured) == pytest.approx(abs(reported), abs=1e-7), (
            f"candidate {record['index']}: twist angle"
        )


def test_the_built_layers_have_the_reported_atom_counts(measured_bilayers):
    for record, measurement in measured_bilayers:
        counts = sorted((int(record["top_atom_count"]), int(record["bottom_atom_count"])))
        assert sorted((measurement["top_atoms"], measurement["bottom_atoms"])) == counts
        assert measurement["top_atoms"] + measurement["bottom_atoms"] == int(
            record["atom_count"]
        )


def test_the_measurement_would_notice_a_wrong_answer(measured_bilayers):
    """A twisted candidate really is twisted, and a strained one really strained."""

    twists = [abs(float(measurement["twist"])) for _record, measurement in measured_bilayers]
    assert max(twists) > 1.0
    strains = [
        float(np.max(np.abs(measurement["top_strain"])))
        for _record, measurement in measured_bilayers
    ]
    assert max(strains) > 1e-4
