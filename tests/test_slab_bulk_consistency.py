"""A slab must be a cut of the bulk, for every crystal and every Miller family.

Slab generation is where a structure tool quietly goes wrong: a plane can be
missed at a termination, an atom can be duplicated across the periodic boundary,
or the in-plane cell can be taken one repeat too large.  None of that shows up
in a picture, and all of it changes a total energy.

The check here needs no reference implementation.  Two slabs of the same crystal
and Miller family that differ by a whole number of stacking periods differ by a
piece of *bulk*: the extra atoms occupy the extra thickness at exactly the bulk
number density.  Comparing a twelve-layer and a twenty-four layer slab therefore
tests the terminations, the layer spacing and the surface cell area at once, and
it does so to machine precision rather than to a tolerance.  The in-plane cell
must also not depend on how thick the slab is.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from cellstine.interface.surface.surface import Surface
from cellstine.io import native as io_mod

from conftest import write_poscar

MILLER_FAMILIES = ["100", "110", "111", "211", "001", "210", "1x11"]


def _bulk_cells(directory):
    """Return the test crystals, as ``name -> (path, number density)`` pairs."""

    definitions = {
        "fcc_aluminium": (
            4.05 * np.eye(3),
            ["Al"],
            [4],
            np.array([[0.0, 0.0, 0.0], [0.0, 0.5, 0.5], [0.5, 0.0, 0.5], [0.5, 0.5, 0.0]]),
        ),
        "bcc_iron": (
            2.87 * np.eye(3),
            ["Fe"],
            [2],
            np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]]),
        ),
        "diamond_silicon": (
            5.43 * np.array([[0.0, 0.5, 0.5], [0.5, 0.0, 0.5], [0.5, 0.5, 0.0]]),
            ["Si"],
            [2],
            np.array([[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]]),
        ),
        "rocksalt_sodium_chloride": (
            5.64 * np.eye(3),
            ["Na", "Cl"],
            [4, 4],
            np.array(
                [
                    [0.0, 0.0, 0.0],
                    [0.0, 0.5, 0.5],
                    [0.5, 0.0, 0.5],
                    [0.5, 0.5, 0.0],
                    [0.5, 0.5, 0.5],
                    [0.5, 0.0, 0.0],
                    [0.0, 0.5, 0.0],
                    [0.0, 0.0, 0.5],
                ]
            ),
        ),
        "hcp_magnesium": (
            np.array(
                [
                    [3.21, 0.0, 0.0],
                    [-1.605, 3.21 * math.sqrt(3.0) / 2.0, 0.0],
                    [0.0, 0.0, 5.21],
                ]
            ),
            ["Mg"],
            [2],
            np.array([[0.0, 0.0, 0.0], [1.0 / 3.0, 2.0 / 3.0, 0.5]]),
        ),
    }
    cells = {}
    for name, (lattice, species, counts, positions) in definitions.items():
        path = write_poscar(directory / f"{name}.vasp", lattice, species, counts, positions)
        density = sum(counts) / abs(float(np.linalg.det(np.asarray(lattice, dtype=float))))
        cells[name] = (path, density)
    return cells


@pytest.fixture(scope="module")
def crystals(tmp_path_factory):
    return _bulk_cells(tmp_path_factory.mktemp("slab-consistency"))


@pytest.fixture(scope="module")
def surface_workflow(tmp_path_factory):
    workspace = tmp_path_factory.mktemp("slab-consistency-runs")
    return Surface(runs_root=str(workspace / "runs"), output_root=str(workspace / "output"))


def _slab_measurements(workflow, bulk_path, miller: str, layers: int):
    """Return the atom count, in-plane area and occupied thickness of a slab."""

    result = workflow.surface(
        bulk_poscar=str(bulk_path), miller=miller, layers=layers, vacuum=12.0
    )
    record = io_mod.read_poscar(str(result.artifacts["slab_poscar"]))
    lattice = np.asarray(record.lattice, dtype=float)
    area = float(np.linalg.norm(np.cross(lattice[0], lattice[1])))
    heights = np.asarray(record.positions_cartesian, dtype=float)[:, 2]
    return int(sum(record.counts)), area, float(heights.max() - heights.min())


@pytest.mark.parametrize("miller", MILLER_FAMILIES)
@pytest.mark.parametrize(
    "crystal",
    ["fcc_aluminium", "bcc_iron", "diamond_silicon", "rocksalt_sodium_chloride", "hcp_magnesium"],
)
def test_thickening_a_slab_adds_exactly_bulk(surface_workflow, crystals, crystal, miller):
    """The extra twelve layers weigh what that much bulk weighs."""

    bulk_path, bulk_density = crystals[crystal]
    thin_atoms, thin_area, thin_thickness = _slab_measurements(
        surface_workflow, bulk_path, miller, 12
    )
    thick_atoms, thick_area, thick_thickness = _slab_measurements(
        surface_workflow, bulk_path, miller, 24
    )

    assert thick_area == pytest.approx(thin_area, rel=1e-12)
    assert thick_atoms > thin_atoms
    assert thick_thickness > thin_thickness

    added_density = (thick_atoms - thin_atoms) / (thin_area * (thick_thickness - thin_thickness))
    assert added_density == pytest.approx(bulk_density, rel=1e-9)


@pytest.mark.parametrize("miller", MILLER_FAMILIES)
def test_a_slab_never_puts_two_atoms_in_one_place(surface_workflow, crystals, miller):
    """No termination may duplicate a plane across the periodic boundary."""

    bulk_path, _ = crystals["rocksalt_sodium_chloride"]
    result = surface_workflow.surface(
        bulk_poscar=str(bulk_path), miller=miller, layers=6, vacuum=12.0
    )
    record = io_mod.read_poscar(str(result.artifacts["slab_poscar"]))
    lattice = np.asarray(record.lattice, dtype=float)
    fractional = np.asarray(record.positions_direct, dtype=float)
    shifts = np.array(
        [[first, second, 0.0] for first in (-1, 0, 1) for second in (-1, 0, 1)], dtype=float
    )
    closest = math.inf
    for index, site in enumerate(fractional):
        deltas = fractional - site
        for shift in shifts:
            distances = np.linalg.norm((deltas + shift) @ lattice, axis=1)
            if not shift.any():
                distances[index] = math.inf
            closest = min(closest, float(distances.min()))
    assert closest > 1.5
