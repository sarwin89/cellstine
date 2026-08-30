"""A vacuum-free ``(hkl)`` cell has to be the bulk crystal, not a squashed slab.

``interface surface --vacuum 0`` asks for the oriented *bulk* cell of a face:
the same crystal, cut so that ``c`` is the plane normal.  The statements below
are the ones that make it a crystal rather than a stack of layers with the ends
glued together:

* the cell tiles space with the bulk atomic density, so no layer is duplicated
  across the boundary and no gap is left behind;
* the layer count is a whole number of stacking periods, and a request that is
  not one is refused with the period it should be a multiple of;
* every atom of the cell sits on a bulk site of the input crystal, and no two
  atoms of the periodic cell overlap;
* asking for a vacuum is unaffected: the cell is then the slab plus exactly the
  requested empty gap.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from cellstine.interface.surface import backend as surface
from cellstine.interface.surface.surface_supercell import build_surface_structure
from cellstine.io import native as io_mod

from conftest import write_poscar

CRYSTALS: dict[str, tuple[np.ndarray, list[str], list[int], np.ndarray]] = {
    "simple_cubic": (
        3.0 * np.eye(3),
        ["Po"],
        [1],
        np.array([[0.0, 0.0, 0.0]]),
    ),
    "face_centred": (
        4.05 * np.eye(3),
        ["Al"],
        [4],
        np.array([[0.0, 0.0, 0.0], [0.0, 0.5, 0.5], [0.5, 0.0, 0.5], [0.5, 0.5, 0.0]]),
    ),
    "body_centred": (
        2.87 * np.eye(3),
        ["Fe"],
        [2],
        np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]]),
    ),
    "rock_salt": (
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
    "diamond": (
        5.431 * np.eye(3),
        ["Si"],
        [8],
        np.array(
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.5, 0.5],
                [0.5, 0.0, 0.5],
                [0.5, 0.5, 0.0],
                [0.25, 0.25, 0.25],
                [0.25, 0.75, 0.75],
                [0.75, 0.25, 0.75],
                [0.75, 0.75, 0.25],
            ]
        ),
    ),
}

MILLERS = [(1, 0, 0), (1, 1, 0), (1, 1, 1), (2, 1, 1)]

MAX_PROBE_LAYERS = 16


@pytest.fixture(scope="module")
def crystals(tmp_path_factory) -> dict[str, str]:
    workspace = tmp_path_factory.mktemp("vacuum_free")
    paths: dict[str, str] = {}
    for name, (lattice, species, counts, positions) in CRYSTALS.items():
        paths[name] = str(
            write_poscar(workspace / f"{name}.vasp", lattice, species, counts, positions, comment=name)
        )
    return paths


def _bulk_density(name: str) -> float:
    lattice, _, counts, _ = CRYSTALS[name]
    return float(sum(counts)) / abs(float(np.linalg.det(np.asarray(lattice, dtype=float))))


def _stacking_period(path: str, miller: tuple[int, int, int]) -> int | None:
    """Return the smallest vacuum-free layer count the builder accepts."""

    for layers in range(1, MAX_PROBE_LAYERS + 1):
        try:
            build_surface_structure(path, miller=miller, layers=layers, vacuum=0.0)
        except ValueError:
            continue
        return layers
    return None


def _expanded_species(structure) -> list[str]:
    labels: list[str] = []
    for symbol, count in zip(structure.species, structure.counts):
        labels.extend([str(symbol)] * int(count))
    return labels


def _minimum_image_separation(structure) -> float:
    lattice = np.asarray(structure.lattice, dtype=float)
    direct = np.asarray(structure.positions_direct, dtype=float)
    shifts = np.array([[i, j, k] for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1)], dtype=float)
    smallest = math.inf
    for first in range(len(direct)):
        for second in range(len(direct)):
            delta = direct[second] - direct[first]
            delta -= np.round(delta)
            images = (delta + shifts) @ lattice
            distances = np.linalg.norm(images, axis=1)
            if first == second:
                distances = distances[distances > 1e-8]
            if distances.size:
                smallest = min(smallest, float(np.min(distances)))
    return smallest


def _sits_on_bulk_site(name: str, structure) -> bool:
    """Return whether every atom of the cell is a site of the input crystal."""

    bulk_lattice, bulk_species, bulk_counts, bulk_direct = CRYSTALS[name]
    bulk_labels: list[str] = []
    for symbol, count in zip(bulk_species, bulk_counts):
        bulk_labels.extend([str(symbol)] * int(count))
    bulk_cartesian = np.asarray(bulk_direct, dtype=float) @ np.asarray(bulk_lattice, dtype=float)
    inverse = np.linalg.inv(np.asarray(bulk_lattice, dtype=float))

    lattice = np.asarray(structure.lattice, dtype=float)
    cartesian = np.asarray(structure.positions_direct, dtype=float) @ lattice
    # The builder emits the cell in its own Cartesian frame -- a along +x and the
    # plane normal along +z -- so each atom is first rotated back into the bulk
    # frame and then tested against the bulk sites modulo the bulk lattice.
    labels = _expanded_species(structure)
    for rotation in _frame_rotations(name, structure):
        rotated = cartesian @ rotation
        placed = True
        for point, symbol in zip(rotated, labels):
            deltas = (point - bulk_cartesian) @ inverse
            residual = deltas - np.round(deltas)
            matches = np.linalg.norm(residual @ np.asarray(bulk_lattice, dtype=float), axis=1) < 1e-4
            if not any(matches[index] and bulk_labels[index] == symbol for index in range(len(bulk_labels))):
                placed = False
                break
        if placed:
            return True
    return False


def _frame_rotations(name: str, structure) -> list[np.ndarray]:
    """Return the rotations that can take the surface frame back to the bulk one.

    The two in-plane vectors of the cell are bulk translations expressed in the
    surface frame, so a rotation is fixed by mapping them, and their normal, onto
    a matching pair of bulk vectors.  Every pair of short bulk translations with
    the same lengths and angle gives one candidate.
    """

    bulk_lattice = np.asarray(CRYSTALS[name][0], dtype=float)
    lattice = np.asarray(structure.lattice, dtype=float)
    shifts = np.array(
        [[i, j, k] for i in range(-4, 5) for j in range(-4, 5) for k in range(-4, 5)],
        dtype=float,
    )
    # Half-integer combinations are needed because the centred crystals above
    # carry translations that are not integer combinations of their cells.
    candidates = np.unique(np.round(np.vstack([shifts, shifts / 2.0]), 6), axis=0) @ bulk_lattice
    lengths = np.linalg.norm(candidates, axis=1)
    rotations: list[np.ndarray] = []
    for first in np.nonzero(np.abs(lengths - np.linalg.norm(lattice[0])) < 1e-6)[0]:
        for second in np.nonzero(np.abs(lengths - np.linalg.norm(lattice[1])) < 1e-6)[0]:
            if abs(float(candidates[first] @ candidates[second]) - float(lattice[0] @ lattice[1])) > 1e-5:
                continue
            target = np.vstack(
                [candidates[first], candidates[second], np.cross(candidates[first], candidates[second])]
            )
            source = np.vstack([lattice[0], lattice[1], np.cross(lattice[0], lattice[1])])
            rotation = np.linalg.solve(source, target)
            if np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-6) and float(np.linalg.det(rotation)) > 0.0:
                rotations.append(rotation)
    return rotations


@pytest.mark.parametrize("name", sorted(CRYSTALS))
@pytest.mark.parametrize("miller", MILLERS)
def test_vacuum_free_cell_has_the_bulk_density(crystals, name, miller):
    period = _stacking_period(crystals[name], miller)
    assert period is not None, "no vacuum-free cell was accepted within the probe range"
    for multiple in (1, 2):
        structure = build_surface_structure(
            crystals[name], miller=miller, layers=period * multiple, vacuum=0.0
        ).structure
        volume = abs(float(np.linalg.det(np.asarray(structure.lattice, dtype=float))))
        assert structure.natoms / volume == pytest.approx(_bulk_density(name), rel=1e-9)


@pytest.mark.parametrize("name", sorted(CRYSTALS))
@pytest.mark.parametrize("miller", MILLERS)
def test_vacuum_free_cell_is_the_bulk_crystal(crystals, name, miller):
    period = _stacking_period(crystals[name], miller)
    assert period is not None
    structure = build_surface_structure(crystals[name], miller=miller, layers=period, vacuum=0.0).structure
    assert _sits_on_bulk_site(name, structure)
    # No layer is repeated across the periodic boundary: the closest pair of the
    # periodic cell is a genuine bond length, not a coincidence of two copies.
    assert _minimum_image_separation(structure) > 0.5


@pytest.mark.parametrize("name", sorted(CRYSTALS))
@pytest.mark.parametrize("miller", MILLERS)
def test_layer_count_scales_with_the_period(crystals, name, miller):
    period = _stacking_period(crystals[name], miller)
    assert period is not None
    single = build_surface_structure(crystals[name], miller=miller, layers=period, vacuum=0.0).structure
    double = build_surface_structure(crystals[name], miller=miller, layers=2 * period, vacuum=0.0).structure
    assert double.natoms == 2 * single.natoms
    assert float(double.lattice[2][2]) == pytest.approx(2.0 * float(single.lattice[2][2]))


@pytest.mark.parametrize("name", sorted(CRYSTALS))
@pytest.mark.parametrize("miller", MILLERS)
def test_a_partial_period_is_refused_with_its_period(crystals, name, miller):
    period = _stacking_period(crystals[name], miller)
    assert period is not None
    if period == 1:
        pytest.skip("every layer count is a whole number of periods for this face")
    with pytest.raises(ValueError) as error:
        build_surface_structure(crystals[name], miller=miller, layers=period + 1, vacuum=0.0)
    message = str(error.value)
    assert f"repeats every {period} atomic layers" in message
    assert "vacuum" in message


@pytest.mark.parametrize("name", sorted(CRYSTALS))
@pytest.mark.parametrize("miller", MILLERS)
def test_a_requested_vacuum_is_the_gap_that_is_left(crystals, name, miller):
    layers = 4
    vacuum = 12.0
    structure = build_surface_structure(crystals[name], miller=miller, layers=layers, vacuum=vacuum).structure
    heights = np.asarray(structure.positions_direct, dtype=float)[:, 2] * float(structure.lattice[2][2])
    thickness = float(np.max(heights) - np.min(heights))
    assert float(structure.lattice[2][2]) - thickness == pytest.approx(vacuum, abs=1e-9)


def test_the_stacking_report_still_probes_any_layer_count(crystals):
    # The stacking probe asks for a fixed number of layers, which need not be a
    # whole number of periods, so it must not be held to the bulk-cell rule.
    for probe_layers in (4, 5, 7, 8):
        analysis = surface.analyse_primitive_surface(
            crystals["face_centred"], miller=(1, 1, 1), probe_layers=probe_layers
        )
        assert len(analysis.stacking_sequence) == probe_layers
        assert analysis.stacking_period == "ABC"


def test_a_written_vacuum_free_cell_reads_back_as_the_same_crystal(tmp_path, crystals):
    structure = build_surface_structure(crystals["face_centred"], miller=(1, 1, 1), layers=3, vacuum=0.0).structure
    path = tmp_path / "al111_bulk.vasp"
    surface.write_surface_poscar(str(path), structure, (1, 1, 1))
    reloaded = io_mod.read_poscar(str(path))
    assert reloaded.natoms == structure.natoms
    volume = abs(float(np.linalg.det(np.asarray(reloaded.lattice, dtype=float))))
    assert reloaded.natoms / volume == pytest.approx(_bulk_density("face_centred"), rel=1e-9)
