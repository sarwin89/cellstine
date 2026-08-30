"""Geometry of molecules placed on and moved over a substrate.

Placement is only trustworthy if the molecule arrives rigid, at the requested
height above the chosen site, and centred over that site in the surface plane.
Each of those is a closed-form statement about the output structure, so it is
checked directly rather than through a picture.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from cellstine.adsorbate.adsorbate import Adsorbate
from cellstine.adsorbate.placement import operations as placement
from cellstine.core.species import expand_species
from cellstine.core.transforms import yaw_pitch_roll_matrix
from cellstine.interface.surface import backend as surface
from cellstine.io import native as io_mod

from conftest import write_poscar


@pytest.fixture(scope="module")
def aluminium_slab(tmp_path_factory) -> str:
    """A four-layer Al(111) slab, repeated so a small molecule fits on it."""

    workspace = tmp_path_factory.mktemp("adsorbate")
    bulk = write_poscar(
        workspace / "al.vasp",
        4.05 * np.eye(3),
        ["Al"],
        [4],
        np.array([[0.0, 0.0, 0.0], [0.0, 0.5, 0.5], [0.5, 0.0, 0.5], [0.5, 0.5, 0.0]]),
    )
    structure = surface.build_surface_structure(
        str(bulk), miller=(1, 1, 1), layers=4, vacuum=16.0, repeat_a=2, repeat_b=2
    ).structure
    slab_path = workspace / "al111.vasp"
    io_mod.write_poscar(
        str(slab_path),
        structure.lattice,
        structure.positions_direct,
        structure.counts,
        structure.species,
        positions_are_cartesian=False,
    )
    return str(slab_path)


@pytest.fixture(scope="module")
def carbon_monoxide(tmp_path_factory) -> str:
    path = tmp_path_factory.mktemp("adsorbate-molecule") / "co.vasp"
    positions = np.array([[0.5, 0.5, 0.5], [0.5, 0.5, 0.5 + 1.128 / 12.0]])
    return str(write_poscar(path, np.diag([12.0, 12.0, 12.0]), ["C", "O"], [1, 1], positions))


def _internal_distances(positions: np.ndarray) -> np.ndarray:
    difference = positions[:, None, :] - positions[None, :, :]
    return np.sort(np.linalg.norm(difference, axis=2).ravel())


def test_yaw_pitch_roll_is_a_proper_rotation():
    for angles in ((0.0, 0.0, 0.0), (30.0, 0.0, 0.0), (12.0, -47.0, 130.0), (359.0, 89.0, -12.5)):
        rotation = yaw_pitch_roll_matrix(*angles)
        assert np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-12)
        assert float(np.linalg.det(rotation)) == pytest.approx(1.0, abs=1e-12)


def test_yaw_composes_about_the_surface_normal():
    """A pure yaw must rotate the xy plane by the requested angle."""

    angle = 37.0
    rotation = yaw_pitch_roll_matrix(angle, 0.0, 0.0)
    vector = np.array([1.0, 0.0, 0.0])
    rotated = rotation @ vector
    assert rotated[2] == pytest.approx(0.0, abs=1e-12)
    assert math.degrees(math.atan2(rotated[1], rotated[0])) == pytest.approx(angle, abs=1e-9)


@pytest.mark.parametrize("site_type", ["top", "bridge", "fcc", "hcp"])
def test_molecule_sits_at_the_requested_height_over_its_site(
    tmp_path, aluminium_slab, carbon_monoxide, site_type
):
    height = 2.1
    run = placement.place_molecule_on_site(
        aluminium_slab,
        carbon_monoxide,
        site_type=site_type,
        height=height,
        output_path=str(tmp_path / f"{site_type}.vasp"),
    )
    structure = io_mod.read_poscar(str(run.output_path))
    assert structure.natoms == run.substrate_atom_count + run.molecule_atom_count

    labels = np.array(expand_species(structure.species, structure.counts))
    cartesian = np.asarray(structure.positions_cartesian, dtype=float)
    molecule = cartesian[np.isin(labels, ["C", "O"])]
    substrate = cartesian[labels == "Al"]
    assert len(molecule) == 2

    site_height = float(run.site_cartesian[2])
    assert float(molecule[:, 2].min()) - site_height == pytest.approx(height, abs=1e-8)
    assert float(molecule[:, 2].min()) > float(substrate[:, 2].max())


def test_placement_keeps_the_molecule_rigid_and_over_the_site(
    tmp_path, aluminium_slab, carbon_monoxide
):
    original = io_mod.read_poscar(carbon_monoxide)
    original_species = expand_species(original.species, original.counts)
    original_distances = _internal_distances(np.asarray(original.positions_cartesian, dtype=float))

    run = placement.place_molecule_on_site(
        aluminium_slab,
        carbon_monoxide,
        site_type="fcc",
        height=2.0,
        rotation_deg=25.0,
        tilt_deg=15.0,
        roll_deg=-40.0,
        reframe_axes=None,
        output_path=str(tmp_path / "tilted.vasp"),
    )
    structure = io_mod.read_poscar(str(run.output_path))
    labels = np.array(expand_species(structure.species, structure.counts))
    molecule = np.asarray(structure.positions_cartesian, dtype=float)[np.isin(labels, ["C", "O"])]
    assert np.allclose(_internal_distances(molecule), original_distances, atol=1e-9)

    lattice = np.asarray(structure.lattice, dtype=float)
    centre = placement.center_of_mass_cartesian(molecule, original_species)
    offset = (centre - np.asarray(run.site_cartesian, dtype=float))[:2]
    fractional = np.linalg.solve(lattice[:2, :2].T, offset)
    assert np.allclose(fractional - np.round(fractional), 0.0, atol=1e-8)


def test_placement_leaves_the_substrate_untouched(tmp_path, aluminium_slab, carbon_monoxide):
    """The substrate atoms keep their Cartesian positions and the cell its shape.

    Only the length of ``c`` may change, and only because the molecule would
    otherwise eat into the vacuum the slab arrived with.
    """

    slab = io_mod.read_poscar(aluminium_slab)
    run = placement.place_molecule_on_site(
        aluminium_slab,
        carbon_monoxide,
        site_type="top",
        height=2.0,
        reframe_axes=None,
        output_path=str(tmp_path / "untouched.vasp"),
    )
    structure = io_mod.read_poscar(str(run.output_path))
    labels = np.array(expand_species(structure.species, structure.counts))
    aluminium = np.asarray(structure.positions_cartesian, dtype=float)[labels == "Al"]
    expected = np.asarray(slab.positions_cartesian, dtype=float)
    difference = np.sort(aluminium, axis=0) - np.sort(expected, axis=0)
    assert np.allclose(difference, 0.0, atol=1e-9)
    assert np.allclose(structure.lattice[:2], slab.lattice[:2], atol=1e-9)
    direction = np.asarray(slab.lattice, dtype=float)[2]
    grown = np.asarray(structure.lattice, dtype=float)[2]
    scale = float(np.linalg.norm(grown) / np.linalg.norm(direction))
    assert scale >= 1.0 - 1e-12
    assert np.allclose(grown, scale * direction, atol=1e-9)


def test_negative_height_is_rejected(tmp_path, aluminium_slab, carbon_monoxide):
    with pytest.raises(ValueError):
        placement.place_molecule_on_site(
            aluminium_slab,
            carbon_monoxide,
            site_type="top",
            height=-1.0,
            output_path=str(tmp_path / "bad.vasp"),
        )


def test_moving_the_top_molecule_is_a_rigid_motion(tmp_path, aluminium_slab, carbon_monoxide):
    placed = placement.place_molecule_on_site(
        aluminium_slab,
        carbon_monoxide,
        site_type="top",
        height=2.0,
        output_path=str(tmp_path / "placed.vasp"),
    )
    before = io_mod.read_poscar(str(placed.output_path))
    labels = np.array(expand_species(before.species, before.counts))
    molecule_before = np.asarray(before.positions_cartesian, dtype=float)[np.isin(labels, ["C", "O"])]

    moved = placement.transform_top_molecule(
        str(placed.output_path),
        target_cartesian=None,
        target_direct=(0.25, 0.75),
        rotation_deg=90.0,
        output_path=str(tmp_path / "moved.vasp"),
    )
    after = io_mod.read_poscar(str(moved.output_path))
    labels_after = np.array(expand_species(after.species, after.counts))
    molecule_after = np.asarray(after.positions_cartesian, dtype=float)[np.isin(labels_after, ["C", "O"])]

    assert np.allclose(_internal_distances(molecule_before), _internal_distances(molecule_after), atol=1e-9)
    lattice = np.asarray(after.lattice, dtype=float)
    centre = placement.center_of_mass_cartesian(
        molecule_after, expand_species(["C", "O"], [1, 1])
    )
    fractional = np.linalg.solve(lattice.T, centre)
    assert fractional[0] % 1.0 == pytest.approx(0.25, abs=1e-8)
    assert fractional[1] % 1.0 == pytest.approx(0.75, abs=1e-8)


def test_placement_writes_in_plane_coordinates_inside_the_cell(
    tmp_path, aluminium_slab, carbon_monoxide
):
    """Recentring the molecule must not push substrate atoms outside [0, 1).

    Reframing shifts the whole structure so the molecule sits at the middle of
    the surface cell.  The shift is a lattice-periodic operation in the plane,
    so the in-plane fractional coordinates are wrapped back into the cell and
    only the aperiodic z coordinate is left alone.
    """

    slab = io_mod.read_poscar(aluminium_slab)
    run = placement.place_molecule_on_site(
        aluminium_slab,
        carbon_monoxide,
        site_type="top",
        height=2.0,
        reframe_axes="xy",
        output_path=str(tmp_path / "wrapped.vasp"),
    )
    structure = io_mod.read_poscar(str(run.output_path))
    direct = np.asarray(structure.positions_direct, dtype=float)

    assert np.all(direct[:, :2] >= 0.0)
    assert np.all(direct[:, :2] < 1.0)

    labels = np.array(expand_species(structure.species, structure.counts))
    molecule = direct[np.isin(labels, ["C", "O"])]
    # The molecule itself is centred, hence never split across the boundary.
    assert np.allclose(molecule[:, :2], molecule[0, :2], atol=1e-9)

    # The substrate is unchanged as a periodic lattice: every atom coincides
    # with a slab atom modulo one lattice translation in the plane.
    aluminium = direct[labels == "Al"]
    aluminium_z = np.asarray(structure.positions_cartesian, dtype=float)[labels == "Al", 2]
    expected = np.asarray(slab.positions_direct, dtype=float)
    expected_z = np.asarray(slab.positions_cartesian, dtype=float)[:, 2]
    for row, height in zip(aluminium, aluminium_z):
        residuals = np.mod(row[:2] - expected[:, :2], 1.0)
        residuals = np.minimum(residuals, 1.0 - residuals)
        matches = np.all(residuals < 1e-9, axis=1) & (np.abs(expected_z - height) < 1e-9)
        assert matches.any()


def _expanded_substrate(tmp_path, slab_path, **kwargs):
    """Run the placement workflow on a slab and return its substrate cell."""

    workflow = Adsorbate(runs_root=str(tmp_path / "runs"), output_root=str(tmp_path / "output"))
    resolved, extra = workflow._resolve_substrate(
        substrate_path=slab_path,
        substrate_kind="slab",
        run_dir=tmp_path / "runs",
        **kwargs,
    )
    return io_mod.read_poscar(resolved), extra


def test_an_unexpanded_slab_substrate_is_used_as_given(tmp_path, aluminium_slab):
    structure, extra = _expanded_substrate(tmp_path, aluminium_slab)
    original = io_mod.read_poscar(aluminium_slab)

    assert sum(structure.counts) == sum(original.counts)
    assert np.allclose(structure.lattice, original.lattice)
    assert "expanded_substrate_poscar" not in extra


@pytest.mark.parametrize("repeat_a,repeat_b", [(2, 2), (3, 1)])
def test_a_slab_substrate_honours_the_requested_repeats(
    tmp_path, aluminium_slab, repeat_a, repeat_b
):
    """A slab given directly is enlarged exactly as a bulk-derived one is.

    The request used to be dropped for slab inputs, so the molecule landed on a
    cell smaller than the one that was asked for.
    """

    original = io_mod.read_poscar(aluminium_slab)
    structure, extra = _expanded_substrate(
        tmp_path, aluminium_slab, repeat_a=repeat_a, repeat_b=repeat_b
    )

    assert sum(structure.counts) == sum(original.counts) * repeat_a * repeat_b
    assert np.allclose(structure.lattice[0], original.lattice[0] * repeat_a)
    assert np.allclose(structure.lattice[1], original.lattice[1] * repeat_b)
    assert np.allclose(structure.lattice[2], original.lattice[2])
    assert extra["substrate_repeat"] == [repeat_a, repeat_b]


def test_a_slab_substrate_honours_an_in_plane_supercell_matrix(tmp_path, aluminium_slab):
    original = io_mod.read_poscar(aluminium_slab)
    matrix = np.array([[2, 0], [1, 2]], dtype=float)
    structure, extra = _expanded_substrate(
        tmp_path, aluminium_slab, supercell_matrix=[2, 0, 1, 2]
    )

    index = int(round(abs(float(np.linalg.det(matrix)))))
    assert sum(structure.counts) == sum(original.counts) * index
    assert np.allclose(np.asarray(structure.lattice)[:2, :], matrix @ np.asarray(original.lattice)[:2, :])
    assert list(extra["substrate_supercell_matrix"]) == [2, 0, 1, 2]


def test_the_expanded_substrate_keeps_every_atom_on_a_slab_site(tmp_path, aluminium_slab):
    """Repeating is a lattice operation, so every atom sits on an original site."""

    original = io_mod.read_poscar(aluminium_slab)
    structure, _ = _expanded_substrate(tmp_path, aluminium_slab, repeat_a=2, repeat_b=3)

    cartesian = np.asarray(structure.positions_cartesian, dtype=float)
    fractional = np.linalg.solve(np.asarray(original.lattice, dtype=float).T, cartesian.T).T
    residuals = np.mod(fractional[:, None, :] - np.asarray(original.positions_direct)[None, :, :], 1.0)
    residuals = np.minimum(residuals, 1.0 - residuals)
    assert np.all(np.any(np.all(residuals < 1e-9, axis=2), axis=1))
