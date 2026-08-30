"""Closest-approach measurements, and what a placement height guarantees.

A structure is only usable if no two atoms are on top of each other, and the
number that decides that is the shortest distance between two atoms -- over the
periodic images as well.  The searches in :mod:`cellstine.core.contacts` restrict
that infinite search to a box of lattice translations, so each of them is checked
here against a brute-force enumeration over a much wider box, on cells that are
deliberately skewed.  The bounds the restrictions rest on are proved in
``RequestProject/ContactDistance.lean``.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from cellstine.adsorbate.placement import operations as placement
from cellstine.core import contacts
from cellstine.core.elements import covalent_radius, element_symbol
from cellstine.interface.surface import backend as surface
from cellstine.io import native as io_mod

from conftest import write_poscar


HEXAGONAL_SLAB_CELL = np.array(
    [[6.0, 0.0, 0.0], [-3.0, 3.0 * math.sqrt(3.0), 0.0], [0.0, 0.0, 18.0]]
)


def _brute_force_cross(lattice, first, second, reach: int = 3) -> float:
    left = np.asarray(first) @ lattice
    right = np.asarray(second) @ lattice
    best = math.inf
    for i in range(-reach, reach + 1):
        for j in range(-reach, reach + 1):
            for k in range(-reach, reach + 1):
                shift = i * lattice[0] + j * lattice[1] + k * lattice[2]
                deltas = left[:, None, :] - (right[None, :, :] + shift)
                best = min(best, float(np.linalg.norm(deltas, axis=2).min()))
    return best


def _brute_force_self(lattice, points, reach: int = 3) -> float:
    cartesian = np.asarray(points) @ lattice
    best = math.inf
    for i in range(-reach, reach + 1):
        for j in range(-reach, reach + 1):
            for k in range(-reach, reach + 1):
                if (i, j, k) == (0, 0, 0):
                    continue
                shift = i * lattice[0] + j * lattice[1] + k * lattice[2]
                deltas = cartesian[:, None, :] - (cartesian[None, :, :] + shift)
                best = min(best, float(np.linalg.norm(deltas, axis=2).min()))
    return best


def test_element_symbol_reads_a_decorated_label():
    assert element_symbol("Fe1") == "Fe"
    assert element_symbol("C_surf") == "C"
    assert element_symbol("Co") == "Co"
    assert element_symbol("Zz", strict=False) is None
    with pytest.raises(ValueError):
        element_symbol("Zz")


def test_covalent_radius_is_unknown_for_an_unknown_label():
    assert covalent_radius("C") == pytest.approx(0.76)
    assert math.isnan(covalent_radius("Zz"))


@pytest.mark.parametrize("seed", range(6))
def test_closest_contact_matches_a_brute_force_search(seed: int):
    rng = np.random.default_rng(seed)
    lattice = HEXAGONAL_SLAB_CELL if seed % 2 else rng.normal(size=(3, 3)) * 3.0 + np.diag([9.0] * 3)
    first = rng.random((4, 3))
    second = rng.random((6, 3))
    found = contacts.closest_contact(lattice, first, second)
    assert found is not None
    assert found.distance == pytest.approx(_brute_force_cross(lattice, first, second), abs=1e-9)
    reference = np.linalg.norm(
        (first[found.first_index] - second[found.second_index]) @ lattice
        - np.round((first[found.first_index] - second[found.second_index])) @ lattice
    )
    assert found.distance <= reference + 1e-9


@pytest.mark.parametrize("seed", range(6))
def test_self_image_contact_matches_a_brute_force_search(seed: int):
    rng = np.random.default_rng(100 + seed)
    lattice = HEXAGONAL_SLAB_CELL if seed % 2 else rng.normal(size=(3, 3)) * 3.0 + np.diag([9.0] * 3)
    points = contacts.unwrap_group(lattice, rng.random((5, 3)))
    found = contacts.self_image_contact(lattice, points)
    assert found is not None
    assert found.distance == pytest.approx(_brute_force_self(lattice, points), abs=1e-9)


def test_self_image_contact_of_a_single_atom_is_the_shortest_translation():
    lattice = HEXAGONAL_SLAB_CELL
    found = contacts.self_image_contact(lattice, np.array([[0.2, 0.3, 0.4]]))
    assert found is not None
    assert found.distance == pytest.approx(6.0, abs=1e-9)


def test_a_molecule_split_across_a_face_is_put_back_together():
    """Fractional coordinates on either side of a face are one rigid body."""

    lattice = np.diag([10.0, 10.0, 20.0])
    split = np.array([[0.98, 0.5, 0.5], [0.02, 0.5, 0.5]])
    assert contacts.group_diameter(lattice, split) == pytest.approx(0.4, abs=1e-9)
    found = contacts.self_image_contact(lattice, split)
    assert found is not None
    assert found.distance == pytest.approx(10.0 - 0.4, abs=1e-9)


def test_self_image_distance_never_exceeds_the_shortest_translation():
    """Translating the whole group by a lattice vector is always a candidate."""

    rng = np.random.default_rng(7)
    lattice = HEXAGONAL_SLAB_CELL
    points = rng.random((6, 3))
    found = contacts.self_image_contact(lattice, points)
    assert found is not None
    assert found.distance <= 6.0 + 1e-9


def test_overlapping_atoms_are_called_out():
    lattice = np.diag([12.0, 12.0, 12.0])
    report = contacts.contact_report(
        lattice=lattice,
        group_direct=np.array([[0.5, 0.5, 0.55]]),
        other_direct=np.array([[0.5, 0.5, 0.5]]),
        group_species=["O"],
        other_species=["Al"],
    )
    assert report["contact_distance"] == pytest.approx(0.6, abs=1e-9)
    assert any("overlap" in note for note in report["notes"])


@pytest.fixture(scope="module")
def aluminium_slab(tmp_path_factory) -> str:
    workspace = tmp_path_factory.mktemp("contacts")
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
    path = tmp_path_factory.mktemp("contacts-molecule") / "co.vasp"
    positions = np.array([[0.5, 0.5, 0.5], [0.5, 0.5, 0.5 + 1.128 / 12.0]])
    return str(write_poscar(path, np.diag([12.0, 12.0, 12.0]), ["C", "O"], [1, 1], positions))


@pytest.mark.parametrize("height", [1.2, 2.0, 2.8])
def test_a_placement_reports_the_contact_it_really_makes(
    aluminium_slab, carbon_monoxide, tmp_path, height: float
):
    """The reported contact is the true closest approach, and at least the height."""

    run = placement.place_molecule_on_site(
        aluminium_slab,
        carbon_monoxide,
        site_type="fcc_hollow",
        site_index=1,
        height=height,
        output_path=str(tmp_path / f"placed_{height}.vasp"),
    )
    written = io_mod.read_poscar(str(run.output_path))
    positions = np.asarray(written.positions_direct, dtype=float)
    substrate = positions[: run.substrate_atom_count]
    molecule = positions[run.substrate_atom_count :]
    measured = _brute_force_cross(np.asarray(written.lattice, dtype=float), molecule, substrate)
    assert run.contact_distance == pytest.approx(measured, abs=1e-6)
    # A substrate that does not rise above its adsorption site cannot be closer
    # than the requested clearance -- `Cellstine.height_le_contactDistance`.
    assert run.contact_distance >= height - 1e-6


def test_a_hollow_site_placement_stands_further_off_than_its_height(
    aluminium_slab, carbon_monoxide, tmp_path
):
    """Over a hollow the nearest atoms are to the side, so the contact is longer."""

    run = placement.place_molecule_on_site(
        aluminium_slab,
        carbon_monoxide,
        site_type="fcc_hollow",
        site_index=1,
        height=1.5,
        output_path=str(tmp_path / "hollow.vasp"),
    )
    assert run.contact_distance > 1.5 + 0.05
    assert any("measured along the surface normal" in note for note in run.notes)


def test_a_placement_reports_its_own_image_separation(
    aluminium_slab, carbon_monoxide, tmp_path
):
    run = placement.place_molecule_on_site(
        aluminium_slab,
        carbon_monoxide,
        site_type="top",
        site_index=1,
        height=2.0,
        output_path=str(tmp_path / "top.vasp"),
    )
    written = io_mod.read_poscar(str(run.output_path))
    lattice = np.asarray(written.lattice, dtype=float)
    molecule = np.asarray(written.positions_direct, dtype=float)[run.substrate_atom_count :]
    reference = _brute_force_self(lattice, contacts.unwrap_group(lattice, molecule))
    assert run.self_image_distance == pytest.approx(reference, abs=1e-6)
    assert run.self_image_distance > run.contact_distance


def test_placement_keeps_the_molecule_rigid(aluminium_slab, carbon_monoxide, tmp_path):
    """A rotation and a translation change no distance inside the molecule."""

    run = placement.place_molecule_on_site(
        aluminium_slab,
        carbon_monoxide,
        site_type="bridge",
        site_index=1,
        height=2.0,
        rotation_deg=31.0,
        tilt_deg=17.0,
        output_path=str(tmp_path / "rigid.vasp"),
    )
    written = io_mod.read_poscar(str(run.output_path))
    lattice = np.asarray(written.lattice, dtype=float)
    molecule = contacts.unwrap_group(
        lattice, np.asarray(written.positions_direct, dtype=float)[run.substrate_atom_count :]
    ) @ lattice
    assert float(np.linalg.norm(molecule[0] - molecule[1])) == pytest.approx(1.128, abs=1e-6)


@pytest.fixture(scope="module")
def aluminium_111_slab(tmp_path_factory) -> str:
    """A six-layer Al(111) slab, built through the surface workflow."""

    from cellstine.interface.surface.surface import Surface

    workspace = tmp_path_factory.mktemp("contacts-interface")
    bulk = write_poscar(
        workspace / "al.vasp",
        4.05 * np.eye(3),
        ["Al"],
        [4],
        np.array([[0.0, 0.0, 0.0], [0.0, 0.5, 0.5], [0.5, 0.0, 0.5], [0.5, 0.5, 0.0]]),
    )
    tool = Surface(runs_root=str(workspace / "runs"), output_root=str(workspace / "output"))
    result = tool.surface(bulk_poscar=str(bulk), miller="111", layers=6, vacuum=12.0)
    return str(result.artifacts["slab_poscar"])


@pytest.mark.parametrize("gap", [2.0, 2.34, 3.0])
def test_an_interface_reports_the_contact_it_really_makes(aluminium_111_slab, tmp_path, gap: float):
    """The reported closest contact is the one the written structure has."""

    from cellstine.interface.workflow.interface import Interface

    tool = Interface(runs_root=str(tmp_path / "runs"), output_root=str(tmp_path / "output"))
    result = tool.build(
        bottom_input=aluminium_111_slab,
        top_input=aluminium_111_slab,
        gap=gap,
        registry="fcc",
    )
    reported = float(result.summary["closest_contact"])
    written = io_mod.read_poscar(str(result.artifacts["interface_poscar"]))
    lattice = np.asarray(written.lattice, dtype=float)
    direct = np.asarray(written.positions_direct, dtype=float)
    # The two slabs are identical, so the contact across the boundary is the
    # shortest distance between two atoms that are not in the same slab; the
    # shortest distance in the whole cell is either that or an intra-slab bond,
    # and the reported number must be one of the distances present.
    deltas = direct[:, None, :] - direct[None, :, :]
    distances = np.linalg.norm(
        (deltas - np.round(deltas)).reshape(-1, 3) @ lattice, axis=1
    ).reshape(direct.shape[0], -1)
    np.fill_diagonal(distances, np.inf)
    # The summary rounds to four decimals, so compare at that resolution.
    assert reported >= float(distances.min()) - 1e-4
    assert np.isclose(distances, reported, atol=1e-4).any()
    # A close-packed contact at a gap of `gap` cannot be shorter than the gap.
    assert reported >= gap - 1e-4


def test_an_interface_at_a_crushing_gap_is_called_out(aluminium_111_slab, tmp_path):
    from cellstine.interface.workflow.interface import Interface

    tool = Interface(runs_root=str(tmp_path / "runs"), output_root=str(tmp_path / "output"))
    result = tool.build(
        bottom_input=aluminium_111_slab,
        top_input=aluminium_111_slab,
        gap=1.0,
        registry="fcc",
    )
    assert float(result.summary["closest_contact"]) < 2.0
    assert any(
        "shorter than" in note or "overlap" in note for note in result.summary["warnings"]
    )
