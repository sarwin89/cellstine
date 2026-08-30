"""Checks on slab construction and adsorption-site analysis.

A slab is correct when its in-plane cell is a two-dimensional sublattice of the
bulk, when every atom sits on a bulk site, and when the vacuum is the gap that
was asked for.  The adsorption sites are checked against the sites of the
low-index faces of a close-packed metal, whose count, position and empty-circle
radius are all known in closed form.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from cellstine.core import symmetry3d as sym
from cellstine.interface.surface import backend as surface
from cellstine.io import native as io_mod

from conftest import write_poscar


@pytest.fixture(scope="session")
def aluminium_poscar(tmp_path_factory) -> str:
    """Face-centred cubic aluminium, conventional four-atom cell."""

    path = tmp_path_factory.mktemp("surface") / "al.vasp"
    lattice = 4.05 * np.eye(3)
    positions = np.array([[0.0, 0.0, 0.0], [0.0, 0.5, 0.5], [0.5, 0.0, 0.5], [0.5, 0.5, 0.0]])
    return str(write_poscar(path, lattice, ["Al"], [4], positions))


def _bulk_sites(lattice: np.ndarray, positions: np.ndarray, reach: int = 3) -> np.ndarray:
    shifts = np.array(
        [[i, j, k] for i in range(-reach, reach + 1) for j in range(-reach, reach + 1) for k in range(-reach, reach + 1)],
        dtype=float,
    )
    return ((positions[None, :, :] + shifts[:, None, :]).reshape(-1, 3)) @ lattice


def _integer_vectors(bulk: np.ndarray, target_length: float, reach: int = 4, tolerance: float = 1e-6) -> np.ndarray:
    """Return every integer combination of the bulk vectors of a given length."""

    grid = np.array(
        [[i, j, k] for i in range(-reach, reach + 1) for j in range(-reach, reach + 1) for k in range(-reach, reach + 1)],
        dtype=float,
    )
    lengths = np.linalg.norm(grid @ bulk, axis=1)
    return grid[np.abs(lengths - target_length) <= tolerance]


def translation_lattice(structure) -> np.ndarray:
    """Return a primitive basis of the translation lattice of a structure.

    A surface vector is a translation of the crystal, which for a centred cell
    such as the conventional face-centred one is not an integer combination of
    the cell vectors, so the primitive basis is the right reference.
    """

    lattice, positions, species = sym.primitive_cell(
        np.asarray(structure.lattice, dtype=float),
        np.asarray(structure.positions_direct, dtype=float),
        list(structure.expanded_species) if hasattr(structure, "expanded_species") else _species_of(structure),
    )
    del positions, species
    return lattice


def _species_of(structure) -> list[str]:
    labels: list[str] = []
    for symbol, count in zip(structure.species, structure.counts):
        labels.extend([str(symbol)] * int(count))
    return labels


def match_inplane_sublattice(bulk: np.ndarray, slab: np.ndarray, tolerance: float = 1e-6):
    """Return ``(coefficients, rotation)`` proving that the in-plane cell of the
    slab is a two-dimensional sublattice of the translation lattice ``bulk``.

    The builder writes the slab in a frame whose third axis is the surface
    normal, so the surface vectors equal integer combinations of the bulk
    vectors only after a rotation.  This recovers both the integer coefficients
    and that rotation, which is what makes the claim checkable.
    """

    bulk = np.asarray(bulk, dtype=float)
    first, second = np.asarray(slab, dtype=float)[:2]
    candidates_a = _integer_vectors(bulk, float(np.linalg.norm(first)), tolerance=tolerance)
    candidates_b = _integer_vectors(bulk, float(np.linalg.norm(second)), tolerance=tolerance)
    target_dot = float(first @ second)
    for coefficients_a in candidates_a:
        vector_a = coefficients_a @ bulk
        for coefficients_b in candidates_b:
            vector_b = coefficients_b @ bulk
            if abs(float(vector_a @ vector_b) - target_dot) > tolerance:
                continue
            source = np.array([vector_a, vector_b, np.cross(vector_a, vector_b)])
            image = np.array([first, second, np.cross(first, second)])
            rotation = np.linalg.solve(source, image)
            if np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-8):
                return np.array([coefficients_a, coefficients_b]), rotation
    raise AssertionError("the slab in-plane cell is not a sublattice of the bulk lattice")


@pytest.mark.parametrize("miller", [(0, 0, 1), (1, 1, 0), (1, 1, 1), (2, 1, 1)])
def test_slab_atoms_sit_on_bulk_sites(aluminium_poscar, tmp_path, miller):
    bulk = io_mod.read_poscar(aluminium_poscar)
    output = tmp_path / f"slab_{miller[0]}{miller[1]}{miller[2]}.vasp"
    surface.build_surface(aluminium_poscar, miller=miller, layers=6, vacuum=15.0, output_path=str(output))
    slab = io_mod.read_poscar(str(output))

    bulk_lattice = np.asarray(bulk.lattice, dtype=float)
    _, rotation = match_inplane_sublattice(translation_lattice(bulk), np.asarray(slab.lattice, dtype=float))
    reference = _bulk_sites(bulk_lattice, np.asarray(bulk.positions_direct, dtype=float), reach=4)
    unrotated = np.asarray(slab.positions_cartesian, dtype=float) @ np.linalg.inv(rotation)
    unrotated = unrotated - unrotated[0]
    for point in unrotated:
        assert np.linalg.norm(reference - point[None, :], axis=1).min() < 1e-6


@pytest.mark.parametrize("miller", [(0, 0, 1), (1, 1, 0), (1, 1, 1)])
def test_slab_inplane_cell_is_a_bulk_sublattice(aluminium_poscar, tmp_path, miller):
    """The two surface vectors must be integer combinations of the bulk vectors."""

    bulk = io_mod.read_poscar(aluminium_poscar)
    output = tmp_path / f"sub_{miller[0]}{miller[1]}{miller[2]}.vasp"
    surface.build_surface(aluminium_poscar, miller=miller, layers=4, vacuum=12.0, output_path=str(output))
    slab = io_mod.read_poscar(str(output))
    coefficients, rotation = match_inplane_sublattice(
        translation_lattice(bulk), np.asarray(slab.lattice, dtype=float)
    )
    assert np.allclose(coefficients, np.round(coefficients), atol=1e-8)
    assert abs(abs(float(np.linalg.det(rotation))) - 1.0) < 1e-9


@pytest.mark.parametrize("miller", [(0, 0, 1), (1, 1, 0), (1, 1, 1)])
def test_slab_vacuum_gap_matches_the_request(aluminium_poscar, tmp_path, miller):
    output = tmp_path / f"vac_{miller[0]}{miller[1]}{miller[2]}.vasp"
    surface.build_surface(aluminium_poscar, miller=miller, layers=5, vacuum=13.0, output_path=str(output))
    slab = io_mod.read_poscar(str(output))
    lattice = np.asarray(slab.lattice, dtype=float)
    normal = np.cross(lattice[0], lattice[1])
    normal = normal / np.linalg.norm(normal)
    height = float(abs(lattice[2] @ normal))
    projections = np.asarray(slab.positions_cartesian, dtype=float) @ normal
    thickness = float(projections.max() - projections.min())
    assert height - thickness == pytest.approx(13.0, abs=1e-6)


def test_close_packed_face_has_one_fcc_and_one_hcp_hollow(aluminium_poscar, tmp_path):
    output = tmp_path / "al_111.vasp"
    surface.build_surface(aluminium_poscar, miller=(1, 1, 1), layers=6, vacuum=15.0, output_path=str(output))
    report = surface.find_adsorption_sites(str(output))

    assert report.site_counts == {"top": 1, "bridge": 3, "fcc_hollow": 1, "hcp_hollow": 1}
    spacing = 4.05 / math.sqrt(2.0)
    assert report.nearest_neighbor_distance == pytest.approx(spacing, abs=1e-6)
    assert report.average_top_layer_coordination == pytest.approx(6.0)
    assert report.detected_layer_count == 6

    hollows = {site.site_type: site for site in report.sites if site.site_type.endswith("hollow")}
    for site in hollows.values():
        assert site.coordination == 3
        assert site.void_radius == pytest.approx(spacing / math.sqrt(3.0), abs=1e-3)
    assert hollows["hcp_hollow"].subsurface_depth == 1
    assert hollows["fcc_hollow"].subsurface_depth == 2


def test_square_face_has_a_single_fourfold_hollow(aluminium_poscar, tmp_path):
    output = tmp_path / "al_001.vasp"
    surface.build_surface(aluminium_poscar, miller=(0, 0, 1), layers=6, vacuum=15.0, output_path=str(output))
    report = surface.find_adsorption_sites(str(output))

    assert report.site_counts == {"top": 1, "bridge": 2, "fourfold_hollow": 1}
    hollow = next(site for site in report.sites if site.site_type == "fourfold_hollow")
    assert hollow.coordination == 4
    assert hollow.void_radius == pytest.approx(4.05 / 2.0, abs=1e-3)


def test_honeycomb_monolayer_exposes_its_hexagon_centre(graphene_poscar):
    """A search for triangles of mutual neighbours finds no hollow on graphene;
    the hollow is the centre of the carbon hexagon."""

    report = surface.find_adsorption_sites(str(graphene_poscar))
    assert report.site_counts == {"top": 2, "bridge": 3, "hollow": 1}
    hollow = next(site for site in report.sites if site.site_type == "hollow")
    assert hollow.coordination == 6
    assert hollow.void_radius == pytest.approx(2.46 / math.sqrt(3.0), abs=1e-3)
    assert np.allclose(np.asarray(hollow.direct)[:2], [0.0, 0.0], atol=1e-3)


def test_buckled_bilayer_hollows_are_not_named_after_close_packing(silicon_poscar, tmp_path):
    """Silicon (111) is a stack of buckled bilayers, not a close-packed stack, so
    its hollows are reported generically together with the depth of the atom
    beneath them rather than as ``fcc``/``hcp``."""

    output = tmp_path / "si_111.vasp"
    surface.build_surface(str(silicon_poscar), miller=(1, 1, 1), layers=6, vacuum=15.0, output_path=str(output))
    report = surface.find_adsorption_sites(str(output))
    hollows = [site for site in report.sites if site.site_type.endswith("hollow")]
    assert len(hollows) == 2
    assert {site.site_type for site in hollows} == {"hollow"}
    depths = sorted(site.subsurface_depth for site in hollows if site.subsurface_depth is not None)
    assert depths == [2, 4]


def test_site_report_records_the_written_slab(aluminium_poscar, tmp_path):
    output = tmp_path / "al_report.vasp"
    run = surface.build_surface(
        aluminium_poscar,
        miller=(1, 1, 1),
        layers=4,
        vacuum=12.0,
        output_path=str(output),
        analyse_sites=True,
    )
    assert run.site_output_path is not None
    import json

    payload = json.loads(run.site_output_path.read_text())
    assert payload["source_poscar"] == str(output.resolve())
    assert payload["detected_layer_count"] == 4
    for entries in payload["sites"].values():
        for entry in entries:
            assert len(entry["direct"]) == 3
            assert len(entry["cartesian"]) == 3


def test_selecting_a_site_by_name_and_index(aluminium_poscar, tmp_path):
    output = tmp_path / "al_select.vasp"
    surface.build_surface(aluminium_poscar, miller=(1, 1, 1), layers=4, vacuum=12.0, output_path=str(output))
    report = surface.find_adsorption_sites(str(output))
    chosen = surface.select_adsorption_site(report, "fcc", 1)
    assert chosen.site_type == "fcc_hollow"
    with pytest.raises(ValueError):
        surface.select_adsorption_site(report, "fcc", 2)
    with pytest.raises(ValueError):
        surface.select_adsorption_site(report, "not_a_site")
