"""Slab geometry across Bravais lattices and high-index faces.

The existing surface tests check face-centred aluminium on a handful of low
Miller indices.  The statements below are the ones that have to hold for *any*
lattice and *any* Miller index, and they are checked against the exact
crystallography of the input rather than against a stored answer:

* the in-plane cell of the slab is the **primitive** cell of the plane
  sublattice, which is the statement ``area * d_hkl == V_primitive``, where
  ``d_hkl`` is the spacing of the lattice planes measured from the translation
  lattice itself;
* both in-plane vectors are translations of the crystal;
* the slab is a piece of the bulk, so every atom sits on a bulk site and the
  atom count is exactly the number of stacking levels times the number of atoms
  on one level;
* the requested vacuum is the gap that is left.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from cellstine.core import symmetry3d as sym
from cellstine.interface.surface import backend as surface
from cellstine.io import native as io_mod

from conftest import write_poscar

MILLERS = [
    (0, 0, 1),
    (1, 0, 0),
    (1, 1, 0),
    (1, 1, 1),
    (2, 1, 0),
    (2, 1, 1),
    (3, 1, 1),
    (3, 2, 1),
    (5, 3, 1),
]


def _hexagonal(constant: float, height: float) -> np.ndarray:
    return np.array(
        [
            [constant, 0.0, 0.0],
            [-0.5 * constant, 0.5 * math.sqrt(3.0) * constant, 0.0],
            [0.0, 0.0, height],
        ]
    )


CRYSTALS: dict[str, tuple[np.ndarray, list[str], list[int], np.ndarray]] = {
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
    "hexagonal_close_packed": (
        _hexagonal(3.21, 5.21),
        ["Mg"],
        [2],
        np.array([[1.0 / 3.0, 2.0 / 3.0, 0.25], [2.0 / 3.0, 1.0 / 3.0, 0.75]]),
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
    "triclinic": (
        np.array([[3.1, 0.0, 0.0], [0.7, 3.4, 0.0], [0.4, -0.6, 4.2]]),
        ["X"],
        [1],
        np.array([[0.0, 0.0, 0.0]]),
    ),
}


@pytest.fixture(scope="module")
def crystals(tmp_path_factory) -> dict[str, str]:
    workspace = tmp_path_factory.mktemp("bravais")
    paths: dict[str, str] = {}
    for name, (lattice, species, counts, positions) in CRYSTALS.items():
        paths[name] = str(
            write_poscar(workspace / f"{name}.vasp", lattice, species, counts, positions, comment=name)
        )
    return paths


def _expanded_species(structure) -> list[str]:
    labels: list[str] = []
    for symbol, count in zip(structure.species, structure.counts):
        labels.extend([str(symbol)] * int(count))
    return labels


def _translation_lattice(structure) -> np.ndarray:
    """Return a primitive basis of the translation lattice of the crystal."""

    lattice, _, _ = sym.primitive_cell(
        np.asarray(structure.lattice, dtype=float),
        np.asarray(structure.positions_direct, dtype=float),
        _expanded_species(structure),
    )
    return np.asarray(lattice, dtype=float)


def _match_inplane_cell(
    translation_lattice: np.ndarray,
    slab_lattice: np.ndarray,
    normal: np.ndarray,
    *,
    reach: int = 8,
    tolerance: float = 1e-7,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return every ``(coefficients, rotation)`` match for a slab in-plane cell.

    The plane sublattice can be more symmetric than the crystal itself -- the
    hexagonal (111) plane of a body-centred lattice is the standard example --
    so several rotations carry the crystal cell onto the surface cell and only
    some of them respect the stacking.  All of them are returned.

    The slab is written in its own frame, so the in-plane vectors equal integer
    combinations of the crystal translations only after a rotation.  Recovering
    both is what makes "the surface cell is a sublattice of the crystal" a
    checkable statement.
    """

    grid = np.stack(
        np.meshgrid(*(np.arange(-reach, reach + 1),) * 3, indexing="ij"), axis=-1
    ).reshape(-1, 3)
    vectors = grid @ np.asarray(translation_lattice, dtype=float)
    in_plane = np.abs(vectors @ np.asarray(normal, dtype=float)) <= 1e-8
    grid = grid[in_plane]
    vectors = vectors[in_plane]
    lengths = np.linalg.norm(vectors, axis=1)

    first, second = np.asarray(slab_lattice, dtype=float)[:2]
    target_dot = float(first @ second)
    candidates_a = np.flatnonzero(np.abs(lengths - float(np.linalg.norm(first))) <= tolerance)
    candidates_b = np.flatnonzero(np.abs(lengths - float(np.linalg.norm(second))) <= tolerance)
    image = np.array([first, second, np.cross(first, second)])
    matches: list[tuple[np.ndarray, np.ndarray]] = []
    for index_a in candidates_a:
        vector_a = vectors[index_a]
        for index_b in candidates_b:
            vector_b = vectors[index_b]
            if abs(float(vector_a @ vector_b) - target_dot) > 1e-6:
                continue
            source = np.array([vector_a, vector_b, np.cross(vector_a, vector_b)])
            if abs(float(np.linalg.det(source))) < 1e-9:
                continue
            rotation = np.linalg.solve(source, image)
            if np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-8):
                matches.append((np.array([grid[index_a], grid[index_b]], dtype=float), rotation))
    if not matches:
        raise AssertionError("the in-plane cell of the slab is not a cell of the crystal lattice")
    return matches


def _lattice_plane_spacing(translation_lattice: np.ndarray, normal: np.ndarray, reach: int = 5) -> float:
    """Return the spacing of the lattice planes perpendicular to ``normal``."""

    grid = np.stack(
        np.meshgrid(*(np.arange(-reach, reach + 1),) * 3, indexing="ij"), axis=-1
    ).reshape(-1, 3)
    levels = np.sort(grid @ translation_lattice @ normal)
    gaps = np.diff(levels)
    return float(gaps[gaps > 1e-6].min())


@pytest.mark.parametrize("name", sorted(CRYSTALS))
@pytest.mark.parametrize("miller", MILLERS)
def test_the_slab_uses_the_primitive_cell_of_the_plane_lattice(crystals, name, miller):
    """``area * d_hkl == V_primitive`` is exactly the statement of primitivity.

    The plane sublattice of a three-dimensional lattice of primitive volume
    ``V`` has covolume ``V / d``, where ``d`` is the interplanar spacing.  An
    in-plane cell of that area therefore carries no repeated content.
    """

    path = crystals[name]
    bulk = io_mod.read_poscar(path)
    build = surface.build_surface_structure(path, miller=miller, layers=3, vacuum=12.0)

    lattice = np.asarray(build.structure.lattice, dtype=float)
    area = float(np.linalg.norm(np.cross(lattice[0], lattice[1])))
    translation = _translation_lattice(bulk)
    normal = surface._reciprocal_normal(np.asarray(bulk.lattice, dtype=float), miller)
    spacing = _lattice_plane_spacing(translation, normal)

    assert area * spacing == pytest.approx(abs(float(np.linalg.det(translation))), rel=1e-9)


@pytest.mark.parametrize("name", sorted(CRYSTALS))
@pytest.mark.parametrize("miller", MILLERS)
def test_the_in_plane_vectors_are_crystal_translations(crystals, name, miller):
    """Both surface vectors must be integer combinations of a primitive basis."""

    path = crystals[name]
    bulk = io_mod.read_poscar(path)
    build = surface.build_surface_structure(path, miller=miller, layers=3, vacuum=12.0)

    lattice = np.asarray(build.structure.lattice, dtype=float)
    translation = _translation_lattice(bulk)
    normal = surface._reciprocal_normal(np.asarray(bulk.lattice, dtype=float), miller)
    coefficients, rotation = _match_inplane_cell(translation, lattice, normal)[0]

    assert np.allclose(coefficients, np.round(coefficients), atol=1e-8)
    assert abs(abs(float(np.linalg.det(rotation))) - 1.0) < 1e-9
    recovered = coefficients @ translation
    assert np.allclose(recovered @ recovered.T, lattice[:2] @ lattice[:2].T, atol=1e-8)
    # The surface vectors are perpendicular to the plane normal of the crystal.
    assert np.allclose(recovered @ normal, 0.0, atol=1e-8)


@pytest.mark.parametrize("name", sorted(CRYSTALS))
@pytest.mark.parametrize("miller", [(1, 0, 0), (1, 1, 1), (2, 1, 1), (3, 2, 1)])
def test_the_slab_is_a_stack_of_identical_levels(crystals, name, miller):
    """``layers`` counts stacking levels, and every level carries the same load."""

    path = crystals[name]
    counts: list[int] = []
    for layers in (2, 3, 5):
        build = surface.build_surface_structure(path, miller=miller, layers=layers, vacuum=12.0)
        counts.append(int(sum(build.structure.counts)))

    per_level = counts[0] / 2.0
    assert per_level == int(per_level)
    assert counts == [int(per_level * layers) for layers in (2, 3, 5)]


@pytest.mark.parametrize("name", sorted(CRYSTALS))
@pytest.mark.parametrize("miller", [(1, 0, 0), (1, 1, 1), (3, 2, 1)])
def test_every_slab_atom_sits_on_a_bulk_site(crystals, name, miller):
    """A slab is a piece of the bulk: no atom may drift off a crystal site.

    The slab is written in a rotated frame, so the check is done on the
    difference vectors, which are frame independent up to that rotation: every
    atom-to-atom vector inside the slab must be a difference of bulk sites.
    """

    path = crystals[name]
    bulk = io_mod.read_poscar(path)
    build = surface.build_surface_structure(path, miller=miller, layers=3, vacuum=12.0)

    translation = _translation_lattice(bulk)
    bulk_lattice = np.asarray(bulk.lattice, dtype=float)
    reach = 5
    grid = np.stack(
        np.meshgrid(*(np.arange(-reach, reach + 1),) * 3, indexing="ij"), axis=-1
    ).reshape(-1, 3)
    sites = (
        np.asarray(bulk.positions_direct, dtype=float)[None, :, :] + grid[:, None, :]
    ).reshape(-1, 3) @ bulk_lattice
    bulk_labels = np.repeat(
        np.array(_expanded_species(bulk))[None, :], len(grid), axis=0
    ).reshape(-1)

    lattice = np.asarray(build.structure.lattice, dtype=float)
    normal = surface._reciprocal_normal(bulk_lattice, miller)
    slab_labels = np.array(_expanded_species(build.structure))
    cartesian = np.asarray(build.structure.positions_cartesian, dtype=float)

    # The slab has its own frame and its own origin, so try every rotation that
    # carries a crystal cell onto the surface cell, and every nearby bulk site
    # of the right species as the image of the first slab atom.
    matched = False
    for _, rotation in _match_inplane_cell(translation, lattice, normal):
        unrotated = cartesian @ np.linalg.inv(rotation)
        origin_candidates = sites[bulk_labels == slab_labels[0]]
        order = np.linalg.norm(origin_candidates, axis=1).argsort()[:4]
        for origin in origin_candidates[order]:
            shifted = unrotated - unrotated[0] + origin
            if all(
                float(np.linalg.norm(sites[bulk_labels == label] - point[None, :], axis=1).min()) < 1e-6
                for point, label in zip(shifted, slab_labels)
            ):
                matched = True
                break
        if matched:
            break
    assert matched, "a slab atom does not sit on a bulk site"


@pytest.mark.parametrize("name", sorted(CRYSTALS))
@pytest.mark.parametrize("miller", [(1, 1, 0), (2, 1, 1)])
def test_the_requested_vacuum_is_the_gap_that_is_left(crystals, name, miller):
    build = surface.build_surface_structure(crystals[name], miller=miller, layers=4, vacuum=13.5)
    lattice = np.asarray(build.structure.lattice, dtype=float)
    normal = np.cross(lattice[0], lattice[1])
    normal = normal / float(np.linalg.norm(normal))
    projections = np.asarray(build.structure.positions_cartesian, dtype=float) @ normal
    height = abs(float(lattice[2] @ normal))
    assert height - float(projections.max() - projections.min()) == pytest.approx(13.5, abs=1e-6)
