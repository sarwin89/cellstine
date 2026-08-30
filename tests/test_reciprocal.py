"""Mathematical checks on reciprocal lattices and Brillouin-zone sampling.

A k-point mesh is only usable if three things hold exactly: the reciprocal basis
is dual to the cell, the mesh really samples at least as finely as it claims,
and the symmetry reduction keeps every point accounted for -- each orbit
represented once, with its exact size as the weight.  These tests check all
three against brute force, against the textbook irreducible counts of the cubic
meshes, and against the folding relation that ties a supercell mesh to the
primitive one.
"""

from __future__ import annotations

import math
from itertools import product

import numpy as np
import pytest

from cellstine.core import reciprocal as rc
from cellstine.core import symmetry3d
from cellstine.io import kpoints as kpoints_io
from cellstine.io import native as io_mod

from conftest import hexagonal_basis


def _cubic(constant: float = 4.05) -> np.ndarray:
    return float(constant) * np.eye(3)


def _face_centred(constant: float = 4.05) -> tuple[np.ndarray, np.ndarray, list[str]]:
    lattice = _cubic(constant)
    positions = np.array(
        [[0.0, 0.0, 0.0], [0.0, 0.5, 0.5], [0.5, 0.0, 0.5], [0.5, 0.5, 0.0]], dtype=float
    )
    return lattice, positions, ["Al"] * 4


def _hexagonal(constant: float = 2.46, height: float = 6.7) -> tuple[np.ndarray, np.ndarray, list[str]]:
    basis = hexagonal_basis(constant)
    lattice = np.array(
        [
            [basis[0][0], basis[1][0], 0.0],
            [basis[0][1], basis[1][1], 0.0],
            [0.0, 0.0, float(height)],
        ],
        dtype=float,
    )
    positions = np.array([[0.0, 0.0, 0.0], [1.0 / 3.0, 2.0 / 3.0, 0.0]], dtype=float)
    return lattice, positions, ["C", "C"]


def _random_lattices(count: int = 12, seed: int = 20240825) -> list[np.ndarray]:
    generator = np.random.default_rng(seed)
    lattices = []
    while len(lattices) < count:
        candidate = generator.normal(scale=3.0, size=(3, 3))
        if abs(float(np.linalg.det(candidate))) > 0.5:
            lattices.append(candidate)
    return lattices


def test_the_reciprocal_basis_is_dual_to_the_cell():
    for lattice in _random_lattices():
        basis = rc.reciprocal_lattice(lattice)
        assert np.allclose(lattice @ basis.T, 2.0 * math.pi * np.eye(3), atol=1e-9)
        assert np.allclose(rc.reciprocal_lattice(basis), lattice, atol=1e-9)


def test_the_zone_volume_is_the_reciprocal_of_the_cell_volume():
    for lattice in _random_lattices():
        volume = rc.cell_volume(lattice)
        assert volume > 0.0
        zone = rc.brillouin_zone_volume(lattice)
        assert zone == pytest.approx((2.0 * math.pi) ** 3 / volume, rel=1e-12)
        assert abs(float(np.linalg.det(rc.reciprocal_lattice(lattice)))) == pytest.approx(
            zone, rel=1e-9
        )


def test_a_singular_cell_is_refused():
    flat = np.array([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    with pytest.raises(ValueError):
        rc.reciprocal_lattice(flat)
    with pytest.raises(ValueError):
        rc.mesh_divisions_for_spacing(_cubic(), 0.0)


def test_the_divisions_meet_the_requested_spacing_and_are_the_smallest_that_do():
    lattice, _, _ = _hexagonal()
    for spacing in (0.5, 0.3, 0.17, 0.05):
        divisions = rc.mesh_divisions_for_spacing(lattice, spacing)
        steps = rc.mesh_spacings(lattice, divisions)
        assert max(steps) <= spacing + 1e-12
        for axis, count in enumerate(divisions):
            if count > 1:
                coarser = list(divisions)
                coarser[axis] = count - 1
                assert max(rc.mesh_spacings(lattice, coarser)) > spacing + 1e-12


def test_a_surface_normal_can_be_pinned_to_one_division():
    lattice, _, _ = _hexagonal(height=20.0)
    divisions = rc.mesh_divisions_for_spacing(lattice, 0.2, minimum=(1, 1, 1))
    assert divisions[2] >= 1
    pinned = rc.build_mesh(lattice, divisions=(divisions[0], divisions[1], 1))
    assert pinned.divisions[2] == 1


def test_the_mesh_is_the_grid_it_promises():
    divisions = (3, 4, 5)
    points = rc.mesh_points(divisions)
    assert len(points) == 3 * 4 * 5
    assert np.all(points >= -0.5 - 1e-15) and np.all(points < 0.5 - 1e-15 + 1e-12)
    assert np.any(np.all(points == 0.0, axis=1)), "a Gamma-centred mesh contains Gamma exactly"
    unique = {tuple(np.round(point, 12)) for point in points}
    assert len(unique) == len(points)
    scaled = points * np.asarray(divisions, dtype=float)
    assert np.allclose(scaled, np.rint(scaled), atol=1e-12)


def test_the_monkhorst_pack_offset_is_half_a_step_on_even_axes():
    assert rc.mesh_shift((4, 5, 6), "monkhorst") == (0.5, 0.0, 0.5)
    assert rc.mesh_shift((4, 5, 6), "gamma") == (0.0, 0.0, 0.0)
    with pytest.raises(ValueError):
        rc.mesh_shift((4, 4, 4), "centred")
    shifted = rc.mesh_points((2, 2, 2), shift=rc.mesh_shift((2, 2, 2), "monkhorst"))
    assert not np.any(np.all(np.isclose(shifted, 0.0), axis=1))
    assert np.allclose(np.sort(np.unique(shifted[:, 0])), [-0.25, 0.25])


def test_an_odd_monkhorst_mesh_is_the_gamma_mesh():
    assert np.allclose(
        np.sort(rc.mesh_points((5, 5, 5), shift=rc.mesh_shift((5, 5, 5), "monkhorst")), axis=0),
        np.sort(rc.mesh_points((5, 5, 5)), axis=0),
    )


def test_a_shift_that_is_not_a_half_step_is_refused():
    with pytest.raises(ValueError):
        rc.mesh_points((4, 4, 4), shift=(0.3, 0.0, 0.0))


def _brute_force_orbits(
    points: np.ndarray, rotations: np.ndarray, *, time_reversal: bool
) -> list[frozenset[int]]:
    """Group mesh points by symmetry with an independent float implementation."""

    keys = {tuple(np.round(np.mod(point + 0.5, 1.0) - 0.5, 9)): index for index, point in enumerate(points)}

    def locate(vector: np.ndarray) -> int | None:
        wrapped = np.mod(np.asarray(vector, dtype=float) + 0.5, 1.0) - 0.5
        return keys.get(tuple(np.round(wrapped, 9)))

    maps = [np.linalg.inv(np.asarray(rotation, dtype=float)) for rotation in rotations]
    orbits: list[frozenset[int]] = []
    seen: set[int] = set()
    for index, point in enumerate(points):
        if index in seen:
            continue
        orbit = {index}
        for matrix in maps:
            image = point @ matrix
            for candidate in ((image,) if not time_reversal else (image, -image)):
                found = locate(candidate)
                if found is None:
                    return []
                orbit.add(found)
        orbits.append(frozenset(orbit))
        seen |= orbit
    return orbits


@pytest.mark.parametrize("mode", ["gamma", "monkhorst"])
@pytest.mark.parametrize("divisions", [(4, 4, 4), (6, 6, 6), (3, 3, 3)])
def test_the_reduction_reproduces_a_brute_force_orbit_count(divisions, mode):
    lattice, positions, species = _face_centred()
    rotations, _ = symmetry3d.symmetry_operations(lattice, positions, species)
    mesh = rc.build_mesh(lattice, divisions=divisions, mode=mode, rotations=rotations)
    whole = rc.mesh_points(divisions, shift=mesh.shift)
    orbits = _brute_force_orbits(whole, rotations, time_reversal=True)
    assert orbits, "every operation must map this mesh onto itself"
    assert mesh.point_count == len(orbits)
    assert sorted(int(item) for item in mesh.weights) == sorted(len(orbit) for orbit in orbits)
    assert int(np.sum(mesh.weights)) == mesh.full_point_count


def test_the_cubic_irreducible_counts_are_the_textbook_ones():
    """For m-3m the Gamma and Monkhorst-Pack meshes have closed-form counts."""

    lattice, positions, species = _face_centred()
    rotations, _ = symmetry3d.symmetry_operations(lattice, positions, species)
    for count in (4, 6, 8, 12):
        gamma = rc.build_mesh(lattice, divisions=(count,) * 3, rotations=rotations)
        monkhorst = rc.build_mesh(
            lattice, divisions=(count,) * 3, mode="monkhorst", rotations=rotations
        )
        assert gamma.point_count == (count + 2) * (count + 4) * (count + 6) // 48
        assert monkhorst.point_count == count * (count + 2) * (count + 4) // 48
        assert gamma.operations_used == 48
        assert monkhorst.operations_used == 48


def test_every_mesh_point_is_equivalent_to_exactly_one_representative():
    lattice, positions, species = _face_centred()
    rotations, _ = symmetry3d.symmetry_operations(lattice, positions, species)
    divisions = (6, 6, 6)
    mesh = rc.build_mesh(lattice, divisions=divisions, rotations=rotations)
    whole = rc.mesh_points(divisions)
    representatives = {tuple(np.round(point, 9)) for point in mesh.points}
    matched = np.zeros(len(whole), dtype=int)
    for index, point in enumerate(whole):
        images = set()
        for rotation in rotations:
            image = point @ np.linalg.inv(np.asarray(rotation, dtype=float))
            for signed in (image, -image):
                wrapped = np.mod(signed + 0.5, 1.0) - 0.5
                images.add(tuple(np.round(wrapped, 9)))
        matched[index] = len(images & representatives)
    assert np.all(matched == 1)


def test_no_two_representatives_are_symmetry_equivalent():
    lattice, positions, species = _face_centred()
    rotations, _ = symmetry3d.symmetry_operations(lattice, positions, species)
    mesh = rc.build_mesh(lattice, divisions=(8, 8, 8), rotations=rotations)
    labels = {}
    for index, point in enumerate(mesh.points):
        for rotation in rotations:
            image = point @ np.linalg.inv(np.asarray(rotation, dtype=float))
            for signed in (image, -image):
                key = tuple(np.round(np.mod(signed + 0.5, 1.0) - 0.5, 9))
                assert labels.setdefault(key, index) == index


def test_time_reversal_alone_halves_a_mesh_without_inversion_symmetry():
    lattice, _, _ = _hexagonal()
    mesh = rc.build_mesh(lattice, divisions=(6, 6, 2), rotations=None, time_reversal=True)
    plain = rc.build_mesh(lattice, divisions=(6, 6, 2), rotations=None, time_reversal=False)
    assert plain.point_count == plain.full_point_count == 72
    assert mesh.full_point_count == 72
    assert mesh.point_count == 40
    assert int(np.sum(mesh.weights)) == 72
    assert set(int(item) for item in mesh.weights) <= {1, 2}


def test_a_graphite_mesh_carries_the_full_point_group():
    lattice, positions, species = _hexagonal()
    rotations, _ = symmetry3d.symmetry_operations(lattice, positions, species)
    mesh = rc.build_mesh(lattice, divisions=(6, 6, 3), rotations=rotations)
    assert mesh.operations_used == mesh.operations_given
    assert mesh.symmetry_complete
    assert int(np.sum(mesh.weights)) == mesh.full_point_count


def test_a_mesh_that_breaks_the_symmetry_reports_the_operations_it_lost():
    """Unequal divisions along symmetry-related axes cannot carry the group."""

    lattice, positions, species = _face_centred()
    rotations, _ = symmetry3d.symmetry_operations(lattice, positions, species)
    lopsided = rc.build_mesh(lattice, divisions=(4, 4, 3), rotations=rotations)
    assert lopsided.operations_used < lopsided.operations_given
    assert not lopsided.symmetry_complete
    assert int(np.sum(lopsided.weights)) == lopsided.full_point_count
    even = rc.build_mesh(lattice, divisions=(4, 4, 4), rotations=rotations)
    assert even.symmetry_complete


def test_a_half_shift_along_one_axis_only_drops_the_operations_that_move_it():
    lattice, positions, species = _face_centred()
    rotations, _ = symmetry3d.symmetry_operations(lattice, positions, species)
    mesh = rc.build_mesh(lattice, divisions=(4, 4, 4), shift=(0.5, 0.0, 0.0), rotations=rotations)
    assert 0 < mesh.operations_used < mesh.operations_given
    assert int(np.sum(mesh.weights)) == mesh.full_point_count


def test_generators_are_closed_into_their_group_before_reducing():
    lattice, positions, species = _face_centred()
    rotations, _ = symmetry3d.symmetry_operations(lattice, positions, species)
    full = rc.build_mesh(lattice, divisions=(6, 6, 6), rotations=rotations)
    generators, _ = symmetry3d.generating_operations(rotations, np.zeros((len(rotations), 3)))
    assert len(generators) < len(rotations)
    partial = rc.build_mesh(lattice, divisions=(6, 6, 6), rotations=generators)
    assert partial.point_count == full.point_count
    assert partial.operations_used == full.operations_used


def test_rotations_that_are_not_a_point_group_are_refused():
    shear = np.array([[1, 1, 0], [0, 1, 0], [0, 0, 1]], dtype=np.int64)
    with pytest.raises(ValueError):
        rc.build_mesh(_cubic(), divisions=(4, 4, 4), rotations=[shear])
    singular = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 2]], dtype=np.int64)
    with pytest.raises(ValueError):
        rc.build_mesh(_cubic(), divisions=(4, 4, 4), rotations=[singular])


def test_the_cartesian_points_are_the_fractional_ones_in_the_reciprocal_basis():
    lattice, positions, species = _hexagonal()
    rotations, _ = symmetry3d.symmetry_operations(lattice, positions, species)
    mesh = rc.build_mesh(lattice, divisions=(4, 4, 2), rotations=rotations)
    cartesian = mesh.cartesian_points(lattice)
    assert np.allclose(cartesian, mesh.points @ rc.reciprocal_lattice(lattice))
    assert np.all(np.linalg.norm(cartesian, axis=1) <= max(
        np.linalg.norm(rc.reciprocal_lattice(lattice), axis=1)
    ))


def test_a_supercell_mesh_samples_the_same_wavevectors_as_the_primitive_one():
    """A 2x2x2 supercell with half the divisions samples a subset of the same k."""

    lattice = _cubic()
    divisions = (8, 8, 8)
    repeats = np.diag([2, 2, 2])
    supercell = repeats @ lattice
    folded = rc.supercell_divisions(divisions, repeats)
    assert folded == (4, 4, 4)
    primitive_points = rc.mesh_points(divisions) @ rc.reciprocal_lattice(lattice)
    supercell_points = rc.mesh_points(folded) @ rc.reciprocal_lattice(supercell)
    coarse = {tuple(np.round(point, 9)) for point in primitive_points}
    for point in supercell_points:
        assert tuple(np.round(point, 9)) in coarse
    assert rc.kpoint_density(lattice, divisions) == pytest.approx(
        rc.kpoint_density(supercell, folded), rel=1e-12
    )


def test_the_supercell_divisions_never_sample_more_coarsely_than_asked():
    lattice, _, _ = _hexagonal()
    divisions = (9, 9, 3)
    for repeat in (2, 3, 4):
        matrix = np.diag([repeat, repeat, 1])
        folded = rc.supercell_divisions(divisions, matrix)
        supercell = matrix @ lattice
        assert max(rc.mesh_spacings(supercell, folded)) <= max(
            rc.mesh_spacings(lattice, divisions)
        ) + 1e-12


def test_a_non_diagonal_supercell_is_refused_with_a_pointer_to_the_right_call():
    with pytest.raises(ValueError, match="mesh_divisions_for_spacing"):
        rc.supercell_divisions((4, 4, 4), [[1, 1, 0], [0, 1, 0], [0, 0, 1]])
    with pytest.raises(ValueError):
        rc.supercell_divisions((4, 4, 4), [[0.5, 0, 0], [0, 1, 0], [0, 0, 1]])


def test_the_density_counts_points_per_unit_zone_volume():
    lattice = _cubic(3.0)
    density = rc.kpoint_density(lattice, (4, 4, 4))
    assert density == pytest.approx(64.0 / rc.brillouin_zone_volume(lattice), rel=1e-12)


def test_an_automatic_kpoints_file_round_trips(tmp_path):
    path = kpoints_io.write_automatic_kpoints(
        tmp_path / "KPOINTS", (6, 6, 2), shift=(0.0, 0.0, 0.0), comment="graphene"
    )
    parsed = kpoints_io.read_kpoints(path)
    assert parsed.mode == "gamma"
    assert parsed.divisions == (6, 6, 2)
    assert parsed.shift == (0.0, 0.0, 0.0)
    assert parsed.comment == "graphene"
    assert parsed.point_count == 0
    shifted = kpoints_io.write_automatic_kpoints(
        tmp_path / "KPOINTS-mp", (6, 6, 2), shift=(0.5, 0.5, 0.0), gamma_centred=False
    )
    reread = kpoints_io.read_kpoints(shifted)
    assert reread.mode == "monkhorst"
    assert reread.shift == (0.5, 0.5, 0.0)


def test_an_explicit_kpoints_file_round_trips_exactly(tmp_path):
    lattice, positions, species = _face_centred()
    rotations, _ = symmetry3d.symmetry_operations(lattice, positions, species)
    mesh = rc.build_mesh(lattice, divisions=(6, 6, 6), rotations=rotations)
    path = kpoints_io.write_mesh(tmp_path / "KPOINTS", mesh)
    parsed = kpoints_io.read_kpoints(path)
    assert parsed.mode == "explicit"
    assert parsed.coordinate_mode == "reciprocal"
    assert parsed.point_count == mesh.point_count
    assert np.allclose(parsed.points, mesh.points, atol=1e-10)
    assert np.array_equal(parsed.weights.astype(np.int64), mesh.weights)
    assert int(parsed.weights.sum()) == mesh.full_point_count
    text = path.read_text(encoding="utf-8")
    assert text.splitlines()[1].strip() == str(mesh.point_count)
    assert "." not in text.splitlines()[3].split()[-1], "an orbit size is written as an integer"


def test_writing_a_mesh_chooses_the_layout_that_keeps_the_reduction(tmp_path):
    lattice = _cubic()
    unreduced = rc.build_mesh(lattice, divisions=(4, 4, 4), time_reversal=False)
    automatic = kpoints_io.write_mesh(tmp_path / "KPOINTS-auto", unreduced)
    assert kpoints_io.read_kpoints(automatic).mode == "gamma"
    reduced = rc.build_mesh(lattice, divisions=(4, 4, 4), time_reversal=True)
    listed = kpoints_io.write_mesh(tmp_path / "KPOINTS-list", reduced)
    assert kpoints_io.read_kpoints(listed).mode == "explicit"
    forced = kpoints_io.write_mesh(tmp_path / "KPOINTS-forced", reduced, explicit=False)
    assert kpoints_io.read_kpoints(forced).divisions == (4, 4, 4)


def test_broken_kpoints_files_are_refused(tmp_path):
    short = tmp_path / "KPOINTS-short"
    short.write_text("comment\n0\n", encoding="utf-8")
    with pytest.raises(ValueError):
        kpoints_io.read_kpoints(short)
    unknown = tmp_path / "KPOINTS-unknown"
    unknown.write_text("comment\n0\nAuto\n10\n", encoding="utf-8")
    with pytest.raises(ValueError):
        kpoints_io.read_kpoints(unknown)
    truncated = tmp_path / "KPOINTS-truncated"
    truncated.write_text("comment\n3\nReciprocal\n0 0 0 1\n", encoding="utf-8")
    with pytest.raises(ValueError):
        kpoints_io.read_kpoints(truncated)
    with pytest.raises(ValueError):
        kpoints_io.write_explicit_kpoints(tmp_path / "KPOINTS-bad", [[0, 0, 0]], [0.0])
    with pytest.raises(ValueError):
        kpoints_io.write_automatic_kpoints(tmp_path / "KPOINTS-bad", (0, 1, 1))


def test_a_generated_slab_gets_a_mesh_with_one_division_along_the_vacuum(silicon_poscar, tmp_path):
    """The workflow structures and the mesh helpers agree on the same cell."""

    from cellstine.interface.surface.surface import Surface

    workflow = Surface(runs_root=str(tmp_path / "runs"), output_root=str(tmp_path / "out"))
    result = workflow.surface(
        bulk_poscar=str(silicon_poscar), miller="111", layers=2, vacuum=12.0
    )
    slab = io_mod.read_poscar(str(result.artifacts["slab_poscar"]))
    spacing = 0.25
    divisions = rc.mesh_divisions_for_spacing(slab.lattice, spacing)
    lengths = np.linalg.norm(rc.reciprocal_lattice(slab.lattice), axis=1)
    assert divisions[2] == max(1, math.ceil(lengths[2] / spacing))
    assert divisions[2] < divisions[0], "the slab is long along c, so its zone is short"
    rotations, _ = symmetry3d.symmetry_operations(
        slab.lattice, slab.positions_direct, list(np.repeat(slab.species, slab.counts))
    )
    pinned = (divisions[0], divisions[1], 1)
    mesh = rc.build_mesh(slab.lattice, divisions=pinned, rotations=rotations)
    assert int(np.sum(mesh.weights)) == mesh.full_point_count
    assert mesh.point_count < mesh.full_point_count
    assert max(mesh.spacings[:2]) <= spacing + 1e-12


def test_the_normalised_weights_are_a_probability_over_the_full_mesh():
    """A Brillouin-zone average is a weighted mean, so the weights must sum to one.

    Each irreducible point stands for its whole orbit, and its share of the
    zone is the size of that orbit divided by the number of mesh points.  Any
    other normalisation would silently rescale every averaged quantity.
    """

    lattice, positions, species = _hexagonal()
    rotations, _ = symmetry3d.symmetry_operations(lattice, positions, species)
    for divisions in [(6, 6, 3), (4, 4, 2), (5, 5, 5), (3, 3, 1)]:
        for time_reversal in (False, True):
            mesh = rc.build_mesh(
                lattice, divisions=divisions, rotations=rotations, time_reversal=time_reversal
            )
            shares = mesh.normalised_weights
            assert shares.shape == (mesh.point_count,)
            assert float(np.sum(shares)) == pytest.approx(1.0, rel=1e-12)
            assert np.all(shares > 0.0)
            assert shares == pytest.approx(mesh.weights / mesh.full_point_count, rel=1e-12)


def test_an_unreduced_mesh_weights_every_point_equally():
    lattice, _, _ = _hexagonal()
    mesh = rc.build_mesh(lattice, divisions=(4, 4, 2), rotations=None, time_reversal=False)
    shares = mesh.normalised_weights
    assert mesh.point_count == mesh.full_point_count == 32
    assert shares == pytest.approx(np.full(32, 1.0 / 32.0), rel=1e-12)
