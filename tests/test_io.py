"""Checks that structure I/O and orientation are lossless and rigid.

Every later stage reads and rewrites structures, so a silent change of cell,
coordinates, or species order there would corrupt every workflow.  The tests
below pin the two properties that matter: a POSCAR round trip returns exactly
what was written, and the orientation helpers are rigid rotations, which is
checked on the full periodic distance spectrum rather than on the cell alone.
"""

from __future__ import annotations

import numpy as np
import pytest

from cellstine.core.species import expand_species, group_species
from cellstine.io import native as io_mod
from cellstine.io.converters import StructureConverter
from cellstine.io.orientation import OrientationNormalizer
from cellstine.io.vasp import VaspIO

from conftest import write_poscar

TRICLINIC = np.array([[3.1, 0.4, 0.2], [-1.0, 4.2, 0.3], [0.2, -0.5, 7.7]])


@pytest.fixture
def triclinic_poscar(tmp_path):
    positions = np.array(
        [
            [0.05, 0.11, 0.23],
            [0.37, 0.52, 0.61],
            [0.71, 0.19, 0.88],
            [0.44, 0.83, 0.05],
            [0.92, 0.66, 0.47],
        ]
    )
    return str(write_poscar(tmp_path / "triclinic.vasp", TRICLINIC, ["Fe", "O"], [2, 3], positions))


def _distance_spectrum(lattice: np.ndarray, direct: np.ndarray) -> np.ndarray:
    """Return every interatomic distance out to the neighbouring cells, sorted."""

    lattice = np.asarray(lattice, dtype=float)
    cartesian = np.asarray(direct, dtype=float) @ lattice
    shifts = np.array(
        [[i, j, k] for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1)], dtype=float
    ) @ lattice
    differences = cartesian[:, None, None, :] - cartesian[None, :, None, :] - shifts[None, None, :, :]
    return np.sort(np.linalg.norm(differences, axis=3).ravel())


def test_direct_and_cartesian_conversions_are_inverse():
    rng = np.random.default_rng(11)
    direct = rng.random((9, 3))
    cartesian = io_mod.direct_to_cartesian(direct, TRICLINIC)
    assert np.allclose(io_mod.cartesian_to_direct(cartesian, TRICLINIC), direct, atol=1e-13)


def test_wrapping_keeps_positions_in_the_cell_and_preserves_sites():
    rng = np.random.default_rng(12)
    direct = rng.random((9, 3)) * 6.0 - 3.0
    wrapped = io_mod.wrap_direct(direct)
    assert np.all(wrapped >= 0.0) and np.all(wrapped < 1.0)
    assert np.allclose(wrapped - direct, np.round(wrapped - direct), atol=1e-9)


def test_poscar_round_trip_is_exact(tmp_path, triclinic_poscar):
    original = io_mod.read_poscar(triclinic_poscar)
    flags = [("T", "T", "F")] * original.natoms
    io_mod.write_poscar(
        str(tmp_path / "again.vasp"),
        original.lattice,
        original.positions_direct,
        original.counts,
        original.species,
        positions_are_cartesian=False,
        selective_flags=flags,
    )
    reloaded = io_mod.read_poscar(str(tmp_path / "again.vasp"))
    assert np.allclose(reloaded.lattice, original.lattice, atol=1e-10)
    assert np.allclose(reloaded.positions_direct, original.positions_direct, atol=1e-10)
    assert reloaded.species == original.species
    assert reloaded.counts == original.counts
    assert reloaded.selective_flags == flags


def test_record_round_trip_through_the_converter(tmp_path, triclinic_poscar):
    converter = StructureConverter()
    record = converter.read(triclinic_poscar)
    VaspIO().write(record, str(tmp_path / "copy.vasp"), positions_are_cartesian=False, wrap_positions=False)
    reloaded = converter.read(str(tmp_path / "copy.vasp"))
    assert np.allclose(reloaded.lattice, record.lattice, atol=1e-10)
    assert np.allclose(reloaded.positions_direct, record.positions_direct, atol=1e-10)
    assert np.allclose(reloaded.positions_cartesian, record.positions_cartesian, atol=1e-9)
    assert reloaded.species == record.species and reloaded.counts == record.counts


@pytest.mark.parametrize("method", ["align_ab_to_xy", "align_c_to_z"])
def test_orientation_helpers_are_rigid_rotations(triclinic_poscar, method):
    record = StructureConverter().read(triclinic_poscar)
    rotated = getattr(OrientationNormalizer(), method)(record)
    before = _distance_spectrum(record.lattice, record.positions_direct)
    after = _distance_spectrum(rotated.lattice, rotated.positions_direct)
    assert np.allclose(before, after, atol=1e-12)
    assert float(np.linalg.det(rotated.lattice)) == pytest.approx(
        abs(float(np.linalg.det(record.lattice))), rel=1e-12
    )


def test_align_ab_to_xy_puts_the_surface_plane_in_xy(triclinic_poscar):
    """The slab convention: a along x, b in the plane, and c above it."""

    record = StructureConverter().read(triclinic_poscar)
    rotated = OrientationNormalizer().align_ab_to_xy(record)
    lattice = np.asarray(rotated.lattice, dtype=float)
    assert np.allclose(lattice[0, 1:], 0.0, atol=1e-12)
    assert lattice[0, 0] > 0.0
    assert abs(float(lattice[1, 2])) <= 1e-12
    assert float(lattice[2, 2]) > 0.0
    assert np.allclose(
        np.asarray(rotated.positions_cartesian, dtype=float),
        np.asarray(rotated.positions_direct, dtype=float) @ lattice,
        atol=1e-10,
    )


def test_align_ab_to_xy_is_the_exact_frame(triclinic_poscar):
    """The alignment is a rotation, and the components it zeroes are already zero.

    Mirrors ``Cellstine.alignFrame_apply_first``,
    ``Cellstine.alignFrame_apply_second_height``,
    ``Cellstine.alignFrame_apply_second_pos`` and ``Cellstine.alignFrame_height``
    in ``RequestProject/SurfaceAlignment.lean``.
    """

    normaliser = OrientationNormalizer()
    record = StructureConverter().read(triclinic_poscar)
    original = np.asarray(normaliser.ensure_right_handed(record).lattice, dtype=float)
    rotated = np.asarray(normaliser.align_ab_to_xy(record).lattice, dtype=float)

    first, second, third = original
    normal = np.cross(first, second)
    area = float(np.linalg.norm(normal))
    length = float(np.linalg.norm(first))

    x_hat = first / length
    z_hat = normal / area
    y_hat = np.cross(z_hat, x_hat)
    frame = np.column_stack((x_hat, y_hat, z_hat))
    assert np.allclose(frame.T @ frame, np.eye(3), atol=1e-12)
    assert float(np.linalg.det(frame)) == pytest.approx(1.0, abs=1e-12)

    raw = original @ frame
    # The implementation sets these three entries to zero; they are already zero.
    assert abs(float(raw[0, 1])) <= 1e-12
    assert abs(float(raw[0, 2])) <= 1e-12
    assert abs(float(raw[1, 2])) <= 1e-12
    assert np.allclose(raw, rotated, atol=1e-12)

    assert float(rotated[0, 0]) == pytest.approx(length, rel=1e-12)
    assert float(rotated[1, 0]) == pytest.approx(float(first @ second) / length, rel=1e-12)
    assert float(rotated[1, 1]) == pytest.approx(area / length, rel=1e-12)
    assert float(rotated[1, 1]) > 0.0
    assert float(rotated[2, 2]) == pytest.approx(float(normal @ third) / area, rel=1e-12)
    assert float(rotated[2, 2]) > 0.0


def test_align_ab_to_xy_rejects_a_degenerate_plane(tmp_path):
    path = write_poscar(
        tmp_path / "flat.vasp",
        np.array([[3.0, 0.0, 0.0], [6.0, 0.0, 0.0], [0.0, 0.0, 9.0]]),
        ["C"],
        [1],
        np.array([[0.0, 0.0, 0.0]]),
    )
    record = StructureConverter().read(str(path))
    with pytest.raises(ValueError):
        OrientationNormalizer().align_ab_to_xy(record)


def test_repeat_along_c_multiplies_atoms_and_height(triclinic_poscar):
    structure = io_mod.read_poscar(triclinic_poscar)
    repeated = io_mod.repeat_structure_along_c(structure, 3)
    assert repeated.natoms == 3 * structure.natoms
    assert np.allclose(repeated.lattice[2], 3.0 * np.asarray(structure.lattice[2]), atol=1e-12)
    assert np.allclose(repeated.lattice[:2], np.asarray(structure.lattice)[:2], atol=1e-12)
    original_heights = np.sort((structure.positions_direct @ structure.lattice)[:, 2])
    repeated_heights = np.sort((repeated.positions_direct @ repeated.lattice)[:, 2])
    assert np.allclose(repeated_heights[: structure.natoms], original_heights, atol=1e-9)


def _xyz(path, rows) -> str:
    lines = [str(len(rows)), "interleaved molecule"]
    lines.extend(f"{symbol} {x:.6f} {y:.6f} {z:.6f}" for symbol, (x, y, z) in rows)
    path.write_text("\n".join(lines) + "\n")
    return str(path)


def test_grouping_species_permutes_the_atoms_as_well_as_counting_them():
    order, counts, atom_order = group_species(["C", "O", "C", "O", "H"])
    assert order == ["C", "O", "H"]
    assert counts == [2, 2, 1]
    assert atom_order == [0, 2, 1, 3, 4]
    expanded = expand_species(order, counts)
    assert [["C", "O", "C", "O", "H"][index] for index in atom_order] == expanded


def test_reading_an_interleaved_xyz_keeps_every_atom_with_its_own_species(tmp_path):
    """A POSCAR lists all atoms of one species together, so the reader must reorder.

    Counting the species without moving the coordinates silently renamed atoms:
    a file of ``C O C O`` came back as two carbons at the carbon *and oxygen*
    positions of the first molecule.
    """

    rows = [
        ("C", (0.0, 0.0, 0.0)),
        ("O", (0.0, 0.0, 1.13)),
        ("C", (3.0, 0.0, 0.0)),
        ("O", (3.0, 0.0, 1.13)),
    ]
    record = StructureConverter().read(_xyz(tmp_path / "co.xyz", rows))
    assert record.species == ["C", "O"]
    assert record.counts == [2, 2]
    cartesian = np.asarray(record.positions_cartesian, dtype=float)
    assert expand_species(record.species, record.counts) == ["C", "C", "O", "O"]
    # Every atom keeps the species it was read with: the two carbons are the two
    # atoms at the lower height, the two oxygens the two above them.
    heights = cartesian[:, 2]
    assert heights[0] == pytest.approx(heights[1])
    assert heights[2] == pytest.approx(heights[3])
    assert heights[2] - heights[0] == pytest.approx(1.13)
    # ... and each carbon still sits directly below its own oxygen.
    assert cartesian[0, 0] == pytest.approx(cartesian[2, 0])
    assert cartesian[1, 0] == pytest.approx(cartesian[3, 0])
    assert abs(cartesian[1, 0] - cartesian[0, 0]) == pytest.approx(3.0)


def test_an_interleaved_xyz_survives_a_round_trip_through_a_poscar(tmp_path):
    rows = [
        ("C", (0.0, 0.0, 0.0)),
        ("O", (0.0, 0.0, 1.13)),
        ("H", (1.0, 0.0, 0.0)),
        ("O", (3.0, 0.0, 1.13)),
    ]
    converter = StructureConverter()
    poscar = converter.convert(_xyz(tmp_path / "mixed.xyz", rows), str(tmp_path / "POSCAR.vasp"))
    reread = converter.read(str(poscar))
    assert reread.species == ["C", "O", "H"]
    assert reread.counts == [1, 2, 1]
    labels = expand_species(reread.species, reread.counts)
    original = {symbol: [] for symbol, _ in rows}
    for symbol, position in rows:
        original[symbol].append(np.asarray(position, dtype=float))
    read_back = {symbol: [] for symbol in original}
    for label, position in zip(labels, np.asarray(reread.positions_cartesian, dtype=float)):
        read_back[label].append(position)
    # The cell origin moved when the molecule was boxed, so compare the shape of
    # each species' point set rather than its absolute coordinates.
    for symbol in original:
        first = np.array(sorted(map(tuple, original[symbol])))
        second = np.array(sorted(map(tuple, read_back[symbol])))
        assert first.shape == second.shape
        assert np.allclose(first - first.mean(axis=0), second - second.mean(axis=0), atol=1e-6)


@pytest.mark.parametrize("vacuum", (6.0, 12.0, 20.0))
def test_the_vacuum_of_a_boxed_molecule_is_a_guaranteed_clearance(tmp_path, vacuum):
    """Reading a bare XYZ pads the bounding box, and the padding is a real gap.

    Every atom must land strictly inside the box, and every atom of every
    periodic image must be at least the requested vacuum away.  Formally
    ``Cellstine.bboxShift_mem_cell`` and ``Cellstine.bbox_image_separation``.
    """

    rows = [
        ("C", (0.0, 0.0, 0.0)),
        ("O", (0.0, 0.0, 1.13)),
        ("H", (2.4, -0.7, 0.0)),
        ("H", (-1.9, 0.3, 4.2)),
    ]
    record = StructureConverter().read(_xyz(tmp_path / "clearance.xyz", rows), vacuum=vacuum)
    lattice = np.asarray(record.lattice, dtype=float)
    cartesian = np.asarray(record.positions_cartesian, dtype=float)
    direct = np.asarray(record.positions_direct, dtype=float)
    assert np.allclose(np.diag(np.diag(lattice)), lattice), "the box is orthorhombic"
    assert np.all(direct > 0.0) and np.all(direct < 1.0), "no atom needs wrapping"

    span = np.arange(-1, 2)
    shifts = np.stack(np.meshgrid(span, span, span, indexing="ij"), axis=-1).reshape(-1, 3)
    shifts = shifts[np.any(shifts != 0, axis=1)]
    closest = min(
        float(np.linalg.norm(second + shift @ lattice - first))
        for first in cartesian
        for second in cartesian
        for shift in shifts
    )
    assert closest >= vacuum - 1e-9, "the padding is the minimum image separation"
