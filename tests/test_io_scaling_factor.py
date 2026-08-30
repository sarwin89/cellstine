"""The second line of a POSCAR is a scaling factor, in all of VASP's forms.

It is a linear map of space, so whatever form it takes it has to act on the cell
and on Cartesian positions alike and leave fractional coordinates untouched:

* a positive number multiplies the cell;
* a **negative** number is not a multiplier at all -- its magnitude is the
  volume the cell is to have, so the cell keeps its shape and its handedness and
  comes out with exactly that volume;
* three numbers scale the three Cartesian components.

Reading a file written in one of these forms and the same structure written in
another must give the same crystal, which is what the tests below check.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from cellstine.io import native as io_mod

LATTICE = np.array([[3.0, 0.0, 0.0], [0.5, 3.2, 0.0], [0.2, -0.4, 4.1]])
DIRECT = np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.0], [0.25, 0.1, 0.5]])
SPECIES = ["Al", "Cu"]
COUNTS = [2, 1]


def _poscar(path, scale_line: str, lattice: np.ndarray, positions: np.ndarray, mode: str) -> str:
    lines = ["scaling factor test", scale_line]
    for row in np.asarray(lattice, dtype=float):
        lines.append("  {:.12f}  {:.12f}  {:.12f}".format(*row))
    lines.append("  " + "  ".join(SPECIES))
    lines.append("  " + "  ".join(str(value) for value in COUNTS))
    lines.append(mode)
    for row in np.asarray(positions, dtype=float):
        lines.append("  {:.12f}  {:.12f}  {:.12f}".format(*row))
    path.write_text("\n".join(lines) + "\n")
    return str(path)


def test_a_positive_factor_multiplies_the_cell(tmp_path):
    path = _poscar(tmp_path / "POSCAR", "2.5", LATTICE, DIRECT, "Direct")
    data = io_mod.read_poscar(path)
    assert np.allclose(data.lattice, 2.5 * LATTICE)
    assert np.allclose(data.positions_direct, DIRECT)
    assert np.allclose(data.positions_cartesian, DIRECT @ (2.5 * LATTICE))


def test_a_positive_factor_also_scales_cartesian_positions(tmp_path):
    cartesian = DIRECT @ LATTICE
    direct_file = _poscar(tmp_path / "POSCAR_direct", "2.5", LATTICE, DIRECT, "Direct")
    cartesian_file = _poscar(tmp_path / "POSCAR_cart", "2.5", LATTICE, cartesian, "Cartesian")
    from_direct = io_mod.read_poscar(direct_file)
    from_cartesian = io_mod.read_poscar(cartesian_file)
    assert np.allclose(from_direct.lattice, from_cartesian.lattice)
    assert np.allclose(from_direct.positions_direct, from_cartesian.positions_direct)
    assert np.allclose(from_direct.positions_cartesian, from_cartesian.positions_cartesian)


def test_a_negative_factor_is_the_volume_the_cell_is_to_have(tmp_path):
    target = 100.0
    path = _poscar(tmp_path / "POSCAR", f"-{target}", LATTICE, DIRECT, "Direct")
    data = io_mod.read_poscar(path)
    volume = float(np.linalg.det(np.asarray(data.lattice, dtype=float)))
    assert volume == pytest.approx(target, rel=1e-12)
    # The cell keeps its shape: every vector grew by the same factor, and the
    # cell is still right-handed rather than turned inside out.
    factor = (target / abs(float(np.linalg.det(LATTICE)))) ** (1.0 / 3.0)
    assert np.allclose(data.lattice, factor * LATTICE)
    assert volume > 0.0
    assert np.allclose(data.positions_direct, DIRECT)


def test_a_negative_factor_scales_cartesian_positions_too(tmp_path):
    target = 100.0
    cartesian = DIRECT @ LATTICE
    path = _poscar(tmp_path / "POSCAR", f"-{target}", LATTICE, cartesian, "Cartesian")
    data = io_mod.read_poscar(path)
    assert np.allclose(data.positions_direct, DIRECT)


def test_three_factors_scale_the_three_cartesian_components(tmp_path):
    factors = np.array([1.5, 2.0, 0.5])
    path = _poscar(tmp_path / "POSCAR", "1.5 2.0 0.5", LATTICE, DIRECT, "Direct")
    data = io_mod.read_poscar(path)
    assert np.allclose(data.lattice, LATTICE * factors[None, :])
    assert np.allclose(data.positions_direct, DIRECT)


def test_three_factors_agree_between_direct_and_cartesian(tmp_path):
    cartesian = DIRECT @ LATTICE
    direct_file = _poscar(tmp_path / "POSCAR_direct", "1.5 2.0 0.5", LATTICE, DIRECT, "Direct")
    cartesian_file = _poscar(tmp_path / "POSCAR_cart", "1.5 2.0 0.5", LATTICE, cartesian, "Cartesian")
    from_direct = io_mod.read_poscar(direct_file)
    from_cartesian = io_mod.read_poscar(cartesian_file)
    assert np.allclose(from_direct.lattice, from_cartesian.lattice)
    assert np.allclose(from_direct.positions_direct, from_cartesian.positions_direct)


def test_a_zero_factor_is_refused(tmp_path):
    path = _poscar(tmp_path / "POSCAR", "0.0", LATTICE, DIRECT, "Direct")
    with pytest.raises(ValueError):
        io_mod.read_poscar(path)


def test_a_scaled_cell_survives_a_round_trip(tmp_path):
    path = _poscar(tmp_path / "POSCAR", "-100.0", LATTICE, DIRECT, "Direct")
    data = io_mod.read_poscar(path)
    out = tmp_path / "POSCAR_out"
    io_mod.write_poscar(
        str(out),
        data.lattice,
        data.positions_direct,
        data.counts,
        data.species,
        positions_are_cartesian=False,
        wrap_positions=False,
    )
    reloaded = io_mod.read_poscar(str(out))
    assert np.allclose(reloaded.lattice, data.lattice)
    assert np.allclose(reloaded.positions_direct, data.positions_direct)
    assert float(np.linalg.det(np.asarray(reloaded.lattice, dtype=float))) == pytest.approx(100.0)


def test_bond_lengths_follow_the_scaled_cell(tmp_path):
    """A physical quantity, not just the numbers on the page."""

    plain = io_mod.read_poscar(_poscar(tmp_path / "POSCAR_one", "1.0", LATTICE, DIRECT, "Direct"))
    scaled = io_mod.read_poscar(_poscar(tmp_path / "POSCAR_two", "2.0", LATTICE, DIRECT, "Direct"))
    first = np.asarray(plain.positions_cartesian, dtype=float)
    second = np.asarray(scaled.positions_cartesian, dtype=float)
    for index in range(1, len(first)):
        short = math.dist(first[0], first[index])
        long = math.dist(second[0], second[index])
        assert long == pytest.approx(2.0 * short)


def test_a_negative_factor_keeps_a_left_handed_cell_left_handed(tmp_path):
    """The factor is a positive number, so it cannot change the sign of the cell.

    Formally ``Cellstine.abs_det_volumeScale_smul`` and
    ``Cellstine.det_volumeScale_smul_pos_iff``; reading the number as a plain
    multiplier instead would flip the sign, which is
    ``Cellstine.det_smul_neg_of_neg``.
    """

    mirrored = np.array([LATTICE[1], LATTICE[0], LATTICE[2]])
    assert float(np.linalg.det(mirrored)) < 0.0
    path = _poscar(tmp_path / "POSCAR", "-100.0", mirrored, DIRECT, "Direct")
    data = io_mod.read_poscar(path)
    determinant = float(np.linalg.det(np.asarray(data.lattice, dtype=float)))
    assert determinant < 0.0, "the handedness of the cell is preserved"
    assert abs(determinant) == pytest.approx(100.0, rel=1e-12)
    assert np.allclose(data.positions_direct, DIRECT)


def test_wrapping_positions_moves_atoms_by_whole_lattice_vectors(tmp_path):
    """Wrapping is a relabelling of the same crystal, not a new structure.

    Formally ``Cellstine.wrap_mem_Ico`` and ``Cellstine.wrap_sub_mem_lattice``.
    """

    outside = np.array([[1.25, -0.5, 0.75], [-2.1, 3.4, 0.0], [0.5, 0.5, -1.5]])
    path = _poscar(tmp_path / "POSCAR", "1.0", LATTICE, outside, "Direct")
    data = io_mod.read_poscar(path)
    out = tmp_path / "POSCAR_wrapped"
    io_mod.write_poscar(
        str(out),
        data.lattice,
        data.positions_direct,
        data.counts,
        data.species,
        positions_are_cartesian=False,
        wrap_positions=True,
    )
    wrapped = io_mod.read_poscar(str(out)).positions_direct
    assert np.all(wrapped >= 0.0) and np.all(wrapped < 1.0)
    shifts = np.asarray(outside, dtype=float) - np.asarray(wrapped, dtype=float)
    assert np.allclose(shifts, np.rint(shifts), atol=1e-9), "each atom moved by a lattice vector"
