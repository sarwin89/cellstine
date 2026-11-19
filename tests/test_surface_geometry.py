"""Adsorption sites follow physical slab geometry, including a tilted c axis."""

from itertools import product

import numpy as np
import pytest

from cellstine.adsorbate.molecule import Molecule
from cellstine.adsorbate.placement.operations import place_molecule_on_site
from cellstine.interface.surface import backend
from cellstine.io import native
from cellstine.io.models import StructureRecord
from cellstine.io.vasp import VaspIO


def _record(cell, direct, symbol="Al", flags=None):
    cell = np.asarray(cell, dtype=float)
    direct = np.asarray(direct, dtype=float)
    return StructureRecord(comment="surface geometry", lattice=cell, species=[symbol], counts=[len(direct)],
        positions_direct=direct, positions_cartesian=direct @ cell,
        selective_dynamics=flags is not None, selective_flags=flags)


def _planar_residual(delta, cell):
    # These small fixtures require at most two a/b image translations.
    shifts = np.array(list(product(range(-2, 3), repeat=2))) @ cell[:2]
    return float(np.linalg.norm(delta - shifts, axis=1).min())


@pytest.mark.parametrize("side,index", [("top", 1), ("bottom", 0)])
@pytest.mark.parametrize("tilt", [0., 1.])
@pytest.mark.parametrize("rotated", [False, True])
def test_top_sites_remain_on_actual_atoms_for_tilted_and_rotated_cells(side, index, tilt, rotated):
    cell = np.array([[2., 0., 0.], [0., 2., 0.], [tilt, 0., 12.]])
    if rotated:
        angle = .37
        rotation = np.array([[1., 0., 0.], [0., np.cos(angle), -np.sin(angle)],
                             [0., np.sin(angle), np.cos(angle)]])
        cell = cell @ rotation.T
    source = _record(cell, [[1.2, -.7, .2], [.2, .3, .6]])
    saved = source.positions_direct.copy()
    report = backend.find_adsorption_sites(source, surface_side=side)
    atop = next(site for site in report.sites if site.site_type == "top")
    assert _planar_residual(np.array(atop.cartesian) - source.positions_cartesian[index], cell) < 1e-10
    assert _planar_residual(np.array(atop.direct) @ cell - source.positions_cartesian[index], cell) < 1e-10
    assert np.array_equal(source.positions_direct, saved)


def test_oblique_layers_do_not_get_false_hcp_or_fcc_names():
    angle = np.deg2rad(70.)
    cell = np.array([[2., 0., 0.], [3. * np.cos(angle), 3. * np.sin(angle), 0.], [0., 0., 12.]])
    source = _record(cell, [[0., 0., .6], [.72425960914581, .5628725120004174, .4],
                            [.72425960914581, .5628725120004174, .2]])
    report = backend.find_adsorption_sites(source)
    assert report.average_top_layer_coordination == 2.
    assert report.site_counts == {"top": 1, "bridge": 1, "hollow": 2}


@pytest.mark.parametrize("cell", [
    [[2., 0., 0.], [0., 2., 0.], [1., 0., -12.]],
    [[2., 0., 0.], [0., -2., 0.], [1., 0., 12.]],
])
def test_planar_projection_respects_the_normal_oriented_towards_c(cell):
    source = _record(cell, [[.2, .3, .6]])
    report = backend.find_adsorption_sites(source)
    atop = next(site for site in report.sites if site.site_type == "top")
    assert _planar_residual(np.array(atop.cartesian) - source.positions_cartesian[0], source.lattice) < 1e-10


@pytest.mark.parametrize("scale", [1e-100, 1e100])
def test_planar_coordinate_conversion_avoids_cross_norm_range_loss(scale):
    cell = np.array([[2., 0., 0.], [0., 2., 0.], [1., 0., 12.]]) * scale
    result = backend._site_plane_coordinates(np.array([[.2, .3, .6]]), cell, backend.surface_normal(cell))
    assert np.allclose(result, [[.5, .3]], rtol=1e-13, atol=1e-13)


@pytest.mark.parametrize("word,counts", [("ABC", {"top": 1, "bridge": 3, "fcc_hollow": 1, "hcp_hollow": 1}),
    ("ABA", {"top": 1, "bridge": 3, "hcp_hollow": 1, "hollow": 1})])
@pytest.mark.parametrize("tilt", [0., 1.])
def test_physical_triangular_stacking_keeps_its_hollow_labels(word, counts, tilt):
    cell = np.array([[2., 0., 0.], [1., np.sqrt(3.), 0.], [tilt, 0., 12.]])
    # Construct the actual ABC/ABA Cartesian planes independently of c shear.
    cartesian = np.array([np.array([(ord(letter) - 65) / 3.] * 2) @ cell[:2] +
        [0., 0., 2.4 + 2.4 * index] for index, letter in enumerate(word)])
    source = _record(cell, cartesian @ np.linalg.inv(cell))
    assert backend.find_adsorption_sites(source).site_counts == counts


def test_buried_registry_fault_does_not_remove_correct_local_hollow_names():
    cell = np.array([[2., 0., 0.], [1., np.sqrt(3.), 0.], [0., 0., 12.]])
    # The exposed three planes are exactly ABC. The deeper D offset does not
    # change which first/second subsurface atom lies below either surface hole.
    direct = np.array([[.13, .07, .1], [0., 0., .1 + 1/6],
        [1/3, 1/3, .1 + 2/6], [2/3, 2/3, .6]])
    report = backend.find_adsorption_sites(_record(cell, direct))
    assert report.site_counts == {"top": 1, "bridge": 3, "fcc_hollow": 1, "hcp_hollow": 1}


def test_zero_layer_tolerance_does_not_regroup_the_selected_rotated_plane():
    angle = .2
    rotation = np.array([[1., 0., 0.], [0., np.cos(angle), -np.sin(angle)],
                         [0., np.sin(angle), np.cos(angle)]])
    cell = np.array([[4., 0., 0.], [2., 2. * np.sqrt(3.), 0.], [0., 0., 12.]]) @ rotation.T
    direct = np.array([[i/2 + layer/6, j/2 + layer/6, .2 + .2 * layer]
                       for layer in range(3) for i, j in product(range(2), repeat=2)])
    report = backend.find_adsorption_sites(_record(cell, direct), layer_tolerance=0.)
    assert report.detected_layer_count == 3
    assert report.site_counts == {"top": 4, "bridge": 12, "hcp_hollow": 4, "fcc_hollow": 4}


@pytest.mark.parametrize("matrix", [
    [[2, 1], [1, 2]], [[4, 1], [1, 4]], [[5, 3], [3, 4]], [[7, 2], [3, 3]],
    [[1, 100], [0, 12]], [[12, 101], [0, 1]], [[31, 32], [32, 34]],
    [[17, 19], [18, 21]], [[8, 13], [13, 23]], [[13, 21], [21, 35]],
])
def test_integer_triangular_supercells_keep_correct_site_counts(matrix):
    matrix = np.array(matrix, dtype=int)
    a, b, c, d = (int(value) for value in matrix.ravel())
    size = a * d - b * c
    adjugate = np.array([[d, -b], [-c, a]], dtype=int)
    # Exact integer quotient keys construct every occupied primitive site.
    keys = {tuple(int(value) for value in (np.array([i, j]) @ adjugate) % size)
            for i in range(size) for j in range(size)}
    assert len(keys) == size
    primitive = np.array([[1., 0., 0.], [.5, np.sqrt(3.) / 2., 0.]])
    cell = np.vstack((matrix @ primitive, [0., 0., 12.]))
    inverse = np.linalg.inv(matrix.astype(float))
    direct = np.array([[*(np.array(key) / size + np.array([layer / 3., layer / 3.]) @ inverse),
        .2 + .2 * layer] for layer in range(3) for key in sorted(keys)])
    report = backend.find_adsorption_sites(_record(cell, direct))
    assert report.site_counts == {"top": size, "bridge": 3 * size, "fcc_hollow": size, "hcp_hollow": size}
    assert report.average_top_layer_coordination == 6.


@pytest.mark.parametrize("workflow", [False, True])
def test_real_placed_atom_is_above_the_actual_tilted_cell_site(tmp_path, workflow):
    cell = np.array([[2., 0., 0.], [0., 2., 0.], [1., 0., 12.]])
    substrate = _record(cell, [[.2, .3, .6]], flags=[("F", "T", "F")])
    molecule = _record(np.eye(3) * 10., [[0., 0., 0.]], "He")
    first, second = tmp_path / "slab.vasp", tmp_path / "molecule.vasp"
    for record, path in ((substrate, first), (molecule, second)):
        VaspIO().write(record, str(path), positions_are_cartesian=False, wrap_positions=True)
    saved = [first.read_bytes(), second.read_bytes()]
    destination = tmp_path / "placed.vasp"
    if workflow:
        result = Molecule(backend="native", runs_root=tmp_path / "runs", output_root=tmp_path / "output").place(
            substrate_poscar=str(first), molecule_poscar=str(second), site_type="top", height=1.7,
            preserve_vacuum=False, output_path=str(destination))
        assert np.allclose(result.payload["site_cartesian"], [1., .6, 7.2], atol=1e-10)
    else:
        place_molecule_on_site(str(first), str(second), site_type="top", height=1.7,
            reframe_axes=None, preserve_vacuum=False, output_path=str(destination))
    output = native.read_poscar(str(destination))
    assert _planar_residual(output.positions_cartesian[1] - output.positions_cartesian[0] - [0., 0., 1.7], cell) < 1e-9
    assert output.selective_flags == [("F", "T", "F"), ("T", "T", "T")]
    assert [first.read_bytes(), second.read_bytes()] == saved
