"""Checks on the native lattice point-group search.

Each test mirrors a machine-checked statement in
``RequestProject/LatticeAutomorphisms.lean``: metric preservation is the
definition of a lattice automorphism and is the same thing as orthogonality of
the induced Cartesian map, the entrywise column test is that condition, the
enumeration box the search ranges over is complete, and searching in a reduced
basis and conjugating back gives the same group.
"""

from __future__ import annotations

import itertools
import math

import numpy as np
import pytest

from cellstine.core import symmetry3d


CUBIC = np.eye(3) * 3.2
TETRAGONAL = np.diag([3.0, 3.0, 5.1])
HEXAGONAL = np.array([[2.5, 0.0, 0.0], [-1.25, 2.5 * math.sqrt(3.0) / 2.0, 0.0], [0.0, 0.0, 4.1]])
TRICLINIC = np.array([[3.1, 0.0, 0.0], [0.7, 3.9, 0.0], [0.4, 0.9, 4.7]])
SKEWED_CUBIC = np.array([[3.2, 0.0, 0.0], [3.2, 3.2, 0.0], [3.2, 3.2, 3.2]])


def _metric(lattice: np.ndarray) -> np.ndarray:
    array = np.asarray(lattice, dtype=float)
    return array @ array.T


@pytest.mark.parametrize(
    "lattice, order",
    [(CUBIC, 48), (SKEWED_CUBIC, 48), (TETRAGONAL, 16), (HEXAGONAL, 24), (TRICLINIC, 2)],
)
def test_every_operation_preserves_the_metric_and_is_unimodular(lattice, order):
    """``Cellstine.PreservesGram`` and ``Cellstine.sq_det_eq_one_of_preservesGram``."""

    metric = _metric(lattice)
    group = symmetry3d.lattice_point_group(lattice)
    assert len(group) == order
    for element in group:
        assert np.allclose(element.T @ metric @ element, metric, atol=1e-9)
        assert abs(round(float(np.linalg.det(element.astype(float))))) == 1


@pytest.mark.parametrize("lattice", [CUBIC, SKEWED_CUBIC, HEXAGONAL, TRICLINIC])
def test_the_induced_cartesian_map_is_orthogonal(lattice):
    """``Cellstine.preservesGram_iff_orthogonal``."""

    columns = np.asarray(lattice, dtype=float).T
    inverse = np.linalg.inv(columns)
    for element in symmetry3d.lattice_point_group(lattice):
        cartesian = columns @ element.astype(float) @ inverse
        assert np.allclose(cartesian.T @ cartesian, np.eye(3), atol=1e-9)


@pytest.mark.parametrize("lattice", [CUBIC, HEXAGONAL, TRICLINIC])
def test_the_column_test_is_the_metric_test(lattice):
    """``Cellstine.preservesGram_iff_columns``."""

    metric = _metric(lattice)
    for element in symmetry3d.lattice_point_group(lattice):
        for i in range(3):
            for j in range(3):
                left = element[:, i].astype(float)
                right = element[:, j].astype(float)
                assert float(left @ metric @ right) == pytest.approx(metric[i, j], abs=1e-9)


@pytest.mark.parametrize("lattice", [CUBIC, TETRAGONAL, HEXAGONAL, TRICLINIC, SKEWED_CUBIC])
def test_the_enumeration_box_holds_every_short_vector(lattice):
    """``Cellstine.abs_coord_le_sqrt_gram_inv_diag``.

    The search ranges over the box ``|n_i| <= sqrt(r * (G^-1)_ii)``.  No integer
    vector of squared metric length at most ``r`` lies outside it, checked here
    against a much wider brute-force box.
    """

    metric = _metric(lattice)
    radius = float(np.max(np.diag(metric)))
    limits = np.sqrt(radius * np.diag(np.linalg.inv(metric)))
    wide = range(-8, 9)
    for vector in itertools.product(wide, repeat=3):
        candidate = np.asarray(vector, dtype=float)
        if float(candidate @ metric @ candidate) <= radius + 1e-9:
            assert np.all(np.abs(candidate) <= limits + 1e-9)


def test_the_search_finds_every_automorphism_of_a_cubic_lattice():
    """Completeness, checked against an exhaustive scan.

    Every automorphism of a cubic lattice has entries in ``{-1, 0, 1}``, so all
    ``3**9`` integer matrices can be tested directly; the search must return
    exactly the ones that preserve the metric.
    """

    metric = _metric(CUBIC)
    exhaustive = set()
    for entries in itertools.product((-1, 0, 1), repeat=9):
        candidate = np.asarray(entries, dtype=np.int64).reshape(3, 3)
        if np.allclose(candidate.T @ metric @ candidate, metric, atol=1e-9):
            exhaustive.add(entries)

    found = {tuple(int(value) for value in element.ravel())
             for element in symmetry3d.lattice_point_group(CUBIC)}
    assert found == exhaustive


@pytest.mark.parametrize(
    "transform",
    [
        [[1, 0, 0], [1, 1, 0], [0, 0, 1]],
        [[2, 1, 0], [1, 1, 0], [0, 0, 1]],
        [[1, 0, 1], [0, 1, 0], [0, 1, 1]],
    ],
)
@pytest.mark.parametrize("lattice", [CUBIC, HEXAGONAL, TRICLINIC])
def test_a_change_of_basis_conjugates_the_point_group(lattice, transform):
    """``Cellstine.preservesGram_conj`` and ``Cellstine.gram_mul``.

    Searching in a reduced basis and conjugating the answer back is exactly what
    the implementation does, and it returns the same group.
    """

    change = np.asarray(transform, dtype=np.int64)
    assert abs(round(float(np.linalg.det(change.astype(float))))) == 1
    inverse = np.rint(np.linalg.inv(change.astype(float))).astype(np.int64)

    changed_lattice = change.astype(float) @ np.asarray(lattice, dtype=float)
    assert np.allclose(_metric(changed_lattice), change @ _metric(lattice) @ change.T, atol=1e-9)

    original = {tuple(int(value) for value in element.ravel())
                for element in symmetry3d.lattice_point_group(lattice)}
    conjugated = set()
    for element in symmetry3d.lattice_point_group(changed_lattice):
        mapped = change.T @ element @ inverse.T
        conjugated.add(tuple(int(value) for value in mapped.ravel()))
    assert conjugated == original


@pytest.mark.parametrize("lattice", [CUBIC, HEXAGONAL, TRICLINIC])
def test_the_point_group_is_closed_under_products_and_inverses(lattice):
    """``Cellstine.preservesGram_one``, ``_mul`` and ``_inv``."""

    group = symmetry3d.lattice_point_group(lattice)
    elements = {tuple(int(value) for value in element.ravel()) for element in group}
    assert tuple(int(value) for value in np.eye(3, dtype=np.int64).ravel()) in elements
    for left in group:
        inverse = np.rint(np.linalg.inv(left.astype(float))).astype(np.int64)
        assert tuple(int(value) for value in inverse.ravel()) in elements
        for right in group:
            product = left @ right
            assert tuple(int(value) for value in product.ravel()) in elements


# ---------------------------------------------------------------------------
# the module split
# ---------------------------------------------------------------------------


def test_point_group_module_is_importable_on_its_own():
    """``core.pointgroup3d`` must not need ``core.symmetry3d`` to be loaded."""

    import subprocess
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[1] / "src" / "cellstine"
    script = (
        "import importlib.util, sys;"
        f"root = {str(root)!r};"
        "spec = importlib.util.spec_from_file_location("
        "'cellstine', root + '/__init__.py', submodule_search_locations=[root]);"
        "module = importlib.util.module_from_spec(spec);"
        "sys.modules['cellstine'] = module; spec.loader.exec_module(module);"
        "import importlib;"
        "pointgroup = importlib.import_module('cellstine.core.pointgroup3d');"
        "assert 'cellstine.core.symmetry3d' not in sys.modules;"
        "import numpy as np;"
        "print(len(pointgroup.lattice_point_group(np.eye(3) * 3.2)))"
    )
    completed = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, cwd=str(root.parent)
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "48"


@pytest.mark.parametrize(
    "name",
    ["lattice_point_group", "rotation_type", "point_group_symbol", "crystal_system_of_point_group"],
)
def test_symmetry3d_reexports_the_same_objects(name):
    """The names moved out of ``symmetry3d`` are still reachable through it."""

    from cellstine.core import pointgroup3d

    assert getattr(symmetry3d, name) is getattr(pointgroup3d, name)
    assert name in pointgroup3d.__all__
