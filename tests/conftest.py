"""Test bootstrap and shared structure fixtures."""

from __future__ import annotations

import itertools
import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "benchmarks"))


def hexagonal_basis(constant: float) -> np.ndarray:
    """Return the 2x2 Cartesian column basis of a hexagonal lattice."""

    return np.array(
        [[constant, -0.5 * constant], [0.0, 0.5 * math.sqrt(3.0) * constant]]
    )


def write_poscar(
    path: Path,
    lattice: np.ndarray,
    species: list[str],
    counts: list[int],
    positions: np.ndarray,
    *,
    comment: str = "test structure",
) -> Path:
    lines = [comment, "1.0"]
    for row in np.asarray(lattice, dtype=float):
        lines.append("  {:.10f}  {:.10f}  {:.10f}".format(*row))
    lines.append("  " + "  ".join(species))
    lines.append("  " + "  ".join(str(int(value)) for value in counts))
    lines.append("Direct")
    for row in np.asarray(positions, dtype=float):
        lines.append("  {:.10f}  {:.10f}  {:.10f}".format(*row))
    path.write_text("\n".join(lines) + "\n")
    return path


def _honeycomb(path: Path, constant: float, species: list[str], vacuum: float) -> Path:
    lattice = np.zeros((3, 3))
    lattice[:2, :2] = hexagonal_basis(constant).T
    lattice[2, 2] = vacuum
    positions = np.array([[1.0 / 3.0, 2.0 / 3.0, 0.5], [2.0 / 3.0, 1.0 / 3.0, 0.5]])
    if species[0] == species[1]:
        return write_poscar(path, lattice, [species[0]], [2], positions)
    return write_poscar(path, lattice, species, [1, 1], positions)


@pytest.fixture(scope="session")
def graphene_poscar(tmp_path_factory) -> Path:
    path = tmp_path_factory.mktemp("structures") / "graphene.vasp"
    return _honeycomb(path, 2.46, ["C", "C"], 20.0)


@pytest.fixture(scope="session")
def hbn_poscar(tmp_path_factory) -> Path:
    path = tmp_path_factory.mktemp("structures") / "hbn.vasp"
    return _honeycomb(path, 2.504, ["B", "N"], 20.0)


@pytest.fixture(scope="session")
def mos2_poscar(tmp_path_factory) -> Path:
    """Return a 1H-MoS2 monolayer with a 20 Angstrom vacuum."""

    path = tmp_path_factory.mktemp("structures") / "mos2.vasp"
    lattice = np.zeros((3, 3))
    lattice[:2, :2] = hexagonal_basis(3.16).T
    lattice[2, 2] = 20.0
    height = 1.56 / 20.0
    positions = np.array(
        [
            [1.0 / 3.0, 2.0 / 3.0, 0.5],
            [2.0 / 3.0, 1.0 / 3.0, 0.5 + height],
            [2.0 / 3.0, 1.0 / 3.0, 0.5 - height],
        ]
    )
    return write_poscar(path, lattice, ["Mo", "S"], [1, 2], positions)


@pytest.fixture(scope="session")
def silicon_poscar(tmp_path_factory) -> Path:
    """Return the two-atom primitive cell of diamond silicon."""

    path = tmp_path_factory.mktemp("structures") / "si.vasp"
    lattice = 5.43 * np.array([[0.0, 0.5, 0.5], [0.5, 0.0, 0.5], [0.5, 0.5, 0.0]])
    positions = np.array([[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]])
    return write_poscar(path, lattice, ["Si"], [2], positions)


# ---------------------------------------------------------------------------
# measuring a built layer back from its coordinates
# ---------------------------------------------------------------------------


def layer_translation_lattice(
    lattice: np.ndarray,
    fractional: np.ndarray,
    labels,
    tolerance: float = 1e-5,
) -> np.ndarray:
    """Return two independent in-plane translations of one layer of atoms.

    A displacement is a translation of the layer when moving every atom by it
    reproduces the layer, species by species, under the periodic boundary of the
    built cell.  The displacements between the atoms are the only candidates,
    and the cell vectors are translations too.  Nothing from the search that
    produced the cell is used, so this measures the structure as written.
    """

    fractional = np.mod(np.asarray(fractional, dtype=float), 1.0)
    labels = np.asarray(labels)
    accepted = [np.zeros(2)]
    for candidate in fractional - fractional[0]:
        moved = np.mod(fractional + candidate, 1.0)
        delta = moved[:, None, :] - fractional[None, :, :]
        delta -= np.rint(delta)
        distance = np.linalg.norm(delta @ lattice, axis=2)
        # Only an atom of the same species can be the image of an atom.
        distance = np.where(labels[:, None] == labels[None, :], distance, np.inf)
        if np.all(distance.min(axis=1) < tolerance):
            accepted.append((candidate @ lattice)[:2])
    cell = np.asarray(lattice, dtype=float)[:2, :2]
    shifts = np.array([[i, j] for i in (-1, 0, 1) for j in (-1, 0, 1)], dtype=float) @ cell
    vectors = (np.array(accepted)[:, None, :] + shifts[None, :, :]).reshape(-1, 2)
    vectors = vectors[np.argsort(np.linalg.norm(vectors, axis=1))]
    first = vectors[np.linalg.norm(vectors, axis=1) > 1e-6][0]
    second = next(
        vector
        for vector in vectors
        if np.linalg.norm(vector) > 1e-6
        and abs(first[0] * vector[1] - first[1] * vector[0]) > 1e-6
    )
    return np.array([first, second])


def unimodular_correspondences(bound: int = 2) -> list[np.ndarray]:
    """Return the integer changes of basis with entries in ``[-bound, bound]``."""

    return [
        np.array(entries, dtype=float).reshape(2, 2)
        for entries in itertools.product(range(-bound, bound + 1), repeat=4)
        if abs(round(entries[0] * entries[3] - entries[1] * entries[2])) == 1
    ]


def rotation_and_log_strain(
    measured: np.ndarray, pristine: np.ndarray, *, bound: int = 2
) -> tuple[float, np.ndarray]:
    """Rotation in degrees and principal log strains from ``pristine`` to ``measured``.

    The correspondence between the measured basis and the pristine one is fixed
    by taking the one that needs the least strain; only orientation-preserving
    deformations are allowed, so the rotation is well defined modulo the point
    group of the pristine lattice.
    """

    best: tuple[float, float, np.ndarray] | None = None
    for correspondence in unimodular_correspondences(bound):
        gradient = np.linalg.solve(correspondence @ pristine, measured)
        if np.linalg.det(gradient) <= 0.0:
            continue
        left, singular, right = np.linalg.svd(gradient)
        strain = np.sort(np.log(singular))
        score = float(np.max(np.abs(strain)))
        if best is None or score < best[0]:
            rotation = left @ right
            angle = math.degrees(math.atan2(rotation[1, 0], rotation[0, 0]))
            best = (score, angle, strain)
    assert best is not None
    return best[1], best[2]


@pytest.fixture(scope="session")
def carbon_monoxide_poscar(tmp_path_factory) -> Path:
    """Return a CO molecule in a box, aligned with the Cartesian z axis."""

    path = tmp_path_factory.mktemp("structures") / "co.vasp"
    lattice = np.diag([12.0, 12.0, 12.0])
    positions = np.array([[0.5, 0.5, 0.5], [0.5, 0.5, 0.5 + 1.128 / 12.0]])
    return write_poscar(path, lattice, ["C", "O"], [1, 1], positions)
