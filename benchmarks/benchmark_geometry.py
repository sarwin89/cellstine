"""Cross-check and time the periodic geometry core against brute force.

Run from the repository root::

    python benchmarks/benchmark_geometry.py

For each test cell the script

1. checks the fast paths of :mod:`cellstine.core.geometry` -- exact minimum
   images, cell-list neighbour images, nearest-point distances and periodic site
   matching -- against exhaustive reference computations, and stops on the first
   mismatch,
2. prints measured wall-clock timings for those routines and for the two heavy
   consumers, ``analyse_symmetry`` and ``find_void_sites``.

Timings are host-dependent measurements, not performance assertions.  The
correctness checks are assertions and the script exits non-zero if any fails.
"""

from __future__ import annotations

import importlib.util
import itertools
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))


def _load_cellstine():
    """Import the repository as the ``cellstine`` package without installation."""

    if "cellstine" in sys.modules:
        return sys.modules["cellstine"]
    spec = importlib.util.spec_from_file_location(
        "cellstine", ROOT / "__init__.py", submodule_search_locations=[str(ROOT)]
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["cellstine"] = module
    spec.loader.exec_module(module)
    return module


_load_cellstine()

from cellstine.core import geometry  # noqa: E402
from cellstine.core import symmetry3d  # noqa: E402
from cellstine.core import voids  # noqa: E402


# ---------------------------------------------------------------------------
# test cells
# ---------------------------------------------------------------------------


def si_supercell(n: int):
    """Return an ``n x n x n`` supercell of cubic diamond silicon."""

    a = 5.43
    lattice = np.eye(3) * a * n
    base = np.array(
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
    )
    shifts = np.array(
        [[i, j, k] for i in range(n) for j in range(n) for k in range(n)], dtype=float
    )
    positions = ((base[None, :, :] + shifts[:, None, :]) / n).reshape(-1, 3)
    return lattice, positions, ["Si"] * len(positions)


def graphene_supercell(n: int):
    """Return an ``n x n`` graphene sheet in a hexagonal cell with vacuum."""

    a = 2.46
    lattice = np.array(
        [[a * n, 0.0, 0.0], [-0.5 * a * n, np.sqrt(3.0) / 2.0 * a * n, 0.0], [0.0, 0.0, 15.0]]
    )
    base = np.array([[0.0, 0.0, 0.5], [1.0 / 3.0, 2.0 / 3.0, 0.5]])
    cells = np.array([[i, j, 0.0] for i in range(n) for j in range(n)], dtype=float)
    positions = base[None, :, :] + cells[:, None, :] / np.array([1.0, 1.0, 1.0])
    positions = positions.reshape(-1, 3)
    positions[:, 0] /= n
    positions[:, 1] /= n
    return lattice, positions, ["C"] * len(positions)


def sheared_cell(rng: np.random.Generator, count: int):
    """Return a strongly sheared triclinic cell with random sites."""

    lattice = np.array([[4.1, 0.0, 0.0], [3.3, 2.9, 0.0], [2.7, 1.9, 3.4]])
    return lattice, rng.random((count, 3)), ["X"] * count


CELLS = {
    "Si 1x1x1": si_supercell(1),
    "graphene 3x3": graphene_supercell(3),
    "sheared triclinic": sheared_cell(np.random.default_rng(20240824), 24),
}


# ---------------------------------------------------------------------------
# references
# ---------------------------------------------------------------------------


def reference_minimum_image_distances(lattice, deltas, reach=4):
    """Shortest image length of each fractional displacement, by enumeration."""

    shifts = np.array(list(itertools.product(range(-reach, reach + 1), repeat=3)), dtype=float)
    cartesian = (deltas[:, None, :] - shifts[None, :, :]) @ lattice
    return np.sqrt(np.einsum("ijk,ijk->ij", cartesian, cartesian)).min(axis=1)


def reference_neighbour_counts(lattice, positions, cutoff, reach=4):
    """Number of neighbour images within ``cutoff`` of each atom, by enumeration."""

    shifts = np.array(list(itertools.product(range(-reach, reach + 1), repeat=3)), dtype=float)
    images = (positions[:, None, :] + shifts[None, :, :]).reshape(-1, 3) @ lattice
    base = positions @ lattice
    deltas = images[None, :, :] - base[:, None, :]
    distances = np.sqrt(np.einsum("ijk,ijk->ij", deltas, deltas))
    return (distances <= cutoff).sum(axis=1)


def timed(label, fn, *args, **kwargs):
    started = time.perf_counter()
    result = fn(*args, **kwargs)
    print(f"  {label:52s} {time.perf_counter() - started:8.3f} s")
    return result


# ---------------------------------------------------------------------------
# checks
# ---------------------------------------------------------------------------


def check_minimum_image(name, lattice, positions) -> None:
    deltas = positions[:, None, :] - positions[None, :, :]
    deltas = deltas.reshape(-1, 3)
    fast = geometry.minimum_image_distances(lattice, deltas)
    slow = reference_minimum_image_distances(lattice, deltas)
    worst = float(np.max(np.abs(fast - slow)))
    assert worst < 1e-9, f"{name}: minimum image differs from brute force by {worst:.3e} A"
    naive = deltas - np.rint(deltas)
    naive = np.linalg.norm(naive @ lattice, axis=1)
    gap = float(np.max(naive - slow))
    print(f"  minimum image exact (worst rounding error avoided: {gap:8.4f} A)")


def check_neighbour_images(name, lattice, positions, cutoff) -> None:
    _, indices, valid = geometry.neighbour_images(lattice, positions, cutoff)
    fast = valid.sum(axis=1)
    slow = reference_neighbour_counts(lattice, positions, cutoff)
    assert np.array_equal(fast, slow), f"{name}: neighbour counts differ from brute force"
    print(f"  neighbour images complete at {cutoff:.1f} A ({int(fast.sum())} pairs)")


def check_nearest_points(name, lattice, positions, rng) -> None:
    points = positions @ lattice
    queries = rng.random((64, 3)) @ lattice
    fast = geometry.nearest_point_distances(queries, points)
    deltas = queries[:, None, :] - points[None, :, :]
    slow = np.sqrt(np.einsum("ijk,ijk->ij", deltas, deltas)).min(axis=1)
    worst = float(np.max(np.abs(fast - slow)))
    assert worst < 1e-9, f"{name}: nearest-point distances differ by {worst:.3e} A"
    print("  nearest-point distances exact")


def check_site_index(name, lattice, positions, rng) -> None:
    index = geometry.PeriodicSiteIndex(lattice, positions, tolerance=1e-4)
    jitter = (rng.random(positions.shape) - 0.5) * 1e-6
    shifted = positions + jitter + rng.integers(-2, 3, positions.shape)
    matched = index.match(shifted)
    assert np.array_equal(matched, np.arange(len(positions))), f"{name}: site index missed a site"
    print("  periodic site index matches every jittered image")


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------


def main() -> int:
    rng = np.random.default_rng(7)
    for name, (lattice, positions, _species) in CELLS.items():
        print(f"{name} ({len(positions)} sites)")
        check_minimum_image(name, lattice, positions)
        check_neighbour_images(name, lattice, positions, 4.0)
        check_nearest_points(name, lattice, positions, rng)
        check_site_index(name, lattice, positions, rng)
        print()

    print("timings")
    for n in (1, 2, 3, 4):
        lattice, positions, species = si_supercell(n)
        timed(
            f"analyse_symmetry Si {len(positions):4d} atoms",
            symmetry3d.analyse_symmetry,
            lattice,
            positions,
            species,
        )
    lattice, positions, _species = si_supercell(2)
    timed(
        f"find_void_sites Si {len(positions)} atoms",
        voids.find_void_sites,
        lattice,
        positions,
    )
    # A disordered cell is the hard case for the void search: the covering
    # radius is large next to a short packing distance, so the neighbour cutoff
    # -- and with it the vertex enumeration -- is set entirely by how tight the
    # covering-radius bound is.
    lattice, positions, _species = sheared_cell(np.random.default_rng(13), 40)
    timed(
        f"find_void_sites disordered {len(positions)} atoms",
        voids.find_void_sites,
        lattice,
        positions,
    )
    lattice, positions, _species = si_supercell(3)
    timed(
        f"neighbour_images Si {len(positions)} atoms, 6 A",
        geometry.neighbour_images,
        lattice,
        positions,
        6.0,
    )
    print("\nall cross-checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
