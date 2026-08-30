"""Cross-check and time the native Gram search against a brute-force reference.

Run from the repository root::

    python benchmarks/benchmark_gram_search.py

For each test system and each length bound the script

1. runs the independent brute-force reference of
   :mod:`benchmarks.reference_moire`,
2. runs the native Gram engine,
3. compares the two sets of physical candidate classes -- twist angle, per-layer
   atom counts, relative principal strains and supercell area -- and stops on
   the first mismatch,
4. prints measured wall-clock timings.  Timings are host-dependent
   measurements, not performance assertions.
"""

from __future__ import annotations

import importlib.util
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = ROOT / "src" / "cellstine"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "benchmarks"))


def _load_cellstine():
    """Import the repository as the ``cellstine`` package without installation."""

    if "cellstine" in sys.modules:
        return sys.modules["cellstine"]
    spec = importlib.util.spec_from_file_location(
        "cellstine", PACKAGE_ROOT / "__init__.py", submodule_search_locations=[str(PACKAGE_ROOT)]
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["cellstine"] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


_load_cellstine()

from cellstine.core.symmetry2d import (  # noqa: E402
    lattice_point_group,
    layer_point_group,
)
from cellstine.moire.search import gram  # noqa: E402
from reference_moire import ReferenceConfig, reference_search  # noqa: E402

ANGLE_TOLERANCE_DEG = 1e-6
STRAIN_TOLERANCE = 1e-7
AREA_TOLERANCE = 1e-6


def _hexagonal(constant: float) -> np.ndarray:
    return np.array(
        [[constant, -0.5 * constant], [0.0, 0.5 * math.sqrt(3.0) * constant]]
    )


def _hexagonal_layer_group(constant: float, sublattices: int) -> np.ndarray:
    """Return the decorated point group of a honeycomb layer.

    ``sublattices == 1`` is a graphene-like layer whose two sites carry the same
    species (six-fold); ``sublattices == 2`` is an hBN-like layer with distinct
    species on the two sites (three-fold).
    """

    lattice = np.zeros((3, 3))
    lattice[:2, :2] = _hexagonal(constant).T
    lattice[2, 2] = 20.0
    positions = np.array([[1.0 / 3.0, 2.0 / 3.0, 0.5], [2.0 / 3.0, 1.0 / 3.0, 0.5]])
    species = ["C", "C"] if sublattices == 1 else ["B", "N"]
    return layer_point_group(lattice, positions, species)


def _engine_signatures(result) -> list[tuple]:
    signatures = []
    for row in range(len(result)):
        strains = np.sort(result.principal_strains[row])
        cell = gram_cell_area(result, row)
        signatures.append(
            (
                float(result.twist_degrees[row]),
                int(result.top_atom_counts[row]),
                int(result.bottom_atom_counts[row]),
                float(strains[0]),
                float(strains[1]),
                cell,
            )
        )
    return signatures


def gram_cell_area(result, row: int) -> float:
    """Return the top supercell area implied by the reported Gram triple."""

    g11, g12, g22 = result.top_gram[row]
    return float(math.sqrt(max(g11 * g22 - g12 * g12, 0.0)))


def _matches(left: tuple, right: tuple) -> bool:
    return (
        abs(left[0] - right[0]) <= ANGLE_TOLERANCE_DEG
        and left[1] == right[1]
        and left[2] == right[2]
        and abs(left[3] - right[3]) <= STRAIN_TOLERANCE
        and abs(left[4] - right[4]) <= STRAIN_TOLERANCE
        and abs(left[5] - right[5]) <= AREA_TOLERANCE
    )


def _missing(wanted: list[tuple], available: list[tuple]) -> list[tuple]:
    return [
        item for item in wanted if not any(_matches(item, other) for other in available)
    ]


def compare(name: str, basis_top, basis_bottom, group_top, group_bottom, *, atoms_top,
            atoms_bottom, max_length: float, budget: float) -> None:
    reference_config = ReferenceConfig(
        top_basis=basis_top,
        bottom_basis=basis_bottom,
        max_length=max_length,
        top_strain=0.5 * budget,
        bottom_strain=0.5 * budget,
        top_atoms=atoms_top,
        bottom_atoms=atoms_bottom,
        top_group=group_top,
        bottom_group=group_bottom,
    )
    started = time.perf_counter()
    reference = reference_search(reference_config)
    reference_seconds = time.perf_counter() - started

    engine_config = gram.SearchConfig(
        top_basis=basis_top,
        bottom_basis=basis_bottom,
        max_length=max_length,
        top_strain=0.5 * budget,
        bottom_strain=0.5 * budget,
        top_atoms=atoms_top,
        bottom_atoms=atoms_bottom,
        top_group=group_top,
        bottom_group=group_bottom,
    )
    started = time.perf_counter()
    engine = gram.search(engine_config)
    engine_seconds = time.perf_counter() - started

    reference_signatures = [item.signature() for item in reference]
    engine_signatures = _engine_signatures(engine)
    lost = _missing(reference_signatures, engine_signatures)
    spurious = _missing(engine_signatures, reference_signatures)
    if lost or spurious:
        print(f"MISMATCH for {name} at max length {max_length:g} Angstrom")
        for item in lost[:5]:
            print(f"  reference class missing from the engine: {item}")
        for item in spurious[:5]:
            print(f"  engine class absent from the reference:  {item}")
        raise SystemExit(1)

    speedup = reference_seconds / engine_seconds if engine_seconds > 0 else float("inf")
    print(
        f"{name:22s} L<={max_length:5.1f} A  "
        f"reference {len(reference_signatures):4d} classes in {reference_seconds:7.3f} s  |  "
        f"engine {len(engine_signatures):4d} classes in {engine_seconds:7.3f} s  "
        f"({speedup:6.1f}x)"
    )


def main() -> int:
    graphene = _hexagonal(2.46)
    hbn = _hexagonal(2.504)
    graphene_group = _hexagonal_layer_group(2.46, 1)
    hbn_group = _hexagonal_layer_group(2.504, 2)
    square = np.array([[3.9, 0.0], [0.0, 3.9]])
    square_group = lattice_point_group(square)

    print("Cross-check of the native Gram search against the brute-force reference.")
    print("Timings are measurements on this host, not performance guarantees.\n")
    for max_length in (8.0, 12.0, 16.0):
        compare(
            "graphene / graphene",
            graphene,
            graphene,
            graphene_group,
            graphene_group,
            atoms_top=2,
            atoms_bottom=2,
            max_length=max_length,
            budget=0.02,
        )
    for max_length in (8.0, 12.0, 16.0):
        compare(
            "graphene / hBN",
            graphene,
            hbn,
            graphene_group,
            hbn_group,
            atoms_top=2,
            atoms_bottom=2,
            max_length=max_length,
            budget=0.02,
        )
    for max_length in (8.0, 12.0, 16.0):
        compare(
            "square / square",
            square,
            square,
            square_group,
            square_group,
            atoms_top=1,
            atoms_bottom=1,
            max_length=max_length,
            budget=0.02,
        )
    print("\nAll compared systems agree class for class.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
