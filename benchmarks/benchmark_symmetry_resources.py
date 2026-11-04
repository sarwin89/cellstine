"""Compare complete symmetry outputs and resource use against frozen Git code."""

from __future__ import annotations

import argparse
from dataclasses import asdict, is_dataclass
import functools
import hashlib
import itertools
import json
import platform
from pathlib import Path
import subprocess
import sys
import types

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from benchmark_geometry_memory import _measure  # noqa: E402
from cellstine.core import pointgroup3d, symmetry2d, symmetry3d  # noqa: E402


def _baseline(ref):
    commit = subprocess.check_output(
        ["git", "rev-parse", "--verify", "--end-of-options", ref + "^{commit}"],
        cwd=ROOT, text=True, encoding="utf-8",
    ).strip()
    modules, hashes = {}, {}
    for name in ("symmetry2d", "pointgroup3d", "symmetry3d"):
        path = f"src/cellstine/core/{name}.py"
        source = subprocess.check_output(
            ["git", "show", commit + ":" + path], cwd=ROOT, text=True, encoding="utf-8",
        )
        module = types.ModuleType("cellstine.core._baseline_" + name)
        module.__package__ = "cellstine.core"
        sys.modules[module.__name__] = module
        exec(compile(source, commit + ":" + path, "exec"), module.__dict__)
        modules[name] = module
        hashes[path] = hashlib.sha256(source.encode("utf-8")).hexdigest()
    for name in ("lattice_point_group", "point_group_symbol", "rotation_type", "crystal_system_of_point_group"):
        setattr(modules["symmetry3d"], name, getattr(modules["pointgroup3d"], name))
    return commit, modules, hashes


def _equal(first, second):
    if isinstance(first, np.ndarray):
        return isinstance(second, np.ndarray) and np.array_equal(first, second)
    if is_dataclass(first):
        return is_dataclass(second) and _equal(asdict(first), asdict(second))
    if isinstance(first, dict):
        return first.keys() == second.keys() and all(_equal(first[key], second[key]) for key in first)
    if isinstance(first, (tuple, list)):
        return len(first) == len(second) and all(_equal(a, b) for a, b in zip(first, second))
    return first == second


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", default="1a2c3bb57039668333740915a824c653b455c313")
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args(argv)
    if args.repeats < 1:
        parser.error("repeats must be positive")
    commit, old, baseline_hashes = _baseline(args.baseline)
    grid = np.array(list(itertools.product(np.arange(32) / 32., repeat=2)))
    lattice = 6. * np.eye(3)
    positions = np.array(list(itertools.product(np.arange(6) / 6., repeat=3)))
    rotations = pointgroup3d.lattice_point_group(lattice)
    shifts = np.zeros((len(rotations), 3))
    species = ["C"] * len(positions)
    operation_args = (lattice, positions, species, rotations, shifts)
    cases = [
        ("2D-small-grid", "symmetry2d", "_short_vectors", (np.diag([1., 2.]), 2.)),
        ("2D-skew-grid", "symmetry2d", "_short_vectors", (np.array([[1., 100.], [100., 10001.]]), 10001.)),
        ("3D-small-grid", "pointgroup3d", "_integer_vectors_within", (np.eye(3), 1.00001)),
        ("3D-anisotropic-grid", "pointgroup3d", "_integer_vectors_within", (np.diag([1., 2., 10000.]), 10000.)),
        ("1024-site-decorated-match", "symmetry2d", "_accepts_translation",
            (grid, grid, np.zeros(len(grid), dtype=int), np.zeros(len(grid)), 1e-4, 1e-3)),
        ("216-site48provided-permutations", "symmetry3d", "site_permutations", operation_args),
        ("216-site48provided-orbits", "symmetry3d", "equivalent_atom_map", operation_args),
        ("native-BCC-dataset", "symmetry3d", "analyse_symmetry",
            (3. * np.eye(3), np.array([[0., 0., 0.], [.5, .5, .5]]), ["Fe", "Fe"])),
        ("planar-gauge", "symmetry3d", "_canonical_planar_gauge", (np.eye(3), np.diag([3., 3., 20.]))),
    ]
    current = {"symmetry2d": symmetry2d, "pointgroup3d": pointgroup3d, "symmetry3d": symmetry3d}
    records = []
    for label, module, function, arguments in cases:
        before = functools.partial(getattr(old[module], function), *arguments)
        after = functools.partial(getattr(current[module], function), *arguments)
        assert _equal(before(), after()), label
        records.append({"case": label, "exact_output": True,
            "old": _measure(before, args.repeats), "new": _measure(after, args.repeats)})
    paths = ("src/cellstine/core/_integer_grid.py", "src/cellstine/core/symmetry2d.py",
             "src/cellstine/core/pointgroup3d.py", "src/cellstine/core/symmetry3d.py",
             "benchmarks/benchmark_symmetry_resources.py")
    print(json.dumps({"baseline": commit, "baseline_source_hashes": baseline_hashes,
        "source_hashes": {path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest() for path in paths},
        "python": sys.version, "numpy": np.__version__, "platform": platform.platform(), "records": records,
        "limits": "Local warm timings, Python-tracked peak not RSS; no CI performance thresholds or universal speedup claim. Three baseline symmetry modules are frozen, other dependencies shared. Exact comparisons cover valid existing behavior; repaired centering/empty-species cases intentionally differ and are tested separately. Enumeration remains box-sized work with output-sized storage; decorated matching remains quadratic per translation in the worst case."}, indent=2))


if __name__ == "__main__":
    main()
