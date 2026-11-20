"""Compare bounded structure helpers with a trusted Git baseline.

Run: python benchmarks/benchmark_structure_helpers.py --baseline 1db9be4
Ordinary outputs and canonical coset order must match exactly. Extreme-scale
correctness has independent geometry tests; timings are not CI thresholds.
"""

from __future__ import annotations

import argparse
import functools
import hashlib
import json
import sys

import numpy as np

from benchmark_geometry_memory import ROOT, _baseline, _measure
from cellstine.core import directions, species, transforms, vacuum


def _repeat_call(operation, count):
    result = None
    for _ in range(count):
        result = operation()
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", default="1db9be4")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--micro-calls", type=int, default=1000)
    args = parser.parse_args(argv)
    if args.repeats < 1 or args.micro_calls < 1:
        parser.error("repeat counts must be positive")
    modules = {"species": species, "transforms": transforms, "directions": directions, "vacuum": vacuum}
    old = {}
    commits = set()
    for name in modules:
        commit, old[name] = _baseline(args.baseline, name)
        commits.add(commit)
    assert len(commits) == 1
    cases = [
        ("bounded-labels", "species", "expand_species", (["C"], [1000000]), {"natoms": 3}, 1),
        ("full-labels", "species", "expand_species", (["C", "O"], [5000, 5000]), {}, 1),
        ("positive-shear", "transforms", "supercell_cosets", ([[2, 100001, 0], [0, 1, 0], [0, 0, 1]],), {}, 1),
        ("negative-shear", "transforms", "supercell_cosets", ([[2, 100001, 0], [0, -1, 0], [0, 0, 1]],), {}, 1),
        ("unimodular-shear", "transforms", "supercell_cosets", ([[1, 100001, 0], [0, 1, 0], [0, 0, 1]],), {}, 1),
        ("ordinary-cosets", "transforms", "supercell_cosets", ([[3, 1, 0], [0, 3, 1], [0, 0, 2]],), {}, 1),
        ("ordinary-rotation", "transforms", "rotation_matrix_about_axis", ([1., 2., 3.], 37.), {}, args.micro_calls),
        ("ordinary-frame", "directions", "orthonormal_frame", (np.array([1., 2., 3.]), np.diag([4., 5., 6.])), {}, args.micro_calls),
        ("ordinary-normal", "vacuum", "surface_normal", (np.array([[2., 1., .1], [0., 3., .5], [.2, 1., 7.]]),), {}, args.micro_calls),
    ]
    records = []
    for name, module, method, inputs, options, calls in cases:
        before = functools.partial(getattr(old[module], method), *inputs, **options)
        after = functools.partial(getattr(modules[module], method), *inputs, **options)
        assert np.array_equal(before(), after()), name
        operations = [functools.partial(_repeat_call, operation, calls) for operation in (before, after)]
        records.append({"case": name, "calls_per_sample": calls, "baseline": _measure(operations[0], args.repeats), "current": _measure(operations[1], args.repeats)})
    hashes = {}
    for name in modules:
        source = (ROOT / "src/cellstine/core" / (name + ".py")).read_text(encoding="utf-8-sig")
        hashes[name] = hashlib.sha256(source.encode()).hexdigest()
    print(json.dumps({"baseline_commit": commits.pop(), "current_source_sha256": hashes, "python": sys.version.split()[0], "numpy": np.__version__, "records": records}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
