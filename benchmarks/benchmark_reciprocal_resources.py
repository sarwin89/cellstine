"""Compare reciprocal mesh time/allocation and exercise the large-grid key fix.

Run: python benchmarks/benchmark_reciprocal_resources.py --baseline dc61527
Local measurements are observations, never CI timing thresholds.
"""

from __future__ import annotations

import argparse
import functools
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from benchmark_geometry_memory import _baseline, _measure  # noqa: E402
from cellstine.core import reciprocal  # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", default="dc61527")
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args(argv)
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    commit, previous = _baseline(args.baseline, "reciprocal")
    lattice = np.eye(3) * 4.0
    rotation = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.int64)
    rng = np.random.default_rng(83)
    numerators = rng.integers(-256, 256, size=(200000, 3), dtype=np.int64)
    cases = [
        ("128x128x16-numerators", "_mesh_numerators", ((128, 128, 16), (0, 1, 0)), {}),
        ("200k-ordinary-keys", "_encode", (numerators, 512), {}),
        ("32x32x16-fourfold-reduced-mesh", "build_mesh", (lattice,),
         {"divisions": (32, 32, 16), "rotations": [rotation], "time_reversal": True}),
    ]
    records = []
    for name, method, positional, keywords in cases:
        operations = [functools.partial(getattr(module, method), *positional, **keywords)
                      for module in (previous, reciprocal)]
        before, after = operations[0](), operations[1]()
        if method == "_mesh_numerators":
            assert before[1] == after[1] and np.array_equal(before[0], after[0]), name
        elif method == "_encode":
            assert np.array_equal(before, after), name
        else:
            assert before.summary() == after.summary(), name
            assert np.array_equal(before.points, after.points), name
            assert np.array_equal(before.weights, after.weights), name
        records.append({"case": name, "baseline": _measure(operations[0], args.repeats),
                        "current": _measure(operations[1], args.repeats), "exact_equal": True})
    # This is a correctness extension, so do not benchmark against an incorrect
    # baseline as if it produced the same answer. The million-point run stays
    # here instead of being multiplied across every CI correctness selection.
    count = 2**20
    operation = functools.partial(reciprocal.build_mesh, lattice, divisions=(count, 1, 1),
                                  time_reversal=False)
    fixed = operation()
    assert fixed.point_count == count and np.all(fixed.weights == 1)
    expected = np.arange(count, dtype=float) / count
    expected[expected >= 0.5] -= 1.0
    assert np.array_equal(fixed.points[:, 0], expected)
    assert np.all(fixed.points[:, 1:] == 0.0)
    probe = np.array([[-count, 0, 0], [0, 0, 0]], dtype=np.int64)
    assert len(np.unique(previous._encode(probe, 2 * count))) == 1
    assert len(np.unique(reciprocal._encode(probe, 2 * count))) == 2
    del fixed, expected
    records.append({"case": "1048576x1x1-identity-only-correctness-extension",
                    "baseline": "two distinct keys collide; not a valid timing comparator",
                    "current": _measure(operation, args.repeats), "point_count": count,
                    "all_weights_one": True})
    source = (ROOT / "src/cellstine/core/reciprocal.py").read_text(encoding="utf-8-sig")
    print(json.dumps({"baseline_commit": commit,
                      "current_source_sha256": hashlib.sha256(source.encode()).hexdigest(),
                      "python": sys.version.split()[0], "numpy": np.__version__,
                      "records": records}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
