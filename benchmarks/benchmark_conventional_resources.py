"""Measure conventional-cell reuse and prompt group-cap rejection against Git.

Run: python benchmarks/benchmark_conventional_resources.py --baseline bba6576
Local observations are not CI timing thresholds; valid outputs must match exactly.
"""

from __future__ import annotations

import argparse
import functools
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from benchmark_geometry_memory import _baseline, _measure  # noqa: E402
from cellstine.core import bravais, idealisation  # noqa: E402


def _over_cap(module, group, limit):
    try:
        module.close_matrix_group(group, max_order=limit, name="bounded")
    except ValueError as error:
        return str(error)
    raise AssertionError("an oversized group must be refused")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", default="bba6576")
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args(argv)
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    commit, old_bravais = _baseline(args.baseline, "bravais")
    _, old_idealisation = _baseline(args.baseline, "idealisation")
    # Baseline bravais imports the current idealisation module by relative name.
    # Bind its original implementation explicitly to compare complete paths.
    old_bravais.symmetrise_basis = old_idealisation.symmetrise_basis
    hexagonal = np.array([[4., 0., 0.], [-2., 2. * math.sqrt(3.), 0.], [0., 0., 12.4]])
    rhombohedral = np.array([[2., 1., 1.], [-1., 1., 1.], [-1., -2., 1.]]) @ hexagonal / 3.
    cases = {
        "cP": np.eye(3) * 4.,
        "oP": np.diag([4., 5.3, 6.7]),
        "tP": np.diag([4., 4., 6.7]),
        "hP": hexagonal,
        "hR": rhombohedral,
        "mP": np.array([[4., 0., 0.], [0., 5.3, 0.], [1.9, 0., 6.7]]),
    }
    records = []
    for symbol, lattice in cases.items():
        original = lattice.copy()
        operations = [functools.partial(module.conventional_cell, lattice)
                      for module in (old_bravais, bravais)]
        before, after = operations[0](), operations[1]()
        assert before.symbol == after.symbol == symbol
        assert before.summary() == after.summary(), symbol
        for field in ("cell", "lattice", "centring_vectors", "to_conventional", "to_primitive"):
            assert np.array_equal(getattr(before, field), getattr(after, field)), (symbol, field)
        assert np.array_equal(original, lattice), symbol
        records.append({"case": symbol, "baseline": _measure(operations[0], args.repeats),
                        "current": _measure(operations[1], args.repeats), "exact_equal": True})
    generators = np.array([[[1, value], [0, 1]] for value in range(1, 101)], dtype=np.int64)
    operations = [functools.partial(_over_cap, module, generators, 12)
                  for module in (old_idealisation, idealisation)]
    assert operations[0]() == operations[1]() == "bounded closure exceeded 12 elements"
    records.append({"case": "100-generators-cap-12", "baseline": _measure(operations[0], args.repeats),
                    "current": _measure(operations[1], args.repeats), "same_error": True})
    quarter = np.array([[0, -1], [1, 0]], dtype=np.int64)
    mirror = np.diag([1, -1])
    group = np.stack([quarter, mirror])
    old = old_idealisation.close_matrix_group(group, max_order=8)
    new = idealisation.close_matrix_group(group, max_order=8)
    assert np.array_equal(old, new) and not old.flags.writeable and not new.flags.writeable
    hashes = {}
    for name in ("bravais", "idealisation"):
        source = (ROOT / "src/cellstine/core" / (name + ".py")).read_text(encoding="utf-8-sig")
        hashes[name] = hashlib.sha256(source.encode()).hexdigest()
    print(json.dumps({"baseline_commit": commit, "current_source_sha256": hashes,
                      "python": sys.version.split()[0], "numpy": np.__version__,
                      "valid_group_order_exact_equal": True, "records": records}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
