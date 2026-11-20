"""Measure reciprocal-path allocations/time against a Git baseline.

Run from the checkout with --baseline 427e2d5. Assertions precede timing;
local medians and tracemalloc peaks are observations, not CI thresholds.
"""
from __future__ import annotations

import argparse
from functools import partial
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import types

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cellstine.core import kpath, strata  # noqa: E402
from benchmark_geometry_memory import _measure  # noqa: E402


def _baseline(ref, name):
    commit = subprocess.check_output(
        ["git", "rev-parse", "--verify", "--end-of-options", ref + "^{commit}"],
        cwd=ROOT, text=True, encoding="utf-8",
    ).strip()
    source = subprocess.check_output(
        ["git", "show", commit + ":src/cellstine/core/" + name + ".py"],
        cwd=ROOT, text=True, encoding="utf-8",
    )
    module = types.ModuleType("cellstine.core._" + name + "_baseline")
    module.__package__ = "cellstine.core"
    sys.modules[module.__name__] = module
    exec(compile(source, commit + ":" + name, "exec"), module.__dict__)
    return commit, module


def _equal(left, right):
    if isinstance(left, np.ndarray):
        assert isinstance(right, np.ndarray) and np.array_equal(left, right)
    elif isinstance(left, (tuple, list)):
        assert type(left) is type(right) and len(left) == len(right)
        for first, second in zip(left, right):
            _equal(first, second)
    else:
        assert left == right


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", default="427e2d5")
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args(argv)
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    commit, old_path = _baseline(args.baseline, "kpath")
    _, old_strata = _baseline(args.baseline, "strata")
    path = kpath.band_path(np.eye(3))
    cases = []
    for count in (20, 20000):
        cases.append((f"path-divisions-{count}",
                      partial(old_path.BandPath.sample, path, count),
                      partial(path.sample, count)))
    rng = np.random.default_rng(913)
    points = rng.uniform(-1, 1, (24000, 3))
    operations = np.tile(np.array([np.eye(3, dtype=int), -np.eye(3, dtype=int)]), (24, 1, 1))
    cases.append(("fixing-masks-24000x48", partial(old_strata._fixing_masks, points, operations),
                  partial(strata._fixing_masks, points, operations)))
    graph = [[(int(j), float(rng.integers(0, 100)))
              for j in rng.choice(900, size=24, replace=False)] for _ in range(900)]
    cases.append(("dijkstra-900x24", partial(old_path._dijkstra, graph, 0, len(graph)),
                  partial(kpath._dijkstra, graph, 0, len(graph))))
    group = np.array([np.eye(3, dtype=int), np.diag([-1, -1, 1])])
    # Warm both caches before measuring the intentional caller-ownership cost.
    _equal(old_strata.grid_strata(group), strata.grid_strata(group))
    cases.append(("grid-strata-warm-owned-directions", partial(old_strata.grid_strata, group),
                  partial(strata.grid_strata, group)))
    records = []
    for name, before, after in cases:
        _equal(before(), after())
        previous = _measure(before, args.repeats)
        current = _measure(after, args.repeats)
        records.append({"case": name, "baseline": previous, "current": current,
                        "speedup": previous["median_ms"] / current["median_ms"],
                        "exact_outputs": True})
    hashes = {"src/cellstine/core/" + name + ".py": hashlib.sha256(
        (ROOT / "src/cellstine/core" / (name + ".py")).read_bytes()).hexdigest()
              for name in ("brillouin", "strata", "kpath")}
    print(json.dumps({"baseline": commit, "repeats": args.repeats,
                      "source_hashes": hashes, "cases": records}, indent=2))


if __name__ == "__main__":
    main()
