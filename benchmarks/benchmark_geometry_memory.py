"""Compare geometry allocation/time against a Git baseline, asserting exact outputs.

Run from the checkout: python benchmarks/benchmark_geometry_memory.py --baseline 9f59701
Timings are local observations, never CI performance thresholds. Baseline geometry
uses the same current reduction module, isolating the geometry implementation.
"""

from __future__ import annotations

import argparse
import functools
import hashlib
import json
import subprocess
import sys
import time
import tracemalloc
import types
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cellstine.core import geometry  # noqa: E402


def _baseline(ref):
    commit = subprocess.check_output(
        ["git", "rev-parse", "--verify", "--end-of-options", ref + "^{commit}"],
        cwd=ROOT, text=True,
    ).strip()
    source = subprocess.check_output(
        ["git", "show", commit + ":src/cellstine/core/geometry.py"], cwd=ROOT, text=True,
    )
    module = types.ModuleType("cellstine.core._geometry_baseline")
    module.__package__ = "cellstine.core"
    sys.modules[module.__name__] = module
    exec(compile(source, commit + ":geometry.py", "exec"), module.__dict__)
    return commit, module


def _measure(operation, repeats):
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        operation()
        samples.append(time.perf_counter() - start)
    tracemalloc.start()
    try:
        operation()
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return {"median_ms": float(np.median(samples)) * 1000.0, "peak_MiB": peak / 1024**2}


def _nearest(module, points, queries, pitch=None, block=1 << 20):
    return module.NearestPointQuery(points, pitch=pitch, block=block).distances(queries)


def _site_matches(module, sites, labels, queries, wanted):
    return module.PeriodicSiteIndex(np.eye(3), sites, labels, 1e-4).match(queries, wanted)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", default="9f59701")
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args(argv)
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    commit, previous = _baseline(args.baseline)
    cases = []
    points = np.zeros((512, 3))
    queries = np.zeros((4096, 3))
    for block in (1024, 16384, 1 << 20):
        cases.append((f"cluster-block-{block}", "_nearest", (points, queries, 1.0, block)))
    line = np.column_stack([np.linspace(0.0, 1.0, 1024), np.zeros((1024, 2))])
    x, y = np.meshgrid(np.arange(32) / 32.0, np.arange(32) / 32.0, indexing="ij")
    plane = np.column_stack([x.ravel(), y.ravel(), np.zeros(1024)])
    for name, cloud in (("line-auto", line), ("plane-auto", plane)):
        cases.append((name, "_nearest", (cloud, cloud[::32] + [0.0, 0.0, 0.1])))
    cases.append(("site-matching", "_site_matches", (
        points, np.arange(len(points)) % 2, queries[:2048], np.arange(2048) % 3,
    )))
    cases.append(("plane-images", "plane_minimum_image", (
        np.array([[1.0, 0.0], [-0.5, np.sqrt(3.0) / 2.0]]),
        np.tile([0.49, -0.49], (120000, 1)),
    )))
    cases.append(("vacuum-pair", "shortest_interatomic_distance", (
        np.diag([1.0, 1.0, 10000.0]), np.array([[0.0, 0.0, 0.5], [0.1, 0.0, 0.5]]),
    )))
    rng = np.random.default_rng(91)
    cases.append(("bulk-auto", "_nearest", (rng.normal(size=(256, 3)), rng.normal(size=(2048, 3)))))
    records = []
    for name, method, arguments in cases:
        operations = []
        for module in (previous, geometry):
            if method == "_nearest":
                operations.append(functools.partial(_nearest, module, *arguments))
            elif method == "_site_matches":
                operations.append(functools.partial(_site_matches, module, *arguments))
            else:
                operations.append(functools.partial(getattr(module, method), *arguments))
        assert np.array_equal(operations[0](), operations[1](), equal_nan=True), name
        records.append({"case": name, "baseline": _measure(operations[0], args.repeats),
                        "current": _measure(operations[1], args.repeats)})
    source_hashes = {}
    for name in ("geometry", "reduction"):
        source = (ROOT / "src" / "cellstine" / "core" / (name + ".py")).read_text(encoding="utf-8-sig")
        source_hashes[name] = hashlib.sha256(source.encode("utf-8")).hexdigest()
    print(json.dumps({"baseline_commit": commit, "current_source_sha256": source_hashes,
                      "python": sys.version.split()[0], "numpy": np.__version__,
                      "records": records}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
