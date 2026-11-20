"""Measure complete production JSON/NPZ persistence, including reconstruction."""

from __future__ import annotations

import argparse
import copy
import functools
import hashlib
import json
import platform
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from benchmark_geometry_memory import _measure  # noqa: E402
from cellstine.moire.search import gram, results  # noqa: E402


def _document():
    basis = np.array([[3., -1.5], [0., 1.5*np.sqrt(3.)]])
    config = gram.SearchConfig(top_basis=basis, bottom_basis=basis, max_length=35., top_strain=.01,
                              bottom_strain=.01, min_twist_angle_deg=5., max_twist_angle_deg=25.)
    return results.build_results_document(top_poscar="top.vasp", bottom_poscar="bottom.vasp", config=config,
        result=gram.search(config), symmetric_requested=False, symmetric_used=False,
        created_at="2000-01-01T00:00:00+00:00")


def _write(path, document, compressed):
    return results.write_results(path, document, validated=True, compressed=compressed)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=20000)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args(argv)
    if args.rows < 1 or args.repeats < 1:
        parser.error("rows and repeats must be positive")
    real = _document()
    cases = [("actual-search", real)]
    expanded = copy.deepcopy(real)
    expanded["candidates"] = [dict(real["candidates"][i % len(real["candidates"])] , index=i+1)
                              for i in range(args.rows)]
    expanded["metadata"].update(candidates_found=args.rows, candidates_recorded=args.rows)
    cases.append(("cycled-real-search-with-unique-indexes", expanded))
    records = []
    with tempfile.TemporaryDirectory(prefix="cellstine-result-formats-") as directory:
        for label, document in cases:
            expected = json.dumps(document)
            for name, suffix, compressed in (("json", ".json", False), ("npz", ".npz", False),
                                              ("npz-compressed", ".npz", True)):
                path = Path(directory)/(name+suffix)
                operation = functools.partial(_write, path, document, compressed)
                operation()
                assert json.dumps(results.read_results(path)) == expected
                assert json.dumps(document) == expected
                records.append({"case": label, "rows": len(document["candidates"]), "format": name,
                    "bytes": path.stat().st_size, "exact_document_and_order": True,
                    "write": _measure(operation, args.repeats),
                    "read_validate_reconstruct": _measure(functools.partial(results.read_results, path), args.repeats)})
    source_paths = ("src/cellstine/moire/search/npz_results.py", "src/cellstine/moire/search/results.py",
                    "benchmarks/benchmark_result_formats.py")
    print(json.dumps({"python": sys.version, "numpy": np.__version__, "platform": platform.platform(),
        "source_hashes": {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in source_paths},
        "records": records, "limits": "Warm filesystem, no fsync; Python-tracked peak, not RSS. Expanded rows cycle actual configurations; compression ratio is workload dependent. Both actual production APIs use their collection-paused context."}, indent=2))


if __name__ == "__main__":
    main()
