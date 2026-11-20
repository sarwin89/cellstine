"""Compare void working allocations with a trusted Git baseline.

Run: python benchmarks/benchmark_voids_memory.py --baseline a455f81
Optional --include-dense-anchor measures the corrected large-neighbour kernel;
its old truncated result is incorrect, so this case has no speedup comparison.
Inputs precede tracing. Timings are observations, never CI thresholds.
"""

from __future__ import annotations

import argparse
import dataclasses
import functools
import hashlib
import json
import math
import sys

import numpy as np

from benchmark_geometry_memory import ROOT, _baseline, _measure
from cellstine.core import geometry, voids


def _cold_triples(module, count):
    module._TRIPLE_INDEX_CACHE.clear()
    return module._triple_indices(count)


def _public(module, lattice, atoms, saddles, merge=None):
    return dataclasses.asdict(module.find_void_sites(lattice, atoms, include_saddles=saddles, merge_distance=merge))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", default="a455f81")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--include-dense-anchor", action="store_true")
    args = parser.parse_args(argv)
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    commit, previous = _baseline(args.baseline, "voids")
    directions = np.tile(np.concatenate((np.eye(3), -np.eye(3))), (34, 1))
    lattice = np.eye(3) * 2.87
    atoms = np.array([[0., 0, 0], [.5, .5, .5]])
    cases = [
        ("cold-triples-120", functools.partial(_cold_triples, previous, 120), functools.partial(_cold_triples, voids, 120)),
        ("classification-204", functools.partial(previous._classify_contact_directions, directions), functools.partial(voids._classify_contact_directions, directions)),
        ("bcc-maxima", functools.partial(_public, previous, lattice, atoms, False), functools.partial(_public, voids, lattice, atoms, False)),
        ("bcc-saddles", functools.partial(_public, previous, lattice, atoms, True), functools.partial(_public, voids, lattice, atoms, True)),
    ]
    supercell = np.eye(3) * 4.0
    repeated = np.indices((4, 4, 4)).reshape(3, -1).T / 4.0
    for name, merge in (("sc64-default", None), ("sc64-wide-merge", 4.0)):
        cases.append((name, functools.partial(_public, previous, supercell, repeated, False, merge), functools.partial(_public, voids, supercell, repeated, False, merge)))
    records = []
    for name, before, after in cases:
        expected, actual = before(), after()
        if isinstance(expected, np.ndarray):
            assert np.array_equal(expected, actual), name
        else:
            assert expected == actual, name
        records.append({"case": name, "baseline": _measure(before, args.repeats), "current": _measure(after, args.repeats)})
    if args.include_dense_anchor:
        lattice = np.eye(3) * 10.0
        atoms = np.full((202, 3), .5)
        atoms[:, 0] = np.linspace(.49, .51, len(atoms))
        images, indices, valid = geometry.neighbour_images(lattice, atoms, 18.)
        anchor = atoms[100] @ lattice
        neighbours = images[indices[100][valid[100]]]
        neighbours = neighbours[np.linalg.norm(neighbours - anchor, axis=1) > 1e-8]
        operation = functools.partial(voids._streamed_void_centres, anchor, neighbours, 9., False)
        centres, radii = operation()
        local_radius = math.sqrt(50.0 + (0.2 / 201 / 2)**2)
        assert len(centres) and local_radius - 1e-9 <= max(radii) <= local_radius + 1e-6
        records.append({"case": "dense-anchor-5453", "baseline": None, "current": _measure(operation, args.repeats), "accepted_raw_centres": len(radii)})
    source = (ROOT / "src/cellstine/core/voids.py").read_text(encoding="utf-8-sig")
    print(json.dumps({"baseline_commit": commit, "current_source_sha256": hashlib.sha256(source.encode()).hexdigest(), "python": sys.version.split()[0], "numpy": np.__version__, "records": records}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
