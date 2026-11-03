"""Measure persistence against Git, asserting exact bytes and record ownership.

Run: python benchmarks/benchmark_persistence.py --baseline 6692121
Measurements are local observations, not CI timing thresholds. Path/kpath cases
isolate their changed encoder expression; other cases call the actual writers.
"""

from __future__ import annotations

import argparse
import copy
import functools
import json
import subprocess
import sys
import tempfile
import types
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from benchmark_geometry_memory import _measure  # noqa: E402
from cellstine.core import jsonio, manifests  # noqa: E402
from cellstine.defect import analysis as defect  # noqa: E402
from cellstine.defect.records import DefectAnalysis, DefectSite  # noqa: E402
from cellstine.interface.surface import backend as surface  # noqa: E402
from cellstine.interface.surface.surface_types import AdsorptionSite, SiteAnalysisRun  # noqa: E402
from cellstine.interface.workflow import lattice_match as matches  # noqa: E402
from cellstine.io.models import StructureRecord  # noqa: E402
from cellstine.moire.search import gram, results  # noqa: E402
from cellstine.symmetry import symmetry  # noqa: E402
from cellstine.symmetry.models import SymmetryAnalysis, SymmetryOperation  # noqa: E402


def _load(commit, package, module):
    path = "src/" + package.replace(".", "/") + "/" + module + ".py"
    source = subprocess.check_output(["git", "show", commit + ":" + path], cwd=ROOT, text=True, encoding="utf-8")
    loaded = types.ModuleType(package + "._persistence_baseline_" + module)
    loaded.__package__ = package
    sys.modules[loaded.__name__] = loaded
    exec(compile(source, commit + ":" + path, "exec"), loaded.__dict__)
    return loaded


def _cases(rows):
    manifest = manifests.RunManifest.create(workflow="defect", stage="generate", run_id="probe", backend="native")
    manifest.created_at = "2000-01-01T00:00:00+00:00"
    yield "manifest", "manifest", manifest
    for count in (100, rows):
        sites = [DefectSite(f"atom_{index}", "Mo", 1, (.1, .2, .3), (1., 2., 3.),
                           equivalent_indices=[index + 1], members=[{"indices": [index + 1], "direct": [.1, .2, .3]}])
                 for index in range(count)]
        yield f"defect-{count}", "defect", DefectAnalysis("source.vasp", "bulk", "native", count, ["Mo"], [count], [], sites)
        run = SiteAnalysisRun(None, "source.vasp", "top", count, 3, 2.4, 2.7, 6., {"top": count},
                              [AdsorptionSite("top", (.1, .2, .3), (1., 2., 3.), 6, 1.2, 1) for _ in range(count)])
        yield f"surface-{count}", "surface", run
        yield f"path-encoding-{count}", "encoding", {
            "schema": "cellstine.defect.path", "version": 1,
            "images": [{"index": index, "path": f"path/{index}/POSCAR", "positions_direct": [[.1, .2, .3]] * 10}
                       for index in range(count)],
        }
    report = SymmetryAnalysis(None, "native", 1, ["Mo"], [1], {"a": 3., "b": 3., "c": 12.},
                             operations=[SymmetryOperation([[1, 0, 0], [0, 1, 0], [0, 0, 1]], (0., 0., 0.))] * 192)
    yield "symmetry", "symmetry", report
    yield "kpath-encoding", "encoding", {"schema": "cellstine.symmetry.kpath", "version": 1,
        "divisions": 32, "sampled_point_count": 192, "ticks": [{"label": "GAMMA", "distance": 0.}],
        "points": [{"label": "X", "fractional": [.5, 0., 0.]}]}
    config = gram.SearchConfig(np.eye(2), np.eye(2), 1.1, 0., 0.)
    document = results.build_results_document(top_poscar="top.vasp", bottom_poscar="bottom.vasp", config=config,
        result=gram.search(config), symmetric_requested=False, symmetric_used=False, created_at=manifest.created_at)
    pair = matches.SlabPair(Path("bottom.vasp"), Path("top.vasp"), (0, 0, 1), (0, 0, 1), 1, 1)
    entries = matches.match_entries(pair, document["candidates"] * rows, results_path=Path("results.json"))
    yield "interface-matches", "matches", matches.build_match_document(bottom_bulk="bottom.vasp", top_bulk="top.vasp",
        request=matches.MatchRequest(10.), entries=entries, vacuum=15., bottom_millers=["001"], top_millers=["001"],
        bottom_layers_list=[1], top_layers_list=[1])
    document["candidates"] = [dict(copy.deepcopy(document["candidates"][0]), index=index + 1) for index in range(rows)]
    document["metadata"].update(candidates_found=rows, candidates_recorded=rows)
    yield "read-results", "read", document


def _write(module, kind, path, payload):
    if kind == "manifest":
        return module.RunManifest.write(payload, path)
    if kind in ("defect", "symmetry"):
        return module._write_analysis_file(path, payload)
    if kind == "surface":
        return module.write_site_report_json(str(path), payload)
    if kind == "matches":
        return module.write_matches(path, payload)
    with path.open("w", encoding="utf-8") as handle:
        if module is jsonio:
            jsonio.write_indented_json(handle, payload)
        else:
            json.dump(payload, handle, indent=2)
    return path


def _record(module, source, labels, positions, flags):
    return module._record_from_atoms(source, labels, positions, flags, comment="regrouped")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", default="6692121")
    parser.add_argument("--rows", type=int, default=2000)
    parser.add_argument("--atoms", type=int, default=40000)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args(argv)
    if min(args.rows, args.atoms, args.repeats) < 1:
        parser.error("rows, atoms and repeats must be positive")
    commit = subprocess.check_output(["git", "rev-parse", "--verify", "--end-of-options", args.baseline + "^{commit}"],
                                     cwd=ROOT, text=True, encoding="utf-8").strip()
    current = {"manifest": manifests, "defect": defect, "surface": surface, "symmetry": symmetry,
               "matches": matches, "encoding": jsonio, "read": results}
    previous = {key: _load(commit, module.__package__, module.__name__.rsplit(".", 1)[-1])
                for key, module in current.items() if key != "encoding"}
    previous["encoding"] = json
    records = []
    with tempfile.TemporaryDirectory(prefix="cellstine-persistence-benchmark-") as temporary:
        for name, kind, payload in _cases(args.rows):
            paths = [Path(temporary) / "baseline.json", Path(temporary) / "current.json"]
            if kind == "read":
                paths[0].write_text(json.dumps(payload, indent=2), encoding="utf-8")
                assert previous[kind].read_results(paths[0]) == current[kind].read_results(paths[0])
                operations = [functools.partial(module.read_results, paths[0]) for module in (previous[kind], current[kind])]
            else:
                operations = [functools.partial(_write, module, kind, path, payload)
                              for module, path in zip((previous[kind], current[kind]), paths)]
                for operation in operations:
                    operation()
                assert paths[0].read_bytes() == paths[1].read_bytes(), name
            records.append({"case": name, "bytes": paths[0].stat().st_size,
                            "baseline": _measure(operations[0], args.repeats), "current": _measure(operations[1], args.repeats)})
        count = args.atoms
        labels = ["Si", "X", "O", "C", "N", "Mo", "S", "Au"] * ((count + 7) // 8)
        labels = labels[:count]
        positions = np.arange(count * 3, dtype=float).reshape(count, 3) / count
        original = positions.copy()
        source = StructureRecord("source", np.eye(3), ["O", "Si", "C", "N", "Mo", "S", "Au", "Cu"],
                                 [0] * 8, np.empty((0, 3)), np.empty((0, 3)))
        flags = [("T", "F", "T")] * count
        for selected_flags in (None, flags):
            operations = [functools.partial(_record, module, source, labels, positions, selected_flags)
                          for module in (previous["defect"], defect)]
            before, after = operations[0](), operations[1]()
            for field in ("positions_direct", "positions_cartesian", "lattice"):
                assert np.array_equal(getattr(before, field), getattr(after, field)), field
            for field in ("species", "counts", "selective_flags", "metadata", "comment", "source_path", "source_format"):
                assert getattr(before, field) == getattr(after, field), field
            assert not np.shares_memory(after.positions_direct, positions)
            assert np.array_equal(positions, original)
            records.append({"case": "regroup-flags" if selected_flags is not None else "regroup", "atoms": count,
                            "baseline": _measure(operations[0], args.repeats), "current": _measure(operations[1], args.repeats)})
    print(json.dumps({"baseline": commit, "python": sys.version, "numpy": np.__version__, "records": records}, indent=2))


if __name__ == "__main__":
    main()
