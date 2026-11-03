"""Compare search arrays and JSON against a complete Git search-package baseline.

Run: python benchmarks/benchmark_moire_fast_paths.py --baseline c57d6a1
Timing observations are local measurements, not CI thresholds.
"""

from __future__ import annotations

import argparse
import dataclasses
import functools
import importlib.util
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from benchmark_geometry_memory import _measure  # noqa: E402
from cellstine.core.symmetry2d import lattice_point_group, proper_subgroup  # noqa: E402
from cellstine.moire.search import gram, results  # noqa: E402


def _baseline(ref, directory):
    commit = subprocess.check_output(
        ["git", "rev-parse", "--verify", "--end-of-options", ref + "^{commit}"],
        cwd=ROOT, text=True,
    ).strip()
    prefix = "src/cellstine/moire/search/"
    paths = subprocess.check_output(
        ["git", "ls-tree", "-r", "--name-only", commit, "--", prefix],
        cwd=ROOT, text=True,
    ).splitlines()
    for path in paths:
        if path.endswith(".py"):
            target = directory / Path(path).relative_to(prefix)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(subprocess.check_output(["git", "show", commit + ":" + path], cwd=ROOT))
    name = "cellstine.moire._benchmark_search_baseline"
    spec = importlib.util.spec_from_file_location(
        name, directory / "__init__.py", submodule_search_locations=[str(directory)],
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return commit, importlib.import_module(name + ".gram"), importlib.import_module(name + ".results")


def _cases():
    square = np.eye(2) * 3.
    hexagonal = np.array([[3., -1.5], [0., 1.5 * np.sqrt(3.)]])
    rectangular = np.diag([3., 4.7])
    oblique = np.array([[3., .7], [0., 4.3]])
    bases = {"square": square, "hexagonal": hexagonal, "rectangular": rectangular, "oblique": oblique}
    for name, basis in bases.items():
        for variant, options in (
            ("strained", {}), ("rigid", {"top_strain": 0., "bottom_strain": 0.}),
            ("unfolded", {"fold_symmetry": False}),
            ("imprimitive", {"primitive_only": False}),
            ("unfolded-imprimitive", {"primitive_only": False, "fold_symmetry": False}),
            ("window", {"min_twist_angle_deg": 8., "max_twist_angle_deg": 23.}),
            ("atom-cap", {"max_atoms": 12}),
        ):
            yield name + "-" + variant, dict(top_basis=basis, bottom_basis=basis, max_length=12.,
                                            top_strain=.02, bottom_strain=.01) | options
    for orientation in (20., 37.):
        angle = np.deg2rad(orientation)
        rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        yield f"paired-mirror-{orientation:g}", dict(
            top_basis=rectangular, bottom_basis=rotation @ rectangular, max_length=14.,
            top_strain=.03, bottom_strain=.02,
            min_twist_angle_deg=orientation - 1., max_twist_angle_deg=orientation + 1.,
        )
    for name, basis in (("square", square), ("hexagonal", hexagonal)):
        yield name + "-symmetric", dict(top_basis=basis, bottom_basis=basis, max_length=18.,
                                        top_strain=.01, bottom_strain=.01, symmetric=True)
        group = lattice_point_group(basis)
        for label, top, bottom in (
            ("proper-full", proper_subgroup(group), group),
            ("identity-full", np.eye(2, dtype=np.int64)[None], group),
            ("proper-identity", proper_subgroup(group), np.eye(2, dtype=np.int64)[None]),
        ):
            yield name + "-" + label, dict(top_basis=basis, bottom_basis=basis,
                top_group=top, bottom_group=bottom, max_length=12., top_strain=.02, bottom_strain=.01)


def _document(module, config, result):
    return module.build_results_document(
        top_poscar="top.vasp", bottom_poscar="bottom.vasp", config=config, result=result,
        symmetric_requested=config.symmetric, symmetric_used=config.symmetric,
        created_at="2000-01-01T00:00:00+00:00",
    )


def _assert_same_result(name, before, after):
    for field in dataclasses.fields(before):
        if field.name != "stats":
            assert np.array_equal(getattr(before, field.name), getattr(after, field.name)), (name, field.name)
    # Stage counts describe work performed, which moves earlier in the pipeline.
    # Retain every changed statistic for audit rather than silently ignoring it.
    return {key: {"baseline": before.stats.get(key), "current": after.stats.get(key)}
            for key in sorted(before.stats.keys() | after.stats.keys())
            if before.stats.get(key) != after.stats.get(key)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", default="c57d6a1")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--measure", action="store_true", help="also measure a larger search and persistence")
    parser.add_argument("--length", type=float, default=35., help="maximum length of the measured search")
    args = parser.parse_args(argv)
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    records = []
    with tempfile.TemporaryDirectory(prefix="cellstine-search-baseline-") as temporary:
        commit, old_gram, old_results = _baseline(args.baseline, Path(temporary))
        for name, options in _cases():
            configs = [module.SearchConfig(**options) for module in (old_gram, gram)]
            before, after = [module.search(config) for module, config in zip((old_gram, gram), configs)]
            changed_stats = _assert_same_result(name, before, after)
            documents = []
            errors = []
            for module, config, result in zip((old_results, results), configs, (before, after)):
                try:
                    documents.append(_document(module, config, result))
                    errors.append(None)
                except ValueError as error:
                    documents.append(None)
                    errors.append(str(error))
            assert errors[0] == errors[1], (name, "document validation", errors)
            if errors[0] is not None:
                records.append({"case": name, "rows": len(after), "exact_arrays": True,
                                "same_document_error": errors[0], "changed_stats": changed_stats})
                continue
            for document in documents:
                document["metadata"].pop("stage_stats", None)
            assert documents[0] == documents[1], (name, "JSON")
            records.append({"case": name, "rows": len(after), "exact_arrays_and_stable_json": True,
                            "changed_stats": changed_stats})
        measurements = []
        if args.measure:
            basis = np.array([[3., -1.5], [0., 1.5 * np.sqrt(3.)]])
            options = dict(top_basis=basis, bottom_basis=basis, max_length=args.length, top_strain=.01,
                           bottom_strain=.01, min_twist_angle_deg=5., max_twist_angle_deg=25.)
            previous_result = previous_document = None
            for label, engine, writer in (("baseline", old_gram, old_results), ("current", gram, results)):
                config = engine.SearchConfig(**options)
                result = engine.search(config)
                document = _document(writer, config, result)
                stable_document = {**document, "metadata": {key: value for key, value in document["metadata"].items()
                                                          if key != "stage_stats"}}
                if previous_result is not None:
                    _assert_same_result("measured-search", previous_result, result)
                    assert previous_document == stable_document, "measured JSON"
                previous_result, previous_document = result, stable_document
                measurements.append({"implementation": label, "rows": len(result),
                    "search": _measure(functools.partial(engine.search, config), args.repeats),
                    "document": _measure(functools.partial(_document, writer, config, result), args.repeats),
                    "write": _measure(functools.partial(writer.write_results, Path(temporary) / "result.json", document), args.repeats)})
        print(json.dumps({"baseline": commit, "comparisons": records, "measurements": measurements}, indent=2))


if __name__ == "__main__":
    main()
