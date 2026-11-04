"""Compare actual builder outputs and resources against a frozen Git baseline."""

from __future__ import annotations

import argparse
import copy
from dataclasses import asdict
import functools
import hashlib
import importlib
import json
import platform
from pathlib import Path
import subprocess
import sys
import tempfile
import types

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from benchmark_geometry_memory import _measure  # noqa: E402
from cellstine.moire.builder import generator, nlayer  # noqa: E402
from cellstine.moire.search.find import run_find  # noqa: E402
from cellstine.moire.search.nlayer import run_findn  # noqa: E402
from cellstine.moire.search.results import read_results, write_results  # noqa: E402

bilayer = importlib.import_module("cellstine.moire.builder.make")


def _baseline(ref):
    commit = subprocess.check_output(
        ["git", "rev-parse", "--verify", "--end-of-options", ref + "^{commit}"],
        cwd=ROOT, text=True, encoding="utf-8",
    ).strip()
    modules, hashes = {}, {}
    for name in ("generator", "make", "nlayer"):
        path = f"src/cellstine/moire/builder/{name}.py"
        source = subprocess.check_output(
            ["git", "show", commit + ":" + path], cwd=ROOT, text=True, encoding="utf-8",
        )
        module = types.ModuleType("cellstine.moire.builder._baseline_" + name)
        module.__package__ = "cellstine.moire.builder"
        sys.modules[module.__name__] = module
        exec(compile(source, commit + ":" + path, "exec"), module.__dict__)
        modules[name] = module
        hashes[path] = hashlib.sha256(source.encode("utf-8")).hexdigest()
    modules["make"].generator_backend = modules["generator"]
    modules["nlayer"].generator = modules["generator"]
    return commit, modules, hashes


def _batch(module, path, workspace, indexes, multilayer=False):
    # The helper owns this newly-created directory and removes only its outputs.
    with tempfile.TemporaryDirectory(prefix="batch-", dir=workspace) as directory:
        options = {"interlayers": [3.35, 4.]} if multilayer else {"interlayer_distance": 3.35}
        runs = module.generate_many_from_results(
            str(path), indexes=indexes, output_dir=directory, vacuum=15., **options,
        )
        outputs = []
        for run in runs:
            metadata = asdict(run)
            metadata.pop("output_path")
            outputs.append((run.output_path.read_bytes(), json.dumps(metadata, sort_keys=True)))
        return outputs


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", default="8bfcf17127fee9c4e865f3c0c3140fbb11112e5b")
    parser.add_argument("--rows", type=int, default=5000)
    parser.add_argument("--indexes", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args(argv)
    if min(args.rows, args.indexes, args.repeats) < 1:
        parser.error("rows, indexes and repeats must be positive")
    commit, old, old_hashes = _baseline(args.baseline)
    records = []
    for matrix in ([[37, 0], [0, 53]], [[1000, 0], [0, 1500]]):
        matrix = np.asarray(matrix, dtype=np.int64)
        assert np.array_equal(old["generator"]._coset_representatives(matrix), generator._coset_representatives(matrix))
        records.append({"case": "exact-ij-cosets", "matrix": matrix.tolist(),
            "old": _measure(functools.partial(old["generator"]._coset_representatives, matrix), args.repeats),
            "new": _measure(functools.partial(generator._coset_representatives, matrix), args.repeats),
            "exact_output": True})
    workspace = ROOT / ".tmp"
    workspace.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="cellstine-builders-", dir=workspace) as directory:
        work = Path(directory)
        layer = work / "layer.vasp"
        layer.write_text("carbon\n1\n2 0 0\n0 2 0\n0 0 12\nC\n1\nSelective dynamics\nDirect\n0 0 0.5 F T F\n", encoding="utf-8")
        found = run_find(top_poscar=str(layer), bottom_poscar=str(layer), max_length=2.1,
            top_strain=0., bottom_strain=0., reduce_layers=False, output_root=str(work / "bilayer"))
        stack = run_findn(base_poscar=str(layer), upper_poscars=[str(layer), str(layer)], max_length=2.1,
            layer_strains=0., reduce_layers=False, output_root=str(work / "nlayer"))
        documents = (("bilayer", read_results(found.result_path)),
                     ("nlayer", json.loads(stack.result_path.read_text(encoding="utf-8"))))
        for kind, original in documents:
            for rows in (len(original["candidates"]), args.rows):
                document = copy.deepcopy(original)
                document["candidates"] = [dict(original["candidates"][i % len(original["candidates"])] , index=i+1)
                                          for i in range(rows)]
                if kind == "bilayer":
                    document["metadata"].update(candidates_found=rows, candidates_recorded=rows)
                for suffix in (("json", "npz") if kind == "bilayer" else ("json",)):
                    path = work / f"{kind}-{rows}.{suffix}"
                    if kind == "bilayer":
                        write_results(path, document)
                    else:
                        path.write_text(json.dumps(document), encoding="utf-8")
                    indexes = [(i % rows) + 1 for i in range(args.indexes)]
                    before = functools.partial(_batch, old["make" if kind == "bilayer" else "nlayer"],
                        path, work, indexes, kind == "nlayer")
                    after = functools.partial(_batch, bilayer if kind == "bilayer" else nlayer,
                        path, work, indexes, kind == "nlayer")
                    assert before() == after(), (kind, rows, suffix)
                    records.append({"case": kind, "rows": rows, "indexes": args.indexes, "format": suffix,
                        "old": _measure(before, args.repeats), "new": _measure(after, args.repeats),
                        "exact_output_and_metadata": True})
    paths = ("src/cellstine/moire/builder/generator.py", "src/cellstine/moire/builder/make.py",
             "src/cellstine/moire/builder/nlayer.py", "benchmarks/benchmark_builder_resources.py")
    print(json.dumps({"baseline": commit, "baseline_source_hashes": old_hashes,
        "source_hashes": {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in paths},
        "python": sys.version, "numpy": np.__version__, "platform": platform.platform(), "records": records,
        "limits": "Warm local filesystem, no fsync; Python-tracked peak not RSS. Baseline builder/geometry modules come from Git; other dependencies shared. Batch timings include POSCAR/metadata readback and owned-directory cleanup, with normal production GC policy. Expanded rows cycle actual small-cell records with unique indexes; sequential builders only, no parallel speedup claim."}, indent=2))


if __name__ == "__main__":
    main()
