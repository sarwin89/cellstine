"""Measure native I/O resources against a Git baseline with exact valid outputs.

Run: python benchmarks/benchmark_io_resources.py --baseline c7dec6c
Local measurements are observations, never CI timing thresholds.
"""

from __future__ import annotations

import argparse
import functools
import hashlib
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
from cellstine.io import converters, kpoints, native  # noqa: E402
from cellstine.io.models import StructureRecord  # noqa: E402


def _baseline_io(ref, module_name):
    if module_name not in {"converters", "kpoints", "native"}:
        raise ValueError("unsupported I/O baseline module")
    commit = subprocess.check_output(
        ["git", "rev-parse", "--verify", "--end-of-options", ref + "^{commit}"],
        cwd=ROOT, text=True,
    ).strip()
    source = subprocess.check_output(
        ["git", "show", commit + ":src/cellstine/io/" + module_name + ".py"],
        cwd=ROOT, text=True,
    )
    module = types.ModuleType("cellstine.io._" + module_name + "_baseline")
    module.__package__ = "cellstine.io"
    sys.modules[module.__name__] = module
    exec(compile(source, commit + ":" + module_name + ".py", "exec"), module.__dict__)
    return commit, module


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", default="c7dec6c")
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args(argv)
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    commit, old_converters = _baseline_io(args.baseline, "converters")
    _, old_native = _baseline_io(args.baseline, "native")
    _, old_kpoints = _baseline_io(args.baseline, "kpoints")
    records = []
    (ROOT / ".tmp").mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="io-benchmark-", dir=ROOT / ".tmp") as scratch:
        folder = Path(scratch)
        trajectory = folder / "trajectory.xyz"
        trajectory.write_text("2\nfirst frame\nC 0 0 0\nC 1 1 1\n" * 100000, encoding="utf-8")
        atom_count = 20000
        poscar = folder / "POSCAR"
        poscar.write_text(
            "large frame\n1\n1 0 0\n0 1 0\n0 0 1\nC\n" + str(atom_count)
            + "\nDirect\n" + "0.1 0.2 0.3\n" * atom_count
            + "\nCartesian\n" + "0 0 0\n" * atom_count, encoding="utf-8",
        )
        xyz_record = StructureRecord(
            comment="large molecule", lattice=np.eye(3), species=["C"], counts=[100000],
            positions_direct=np.zeros((100000, 3)), positions_cartesian=np.zeros((100000, 3)),
            coordinate_mode="Direct", selective_dynamics=False, selective_flags=None,
        )
        endpoints = np.arange(128 * 3, dtype=float).reshape(128, 3) / 128
        cases = []
        cases.append(("first-XYZ-frame-with-100k-frame-tail", [
            functools.partial(module.StructureConverter()._read_xyz, trajectory, vacuum=20.0)
            for module in (old_converters, converters)
        ], "structure"))
        cases.append(("XYZ-100k-label-write", [
            functools.partial(module.StructureConverter()._write_xyz, xyz_record, folder / (str(i) + ".xyz"))
            for i, module in enumerate((old_converters, converters))
        ], "bytes"))
        cases.append(("POSCAR-20k-atoms-with-velocity-tail", [
            functools.partial(module.read_poscar, str(poscar)) for module in (old_native, native)
        ], "structure"))
        source = native.read_poscar(str(poscar))
        cases.append(("homogeneous-c-repeat-20k-by-3", [
            functools.partial(module.repeat_structure_along_c, source, 3)
            for module in (old_native, native)
        ], "structure"))
        cases.append(("64-line-segments-by-2048-points", [
            module.KpointsFile(comment="path", mode="line", line_divisions=2048, endpoints=endpoints).line_points
            for module in (old_kpoints, kpoints)
        ], "array"))
        for name, operations, kind in cases:
            before, after = operations[0](), operations[1]()
            if kind == "bytes":
                assert before.read_bytes() == after.read_bytes(), name
            elif kind == "array":
                assert np.array_equal(before, after), name
            else:
                for field in ("lattice", "positions_direct", "positions_cartesian"):
                    assert np.array_equal(getattr(before, field), getattr(after, field)), (name, field)
                for field in ("comment", "species", "counts", "coordinate_mode", "selective_dynamics", "selective_flags"):
                    assert getattr(before, field) == getattr(after, field), (name, field)
            records.append({"case": name, "baseline": _measure(operations[0], args.repeats),
                            "current": _measure(operations[1], args.repeats)})
    hashes = {}
    for name in ("converters", "native", "kpoints", "orientation"):
        source = (ROOT / "src/cellstine/io" / (name + ".py")).read_text(encoding="utf-8-sig")
        hashes[name] = hashlib.sha256(source.encode("utf-8")).hexdigest()
    print(json.dumps({"baseline_commit": commit, "current_source_sha256": hashes,
                      "python": sys.version.split()[0], "numpy": np.__version__,
                      "records": records}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
