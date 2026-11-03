"""Compare guided file discovery against an exact Git baseline on owned trees."""

from __future__ import annotations

import argparse
import ast
import functools
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from benchmark_geometry_memory import _measure  # noqa: E402
from cellstine.cli.interactive.prompts import _find_candidates  # noqa: E402


def _baseline(commit):
    source = subprocess.check_output(
        ["git", "show", commit + ":src/cellstine/cli/interactive/prompts.py"],
        cwd=ROOT, text=True, encoding="utf-8",
    )
    function = next(node for node in ast.parse(source).body
                    if isinstance(node, ast.FunctionDef) and node.name == "_find_candidates")
    namespace = {"Path": Path, "Sequence": Sequence}
    exec(compile(ast.Module(body=[function], type_ignores=[]), commit + ":prompts.py", "exec"), namespace)
    return namespace["_find_candidates"]


def _populate(directory, count):
    directory.mkdir()
    for index in range(count):
        path = directory / f"POSCAR{index:06d}.vasp"
        path.touch()
        stamp = 100000 + index % 13
        os.utime(path, (stamp, stamp))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", default="837be429")
    parser.add_argument("--files", type=int, default=2000)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args(argv)
    if args.files < 16 or args.repeats < 1:
        parser.error("files must be at least16 and repeats must be positive")
    commit = subprocess.check_output(
        ["git", "rev-parse", "--verify", "--end-of-options", args.baseline + "^{commit}"],
        cwd=ROOT, text=True, encoding="utf-8",
    ).strip()
    old = _baseline(commit)
    evidence = []
    # Own all generated files; never traverse or modify the user's run folders.
    with tempfile.TemporaryDirectory(prefix="cellstine-discovery-") as temporary:
        base = Path(temporary)
        small, partial, dense = base / "input", base / "partial", base / "runs"
        _populate(small, 16)
        _populate(partial, 3)
        _populate(dense, args.files)
        for name, roots in (("filled-input", (small, dense)), ("dense-input", (dense,)),
                            ("partial-input", (partial, dense)), ("overlapping-roots", (partial, base))):
            operations = [functools.partial(function, ("*.vasp", "POSCAR*"), roots, limit=8)
                          for function in (old, _find_candidates)]
            assert operations[0]() == operations[1](), name
            evidence.append({"case": name, "exact_paths_and_order": True,
                             "baseline": _measure(operations[0], args.repeats),
                             "current": _measure(operations[1], args.repeats)})
    print(json.dumps({"baseline": commit, "python": sys.version, "files_in_dense_tree": args.files,
                      "records": evidence, "limits": "Warm filesystem metadata; Python-tracked peak, not RSS."}, indent=2))


if __name__ == "__main__":
    main()
