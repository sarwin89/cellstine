"""Compare provenance resources with a trusted Git baseline; assert exact outputs.

Run: python benchmarks/benchmark_infrastructure.py --baseline 4d005c0
Inputs precede tracing. Local timings are observations, never CI thresholds.
"""

from __future__ import annotations

import argparse
import functools
import hashlib
import json
import sys

import numpy as np

from benchmark_geometry_memory import ROOT, _baseline, _measure
from cellstine.core import provenance


def _repeat_call(operation, count):
    result = None
    for _ in range(count):
        result = operation()
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", default="4d005c0")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--micro-calls", type=int, default=1000)
    args = parser.parse_args(argv)
    if args.repeats < 1 or args.micro_calls < 1:
        parser.error("repeat counts must be positive")
    commit, previous = _baseline(args.baseline, "provenance")
    oversized = [f"discard-{index}-" + "x" * 1000 for index in range(1000)]
    oversized += ["kept:first", "kept:second"]
    unique = [f"note-{index}" for index in range(8000)]
    repeated = " | ".join(unique[:4000] * 2)
    history = " | ".join(unique[:5000])
    appended = unique[:5000] + ["new", "new"]
    cases = [
        ("bounded-history", "_assemble", ("build", oversized), 1),
        ("unique-notes", "comment_notes", (" | ".join(unique),), 1),
        ("repeated-notes", "comment_notes", (repeated,), 1),
        ("restage-history", "restage_comment", (history, "build", *appended), 1),
        ("ordinary-stage", "stage_comment", ("build", "Cú", "Miller (1 1 1)"), args.micro_calls),
        ("ordinary-restage", "restage_comment", ("handwritten", "build", "new"), args.micro_calls),
    ]
    records = []
    for name, method, inputs, calls in cases:
        before = functools.partial(getattr(previous, method), *inputs)
        after = functools.partial(getattr(provenance, method), *inputs)
        assert before() == after(), name
        operations = [functools.partial(_repeat_call, operation, calls) for operation in (before, after)]
        records.append({"case": name, "calls_per_sample": calls,
                        "baseline": _measure(operations[0], args.repeats),
                        "current": _measure(operations[1], args.repeats)})
    hashes = {}
    for name in ("base", "validation", "provenance"):
        source = (ROOT / "src/cellstine/core" / (name + ".py")).read_text(encoding="utf-8-sig")
        hashes[name] = hashlib.sha256(source.encode()).hexdigest()
    print(json.dumps({"baseline_commit": commit, "current_source_sha256": hashes,
                      "python": sys.version.split()[0], "numpy": np.__version__,
                      "records": records}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
