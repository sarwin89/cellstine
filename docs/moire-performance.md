# Moire search performance

The native Gram-form search enumerates reduced bilayer cell shapes up to the
physical length bound and joins compatible top and bottom forms. Search cost
therefore depends on `--length`, the lattice geometry, the sum of
`--top-strain` and `--bottom-strain`, and optional cell-shape or atom-count
bounds.

Here **strain** is principal logarithmic strain, `h = log(lambda)`. The accepted
relative principal strain is bounded by the sum of the two layer budgets and
shared optimally between the layers. A representative native workflow is:

```bash
cellstine moire search input/examples/mos2.vasp input/examples/mos2.vasp --length 20 --strain 0.01 --progress
cellstine moire build runs/moire/<run-id>/results.npz --indexes 1 --interlayer-distance 3.35
```

The default is NPZ. Add `--results-format json` to the search for readable
`results.json`; build and view accept either format or a run manifest.

The finder writes portable `results.npz` by default. `--symmetric` requests a
restricted square/hexagonal family and falls back to the general search when it
is inapplicable. Experimental N-layer commands are exposed as
`moire stack-search` and `moire stack-build`, but performance claims here are
only for the native bilayer Gram-form search.

## Reproducible benchmark

Run from the repository root:

```bash
python benchmarks/benchmark_gram_search.py
```

The benchmark compares independent canonical candidate classes from a direct
reference enumeration with the native Gram search at three increasing length
bounds. It stops on mismatch. Reported timings and scaling ratios are measured
wall-clock values, so compare candidate-class equality first and treat speed as
host-dependent evidence rather than a fixed performance promise.

For the algorithm and JSON fields, see the
[native Gram-form search note](../src/cellstine/moire/MOIRE_SEARCH.md).


## Concurrent structure builds

`moire build` calculates and writes each selected structure independently. Use
`--workers 4` for a process pool or `--workers 4 --parallel-backend thread` for
threads. Both return results in the requested index order and accept JSON, NPZ,
and compressed NPZ. The Python equivalent is
`Moire.make(..., workers=4, parallel_backend="thread")`.

The defaults remain one worker and the process backend. Process pools require a
normal multiprocessing main guard in Python scripts on Windows; threads avoid
that requirement. Pools start for each batch, so small selections can be faster
with one worker. Threads share the caller's numerical runtime and do not change
its native thread limits. Set numerical-library environment variables before
importing NumPy if controlling native threads yourself. The process initializer's
environment defaults cannot reconfigure a native runtime that is already loaded.

Actual pool size is capped to the selection length, and at most twice that many
tasks are pending. The result snapshot, requested indexes, and returned result
list still occupy memory proportional to the batch size. Each task reserves its
own output file. A failure can leave partial files and outputs from already
started tasks; pending tasks are cancelled and completed writes are never retried.
Duplicate candidates can receive filename suffixes in completion-dependent order,
as with the existing process backend; the returned list preserves request order.

A local warm Windows fixture with 12 structures of roughly 600 atoms each retained
exact coordinates, species, flags, contacts and requested order across backends.
The final implementation with inherited native-thread settings took 2.60s serial,
1.88s with two threads and 1.23s with four threads (three-repeat medians). A separate
controlled run with native threads set to one before import also retained exact
outputs and showed useful thread scaling. These local observations include pool
startup and writes but exclude fsync and peak worker memory measurement; they are
not a universal speedup promise. See the [optimization checkpoint](optimization/part-05g-bounded-concurrency.md)
for the fixture, baseline comparisons and limitations.
