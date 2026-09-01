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
cellstine moire build runs/moire/<run-id>/results.json --indexes 1 --interlayer-distance 3.35
```

The finder writes schema-versioned `results.json`. `--symmetric` requests a
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
