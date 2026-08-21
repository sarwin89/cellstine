# Native Gram-form moire search

CELLSTINE 4 searches commensurate **bilayer** supercells directly in Gram form.
The twist angle is an output of each accepted pair of integer supercell
matrices, not an angle swept by the user.

## Native workflow

Run a bounded search and then build candidates from its JSON:

```bash
cellstine moire find TOP.vasp BOTTOM.vasp --max-length 20 --top-strain 0.01 --bottom-strain 0.01
cellstine moire make runs/moire/<run-id>/results.json --indexes 1 --interlayer-distance 3.35
```

The result is schema `cellstine.moire.gram`, version 1, in `results.json`.
Previews, the builder, and both visualization paths use the same validated
central reader. Legacy positional `.dat` input is rejected with an instruction
to rerun native `moire find`.

The JSON records the top and bottom integer matrices, their Gram triples, angle
in degrees, relative principal strain pair, strain budgets and sharing fraction,
layer and total atom counts, rank and Pareto flag, Löwner certification,
recorded affine maps, shared lattice, and search metadata. No reader depends on
legacy positional columns.

## What strain means

In the moire CLI, **strain** is the principal logarithmic strain

```text
h = log(lambda)
```

of the relative deformation's principal stretch `lambda`. A budget `e`
allows stretches from `exp(-e)` through `exp(e)`; it is not engineering
strain. Total accepted relative principal strain is bounded by the sum of the
top and bottom strain budgets. The engine shares that strain optimally between
the layers for each candidate. This naming is deliberate: it is scientifically
precise while keeping the CLI readable.

## Search outline

1. Enumerate reduced positive-definite Gram forms whose basis lengths satisfy
   `--max-length` and the optional cell-shape and atom-count bounds.
2. Join top and bottom forms using the Löwner inequalities implied by the sum of
   `--top-strain` and `--bottom-strain`.
3. Recover candidate matrices, principal stretches, relative logarithmic
   strains, optimal sharing, twist angle, and a common lattice.
4. Fold proper point-group equivalents when enabled, rank independent canonical
   candidate classes, and mark the Pareto frontier.

`--symmetric` is a restricted square/hexagonal symmetry-preserving family. If
the inputs or requested bounds make that family inapplicable, CELLSTINE records
the reason and falls back to the general search.

N-layer moire workflows are not supported in this release. Use the bilayer
`moire find` and JSON `moire make` workflow.

## Mathematical provenance

[Harmonic's Aristotle](https://aristotle.harmonic.fun/) and
[Lean 4](https://lean-lang.org/lean4/doc/) are cited only as an external
mathematical reference for the derivation and checking that informed the native
implementation. CELLSTINE executes Python/NumPy code and does not copy, vendor,
or require Lean source files.

## Reproducible comparison

From the repository root, run:

```bash
python benchmarks/benchmark_gram_search.py
```

The script compares independent canonical candidate classes from the reference
enumeration and the native Gram search. It stops on mismatch, reports actual
timings at three increasing length bounds, and treats speed numbers as
host-dependent measurements rather than fixed performance assertions.

The guided interface remains a simple launcher over the same native backend.
Its preview and static/interactive visualizations read `results.json`, show
angles in degrees, and use the terms top strain, bottom strain, and relative
principal strain.
