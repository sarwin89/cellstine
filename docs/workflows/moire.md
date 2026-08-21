# Moire workflow

The `moire` workflow searches and builds commensurate **bilayer** supercells
with the native Gram-form engine.

## Find and make

```bash
cellstine moire find input/examples/mos2.vasp input/examples/mos2.vasp --max-length 20 --top-strain 0.01 --bottom-strain 0.01
cellstine moire make runs/moire/<run-id>/results.json --indexes 1 --interlayer-distance 3.35
```

`moire find` writes schema-versioned `results.json`. Pass that JSON directly to
`moire make`; previews and visualizers use the same validated results reader.
The guided interface is a simple launcher over this workflow.

## Strain and symmetry

In the moire CLI, **strain** means principal logarithmic strain,
`h = log(lambda)`, for a principal stretch `lambda`. An accepted candidate's
relative principal strain is bounded by the sum of `--top-strain` and
`--bottom-strain`. The engine shares it optimally between the two layers.

`--symmetric` requests a restricted square/hexagonal symmetry-preserving
family. When that family does not apply to the inputs or bounds, the search
records the reason and falls back to the general search.

N-layer moire workflows are not supported in this release. Use native bilayer
`moire find` followed by JSON `moire make`.

## Related commands

Translate the top group of an existing bilayer stack in direct coordinates:

```bash
cellstine moire translate output/examples/mos2x_mos2_stack_0deg_atoms6.vasp --shift-direct 0.333,0.667
```

Adsorbate assembly may reuse commensuration machinery, but molecule placement
lives in the `adsorbate` workflow. For reproducible scaling and candidate-class
checks, see [moire search performance](../moire-performance.md).
