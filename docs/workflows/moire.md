# Moire workflow

The `moire` workflow searches and builds commensurate **bilayer** supercells
with the native Gram-form engine.

## Search and build

```bash
cellstine moire search input/examples/mos2.vasp input/examples/mos2.vasp --length 20 --strain 0.01
cellstine moire build runs/moire/<run-id>/results.json --indexes 1 --interlayer-distance 3.35
```

`moire search` writes schema-versioned `results.json`. Pass that JSON directly
to `moire build`; previews and visualizers use the same validated results reader.
The guided interface is a simple launcher over this workflow.

## Strain and symmetry

In the moire CLI, **strain** means principal logarithmic strain,
`h = log(lambda)`, for a principal stretch `lambda`. An accepted candidate's
relative principal strain is bounded by the selected strain budget. Use
`--strain E` for the common equal-budget case, `--rigid` for zero strain, or
`--top-strain` with `--bottom-strain` for asymmetric expert searches. The
engine shares accepted strain optimally between the two layers.

`--symmetric` requests a restricted square/hexagonal symmetry-preserving
family. When that family does not apply to the inputs or bounds, the search
records the reason and falls back to the general search.

N-layer moire workflows are not supported in this release. Use native bilayer
`moire search` followed by JSON `moire build`.

## Related commands

Translate the top group of an existing bilayer stack in direct coordinates:

```bash
cellstine moire shift output/examples/mos2x_mos2_stack_0deg_atoms6.vasp --shift-direct 0.333,0.667
```

Adsorbate assembly may reuse commensuration machinery, but molecule placement
lives in the `adsorbate` workflow. For reproducible scaling and candidate-class
checks, see [moire search performance](../moire-performance.md).
