# Moire Workflow

The `moire` workflow finds and builds commensurate bilayer and N-layer supercells.

## Common Commands

```bash
cellstine moire find input/examples/mos2.vasp input/examples/mos2.vasp --nindex 8
cellstine moire make runs/moire/<run-id>/manifest.json --indexes 1 --interlayer-distance 3.35
cellstine moire findn input/examples/mos2.vasp input/examples/graph.vasp input/examples/mos2.vasp --match-mode base_shared --nindex 8
cellstine moire translate output/examples/mos2x_mos2_stack_0deg_atoms6.vasp --shift-direct 0.333,0.667
```

## Notes

- Strain tolerances are fractions, so `0.02` means 2 percent.
- `findn` defaults to `base_shared`, which is the recommended N-layer mode.
- Adsorbate assembly may reuse the moire finder engine, but molecule placement lives in the `adsorbate` workflow.
- See [moire search performance](../moire-performance.md) for the optimized angle-search model and performance knobs.
