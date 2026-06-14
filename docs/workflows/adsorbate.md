# Adsorbate Workflow

The `adsorbate` workflow places and moves molecules on substrate slabs.

## Common Commands

```bash
cellstine adsorbate place output/examples/Au_Bulk_111_surface.vasp input/examples/graph.vasp --site-type top --site-index 1 --height 2.5
cellstine adsorbate move output/examples/mos2x_mos2_stack_0deg_atoms6.vasp --target-direct 0.5,0.5 --rotate 30
cellstine adsorbate assemble input/examples/Au_Bulk.vasp --a-length 12 --b-length 12 --angle 60
```

## Notes

- Placement can use Cartesian or Direct coordinates depending on the command.
- Molecule movement is center-of-mass based.
- The substrate can be an existing slab or a bulk cell that is converted through the interface surface workflow.
- Site choices should come from analysed sites rather than being guessed blindly.
