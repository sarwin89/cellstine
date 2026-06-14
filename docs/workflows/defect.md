# Defect Workflow

The `defect` workflow analyses inequivalent defect sites and generates representative defect structures.

## Common Commands

```bash
cellstine defect analyse output/examples/Au_Bulk_111_surface.vasp --structure-kind surface
cellstine defect preview runs/defect/<run-id>/manifest.json
cellstine defect generate runs/defect/<run-id>/manifest.json --defect-type vacancy
cellstine defect generate runs/defect/<run-id>/manifest.json --defect-type substitution --substitution-species Pt
```

## Notes

- Default generation is inequivalent-only.
- Native analysis is available without optional dependencies.
- Exact Wyckoff-style metadata requires optional symmetry backends where available.
- Surface adatoms reuse the interface adsorption-site analysis.
