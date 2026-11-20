# CELLSTINE v3.1.0

This release updates moiré result storage, adds parallel bilayer builds, and
improves the correctness and memory use of structure workflows.

## Migration

Bilayer search and molecular assembly now default to NPZ result archives,
reported as `results_npz` in manifests. Existing JSON results remain readable.
Pass `--results-format json` to retain `results.json` and the `results_json`
artifact key in scripts that depend on them.

## Highlights

- Bilayer builds can use multiple workers for calculation and file writes.
  Use `--workers` and `--parallel-backend` to select the worker count and
  process or thread execution. Outputs retain candidate order.
- Periodic geometry, surface-site enumeration and divacancy discovery use less
  temporary memory. In the measured 1,000-atom no-pair divacancy case, traced
  peak allocation fell by 71% and discovery time by 47%.
- Interface builds honor requested stacking senses and improve physical
  placement for tilted cells. Manual divacancy selection resolves displayed
  site IDs to their representative atoms.
- Guided navigation and visualization handle more edge cases. Generated
  viewers preserve text and geometry, and Matplotlib respects a configured
  interactive backend when `--show` is requested.

The [release preparation audit](optimization/release-preparation-audit.md)
documents verification and remaining limitations, including skewed surface
fallbacks, extreme numerical ranges and the secondary supercell shape ranking.
