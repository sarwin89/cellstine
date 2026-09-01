# Codebase inventory

This inventory records the current repository shape after the Aristotle
replacement, CLI rework, and cleanup passes. It is meant to answer two
questions before future refactors start:

1. What is source code, test code, documentation, or local generated material?
2. Which areas are clean enough to keep stable, and which areas should be
   reorganized next?

The snapshot was taken on 2026-09-01 from `dev`, with the worktree clean against
`origin/dev`.

## Tracked repository shape

| Area | Tracked files | Role |
| --- | ---: | --- |
| `src/` | 139 | Package source and source-adjacent mathematical notes. |
| `tests/` | 87 | Regression, kernel, CLI, and end-to-end tests. |
| `docs/` | 13 | User-facing workflow, CLI, architecture, migration, inventory, technical, and performance docs. |
| `benchmarks/` | 3 | Reproducible moire search benchmark and oracle support. |
| `input/` | 4 | Tracked example inputs only. |
| root files | 8 | Project metadata, launch script, README, roadmap, license. |

The package uses a `src/` layout. The root [cellstine.py](../cellstine.py) is a
checkout-local launcher for `cellstine.cli.main:main`; the installed console
script remains `cellstine`.

## Package layout

```text
src/cellstine/
  cli/          CLI entrypoints, shared parser/spec, and guided builders
  core/         generic geometry, lattice, symmetry, manifests, reporting
  io/           VASP/native I/O and optional structure conversion
  moire/        bilayer Gram search, stack search, JSON results, builders
  interface/    surface generation, stacking registry, interface matching
  adsorbate/    molecule placement, movement, and assembly workflows
  defect/       defect-site analysis, generation, supercells, reporting
  symmetry/     public symmetry workflow and KPOINTS/KPATH stages
  visualize/    static and optional interactive visualization backends
```

Largest tracked source files, and the reason to watch them:

| File | Approx. lines | Why it matters |
| --- | ---: | --- |
| `core/geometry.py` | 947 | Central periodic-geometry kernels used by several workflows. |
| `symmetry/symmetry.py` | 887 | Public symmetry workflow plus optional/backend logic. |
| `adsorbate/placement/operations.py` | 880 | Substrate preparation, molecule placement, and movement helpers. |
| `interface/surface/surface_cell.py` | 866 | Miller-plane and slab-cell construction logic. |
| `cli/plain.py` | 860 | Full stdlib command grammar. |
| `core/kpath.py` | 853 | Brillouin-zone, special-point, and band-path derivation. |
| `core/symmetry3d.py` | 848 | Native 3D symmetry operations, primitive cells, and planar gauges. |

These files are not inherently wrong, but each combines enough concerns that
future edits should be isolated and heavily tested.

## Documentation layout

Current public docs are split across three locations:

- root: [README.md](../README.md), [USAGE_GUIDE.md](../USAGE_GUIDE.md),
  [ROADMAP.md](../ROADMAP.md)
- `docs/`: quickstart, CLI, workflow, architecture, performance, and migration
  notes
- `src/cellstine/**.md`: source-adjacent mathematical notes

The preferred direction is:

```text
docs/
  README.md
  quickstart.md
  cli.md
  codebase-inventory.md
  architecture.md
  roadmap.md or root ROADMAP.md
  workflows/
  technical/
```

Source-adjacent `.md` files should stay only when they are valuable beside the
implementation. The canonical reader-facing technical map should live under
`docs/technical/`.

## Ignored local material

These directories are intentionally ignored and should not be committed:

| Path | Files | Approx. size | Notes |
| --- | ---: | ---: | --- |
| `runs/` | 477 | 569 MB | Local manifests and generated workflow artifacts. |
| `scratch/` | 10007 | 400 MB | Includes `scratch/moire_manim/.venv` and rendered media. |
| `benchmark_nindex10_moire/` | 6533 | 99 MB | Legacy benchmark corpus and external-tool outputs. |
| `aristotle-lean-reference/` | 62 | 0.6 MB | External Lean proof/reference project; not package source. |
| `cellstine-report/` | 23 | 1.4 MB | Local report/export material. |
| `output/` | 0 | 0 MB | Generated structures/plots; ignored except curated examples elsewhere. |

Before deleting any of these, preserve anything the user wants to keep. The
safe cleanup target is Python cache folders such as `__pycache__/`.

## Current consistency findings

1. The tracked package and tests are cleanly separated from generated artifacts.
2. `USAGE_GUIDE.md` has been brought back in line with the CLI rework command
   names.
3. The N-layer moire implementation exists in code and tests, while some docs
   historically described it as unsupported/deferred. It is now documented as
   experimental, but the product decision remains: either stabilize its public
   contract or hide it from the public CLI.
4. The flat `tests/` directory is still workable, but it is ready for a
   mechanical domain split.
5. A single technical algorithm map was missing; see
   [technical/algorithms.md](technical/algorithms.md).

## Recommended reorganization sequence

Use one commit per phase:

1. Documentation consistency: inventory, technical map, updated usage guide.
2. Public contract decision: settle N-layer as either experimental or disabled.
3. Mechanical test split: move tests into domain folders without changing test
   logic.
4. Parser split: move group parser construction out of `cli/plain.py`.
5. Large-module splits by domain, starting with files that see the most edits.
