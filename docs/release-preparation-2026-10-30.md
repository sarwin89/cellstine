# Release preparation review — 2026-10-30

## Functionality and scope

CELLSTINE exposes moire bilayer/stack searches and construction, adsorbate placement and movement, surface generation and interface matching, defect analysis/generation and migration paths, symmetry/cell reduction and reciprocal-space sampling, and optional visualization. Shared geometry, lattice reduction, contacts, validation, reporting, and I/O support these workflows. The public API uses lazy workflow imports; NumPy is the sole mandatory runtime dependency. N-layer moire support remains experimental.

This pass preserves public signatures, scientific tolerances, versions, and output schemas. The checkout already contained moire equivalence/search, symmetry2d, benchmark, documentation, and example-input work. Additional guided-navigation changes appeared during review. Those changes were retained; they are not attributed to this pass. Ignored scientific runs, output directories, and external Lean reference material were left untouched.

## Changes from this pass

- Fixed the guided adsorbate/defect migration menus: both referenced `_build_migration_path` without importing it. Regression tests exercise real menu answers and parse the resulting commands. Both tests failed with `NameError` before the fix.
- Replaced the full atom-pair displacement allocation in `core.contacts.group_diameter` with upper-triangle tiles. Unwrapping and Euclidean distance calculations retain their existing definitions. Pair computation remains quadratic, but temporary pair storage is bounded by tile size; the unwrapped coordinates still require linear storage. Only the final maximum needs a square root.
- Added independent diameter checks for empty, singleton, small, and multi-tile groups, plus a 1,600-atom peak-allocation regression. The memory test failed at 102,443,222 bytes before the change; its budget is 16 MiB.
- Resolved undefined annotation names in lattice reduction and symmetry, and removed unused imports. Retained `symmetry.vector_angle_deg` as an explicit re-export because existing callers/tests rely on that exposed name.

## Measured contact calculation

Windows, Python 3.14, NumPy 2.4.3; five calls per implementation, median wall time. Input: seeded random fractional coordinates in `[0.4, 0.6]`, a diagonal 10-Angstrom cell, seed 15. Original calculation: unwrap, allocate `points[:, None, :] - points[None, :, :]`, compute pair norms and maximum. Allocation measured separately with `tracemalloc`.

| Atoms | Original median | Tiled median | Original peak | Tiled peak |
| ---: | ---: | ---: | ---: | ---: |
| 32 | 0.121 ms | 0.152 ms | 0.073 MiB | 0.073 MiB |
| 513 | 8.610 ms | 4.714 ms | 10.05 MiB | 3.63 MiB |
| 1,600 | 58.621 ms | 34.105 ms | 97.69 MiB | 3.66 MiB |

All three returned identical diameters. These are local microbenchmarks, not whole-workflow performance guarantees. Small-group loop overhead is about 0.03 ms in this sample.

## Verification

- Initial full-suite run: 3,254 passed, 2 skipped, 81 dependency deprecation warnings, 309.93 seconds.
- Latest full-suite run: **3,273 passed, 2 skipped, 81 dependency deprecation warnings**, 352.10 seconds. Skips are the Lean index self-import case and a surface face whose layer counts are all whole periods.
- Red/green checks: both migration-menu failures and the memory-budget failure reproduced before fixes; focused tests passed afterward.
- Combined angle, guided navigation, migration, and contacts checks: 76 passed at that checkpoint.
- Closing CLI checks after concurrent navigation changes: **92 passed**, including end-to-end workflows; a separate CLI/startup/moire-results check passed 283 tests. An intermediate combined run had five failures: one removed angle-helper re-export and four guided-search answer scripts out of step with concurrent new prompts. The re-export was restored and the concurrently updated scripts pass in the later runs.
- `python -m ruff check src --select F`: passed after cleanup. This checks undefined names and related errors, not a new repository-wide style policy.
- All 140 package Python files parse with Python 3.10 syntax. Runtime type-hint resolution for `Symmetry.kpoints` succeeds.
- Wheel and source distribution build successfully; `twine check` passes for both. Wheel contents were compared byte-for-byte with all 140 package source files. Generated runs/output/temp data are excluded from both artifacts.

Clean core-only wheel installation passes console version/help, imports all eight primary workflow classes, and checks the diameter calculation with NumPy 2.5.4 and no Rich/spglib installed. Clean CLI-extra wheel installation passes version, top-level help, and help for all seven command groups, with Typer 0.27.3 and Rich 14.3.4. The final wheel matches current package source byte-for-byte; closing source/test hashes showed no additional changes after the final targeted checks.

Clean source-distribution installation also builds and installs successfully using offline build dependencies and a repository-local pip cache. The first installation attempt stalled in the sandbox's default wheel cache and was interrupted; the local-cache retry completed. Installed version/help and the numerical smoke pass, and dependency checks pass in all three environments.

Verification logs and local artifacts are under ignored `.tmp/release-prep-20261030/` and `.tmpdist/release-prep-20261030/` directories. Build-validation tools, wheel downloads, and clean virtual environments are local ignored artifacts; they do not alter project dependencies.

## Remaining release gates

Only Python 3.14 is installed locally. The supported Python 3.10/3.11/3.12 runtime matrix must pass CI before release; syntax parsing does not replace those runs. The original review did not commit or publish. The subsequent user-authorized dev follow-up is scoped separately below; it does not change versions, publish packages, or deploy anything.


## Scoped dev follow-up

The dev candidate is prepared in an isolated local checkout from `origin/dev` at `bb90941`. It includes this chat's contact optimizations, static import/annotation cleanup, guided migration-helper import fix, and their tests. Separate moire equivalence/search and guided-navigation work in the shared checkout is intentionally excluded from the commit.

`closest_contact` now reduces strips of at most 65,536 pairs (or one wider row). After finding an initial contact, it uses the existing bounded minimum-image kernel, which is exact inside the current winning radius and guarantees all other distances stay outside it. The result retains original row-major tie ordering, indices, species, and covalent-radius reporting. Working storage no longer contains the full cross-group matrix; one exceptionally wide second group still requires linear row storage.

Cross-group microbenchmark: Python 3.14 / NumPy 2.4.3, seed 12, uniform fractional coordinates, diagonal 10-Angstrom cell, C/O labels; three calls per implementation, median timing, separate tracemalloc measurement. Every Contact record matched the original full-matrix implementation exactly.

| Atoms per group | Original median | Bounded strips | Original peak | New peak |
| ---: | ---: | ---: | ---: | ---: |
| 32 | 0.714 ms | 0.722 ms | 0.56 MiB | 0.56 MiB |
| 513 | 148.550 ms | 50.412 ms | 19.32 MiB | 10.94 MiB |
| 1,600 | 1,537.282 ms | 199.626 ms | 65.72 MiB | 10.81 MiB |

The new memory regression failed before the optimization at 70,643,413 bytes against a 16-MiB budget. All 38 contact tests pass, including exact references on orthogonal, hexagonal, and strongly sheared cells with a winning pair beyond the first strip. Independent read-only review also found bit-for-bit agreement in fifteen additional multi-strip comparisons. Timings remain host- and workload-dependent.


Final isolated dev-candidate verification: **3,139 passed, 4 skipped, 81 spglib deprecation warnings**, 311.89 seconds on Python 3.14. Three skips concern the absent external Aristotle Lean reference; one is the whole-period surface face case. No generated or external reference files are included. Source/test hashes remained stable across verification. Ruff F checks and whitespace checks pass. Wheel/sdist builds and Twine checks pass; the wheel matches every package source file byte-for-byte. Clean core-only, CLI-extra, and source-distribution installations pass numerical/version/help checks and dependency checks with NumPy 2.5.4. The supported Python 3.10/3.11/3.12 matrix runs in CI after the requested dev push.
