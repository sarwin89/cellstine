# Aristotle migration timeline

This is the condensed timeline for the optimized Aristotle delivery and the
repository work that led into this migration. The Lean sources remain outside
the Python package, under the ignored local folder
`aristotle-lean-reference/`.

## Repository history before the optimized replacement

- 2025-05-12 to 2025-05-13: initial moiré package uploads and early refactors,
  including `cellfind.py`, `results.dat` cleanup, and the first `.gitignore`.
- 2026-03-17 to 2026-04-11: versioned releases through the 4.x line, plus
  generated-file cleanup and removal of legacy trilayer modules.
- 2026-06-14 to 2026-07-01: 6.x releases and the package-tree restructure that
  became the shared base for the current `main`/`dev` branches.
- 2026-08-21: first Gram-form replacement line on `dev`: generated artifacts
  were untracked, the native Gram search and JSON workflow were introduced,
  legacy `.dat` readers were retired, CLI controls were moved to physical
  `--max-length`/`--*-strain` bounds, and public docs were updated.

## Aristotle optimized delivery

- Initial optimized tree: Aristotle delivered a full package-shaped
  implementation under `src/cellstine-optimised`, with the Lean library in
  `aristotle-lean-reference/RequestProject/` and Python modules covering `core`, `io`, `moire`,
  `adsorbate`, `interface`, `defect`, `symmetry`, and `visualize`.
- Formal reference pass: the Lean project was reported as building end to end;
  the donor notes record 58 Lean sources, every one imported by
  `aristotle-lean-reference/RequestProject/Main.lean`, and no unfinished `sorry`, `admit`, or `axiom`.
- Moiré engine: the old angle/`nindex` path was replaced by staged Gram-form
  search, symmetry folding, exact acceptance checks, JSON result persistence,
  affine-based builders, symmetric-subfamily search, and an exact base-anchored
  N-layer extension using sublattice intersections.
- Interface work: slab and surface construction were expanded with exact
  surface-plane cells, stacking registry enumeration, mirrored/relative
  stacking controls, registry-based interface building, and interface match
  ranking.
- Defect work: defect analysis and generation were expanded around native
  layer partitioning, supercell/image-separation logic, vacancy/substitution/
  interstitial/adatom/divacancy handling, migration paths, and structured
  JSON records.
- Adsorbate work: molecule framing and placement were consolidated around
  contact distances, rigid orientation controls, adsorbate movement, and
  substrate fit/repeat behavior.
- Core and symmetry work: reusable geometry, layer, reciprocal, Bravais,
  K-path, covering-radius, void-search, and point-group helpers were split
  into focused modules. `core/symmetry3d.py` was later split so bare-lattice
  point-group logic lives in `core/pointgroup3d.py`.
- Verification growth: the donor notes show the suite growing from hundreds of
  tests to more than 3,000 tests, with separate tests for Lean-source hygiene,
  import isolation, numerical kernels, CLI behavior, builders, JSON schemas,
  and end-to-end workflows.

## Migration into the normal repository layout

- The Python package was migrated from the donor tree into `src/cellstine`.
  The donor folder itself remains ignored and untracked.
- The Lean source was copied to `aristotle-lean-reference/`, also ignored and
  untracked, so the Python package cites it as provenance without shipping it
  as package source.
- The optimized tests were moved into the normal `tests/` layout and their
  flat-tree bootstrap was adapted to load `src/cellstine`.
- One real migration fix was required: the spglib symmetry-analysis path now
  records centering translation count and symmorphic-setting metadata, matching
  the native analysis metadata expected by the optimized tests.
- Current migration verification:
  - `python -m pytest tests -q --basetemp='.pytest-migrated-full' -p no:cacheprovider`
    passed with 3188 passed and 2 skipped.
  - `python benchmarks/benchmark_gram_search.py` agreed with the independent
    brute-force oracle class-for-class. On this host, representative hexagonal
    searches measured roughly 9x to 128x faster than the reference baseline.
