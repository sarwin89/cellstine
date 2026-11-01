# CELLSTINE systematic optimization plan

**Goal:** Review every tracked non-test Python function in dependency order, retaining behavior and scientific conventions, with measurable changes, tests, and separately verified dev commits.

**Architecture:** Continue in the isolated dev checkout; keep the shared dirty main checkout untouched. Generate the complete function inventory from Git and Python ASTs, including methods, nested functions, async functions, and lambdas. Review evidence lives beside the machine-readable ledger, with module hashes preventing stale review claims.

**Tech stack:** Python 3.10–3.14, NumPy, pytest/coverage, setuptools, GitHub Actions.

**Spec:** The active chat goal; its complete requirements are recorded in `docs/optimization/README.md`.

## Constraints

- Preserve APIs, scientific tolerances, file formats, ordering, and user-owned scientific data.
- Include support/benchmark/tool functions; exclude tests from the review denominator.
- Pending review is explicit deferral with a reason, never inferred completion from coverage.
- Full tests and relevant package checks precede each commit; dev CI must pass before the next part.
- No package/release publication, tags, deployment, or credentials configuration.

## Review focus

Extreme valid numerical scales, skew/near-boundary cells, deterministic selection/ties, malformed inputs and atomic failure, optional-dependency and installed-artifact behavior.

## Ordered parts

- [x] 01: Inventory/tooling, lattice helpers and reduction kernels; add Python 3.13/3.14 CI and line/branch coverage; write publication plan. Commit `9f59701`; all five Python jobs, package checks and retained coverage artifact verified in CI run `41050772517`.
- [x] 02: Periodic geometry, site indexes, neighbor kernels, contacts and covering/void kernels.
  - [x] 02a: Complete function review of `core/geometry.py`, bounded query/image allocations, sparse-grid heuristics and exact comparative benchmark. Commit `be43f26`; all five Python jobs and package checks green in run `41056032389`; coverage artifact `12976848250` retained.
  - [x] 02b: Remaining contact functions, covering/void kernels and exceptional reduction-fallback guarantee investigation. Commit `a455f81`; all five Python jobs and package build green in run `41175668140`; coverage artifact `13033086742` retained. Local full suite passed 3,175 tests; wheel/sdist/Twine and installed smokes pass.
  - [x] 02c: Complete void support and bounded candidate processing. Commit `1db9be4`; all five Python jobs and package build pass in run `41283962742`; coverage artifact `13074945312` matches the exact head. Local suite: 3,190 passed; package/install and ledger checks pass.
- [ ] 03: Remaining shared models, validation, species, transforms, lattice/symmetry primitives, provenance/reporting.
  - [ ] 03a: Structure models, species/element helpers, transforms, observation directions and vacuum primitives (40 existing functions); evidence-first resource/numerical repairs.
  - [ ] 03b onward: Workflow base, dependency/validation utilities, metadata/reporting and remaining lattice/symmetry primitives.
- [ ] 04: I/O registry, VASP/native/conversion/orientation and KPOINTS writers.
- [ ] 05: Moire search, ranking/results, builders, transforms and public workflows.
- [ ] 06: Surfaces and interface matching/building/registry workflows.
- [ ] 07: Molecules, adsorbate placement/assembly/movement.
- [ ] 08: Defect analysis, sites, supercells, generation and paths.
- [ ] 09: Symmetry workflows, records/adapters and reciprocal-space stages.
- [ ] 10: Visualization backends/results, CLI spec/parser/dispatch/guided flows, launcher and benchmark helpers.
- [ ] 11: Reconcile all ledger rows against tracked source; audit every requirement; final coverage, full supported-version CI, package/install checks and final report of findings/limitations.

Each part may split into smaller independently reviewed commits. For each changed function: reproduce the defect or resource regression first, implement narrowly, compare old/new behavior and benchmark when applicable, run focused/full/package checks, update evidence, commit/push dev, then inspect CI. Leave sound functions unchanged.
