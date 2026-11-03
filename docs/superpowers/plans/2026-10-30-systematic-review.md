# CELLSTINE systematic optimization plan

**Goal:** Review every tracked non-test Python function in dependency order, retaining behavior and scientific conventions, with measurable changes, tests, and separately verified dev commits.

**Architecture:** Continue in the isolated dev checkout; keep the shared dirty main checkout untouched. Generate the complete function inventory from Git and Python ASTs, including methods, nested functions, async functions, and lambdas. Review evidence lives beside the machine-readable ledger, with module hashes preventing stale review claims.

**Tech stack:** Python 3.10–3.14, NumPy, pytest/coverage, setuptools, GitHub Actions.

**Spec:** The active chat goal; its complete requirements are recorded in `docs/optimization/README.md`.

## Constraints

- Preserve APIs, scientific tolerances, file formats, ordering, and user-owned scientific data.
- Include support/benchmark/tool functions; exclude tests from the review denominator.
- Pending review is explicit deferral with a reason, never inferred completion from coverage.
- Batch related code work into larger checkpoints. One full test/package pass precedes each checkpoint commit; exact-head dev CI is checked before the next code checkpoint. Use CI wait time for the next area's read-only inspection and avoid repeated passing checks without new changes or evidence, following the user's2026-11-02 instruction.
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
  - [x] 03a: Structure models, species/element helpers, transforms, observation directions and vacuum primitives (40 original functions). Commit `4d005c0`; all seven jobs pass in run `41289513562`, coverage artifact `13076513125` exact-head XML matches local12377/14290statements and3206/4502branches.
  - [x] 03b: Workflow base, dependency/validation utilities, reports, provenance and manifests (44 original functions). Commit `a4a3a0e`, all seven CI jobs pass in run `41398913199`, exact-head coverage artifact `13120128556`; minimum NumPy lane257tests pass. Two evidenced deferrals were recorded for follow-up.
  - [x] 03c: Default output allocation/naming across consumers. Commit `c7dec6c`; all seven jobs passed in run `41405522888`, exact-head coverage artifact `13123961765` XML read back (12462/14352 statements,3238/4512 branches). Local suite3,270passed;25 new original reviews and two helpers, legacy parser values retained.
  - [x] 03d: Reciprocal checkpoint `bba6576`: all seven jobs passed in CI41427690267, exact-head artifact13133367158 XML matches12661/14531 statements,3340/4618 branches.26 originals/two helpers/benchmark;85 focused and3,331 full tests, package/install checks pass. See [report](../../optimization/part-03d-reciprocal.md).
  - [x] 03e: Conventional cells/idealisation checkpoint `c57d6a1`:21 originals/two benchmark functions;456 focused and3,339 full tests, package/install checks pass;257 frozen Python hashes match. All seven jobs pass in CI41439087850; downloaded exact-head coverage artifact13139281955 matches12665/14534 statements and3341/4618 branches ([plan](2026-11-02-conventional-cell-review.md)). Remaining shared/numerical-domain work stays pending; preserve scientific tolerances.
- [x] 04: I/O checkpoint `dc61527`: all seven jobs passed in CI41423449104; exact-head coverage artifact13130624746 XML matches12613/14474 statements,3321/4592 branches.37 originals/three helpers/two benchmark functions;127 focused and3,316 full tests, package/install checks pass. See [part report](../../optimization/part-04-io-integrity.md). Exceptional numerical-domain follow-up remains separate.
- [ ] 05: Moire search, ranking/results, builders, transforms and public workflows. Part05a checkpoint6692121:119function reviews,541 focused and3,374 full tests; package/install checks and all seven CI41585027655jobs pass, retained exact-head XML matches12927/14794statements3410/4686branches. Part05b local checkpoint:121reviews,273focused and3,406full tests,142module archive bytes/installed smokes pass;538/1148functions have inspection evidence. Scoped push and exact-head CI remain next. User handoff order:engine/results first, then JSON writer hunks by hand, then CLI. Scratch merge prepared against verified bases; preserve current dev mirror/fold/schema semantics and compare actual integrated outputs before closing rows ([integration plan](2026-11-02-claude-integration.md)).
- [ ] 06: Surfaces and interface matching/building/registry workflows.
- [ ] 07: Molecules, adsorbate placement/assembly/movement.
- [ ] 08: Defect analysis, sites, supercells, generation and paths.
- [ ] 09: Symmetry workflows, records/adapters and reciprocal-space stages.
- [ ] 10: Visualization backends/results, CLI spec/parser/dispatch/guided flows, launcher and benchmark helpers.
- [ ] 11: Reconcile all ledger rows against tracked source; audit every requirement; final coverage, full supported-version CI, package/install checks and final report of findings/limitations.

Each part may split into smaller independently reviewed commits. For each changed function: reproduce the defect or resource regression first, implement narrowly, compare old/new behavior and benchmark when applicable, run focused/full/package checks, update evidence, commit/push dev, then inspect CI. Leave sound functions unchanged.
