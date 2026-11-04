# Systematic function review

The goal is to inspect every tracked non-test Python function, one subsystem at a time, and optimize only when evidence supports a change. APIs, scientific tolerances, file formats, output ordering, unrelated edits, and scientific run data must be preserved. Each part requires focused tests, a full suite, relevant build/install checks, a separate dev commit, and passing CI before the next part. Completion also requires a requirement-by-requirement final audit, a final findings/limitations report, broader Python/test coverage, and a publication plan. Publication is not authorized.

## Auditable scope

`function-review.json` includes every `FunctionDef`, `AsyncFunctionDef`, and `Lambda` in tracked Python files outside `tests/`: package source, checkout launcher, benchmark helpers, and review tooling. Nested functions and methods are separate entries. Python files with no functions are covered by the source inventory but contribute no function rows. Generated and ignored data is never scanned.

The authoritative target is `dev`'s tracked source. Unmerged concurrent edits in the shared `main` checkout remain preserved. If integrated later, changed module hashes require re-review; they cannot silently inherit the dev ledger's conclusions.

Every row records the path, qualified name, source lines, normalized module SHA256, status, reason, and evidence. All existing records start as **deferred—with reason** (pending review). Reviewed functions become **reviewed—unchanged** or **optimized—verified**. A substantive investigation deferred for a documented reason carries evidence; a pending entry does not. Counts must distinguish those cases.

Run from the repository root:

```powershell
python tools/review_inventory.py
python tools/review_inventory.py --check
```

The generator preserves evidence only when the normalized source module is unchanged. Changed source invalidates its module's reviews. New/deleted functions alter the ledger. `--check` fails on drift, invalid status, missing reason, or completed reviews without evidence. Checkout line-ending differences do not invalidate reviews. After adding a source file, stage it before regenerating because the scope is Git-tracked files.

Passing tests or obtaining high coverage never automatically marks a function reviewed. Completion must also check external dependencies/callers and the actual reasoning in each part report.

At the Part03f local checkpoint, the ledger contains **1,215 functions**: **566 reviewed unchanged**, **270 optimized and verified**, **2 investigated deferrals**, and **377 pending reviews**. Thus **838 functions have inspection evidence**; the project-wide review is incomplete. Part03f reviews47 callables, with3,610 full tests, package/three installed checks and280 frozen Python hashes passed; scoped commit/push and exact-head16-job CI remain next. Part05e commit1a2c3bb passed all16 CI41782577167 jobs and downloaded coverage artifact13299884630 has verified head and ZIPdigest7545a47e915103c01e395786975db837207772d20084172330ca5b1b24362fb4. Part05d commit8bfcf17 likewise passed all16 jobs, including nine Windows/Linux/macOS portable-result lanes; earlier checkpoints remain linked below. The investigated deferrals are rational lattice basis and extreme-scale structure validation. Part03c resolved the output-suffix deferral in actual default-path consumers; lower-level working-directory/failure-artifact limits remain explicit. Part02c resolved neighbour-count truncation within its documented limits.

## Dependency order and evidence

The full sequence is in [the implementation plan](../superpowers/plans/2026-10-30-systematic-review.md). Part reports link exact reviewed functions, callers, invariants, tests, changes, measurements and limitations.

| Part | Area | Evidence |
| --- | --- | --- |
| Prior scoped pass | Contact diameter and cross-group minimum search | [release-preparation report](../release-preparation-2026-10-30.md), commit `b749da1`, green supported-version CI |
| 01 | Lattice helpers, reduction kernels, review tooling, coverage/CI setup | [part-01-lattice.md](part-01-lattice.md) |
| 02a | Periodic geometry, site indexes, neighbor queries and comparative benchmark | [part-02a-geometry.md](part-02a-geometry.md) |
| 02b | Contacts, covering, void review and exact reduction fallback | [part-02b-contacts-covering.md](part-02b-contacts-covering.md) |
| 02c | Complete void support, bounded enumeration and priority-preserving classification | [part-02c-void-completeness.md](part-02c-void-completeness.md) |
| 03a | Shared structure transforms, directions/vacuum, species/elements and canonical structure model | [part-03a-shared-structure.md](part-03a-shared-structure.md) |
| 03b | Workflow base, dependency discovery, validation, reports, provenance and manifests | [part-03b-shared-infrastructure.md](part-03b-shared-infrastructure.md) |
| 03c | Actual default-output reservations across workflow and direct API consumers | [part-03c-safe-default-outputs.md](part-03c-safe-default-outputs.md) |
| 03d | Exact reciprocal keys, rotations/images, supported integer ranges and bounded grid allocations | [part-03d-reciprocal.md](part-03d-reciprocal.md) |
| 03e | Shared conventional-axis clouds, prompt group caps and complete Bravais/idealisation review | [part-03e-conventional.md](part-03e-conventional.md) |
| 03f | Shared2D/3D symmetry primitives, bounded ordered integer boxes/matching, reused site indexes and exact componentwise centering | [part-03f-shared-symmetry.md](part-03f-shared-symmetry.md) |
| 04 | Native I/O, molecular temporary ownership, frame/format integrity and bounded allocations | [part-04-io-integrity.md](part-04-io-integrity.md) |
| 05a | Engine/results integration, exact signed primitive keys, early invariant filters and bounded JSON encoding | [part-05a-search-fast-paths.md](part-05a-search-fast-paths.md) |
| 05b | Bounded report writing, scalar-fallback validation, unfolded persistence and defect regrouping | [part-05b-persistence.md](part-05b-persistence.md) |
| 05c | Complete guided CLI integration, actual-row index bounds, replay/retry recovery and bounded recent-file selection | [part-05c-guided-cli.md](part-05c-guided-cli.md) |
| 05d | Optional no-pickle NPZ, complete consumer routing and cross-platform format checks | [part-05d-portable-results.md](part-05d-portable-results.md) |
| 05e | Exact HNF, bounded ordered cosets, read-once batch builders, safe worker failures and complete layer helper review | [part-05e-builder-resources.md](part-05e-builder-resources.md) |
| Remaining | Numerical-domain follow-ups and remaining shared code, moire, interfaces, adsorbates, defects, symmetry, visualization and CLI/support | Pending dependency-ordered review |

Portable-format follow-up: [I/O measurements and optional NPZ plan](portable-io-options.md). The prototype investigation is followed by the production implementation and evidence in Part05d.

## Coverage and release boundaries

CI runs the full suite on Python 3.10–3.14 on Windows. The 3.12 job additionally measures package line/branch coverage and uploads `coverage.xml`. A separate Python 3.12 / NumPy 1.26.4 lane exercises structure, infrastructure and persistence helpers against the minimum supported NumPy series. Nine portable-result lanes additionally exercise Windows, Linux and macOS with Python3.10,3.12 and3.14 (3.12 pins NumPy1.26.4); all nine passed at Part05d commit8bfcf17, with36tests and2 optional-render skips each. Coverage exposes gaps to investigate; no arbitrary percentage is used as a substitute for scientific assertions or function review. Subprocess-only CLI execution may require separate measurements before a claim of complete coverage.

See [the publication plan](../publication-plan.md) for candidate selection, versioning, validation, installed artifact checks, and the later explicitly authorized publication sequence. Local review artifacts and scientific outputs remain ignored.
