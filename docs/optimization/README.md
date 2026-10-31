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

At the Part-02b local checkpoint, the ledger contains **1,086 functions**: **100 reviewed unchanged**, **15 optimized and verified**, **2 investigated deferrals**, and **969 pending reviews**. Thus **117 functions have inspection evidence**; the project-wide review is incomplete. The new exact fallback helper is included in the denominator. The void-search neighbour cap is an explicit investigated deferral for Part 02c.

## Dependency order and evidence

The full sequence is in [the implementation plan](../superpowers/plans/2026-10-30-systematic-review.md). Part reports link exact reviewed functions, callers, invariants, tests, changes, measurements and limitations.

| Part | Area | Evidence |
| --- | --- | --- |
| Prior scoped pass | Contact diameter and cross-group minimum search | [release-preparation report](../release-preparation-2026-10-30.md), commit `b749da1`, green supported-version CI |
| 01 | Lattice helpers, reduction kernels, review tooling, coverage/CI setup | [part-01-lattice.md](part-01-lattice.md) |
| 02a | Periodic geometry, site indexes, neighbor queries and comparative benchmark | [part-02a-geometry.md](part-02a-geometry.md) |
| 02b | Contacts, covering, void review and exact reduction fallback | [part-02b-contacts-covering.md](part-02b-contacts-covering.md) |
| 02c onward | Void completeness investigation, remaining shared code, I/O, moire, interfaces, adsorbates, defects, symmetry, visualization and CLI/support | Pending dependency-ordered review |

## Coverage and release boundaries

CI runs the full suite on Python 3.10–3.14 on Windows. The 3.12 job additionally measures package line/branch coverage and uploads `coverage.xml`. Coverage exposes gaps to investigate; no arbitrary percentage is used as a substitute for scientific assertions or function review. Subprocess-only CLI execution may require separate measurements before a claim of complete coverage.

See [the publication plan](../publication-plan.md) for candidate selection, versioning, validation, installed artifact checks, and the later explicitly authorized publication sequence. Local review artifacts and scientific outputs remain ignored.
