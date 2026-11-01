# Shared structure helpers implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Review all 40 existing functions in structure transforms, direction/vacuum primitives, species/element helpers and the canonical structure model, retaining scientific tolerances, ordering and public formats.

**Architecture:** Continue in the isolated dev clone after verified Part02c CI. Review each function and its callers/tests before deciding whether to change it. Resource improvements must enumerate the same mathematical domain; unusual finite-scale repairs must preserve ordinary arithmetic and all documented thresholds. Work is authorized to proceed autonomously by the active goal.

**Tech stack:** Python3.10–3.14, NumPy>=1.26,<3, pytest/coverage, existing build/Git tools.

**Spec:** Active systematic-review goal, `docs/optimization/README.md`.

## Files and scope

`core/transforms.py`11 functions; `core/directions.py`15; `core/vacuum.py`6; `core/species.py`2; `core/elements.py`4; `io/models.py`2. Read the declarative core models/constants and exports as dependencies even though they add no AST function rows. The workflow base, dependency manager and validation module received an initial scan but remain for Part03b; do not mark them reviewed from that scan.

## Constraints and review focus

- Preserve scalar/array return types, species and image order, selective flags, shallow metadata-copy conventions, right-handedness and lattice row conventions.
- Keep existing scientific degeneracy/rank and integer-acceptance tolerance policies unless an actual compatibility-preserving numerical repair is proved. Do not silently tighten default allclose acceptance.
- Preserve ordinary outputs exactly where feasible; compare floating transformations with independent geometry and stated tolerances.
- Keep optional dependencies optional. Check any proposed NumPy call against the declared minimum version; do not infer compatibility from one recent interpreter.
- No shared-root cleanup, scientific runs, release/package publication or force pushes.

## Task1: function inspection and reproduced regressions

- [x] Read all selected modules initially; trace species and transform callers in renderers, interface assembly, moire generation and defect supercells.
- [x] Complete caller/test/edge-case review for every function and record reasons in the part report and ledger.
- [x] Reproduce bounded-label allocation: expand_species(['C'],[1000000],natoms=3) must return three labels without constructing the discarded million entries. Keep negative-slice behavior and full expansion when natoms is unset; keep validating later counts rather than breaking early.
- [x] Investigate supercell-coset box amplification with sheared integer matrices. Prove that any replacement returns the exact old canonical representatives in lexicographic order, including negative determinant. A possible complete group walk uses adjugate residues modulo abs(det), but retain a fast small-box path if benchmarks justify it. Do not implement without a failing allocation/correctness test.
- [x] Probe finite extreme-scale axis normalization and frame/surface directions against scale invariance, norm and handedness. Preserve ordinary evaluation paths; leave sound functions unchanged.
- [x] Verify the declared NumPy minimum for strained_copy; current np.asarray(...,copy=True) deserves an explicit compatibility check.

## Task2: verification and scoped checkpoint

- [x] Add meaningful red/green regressions for selected changes, compare old/new outputs and benchmark resources on reproducible cases.
- [x] Independently review the full scoped implementation, including exact integer membership/order and numerical boundary behavior.
- [x] Update the source-function ledger only with actual inspection evidence; include any new helper/benchmark functions. Distinguish pending reviews from investigated deferrals.
- [x] Freeze source/tests; run focused then full coverage suite, check frozen hashes, build wheel/sdist/Twine and relevant installed smokes.
- [ ] Commit scoped changes, push dev, and verify all required supported-version CI jobs and exact-head coverage artifact before Part03b.


## Current implementation rulings and focused evidence

Ruling: cap only nonnegative requested label lengths, retain full expansion when unset and preserve negative slicing — this bounds discarded headers without altering valid label order/padding or skipping later count validation. The allocation regression failed at16,000,112bytes then passed;2000exact old/new comparisons pass.25species/I/O tests pass after using a writable project temporary directory; the first broader attempt failed only fixture setup in the sandbox temp directory.

Ruling: retain existing integer-matrix acceptance and adjugate validation policies. Oversized boxes use an adjugate-residue quotient walk; verify closure/cardinality before converting to integer canonical representatives and lexicographically sorting. Small boxes retain the existing vectorized path, bounded by65,536candidates; determinant±1returns the unique origin after validation. Three signed/sheared allocation regressions failed at23.6–35.0MB then passed.100independent box references and full atom-block/selective-flag/folded-geometry checks pass;56existing defect-supercell tests pass. Existing floating determinant/int64 representability limits remain and must be documented rather than silently claiming arbitrary exact integer input support.

Ruling: normalize via largest-component scaling only when existing finite-vector arithmetic overflows/underflows; preserve ordinary arithmetic, absolute zero thresholds, handedness and frame orientation. Surface-plane fallback rescales the cross product or its input rows, as needed.17scale/minimum-API regressions failed before implementation; all18then passed. Combined helper/rotation/species/coset selection:200passed. The final numerical diff and all 40 original functions have received independent read-only review; no material finding remains. Two intermediate tiny/partially-subnormal cross-product findings were reproduced red/green before the final review. The exceptional fallback computes exact cross-component ratios from the supplied binary64 values when row scaling alone loses them.

Ruling: replace np.asarray(copy=True) in strained_copy with np.array(copy=True) — installed NumPy documentation marks the former keyword as added in2.0, contrary to declared>=1.26. The dependency-boundary test reproduces the old signature failure and verifies non-aliasing. Add a real Python3.12/NumPy1.26.4 CI job for the four focused helper test modules; the existing five-version full suite and package/coverage jobs remain required. Do not report minimum-version runtime success before that new job passes.

Corrected the invalid Miller example1x1in direction documentation/help to11x1, which already means(1,-1,1) under the existing trailing-x convention. No parser acceptance or sign convention changed.

All 40 original functions, the new quotient helper and two benchmark functions have now been inspected. The final focused caller selection passed 400 tests with eight existing warnings. The reproducible final-source benchmark and independent review are recorded in docs/optimization/part-03a-shared-structure.md. Remaining before checkpoint: local full frozen suite, packaging/install checks, ledger update, scoped dev commit and all required CI jobs including the new minimum-NumPy lane.


Local checkpoint: 3230 passed,4 skipped,81 warnings in706.57s. All245 frozen hashes and final benchmark digests match. Ledger1099total/170inspected/929pending; canonical drift check passes. Wheel/sdist/Twine,140module bytes and all three installed structure-helper/CLI/dependency smokes pass. Final archive docs/ledger refresh and remote CI remain before the final integration checkbox.
