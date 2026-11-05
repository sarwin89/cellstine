# Stacking mirror enumeration and complete callable review

**Goal:** Correct missing in-plane mirrors in ordinary sheared cells and reduce the repeated matrix/determinant work while retaining scientific acceptance, matrix order and structure metadata.

**Architecture:** Replace the fixed coefficient box for finite, well-resolved metrics with conservative row annuli. Pair only possible rows; retain the original determinant and `M @ G @ M.T` residual expressions. Preserve the original small-box path for numerically unresolved or excessively broad tolerance domains. Exact rational bound construction and integer annulus enumeration avoid inward-rounded search limits. No new dependency or parallelism default.

**Stack:** Python >=3.10, NumPy >=1.26,<3, standard-library integer/Fraction arithmetic.

**Spec:** The active systematic CELLSTINE review goal; predecessor dev `4c983a33edc02e938aa817917d5d93a6f0063400`; prepared exact-integer certificates in the ignored `part06d-mirror-baseline.json` artifact. Part03h all 16 CI jobs and downloaded coverage checks closed before this batch.

## Global constraints

- Review all 29 existing callable bodies in `src/cellstine/interface/surface/stacking.py`, including methods and the sort lambda; independently review any new helper too.
- Preserve the final componentwise Gram residual, tolerance scaling, determinant predicate, lexicographic output and first default mirror for already-complete ordinary cases.
- Preserve source positions, flags, species/count order and shallow metadata-copy semantics. No releases, tags, package uploads, new dependencies or changes to shared main/scientific data.
- Continue natively, with one independent whole-batch review, one frozen full-suite checkpoint and package checks before a scoped dev commit; exact-head CI/coverage gates the following runtime batch.

## Review focus

1. Nearly singular, zero and nonfinite metrics must not enter an unjustified finite-domain proof; preserve their original fallback behavior (Task 1 compatibility cases).
2. Tolerance boundaries must not discard matrices accepted by the original floating expressions (Task 1 exhaustive bounded-reference comparisons).
3. Very small physical scales and broad tolerances can admit huge search domains; retain the established path when the numerical proof/domain is unsuitable (Task 1 scale/tolerance cases).
4. Returned matrices must be independently writable; mutations cannot poison later calls or the caller's lattice (Task 1 ownership case).
5. Default mirroring must preserve compound headers/flags and be an isometry, including a sheared cell previously rejected (Task 1 structure-level case).

## Task 1: Bound and filter mirror rows

**Files:** modify `src/cellstine/interface/surface/stacking.py`; create `tests/test_stacking_mirror_resources.py`.

**Interfaces:** public `inplane_mirror_matrices(lattice, tolerance=1e-6) -> list[np.ndarray]` and `mirror_structure(record, *, matrix=None)` stay unchanged. A private row-search helper returns two lists of integer row tuples, or `None` for the legacy fallback.

- [ ] Write literal independently derived mirror sets for square/rectangular/oblique and four sheared/orientation variants. Include structure-level default reflection and compatibility/ownership cases.
- [ ] Run `python -m pytest -q tests/test_stacking_mirror_resources.py`, record missing sheared mirrors/default reflection failures before runtime changes.
- [ ] Derive a conservative floating roundoff envelope, form exact rational inner/outer quadratic bounds, enumerate integer annuli by interval square roots, then apply the unchanged original final predicates. Keep a justified numerical/broad-tolerance fallback; document its remaining completeness limits honestly.
- [ ] Clarify the reduced-basis precondition and geometry-only point-lattice documentation without changing stacking gauges, chemical policy or primitive-lattice selection.
- [ ] Run the new tests and the existing interface stacking/CLI/build/sequence/layer tests as one focused batch. Expected: no failures.

## Task 2: Review, measure and close the checkpoint

**Files:** new `docs/optimization/part-06d-stacking-mirrors.md`; update function ledger, optimization README, CI selections and preceding checkpoint's CI closure.

**Interfaces:** retain all 29 original IDs; reconcile every new helper ID and normalized module hash. Prepared notes do not count until independently reviewed and gated.

- [ ] Compare predecessor/final results and timings for ordinary and corrected sheared cells; validate accepted matrices against the independent integer oracle and meaningful floating tolerance cases. No CI timing thresholds.
- [ ] Obtain one read-only complete-body review of the full module and additions; explicitly resolve or document findings.
- [ ] Freeze tracked Python plus the new test, run one full branch-coverage suite and wheel/sdist/Twine checks with three installed environments. Expected: full suite green, archive runtime bytes equal, dependency checks clean.
- [ ] Promote individually reasoned review rows only after final local gates; refresh final sdist support files without repeating unchanged runtime gates.
- [ ] Commit the scoped files, push dev, verify all supported-version/portable CI jobs and downloaded coverage digest/head/modules/statements before the next runtime batch. Keep the overall goal active until every required callable and final audit is complete.

## Self-review

All five review-focus inputs have Task 1 coverage. The public signatures and final arithmetic are retained. The confirmed missing-mirror defect is addressed within a proved ordinary numerical domain; wider primitive/registry policies remain separate investigated work rather than silent behavior changes. Previous review artifacts and the shared checkout remain preserved.

## Execution rulings and evidence

- Initial literal/consumer/resource regressions: six failures and18passes before source changes; then24passes. The larger shear300 case exposed an unnecessarily restrictive broad-tolerance guard (one fail/one pass); the final diagonal error envelope uses positive-definite Gram diagonals rather than an isotropic maximum, and broad means threshold exceeds a target diagonal.
- Independent review identified infeasible coordinate scans for valid anisotropic/near-collinear finite metrics. Two scoped range-spy regressions failed immediately without launching enormous loops. The fix scans the shorter axis and bounds scans to65,536 steps, rows/pairs to2,401 (the old matrix-box count); infeasible numerical/work domains retain explicit legacy fallback. Final focused97tests pass in6.10s.
- Final timing evidence covers seven ordinary/corrected integer cells,180scalar floating-final-predicate comparisons and15exact integer conjugations. Ordinary square/hexagonal compound default records remain byte-exact. Only three docstrings changed afterward; the measured raw source was reconstructed and hash-matched, with executable AST equality proved. No unchanged runtime benchmark rerun for doc-only edits.
- Existing Important selection defect: a60degree triangular lattice's first geometric mirror can preserve hollow cosets. `apply_relative_stacking` refuses the wrong requested sense, but assembly `prepare_stacking` can silently reportABC afterCBA was requested. This batch preserves low-level default ordering; the high-level selection/validation fix is explicitly the next priority. Correct false universal reflection wording now; do not count this defect as solved or claim release readiness.
