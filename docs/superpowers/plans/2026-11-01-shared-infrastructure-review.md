# Shared infrastructure review implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. The active goal explicitly authorizes autonomous inline execution and scoped dev pushes; no additional execution-choice approval is required.

**Goal:** Individually review all 44 existing functions in workflow base, dependency discovery, validation, reporting, provenance and manifests, making only evidenced improvements.

**Architecture:** Continue from verified `4d005c0` in the isolated dev clone. Keep public signatures, filename suffixes, manifest fields, formatting, scientific tolerances and provenance ordering. Prefer small local changes over new infrastructure. Keep compatibility-sensitive copying and serialization semantics intact.

**Tech stack:** Python3.10–3.14, NumPy>=1.26,<3, pytest/coverage, existing build/install and GitHub CI tools.

**Spec:** Active systematic-review goal and `docs/optimization/README.md`.

## Global constraints

- Read every scoped function and trace callers/tests; passing coverage alone is not review evidence.
- Preserve shared-root edits, scientific run data, APIs, scientific thresholds, file formats and output ordering.
- Do not publish packages/releases or force push. Local full/package verification precedes commits; exact-head CI and coverage artifact verification precede the next part.
- Existing `asdict` copying, dependency-cache lifetime and manifest representation are compatibility boundaries; do not simplify them without equivalence evidence.

## Scope and files

Inspect `core/base.py`9, `dependencies.py`6, `validation.py`5, `report.py`12, `provenance.py`8, `manifests.py`4. Create focused infrastructure/provenance tests and `benchmarks/benchmark_infrastructure.py`; modify production bodies only after reproduction. Maintain the ledger and part report, and record Part03a CI/readback in its report and parent plan.

## Review focus

1. Concurrent workflows choosing the same minute-based run ID must create distinct directories, without swallowing permission or parent-path failures.
2. NaN is not positive; preserve positive infinity and existing finite-number policy rather than silently imposing a new finite-only contract.
3. Oversized provenance histories must retain the exact newest suffix under the existing character limit, including Unicode, empty notes and an overlong stage name.
4. Provenance deduplication preserves case-sensitive first occurrence; repeated newly appended notes retain their existing behavior.
5. Optional dependency errors/cache behavior, nested manifest values/copy conventions and CLI report ordering need direct edge-case exercises. Investigate extreme validation scales without claiming an isolated predicate repair proves the downstream geometry domain.

## Task1: atomic run-directory creation and scalar validation

Interfaces remain `Base.create_run_dir(stage, label=None) -> (str, Path)` and `ensure_positive(value, *, name) -> float`.

- [x] Add deterministic race regression: reserve the first target between selection and mkdir; expect the existing `_02_` suffix and leave the competing directory intact. Add a real concurrent-creation case and parent/permission error propagation checks.
- [x] Run the new tests before implementation and record the actual failure. Replace the check-then-create sequence with mkdir-first collision retry; retry only a confirmed occupied target, not parent/permission errors.
- [x] Add NaN rejection with the established named error, plus finite positive/zero/negative/infinity cases. Observe NaN failure before changing the positivity predicate; retain the remaining policy.
- [x] Run focused tests; trace all workflow directory consumers and inspect the remaining base/dependency/validation functions individually.

## Task2: bounded provenance processing

Interfaces remain `comment_notes(comment) -> list[str]`, `_assemble(stage, notes) -> str`, `stage_comment(stage, *notes) -> str`, `restage_comment(previous, stage, *notes) -> str`.

- [x] Reproduce `_assemble` allocation with 1,000 oversized notes followed by two short retained notes; prepare inputs before tracing, assert exact output and peak below128KiB.
- [x] Compare note deduplication and restaging on long unique/repeated histories with the baseline. Preserve field filtering, exact case-sensitive first-occurrence order and repeated appended-note semantics.
- [x] Keep only the newest suffix whose existing character accounting fits; avoid rebuilding/dropping the entire joined string repeatedly. Use set membership only where string equality/order semantics remain unchanged.
- [x] Add exact-limit, one-character-over, Unicode, empty-history, overlong-stage and duplicate-note checks; compare randomized baseline outputs.
- [x] Add a reproducible benchmark against `4d005c0` using trusted baseline loading, exact equality, inputs outside tracing, three timing samples and environment/source hashes. Do not add unstable CI timing thresholds.

## Task3: complete inspection and verification checkpoint

- [x] Exercise all44 original functions, including optional dependency/cache/error branches, manifests/JSON ownership, report recursion/units/warnings and ordinary validation. Record a reason for every unchanged function and any investigated deferral.
- [x] Obtain independent read-only code review; fix material findings before freezing.
- [x] Stage new benchmark before inventory regeneration. Reconcile source hashes from an executable AST comparison; mark only actually inspected functions. Check canonical ledger drift.
- [x] Run focused caller tests, freeze all source/tests, then run the full coverage suite with project-local temporary files. Verify frozen hashes after terminal success.
- [x] Build/check wheel and sdist, compare packaged/support bytes and run isolated core/CLI/sdist installation smokes and dependency/help checks.
- [ ] Finalize evidence/counts, commit scoped files, push dev, verify all seven exact-head CI jobs and the retained coverage artifact before proceeding.

Self-review: all active-goal gates map to Task3; each proposed change has reproduction/equivalence checks; all five review-focus cases map to a task. No implementation-agent delegation or publication is planned.

Rulings: independent review inspected and directly exercised44/44original functions and reviewed both benchmark functions;49focusedtests pass. The coordinator selection passes62tests.30000provenance and1007positivity unchanged-output/error comparisons pass. Exactly five original executable bodies changed. No implementation finding remains. Defer run_output_suffix with proven run100/cross-label overwrite and numeric/Miller-label ambiguity; defer structure_errors with demonstrated determinant/scale overflow and downstream distance overflow. Track both as inspected deferrals in the ledger; next scoped reviews must address these release follow-ups without renaming ordinary noncolliding outputs or changing scientific tolerance. Full suite runs against248frozenPython files.

Local verification checkpoint:3257passed,4skipped,81warnings in727.34s. All248frozenPython hashes and finalbenchmark digests match. Ledger1101total/216inspected/885pending with3investigated deferrals;39optimized/174unchanged. Package12412/14301statements and3229/4504branches. Wheel/sdist/Twine and140module bytes pass;all3installedcore/CLI/sdist smokes/help/dependency checks pass. ExpandedminimumNumPy lane collects257tests;independent49added tests pass withoptionalimportsblocked. Finalsupportarchive bytes,commit/push and seven-job CI/artifact readback remain.
