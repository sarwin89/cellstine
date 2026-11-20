# Shared migration, record and compound-surface resource implementation plan

> **For agentic workers:** Use superpowers:executing-plans inline, with one independent read-only whole-batch review. No implementation delegation.

**Goal:** Preserve exact migration pairing/arithmetic and valid structure ordering while reducing temporary memory and fixing malformed-header/duplicate-species handling.

**Architecture:** Retain the assignment solver and metric kernels; block only cost and certificate temporaries. Group species/index data once. Repeat compound structures by source header block, cell, then atom. Aggregate duplicate chemical labels where a grouped physical result is required. Keep scientifically sound primitive-surface routines unchanged; fix the shared grouping and scalar-level ownership/tolerance boundary that their callers use.

**Tech Stack:** Python >=3.10, NumPy >=1.26,<3; no new dependency or worker default.

**Spec:** Persistent systematic-review goal and prepared source-hashed40+24 callable inspections. Base604d660 passed all16CI42020605925. Downloaded coverage13411738331 ZIP SHA87a4e963d44cf1289dd6317c61223805be7786ac14187b0151eb6338e12cba6a verified head/module/statement sets; only known prompts501-502 CI-only path fallback, no branch delta.

## Current scope and rulings

- Inspect all64 original complete callable bodies in core/pathway(18), core/naming/root exports(2), surface_supercell/termination(11), surface_cell(24), symmetry models/records/reporting/spglib_adapter(9). New callables require their own rows.
- Runtime changes are limited to pathway, surface_supercell, termination, surface_cell grouping/scalar levels plus the review-discovered bijection/nonplanar enumeration repairs, and symmetry records. Other bodies receive current individual inspection decisions, not automatic passing-test promotion.
- Preserve exact assignment/ties, periodic minimum-image arithmetic, wrapping, source-species/header/cell/atom order, flags, JSON schemas and valid public defaults. Keep old unusual/nonfinite certificate behavior through an explicit fallback.
- Surface repetition keeps identity object for1x1, valid empty shapes and duplicate source-header blocks. Surface regrouping instead emits each species once, maintaining within-species atom/flag order.
- Reject impossible address-sized repeat/image output before copying/assignment. Bound pair-cost/certificate temporary blocks to roughly65536 values; returned costs/images still require output-sized storage.
- Scalar-level clustering keeps the last-kept greedy rule, accepts zero tolerance, refuses negative/nonfinite tolerance before looping and stops mutating caller arrays.
- Do not claim to repair arbitrary huge-integer surface transforms or all primitive-surface fallback bounds by changing a wrapper. Document any actual investigated limitation distinctly from pending review. Stacking/registry mirror and matching policies remain future batches.
- No scientific run data, unrelated shared-root source, publication, tags or package uploads.

## Task1: Shared data/resource correctness and scoped optimization

Files: src/cellstine/core/pathway.py; src/cellstine/interface/surface/{surface_supercell,termination,surface_cell}.py; src/cellstine/symmetry/records.py; unchanged reviewed source modules listed above; tests/test_shared_record_surface_resources.py; CI minimum-NumPy selection; scoped benchmark/report/ledger.

- [x] Observe red regressions: initial24cases20failed/4passed, corrected nonmatrix target4failed and4additional scalar-level cases failed. No runtime edit preceded those failures.
- [ ] Implement bounded cost/certificate computation, exact species grouping, early migration metadata/cell/image guards, block-ordered surface repetition, aggregated report totals, scalar-level copy/guard, and record cardinality checks.
- [ ] Add valid/empty/tie/legacy fallback/ownership equivalence cases where needed; benchmark source-hashed predecessor and corrected cases with independent physical oracles. No timing thresholds in CI.
- [ ] One independent complete-body review of whole batch; reconcile all64 originals/new helpers and retain prior valid evidence only after actual review.
- [ ] Focused tests, frozen full suite and coverage, wheel/sdist/Twine/archive and three installed environments. Promote ledger after gates; refresh only final sdist for doc/ledger closure, avoiding unchanged gate reruns.
- [ ] Separate dev commit/push, all16 exact-head CI and downloaded coverage digest/head/module/statement readback before further runtime changes.

## Original prepared rationale (historical prior-base draft; current base and rulings above apply)

# Path, record and compound-surface resource plan

Base:1e01173aa2725431d035d8ecfdcc519fc529ab11. All16 exact-head CI41820006659 jobs passed. Downloaded coverage13316437813 ZIP SHA256d1564189d31d0adb3c7f1683b3c8d546d10e8ec489944eef7fd2136b3cfb8db0 and module/statement sets verified; only known path-display fallback differs locally. Remote dev rechecked at the same commit before edits on2026-11-05.

## Scope and invariants

Complete individually source-hashed inspections cover40 original functions:18core/pathway,core/naming and root lazy export (20 total); surface_supercell/termination (11); symmetry models/records/reporting/spglib_adapter (9). New benchmark callables require separate review. The shared-root notes are .tmp/part03h-combined-inspections.json. Work only in the isolated dev clone; preserve user/Claude source and scientific data.

Preserve exact assignment solver/tie policy, periodic minimum-image conventions, species ordering, wrapping and flags, headers/schema, public defaults, all scientific tolerances and finite valid numerical arithmetic. Leave sound functions unchanged. Existing huge-integer surface transforms and extreme lattice numeric domains are not repaired by changing only a wrapper.

## Planned implementation

- Add independent red regressions before source edits for compound repetition (including source header blocks/duplicate labels, per-cell order, selective flags, empty and identity behavior), duplicate bulk/slab count aggregation, direct migration header/position consistency, finite same-cell comparison and NumPy end_counts with legacy empty-count fallback.
- Preallocate surface repeat output in source-species-block order, then preserve original cell-major/atom order within each block. Bound vectorized offset work; retain the1x1 identical-object fast path and valid coordinate arithmetic. Reject impossible address-sized output before allocation. Core.repeat_structure uses atom-major order and is not a drop-in substitute.
- Aggregate repeated chemical labels in termination totals without altering ordinary unique-header output or floor/excess conventions. Correct the fractional-unit docstring: an integer limiting ratio alone does not imply zero excess.
- Bound migration pair-cost and certificate slack temporaries, retaining all output-sized costs and the exact assignment solver. Preserve the old certificate behavior for unusual/nonfinite inputs through the original fallback. Group matching indices once in sorted-species order and remove the overwritten displacement allocation.
- Validate migration endpoint/header cardinality and finite3x3 same-cell comparison before work; permit zero cell tolerance, preserve integer coercions and empty end_counts fallback. Reject unaddressable image chains before matching/allocation. Preserve chain arithmetic and wrapping.
- Group symmetry species/index records once and gather positions directly, preserving source-species priority, within-species atom order, metadata-copy contract and model output types. Refuse mismatched labels/positions before silently discarding atoms. Keep optional-adapter access behavior unchanged unless a measured need justifies a separate change.
- Add meaningful resource/equality regressions, production old/new benchmarks (including small cases and corrected-case independent oracles), and extend minimum-NumPy CI selection. No timing thresholds in CI.
- Obtain read-only whole-function review, freeze tracked Python bytes, run one full/coverage/package/three installed checkpoint, then promote the individually reasoned rows. Refresh only final sdist for docs/ledger closure. Commit/push dev, all16 exact-head CI and downloaded coverage readback before another production batch.

## Explicit boundaries

Registry/plane-coincidence policy and defect legacy-orbit reconstruction remain future batches with saved probes; this checkpoint does not invent missing symmetry members. No releases, tags, uploads, broad cleanup or changes to run data. The goal remains active until every tracked source function has inspection evidence and the final requirement audit passes.


## Review-driven rulings

- Ordinary nonplanar transformdet27 returned23 rows. After red regression, retain complete legacy first-pass results exactly; incomplete results retry a cube derived from transformed-corner bounds, with address guards and a final cardinality refusal. Finite/nonnegative tolerance is checked before dispatch. Three compound/orientation oracle cases and a complete legacy diagonal comparison pass. This is not a claim of arbitrary huge-shear/floating transform completeness.
- Any-match translation neighborhoods violated Hall matching. After red regression, retain the exact fractional/componentwise/species graph; unique first choices prove a bijection, otherwise the existing exact0/1 assignment solver decides feasibility. No Cartesian-tolerance substitution.
- Small cost/certificate paths retain exact legacy arithmetic under the65536-value budget after measured overhead. Larger blocking trades some CPU for substantially lower temporary memory.
- Seven direct/related surface algorithms retain individually evidenced investigated deferrals; the finite fallback search has a concrete nonminimal skew example. Wrappers are judged against their actual finite ordinary contract, not blanket-deferred merely for calling a limited helper.
- Existing minimum-NumPy and nine portable-result jobs gain the new core-only regressions, without new jobs/dependencies.
