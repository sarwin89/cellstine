# Shared symmetry resource and tolerance implementation plan

**Goal:** Complete43 individual shared2D/3D symmetry inspections, optimize measured allocations/index reuse, and repair reproduced tolerance/empty-species inconsistencies while preserving public APIs and scientific conventions.

**Architecture:** Work only in the isolated dev clone. Share an internal ordered integer-box block iterator between planar and3D vector enumeration. Keep all existing norm/metric expressions, inclusive boundaries, signs, group ordering and gauge keys. Reuse one call-local site index, preserve decorated matching's existing row-and-column rule, and retain the fast centering lookup with bounded exact fallback for ambiguous nearest matches.

**Tech stack:** Python3.10–3.14, required NumPy1.26–2.x, pytest/coverage, existing build/Twine tools. No new runtime dependency or publication.

**Spec:** Active systematic-review goal and docs/optimization/README.md. Base1a2c3bb57039668333740915a824c653b455c313 passed all16 CI41782577167 jobs and artifact13299884630 ZIPdigest7545a47e915103c01e395786975db837207772d20084172330ca5b1b24362fb4 is independently verified. Only additional CI statements are prompts.py501–502's environment-dependent path fallback; branches agree. The next-source prerequisite is satisfied.

## Scope and invariants

43 complete source callables in core/symmetry2d.py, core/pointgroup3d.py and core/symmetry3d.py have individual prepared reasons in shared-workspace .tmp/part03f-prepared-reviews.json, including the nested planar-gauge lambda. Review the new internal iterator and benchmark functions separately before promotion.

Preserve row/column lattice conventions, signed mirrors, proper subgroup policy, symprec and all existing metric/position/z/gauge tolerances, vector enumeration order, atom/species ordering, direct API signatures, identity/no-operation fallbacks and experimental N-layer policy. No arbitrary scientific-domain or atom-count caps. Remaining rational-basis/extreme-float limits are not fixed by this resource batch.

## Tasks

- [x] Inspect complete bodies/callers and prepare individual reasons. Reproduce nearest-sphere/componentwise disagreement, empty-species validation inconsistency, repeated index construction and full-grid/pairwise allocations.
- [x] Prepare51 regression cases. Prototype300 exact vector/matching comparisons and16 integer-box ordering checks; observe local allocation reductions without claiming production verification.
- [ ] Add core/_integer_grid.py with a bounded rectangular iterator: keep later axes whole to concatenate in meshgrid C order, use scalar prefix decoding and broadcast coordinate axes, allocate fresh int64 blocks, and reject noninteger/negative/out-of-int64 coordinate bounds. Huge box products stream without materializing full axes or linear-index arrays.
- [ ] Use the iterator in2D short-vector and3D metric-vector enumeration. Preserve scalar arithmetic/einsum expressions and inclusion boundaries, convert3D coordinates to float once per block, collect accepted outputs in original order and retain correct empty shapes/dtypes. Output-sized storage remains unavoidable.
- [ ] Block decorated2D translation matching to target65,536 pairs per block (at least one row), retain same partner order, labels/heights and both row/column existential conditions. Do not substitute a different assignment algorithm.
- [ ] Reuse a lazily built call-local PeriodicSiteIndex in equivalent_atom_map/site_permutations; no index for empty operations/positions and no persistent/global cache. Preserve invalid-operation fallback and image arithmetic.
- [ ] Centering: retain nearest periodic lookup for normal separated candidates. If its selected point fails the existing componentwise bound, scan only ambiguous query rows against bounded center blocks using the exact legacy residual/boundary rule. Missing radius candidates still fail immediately. No tolerance widening or full operations×centerings allocation; rare fallback may cost their product.
- [ ] Validate species count before translation_group's empty return, aligning its reduction callers with primitive_cell. Move planar-gauge transform allocation after exact integer determinant rejection without changing traversal or tie keys.
- [ ] Run focused regressions and affected scientific callers. Benchmark actual production APIs against frozen Git modules, exact outputs first; include small/large grids, decorated matching, provided operations and native datasets, with source hashes and honest CPU/memory limits. Extend minimum-NumPy helper lane with new tests.
- [ ] Independent whole-batch read-only review. Fix concrete findings, then freeze all tracked Python bytes and run one full/coverage/build/installed checkpoint using workspace TEMP/TMP/PIP_CACHE_DIR.
- [ ] Promote only individually reviewed source-hashed ledger rows after local gates pass; refresh final sdist for documents/ledger without rerunning unchanged runtime tests. Scoped commit/push dev; verify all16 exact-head CI jobs and downloaded coverage ZIP before next production edits.

## Judgments and limits

Point-group symbol/type classifiers assume actual finite crystallographic rotations/groups, not arbitrary unimodular matrices; adding a general closure validator is outside this resource change. Decorated2D matching's row/column existence rule remains as implemented, including ambiguous duplicate-site inputs. Native3D nearest-site assignment remains its existing bijection policy. Normal valid separated crystallographic centering sets did not reproduce the tolerance edge; regressions establish the helper's advertised componentwise contract without claiming a widespread physical classification error.

Bounded enumeration still costs the number of integer points in its box, plus storage for all accepted output vectors; no universal RSS/time bound is claimed. Warm local prototype measurements (not final production claims) cut2D-skew tracked peak155→3.6MiB and1024-site matching48→4.1MiB while preserving outputs. Rectangular blocks avoid per-point integer division; timing gains need actual production measurements. No release/tag/package upload, user-run cleanup or unrelated shared-root edits.


Ruling after reproduced cube-corner failure: use a conservatively enclosing2*tolerance candidate radius. The final componentwise threshold remains unchanged. Added three red/green boundary regressions; all54 new cases pass. This resolves a prefilter rounding miss alongside the nearest-candidate ambiguity.


## Local execution checkpoint

- [x] Implement ordered boxes, bounded matching, index reuse, exact centering, empty labels and gauge allocation changes.
- [x]54 new regressions pass;914 affected scientific cases pass in117.53seconds.
- [x] Independent47-callable and one-line minimum-NumPy CI review: no actionable findings.
- [x] Actual old/new production benchmark outputs and source hashes verified; memory improvements and workload-specific timings recorded honestly.
- [x]280 Python files frozen; full3,610passed4skipped81warnings in721.29seconds.
- [x] Wheel/sdist/Twine,144runtime archive bytes and3installed environments pass; workspace caches used.
- [x] Promote47 individually reasoned rows; ledger838/1215with evidence,377pending+2investigated deferrals.
- [ ] Refresh closure-document sdist/readback, scoped commit/push and all16exact-head CI jobs plus downloaded coverage ZIP.
