# Realized stacking selection and interface assembly resources

**Goal:** Fulfil requested stacking senses, preserve selective flags across mixed inputs, and remove avoidable per-atom copying in interface assembly.

**Architecture:** Reuse the incoming same-gauge analysis, select a deterministic geometric mirror whose actual layer increments reverse in that gauge, and validate realized requests in every high-level caller. Keep low-level geometric-mirror ordering unchanged. Group coordinates by validated header blocks, retain the original Cartesian/direct operations, and preallocate final arrays.

**Stack:** Python >=3.10; NumPy >=1.26,<3; no new dependencies.

**Spec:** Active systematic CELLSTINE review goal; prepared `part06e-next-stacking-selection.md`, confirmed selection and assembly probes under the ignored root `.tmp`. Base dev `b34cfe3e5d392c9d739c6d9225c30ead21c5841b`; all16 CI42097678458 and downloaded coverage artifact13449607294 digest/head/module/statement checks passed before this batch.

## Global constraints

- Inspect every current callable in stacking, assembly and Interface:52 existing bodies plus new helper(s). Existing two optimized Interface bodies retain individually justified status when unchanged. Prepared notes alone do not promote ledger entries.
- Preserve public signatures, scientific tolerances, gauge conventions, valid output ordering, paths, metadata and low-level default `mirror_structure` behavior.
- Keep both existing Cartesian-to-direct operations and arithmetic order when stacking. Empty grouping shapes and zero/duplicate header behavior remain compatible.
- Unflagged sites are physically free: when any slab carries selective flags, fill the other slab's sites with `(T,T,T)` in final species/atom order.
- Preserve main checkout, Claude patches and run data; no publication, release or tag. Native execution continues autonomously with one whole-batch independent review, one frozen full-suite/package checkpoint, scoped dev push and exact-head CI before further runtime work.

## Review focus

1. Uniform ABC/CBA, HCP and mixed words: check complete increment negation in one fixed gauge, not only a sign or determinant (Task1).
2. Nonprimitive or non-ideal cells may lack a realizable reversal: fail clearly without returning a contradictory requested-sense summary (Task1).
3. Duplicate/zero chemical headers, mixed flags and unequal source species must preserve coordinate/flag alignment and ownership (Task2).
4. Huge or malformed counts must be rejected before label expansion/allocation, without silently dropping coordinates (Task2).
5. Actual written direct and matched-interface slab files must realize the selected sense, preserve flags and leave source bytes untouched (Task1).

## Task1: Realize stacking requests

**Files:** modify `src/cellstine/interface/surface/stacking.py`, `src/cellstine/interface/workflow/assembly.py`, `src/cellstine/interface/workflow/interface.py`; create `tests/test_interface_assembly_resources.py`.

**Interfaces:** add private `_reverse_stacking(record, analysis, *, reference_hollow=None, layer_tolerance=LAYER_TOLERANCE)` returning the mirrored record and validated same-gauge analysis. Public `apply_relative_stacking`, `prepare_stacking` and Interface signatures stay unchanged.

- [ ] Write the exact60degree triangular regression: input ABC, supplied incoming gauge, requested CBA must yield increments(-1,-1)mod3 and summary CBA. Cover60/120degree bases, both incoming senses, bottom mirror/registry, HCP/mixed increments, and written slab readback.
- [ ] Run the new test file, record failures before source changes.
- [ ] Select the first sorted mirror whose actual analysis reverses all incoming increments in the fixed gauge. Reuse supplied analysis. Keep ordinary first-mirror successful outputs; fail if none works. Start with actual candidate readback, adding no speculative geometric prefilter.
- [ ] Use the shared selection in relative stacking, both assembly mirrored choices and `_mirrored_slab_copy`. Reuse analyses where they already describe the returned records. Validate explicit final ABC/CBA requests after selection/registry translation. Preserve standalone mirrored analysis's own-gauge semantics when no external gauge was supplied.

## Task2: Group and merge owned blocks

**Files:** same assembly module and test file.

**Interfaces:** `group_by_species(record, positions_direct, selective_flags)` and `stack_structures(bottom, top, *, gap, vacuum)` keep tuple shapes/order. Non-string all-zero Miller indices gain the same invalid-plane rejection as string `000`.

- [ ] Add mixed flagged/unflagged source tests with literal full-length flag oracles, both source directions and disjoint/duplicate headers. Add empty/zero grouping and ownership cases, malformed/huge header/cardinality refusal, and all-zero sequence Miller rejection. Observe failures first.
- [ ] Validate header lengths, nonnegative integer-cast counts, coordinate and flag cardinality before expansion/allocation. `expand_species` is deliberately permissive for renderers and cannot validate structure headers.
- [ ] Fill preallocated grouping arrays from header slices, retaining first positive species occurrence and within-species atom order. Merge grouped slabs by header blocks without per-atom NumPy arrays or repeated per-species header rescans. Preserve existing direct conversion calls and free-site TTT semantics.
- [ ] Run one focused assembly/stacking/interface/matching/flags batch. Expected: no failures. Benchmark representative small/large grouping and complete stacks against b34cfe3, including flags/ownership/exact coordinate fields and corrected-case independent oracles.

## Task3: Review and checkpoint

- [ ] Independently review all52 original callable bodies plus additions, source-hash every decision and explicitly retain unresolved numerical/primitive limitations. Do not silently carry the solved CBA defect as open, or conflate it with unsupported numerical domains.
- [ ] Freeze tracked Python plus the new test; run one full branch-coverage pass, wheel/sdist/Twine and installed core/CLI/sdist scientific readbacks.
- [ ] Promote individually reasoned rows after terminal success, refresh only final sdist for ledger/docs, commit/push scoped dev changes, check all supported-version/portable jobs and downloaded coverage artifact before next runtime edits.

## Self-review

All five review-focus inputs have tests in their owning task. The52-body scope covers all mutated modules. Unproven registry policy changes are excluded: the prepared ABC/HCP profile example gives the same six option rows under either interchangeability flag. Passive nonideal registry reporting is not silently changed. Whole-goal completion remains subject to every pending function and final audit.
