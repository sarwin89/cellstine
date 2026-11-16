# Part06e: realized stacking requests and interface assembly resources

Base: dev `b34cfe3e5d392c9d739c6d9225c30ead21c5841b`; all 16 CI42097678458 jobs and downloaded coverage artifact13449607294 digest/head/module/statement checks closed before runtime edits. Independent review covers all 52 original bodies in stacking, assembly and Interface plus the new selector:53 current callables.

## Changes

High-level stacking reversal now selects the first sorted lattice mirror whose actual same-gauge layer increments negate every incoming step. This fixes the 60° triangular CBA request which previously returnedABC or raised despite three valid reversing operations. The low-level default geometric mirror stays unchanged. Relative requests, direct assembly and matched-slab copies use the shared selector; explicit requested senses and realized registry configurations are checked. Existing analyses are reused rather than recalculated for unchanged records. Standalone mirror results retain their own-gauge analysis semantics.

Grouping validates header/position/flag cardinality before label expansion or output allocation, copies header blocks into owned arrays, and preserves duplicate/zero header behavior, species order, atom order and empty grouping shape `(0,)`. The complete stack merges grouped blocks without per-atom NumPy arrays or repeated species/header scans. Both existing Cartesian-to-direct calls and arithmetic order remain. When only one slab has selective flags, the other slab's atoms receive free `(T,T,T)` flags in the actual final order; previously a valid four-atom input returned only two flags. Non-string zero Miller indices now receive the same invalid-plane refusal as `000`.

## Measurements

Three interleaved warm unprofiled predecessor/final timings, native threading limited before NumPy import; separate Python/NumPy tracked peaks, not RSS:

| Case | Before | After | Peak before / after |
| --- | ---: | ---: | ---: |
| group_8_2_flagsTrue | 0.018 ms | 0.010 ms | 0.002 / 0.001 MiB |
| group_20000_100_flagsTrue | 31.412 ms | 1.700 ms | 4.476 / 0.644 MiB |
| group_20000_100_flagsFalse | 31.154 ms | 0.349 ms | 4.305 / 0.472 MiB |
| group_20000_duplicate_zero_headers | 41.468 ms | 2.315 ms | 4.457 / 0.700 MiB |
| complete_stack_8_2_overlap | 0.174 ms | 0.119 ms | 0.004 / 0.002 MiB |
| complete_stack_20000_50_overlap | 83.602 ms | 4.471 ms | 5.548 / 2.170 MiB |
| prepare_6_abc_alreadycorrect | 13.397 ms | 6.209 ms | 0.017 / 0.011 MiB |
| prepare_6_cba_alreadycorrect | 15.000 ms | 12.288 ms | 0.018 / 0.018 MiB |
| prepare_600_abc_alreadycorrect | 200.506 ms | 99.863 ms | 0.252 / 0.215 MiB |
| prepare_600_cba_alreadycorrect | 226.371 ms | 178.261 ms | 0.289 / 0.274 MiB |

Valid grouping, complete stack arrays/headers/flags, empty stack shapes and already-correct 120° preparation records/summaries match the predecessor exactly. Corrected 60° and mixed-flag cases use independent actual increment and literal flag oracles, not equality to wrong predecessor data. Small cases are included; no timing thresholds or universal speedup claim.

## Verification and limitations

Initial 32 regressions gave 22 failures / 10 passes. One nonprimitive fixture initially supplied native I/O data lacking `.copy`; it was adapted to the canonical workflow record type, and the benchmark independently confirms the original canonical case refused CBA while the new one realizes it. The broader interface/matching/stacking batch passed 144 tests in 12.62s. New 32 core-only regressions also enter minimum-NumPy and nine portable CI lanes. The completed local checkpoint is recorded below; scoped dev push and exact-head CI/artifact closure remain.

The ordinary60degree requested-sense defect is resolved. Existing `_gauss_reduce` extreme convergence and `_layer_point_lattice` arbitrary primitive-discovery/resource guarantees remain investigated deferrals. The coordinator adds three individually evidenced reporting deferrals after inspecting the peer review's declined scope: passive reporting infers contact from separately origin-reset layer labels. Top translations `0,h,2h` require actual contact deltas `1,2,0`, but both predecessor and current code report `1,1,1`. Actual registry-shift checks require zero correction for each independently expected delta. `contact_summary`, `realised_stacking_summary` and `prepare_stacking` therefore remain deferred for physical contact-origin reporting; this does not negate verified request selection, geometry or flag fixes. The next registry/sequence reporting batch must fix that metadata behavior and define nonideal-offset reporting before release readiness. No policy change is justified by the separate ABC/HCP profile example, whose six option rows are unchanged.

## Coordinator rulings

- **Passive physical contact-origin reporting**: Confirmed existing Important metadata defect, including ideal shifted contacts. Defer contact_summary, realised_stacking_summary and prepare_stacking individually to the next registry/sequence reporting batch. Preserve this batch's public label-only helper signature and record actual error rather than claim release readiness. Cost if wrong: Reported registry can contradict the actual boundary; users may choose an inappropriate contact even though structure coordinates and measured distances are retained.
- **External arbitrary mirror matrices/non-normalized/nonfinite input hardening**: Current workflows normalize geometry and select enumerated integer mirrors. Keep external low-level behavior and do not certify arbitrary supplied transforms or numerical domains; the existing numerical/primitive deferrals remain. Cost if wrong: A caller bypassing the established preconditions can obtain errors or nonphysical geometry; no universal public-helper hardening claim.
- **Registry interchangeability and abstract table materializability**: Do not change policy from the ABC/HCP profile suspicion: its six output options are identical with either flag. Validate selected builds and refuse unavailable cell reversals; broader registry proof stays in its pending review. Cost if wrong: Unsupported cells may refuse abstract options; completeness/congruence outside the supported primitive contract is not certified.
- **Coordinator verification gates**: Peer review is source-only. Actual144focused and interleaved benchmarks pass; full/package/installed/exact-headCI gates remain separate until terminal evidence. Cost if wrong: Unproven gates would misrepresent package/scientific acceptance; no promotion until those pass.
- **Nonprimitive test data type**: Convert native repetition payload into the canonical StructureRecord used by workflows. The initial copy AttributeError was a fixture mismatch; canonical predecessor refusal and corrected same-gauge CBA are separately confirmed by the benchmark. Cost if wrong: Direct mirroring of native I/O payload types remains an existing unsupported-copy path; do not present it as a tested canonical workflow failure.

The overall goal remains incomplete. No release, tag or package upload is authorized.

## Local checkpoint

One clean frozen full run passed **3814 tests**, skipped 4, with 81 warnings in 819.76s. All 287 Python source hashes remained fixed. Local coverage is 14261/15909 statements and 4019/5206 branches. Wheel/sdist builds and Twine checks passed. All three installed core/CLI/sdist environments passed actual requested-sense files, matched-slab copies, mixed constraints, header guards and native serial/process/thread builds with dependency checks. All 144 runtime modules match the wheel; final ledger/docs update refreshes only sdist support files without repeating unchanged runtime/full/install checks.

Ledger: **1226 functions**, **1041 with inspection evidence**; 720 reviewed—unchanged, 307 optimized—verified, 199 deferred—with reason. Of deferred rows, 185 are pending and 14 are investigated. This batch refreshes 53 body inspections and adds 21 inspection records; the project-wide goal remains incomplete. The ordinary CBA-selection deferral is resolved, while three concrete passive registry-reporting bodies enter explicit investigated deferral for the next batch.
