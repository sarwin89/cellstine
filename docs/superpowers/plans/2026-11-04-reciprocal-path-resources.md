# Reciprocal geometry and band-path resource plan

**Goal:** Complete62 individual Brillouin/strata/kpath inspections; repair reproduced ray/cache ownership defects and reduce measured sampling, mask and graph work while preserving public APIs, conventions, ordering and scientific thresholds.

**Architecture:** Modify only the isolated dev clone. Normalize ray magnitude safely before existing halfspace intersection logic; keep the same unit-direction angular tolerance. Keep cached grid/strata arrays readonly, returning fresh writable direction containers with shared aliases preserved within each call. Preallocate sampled paths with the same Python scalar fractions, block floating point-operation masks, use tuple-keyed heap Dijkstra, and reduce integer coefficients modulo48 before periodic grid products.

**Spec:** Active systematic-review goal and docs/optimization/README.md. Base427e2d5d6bd8cbec41a8453c53a1fe1f2b964b4f passed all16 CI41787883849 jobs. Coverage artifact13302931287 ZIPdigest e78173b1a7de0f72ab8504e8fa89ce46159b5e05328cb8e3e8d892d712eb1c05/head/module/statement sets verified; only CI-extra statements prompts.py501–502 path fallback, no branch difference. Production prerequisite satisfied.

**Stack:** Python3.10–3.14, required NumPy1.26–2.x, stdlib heapq, existing pytest/coverage/build/Twine tools. No new runtime dependency or publication.

## Scope and constraints

62 complete callable bodies including methods, nested functions and lambdas have individually prepared reasons in shared-workspace .tmp/part03g-prepared-reviews.json. New normalization/cache-copy helpers and benchmark callables require separate whole-function review. Preserve row reciprocal2pi convention, halfspace/vertex/rank/periodic tolerances, named points/aliases, first-tie policies, requested path/run/segment order and output shapes/dtypes. Keep ordinary ray unit-direction tolerance unchanged; magnitude must not define direction. Scalar boundary_scale can exceed floating range for subnormal rays; physical point/distance should avoid that intermediate. Existing extreme lattice/SVD/rational limits remain explicit.

## Tasks

- [x] Inspect complete bodies/callers and reproduce ray-scale errors, writable cached direction poisoning and isolated integer-modulo overflow; evidence .tmp/part03g-geometry-probes.json.
- [x] Add red regressions for scaled rays, invalid tolerances, cache list/array isolation with writable return compatibility, sampled output/labels/breaks/empty shape and impossible counts, bounded fixing masks, exact Dijkstra ties/relaxation and modulo grid congruence. Use independent or frozen references; mocks isolate only unrelated huge-coefficient SVD behavior.
- [x] Brillouin: private max-component ray normalization; intersect using the same unit-direction angular threshold, compute point/distance via normalized rays, reject nonfinite custom slack. Preserve basis, shell, volume/face construction and all numerical constants.
- [x] Strata: copy one writable array per distinct cached direction identity plus a fresh list per call; retain readonly grid/strata arrays and within-call aliases. Block fixing masks with unchanged einsum/residual expression. Reduce each integer operation modulo48 before grid multiplication, isolating congruence correctness from existing huge-coefficient float SVD limitations.
- [x] Kpath: precompute output size from zipped runs, reject unaddressable arrays, allocate points/distances/labels once, generate exact old Python fractions with fromiter and fill segments in the original order. Preserve legacy empty point shape, skipped junctions, run-break distances and label tuple. Replace sorted queue with heapq using identical(distance,node) ordering, seen and1e-12 relaxation.
- [x] Focused scientific tests and actual old/new production benchmarks with source hashes. Compare path metadata/labels exactly, normal geometry within existing numerical tolerance; corrected defects intentionally differ. Report warm-cache ownership overhead and output-sized memory honestly. Extend NumPy1.26.4 helper lane with new tests.
- [x] Independent whole-batch read-only review; resolve findings, freeze tracked Python bytes, one full/coverage/package/three installed checkpoint with workspace TEMP/TMP/PIP_CACHE_DIR.
- [ ] Promote only reviewed source-hashed rows after local gates; refresh final sdist docs/ledger, scoped commit/push dev, all16exact-head CI plus downloaded coverage ZIP before next production edits.

## Retained limits and judgments

Do not retune Wigner face residual/volume thresholds, grid denominator48, rank1e-9 rule, primitive rationalization, named-table or graph tie policies. Integer congruence repair does not establish arbitrary huge-operation floating SVD correctness. Cache safety preserves mutable return types at the cost of one point-direction list copy and a few unique vectors per call; benchmark this cost rather than claim universal speedup. Dense finite zone-node distance tables and first-hit node clustering stay unchanged unless measurements justify altering them. No scientific run cleanup, shared-root source edits, release/tag/upload or account changes.

Local gates passed;67 rows promoted after terminal full/install success. Final sdist, scoped commit/push and exact-head16 CI/artifact remain the final checklist item.
