# Part06d: bounded stacking mirror enumeration

Predecessor: dev `4c983a33edc02e938aa817917d5d93a6f0063400`. All 16 exact-head CI jobs and downloaded coverage checks closed before runtime changes. Independent complete-body review covers 29 original stacking callables and one new private helper.

## Behavior and numerical bounds

The fixed `[-3,3]^4` box missed all four mirrors of the square cell with rows `(1,5,0),(0,1,0),(0,0,12)`. It also missed reflected/reversed-basis square and sheared rectangular cases. The new ordinary path finds the complete independent integer-certified sets, including shears 100 and 300. Matrix ordering is still lexicographic; the final NumPy determinant and `M @ G @ M.T` residual/tolerance expressions are retained. Ordinary square/hexagonal compound default records match predecessor fields and arrays exactly.

For positive definite G, its absolute quadratic sum is bounded by twice the diagonal quadratic sum. A conservative diagonal floating error envelope E includes the two dot products and subnormal error; final subtraction rounding also expands the target interval. Accepted rows therefore lie inside a `G-E` ellipse and outside a `G+E` ellipse. Fraction setup and integer square roots derive outward annulus intervals without floating bound rounding. Enumeration scans the tighter coordinate, then maps and sorts rows in their original axes. Only candidate row pairs reach the original final predicates.

Each of the two target-row scans is bounded at 65,536 coordinate iterations. Each candidate row list and the final matrix-pair count are bounded at 2,401, the legacy matrix-box count. Unresolved/nonfinite metrics, broad tolerances, coordinates outside exact float integer representation, or work exceeding these budgets retain the established box. This preserves resource behavior; completeness outside the accepted ordinary domain remains unproven. No all-domain point-group claim is made.

Documentation now states the actual reduced-basis/short-representative precondition and that projected point-lattice helpers do not receive chemistry or heights. Existing stacking gauges and chemical policy are unchanged.

## Measurements

Three interleaved warm unprofiled predecessor/final measurements, native threading limited before NumPy import, show:

| Cell | Before | After | Mirror counts before / after |
| --- | ---: | ---: | ---: |
| square | 32.118 ms | 0.969 ms | 4 / 4 |
| rectangle | 38.627 ms | 0.709 ms | 2 / 2 |
| oblique | 41.506 ms | 0.880 ms | 2 / 2 |
| sheared_square | 47.841 ms | 2.151 ms | 0 / 4 |
| sheared_rectangle | 34.392 ms | 1.086 ms | 0 / 2 |
| reverse_sheared_square | 32.316 ms | 1.363 ms | 0 / 4 |
| left_handed_square | 31.556 ms | 1.553 ms | 0 / 4 |

Separate Python/NumPy tracked peaks rise from about 1.4–2.1 KiB to 6.6–7.5 KiB because of exact bound construction; these are not process RSS measurements. Corrected cases are compared to independent exact certificates, not to incorrect predecessor outputs. The operation-count regression drops normal determinant calls from 2,401 to fewer than 100. No CI timing thresholds or new dependencies.

## Verification and open science issues

Thirty regressions cover seven exact certificate sets, larger shears, compound default isometry/headers/flags/metadata, writable ownership, invalid/wide/zero/tiny/nonfinite domains, near-hexagonal tolerance boundaries and bounded resource behavior. Red evidence: 6 initial failures / 18 passes, then a larger-shear failure, then two review-discovered resource failures. The focused batch passed 97 tests; 180 scalar original-predicate comparisons and 15 exact integer conjugations also passed. Independent review covers all 30 current bodies, with the resource finding resolved and three explicit investigated deferrals. One clean frozen full run passed **3782 tests**, skipped 4, with 81 warnings in 951.45s. All 286 Python hashes stayed fixed. Local coverage is 14218/15875 statements and 3986/5180 branches. Wheel/sdist builds, Twine and three installed core/CLI/sdist environments passed actual mirror, compound/resource, migration/record and native serial/process/thread checks plus dependency checks. All 144 runtime modules match the wheel. Only final documentation/ledger support changes refresh the sdist; unchanged runtime/full/install gates are retained. Scoped dev push and exact-head CI/artifact closure remain.

Three investigated limitations remain: extreme `_gauss_reduce` convergence/shortest-vector guarantees, arbitrary-supercell primitive discovery/resource behavior in `_layer_point_lattice`, and sense-changing default reflection selection. The last is a concrete existing Important defect: on a 60° triangular slab, a geometric mirror can preserve the hollow coset, so assembly `prepare_stacking(...,top_stacking='cba')` can return ABC with `top_mirrored=True`. The lower-level relative helper detects/refuses that result. The next batch must select a coset-reversing operation and validate the realized high-level request; this batch retains default geometric-mirror order and does not claim that defect solved.

The overall goal remains incomplete. No release, tag or package upload is authorized.


Ledger: **1225 functions**, **1020 with inspection evidence**; 707 reviewed—unchanged, 301 optimized—verified, 217 deferred—with reason. Deferred rows comprise 205 pending and 12 investigated limitations. This batch adds 30 inspection records; the overall goal remains incomplete.
