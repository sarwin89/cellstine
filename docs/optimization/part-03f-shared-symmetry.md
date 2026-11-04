# Part03f: shared symmetry resources and exact tolerance checks

Base1a2c3bb57039668333740915a824c653b455c313 passed all16 CI41782577167 jobs. Downloaded coverage artifact13299884630 has verified ZIP SHA2567545a47e915103c01e395786975db837207772d20084172330ca5b1b24362fb4 and exact head/module/statement sets. CI covers13,812/15,510 statements and3,797/5,010 branches; its only extra local statements are prompts.py501–502's established absolute-path fallback.

This batch passed the local full/package checkpoint and47 individually reasoned function reviews are promoted. It remains uncommitted; scoped commit/push and exact-head16-job CI/artifact checks are next. No project-wide completion, release or package publication is claimed.

## Implementation and invariants

The internal integer-box iterator yields bounded fresh int64 rectangles in meshgrid ij/ijk order, with later axes varying fastest. It keeps whole suffix axes and a slice of the preceding axis, avoiding per-point integer division and full axis/linear-index arrays. Coordinate bounds must be representable as int64; huge box products can stream without a full product-sized allocation. Output vectors remain in exactly the old order.2D and3D enumeration retain the same coordinate bounds, norm expressions, inclusive thresholds and empty shapes;3D uses one float conversion per block. Returning every accepted vector still needs output-sized storage, and enumeration time still depends on box size.

Decorated2D matching targets65,536 pair comparisons per row block, with at least one row for larger site sets. Original reference/partner order, shifted coordinates, species and height constraints, centered wrapping and both row-and-column existence tests are retained. The change bounds temporary pairwise arrays; it does not replace the existing rule with another assignment algorithm or remove quadratic worst-case work per translation.

Provided-operation permutation/orbit helpers lazily construct one call-local PeriodicSiteIndex, reusing it across operations. Empty operation sets and empty positions build no index. Image arithmetic, invalid-operation identity/skip behavior, lowest-index orbit labels and zipped operation order remain unchanged. No persistent/global cache is introduced.

The symmorphic-setting helper intends a componentwise periodic tolerance. Its old Euclidean nearest match could lie outside that cube even when another center satisfies it. A further regression showed the cube corner at the existing1e-6 threshold could be lost to floating rounding in a tightly circumscribed sphere. The candidate search now conservatively encloses the cube with a2*tolerance radius, then applies the unchanged componentwise acceptance threshold. Only ambiguous selected rows scan additional centers in bounded blocks; missing enclosing-ball candidates still fail immediately. Typical separated centering sets retain the fast path. The constructed near-center examples establish the helper contract; no widespread physical classification error is claimed.

Species labels are validated before translation_group's empty return, aligning its reduction callers with primitive_cell's one-label-per-atom rule. Valid empty structures remain valid. The planar-gauge loop creates its float transform after rejecting nonunimodular integer pairs; traversal,625 rewriting candidates, reduction criteria, rounded keys and tie policy are unchanged.

## Regressions and review scope

43 original callables have individual prepared reasons across symmetry2d, pointgroup3d and symmetry3d, including the nested planar-gauge lambda. The new grid iterator and three benchmark functions are reviewed separately. Scientific callers include native symmetry datasets, site/orbit generators, centering/primitive layers, moire search/reduction, surfaces, reciprocal meshes and kpaths.

The prepared51-case suite first produced37 expected failures and14 passes: nearest/cube mismatch, empty-species inconsistency, repeated index construction, missing bounded iterator and full pair/grid allocations. Three further cube-boundary cases produced1 expected failure/2 passes before the enclosing-radius correction. All54 now pass. Tests include independent integer ellipsoid/planar-form oracles, row ordering, empty outputs, streaming a box beyond full int64 index size, invalid bounds, no-operation behavior, late row failure and a missing target column across matching blocks. The minimum-NumPy helper lane is extended with this file; its one-line configuration change was independently reviewed. The affected symmetry/centering/orbit/primitive-layer/surface/reciprocal/kpath selection passed914 cases with11 spglib deprecation warnings in117.53seconds.

## Production measurements

`python benchmarks/benchmark_symmetry_resources.py --repeats 3` loads the three old modules from the frozen base, reconnects their point-group exports and compares complete outputs before timing actual old/new functions. Other dependencies are shared. Three-sample Windows/Python3.14.3/NumPy2.4.3 observations use warm local execution; peaks are Python-tracked allocations, not RSS.

| Case | Old ms | New ms | Old peak MiB | New peak MiB |
| --- | ---: | ---: | ---: | ---: |
|2D small grid|0.053|0.052|0.0063|0.0050|
|2D skew grid|115.91|117.37|154.98|3.61|
|3D small grid|0.104|0.083|0.0294|0.0266|
|3D anisotropic grid|13.03|10.60|11.20|5.32|
|1,024-site decorated match|42.87|40.31|48.02|4.09|
|216 sites/48 provided permutations|178.03|18.15|0.784|0.701|
|216 sites/48 provided-operation orbits|185.20|14.77|0.783|0.699|
|Native BCC dataset|21.55|21.59|0.0482|0.0481|
|Planar gauge|1.435|0.748|0.00253|0.00253|

The skew case establishes a memory improvement with comparable/slightly slower observed runtime. BCC and very small cases do not establish universal speedup. Provided-operation gains follow removing repeated index construction. Corrected centering and invalid-species cases intentionally differ from the baseline and have separate regressions; normal valid benchmark outputs compare exactly. Final measured source hashes are recorded in shared-workspace .tmp/part03f-benchmark.json. No timing threshold is imposed in CI.

## Retained limits

All scientific tolerances, lattice row/column conventions, signed mirror policy, proper subgroup handling, species/atom ordering, primitive c-vector treatment and experimental N-layer policy remain unchanged. Type/symbol classifiers assume actual finite crystallographic rotations/groups; they are not generic closure validators for arbitrary unimodular matrices. Native3D nearest-site bijection policy and decorated2D row/column existence policy are retained, including their different ambiguous-site behavior. Rational-lattice and extreme floating-scale guarantees remain the existing investigated deferrals. Ambiguous centering fallback can cost a product of query/center counts, but does not allocate that full product.


## Local checkpoint closure

The frozen full suite passed **3,610 tests,4 skips,81 deprecation warnings in721.29seconds**. The original run completed successfully after an observational faulthandler stack in the existing dense-void case; it was not restarted. Coverage records13,889/15,583 statements and3,845/5,052 branches, compared with13,810/15,510 statements and3,797/5,010 branches at the preceding local checkpoint. A coverage percentage does not replace scientific assertions.

All280 tracked Python files match the frozen set/byte hashes. Wheel and sdist builds plus Twine checks pass;144 package modules match both archives. Core wheel, CLI wheel and sdist installed smokes under NumPy2.5.4 pass NPZ and N-layer builds, ordered integer blocks, exact centering boundaries, provided-operation permutations/orbits, native BCC dataset, empty-species rejection, command help and dependency checks. Workspace TEMP/TMP/PIP_CACHE_DIR were used, with offline cached isolated sdist build dependencies. Final closure documents/ledger are read back from the refreshed sdist without repeating unchanged runtime gates.

The47-callable independent review includes every original function, the nested gauge lambda, the new iterator and three benchmark functions, with individual source-hashed reasons and no actionable findings. The final benchmark and reviewed module byte hashes match frozen source. The ledger now has1,215 functions:566 reviewed unchanged,270 optimized verified,2 investigated deferrals and377 pending reviews (838 with inspection evidence). The47 additional entries have inspection evidence; untouched entries remain pending.

Scoped commit/push and all16 exact-head CI jobs, including the expanded NumPy1.26.4 helper lane and nine portable-format lanes, remain required before the next production batch. Download and verify the coverage ZIP digest/head/module/statement sets and investigate any environmental difference.
