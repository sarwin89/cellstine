# Part03h: shared migration, record and surface resources

Baseline: dev `604d6609c5333b360826789eba6efef62fdbdcf5`. Its sixteen CI jobs and downloaded coverage digest, head, modules and statements were verified before these runtime edits. This batch reviews 64 original callable bodies across ten modules plus one new nested collector. Independent source-hashed review covers all 65 current bodies. Prepared notes alone do not count as completed review.

## Changes and compatibility

Migration pair costs and assignment certificates use bounded row blocks above 65,536 matrix entries. Small inputs keep their original expressions. Large normal inputs preserve arithmetic, solver/tie order and output-sized matrices while reducing temporary arrays. Custom, nonfinite and unusual certificate inputs retain the original fallback. Species matching groups indices once. Migration validates finite cell shape, nonnegative tolerance, species/count cardinality and addressable image counts early; NumPy endpoint counts and empty fallback counts work without ambiguous truthiness.

In-plane slab repetition preallocates and fills header-block/cell/atom order, preserving flags and zero-count or duplicate headers. This corrects compound slabs whose old coordinate order disagreed with species counts. Identity repeats return the original object. Symmetry records group species once, retaining first-seen ordering, wrapping and metadata policy, and reject mismatched atom labels. Termination totals aggregate duplicate chemical headers.

Scalar-level reduction copies before sorting, preserving caller ownership, and rejects invalid tolerances. Translation equivalence now requires a species-preserving bijection within the original fractional tolerance graph. Cheap missing-row/unique-choice paths precede exact assignment for ambiguous graphs; this fixes a Hall-deficit false positive without changing to Cartesian tolerance.

Nonplanar transforms first retain the legacy collection order. Only an incomplete atom count triggers a conservatively enlarged ordinary integer-transform search. The determinant-27 missing-image regression now contains all 27 atoms; compound positive/negative determinant cases preserve species, flags and periodic images. Incomplete final output raises instead of returning contradictory counts. This is not a general exact integer-lattice algorithm.

## Measurements

Interleaved three-repeat warm timings compare the predecessor with the final source, native threading limited before NumPy import. Allocation probes are separate Python/NumPy tracked peaks, not process RSS. Exact complete legacy migration and already-complete nonplanar outputs were compared; corrected incorrect outputs use independent physical oracles.

| Case | Before | After | Tracked peak before / after |
| --- | ---: | ---: | ---: |
| Pair costs, 24 atoms | 0.176 ms | 0.180 ms | 0.054 / 0.054 MiB |
| Pair costs, 600 atoms | 39.45 ms | 43.64 ms | 32.96 / 9.23 MiB |
| Certificate, 16 rows | 0.020 ms | 0.023 ms | 0.007 / 0.007 MiB |
| Certificate, 1,000 rows | 8.62 ms | 14.28 ms | 15.32 / 1.55 MiB |
| Elemental repeat, four atoms, 60 by 60 | 65.95 ms | 1.55 ms | 1.57 / 0.80 MiB |
| Species codes, 20,000 labels / 100 types | 51.70 ms | 3.71 ms | 0.170 / 0.173 MiB |
| Record grouping, 20,000 atoms / 100 types | 287.44 ms | 15.82 ms | 4.01 / 2.47 MiB |

The large blocked calculations trade CPU time for lower temporary memory. Identity repetition has no meaningful speedup. No blanket parallelism or changed worker default is introduced here. Existing bounded process/thread builds and uncompressed NPZ defaults remain.

## Verification and limits

Initial regressions failed on the predecessor; review-discovered Hall, missing-image and invalid-tolerance cases also have red/green evidence. Final broad focused verification passed 750 tests with one skip; three additional compound/orientation oracle cases passed on unchanged runtime source. There are 51 new regression cases, included in minimum-NumPy and nine portable CI lanes. One clean frozen full run passed **3752 tests**, with 4 skips and 81 warnings in 886.59s. All 285 Python source hashes remained fixed. Local branch coverage is 14153/15806 statements and 3964/5154 branches. Wheel/sdist builds and Twine checks passed; installed core, CLI and sdist environments passed new compound/migration/record/Hall/nonplanar cases and native serial/process/thread builds with matching files and clean dependency checks. Final documentation/ledger refresh changes only sdist support files; unchanged runtime/full/install gates are retained.

Seven reviewed callables remain investigated deferrals: `_primitive_surface_vectors_from_lattice`, `_primitive_surface_vectors`, `_build_native_primitive_surface_cell`, its `_layer_atoms`, `_structure_from_transform`, its `_collect`, and `_structure_from_inplane_transform`. The finite-box primitive fallback can miss shorter vectors in highly sheared cells; primary rational reduction is unchanged. Floating determinant/membership arithmetic, cubic image searches and arbitrary large/sheared transform completeness remain limitations. Ambiguous translation graphs retain dense graph/assignment resource costs. These are explicit incomplete algorithm reviews, not claims of universal scientific completeness.

No release, tag or package upload is part of this checkpoint. A scoped dev push, exact-head supported-version CI and downloaded coverage readback must close before the next runtime batch.


Ledger: **1224 functions**, **990 with inspection evidence**; 682 reviewed—unchanged, 299 optimized—verified, 243 deferred—with reason. Deferred rows include 234 pending and 9 investigated limitations (two prior plus seven this batch). The project-wide review remains incomplete.


Closed on dev4c983a33edc02e938aa817917d5d93a6f0063400: all16 jobs in CI42032550909 passed. Downloaded artifact13417667678 SHA256e26c53e7353239781e6eab6043750835e68c28bb3ecbd76604ff607f3af78c87 matched digest/head/job/module/statement expectations. CI14155/15806 statements and3964/5154 branches; local14153 differs only by known prompts.py501-502 fallbacks, with no branch delta. Closure preceded Part06d source edits.
