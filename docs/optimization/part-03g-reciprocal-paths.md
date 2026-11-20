# Part03g: reciprocal geometry and path resources

Base427e2d5d6bd8cbec41a8453c53a1fe1f2b964b4f passed all16 exact-head CI41787883849 jobs. Downloaded coverage13302931287 SHA256e78173b1a7de0f72ab8504e8fa89ce46159b5e05328cb8e3e8d892d712eb1c05 has verified head/module/statement sets; only two already-investigated path-display fallback statements differ from local coverage, with identical covered branches. Local full/package/installed gates passed and67 individually reasoned reviews are promoted. Commit1e01173aa2725431d035d8ecfdcc519fc529ab11 is pushed to dev; all16 exact-head CI41820006659 jobs passed. Downloaded coverage13316437813 ZIP SHA256d1564189d31d0adb3c7f1683b3c8d546d10e8ec489944eef7fd2136b3cfb8db0, head/module/statement sets verified; only known prompts.py501-502 fallback adds two CI statements without branch delta. No release or package publication is authorized.

## Behavior and invariants

Wigner-Seitz boundary rays are scaled by their largest absolute component before halfspace intersection. A finite nonzero ray therefore defines the same physical boundary point regardless of magnitude, including1e-300 and1e300 directions on ordinary cells. The angular acceptance threshold remains the existing unit-direction tolerance: projections of the scaled ray are compared with tolerance times its norm. boundary_scale converts the final scalar back to original-ray units; boundary_point and zone_boundary_distance avoid an overflowing scalar intermediate. A true scalar scale beyond floating range remains unrepresentable. Custom cell tolerance must be finite and positive. Basis/shell/face/volume thresholds and reciprocal2pi convention are unchanged.

The48-denominator grid first reduces integer operation coefficients modulo48, preventing int64 dot-product overflow before the congruence test. This establishes the integer stage, not floating SVD classification for arbitrarily huge conjugated operations. Floating little-group masks use the same einsum, centered residual and componentwise tolerance expression in blocks of at most4,096 rows, targeting65,536 point-operation pairs. Returned masks still require output-sized storage; operations and input coordinates are cast once.

Cached line directions are private. Each result contains a fresh writable list and one fresh writable array per distinct direction object; aliases remain shared within that one return. Readonly grid/strata arrays continue to be shared. This prevents returned-list/array edits from changing later classifications. It intentionally adds a list-copy cost per warm lookup; no universal cache speedup is claimed.

BandPath.sample allocates points/distances/labels once and fills one segment at a time. Fractions retain the old Python scalar division, then NumPy multiplication/addition in the original order. Junction omission, path/run order, break distances, endpoint labels, tuple labels, float64 arrays and the legacy empty point shape(0,) are retained. Impossible address-sized requests fail before per-segment work. Memory still scales with the returned samples and per-segment fractions.

Dijkstra uses a heap of the same(distance,node) tuples in place of repeated full sorting. Neighbor traversal, first-visit handling, predecessor updates and1e-12 relaxation remain unchanged. This is an internal queue optimization; no new path selection policy is introduced.

## Regression and review scope

The62 original complete callable bodies in brillouin/strata/kpath, including nested functions and distinct lambdas, have individual inspection notes. Two private helpers and three benchmark functions are separately reviewed. Scientific callers include reciprocal boundary scaling, kpoint writing, band sampling, native symmetry and path derivation. Named tables, aliases, representative selection and all other inspected sound bodies remain unchanged.

The new39-case suite covers scaled/nonfinite/zero rays, custom tolerances, cache ownership and alias compatibility, exact scalar-reference samples across16 run/count combinations, empty shapes, early impossible-size rejection, chunked masks including empty input and tolerance boundaries, exact sorted-queue reference distances/predecessors and isolated integer congruences. The initial red run had12 failures/27 passes; its order-three fixture incorrectly applied the2D minimal polynomial to the invariant z axis. Correcting that fixture's z entry to1 made its exact Python-object identity check meaningful. After implementation and fixture correction, the complete Brillouin/kpath/new-regression selection passed165 tests in31.92seconds. No claim is based on that invalid initial fixture. The NumPy1.26 helper CI lane includes the new regression file.

## Production measurements

`python benchmarks/benchmark_reciprocal_paths.py --baseline 427e2d5 --repeats 3` loads actual old modules from the verified commit and compares complete outputs before timing. Sampling uses the same prepared BandPath instance; mask/graph/cache inputs are identical. Other dependencies are shared and setup is excluded. Three-sample local Windows/Python3.14.3/NumPy2.4.3 observations use warm execution and tracemalloc peaks, not RSS or CI performance thresholds.

| Case | Old ms | New ms | Old peak MiB | New peak MiB |
| --- | ---: | ---: | ---: | ---: |
| Small path,20 divisions |0.377|0.138|0.0248|0.0079|
| Large path,20,000 divisions |341.26|20.15|27.62|5.71|
|24,000 points x48 masks |145.59|128.15|105.47|7.10|
| Synthetic900-node graph,24 neighbors |77.20|3.50|0.116|0.116|
| Warm cached grid with owned directions |0.001|3.287|0.00017|0.860|

Large sampling is16.93x faster locally; the synthetic graph is22.09x faster. The latter isolates queue work and is not a claim that typical small crystal paths improve by that factor. Warm cache ownership is deliberately slower. Complete point/distance/label/mask/predecessor/direction outputs compare exactly. Corrected ray/cache/congruence behavior is asserted separately, since reproducing the old defects is not the goal. Numerical scaling improvements do not establish all extreme-lattice geometry domains.

## Checkpoint and retained limits

Independent review covers all67 current source-hashed callables with no actionable findings. Its small probes confirm near-threshold angular masks, smallest subnormal axis intersection, huge mixed rays,192 within-call direction aliases and object-integer congruences. The reviewer explicitly leaves full/package/benchmark execution to this checkpoint and retains the stated numerical-domain/storage limits; these exclusions are accepted without broader claims.

The one frozen full run passed **3,649 tests, 4 skipped**, with81 warnings in1141.35 seconds. Local coverage is **13924/15611 statements** and **3856/5058 branches**. All282 tracked Python byte hashes matched the tested freeze. Wheel/sdist build and Twine passed; all144 runtime modules match both archives. Three isolated installed core-wheel/CLI-wheel/sdist environments passed representative NPZ search/build, native symmetry and reciprocal/path probes, root help and dependency checks. Installed environments use Python3.14.3/NumPy2.5.4.

Final ledger:905/1,220 functions with inspection evidence,623 reviewed unchanged,280 optimized verified,315 pending and two investigated deferrals. Documentation/ledger closure refreshes only the sdist and final archive readback, without repeating unchanged runtime full/install gates. Exact-head16-job CI and downloaded coverage digest/module/statement readback remain required before another production batch.

Retain the existing grid denominator, SVD/rationalization limits, Wigner geometric construction thresholds, first-hit clustering and finite graph domain. Cached grid/strata arrays retain their readonly-flag contract; this is not an immutable-storage security boundary. Output-sized allocations remain. The two earlier investigated numerical-domain deferrals and project-wide pending reviews remain open.
