# Part 02c: complete void support and bounded candidate processing

Local verification passed. This checkpoint resolves the neighbour-count truncation investigated in Part 02b, within the documented radius and numerical limits. The pushed supported-version CI gate passed; the project-wide systematic review is incomplete.

## Reproduced omission

The old `_MAX_NEIGHBOURS = 200` branch silently removed every neighbour after the nearest 200, both as candidate supports and as blockers in the emptiness test. In a 10 Angstrom cube with 202 atoms at x=linspace(.49,.51), y=z=.5, the public search returned no sites with valid cutoff 18 and vacuum filtering explicitly disabled. Periodic geometry gives a largest empty-sphere radius sqrt(4.9^2+5^2+5^2)=8.602906485601247 Angstrom. The first regression was verified red on a455f81. The full original 202 atom regression is retained.

A second independently analytic periodic line, x=arange(202)/202, should have 202 distinct transverse maxima of radius sqrt(50+(5/202)^2). The current implementation returns all 202 with the correct radii and positions when merging is below the site spacing. This new regression passes; it supplements the original dense-cluster test.

## Algorithms and preserved behavior

Neighbour count now selects an algorithm, never a truncated mathematical domain. Above 200 neighbours, conservative Voronoi support bounds remove only contact planes proven unable to meet an accepted centre. All original neighbours remain available for emptiness checking. Compatible triples remain lexicographic; pair and triangle saddles retain their category/order. Equality and uncertain planes remain eligible. The small-neighbour path retains its established enumeration arithmetic.

Relative support edge e and ideal centre c obey e.c=|e|^2/2. The existing nearest-distance emptiness tolerance 1e-6 implies a relaxed halfspace. Starting with a radius box, interval contraction minimizes the other two coordinate contributions to tighten each axis. Only an outer box is maintained; stopping after 32 sweeps weakens pruning rather than limits candidates. A plane is discarded only when its maximum over this outer box is strictly below its support value with a numerical guard.

Finite absolute-coordinate arithmetic requires distinguishing solved offsets, returned rounded centres and triangle radii. A three-ULP pair near 1e10 exposed an exclusion despite acceptance by the old emptiness tolerance; that new red/green regression now passes. The implementation enlarges the radius box by a coordinate guard delta and uses relaxed unnormalized RHS allowance (R+delta)*(1e-6+delta), plus roundoff guards. Short coefficients are skipped conservatively. Nonfinite/invalid box bounds retain all supports. These are numerical safeguards, not a formal exact-real proof for arbitrary floating-point conditioning.

Compatible triples use bounded chunks. Large support sets use compatible-graph triangle enumeration without a full cubic index table; small sets may use a fixed-size face table. Candidate arrays are solved/refined in blocks, and identical Cartesian nearest queries are deduplicated. A spatial memo retains bounded buckets; a verdict is reused only when the 1-Lipschitz nearest-distance bound certifies it. Near the emptiness threshold, the original query is evaluated exactly. Bucket width changes reuse opportunities and never the acceptance tolerance.

Triple-index construction streams integers directly into NumPy instead of building one Python tuple per combination. Aggregate retained triple-cache arrays are capped at 8 MiB. An uncached 200-neighbour table can still require about 30 MiB transiently; the large path does not request such a full table. Contact classification retains the same rays/constraint test while evaluating the ray-by-contact matrix in strips. Quadratic contact-pair/ray storage remains; the cubic comparison matrix is removed.

Large maxima-only candidate sets use stable radius-priority chunks of 128. The periodic-neighbour relation is the same as final merging. Already accepted maxima suppress later nearby candidates. Within a chunk, candidates without earlier pending neighbours form a mutually nonadjacent frontier and can be classified together; only accepted maxima suppress later nodes. Survivors are restored to stable priority order. Saddle searches keep classification before kind-specific merging, and nonpositive merge distances retain batch classification. A singleton implementation caused a measured6x ordinary-supercell slowdown and was replaced with this frontier batching.

## Function review scope

`core/voids.py` now has 21 AST function/lambda entries, including 7 new helper/nested functions. Existing `material_bounds`, `_origin_in_flat_hull`, `_critical_point_classes`, `_atom_images`, `_circumcentres`, `_triangle_circumcentres`, `_merge_void_centres`, its sorting lambda, `find_void_sites` sorting lambda and `_shortest_interatomic_distance` were re-read; their bodies remain unchanged apart from lambda source positions. Existing `_classify_contact_directions`, `_triple_indices`, `_empty_sphere_centres` and `find_void_sites` carry the scoped optimizations described above. New `_void_support_bounds`, `_compatible_void_faces`, `_void_triple_batches`, `_empty_void_candidates`, `_streamed_void_centres`, nested `retain` and `_maxima_before_merge` were individually inspected and independently reviewed.

The three functions in `benchmarks/benchmark_voids_memory.py` are included in the inventory: `_cold_triples` clears only the compared module's local cache before measurement; `_public` returns complete dataclass dictionaries for exact comparisons; `main` validates repeats, resolves a trusted Git baseline, constructs inputs before tracing, asserts outputs, records versions/hash and distinguishes the corrected dense case from speedup comparisons. The benchmark shares current geometry/reduction dependencies to isolate void-module changes.

## Measurements

Reproduce from a Git checkout:

```powershell
python benchmarks/benchmark_voids_memory.py --baseline a455f81 --include-dense-anchor
```

Python 3.14.3 / NumPy 2.4.3; three-call median timings, traced allocations measured separately, inputs excluded. The baseline cold-cache case includes recomputation each time. Other caches are warmed by equality assertions. Fixed before/after order and concurrent machine load influence timings; no timing threshold is a CI gate. Traced allocation is not process RSS.

| Case | Before ms | After ms | Before MiB | After MiB |
| --- | ---: | ---: | ---: | ---: |
| cold-triples-120 | 101.781 | 34.926 | 36.350 | 6.429 |
| classification-204 | 18.786 | 9.968 | 49.693 | 2.851 |
| bcc-maxima | 26.138 | 25.273 | 8.502 | 8.502 |
| bcc-saddles | 36.371 | 33.266 | 8.501 | 8.501 |
| sc64-default | 137.899 | 150.960 | 5.279 | 5.280 |
| sc64-wide-merge | 173.148 | 147.141 | 19.433 | 19.421 |
| dense-anchor-5453 | not a valid old result | 2584.205 | n/a | 13.129 |

Baseline/public comparisons assert every field and site order. The dense-anchor case measures one anchor of the original 202 atom geometry, with 5,453 original neighbours, and validates its local radius within the pre-existing1e-6 emptiness tolerance. It is a correctness repair with combinatorial cost, not a speedup over the old incorrect empty result. Returned raw centre storage remains output-sized. NearestPointQuery may use one query's linear storage for an overfull bucket even when that exceeds 65,536 entries; no constant-RSS or universal runtime guarantee is claimed.

## Independent review and test evidence

The reviewer found no remaining material issue after the coordinate guard and frontier changes. Checks include 250 small direct-uncapped maxima/saddle comparisons; 205-neighbour lexicographic triples; 24 original public crystal/random-skew exact baseline searches; 220 classifier comparisons including tolerance perturbations; 500 large-offset clouds; 24 additional public searches across negative/zero/small/large merge distances; 105 synthetic merge comparisons and 36 additional comparisons with 129–385 candidates frontier cases including chains crossing 128 and 256. Every public comparison retained full field/site-order equality. The coordinate-roundoff bug was found separately by the parent and fixed with a new regression, followed by fresh review.

The focused fast selection passed **14 tests** (the original dense case deselected), and **63 existing void tests** passed. The original dense focused run passed 13 tests in 481.34 seconds on an intermediate implementation. The final frozen full suite then passed **3,190 tests**, with **4 skips** and **81 spglib deprecation warnings**, in **717.44 seconds**. All 241 Python source/test hashes remained unchanged through and after the run. The long-test diagnostic trace confirmed active enumeration; it was not a failed assertion or a killed test. Three skips concern the absent external Lean reference and one the whole-period surface case. No formal Lean execution is claimed.

New regressions cover triple and classification allocations, independent uncapped geometry, both sides of the 1e-6 emptiness boundary, streamed triples above the old cap, cospherical/short edges, large coordinate offsets, early classification, stable equal-radius/nontransitive merging and all 202 periodic maxima. Interrupted exploratory runs after source changes are not counted as passed verification.

| Scope | Covered statements | Statement coverage | Covered branches | Branch coverage |
| --- | ---: | ---: | ---: | ---: |
| Package | 12,292 / 14,200 | 86.56% | 3,161 / 4,454 | 70.97% |
| `core/voids.py` | 437 / 455 | 96.04% | 138 / 162 | 85.19% |

Coverage diagnoses gaps; it does not prove universal numerical completeness. Ruff F checks, the tracked-function inventory check and Git whitespace checks pass. Source-level comparison establishes that exactly four existing void function bodies changed and the other ten remain AST-identical to `a455f81`; seven helper/nested functions were added. Together with the three benchmark functions, the ledger now has **1,096 functions**, **127 with inspection evidence** and **969 pending reviews**. The one investigated deferral is the separate rational-lattice-basis issue from Part 01.

Wheel/sdist builds and Twine checks pass. All 140 packaged Python modules match frozen source bytes. The source archive includes the new tests, benchmark, review tooling, report and ledger, and excludes generated run/output/temp data. Fresh core-only, CLI-extra and source-distribution installations pass the 202-maxima, large-offset, classification, bounded-cache, CLI-help and dependency checks with NumPy 2.5.4. Documentation-inclusive source archive regeneration follows final ledger/report updates; package code is unchanged.

## Pushed CI gate

The scoped commit must pass all five Python 3.10–3.14 jobs and package checks, with the retained coverage artifact's head SHA verified, before Part 03 starts. This checkpoint does not authorize publication or assert that the remaining 969 source functions have been reviewed.


## Supported-version CI readback

Commit `1db9be4159c413b45adf7f8fa0c5376c7fdb5966` passed all Python 3.10–3.14 jobs and package checks in [CI run 41283962742](https://github.com/sarwin89/cellstine/actions/runs/41283962742). The retained, nonexpired coverage artifact `13074945312` records this exact head SHA. This gate was checked before Part 03 began.
