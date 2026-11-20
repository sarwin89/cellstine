# Part 02b: contacts, covering, void kernels and reduction fallback

This checkpoint inspects all 45 function/lambda entries in contacts (15), covering (6), bulk voids (14) and planar voids (10), plus the exceptional geometry fallback. The previous contact optimizations are retained. The new fallback helper adds one function to the inventory. The complete ledger now has 1,086 entries: 100 reviewed unchanged, 15 optimized and verified, two investigated deferrals and 969 pending reviews. Inspection evidence exists for 117 entries; the full review remains incomplete.

## Changes and invariants

`self_image_contact` now processes first-atom rows in strips targeting at most 65,536 pairs. It retains translation order, row-major argmin order, strict winner replacement, distances, indices, species, ratios and report schemas. Inputs and required outputs remain linear; this is bounded working allocation, not constant process memory.

`branch_and_bound_maximum` checks the next child count against the existing probe budget before allocating children. Callback order, stopping condition and returned rigorous Lipschitz bound are unchanged. Planar covering starts with at most 12 coarse boxes per axis, matching the bulk heuristic. It retains the existing refinement tolerance and probe budget. Its valid upper bound may differ or be looser; public hollow geometry is checked separately.

The `_reduced_cell` RuntimeError/ValueError fallback previously applied a reduced-basis coefficient box to an unreduced basis. For rows `(10,0,0)`, `(94,1,0)`, `(0,0,10)`, it reported shortest translation 10 and a tested minimum image 3, versus true values sqrt(8) and sqrt(2). The new helper starts with a basis-row upper bound U and uses reciprocal Cauchy-Schwarz bounds `abs(n_i) <= U * reciprocal_norm_i` to enumerate the complete coefficient box. It tightens bounds as candidates improve and streams large boxes in chunks of at most 65,536 rows. Successful Delaunay reduction retains the previous arithmetic. Nonfinite/zero bounds or coefficients beyond float64's exact integer range raise ArithmeticError rather than claim an exact answer. Enumeration can still be expensive for exceptionally ill-conditioned unreduced bases; this is a correctness fallback, not a general speedup. Floating-point operations retain their numerical limits; no formal proof execution is claimed.

## Function review

The entries below include nested callbacks and sorting lambdas. Dataclass-generated methods are outside the AST function denominator. Callers in surface sites, defects, contacts and periodic geometry were checked alongside the existing tests.

Contact review preserves missing-label defaults and mismatch rejection, nonfinite covalent-ratio semantics, Cartesian/direct conversion, unwrap reference conventions, empty/singleton handling and the warning/note order. Report builders delegate to the same exact geometry kernels and retain their JSON fields. No scalar validation or scientific thresholds were changed.

Covering review checked box-reach corner enumeration, monotone best/bound updates, live-box pruning and the cached nearest-image expansion. The early budget check uses exactly the number of children previously materialized. Bulk covering initial counts, periodic seeds and refinement accuracy remain unchanged.

Bulk void review checked material bounds, flat-hull membership, rank/contact-direction classification, image reach, circumcentre filtering and empty-sphere validation. Merge order remains stable descending radius; final kind order survives equal-radius ties. Planar review checked frame/pseudoinverse conventions, reduced-cell transforms, circumcentre filters, local-maximum tests and stable `(-radius, uv)` ordering. Its nested evaluator still wraps fractional samples before nearest Cartesian queries. The documented neighbour cap is the unresolved completeness boundary.

### `core/contacts.py`

- `Contact.ratio`: reviewed—unchanged. Inspected scientific algorithm, callers, edge cases, tests and ordering/tolerance conventions; retain established behavior
- `Contact.as_dict`: reviewed—unchanged. Inspected scientific algorithm, callers, edge cases, tests and ordering/tolerance conventions; retain established behavior
- `_labels`: reviewed—unchanged. Inspected scientific algorithm, callers, edge cases, tests and ordering/tolerance conventions; retain established behavior
- `_covalent_sum`: reviewed—unchanged. Inspected scientific algorithm, callers, edge cases, tests and ordering/tolerance conventions; retain established behavior
- `closest_contact`: optimized—verified. Retained prior verified exact contact reduction and diameter tiling; this module review revalidated existing contracts and reports
- `closest_contact_cartesian`: reviewed—unchanged. Inspected scientific algorithm, callers, edge cases, tests and ordering/tolerance conventions; retain established behavior
- `layer_contact_report`: reviewed—unchanged. Inspected scientific algorithm, callers, edge cases, tests and ordering/tolerance conventions; retain established behavior
- `unwrap_group`: reviewed—unchanged. Inspected scientific algorithm, callers, edge cases, tests and ordering/tolerance conventions; retain established behavior
- `group_diameter`: optimized—verified. Retained prior verified exact contact reduction and diameter tiling; this module review revalidated existing contracts and reports
- `self_image_contact`: optimized—verified. Verified bounded pair/child allocations or coarse-grid setup with regressions, full-suite coverage, independent review and comparative measurements
- `structure_contact`: reviewed—unchanged. Inspected scientific algorithm, callers, edge cases, tests and ordering/tolerance conventions; retain established behavior
- `structure_contact_report`: reviewed—unchanged. Inspected scientific algorithm, callers, edge cases, tests and ordering/tolerance conventions; retain established behavior
- `contact_notes`: reviewed—unchanged. Inspected scientific algorithm, callers, edge cases, tests and ordering/tolerance conventions; retain established behavior
- `contact_report`: reviewed—unchanged. Inspected scientific algorithm, callers, edge cases, tests and ordering/tolerance conventions; retain established behavior
- `merge_notes`: reviewed—unchanged. Inspected scientific algorithm, callers, edge cases, tests and ordering/tolerance conventions; retain established behavior

### `core/covering.py`

- `grid_box_reach`: reviewed—unchanged. Inspected scientific algorithm, callers, edge cases, tests and ordering/tolerance conventions; retain established behavior
- `branch_and_bound_maximum`: optimized—verified. Verified bounded pair/child allocations or coarse-grid setup with regressions, full-suite coverage, independent review and comparative measurements
- `NearestAtomDistance.__init__`: reviewed—unchanged. Inspected scientific algorithm, callers, edge cases, tests and ordering/tolerance conventions; retain established behavior
- `NearestAtomDistance._rebuild`: reviewed—unchanged. Inspected scientific algorithm, callers, edge cases, tests and ordering/tolerance conventions; retain established behavior
- `NearestAtomDistance.__call__`: reviewed—unchanged. Inspected scientific algorithm, callers, edge cases, tests and ordering/tolerance conventions; retain established behavior
- `bulk_covering_radius_bound`: reviewed—unchanged. Inspected scientific algorithm, callers, edge cases, tests and ordering/tolerance conventions; retain established behavior

### `core/voids.py`

- `material_bounds`: reviewed—unchanged. Inspected scientific algorithm, callers, edge cases, tests and ordering/tolerance conventions; retain established behavior
- `_origin_in_flat_hull`: reviewed—unchanged. Inspected scientific algorithm, callers, edge cases, tests and ordering/tolerance conventions; retain established behavior
- `_classify_contact_directions`: reviewed—unchanged. Inspected scientific algorithm, callers, edge cases, tests and ordering/tolerance conventions; retain established behavior
- `_critical_point_classes`: reviewed—unchanged. Inspected scientific algorithm, callers, edge cases, tests and ordering/tolerance conventions; retain established behavior
- `_atom_images`: reviewed—unchanged. Inspected scientific algorithm, callers, edge cases, tests and ordering/tolerance conventions; retain established behavior
- `_circumcentres`: reviewed—unchanged. Inspected scientific algorithm, callers, edge cases, tests and ordering/tolerance conventions; retain established behavior
- `_triple_indices`: reviewed—unchanged. Inspected scientific algorithm, callers, edge cases, tests and ordering/tolerance conventions; retain established behavior
- `_triangle_circumcentres`: reviewed—unchanged. Inspected scientific algorithm, callers, edge cases, tests and ordering/tolerance conventions; retain established behavior
- `_empty_sphere_centres`: deferred—with reason. Inspected; existing 200-neighbor truncation prevents a universal completeness guarantee beyond the cap; dedicated Part02c will investigate a complete resource-safe replacement
- `_merge_void_centres`: reviewed—unchanged. Inspected scientific algorithm, callers, edge cases, tests and ordering/tolerance conventions; retain established behavior
- `_merge_void_centres.<lambda>@428:47`: reviewed—unchanged. Inspected scientific algorithm, callers, edge cases, tests and ordering/tolerance conventions; retain established behavior
- `find_void_sites`: reviewed—unchanged. Inspected scientific algorithm, callers, edge cases, tests and ordering/tolerance conventions; retain established behavior
- `find_void_sites.<lambda>@559:21`: reviewed—unchanged. Inspected scientific algorithm, callers, edge cases, tests and ordering/tolerance conventions; retain established behavior
- `_shortest_interatomic_distance`: reviewed—unchanged. Inspected scientific algorithm, callers, edge cases, tests and ordering/tolerance conventions; retain established behavior

### `core/planar_voids.py`

- `_plane_frame`: reviewed—unchanged. Inspected scientific algorithm, callers, edge cases, tests and ordering/tolerance conventions; retain established behavior
- `_plane_images`: reviewed—unchanged. Inspected scientific algorithm, callers, edge cases, tests and ordering/tolerance conventions; retain established behavior
- `_planar_nearest_neighbour`: reviewed—unchanged. Inspected scientific algorithm, callers, edge cases, tests and ordering/tolerance conventions; retain established behavior
- `_planar_covering_radius_bound`: optimized—verified. Verified bounded pair/child allocations or coarse-grid setup with regressions, full-suite coverage, independent review and comparative measurements
- `_planar_covering_radius_bound.evaluate`: reviewed—unchanged. Inspected scientific algorithm, callers, edge cases, tests and ordering/tolerance conventions; retain established behavior
- `_planar_circumcentres`: reviewed—unchanged. Inspected scientific algorithm, callers, edge cases, tests and ordering/tolerance conventions; retain established behavior
- `_is_local_maximum`: reviewed—unchanged. Inspected scientific algorithm, callers, edge cases, tests and ordering/tolerance conventions; retain established behavior
- `_reduced_planar_voids`: reviewed—unchanged. Inspected scientific algorithm, callers, edge cases, tests and ordering/tolerance conventions; retain established behavior
- `_reduced_planar_voids.<lambda>@271:21`: reviewed—unchanged. Inspected scientific algorithm, callers, edge cases, tests and ordering/tolerance conventions; retain established behavior
- `find_planar_voids`: reviewed—unchanged. Inspected scientific algorithm, callers, edge cases, tests and ordering/tolerance conventions; retain established behavior

### Geometry and benchmark support

`_reduced_cell` and `_unreduced_shortest_vector_length` are optimized and verified for the fallback correction. The other 35 pre-existing geometry function bodies are AST-identical to commit `be43f26`; their Part-02a decisions are retained with fresh module hashes, whole-suite verification and all nine original benchmark equality comparisons. The benchmark's five functions were re-read. `_baseline(ref, module_name="geometry")` adds an optional trusted-Git module selector to support isolated comparisons; the original command and nine geometry cases retain exact-output behavior. No new CLI interface is introduced.

## Measurements and regression evidence

Python 3.14.3 / NumPy 2.4.3, three-call medians; allocations traced separately with input arrays excluded and caches warmed. The existing benchmark loader loads baseline module bodies from `be43f26`; contacts/covering/planar comparisons share current dependencies to isolate each changed body. Fixed baseline-first ordering and machine load affect timings. Traced allocation is not process RSS; no timing threshold runs in CI.

| Case | Before ms | After ms | Before MiB | After MiB |
| --- | ---: | ---: | ---: | ---: |
| Self-image contact, 1,200 atoms | 228.955 | 218.522 | 77.104 | 3.674 |
| Covering, exhausted 50,000-probe budget | 7.451 | 0.467 | 10.856 | 0.812 |
| Planar covering, close pair at spacing 0.001 | 261.634 | 6.611 | 58.119 | 0.345 |

Contact dictionaries and the covering scalar matched the baseline exactly. Planar bounds were each checked against the analytic covering radius, rather than required to be identical. The public result retains both fourfold hollows at `(0.5005,0.5)` and `(0.0005,0.5)` with analytic radii to absolute 1e-10. The streamed fallback at shear 100,004 took 127.071 ms and 6.001 MiB; the old result was wrong, so no speedup comparison is claimed.

New regression cases reproduced old incorrect fallback values and excessive contact/covering/planar allocations before implementation. A separate old-code check confirmed the unsupported-coefficient guard previously failed to raise. All eight added cases pass. The focused combined selection passed 97 tests, and the original nine-case geometry comparison also passed.

Independent read-only review found no actionable regression. Checks included old/new self-image fields for sizes 1/2/7/257/400; 50 integer-sheared fallback cells against an independent enumeration; four large positive/negative streamed shears; old/new covering across five budgets in 2D and 3D; and square/skew planar bounds against 10,000 random probes plus independent periodic hollow radii. These checks do not establish universal completeness beyond the documented cap below.

## Verification

The frozen full suite passed **3,175 tests**, with four skips and 81 spglib deprecation warnings, in 260.73 seconds. Three skips concern the absent external Lean reference; one concerns the whole-period surface case. Python source/test SHA256 values remained unchanged during and after coverage collection. Ruff F checks and the inventory drift check pass.

| Scope | Statements | Coverage | Branches | Coverage |
| --- | ---: | ---: | ---: | ---: |
| Package | 12,107 / 14,012 | 86.40% | 3,085 / 4,370 | 70.59% |
| Contacts | 189 / 196 | 96.43% | 50 / 64 | 78.13% |
| Covering | 73 / 77 | 94.81% | 9 / 12 | 75.00% |
| Geometry | 419 / 445 | 94.16% | 96 / 118 | 81.36% |
| Planar voids | 142 / 152 | 93.42% | 21 / 32 | 65.63% |
| Bulk voids | 253 / 267 | 94.76% | 62 / 78 | 79.49% |

Coverage is diagnostic, not a completeness proof. Wheel and source archives build and pass Twine checks; all 140 packaged Python modules match the frozen source byte-for-byte. The source archive contains the ledger, report, regression tests, benchmark and review tooling, and excludes generated runs/output/temp data. Fresh core-only, CLI-extra and source-distribution installations pass numerical fallback, contact, covering, planar hollow, CLI help and dependency checks with NumPy 2.5.4. The initial standalone smoke driver had reversed minimum-image arguments; that driver was corrected without changing package sources. Pushed Python 3.10–3.14 and package CI must pass before the next part.

## Investigated deferral and next part

`_empty_sphere_centres` retains the existing `_MAX_NEIGHBOURS = 200` truncation. Inspection established a completeness boundary, not a concrete missed-void reproduction. It cannot inherit a universal completeness claim from passing finite tests. Part 02c must investigate the cap and candidate/triple enumeration, reproduce any defect and develop a complete resource-safe strategy before closing this row. The cap is not silently relaxed, replaced with a narrower workflow, or declared resolved here. The other investigated deferral remains the rational lattice basis issue from Part 01.


## Pushed CI checkpoint

Commit `a455f81447167791230d4c1811ac54a0e2040023` passed all five Python 3.10–3.14 jobs and the package build in [CI run 41175668140](https://github.com/sarwin89/cellstine/actions/runs/41175668140). The retained coverage artifact `13033086742` is non-expired and its recorded head SHA matches this commit.
