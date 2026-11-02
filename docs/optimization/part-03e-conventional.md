# Part 03e: conventional-cell work reuse and prompt closure limits

Baseline: `bba657631259dfff11f3accb15ee67a0264c84b8`. All seven CI jobs passed in run `41427690267`; exact-head coverage artifact `13133367158`, digest `1592a0121be0083822b6680ee690d1afedf45174d154371cd8f40cf2a785ddfd`. Claude's uncommitted root work remains separate; its final patch handoff is being prepared in a scratch tree while this source checkpoint is frozen.

## Changes and preserved behavior

Orthogonal conventional-cell construction previously repeated the same Delaunay reduction and lattice-point shell for each of its three axes. Axial and monoclinic branches repeated it twice. These branches now prepare the identical cloud once and pass it to the private shortest-vector helpers. Standalone calls retain their previous default construction. This is branch-local reuse with no persistent cache or new public API.

The shared cloud is read without mutation. Candidate order, stable shortest-vector ties, direction signs, thresholds, cell handedness and output arrays retain their old calculations. Coset enumeration, centring, primitive/conventional transforms and idealisation reporting are unchanged.

`close_matrix_group` previously checked its cap after an entire product round, including when the supplied generators already exceeded the cap. It now checks each new distinct insertion. Duplicates at the limit remain accepted; an excess element produces the same error message promptly. Accepted groups retain the exact insertion/product traversal order, which also preserves metric averaging order. Return stacking still owns independent, read-only storage.

## Whole-function inspection

All18 functions in `core/bravais.py` and all three in `core/idealisation.py`, including nested `_add`, were inspected with callers. The two benchmark functions are additional inventory entries.

| Functions | Decision |
| --- | --- |
| `ConventionalCell.parameters`, `.summary` | Retain parameter calculation, report names/types and deviation reporting. |
| `_cartesian_rotations`, `_rotation_axis`, `_axes_of_order` | Retain column-fractional/Cartesian conversion, eigenvector signs and axis deduplication tolerance. |
| `_lattice_points` | Retain Delaunay shell/radius and iteration order; caller-level reuse avoids changing scientific candidate selection. |
| `_shortest_along`, `_shortest_perpendicular` | Accept an optional private prepared cloud; preserve default behavior, selection arithmetic and tie order. |
| `_rotate` | Retain the right-hand rotation calculation. |
| `_centring_from_cosets` | Retain multiplicity, exact-twelfths keys, sorted representatives and recognised centring policy. |
| `_orthogonal_cell`, `_axial_cell`, `_monoclinic_cell` | Prepare one cloud per branch; retain constructed axes, reductions, signs and handedness. |
| `_standard_centring`, `_sort_base_centred`, `_obtuse_monoclinic` | Retain standard C-centring, stable base-axis order and b-unique obtuse monoclinic convention. |
| `conventional_cell`, `bravais_symbol` | Retain public signatures, all family dispatch, coset correction, transforms and classification metadata. |
| `close_matrix_group`, `close_matrix_group._add` | Enforce the existing cap before insertion; preserve accepted closure order and immutable result ownership. |
| `symmetrise_basis` | Retain metric average, Procrustes orientation/handedness, deviation definition and immutable output. |

Benchmark `_over_cap` compares the existing rejection message; `main` loads trusted baseline modules, explicitly binds baseline idealisation into baseline classification, prepares inputs outside tracing and checks exact fields/arrays and valid-group order before measurement.

Callers inspected include planar group closure/metric idealisation, native Bravais/path construction and the public conventional/refined symmetry workflow. Existing tests cover all14 Bravais types under rebasing, centring multiplicity/cosets, primitive conventional axes, rounded cells, invariant positive metrics, orientation/deviation and rounded moiré inputs.

## Evidence

All eight initial resource regressions failed against the baseline. After implementation, the combined resource/Bravais/rebased/rounded/planar idealisation/rounded-moiré selection passed **456 tests in6.49seconds**. The six branch tests assert one cloud and input preservation; two cap tests assert excess construction stops promptly. There are no timing thresholds in CI.

One independent reviewer inspected every scoped function, tests, benchmark, callers and reduction purity. It checked exact D4 closure order with repeated generators and limits `None`,8,9,8.9 and `"8"`; exact-cap duplicates, over-cap/zero/negative and empty-generator errors; shape/dimension error precedence; independent read-only closure storage; prepared-cloud first ties, read-only input and mutation isolation. All narrow probes passed, with no findings. Its first attempt hit a usage cap; the resumed review completed successfully. It did not rerun the broad suite or benchmark.

`python benchmarks/benchmark_conventional_resources.py --baseline bba6576 --repeats 3`, Python3.14.3 / NumPy2.4.3. Three local timing samples and one traced allocation sample per implementation; all classification summaries, five array fields and group order match exactly, and input arrays remain unchanged.

| Case | Baseline ms | Current ms | Baseline peak MiB | Current peak MiB |
| --- | ---: | ---: | ---: | ---: |
| cP | 9.81 | 9.32 | .0580 | .0586 |
| oP | 2.73 | 2.16 | .0548 | .0539 |
| tP | 3.20 | 2.98 | .0562 | .0543 |
| hP | 4.33 | 4.51 | .0899 | .0899 |
| hR | 5.17 | 4.46 | .4074 | .4076 |
| mP | 2.91 | 2.46 | .0534 | .0533 |
| 100 generators, cap12 | 20.91 | .0149 | .0507 | .00577 |

Valid-case gains are modest, and hP was slightly slower locally despite removing repeated work. Peak memory is essentially unchanged for accepted cells. The substantial rejection improvement is for an oversized invalid group, not a universal classification speedup. Tracemalloc measures Python-tracked allocation, not process RSS.

## Limits and checkpoint

Existing `int()` conversion/coercion, scientific tolerances, lattice shell policy, integer/floating-point numerical domains and uncapped infinite generators are not changed. Arbitrary externally supplied private clouds remain the private caller's responsibility. This batch does not claim to resolve the existing global numerical-domain deferrals.

The single full coverage checkpoint passed **3,339 tests, four skipped,81 warnings in769.26seconds**. The skips retain absent external Lean references and the existing surface-period case. All257 frozen Python hashes and the exact tracked file set match after terminal success. Ruff F and canonical ledger checks pass. Coverage is **12,665/14,534 statements** and **3,341/4,618 branches**; it is not a universal numerical-domain guarantee.

One wheel/source-distribution build and Twine metadata check passed. All141 package module bytes and the new test/benchmark match the archives; scientific run/environment paths are excluded. Installed core-wheel, CLI-wheel and source-distribution environments passed conventional-classification, one-cloud reuse, transform consistency, group cap/order/ownership, metric idealisation, help and dependency smokes under NumPy2.5.4. Isolated Python imports were asserted to come from each installed environment. The final source archive is refreshed only for documentation/ledger support bytes, with no repeated unchanged installs.

The ledger contains **1,113 functions:233 reviewed unchanged,102 optimized verified,two investigated deferrals and776 pending**. Thus **337 functions have inspection evidence**. This batch contributes21 original functions and two benchmark functions. The minimum NumPy lane adds the8 new resource tests; its actual runtime and all exact-head CI jobs/coverage artifact remain to be read after the push.

No source function is marked completed from coverage alone. Scientific runs and concurrent root edits remain preserved. Publication is unauthorized.
