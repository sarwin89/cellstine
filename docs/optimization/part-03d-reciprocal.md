# Part 03d: exact reciprocal meshes with bounded temporary storage

Baseline: `dc615276878cb1fe05e4ecaf5886188d73ad8d72` (Part04 I/O). All seven jobs passed in CI run `41423449104`; its exact-head coverage artifact is `13130624746`, digest `82252595f6335d1595d79ee44a49baec265b181ef48d7ddd0e8c99f7ed80093f`. Concurrent Claude root search/CLI patches remain separate and unmerged.

## Demonstrated faults and resulting behavior

Packed int64 keys were not collision-free. At denominator2097152, `[-1048576,0,0]` and `[0,0,0]` both encoded to8796095119360. Identity-only orbit assignment consequently merged unrelated points, even with time reversal disabled. Both rows occur in a valid `(1048576,1,1)` mesh.

Ordinary denominators retain the original numeric key and ordering, using one output column with in-place arithmetic. For `D <= 2**20`, shifted coordinates lie in `[0,2D)`, so the key is bounded by `(2D)**3-1 <= 2**63-1`. Larger denominators use an owned structured array of signed integer triples, whose numeric lexicographic sort/search is exact. There is no per-point Python integer key and no probabilistic hash collision policy. Orbit representatives remain the lowest original mesh indices.

Two other integer paths were inexact. An image under a large integral finite-order rotation overflowed before the modulus, returning149041 instead of540885 for one demonstrated component. A conjugated order-four matrix with `n=2**28`, rows `[[n,-n*n-1,0],[1,-n,0],[0,0,1]]`, was rejected after float conversion lost its determinant. Integral input now stays integral. The 3x3 determinant/adjugate is computed with Python integers; ordinary matrix products use int64 under a conservative bound, while exceptional products are checked exactly. Modular image products use bounded8192-row object workspaces only when the int64 bound is exceeded. Results return to the existing canonical int64 numerator representation.

An unrepresentable spacing request, such as `mesh_divisions_for_spacing(np.eye(3),1e-20)`, formerly cast to the minimum signed integer and returned `(1,1,1)`, silently undersampling. It now raises a clear range error. Finite-shift/denominator/addressable-array checks reject unsupported requests before conversion/allocation. Half-interval wrapping compares to an integer midpoint rather than overflowing `2*wrapped`. No practical RAM quota is invented; representable but unaffordable requests can still raise `MemoryError`.

Numerator construction preallocates its final tensor and broadcasts one axis at a time. It preserves the original C-order mesh iteration, half shifts and exact division arithmetic without retaining three full meshgrid arrays alongside the output.

## Whole-function inspection

All26 original functions were read with their callers and existing invariants. Two source helpers and one benchmark entry are additional function rows. Tests are excluded from the denominator.

| Functions | Decision and contract |
| --- | --- |
| `_as_lattice`, `cell_volume`, `reciprocal_lattice`, `brillouin_zone_volume` | Retain finite3x3 validation, row-vector duality, solve convention and volume relations. Pathological floating-point lattice range remains the existing end-to-end investigation. |
| `mesh_divisions_for_spacing` | Add representability checks before casting; preserve the existing spacing formula and tolerance. |
| `_as_divisions`, `mesh_spacings`, `kpoint_density`, `mesh_shift` | Retain integer conversion, positive divisions, reciprocal step/density and standard Gamma/Monkhorst policy. |
| `_shift_numerators` | Add finite/range errors; preserve existing float acceptance tolerance and half-step modulo convention. |
| `_mesh_numerators`, `_wrap_numerators`, `_encode` | Reduce retained arrays and remove integer-overflow errors while preserving ordinary exact mesh/key order. |
| `_integer_inverse`, `_mesh_images` | Compute integer inverses and modular images exactly within the stated representation, with bounded exceptional workspaces. |
| `KpointMesh.point_count`, `.normalised_weights`, `.symmetry_complete`, `.cartesian_points`, `.summary` | Retain counts, orbit weights, operation reporting, JSON field types and Cartesian reciprocal conversion. No new model copying or mode is needed. |
| `mesh_points` | Retain numerator-to-float conversion and row order; delegates to corrected construction. |
| `_close_group`, `_usable_operations` | Preserve finite-group closure order and mesh-preserving subgroup selection; validate exact integers and prevent silent product wrap. |
| `_orbit_labels`, `build_mesh` | Retain complete symmetry/time-reversal images, lowest-index representatives and exact weight assembly; consume corrected keys/products. |
| `supercell_divisions` | Retain diagonal-only supported policy, absolute repeats and ceiling division; general matrices still direct callers to spacing-based construction. |

New `_as_rotation` preserves signed integral input and the existing float-rounding tolerance; `_integer_product` selects checked arithmetic. Benchmark `main` loads a trusted Git baseline using the already reviewed helper, prepares inputs outside tracing and distinguishes exact ordinary comparisons from the large-grid correctness extension.

Callers include native KPOINTS automatic/explicit layouts and their sampling methods, `Symmetry.kpoints` with supplied crystallographic rotations, reciprocal consumers in Brillouin/path code, and the public CLI. Existing tests independently check reciprocal duality, spacing, cubic counts, brute-force orbit partitions, shifted/anisotropic subgroup loss, generator closure, representative uniqueness, weight sums and diagonal-supercell folding.

## Focused verification and independent review

All nine initial correctness/resource regressions failed against the baseline. Fifteen new cases now cover those findings, near-int64 wrapping, unsupported shifts and pre-allocation range errors. A first combined selection passed56 tests; after completing boundary cases, the affected reciprocal/KPOINTS/CLI/symmetry selection passed **85 tests,11 existing warnings in7.60seconds**. No broad selection was repeated for documentation-only work.

One independent reviewer read every scoped function, new tests, benchmark and relevant callers, finding no issues. Its narrow read-only probes checked key uniqueness, ownership and numeric ordering at `2**20`, `2**20+2`, `3*2**60` and `2**63-2`; an independent Python-integer dictionary oracle checked large-rotation orbit labels with and without time reversal. It did not rerun the suite or benchmark. Full/package gates below remain executor evidence.

## Resource evidence

`python benchmarks/benchmark_reciprocal_resources.py --baseline dc61527 --repeats 3`, Python3.14.3 / NumPy2.4.3. Inputs are prepared outside tracing; three timing samples and one traced allocation sample per implementation. Ordinary arrays, complete mesh summaries, representatives and weights match exactly.

| Case | Baseline median ms | Current median ms | Baseline peak MiB | Current peak MiB |
| --- | ---: | ---: | ---: | ---: |
| `128x128x16` numerator grid | 4.24 | 2.52 | 12.00 | 6.01 |
| 200k ordinary keys | 4.64 | 1.87 | 9.16 | 1.53 |
| `32x32x16` fourfold-reduced mesh | 23.93 | 22.97 | 3.30 | 3.30 |

The actual1048576-point identity-only public mesh retains every point, in original coordinate order, with every weight1. Current median3.77seconds/peak187.00MiB. Its old keys collide, so the incorrect baseline is not presented as an equivalent timing comparator. The final benchmark additionally asserts those coordinates against a direct independent uniform-grid formula. Measurements describe local Python-tracked allocations, not RSS, portable speed guarantees or CI thresholds.

## Deliberate limits and remaining gates

Existing fractional division/minimum integer casts and default relative tolerances in float rotation/shift acceptance are retained; changing those invalid-input/tolerance policies is outside this overflow/allocation batch. Floating determinant/norm extreme-range behavior remains separate. Rotations, inverses and generated group elements must fit signed int64; unsupported values raise rather than wrap. Mesh denominators and array byte counts must fit the checked representation/addressable range.

One wheel/source-distribution build and Twine metadata check passed. All141 package module bytes in both archives match the frozen checkout; the new test and benchmark are present in the source distribution, with no scientific run/environment paths. All255 frozen Python hashes and the tracked file set match after the build.

Installed core-wheel, CLI-wheel and source-distribution environments passed reciprocal correctness, help and dependency smokes under Python3.14.3 / NumPy2.5.4. Each asserted imports came from its own installed environment in isolated Python mode. The core environment additionally executed the actual1048576-point public mesh and checked every coordinate independently and every weight1; all three covered large-key orbits, exact inverse/group arithmetic, overflowed modular images, spacing rejection and ordinary shifted sampling. The final benchmark's independent large-grid assertions also passed in a separate one-sample invocation after being added; the original three-sample ordinary measurements above remain unchanged-source measurements.

The one full coverage checkpoint passed **3,331 tests, four skipped,81 warnings in715.31seconds**. The skips retain the absent external Lean references and existing surface-period case. All255 frozen Python hashes and the exact tracked file set match after terminal success. Ruff F checks and canonical ledger drift checks pass.

Coverage is **12,661/14,531 statements** and **3,340/4,618 branches**. This does not prove universal numerical range or every user-supplied group/input policy. The minimum NumPy lane adds the15 new reciprocal tests; all exact-head supported-version/package/minimum-dependency jobs and coverage artifact must still be read after pushing this checkpoint.

The ledger contains **1,111 functions:217 reviewed unchanged,95 optimized verified,two investigated deferrals and797 pending**. Thus **314 functions have inspection evidence**. This part adds26 original functions, two helpers and one benchmark function. Initial source reading of conventional-cell/idealisation modules identified repeated point-cloud construction and late group-cap enforcement for the next batch; those21 functions remain pending, with no source change or completion claim here.

Final documentation/ledger support bytes are compared after a documentation-only source-archive refresh; repeating the unchanged installs is unnecessary. Whole-project review remains incomplete, and publication is unauthorized.
