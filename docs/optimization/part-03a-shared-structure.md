# Part 03a: shared structure helpers

## Scope and review decisions

Baseline: `1db9be4159c413b45adf7f8fa0c5376c7fdb5966`. This part individually inspects all 40 pre-existing functions below, one new quotient helper, and the two functions in `benchmark_structure_helpers.py`. Declarative `core/models.py`, `core/constants.py` and exports were also read; they contribute no function rows. Workflow base classes, dependency management and validation remain pending for Part 03b.

| Module | Individually inspected functions | Decision and invariants |
| --- | --- | --- |
| `core/transforms.py` | `rotation_matrix_about_axis`, `rotation_matrix_z`, `rotation_matrix_x`, `rotation_matrix_y`, `yaw_pitch_roll_matrix`, `right_handed_lattice`, `strained_copy`, `integer_supercell_matrix`, `supercell_cosets`, `supercell_structure`, `repeat_structure`; new `_cosets_by_residues` | Change axis overflow fallback, strain copy API and large-box coset enumeration. Keep rotation order, row lattice convention, handedness, atom/species/image order, selective flags, copy boundaries and existing integer acceptance. |
| `core/directions.py` | `ViewDirection.is_lattice_plane_normal`, `.project`, `.frame`, `.describe`, `.as_dict`; `project_along`, `orthonormal_frame`, `_reciprocal_rows`, `_unit`, `_parse_triple`, `_parse_floats`, `_reduce_indices`, `_plane_family`, `_reads_as_signed_triple`, `resolve_direction` | Change finite overflow normalization and frame candidate fallback. Preserve parsing, Miller/real-space distinction, spacing, frame orientation and tolerance policies. Correct the documentation example to `11x1`, already meaning `(1,-1,1)` under the existing trailing-x convention. |
| `core/vacuum.py` | `surface_normal`, `cell_height`, `normal_heights`, `occupied_span`, `vacuum_gap`, `fit_cell_to_vacuum` | Repair finite-scale surface normal overflow/underflow; preserve normal orientation, projected heights, slab gap and rigid translations. Other five bodies are unchanged. |
| `core/species.py` | `expand_species`, `group_species` | Bound discarded labels for nonnegative requested sizes. Preserve full expansion, negative slicing, count conversion, padding, ordering and species-block grouping. |
| `core/elements.py` | `element_symbol`, `covalent_radius`, `covalent_radii`, `species_masses` | Unchanged: retain label normalization, element table values, unknown-element fallbacks, ordering and mass/radius conventions. |
| `io/models.py` | `StructureRecord.natoms`, `StructureRecord.copy` | Unchanged: integer count sum; owned arrays/lists and shallow metadata copy. |
| `benchmarks/benchmark_structure_helpers.py` | `_repeat_call`, `main` | Review baseline loading, equality assertions, timing/allocation boundaries, positive arguments and reproducibility metadata. |

Callers traced include rendering/projection, observation-direction layer census, interface strain/assembly, moire construction, adsorbate vacuum fitting, POSCAR I/O and defect supercells. The focused integration selection exercises those shared contracts. The unchanged element tables and dataclass defaults were checked directly, rather than inferred from test coverage.

## Changes and compatibility

`expand_species` caps each list multiplication to the remaining requested length before allocation. It still converts every later zipped count, even after enough labels exist; invalid later counts therefore retain their errors. Unset sizes and historical negative slices retain full expansion. This is a bounded-size improvement, not a new header validator.

For a validated integer matrix M and signed determinant d, translation residues are `t @ adj(M) * sign(d) mod abs(d)`. The three unit-translation generators reach the finite quotient. The walk processes closure, rejecting a count different from `abs(d)`. Each residue r gives the canonical integer row `r @ M / abs(d)`; exact Python integer division checks integrality. Lexicographic sorting returns the same representatives and image order as the original bounding-box membership rule, including negative determinants. Boxes of at most 65,536 candidates retain the vectorized path; determinant ±1 returns the unique origin after validation. Storage on the large-box path scales with output cardinality instead of shear-amplified box volume.

This does not replace the existing floating determinant/inverse construction, int64 representability checks or default `allclose` integer acceptance policy. Arbitrarily large integer matrices are not newly guaranteed. The quotient walk deliberately validates closure rather than stopping as soon as a possibly incorrect determinant count is reached.

Ordinary normalization arithmetic remains in place. Overflowed finite rotation/direction vectors use largest-component scaling. Overflowed frame candidates are projected after scaling, preserving the existing relative parallelism criterion and first suitable lattice-row orientation. Surface normals recompute from scaled rows when a small cross product may have lost its direction. An exceptionally anisotropic, subnormal scaled cross product uses exact rational arithmetic on the six supplied binary64 components, then converts bounded component ratios back to floats. This fallback distinguishes true parallelism from erased or partially quantized components; it does not claim recovery of information already absent from the supplied floats. Existing absolute zero-axis thresholds remain unchanged.

`strained_copy` now uses `np.array(..., copy=True)` instead of the NumPy-2-only `np.asarray(..., copy=True)`. It retains independent direct/lattice/Cartesian arrays and a new shallow metadata mapping. Existing sharing of species/counts/selective flags through `dataclasses.replace` remains unchanged.

## Regressions and independent review

The allocation, finite-scale and minimum-API regressions were observed failing before their respective fixes. The new modules contain 40 tests. They cover discarded million-label headers, later count validation, signed/unimodular large shears, 100 independent small-box references, atom blocks/flags/folded geometry, huge axes/frames, signed plane scales, tilted tiny planes, extreme anisotropy, partially subnormal unequal cross components and strain array ownership. The NumPy API proxy is useful locally but is not an actual minimum-version runtime result.

The final focused helper/rotation selection passed **208 tests**. The broader selection, adding I/O, defect cells/layers, visualization directions/projections, contacts and adsorbates, passed **400 tests**, with eight existing spglib deprecation warnings, in 21.47 seconds. A prior fixture setup failure came from an unwritable sandbox temporary directory; the successful run used project-local temporary files.

Independent read-only review inspected and invoked all 40 original functions and the new helper, plus the benchmark and CI lane. Ordinary baseline comparisons passed exactly in 2,000 rotation/unit/frame/normal cases and another 1,000 normal cases. A 14,641-case tilted-row scale sweep had worst error 2.22e-16; 105 anisotropic cases against exact-rational cross references had worst error 1.11e-16. Review found two intermediate underflow errors (reuse of a quantized tiny cross product and a partially subnormal rescaled cross product); both were reproduced as failing tests and corrected before the final review. No material finding remained on the frozen implementation.

## Reproducible measurements

Run `python benchmarks/benchmark_structure_helpers.py` from the checkout. The default baseline is the commit above; each case asserts old/new output equality. Inputs are prepared before allocation tracing. Three timing samples are used; micro-operation samples batch 1,000 calls, while allocation traces measure a single call. Timings are observations, not CI thresholds. Final-source run: Python 3.14.3, NumPy 2.4.3.

| Case | Baseline median ms | Current median ms | Baseline peak MiB | Current peak MiB |
| --- | ---: | ---: | ---: | ---: |
| Million labels, three requested | 5.6101 | 0.0025 | 15.25903 | 0.000290 |
| Full labels | 0.0287 | 0.0178 | 0.11453 | 0.11453 |
| Positive shear | 38.0169 | 0.4077 | 33.38136 | 0.011916 |
| Negative shear | 38.9606 | 0.3057 | 33.38132 | 0.011916 |
| Unimodular shear | 26.7015 | 0.1857 | 22.50903 | 0.006125 |
| Ordinary cosets | 0.3214 | 0.3569 | 0.016787 | 0.016656 |
| Ordinary rotation, 1,000 calls | 11.1184 | 11.1905 | 0.001053 | 0.001053 |
| Ordinary frame, 1,000 calls | 70.4161 | 52.4884 | 0.007652 | 0.007675 |
| Ordinary normal, 1,000 calls | 31.9101 | 34.5059 | 0.006592 | 0.006950 |

Ordinary cosets and normals have small measured overheads; no universal speedup is claimed. Extreme-scale repairs improve correctness and are not reported as speedups against incorrect baseline outputs.

## Checkpoint gates

Source/tests were frozen before the full suite (245 Python files). The full coverage run passed **3,230 tests, four skipped, 81 warnings in 706.57 seconds**. Three skips require the absent external Lean reference; the fourth is an existing face/layer-period case. The two-minute faulthandler dump in the dense-void test was diagnostic; the process subsequently exited successfully. All frozen hashes and final benchmark source digests match.

Package statements covered: **12,377/14,290 (86.61%)**; branches: **3,206/4,502 (71.21%)**. Module statement/branch percentages are directions 91.28/80.68, elements 94.29/75.00, species 96.97/93.75, transforms 90.13/75.00, vacuum 90.67/73.08; `StructureRecord` has 24/24 statements and no branch rows. These measurements expose remaining gaps; they do not establish complete review or correctness.

Wheel and sdist build and Twine checks passed. All 140 packaged Python modules match source bytes; the sdist contains the new tests and benchmark and excludes scientific runs and temporary environments. Offline wheel core-only, wheel CLI and sdist installations each passed structure-helper numerical/ownership/ordering smokes, CLI help and dependency checks under NumPy 2.5.4. Final documentation/ledger archive contents are checked again after this report update. Ruff F checks and whitespace checks passed.

The ledger contains **1,099 functions: 135 reviewed unchanged, 34 optimized verified, one investigated deferral and 929 pending**. This part adds 43 inspection records: 40 original functions, one helper and two benchmark functions. The seven changed original executable bodies and new helper were checked by AST comparison; all other scoped executable bodies remain identical to baseline, with the parser documentation correction explicitly accounted for. The canonical ledger drift check passes.

The workflow adds a separate Python 3.12 / NumPy 1.26.4 helper lane. Actual minimum-runtime compatibility remains pending until that lane succeeds, alongside all five full supported-version jobs and package build on the pushed commit. No package or release publication is authorized.
