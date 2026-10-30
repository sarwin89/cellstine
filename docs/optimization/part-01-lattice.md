# Part 01 — lattice foundations and review instrumentation

Baseline: `b749da1`, confirmed successful on Python 3.10/3.11/3.12 plus package build in [CI run 41047110782](https://github.com/sarwin89/cellstine/actions/runs/41047110782).

## Inspected behavior and callers

| Function(s) | Invariants/callers and evidence | Decision |
| --- | --- | --- |
| `core.lattice.build_target_lattice` | Row-vector planar cell, third vector along z; adsorbate assembly consumes requested lengths/angle. Existing adsorbate and functional API tests exercise cell construction. | Retain arithmetic and accepted input domain. |
| `core.lattice.vector_angle_deg` | Shared cell-angle/twist helper, re-exported by symmetry/surface/moire reports. Kahan chord/span formula and existing boundary tests retained. | Repair norm underflow/overflow for finite nonzero vectors; ordinary arithmetic unchanged. |
| `core.lattice.inplane_principal_log_strains` | Interface build maps top to bottom with `solve(top,bottom)`, SVD singular values, sorted logarithmic strains. Top-plane rank check and stretch floor are existing conventions. Interface stacking/build tests cover downstream strain behavior. | Retain conventions and solve/SVD path. |
| `core.reduction.as_lattice` | 3x3 row basis, finite entries, determinant independence check; geometry and all reduction entrypoints share it. | Retain validation and determinant threshold. |
| `core.reduction.wrap_to_cell`, `wrap_fractional` | `[0,1)` and rint half-cell representatives; geometry callers use exact minimum-image kernels for skew cells. | Retain endpoints and tie conventions. |
| `core.reduction.reciprocal_norms`, `axis_spacings` | Reciprocal row norms and inverse plane spacings, verified against volume/opposite-face area in `test_reduction.py`. | Retain inverse and norm computation; no measured bottleneck justifying changed rounding. |
| `core.reduction.niggli_reduce`, nested `metric`, nested `apply` | Unimodular integer transforms, canonical metric, handedness and existing tolerance/termination rules; primitive/symmetry cells and periodic geometry depend on them. `test_reduction.py` checks volume, transformed lattice, inequalities and basis invariance. | Retain proof-backed algorithm and tolerances. |
| `core.reduction.delaunay_reduce` | Obtuse superbase, shortest independent triple and oriented unimodular transform; exact periodic search bounds depend on it. Shell-one vs shell-four comparisons and unimodular tests in `test_reduction.py`. | Retain enumeration/order/selection rules. |
| `core.reduction.integer_lattice_basis`, nested sorting lambda | Integer row Hermite basis; geometry tests cover nontrivial spanning sets, determinant index, shuffle invariance, and rank failure. Native symmetry reduction consumes this through rational bases. | Retain int64 arithmetic/domain. Extreme int64 overflow is outside present tested domain; not claiming unbounded-integer support. |
| `core.reduction.rational_lattice_basis` | Common-denominator scaling, rounding, integer basis and fractional restoration. Geometry tests cover centering generators and rejected mismatches. | Inspected but deferred: `allclose` combines its explicit absolute tolerance with NumPy's default relative tolerance. Changing that scientific acceptance rule needs a separate contract analysis. |
| `core.reduction._extended_gcd`, `plane_form_kernel_basis` | Exact Bezout coefficients and saturated integer kernel, including signed/axial forms; exhaustive small forms and recovery of all sampled plane points in `test_plane_sublattice.py`. Surface construction consumes these. | Retain exact construction and sign conventions. |
| `core.reduction.gauss_reduction_multiplier`, `plane_reduce` | Boundary slack, reduced shortest vector, unimodular row transform, canonical obtuse hexagonal boundary. Reduction and boundary-convention tests explicitly pin 60/120-degree behavior and preserved lattice. | Retain slack, shear direction, termination and canonical angle. |
| `core.reduction.plane_reciprocal_norms` | Pseudo-inverse columns in embedded planes; area/edge-height comparisons in reduction tests and plane minimum-image consumers. | Retain pseudoinverse path. |
| `tools.review_inventory` functions/methods | AST traversal includes nested/async/lambda functions, tracks source changes and retains only current review evidence, checks statuses, and restricts scans to tracked non-test Python. Unit fixtures verify entries/lines, retained vs invalidated reviews, missing evidence rejection, and LF/CRLF stability. | New audit infrastructure; independently review before marking its rows complete. |

## Reproduced defect and implementation

Finite nonzero vectors scaled by `1e-200` raised the zero-vector error; `1e200` inputs returned 0 degrees for 30/90/180-degree pairs. Mixed scales around `1e-160` produced inaccurate subnormal squared norms (including approximately 179.999545 degrees for an antiparallel pair). Regression tests demonstrate each case before the fix.

The ordinary norm computation and Kahan formula remain unchanged for representable normal squared norms. Only vectors whose squared norm is subnormal/zero or whose norm overflows are rescaled by their largest absolute finite component before normalization. True zero-vector rejection is retained. Ordinary scalar tolerances and caller output ordering do not change.

## Validation record

Focused lattice/reduction/geometry/plane/adsorbate/interface tests: 215 passed at the initial checkpoint. Final angle/reduction/inventory checks: 52 passed. Independent review also passed all 28 angle tests and all 4 inventory tests at its checkpoint, verified the 1,079-function AST count across 145 non-test modules, and compared 4,000 ordinary vector pairs (dimensions 2/3/10/100, scales `1e-100`–`1e100`) with exactly equal baseline/new angles.

Final full suite with coverage on Python 3.14 / NumPy 2.4.3: **3,156 passed, 4 skipped, 81 spglib deprecation warnings**, 247.61 seconds. Three skips concern absent external Lean reference material, and one is a whole-period surface case. Additional validation fixtures cover rank-deficient top cells, insufficient integer generators and nonpositive rational denominators.

| Scope | Statements | Statement coverage | Branches | Branch coverage |
| --- | ---: | ---: | ---: | ---: |
| Full package | 12,044 / 13,948 | 86.35% | 3,052 / 4,332 | 70.45% |
| `core/lattice.py` | 39 / 39 | 100% | 11 / 12 | 91.67% |
| `core/reduction.py` | 205 / 207 | 99.03% | 83 / 86 | 96.51% |

These are local package measurements, not a claim that unreviewed functions are complete or that subprocess CLI paths are fully measured. CI measures its own Python 3.12 baseline and retains the XML artifact.

The ordinary angle microbenchmark (two fixed 3D vectors, minimum of three 5,000-call runs) measured about 13.78 microseconds before and 16.17 after: a 2.39-microsecond guard cost. This is a correctness repair; no speedup is claimed. Ordinary outputs are unchanged bit-for-bit in the independent comparison.

Wheel/sdist builds, Twine metadata checks, wheel source byte comparisons, and installed core-only/CLI-extra/source-distribution numerical/version/help smokes pass. The source archive now includes maintained docs, benchmark helpers, tracked examples, launcher and review tooling required by its tests; extracted inventory/docs tests pass (6 tests). Generated run/output/temp files remain excluded. Ruff F checks, whitespace checks and ledger drift checks pass.

The authoritative supported-version CI gate passed on commit `9f59701`: [run 41050772517](https://github.com/sarwin89/cellstine/actions/runs/41050772517) succeeded for Python 3.10/3.11/3.12/3.13/3.14 and package builds. The nonexpired `python-3.12-coverage` artifact was read back as artifact `12974319031`, associated with this exact commit. This gate was verified before Part 02 began.

## Limits

An external Lean source tree is absent in the isolated checkout; formal proofs are references rather than a locally executed verification gate. Existing int64 domain limits, determinant/rank thresholds and reduction tolerance conventions were retained. The rational-basis tolerance investigation remains explicitly deferred, with this evidence, rather than silently changing which scientific inputs are accepted.
