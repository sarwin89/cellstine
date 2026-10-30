# Part 02a — periodic geometry and query storage

Baseline: `9f59701`, confirmed successful for Python 3.10–3.14 and package builds in [run 41050772517](https://github.com/sarwin89/cellstine/actions/runs/41050772517). This part inspects all 36 functions/methods defined in `core/geometry.py` and five functions in the new comparative benchmark. The overall goal remains incomplete.

## Reviewed function map

| Functions | Callers, invariants and evidence | Decision |
| --- | --- | --- |
| `shortest_lattice_vector_length`, `_shortest_vector_length`, `shortest_plane_vector_length` | Contact self-images and defect/geometry bounds; reduced-basis shortest-vector enumeration and planar first-vector bound. Reduction/geometry/contacts tests compare shortest translations. | Retain normal-path algorithms and enumeration order. |
| `image_shift_reach`, `lattice_shifts`, `plane_shift_reach`, `plane_shifts`, `atom_images` | Neighbor images, surfaces and void search; reciprocal spacing bounds and shift-major/site-major ordering. Wide-box tests verify reach completeness. | Retain bounds, margins and ordering. |
| `plane_minimum_image`, `plane_minimum_image_distances` | Surface-site distances and geometry reports; exact reduced-plane image enumeration, first winning image and half-cell conventions. Plane tests and independent old/new comparisons cover 2D/3D embeddings, skew and boundary ties. | Block image candidate tensors before allocation; keep all comparisons and tie ordering. |
| `_reduced_cell` | Cached reduction data shared by exact minimum-image and shortest-vector routines. Normal-path transform/cache behavior inspected. | Investigated deferral: if reduction fails, the original unreduced basis and a small-box shortest-vector estimate are used. A general guarantee for that exceptional regime requires a dedicated contract/regression investigation; no expanded guarantee is claimed here. |
| `minimum_image_displacements`, `minimum_image_distances`, `minimum_image_fractional`, `bounded_minimum_image_squared` | Contacts, symmetry site matching, molecular paths, and duplicate merging. Existing 3D candidate blocking, half-shortest settlement, radius exactness, and fractional return transforms remain intact. Tests compare exhaustive shifts, bounded screens, skew cells and reversed inputs. | Retain mathematical screening, reduction, ordering and tolerances. |
| `periodic_midpoints`, `pairwise_minimum_image_distances` | Path/site construction and workflows needing a complete matrix. Existing shape validation and row-blocked internal work retained. The full returned matrix is required by the public API. | Retain result shapes and row conventions; no misleading claim of constant memory for required outputs. |
| `_bucket_table`, `_bucket_slices`, new `_bucket_candidates` | Stable sorted keys/offsets/entry order used by Cartesian and periodic queries. Missing buckets map to empty slices; padding is explicitly masked. | Retain table/slice ordering; share padding materialization so reducing callers can request only a bounded query subset. |
| `PeriodicSiteIndex.__init__`, `_encode`, `match`, `find` | Native symmetry, defects and coincident-site representatives. Wrapping, label requirements, capped periodic bins, duplicate `(bucket,site)` removal, distance-vs-lowest-index selection. | Retain construction/encoding/find behavior; block `match` padding, distances and scores while preserving labels and ties. |
| `CartesianGrid.__init__`, `_cells`, `_encode`, `candidates` | Neighbor images and repeated nearest queries. Cartesian pitch cells, 27-bucket coverage and stable entry order. Exact distance checks reject extra border candidates for distant queries. | Retain construction/encoding. Share candidate padding helper; clarify border behavior without changing outputs. Public candidate tables retain their required full shape. |
| `_occupancy_pitch`, `NearestPointQuery.__init__`, `_grid`, `distances`, `nearest_point_distances` | Covering-radius probes, planar voids and nearest-distance queries. Cached grids expand until a candidate is certified within pitch, with an exact fallback. | Use active-axis density for axis-aligned flat/line clouds; allocate candidate tables in blocks before displacement work; bound fallback work and safely reuse owned gather buffers. Constructor/default floors and certified distance rules remain intact. |
| `neighbour_images`, `periodic_neighbour_pairs`, `shortest_interatomic_distance` | Void geometry, validation and contacts. Shift-major ownership, `first < second`, unique sorted pair codes, and ignored coincident sites. | Retain public output ordering and required tables; cap the initial shortest-pair radius by the already known self-image ceiling to avoid vacuum-dependent image growth that cannot improve the returned result. |
| Benchmark `_baseline`, `_measure`, `_nearest`, `_site_matches`, `main` | Checkout-only deterministic comparisons against actual Git geometry source, with a disclosed shared current reduction dependency. Source hashes and baseline commit accompany results. | Inspected by an independent reviewer; all nine exact-output assertions pass. Git access is read-only and output is JSON on stdout. |

## Reproduced allocation regressions

Before the fixes, resource tests failed at approximately 17/49 MiB for nearest query/fallback batches with a small block budget, 36 MiB for 120,000 plane displacements, 10/8 MiB for axis-aligned line/plane cached grids, 19 MiB for two sites in a vacuum-heavy cell, and 113 MiB for dense labeled site matching. The tests assert analytic distances, known first labeled matches, and exact Cartesian references as well as memory budgets. All eleven new regressions pass after the changes.

Nearest queries now look up bucket ranges with linear storage and materialize padded entries only for the current block. An exceptionally wide bucket still needs one query's linear storage. Site matches use the same helper with a fixed 65,536-pair working target. Required input/output arrays and cached grids remain linear or output-sized; process memory is not claimed to be constant.

## Comparative benchmark

Reproduce from a Git checkout:

```powershell
python benchmarks/benchmark_geometry_memory.py --baseline 9f59701
```

The driver records baseline SHA, current normalized geometry/reduction source hashes, Python/NumPy versions, three-call median timings, and separately traced peak allocations. Each nearest/site call constructs a fresh index; inputs are allocated before tracing, and global geometry reduction caches are warmed by the equality/timing checks. The metric is traced allocation, not process RSS. Fixed baseline-first order and machine load affect timings. No timing thresholds run in CI.

Python 3.14.3 / NumPy 2.4.3 local measurement; all nine cases returned exactly equal old/new outputs:

| Case | Before ms | After ms | Before MiB | After MiB |
| --- | ---: | ---: | ---: | ---: |
| Dense nearest, block 1,024 | 97.033 | 115.300 | 34.458 | 1.265 |
| Dense nearest, block 16,384 | 97.036 | 60.437 | 34.458 | 1.265 |
| Dense nearest, default block | 85.725 | 78.401 | 98.293 | 41.265 |
| Axis-aligned line, automatic pitch | 52.656 | 25.850 | 10.589 | 4.166 |
| Axis-aligned plane, automatic pitch | 32.774 | 9.424 | 8.171 | 2.898 |
| Dense site matching | 90.813 | 94.432 | 113.265 | 6.824 |
| Plane image displacements | 32.004 | 31.709 | 35.826 | 10.178 |
| Vacuum-cell shortest pair | 33.131 | 0.735 | 19.415 | 0.382 |
| Random bulk nearest | 12.552 | 11.272 | 6.993 | 4.008 |

The small-block and dense matching cases expose explicit time/memory tradeoffs; this is not a universal speedup claim.

## Independent review and validation

The independent reviewer inspected every function listed above, the geometry/plane/query tests and callers in covering, planar voids, voids, native symmetry, defects and surface sites. **356 exact-output comparisons** passed across cubic, hexagonal, sheared and slab cells; empty/singleton/labeled/tie cases; line/plane/dense/sparse clouds; distant and reversed/reused queries; and blocks 1/127/4096. Plane batches crossed the new block boundary and included half-cell ties and strongly sheared 2D/3D bases. Nearest results also matched an independent Cartesian reference within `1e-14`.

The frozen full suite on Python 3.14.3 / NumPy 2.4.3 passed **3,167 tests**, with **4 skips** and **81 spglib deprecation warnings**, in **261.57 seconds**. Three skips concern the absent external Lean reference, one the whole-period surface case. Source/test SHA256 values remained unchanged during the run. An earlier full run also passed 3,167 tests, but its coverage line map was discarded because comment lines changed while tracing; that stale percentage is not used as evidence.

| Scope | Covered statements | Statement coverage | Covered branches | Branch coverage |
| --- | ---: | ---: | ---: | ---: |
| Full package | 12,067 / 13,971 | 86.37% | 3,065 / 4,346 | 70.52% |
| `core/geometry.py` | 382 / 407 | 93.86% | 78 / 96 | 81.25% |

Coverage includes normal paths and the newly exercised distant-query fallback; it does not establish guarantees for untested numerical domains or unreviewed modules. Wheel/sdist builds and Twine checks pass. The wheel matches every package Python source byte-for-byte; the source archive includes the comparative benchmark and excludes generated run/output/temp data. Core-only, CLI-extra and source-distribution installed smokes and dependency checks pass with NumPy 2.5.4. Core-only smoke imports all eight primary workflow classes and exercises blocked nearest queries, labeled periodic lookup and the vacuum-cell shortest pair. The pushed commit must pass all five supported-version CI jobs and package checks before Part 02b begins.

## Remaining limits

The pitch improvement counts active Cartesian axes; it is not an intrinsic-rank calculation, so rotated planes/lines with three nonzero extents keep the prior 3D density heuristic. Required public candidate/image/matrix outputs may still be large. Negative/nonfinite-coordinate policy, extreme integer encodings, and exceptional reduction fallback behavior were not broadened by this patch. The `_reduced_cell` failure-regime guarantee is recorded as an investigated deferral for Part 02b; no formal Lean proof was executed in the isolated checkout.
