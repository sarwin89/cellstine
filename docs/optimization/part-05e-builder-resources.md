# Part05e: exact builder resources and batch persistence

Base:8bfcf17127fee9c4e865f3c0c3140fbb11112e5b. Its16 exact-head CI41777781487 jobs passed, including all nine portable-format lanes. Coverage artifact13298477050 has verified head and ZIP SHA2564b0331d856827b200a539a7a1e54edcb70a00a1f63584043cff213690588caae. The only coverage difference from local is prompts.py501–502's environment-dependent path fallback, with identical statement sets and branch counts.

This batch passed its local full/package checkpoint; scoped commit/push and exact-head CI remain pending. The48 independently reasoned function reviews are promoted after all local gates. No project-wide completion is claimed. The minimum-NumPy helper lane includes the new46-case builder file; all16 exact-head CI jobs remain required.

## Behavior and changes

Column HNF now computes the determinant using Python integer products. The previous NumPy scalar products could overflow before conversion and falsely report a nonsingular matrix as singular. Signed diagonal2**32 and large-entry determinant-one regressions establish exact HNF arithmetic without overflow warnings. This does not establish arbitrary-scale floating-point geometry support.

Coset enumeration retains exactly the previous ij box and row order, with h22 varying fastest. A single C-contiguous int64 output grid is filled by broadcasting its two axes, avoiding meshgrid's extra full coordinate buffers. An exact Python-integer byte-count check rejects arrays exceeding np.intp addressability before allocating axes or output. This is an addressability guard, not an available-RAM guarantee or arbitrary atom-count cap. Per-atom affine conversion is hoisted for nonempty layers while preserving arithmetic, copied positions, z, flags and empty-input behavior.

Bilayer JSON/NPZ batch builds parse and validate one document snapshot and build one index lookup per nonempty call. Each spawned worker receives only its selected candidate, paths and options; no full document is shipped per task and no persistent global cache is introduced. The public single-candidate API still reads fresh results. N-layer batches likewise reuse one validated document/index map; their separate experimental schema, orientation and scientific policy remain unchanged. Supermoire.maken retains its initial layer-count read plus one batch read.

The old parallel fallback caught OSError around task execution and retried the entire batch, including outputs already written before a worker's file-write error. Fallback is now limited to executor construction, before task dispatch. Task, context/dispatch and later write errors propagate; a spawn error after construction is not automatically retried because some output work may have started. Already completed outputs and an empty reserved failed destination retain the direct writer's existing failure semantics. Default filenames, collision reservation, explicit single-output paths, requested index order/duplicates and late invalid-index behavior are preserved.

Layer selection, partitioning, coordinate shifts, zfix thresholds, handedness correction, contact reporting, species grouping and transform wrappers are inspected and retained where sound. New boundary assertions preserve strict cutoff, first-largest-gap tie, minimum-gap equality, signed relaxation thresholds and Cartesian geometry under row-basis swaps.

## Evidence and scope

43 original callable inspections were prepared with individual source-hashed reasoning before edits. The complete batch includes the two private parsed-context helpers and three benchmark functions. Callers include Moire.make, Supermoire.maken, result plotting, N-layer replication and layer/defect/surface consumers. Existing exact-coset tests exercise signed matrices and lattice transversals; built-geometry and contact suites protect numerical conventions.

The new46-case regression file passed in11.44seconds, including real two-worker JSON/NPZ/compressed-NPZ builds with byte-identical POSCARs and metadata, duplicate output preservation, guarded allocation, late invalid index, constructor fallback and non-retried task/dispatch failures. Before implementation,13 cases failed for the expected overflow, allocation, repeated parsing and unsafe retry reasons;7 existing-behavior assertions passed. Another26 boundary cases passed against unchanged helpers before implementation. The broader moire/layer/contact/output/preview selection passed691 cases with16 spglib deprecation warnings in86.50seconds.

Actual old/new production measurements use benchmark_builder_resources.py and frozen Git modules from the base commit. Outputs and metadata compare exactly before timing. Final three-sample Windows/Python3.14.3/NumPy2.4.3 observations:

| Case | Old ms | New ms | Old peak MiB | New peak MiB |
| --- | ---: | ---: | ---: | ---: |
|1,961 exact cosets|0.044|0.014|0.062|0.031|
|1,500,000 exact cosets|14.24|7.84|45.80|22.90|
|8 bilayer builds,5,000 JSON rows|1720.21|450.78|23.27|23.25|
|8 bilayer builds,5,000 NPZ rows|2255.65|300.23|15.19|15.11|
|8 N-layer builds,5,000 JSON rows|1131.34|177.49|23.65|23.63|

All output bytes and metadata match. Large documents cycle actual small-cell records with unique indexes; these measurements establish parsing/resource improvements for that workload, not large-geometry or parallel speedups. Peaks are Python-tracked allocations, not RSS. Timings include output readback and owned-directory cleanup, use warm filesystem caches and no fsync, and impose no CI threshold.

Small cases are timing-sensitive: three-sample1-row/eight-output JSON bilayer was88.97→88.67ms; NPZ206.77→119.61ms; N-layer114.01→197.34ms. The apparent N-layer slowdown prompted a targeted12-pair run with alternating old/new order; medians were73.44→65.81ms, minima63.14→60.12ms. Thus no consistent small-case regression or universal small-case gain is established; raw samples are retained rather than hiding variance. Artifacts: shared-workspace .tmp/part05e-builder-benchmark.json and part05e-small-nlayer-timing.json, with the first concurrent benchmark retained separately. Final benchmark and48 independent review source hashes match the frozen bodies.

The frozen full suite passed **3,556 tests,4 skips,81 deprecation warnings in938.97seconds**. Coverage records13,810/15,510 statements and3,797/5,010 branches, compared with13,767/15,492 statements and3,777/5,006 branches at the preceding local checkpoint. The existing observational faulthandler report at99% did not stop the suite; the original full run completed successfully and was not restarted.

All277 tracked Python files match the frozen set and byte hashes. Wheel/sdist builds and Twine checks pass;143 package modules match both archives. Core wheel, CLI wheel and sdist installed smokes pass under NumPy2.5.4, including duplicate NPZ and N-layer builds, HNF/guard behavior, command help and dependency checks. The final sdist is refreshed for closure documents/ledger without rerunning unchanged runtime checks.

The sdist install first exposed a verification setup error: disabling build isolation in the lean environment omitted setuptools. The isolated retry then spent CPU in tempfile.mkdtemp against pip's default sandbox cache outside the writable workspace. A read-only profiler identified the exact build process/path; only that verified owned backend was stopped. Offline isolated installation succeeded after setting workspace-local PIP_CACHE_DIR as well as TEMP/TMP. No production source, scientific data or full-suite process was changed for either setup correction. The profiler is an ignored diagnostic tool, not a package dependency.

The ledger now contains1,211 functions:528 reviewed unchanged,261 optimized verified,2 investigated deferrals and420 pending reviews (791 with inspection evidence). The48 batch reviews include re-reviews;44 net new entries have inspection evidence. Global review remains incomplete. Scoped commit/push and exact-head16-job CI/artifact readback remain pending. The initial benchmark overlapped focused tests, so small-case timing variation prompted one serialized run; do not treat the initial observations as stable speedup claims.

## Retained limits

Scientific agreement thresholds, zfix boundaries, c-vector direction/tie selection, species and within-species ordering, repeated-layer behavior and experimental N-layer shortlist policy are unchanged. Extreme float inverse/geometry scales and tilted-c padding remain existing numerical-domain concerns. The thread initializer respects caller environment settings, but runs after module imports and cannot guarantee universal native-library thread control. Batch snapshots are per-call and released afterward; source POSCARs remain read by each candidate's geometry stage. No release, package upload or new runtime dependency is performed.
