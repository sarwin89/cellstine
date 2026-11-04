# Part05d: optional portable result archives

Base:86adf7ae1d6e95ab528917138f3d326da4f39dfd. All seven exact-head CI41679502002 jobs passed; downloaded artifact13252837536 has SHA256b8f9e02d6121971f42870c97541d8bf084926b1f6e36fc9635215477765602d0. Its13,535 covered statements include two path-display fallback statements beyond local13,533; statement sets/covered branches agree. Commit8bfcf17127fee9c4e865f3c0c3140fbb11112e5b is pushed to dev. All16 exact-head CI41777781487 jobs passed, including nine Windows/Linux/macOS portable-result lanes.

## Behavior

Bilayer search accepts `--results-format json|npz|npz-compressed`, with the same `results_format` keyword on `Moire.find` and `run_find`. JSON remains the default, with existing byte layout, schema, artifacts and direct API shortlist defaults unchanged. Guided advanced questions expose the choice; the short guided path is unchanged. New archives use results.npz and the explicit results_npz artifact, and progress messages name the actual file. Format validation precedes creating a run directory.

NPZ archives preserve the complete JSON v2 document contract. Ordinary uniform numerical fields use primitive fixed-width NPY arrays. Metadata and irregular/mixed/ragged/large-integer extension values use JSON in uint8 NPY members; varying per-row key layouts use the JSON candidate-list fallback. Top-level and candidate key ordering, numeric values/types, signed zero and metadata survive reconstruction. Unknown objects cannot execute NumPy conversion hooks. No object array or pickle is read or written. The archive has its own version1 schema, distinct from the enclosed JSON v2 schema; arbitrary unrelated NPZ files are not CELLSTINE results.

Writing preflights conversion and JSON serialization before opening a destination. Both encoders use allow_nan=False, matching the existing JSON writer, including extension fields. Numerical arrays are retained while writing to avoid repeated conversion; this uses memory proportional to encoded numerical data. JSON encoding streams, with a count pass to write its NPY header; large atomic strings can still require their own encoder chunk. Later filesystem failures retain ordinary explicit-file write semantics; no transaction or universal RSS bound is claimed.

Reading checks schema/version, duplicate/excess/missing members, compression methods, declared column layout, primitive dtype, bounded NPY1.0 header, shape, row count and exact declared payload bytes. JSON members are decoded from the post-header stream without a second NumPy-byte-array copy. Numerical headers are checked before candidate allocation, and member EOF/CRC is read. The reconstructed document runs the original scientific validator. NumPy1.x's32-dimensional limit is retained for physical arrays; deeper values use lossless JSON fallback. This also bounds cyclic trusted-list scanning, after which JSON reports circularity before destination open. [NumPy's migration guide](https://numpy.org/doc/stable/numpy_2_0_migration_guide.html#increased-maximum-number-of-dimensions) explains the differing1.x/2.x dimension limits.

Manifest-based preview/counts, build, PNG and Plotly consumers resolve the new artifact. Direct archive builds match JSON builds in species/counts, lattice and both coordinate arrays. File discovery includes results.npz. Interface searches continue producing their existing JSON source records; N-layer retains its separate experimental JSON schema and its existing shortlist/scientific policy. No numerical search algorithm, tolerances, matrix orientation or structure output changed.

## Measurements

`python benchmarks/benchmark_result_formats.py --rows 20000 --repeats 3` calls the actual production writer and reader, including conversion, reconstruction and validation. JSON and NPZ both use their production collection-paused context. The final artifact records exact measured-source hashes. Three-sample Windows/Python3.14.3/NumPy2.4.3 observations use warm filesystem caches and no fsync. Peaks are Python-tracked allocations, not RSS.

| Rows | Format | MB | Write ms | Read+validate ms | Write peak MiB | Read peak MiB |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 365 | JSON | 0.644 | 10.68 | 13.64 | 1.908 | 1.917 |
| 365 | NPZ | 0.140 | 12.29 | 18.01 | 0.280 | 1.280 |
| 365 | compressed NPZ | 0.061 | 17.23 | 16.17 | 0.644 | 1.280 |
| 20000 | JSON | 35.194 | 530.46 | 759.06 | 10.097 | 98.594 |
| 20000 | NPZ | 7.110 | 482.86 | 487.20 | 8.774 | 60.436 |
| 20000 | compressed NPZ | 0.156 | 500.17 | 482.16 | 8.774 | 60.436 |

All documents and key ordering compare exactly, and inputs remain unchanged. The365 rows are actual length35 hexagonal search results;20000 rows cycle them with unique indexes. Repetition makes the compressed ratio unusually favorable. Small results are faster as JSON on this host; large NPZ improves file size, read time and tracked allocation, with a modest write-time gain. Compression trades write time for disk space, and equal read timings can be noise. No universal speedup, equal gains across Python versions, or representative20k-distinct-candidate compression ratio is claimed. Initial prototype timings omitted production preflight and are not production performance claims.

Evidence: shared-workspace `.tmp/part05d-production-benchmark-final.json`, with initial pre-guard observations retained separately. The scalar-column plain-type fast path was reviewed with unknown-object, mixed-number and large-integer regressions. No timing threshold is imposed in CI.

## Review and validation state

Independent whole-function review covers99 source callables:42 archive/results,36 workflow/preview,21 CLI/visualization/benchmark. The review maps include unchanged functions, nested callbacks and lambdas; individual source-hashed notes are prepared before ledger promotion. Findings resolved: nonfinite extension writer policy, actual guided NPZ discovery evidence, cross-version depth handling and cyclic preflight termination. Absurd NPY payload declarations are rejected before array allocation; tampered scientific matrices still fail the established scalar validator.

The affected Moire/CLI/interface/preview selection passed782 cases/29 deprecation warnings in50.54seconds. All38 new archive/consumer cases pass, including actual PNG/Plotly generation, foreign byte order/Fortran layout, malformed archives, JSON bytes/defaults, nonfinite values, destination preservation and hooks/cycles. The new portable CI matrix has nine lanes: Windows/Linux/macOS with Python3.10/latest-compatible NumPy, Python3.12/NumPy1.26.4, andPython3.14/latest-compatible NumPy. Core-only portable lanes may skip the two rendering cases; existing all-dependency full Windows jobs execute them. All nine lanes executed successfully:36passed and2 optional-rendering skips each. Log readback confirms NumPy2.2.6 for Python3.10,1.26.4 for3.12, and2.5.4 for3.14 on each OS. No new runtime dependency was added.

The frozen full suite passed **3,510 tests,4 skips,81 deprecation warnings in686.46seconds**. Coverage records13,767/15,492 statements and3,777/5,006 branches, compared with13,533/15,256 statements and3,666/4,886 branches at the preceding local checkpoint. No percentage alone establishes correctness. The existing dense-void case emitted an observational faulthandler stack and completed successfully; it was not restarted.

All275 tracked Python files match the frozen set and byte hashes. Wheel and source distribution passed build/Twine checks;143 package modules match both archives. Core wheel, CLI-extra wheel and sdist installed environments passed NPZ search/read/manifest preview/build, command help and external pip dependency checks under NumPy2.5.4. Final documentation/ledger inclusion is checked in the refreshed sdist without rerunning unchanged runtime gates.

The99 individually reasoned reviews are promoted only after these gates. The ledger has1,206 functions:492 reviewed unchanged,253 optimized verified,2 investigated deferrals and459 pending (747 with inspection evidence). New functions count separately; re-reviews do not inflate total completed coverage. All16 exact-head CI jobs passed. Downloaded artifact13298477050 has independently verified ZIP SHA2564b0331d856827b200a539a7a1e54edcb70a00a1f63584043cff213690588caae and matching head/module/statement sets. CI covers13,769 statements and3,777 branches; the only local difference is prompts.py501–502, the environment-dependent absolute-path fallback already investigated in Part05c. No covered-branch difference exists. The repository-wide review is incomplete. No release, package upload or new runtime dependency is authorized/performed.
