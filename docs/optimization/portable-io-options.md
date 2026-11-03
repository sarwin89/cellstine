# Portable result I/O investigation

The user authorized investigating and switching to faster portable I/O on 2026-11-03. Current production remains JSON v2 at dev837be429. This investigation does not change any reader, writer, schema, dependency or default.

## Measurements and decision

Prototype run: Windows, CPython3.14.3, NumPy2.4.3; three timed samples plus a separate Python-tracked allocation sample. Every complete read/write operation uses the same production garbage-collection policy; the initial exploratory run is retained separately because it did not yet equalize that policy. File operations include opening/closing and the filesystem cache; no fsync, cold-cache or RSS claim. Every format reconstructs the current document and runs the same production validation. Row-to-column conversion is included in writing; array-to-row reconstruction is included in reading. All selected document fields, numerical dtypes and numerical bytes match exactly, including a separate foreign-endian NPY signed-zero/subnormal probe.

| Workload | Format | File MB | Write ms | Read + reconstruct + validate ms | Write peak MiB | Read peak MiB |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| actual365 | current indented JSON | 0.644 | 11.51 | 13.56 | 1.91 | 1.92 |
| actual365 | compact JSON | 0.356 | 9.39 | 13.08 | 0.81 | 1.64 |
| actual365 | gzip compact JSON level1 | 0.120 | 10.20 | 14.79 | 1.64 | 1.81 |
| actual365 | columnar JSON | 0.218 | 25.27 | 11.82 | 0.33 | 1.30 |
| actual365 | NPZ | 0.138 | 4.66 | 10.29 | 0.35 | 1.31 |
| actual365 | compressed NPZ | 0.061 | 8.12 | 10.97 | 0.72 | 1.31 |
| expanded20000 | current indented JSON | 35.194 | 607.11 | 716.61 | 10.10 | 98.59 |
| expanded20000 | compact JSON | 19.433 | 433.62 | 703.34 | 3.92 | 83.56 |
| expanded20000 | gzip compact JSON level1 | 6.477 | 499.71 | 732.33 | 6.73 | 83.73 |
| expanded20000 | columnar JSON | 11.834 | 1299.57 | 700.87 | 4.46 | 69.03 |
| expanded20000 | NPZ | 7.109 | 202.92 | 468.41 | 12.73 | 63.31 |
| expanded20000 | compressed NPZ | 0.156 | 219.53 | 472.11 | 12.73 | 63.31 |

The365 rows come from the actual hexagonal length35 search, strain budgets.01/.01, twist5:25. The20000-row workload cycles those candidates with unique indexes; it is a size-scaling probe, not a20000-distinct-candidate search. Its repetition makes the compression ratio especially favorable. Empty and one-candidate boundaries also roundtrip in all six formats. Raw NPZ array loading takes8.39ms for20000 rows, but excludes validation/reconstruction and is not comparable to the complete716.61ms JSON read.

Decision: implement an **optional versioned NPZ result format in a separate production batch after CLI integration**, retain current JSON reading/writing and defaults, and decide compression/defaults using additional representative data. Compact JSON is a useful lower-cost export option but does not remove Python object materialization. Columnar JSON needs a better encoder to improve this prototype's write time. These host measurements do not establish equal performance on other systems.

## Compatibility boundaries and implementation plan

NumPy's [NPY/NPZ specification](https://numpy.org/doc/stable/reference/generated/numpy.lib.format.html) stores shape, dtype and byte order across machine architectures; NPZ is a ZIP of NPY members. NumPy is already the only required runtime dependency. Use numeric/bool arrays with explicit sized dtypes and a UTF-8 JSON metadata member represented as uint8; never store object arrays. Read with allow_pickle=False and explicitly reject object dtypes. Preserve candidate indexes/order, matrix orientation, physical units, full float64 precision, exact integer values, search/metadata, and all supported optional fields. Do not silently narrow arbitrary JSON integers to int64.

Minimum NumPy1.26 [savez](https://numpy.org/doc/1.26/reference/generated/numpy.savez.html) accepts arbitrary keyword array names; do not assume newer allow_pickle writer keyword support. Write individual members through numpy.save(..., allow_pickle=False) in a standard ZIP, or enforce nonobject arrays before the minimum-version savez call. Archive members require a versioned schema, exact required names/shapes/dtypes/count consistency, duplicate-name rejection, finite/physics validation and appropriate allocation limits. Malformed/incompatible inputs must fail clearly. Keep public dictionary results and JSON consumers working; extension dispatch, manifest artifact routing, previews, plotting and downstream build readers must all use the same validated loading path. Preserve output reservations/overwrite policy and do not create extra sidecar files implicitly.

Add meaningful boundary/corruption/foreign-endian/optional-field roundtrip regressions, compare actual candidate ordering and scientific readbacks, and verify portable-format tests on Windows/Linux/macOS, supported Python versions and NumPy1.26/latest where compatible. Existing CI currently runs Windows: portability documentation alone is not an executed cross-platform gate. Perform one full/package checkpoint for the eventual format batch, push dev and inspect exact-head CI before promoting ledger entries. No release/package publication is authorized.

An accelerated JSON library is another optional avenue, but is not an immediate replacement: [orjson](https://github.com/ijl/orjson) has native wheels for common desktop platforms, different integer/nonfinite/Unicode behavior and no PyPy support. It would require a scoped compatibility policy, dependency fallback and workload-specific measurements. No native JSON dependency is added by this investigation.

## Evidence

Owned ignored artifacts in the shared workspace: `.tmp/portable_io_probe.py`, `.tmp/portable-io-probe.json`, `.tmp/portable-io-probe-final.log`. The script runs from the isolated dev checkout and writes only its own temporary files, removed through TemporaryDirectory. All24 format/workload cases completed successfully; no background process remains. These artifacts are exploratory evidence rather than shipped production code or completed function reviews. The publication and full-repository review goals remain open.

Prototype SHA256: `b907f1249dadbf9323d82c896094164138365fb9ee493d6cf68de06f505b36bc`. Final machine-readable evidence records the same hash and equal-GC policy.

## Consumer map inspected during the CLI checkpoint

The next format batch must cover these current paths together; replacing only
the JSON encoder would leave manifest-based commands and guidance incomplete.

| Path | Integration requirement |
| --- | --- |
| `moire/search/results.py` | Preserve JSON v2 document API; dispatch validated reading/writing by explicitly supported format. Archive schema/version is distinct from the existing JSON schema version. |
| `moire/search/find.py`, `moire/moire.py` | Select output extension/format explicitly and retain output reservations, stored parameters, candidate count, manifest and result artifacts. Preserve the public API default and existing JSON calls. |
| `core/base.py` | `resolve_results_file` currently uses caller-supplied artifact keys; update affected callers rather than silently changing unrelated workflows. |
| `core/previews.py`, guided builders | Recognize the new result artifact/file pattern and reuse actual validated candidate count. If aliases/dual formats are present, avoid duplicate logical previews and ambiguous counts. |
| `moire/builder/generator.py` | `parse_results` consumes `read_results` and returns original POSCAR paths/candidate ordering; retain geometry/matrix conventions and downstream generation readbacks. |
| `moire/moire.py`, `visualize/visualize.py` | Manifest-based build/plot currently resolves `results_json` (or legacy DAT in the plot wrapper); explicitly add new artifact selection while keeping JSON compatibility. |
| `visualize/backends/matplotlib.py`, `visualize/results/plotly.py` | Both call `read_results`; check actual plot input data and selected candidate indexes after archive reconstruction. |
| `interface/workflow/lattice_match.py`, `interface/workflow/interface.py` | Match records currently store source paths under the existing `results_json` field. Preserve existing match JSON readers/schema; decide whether an explicit new source-format field/version is required before emitting NPZ paths there. |
| `moire/supermoire.py`, `moire/search/nlayer.py` | N-layer also uses a `results_json` artifact but has a separate shallowly validated schema. Do not accidentally dispatch it through the bilayer archive reader. Its experimental status remains explicit. |
| CLI parser/dispatch/help and publication plan | Route any format option end to end, document optional dependencies (none new planned), keep old commands working, and retain the no-publication boundary. |

The prototype only handles uniform native numerical candidate fields. The
production format must also preserve accepted JSON extension fields, per-row
ordering where observable, ragged values and integers beyond int64, or reject
such opt-in archive writes explicitly before creating a misleading partial
output. An all-object NumPy array or lossy coercion is not an acceptable
fallback. Loading a few arrays quickly does not establish complete validated
consumer performance. Keep selected-row optimizations separate from any claim
that unselected malformed scientific rows were checked.
