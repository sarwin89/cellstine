# Part 04: native I/O integrity and resource use

Baseline: `c7dec6cc0d83d0183e9f9b438d0ba660b10954d6`. All seven supported-version, minimum-NumPy and package jobs passed in CI run `41405522888`. Its Python 3.12 coverage artifact is `13123961765`, digest `efc16e6af27ccc42a763d41ad5437887a7ac2e3efb8af3b26a0e74ffa7580c31`. This batch is implemented in the isolated dev checkout; concurrent root moiré/search/CLI edits are excluded.

## Problems and resulting behavior

Molecular fallback previously overwrote and deleted a sibling `.tmp.xyz`. It now closes a uniquely owned temporary file before the external writer runs and removes only that file in `finally`. The requested vacuum and original input metadata survive the fallback. The external-dependency regression uses a small fake pymatgen adapter to exercise ownership and error paths without requiring that optional package; it does not claim coverage of every real pymatgen format.

XYZ parsing reads only the first declared frame, rejects missing/nonfinite coordinates and invalid padding, and preserves stable species grouping and the existing box arithmetic for ordinary valid inputs. XYZ writing validates record shape, coordinates and the single-line comment before opening the destination, then streams labels rather than allocating a label per atom. Valid ten-decimal output is unchanged. Explicit destination overwrite semantics remain; this is not a transaction for later disk failures.

POSCAR parsing reads the required header and atom rows rather than retaining trailing velocities or other data. It rejects malformed/unsupported scaling lines, incomplete positions and nonfinite raw data. The [VASP POSCAR specification](https://vasp.at/wiki/POSCAR) permits one scale or three positive component scales, a negative single target-volume scale, and C/c/K/k Cartesian modes; other mode initials select Direct. Reading `Fractional` now follows that Direct convention. Existing label/count tolerance in the raw reader remains because renderer consumers handle missing/inconsistent labels themselves.

Repeating a mixed-species structure along c previously emitted repeat-major coordinates with species-major counts, silently assigning the wrong species to positions. It now fills species blocks, carrying each atom's flags with it. A homogeneous structure keeps its old numerical order exactly. Repeat-one preserves copy semantics; inputs are not mutated.

KPOINTS parsing streams explicit data, accepts whole decimal/exponent count spellings, and rejects fractional/nonfinite counts, missing coordinate/weight fields, nonpositive grid divisions, incomplete shifts and nonfinite sampling data. Blank/comment rows, paired line endpoints, labels, explicit trailing data and existing explicit weight-reading policy are retained. Writer finite checks precede opening the destination. Line sampling preallocates its final array and one segment workspace, preserving shared endpoint duplication and arithmetic.

Orientation uses the original arithmetic and absolute degeneracy thresholds for ordinary cells. Only overflowed norm/cross/projection paths rescale finite vectors. This extends demonstrated large-cell cases without claiming that every finite float cell has an invertible, representable derived geometry.

## Individually inspected functions

All 37 original functions were read with their data contracts and relevant callers; declarative registry/exports were read too. Three new source helpers and two benchmark functions are additional inventory entries. The existing `StructureRecord` reviews are not counted again.

| Module | Original functions changed | Original functions inspected and retained |
| --- | --- | --- |
| `io/converters.py` | `StructureConverter.read`, `_read_xyz`, `_write_xyz`, `_read_with_pymatgen` | `__init__` preserves dependency injection; `_detect_format` retains registry precedence; `write` and `convert` retain format/canonicalization delegation; `_write_with_pymatgen` retains optional format conversion and grouped species construction. |
| `io/native.py` | `_apply_scaling_factor`, `read_poscar`, `repeat_structure_along_c`, `_normalise_flags` | `PoscarData.natoms` retains integer total; `cartesian_to_direct` and `direct_to_cartesian` retain row-vector algebra; `wrap_direct` retains existing tolerance semantics; `write_poscar` retains pre-open validation, formatting, wrapping and explicit flags. |
| `io/kpoints.py` | `KpointsFile.line_points`, `write_automatic_kpoints`, `write_explicit_kpoints`, `write_line_mode_kpoints`, `read_kpoints` | `point_count`, `total_shift`, `mesh_points` preserve automatic/explicit/line delegation; `_format_number`, `_format_weight`, `_coordinate_header`, `_tokens` preserve accepted spelling/precision; `write_band_path` preserves divisions/spacing and path order; `write_mesh` retains Monkhorst implied-shift handling. |
| `io/orientation.py` | `OrientationNormalizer.align_ab_to_xy`, `align_c_to_z` | `ensure_right_handed` retains copied record, position transform and metadata behavior. |
| `io/vasp.py` | None | `VaspIO.read` retains owned arrays/flags and resolved provenance; `write` retains validation delegation, Cartesian/direct selection and destination semantics. |

New helpers: `read_poscar.next_line` reports required-section EOF, `_integer_token` validates exact whole-number tokens, and `_projected_axis` preserves ordinary projection while bounding overflow. Benchmark `_baseline_io` validates the requested Git commit and an explicit module allowlist; `main` prepares inputs outside tracing and asserts valid output equality before measurements.

Callers inspected include `VaspIO` and converter workflows, native c-repeat use in structure building, symmetry mesh/path serialization, and existing I/O tests of triclinic roundtrips and full periodic distance spectra. Broader moiré, slab, adsorbate, defect, renderer and CLI consumers are exercised by the full checkpoint rather than inferred safe from unit coverage alone.

## Resource evidence

`python benchmarks/benchmark_io_resources.py --baseline c7dec6c --repeats 3` compares trusted baseline modules in-process with prepared identical inputs. All record arrays/fields and valid XYZ output bytes match exactly in these cases. The mixed-species correction is separately tested against expected chemistry, not against the incorrect baseline order. Python 3.14.3 / NumPy 2.4.3, three timing samples plus one traced allocation sample per implementation:

| Prepared case | Baseline median ms | Current median ms | Baseline peak MiB | Current peak MiB |
| --- | ---: | ---: | ---: | ---: |
| First XYZ frame, 100k-frame file | 77.09 | 0.36 | 17.39 | 0.15 |
| XYZ write, 100k atom labels | 192.71 | 193.12 | 1.53 | 0.15 |
| POSCAR, 20k atoms and velocity tail | 23.36 | 32.56 | 6.33 | 4.30 |
| Homogeneous 20k atoms repeated three times | 2.41 | 1.68 | 4.12 | 2.75 |
| 64 band segments, 2048 points each | 5.58 | 5.38 | 6.02 | 3.16 |

These are local observations, not CI timing thresholds or universal speedup claims. The streamed POSCAR parser trades approximately 9 ms in this case for lower retained input memory and precise row errors. XYZ writer runtime is effectively unchanged while label storage is bounded. Tracemalloc is Python-tracked allocation, not a process RSS measurement.

## Verification and limits

The initial demonstrated integrity cases failed before implementation (30 failures, followed by three orientation and seven additional boundary failures). One combined focused I/O selection then passed 80 tests in 6.75 seconds. Independent review identified an additional writer flag-token gap; its two invalid-token regressions failed before the fix. The final affected-caller selection passed **127 tests, one warning in 12.82 seconds**, including I/O, KPOINTS CLI, workflow interoperability and molecular framing.

The one full coverage checkpoint passed **3,316 tests, four skipped, 81 warnings**. Its recorded wall time was 4324.78 seconds; this is verification timing, not an I/O benchmark. The unchanged dense geometry regression was retained. The skips cover the absent external Lean references and the existing surface-period case. All 253 frozen Python hashes and the exact tracked/new file set match after terminal success. Ruff F checks and canonical ledger drift checks pass.

Coverage is **12,613/14,474 statements (87.14%)** and **3,321/4,592 branches (72.32%)**. This measures exercised package code, not universal format/dependency or numerical-domain completeness. The minimum NumPy lane additionally includes the completed 84-test I/O group; its runtime and all supported-version/package/coverage results must be read from the pushed commit before claiming this checkpoint remotely verified.

The ledger now has **1,108 functions:199 reviewed unchanged,84 optimized verified,two investigated deferrals and823 pending**. Thus **285 functions have inspection evidence**. Part04 contributes37 original functions, three helpers and two benchmark functions; untouched pending modules remain pending. Initial reciprocal source inspection has reproduced an integer-key collision and unrepresentable-spacing undersampling, recorded in the next plan without source edits or completed ledger claims.

The independent reviewer read all scoped modules, new tests, benchmark and plan; it performed narrow in-memory malformed-input and Decimal probes, with no duplicate suite/build. Its initial request to tighten raw header/flag reading was withdrawn after checking the documented rendering contract in `core/species.py` and the explicit compatibility regression. Writer validation now rejects unsupported flags before opening output while preserving valid capitalization and existing extra-column trimming. Revised verdict: no outstanding blocking findings; runtime evidence remains the executor's, not an independent rerun.

Malformed records intentionally fail earlier. Empty XYZ molecules still cannot infer a cell; writing a coherent empty record retains its earlier behavior. Explicit KPOINTS reading retains zero/negative finite weights while its existing writer requires positive weights. Native raw selective flags remain permissive on read; a writer now refuses fewer than three flags or tokens other than T/F (case-insensitive) before truncating a destination. Extra native flag columns are still ignored.

Global extreme-scale geometry/validation remains the separately investigated deferral. For example, finite near-maximum fractional input can still overflow derived Cartesian coordinates. No new relative thresholds, format precision changes, or blanket validation policy have been introduced. Scientific runs and root concurrent changes remain untouched. Publication is unauthorized, and the whole-project function review remains incomplete.

## Package checkpoint

One wheel/source-distribution build and Twine metadata check passed. All 141 package module bytes match the frozen checkout in both archives; the two new regression modules and benchmark are present in the source distribution. Archive checks exclude scientific run/environment paths. All 253 frozen tracked/new Python file hashes match after the build.

Installed core-wheel, CLI-wheel and source-distribution environments each passed the I/O smoke under Python 3.14.3 / NumPy 2.5.4: mixed-species repeats and flags, POSCAR roundtrip/pre-open writer protection, XYZ frame/vacuum behavior and incomplete-frame rejection, line sampling and fractional count rejection, and large-cell orientation. Each used isolated Python mode and asserted imports came from its installed environment. Help and dependency checks passed in all three. Documentation-only source-archive refresh after final report/ledger updates does not require repeating these installs.

The resource measurements precede the final independent-review writer flag-token guard; none of the measured read/repeat/XYZ/line-sampling bodies changed afterward. The benchmark records its measured module hashes; it does not silently attribute those measurements to a different native writer validation body.
