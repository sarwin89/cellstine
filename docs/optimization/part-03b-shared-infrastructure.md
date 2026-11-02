# Part 03b: shared workflow infrastructure

Baseline: `4d005c06c30b1c2e71a7ea9cc0eaa2f0f16648d5`. Before starting this part, all seven jobs passed in CI run `41289513562`; all five full suites reported 3,230 passed/four skipped. The unexpired coverage artifact `13076513125` names that exact head, and its downloaded XML agrees with local coverage: 12,377/14,290 statements and 3,206/4,502 branches. The actual minimum lane used Python3.12.10/NumPy1.26.4 and passed208 tests.

## Individually inspected functions

All44 pre-existing functions were read, their callers/tests traced, and independently invoked during read-only review. Test coverage alone was not used to infer inspection. The new benchmark's two functions were also reviewed.

| Module | Functions | Decisions and preserved contracts |
| --- | --- | --- |
| `core/base.py`9 | `_slug`, `run_output_suffix`, `_json_ready`, `Base.__init__`, `.choose_backend`, `.create_run_dir`, `.write_manifest`, `.result`, `.resolve_results_file` | Change directory reservation to atomic mkdir-first retry. Defer output-suffix ambiguity below. Preserve slug policy, root resolution, backend delegation, recursive conversion/copy conventions, manifest fields, result ownership and ordered artifact-key lookup. |
| `core/dependencies.py`6 | `DependencyManager.__init__`, `.status`, `.has`, `.versions`, `.choose_backend`, `.choose_symmetry_backend` | Unchanged: per-instance cached discovery, lazy metadata import, available-without-version case, backend priorities and error messages. No eager import of optional implementations or new cache invalidation policy. |
| `core/validation.py`5 | `ensure_existing_file`, `ensure_positive`, `coincident_site_pairs`, `structure_errors`, `validate_structure` | Reject NaN in positivity check; preserve positive infinity. Defer extreme determinant/scale investigation below. Keep path existence semantics, periodic duplicate pairs/order, 1e-4 duplicate tolerance and contextual errors. |
| `core/report.py`12 | `_is_pathlike`, `format_path`, `format_number`, `_unit_for`, `format_scalar`, `_is_scalar`, `format_label`, `_render_sequence`, `_wrap`, `render_mapping`, `render_timings`, `format_result` | Unchanged: six significant digits, unit keys, path relativity, bool/None presentation, nested indentation, sequence order, width96, timing order, preview blocks and warnings last. |
| `core/provenance.py`8 | `_fields`, `_is_signature`, `_is_stage_field`, `stage_of`, `comment_notes`, `_assemble`, `stage_comment`, `restage_comment` | Change membership and suffix assembly only. Preserve field splitting/filtering, case-sensitive deduplication, stage parsing, mandatory signature, newest retained suffix and repeated newly appended notes. |
| `core/manifests.py`4 | `RunManifest.create`, `.to_dict`, `.write`, `.load` | Unchanged: UTC timestamp, string fields, shallow input mapping ownership, exact serialized field set/order/indentation, resolved path and optional-field defaults. |
| `benchmarks/benchmark_infrastructure.py`2 | `_repeat_call`, `main` | Trusted Git baseline resolution, exact output comparison, positive argument checks, prepared inputs, allocation boundary, microbatch counts and source/environment metadata checked. |

Callers include every public workflow's run setup/manifest handoff, I/O converters, POSCAR writing/validation, CLI result rendering, surface/moire builders, defect generation and pipeline interoperation. `_json_ready` retains `asdict` semantics: avoiding its array copy alone would alter copying of unhandled dataclass leaves. Direct manifest writes and dependency-cache lifetime also retain existing contracts; this part introduces no new transactional or refreshing API.

## Reproduced fixes

`Base.create_run_dir` previously checked existence before mkdir, allowing another workflow to win between those operations. A deterministic competing reservation and a real eight-worker/24-run test both failed with `FileExistsError`. The new loop attempts mkdir first, retries a confirmed occupied target with the existing `_02_`, `_03_`, ... convention, and propagates permission or parent-path failures. Existing competing directories/files and their contents are preserved. This guarantees distinct reserved run directories; it does not establish uniqueness of separately named flat output artifacts.

`ensure_positive` used `number <= 0`, which is false for NaN. Three representations of NaN failed the new rejection tests. The predicate now requires `number > 0`; existing positive infinity remains accepted. Another1,007 baseline comparisons cover unchanged finite/infinity/string/error behavior. This is not a new finite-only policy.

`_assemble` repeatedly joined the full history and removed the first note until the line fit. The failing allocation case prepares1,000 oversized notes plus two retained short notes before tracing: peak2,045,930bytes. The replacement accounts for the existing mandatory fields and delimiters, scans backward to the exact longest permitted suffix, and joins once. It still constructs the normalized notes list and consumes the whole supplied iterable, preserving conversion behavior. Mandatory fields can exceed the limit when the stage itself is too long, as before.

`comment_notes` uses a set alongside the ordered result to avoid repeated linear membership scans. `restage_comment` uses a set of existing notes only; it deliberately does not add new notes to that set while appending, because repeated newly appended notes have always survived. String equality remains case-sensitive; signature/stage filtering retains its existing rules.

## Verification and measurements

The new tests initially reported **six failed,21 passed**: two directory races, three NaN cases and the assembly allocation. After the changes, the focused integration selection passed **62 tests**, one existing spglib warning, in6.15seconds. Independent review passed49 focused tests and used additional direct assertions plus function-entry profiling to exercise **44/44 original functions**. No material finding remained in the five changed bodies. Two pre-existing scoped defects remain explicitly deferred below.

An additional **30,000 exact old/new comparisons** covered assembly, note parsing and restaging with Unicode, iterables, duplicates, empty/oversized stages, embedded separators and character limits from negative values through500. Finite positivity/error comparisons also pass. AST comparison identifies exactly five changed original executable bodies and no new/deleted production function. Ruff F checks and whitespace checks pass.

Run `python benchmarks/benchmark_infrastructure.py --baseline 4d005c0`. Each of six cases asserts exact output equality, uses three timing samples and prepares inputs before tracing. Ordinary micro-operation samples batch1,000 calls; allocation traces cover a single call. Python3.14.3/NumPy2.4.3 final-source measurements:

| Case | Baseline median ms | Current median ms | Baseline peak MiB | Current peak MiB |
| --- | ---: | ---: | ---: | ---: |
| Oversized history, two short notes retained | 49.5825 | 0.1634 | 1.951151 | 0.008647 |
| 8,000 unique notes | 364.2917 | 4.1275 | 0.904375 | 1.109560 |
| 8,000 fields,4,000 unique notes | 170.1913 | 4.0405 | 0.902258 | 0.902464 |
| Long restaging history | 391.2526 | 4.2128 | 0.602365 | 0.980585 |
| Ordinary stage,1,000 calls | 0.8100 | 1.1915 | 0.000428 | 0.000428 |
| Ordinary restage,1,000 calls | 2.2340 | 2.8840 | 0.000672 | 0.000878 |

Set membership trades extra memory for linear-time note processing. Ordinary cases add about0.38/0.65microseconds per call in this run; no universal speedup or constant-memory history normalization is claimed. Timings are observations, not CI gates.

## Investigated deferrals and release follow-ups

**`run_output_suffix`: filename ambiguity and overwrites.** Directory reservation100 produces `build_sample_100_TIMESTAMP`, but the existing parser recognizes only two-digit counters; the first and hundredth reservations return the same suffix. Independent reproduction through `Visualize.structure`, with real directory/manifests and a mocked HTML writer, confirmed both runs reference `structure_viewer_TIMESTAMP.html` and the second overwrites the first. Different input/stage labels can also share a flat artifact basename within the same minute.

A pure parser cannot distinguish a numeric label from a collision counter: `create_run_dir('build','111')` and reservation111 of `create_run_dir('build')` have identical run-ID syntax. Numeric labels are common: surface generation ends its label with Miller indices such as111. Expanding the parser to all digit lengths would rename valid ordinary surface outputs and would still not solve general cross-label output collisions. This function remains **inspected, deferred**. The next scoped output-allocation review must protect existing artifacts, retain ordinary noncolliding filenames, cover run100/cross-label/concurrent cases and decide an explicit unambiguous naming/reservation contract across consumers. This is a release follow-up, not a resolved uniqueness guarantee.

**`structure_errors`: representable numerical domain.** A one-atom cubic cell scaled by1e100 passes;1e103 is falsely called degenerate with volume`inf` because determinant/product arithmetic overflows. Further independent probes show `minimum_image_distances(eye(3)*1e155, [[.25,0,0]])` returns`inf`, although the true2.5e154 is representable;1e200 has the same problem. An isolated validation predicate repair could admit cells into known downstream overflow paths. This function remains **inspected, deferred** pending an end-to-end validation/geometry range investigation. Preserve the established `1e-8 * max(product_of_row_lengths,1)` degeneracy threshold and duplicate-distance policy while doing that work. The ordinary validation path is revalidated here; the extreme defect is not claimed fixed.

Other review decisions: positive infinity remains valid by explicit policy; `ensure_existing_file` retains existence-only behavior; heterogeneous set sorting and expansion of unsupported nested report sequences are outside the current output contract; manifest deep-copy/cache-refresh changes are not introduced; overlong mandatory provenance fields are retained. These decisions preserve demonstrated behavior and are not silent omissions from the wider review.

## Checkpoint gates

The full frozen coverage run passed **3,257 tests, four skipped,81 warnings in727.34seconds**. Three skips require the absent external Lean reference; one is the existing surface layer/period case. The diagnostic faulthandler dump in the dense-void test was followed by terminal success. All248 frozen Python hashes match after the run; benchmark source digests match. Ruff F, whitespace and canonical inventory checks pass.

Package coverage: **12,412/14,301 statements (86.79%)** and **3,229/4,504 branches (71.69%)**. Base has77/77 statements and28/28 branches; provenance50/50 and14/14; dependencies61/62 and23/24; manifests37/37 and no branch rows; report136/159 and60/76; validation62/68 and25/30. Remaining gaps remain visible and do not inherit correctness claims from these percentages.

Wheel and source distribution build and Twine metadata checks pass. All140 packaged Python modules match source bytes; new benchmark/tests are included and scientific runs/temp environments excluded. Wheel core-only, wheel CLI and source-distribution installations under NumPy2.5.4 each pass concurrent directory, JSON handoff, positivity, provenance, help and dependency checks. The installed assembly case peaks at9,067bytes. The final source archive's documentation/ledger/support files are compared again after this checkpoint text is written.

The ledger contains **1,101 functions:174 reviewed unchanged,39 optimized verified,three investigated deferrals and885 pending**. Thus **216 functions have inspection evidence**. This part adds44 original reviews and two benchmark reviews; the two investigated defects are not counted as optimized or unchanged. Changed-module evidence is reconciled only after direct inspection and the executable AST comparison. Publication remains unauthorized.

The minimum NumPy1.26.4 CI lane is extended to the two new infrastructure/provenance modules and existing validation/report tests, alongside the208 structure-helper tests. These modules require no optional runtime package; the backend tests explicitly cover absent optional implementations. This expanded lane must pass on the final Part03b commit; the earlier208-test Part03a result does not establish its success.

Incremental independent review of that CI change found no issue;49 added tests passed with optional imports blocked and pytest plugin autoload disabled. Local collection confirms257 tests in the expanded lane. Exact-version runtime confirmation and all five full supported-version jobs/package/coverage remain the pushed-commit gates before Part03c.
