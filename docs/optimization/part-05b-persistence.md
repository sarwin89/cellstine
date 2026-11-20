# Part 05b: bounded report encoding and persistence validation (local checkpoint verified)

Baseline: `66921213c3e1f2421b792aa165e945cf02b22a16`. That search checkpoint passed all seven jobs in CI41585027655. Exact-head coverage artifact13207679862, SHA-256 `a8e269f89c0aa39bc53083509fd56d845a9714e9c161ecc92fb165441d58b328`, was downloaded; XML matches12,927/14,794 statements and3,410/4,686 branches.

This batch manually integrates Claude's seven writer hunks and only the validation portion of follow-up04. The shared root source and scientific data remain preserved. Existing dev signed-orbit code, candidate schema and output reservations remain intact. CLI01 and follow-up05 remain separate.

## Whole-function scope and decisions

Independent review inspected108 production functions plus13 benchmark functions, including complete functions beyond changed lines, their callers and tests. All121 source-hashed rows are promoted after the full/package checkpoint:91 unchanged reviews and30 optimized verifications, including12 retained prior optimizations with unchanged bodies. Previously verified optimizations with unchanged function bodies retain their evidence when their module is re-reviewed. The inventory contains1,148 functions:392 reviewed unchanged,144 optimized verified,two investigated deferrals and610 pending. Thus538 functions have inspection evidence; the project-wide review remains incomplete.

| Module | Functions | Reviewed behavior |
| --- | ---: | --- |
| core/jsonio | 2 | Conservative small-plain-document budget and caller-owned indented writing. |
| core/manifests | 4 | Metadata creation/copies, serialization field order, persistence/path state and loading. |
| core/path_stage | 5 | Path/moved-atom tables, complete image/matching payload, migration outputs/warnings/provenance and warning tie selection. |
| defect/analysis | 20 | Layer census, symmetry orbit grouping, record/selection helpers, report persistence and position/flag/species reconstruction. |
| interface/surface/backend | 9 | Site reports, aliases/order/selection, surface/hollow detection, slab persistence and reserved default outputs. |
| interface/workflow/lattice_match | 20 | Requests, slab search/candidate conversion, ranking, schema validation/read/write/selection and formatting. |
| symmetry/kpath_stage | 3 | Point/segment display, decorated path sampling, KPOINTS reservations and workflow reports. |
| symmetry/symmetry | 15 | Backend/record analysis, primitive/conventional/lattice reductions, meshes, reporting and output/provenance consumers. |
| moire/search/results | 30 | JSON conversion/shapes, scalar/array/plain validators, GC/ownership, shortlist, document construction, read/write and chunked encoding. |
| benchmark helpers | 13 | Exact Git baselines, representative payloads, isolated timing, assertions, scoped unfolded validation oracle and cleanup. |

`core.jsonio` uses whole encoding only for bounded exact builtins. The preflight rejects containers over1,024 entries, nesting beyond32, nonfinite floats, integers over1,024 bits and custom types; other values retain streaming. Conservative costs include astral Unicode escapes, non-string key quotes and closing indentation. Independent review found underestimates in the prototype; the actual implementation corrects them. A further1,004 narrow probes bounded every accepted encoded length. The1MiB eligibility budget limits whole encoded text, not universal peak RSS. Callers retain opening/newline/trailing-newline policy and `allow_nan`; partial-write boundaries on I/O failure naturally differ between one and many writes.

The validator follow-up proves ordinary parsed candidate rows pass before skipping scalar helper calls. Anything outside that fast subset, or failing a check, invokes the original scalar error oracle. The existing int64 wrapped-zero determinant fix remains. Three hundred deterministic malformed-row mutations compare verdicts and exception classes/text against scalar validation, including missing fields, bools, tuples, subclasses, huge integers and singular matrices.

The existing unfolded-search persistence defect is corrected explicitly: unfolded recorded angles may span a signed half turn; folded records retain the old half-symmetry-period bound and1e-6degree tolerance. The stored physical `angle_period_deg` is unchanged. Square/hexagonal round trips failed before the fix. All38 comparative search cases preserve every scientific array/order and stable JSON field; for the eight old unfolded failures, a scoped baseline oracle corrects only scalar/array validation bounds and restores both wrappers in `finally`.

Defect reconstruction groups string labels once and applies one advanced-index permutation instead of rescanning all atoms per species and allocating a NumPy object for each row. Source-species priority, absent/new species, positions/flags, empty shape, optional lattice, shallow metadata and array ownership are preserved. Existing return conventions and scientific arithmetic remain unchanged.

## Evidence and measurements

The affected consumer selection passed273 tests/33 warnings in39.75seconds. All32 new regression cases also passed independently with a workspace temporary directory; the initial independent run's three fixture errors were the known sandbox default-TEMP issue. A final22-case helper run verifies the integer test also respects Python builds/settings that allow large integer serialization.

Final three-sample local measurements use explicit UTF-8 Git decoding against6692121 on Windows/Python3.14.3/NumPy2.4.3, after the full test process ended. Every writer byte/readback comparison and selected record-field/ownership assertion passes. The table reports milliseconds and Python-tracked peak MiB, not RSS or CI thresholds.

| Case | Old ms | New ms | Old peak MiB | New peak MiB |
| --- | ---: | ---: | ---: | ---: |
| manifest | 0.95 | 0.98 | 0.133 | 0.128 |
| defect-100 | 7.02 | 4.57 | 0.268 | 0.427 |
| surface-100 | 3.73 | 2.19 | 0.200 | 0.227 |
| path-encoding-100 | 7.14 | 2.84 | 0.179 | 0.351 |
| defect-2000 | 118.28 | 118.89 | 2.333 | 2.333 |
| surface-2000 | 41.00 | 42.83 | 0.866 | 0.866 |
| path-encoding-2000 | 127.63 | 125.16 | 0.178 | 0.178 |
| symmetry | 6.99 | 3.61 | 0.265 | 0.378 |
| kpath-encoding | 0.46 | 0.40 | 0.134 | 0.128 |
| interface-matches | 166.67 | 186.17 | 0.179 | 0.179 |
| read-results | 105.75 | 67.93 | 9.372 | 9.371 |
| regroup | 67.20 | 10.27 | 7.854 | 3.668 |
| regroup-flags | 78.58 | 15.67 | 8.523 | 4.337 |

Result-file reads and regrouping improve materially. Writer timings are mixed: tiny manifests have no useful gain, interface-match writing was slower locally and small-document temporary peaks sometimes increased. Large documents retain streaming allocation. No universal writer speedup or memory reduction is claimed. Path/kpath cases isolate their encoder expression; other cases call actual writers. The record benchmark checks selected fields and direct-array detachment; broader ownership/flags assertions are in the regressions. Encoder performance depends on Python version; these host observations do not prove equal speedups across supported versions.

Git subprocess decoding in the two benchmark files now explicitly uses UTF-8. Standalone38-case and13-case post-fix checks pass. This support-only correction occurred after the full run started: the original freeze manifest is retained,263 production/test/other Python files stay unchanged, and the two benchmark-only changes are recorded separately with fresh standalone verification. The full suite does not import either benchmark. Post-fix smoke timings overlap the full run and are not used as uncontended performance observations.

One wheel/source-distribution build and Twine checks pass. All142 package modules and four new regression/benchmark support files match the initial archives; run/environment data are excluded. Installed core-wheel, CLI-wheel and source-distribution environments pass persistence/unfolded/memory-policy/regrouping smokes, help and dependency checks under NumPy2.5.4; isolated imports are asserted. Final source-support bytes and the benchmark decoding fix are included in the refreshed source archive without repeating unchanged installed runtime checks. The minimum-NumPy CI lane adds all32 new cases.

The single full branch-coverage run passed **3,406 tests, four skipped,81 warnings in718.45seconds**. Coverage is **13,061/14,927 statements** and **3,505/4,778 branches**, adding134 covered statements and95 covered branches over6692121. All265 amended hashes and the exact tracked set match after terminal success, with the two benchmark-only changes explicitly recorded above and the original263 runtime/test/other Python hashes unchanged. The final source archive refresh matches142 package modules and15 support files; no repeated full run or installed runtime checks are needed for documentation/benchmark support changes. Commit837be429c51639ce1ce873611f12d6d8167164f3 is pushed to dev. All seven exact-head CI41591743264 jobs passed. Downloaded Python3.12 coverage artifact13211186847 (SHA25687daca9056e3f54af372826f89b7bf8e5ad336adb1ffab5ab1f3b18ffd5fc4ad) matches the local statement/branch counts. Unchanged finite numerical/group domains and the previously documented global deferrals remain; the unfolded persistence limitation itself is resolved by this batch's regression.
