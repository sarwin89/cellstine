# Part05f: portable NPZ by default

Base1e01173aa2725431d035d8ecfdcc519fc529ab11 passed all16 exact-head CI41820006659 jobs. Coverage artifact13316437813 SHA256d1564189d31d0adb3c7f1683b3c8d546d10e8ec489944eef7fd2136b3cfb8db0 has verified head/module/statement sets; only the known path-display fallback differs locally. Local test-scope/package/installed checks are complete, using the full execution plus successful scoped reruns described below. Individually reasoned rows are promoted. Scoped commit/push and clean16-job exact-head CI/artifact remain next.

## Authorized behavior change

On2026-11-05 the user explicitly requested NPZ as the default, with calculation/I/O parallelization next for speed. This supersedes the earlier JSON-default constraint. NumPy is already required. A dependency-free constant supplies the default to Moire.find,run_find,default result paths,CLI search and molecular assembly. Guided search offers NPZ first and explicitly forwards a chosen JSON override. Search and assembly parameters record the normalized choice, including explicit JSON.

The default file/artifact is now results.npz/results_npz. Existing automation that hardcodes results.json/results_json must request --results-format json (or results_format='json'). This intentional default change is documented rather than hidden behind a mislabeled alias or duplicate automatic export. JSON schema/encoding/read support remains intact; explicit .json destinations still write JSON. The archive still contains the same validated Gram v2 document with the existing no-pickle versioned format. Compression remains opt-in. Public shortlist/search/geometry defaults and numerical tolerances are unchanged.

Molecular assembly shares the bilayer writer and now exposes the same normalized option through API/CLI, publishing the actual format's artifact. Preview/build/PNG/Plotly accept default archives directly or by manifest. N-layer top-level documents retain their separate experimental JSON schema, while intermediate bilayer pair paths now use valid NPZ. Interface v1 records retain actual JSON files for their results_json pointer by explicitly selecting JSON. No JSON is stored under an NPZ suffix or archive under a JSON artifact label.

## Review and regressions

Independent complete-body review covers93 current callables across10 runtime modules, including unchanged bodies, nested functions and lambdas. core/constants.py has no functions but its raw bytes/import independence are checked. Existing source-hashed review statuses are retained when appropriate; only five previously pending functions (four Adsorbate callables and one parser group) gain new inspection evidence. Sound unrelated code remains unchanged. The earlier40-function path/record/surface plan is parked without production edits.

Nine new default/guided/manifest regressions failed before the default changes; five assembly regressions then exposed its inherited archive/JSON-label mismatch and missing override; one interface regression caught the same default propagation under a fixed JSON-pointer schema. A path-helper test was corrected to unpack its existing(run_id,path) return before using its red result; no claim relies on the initial invalid assertion. The16 new cases cover API/CLI/guided defaults and overrides, metadata, no-pickle readback, exact JSON-export/build equivalence, preview, interface compatibility and N-layer pair archives. Existing JSON-specific encoder/schema fixtures remain explicit JSON. Format-neutral pipeline fixtures now use read_results and verify the same scientific assertions on default archives; N-layer top-level assertions retain JSON.

407 unique focused tests passed (244scientific/CLI/interface/rendering and163persistence/resource cases), with30 spglib deprecation warnings. Independent review found no runtime blocker; one P2 current-doc migration finding was fixed in quickstart, workflow, performance and source-adjacent moire guides, then independently rechecked. Dated optimization reports retain their historical descriptions; this report records the changed default. Minimum-NumPy and all nine portable CI lanes include the default regression file.

## Current production measurements

`python benchmarks/benchmark_result_formats.py --rows 20000 --repeats 3` compares complete production writing, reading, reconstruction and validation with exact document/key ordering assertions. Current source hashes are recorded in shared-workspace .tmp/part05f-production-benchmark.json. Three-sample local Windows/Python3.14.3/NumPy2.4.3 observations use warm caches/no fsync and Python-tracked peaks, not RSS.

| Rows | Format | MB | Write ms | Read/validate ms | Write peak MiB | Read peak MiB |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
|365|JSON|0.644|10.80|12.78|1.908|1.917|
|365|NPZ|0.140|12.06|19.06|0.280|1.280|
|365|compressed NPZ|0.061|16.12|15.79|0.644|1.280|
|20,000|JSON|35.194|553.83|863.54|10.097|98.594|
|20,000|NPZ|7.110|503.92|503.23|8.774|60.436|
|20,000|compressed NPZ|0.156|537.48|554.45|8.774|60.436|

Large NPZ reads are about1.72x faster and files about4.95x smaller locally. Small JSON remains faster. The expanded data cycles actual configurations with unique indexes, making compression unusually favorable; no universal speedup or representative distinct-candidate compression ratio is claimed. The encoder and calculation algorithm are unchanged; the default now selects an existing validated format.

## Next speed work and checkpoint

Calculation/I/O concurrency is authorized next, after profiling. Avoid serializing large Python documents per task, nested numerical-thread/process oversubscription, shared file handles, uncontrolled buffer multiplication and out-of-order merges. Existing explicit builder workers remain available. Measure complete scientific readbacks and representative small/large cases before selecting automatic thresholds. This checkpoint does not claim blanket parallelism or a calculation speedup.

The initial full/coverage run passed **3,663 tests**, failed two, and skipped four, with81 warnings in1143.65 seconds. One fixture still decoded NPZ as text; its reader was corrected without changing scientific assertions and all five rounded-input cases passed. The other failure was a1.5MiB allocation in unchanged dense-void code. Windows commit after the failure was27.27GB against30.98GB; the exact allocation-failure cause is not proven. Both failed node IDs then passed in an isolated coverage-appending run in845.65 seconds. The observed isolated worker held about0.5-0.6GiB private memory while CPU advanced. Geometry/void source is byte-identical to the verified base. Only the corrected test fixture differs from the initial283-file freeze; all runtime code is unchanged.

Thus all **3,665 test cases** have successful execution evidence from the complete run plus successful scoped resumption. **No clean single local full run is claimed.** The original failure log, freeze and coverage data are retained; a clean final-candidate supported-version CI run is required before the next production batch. Repeating the other3,663 passing cases after a test-reader-only correction was avoided. This preserves the user's request to batch verification while retaining every failed case in the audit.

Combined local coverage is **13939/15622 statements** and **3861/5058 branches**. All283 final Python hashes match the tested scope. Build/Twine and all144 runtime archive module comparisons passed. Core-wheel,CLI-wheel andsdist installed environments each passed lazy CLI/default checks, no-pickle NPZ/JSON equivalence, default search/preview/build/native symmetry/path checks, default and explicit JSON assembly, root help and dependency checks (Python3.14.3/NumPy2.5.4). Test/doc-only closure refreshes the sdist without reinstalling unchanged runtime. Final ledger:910/1,220 functions with inspection evidence,625 reviewed unchanged,283 optimized verified,310 pending and two investigated deferrals. Final docs/ledger closure refreshes only sdist rather than repeating unchanged runtime gates. All16 exact-head CI and downloaded coverage readback are required before next production changes. No publication/tag/upload is authorized; candidate versioning must account for the explicit default-filename migration.
