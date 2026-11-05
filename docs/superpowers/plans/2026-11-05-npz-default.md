# Make portable NPZ the default bilayer result format

User steering on2026-11-05 explicitly authorizes NPZ as default if existing functionality remains working, with calculation/I/O parallelization next for speed. This supersedes the earlier JSON-default constraint only for the bilayer search format choice. NumPy is already required. Existing JSON reader/writer/schema support remains available explicitly. N-layer and interface records use different schemas and keep their current format pending a separate complete implementation; do not put JSON under an NPZ filename.

Base dev1e01173aa2725431d035d8ecfdcc519fc529ab11 is clean and remote-matched. All16 CI41820006659 jobs and downloaded coverage13316437813 digest/head/module/statement readback passed. The prepared40-function path/record/surface plan is parked, with no production edits, until this explicit priority is complete.

## Implementation and verification

- Centralize a dependency-free DEFAULT_RESULTS_FORMAT='npz' constant. Use it consistently for Moire.find, run_find, default result path, CLI registration and guided advanced choice. Keep explicit json and npz-compressed choices working; guided JSON choice must be forwarded even though it is no longer the default. Leave standalone explicit .json destination encoding unchanged.
- First add regressions proving defaults through direct/API/CLI/guided paths, default manifests/preview/build and rendering, with legacy JSON equivalence and real scientific readbacks. Adapt existing JSON-specific tests to request json explicitly; use read_results for format-neutral scientific assertions. Do not weaken assertions or add a mislabeled results_json archive alias.
- Update current CLI/usage/migration documentation. Historical checkpoint reports retain their dated/default descriptions, with a linked follow-up note. Document that the default result file/artifact changes to results.npz/results_npz and that automation needing JSON must request it.
- Reuse prior full production NPZ comparisons; no universal speedup claim because small JSON can be faster. Profile calculation and I/O paths before new concurrency. Preserve result order and exact scientific outputs, limit simultaneous buffers/workers, avoid nested BLAS/process oversubscription and never share file handles between writers.
- Independently review every callable in changed source modules, including unchanged bodies. Freeze Python bytes; one focused selection and one full/coverage/package/three installed checkpoint. Promote only source-hashed reviewed rows, refresh final sdist docs/ledger, scoped commit/push dev, then16 exact-head CI and downloaded coverage before next production batch. No release/tag/upload.

## Verification execution adjustment

The complete local run had3,663passes plus a stale JSON-reading test fixture and an allocation failure in unchanged dense-void code. Only the fixture reader changed; its five cases and both initially failed node IDs passed in scoped runs, with original coverage appended. Allruntime hashes remained unchanged. Record this as complete test-scope evidence via successful resumption, not a clean singlelocalfullrun. Keep the original failure and resource evidence; clean full supported-version CI is mandatory before another production change. This avoids repeating unaffected passing tests after a test-only repair.
