# Safe default outputs implementation plan

> **Execution:** Continue autonomously inline using the existing executing-plans workflow. The user's latest instruction is to batch related code work, reduce repetitive verification and inspect the next area during CI. Use focused regressions while implementing, then one full/package/CI gate for this larger checkpoint.

**Goal:** Prevent generated default artifacts from overwriting existing files, including run100, different input labels and concurrent workflows, while preserving ordinary filenames and explicit output-path behavior.

**Baseline:** Part03b `a4a3a0ec4d24b782f79c1249659da1d0f459ec6d`; CI41398913199 runs in the background. Read/plan during that run; confirm its terminal result before the next code checkpoint.

**Contract:** Keep `run_output_suffix` parsing unchanged: numeric Miller/input labels cannot be distinguished from counters in historical IDs. Reserve the actual default destination through exclusive filesystem creation. The first available preferred filename stays identical; occupied names get `_02`, `_03`, ... before the final extension. Explicit file paths retain their existing overwrite semantics. Default batch directories are reserved atomically too. No user/scientific file is removed. A failed writer may leave its newly owned empty/partial output, consistent with retaining failed run artifacts; reservation does not promise transactional publication.

## Files and work

- Add `core/outputs.py` with `reserve_default_file(preferred: str|Path) -> Path` and `reserve_default_directory(preferred: str|Path) -> Path`. Create parents once; attempt exclusive leaf creation; retry only leaf collisions. Preserve existing files, directories and symlinks. Permission/parent failures propagate.
- Apply file reservation to default artifact choices in visualization, adsorbate placement/movement, surface/sites, interface building, symmetry reduction/KPOINTS, moire translation and defect generation. Keep explicit path branches unchanged.
- Reserve default batch directories for bilayer/N-layer make. Inspect lower-level automatic writers in builders, placement, layer shifting and visualization/surface backends so default API entry points do not bypass the same protection. Internal artifacts inside a freshly reserved run directory keep their existing names.
- Trace result/manifest paths to the selected destinations. Review caller return types, ordering, associated files and rename operations, particularly batch generation and surface/sites pairs. Record remaining working-directory/explicit-directory semantics in the report instead of claiming universal protection of every writer.

## Focused regression batch

- Public visualization: different inputs sharing the same timestamp preserve both outputs; first filename stays unchanged; legacy run100 and numeric-label111 cases stay safe without parser changes; explicit output paths retain their behavior.
- Shared helpers: existing bytes and directories survive, Unicode/extensionless names work, threads reserve distinct paths, and parent/permission failures do not loop.
- Exercise affected native workflow callers with their existing tests; add a regression for batch-directory reuse and generated-file collisions where applicable. Avoid separate full-suite/build cycles for each helper or caller.

## Larger checkpoint

- Independently review the complete change once, fix material findings and finish the actual function-level inspection ledger.
- Run one combined focused selection, freeze source/tests, then one full coverage suite and package/install pass. Update hashes/evidence/counts and resolve the output-suffix deferral only to the demonstrated scope.
- Commit/push the complete batch to dev and inspect exact-head CI/coverage. Use waiting time for the next area's read-only inspection; do not repeatedly rebuild or rerun already passing checks without new changes or evidence.

Local checkpoint: all26original functions and2helpers statically reviewed independently with no blockers;210caller tests plus3targetedbatch/siblingcases pass. One full suite3270passed4skipped81warnings857.74s;all250frozenfiles/file-setmatch. Ledger1103total243inspected860pending;175unchanged66optimized2investigateddeferrals. One wheel/sdist/Twine/content/install pass succeeded;141package modules match source. MinimumNumPylane adds13tests(total270). Finalsupportarchive refresh,scopedpush and remoteCI/readback remain.
