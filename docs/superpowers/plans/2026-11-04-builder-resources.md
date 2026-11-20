# Builder and layer review preparation

Base:8bfcf17127fee9c4e865f3c0c3140fbb11112e5b. Implementation plan. Part05d CI41777781487 passed all16 jobs and artifact13298477050 digest/head/readback is verified. The prerequisite is satisfied. User authorizes autonomous implementation and pushes to dev; no publication.

## Scope and evidence

43 individually reasoned, source-hashed rows in part05e-prepared-reviews.json cover core/layers.py, moire/builder/{generator,make,nlayer}.py, moire/supermoire.py and both transform wrappers.39 were pending;4 were already reviewed. Do not promote prepared rows as completed. Existing exact-coset, built geometry, contacts, reduction, N-layer, rotated/left-handed layer and output-reservation tests define invariants. Coverage holes are recorded from Part05d JSON.

Read-only probes:

- part05e-builder-probe.json: exact ij-order comparisons for5 signed/skew/diagonal matrices.1.5million cosets: old13.75ms/45.80MiB versus proposed7.12ms/22.90MiB (5sample warm local timings, tracked allocations). Candidate only; rerun actual changed production body against frozen reference before claiming optimization.
- Same probe: diagonal2**32 has exact determinant2**64; old HNF overflows NumPy scalar product and falsely reports singular. Python integer determinant plus representable-output guard is needed; never try to allocate this coset array.
- part05e-worker-probe.json: actual batch orchestration with fake executor and injected candidate-output OSError attempts indexes1,2,1,2,3 and swallows failure. Broad pool fallback retries previously completed writes. Reproduce as a regression before narrowing retry scope.

## Implementation tasks after CI gate

1. Copy plan into docs/superpowers/plans/2026-11-04-builder-resources.md and add Part05e report. Keep source module hashes in prepared review notes; all changes remain in isolated clone.
2. Add meaningful red regressions: oversized exact HNF, coset allocation guard before NumPy allocation, actual batch parser-count for multiple/duplicate indexes, empty no-I/O behavior, late invalid-index preserving earlier outputs, pool-construction fallback versus task OSError no retry. Use actual JSON/NPZ and N-layer fixtures for output equivalence; fakes only for scheduling/error routing.
3. generator: compute HNF determinant with Python integers; precheck final coset shape bytes against np.intp addressability; fill one C-order(h11,h22,2) grid by broadcasting axes, retaining original ij enumeration. Hoist affine conversion for nonempty layers without changing arithmetic or empty-input permissiveness. Do not vectorize all atoms or impose arbitrary atom caps.
4. bilayer builder: keep public single-candidate signature/fresh reads, extract private parsed-record writer. Parse/index once per nonempty batch, pass selected record/path/scalar tasks to processes without shipping full document or persistent global cache. Preserve requested order/duplicates, default name reservation, single-index explicit path, and late validation failure behavior. Restrict synchronous OSError fallback to safe executor construction; errors after task dispatch propagate to avoid duplicate writes. Document spawn-time fallback limits and existing native thread initializer limits.
5. N-layer builder: same private parsed-record reuse for nonempty batch, retain distinct schema/orientation and all per-layer gaps/repeats/flags/counts. Supermoire.maken may still perform its initial layer-count read plus one batch read; avoid expanding public API merely to eliminate one constant-cost read. Shared layer helpers remain unchanged if sound, add useful boundary assertions for currently missing branches.
6. Compare actual old/new full outputs and metadata in JSON/NPZ builds, selected order/duplicates and errors; benchmark old modules loaded from git base versus actual production source, including realistic and large repeated result documents. Clearly label repeated candidate compression/workload limitations.
7. One independent whole-batch read-only review under requesting-code-review skill; fix concrete findings, then one frozen full/coverage/package checkpoint. Promote complete individually reasoned rows only after focused/full/package success; verify artifacts and installed smokes. Scoped commit/push dev, all16 exact-head CI plus coverage artifact before next production batch.

## Deferred scientific boundaries

Keep numerical agreement thresholds, zfix sign/boundary rules, c-vector choice/tie, output species/within-species order, exact coset ordering and experimental N-layer shortlist policy. Extreme-scale float inverse/geometry and tilted-c padding are existing domain concerns, not resolved by exact HNF arithmetic alone. Oversized exact HNF can be calculated, but impossible output arrays must fail before allocation. No claim of arbitrary-scale end-to-end support.


## Local execution checkpoint

- [x] Red regressions reproduced13expected failures;46new tests and691broader cases pass.
- [x] Implement exact HNF/one-buffer ordered cosets/affine hoist, parsed batch contexts and safe worker fallback.
- [x] Independent48-callable review and minimum-NumPy CI delta review: no findings.
- [x] Production benchmark source hashes and exact bytes/metadata verified; small-case variation investigated with12alternating pairs.
- [x] Frozen277Python files; full3,556passed4skipped81warnings in938.97seconds.
- [x] Wheel/sdist/Twine,143runtime archive bytes and3installed environments pass; writable-cache sdist correction documented in part report.
- [x] Promote48 individually reasoned rows; ledger791/1211with evidence,420pending+2investigated.
- [ ] Final closure-document/ledger sdist readback, scoped commit/push, exact-head16CI jobs and downloaded coverage artifact.

Ruling: keep constructor-only process fallback and snapshot-per-call result reuse; changing input files during one batch is not used as a per-candidate refresh channel. Preserve direct single-candidate fresh reads and all output ordering. Do not change sources after freeze for documentation-only closure.
