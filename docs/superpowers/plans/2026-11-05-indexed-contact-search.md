# Indexed contact search implementation plan

> **For agentic workers:** Use superpowers:executing-plans inline, with one independent read-only whole-batch reviewer. No implementation delegation.

**Goal:** Reduce exact contact calculation work using conservative periodic bucket candidates, while retaining the existing exhaustive path and all result conventions.

**Architecture:** Add a candidate-only method to existing `PeriodicSiteIndex`, returning original site indexes in ascending order within each query row. In `closest_contact`, retain the original first strip and strip boundaries; once its finite positive best distance supplies a conservative radius, use a bounded candidate index for remaining strips. Score candidates from the original unwrapped coordinate differences with the same bounded minimum-image kernel, square root and row-major argmin rules. Use the legacy dense strip path whenever indexing cannot safely help. Repair the existing geometry benchmark's stale package loader as part of making the measurement tool usable from the current src layout.

**Tech Stack:** Python >=3.10, NumPy >=1.26,<3 and existing geometry index; no dependencies or new worker defaults.

**Spec:** Persistent systematic review goal and user's request for calculation/I/O speed while preserving scientific behavior. Parent checkpoint6d2b641 passed all16 CI41998047816 and verified coverage artifact13401697811 before this batch.

## Global constraints

- Row lattices, exact periodic images, tolerance and stable first-pair ties remain unchanged.
- Keep empty groups, labels/count validation, nonfinite/extreme input behavior and public Contact schema. No nearest200 cap, dropped atoms, relaxed scientific precision or changed acceptance policy.
- Index is a conservative candidate filter only; never score wrapped/reconstructed coordinates in place of original raw differences.
- Preserve the first full strip and later strip update order. Index candidate rows must be original-index sorted, with duplicate bucket copies removed. Sqrt ties must retain old row-first/column-first argmin behavior.
- Expand only the candidate search radius conservatively for floating bucket boundaries; exact scoring still uses the old best distance and acceptance logic.
- Preserve legacy fallback for zero/nonfinite best distance, unsuitable numerical scale/conditioning or index-grid encoding range, tiny/fully completed batches. Do not add domain rejection through an optimization guard.
- Bound candidate tables using existing65536-pair strip budget; one very wide row remains linear as before. Measure index storage as well as scored pair reduction. No hidden automatic process pools/native thread changes.
- Full baseline/final verification, one final frozen full suite, package/readback/installed gates and scoped dev push followed by exact-head CI/artifact. No publishing.

## Review focus

- Fractional boundary/large-offset/skew-cell candidates cannot omit any pair that can beat or tie the first-strip best.
- Exact equal and sqrt-rounded equal distances retain original site and pair order; original arrays/labels are not mutated.
- NaN/Inf and unsafe numeric/index configurations use existing behavior rather than entering integer bucket casts.
- Empty/duplicate labels/sites, zero distance and one-wide-row memory cases keep their contract.
- Geometry benchmark must import this checkout's src package from an arbitrary current directory without running its heavy driver during unit import checks.

## Task1: Candidate enumeration, exact contact pruning and working reference benchmark

Files: src/cellstine/core/geometry.py (`PeriodicSiteIndex.candidates` only; existing match semantics retained); src/cellstine/core/contacts.py (`closest_contact` and small private guard/reduction helpers if needed); benchmarks/benchmark_geometry.py loader; tests/test_geometry_contact_index.py; docs/optimization checkpoint and source-hashed ledger.

- [x] Write and observe failing tests for candidate completeness/order, exact contacts on enough rows to cross the strip threshold, nonfinite/zero-distance fallback, scientific ties, no mutation, reduced exact scored-pair count and stale src-layout benchmark loading.
- [x] Implement candidate-only enumeration from existing bucket table and conservative contact use; keep original dense fallback and exact kernel arithmetic. Avoid cross-module access to geometry private index fields from contacts.
- [x] Compare new contact results to baseline6d2b641 on deterministic random/skew/boundary/tie/large-offset inputs and full native builder scientific readbacks.
- [x] Run repaired benchmark's independent image/neighbour/nearest/site checks once; inspect all12 pending benchmark callables and all complete bodies in changed runtime modules, preserving prior inspection evidence where exact bodies are unchanged.
- [x] Benchmark complete12-structure600-atom and2500-atom workloads, serial/process/thread as justified by the changed code; compare source-hashed predecessor, report wall-time variability and storage bounds, no universal speed promise.
- [ ] Independent whole-batch review, focused gates,283-plus-new-test Python freeze, one full suite/coverage and relevant package/installed checks, ledger promotion, separate dev commit/push, exact-head all16CI/artifact before further production work.

## Measurement ruling

The70566-row/12-structure large workload gives process48.31s serial/21.66s four workers and a separate thread51.13s serial/25.13s four workers, exact scientific signatures. CPU profile suggests contact minimum-image work dominates. An independent unprofiled result read validates all70566 records in1.79s and builder parsing in2.05s (three-repeat medians), not the9.6s shown under call-heavy cProfile instrumentation. Therefore do not implement a partial reader or skip whole-document validation based on the profile. General search was about2s. Threads won the smaller local fixture and processes won the larger; defaults stay explicit and unchanged.

## Pending dependency work

Prepared surface-cell24 and adsorbate-helper17 inspection notes are outside this runtime change and remain unpromoted. They record reproducible duplicate-header grouping/nonbijective matching issues and resource/validation opportunities for a later dependency-ordered surface/placement batch.

Ruling: Add the20 new contact/index/loader regressions to both the existing minimum-NumPy lane and all nine portable-result lanes; they use core NumPy only. This checks the new candidate API on Windows/Linux/macOS and NumPy1.26 without adding jobs or new dependencies.
