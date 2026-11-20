# Part05g: bounded bilayer build concurrency

Base `ffafb352e684ee73eca5aad514db9d3dfbfb7015` passed all 16 exact-head CI jobs 41987107250, and coverage artifact 13396923732 passed digest/head/module/statement comparison before runtime changes. The two known CI-only path-display fallback statements have no branch delta.

## Behavior and invariants

Bilayer build accepts `parallel_backend="process"` or `"thread"` through the direct batch API, `Moire.make` and `--parallel-backend`. Existing defaults remain process and one worker. Each task performs the full selected-structure calculation and writes to its own reserved file; returned results preserve requested order, including duplicates. The manifest records the selected backend and requested worker count. Invalid backends fail before input reads/run artifacts.

The actual pool is capped to the selection length. A lazy queue holds at most twice the effective worker count in futures/task payloads. This bounds scheduling memory, not all batch memory: requested indexes, result snapshot/index map and the public returned list remain proportional to the batch. Threads share the caller's numerical runtime and do not run an initializer or change native thread limits/environment. No dependency was added. Processes retain their existing initializer; environment defaults cannot reconfigure preloaded BLAS libraries. A diagnostic proved that import order matters; no claim this batch fixes native-thread oversubscription.

On construction-only OSError, no work was dispatched and the serial fallback is safe. Submission/worker/write errors propagate; unstarted pending tasks are cancelled and active writers finish on executor shutdown. Files from earlier or already-started work, including partial reserved files, remain. No completed batch is replayed. Duplicate candidate filename suffix assignment remains scheduling-dependent, as in the prior process backend; requested result order is stable. Each future is consumed in request order, so an early slow task can delay reporting later completions; this is intentional order preservation.

Scientific algorithms, tolerances, interlayer separation, atom/candidate order, JSON/NPZ/compressed readers, POSCAR bytes/flags, contacts and shortlist policies are unchanged. This is selected bilayer build concurrency, not blanket parallelism of search, N-layer builds, symmetry or every application routine.

## Measurement

Python 3.14.3 / NumPy 2.4.3 on local Windows. A two-atom hexagonal carbon layer with 3 Å in-plane vectors and 12 Å c is searched to 35 Å with 0.01/0.01 logarithmic strain budgets and 5–25° twist window. The 365 actual candidates are persisted as NPZ; the 12 largest by descending atom_count then stable index are built, roughly 600 atoms each. Three unprofiled repeats use fresh output directories and fresh pools; startup and writes are timed, caches warm, no fsync. Profiles are separate serial process-CPU observations; no RSS or peak worker-memory measurement.

| Code/configuration | Serial s | 2 workers s | 4 workers s |
| --- | ---: | ---: | ---: |
| Base process / native threads 1 before import |2.8353|2.4715|2.0154|
| Final process / native threads 1 before import |2.7418|2.3291|1.8399|
| Final thread / native threads 1 before import |3.1148|2.0215|1.3717|
| Final thread / inherited native threads 12 configured |2.6038|1.8846|1.2294|

All base/prototype/final modes have the exact same candidate signature, selection order, native readback coordinates/cartesian positions/lattice/species/counts/selective flags, angle/contact values/species/notes. The per-build scientific signature is `e3aeb2f5a4c0442c6748d2242dff136b58c3bb9b16729a9f8e6e466db5a2edea`. The three repeats write 486,758 bytes each. No equality of filename timestamps or pool scheduling is claimed. Serial timings across consecutive runs vary; the small process timing difference is not attributed solely to the bounded scheduler. The thread path showed useful local scaling, but the fixture is not a universal crossover or speedup guarantee.

Source-hashed raw reports and disposable harnesses are retained in shared-workspace `.tmp/part05g-*concurrency*`; unprofiled wall medians are separate from process-CPU profiles. Search on this fixture was only about 25 ms; no search concurrency policy is inferred. Larger search/native-thread scaling needs its own measurements before implementation. The only configured thread-limit probe dependency lives in ignored diagnostic tools, outside package requirements.

## Review and validation

Independent complete-body review inspected all 26 callables in the four changed modules plus relevant generator, cache, file-reservation and writer paths. No active findings; normalized inventory/module hashes and raw file hashes are recorded. The shared geometry cache publishes fully constructed entries and its consumers read arrays only; simultaneous misses can repeat a reduction or clear reusable entries without changing arithmetic.

The focused 93-case batch passes. Coverage includes actual processes/threads across all three formats, nonmonotonic duplicate selections, one snapshot read, exact POSCAR bytes/contacts/flags, CLI forwarding, API manifests, early invalid backend rejection, capped workers, bounded consumption with out-of-order completion, cancellation and non-replayed failure behavior. A previous inline fake assumed later tasks never started before an observed failure; it now permits the three already-started requested writes and still prohibits replay. The red failures were missing keyword/helper/CLI support, not scientific mismatches.

One complete frozen full run passed **3681 tests**, skipped 4, with 81 warnings in 740.51s. Local coverage is **13961/15644 statements** and **3875/5072 branches**. All 283 frozen Python files still match; wheel/Twine, 144 runtime wheel byte comparisons, and installed core/CLI environments passed actual serial/process/thread exact-byte builds and dependency checks. Final sdist/install and exact-head CI/artifact remain required before further production changes. No release/tag/package publication is authorized.


The refreshed ledger contains 1221 functions and 911 with inspection evidence. Counts: 625 reviewed—unchanged, 312 deferred—with reason, 284 optimized—verified. Of the deferred rows, 310 remain pending and two are investigated limitations (rational lattice basis and extreme-scale structure validation). This batch refreshes 26 complete reviews and adds one private scheduler helper; it does not close the project-wide review.


Final sdist build/Twine and the third installed environment passed the same serial/process/thread exact-byte and dependency checks. All three isolated installed environments are verified. Documentation/ledger closure refreshes only the sdist; unchanged runtime wheel/install/full-suite gates are not repeated. The tested 283 Python-file freeze and all 144 runtime module bytes remain authoritative. Scoped dev push and exact-head 16-job CI plus downloaded coverage identity/digest/readback are required before the next production batch.


### Exact-head CI closure

Commit `6d2b6414b248cfa71220e58a411fcef9ba0416d6` passed all 16 jobs in [CI 41998047816](https://github.com/sarwin89/cellstine/actions/runs/41998047816). Downloaded coverage artifact 13401697811 has verified ZIP SHA256 `56b21312ae7a62a8a3d54a22d2c33d0639d854fadb14640d3f68f77f5db5b460`, commit identity and matching module/statement sets. CI coverage is 13963/15644 statements and 3875/5072 branches; only known `prompts.py`501-502 fallback statements are CI-only, with no branch delta. This closed the gate before the indexed-contact runtime batch.
