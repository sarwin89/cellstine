# Bounded bilayer concurrency implementation plan

> **For agentic workers:** Use superpowers:executing-plans inline. No implementation delegation.

**Goal:** Offer explicit threads for concurrent bilayer calculations and private file writes, and bound pending tasks while retaining ordered results.

**Architecture:** Extend the existing batch builder with `parallel_backend="process"` or `"thread"`. Keep workers=1 and process defaults. Use a shared lazy submission iterator bounded to twice the effective worker count; cap actual workers to the selection length. Threads do not run the process initializer or change environment settings.

**Tech Stack:** Python >=3.10, concurrent.futures, existing NumPy >=1.26,<3; no new dependency.

**Spec:** User's 2026-11-05 request: NPZ default provided nothing breaks, followed by parallel calculation and I/O for speed. NPZ commit ffafb35 closed with all16CI success and verified coverage artifact before this batch.

## Global constraints

- Preserve science, tolerances, candidate selection/order, duplicate selections, JSON/NPZ readers and file contents.
- Process backend remains default; explicit threads also work without multiprocessing main guards.
- No parent environment mutation, global BLAS changes, automatic worker count, or nested pools.
- Constructor-only OSError fallback is safe before dispatch; later failures propagate without retrying outputs.
- Each writer owns a separately reserved destination and file handle. Filename suffix assignment for duplicate candidates is scheduling-dependent as in the existing process backend.
- Bound queued futures and task payloads, not all result memory: the return API remains a complete list and keeps selection/result snapshot data.
- Preserve shared-root user/Claude work and scientific runs. No release/package publishing.

## Review focus

- Invalid backend must fail before reading inputs or creating run artifacts.
- Out-of-order task completion must retain requested result order.
- Partial write/dispatch errors must not replay prior writes; cancel work not yet started, allow active writers to finish.
- Huge worker requests and many selections must not create more workers than tasks or unbounded pending futures.
- Thread workers must not mutate shared result records or native-thread environment settings.

## Task 1: Concurrent builder, public API, CLI and documentation

Files: src/cellstine/moire/builder/make.py, src/cellstine/moire/moire.py, src/cellstine/cli/plain_moire.py, src/cellstine/cli/main.py; tests/test_moire_builder_resources.py plus focused concurrency/API tests; docs/moire-performance.md and docs/cli.md.

- [x] Add failing regression coverage for explicit thread backend, invalid early rejection, ordered bounded submission, cancellation/failure behavior, public forwarding/manifest and CLI parsing.
- [x] Add lazy ordered bounded dispatch and explicit backend selection; cap effective workers to selection length; preserve defaults and safe fallback boundary.
- [x] Exercise all3 formats, duplicate/nonmonotonic selections, exact POSCAR readbacks and contacts with actual threads/processes.
- [x] Benchmark final implementation against ffafb35 with the same owned fixtures; separate startup and native-thread observations, report local limits.
- [x] Independently review every callable in changed source modules; update source-hashed inventory and coverage evidence.
- [ ] Run focused tests, frozen full suite and package/install checks once for the final candidate. Commit/push dev, then check exact-head supported-version CI and coverage artifact before further production changes.

## Evidence before implementation

Ignored benchmark reports part05g-concurrency-{inherit,1,1-thread}.json use 365 actual candidates,12 largest selected structures (about600 atoms each),3 unprofiled repeats, exact native scientific readbacks and candidate signatures. In early-one-thread mode serial2.84s,2process2.47s,4process2.02s,2threads1.85s,4threads1.32s. This is a warm Windows local fixture with pool startup timed, no fsync or peak worker memory measurements, no universal speed claim. Existing late worker initializer leaves preloaded OpenBLAS at12 configured threads; no dependency change proposed without broader benefit evidence.
