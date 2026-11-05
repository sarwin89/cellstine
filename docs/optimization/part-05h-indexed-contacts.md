# Part05h: conservative indexed contact search and usable geometry benchmark

Base `6d2b6414b248cfa71220e58a411fcef9ba0416d6` passed all 16 jobs in [CI 41998047816](https://github.com/sarwin89/cellstine/actions/runs/41998047816). Coverage artifact 13401697811 ZIP SHA256 `56b21312ae7a62a8a3d54a22d2c33d0639d854fadb14640d3f68f77f5db5b460` matches its head and module/statement sets. CI coverage 13963/15644 statements and 3875/5072 branches differ from local only by known prompts.py501-502 path-display fallbacks, no branch delta. This closed the previous gate before this runtime batch.

## Behavior and invariants

`PeriodicSiteIndex.candidates` exposes conservative original-index-ordered bucket candidates with validity masks. Every site within the index tolerance appears, independent of labels; extra sites still require exact scoring. Returned tables own their storage and have output-sized memory. The existing matcher semantics are unchanged.

`closest_contact` still computes the first entire legacy strip, with the original 65,536-pair budget and row boundaries. A positive finite best distance can seed an expanded conservative bucket radius for later strips. Bucket queries wrap coordinates, but scoring uses original unwrapped input differences and the existing bounded minimum-image kernel, square root, argmin and strict best update. Original-index candidate order preserves row-first/column-first ties, including different squared distances with equal rounded square roots. Invalid padded lanes are zeroed before the kernel and masked to infinity afterward. No nearest-neighbor cap or changed distance/acceptance policy is introduced.

Zero/nonfinite best values, nonfinite inputs, unsuitable scale/conditioning, unsafe bucket integer ranges and more than 16,384 second sites retain the legacy dense path. These are performance guards, not new domain rejections. The index spread is bounded by that site cap; candidate strips remain at most 65,536 entries except one inherently wide row. The first full strip and index construction still cost time/storage, so tiny or dense cases need not improve. Geometry reduction's preexisting extreme-domain limitations are not newly certified.

The geometry reference benchmark now loads `src/cellstine` from its own checkout instead of assuming a removed root `__init__.py`. Its 12 complete callable bodies were inspected, with the loader repaired and sound scientific references/driver retained. Existing loaded-module cache behavior is unchanged; a fresh subprocess verifies the intended checkout from another working directory.

## Measurements and scope

All timings below are three-repeat warm Windows / Python 3.14.3 / NumPy 2.4.3 observations with numerical threads set to one before import. Pool startup and actual structure writing are timed; no fsync. Profiles are separate CPU observations, not wall timings or RSS. Baseline source is the verified predecessor 6d2b641 and final reports contain changed runtime hashes. No universal speed or crossover claim is made.

Twelve actual structures at 35 Å (365 candidates, roughly 600 atoms each) and 70 Å (70,566 candidates, 2,510–2,538 atoms each) retain the exact candidate, selection and native structure/contact signature across every baseline/final worker mode, including species/counts/lattice/cartesian/direct coordinates/flags/angles/contact values/species/notes. Baseline file output sizes are retained.

| Fixture / backend | Baseline serial s | Final serial s | Baseline 4 workers s | Final 4 workers s |
| --- | ---: | ---: | ---: | ---: |
|35 Å process|2.7418|2.6573|1.8399|1.6728|
|35 Å thread|3.1148|2.4728|1.3717|1.2138|
|70 Å process|48.3116|10.3453|21.6602|6.2268|
|70 Å thread|51.1286|10.4188|25.1343|5.8472|

These sequential whole-build samples vary in serial baseline; do not attribute every small difference to the index. Same-process interleaved baseline/new contact calls provide a second check on structured aligned-plane inputs. The original baseline function AST is loaded from Git and uses unchanged metric functions. Each measured call returns identical finite carbon contact metadata. Separate allocation probes are Python/NumPy-tracked peaks, not process RSS:

| Sites per group | Baseline ms | Indexed ms | Baseline tracked MiB | Indexed tracked MiB |
| ---: | ---: | ---: | ---: | ---: |
|100|21.38|21.58|4.792|4.792|
|400|333.83|145.84|14.373|11.154|
|1296|3656.77|178.43|14.805|11.109|

The 100-site path is effectively unchanged and can be slightly slower in individual observations. Sparse larger groups show useful pruning; no output-sized table or all-input memory guarantee is removed. The sparse regression counts exact scored rows, not elapsed-time assertions.

A separate unprofiled full NPZ read validates all 70,566 records in 1.79s and complete builder parsing in 2.05s, with exact document/key order. cProfile's 9.6s validation figure is instrumented overhead, not raw I/O wall time. Full-document validation remains; no partial/trusted reader was added. General search was about 2s, so blanket search parallelism was not selected from this evidence. Existing explicit process/thread build backends and their defaults remain.

## Regression and review

The initial red run had 6 failures / 1 pass: absent candidate API, unpruned scored rows and stale benchmark package loader. Twenty focused regressions now pass: three periodic cell geometries, boundary and duplicate sites, returned ownership/order, empty candidate shapes, sparse pruning, sqrt-one-ulp tie, eight exact dense-reference late-tie cases with skew/offset coordinates, three nonfinite fallbacks, tiny/zero radii and fresh-process src loading. An allocation harness initially compared dictionaries containing legacy unknown-species NaNs; matching diagnostic values showed no scientific difference. It now supplies carbon labels so every compared metadata field is finite. No runtime change was needed for that harness correction.

The broader 243-case contact/geometry/molecule/native builder/NPZ batch passes. The repaired reference driver passes independent brute-force minimum-image, neighbor counts, nearest-point and site matching for its three cells, plus native symmetry/void consumers. Independent review inspected all 66 current callable bodies in geometry (38), contacts (16), benchmark (12), including exact AST/body changes and per-function source hashes; no actionable findings. Four bodies changed/added: candidate API, private guard, closest contact integration and loader. Fourteen functions gain new inspection evidence (12 benchmark functions and two new helpers); sound inspected bodies retain their prior status. Prepared future surface/placement notes are not promoted here.

The new 20 regressions also run in the existing minimum-NumPy job and nine cross-platform result jobs. One clean frozen full run passed **3,701 tests**, skipped 4, with 81 warnings in 727.32s. Local coverage is **14003/15687 statements** and **3893/5092 branches**. All 284 Python hashes remain fixed. Wheel/Twine, 144 runtime wheel byte comparisons and two installed core/CLI environments passed actual candidate/sqrt-tie/native serial/process/thread builds plus dependency checks. The sdist build/Twine and third installed environment also passed the same candidate, exact tie, native serial/process/thread and dependency checks. All three installed environments are verified. Documentation/ledger closure refreshes only the sdist; unchanged runtime wheel/install/full-suite gates are not repeated. Scoped dev commit/push remains next. Exact-head supported-version CI and downloaded coverage readback gate the next production batch. No publication/tag/package upload is authorized.


Ledger: **1,223 functions**, **925 with inspection evidence**; 636 reviewed—unchanged, 287 optimized—verified, 300 deferred—with reason. Of deferred rows, 298 are pending and two are investigated numerical limitations. This batch refreshes 66 complete reviews and adds 14 inspection records; the project-wide goal remains incomplete. Prepared surface/placement notes are outside this promotion.
