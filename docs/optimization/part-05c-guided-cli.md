# Part05c: guided CLI integration and bounded discovery

Base checkpoint:837be429c51639ce1ce873611f12d6d8167164f3. Its seven exact-head CI41591743264 jobs and downloaded coverage artifact13211186847 are verified. This part integrates the verified Claude01/05 exports and their required pre-existing CLI dependencies into the isolated dev checkout, preserving shared-root edits/scientific data. Local full/package verification is complete. Scoped commit/push and exact-head CI are **pending**.

## Behavior and integration decisions

Guided Back replays validated answers and remembered in-memory previews; scientific execution remains outside the replay boundary. Bad paths, invalid/nonfinite numbers and parser errors retry at the question. Primary and follow-up execution failures retain editable answers, with completed commands never replayed. Declining a command returns to the menu. Rich renders scientific bracket text literally, and confirms through the shared navigation-aware loop. Existing plain/optional-Rich import boundaries remain.

Root's required history/decorator/default/prefilled-builder dependencies were absent from dev; a text-clean three-way merge alone left missing imports, discarded decorators, wrong follow-up routing and duplicate confirmation. The complete scoped prerequisite review restored those dependencies. Guided defect inspection deliberately uses pure `_analysis_from_input` returning DefectAnalysis, since the public preview command persists a run and returns CommandResult. Confirmed generation still uses the original source/settings; public preview behavior is unchanged.

Bilayer CLI shortlisting is opt-in and explicitly forwards parserNone for save-all; the direct Python `Moire.find` default remains500. A regression found CLI search parsed the limit without forwarding it; dispatch now forwards it. The guided rigid short path retains its existing400-atom default; advanced explicit opt-out remains possible. N-layer float lists now support arbitrary stack heights and one-value strain broadcasting, without changing its experimental/shortlist-limited numerical policy.

The original64KB count regex gave false bounds for nested keys and split number tokens. Prefix parsing also cannot rule out duplicate top-level keys later in a large file. Instead, the core private preview helper returns text and **actual candidate-row count from the same validated read**, retaining public preview text/API. Build reuses this count/cache; selected visualization reads once. Multi-result manifests have no single bound. Existing metadata validation permits some declared-count/actual-row discrepancies, so metadata alone is not a selection proof. New valid-file probes cover nested/split/duplicate/misreported counts and zero rows. No result schema or validator policy changed.

Command previews use POSIXshlex.join or literal PowerShell single quotes, escaping apostrophes and quoting shell-active characters. Independent review caught leading@ splatting, now quoted. POSIX splitting and an actual PowerShell7-to-Python native argument readback cover dollar/backtick/quotes/apostrophe/brackets/semicolon/path/@ cases. PowerShell's reserved exact`--%` token is dropped by its native binder even when quoted or expression-produced; that copied-command edge and legacy native quote handling remain explicit limitations. Guided execution passes original argv directly and is unaffected.

## Discovery measurements

File selection keeps original root priority, descending modification times, first-encounter ties, resolved duplicates across patterns/overlapping roots, and historical negative slicing. For a positive limit, a stable bounded shortlist is selected per root; lower-priority roots are skipped once full. A seen set is still proportional to unique inspected paths; this is not a universal bounded-memory filesystem scan. Modification times retain the existing non-atomic filesystem observation semantics.

Three-sample local benchmark against exact baseline837be429, Windows/Python3.14.3, two overlapping patterns and2000 files in an owned tree. Every selected path/order matches. Timings use warm filesystem metadata; peaks are Python-tracked allocations, not RSS.

| Case | Old ms | New ms | Old peak MiB | New peak MiB |
| --- | ---: | ---: | ---: | ---: |
| filled higher-priority input | 1515.39 | 12.18 | 3.479 | 0.032 |
| dense input only | 1475.20 | 1499.05 | 3.458 | 3.444 |
| partial input then dense run tree | 1485.61 | 1487.03 | 3.462 | 3.447 |
| overlapping roots | 1522.81 | 1533.47 | 3.488 | 3.470 |

The material gain is avoiding an irrelevant lower-priority tree. Dense-tree resolution/stat calls dominate, and heap selection provides no demonstrated timing improvement there. No universal speedup is claimed. `python benchmarks/benchmark_cli_discovery.py --baseline 837be429 --files 2000 --repeats 3` reproduces the cases. Evidence is `.tmp/part05c-discovery-benchmark.json` in the shared workspace; no scientific run tree was read or modified.

## Review and validation state

Independent inspection covered182 production callables, including all unchanged/nested/lambda bodies:92 navigation/preview,42 builders,48 parser/dispatch/N-layer. Three new benchmark helpers were inspected separately, making185 source-hashed individual rationale entries. Review artifacts are `.tmp/part05c-{navigation,builders,dispatch}-review.json` and combined `.tmp/part05c-function-notes.json`. No ledger promotion occurs before the frozen full/package gate.

Reviewer findings fixed: rigid atom cap, follow-up builder recovery, follow-up answer retention/retry and leading@ command quoting. Red-before/green-after regressions cover discovery's unwanted lower-root scan, count/preview helper, failed follow-ups and copied argument expansion. Existing CLI test adaptations keep the public API default, supply the new common/separate-strain answer and replace incomplete fabricated JSON headers with validated results; they do not discard old scientific assertions.

Affected CLI/consumer selection:196 passed,13 optional-library deprecation warnings in22.37seconds. The final special-token quoting extension separately passes3 cases after the@ fix. Stdlib-only reviewer probing resolved27 workflow calls with no missing/unknown keywords or missing parser attributes and imported no NumPy/Rich/ASE/spglib/plotting packages. Minimum-NumPy CI adds20 discovery/preview/quoting cases. All271 tracked Python files are frozen before the single full branch-coverage run; no subsequent Python edits are planned.

Existing numerical/domain limitations remain explicit: N-layer integer helpers cast/store int64, some screening uses floating determinants, N-layer search is experimental and shortlist-limited, and nested N-layer persistence validation is shallow. Saved defect-analysis reuse checks view direction rather than all new kind/backend/side/saddle settings. Three pre-existing unreachable legacy kpath dispatch branches remain dormant. These findings do not establish arbitrary-size integer safety, exhaustive stack enumeration, stricter schema validation or completion of the repository-wide review.

Portable no-pickle NPZ remains a separately planned user-authorized format batch after this checkpoint; see [portable I/O options](portable-io-options.md). Current JSON APIs/formats/defaults remain compatible. No release/package publication.


## Frozen local checkpoint

The single full branch-coverage run passed **3,472 tests, four skips and81 warnings in698.82seconds**. The120second faulthandler stack during the existing dense-void test was observational; the original process finished successfully and was not restarted. All271 frozen tracked Python hashes and the exact tracked set match after terminal success. Coverage is **13,533/15,256 statements** and **3,666/4,886 branches**, adding472 covered statements and161 covered branches over837be429.

Wheel/source builds and Twine checks pass;142 package modules and six new support files match archives byte for byte, with local environments/run data excluded. Initial build failure was the known sandbox default-TEMP permission issue before compilation; the corrected workspace-TEMP build passed. Core-wheel, CLI-wheel and source-install environments pass isolated CLI/replay/discovery/count/API smokes, public help and dependency checks under NumPy2.5.4. The lean core environment deliberately lacks an installed pip module, so dependency checks use external pip with --python; no runtime package repair was needed. Documentation/ledger-only source archive refresh does not require repeating unchanged installed runtime checks.

All185 current source-hashed reviews are promoted after these gates. The ledger contains **1,191 functions:483 reviewed unchanged,238 optimized verified,two investigated deferrals and468 pending**. Thus **723 functions have inspection evidence**; this is not whole-repository completion. The next production batch waits for this commit's exact-head supported-version CI and coverage artifact readback. No release/package publication.
