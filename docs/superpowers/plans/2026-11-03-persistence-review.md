# Persistence review implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Reviewers are read-only; the user requests autonomous implementation and larger related batches with one full/package checkpoint.

**Goal:** Integrate Claude's JSON-writer work with bounded temporary allocation, preserve persistence contracts, and resolve the observed unfolded-result rejection while reviewing complete touched modules.

**Architecture:** Introduce a small private JSON-writing policy shared by the seven existing writers. Plain bounded documents use whole-string encoding; large/unusual documents retain the original stream encoder. Callers retain ownership of paths, handles, newline conventions, validation and provenance. Adapt the independently supplied validation follow-up to current dev, and retain its scalar validator as the error oracle.

**Tech stack:** Python3.10–3.14, standard-library JSON, existing NumPy1.26–2.x package contract; no new dependency. Existing `core.outputs` default reservations stay in their actual callers.

**Spec:** The active user goal and [the integration plan](2026-11-02-claude-integration.md). Codex chooses the bounded helper based on measured memory, rather than importing patch03's seven unconditional whole-document allocations. This is an implementation choice under the user's autonomous optimization instruction.

## Constraints and evidence

- The isolated dev baseline is6692121; all seven exact-head CI jobs must succeed before applying the next production batch. At preparation time the minimum-NumPy/package jobs succeed and five Python jobs are running.
- Preserve public signatures, scientific conventions, schema fields, ordering, default-output reservations, allow_nan behavior, existing valid bytes, handle ownership and platform newline choices.
- Full scope includes the76 functions in manifests/path_stage/defect.analysis/surface.backend/interface.lattice_match/symmetry.kpath_stage/symmetry.symmetry, the entire results module if changed, the new helper and benchmark functions. Re-review unchanged functions when module hashes change.
- Root source/patches/run data remain preserved. Original30 result hashes were revalidated with six follow-up overlays; `MANIFEST-followups.tsv` bases match the original exported results. Patch04's signed formula agrees with the already integrated dev version; retain dev's matched-sign policy.
- Representative real-record encoding probe:4.36MB defect report, streaming .175MiB temporary allocation versus8.437MiB whole-string encoding; whole-string throughput improves substantially. A bounded prototype improves smaller report encoding but provides no useful tiny-manifest gain; large-file timing varies. Report measurements honestly and use the final implementation benchmark as evidence.

## Review focus

- Astral Unicode, control characters, quotes, non-string primitive keys and nested closing indentation must not evade the size estimate or change bytes.
- Cycles, custom types, nonfinite values and very large integers must retain the original stream encoder's verdict/error behavior, including failed-output prefixes where practical.
- Deep and high-cardinality payloads must select streaming without a full recursive traversal; the estimate is not a universal RSS guarantee.
- Unfolded search candidates are physical angles within180degrees, while angle_period_deg remains the actual symmetry period. Folded candidates retain the current narrower limit.
- Defect record optimization must preserve source-species-first ordering, matching position/flag permutation, metadata ownership and optional lattice behavior.

## Task1: bounded JSON policy and writer integration

- [x] Create `src/cellstine/core/jsonio.py`: `_small_json_budget(value, remaining=1<<20, depth=0)` and `write_indented_json(handle, value, *, allow_nan=True)`; fixed existing indent2, caller-owned handle.
- [x] Restrict fast eligibility to exact plain types; depth>32 or container cardinality>1024 streams. Use12*len(str)+2 for worst ensure_ascii escaping, conservative opening/closing indentation and separators, quoted non-string key costs. Restrict fast integer bit length to1024; nonfinite floats stream. Catch preflight RecursionError and use the baseline encoder.
- [x] Whole encoding is permitted only for a nonnegative budget. Other objects go directly through `json.dump(..., indent=2, allow_nan=...)`. Do not claim peak allocation equals the output budget.
- [x] Modify only the seven writer expressions/imports from03 by hand. Preserve caller `json.load` uses and all output-reservation/provenance code.
- [x] Add `tests/test_json_writer_resources.py` with actual writer byte comparisons, native/LF/trailing-newline behavior, escaping, primitive keys, cyclic/custom/nonfinite/huge-integer errors, handle ownership and a meaningful large-report temporary-allocation regression.
- [x] Benchmark final helper against6692121 with representative small/large records; compare exact bytes and memory, report platform/version and I/O variability.

## Task2: persistence validation follow-up and unfolded correctness

- [x] Import only patch04's `_plain_numbers`, `_plain_matrix`, `_plain_integer_matrix`, `_plainly_valid_candidate` and scalar-fallback wiring after checking their base/result hashes. Preserve dev schema and scalar oracle; retain existing signed orbit code and int64 wrapped-zero validation fix.
- [x] Add a failing persistence regression for a valid fold_symmetry=False search whose raw angle exceeds half the physical symmetry period.
- [x] Make scalar and array validators use180degrees for unfolded candidates, retaining the existing half-period rule for folded candidates. Keep the stored angle_period_deg field unchanged. Make the optional plain-candidate predicate coherent with those rules.
- [x] Compare verdicts, exception classes/text and input ownership against the full scalar validator for ordinary parsed records, malformed rows, missing fields, NaN/inf, bools, tuples, subclasses, huge ints, singular matrices and exact tolerance edges.
- [x] Update `benchmarks/benchmark_moire_fast_paths.py` to distinguish the intentional unfolded validation fix from accidental compatibility changes: compare all scientific arrays exactly and preserve every other validation check in a scoped baseline oracle.

## Task3: defect record resource reuse

- [x] Inspect `_record_from_atoms` and its generation callers; capture exact baseline outputs and resource measurements first.
- [x] Replace repeated species scans and per-row NumPy object construction with one grouping/permutation and a single position gather if measurements justify it.
- [x] Add meaningful position/flag/species/ownership regressions for missing/new/interleaved species, empty arrays, metadata and optional lattice. Preserve all public caller behavior and accepted string-label conventions.

## Task4: whole-function review and final gate

- [x] Inspect complete touched functions/callers and classify each with a source hash, individual reason and evidence. Obtain independent read-only review; resolve findings before freezing.
- [x] Run one affected test selection, then freeze all tracked Python bytes/file set and run one full branch-coverage checkpoint.
- [x] Build wheel/sdist, check Twine and exact archive modules/support, and smoke isolated core/CLI/sdist installs. Refresh only documentation in the final sdist after ledger/report updates.
- [ ] Commit the verified related persistence batch and push dev; inspect all seven exact-head CI jobs and retained coverage XML. Do not publish packages/releases.
- [ ] Continue with CLI01 and follow-up05 after this checkpoint; remaining function review stays active.


Local checkpoint:273focused/32independent/3406full tests pass;265amended hashes match with263original Python files unchanged and2UTF8benchmark-support corrections verified separately.142module and15finalsupport bytes match archives; installed core/CLI/sdist smokes pass. Full coverage13061/14927statements3505/4778branches. Scoped commit/push and CI remain.
