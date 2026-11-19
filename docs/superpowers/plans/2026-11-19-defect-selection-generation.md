# Defect selection and generation implementation plan

> Execute inline with superpowers:executing-plans; one independent whole-batch
> source review after implementation. User authorizes autonomous scoped dev
> integration; no routine confirmation or publication.

Base dev `5fd5be33f892004b7d6934029adf1a1eb6cf943b` passed all16 CI44479741997.
Coverage artifact14507809942 SHA256
`01155cfa9e6eb62b52b865cef9d537c29a05a5b470f157269651ba015c9ed4a6`
matches the prior runtime coverage exactly. No source work precedes that closure.

## Scope and invariants

Change `src/cellstine/defect/layers.py` and `generation.py` only where evidence
supports it. Review every body there plus unchanged records.py/dilution.py and
relevant producers/saved-input/workflow callers. Inventory the final scope,
including new helpers; refresh prepared14+7 body notes against current source.
Preserve public signatures, JSON schema, valid ordering, physical tolerances,
flags, native output, IDs for complete member metadata, and source ownership.
Preserve the public standalone parser's full-tuple API. No orbit-size caps,
guessed unknown members, automatic reanalysis with different symmetry settings,
new concurrency mode, or scientific run edits.

## Task 1: meaningful regression evidence

- [x] Add `tests/test_defect_selection_resources.py` for compact rejected ranges,
  ordinal/negative/reversed selection, duplicate/noncontiguous plane IDs,
  site/requested-plane/member ordering, multi-plane membership, duplicate plane
  requests and ownership. Allocation guard must catch expansion of unavailable
  ranges without allocating a huge predecessor object.
- [x] Add `tests/test_defect_generation_integrity.py` with real saved JSON/POSCAR
  readbacks: missing multi-member orbit cannot silently drop or claim complete
  expansion; known single representatives can be retained accurately; genuine
  unplaced adatoms retain their intended behavior. Preserve unfiltered
  inequivalent generation for older analyses.
- [x] Demonstrate invalid 1-based vacancy/substitution/pair indices cannot create
  no-op or wrong-cardinality structures. Preserve first-two pair semantics but
  require valid distinct selected atoms.
- [x] Test top/bottom physical adatom height in rotated, tilted and negative-c
  cells, with and without vacuum preservation. Non-preserving bounds use normal
  projections and c's normal height with the existing1e-6 tolerance. Preserve the
  original normal's area-validation domain and ordinary arithmetic before
  orienting it towards c.

Scratch public witnesses: `.tmp/part08a-saved-orbit-probe.json` and
`part08a-generation-domain-probe.json` in the shared workspace. Scratch compact
prototype matches5995 small current outcomes, including exact errors, and rejects
the trillion-entry invalid range with1407 traced bytes; this is preparation,
not production verification or ledger promotion.

## Task 2: scoped implementation

- [x] Parse compact range parts internally, fully converting tokens before
  resolver bounds checks. Validate the first unavailable/zero ordinal before
  expansion. Keep standalone parser output, keyword/empty/error conventions and
  sorted distinct resolver outputs.
- [x] Collect member layer IDs and index members by requested layer once.
  Preserve site-major/requested-plane-major/member order, including repeated
  requests, multi-plane members and shallow-copy boundaries.
- [x] Handle missing metadata explicitly: restore only a known single
  representative; require reanalysis for unknown multi-member expansion. Keep
  intentional unplaced-adatom handling. Emit the complete-expansion note only
  when member expansion actually occurred. Do not infer unknown orbit geometry.
- [x] Validate atom bounds/distinct pairs in `_generate_one` before edits.
  Orient its existing normal towards c and replace global-z non-preserving
  adatom bounds with corresponding normal projections. Keep finite thresholds,
  vacuum fitting, coordinates/flags/species ordering and output conventions.
- [x] Inspect generation/record/dilution bodies completely. Leave sound code
  unchanged; record retained domains instead of claiming universal correctness.

## Task 3: batch evidence and integration

- [x] Measure quiet interleaved predecessor/current small/large layer selection
  and orbit expansion cases, separate allocations, exact accepted outputs.
  Correctness fixes use independent written physical/ownership oracles.
- [x] One independent source review over every final scoped body and tests;
  resolve findings and refresh exact hashes before freezing.
- [x] Add relevant portable/minimum-NumPy CI cases; freeze Python source, run
  affected/full coverage/build/Twine/three installed workflows once. Promote
  individual ledger rows only after observed gates.
- [ ] Finish report/counts, support-only sdist/archive readback, scoped dev
  commit/push, all16 exact-head CI and downloaded coverage. Preserve main/Claude
  edits and scientific data. Goal stays active while other work remains.

Local frozen gates pass. Final support/archive/dev integration and exact-head CI remain the final checklist item; no further runtime edits precede it.
