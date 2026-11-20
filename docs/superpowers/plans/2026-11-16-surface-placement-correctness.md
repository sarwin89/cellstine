# Surface geometry and placement implementation plan

> **For agentic workers:** Use superpowers:executing-plans for inline execution.
> One independent whole-batch source review precedes the frozen verification gate.

**Goal:** Resolve actual tilted-cell site displacements and false physical hollow
labels, then carry correct sites through molecule placement with stable API inputs
and less repeated species work.

**Architecture:** Keep Cartesian geometry and existing cell vectors authoritative.
Prepare physical planar UV in the site backend, reuse established triangular
stacking analysis before named hollows, and preserve the current ordered site
pipeline. Improve placement helpers in place; retain valid arithmetic, mass lookup,
species/atom order, flags, filenames and metadata schemas.

**Tech stack:** Python3.10–3.14, NumPy1.26–2.x, native POSCAR and stdlib only.

**Evidence/spec:** Part06g's F1/F2 witnesses in the surface-site report; read-only
design/projection/ten-supercell probes under the owned root `.tmp` directory.
Base dev `3337956`; all16CI43864731187 jobs and coverage artifact14225332908
closed before implementation. Shared main/Claude/run data remain untouched.

## Files and complete review scope

* `interface/surface/backend.py`: physical planar UV and actual packing gate;
  all9 current bodies (including lambda) reopened.
* `interface/surface/surface_sites.py`: clarify profile/planar contracts;
  all17 current bodies refreshed, including moved lambda accounting as needed.
* `adsorbate/placement/operations.py`: all20 bodies; stable first disjoint gap,
  COM/cardinality, finite rotations, axis parsing, species grouping/flag storage.
* `adsorbate/adsorbate.py`: all7 bodies; replace ambiguous NumPy truth-value
  checks for annotated sequence targets, Miller indices and supercell matrices.
* Functional assemble/place/move wrappers:3 bodies, inspect unchanged forwarding.
* Two new test files for physical geometry and placement resource/API behavior.
  Add both to minimum-NumPy and nine portable lanes before source freeze.

Expected original scope56 bodies plus new helpers; derive final inventory totals.

## Task 1: physical sites and naming

- [x] Add failing literal tilted-c and oblique-label regressions first.
- [x] Prepare planar UV from the actual cell/record, including w*c_parallel.
  Preserve the exact ordinary orthogonal-c path. Dual vectors avoid a squared
  Gram condition or SVD rank truncation; verify rotated and sheared frames and
  state numerical-zero handling explicitly without a new physical cutoff.
- [x] Keep count/spacing profile checks, but require established physical
  triangular exposed-layer certification before fcc/hcp labels. Reuse existing
  planar lattice/hollow helpers and their position tolerance; do not regroup
  the selected layer's floating normal residues. Keep the caller's exact layer
  tolerance and actual first/second subsurface depths. A buried
  registry fault must not remove correct local names; do not require a whole-
  slab ABC word or invent neighbor criteria.
- [x] Verify top/bottom faces, periodic images, translations, layer-dependent
  offsets, ordinary FCC/HCP/square/honeycomb/buckled cases and the ten exact
  triangular supercells. Test actual JSON and placed POSCAR coordinates, not
  merely the new helper's self-consistency. Keep cell/source bytes unchanged.

## Task 2: placement inputs and repeated work

- [x] Add regressions for valid sequence axes/NumPy targets and matrices,
  mismatched COM data, silent NaN rotations and real written workflow output.
- [x] Parse supported axis sequences and string separators in stable order;
  reject invalid tokens. Preserve zero-rotation arithmetic; add local finite
  angle validation (shared rotation helpers do not currently validate angles).
- [x] Validate COM shape/cardinality/finiteness without changing valid reduction
  order, species mass lookup or unknown-species behavior.
- [x] Replace repeated disjoint-species gap scans with an exact cumulative
  count sweep. Preserve current argsort, first disjoint gap, min_gap, ties,
  shared-species rejection and largest-gap fallback semantics.
- [x] Use stable atom-index grouping for combined species and avoid unused
  default flag lists. Preserve coordinate conversions/reframing, zero headers,
  mixed flags and molecule indices. Leave multi-image unwrapping, vacuum fitting
  and height/rotation conventions unchanged.
- [x] Replace only ambiguous optional-sequence truth checks in the public wrapper
  with explicit None handling; retain actual filenames/manifest field order.
  Normalize NumPy manifest scalars while preserving built-in scalar values;
  exercise float32 as well as float64 targets and integer matrices.

## Task 3: evidence and integration

- [x] Measure predecessor/current small and large selection/grouping workloads,
  plus cost of physical packing validation. Exact comparison only where the
  predecessor is correct; corrected geometry/inputs require independent oracles.
- [x] One independent complete source review for the whole inventory-derived
  scope; resolve blockers and explicitly account for any retained limitations.
- [ ] Freeze all Python files; one affected/full coverage/build/install batch,
  final support-only sdist/readback, ledger promotion and scoped dev commit/push.
- [ ] Close all16 exact-head supported-version CI jobs and downloaded coverage
  before another runtime batch. Resolve F1/F2's four investigated ledger rows
  only from physical readbacks and source reasoning. Goal remains incomplete
  while other functions or required work remain. No release, tag or upload.

Local gates are recorded in Part06h; final support readback/dev integration and exact-head CI closure follow without another runtime edit.
