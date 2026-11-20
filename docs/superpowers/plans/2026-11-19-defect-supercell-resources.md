# Defect supercell allocation implementation plan

Execute inline using the existing writing/executing-plans workflow, followed
by one independent complete-batch source review. Autonomous dev integration is
authorized; no release, tag, main merge or package publication.

Base e01be6f63e8d5b66e85ed289f145a6956915b706 passed all16 CI44513902716
jobs. Coverage artifact14525346664 SHA256
83967cb5cb0ffa18bba06e1a5000e023678526e69bbb15922dfdcaea5c02389f
was verified before this batch:14532/16158 statements,4198/5368 branches;
only the known prompts.py501–502 difference from local coverage.

Scope: all28 supercell.py bodies and all6 unchanged reporting.py bodies.
Prepared source/hash notes and scratch measurements are under shared .tmp/
part08b-*. Refresh those hashes; preparation does not promote ledger rows.

- [x] Add meaningful tests in tests/test_defect_supercell_resources.py:
  independent ordered HNF parameter values/counts, allocation budgets, bounded
  real host-vector blocks, scalar periodic-vector references, empty/negative
  radius behavior, cutoff/tie ordering, real selected/written supercells and
  report presentation limits. Watch allocation/mesh regressions fail first.
- [x] Preallocate exact HNF2D/3D parameter output arrays; fill existing divisor
  blocks with broadcast views rather than meshgrids and concatenated copies.
  Preserve all row/column order, int64 domain and public enumeration behavior.
- [x] Reuse reviewed core._integer_grid._integer_box_blocks to bound temporary
  host search grids. Preserve spacing/reach arithmetic, lexicographic generation,
  zero exclusion, norm/cutoff1e-9 and default argsort ordering. Materialized
  accepted vectors/norms and sorting still require proportional storage.
  Reject nonfinite/int64-unrepresentable floating reach before casting, so a
  coordinate overflow cannot be confused with an ordinary negative-radius
  empty search. Four failing edge regressions require a renewed final freeze.
- [x] Do not change maximum-distance selection, roundness shortlist32, tie
  priorities, user cell limits, integer membership or scientific tolerances.
  Explicitly record secondary roundness's existing heuristic limitation rather
  than claim a global guarantee. No new whole-domain numerical certificate.
- [x] Compare actual predecessor/current arrays, norms, selected matrices and
  report strings; run quiet warm interleaved timing/separate allocation cases.
  Small-case and output-memory costs must remain visible.
- [x] One independent complete34-body review plus callers/tests; resolve real
  findings before freezing. Run affected/full coverage/build/Twine/three
  installed workflows once; promote individual ledger rows only after gates.
- [ ] Final support-only sdist/raw artifact readback, scoped dev commit/push,
  all16 exact-head CI plus downloaded coverage before further runtime changes.
  Preserve all shared main/Claude edits and scientific run data.

Corrected local gates pass. Final support/archive/dev integration and exact-head CI remain the final item; no further runtime edits precede closure.
