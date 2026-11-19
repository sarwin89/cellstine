# Defect pair allocation and workflow provenance implementation plan

Execute inline with writing/executing-plans and regression-first checks, then
one independent complete-batch source review. Base cda1bc5 passed all16 jobs
of CI44522433664; artifact14529048829 digest
bacb7f68e1bd72405a44d5f02aca5b910fe030911089af81a2880082174f4407 verified.

Goal: inspect the remaining8 sites.py and8 workflow.py bodies, reduce temporary
pair enumeration storage, and keep preview summaries and requested parameters
accurate. Re-review generation.py's8 bodies when adding its provenance field.

Architecture: use the existing exact periodic-distance kernel in bounded row
strips against the remaining upper triangle. Materialize only accepted pairs,
in the original lexicographic order. Keep orbit grouping, cutoffs, tolerances,
midpoints, species, member layers and public APIs unchanged. Dense accepted
pair outputs can still be quadratic. No spatial approximation or new cap.

- [x] Add real no-pair allocation and multiblock ordering/cutoff regressions to
  tests/test_defect_pair_resources.py; watch allocation checks fail first.
- [x] Add preview limit and analyse/preview/generate saddle-request manifest
  regressions to tests/test_defect_workflow_provenance.py; observe failures.
- [x] Implement private ordered pair helper in sites.py using reviewed core
  geometry, with65536-distance target and one linear row for wider inputs.
  Test duplicate positions, skew cells, exact boundaries, infinities/NaN,
  empty/singleton inputs, actual symmetry orbits and member metadata.
- [x] Clamp shown_sites consistently with the report. Add native bool requested
  interstitial_saddles to existing extensible manifest parameter dictionaries;
  also record the requested preserve_vacuum policy in generation manifests.
  saved-analysis reuse remains unchanged and the field describes the request.
- [x] Resolve independent review's preexisting wrong-atom manual divacancy:
  exact displayed atom site IDs win over numeric aliases; retain plain-number
  and unmatched atom_N aliases. Test the actual written surviving atoms.
- [x] Compare predecessor/current ordered pair values and full site records,
  benchmark quiet interleaved timings and separate allocations, report costs.
- [x] Complete individual source/caller review, one independent batch review,
  resolve findings, freeze Python hashes; focused/full/package/Twine and three
  installed workflows before scoped commit/push to dev.
- [ ] Update hashed ledger/report, refresh final support sdist and raw archive
  comparison; require16 exact-head CI jobs and downloaded coverage closure
  before further runtime changes. Overall goal remains active.

Preserve shared main/Claude edits and scientific data. No main merge, release,
tag, package publication or change to adatom expansion policy.

Local gates pass on297-Python freeze; final support/readback/dev integration and exact-head CI remain. Overall goal active.
