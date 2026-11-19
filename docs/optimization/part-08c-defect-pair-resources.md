# Defect pair resources, physical selection and workflow metadata

Base: dev cda1bc5a588e754513a0dcf2e1f34fbad042c4ca. Its16 jobs in
CI44522433664 passed; artifact14529048829 SHA256
bacb7f68e1bd72405a44d5f02aca5b910fe030911089af81a2880082174f4407 was
downloaded and verified against the exact head and coverage job. Coverage was
14549/16175 statements,4208/5378 branches; only prompts.py501–502 differed
from local coverage, with no branch delta.

## Complete review scope

All9 bodies in defect/sites.py,8 in workflow.py and8 in generation.py were
reviewed, including nested pair-orbit find/union. This closes16 previously
pending defect bodies, re-reviews8 generation bodies after source changes,
and introduces one private helper. Individual raw/LF-normalized module hashes,
AST body hashes, baseline body IDs and rationales are recorded by the independent
reviewer in the ignored part08c-independent-review.json evidence. Canonical
ledger rows are promoted only after local source/package gates pass.

Callers and dependencies include native/direct-spglib atom orbit producers,
layer/member metadata, periodic void/surface sites, geometry's exact image
kernel, PeriodicSiteIndex, generating_operations, defect selection/editing,
saved-analysis loading, manifests, CLI --site-ids and interactive displayed IDs.
Atom/site numbering, pair midpoint geometry and grouping remain explicit.

## Changes and contracts

The old divacancy discovery allocated an N-by-N distance matrix plus full
upper-triangle index and distance arrays. `_divacancy_pairs` uses the same
exact periodic-distance kernel on ascending row strips against remaining
columns, discarding each strip's lower triangle before threshold selection.
The target is65536 distances per strip; hosts wider than65536 require one
linear row. No accepted-pair cap or spatial approximation is introduced.
All pairs retain i<j lexicographic ordering, native float values and inclusive
cutoff. Species/distance fallback grouping, real symmetry components, site IDs,
representative pairs, members, layer IDs and physical midpoints are unchanged.
Accepted-pair lists/orbit metadata can still require quadratic output storage.

Preview's shown_sites now clamps a negative limit to zero, consistently with
its text formatter; total_sites and the actual requested limit are retained.
Analyse/preview/generate manifests add a native boolean interstitial_saddles
request. Generate also records preserve_vacuum. These are additions to existing
extensible parameter dictionaries, not a claim that a reused saved analysis was
re-enumerated with the new request. Saved-analysis reuse policy is unchanged.

Independent review found a preexisting wrong-atom edit: `atom_002`, a displayed
orbit/site ID, was parsed as physical atom2 even when its representative was
atom3. Manual two-site divacancy could remove the wrong atom. Exact existing
atom-site IDs now resolve before the atom_N numeric-alias fallback. Bare
numbers remain physical indices, and unmatched atom_N aliases retain their
previous interpretation. The CLI calls these site IDs and displays those
same identifiers, so exact site identity takes precedence when ambiguous.
The actual written surviving coordinates and flags are checked. This scientific
correctness correction belongs in future candidate release notes.

No other layer/adatom expansion policy, tolerance, backend selection, public
signature, file encoding or cell-selection behavior changes. Remaining numeric
conditioning and metadata-trust limitations documented in earlier parts persist.

## Regression and predecessor evidence

Initial regressions:12 failed/12 passed for allocation, bounded real kernel
calls, preview count and missing saddle provenance. Vacuum-policy serialization
then failed3 cases; the review finding failed1 actual survivor case while2
numeric-alias cases passed. All failures were observed before their fixes.

The final affected batch passes193 tests with38 spglib deprecation warnings in
21.37s. The44 new cases cover real1000-atom no-pair allocations, real strip work
volume, duplicate positions, empty/singleton hosts, negative/zero/finite/infinite/
NaN cutoffs, skew images, nextafter boundaries, multiblock float.hex pair values,
negative/zero/positive previews, requested native/NumPy booleans and actual
written survivor coordinates. Existing full-group orbit, layers, physical
ferrite saddles, editing integrity, path and record tests cover downstream use.
Both files run in minimum NumPy and9 portable OS/Python lanes.

Actual predecessor pair-discovery AST statements were compiled separately:
108 ordered pair-array comparisons and12 complete site-record comparisons pass,
including skew cells, n=0/1/2/7/257/513, cutoff edge classes, fallback and native
symmetry. The reviewer independently checked20 small exact-pair cases and all25
complete source bodies; its wrong-atom finding was fixed and the final snapshot
re-reviewed. No unresolved review finding remains in this batch.

## Measured allocation and time

Python3.14.3/NumPy2.4.3, native thread environment1, seven warm interleaved
medians. Traced peak allocation is measured separately and is not process RSS.
The final benchmark ran without another local verification runtime. Env1 is a
verification setting; runtime concurrency defaults remain unchanged.

| Case | Atoms | Accepted pairs | Old ms | New ms | Old traced bytes | New traced bytes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Sparse grid |27|0|0.8012|0.9044|367089|350995|
| Sparse grid |216|0|31.0231|30.7350|9353184|9301281|
| Sparse grid |1000|0|654.8391|347.8797|39817758|11505501|
| Finite sparse cutoff |300|210|58.1805|46.2921|12947143|11481192|
| Dense accepted output |300|44850|93.8439|84.7414|12947192|11481239|
| Small host |2|0|0.2679|0.2199|13473|13455|
| Small host |7|15|0.3139|0.2918|75432|60403|
| Small host |32|362|1.3514|1.6344|767297|715634|

The1000-atom discovery case reduces traced peak by71% and elapsed time by47%.
The27-atom and32-atom cases cost about0.10ms and0.28ms respectively. This is a
bounded-temporary resource improvement with measured workload costs, not a
universal speedup or a claim about complete workflow/full-suite elapsed time.

## Checkpoint gates

The frozen full suite passes **4072 tests**, with 4 skips and
81 dependency warnings in 1070.54s. All297 Python hashes
remain unchanged. Local coverage: **14593/16183
statements**, **4232/5384 branches**.
Wheel/sdist/Twine and installed core/CLI/sdist workflows pass actual multiblock
pair values, displayed-site survivor coordinates/flags, clamped preview and
requested boolean manifests, plus earlier native workflows/dependency checks.
Final support changes refresh only the sdist, followed by raw144-runtime and
8-support archive readback before scoped dev push and all16 exact-head CI jobs
with downloaded coverage verification. CI closure still gates further edits.

Ledger: **1237 functions**, **1181 with inspection evidence**,
**817 reviewed unchanged**, **350 optimized and verified**,
**14 investigated deferrals**, **56 pending**. This batch reviews25 complete
bodies, adding17 newly inspected bodies including1 new private helper.
Remaining pending: visualization23, CLI12 and benchmark helpers21. Final
requirement-level audit and unresolved-limitations report still gate completion.
No release readiness claim, main merge, tag or package publication.
