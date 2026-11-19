# Defect plane selection and generation integrity

Implemented against dev `5fd5be33f892004b7d6934029adf1a1eb6cf943b`. All16
permissions-fix CI44479741997 jobs passed before runtime edits. Downloaded
coverage artifact14507809942 has verified SHA256
`01155cfa9e6eb62b52b865cef9d537c29a05a5b470f157269651ba015c9ed4a6`
and exactly matches the prior runtime checkpoint:14451/16081 statements,
4138/5310 branches. The workflow's contents:read token still uploads coverage.
GitHub's two default-branch alerts remain a separate main integration/rescan
matter; neither alert was dismissed and no main merge or publication occurred.

## Selection resources and compatibility

The internal parser retains compact inclusive ranges until actual ordinal
bounds are known. The resolver rejects the first unavailable/zero ordinal
without enumerating a huge request. Public parse_layer_selection still returns
its full tuple, with existing token conversion, keyword aliases, empty behavior,
negative/reversed ranges and malformed-token error priority. Resolver output
remains sorted/distinct by the existing ordinal mapping, including duplicate
and noncontiguous census IDs. No selection cap or silent truncation is added.

Production/predecessor comparisons cover5995 exact resolver outcomes and1199
public-parser outcomes, including exact error types/messages. The corrected
allocation guard allows lazy range creation and forbids unavailable-range
iteration; it fails all four predecessor cases. Production rejects
1-1000000000000 for five available planes using1463 traced bytes. This concerns
resolver allocation, not a universal input/output/RSS bound or a new bound on
the public standalone parser's intentionally materialized tuple.

Member layer IDs use a set, and per-site members are indexed by plane once.
Site-major, requested-plane-major and member order remain, including repeated
plane requests, multi-plane pairs, representative indices, pair indices,
multiplicities and shallow dictionary-copy behavior. Existing helpers remain
available even when no longer used internally.

## Saved analysis and physical corrections

Public saved-input probes established these predecessor errors:

* Missing orbit members caused a known selected atom plane to disappear; all
  mode wrote one structure for multiplicity2 while also claiming full expansion.
* Vacancy index0 wrote an unchanged host; duplicate divacancy indices removed
  only one atom. Substitution/other out-of-range index cases were unguarded.
* A negative-c slab placed its top adatom towards the host rather than along+c.
  Rotated non-preserving bounds used global z and accepted an over-height atom.
* Mixed-case antisite input reached normalized scientific dispatch but missed
  automatic replacement discovery and raised a missing-species error.

Known legacy single representatives are retained without inventing an orbit.
All mode preserves their original site identity/ID; plane selection can use a
known singleton plane. Unknown multi-member geometry requires actionable
reanalysis. Missing divacancy plane metadata cannot be inferred from a scalar
layer ID, even for one pair; a pair can touch two planes. Single-pair all or
unfiltered generation remains usable when plane membership is not requested.
Intentional unplaced-adatom handling remains. Full-expansion notes require
actual member expansion. Unfiltered inequivalent legacy generation is retained.

Explicit complete selections must not be blocked by unrelated missing metadata.
A prefilter skips only provably irrelevant legacy sites; normal complete sites
retain the original post-expansion ID filter. Exact known base/layer/member and
numeric matches remain. Missing/short atom equivalence lists leave unowned
numeric requests ambiguous and require reanalysis, rather than partial output.
Indices explicitly assigned by existing atom member records belong to disjoint
producer atom orbits and can be excluded from that ambiguity. This does not
infer ownership from scalar representatives or pair/void records.

Atom edits now require valid 1-based indices and distinct first-two pair atoms.
Adatoms retain the existing normal calculation/area threshold, orient it towards
c, and use normal projections for non-preserving bounds. The existing policy
is preserved in rotationally invariant quantities: top <= min(host@n)+abs(c@n)
+1e-6; bottom >= max(host@n)-abs(c@n)-1e-6. This is the existing occupied-slab
relative capacity policy, not forced fractional[0,1) wrapping. Four boundary
tests pin the unchanged slack. Vacuum fitting, output coordinate/flag/species
ordering and normal positive-c arithmetic are retained. Only automatic antisite
discovery's comparison is normalized; caller-spelled filenames remain.

## Independent review and measurements

All24 complete bodies in layers.py, generation.py, records.py and dilution.py
received individual review:21 predecessors and3 helpers. Review found actual
selection/incomplete-metadata issues R1/R2/R3; each gained failing regressions
before correction. Final source/test hashes are refreshed and no unresolved
finding remains. The final affected batch passed202 tests with40 dependency
warnings in24.30s;60 new tests include written POSCAR/JSON/manifest-facing
behavior, actual coordinates/flags and immutable source bytes. Both new files
enter minimum-NumPy and nine portable CI lanes.

Eleven quiet warm interleaved predecessor/current cases compare exact accepted
outputs, including canonical generated coordinates/flags. Five timing samples
and separate tracemalloc probes use native environment1 before NumPy, with no
concurrent runtime probes/tests/builds. Traced allocations are not RSS.

| Case | Previous ms | Current ms | Previous traced KiB | Current traced KiB |
| --- | ---: | ---: | ---: | ---: |
| resolve, 5 planes, 1 tokens | 0.0035 | 0.0040 | 0.79 | 1.15 |
| resolve, 2000 planes, 1 tokens | 17.7993 | 0.7505 | 149.26 | 192.29 |
| resolve, 1000 planes, 10 tokens | 43.6819 | 2.7859 | 399.20 | 66.07 |
| member-layer-ids, 2 planes, 8 members | 0.0022 | 0.0023 | 0.38 | 0.55 |
| sites-by-layer, 2 planes, 8 members | 0.0262 | 0.0229 | 5.49 | 5.73 |
| sites-by-member, 2 planes, 8 members | 0.0934 | 0.0949 | 8.82 | 8.82 |
| member-layer-ids, 50 planes, 4000 members | 1.6245 | 0.7990 | 1.51 | 3.27 |
| sites-by-layer, 50 planes, 4000 members | 69.5373 | 4.5138 | 819.22 | 858.89 |
| sites-by-member, 50 planes, 4000 members | 31.8813 | 34.1740 | 3154.87 | 3154.87 |
| generate-one-vacancy, 4000 atoms | 2.1335 | 2.1108 | 901.42 | 901.40 |
| generate-one-adatom, 2 atoms | 0.1302 | 0.1370 | 8.61 | 8.61 |

The 2,000-plane set uses about44KiB more traced allocation while reducing
selection CPU substantially. The 4,000-member plane index uses about40KiB more
than repeated scans. All-member expansion costs about2.29ms more in that case,
with unchanged traced peak; the two-atom adatom check costs about0.007ms more.
Tiny-case overheads and these tradeoffs remain explicit. No universal speedup
or globally bounded output/worker-memory claim follows from these samples.
Source-hashed raw artifacts remain under ignored shared .tmp/part08a-* paths.

## Retained boundaries and gates

Saved scientific metadata is trusted to describe disjoint atom orbits. This
pass does not validate host-file freshness, repair arbitrary corrupt/partial
nonempty member records, or infer missing orbit physics. Existing extreme
floating/integer domains, 1e-12 normal area threshold, dilution heuristics,
explicit-output failure ownership and reservation behavior remain. No new
format, dependency, public parameter, parallel mode or scientific cutoff.

## Local checkpoint

The frozen full run passed **3994 tests**, skipped 4,
with 81 dependency warnings in 731.62s. All **294
Python hashes** remain unchanged. Local coverage is **14530/16158
statements** and **4198/5368 branches**.
Verification sets native thread environment1 before NumPy; runtime defaults are
unchanged and suite-time differences are not attributed without controlled data.

Wheel/sdist builds, Twine and three installed core/CLI/sdist environments pass
real saved-input selection, hidden-numeric rejection without partial output,
invalid edit rejection and negative-c physical adatom readback, dependency
checks, and previous native serial/process/thread workflows. Final report and
ledger edits refresh only the sdist; no unchanged full/runtime/install repetition.
Raw runtime/support archive readback precedes scoped dev integration. All16
exact-head CI jobs and downloaded coverage gate the next runtime batch.

Ledger: **1236 functions**, **1130 with inspection
evidence**, **776 reviewed unchanged**,
**341 optimized and verified**,
**13 investigated deferrals** and **106 pending**.
This batch reviews24 bodies, adds23 inspection records (20 prior pending bodies
and3 helpers), and reassesses one existing reviewed body. Hashes scope evidence.
The remaining pending work is50 defect bodies,23 visualization,12 CLI and21
benchmark helpers; the13 prior investigated limitations remain separate.

No release readiness, publication, tag, main merge or package upload is claimed.
The overall goal remains active and incomplete.
