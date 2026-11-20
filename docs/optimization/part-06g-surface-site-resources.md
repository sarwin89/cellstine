# Surface-site memory and exact ordered deduplication

Work in progress against dev `8b04d03`. The previous checkpoint passed all 16
CI 43857657826 jobs; coverage artifact 14221503471 SHA256
485585daeabac39b0eaa879444bb754301851db2da7a2ee9b0ce11f6768501a2
verified head/job/digest/module/statement sets. CI 14330/15977 statements and
4052/5242 branches differ locally only by known prompts.py:501–502 fallbacks,
with no branch delta. Closure preceded this batch's runtime edits.

## Changes

The rounded duplicate prefilter can discard a valid first-surviving site. In a
unit cell at tolerance 1e-4, ordered x positions 0, 1e-4-2e-10, 1e-4+2e-10 should
keep the first and last. Rounding collapses the latter pair, then the retained
bucket representative is itself rejected against 0. Only exact duplicates are
now removed before the original ordered minimum-image distance pass. Accepted
coordinate storage is preallocated; returned points remain independent of input.

Distinct-atom pair displacements are streamed in upper-triangle order, retaining
the original small-array path. Site coordination and bridges use at most 65,536
anchor/image pairs in each distance matrix. Complete image rows stay together
when they fit; larger rows split their images one anchor at a time. This retains
global anchor-major/image-major order, >1e-8 exclusions, cutoff+1e-12 and original
elementwise Cartesian/norm arithmetic. No neighbors or images are truncated.
The dense matrix helper's API remains available.

The pair budget does not bound expanded periodic-image arrays, all accepted
midpoints or final output size. Nearest-neighbor work remains quadratic in atom
count; this is a memory reduction, not a subquadratic-search claim.

## Current evidence

Initial six regressions gave 5 failures and 1 pass. The 18 new tests cover literal
losses, ownership/order, real-kernel allocation guards, scalar periodic counts
and ordered bridge references at 60/90/120 degrees, forced image/row blocks, exact
cutoff slack, duplicate/self/sheared images, empties and a real written
Surface.sites JSON/manifest preserving source bytes. The final affected surface,
adsorbate, framing, stacking and termination batch passed 105 tests in 5.46s.
The new file enters the minimum-NumPy and all nine portable lanes.

Five warm interleaved predecessor/current measurements use Python 3.14.3 /
NumPy 2.4.3 with native thread environment 1 before import; separate tracemalloc
probes measure allocations, not RSS. Ordinary accepted outputs compare exactly;
corrected losses use literal oracles. One tiny reviewer probe may overlap timing;
no broad tests/builds/installs were active. Do not infer universal hardware speed.

| Case | Previous ms | Current ms | Previous traced MiB | Current traced MiB |
| --- | ---: | ---: | ---: | ---: |
| top_layer_coordination_counts, 1 atoms | 0.225 | 0.236 | 0.009 | 0.009 |
| find_bridge_sites, 1 atoms | 0.836 | 0.823 | 0.015 | 0.015 |
| top_layer_coordination_counts, 100 atoms | 12.477 | 11.838 | 7.570 | 5.456 |
| find_bridge_sites, 100 atoms | 70.185 | 68.188 | 7.570 | 5.456 |
| top_layer_coordination_counts, 400 atoms | 182.110 | 164.617 | 120.908 | 6.056 |
| find_bridge_sites, 400 atoms | 431.061 | 431.878 | 120.908 | 6.080 |
| nearest_neighbor, 4 atoms | 0.406 | 0.346 | 0.011 | 0.012 |
| nearest_neighbor, 400 atoms | 19.198 | 17.091 | 11.934 | 9.496 |
| nearest_neighbor, 1600 atoms | 282.051 | 252.751 | 134.063 | 10.511 |
| deduplication, 400 candidates | 126.086 | 122.074 | 0.248 | 0.243 |
| deduplication, 2000 candidates | 135.815 | 128.624 | 0.476 | 0.475 |

## Independent review and explicit dispositions

All 25 current bodies were independently read (surface_sites 17, Surface 6 and
build/match wrappers 2). The unchanged actual backend caller was also read and
reassessed by the coordinator, for 26 individual dispositions. No introduced
correctness blocker was found. Two concrete preexisting P2 limitations are
retained as investigated follow-ups, rather than universal correctness claims:

* F1: equal layer counts and spacing do not prove triangular packing. An oblique
  2x3/70degree lattice with three one-atom layers is given an hcp hollow despite
  top coordination 2. `_is_close_packed_stacking` and its owning backend caller
  are deferred; `_classify_hollow` retains its truthful-input-flag contract.
* F2: raw first-two fractional coordinates omit the parallel component of a
  tilted third vector. For lattice[[2,0,0],[0,2,0],[1,0,12]], direct[.2,.3,.6]
  is Cartesian[1,.6,7.2], but the top site is reported near[.4,.6,7.2]. Planar
  extraction, reconstruction and the owning backend caller are deferred. This
  affects accepted input, not merely hypothetical invalid data.

Practical costs are misleading physical hollow labels and displaced adsorption
positions/subsurface registry. These are priority owning-backend correctness
follow-ups before publication or final goal closure. The resource changes do not
resolve or introduce them. Numerical/malformed-domain hardening is not newly
certified, and the peer source review does not certify coordinator test/package
gates.

## Local checkpoint

One frozen full run passed **3873 tests**, skipped 4,
with 81 warnings in 925.00s. All 290 Python hashes
remain fixed. Coverage is **14368/16012
statements** and **4083/5268
branches**. Native thread environment variables were set to 1 before NumPy for
local verification only; runtime defaults remain unchanged.

Wheel/sdist builds and Twine pass. All three installed core/CLI/sdist environments
pass actual surface JSON/manifest cases, literal dedup witnesses, block-size
checks and prior interface/registry/native serial/process/thread workflows, with
dependency checks. Final support/ledger edits refresh only the sdist; unchanged
runtime/full/install checks are not repeated. Raw archive readback and scoped
dev push precede exact-head 16-job CI and downloaded coverage closure.

Ledger: **1230 functions**, **1084
with inspection evidence**, 747 reviewed unchanged,
320 optimized and verified,
17 investigated deferrals and
146 pending reviews. This batch inspects 25 complete bodies,
adds 23 inspection records and reassesses one previously reviewed caller. Four
rows carry explicit F1/F2 investigated dispositions; those actual scientific
issues require the priority owning-backend follow-up before final goal closure.

No release, tag or package upload is authorized; the overall goal is incomplete.

## Exact-head CI closure

Dev `3337956cfdb1c59d21e000a34bdf5864c967d0c6` passed all 16 jobs in
CI 43864731187. Downloaded coverage artifact 14225332908 has verified SHA256
`439e9e9d1caf60805514dcd1fc486f88f163e7733f36fd5fbe721285131b8fbf`,
matching head/job/module/statement sets. CI covered 14370/16012 statements and
4083/5268 branches; the only local difference was the known prompts.py:501–502
fallback, with no branch delta. Closure preceded Part06h runtime edits.
Part06h subsequently resolves this report's F1/F2 owning-backend follow-ups.
