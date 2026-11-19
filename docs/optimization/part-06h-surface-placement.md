# Physical surface sites and placement resources

Implemented against dev `3337956`. That checkpoint closed all 16 jobs in
CI 43864731187 and coverage artifact 14225332908, SHA256
439e9e9d1caf60805514dcd1fc486f88f163e7733f36fd5fbe721285131b8fbf.
CI coverage14370/16012 statements and4083/5268 branches differs locally only by
known prompts.py:501–502 fallbacks, with no branch delta. Closure preceded edits.

## Physical corrections

The site backend now includes the third lattice vector's in-plane contribution.
A row-scaled signed dual respects the shared normal oriented towards c, without
an overflowing cross-product norm or an arbitrary new physical tilt cutoff.
Exactly zero coefficients retain the canonical orthogonal-c arithmetic path.
Rotated roundoff-scale coefficients remain representable; corrected outputs are
checked in Cartesian space modulo a/b, not compared with incorrect predecessors.
For lattice[[2,0,0],[0,2,0],[1,0,12]], direct[.2,.3,.6] now reports its actual
top position[1,.6,7.2], replacing[.4,.6,7.2]. Negative orientation, rotated frames,
top/bottom faces, periodic representatives and real placed atoms are covered.
Pure coordinate conversion at scales1e-100/1e100 is tested separately; it does not
certify the entire site workflow or every numerical scale.

Equal counts/heights remain the existing layer-profile condition. Before fcc/hcp
names, the selected exposed plane must also have the established triangular
point lattice, using shared planar primitives and POSITION_TOLERANCE. Measured
subsurface depths retain their existing meaning. An oblique2x3/70degree plane
with coordination2 now receives generic hollows. Valid local names survive a
buried registry fault; no global ABC-word constraint was introduced.

Independent review found an actual uncommitted regression in the initial
synthetic-plane approach: at exact zero layer tolerance, floating normal
residues split an already-selected rotated plane. The final implementation
directly certifies planar points and never regroups their heights. The exact
rotated2x2 witness first failed then passed. Refreshed source/test hashes and the
resolved finding, rather than the earlier no-findings snapshot, are authoritative.

## Placement behavior and resources

* COM input shape/cardinality/finiteness are checked without changing the valid
  mass lookup or reduction order. Undefined NaN rotations now fail explicitly;
  finite zero rotations retain their original arithmetic path.
* List/array axis sequences preserve order and deduplication. Supported string
  separators/case remain; invalid characters no longer silently disappear.
* The first acceptable disjoint-species gap is selected with a cumulative count
  sweep. Existing argsort, cutoff, first-gap priority and largest-gap fallback
  semantics remain, including ties and original atom indices.
* Combined atoms use a stable species-index map and emit mixed flags directly.
  Species/atom order, zero headers, coordinate conversions, reframing, immutable
  flag values and molecule indices remain. Fewer redundant scans/flag lists are
  required; the multi-image unwrapping and vacuum/height conventions are unchanged.
* Public NumPy Miller indices, supercell matrices and move targets no longer
  encounter ambiguous truth-value checks. Manifest scalars are normalized while
  preserving built-in values; real float32/64 and integer-array cases are tested.

Ten final warm interleaved predecessor/current timing cases, with separate
tracemalloc allocation probes, compare exact accepted arrays/fields. The frozen
baseline placement module is rewired to its baseline site backend; other shared
dependencies are unchanged/current. Native environment1 precedes NumPy. No other
runtime probes/tests/builds were active. Allocations are not RSS or universal
scalability claims. All ten cases are shown below. Traced allocation is not process RSS.

| Case | Previous ms | Current ms | Previous traced MiB | Current traced MiB |
| --- | ---: | ---: | ---: | ---: |
| species-gap, 8 atoms, disjoint | 0.033 | 0.031 | 0.003 | 0.003 |
| species-gap, 2000 atoms, disjoint | 245.256 | 0.909 | 0.169 | 0.106 |
| species-gap, 2000 atoms, shared species | 451.378 | 1.351 | 0.169 | 0.101 |
| species-grouping, 6 atoms, 2 headers/source, no flags | 0.036 | 0.034 | 0.007 | 0.007 |
| species-grouping, 20000 atoms, 50 headers/source, no flags | 80.006 | 6.292 | 4.380 | 4.231 |
| species-grouping, 20000 atoms, 50 headers/source, mixed flags | 81.827 | 8.129 | 4.549 | 4.396 |
| species-grouping, 20000 atoms, 2 headers/source, mixed flags | 8.002 | 8.219 | 4.556 | 4.367 |
| complete-site-analysis, 1 atoms/layer | 7.543 | 8.173 | 0.274 | 0.274 |
| complete-site-analysis, 16 atoms/layer | 72.419 | 64.739 | 3.132 | 3.132 |
| complete-site-analysis, 100 atoms/layer | 425.622 | 425.911 | 19.413 | 19.413 |

The 20,000-atom/two-header mixed-flags case costs about 0.22ms more CPU;
the one-atom/layer site analysis costs about 0.63ms for physical packing
validation. The 100-atom/layer site case is effectively unchanged in this
measurement. Larger multi-header grouping and repeated gap selection improve
substantially; these measurements do not promise universal speedups.

## Current evidence and gates

The61 new tests cover actual physical records/output, ten exact integer
triangular supercells, scalar gap references, metadata and ownership. Initial
geometry regressions gave8 failures/7 passes. Placement gave20 failures/6passes,
with separate failing Miller/float32-manifest cases. Orientation/range checks
gave4 failures. Buried-fault and zero-tolerance compatibility cases each failed
before their corrections. The final affected batch passed137 tests in4.40s.
Both new files enter minimum-NumPy and all nine portable CI lanes.

All59 bodies in seven modules received independent complete source review,
including59 unique IDs, exact hashes and the moved lambda mapping. No unresolved
review finding remains. The four prior F1/F2 owning-body dispositions are resolved by the physical
projection/packing corrections, source reasoning and passing full/package gates.
Doc-only helper contracts retain reviewed-unchanged dispositions; callers own
the behavioral corrections. The ledger records every body separately.

Retained limits: existing primitive-discovery/numerical domains; no universal
decorated/relaxed packing theorem; existing multi-image and output-failure
ownership behavior; unchanged expanded-image/output memory bounds. Invalid axis,
rotation and COM data now raise errors. No new mode, dependency, format or public
parameter is introduced. No release readiness is claimed.

## Local checkpoint

The frozen full run passed **3934 tests**, skipped 4,
with 81 warnings in 906.48s on **292 frozen Python files**.
Runtime bytes remain unchanged; the test-only CI correction below follows that
run. Local coverage is **14449/16081
statements** and **4138/5310 branches**.
Native thread environment variables were set to 1 before NumPy for verification;
runtime defaults are unchanged. Full-run elapsed differences are not attributed
to this environment without a controlled comparison.

Wheel/sdist builds and Twine pass. Core-wheel, CLI-wheel and isolated-sdist
installations pass physical tilted-cell positions, actual placement/flags,
zero-layer-tolerance hollow labels, NumPy matrices and float32 move manifests,
dependency checks, and earlier surface/registry/native serial/process/thread
workflows. Final support edits refresh only the sdist; unchanged runtime checks
are not repeated. Raw 144-module wheel/sdist and final support readback precede
the scoped dev push. All 16 exact-head CI jobs and downloaded coverage remain
the separate gate before the next runtime batch.

Ledger: **1233 functions**, **1107 with inspection
evidence**, **763 reviewed unchanged**,
**331 optimized and verified**,
**13 investigated deferrals** and **126 pending reviews**.
The 59-body scope covers 56 predecessor bodies and 3 new helpers, adds 23
inspection records, resolves four prior investigated dispositions, and records
the unchanged moved lambda by its current ID. Evidence is module-hash scoped.

## CI test portability correction

Initial dev `901530f` exposed a platform-sensitive assumption in the new
zero-tolerance test. Minimum NumPy on Windows and macOS reported five initial
height groups rather than three: exact-zero grouping can separate Cartesian
dot-product roundoff differently by NumPy/BLAS/platform. The production initial
layer-selection algorithm is unchanged; its tolerance must not be widened to
satisfy this fixture.

The corrected test explicitly supplies stable, already-selected planes through
the real selector, verifies that its caller requested zero exactly once, and
fails if packing certification calls the synthetic height grouper. It retains
the complete four-top/twelve-bridge/four-hcp/four-fcc physical assertions. This
deterministically detects the removed regrouping regression without claiming
portable exact-zero initial plane membership. Other physical integration tests
continue to use normal unmodified selection.

An independent source review approved this boundary and refreshed test hashes;
all seven source-module hashes and 59 body reviews remain unchanged. The revised
affected batch passed **93 tests in 3.55s**. Only one test file and this report
change after the full run; the archive is refreshed for those support bytes.
No unchanged runtime/full/install/benchmark work is repeated locally. Fresh
exact-head CI, including full supported-version suites, remains mandatory.

No release, tag or package upload is authorized; the overall goal is incomplete.

## Exact-head CI closure

Corrected dev a18b2b8baaea4b874208a16f9a13aa618f5b40c5 passed all16 jobs in
CI44477217790. Downloaded artifact14507627405 has verified SHA256
f782daea039639697806bd396bd73fd86bf9f4349be62b95739e534447e3a4c4,
matching head/job/module/statement sets. CI covers14451/16081 statements and
4138/5310 branches; only known prompts.py:501–502 differs locally, no branch
delta. Closure preceded the CI permissions fix and Part08a runtime work.
