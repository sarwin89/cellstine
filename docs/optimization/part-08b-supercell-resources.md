# Defect supercell allocation and reporting review

Base e01be6f63e8d5b66e85ed289f145a6956915b706 passed all16 CI44513902716
jobs before runtime edits. Downloaded coverage artifact14525346664 SHA256
83967cb5cb0ffa18bba06e1a5000e023678526e69bbb15922dfdcaea5c02389f
was verified:14532/16158 statements,4198/5368 branches; only known
prompts.py501–502 differs locally, with no branch delta.

## Scope, behavior and resources

All28 supercell.py bodies and6 reporting.py bodies were read completely,
including public selectors, HNF generators/counts, integer membership, bounds,
reduction, nested tie ranking, serialization and presentation. Relevant
geometry/grid/workflow and existing exhaustive/written-output tests were read.

The2D/3D HNF parameter arrays now allocate their exact final row counts once.
Divisor ordering and all parameter columns remain. A shaped view broadcasts the
ab/ac/bc axes in the same ij/C order without an auxiliary meshgrid, block list
and final full concatenation. Returned arrays remain contiguous int64. Positive
scientific inputs are unchanged; private nonpositive counts still raise
ValueError, now earlier through the public count validator.

Host-vector search retains inverse/pseudoinverse spacing, floor reach+1e-9,
lexicographic coefficients, zero removal, floating norm, radius+1e-9 and default
argsort. The reviewed integer-box iterator bounds temporary grids to65536 rows.
Accepted chunks concatenate in original lex order before sorting. Empty and
ordinary negative-radius searches retain empty arrays with correct dimensions.
Accepted vectors/norms, concatenation, sort indices and final arrays still use
storage proportional to output; total memory is not universally bounded.

Maximum-distance selection, user cell limits, integer membership, basis
reduction, scientific tolerances and existing roundness policy are unchanged.
The report now distinguishes genuinely empty discovery from zero displayed
rows: limit0 or a clamped negative limit keeps the header and omitted count.
Positive-limit text and the actual analysis list remain unchanged. The two new
report regressions failed before correction.

## Evidence and independent review

Initial resource regressions gave5 failures/23 passes: duplicate HNF storage,
unbounded meshes and skew host-vector peak allocation. Real integer blocks are
observed with their row cap, rather than merely replacing a helper with a stub.
Independent integer enumeration checks ordered HNF values/counts/dtype; scalar
vectors cover empty, negative, zero and1e-9 cutoff behavior. Existing exhaustive
FCC/slab choices and written workflow readbacks remain part of affected checks.
There are34 new cases; they enter minimum-NumPy and nine portable CI lanes.

Independent review covers34 unique bodies/current raw and normalized hashes,
including callers/tests and the shared iterator. No introduced blocking finding
was found. One preexisting investigated deferral is recorded for _roundest:
the32-candidate unreduced-product shortlist does not certify globally optimal
reduced-product roundness or order independence at cheap ties. Primary maximum
image separation remains separate. This pass does not silently change ranking
policy or claim a missing global certificate was proved.

After the initial freeze, a coordinator probe found an introduced false-empty
edge missed by the initial review: radius1e20 on the unit cell previously raised
ValueError, but the new negative-reach path returned an empty result after
int64 conversion overflow. Four regressions failed for oversized positive and
negative radii, infinity and NaN. The corrected path checks finite floating
reach and its int64 coordinate range before casting, then rejects unsupported
requests explicitly. Ordinary negative-radius empty results remain. The owned
obsolete full process was stopped, its logs/artifacts retained, and the final
review hashes refreshed with the resolved finding. Initial gate results do not
certify this correction; all final source/package gates are renewed.

Actual predecessor/current comparisons cover70 complete parameter arrays,
20 vector arrays/norm arrays (including multi-block skew and ties), and12 final
selected matrices/distances. Raw signatures include dtype, shape and bytes.
Eleven quiet warm interleaved measurements compare accepted outputs exactly.
Five timing samples and separate tracemalloc probes use native env1 before
NumPy, without other local runtime tests/probes/builds. Allocations are not RSS.

| Case | Previous ms | Current ms | Previous traced KiB | Current traced KiB |
| --- | ---: | ---: | ---: | ---: |
| HNF3D, 1 cells | 0.0238 | 0.0093 | 10.72 | 1.20 |
| HNF3D, 64 cells | 1.3949 | 0.3136 | 1018.67 | 508.00 |
| HNF3D, 256 cells | 8.4522 | 2.7577 | 16346.95 | 8172.28 |
| HNF2D, 16 cells | 0.0184 | 0.0166 | 2.58 | 1.70 |
| HNF2D, 1024 cells | 0.0479 | 0.0436 | 98.06 | 56.99 |
| host-vectors-unit3D | 0.1161 | 0.1212 | 11.56 | 9.30 |
| host-vectors-hex2D | 0.1200 | 0.1378 | 8.95 | 6.73 |
| host-vectors-skew3D | 46.4226 | 42.8691 | 43454.12 | 7703.75 |
| selected-cell-FCC, 8 cells | 2.7772 | 2.5236 | 38.48 | 37.06 |
| selected-cell-hex, 12 cells | 1.6015 | 1.4157 | 12.29 | 13.92 |
| report-positive-limit | 0.1679 | 0.1782 | 19.09 | 19.09 |

The extra reach checks cost roughly0.005ms in the unit3D case and0.018ms
in the small hex2D case. Selected hex-cell traced storage rises about1.6KiB
in this sample. Larger HNF storage falls by about half and the skew search
temporary peak falls substantially. Small report timing variation is not a
general speed claim. All four unsupported-radius cases match the predecessor
ValueError category, with no new narrowing warnings or false-empty results.

## Retained limits and gates

The32-candidate secondary roundness heuristic is an explicit investigated
deferral. Existing determinant/inverse conditioning, floating bounds, int64
coefficients/intermediates and reduction acceptance domains remain. No arbitrary
integer or all-finite-scale certificate follows. No format, dependency, public
parameter, parallel mode, cutoff or output-enumeration limit is added.

## Local checkpoint

The corrected frozen full run passed **4028 tests**, skipped 4,
with 81 dependency warnings in 710.54s. All **295
Python hashes** remain unchanged. Local coverage is **14547/16175
statements** and **4208/5378 branches**.
The affected batch passed130 tests with2 warnings in14.34s. Native env1 is a
verification setting; runtime defaults and scientific cutoffs are unchanged.
Initial stopped-source gates are retained only as invalidated history.

Corrected wheel/sdist/Twine and all three installed core/CLI/sdist environments
pass ordered HNF arrays, real multi-block host vectors, rejected unsupported
radii, selected/written FCC cells/flags, zero-limit reporting, source ownership
and previous native workflows, with dependency checks. Final support changes
refresh only the sdist. Raw144-runtime/support archive readback precedes scoped
dev push, followed by all16 exact-head CI and downloaded coverage.

Ledger: **1236 functions**, **1164 with inspection
evidence**, **805 reviewed unchanged**,
**345 optimized and verified**,
**14 investigated deferrals** and **72 pending**.
All34 bodies add inspection evidence. The existing roundness heuristic now has
an explicit investigated disposition; it is not mislabeled globally verified.
Remaining pending work:16 defect sites/workflow bodies,23 visualization,
12 CLI and21 benchmark helpers. No release readiness, main merge, tag or upload.
The overall goal remains active and incomplete.

## Exact-head CI closure

Dev cda1bc5a588e754513a0dcf2e1f34fbad042c4ca passed all16 jobs of
CI44522433664. Downloaded artifact14529048829 SHA256
bacb7f68e1bd72405a44d5f02aca5b910fe030911089af81a2880082174f4407
matches the head/job/digest and module statement sets. CI coverage is
14549/16175 statements,4208/5378 branches. Only the known prompts.py501–502
fallback differs from local coverage; no branch delta. This closes Part08b's
gate before Part08c runtime edits. Overall review remains incomplete.
