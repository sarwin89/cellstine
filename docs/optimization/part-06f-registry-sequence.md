# Registry origins and plane sequence matching

Base dev `3a9afc0bb69edaa7972579963b1ecccc3ad8e69f`. This batch inspects **30
complete bodies**: assembly 12 (including 2 new helpers), registry 13, sequence 4
and the sites wrapper 1. Independent source review found no unresolved blocker;
its additional 150 exhaustive matching comparisons and 50 long-prefix oracle
cases passed. Package and CI gates are separate from source review.

## Correct physical contact reporting

Separate origin-reset layer labels cannot encode two slabs' relative translation.
For an ABC bottom ending at C and top translated by 0, h, 2h, the predecessor
reports delta 1 for all three. Independent expected contacts are 1, 2, 0. The
record-owning paths now measure actual boundary anchors in the same primitive
gauge, projection, componentwise physical tolerance and integer-lattice test
used by registry_shift_direct. Geometry and ordering are preserved. Explicit
requests validate the chosen delta after translation, including overlapping
tolerance neighborhoods. Public contact_summary keeps its signature, label-only
inference, values and field order.

A nonideal unrequested contact remains a valid input with `delta: null`,
`contact: "unclassified"`, `kind: "general_offset"` and an explanatory note.
This intentional metadata value-domain expansion replaces misleading guessed
ideal contacts. It is documented in the publication plan for consumer/schema
review. No registry enumeration policy or geometry transformation changed.

## Exact plane matching and prefix resources

Nearest-first pairing can steal the only partner of another atom. The witness
first[.04,0], second[.01,.10] at tolerance .07 has a valid bijection with distances
.06 and .01, but previously returned False. A failed greedy assignment now uses
an iterative augmenting path. Every atom still needs a unique positional
periodic partner under the existing Euclidean tolerance. Rows are generated
on demand; auxiliary storage is linear, without a dense pair matrix or recursion.
The reverse map is allocated only when repair is necessary. This is a correctness
fix; no universal plane-matching speedup is claimed. Dense ambiguous cases can
still require repeated distance rows and substantial CPU.

Prefix search retains native string checks for periods through 256, then uses a
packed unsigned prefix-function table. A partial last repetition remains valid;
there is no divisibility requirement. The bounded preliminary search plus linear
fallback avoids quadratic work for long nonperiodic words. The table increases
temporary memory and can cost CPU for periods just above the direct-path bound;
all measured cases are disclosed below rather than only the favorable case.

## Measurements

Five warm interleaved medians, with separate traced allocation probes and native
thread environment variables set to 1 before importing NumPy. Python 3.14.3 /
NumPy 2.4.3, same process and
shared dependencies. Only the two predecessor modules are loaded from base Git
source. Measurements ran after the full suite stopped. Traced allocations are
Python/NumPy allocations, not RSS; no timing test or universal hardware claim.
Accepted ordinary outputs compare exactly with the predecessor; corrected
contacts and bijections use literal independent oracles.

| Case | Previous ms | Current ms | Previous traced KB | Current traced KB |
| --- | ---: | ---: | ---: | ---: |
| prefix-small | 0.0010 | 0.0011 | 0.33 | 0.33 |
| prefix-long-crystal | 0.0073 | 0.0074 | 90.37 | 90.37 |
| prefix-nonperiodic | 118.2815 | 8.6970 | 150.36 | 270.83 |
| prefix-medium-period | 0.1445 | 0.1450 | 61.17 | 61.17 |
| prefix-long-period | 0.7672 | 6.1825 | 91.27 | 272.10 |
| plane-ordinary (4 atoms) | 0.5130 | 0.4866 | 14.06 | 14.24 |
| plane-ordinary (100 atoms) | 12.9544 | 13.2677 | 28.93 | 29.97 |
| plane-ordinary (1000 atoms) | 175.1281 | 180.9852 | 159.56 | 167.75 |
| contact-ordinary | 4.1159 | 4.4516 | 19.15 | 19.15 |
| plane-corrected-bijection | 0.1458 | 0.3885 | 5.40 | 5.60 |

## Verification and ledger

Initial sequence regressions: 1 failed / 9 passed. Initial registry cases:
15 failed / 7 passed. The final affected suite passed **178 tests in 6.23s**. New tests cover
orthogonal/skew exhaustive small bijections, Hall deficits, duplicates, periodic
boundaries, Unicode/partial/long periods, translated 60/120-degree contacts,
common-origin and lattice-translation invariance, .049/.051 componentwise
physical tolerance, public label-only compatibility, and actual POSCAR/manifest
metadata with unchanged planar geometry. Both new files enter the minimum NumPy
lane and all nine portable OS/Python lanes.

One frozen full run passed **3855 tests**, skipped 4,
with 81 warnings in 1143.69s. All 289 Python hashes
remain fixed. Local coverage: **14328/15977
statements** and **4052/5242
branches**. Wheel/sdist/Twine and all 3 installed core/CLI/sdist environments
passed actual registry and prefix regressions, native direct/matched-slab builds,
mixed flags, native process/thread builds and dependency checks. Final support
documents/ledger refresh only the sdist; unchanged runtime/full/install checks
are not repeated. Final archive raw-byte readback and scoped dev push precede
the exact-head 16-job CI/downloaded coverage closure.

Ledger: **1228 functions**, **1061
with inspection evidence**, 734 reviewed unchanged,
314 optimized and verified,
13 investigated deferrals and
167 pending reviews. This batch adds 20 inspection records.
The former three-body actual-origin reporting deferral is resolved.

## Explicit dispositions

* `stacking_sequence`: the existing one-character alphabet cycles after 26
  distinct planes. Preserve the format pending a representation decision;
  do not certify unbounded distinct-plane labelling. Investigated deferral.
* `slabs_are_interchangeable`: chemistry multisets and layer spacings do not
  prove arbitrary in-plane isometry or decorations. The prepared ABC/HCP
  suspicion produces the same option rows under either flag, so it is not
  evidence of a dropped class. Preserve policy without inventing a replacement
  or claiming a complete physical proof. Investigated deferral.
* Generic nonfinite/external invalid-domain hardening and tilted/unmatched raw
  assembly are not newly certified. Actual workflow callers provide aligned,
  common planar cells; the existing numerical-domain limitations remain.
* The independent review establishes source reasoning and narrow probes;
  the coordinator owns full/package/installed/CI verification separately.

The project-wide goal remains incomplete. No release, tag or upload is authorized.
