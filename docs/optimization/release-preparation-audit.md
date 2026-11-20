# Release preparation audit and initial-to-current progress

This report describes the reviewed dev candidate, not a published release or
an unconditional certification of every numerical domain. The source-function
ledger, part reports, exact-head CI and artifact readbacks are separate evidence.
The final local/CI checkpoint is filled only after the corresponding gates.

## Starting point and delivered changes

The initial shared checkout included concurrent uncommitted work. The isolated
dev preparation started from bb90941; that ownership boundary remains explicit.
Initial local reports recorded3254 passing tests in the shared checkout and
3156 at the early isolated Part01 checkpoint. Those are different source scopes,
so their counts are not interchangeable benchmarks or proof of regressions.
Claude's reported3334-test handoff was integrated and reviewed in scoped later
parts; it is not substituted for the current candidate's own verification.

| Area | Initial problem or limit | Reviewed candidate behavior/evidence |
| --- | --- | --- |
| Auditability | Broad cleanup lacked a complete function-level disposition | LF-normalized source-hashed ledger covers tracked non-test callables, nested functions, lambdas, benchmark helpers and review tooling; unchanged sound bodies are retained |
| Geometry and lattice | Whole pair grids, repeated transformations and fragile extreme-scale vector normalization | Bounded geometry/contact working arrays, exact periodic kernels and improved finite-vector normalization; scientific tolerance/domain exceptions remain explicit in the limitations below |
| I/O and persistence | JSON-only default and expensive large-document materialization | User-authorized no-pickle NPZ default, validated portable consumers, explicit JSON compatibility and bounded indented JSON encoding; native formats/ordering remain checked |
| Moire search/build | Repeated inputs and intermediates, primitive/sign key weaknesses, coupled search/output limits | Exact signed/primitive keys and cosets, invariant pruning, read-once batch preparation and honest candidate totals/shortlisting; builder/visualizer scientific readbacks are tested |
| Parallel work | Independent bilayer calculation and private writes were serial | Explicit bounded process/thread scheduling covers both calculation and I/O, preserving output order, source ownership and failure propagation; small or coupled work retains serial paths |
| Surfaces/interfaces | Repeated layer/site work and unrealized reverse stacking in some paths | Bounded site resources, physical CBA/mirror behavior, registry-origin/sequence reporting and tilted-cell projection/placement corrections; full-domain native surface fallbacks remain investigated limitations |
| Defects | Large pair matrices, incomplete orbit handling and ambiguous displayed atom IDs | Bounded ordered pairs, physical plane/member selection, explicit legacy-orbit handling, invalid-index rejection, correct manual divacancy representatives, normal-height placement and accurate counts/provenance |
| Symmetry/reciprocal | Large enumeration intermediates and weak resource/domain boundaries | Bounded ordered integer enumeration, owned caches and exact matching/reciprocal keys while retaining established symmetry/cell conventions |
| CLI | Guided migration/back/replay gaps and expensive eager imports | Shared normal builders, lowercase back navigation with uppercase chemical symbols retained, plain/rich parity, explicit limits and lazy dependency-free parser startup |
| Visualization | Inline JSON could break HTML script boundaries; repeated label work and unused row copies | Safe inline JSON boundaries, unchanged decoded physical data, unique-label sizes, early selected-row preparation, owned-figure cleanup and configured-backend --show policy |
| Verification/package | Limited supported-version instrumentation and incomplete artifact provenance | Full Windows Python3.10–3.14 jobs, minimum NumPy1.26 lane,9 portable Windows/Linux/macOS jobs, package checks, downloaded line/branch coverage and installed artifact readbacks |

Detailed function/caller reasoning and measurements remain in the linked
[part reports](README.md). Changes deliberately repair some preexisting wrong
results/errors; compatibility does not mean retaining the wrong atom selection,
false empty reports, unsupported-bound false empties or unsafe HTML boundaries.

## Measured examples and their scope

Part08c's1000-atom no-pair discovery:654.84ms to347.88ms, traced peak39.82MB to
11.51MB (about47% less time/71% less traced allocation). These are per-part
measurements against the actual predecessor, not a full-workflow speed claim.

Part09's final-source measurements: preparing one selected plotting row out of
10000 goes from12.4028ms/4725712 traced bytes to0.8685ms/1200 bytes. This isolates
row preparation after validated payload decoding; disk reading is unchanged.
Actual10000-atom HTML writing/readback goes from80.6949ms to70.2450ms. Two-atom
HTML costs about0.25ms and176 traced bytes in that sample. Two representative
static PNGs are byte-identical. Timing medians and allocation peaks are measured
separately; none of these figures is a process-RSS or universal speed guarantee.

## Investigated limitations retained in the ledger

These14 bodies have real inspection evidence and concrete reasons for deferral.
They are not pending/unread functions and are not silently certified.

| Function | Retained limitation |
| --- | --- |
| core.reduction.rational_lattice_basis | Existing allclose relative-plus-absolute acceptance needs a separate scientific tolerance contract |
| core.validation.structure_errors | Extreme finite determinant/product and downstream distance overflow require an end-to-end representable-domain repair |
| defect.supercell._roundest | The32-entry unreduced-product shortlist does not certify global reduced-product roundness or arbitrary-order independence; image separation is selected separately |
| surface.registry.slabs_are_interchangeable | Species multisets/spacings do not establish complete decorated in-plane isometry; no dropped-class witness was established |
| surface.sequence.stacking_sequence | One-character plane labels cycle after26 distinct planes; unbounded plane identity needs a format decision |
| surface.stacking._gauss_reduce | Fixed iterations/cutoff do not certify arbitrary-conditioning convergence |
| surface.stacking._layer_point_lattice | Finite candidate discovery and9N image vectors do not establish completeness for arbitrary supercells |
| surface_cell._primitive_surface_vectors_from_lattice | Fixed coefficient-box fallback misses shorter vectors even for a finite strongly skewed ordinary lattice; valid output alone is not a successive-minimum proof |
| surface_cell._primitive_surface_vectors | Its fallback inherits that heuristic; ordinary recognized rational paths remain separate |
| surface_cell._build_native_primitive_surface_cell | Finite cubic image searches can omit representatives and have large intermediate storage |
| surface_cell._build_native_primitive_surface_cell._layer_atoms | Growing arrays/componentwise skew wrapping need a separate equivalence/resource investigation |
| surface_cell._structure_from_transform | Ordinary missing images were repaired and final cardinality checked; extreme conditioning/integers, practical cubic storage and cross-species policy remain uncertified |
| surface_cell._structure_from_transform._collect | Original order/flags/arithmetic remain, with address guards; inherited resource and cross-species dedup limits persist |
| surface_cell._structure_from_inplane_transform | Floating determinant/inverse and square image enumeration are not certified for arbitrary huge integer shears |

In particular, zero pending function reviews does not make the native surface
fallbacks complete for every skew lattice. These limitations must accompany
any future release decision. Dependency deprecation warnings and externally
unavailable Lean-reference cases are reported separately from failures.

## GitHub security state and publication boundary

Readback on2026-11-20 found alerts1 and2 still open on main at
bb90941471c96d6f93426107e06c1caba5208b56. Both are medium
actions/missing-workflow-permissions in .github/workflows/ci.yml. The underlying
dev workflow has explicit root permissions: contents: read, with successful CI
since the dedicated5fd5be3 fix. The alerts have not been dismissed or falsely
reported closed. Main integration and a corresponding scan are future work;
this task does not authorize a main merge.

The [publication plan](../publication-plan.md) covers candidate selection,
version/name revalidation, wheel/sdist/Twine and installed checks, future trusted
publisher setup, optional TestPyPI rehearsal and exact artifact provenance.
Public PyPI JSON returned404 for cellstine on2026-11-20. That does not establish
name eligibility or reserve it. No account settings, publisher, package upload,
tag, release or main merge were created. Scientific run data and unrelated
shared-checkout edits remain outside the scoped dev commits.

## Current checkpoint

The frozen full suite passed **4113 tests**, with 4 skips and
81 dependency warnings in 747.81s. All299 Python hashes
remain unchanged. Local package coverage: **14604/16192
statements**, **4237/5386 branches**.
Wheel/sdist/Twine and three installed core/CLI/sdist environments pass actual HTML
boundary/decoded-coordinate/native-NPZ summary checks, previous physical
workflows and dependency checks. Final documentation/bootstrap packaging changes
refresh the source archive; byte readback must match all299 frozen Python files,
all144 runtime modules in both archives and8 final support files before push.

The canonical ledger now records **1238 functions with inspection evidence**:
**865 reviewed unchanged**, **359 optimized and verified**,
**14 investigated deferrals**, **0 pending**. This batch reviews96 bodies and
adds57 newly inspected bodies, including one private helper. All-function
inspection is complete; it does not remove the14 documented limitations or
certify an unrestricted release. [The comprehensive audit](release-preparation-audit.md)
describes the initial/current boundaries and outstanding maintainer decisions.

For a candidate commit, all16 exact-head CI jobs and downloaded coverage remain
separate required evidence; their status must be read from that commit rather
than inferred from this source-tree report. The completion readback also checks
the objective requirement by requirement. No main merge, tag, upload, publisher
configuration or release is authorized by these local gates.
