# Next shared-core batch: conventional cells and idealisation

Initial read-only inspection during the frozen reciprocal checkpoint covered all source in `core/bravais.py` (18 functions) and `core/idealisation.py` (three including nested `_add`). These remain pending in the ledger; no source changes are part of Part03d.

## Findings to pursue

- Conventional cubic construction calls `_lattice_points` three times with the identical idealised lattice, repeating Delaunay reduction and the same shell. A wrapped runtime probe on `4*np.eye(3)` returned `cP` and confirmed three calls. Axial and monoclinic branches similarly repeat the same cloud. Assess computing it once per construction branch and supplying it to private helpers while preserving candidate iteration, ties, axis signs and public API.
- `close_matrix_group` tests `max_order` only after all products of the current element list have been computed. Assess enforcing the same cap at insertion, before invalid/noisy detections create many more elements. Preserve closure order and immutable return storage for valid groups; inspect input conversion and exact-product domains as part of the same whole-function review rather than silently tightening scientific tolerances.
- Read existing all14-Bravais, rebase, coset/multiplicity, rounded-input, metric-averaging and orientation tests; preserve standard C-centring, b-unique obtuse monoclinic conventions, idealisation deviation reporting, and column/row conversions.

Benchmark actual conventional-cell results and allocation against the next committed dev baseline. Test genuine repeated-work/cap findings, obtain whole-batch independent review, then one frozen full/package checkpoint. Leave sound functions unchanged with per-function reasons. Keep global floating-point representable-range limitations explicit, preserve concurrent root work/scientific data, and do not publish.

Continue using the separate Claude integration plan after these shared dependencies; the proper/improper moiré equivalence boundary must remain explicit.

## Execution checkpoint

Shared deterministic clouds and insertion-time caps are implemented. Eight initial regressions failed first;456 affected tests passed afterward. Independent whole-function review and narrow cap/order/tie/ownership probes found no issues. The one full suite passed3,339 tests; package/install checks passed;257 frozen Python hashes and file set match. The ledger adds21 original functions and two benchmark entries. Pushed exact-head CI remains the next gate; see the [part report](../../optimization/part-03e-conventional.md).
