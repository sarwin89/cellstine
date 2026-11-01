# Void completeness implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove silent neighbour truncation while preserving complete maximum/saddle candidate support and bounded working allocations.

**Architecture:** Keep the existing small-neighbour path where possible. Above the old 200-neighbour threshold, conservatively prune geometrically impossible supporting planes, stream compatible triples/pairs, and check emptiness against all original neighbours in bounded blocks. A threshold may select an algorithm but must never drop candidates merely by count. Keep NumPy as the only mandatory dependency.

**Tech stack:** Python 3.10â€“3.14, NumPy, pytest/coverage.

**Spec:** Active systematic-review goal and `docs/optimization/README.md`; algorithm/proof conditions below are the design chosen under the user's authorization to continue autonomously.

## Constraints and proof conditions

- Preserve public signatures, radius/contact/degeneracy tolerances, file schemas and normal-case output ordering.
- Existing `_MAX_NEIGHBOURS = 200` truncation can delete every transverse image in a dense collinear cluster. It also removes atoms from the emptiness check. No count-based truncation may remain.
- Relative centre c and neighbour edge e satisfy `e.c = |e|^2/2` when the neighbour supports the sphere. Radius is at most R. The existing emptiness tolerance accepts nearest distance at least r-1e-6, implying `e.c <= |e|^2/2 + R*1e-6`. The expansion belongs to the unnormalized dot-product RHS; dividing by edge length changes its units.
- Start an outer box at [-R,R]^3 with outward roundoff guards. For each expanded halfspace, minimize contributions of the other two coordinates to derive a safe upper/lower bound on the remaining coordinate. Apply bounded repeated sweeps; incomplete contraction only weakens pruning. Skip numerically uncertain divisions and retain uncertain cases.
- Remove a support only if the maximum of e.c over the final outer box is strictly below |e|^2/2 minus a numerical guard. Equality and near equality remain. Apply the mask in original neighbour order. This argument applies equally to tetrahedron maxima, pair midpoints and triangle saddles.
- Emptiness checks must retain all original neighbours, including those removed from candidate support. No accumulated full candidate-by-neighbour product; no full cubic triple table above the bounded small path.
- Review the existing multi-size triple cache for an explicit aggregate memory bound; preserve enumeration order.
- A polyhedron-clipping approach was considered but rejected for this pass because vertex/face degeneracies add unnecessary numerical risk. Pure streaming without conservative pruning remains complete but can waste combinatorial work on dense clusters. Mandatory SciPy is incompatible with the package's NumPy-only core.

## Review focus

Near-duplicate/short edges need the correct unnormalized tolerance expansion. Cospherical contacts must survive equality. Rotated/skewed cells may make box pruning weak; streaming still must work. Pair and triangle saddles need the same full-neighbour emptiness checks. Ill-conditioned solves and exact ties must not silently reorder normal outputs.

## Task 1: reproduce the omission and implement complete bounded enumeration

**Files:** `src/cellstine/core/voids.py`, `tests/test_voids_dense_neighbours.py`, existing void tests.

- [x] Add public regression with 202 atoms at x=linspace(.49,.51), y=z=.5 in a 10 Angstrom cube. Disable vacuum filtering with infinity; use minimum radius 1 and valid cutoff18. Assert nonempty maxima and largest radius sqrt(4.9^2+5^2+5^2), centre(0,0,0), absolute 1e-9.
- [x] Run `python -m pytest tests/test_voids_dense_neighbours.py -q`: fails on baseline a455f81 with empty sites; 7.76 seconds. Log is ignored local `E:/moire-supercell/.tmp/part02c-red.log`.
- [x] Independently review pruning proof. Reviewer verified 5,453 neighbours per sampled anchor and interval contraction retaining 109/192/109 after two sweeps. These are support counts only; not yet a verified production fix.
- [x] Implement conservative box pruning in a private helper; invalid/nonfinite/uncertain box state must retain neighbours rather than assert exclusion.
- [x] Implement lexicographically ordered compatible triple batches, pair/triangle batches, and blocked full-neighbour emptiness. Keep required returned output storage separate from bounded working memory claims.
- [x] Bound aggregate triple-index cache memory or replace it with bounded enumeration without breaking its consumers. The current only caller is `_empty_sphere_centres`.
- [x] Run the public regression and existing `tests/test_voids*.py`; reproduce normal outputs against trusted Git baseline a455f81 for maxima and saddles. Add independent direct uncapped enumeration for small clouds and edge cases that probe tolerance, cospherical support, rotations and blockers excluded from support.
- [x] Measure traced allocations and time versus baseline on dense/ordinary cases; compare corrected dense case against its analytic geometry rather than claim identical wrong outputs.

## Task 2: verify and checkpoint the scoped part

**Files:** function ledger, Part02c report, summary/plan, benchmark support if needed.

- [x] Request independent code review after implementation; resolve material findings. Current final frontier batching independently reviewed; full dense/frozen test gates remain pending.
- [x] Freeze Python source/test hashes, run focused tests, then full suite with branch coverage; require unchanged hashes after the run. Final result: 3,190 passed, 4 skipped; 717.44 seconds; all 241 Python hashes stable.
- [x] Build wheel/sdist, Twine checks, source-byte comparison and installed numerical/CLI/dependency smokes. All pass; ledger regenerated to 1,096 functions with 127 inspection records and 969 pending reviews. Final documentation-inclusive source archive rebuild follows report updates.
- [x] Record concrete proof scope, timings, memory and remaining limits in Part02c report. Removed the neighbour-count completeness deferral after the full frozen regression and reference checks passed.
- [ ] Commit only scoped files, push dev, await all Python 3.10â€“3.14 and package jobs, and verify coverage artifact head SHA before the next part.


## Implementation rulings and current evidence

Ruling: carry a guard for absolute-coordinate roundoff into the outer radius box and expanded halfspaces — a three-ULP pair at absolute coordinate1e10 was accepted by the legacy emptiness rule but excluded by a radius-only box — skipping this can silently remove tolerance-accepted saddles. The new red/green test pins that case. Expansion uses outer radius R+delta and emptiness allowance1e-6+delta. Cache uncertainty likewise triggers an exact original-point query.

Ruling: process large maxima-only candidate sets in stable radius-priority chunks of128 before merge — dense local contacts otherwise force classification of candidates known to be suppressed later — the risk is changed greedy winner/tie ordering, pinned by a64sitepublic regression and a nontransitive equal-radius chain. Use the same periodic-neighbour relationship as final merging. Build a bounded local adjacency table; classify mutually nonadjacent frontier candidates lacking earlier pending neighbours; only accepted maxima suppress later nodes. Append survivors in original priority order. Saddles retain classification before kind-specific merging, and merge<=0retains batch classification. A singleton version caused a measured6x ordinary-supercell slowdown and was replaced; the current SC64 default benchmark improves155.08→135.64ms on the observed run. This is not a universal speedup claim.

Ruling: keep the exact202atom dense collinear regression, including the full original point set and analytic radius — do not replace it with an easier geometry merely for a passing or faster test. Outdated exploratory processes were explicitly stopped after algorithm changes; their interrupted runs are not verification. Current dense run is session48022, log E:/moire-supercell/.tmp/part02c-regression-v2.log, imported the singleton pre-merge version. Final source must receive its own frozen full-suite run after focused success.

Fast current evidence:13 focused tests pass (one dense case still pending),63 existing void tests pass. Reviewer evidence includes250smalluncapped comparisons,205neighbour triple reference,24original public comparisons,220classifier comparisons,500large-offset clouds, and additional24public+36large synthetic frontier comparisons. No material review finding remains. Source/test freeze is ignored local E:/moire-supercell/.tmp/part02c-freeze.json; any further Python edit invalidates it. Package/CI/ledger closure is still pending.


## Final local gate readback

The final frontier implementation passed 14 fast new cases, 63 existing void tests and the full 3,190-test frozen coverage suite. All three installed artifact environments pass with NumPy 2.5.4. The original dense intermediate run completed in 481.34 seconds; the final full run included that same geometry and completed in 717.44 seconds overall. Earlier pending/process notes above are historical, not current failures. Pushed CI remains the final gate before Part 03.
