# Native Gram-form moire search

CELLSTINE 4 searches commensurate **bilayer** supercells directly in Gram form.
The twist angle is an output of each accepted pair of integer supercell
matrices, not an angle swept by the user.

## Native workflow

Run a bounded search and then build candidates from its JSON:

```bash
cellstine moire find TOP.vasp BOTTOM.vasp --max-length 20 --top-strain 0.01 --bottom-strain 0.01
cellstine moire make runs/moire/<run-id>/results.json --indexes 1 --interlayer-distance 3.35
```

Candidate selection is spelled the same way everywhere: `make`, `maken`, and
`visualize` all accept either `--indexes` or `--indices`, with comma-separated
values and ranges such as `1,3-5`.

The result is schema `cellstine.moire.gram`, version 1, in `results.json`.
Previews, the builder, and both visualization paths use the same validated
central reader. Legacy positional `.dat` input is rejected with an instruction
to rerun native `moire find`.

The JSON records the top and bottom integer matrices, their Gram triples, angle
in degrees, relative principal strain pair, strain budgets and sharing fraction,
layer and total atom counts, rank and Pareto flag, Löwner certification,
recorded affine maps, shared lattice, and search metadata. No reader depends on
legacy positional columns.

## Where the stages live

The engine is one pipeline split over five modules of `moire/search/`, in
dependency order:

| module | contents |
| --- | --- |
| `gram_config.py` | `SearchConfig`, `SearchResult`, input validation, shared tolerances |
| `gram_lattice.py` | gauge reduction, vector shells, point groups, Hermite normal forms, the folded basis tables |
| `gram_pairs.py` | the Löwner acceptance test, the bucketed join, principal stretches, twist angles, canonical pair keys |
| `gram_report.py` | matrix powers, the affine maps, the coincidence index, angle folding, the assembled result |
| `gram.py` | the general and symmetric search drivers, and the public `search` entry point |

Two more modules carry the stages built on top of that pipeline:

| module | contents |
| --- | --- |
| `find.py` | the `moire find` stage: reading the layers, detecting and idealising their symmetry, running one search, writing `results.json` |
| `nlayer.py` | the `moire findn` stage: matching every upper layer against a rigid base layer and intersecting the per-layer base supercells |

The older coincidence-lattice engine that swept twist angles has been removed;
the Gram search supersedes it, reports the twist as an output of each candidate,
and can be restricted to a window of twists (see below). An independent
brute-force implementation of the same problem is kept in
`benchmarks/reference_moire.py` and the test suite checks the engine against it.

## What strain means

In the moire CLI, **strain** is the principal logarithmic strain

```text
h = log(lambda)
```

of the relative deformation's principal stretch `lambda`. A budget `e`
allows stretches from `exp(-e)` through `exp(e)`; it is not engineering
strain. Total accepted relative principal strain is bounded by the sum of the
top and bottom strain budgets. The engine shares that strain optimally between
the layers for each candidate. This naming is deliberate: it is scientifically
precise while keeping the CLI readable.

### Rigid searches

Both budgets may be zero. `--top-strain 0 --bottom-strain 0` runs a **rigid**
search: no layer is deformed at all and only exactly commensurate cells are
kept, so every candidate is a true coincidence lattice rather than a strained
approximation. That is the right mode for a twisted homobilayer, where the
commensurate twists are dictated by arithmetic alone; see
`TWISTED_BILAYER.md`. In a rigid search the reported strains are zero and the
sharing fraction is reported as one half, since there is nothing to share.

## Search outline

1. Enumerate reduced positive-definite Gram forms whose basis lengths satisfy
   `--max-length` and the optional cell-shape and atom-count bounds.
2. Join top and bottom forms using the Löwner inequalities implied by the sum of
   `--top-strain` and `--bottom-strain`.
3. Recover candidate matrices, principal stretches, relative logarithmic
   strains, optimal sharing, twist angle, and a common lattice.
4. Fold proper point-group equivalents when enabled, rank independent canonical
   candidate classes, and mark the Pareto frontier.

## Asking for a particular twist

The twist angle is an output, but it can be *selected* on: `--min-twist-angle`
and `--max-twist-angle` keep only the candidates whose reported twist lies in
that window, in degrees.

```bash
cellstine moire find LAYER.vasp LAYER.vasp --max-length 40 \
    --top-strain 0 --bottom-strain 0 --min-twist-angle 9 --max-twist-angle 14
```

The window is read on the *folded* angle, the one the candidate table shows, and
is applied after folding and before ranking. It therefore returns exactly the
candidates the unrestricted search would have reported inside the window --
the same integer matrices and the same angles -- rather than steering the
enumeration. Either bound may be given alone. Both are recorded in
`results.json` as `search.min_twist_angle_deg` and `search.max_twist_angle_deg`,
which are `null` when no window was asked for.

`--symmetric` is a restricted square/hexagonal symmetry-preserving family. Both
layers must carry the same rotation -- a quarter turn or a sixth turn -- and the
search then enumerates only the supercells `(v, R v)` that the rotation maps onto
themselves. If the inputs or requested bounds make that family inapplicable,
CELLSTINE records the reason and falls back to the general search.

The restriction is exact rather than heuristic. A sublattice of a square or
hexagonal lattice is rotation invariant *precisely* when it is `(v, R v)` for one
of its own vectors, so nothing rotation invariant is missed; such a cell holds
`Q(v)` primitive cells with `Q` the invariant form (`x^2 + y^2` for a quarter
turn, `x^2 - xy + y^2` for a sixth turn), which is the squared length the tables
are already sorted on; and two such cells, one per layer, are always related by a
rotation and one overall scale, so the join needs only a band on squared lengths
and every strain it reports is isotropic. All three statements are proved in
`RequestProject/SymmetricSupercell.lean`, and
`tests/test_moire_symmetric_branch.py` checks them against the running engine and
against the general search, which must report exactly the same bilayers whenever
both engines apply.

When imprimitive cells are asked for (`primitive_only=False`, an API-level
option) the two engines differ on purpose: a plain multiple of a commensurate
cell need not be rotation invariant, so the restricted branch reports a subset of
the general search rather than the same list.

Stacks of three or more layers use `moire findn` and `moire maken`, which match
every upper layer against a rigid base layer and intersect the per-layer base
supercells exactly.

## Symmetry tolerance and layer idealisation

A POSCAR is a finite-precision file. Printed with the customary six decimal
places, a hexagonal cell of side 2.468 A is only hexagonal to about one part in
`1e7`, and a relaxed DFT cell is usually worse. Symmetry detection therefore uses
a *physical* relative tolerance on the metric, `--symmetry-tolerance`, whose
default plays the same role as spglib's `symprec`. Detecting at machine
precision instead silently reduces graphene from six-fold to two-fold, which
costs the search its angle folding: the same bilayer is then reported once per
symmetry image, and the residual non-ideality shows up as a fake anisotropic
strain.

Because the rest of the engine treats the detected operations as exact, each
layer is then *idealised* onto the metric its own group preserves identically.
The idealised metric is the group average

```text
g_sym = (1 / |G|) * sum_G  G.T @ g @ G
```

which is the closest invariant metric, and the basis realising it is chosen by
orthogonal Procrustes so the layer is not rotated. `results.json` records
`metadata.symmetry_tolerance` together with `metadata.top_idealisation` and
`metadata.bottom_idealisation`, the largest relative change made to either
metric, so the size of the correction is always visible.

## Reporting an isotropic match as isotropic

The two principal stretches of a candidate are read straight off the two Gram
forms, as the roots of the pencil `det(P) * m^2 - T * m + det(Q)` with
`T = p22 q11 - 2 p12 q12 + p11 q22`. Their *difference* is the delicate part:
it is a square root of the discriminant `T^2 - 4 det(P) det(Q)`, and for an
aligned commensurate cell the two terms of that subtraction agree to the last
stored digit. Evaluating it as written therefore returns rounding noise, and
the square root turns a relative error of `eps` into a reported anisotropy of
`sqrt(eps)`: an exactly isotropic match came out with two principal strains
differing in the tenth decimal.

The engine instead evaluates the same polynomial in the rearranged form

```text
(p22 q11 - p11 q22)^2 + 4 (p22 q12 - p12 q22) (p11 q12 - p12 q11)
```

which is an identity (`Cellstine.stretchDiscriminant_eq`). Every bracket is a
cross difference of the two forms, so each one vanishes *identically* when `Q`
is a multiple of `P` -- exactly the isotropic case -- rather than by
cancellation. An isotropic match now comes back isotropic to a few units in the
last place, and a genuine anisotropy of `1e-11` is still resolved instead of
being swamped; `tests/test_moire_stretch_conditioning.py` pins both.

## Mathematical provenance

The statements the engine relies on are proved in Lean 4 with Mathlib in the
`RequestProject` directory of this repository; `RequestProject/Main.lean` lists
them against the search step or reported quantity each one justifies.

| Reported quantity / search step | Formal statement |
| --- | --- |
| commensurate pair = equal Gram forms | `Cellstine.gram_eq_iff_exists_orthogonal` |
| the Loewner join bounds the relative deformation | `Cellstine.loewner_sandwich_iff_deformation_sandwich` |
| principal stretches obey the budget | `Cellstine.abs_le_exp_of_deformation_bound` |
| a budget `e` is the stretch window `[exp(-e), exp(e)]` | `Cellstine.abs_log_le_iff_mem_exp_interval` |
| the stretches are the roots of the Gram pencil | `Cellstine.det_pencil_sub_smul` |
| the discriminant the engine evaluates is the pencil discriminant | `Cellstine.stretchDiscriminant_eq` |
| a proportional pair of Gram forms is exactly isotropic | `Cellstine.stretchDiscriminant_isotropic` |
| the principal stretches are always real | `Cellstine.stretchDiscriminant_nonneg` |
| the two reported numbers are the principal stretches | `Cellstine.stretch_sq_add_sq`, `Cellstine.stretch_mul` |
| the twist angle exists and is unique mod `2*pi` | `Cellstine.exists_twist_angle`, `Cellstine.twist_angle_unique` |
| the joint budget is the sum of the two budgets | `Cellstine.exists_strain_split_iff` |
| optimal strain sharing between the layers | `Cellstine.isLeast_shared_strain` |
| `--max-length` on a reduced basis bounds the lattice | `Cellstine.first_minimum`, `Cellstine.second_minimum` |
| per-layer atom counts scale by `|det M|` | `Cellstine.card_quotient_range_eq_natAbs_det` |
| symmetry folding preserves the atom count | `Cellstine.coincidenceIndex_unimodular_invariant` |
| relabelling the shared cell leaves the class key alone | `Cellstine.classKey_mul_unimodular` |
| and equal keys mean exactly one relabelling apart | `Cellstine.exists_unimodular_of_classKey_eq` |
| turning both supercells round repeats a key, so half the symmetry pairs are redundant | `Cellstine.classKey_neg` |
| `--symmetric` misses no rotation-invariant supercell | `Cellstine.square_invariant_iff_exists_generator`, `Cellstine.hex_invariant_iff_exists_generator` |
| a rotation-invariant supercell holds `Q(v)` primitive cells | `Cellstine.index_square`, `Cellstine.index_hex` |
| joining two of them can only strain isotropically | `Cellstine.similarity_of_generators`, `Cellstine.norm_similarity` |
| a stack cell sits inside a layer cell exactly when it factors over the integers | `Cellstine.rowLattice_le_iff_exists_factor`, `Cellstine.factor_unique` |
| the folded in-plane cell is the same whichever cell the layer was written in | `Cellstine.Gauge.gaugeOrbit_eq_of_det_one`, `Cellstine.Gauge.selection_eq_of_det_one` |
| two reduced bases of one plane lattice differ by entries in `{-1, 0, 1}` | `Cellstine.Gauge.abs_entries_le_one` |
| `findn` intersects the per-layer base supercells through the kernel of the stacked matrix | `Cellstine.rowLattice_eq_inf_of_kernel_spanning` |
| the intersection is the largest cell common to the layers | `Cellstine.isGreatest_inf_rowLattice` |
| it is a genuine cell, and the smallest common one | `Cellstine.det_ne_zero_of_rowLattice_eq_inf`, `Cellstine.isLeast_abs_det_inf` |
| it holds no more base cells than the two layer cells together | `Cellstine.card_quotient_inf_le` |
| so every rebuilt layer carries one and the same cell | `Cellstine.stack_shares_cell` |
| a partial stack already bounds the atom count of every stack extending it | `Cellstine.prefixAtoms_le_stackAtoms`, `Cellstine.prune_atoms_sound` |
| and already bounds its cell length, so the combination walk may prune | `Cellstine.second_gram_le_of_sublattice`, `Cellstine.prune_length_sound` |
| the twisted-graphene series the search reports: cosine `(4mn - m^2 - n^2) / (2 Q)`, cell `a sqrt(Q)`, `Q = m^2 - mn + n^2` | `Cellstine.HexTwist.twistCos_eq`, `Cellstine.HexTwist.coincidence_det`, `Cellstine.HexTwist.coincidence_length` |
| each of those twists is a rotation and is commensurate | `Cellstine.HexTwist.twistMatrix_gram`, `Cellstine.HexTwist.twistMatrix_det`, `Cellstine.HexTwist.twistMatrix_mulVec` |

`tests/test_moire_theory.py` checks the running engine against those same
statements, so the proofs and the code are tied together rather than merely
cited. CELLSTINE itself executes Python/NumPy only: the Lean sources are a proof
artefact and are not needed at run time.

[Harmonic's Aristotle](https://aristotle.harmonic.fun/) and
[Lean 4](https://lean-lang.org/lean4/doc/) are cited as the tools used for that
derivation and checking.

## Reproducible comparison

From the repository root, run:

```bash
python benchmarks/benchmark_gram_search.py
```

The script compares independent canonical candidate classes from the reference
enumeration and the native Gram search. It stops on mismatch, reports actual
timings at three increasing length bounds, and treats speed numbers as
host-dependent measurements rather than fixed performance assertions.

The guided interface remains a simple launcher over the same native backend.
Its preview and static/interactive visualizations read `results.json`, show
angles in degrees, and use the terms top strain, bottom strain, and relative
principal strain.
