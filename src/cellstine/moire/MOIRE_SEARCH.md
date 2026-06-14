# Moiré commensuration search — algorithm & options

This note documents the reworked `cellstine moire find` search engine
(`moire/commensurate.py`, driven by `moire/finder.py` and `moire/find.py`).

## The idea: the twist angle is an output, not a swept input

Two layers (top rotated by `θ`, bottom fixed) are commensurate when a lattice
vector of the rotated top layer coincides with a lattice vector of the bottom
layer. Write a top vector as `u = m a₁ + n a₂` and a bottom vector as
`w = p b₁ + q b₂`. Length is rotation invariant, so a coincidence requires
`|u| ≈ |w|`. **Given** such an equal-length pair, the twist that makes them
parallel is fixed analytically:

```
θ = angle(w) − angle(u) = atan2(u × w, u · w)
```

So every integer vector pair *determines* a candidate twist angle — there is
nothing to sweep. Two non-collinear coincidences sharing the same `θ` span a
commensurate supercell.

### Pipeline

1. Enumerate the in-plane integer vectors of both layers once.
2. Keep only equal-length pairs. This is a rotation-invariant prune done by
   sorting the norms and binary-searching the admissible length band — the dense
   `N₁ × N₂` mismatch matrix is never formed.
3. Tag every surviving pair with its analytic twist `θ` and sort by it.
4. For each candidate angle, gather the pairs in a narrow angular window around
   it with a binary search (a contiguous slice) and build the supercells.

This replaces the old "shortlist a few angles, then re-rotate and re-scan *all*
length-matched pairs at every angle" loop, whose cost was `O(angles × pairs)`.

### Why this fixes the "only some results" problem

The previous code thinned the searched-angle list as `nindex` grew (down to
~10 angles at `nindex = 100`) and silently capped pair matches. The new engine
searches **every** commensurate angle by default; thinning is now strictly
opt-in via `--max-search-angles`.

## Redundancy culling (on by default)

At one twist angle the raw search returns many supercells that beat one another:
a larger cell that *also* carries more strain is never useful. The cull keeps,
per angle, only the `(atom count, strain)` Pareto frontier and drops every
dominated cell. The "same angle, again and again, with almost identical strain"
duplicates disappear, while genuinely different trade-offs (a small slightly
strained cell vs a larger strain-free cell) are both kept.

### Where the cull happens (the main speed-up)

The per-angle cull now runs **inside the builder, in NumPy, before any Python
candidate objects are created** (`build_supercell_candidates`, gated by the
`frontier_only` flag the finder sets whenever culling is on). For each angle it,
in order:

1. drops unphysical cells whose `strain_avg` exceeds `MAX_PHYSICAL_STRAIN`
   (a numerical artifact of the strain metric inverting an ill-conditioned
   near-collinear "sliver" basis — never a real cell);
2. removes `(strain, ratio)` duplicates, keeping the most compact (smallest
   vector-product) representative — this is what suppresses the spurious slivers,
   since a sliver and the genuine compact supercell of the same superlattice
   share strain and ratio and the genuine one has the shorter vectors;
3. keeps only the `(atoms, strain)` Pareto frontier.

Doing this per angle, vectorised, eliminates the old global `O(candidates²)`
de-duplication pass (which dominated the run time at large `nindex`, exploding at
the degenerate angles). The global `deduplicate_candidates` / `pareto_cull`
passes still run afterwards but are now near-instant clean-ups.

`reduce_candidate` Lagrange–Gauss-reduces every reported integer basis, so
skewed bases such as `(-16, 17), (-15, 16)` are reported as their shortest /
most orthogonal equivalent (here `(0, 1), (1, 1)`). Reduction is a unimodular
change of basis, so strain, area ratios, atom count and angle are unchanged.

The result is identical to the previous exhaustive cull except that the
unphysical garbage-strain sliver rows are no longer emitted; every genuine
candidate is preserved.

## Performance knob: `--max-pair-matches`

Highly symmetric ("degenerate") twist angles (0°, 30°, 60° … for a homobilayer)
have hundreds of coincident vectors, which would blow up the `O(M²)` pairing.
`--max-pair-matches K` bounds, per angle, how many coincident vectors are paired.
The cap keeps two pools and unions them:

* the shortest vectors overall (the small cells), and
* the shortest among the **near-exact** coincidences (so a long-period but
  strain-free supercell is never lost behind a crowd of short, strained
  near-coincidences).

Because of the second pool, **every zero-/low-strain commensurate cell is
retained regardless of `K`**; raising `K` only adds more high-strain frontier
cells at crowded angles. Default `K = 128` reproduces the exhaustive culled
result through moderate `nindex`. Pass `--max-pair-matches 0` for an exhaustive
(unbounded, potentially very slow) search.

## CLI options added

| flag | meaning |
|------|---------|
| `--no-cull` | keep every candidate per angle (disable the Pareto frontier cull) |
| `--no-reduce` | report raw integer bases instead of Lagrange–Gauss-reduced ones |
| `--max-pair-matches K` | per-angle pairing bound (`<= 0` = unlimited/exhaustive); default 128 |

Large `nindex` (e.g. 100) benefits from `--workers N`: the angle list is split
across processes. Example: `nindex = 100`, MoS₂/MoS₂, `--workers 8` searches
~7,600 commensurate angles in well under a minute (≈25 s on an 8-core box),
versus the ~10 angles the old cap reported and the couple of minutes the earlier
un-vectorised cull took on the full angle list.
