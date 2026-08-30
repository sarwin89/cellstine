# `cellstine.core.kpath` — the Brillouin zone and the band-structure path

A band structure is a plot of the eigenvalues along a path through the Brillouin
zone. The path is not free: it is supposed to visit the points and lines that the
symmetry of the crystal singles out, and the segment lengths it reports are the
abscissa of the plot, so they have to be right.

CELLSTINE does not look the path up in a table. It **derives** the special points
from the symmetry of the lattice, and only the *names* — and, for the Bravais
types that have a conventional order of visits, that order — come from
convention. `cellstine.core.brillouin` supplies the zone itself, and
`cellstine.core.strata` the linear algebra of the symmetry action: little
co-groups, fixed spaces, the exhaustive grid sweep and the segment
classification. `cellstine.core.kpath` is then only the naming and the walking.

```
cellstine symmetry kpath POSCAR --spacing 0.03
cellstine symmetry kpath POSCAR --divisions 40 --path GAMMA-X-M-GAMMA-R
cellstine symmetry kpath POSCAR --no-standard      # derive the walk as well
```

Conventions follow `core/RECIPROCAL.md`: lattices are **row** lattices, a
wavevector is a row `k` of *fractional* reciprocal coordinates whose Cartesian
value is `k @ B` with `B = 2 pi inv(A).T`, and a crystal operation `x -> W x + w`
acts on it by the integer matrix `W^-1` on the right, `k -> k W^-1`. Time
reversal is `k -> -k`. Write `P` for the group of integer matrices so obtained.

## The Wigner–Seitz cell

The first Brillouin zone is the Wigner–Seitz cell of the reciprocal lattice:

```
WS = { x : |x| <= |x - g| for every reciprocal lattice vector g != 0 }
```

Squaring the inequality turns each condition into a half space,

```
|x| <= |x - g|   <=>   <x, g> <= |g|^2 / 2,
```

so the zone is an intersection of half spaces — a convex polytope, symmetric
under `x -> -x` and containing the ball of radius `min |g| / 2`. Only finitely
many `g` matter: a `g` longer than twice the circumradius of the candidate cell
cuts nothing, so the search shell is grown until it stops removing faces, and the
half spaces that turn out not to touch the polytope are dropped. The vertices,
edges and faces are then built from the surviving planes, and the result is
checked, not assumed: Euler's formula `V - E + F = 2`, a volume equal to
`det(B)`, and the tiling of space by the translates.

* Lean: `Cellstine.wignerSeitz_eq_halfSpaces`, `Cellstine.convex_wignerSeitz`,
  `Cellstine.neg_mem_wignerSeitz`, `Cellstine.mem_wignerSeitz_of_norm_lt_half`,
  `Cellstine.norm_half_le_of_mem_bisector`, `Cellstine.sub_mem_wignerSeitz_of_min`,
  `Cellstine.exists_isLeast_boundaryScale`, `Cellstine.det_reciprocalBasis`

The last of those is what makes a ray exit the zone exactly once: along a ray the
constraints are linear, so the boundary is reached at the least positive scale
that saturates one of them, and beyond it the point is outside. That is the
routine `brillouin` uses to draw the zone and to place the ends of the symmetry
lines.

## Strata: what a point lies on

The little co-group of a wavevector is

```
L(k) = { M in P : k M = k  mod Z^3 },
```

and the wavevectors that share it form an affine subspace through `k` with
direction space

```
V(k) = { v : v M = v for every M in L(k) }.
```

`dim V(k)` classifies the point: `0` is an isolated high-symmetry point, `1` a
symmetry line, `2` a mirror plane, `3` a generic point. This dimension, and not a
list of letters, is what the module computes.

**A finite search finds every isolated point.** If `V(k) = 0` then the average
`(1/|L|) sum_M M` projects onto `V(k)`, hence vanishes, so

```
sum_{M in L} k (M - I) = -|L| k
```

is an integer vector: a zero-dimensional stratum has rational coordinates whose
denominator divides `|L(k)|`. Every subgroup order of a crystallographic point
group extended by time reversal divides `48`, so a sweep of the grid of
denominator `48` — `GRID_DENOMINATOR` — is **exhaustive**, not a sampling.

* Lean: `Cellstine.card_smul_isIntegerVector`,
  `Cellstine.isIntegerVector_nsmul_of_dvd`, `Cellstine.FixesModLattice`,
  `Cellstine.fixesModLattice_add_int`, `Cellstine.fixesModLattice_add_fixed`

The sweep is `48**3` points, so it is done with linear algebra in bulk rather
than point by point: the "which operations fix me" boolean rows are computed as
one `einsum`, packed into bytes and grouped, and one fixed-space SVD is taken per
*distinct* little co-group. Points sharing a co-group share a stratum dimension,
and there are only a few dozen distinct co-groups, so the number of SVDs drops
from a hundred thousand to a handful. The result is memoised per point group.

## Naming

The familiar letters belong to the *conventional* cell, so the tabulated
coordinates of the Bravais type (from `cellstine.core.bravais`) are carried into
the primitive reciprocal basis by `inv(to_primitive.T)` and matched against each
point *as a whole symmetry orbit*. Anything no standard name covers is named
`P1`, `P2`, … in order of increasing `|k|` and flagged as derived.

Two names may sit on one orbit. The fcc `K = (3/4, 3/8, 3/8)` and
`U = (5/8, 5/8, 1/4)` differ by a reciprocal lattice vector followed by a point
operation, so a band structure cannot tell them apart and they are reported as
aliases. They are nonetheless different **places**: `U` sits on a square face and
`K` on the edge between two hexagons, and the lines through them are different
lines. A path naming both must therefore walk through both, and when a label is
resolved to an actual zone point the candidates are restricted to the copies that
are point-group images of the *tabulated* target as a wavevector — a lattice
translation is not allowed to move it. Without that restriction `U` collapses
onto `K` and the standard fcc path silently walks four duplicated segments.

The standard fcc walk is `GAMMA-X-W-K-GAMMA-L-U-W-L-K|U-X`, and all ten of its
segment lengths now agree with the published table.

## Building the walk

Nodes are the special points taken as actual points of the zone, one node per
copy. Two nodes are joined by an edge when the midpoint between them lies on a
symmetry line — the ends being on it too, this makes the whole segment part of
the line — and no third node lies between them on it, so an edge is one piece of
line and not a shortcut across several. The pairwise test is blocked NumPy rather
than a triple Python loop.

A derived walk is then grown from `GAMMA`, always stepping to the nearest point
whose name has not yet been visited, along symmetry lines when a chain of them
reaches it (shortest path by Dijkstra over the edges) and in a straight line when
none does. A break `|` is written whenever the walk has to jump. An explicit
`--path` is followed instead when one is given, and the tabulated path is used by
default for the Bravais types that have one.

Because a walk may visit two copies of one orbit — the `L` and `L1` of the usual
tables — every name it uses is listed with the coordinates of the copy actually
visited.

## Classifying the segments

A segment is classified by **where its interior sits**, never by the names of its
ends: the interior is sampled at three interior fractions, the stratum dimension
is taken at each, and the largest wins (a segment may cross a higher-symmetry
point without becoming one). Dimension `1` is reported as a symmetry line, `2` as
a mirror plane, `3` as a plain chord; a sample of dimension `0` is read as a line,
since a segment cannot be a point. `BandPath.segment_strata` carries the numbers,
`segment_symmetry` the `dimension == 1` shorthand, and both the table and
`kpath.json` report them.

This is what a name-based test cannot do. Under aliasing the segment `U -> X` is
named with a label that never matches the alias table, so it used to be reported
as a plain chord; it is in fact a symmetry line, and `W -> K`, `L -> U`, `U -> W`
and `L -> K` lie in mirror planes.

## Sampling and what is written

`divisions_for_spacing(s)` returns the per-segment division count, which a
line-mode file carries once for all segments and is therefore set by the longest
one:

```
n = max(2, ceil(longest / s + 1)),
```

so no step exceeds `s`; shorter segments are then sampled more finely, never
less, and a warning is printed when a segment is shorter than
`SHORT_SEGMENT_FRACTION` of the longest. `sample(n)` returns the fractional
points, the cumulative Cartesian distance — the plot abscissa, which does not
advance across a break — and the tick labels.

The stage writes a line-mode `KPOINTS` through `cellstine.io.kpoints`, and a
`kpath.json` recording the Bravais symbol, the path string and where it came
from, every point with its little-group order and stratum dimension, every
segment with its length and stratum, the tick positions, and the zone summary.

* Lean: `Cellstine.pathPoint`, `Cellstine.pathPoint_last`,
  `Cellstine.dist_pathPoint_succ`, `Cellstine.sum_dist_pathPoint`,
  `Cellstine.vecMul_interpolate`

## What is checked

`tests/test_brillouin.py` checks the zone against its definition: the half-space
form, the Euler characteristic, the volume against `det(B)`, the inradius, that
the translates tile space and that a ray leaves exactly once.

`tests/test_kpath.py` checks that every special point is fixed by exactly the
operations counted in its little group and that its stratum dimension is the
dimension of the fixed space; that an isolated point has a denominator dividing
its little-group order, which is the theorem the search grid rests on; that the
points lie in the zone; that the cubic, fcc, bcc, hexagonal and tetragonal
coordinates and *all* segment lengths agree with the standard tables; that `U`
and `K` are one orbit but two places; that no segment is walked twice; that each
segment's reported stratum is the one seen at generic interior points and that
every operation fixing an interior point fixes the direction walked, so a segment
really lies in the stratum it claims; and that nothing depends on how the cell was
written down, by rotating it.
