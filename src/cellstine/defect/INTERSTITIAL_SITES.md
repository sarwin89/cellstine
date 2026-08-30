# Where an interstitial atom can sit

An interstitial site of a structure is a point that is locally as far as
possible from every atom.  CELLSTINE finds those points in closed form, from the
geometry of the atoms themselves, and reports for each one the radius of the
empty sphere it carries, how many atoms lie on that sphere, and whether the
sphere shrinks in *every* direction out of the site or only in some of them.

## Commands

```bash
cellstine defect analyse HOST.vasp                          # the widest holes
cellstine defect analyse HOST.vasp --interstitial-saddles   # and the rest
cellstine defect preview HOST.vasp --interstitial-saddles
cellstine defect generate HOST.vasp --defect-type interstitial \
    --species C --interstitial-saddles --site-ids interstitial_003
```

Both are offered in the guided flow (*Defect Site Analysis* and *Defect
Structure Generation*).  The site table gains an `empty sphere` column, shown
here with the other columns dropped:

```text
 site_id             kind          mult  empty sphere  direct (u, v, w)
 interstitial_001    interstitial    12  1.60 max(4)   ( 0.5000,  0.7500,  0.0000)
 interstitial_002    interstitial    24  1.52 sad(3)   ( 0.5000,  0.8750,  0.8750)
 interstitial_003    interstitial     6  1.44 sad(2)   ( 0.5000,  0.0000,  0.0000)
 interstitial_004    interstitial     8  1.24 sad(2)   ( 0.7500,  0.7500,  0.7500)
```

That is body-centred cubic iron: `max` marks a local maximum, `sad` a saddle,
and the number in brackets is how many atoms touch the sphere.

## Why saddles are not optional

Write `d(x)` for the distance from a point to the nearest atom, images included.
A site is a *critical point* of `d`: a point no small move makes emptier.  Move
along `v`; the distance to a touching atom in direction `u` grows exactly when
`v · u < 0`, so the sphere can be enlarged precisely when some `v` satisfies
`v · u < 0` for every contact direction `u` at once.  By duality that fails
exactly when the origin lies in the convex hull of the contact directions, which
is the classification CELLSTINE applies:

| contacts hold the centre | contact directions | site |
| --- | --- | --- |
| in every direction | origin interior to their hull | **maximum** — a vertex of the Voronoi diagram, four or more atoms |
| along a plane or a line only | origin on the boundary of their hull | **saddle** — two or three atoms |
| not at all | origin outside their hull | not a site: the sphere slides and grows |

Keeping only the maxima is the usual shortcut, and it loses real chemistry:

* **Carbon in ferrite.**  The octahedral site of a body-centred cubic metal,
  `(1/2, 0, 0)` and its five copies, is the midpoint of two second-neighbour
  atoms.  Only those two touch its sphere, so it is a saddle, and a
  Voronoi-vertex search returns the twelve tetrahedral sites and misses it --
  even though carbon and nitrogen in α-iron sit at the octahedral one.
* **Hydrogen at a bond centre.**  In diamond silicon the midpoint of a Si-Si
  bond carries an empty sphere of half the bond length; it is a two-fold saddle.
* **The hollow of a monolayer.**  Graphene has no local maximum at all -- every
  sphere grows by drifting into the vacuum -- but the centre of its hexagon is a
  six-fold saddle whose sphere is as wide as the bond.

## What the search reports, checked against closed forms

Each row was computed by `cellstine.core.voids.find_void_sites` and each radius
is the exact value for that structure; `tests/test_voids.py`,
`tests/test_voids_saddles.py` and `tests/test_defect_interstitial_saddles.py`
assert them.

| structure | site | radius | contacts | kind | per cell |
| --- | --- | --- | --- | --- | --- |
| fcc, constant `a` | octahedral | `a/2` | 6 | maximum | 1 per atom |
| fcc | tetrahedral | `a√3/4` | 4 | maximum | 2 per atom |
| fcc | face centre | `a/√6` | 3 | saddle | 8 per atom |
| fcc | bond centre | `a/(2√2)` | 2 | saddle | 6 per atom |
| bcc, constant `a` | tetrahedral | `a√5/4` | 4 | maximum | 12 |
| bcc | octahedral | `a/2` | 2 | saddle | 6 |
| bcc | bond centre | `a√3/4` | 2 | saddle | 8 |
| hcp, ideal, constant `a` | octahedral | `a/√2` | 6 | maximum | 2 |
| hcp | tetrahedral | `a√(3/8)` | 4 | maximum | 4 |
| diamond, constant `a` | tetrahedral | `a√3/4` | 4 | maximum | 2 |
| diamond | hexagonal | `a√11/8` | 6 | maximum | 4 |
| diamond | bond centre | `a√3/8` | 2 | saddle | 4 |
| rocksalt, constant `a` | tetrahedral | `a√3/4` | 8 | maximum | 2 |
| graphene, bond `b` | hexagon centre | `b` | 6 | saddle | 1 |

## Completeness and cost

The enumeration is exact rather than sampled.  Every centre equidistant from
four atoms is the circumcentre of a tetrahedron of them, every three-fold centre
is the in-plane circumcentre of a triangle, and every two-fold centre is the
midpoint of a pair; all three are enumerated, kept when the sphere through them
holds no atom, and then classified by the rule above.  The classification itself
is decided exactly -- the contact directions either span space, in which case
the cone of directions that hold the sphere is generated by the cross products
of pairs of them, or they span a plane or a line, in which case the answer is an
angular-gap test -- so no direction sampling can miss a narrow escape route.

Nothing is enumerated beyond the reach a site can have: four atoms on a sphere
of radius `r` lie within `2 r` of one another, and the neighbour cutoff is twice
a proven upper bound on the covering radius of the cell
(`RequestProject/CoveringRadius.lean`).

That bound is found adaptively, because how *tight* it is decides how much the
enumeration costs.  The distance to the nearest atom is 1-Lipschitz, so its
value at the centre of a box plus the reach of the box bounds it over the whole
box, while the largest value seen anywhere bounds it from below; a box whose own
bound does not beat that lower bound cannot hold the maximum and is dropped
unrefined, and the rest are cut in half along every axis and looked at again.
The sweep stops when the two bounds meet, or when it has spent its probe budget
-- and stopping early only loosens the answer, never invalidates it.  All of
that is proved in `RequestProject/CoveringBound.lean`: the per-box bound, that
the reach of a box is attained at a corner (which is what `_grid_box_reach`
computes, and is up to `sqrt(3)` smaller than the sum of the half-edges), that
the eight children cover their parent, and that pruning loses no maximum.

Against the uniform grid it replaces this is both tighter and cheaper: on
diamond silicon the bound went from 2.566 A to 2.384 A against a true covering
radius of 2.351 A, and on a disordered 40-atom triclinic cell -- the hard case,
where a short packing distance sits next to a wide hollow -- the whole search
went from 11.0 s to 1.9 s.  Asking for the saddles costs about a third more.

The criterion itself, and the two body-centred cubic sites, are proved in Lean 4
in `RequestProject/CriticalVoids.lean`.
