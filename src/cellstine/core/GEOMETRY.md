# `cellstine.core.geometry` — the periodic geometry core

Every part of CELLSTINE that asks a geometric question about a periodic cell —
"how far apart are these two atoms?", "which atoms are within 4 Å of this one?",
"does this symmetry operation map the structure onto itself?", "is this
interstitial site really empty?" — goes through this one module. Collecting the
answers in one place is what makes them consistent across the library, and it is
what makes it worth doing them exactly and quickly.

All lattices are **row** lattices: `lattice[i]` is the Cartesian vector of basis
vector *i*, and a site with fractional coordinates `x` sits at `x @ lattice`.
That is the convention of `cellstine.io.models.StructureRecord` and of POSCAR.

## The three facts everything rests on

The module is small because three inequalities do all the work. All three are
stated and proved in Lean, in `aristotle-lean-reference/RequestProject/PeriodicGeometry.lean`.

### 1. The reach bound

Let `b_i` be the rows of `inv(lattice).T` (the reciprocal basis) and
`d_i = 1 / ‖b_i‖` the spacing of the lattice planes normal to axis *i*. A
displacement with fractional coordinates `f` satisfies `f_i = b_i · (f @ lattice)`,
so Cauchy–Schwarz gives

```
|f_i| ≤ ‖f @ lattice‖ / d_i.
```

A displacement shorter than `r` therefore cannot reach further than `r / d_i`
cells along axis *i*. Every image enumeration in the library — `image_shift_reach`,
`atom_images`, `neighbour_images` — sizes its box from this, so none of them can
miss an image. What is *not* enough is the naive `ceil(r / ‖lattice[i]‖)`, which
undercounts badly in a skewed cell.

* Lean: `Cellstine.abs_coord_le_euclidNorm_mul_reciprocal`
* Lean (cutoff form): `Cellstine.abs_shift_le_of_cartesian_le`

### 2. Rounding is not the minimum image

The textbook "minimum image convention" — replace the fractional displacement
`f` by `f - rint(f)` — is **wrong in a non-orthogonal cell**. In a hexagonal
cell it can overstate a distance by more than 30 %; in a 3×3 graphene supercell
18 of 324 pair distances came out too large, by up to 1.92 Å, and a 4 Å
neighbour search consequently found only 126 of the 135 genuine pairs.

What `f - rint(f)` *is* good for is an upper bound `d0` on the true shortest
distance. Combining that with the reach bound confines the shortest image to a
small box of lattice shifts, and `minimum_image_displacements` searches that box
exhaustively. It does so in a Delaunay-reduced basis of the same lattice (cached
per lattice), where the box is tiny; the Cartesian answer does not depend on
which basis of the lattice the search uses.

* Lean (the counterexample): `Cellstine.rounding_is_not_the_minimum_image`
* Lean (the box is complete): `Cellstine.abs_shift_le_of_le_guess`

### 3. Bucketing is complete

Two points closer than the bucket pitch differ by at most one bucket along each
axis, in Cartesian coordinates for `CartesianGrid` and in fractional coordinates
for `PeriodicSiteIndex` provided the bins are no finer than the tolerance
allows. So the 27 buckets around a query hold all of its near neighbours, and
nothing can hide in a bucket that is never scanned. This is what turns the
`O(n²)` distance matrices the library used to build into `O(n)` scans.

* Lean: `Cellstine.abs_cartesian_bucket_diff_le_one`,
  `Cellstine.abs_fractional_bucket_diff_le_one`

## What the module provides

| Name | Purpose |
| --- | --- |
| `as_lattice` | Validate and normalise a 3×3 row lattice. |
| `wrap_to_cell`, `wrap_fractional` | Fractional coordinates into `[0, 1)` and into `[-1/2, 1/2)`. |
| `axis_spacings`, `reciprocal_norms` | Interplanar spacings `d_i` and their reciprocals. |
| `niggli_reduce`, `delaunay_reduce` | Reduced bases of the same lattice, with the integer transformation. |
| `image_shift_reach`, `lattice_shifts`, `atom_images` | Complete enumeration of periodic images within a cutoff. |
| `minimum_image_displacements`, `minimum_image_distances`, `pairwise_minimum_image_distances` | Exact shortest images and distances. |
| `PeriodicSiteIndex` | `O(n)` periodic site matching with a tolerance, optional species labels, and `prefer_lowest` for collapsing coincident sites. |
| `CartesianGrid` | Cell list over a fixed point set; one bucket read per query. |
| `nearest_point_distances` | Exact nearest-point distances, with a doubling pitch. |
| `neighbour_images` | Per-atom neighbour images within a cutoff. |

`core.symmetry3d` re-exports `niggli_reduce` and `delaunay_reduce`, so the older
import path still works.

## Consumers

* `core/symmetry3d.py` — site permutations, orbits, primitive cells.
  `analyse_symmetry` on 216-atom silicon: 1.20 s → 0.17 s.
* `core/voids.py` — neighbour lists, empty-sphere estimates, void merging.
  `find_void_sites` on 64-atom silicon: 3.03 s → 0.23 s.
* `defect/analysis.py` — neighbour separations, distance fingerprints and
  symmetry grouping; this is where the rounding bug had been changing results.
* `interface/surface/surface_sites.py` and `interface/surface/surface_cell.py` —
  slab site deduplication and the test that a translation maps a structure onto
  itself.

## Testing

`tests/test_geometry.py` checks the exact minimum image against brute-force
enumeration on cubic, hexagonal and sheared cells, pins the rounding
counterexample, checks that the image-shift reach is complete, and checks the
bucket structures against exhaustive scans. Refactors of the consumers above
were additionally verified to reproduce the previous results bit-for-bit on
silicon, fcc aluminium, hBN, SrTiO₃ and random cells.
