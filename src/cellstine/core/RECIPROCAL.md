# `cellstine.core.reciprocal` — reciprocal lattices and k-point meshes

A plane-wave calculation needs two things from a structure: the cell, and a
sampling of its Brillouin zone. CELLSTINE writes the cell everywhere; this
module writes the sampling, and `symmetry kpoints` is the command that puts the
two together.

All lattices are **row** lattices: `lattice[i]` is the Cartesian vector of basis
vector *i*, so a site with fractional coordinates `x` sits at `x @ lattice`.
That is the convention of `cellstine.io.models.StructureRecord` and of POSCAR.

## The reciprocal basis

```
a_i . b_j = 2 pi delta_ij,      B = 2 pi inv(A).T
```

so a plane wave `exp(i G . r)` with `G = m @ B` and integer `m` has the
periodicity of the cell. A wavevector is carried in **fractional** reciprocal
coordinates `k` — the Cartesian vector is `k @ B` — and the first Brillouin zone
is `k` in `[-1/2, 1/2)`.

The inverse is taken through a solve rather than an explicit matrix inverse,
which is both faster and better conditioned for a skewed cell.

* Lean: `Cellstine.mul_reciprocalBasis_transpose`,
  `Cellstine.reciprocalBasis_reciprocalBasis`

## Choosing the divisions

`mesh_divisions_for_spacing(lattice, s)` returns

```
n_i = max(1, ceil(|b_i| / s))
```

with `s` a largest allowed step in reciprocal space, in inverse angstrom and in
the `2 pi` convention — the quantity VASP calls `KSPACING`. Every step is then
at most `s` long, and one division fewer along any axis would overshoot it, so
these are the smallest divisions that meet the request.

Because the divisions are read off the *cell*, the same `s` automatically
samples a supercell more coarsely: a `2x2x2` supercell of a crystal gets half
the divisions and the same number of points per unit volume of its own (eight
times smaller) Brillouin zone. That is the sense in which a defect supercell, a
moire supercell and the primitive cell it came from are sampled equally well.

* Lean: `Cellstine.mesh_spacing_le`, `Cellstine.lt_mesh_spacing_of_pred`,
  `Cellstine.dist_meshPoint_succ`
* Lean (the folding relation): `Cellstine.reciprocalBasis_supercell`,
  `Cellstine.reciprocal_mem_supercell_lattice`,
  `Cellstine.fold_index_eq_natAbs_det`

A slab is the exception worth knowing about: its bands do not disperse along the
surface normal, so sampling that direction only multiplies the cost. `--surface`
pins the third division to one.

## Gamma-centred and Monkhorst-Pack

The mesh is

```
k(i) = ((i_1 + s_1) / n_1, (i_2 + s_2) / n_2, (i_3 + s_3) / n_3)
```

with `s = 0` the Gamma-centred mesh and `s_j = 1/2` on even axes the original
Monkhorst-Pack choice, which centres the mesh on the zone rather than on Gamma.
For an odd division the two coincide. The shifted mesh is usually smaller after
reduction — for `m-3m` the counts are the textbook `(n+2)(n+4)(n+6)/48` and
`n(n+2)(n+4)/48` — but it need not carry the whole point group, and when it does
not, the report says so instead of quietly reducing by less.

## The reduction is exact

Two facts make the reduction integer arithmetic rather than a tolerance-driven
search.

1. A mesh point is a rational vector whose denominators divide `2 n_j`, so every
   point is an integer vector over the common denominator
   `D = lcm(2 n_1, 2 n_2, 2 n_3)` and two points can be compared exactly.
2. A crystal symmetry `x -> W x + w` acts on fractional reciprocal coordinates
   by the **integer** matrix `W^-1` on the right, `k -> k W^-1`, because `W` is
   unimodular. That is exactly the map that leaves the plane-wave phase `k . x`
   alone, which is why the two wavevectors may be identified.

So the reduction is a hash of integer triples modulo `D`, with no distance
tolerance anywhere, and the weights it reports are exact orbit sizes that add up
to the size of the unreduced mesh.

The operations used are the rotation parts of the space group of the *decorated*
cell, so a cell keeps only the symmetry its atoms actually have; the translation
parts change a phase, not an orbit. Time reversal `k -> -k` is used unless it is
turned off. Rotations handed in are closed into their group first, so a set of
generators reduces exactly as well as the whole group, and an operation that
does not map the mesh onto itself is dropped and counted.

* Lean: `Cellstine.dotProduct_vecMul_inv_mulVec`,
  `Cellstine.meshSet_vecMul_mem`, `Cellstine.meshSet_neg_mem`,
  `Cellstine.meshSet_vecMul_bijective`, `Cellstine.sum_orbit_card_eq_card`,
  `Cellstine.mesh_card_eq_prod`

## What is written

`cellstine.io.kpoints` writes either layout of a VASP KPOINTS file: the
automatic mesh line, or the explicit irreducible list with weights. The default
writes the list exactly when the reduction removed points, since an automatic
line would then discard the saving. Weights are written as integers, so a
reduced mesh round-trips through the file exactly.

```
cellstine symmetry kpoints POSCAR --spacing 0.25
cellstine symmetry kpoints POSCAR --divisions 6,6,1 --mesh monkhorst
cellstine symmetry kpoints slab.vasp --spacing 0.2 --surface
```

## What is checked

`tests/test_reciprocal.py` and `tests/test_kpoints_cli.py` check the duality of
the basis, the minimality of the divisions, the reduction against a brute-force
orbit computation with an independent floating-point implementation, the
textbook cubic irreducible counts, that every mesh point is equivalent to
exactly one representative and no two representatives are equivalent, that the
weights add up, the folding relation between a cell and its supercell, and the
KPOINTS round trip.
