# Migration paths — `cellstine defect path`, `cellstine adsorbate path`

```bash
cellstine defect path START.vasp END.vasp --images 5
cellstine adsorbate path START.vasp END.vasp --images 5
```

The two spellings are one stage: the engine is `core/pathway.py`, the workflow
stage `core/path_stage.py`, and both the defect and the adsorbate workflow class
carry it. Use whichever group the structures came from.

Either writes the chain of structures a nudged-elastic-band run starts from:

```
00/POSCAR   the initial structure
01/POSCAR   ...
06/POSCAR   the final structure
```

`--images` counts the *intermediate* images, so `--images 5` writes seven
folders. A `path.json` beside the manifest records the whole chain — the
pairing, the per-atom displacements, the spacing of the images, and the closest
contact inside each one — and the run prints them as a table.

The stage was written for a defect hop (a vacancy exchanging with a neighbour,
an interstitial moving to the next void) and for an adsorbate sliding from one
site to the next, but nothing in it is specific to either: any two structures
that share one cell and one composition can be chained, a molecule turning over
included.

## What the chain is

Write the initial structure as a point `x` in configuration space and the final
one as `x + d`. Image `k` of `N + 1` steps is

```text
x_k = x + (k / (N + 1)) d.
```

Consecutive images are then exactly `‖d‖ / (N + 1)` apart and the chain is
`‖d‖` long — the even spacing a band expects, and no image is a detour. Both
facts are proved in `aristotle-lean-reference/RequestProject/MigrationPath.lean`
(`Cellstine.pathImage_step_norm`, `Cellstine.pathImage_total_length`), together
with the statement that no chain between the same endpoints can be shorter
(`Cellstine.norm_sub_le_chain_length`).

Everything therefore turns on getting `d` right, and `d` is *not* the difference
of the two coordinate lists.

## Which atom becomes which

A POSCAR is a list of atoms, and nothing in the file says that atom 7 of the
initial structure is atom 7 of the final one. Reading the two files in order is
only right when the caller wrote them that way; a structure that came out of a
relaxation, a defect generator, or another program need not agree.

CELLSTINE therefore *pairs* the atoms, choosing the pairing that makes the path
shortest. With `c[i][j]` the squared minimum-image distance from atom `i` of the
initial structure to atom `j` of the final one, the path length is

```text
‖d‖² = Σ_i c[i][σ(i)]
```

for the pairing `σ`, so the shortest path is the minimum-cost perfect matching —
a linear assignment problem. It is solved exactly, one species at a time (an
aluminium atom may only become an aluminium atom), by the Jonker–Volgenant
shortest-augmenting-path algorithm in `core/pathway.py`; a 200-atom species
block takes a few tens of milliseconds.

**The answer comes with a proof.** The solver returns potentials `u`, `v` with

```text
u_i + v_j ≤ c[i][j]   for every pair of the same species,
u_i + v_σ(i) = c[i][σ(i)]  on the pairing itself.
```

The first line makes `Σ u + Σ v` a lower bound on the cost of every pairing, and
the second says the bound is reached, so `σ` is optimal. That implication is
`Cellstine.assignment_cost_le_of_labelled_certificate`, and
`Assignment.certificate_error` measures how far the two lines are from holding —
it is reported in every run, and is zero to floating-point noise. The pairing is
not "what the solver said": it is checkable, and checked.

Pass `--no-match` to keep the file order instead. That is the right choice when
the two files already correspond atom for atom and the order matters downstream.

## Which periodic image it moves to

An atom near a cell face may be closest to an image of its partner in the *next*
cell. Every displacement is therefore taken as the exact minimum image, through
`minimum_image_fractional`, never as the difference of two wrapped coordinates:
an atom at `u = 0.95` whose partner sits at `u = 0.05` moves `+0.10`, not
`−0.90`, and its intermediate images cross the face.

Choosing the shortest image of each atom separately also minimises the total,
so the whole path is the shortest one available
(`Cellstine.sum_norm_sq_min_le`).

## What the run tells you

| reported | meaning |
| --- | --- |
| `path length` | `‖d‖`, the straight-line distance between the endpoints in configuration space |
| `image spacing` | `‖d‖ / (N + 1)`, and the table lists the measured step of every pair |
| `moving atoms` | how many atoms move at all — a clean hop moves one |
| `maximum atom displacement` | the longest single-atom step |
| `closest contact` | the shortest interatomic distance in each image |
| `matching certificate error` | the residual of the optimality proof above |

Three things are called out as warnings:

* **a pinched image.** A straight line between two relaxed structures can drive
  two atoms much closer than either endpoint does. The image and the distance
  are named, so the chain can be replaced by one through an intermediate
  structure. If a straight line puts two atoms *on the same site* the chain is
  refused outright rather than written.
* **a step longer than half the shortest lattice vector.** The chain takes the
  shortest periodic image, which need not be the hop that was meant.
* **a re-paired final structure.** When the matching is not the file order, the
  number of atoms it moved is reported; the images are always written in the
  atom order of the initial structure, so the chain reads as one structure
  evolving.

## A worked example

A vacancy hop in a 2×2×2 aluminium cell — the vacancy at the origin exchanging
with its neighbour at `(0, 0, ½)`:

```text
 image  reaction coordinate  travelled (A)   step (A)  closest contact (A)
     0               0.0000         0.0000          -               2.8638
     1               0.1667         0.4773     0.4773               2.6575
     2               0.3333         0.9546     0.4773               2.5256
     3               0.5000         1.4319     0.4773               2.4801
     4               0.6667         1.9092     0.4773               2.5256
     5               0.8333         2.3865     0.4773               2.6575
     6               1.0000         2.8638     0.4773               2.8638
```

The path length is `a / √2 = 2.864 Å`, the nearest-neighbour distance of the
face-centred cubic lattice; one atom moves; the closest approach is at the
half-way image, and the chain is symmetric about it, as the symmetry of the hop
requires.
