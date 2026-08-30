# Slabs and vacuum-free cells

`cellstine surface build BULK.vasp --miller HKL --layers N --vacuum V` cuts
the primitive `(hkl)` cell of a crystal and stacks `N` atomic layers of it. The
cell is written in the canonical surface frame: both in-plane vectors lie in the
Cartesian `xy` plane with `a` along `+x`, and `c` is the plane normal along `+z`.

## What the two modes mean

| `--vacuum` | what the cell is | its height |
| --- | --- | --- |
| `V > 0` | a **slab**: `N` layers with an empty gap | span of the layers `+ V` |
| `0` | the **bulk** crystal in that orientation | first layer to its first repeat |

With a vacuum the requested value is the gap that is left: the distance from the
topmost atom to the bottom atom of the cell above is exactly `V`, which is the
quantity a plane-wave calculation has to converge.

With no vacuum the cell is not a slab with its ends glued together but the bulk
crystal itself, cut so that `c` is the plane normal. Its height therefore runs
from the first layer to the first *repeat* of that layer, not to the last layer
of the stack, so the cell tiles space with the bulk atomic density and no layer
is counted twice across the boundary. That is the cell to hand to
`interface build --bottom-kind bulk`, or to relax as a reference bulk in the same
orientation as a slab.

## Why a vacuum-free cell needs a whole number of periods

Layers of a close-packed face are laterally offset from one another: an `fcc`
`(111)` stack runs `ABCABC`, so translating it along the normal maps the crystal
onto itself only after three layers. Since `c` is perpendicular to the plane,
the stack can only close up on itself after a whole number of such stacking
periods; `4` layers of `ABCABC` would emit an `ABCA` cell, which is a stacking
fault rather than aluminium.

CELLSTINE detects the period from the layer contents themselves -- it compares
the in-plane positions and the spacings of the layers, not just their letters --
and refuses a request that would silently produce a fault:

```text
$ cellstine surface build Al.vasp --miller 111 --layers 4 --vacuum 0
Error: a vacuum-free (1 1 1) cell repeats every 3 atomic layers, so 4 layers
would emit a stacking fault; ask for 3 or 6 layers, or for a vacuum
```

Some faces have no vacuum-free cell at all in this frame. A hexagonal crystal
whose `c/a` ratio is irrational has `(111)` planes whose lateral offsets never
sum to a lattice translation, so no perpendicular `c` closes the stack; the
request is refused with that reason, and asking for a vacuum builds the slab as
usual.

The periods are a property of the face, not of CELLSTINE: `2` for `fcc` `(100)`
and `(110)`, `3` for `fcc` `(111)`, `4` for diamond `(100)`, `6` for rock-salt
`(111)`, `2` for `hcp` `(0001)`.

## Checked, not asserted

The rule is proved in Lean 4 / Mathlib in the external reference at
`aristotle-lean-reference/RequestProject/StackingPeriod.lean`:
that a stack of least period `p` closes up after `L` layers exactly when `p`
divides `L` (`Cellstine.isStackPeriod_iff_dvd`), that an `ABCABC` stack has least
period three so four layers of it are a fault
(`Cellstine.fcc111_isStackPeriod_iff`), that a cell of any whole number of
periods holds the bulk atomic density (`Cellstine.stack_density`), and that
stopping the cell at its last layer instead of at the repeat of its first --
what a slab height would do -- reports a density strictly above the bulk one
(`Cellstine.density_span_lt_density_repeat`).

## What the stacking report does instead

`analyse_primitive_surface` probes a fixed number of layers to read off the
stacking sequence (`ABCABC`, `ABAB`, ...) and its shortest repeating prefix. A
probe is a report, not a structure to compute with, so it is free to stop
part-way through a period and is not held to the rule above.

## What the slab exposes

Two further properties decide whether a cut of a *compound* is a usable model of
a surface, and neither of them is visible in the atom count. `termination.py`
reports both, and `surface build` prints them next to the structure it
wrote.

**Stoichiometry.** A slab need not hold a whole number of formula units. The
`(1 0 0)` cut of rocksalt does, whatever its thickness, because every plane
holds one cation and one anion. The `(1 1 1)` cut alternates pure planes, so an
odd number of layers always carries one species in excess, and a perovskite
`(0 0 1)` cut is short of a formula unit whichever way it is terminated. The
formula unit is the bulk counts divided by their greatest common divisor, so the
test is exact integer arithmetic; the report gives the leftover per species.
A total energy of a non-stoichiometric cell only means something against a
chemical potential for the excess atoms.

**The two faces.** A slab has two of them, and only if they are the same
termination does the cell model one surface twice. The `(1 1 1)` cut of rocksalt
puts all the cations on one face and all the anions on the other: the cell
carries a dipole along the normal, and a plane-wave calculation needs a dipole
correction or a symmetric slab before its energy means anything. This is
Tasker's type III surface, and the report names it by comparing the species
multisets of the outermost layer on either side.

An elemental crystal passes both tests always, so its report carries no notes.
