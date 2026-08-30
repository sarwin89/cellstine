# `cellstine.core.contacts` — what the built structure actually looks like

A builder is asked for a *height*, a *gap* or a *site*. None of those three
numbers is the distance between the atoms it puts next to each other, and it is
the distance between the atoms that decides whether the cell is worth an SCF
cycle at all. This module measures that distance and says, in words, when it is
too short.

Everything here is a statement about the finished structure, not about the
search that produced it: the input is a lattice and two groups of atoms, and the
output is the closest approach between them, the pair of atoms that makes it,
and the sum of their covalent radii to compare it with.

## Why a height is not a contact

A placement puts an atom `h` along the outward normal `u` above a site `q` of the
substrate. Every substrate atom `x` satisfies

```
h = u · (p − q) ≤ ‖p − x‖ + u · (x − q)
```

so the height is a *guaranteed clearance*: no substrate atom in the surface plane
can be closer than `h`. That is `Cellstine.height_le_euclidDist` and
`Cellstine.height_le_contactDistance` in `aristotle-lean-reference/RequestProject/ContactDistance.lean`.

The clearance is only attained when the atom is placed directly over a
substrate atom. Over a bridge or a hollow site the nearest substrate atom is
`d > 0` to one side, and

```
‖p − x‖ = sqrt(h² + d²) > h,
```

proved as `Cellstine.euclidDist_offset_eq_sqrt` and
`Cellstine.lt_euclidDist_offset`. A 2.0 Å hollow-site adsorption is therefore not
a 2.0 Å bond; the measured contact can easily be 2.5 Å, and a user who wanted a
bond length got neither an error nor the number they wanted. Both numbers are
now reported, and a note says which is which.

## The three contacts that get measured

* **cross-group** — `closest_contact(lattice, first_direct, second_direct)`.
  The minimum over all pairs of the minimum-image distance, so a contact made
  through a cell face counts. It is used for molecule↔substrate
  (`adsorbate`), layer↔layer (`moire`, `interface`) and defect↔host (`defect`).
* **self-image** — `self_image_contact(lattice, group_direct)`. A molecule in a
  periodic cell also sees its own translated copies, and a cell that is too
  small makes a chain rather than an isolated molecule. This is *not* a
  minimum-image calculation: the copies of a molecule of diameter `D` are found
  by enumerating the lattice translations with `‖t‖ ≤ λ₁ + D`, which is exact
  because `Cellstine.lt_euclidDist_add_of_lt` shows a longer translation cannot
  beat the shortest one already found. `Cellstine.molecule_image_separation` is
  the companion bound `λ₁ − D`.
* **requested vs measured** — when a builder was given a target clearance, the
  note records that the measured contact is the larger number.

The molecule is unwrapped before its diameter is taken
(`unwrap_group`, `group_diameter`): a molecule written across a cell boundary
has fractional coordinates that jump by one, and a naive diameter would be the
size of the cell rather than the size of the molecule.

## When a contact is called too short

The comparison is against the sum of the two covalent radii `r₁ + r₂` from
`cellstine.core.elements`, which is the length of a normal single bond between
those two elements.

| ratio `d / (r₁ + r₂)` | verdict |
| --- | --- |
| `< 0.75` | the atoms overlap; the cell will not converge sensibly |
| `< 0.90` | shorter than a single bond; flagged as chemistry to check |
| otherwise | no note |

The thresholds are `OVERLAP_RATIO` and `BOND_RATIO`. They are deliberately
generous: a genuine short bond (an H-bonded contact, a strained ring) should not
be rewritten by the tool, only mentioned. Nothing here changes a structure —
every note is advisory and the written POSCAR is exactly what was asked for.

## Where it appears

| workflow | key in the summary |
| --- | --- |
| `adsorbate place`, `adsorbate move` | `closest_contact`, `closest_contact_pair`, `molecule_image_distance` |
| `moire make` | `closest_interlayer_contact` |
| `interface build` | `closest_contact`, `closest_contact_pair` |
| `defect generate` | `closest_defect_contact`, `closest_defect_contact_pair` |

Every one of them also fills `warnings`, which
`cellstine.core.report.format_result` prints last, under its own `Warnings:`
heading, rather than between two numbers where it would be missed.

## Element symbols

Contacts need a covalent radius, and a radius needs an element. Structure files
in the wild label atoms `C`, but also `C_surf`, `Fe2+`, `O1` and `Mo/W`.
`cellstine.core.elements.element_symbol` reads the leading alphabetic run before
the first separator and then matches the whole token, its first two letters, and
finally its first letter. The order matters: taking letters greedily turns
`C_surf` into `Cs`, caesium, whose mass is ten times carbon's and whose covalent
radius is twice as large.

## Formal statements

`aristotle-lean-reference/RequestProject/ContactDistance.lean` proves, with no axioms beyond Mathlib's:

| claim | statement |
| --- | --- |
| a placement reaches exactly the requested height | `Cellstine.placedHeights_inf'` |
| the height is a guaranteed clearance | `Cellstine.height_le_euclidDist`, `Cellstine.height_le_contactDistance` |
| an off-site atom is further away than its height | `Cellstine.euclidDist_offset_eq_sqrt`, `Cellstine.lt_euclidDist_offset` |
| placing a molecule does not deform it | `Cellstine.euclidDist_orthogonal_add` |
| the self-image search may stop at `U + D` | `Cellstine.lt_euclidDist_add_of_lt` |
| a molecule of diameter `D` keeps `λ₁ − D` from its images | `Cellstine.molecule_image_separation` |

`tests/test_contacts.py` checks the implementation against brute-force
enumeration over a 7³ block of translations in skewed cells, where the
minimum-image shortcut is not valid, so the fast path is verified against the
slow one rather than against itself.
