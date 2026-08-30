# Stacking order and interface registry

CELLSTINE can reverse the close-packed stacking order of a slab -- `ABCABC`
against `CBACBA` -- and set which layer meets which across the contact, while
reporting only the options that are genuinely different structures.

## Commands

List the distinct ways two close-packed slabs can be put in contact:

```bash
cellstine interface registries BOTTOM.vasp TOP.vasp
```

```text
idx  bottom  top                contact  kind        note
---  ------  -----------------  -------  ----------  ----
1    ABCABC  ABCABC             C-C      eclipsed
2    ABCABC  ABCABC             C-A      fcc_hollow
3    ABCABC  ABCABC             C-B      hcp_hollow
4    ABCABC  ACBACB (mirrored)  C-C      eclipsed
5    ABCABC  ACBACB (mirrored)  C-A      fcc_hollow
```

Build one of them:

```bash
cellstine interface build BOTTOM.vasp TOP.vasp \
    --gap 2.34 --top-stacking cba --registry hcp
```

* `--bottom-kind`/`--top-kind` -- `auto` by default: a cell with no vacuum
  along its normal is read as bulk and cut into a slab with `--bottom-miller`
  and `--bottom-layers`, and a cell that already has vacuum is used as it is.
  Say `bulk`, `slab` or `surface` to force the reading.
* `--bottom-stacking {keep,mirror}` -- keep the bottom slab as it arrives, or
  reflect it, which turns `ABCABC` into `CBACBA`.
* `--top-stacking {keep,mirror,abc,cba}` -- `abc` stacks the top slab the same
  way as the bottom one, `cba` the opposite way (a twin across the contact);
  the slab is reflected only if it is not already stacked that way.
* `--registry` -- a contact such as `C-A`, a kind (`eclipsed`, `fcc`, `hcp`),
  or an index from `interface registries`.  Only the difference of the two
  letters is read, so `A-A`, `B-B` and `C-C` all name the eclipsed contact.
* `--include-equivalent` -- number and list the removed combinations too.

The same questions are offered by the interactive mode, under
*Interface workflow → Build a slab-on-slab interface* and
*→ List the distinct stacking options of two slabs*.

Both slabs must share their in-plane cell for a contact to be meaningful, so
`--registry` is refused together with `--match`; the two stacking senses
`keep` and `mirror` are still available for a matched supercell.

## Why some options are removed

The layers of a close-packed slab sit on three cosets of a triangular lattice,
labelled `A`, `B`, `C`. Two gauge freedoms act on those labels.

* **The origin is arbitrary.** Translating every label by one constant leaves
  the structure alone, so only differences are physical: `A-A`, `B-B` and `C-C`
  are one and the same contact, and the nine labelled contacts are three.
* **The sense of `A → B → C` is arbitrary**, and swapping it is realised by a
  reflection, which negates every layer-to-layer step. A slab on its own
  therefore has no handedness: every uniform close-packed slab may be read as
  `ABCABC`. Handedness is meaningful only *between* the two slabs, which is why
  the bottom slab fixes the gauge and the top slab is described relative to it.

Writing the whole interface as one word of steps in `Z/3` -- the steps inside
the bottom slab, the contact step, then the steps inside the top slab --
reflection is `w ↦ -w`. Two chiral slabs give twelve labelled combinations and
six distinct structures.

If in addition the two slabs can trade places (the same slab on both sides, as
tested by `slabs_are_interchangeable`), the interface can be turned over, which
is `w ↦ reverse(-w)`. That merges exactly one further pair, the two twinned
contacts, leaving **five** distinct interfaces. Two monolayers, having no
stacking sense at all, leave two: the eclipsed contact and the hollow one.

## Machine-checked statements

The counting above is proved in the external Lean reference kept locally at
`aristotle-lean-reference/RequestProject/StackingRegistry.lean`:

| Statement | Formal name |
| --- | --- |
| `A-A`, `B-B` and `C-C` are one contact | `Cellstine.contact_classes_card` |
| a global relabelling leaves every step alone | `Cellstine.translate_increments` |
| reversing a stacking order negates every step | `Cellstine.mirrorWord_wordOf` |
| turning the interface over reverses and negates the word | `Cellstine.flipWord_wordOf` |
| the two operations generate a Klein four-group | `Cellstine.mirror_flip_comm` |
| the labelled combinations are all different words | `Cellstine.wordOf_injective` |
| twelve labelled, six distinct | `Cellstine.chiral_card`, `Cellstine.mirror_classes_card` |
| five distinct for interchangeable slabs | `Cellstine.full_classes_card` |
| exactly one further pair merges | `Cellstine.full_classes_succ`, `Cellstine.twin_contacts_merge` |
| two monolayers give two contacts | `Cellstine.monolayer_classes_card` |

The reading of a stacking sequence and its reversal live in
`interface/surface/stacking.py`; the enumeration, the deduplication and the
contact naming live in `interface/surface/registry.py`.

## Which way is up

Layers are separated, and ordered, by height along the surface normal
`n = a x b / ‖a x b‖`, oriented so that it points the same way as `c`. That
orientation is what makes a POSCAR read from the bottom of its cell upwards
whatever basis it was written in: swapping `a` and `b` renames the basis
without moving an atom, but it flips the sign of `a x b`, and an unoriented
normal would then report the slab upside down -- its top layer as its bottom,
and its in-plane frame mirrored. `cellstine.core.vacuum.normal_heights` is the
one place that measures those heights, and `tests/test_layer_heights.py` checks
that a left-handed cell and a rigidly rotated cell both read exactly as the
upright one does.

A slab such as `ABABAB` has no stacking sense of its own, so `abc` and `cba`
are refused for it, but its outermost step still says which hollow continues
the sequence, and the two hollows are named `fcc_hollow` and `hcp_hollow`
accordingly.

That the removed options really do rebuild the structures they duplicate is
checked on the written POSCARs by `tests/test_interface_stacking_build.py`,
using a per-atom minimum-image distance fingerprint that no isometry can
change.
