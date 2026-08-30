# Twisted bilayers of a hexagonal monolayer

A twisted homobilayer is the one moire problem where the answer is known in
closed form, so it is also the best available check that the CELLSTINE moire,
builder and defect stages agree with reality. This note records the arithmetic,
the exact commands, and the reference numbers the test suite pins.

## The arithmetic

Take a triangular lattice with primitive vectors `a1`, `a2` at 60 degrees. For
coprime integers `m < n` put

```text
N = m^2 + m n + n^2
cos(theta) = (m^2 + 4 m n + n^2) / (2 N)
```

Rotating the top layer by `theta` makes the two layers exactly commensurate:
the smallest common cell holds `N` primitive cells of each layer. Because the
layers have three-fold, not six-fold, symmetry, the supplement `60 - theta` is
a second, physically different structure with a cell of the same size, and the
search reports both.

For MoS2 (three atoms per primitive cell: one Mo plane, two S planes per layer,
four planes in a bilayer) the cell therefore holds

```text
6 N atoms,  4 N of them sulfur,  N in each of the four sulfur planes.
```

The twist leaves only the three-fold rotation about the common axis (point
group 32 = D3, six operations, no inversion — the twist breaks it). A three-fold
rotation on `N` sites of one plane has one fixed point and `(N - 1) / 3` free
orbits when `3` does not divide `N`, so a plane holds

```text
(N + 2) / 3 inequivalent sites.
```

This needs `3` not to divide `N`, which costs nothing: when `m` and `n` are
coprime and `m == n (mod 3)` the pair `(m, n)` is not a reduced one — three
divides `N` and the true cell is the `N / 3` cell of a smaller pair, which the
search returns instead. (`(1, 4)` has `N = 21`, and what the search reports at
its angle of 38.213 degrees is the 42-atom `N = 7` cell.) The derivation —
the index arithmetic, the cosine formula, the labelling of the moire cell by
`Z^2 / A Z^2`, and the orbit count — is machine-checked in
`aristotle-lean-reference/RequestProject/TwistedBilayer.lean`.

## Reference numbers

Known results for AA'-stacked (2H) MoS2:

| twist angle | (m, n) | N  | atoms | S atoms | S per plane | inequivalent S sites per plane |
|-------------|--------|----|-------|---------|-------------|--------------------------------|
| 21.787      | (1, 2) |  7 |    42 |      28 |           7 |                              3 |
| 27.796      | (1, 3) | 13 |    78 |      52 |          13 |                              5 |
| 13.173      | (2, 3) | 19 |   114 |      76 |          19 |                              7 |
|  9.430      | (3, 4) | 37 |   222 |     148 |          37 |                             13 |
| 16.426      | (3, 5) | 49 |   294 |     196 |          49 |                             17 |
|  7.341      | (4, 5) | 61 |   366 |     244 |          61 |                             21 |

CELLSTINE reproduces every row: `tests/test_moire_mos2_reference.py` runs the
search, the build and the defect analysis and asserts the whole table, plus the
cell area `N * sqrt(3) a^2 / 2`, the Mo-S bond, the Mo-Mo distance and the
interlayer gap.

## Doing it yourself

Write the two monolayers first. AA' stacking means the upper layer is the
in-plane 180 degree partner of the lower one, i.e. its sulfur column sits in
the other hollow of the molybdenum lattice: `S` at `(1/3, 2/3)` below and at
`(2/3, 1/3)` above, with `Mo` at the origin in both.

Then run a **rigid** search — zero strain budgets, so nothing is deformed and
only exactly commensurate cells survive:

```console
$ cellstine moire find mos2_Aprime.vasp mos2_A.vasp \
      --max-length 26 --top-strain 0 --bottom-strain 0 --max-atoms 400
```

```text
 idx  angle (deg)   moire a x b (Ang)   gamma  top/bottom/total atoms  relative principal strain (%)
   1       0.0000     3.160 x   3.160   60.00      3/     3/    6  (  +0.0000,   +0.0000)
   2      60.0000     3.160 x   3.160   60.00      3/     3/    6  (  +0.0000,   +0.0000)
   3      21.7868     8.361 x   8.361  120.00     21/    21/   42  (  +0.0000,   +0.0000)
   4      38.2132     8.361 x   8.361   60.00     21/    21/   42  (  +0.0000,   +0.0000)
   5      27.7958    11.394 x  11.394   60.00     39/    39/   78  (  +0.0000,   +0.0000)
   6      32.2042    11.394 x  11.394  120.00     39/    39/   78  (  +0.0000,   +0.0000)
   7      13.1736    13.774 x  13.774   60.00     57/    57/  114  (  +0.0000,   +0.0000)
   ...
  17       7.3410    24.680 x  24.680  120.00    183/   183/  366  (  +0.0000,   +0.0000)
```

Every reference angle is there, each with its 60-degree partner, and every
strain is exactly zero. Build the one you want and inspect its sites:

```console
$ cellstine moire make runs/moire/find_*/results.json \
      --indexes 3 --interlayer-distance 3.1 --output-dir out42
$ cellstine defect analyse out42/stack_idx003_*.vasp --structure-kind bulk
```

```text
Atomic planes (height along the surface normal, atoms, and how many inequivalent sites each plane holds)
 layer  height (A)  atoms  composition                inequivalent sites
--------------------------------------------------------------------------------------------
     1      7.3228      7  S7                         S:3
     2      8.8864      7  Mo7                        Mo:3
     3     10.4500      7  S7                         S:3
     4     13.5500      7  S7                         S:3
     5     15.1136      7  Mo7                        Mo:3
     6     16.6772      7  S7                         S:3

Notes:
- Site equivalence uses the 6 space-group operations of the cell (point group 32).
```

Seven sulfur atoms per plane and three inequivalent ones, as the table
requires. Over the whole cell the analysis reports six inequivalent sulfur
sites, not twelve: the three two-fold axes of group 32 relate the four sulfur
planes in pairs, so a vacancy in the top plane of one layer is equivalent to
one in the bottom plane of the other.

## Caveats worth knowing

- The interlayer distance is a build parameter, not a prediction. `3.1 A`
  between the facing sulfur planes is the usual starting geometry for MoS2;
  relax it in your DFT code.
- The search is rigid, so the cells are exact. If instead you allow a strain
  budget you will also get near-commensurate cells, which are smaller but carry
  a real deformation — see `MOIRE_SEARCH.md`.
- `--max-length` bounds the moire cell, and small angles need large cells: the
  7.341 degree row already needs a 24.7 A cell. Raise both `--max-length` and
  `--max-atoms` to reach smaller angles.
