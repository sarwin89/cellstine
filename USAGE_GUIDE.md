# CELLSTINE Usage Guide

This guide covers both the guided interface and the command-line workflow in more detail than the README.

## 1) Standard Working Layout

CELLSTINE assumes this folder structure by default:

```text
input/   -> source POSCAR files
runs/    -> saved finder result tables (.dat)
output/  -> generated and adjusted POSCAR files
```

Examples:

- place your starting structures in `input/`
- run the finder and keep the saved `.dat` files in `runs/`
- generate final POSCAR files into `output/`

If you pass only a bare filename, CELLSTINE first checks these standard folders.

## 2) Interactive Interface

Launch the interface with:

```bash
python moire_cli.py
```

You will see three workflows:

1. Search for commensurate candidates
2. Generate the commensurate superlattice from a saved search
3. Move a top molecule or shift the upper layer in a stacked POSCAR

### Workflow 1: Search For Commensurate Candidates

CELLSTINE asks for:

- the first POSCAR path
- the second POSCAR path
- which one should be the bottom layer
- `nindex`

It then detects the symmetry-limited angle range and shows recommended finder settings.

If you keep the defaults, it uses a practical search profile automatically.
If you choose the manual path, it lets you set:

- angle window
- angle shortlist tolerances
- vector pairing tolerances
- final strain cutoff
- atom-count limit
- output folder
- optional matrix-value filter

After the search:

- the candidate table is printed
- the run is saved to `runs/<run_id>.dat`
- you can immediately continue to generating the commensurate superlattice

### Workflow 2: Generate The Commensurate Superlattice

This workflow loads a saved `.dat` file and asks for:

- which candidate rows to generate
- the interlayer gap in angstrom
- the output folder
- optional advanced generator tolerances

Examples of valid row input:

- `1`
- `1,2,5`
- `1,2,5-7`

Generated POSCAR files go to `output/` by default.

### Workflow 3: Move A Molecule Or Shift The Upper Layer

This workflow works on an existing stacked POSCAR, usually from `output/`.

For a molecule:

- CELLSTINE detects the top-side adsorbate
- reports its center of mass in Cartesian and Direct coordinates
- asks whether you want to work in Direct or Cartesian coordinates
- asks where to move the center of mass
- optionally rotates the molecule about `z`
- optionally reframes the visible periodic image

For a bilayer or general stacked slab:

- CELLSTINE detects the upper group
- asks for a Direct or Cartesian shift vector
- moves only the upper layer

## 3) Command-Line Workflow

The help pages are:

```bash
python moire_cli.py --help
python moire_cli.py find --help
python moire_cli.py make --help
python moire_cli.py molecule --help
python moire_cli.py layer --help
python -m moire.angles --help
```

## 4) Finder Stage In Detail

Basic search:

```bash
python moire_cli.py find input/mos2.vasp input/mos2.vasp --nindex 12 --max-atoms 300
```

Search only a few explicit angles:

```bash
python moire_cli.py find input/mos2.vasp input/mos2.vasp --angles 13.15,21.787,27.9 --nindex 12
```

Search with a custom atom window:

```bash
python moire_cli.py find input/a.vasp input/b.vasp --nindex 16 --min-atoms 50 --max-atoms 800
```

### Finder Units

- `--min-angle`, `--max-angle`, `--angles`, `--angle-step`, and `--angle-merge-tolerance` are in **degrees**
- `--angle-length-tolerance` is an **absolute** tolerance in **angstrom**
- strain and mismatch tolerances are **fractions**

Fraction examples:

- `0.002 = 0.2%`
- `0.01 = 1%`
- `0.05 = 5%`

### Finder Tolerances

`--angle-length-tolerance`

- absolute mismatch between two span lengths during the fast angle shortlist
- example: `12.000 A` vs `12.003 A` means a mismatch of `0.003 A`

`--angle-strain-tolerance`

- relative span-length mismatch allowed during the fast angle shortlist

`--vector-tolerance`

- relative geometric tolerance used while pairing nearly coincident vectors

`--vector-strain-tolerance`

- relative vector-length mismatch allowed while pairing vectors

`--candidate-tolerance`

- relative tolerance used while assembling candidate supercells from the matched vectors

`--strain-tolerance`

- final strain cutoff after the candidates have already been built
- controlled by `--strain-layer avg|1|2`

### Matrix-Value Finder Mode

This is the new mode for cases where you already know the four entries that should appear in a 2x2 supercell matrix, but not their order.

Example:

```bash
python moire_cli.py find input/a.vasp input/b.vasp --matrix-values 1,2,3,4 --matrix-layer either
```

This means:

- look for commensurate candidates as usual
- keep only candidates whose layer-1 matrix, layer-2 matrix, either matrix, or both matrices match those four values
- ignore entry order

Control options:

- `--matrix-values 1,2,3,4`
  supplies exactly four integers
- `--matrix-layer 1|2|either|both`
  chooses which layer matrix the filter should apply to
- `--matrix-match-mode absolute|exact`
  controls whether signs are ignored

Use `absolute` when you care about the magnitudes but not the signs:

```bash
python moire_cli.py find input/a.vasp input/b.vasp --matrix-values 1,2,3,4 --matrix-layer 2 --matrix-match-mode absolute
```

Use `exact` only when you know the signed matrix entries themselves:

```bash
python moire_cli.py find input/a.vasp input/b.vasp --matrix-values -3,-2,4,-1 --matrix-layer 2 --matrix-match-mode exact
```

## 5) Reading The Finder Output

The printed table and the saved `.dat` file include:

- `angle`: twist angle in degrees
- `strain_avg`: symmetric strain measure used for ranking
- `strain1`: one-sided relative length strain for layer 1 toward layer 2
- `strain2`: one-sided relative length strain for layer 2 toward layer 1
- `atoms`: total atoms in the combined supercell
- `ratio`: primitive-cell ratio between the two layers
- `i11 i12 / i21 i22`: layer-1 2x2 matrix entries
- `j11 j12 / j21 j22`: layer-2 2x2 matrix entries
- `eps1`, `eps2`: residual matching errors for the two basis directions

The saved `.dat` file also records the parameters used for the run.

## 6) Make Stage In Detail

Generate one candidate:

```bash
python moire_cli.py make runs/<run_id>.dat --index 1 --interlayer 3.35
```

Generate several candidates:

```bash
python moire_cli.py make runs/<run_id>.dat --index 1,2,5-7 --interlayer 3.35 --output-dir output
```

Important points:

- `--index` is required in command mode
- `--interlayer` is in angstrom
- output files go to `output/` by default
- `--generator-tolerance` and `--generator-tolerance-float` are advanced image-search settings

## 7) Molecule Stage In Detail

Move a molecule by Direct center-of-mass coordinates:

```bash
python moire_cli.py molecule output/stacked.vasp --target-direct 0.5,0.5 --rotate 30 --reframe xy
```

Move a molecule by Cartesian center-of-mass coordinates:

```bash
python moire_cli.py molecule output/stacked.vasp --target-cart 12.0,8.0,10.5 --rotate 45
```

Useful options:

- `--target-direct`
- `--target-cart`
- `--rotate`
- `--z-cutoff`
- `--min-gap`
- `--reframe`

Notes:

- the molecule is moved rigidly by its center of mass
- rotation is about the moved center of mass around an axis parallel to `z`
- reframing changes the visible periodic image, not the lattice itself

## 8) Upper-Layer Shift Stage In Detail

Shift the upper layer by Direct coordinates:

```bash
python moire_cli.py layer output/stacked.vasp --shift-direct 0.333,0.667
```

Shift the upper layer by Cartesian coordinates:

```bash
python moire_cli.py layer output/stacked.vasp --shift-cart 1.2,0.5
```

Notes:

- only the detected upper group is moved
- this is useful for commensurate bilayers and origin-shift studies

## 9) Fast Angle Shortlist Helper

If you only want a quick angle shortlist, you can use:

```bash
python -m moire.angles input/a.vasp input/b.vasp 12 --strain_tolerance 0.002
```

This does not generate structures. It only reports likely commensurate angles from span matching.

## 10) Troubleshooting

If `--help` works but a real command fails immediately with a dependency error:

- the parser is fine
- the runtime scientific dependencies still need to be available in that Python environment

If the finder returns too many candidates:

- lower `--max-atoms`
- tighten `--strain-tolerance`
- reduce the angle window
- use `--matrix-values` if you already know the matrix family you want

If the finder returns nothing:

- increase `nindex`
- widen the angle window
- relax `--vector-strain-tolerance` and `--strain-tolerance`
- remove the matrix-value filter and try again
