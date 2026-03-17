# CELLSTINE

CELLSTINE stands for **CELL Superlattice Transformation INterface and Engine**.

It is a guided workflow for:

- finding commensurate superlattice candidates between two POSCAR structures
- generating the commensurate superlattice from saved candidate rows
- moving a top-side molecule or shifting an upper layer in a stacked POSCAR

**Made by Sarwin Chandran**

## Standard Folder Flow

CELLSTINE uses these folders by default:

- `input/` for source POSCAR files
- `runs/` for saved finder result tables (`.dat`)
- `output/` for generated and adjusted POSCAR files

If you pass a bare filename, the CLI will try these standard folders first.

## Quick Start

```bash
python moire_cli.py
python moire_cli.py --help
python moire_cli.py find --help
```

The interactive workflow covers the normal path:

1. Search for commensurate candidates.
2. Review the saved rows in `runs/`.
3. Generate the commensurate superlattice into `output/`.
4. Optionally move a molecule or shift an upper layer afterwards.

## Simple CLI Examples

Find candidates:

```bash
python moire_cli.py find input/mos2.vasp input/mos2.vasp --nindex 12 --max-atoms 300
```

Find candidates only near specific angles:

```bash
python moire_cli.py find input/mos2.vasp input/mos2.vasp --angles 13.15,21.787,27.9 --nindex 12
```

Check whether a commensuration exists with a specific set of matrix values:

```bash
python moire_cli.py find input/a.vasp input/b.vasp --matrix-values 1,2,3,4 --matrix-layer either
```

Generate the commensurate superlattice from saved results:

```bash
python moire_cli.py make runs/<run_id>.dat --index 1 --interlayer 3.35
```

Move a top-side molecule by its center of mass:

```bash
python moire_cli.py molecule output/stacked.vasp --target-direct 0.5,0.5 --rotate 30 --reframe xy
```

Shift only the upper layer in a bilayer:

```bash
python moire_cli.py layer output/stacked.vasp --shift-direct 0.333,0.667
```

## Important Units

- angles are in **degrees**
- `--angle-length-tolerance` is in **angstrom**
- strain and mismatch tolerances are **fractions**, so `0.01 = 1%`
- Cartesian move and shift values are in **angstrom**
- Direct coordinates are fractional coordinates of the current cell

## Matrix-Value Finder Mode

The finder can optionally test whether a commensuration exists whose 2x2 supercell matrix uses one requested set of four values, ignoring entry order.

- `--matrix-values 1,2,3,4` supplies the four values
- `--matrix-layer 1|2|either|both` chooses which layer matrix to match
- `--matrix-match-mode absolute|exact` chooses whether signs are ignored

`absolute` is the default, so a matrix like `[-3, -2; 4, -1]` matches `1,2,3,4`.

## Documentation

- For a detailed walkthrough of the interface and CLI, see [USAGE_GUIDE.md](USAGE_GUIDE.md).
- For quick command summaries, use `python moire_cli.py --help` and the subcommand `--help` pages.

## Testing

```bash
python -m unittest discover -s tests -q
```
