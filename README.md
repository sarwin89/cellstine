# CELLSTINE

CELLSTINE stands for **CELL Superlattice Transformation INterface and Engine**.

It is a guided POSCAR workflow for:

- finding commensurate bilayer superlattice candidates
- finding bottom-reference `N`-layer commensuration with `findn`
- generating bilayer stacks with `make` and `N`-layer stacks with `maken`
- extending each input structure along the `c` axis before counting and generation
- moving a top-side molecule or shifting an upper layer in a stacked POSCAR
- building an experimental surface slab from an orthogonal bulk cell and Miller plane
- creating an interactive Plotly HTML viewer for commensurate twist-angle results

**Made by Sarwin Chandran**

## Standard Folder Flow

CELLSTINE uses these folders by default:

- `input/` for source POSCAR files
- `runs/` for saved search results
- `output/` for generated POSCARs, adjusted POSCARs, slabs, and HTML visualizers

Bare filenames are resolved against these folders first.

## Quick Start

```bash
python moire_cli.py
python moire_cli.py --help
python moire_cli.py find --help
python moire_cli.py findn --help
```

The interactive interface still covers the common bilayer workflow.
The newer `findn`, `maken`, `surface`, and `visualize` workflows are CLI-first.

## Simple Examples

Find bilayer candidates:

```bash
python moire_cli.py find input/mos2.vasp input/mos2.vasp --nindex 12 --max-atoms 300
```

Find bilayer candidates while thickening the top and bottom slabs along `c`:

```bash
python moire_cli.py find input/top.vasp input/bottom.vasp --nindex 12 --top-c-repeat 2 --bottom-c-repeat 4
```

Use multiple workers for a larger bilayer search:

```bash
python moire_cli.py find input/top.vasp input/bottom.vasp --nindex 18 --workers 4
```

Find bottom-reference `N`-layer candidates:

```bash
python moire_cli.py findn input/substrate.vasp input/layer_b.vasp input/layer_c.vasp --nindex 12
python moire_cli.py findn input/substrate.vasp input/layer_b.vasp input/layer_c.vasp input/layer_d.vasp --angles 13.2 --angles 21.8 --angles 5.5
```

Generate the commensurate bilayer superlattice:

```bash
python moire_cli.py make runs/<run_id>.dat --index 1 --interlayer 3.35
```

Generate the commensurate `N`-layer superlattice:

```bash
python moire_cli.py maken runs/<run_id>.json --index 1 --interlayer 3.35 --interlayer 3.35
```

Build an experimental `(110)` slab from an orthogonal bulk cell:

```bash
python moire_cli.py surface input/bulk.vasp --miller 1,1,0 --layers 6 --vacuum 15
```

Create an interactive HTML viewer for commensurate frames:

```bash
python moire_cli.py visualize runs/<run_id>.dat --index 1,2,3
python moire_cli.py visualize runs/<run_id>.json --index 1,2
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
- interlayer distances, Cartesian shifts, vacuum, and `zfix` are in **angstrom**
- Direct coordinates are fractional coordinates of the current cell
- `--top-c-repeat`, `--bottom-c-repeat`, and `--upper-c-repeat` are integer repeat counts, not angstrom values

## Parallel And Single-Threaded Modes

- `--workers 1` keeps the original single-threaded behavior
- `--workers > 1` enables multiprocessing in the angle search or batch generation paths
- if multiprocessing is unavailable in a restricted environment, CELLSTINE falls back to the serial path instead of failing outright

## Documentation

- For a detailed walkthrough of both the interface and the CLI, see [USAGE_GUIDE.md](USAGE_GUIDE.md).
- For quick command summaries, use `python moire_cli.py --help` and the subcommand `--help` pages.

## Testing

```bash
python -m unittest discover -s tests -q
```
