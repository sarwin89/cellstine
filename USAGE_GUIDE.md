# CELLSTINE Usage Guide

This guide covers both the guided interface and the command-line workflow in more detail than the README.

## 1. Standard Working Layout

CELLSTINE assumes this folder structure by default:

```text
input/   -> source POSCAR files
runs/    -> saved search results (.dat for bilayers, .json for findn)
output/  -> generated POSCARs, adjusted POSCARs, slabs, and HTML visualizers
```

If you pass only a bare filename, CELLSTINE checks these folders first.

## 2. Interactive Interface

Launch the interface with:

```bash
python moire_cli.py
```

The current interactive interface is focused on the common bilayer workflow:

1. Search for commensurate candidates
2. Generate the commensurate superlattice from a saved search
3. Move a top molecule or shift the upper layer in a stacked POSCAR

The newer `findn`, `maken`, `surface`, and `visualize` workflows are CLI-first.

## 3. CLI Help Pages

```bash
python moire_cli.py --help
python moire_cli.py find --help
python moire_cli.py findn --help
python moire_cli.py make --help
python moire_cli.py maken --help
python moire_cli.py molecule --help
python moire_cli.py layer --help
python moire_cli.py surface --help
python moire_cli.py visualize --help
python -m moire.angles --help
```

## 4. Bilayer Finder Stage

Basic search:

```bash
python moire_cli.py find input/mos2.vasp input/mos2.vasp --nindex 12 --max-atoms 300
```

Search only a few explicit angles:

```bash
python moire_cli.py find input/mos2.vasp input/mos2.vasp --angles 13.15,21.787,27.9 --nindex 12
```

Search with thicker input slabs along `c`:

```bash
python moire_cli.py find input/top.vasp input/bottom.vasp --nindex 12 --top-c-repeat 2 --bottom-c-repeat 4
```

Use multiple workers for a larger angle search:

```bash
python moire_cli.py find input/top.vasp input/bottom.vasp --nindex 18 --workers 4
```

### Bilayer Finder Units

- `--min-angle`, `--max-angle`, `--angles`, `--angle-step`, and `--angle-merge-tolerance` are in **degrees**
- `--angle-length-tolerance` is an **absolute** tolerance in **angstrom**
- strain and mismatch tolerances are **fractions**
- `--top-c-repeat` and `--bottom-c-repeat` are integer repeat counts

Fraction examples:

- `0.002 = 0.2%`
- `0.01 = 1%`
- `0.05 = 5%`

### Matrix-Value Finder Mode

This is the mode for cases where you already know the four entries that should appear in a `2x2` supercell matrix, but not their order.

Example:

```bash
python moire_cli.py find input/a.vasp input/b.vasp --matrix-values 1,2,3,4 --matrix-layer either
```

Control options:

- `--matrix-values 1,2,3,4`
- `--matrix-layer 1|2|either|both`
- `--matrix-match-mode absolute|exact`

## 5. Bilayer Generation Stage

Generate one candidate:

```bash
python moire_cli.py make runs/<run_id>.dat --index 1 --interlayer 3.35
```

Generate several candidates:

```bash
python moire_cli.py make runs/<run_id>.dat --index 1,2,5-7 --interlayer 3.35 --output-dir output
```

Override the slab thickness used during generation:

```bash
python moire_cli.py make runs/<run_id>.dat --index 1 --bottom-c-repeat 4 --top-c-repeat 2
```

Parallel batch generation:

```bash
python moire_cli.py make runs/<run_id>.dat --index 1,2,5-7 --workers 4
```

## 6. General N-Layer Commensuration With `findn`

`findn` is a **bottom-reference N-layer search**.

It does not try to solve a completely free all-to-all commensuration problem at once.
Instead it:

1. searches each upper layer against the same bottom layer
2. groups those pairwise candidates by the bottom-layer supercell
3. keeps only combinations that share that same bottom-layer commensurate cell

That is the scalable version of the earlier trilayer idea, and it now covers `3`, `4`, `5`, or more layers in the same command.

Basic usage:

```bash
python moire_cli.py findn input/substrate.vasp input/layer_b.vasp input/layer_c.vasp --nindex 12
```

Four-layer example with explicit angle sets:

```bash
python moire_cli.py findn input/substrate.vasp input/layer_b.vasp input/layer_c.vasp input/layer_d.vasp --angles 13.2 --angles 21.8 --angles 5.5
```

Use one shared min/max angle for all upper layers:

```bash
python moire_cli.py findn input/substrate.vasp input/layer_b.vasp input/layer_c.vasp --min-angle 0 --max-angle 30
```

Use separate min/max values for different upper layers:

```bash
python moire_cli.py findn input/substrate.vasp input/layer_b.vasp input/layer_c.vasp --min-angle 0 --min-angle 5 --max-angle 30 --max-angle 20
```

Search with thicker slabs:

```bash
python moire_cli.py findn input/substrate.vasp input/layer_b.vasp input/layer_c.vasp input/layer_d.vasp --bottom-c-repeat 5 --upper-c-repeat 2 --upper-c-repeat 2 --upper-c-repeat 3
```

Use multiple workers inside each pairwise search:

```bash
python moire_cli.py findn input/substrate.vasp input/layer_b.vasp input/layer_c.vasp input/layer_d.vasp --workers 4
```

### `findn` Output

`findn` writes a JSON results file to `runs/`.

Each candidate records:

- `strain_max`
- `strain_mean`
- `ratio_bottom`
- `total_atoms`
- the common bottom-layer `2x2` supercell matrix
- an `upper_layers` list containing one entry per upper layer with its angle, strain, ratio, and `2x2` matrix

## 7. N-Layer Generation With `maken`

Generate one `N`-layer candidate:

```bash
python moire_cli.py maken runs/<run_id>.json --index 1 --interlayer 3.35 --interlayer 3.35
```

Generate several candidates:

```bash
python moire_cli.py maken runs/<run_id>.json --index 1,2 --output-dir output --interlayer 3.35 --interlayer 3.35
```

Four-layer example with three independent gaps:

```bash
python moire_cli.py maken runs/<run_id>.json --index 1 --interlayer 3.0 --interlayer 3.2 --interlayer 3.4
```

Override the stored slab thickness:

```bash
python moire_cli.py maken runs/<run_id>.json --index 1 --bottom-c-repeat 5 --upper-c-repeat 2 --upper-c-repeat 2 --upper-c-repeat 3
```

Notes:

- `maken` preserves the shared bottom-layer in-plane cell
- `--interlayer` is repeated once per gap between consecutive layers
- if you give a single `--interlayer`, it is reused for every gap

## 8. Extending Structures Along `c`

CELLSTINE now lets you thicken each input structure along the `c` axis before atom counting and generation.

This is useful for:

- metal substrates where you want more slab layers
- oxide or perovskite top slabs that should not be treated as single-layer sheets
- multi-layer stacks where each component needs a different slab thickness

Examples:

```bash
python moire_cli.py find input/top.vasp input/bottom.vasp --top-c-repeat 2 --bottom-c-repeat 5
python moire_cli.py make runs/<run_id>.dat --index 1 --top-c-repeat 2 --bottom-c-repeat 5
python moire_cli.py findn input/substrate.vasp input/layer_b.vasp input/layer_c.vasp input/layer_d.vasp --bottom-c-repeat 5 --upper-c-repeat 2 --upper-c-repeat 2 --upper-c-repeat 3
```

These are repeat counts, not vacuum distances and not angstrom values.

## 9. Experimental Surface Builder

Build a `(100)` slab:

```bash
python moire_cli.py surface input/bulk.vasp --miller 1,0,0 --layers 6 --vacuum 15
```

Build a `(110)` slab and repeat it in-plane:

```bash
python moire_cli.py surface input/bulk.vasp --miller 1,1,0 --layers 4 --repeat-a 2 --repeat-b 2 --vacuum 18
```

Current limitations:

- this is marked experimental on purpose
- it currently supports **orthogonal bulk cells only**
- that means cubic, tetragonal, and orthorhombic-like inputs are the intended first target
- it is not yet a general arbitrary-crystal slab generator

## 10. Interactive Plotly Visualizer

Create an HTML viewer for bilayer results:

```bash
python moire_cli.py visualize runs/<run_id>.dat --index 1,2,3
```

Create an HTML viewer for `findn` results:

```bash
python moire_cli.py visualize runs/<run_id>.json --index 1,2
```

The viewer:

- snaps one frame at a time through the selected commensurate twist angles
- supports free mouse rotation because it uses a Plotly `scatter3d` scene
- draws the commensurate unit cell only on those saved commensurate frames
- writes an HTML file into `output/` by default

For `findn` results, the visualizer uses the saved bottom-reference stack order and interlayer gaps.
If there are more than three layers, it currently reuses the `--interlayer-bottom-middle` value for all gaps beyond the first two.

## 11. Molecule Stage In Detail

Move a molecule by Direct center-of-mass coordinates:

```bash
python moire_cli.py molecule output/stacked.vasp --target-direct 0.5,0.5 --rotate 30 --reframe xy
```

Move a molecule by Cartesian center-of-mass coordinates:

```bash
python moire_cli.py molecule output/stacked.vasp --target-cart 12.0,8.0,10.5 --rotate 45
```

Notes:

- the molecule is moved rigidly by its center of mass
- rotation is about the moved center of mass around an axis parallel to `z`
- reframing changes the visible periodic image, not the lattice itself

## 12. Upper-Layer Shift Stage In Detail

Shift the upper layer by Direct coordinates:

```bash
python moire_cli.py layer output/stacked.vasp --shift-direct 0.333,0.667
```

Shift the upper layer by Cartesian coordinates:

```bash
python moire_cli.py layer output/stacked.vasp --shift-cart 1.2,0.5
```

## 13. Parallel Execution

CELLSTINE now keeps both the original serial mode and an optional multiprocessing mode.

Rules of thumb:

- `--workers 1` gives the original single-threaded behavior
- use `--workers 2`, `--workers 4`, and so on when you have many angles to check
- the biggest speedup usually comes from larger angle lists or larger `nindex` searches
- batch `make` generation can also use multiple workers

If a restricted environment blocks process-pool creation, CELLSTINE falls back to the serial path rather than failing outright.

## 14. Troubleshooting

If the finder returns too many candidates:

- lower `--max-atoms`
- tighten the strain cutoff
- reduce the angle window
- use explicit angle lists if you already know the interesting region

If the slab builder fails:

- check that the bulk input cell is orthogonal
- start with simple Miller planes like `(100)` or `(110)`

If the visualizer HTML opens but appears blank:

- make sure the results file actually contains candidates
- try `--index 1` first
- if the output HTML is opened without internet access, the Plotly CDN script may not load

## 15. Testing

```bash
python -m unittest discover -s tests -q
```
