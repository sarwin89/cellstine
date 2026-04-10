# CELLSTINE

CELLSTINE stands for **CELL Superlattice Transformation INterface and Engine**.

It is a class-first Python package and guided CLI for three related workflows built around VASP-style structures:

- `moire` for commensurate bilayer and `N`-layer supercell search and construction
- `adsorbate` for molecule-on-substrate placement and rigid molecular movement
- `interface` for bulk-to-surface slab generation, adsorption-site analysis, and slab-on-slab interface construction

The current package keeps the proven NumPy-based search/build kernels as the baseline, adds a publishable `src/cellstine` package layout, and leaves room for optional `pymatgen`-powered backends where they are useful and validated.

**Made by Sarwin Chandran**

## Installation

Inside this repository:

```bash
pip install -e .
```

Install optional extras as needed:

```bash
pip install -e ".[pymatgen]"
pip install -e ".[viz]"
pip install -e ".[all]"
```

Once installed, use:

```bash
cellstine --help
```

When working directly inside the repo, you can also use the compatibility shim:

```bash
python moire_cli.py --help
```

The examples below use `cellstine`, but `python moire_cli.py` works too.

## Standard Folder Flow

CELLSTINE uses these folders by default:

- `input/` for source structures and local example POSCARs
- `runs/` for manifests and saved intermediate workflow artifacts
- `output/` for generated POSCARs, slabs, interfaces, adjusted structures, and HTML visualizers

Each grouped workflow writes a manifest to `runs/<workflow>/<run-id>/manifest.json`.

## Quick Start

Launch the grouped interactive interface:

```bash
cellstine
```

Inspect the CLI:

```bash
cellstine --help
cellstine moire --help
cellstine adsorbate --help
cellstine interface --help
cellstine --version
```

## Simple Examples

Search bilayer commensurate candidates:

```bash
cellstine moire find input/mos2.vasp input/mos2.vasp --nindex 12 --max-angle 30
```

Search with prestrain on the top layer before commensuration:

```bash
cellstine moire find input/graph.vasp input/mos2.vasp --nindex 12 --prestrain-top-mode biaxial --prestrain-top-value 0.01
```

Search a bottom-reference `N`-layer stack:

```bash
cellstine moire findn input/mos2.vasp input/graph.vasp input/mos2x.vasp --match-mode base_shared --nindex 12
```

Search `N`-layer candidates at explicit per-layer angles:

```bash
cellstine moire findn input/mos2.vasp input/graph.vasp input/mos2x.vasp --angles-by-layer "13.2;21.8"
```

Generate the commensurate bilayer superlattice from a manifest:

```bash
cellstine moire make runs/moire/<run-id>/manifest.json --indexes 1 --interlayer-distance 3.35
```

Generate the commensurate `N`-layer superlattice:

```bash
cellstine moire maken runs/moire/<run-id>/manifest.json --indexes 1 --interlayers 3.35,3.35
```

Shift the upper layer in a stacked structure:

```bash
cellstine moire translate output/stacked.vasp --shift-direct 0.333,0.667
```

Build an `Au(111)` slab from a bulk cell and detect adsorption sites:

```bash
cellstine interface surface input/Au_Bulk.vasp --miller 1,1,1 --layers 6 --vacuum 15 --analyse-sites
```

Analyse sites on an existing slab:

```bash
cellstine interface sites output/surface_Au_Bulk_111_layers4.vasp
```

Place a molecule on a bulk-derived substrate:

```bash
cellstine adsorbate place input/Au_Bulk.vasp input/papd_gasp_mol2_final-coor_at_.vasp --substrate-kind bulk --miller 1,1,1 --layers 4 --site-type fcc --height 2.3
```

Move a top-side molecule by its center of mass:

```bash
cellstine adsorbate move output/stacked.vasp --target-direct 0.5,0.5 --rotate 30
```

Build a slab-on-slab interface from two bulk inputs:

```bash
cellstine interface build input/Au_Bulk.vasp input/Au_Bulk.vasp --bottom-kind bulk --top-kind bulk --bottom-miller 1,1,1 --top-miller 1,0,0 --gap 3.0
```

Scan bulk-derived surfaces for possible interface matches:

```bash
cellstine interface match input/Au_Bulk.vasp input/Au_Bulk.vasp --bottom-millers 1,1,1 1,0,0 --top-millers 1,1,1 1,1,0 --max-strain 0.05
```

Create an HTML viewer for commensurate moire results:

```bash
cellstine moire visualize runs/moire/<run-id>/manifest.json --indices 1,2,3
```

## Important Units

- Angles are in **degrees**
- `--angle-length-tolerance` style length checks are in **angstrom**
- Strain and mismatch values are **fractions**, so `0.01 = 1%`
- Interlayer distances, adsorption heights, vacuum, and Cartesian shifts are in **angstrom**
- Direct coordinates are fractional coordinates of the current cell
- `--top-c-repeat`, `--bottom-c-repeat`, and `--upper-c-repeats` are integer repeat counts, not angstrom values

## Workflow Notes

`moire`

- `find` handles two layers
- `findn` supports `base_shared`, `base_independent`, and `pairwise`
- `pairwise` is available but not recommended for routine use
- serial and multiprocessing paths are both kept available through `--workers`

`adsorbate`

- substrate input can be a slab, a primitive surface cell, a full patch, or a bulk cell
- if you pass a bulk, the workflow first generates a surface slab under the hood
- molecule placement supports site-based adsorption, COM moves, and rigid rotation

`interface`

- `surface` builds slabs from bulk Miller planes, including negative-index notation like `1,1,2x`
- `sites` reports top, bridge, `hcp`, `fcc`, hollow, and fourfold hollow sites
- `build` fixes the bottom slab and strains the top slab in-plane to it
- `match` ranks bulk-derived surface combinations primarily by strain and then by commensurate size

## Optional Backends

- `numpy` is required
- `pymatgen` is optional and is used first for broad structure-format conversion when available
- `matplotlib` and `plotly` are optional visualization dependencies

Check what CELLSTINE can see in the current environment with:

```bash
cellstine --version
```

## Documentation

- A longer walkthrough of grouped CLI usage is in [USAGE_GUIDE.md](USAGE_GUIDE.md).
- Every command also has its own `--help` page.

## Testing

```bash
python -m unittest discover -s tests -q
```
