# CELLSTINE Usage Guide

This guide covers the current grouped CELLSTINE package and CLI in more detail than the README.

The package is organized around three top-level workflows:

- `moire`
- `adsorbate`
- `interface`

Examples below use the installed command `cellstine`. Inside the repository, `python moire_cli.py ...` works as a compatibility entrypoint too.

## 1. Installation And Entry Points

Base install:

```bash
pip install -e .
```

Optional extras:

```bash
pip install -e ".[pymatgen]"
pip install -e ".[viz]"
pip install -e ".[all]"
```

Main entry points:

```bash
cellstine --help
cellstine --version
python moire_cli.py --help
```

## 2. Standard Working Layout

CELLSTINE assumes this folder structure by default:

```text
input/   -> source structures and local sample POSCARs
runs/    -> manifests and saved workflow artifacts
output/  -> generated POSCARs, slabs, interfaces, moved structures, and HTML viewers
```

Each grouped stage writes a manifest to:

```text
runs/<workflow>/<run-id>/manifest.json
```

Build stages can consume either a raw results file or a manifest from the previous stage.

## 3. Interactive Interface

Launch the guided interface with:

```bash
cellstine
```

The interactive flow is now grouped:

1. choose `moire`, `adsorbate`, or `interface`
2. choose the stage within that workflow
3. enter only the inputs needed for that stage
4. use the manifest or generated artifact for the next step

It is meant to be a guided launcher over the same backend classes used by the CLI.

## 4. CLI Help Pages

```bash
cellstine --help
cellstine moire --help
cellstine moire find --help
cellstine moire findn --help
cellstine moire make --help
cellstine moire maken --help
cellstine moire translate --help
cellstine moire visualize --help
cellstine adsorbate --help
cellstine adsorbate place --help
cellstine adsorbate move --help
cellstine adsorbate assemble --help
cellstine interface --help
cellstine interface surface --help
cellstine interface sites --help
cellstine interface build --help
cellstine interface match --help
cellstine interface visualize --help
```

## 5. `moire` Workflow

### 5.1 Bilayer Search With `moire find`

Basic search:

```bash
cellstine moire find input/mos2.vasp input/mos2.vasp --nindex 12 --max-angle 30
```

Search explicit angles only:

```bash
cellstine moire find input/mos2.vasp input/mos2.vasp --angles 13.15,21.787,27.9 --nindex 12
```

Search with thicker slabs along `c`:

```bash
cellstine moire find input/graph.vasp input/mos2.vasp --nindex 12 --top-c-repeat 2 --bottom-c-repeat 4
```

Use optional prestrain before the commensuration search:

```bash
cellstine moire find input/graph.vasp input/mos2.vasp --nindex 12 --prestrain-top-mode biaxial --prestrain-top-value 0.01
cellstine moire find input/graph.vasp input/mos2.vasp --nindex 12 --prestrain-top-mode uniaxial --prestrain-top-value 0.01 --prestrain-top-axis a
```

Use multiple workers for angle-parallel search:

```bash
cellstine moire find input/top.vasp input/bottom.vasp --nindex 18 --workers 4
```

### 5.2 Bilayer Finder Units And Tolerances

- twist angles are in **degrees**
- length tolerances are in **angstrom**
- strain and mismatch tolerances are **fractions**
- `0.002 = 0.2%`
- `0.01 = 1%`
- `0.05 = 5%`
- `--top-c-repeat` and `--bottom-c-repeat` are integer repeat counts

### 5.3 Matrix-Value Finder Mode

Use this when you know the four entries of a `2x2` supercell matrix but not their order.

```bash
cellstine moire find input/mos2.vasp input/mos2.vasp --matrix-values 1,2,3,4 --matrix-layer either --matrix-match-mode absolute
```

### 5.4 N-Layer Search With `moire findn`

`findn` is a bottom-reference multi-layer workflow. It supports three modes:

- `base_shared`
  This is the default and recommended mode. All upper layers must share the same base-layer supercell before they can be built together.
- `base_independent`
  Each upper layer is matched against the same bottom layer and reported independently.
- `pairwise`
  All layer pairs are searched. This is available, but not recommended for routine use.

Basic search:

```bash
cellstine moire findn input/mos2.vasp input/graph.vasp input/mos2x.vasp --match-mode base_shared --nindex 12
```

Explicit per-layer angles:

```bash
cellstine moire findn input/mos2.vasp input/graph.vasp input/mos2x.vasp --angles-by-layer "13.2;21.8"
```

Per-layer angle windows:

```bash
cellstine moire findn input/mos2.vasp input/graph.vasp input/mos2x.vasp --min-angles 0,5 --max-angles 30,20
```

Per-layer prestrain:

```bash
cellstine moire findn input/mos2.vasp input/graph.vasp input/mos2x.vasp --prestrain-modes none,biaxial,uniaxial --prestrain-values 0,0.01,0.005 --prestrain-axes a,a,b
```

Thicker bottom and upper slabs:

```bash
cellstine moire findn input/mos2.vasp input/graph.vasp input/mos2x.vasp --bottom-c-repeat 4 --upper-c-repeats 2,3
```

### 5.5 Generation With `moire make` And `moire maken`

Generate one bilayer candidate from a manifest:

```bash
cellstine moire make runs/moire/<run-id>/manifest.json --indexes 1 --interlayer-distance 3.35
```

Generate several bilayer candidates:

```bash
cellstine moire make runs/moire/<run-id>/manifest.json --indexes 1,2,5-7 --interlayer-distance 3.35 --workers 4
```

Generate one `N`-layer candidate:

```bash
cellstine moire maken runs/moire/<run-id>/manifest.json --indexes 1 --interlayers 3.0,3.2
```

Notes:

- `make` uses `--interlayer-distance`
- `maken` uses `--interlayers`
- `maken` is only meaningful for shared-base candidates
- manifests are often easier to use than hunting for the raw `.dat` or `.json` file yourself

### 5.6 Translation And Visualization

Shift the upper layer in a stacked structure:

```bash
cellstine moire translate output/stacked.vasp --shift-direct 0.333,0.667
cellstine moire translaten output/stacked.vasp --shift-cart 1.2,0.5
```

Create a commensurate-results HTML viewer:

```bash
cellstine moire visualize runs/moire/<run-id>/manifest.json --indices 1,2,3
```

## 6. `adsorbate` Workflow

### 6.1 Placement With `adsorbate place`

The substrate input can be:

- `substrate`
- `patch`
- `surface`
- `slab`
- `bulk`

If you pass `bulk`, CELLSTINE first generates a surface slab through the `interface surface` machinery.

Place on an existing slab:

```bash
cellstine adsorbate place output/surface_Au_Bulk_111_layers4.vasp input/papd_gasp_mol2_final-coor_at_.vasp --site-type fcc --site-index 1 --height 2.3
```

Place on a bulk-derived substrate:

```bash
cellstine adsorbate place input/Au_Bulk.vasp input/papd_gasp_mol2_final-coor_at_.vasp --substrate-kind bulk --miller 1,1,1 --layers 4 --vacuum 15 --site-type top --height 2.0
```

How placement works:

- the site list is determined from the slab geometry
- the molecule is rotated rigidly about its center of mass
- the molecule is aligned in-plane to the chosen site
- the closest molecule atom is then placed `--height` angstrom above the selected surface plane

This means `--height` is a molecule-to-surface gap, not a COM height.

### 6.2 Rigid Molecular Movement With `adsorbate move`

Move a top-side molecule by Direct coordinates:

```bash
cellstine adsorbate move output/stacked.vasp --target-direct 0.5,0.5 --rotate 30
```

Move by Cartesian coordinates:

```bash
cellstine adsorbate move output/stacked.vasp --target-cart 12.0,8.0,10.5 --rotate 45
```

### 6.3 Molecular Assembly With `adsorbate assemble`

This mode uses an experimental target lattice to search for a commensurate substrate supercell beneath it.

```bash
cellstine adsorbate assemble input/Au_1x1.vasp --a-length 12.0 --b-length 12.0 --angle 60 --max-strain 0.05
```

By default, strain above `5%` is rejected unless you override `--max-strain`.

### 6.4 Adsorbate Visualization

```bash
cellstine adsorbate visualize output/stacked.vasp
```

## 7. `interface` Workflow

### 7.1 Surface Generation With `interface surface`

Build an `Au(111)`-style slab:

```bash
cellstine interface surface input/Au_Bulk.vasp --miller 1,1,1 --layers 6 --vacuum 15
```

Use negative-index notation with `x`:

```bash
cellstine interface surface input/Au_Bulk.vasp --miller 1,1,2x --layers 6 --vacuum 15
```

Repeat in plane:

```bash
cellstine interface surface input/Au_Bulk.vasp --miller 1,1,0 --layers 4 --repeat-a 2 --repeat-b 2
```

Apply an explicit in-plane `2x2` supercell matrix:

```bash
cellstine interface surface input/Au_Bulk.vasp --miller 1,0,0 --layers 6 --supercell-matrix 2,0,0,3
```

Also write the adsorption-site report:

```bash
cellstine interface surface input/Au_Bulk.vasp --miller 1,1,1 --layers 6 --vacuum 15 --analyse-sites
```

Current practical limitation:

- use a conventional **orthogonal** bulk cell for the native surface builder

### 7.2 Site Analysis With `interface sites`

Analyse an existing slab:

```bash
cellstine interface sites output/surface_Au_Bulk_111_layers4.vasp
cellstine interface sites output/surface_Au_Bulk_111_layers4.vasp --surface-side bottom
```

The site finder reports:

- `top`
- `bridge`
- `hcp_hollow`
- `fcc_hollow`
- `hollow`
- `fourfold_hollow`

For close-packed surfaces such as `(111)`, the native classifier distinguishes `hcp` and `fcc` hollows by subsurface registry.

### 7.3 Interface Construction With `interface build`

Build from two existing slabs:

```bash
cellstine interface build output/bottom_slab.vasp output/top_slab.vasp --gap 3.0
```

Build from two bulks:

```bash
cellstine interface build input/Au_Bulk.vasp input/Au_Bulk.vasp --bottom-kind bulk --top-kind bulk --bottom-miller 1,1,1 --top-miller 1,0,0 --bottom-layers 6 --top-layers 4 --gap 3.0
```

Behavior:

- the bottom slab is fixed
- the top slab is strained in-plane to the bottom slab lattice
- the final heterostructure is written as a VASP structure in `output/`

### 7.4 Bulk Surface Matching With `interface match`

Scan surface combinations from two bulks:

```bash
cellstine interface match input/Au_Bulk.vasp input/Au_Bulk.vasp --bottom-millers 1,1,1 1,0,0 --top-millers 1,1,1 1,1,0 --bottom-layers-list 4 6 --top-layers-list 4 6 --max-strain 0.05
```

Ranking priority:

1. strain
2. total atom count
3. surface area

This means smaller commensurate cells are preferred when strain is otherwise comparable.

### 7.5 Interface Visualization

```bash
cellstine interface visualize output/interface.vasp
```

## 8. Python API Examples

The grouped classes are also usable directly from Python.

Bilayer moire:

```python
from cellstine import Moire

result = Moire().find(
    top_poscar="input/mos2.vasp",
    bottom_poscar="input/mos2.vasp",
    nindex=12,
    explicit_angles=[13.15],
)
print(result.manifest_path)
```

Adsorbate placement:

```python
from cellstine import Molecule

result = Molecule().place(
    substrate_poscar="input/Au_Bulk.vasp",
    molecule_poscar="input/papd_gasp_mol2_final-coor_at_.vasp",
    substrate_kind="bulk",
    miller="1,1,1",
    layers=4,
    site_type="fcc",
    height=2.3,
)
print(result.artifacts["output_poscar"])
```

Surface generation:

```python
from cellstine import Surface

result = Surface().surface(
    bulk_poscar="input/Au_Bulk.vasp",
    miller="1,1,1",
    layers=6,
    vacuum=15.0,
    analyse_sites=True,
)
print(result.summary)
```

## 9. Optional Dependencies And Backends

Current backend behavior:

- `numpy` is required
- VASP I/O is native and always available
- XYZ conversion is handled natively
- `pymatgen` is used first for broad-format conversion when installed
- `plotly` is used for HTML visualizations
- `matplotlib` is reserved for future fallback/static visualization backends

Check installed versions:

```bash
cellstine --version
```

## 10. Testing

Run the current test suite with:

```bash
python -m unittest discover -s tests -q
```

## 11. Troubleshooting

If a moire search returns too many candidates:

- tighten the strain cutoff
- narrow the angle window
- lower `--max-atoms`
- use explicit angle lists where possible

If slab generation fails:

- check that the bulk input is conventional and orthogonal
- start with simple Miller planes like `1,0,0`, `1,1,0`, or `1,1,1`

If a visualization HTML opens blank:

- verify the results file actually contains candidates
- try a single index first
- if you are offline, remember that the current HTML viewer uses the Plotly CDN
