# CELLSTINE Usage Guide

This guide covers the current grouped CELLSTINE package and CLI in more detail than the README.

The package is organized around four top-level workflows:

- `moire`
- `adsorbate`
- `interface`
- `defect`

Examples below use the installed command `cellstine`. Inside the repository, `python moire_cli.py ...` works as a compatibility entrypoint too.

The older top-level `moire` Python package has been retired. Import workflow classes and helpers from `cellstine...` modules going forward.

## 1. Installation And Entry Points

Base install:

```bash
pip install -e .
```

Optional extras:

```bash
pip install -e ".[pymatgen]"
pip install -e ".[viz]"
pip install -e ".[plotly]"
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
output/  -> generated POSCARs, slabs, interfaces, moved structures, static plots, and optional HTML viewers
```

Each grouped stage writes a manifest to:

```text
runs/<workflow>/<run-id>/manifest.json
```

Build stages can consume either a raw results file or a manifest from the previous stage.
Generated outputs include a run-specific identifier with a short `yymmdd-hhmm` timestamp at the end of the filename, or are written into a run-specific output subfolder, which keeps repeated tests with different parameters from overwriting each other.

## 3. Interactive Interface

Launch the guided interface with:

```bash
cellstine
```

You can also jump directly into one workflow group and let CELLSTINE ask only for that group:

```bash
cellstine moire
cellstine adsorbate
cellstine interface
cellstine defect
```

The interactive flow is now grouped:

1. choose `moire`, `adsorbate`, `interface`, or `defect`
2. choose the stage within that workflow
3. enter only the inputs needed for that stage
4. use the manifest or generated artifact for the next step

The guided picker uses the standard folder flow:

- new source structures are suggested from `input/` first
- generated slabs, interfaces, adsorbates, and stacked structures are suggested from `output/` first
- saved searches, manifests, and intermediate workflow artifacts are suggested from `runs/` first

At any picker, type `q`, `quit`, or `exit` to close the interactive interface cleanly.

It is meant to be a guided launcher over the same backend classes used by the CLI, not a separate hidden workflow.

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
cellstine defect --help
cellstine defect analyse --help
cellstine defect generate --help
cellstine defect preview --help
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

Create a commensurate-results Matplotlib summary plot:

```bash
cellstine moire visualize runs/moire/<run-id>/manifest.json --indices 1,2,3
```

The default plot includes labelled strain-vs-angle, atom-count, ranking, and twist-angle distribution panels. Use the optional Plotly path only when you want the interactive 3D frame viewer:

```bash
cellstine moire visualize runs/moire/<run-id>/manifest.json --indices 1,2,3 --plotly
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
For bulk-derived substrates, the primitive surface is generated first; then optional `--substrate-repeat-a`, `--substrate-repeat-b`, or `--substrate-supercell-matrix` expansion is applied before site detection and placement.

Place on an existing slab:

```bash
cellstine adsorbate place output/surface_Au_Bulk_111_layers4.vasp input/papd_gasp_mol2_final-coor_at_.vasp --site-type fcc --site-index 1 --height 2.3
```

Place on a bulk-derived substrate:

```bash
cellstine adsorbate place input/Au_Bulk.vasp input/papd_gasp_mol2_final-coor_at_.vasp --substrate-kind bulk --miller 1,1,1 --layers 4 --vacuum 15 --site-type top --height 2.0
```

Place on a bulk-derived substrate with a non-diagonal in-plane surface matrix:

```bash
cellstine adsorbate place input/Au_Bulk.vasp input/molecule.vasp --substrate-kind bulk --miller 111 --layers 4 --substrate-supercell-matrix 1,1,0,2 --site-type top --height 2.5
```

Place on a primitive slab and let CELLSTINE enlarge the substrate patch if the molecule is too wide for one periodic image:

```bash
cellstine adsorbate place output/Au_111_primitive.vasp input/molecule.vasp --site-type top --height 2.5 --auto-repeat-substrate
```

How placement works:

- the site list is determined from the slab geometry
- guided mode analyses the chosen substrate first and only offers site families found in that specific cell
- the molecule is rotated rigidly about its center of mass
- the molecule is aligned in-plane to the chosen site
- the closest molecule atom is then placed `--height` angstrom above the selected surface plane
- without `--auto-repeat-substrate`, CELLSTINE rejects molecules that cannot fit into one periodic image of the substrate cell instead of folding them into many apparent copies

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

This writes a labelled Matplotlib multi-view PNG by default: top view `x-y`, side view `x-z`, side view `y-z`, and a 3D overview. Add `--plotly` only when you want an optional interactive 3D HTML viewer.

## 7. `interface` Workflow

### 7.1 Surface Generation With `interface surface`

Build an `Au(111)`-style slab:

```bash
cellstine interface surface input/Au_Bulk.vasp --miller 111 --layers 6 --vacuum 15
```

CELLSTINE first builds the primitive surface cell for the requested direction, then applies any requested repeats or supercell matrix. For conventional fcc metals such as Au, Au(111) with four layers gives one atom per layer and `ABCA` stacking; Au(001) with four layers gives `ABAB`. Use `--repeat-a`, `--repeat-b`, or `--supercell-matrix` when you intentionally want a larger surface patch.

In guided interactive mode, the surface workflow previews the primitive cell first: detected centering, stacking sequence, repeating stacking period, atoms per layer, and in-plane angle. It then asks whether to keep the primitive cell, repeat it, or apply a matrix, followed by layer count and vacuum.

`--vacuum` is the empty space above the slab. CELLSTINE places the bottom layer one interlayer spacing above the lower cell boundary, so the same spacing exists below the slab while the requested vacuum remains above the active surface.

Miller notation can be compact or comma-separated:

```bash
cellstine interface surface input/Au_Bulk.vasp --miller 001 --layers 4
cellstine interface surface input/Au_Bulk.vasp --miller 111x --layers 4
cellstine interface surface input/Au_Bulk.vasp --miller 1,1,2x --layers 6
```

Repeat in plane:

```bash
cellstine interface surface input/Au_Bulk.vasp --miller 110 --layers 4 --repeat-a 2 --repeat-b 2
```

Apply an explicit in-plane `2x2` supercell matrix:

```bash
cellstine interface surface input/Au_Bulk.vasp --miller 100 --layers 6 --supercell-matrix 2,0,0,3
```

Also write the adsorption-site report:

```bash
cellstine interface surface input/Au_Bulk.vasp --miller 111 --layers 6 --vacuum 15 --analyse-sites
```

Native surface generation detects primitive, body-centred, face-centred, and base-centred translational lattices from the input structure before cutting the surface slab.

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

This uses the same labelled Matplotlib multi-view format as adsorbate visualization, which is usually more useful for checking slab orientation, vacuum, adsorption height, and layer separation than a free-rotating viewer alone.

## 8. `defect` Workflow

The `defect` workflow analyses valid defect sites first, then generates structures from the discovered site IDs. This is deliberately a two-step flow so users are not asked to guess site indices blindly.

### 8.1 Analyse Defect Sites

```bash
cellstine defect analyse output/Au_Bulk_111_surface.vasp --structure-kind surface
```

Useful options:

- `--structure-kind auto|bulk|surface|slab|molecule-on-substrate`
- `--backend auto|native|pymatgen`
- `--surface-side top|bottom`
- `--layer-tolerance 0.35`
- `--symprec 0.01`

Backend behavior:

- `auto` uses `pymatgen` for bulk equivalence when it is installed.
- slab and surface inputs prefer native layer-aware grouping because 3D bulk symmetry is often the wrong mental model for a finite slab.
- exact Wyckoff labels are only guaranteed with the `pymatgen` backend.

The analysis writes:

```text
runs/defect/<run-id>/manifest.json
runs/defect/<run-id>/defect_analysis.json
```

### 8.2 Preview Site IDs

```bash
cellstine defect preview runs/defect/<run-id>/manifest.json
```

The preview table lists:

- `site_id`
- defect-site kind: `atom`, `interstitial`, or `adatom`
- species and layer where relevant
- multiplicity and represented atom indices
- Wyckoff label when available from `pymatgen`

### 8.3 Generate Vacancy And Substitution Defects

Generate one vacancy POSCAR from one inequivalent atom site:

```bash
cellstine defect generate runs/defect/<run-id>/manifest.json --defect-type vacancy --site-ids atom_001
```

Generate all inequivalent vacancies:

```bash
cellstine defect generate runs/defect/<run-id>/manifest.json --defect-type vacancy
```

Generate a substitution:

```bash
cellstine defect generate runs/defect/<run-id>/manifest.json --defect-type substitution --site-ids atom_001 --substitution-species Pt
```

Restrict substitutions to one original species:

```bash
cellstine defect generate runs/defect/<run-id>/manifest.json --defect-type substitution --original-species Au --substitution-species Pt
```

### 8.4 Generate Interstitials And Adatoms

Generate interstitial candidate structures:

```bash
cellstine defect generate runs/defect/<run-id>/manifest.json --defect-type interstitial --species H
```

Generate an adatom on a detected surface site:

```bash
cellstine defect generate runs/defect/<run-id>/manifest.json --defect-type adatom --site-ids adatom_fcc_hollow_001 --species H --height 2.2
```

For adatoms, `--height` is in angstrom above the detected top/bridge/hollow site. If the chosen height falls outside the current cell, increase the slab vacuum first.

Generated structures are written to:

```text
output/defect_<type>_<site-id>_<species-info>_<yymmdd-hhmm>.vasp
```

## 9. Python API Examples

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

Defect analysis and generation:

```python
from cellstine import Defect

analysis = Defect().analyse(
    structure_path="output/Au_Bulk_111_surface.vasp",
    structure_kind="surface",
)

generated = Defect().generate(
    str(analysis.manifest_path),
    defect_type="vacancy",
    site_ids=["atom_001"],
)
print(generated.artifacts["structures"])
```

## 10. Optional Dependencies And Backends

Current backend behavior:

- `numpy` is required
- VASP I/O is native and always available
- XYZ conversion is handled natively
- `pymatgen` is used first for broad-format conversion and exact bulk defect equivalence when installed
- `matplotlib` is used by default for static, labelled PNG plots
- `plotly` is secondary and only used when `--plotly` is requested for interactive 3D HTML viewers

Check installed versions:

```bash
cellstine --version
```

## 11. Testing

Run the current test suite with:

```bash
python -m unittest discover -s tests -q
```

## 12. Troubleshooting

If a moire search returns too many candidates:

- tighten the strain cutoff
- narrow the angle window
- lower `--max-atoms`
- use explicit angle lists where possible

If slab generation fails:

- check that the bulk input is conventional and orthogonal
- start with simple Miller planes like `1,0,0`, `1,1,0`, or `1,1,1`

If a visualization command says Matplotlib is missing:

- install the visualization extra with `pip install -e ".[viz]"`
- or rerun with `--plotly` if you want the optional HTML viewer instead

If a Plotly visualization HTML opens blank:

- verify the results file actually contains candidates
- try a single index first
- if you are offline, remember that the optional HTML viewer uses the Plotly CDN
